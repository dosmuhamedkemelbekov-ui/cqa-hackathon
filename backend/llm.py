"""Клиент LLM. Два провайдера:
- openai — любой OpenAI-совместимый API: OpenAI в облаке или локальная модель (Ollama, LM Studio, vLLM);
- anthropic — Claude через официальный SDK Anthropic (CQA_PROVIDER=anthropic).
Остальной код вызывает только complete_json() и не зависит от провайдера."""
import asyncio
import json
import re
import sys
import time

from openai import APIConnectionError, APIStatusError, AsyncOpenAI, BadRequestError, RateLimitError

from . import config

_client: AsyncOpenAI | None = None
_sem: asyncio.Semaphore | None = None
# Порядок, в котором снимаем параметры при «немой» ошибке 400.
_OPTIONAL_ORDER = ("prompt_cache_breakpoint", "prompt_cache_key", "seed", "json_schema", "temperature")
# Некоторые модели (reasoning) не принимают temperature/seed или json_schema — запоминаем и больше не шлём.
_unsupported: set[str] = set()


def client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=config.LLM_API_KEY or "local", base_url=config.LLM_BASE_URL, max_retries=3,
                              timeout=config.LLM_TIMEOUT)
    return _client


_rate_lock: asyncio.Lock | None = None
_next_slot = 0.0


async def rate_limit() -> None:
    """Равномерно распределяет запросы: не больше CQA_RPM в минуту (бесплатные тарифы)."""
    global _rate_lock, _next_slot
    if config.LLM_RPM <= 0:
        return
    if _rate_lock is None:
        _rate_lock = asyncio.Lock()
    loop = asyncio.get_running_loop()
    async with _rate_lock:
        now = loop.time()
        wait = _next_slot - now
        _next_slot = max(now, _next_slot) + 60.0 / config.LLM_RPM
    if wait > 0:
        await asyncio.sleep(wait)


def semaphore() -> asyncio.Semaphore:
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(config.LLM_CONCURRENCY)
    return _sem


def is_configured() -> bool:
    if config.LLM_PROVIDER == "anthropic":
        return bool(config.ANTHROPIC_API_KEY)
    return bool(config.LLM_API_KEY or config.LLM_BASE_URL)


def provider_info() -> dict:
    if config.LLM_PROVIDER == "anthropic":
        return {"provider": "anthropic", "model": config.CLAUDE_MODEL, "base_url": config.ANTHROPIC_BASE_URL,
                "effort": config.CLAUDE_EFFORT}
    return {"provider": "openai", "model": config.LLM_MODEL, "base_url": config.LLM_BASE_URL or "https://api.openai.com/v1"}


def log(text: str) -> None:
    """Короткое сообщение в консоль сервера: видно, что происходит с запросами к модели."""
    print(f"[LLM {time.strftime('%H:%M:%S')}] {text}", file=sys.stderr, flush=True)


def _retry_in_text(msg: str) -> float | None:
    """Gemini пишет паузу в теле: «retry in 35.2s» или "retryDelay": "35s"."""
    m = re.search(r"retry in ([\d.]+)\s*s", msg) or re.search(r'retrydelay"?\s*:\s*"?([\d.]+)s', msg)
    return min(90.0, float(m.group(1)) + 1) if m else None


def _retry_after(e: Exception) -> float | None:
    """Пауза из заголовка retry-after (сек), если провайдер её прислал."""
    try:
        v = e.response.headers.get("retry-after")  # type: ignore[attr-defined]
        return min(90.0, float(v)) if v else None
    except Exception:  # noqa: BLE001
        return None


def _usage(resp) -> dict:
    u = resp.usage
    if not u:
        return {"input": 0, "cached": 0, "output": 0}
    cached = 0
    details = getattr(u, "prompt_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", 0) or 0
    return {"input": u.prompt_tokens or 0, "cached": cached, "output": u.completion_tokens or 0}


def cost_usd(usage: dict) -> float:
    uncached = max(0, usage["input"] - usage["cached"])
    if config.LLM_PROVIDER == "anthropic" and config.CLAUDE_MODEL in config.CLAUDE_PRICES:
        p_in, p_read, _p_write, p_out = config.CLAUDE_PRICES[config.CLAUDE_MODEL]
        return (uncached * p_in + usage["cached"] * p_read + usage["output"] * p_out) / 1_000_000
    return (
        uncached * config.PRICE_INPUT_PER_M
        + usage["cached"] * config.PRICE_CACHED_INPUT_PER_M
        + usage["output"] * config.PRICE_OUTPUT_PER_M
    ) / 1_000_000


async def complete_json(system: str, user: str, schema: dict, name: str, model: str | None = None,
                        cache_key: str | None = None, user_prefix: str | None = None) -> tuple[dict, dict]:
    """Возвращает (JSON по схеме, usage). user_prefix — общая для всех критериев часть (транскрипт + факты),
    на её конце ставится точка кэша."""
    if config.LLM_PROVIDER == "anthropic":
        return await _complete_anthropic(system, user, schema, user_prefix)
    primary = model or config.LLM_MODEL
    fb = config.LLM_FALLBACK_MODEL
    if fb and fb != primary and primary in _exhausted:
        return await _complete_openai(system, user, schema, name, fb, cache_key, user_prefix)
    try:
        return await _complete_openai(system, user, schema, name, model, cache_key, user_prefix)
    except RuntimeError as e:
        # Дневной лимит/лимит 0 у основной модели — переключаемся на резервную до перезапуска сервера.
        if fb and fb != primary and ("дневной лимит" in str(e) or "лимит 0" in str(e)):
            if primary not in _exhausted:
                log(f"у {primary} кончился дневной лимит — дальше оцениваю резервной моделью {fb}")
            _exhausted.add(primary)
            return await _complete_openai(system, user, schema, name, fb, cache_key, user_prefix)
        raise


# Модели, у которых в этом запуске кончился дневной лимит (переходим на CQA_FALLBACK_MODEL).
_exhausted: set[str] = set()


def models_used() -> list[str]:
    """Какие модели реально оценивали: основная и, если включалась, резервная."""
    out = [config.LLM_MODEL]
    if config.LLM_FALLBACK_MODEL and _exhausted:
        out.append(config.LLM_FALLBACK_MODEL)
    return out


async def _complete_openai(system: str, user: str, schema: dict, name: str, model: str | None = None,
                           cache_key: str | None = None, user_prefix: str | None = None) -> tuple[dict, dict]:
    """user_prefix — общая часть (транскрипт + факты) для всех критериев диалога. Для gpt-5.6+ на её конце
    ставится явная точка кэша: остальные критерии читают префикс из кэша, а не обрабатывают заново."""
    """Возвращает (распарсенный JSON, usage). Строгая JSON Schema; при отказе модели — json_object."""
    model = model or config.LLM_MODEL
    last_err: Exception | None = None
    rate_waits = 0
    attempt = 0
    if config.LLM_BASE_URL and "api.openai.com" not in config.LLM_BASE_URL:
        _unsupported.update({"prompt_cache_breakpoint", "prompt_cache_key"})
        if "generativelanguage.googleapis.com" in config.LLM_BASE_URL:
            _unsupported.add("seed")  # Gemini не принимает seed — не тратим запрос на пробу
    while attempt < 4:
        attempt += 1
        kwargs: dict = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if user_prefix is not None:
            if "prompt_cache_breakpoint" not in _unsupported:
                kwargs["messages"][1]["content"] = [
                    {"type": "text", "text": user_prefix, "prompt_cache_breakpoint": {"mode": "explicit"}},
                    {"type": "text", "text": user},
                ]
            else:
                kwargs["messages"][1]["content"] = user_prefix + user
        if "temperature" not in _unsupported:
            kwargs["temperature"] = 0
        if "seed" not in _unsupported:
            kwargs["seed"] = 7
        # Один ключ на диалог: запросы по всем критериям попадают на один кэш промпта.
        if cache_key and "prompt_cache_key" not in _unsupported:
            kwargs["prompt_cache_key"] = cache_key
        if "json_schema" not in _unsupported:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": name, "schema": schema, "strict": True},
            }
        else:
            kwargs["response_format"] = {"type": "json_object"}
            kwargs["messages"][0]["content"] += (
                "\n\nВерни ТОЛЬКО JSON-объект строго по схеме:\n" + json.dumps(schema, ensure_ascii=False)
            )
        try:
            async with semaphore():
                await rate_limit()
                resp = await client().chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ""
            return json.loads(text), _usage(resp)
        except BadRequestError as e:
            # Параметр, который модель не принимает, запоминаем и повторяем без него.
            # Проверяем по тому, что было отправлено в ЭТОМ запросе: параллельные вызовы
            # могли уже пометить параметр, но им тоже нужен повтор.
            msg = (str(e) + " " + json.dumps(getattr(e, "body", None), ensure_ascii=False, default=str)).lower()
            sent = {p for p in ("temperature", "seed", "prompt_cache_key") if p in kwargs}
            if isinstance(kwargs["messages"][1]["content"], list):
                sent.add("prompt_cache_breakpoint")
            if kwargs["response_format"]["type"] == "json_schema":
                sent.add("json_schema")
            hit = {p for p in ("temperature", "seed", "prompt_cache_key", "prompt_cache_breakpoint")
                   if p in msg and p in sent}
            if "json_schema" in sent and ("json_schema" in msg or "response_format" in msg):
                hit.add("json_schema")
            if not hit:
                # Некоторые провайдеры (Gemini, Groq) не называют параметр в ошибке 400:
                # снимаем необязательные параметры по одному, от самого специфичного для OpenAI.
                hit = {next((p for p in _OPTIONAL_ORDER if p in sent), "")} - {""}
            last_err = e
            if not hit:
                raise RuntimeError(f"Провайдер отклонил запрос (400): {msg[:400]}") from e
            _unsupported.update(hit)
            log(f"провайдер не принял параметры {sorted(hit)} — повторяю без них")
            attempt -= 1  # снятие параметра не тратит попытку
        except RateLimitError as e:
            # Gemini и др. кладут причину в тело ответа, а не в текст исключения.
            msg = (str(e) + " " + json.dumps(getattr(e, "body", None), ensure_ascii=False, default=str)).lower()
            # Закончились кредиты (OpenAI) — повторять бессмысленно, сразу понятная ошибка.
            if any(s in msg for s in ("insufficient_quota", "no credits", "credit_balance")) or (
                    not config.LLM_BASE_URL and "exceeded your current quota" in msg):
                raise RuntimeError("На ключе закончились кредиты — пополните баланс или смените провайдера в .env") from e
            # Модель недоступна на бесплатном тарифе (лимит 0).
            if re.search(r"limit:\s*0\b", msg):
                raise RuntimeError(f"Модель {model} недоступна на бесплатном тарифе (лимит 0) — "
                                   f"укажите другую в CQA_MODEL. {msg[:300]}") from e
            # Дневной лимит бесплатного тарифа — до завтра не пройдёт.
            if any(s in msg for s in ("per day", "perday", "requests per day", "tokens per day", "free-models-per-day")):
                raise RuntimeError(f"Исчерпан дневной лимит бесплатного тарифа для {model}: {msg[:300]}") from e
            # Поминутный лимит: ждём, сколько просит сервер, и повторяем.
            rate_waits += 1
            if rate_waits > config.LLM_RATE_RETRIES:
                raise RuntimeError(f"Лимит запросов провайдера не освободился: {msg[:300]}") from e
            wait = _retry_after(e) or _retry_in_text(msg) or min(30.0, 5.0 * rate_waits)
            log(f"лимит запросов {model}: жду {wait:.0f} с (повтор {rate_waits}/{config.LLM_RATE_RETRIES})")
            last_err = e
            await asyncio.sleep(wait)
            attempt -= 1  # ожидание лимита не тратит попытки на исправление параметров
            continue
        except APIConnectionError as e:
            last_err = e
            await asyncio.sleep(2 * (attempt + 1))
        except APIStatusError as e:
            if e.status_code >= 500:
                last_err = e
                await asyncio.sleep(2 * (attempt + 1))
            else:
                raise
        except json.JSONDecodeError as e:
            last_err = e
    raise RuntimeError(f"LLM не вернула корректный ответ: {last_err}")


# ======================= Claude (Anthropic) =======================
_aclient = None
# Модели, для которых включаем серверный fallback при отказе классификаторов безопасности.
_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


def aclient():
    global _aclient
    if _aclient is None:
        import anthropic
        # Ключ и адрес передаются явно, чтобы не подхватить чужие переменные окружения.
        _aclient = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY, base_url=config.ANTHROPIC_BASE_URL,
                                            max_retries=3, timeout=300)
    return _aclient


def _claude_usage(resp) -> dict:
    u = resp.usage
    read = getattr(u, "cache_read_input_tokens", 0) or 0
    write = getattr(u, "cache_creation_input_tokens", 0) or 0
    return {"input": (u.input_tokens or 0) + read + write, "cached": read, "output": u.output_tokens or 0}


async def _complete_anthropic(system: str, user: str, schema: dict, user_prefix: str | None) -> tuple[dict, dict]:
    import anthropic
    content = []
    if user_prefix:
        # Транскрипт и факты одинаковы для всех критериев диалога — кэшируем их.
        content.append({"type": "text", "text": user_prefix, "cache_control": {"type": "ephemeral"}})
    content.append({"type": "text", "text": user})
    kwargs: dict = {
        "model": config.CLAUDE_MODEL,
        "max_tokens": 16000,
        "system": [{"type": "text", "text": system}],
        "messages": [{"role": "user", "content": content}],
        "output_config": {"effort": config.CLAUDE_EFFORT, "format": {"type": "json_schema", "schema": schema}},
    }
    use_fallback = config.CLAUDE_MODEL in _FALLBACK_MODELS and "fallbacks" not in _unsupported
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            async with semaphore():
                if use_fallback:
                    resp = await aclient().beta.messages.create(**kwargs, betas=[_FALLBACK_BETA], fallbacks="default")
                else:
                    resp = await aclient().messages.create(**kwargs)
            if resp.stop_reason == "refusal":
                raise RuntimeError("Claude отказался отвечать (refusal)")
            if resp.stop_reason == "max_tokens":
                raise RuntimeError("Claude не уложился в max_tokens")
            text = next((b.text for b in resp.content if b.type == "text"), "")
            return json.loads(text), _claude_usage(resp)
        except anthropic.AuthenticationError as e:
            raise RuntimeError("Неверный ключ Claude: проверьте ANTHROPIC_API_KEY в .env") from e
        except anthropic.BadRequestError as e:
            msg = str(e).lower()
            if "credit balance" in msg:
                raise RuntimeError("На аккаунте Anthropic закончились кредиты — пополните баланс в console.anthropic.com") from e
            if use_fallback and ("fallback" in msg or "beta" in msg):
                _unsupported.add("fallbacks")  # аккаунт/платформа без fallback — работаем без него
                use_fallback = False
                last_err = e
                continue
            raise
        except anthropic.RateLimitError as e:
            last_err = e
            await asyncio.sleep(5 * (attempt + 1))
        except anthropic.APIConnectionError as e:
            last_err = e
            await asyncio.sleep(3 * (attempt + 1))
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                last_err = e
                await asyncio.sleep(5 * (attempt + 1))
            else:
                raise
        except json.JSONDecodeError as e:
            last_err = e
    raise RuntimeError(f"Claude не вернул корректный ответ: {last_err}")
