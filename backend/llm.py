"""Клиент LLM через OpenAI-совместимый API: OpenAI в облаке или локальная модель (Ollama, LM Studio, vLLM)."""
import asyncio
import json

from openai import APIConnectionError, APIStatusError, AsyncOpenAI, BadRequestError, RateLimitError

from . import config

_client: AsyncOpenAI | None = None
_sem: asyncio.Semaphore | None = None
# Некоторые модели (reasoning) не принимают temperature/seed или json_schema — запоминаем и больше не шлём.
_unsupported: set[str] = set()


def client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=config.LLM_API_KEY or "local", base_url=config.LLM_BASE_URL, max_retries=3)
    return _client


def semaphore() -> asyncio.Semaphore:
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(config.LLM_CONCURRENCY)
    return _sem


def is_configured() -> bool:
    return bool(config.LLM_API_KEY or config.LLM_BASE_URL)


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
    return (
        uncached * config.PRICE_INPUT_PER_M
        + usage["cached"] * config.PRICE_CACHED_INPUT_PER_M
        + usage["output"] * config.PRICE_OUTPUT_PER_M
    ) / 1_000_000


async def complete_json(system: str, user: str, schema: dict, name: str, model: str | None = None) -> tuple[dict, dict]:
    """Возвращает (распарсенный JSON, usage). Строгая JSON Schema; при отказе модели — json_object."""
    model = model or config.LLM_MODEL
    last_err: Exception | None = None
    for attempt in range(4):
        kwargs: dict = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if "temperature" not in _unsupported:
            kwargs["temperature"] = 0
        if "seed" not in _unsupported:
            kwargs["seed"] = 7
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
                resp = await client().chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ""
            return json.loads(text), _usage(resp)
        except BadRequestError as e:
            # Параметр, который модель не принимает, запоминаем и повторяем без него.
            # Проверяем по тому, что было отправлено в ЭТОМ запросе: параллельные вызовы
            # могли уже пометить параметр, но им тоже нужен повтор.
            msg = str(e).lower()
            sent = {p for p in ("temperature", "seed") if p in kwargs}
            if kwargs["response_format"]["type"] == "json_schema":
                sent.add("json_schema")
            hit = {p for p in ("temperature", "seed") if p in msg and p in sent}
            if "json_schema" in sent and ("json_schema" in msg or "response_format" in msg):
                hit.add("json_schema")
            last_err = e
            if not hit:
                raise
            _unsupported.update(hit)
        except (RateLimitError, APIConnectionError) as e:
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
