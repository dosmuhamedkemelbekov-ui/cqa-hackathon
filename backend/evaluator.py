"""Оценка диалога по чек-листу: один вызов LLM на критерий (параллельно), итог считает код."""
import asyncio
import re
import time
from collections import Counter

from . import config, llm, preprocess, scoring

SYSTEM_PROMPT = """Ты — старший аналитик контроля качества контакт-центра брокерской компании Freedom Broker.
Ты оцениваешь работу ОПЕРАТОРА (живого сотрудника) в диалоге с клиентом по одному критерию чек-листа.

Общие правила оценки:
1. Оценивается только оператор. Реплики БОТА (AI-ассистент) и СИСТЕМЫ — контекст: что клиент уже сообщил, когда подключился оператор, подтверждена ли личность, как закончился чат. Действия бота оператору не засчитываются и не ставятся в вину.
2. Один и тот же чек-лист применяется к диалогам на русском, казахском и английском. Объяснение всегда пиши на русском, цитаты — на языке оригинала.
3. «Нет» — действие было нужно, но оператор его пропустил или выполнил неправильно. «N/A» — критерий неприменим ТОЛЬКО по условию неприменимости, указанному в критерии. Если N/A для критерия не разрешён, выбирать его нельзя.
4. Опирайся только на транскрипт. Ничего не додумывай: если подтверждения в диалоге нет — действие не выполнено.
5. Для звонков текст получен автоматическим распознаванием речи: возможны искажения слов. Не штрафуй за ошибки распознавания, оценивай смысл. Маски вида [Имя скрыто] / [Персональные данные скрыты] означают, что информация была произнесена.
6. Блок ФАКТЫ посчитан программно по меткам времени и системным событиям — им можно доверять.

Порядок работы:
- Сначала найди реплики, относящиеся к критерию (evidence): id реплики в формате m<номер> из транскрипта и точная короткая цитата. Включай и подтверждающие, и опровергающие реплики.
- Если действие пропущено — укажи в checked_range участок диалога, где оно ожидалось (первый и последний id). Иначе укажи участок, который ты проверил.
- В analysis кратко рассуждай по правилу критерия, затем выбери answer.
- rationale — 1–3 предложения для отчёта в стиле: «Завершение — Нет. После ответа на вопрос оператор закрыл чат без прощания. Основание — последняя реплика оператора и сообщение о закрытии чата». Для N/A объясни, какое условие неприменимости выполнено."""


def _answer_label(criterion: dict, answer: str | None) -> str:
    if answer is None:
        return "—"
    if answer == "na":
        return "N/A"
    if criterion["type"] == "scale":
        return f"{answer} из {int(scoring.scale_max(criterion))}"
    if criterion["type"] == "flag":
        return "Да — нарушение" if answer == "yes" else "Нет"
    return "Да" if answer == "yes" else "Нет"


def criterion_prompt(c: dict) -> str:
    lines = [f"КРИТЕРИЙ: {c['name']}"]
    if c["type"] == "binary":
        lines.append("Тип ответа: Да/Нет" + ("/N/A" if c.get("na_allowed") else "") + ".")
        lines.append(f"Правило: {c['rule']}")
        lines.append("answer: \"yes\" = Да, \"no\" = Нет" + (", \"na\" = N/A." if c.get("na_allowed") else "."))
    elif c["type"] == "flag":
        lines.append("Тип ответа: флаг нарушения (Да/Нет), баллов не даёт.")
        lines.append(f"Правило: {c['rule']}")
        lines.append("answer: \"yes\" = нарушение выявлено, \"no\" = не выявлено.")
    else:
        lines.append("Тип ответа: шкала" + (" или N/A" if c.get("na_allowed") else "") + ". Выбери ровно одну ступень:")
        lines.append(f"Правило: {c['rule']}")
        for s in sorted(c["scale"], key=lambda s: -float(s["value"])):
            lines.append(f"  {s['value']} — {s['label']}")
        lines.append("answer: значение ступени строкой" + (", либо \"na\"." if c.get("na_allowed") else "."))
    if c.get("na_allowed"):
        lines.append(f"Условие N/A: {c.get('na_condition') or 'не задано'}")
    else:
        lines.append("N/A для этого критерия не допускается.")
    if c.get("guidance"):
        lines.append(f"Уточнения к оценке: {c['guidance']}")
    lines.append("Допустимые значения answer для ЭТОГО критерия (только они): "
                 + ", ".join(f'"{a}"' for a in scoring.allowed_answers(c)) + ".")
    return "\n".join(lines)


def response_schema(answers: list[str]) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["evidence", "checked_range", "analysis", "answer", "confidence", "rationale"],
        "properties": {
            "evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["msg_id", "quote", "why"],
                    "properties": {
                        "msg_id": {"type": "string"},
                        "quote": {"type": "string"},
                        "why": {"type": "string"},
                    },
                },
            },
            "checked_range": {
                "type": "object",
                "additionalProperties": False,
                "required": ["from_msg", "to_msg"],
                "properties": {"from_msg": {"type": "string"}, "to_msg": {"type": "string"}},
            },
            "analysis": {"type": "string"},
            "answer": {"type": "string", "enum": answers},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "rationale": {"type": "string"},
        },
    }


def _vote(c: dict, answers: list[str]) -> str:
    if c["type"] == "scale":
        nums = sorted(float(a) for a in answers if a != "na")
        if len(nums) * 2 <= len(answers):  # большинство за N/A
            return "na"
        med = nums[len(nums) // 2]
        return str(int(med)) if med.is_integer() else str(med)
    return Counter(answers).most_common(1)[0][0]


def checklist_schema(checklist: dict) -> dict:
    """Одна схема на весь чек-лист: схема входит в начало промпта, и если она одинакова для всех
    критериев, общий префикс кэшируется. Допустимость ответа для конкретного критерия проверяется в коде."""
    vals: list[str] = []
    for c in checklist["criteria"]:
        for a in scoring.allowed_answers(c):
            if a not in vals:
                vals.append(a)
    return response_schema(vals)


# ---------------- проверка цитат (без LLM) ----------------
def _norm(s: str) -> str:
    s = (s or "").lower().replace("ё", "е")
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def check_quotes(evidence: list[dict], messages: dict[str, dict]) -> list[dict]:
    """Сверяет каждую цитату-доказательство с реальным текстом реплики: ловит выдуманные цитаты."""
    out = []
    for e in evidence:
        m = messages.get(e.get("msg_id"))
        q, t = _norm(e.get("quote", "")), _norm(m["text"]) if m else ""
        ok = False
        if m and q:
            if q in t:
                ok = True
            else:
                qt = [w for w in q.split() if len(w) > 2]
                tt = set(t.split())
                ok = bool(qt) and sum(w in tt for w in qt) / len(qt) >= 0.7
        out.append({**e, "verified": ok})
    return out


# ---------------- прецеденты: решения супервизора как пример трактовки ----------------
def precedents_block(c: dict, precedents: list[dict] | None, conv_id: str) -> str:
    from . import config
    if not precedents:
        return ""
    same = [p for p in precedents if p.get("criterion_id") == c["id"] and p.get("conversation_id") != conv_id]
    same = same[-config.PRECEDENTS_PER_CRITERION:]
    if not same:
        return ""
    lines = ["", "ПРЕЦЕДЕНТЫ ПРОВЕРЯЮЩЕГО по этому критерию (решения супервизора в других диалогах — следуй этой трактовке правила):"]
    for p in same:
        frag = f" Фрагмент: «{p['quote'][:200]}»." if p.get("quote") else ""
        lines.append(f"- ИИ ответил \"{p['ai_answer']}\", проверяющий решил \"{p['decided']}\". Причина: {p['reason'][:300]}.{frag}")
    return "\n".join(lines)


def _is_risky(c: dict, answer: str | None, confidence: str | None) -> bool:
    if answer is None:
        return False
    if confidence and confidence != "high":
        return True
    if answer == "na":
        return True
    if c["type"] == "binary":
        return answer == "no"
    if c["type"] == "flag":
        return answer == "yes"
    if c["type"] == "scale":
        return float(answer) < scoring.scale_max(c)
    return False


VERIFY_PROMPT = """

САМОПРОВЕРКА. Другой аналитик уже оценил этот критерий:
answer: "{answer}"
Обоснование: {rationale}
Доказательства: {evidence}

Ты — независимый контролёр качества. Заново проверь транскрипт строго по правилу, условию N/A и уточнениям критерия.
Типичные ошибки: перепутать N/A и «Нет»; засчитать или поставить в вину оператору действия бота; не заметить, что оператор переспросил уже известное;
поставить «Да» без всех обязательных элементов правила; принять устную проверку личности за её отсутствие; ошибки распознавания речи принять за ошибки оператора.
Если вывод коллеги верен — верни тот же answer. Если ошибочен — верни правильный answer, а в rationale объясни, что исправлено и почему."""


async def evaluate_criterion(c: dict, prepared: dict, samples: int = 1, cache_key: str | None = None,
                             schema: dict | None = None, precedents: list[dict] | None = None,
                             conv_id: str = "", verify: bool | None = None) -> dict:
    # Общая для всех критериев часть — отдельно: на её конце ставится точка кэша.
    prefix = (
        "ТРАНСКРИПТ (формат: [id | время | роль] текст):\n"
        + prepared["transcript_text"]
        + "\n\nФАКТЫ:\n"
        + preprocess.facts_text(prepared["facts"])
        + "\n\n"
    )
    user = criterion_prompt(c) + precedents_block(c, precedents, conv_id)
    schema = schema or response_schema(scoring.allowed_answers(c))
    allowed = set(scoring.allowed_answers(c))

    async def call():
        return await llm.complete_json(SYSTEM_PROMPT, user, schema, name="criterion_evaluation",
                                       cache_key=cache_key, user_prefix=prefix)

    runs = await asyncio.gather(*[call() for _ in range(samples)], return_exceptions=True)
    # Ответ вне допустимых для этого критерия (схема общая) — один повтор.
    for idx, r in enumerate(runs):
        if not isinstance(r, Exception) and r[0].get("answer") not in allowed:
            try:
                runs[idx] = await call()
            except Exception as e:  # noqa: BLE001
                runs[idx] = e
    ok = [r for r in runs if not isinstance(r, Exception)]
    usage = {"input": 0, "cached": 0, "output": 0}
    for _, u in ok:
        for k in usage:
            usage[k] += u[k]
    if not ok:
        return {"criterion_id": c["id"], "answer": None, "error": str(runs[0]), "usage": usage}

    answers = [r["answer"] for r, _ in ok if r.get("answer") in allowed]
    if not answers:
        return {"criterion_id": c["id"], "answer": None, "error": "Модель вернула недопустимый ответ", "usage": usage}
    final = _vote(c, answers)
    chosen = next(r for r, _ in ok if r.get("answer") == final) if final in answers else ok[0][0]

    ids = {m["id"]: m for m in prepared["messages"]}

    def pack(r: dict) -> dict:
        ev_ = check_quotes([e for e in r.get("evidence", []) if e.get("msg_id") in ids], ids)
        rng = r.get("checked_range") or {}
        chk = {"from_msg": rng["from_msg"], "to_msg": rng["to_msg"]} if rng.get("from_msg") in ids and rng.get("to_msg") in ids else None
        return {"evidence": ev_, "checked_range": chk, "analysis": r.get("analysis", ""),
                "rationale": r.get("rationale", ""), "confidence": r.get("confidence")}

    result = {"criterion_id": c["id"], "answer": final, "votes": dict(Counter(answers)) if samples > 1 else None,
              **pack(chosen), "verification": None, "usage": usage}

    # Самопроверка рискованных ответов независимым «контролёром» (тот же кэшируемый префикс).
    from . import config
    if (config.LLM_VERIFY if verify is None else verify) and _is_risky(c, final, result["confidence"]):
        ev_txt = "; ".join(f"{e['msg_id']}: «{e.get('quote', '')}»" for e in result["evidence"]) or "нет"
        vuser = user + VERIFY_PROMPT.format(answer=final, rationale=result["rationale"], evidence=ev_txt)
        try:
            vr, vu = await llm.complete_json(SYSTEM_PROMPT, vuser, schema, name="criterion_evaluation",
                                             cache_key=cache_key, user_prefix=prefix)
            for k in usage:
                usage[k] += vu[k]
            va = vr.get("answer")
            if va in allowed and va != final:
                result.update({"answer": va, **pack(vr)})
                result["verification"] = {"verdict": "changed", "from": final, "to": va, "reason": vr.get("rationale", "")}
            else:
                result["verification"] = {"verdict": "confirmed", "reason": vr.get("rationale", "")}
        except Exception as e:  # noqa: BLE001 — самопроверка не должна ронять оценку
            result["verification"] = {"verdict": "error", "reason": str(e)[:300]}
    return result


def build_result(conv: dict, checklist: dict, crit_results: dict[str, dict], meta: dict) -> dict:
    criteria = []
    answers = {}
    for c in checklist["criteria"]:
        r = crit_results.get(c["id"], {"answer": None, "error": "не оценивался"})
        if r.get("answer") is not None:
            answers[c["id"]] = r["answer"]
        criteria.append(
            {
                "criterion_id": c["id"],
                "name": c["name"],
                "section": c.get("section"),
                "type": c["type"],
                "weight": c.get("weight", 0),
                "na_allowed": c.get("na_allowed", False),
                "answer": r.get("answer"),
                "answer_label": _answer_label(c, r.get("answer")),
                "evidence": r.get("evidence", []),
                "checked_range": r.get("checked_range"),
                "rationale": r.get("rationale", ""),
                "analysis": r.get("analysis", ""),
                "confidence": r.get("confidence"),
                "votes": r.get("votes"),
                "verification": r.get("verification"),
                "error": r.get("error"),
            }
        )
    score = scoring.compute(checklist, answers)
    return {
        "conversation_id": conv["conversation_id"],
        "checklist_id": checklist["id"],
        "checklist_version": checklist["version"],
        "checklist_snapshot": checklist,
        "criteria": criteria,
        "original_answers": answers,
        "final_answers": dict(answers),
        "original_score": score,
        "final_score": score,
        "overrides": {},
        **meta,
    }


PROMPT_FIELDS = ("name", "type", "rule", "na_allowed", "na_condition", "guidance", "scale")


def _same_prompt(a: dict, b: dict) -> bool:
    """Критерий не менялся по смыслу (вес и секция на ответ ИИ не влияют)."""
    return all(a.get(f) == b.get(f) for f in PROMPT_FIELDS)


async def evaluate(conv: dict, checklist: dict, samples: int | None = None, previous: dict | None = None,
                   reuse_answers: bool = True, precedents: list[dict] | None = None) -> dict:
    """previous — прошлая оценка диалога: неизменённые критерии берутся из неё без вызова LLM,
    решения супервизора по ним сохраняются. Новые и изменённые критерии оцениваются заново."""
    samples = samples or int(getattr(config, "LLM_SAMPLES", 1) or 1)
    t0 = time.time()
    prepared = preprocess.prepare(conv)
    reused: dict[str, dict] = {}
    unchanged: set[str] = set()  # критерии, по которым вердикт супервизора остаётся в силе
    if previous:
        prev_defs = {c["id"]: c for c in previous["checklist_snapshot"]["criteria"]}
        prev_items = {i["criterion_id"]: i for i in previous["criteria"]}
        for c in checklist["criteria"]:
            p, item = prev_defs.get(c["id"]), prev_items.get(c["id"])
            if p and _same_prompt(c, p):
                unchanged.add(c["id"])
            if reuse_answers and p and item and item.get("answer") is not None and _same_prompt(c, p):
                reused[c["id"]] = {
                    **{k: item.get(k) for k in ("answer", "evidence", "checked_range", "rationale", "analysis",
                                                "confidence", "votes", "verification")},
                    "criterion_id": c["id"],
                    "reused": True,
                    "usage": {"input": 0, "cached": 0, "output": 0},
                }
    todo = [c for c in checklist["criteria"] if c["id"] not in reused]
    # Прогрев кэша: общий префикс (системный промпт + транскрипт + факты) одинаков для всех критериев.
    # Первый вызов идёт один и кладёт префикс в кэш, остальные параллельно читают его — дешевле и быстрее.
    key = f"cqa-{conv['conversation_id']}"
    schema = checklist_schema(checklist)
    results = []
    if todo:
        kw = {"precedents": precedents, "conv_id": conv["conversation_id"]}
        results.append(await evaluate_criterion(todo[0], prepared, samples, key, schema, **kw))
        results += await asyncio.gather(*[evaluate_criterion(c, prepared, samples, key, schema, **kw) for c in todo[1:]])
    crit_results = {r["criterion_id"]: r for r in results} | reused
    usage = {"input": 0, "cached": 0, "output": 0}
    for r in results:
        for k in usage:
            usage[k] += r["usage"][k]
    meta = {
        "model": config.LLM_MODEL,
        "samples": samples,
        "llm_calls": len(results) * samples + sum(
            1 for r in results if (r.get("verification") or {}).get("verdict") in ("confirmed", "changed")),
        "reused_criteria": sorted(reused),
        "usage": usage,
        "cost_usd": round(llm.cost_usd(usage), 5),
        "duration_s": round(time.time() - t0, 1),
        "errors": [r["criterion_id"] for r in results if r.get("answer") is None],
    }
    ev = build_result(conv, checklist, crit_results, meta)
    for item in ev["criteria"]:
        item["reused"] = item["criterion_id"] in reused
    # Вердикты супервизора по неизменённым критериям остаются в силе.
    if previous:
        for cid, ans in (previous.get("overrides") or {}).items():
            if cid in unchanged:
                try:
                    apply_override(ev, cid, ans)
                except (ValueError, StopIteration):
                    pass
        ev["decisions"] = [d for d in previous.get("decisions", []) if d["criterion_id"] in unchanged]
        ev["previous_evaluation_id"] = previous.get("id")
        ev["previous_total"] = previous["final_score"]["total"]
    return ev


def apply_override(ev: dict, criterion_id: str, answer: str) -> dict:
    """Пересчёт итога после решения проверяющего по апелляции."""
    cl = ev["checklist_snapshot"]
    c = next(c for c in cl["criteria"] if c["id"] == criterion_id)
    if answer not in scoring.allowed_answers(c):
        raise ValueError(f"Недопустимый ответ {answer} для критерия {c['name']}")
    ev["final_answers"][criterion_id] = answer
    if answer != ev["original_answers"].get(criterion_id):
        ev["overrides"][criterion_id] = answer
    else:
        ev["overrides"].pop(criterion_id, None)
    ev["final_score"] = scoring.compute(cl, ev["final_answers"])
    for item in ev["criteria"]:
        if item["criterion_id"] == criterion_id:
            item["final_answer"] = answer
            item["final_answer_label"] = _answer_label(c, answer)
    return ev


# ---------------- надёжность оценки ----------------
def reliability(ev: dict) -> dict:
    """Сводка доверия к оценке: проверенные цитаты, самопроверка, неуверенные ответы."""
    quotes = [e for i in ev["criteria"] for e in (i.get("evidence") or [])]
    ver = [i.get("verification") for i in ev["criteria"] if i.get("verification")]
    return {
        "quotes_total": len(quotes),
        "quotes_verified": sum(1 for e in quotes if e.get("verified")),
        "self_checked": sum(1 for v in ver if v["verdict"] in ("confirmed", "changed")),
        "self_corrected": sum(1 for v in ver if v["verdict"] == "changed"),
        "low_confidence": sum(1 for i in ev["criteria"] if i.get("confidence") == "low"),
    }


# ---------------- коучинг оператора ----------------
COACH_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["summary", "strengths", "tips"],
    "properties": {
        "summary": {"type": "string"},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "tips": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["criterion", "msg_id", "problem", "better_phrase"],
            "properties": {"criterion": {"type": "string"}, "msg_id": {"type": "string"},
                           "problem": {"type": "string"}, "better_phrase": {"type": "string"}}}},
    },
}

COACH_SYSTEM = """Ты — наставник операторов контакт-центра брокерской компании Freedom Broker.
По итогам оценки диалога дай оператору короткую, конкретную и доброжелательную обратную связь.
summary — 1–2 предложения на русском: главный вывод. strengths — 1–3 сильные стороны (на русском).
tips — до 3 советов по критериям, где потеряны баллы (самые дорогие ошибки первыми): problem — что не так (на русском, со ссылкой на ситуацию),
msg_id — реплика, где это произошло (m<номер>), better_phrase — готовая фраза, которую стоило сказать, НА ЯЗЫКЕ КЛИЕНТА, в тоне компании.
Если ошибок нет — tips пустой."""


async def coach(conv: dict, ev: dict) -> tuple[dict, dict]:
    prepared = preprocess.prepare(conv)
    cl = ev["checklist_snapshot"]
    lines = []
    for item in ev["criteria"]:
        c = next(x for x in cl["criteria"] if x["id"] == item["criterion_id"])
        a = ev["final_answers"].get(c["id"])
        lines.append(f"- {c['name']}: {_answer_label(c, a)}. {item.get('rationale', '')}")
    user = ("ТРАНСКРИПТ:\n" + prepared["transcript_text"] + "\n\nЯзык клиента: " + prepared["facts"]["client_language"]
            + f"\n\nИТОГ: {ev['final_score']['total']} из 100. Грубая ошибка: {'да' if ev['final_score']['gross_error'] else 'нет'}."
            + "\nОЦЕНКА ПО КРИТЕРИЯМ:\n" + "\n".join(lines))
    return await llm.complete_json(COACH_SYSTEM, user, COACH_SCHEMA, name="operator_coaching")
