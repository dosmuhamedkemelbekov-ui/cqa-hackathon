"""Тесты для критерия: проверяем, правильно ли ИИ понял формулировку, ДО применения к реальным диалогам.

1. ИИ одним вызовом пишет несколько коротких реалистичных диалогов «бот → оператор» с заранее заданным
   правильным ответом по этому критерию: каждое допустимое значение хотя бы раз + пограничный случай,
   языки RU/KZ/EN. В тексте не упоминается название критерия и не подсказывается ответ.
2. Каждый диалог оценивается ТЕМ ЖЕ путём, что и реальные (предобработка, промпт критерия, проверка цитат).
3. Сравниваем ответ с заданным: «ИИ понял критерий: 5 из 6, вот где путается».

Ограничение: тесты пишет и проверяет ИИ — это проверка понимания формулировки, а не гарантия точности.
"""
import asyncio
import time

from . import evaluator, llm, preprocess, scoring

GEN_SYSTEM = """Ты — методист контроля качества контакт-центра брокерской компании Freedom Broker.
Пишешь короткие тестовые диалоги, чтобы проверить, правильно ли аналитик применяет критерий оценки оператора.
Диалог реалистичный: клиент пишет боту, бот передаёт чат оператору (системное сообщение), дальше клиент и оператор.
Темы — брокерские: тарифы, пополнение, вывод, отчёты, поиск бумаги, приказы, персональные данные.
Каждый диалог должен однозначно давать заданный ответ по правилу критерия (кроме пограничного — там ответ тоже
должен следовать из правила, но ситуация на грани). Не упоминай в тексте название критерия и не подсказывай ответ."""


def _gen_schema(allowed: list[str]) -> dict:
    return {
        "type": "object", "additionalProperties": False, "required": ["cases"],
        "properties": {"cases": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["expected", "language", "kind", "scenario", "transcript"],
            "properties": {
                "expected": {"type": "string", "enum": allowed},
                "language": {"type": "string", "enum": ["RU", "KZ", "EN"]},
                "kind": {"type": "string", "enum": ["typical", "borderline"]},
                "scenario": {"type": "string"},
                "transcript": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False, "required": ["speaker", "text"],
                    "properties": {"speaker": {"type": "string", "enum": ["client", "bot", "system", "operator"]},
                                   "text": {"type": "string"}}}},
            }}}},
    }


def _label(c: dict, a: str) -> str:
    if a == "na":
        return "N/A"
    if c["type"] == "scale":
        return f"{a} из {int(scoring.scale_max(c))}"
    if c["type"] == "flag":
        return "Да — нарушение" if a == "yes" else "Нет"
    return "Да" if a == "yes" else "Нет"


async def run(criterion: dict, n: int | None = None) -> dict:
    t0 = time.time()
    c = criterion
    allowed = scoring.allowed_answers(c)
    n = n or min(7, max(5, len(allowed) + 2))
    plan = (f"Сгенерируй {n} диалогов. Каждое допустимое значение answer ({', '.join(allowed)}) — хотя бы в одном диалоге; "
            "ровно один диалог kind=borderline (пограничный), остальные typical. Языки: в основном RU, "
            "хотя бы один KZ (казахский) и один EN. В каждом диалоге 5–10 реплик, у оператора — имя.\n\n"
            + evaluator.criterion_prompt(c))
    data, usage = await llm.complete_json(GEN_SYSTEM, plan, _gen_schema(allowed), name="criterion_tests")
    cases = [x for x in data.get("cases", []) if x.get("transcript") and x.get("expected") in allowed][:n]
    calls = 1

    async def check(i: int, case: dict) -> dict:
        tr, sec = [], 0
        for m in case["transcript"]:
            sec += 7 if m["speaker"] == "client" else 4
            tr.append({"timestamp": f"{sec // 60:02d}:{sec % 60:02d}", "speaker": m["speaker"], "text": m["text"]})
        conv = {"conversation_id": f"test-{i + 1}", "channel": "agent chat", "language": case["language"],
                "transcript": tr, "audio_url": None}
        prepared = preprocess.prepare(conv)
        r = await evaluator.evaluate_criterion(c, prepared, 1, None, None, precedents=None,
                                               conv_id=conv["conversation_id"], verify=False)
        got = r.get("answer")
        return {"n": i + 1, "language": case["language"], "kind": case["kind"], "scenario": case["scenario"],
                "expected": case["expected"], "expected_label": _label(c, case["expected"]),
                "got": got, "got_label": _label(c, got) if got else "ошибка",
                "pass": got == case["expected"], "rationale": r.get("rationale") or r.get("error", ""),
                "messages": prepared["messages"], "usage": r.get("usage", {})}

    results = await asyncio.gather(*[check(i, x) for i, x in enumerate(cases)])
    for r in results:
        u = r.pop("usage", None) or {}
        for k in usage:
            usage[k] += u.get(k, 0)
    calls += len(results)
    passed = sum(1 for r in results if r["pass"])
    return {"criterion_id": c.get("id"), "criterion_name": c.get("name"), "total": len(results), "passed": passed,
            "cases": results, "llm_calls": calls, "cost_usd": round(llm.cost_usd(usage), 5),
            "duration_s": round(time.time() - t0, 1)}
