"""«Суд» из двух ИИ для спорного критерия: обвинитель и защитник, судья — человек.

Два независимых вызова модели идут параллельно:
- обвинитель отстаивает самый СТРОГИЙ обоснованный ответ (меньше баллов оператору / нарушение есть);
- защитник — самый БЛАГОПРИЯТНЫЙ для оператора обоснованный ответ.
Оба опираются на цитаты (код сверяет их с текстом реплик) и обязаны признать, если их позиция не защищается.
Решение принимает супервизор — ИИ за него не решает.
"""
import asyncio
import time

from . import evaluator, llm, preprocess, scoring

ROLE = {
    "prosecution": ("ОБВИНИТЕЛЬ",
                    "Твоя задача — найти и обосновать самый СТРОГИЙ ответ по критерию, который реально следует из правила: "
                    "меньше баллов оператору, «Нет» вместо «Да», нарушение вместо его отсутствия. Ищи, что оператор пропустил или сделал не так."),
    "defense": ("ЗАЩИТНИК",
                "Твоя задача — найти и обосновать самый БЛАГОПРИЯТНЫЙ для оператора ответ по критерию, который реально следует из правила: "
                "больше баллов, «Да» вместо «Нет», N/A, если условие неприменимости выполнено. Ищи, что оператор сделал правильно."),
}

SYSTEM = """Ты участвуешь в разборе спорной оценки работы оператора контакт-центра Freedom Broker.
Роль: {role}. {task}
Правила честного спора: опирайся только на транскрипт и правило критерия; каждое утверждение подкрепляй точной цитатой
с id реплики; не выдумывай. Реплики бота и системы — контекст, оператору не засчитываются и в вину не ставятся.
Если честно отстоять свою позицию невозможно — поставь concede=true, выбери ответ, который действительно верен,
и объясни это. Пиши по-русски, цитаты — на языке оригинала. argument — 2–4 предложения для супервизора."""


def _schema(allowed: list[str]) -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["position", "argument", "evidence", "weakest_point_of_other_side", "concede"],
        "properties": {
            "position": {"type": "string", "enum": allowed},
            "argument": {"type": "string"},
            "evidence": {"type": "array", "items": {
                "type": "object", "additionalProperties": False, "required": ["msg_id", "quote"],
                "properties": {"msg_id": {"type": "string"}, "quote": {"type": "string"}}}},
            "weakest_point_of_other_side": {"type": "string"},
            "concede": {"type": "boolean"},
        },
    }


async def run(conv: dict, ev: dict, criterion_id: str, appeal_comment: str | None = None) -> dict:
    t0 = time.time()
    cl = ev["checklist_snapshot"]
    c = next(x for x in cl["criteria"] if x["id"] == criterion_id)
    item = next((i for i in ev["criteria"] if i["criterion_id"] == criterion_id), {})
    allowed = scoring.allowed_answers(c)
    prepared = preprocess.prepare(conv)
    prefix = ("ТРАНСКРИПТ (формат: [id | время | роль] текст):\n" + prepared["transcript_text"]
              + "\n\nФАКТЫ:\n" + preprocess.facts_text(prepared["facts"]) + "\n\n")
    current = ev["final_answers"].get(criterion_id)
    case = (evaluator.criterion_prompt(c)
            + f"\n\nТЕКУЩАЯ ОЦЕНКА ИИ: answer=\"{current}\". Обоснование: {item.get('rationale', '')}"
            + (f"\nКОММЕНТАРИЙ ОПЕРАТОРА В АПЕЛЛЯЦИИ: {appeal_comment}" if appeal_comment else ""))
    msgs = {m["id"]: m for m in prepared["messages"]}

    async def side(key: str) -> tuple[str, dict, dict]:
        role, task = ROLE[key]
        data, usage = await llm.complete_json(SYSTEM.format(role=role, task=task), case, _schema(allowed),
                                              name="debate_side", cache_key=f"debate-{conv['conversation_id']}",
                                              user_prefix=prefix)
        if data.get("position") not in allowed:
            data["position"] = current
        data["evidence"] = evaluator.check_quotes(
            [e for e in data.get("evidence", []) if e.get("msg_id") in msgs], msgs)
        data["position_label"] = evaluator._answer_label(c, data["position"])
        return key, data, usage

    res = await asyncio.gather(side("prosecution"), side("defense"))
    usage = {"input": 0, "cached": 0, "output": 0}
    out: dict = {}
    for key, data, u in res:
        out[key] = data
        for k in usage:
            usage[k] += u.get(k, 0)
    p, d = out["prosecution"]["position"], out["defense"]["position"]
    out.update({
        "criterion_id": criterion_id, "criterion_name": c["name"],
        "current": current, "current_label": evaluator._answer_label(c, current),
        "agree": p == d,  # обе стороны пришли к одному ответу — спор, по сути, снят
        "llm_calls": 2, "cost_usd": round(llm.cost_usd(usage), 5), "duration_s": round(time.time() - t0, 1),
    })
    return out
