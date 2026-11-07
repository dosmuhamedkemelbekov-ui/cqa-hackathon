"""Применить актуальные правила (backend/rules.py) к уже сделанным оценкам — без новых запросов к модели.

    py scripts/reapply_rules.py            # все диалоги
    py scripts/reapply_rules.py DEV        # только набор DEV

Берёт исходный ответ модели по каждому критерию (до прежних правил), заново применяет правила,
пересчитывает итог и сохраняет новую оценку. Решения супервизора по апелляциям сохраняются.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend import db, evaluator, preprocess, rules  # noqa: E402


def main() -> None:
    db.init()
    ds = sys.argv[1] if len(sys.argv) > 1 else None
    latest = db.latest_evaluations_map()
    n = changed = 0
    for c in db.list_conversations(ds):
        ev = latest.get(c["conversation_id"])
        if not ev:
            continue
        conv = db.get_conversation(c["conversation_id"])
        prep = preprocess.prepare(conv)
        cl = ev["checklist_snapshot"]
        crit = {}
        for item in ev["criteria"]:
            r = {k: item.get(k) for k in ("answer", "evidence", "checked_range", "rationale", "analysis",
                                          "confidence", "votes", "verification")}
            ru = item.get("rule")
            if ru and ru.get("changed"):  # вернуть ответ модели до прежнего правила
                r["answer"] = ru["from"]
                r["rationale"] = ru.get("llm_rationale", r.get("rationale"))
            r["criterion_id"] = item["criterion_id"]
            crit[item["criterion_id"]] = r
        fired = rules.apply(conv, cl, prep, crit)
        meta = {k: ev.get(k) for k in ("model", "samples", "eval_mode", "llm_calls", "reused_criteria", "usage",
                                       "cost_usd", "duration_s", "errors")}
        new = evaluator.build_result(conv, cl, crit, {**meta, "rules_fired": fired, "rules_reapplied": True})
        for cid, ans in (ev.get("overrides") or {}).items():
            try:
                evaluator.apply_override(new, cid, ans)
            except (ValueError, StopIteration):
                pass
        new["decisions"] = ev.get("decisions", [])
        new["reliability"] = evaluator.reliability(new)
        if new["original_answers"] != ev["original_answers"]:
            changed += 1
            db.save_evaluation(new)
        n += 1
    print(f"Проверено оценок: {n}, изменено правилами: {changed}. Запустите: py scripts/accuracy.py --details")


if __name__ == "__main__":
    main()
