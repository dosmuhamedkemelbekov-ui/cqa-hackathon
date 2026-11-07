"""Проверка детерминированных правил (backend/rules.py) на всех диалогах DEV против эталона — без LLM.

    py scripts/check_rules.py

Подаёт правилам заведомо неверные ответы «модели» и считает, сколько из них правила исправили к эталону
и не увели ли хоть один ответ от эталона. Запускайте после любой правки правил или чек-листа.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backend import preprocess, rules  # noqa: E402


def main() -> int:
    cl = json.loads((ROOT / "checklists" / "default.json").read_text(encoding="utf-8"))
    recs = json.loads((ROOT / "data" / "dev.json").read_text(encoding="utf-8-sig"))["data"]["records"]
    gold = json.loads((ROOT / "gold" / "gold.json").read_text(encoding="utf-8"))
    flip = {"yes": "no", "no": "yes", "na": "yes"}
    fixed = broken = touched = 0
    for r in recs:
        g = gold.get(r["conversation_id"], {}).get("answers")
        if not g:
            continue
        prep = preprocess.prepare(r)
        # «модель» ошиблась везде, где правило может что-то проверить
        res = {c["id"]: {"criterion_id": c["id"], "answer": g.get(c["id"])} for c in cl["criteria"]}
        res["c01"]["answer"] = flip[g["c01"]]
        res["c03"]["answer"] = "na"
        res["c07"]["answer"] = "yes" if g["c07"] != "yes" else "no"
        rules.apply(r, cl, prep, res)
        for cid in ("c01", "c03", "c07", "c02"):
            ru = res[cid].get("rule")
            if not ru or not ru["changed"]:
                continue
            touched += 1
            if res[cid]["answer"] == g[cid]:
                fixed += 1
            else:
                broken += 1
                print(f"✘ {r['conversation_id']} {cid}: правило → {res[cid]['answer']}, эталон {g[cid]} ({ru['why']})")
    print(f"Правила сработали {touched} раз: исправили к эталону {fixed}, увели от эталона {broken}.")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
