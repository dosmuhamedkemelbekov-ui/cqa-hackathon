"""Точность ИИ против эталона команды (gold/gold.json) по последним оценкам в базе.

    python scripts/accuracy.py              # все размеченные диалоги
    python scripts/accuracy.py --sure       # без спорных ответов эталона (поле uncertain)
    python scripts/accuracy.py --details    # список расхождений по каждому критерию

Сравниваются исходные ответы ИИ (до апелляций). MAE — средняя абсолютная ошибка итогового балла.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend import config, db, scoring  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sure", action="store_true", help="исключить спорные ответы эталона")
    ap.add_argument("--details", action="store_true")
    args = ap.parse_args()

    gold = json.loads((config.ROOT / "gold" / "gold.json").read_text(encoding="utf-8"))
    db.init()
    latest = db.latest_evaluations_map()
    per: dict[str, dict] = {}
    maes, exact, n_conv = [], 0, 0
    for cid, g in sorted(gold.items()):
        ev = latest.get(cid)
        if not ev:
            continue
        n_conv += 1
        cl = ev["checklist_snapshot"]
        unc = set(g.get("uncertain") or [])
        for c in cl["criteria"]:
            if c["id"] not in g["answers"] or (args.sure and c["id"] in unc):
                continue
            pred, true = ev["original_answers"].get(c["id"]), g["answers"][c["id"]]
            r = per.setdefault(c["id"], {"name": c["name"], "n": 0, "ok": 0, "miss": []})
            r["n"] += 1
            if pred == true:
                r["ok"] += 1
            else:
                r["miss"].append(f"{cid}: ИИ {pred} / эталон {true}{' (?)' if c['id'] in unc else ''}")
        gt = scoring.compute(cl, g["answers"])["total"]
        pt = ev["original_score"]["total"]
        if gt is not None and pt is not None:
            maes.append(abs(gt - pt))
            exact += abs(gt - pt) < 0.01
    if not n_conv:
        print("Нет оценённых диалогов с эталоном — сначала запустите оценку (scripts/batch_eval.py DEV).")
        return
    tot_n = sum(r["n"] for r in per.values())
    tot_ok = sum(r["ok"] for r in per.values())
    print(f"Диалогов: {n_conv} · совпадение ответов: {tot_ok}/{tot_n} = {tot_ok / tot_n:.1%} · "
          f"MAE итога: {sum(maes) / len(maes):.2f} · итог совпал точно: {exact}/{len(maes)}")
    for cid, r in per.items():
        print(f"  {cid} {r['name'][:42]:<42} {r['ok']:>3}/{r['n']:<3} {r['ok'] / r['n']:.0%}")
        if args.details:
            for m in r["miss"]:
                print("       ", m)


if __name__ == "__main__":
    main()
