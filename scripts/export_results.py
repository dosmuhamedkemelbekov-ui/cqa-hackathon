"""Выгрузка результатов в формат организаторов (JSON 1.0) и проверка файла.

    python scripts/export_results.py DEV                     # → results/dev/results.json
    python scripts/export_results.py PROD --folder prod      # → results/prod/results.json, ожидается 10 записей
    python scripts/export_results.py DEV --use final         # с учётом решений супервизора по апелляциям
    python scripts/export_results.py --check results/prod/results.json --expect 10   # только проверить файл
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend import config, db, results_export  # noqa: E402


def check(path: Path, expected_ids: list[str] | None, expect: int | None) -> bool:
    doc = json.loads(path.read_text(encoding="utf-8"))
    warnings: list[str] = []
    errs = results_export.validate(doc, expected_ids, expect, warnings)
    res = doc.get("results") or []
    ok_n = sum(1 for r in res if r.get("status") == "ok")
    print(f"{path}: записей {len(res)}, ok {ok_n}, error {len(res) - ok_n}")
    for w in warnings:
        print("  предупреждение:", w)
    for e in errs:
        print("  ОШИБКА:", e)
    print("  ✔ формат JSON 1.0 соблюдён" if not errs else f"  ✘ проблем: {len(errs)}")
    return not errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", help="набор в базе (DEV, PROD-группа и т.п.)")
    ap.add_argument("--folder", help="dev или prod (по умолчанию: dev для DEV, иначе prod)")
    ap.add_argument("--out", help="явный путь к файлу")
    ap.add_argument("--use", choices=["ai", "final"], default="ai",
                    help="ai — исходные ответы ИИ (по умолчанию); final — после решений супервизора")
    ap.add_argument("--team", default=os.getenv("CQA_TEAM_NAME", ""), help="название команды (или CQA_TEAM_NAME)")
    ap.add_argument("--expect", type=int, help="ожидаемое число записей (для prod — 10)")
    ap.add_argument("--check", help="только проверить существующий файл")
    args = ap.parse_args()

    if args.check:
        return 0 if check(Path(args.check), None, args.expect) else 1
    if not args.dataset:
        ap.error("укажите набор или --check")
    if not args.team:
        ap.error("укажите название команды: --team \"...\" или CQA_TEAM_NAME в .env")

    db.init()
    convs = db.list_conversations(args.dataset)
    if not convs:
        print(f"В базе нет диалогов набора {args.dataset}. Наборы: {db.list_datasets()}")
        return 1
    folder = args.folder or ("dev" if args.dataset.upper() == "DEV" else "prod")
    out = Path(args.out) if args.out else config.ROOT / "results" / folder / "results.json"
    full = [db.get_conversation(c["conversation_id"]) for c in convs]
    doc = results_export.build(full, db.latest_evaluations_map(), args.team, args.use)
    results_export.write(doc, out)
    expect = args.expect if args.expect is not None else (10 if folder == "prod" else None)
    return 0 if check(out, [c["conversation_id"] for c in convs], expect) else 1


if __name__ == "__main__":
    sys.exit(main())
