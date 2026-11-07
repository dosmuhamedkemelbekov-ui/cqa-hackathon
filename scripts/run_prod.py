"""Контрольный этап одной командой: получить набор → оценить → выгрузить → проверить → (опц.) commit и push.

    python scripts/run_prod.py <group_id> --team "Название команды"
    python scripts/run_prod.py <group_id> --team "..." --push        # сразу git add/commit/push
    python scripts/run_prod.py --file prod.json --team "..."          # если набор скачан вручную

Оценка идёт активным чек-листом из базы (тем, что сохранён в интерфейсе). Диалоги, которые не удалось
оценить, повторяются до двух раз; если ошибка осталась — диалог попадает в файл со status="error"
(так требует регламент: не скрывать и не заменять нулём).
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend import config, dataset, db, evaluator, llm, results_export  # noqa: E402


async def evaluate_all(ids: list[str], cl: dict, reasons: dict[str, str]) -> list[str]:
    precedents = db.get_setting("precedents", [])
    sem = asyncio.Semaphore(config.DIALOG_CONCURRENCY)
    failed: list[str] = []
    t0 = time.time()

    async def one(cid: str):
        async with sem:
            try:
                ev = await evaluator.evaluate(db.get_conversation(cid), cl, None, None, precedents=precedents)
                ev["reliability"] = evaluator.reliability(ev)
                if ev.get("errors"):
                    raise RuntimeError(f"нет ответа по критериям {ev['errors']}")
                db.save_evaluation(ev)
                print(f"  ✔ {cid}: {ev['final_score']['total']} · грубая ошибка: "
                      f"{'да' if ev['final_score']['gross_error'] else 'нет'} · ${ev['cost_usd']:.4f} · "
                      f"{int(time.time() - t0)} с", flush=True)
            except Exception as e:  # noqa: BLE001
                failed.append(cid)
                reasons[cid] = str(e)
                print(f"  ✘ {cid}: {e}", flush=True)

    await asyncio.gather(*[one(c) for c in ids])
    return failed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("group_id", nargs="?")
    ap.add_argument("--file", help="JSON набора, если API недоступен")
    ap.add_argument("--team", default=os.getenv("CQA_TEAM_NAME", ""))
    ap.add_argument("--folder", default="prod")
    ap.add_argument("--expect", type=int, default=10)
    ap.add_argument("--push", action="store_true", help="git add + commit + push файла результатов")
    args = ap.parse_args()
    if not args.team:
        ap.error("укажите --team \"Название команды\" или CQA_TEAM_NAME в .env")
    if not (args.group_id or args.file):
        ap.error("укажите group_id или --file")
    if not llm.is_configured():
        print("LLM не настроена: задайте OPENAI_API_KEY (или OPENAI_BASE_URL) в .env")
        return 1

    db.init()
    print("1/4 Получение набора…", flush=True)
    if args.file:
        ds, records = dataset.parse_payload(json.loads(Path(args.file).read_text(encoding="utf-8-sig")))
        ds = ds or Path(args.file).stem
    else:
        ds, records = asyncio.run(dataset.fetch_group(args.group_id.strip()))
    ids = [r["conversation_id"] for r in records]
    db.upsert_conversations(records, ds)
    print(f"    набор {ds}: {len(ids)} диалогов", flush=True)
    if len(ids) != args.expect:
        print(f"    внимание: ожидалось {args.expect} диалогов", flush=True)

    cl = db.active_checklist()
    print(f"2/4 Оценка: модель {config.LLM_MODEL}, чек-лист «{cl['name']}» v{cl['version']}", flush=True)
    reasons: dict[str, str] = {}
    failed = asyncio.run(evaluate_all(ids, cl, reasons))
    for attempt in (1, 2):
        if not failed:
            break
        print(f"    повтор {attempt} для {len(failed)} диалогов", flush=True)
        failed = asyncio.run(evaluate_all(failed, cl, reasons))

    print("3/4 Выгрузка и проверка формата…", flush=True)
    out = config.ROOT / "results" / args.folder / "results.json"
    convs = [db.get_conversation(i) for i in ids]
    doc = results_export.build(convs, db.latest_evaluations_map(), args.team, "ai",
                               {c: reasons[c] for c in failed})
    results_export.write(doc, out)
    warnings: list[str] = []
    errs = results_export.validate(doc, ids, args.expect, warnings)
    for w in warnings:
        print("    предупреждение:", w)
    for e in errs:
        print("    ОШИБКА:", e)
    n_ok = sum(1 for r in doc["results"] if r["status"] == "ok")
    print(f"    {out}: {len(doc['results'])} записей, ok {n_ok}", flush=True)
    if errs:
        return 1

    rel = out.relative_to(config.ROOT)
    if args.push:
        print("4/4 git commit + push…", flush=True)
        subprocess.run(["git", "add", str(rel)], cwd=config.ROOT, check=True)
        subprocess.run(["git", "commit", "-m", f"Результаты контрольного набора {ds}"], cwd=config.ROOT, check=True)
        subprocess.run(["git", "push"], cwd=config.ROOT, check=True)
        print("    ✔ отправлено", flush=True)
    else:
        print(f"4/4 Готово. Отправьте файл:\n    git add {rel} && git commit -m \"prod results\" && git push")
    return 0


if __name__ == "__main__":
    sys.exit(main())
