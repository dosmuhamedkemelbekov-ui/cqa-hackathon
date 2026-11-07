"""Пакетная оценка без веб-сервера — для долгих прогонов (весь набор, несколько прогонов на критерий).

    python scripts/batch_eval.py DEV                 # все диалоги набора, только неоценённые/устаревшие
    python scripts/batch_eval.py DEV --all           # переоценить все заново
    python scripts/batch_eval.py --ids p1-ru-001,p4-ru-002 --all

Каждый диалог сохраняется в базу сразу после оценки: остановка не теряет прогресс,
повторный запуск продолжит с неоценённых. Интерфейс увидит результаты после обновления страницы.
"""
import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend import config, db, evaluator  # noqa: E402


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("--ids")
    ap.add_argument("--all", action="store_true", help="переоценить заново, даже если уже оценено текущей версией")
    args = ap.parse_args()

    db.init()
    cl = db.active_checklist()
    ids = args.ids.split(",") if args.ids else [c["conversation_id"] for c in db.list_conversations(args.dataset)]
    if not args.all:
        latest = db.latest_evaluations_map()
        ids = [i for i in ids if not (latest.get(i) and latest[i]["checklist_id"] == cl["id"]
                                      and latest[i]["checklist_version"] == cl["version"]
                                      and latest[i].get("samples") == config.LLM_SAMPLES)]
    print(f"Модель {config.LLM_MODEL} · прогонов на критерий {config.LLM_SAMPLES} · самопроверка {config.LLM_VERIFY} · "
          f"чек-лист {cl['id']} v{cl['version']} · диалогов {len(ids)}", flush=True)
    precedents = db.get_setting("precedents", [])
    sem = asyncio.Semaphore(config.DIALOG_CONCURRENCY)
    done, failed, cost, t0 = 0, [], 0.0, time.time()

    async def one(cid: str):
        nonlocal done, cost
        async with sem:
            try:
                conv = db.get_conversation(cid)
                ev = await evaluator.evaluate(conv, cl, None, db.latest_evaluation(cid), reuse_answers=False,
                                              precedents=precedents)
                ev["reliability"] = evaluator.reliability(ev)
                db.save_evaluation(ev)
                cost += ev["cost_usd"]
                done += 1
                print(f"[{done}/{len(ids)}] {cid}: {ev['final_score']['total']} · вызовов {ev['llm_calls']} · "
                      f"{ev['duration_s']} с · ошибок {len(ev['errors'])} · {int(time.time() - t0)} с всего", flush=True)
            except Exception as e:  # noqa: BLE001
                failed.append(cid)
                print(f"ОШИБКА {cid}: {e}", flush=True)

    await asyncio.gather(*[one(c) for c in ids])
    print(f"ГОТОВО: {done} оценено, ошибок {len(failed)} {failed[:5]}, ≈${cost:.2f} (по ценам из .env), "
          f"{int(time.time() - t0)} с", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
