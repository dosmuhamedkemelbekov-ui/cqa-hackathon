"""HTTP API + раздача фронтенда. Запуск: python -m uvicorn backend.app:app --port 8000"""
import asyncio
import json
import uuid
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, dataset, db, evaluator, intake, llm, preprocess, results_export, review_queue, scoring

app = FastAPI(title="CQA — Conversation Quality Analytics")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
db.init()

JOBS: dict[str, dict] = {}
_TASKS: set = set()  # ссылки на фоновые задачи, чтобы их не собрал сборщик мусора


# ---------------- служебное ----------------
@app.get("/api/health")
def health():
    return {
        "ok": True,
        "llm_configured": llm.is_configured(),
        **llm.provider_info(),
        "samples": config.LLM_SAMPLES,
        "stt_configured": intake.stt_configured(),
        "stt_model": config.STT_MODEL,
        "verify": config.LLM_VERIFY,
        "eval_mode": config.EVAL_MODE,
        "rpm": config.LLM_RPM,
        "precedents": len(db.get_setting("precedents", [])),
    }


# ---------------- наборы и диалоги ----------------
class FetchReq(BaseModel):
    group_id: str


@app.post("/api/datasets/fetch")
async def fetch_dataset(req: FetchReq):
    gid = req.group_id.strip()
    source = "api"
    try:
        ds, records = await dataset.fetch_group(gid)
    except Exception as e:  # noqa: BLE001
        # DEV есть в репозитории (data/dev.json): если API организаторов недоступен, берём его оттуда.
        local = config.ROOT / "data" / "dev.json"
        if gid.upper() != "DEV" or not local.exists():
            raise HTTPException(502, f"Не удалось получить набор «{gid}» из API организаторов: {e}")
        ds, records = dataset.parse_payload(json.loads(local.read_text(encoding="utf-8-sig")))
        ds, source = ds or "DEV", "file"
    n = db.upsert_conversations(records, ds)
    return {"dataset": ds, "loaded": n, "source": source}


@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile = File(...), name: str | None = None):
    try:
        payload = json.loads((await file.read()).decode("utf-8-sig"))
        ds, records = dataset.parse_payload(payload)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Файл не распознан: {e}")
    ds = name or ds or (file.filename or "UPLOAD").rsplit(".", 1)[0]
    return {"dataset": ds, "loaded": db.upsert_conversations(records, ds)}


@app.get("/api/datasets")
def datasets():
    return db.list_datasets()


@app.get("/api/conversations")
def conversations(dataset: str | None = None):
    items = db.list_conversations(dataset)
    active = db.active_checklist()
    latest = db.latest_evaluations_map()
    open_appeals: dict[str, int] = {}
    for a in db.list_appeals(status="open"):
        open_appeals[a["conversation_id"]] = open_appeals.get(a["conversation_id"], 0) + 1
    for it in items:
        ev = latest.get(it["conversation_id"])
        if ev:
            it["evaluation"] = {
                "id": ev["id"],
                "total": ev["final_score"]["total"],
                "original_total": ev["original_score"]["total"],
                "gross_error": ev["final_score"]["gross_error"],
                "checklist_id": ev["checklist_id"],
                "checklist_version": ev["checklist_version"],
                "stale": ev["checklist_id"] != active["id"] or ev["checklist_version"] != active["version"],
                "answers": ev["final_answers"],
                "errors": ev.get("errors", []),
                "overridden": bool(ev.get("overrides")),
            }
        it["open_appeals"] = open_appeals.get(it["conversation_id"], 0)
        if ev:  # риск для сортировки «очередь проверки» (без профиля — быстро)
            it["risk"], it["risk_reasons"] = review_queue.risk(ev, None, it["open_appeals"])
        if not ev:
            it["status"] = "Не оценено"
        elif it["open_appeals"]:
            it["status"] = "Апелляция"
        elif ev.get("decisions"):
            it["status"] = "Пересмотрено"
        else:
            it["status"] = "Оценено"
    return items


@app.get("/api/conversations/{conv_id}")
def conversation(conv_id: str):
    conv = db.get_conversation(conv_id)
    if not conv:
        raise HTTPException(404, "Диалог не найден")
    prepared = preprocess.prepare(conv)
    return {
        "conversation_id": conv_id,
        "dataset": conv.get("dataset"),
        "channel": conv.get("channel"),
        "language": conv.get("language"),
        "has_audio": bool(conv.get("audio_url")),
        "audio_page": conv.get("audio_url"),
        "messages": prepared["messages"],
        "facts": prepared["facts"],
        "profile": preprocess.compute_profile(prepared["messages"], prepared["is_voice"]),
        "evaluation": db.latest_evaluation(conv_id),
        "history": db.evaluation_history(conv_id),
        "appeals": db.list_appeals(conversation_id=conv_id),
    }


@app.get("/api/conversations/{conv_id}/audio")
async def audio(conv_id: str):
    conv = db.get_conversation(conv_id)
    if not conv or not conv.get("audio_url"):
        raise HTTPException(404, "У диалога нет записи")
    local = intake.upload_path(conv["audio_url"])
    if local:
        return FileResponse(local)
    if conv["audio_url"].startswith(intake.UPLOAD_PREFIX):
        raise HTTPException(404, "Файл записи не найден на сервере")
    try:
        url = await dataset.resolve_audio(conv["audio_url"])
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"Не удалось получить запись: {e}")
    return RedirectResponse(url, status_code=302)


# ---------------- новый диалог: текст или аудио ----------------
MANUAL_DATASET = "MANUAL"


class NewConvReq(BaseModel):
    conversation_id: str
    channel: str = "chat"  # chat | call
    language: str = "RU"
    transcript: list[dict]
    audio_file: str | None = None  # имя файла из /api/transcribe
    replace: bool = False


@app.post("/api/conversations")
def create_conversation(req: NewConvReq):
    if not req.replace and db.get_conversation(req.conversation_id):
        raise HTTPException(409, f"Диалог с ID «{req.conversation_id}» уже есть")
    try:
        rec = intake.make_record(req.conversation_id, req.channel, req.language, req.transcript, req.audio_file)
    except ValueError as e:
        raise HTTPException(400, str(e))
    db.upsert_conversations([rec], MANUAL_DATASET)
    return {"conversation_id": rec["conversation_id"], "dataset": MANUAL_DATASET, "messages": len(rec["transcript"])}


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = "AUTO", operator: str | None = None):
    """Аудио → Whisper → роли говорящих (LLM). Возвращает транскрипт для проверки и имя сохранённого файла."""
    if not intake.stt_configured():
        raise HTTPException(400, "Распознавание речи не настроено: задайте OPENAI_API_KEY (или CQA_STT_API_KEY) в .env")
    try:
        name = intake.save_upload(await file.read(), file.filename or "audio")
    except ValueError as e:
        raise HTTPException(400, str(e))
    path = intake.UPLOAD_DIR / name
    language = (language or "AUTO").upper()
    try:
        stt = await intake.transcribe(path, language)
    except Exception as e:  # noqa: BLE001
        path.unlink(missing_ok=True)
        raise HTTPException(502, f"Распознавание речи не удалось: {e}")
    warnings = []
    lang = language if language in intake.LANG_CODE else intake.LANG_FROM_WHISPER.get(stt["language"].lower())
    if not lang:
        lang = "RU"
        warnings.append(f"Язык «{stt['language'] or 'не определён'}» не входит в RU/KZ/EN — установлен RU")
    segs = stt["segments"]
    roles_by = "llm"
    try:
        if not llm.is_configured():
            raise RuntimeError("LLM не настроена")
        labels, fixed = await intake.diarize(segs, lang, (operator or "").strip() or None)
        if fixed:
            warnings.append("Роли части реплик определены по чередованию — проверьте разметку")
    except Exception as e:  # noqa: BLE001
        labels, roles_by = intake.heuristic_roles(segs), "heuristic"
        warnings.append(f"Роли говорящих размечены по паузам, без ИИ ({e}) — проверьте operator/client")
    transcript = intake.build_transcript(segs, labels)
    roles_ok = any(t["speaker"] == "operator" for t in transcript) and any(t["speaker"] == "client" for t in transcript)
    if not roles_ok:
        warnings.append("Не удалось выделить обе роли (operator и client) — исправьте разметку вручную")
    return {"audio_file": name, "language": lang, "duration": round(stt["duration"], 1), "transcript": transcript,
            "roles_by": roles_by, "roles_ok": roles_ok, "warnings": warnings}


# ---------------- чек-листы ----------------
def _validate_checklist(cl: dict) -> dict:
    if not cl.get("id") or not cl.get("name"):
        raise HTTPException(400, "У чек-листа должны быть id и name")
    crits = cl.get("criteria") or []
    if not crits:
        raise HTTPException(400, "Добавьте хотя бы один критерий")
    seen = set()
    section_ids = {s["id"] for s in cl.get("sections") or []}
    for c in crits:
        if not c.get("id") or c["id"] in seen:
            raise HTTPException(400, f"Пустой или повторяющийся id критерия: {c.get('id')}")
        seen.add(c["id"])
        if c.get("type") not in ("binary", "scale", "flag"):
            raise HTTPException(400, f"{c['id']}: тип должен быть binary, scale или flag")
        if not c.get("name") or not c.get("rule"):
            raise HTTPException(400, f"{c['id']}: заполните название и правило")
        c["weight"] = float(c.get("weight") or 0) if c["type"] != "flag" else 0
        if c["weight"] < 0:
            raise HTTPException(400, f"{c['id']}: вес не может быть отрицательным")
        if c["type"] == "scale":
            scale = c.get("scale") or []
            if len(scale) < 2:
                raise HTTPException(400, f"{c['id']}: у шкалы должно быть минимум две ступени")
            for s in scale:
                s["value"] = float(s["value"])
                if s["value"].is_integer():
                    s["value"] = int(s["value"])
        if c.get("section") and section_ids and c["section"] not in section_ids:
            raise HTTPException(400, f"{c['id']}: неизвестная секция {c['section']}")
    weight_sum = sum(c["weight"] for c in crits)
    if weight_sum <= 0:
        raise HTTPException(400, "Сумма весов должна быть больше нуля")
    return cl


@app.get("/api/checklists")
def checklists():
    active = db.get_setting("active_checklist")
    return [{**c, "active": c["id"] == active, "versions": db.checklist_versions(c["id"])} for c in db.list_checklists()]


@app.get("/api/checklists-default")
def checklist_default():
    return json.loads(config.DEFAULT_CHECKLIST.read_text(encoding="utf-8"))


@app.get("/api/checklists/{cl_id}")
def checklist(cl_id: str, version: int | None = None):
    cl = db.get_checklist(cl_id, version)
    if not cl:
        raise HTTPException(404, "Чек-лист не найден")
    return cl


@app.post("/api/checklists")
def save_checklist(cl: dict[str, Any], activate: bool = True):
    cl = _validate_checklist({k: v for k, v in cl.items() if k not in ("version", "saved_at", "active", "versions")})
    saved = db.save_checklist(cl)
    if activate:
        db.set_setting("active_checklist", saved["id"])
    return saved


@app.post("/api/checklists/{cl_id}/activate")
def activate_checklist(cl_id: str):
    if not db.get_checklist(cl_id):
        raise HTTPException(404, "Чек-лист не найден")
    db.set_setting("active_checklist", cl_id)
    return {"active": cl_id}


# ---------------- оценка ----------------
class EvalReq(BaseModel):
    conversation_ids: list[str] | None = None
    dataset: str | None = None
    checklist_id: str | None = None
    only_missing: bool = False
    samples: int | None = None
    reuse: bool = True  # неизменённые критерии берутся из прошлой оценки без вызова LLM


def _checklist_for(req_id: str | None) -> dict:
    cl = db.get_checklist(req_id) if req_id else db.active_checklist()
    if not cl:
        raise HTTPException(404, "Чек-лист не найден")
    return cl


@app.post("/api/evaluate/{conv_id}")
async def evaluate_one(conv_id: str, checklist_id: str | None = None, samples: int | None = None, fresh: bool = True):
    if not llm.is_configured():
        raise HTTPException(400, "LLM не настроена: задайте OPENAI_API_KEY (или OPENAI_BASE_URL для локальной модели) в .env")
    conv = db.get_conversation(conv_id)
    if not conv:
        raise HTTPException(404, "Диалог не найден")
    # fresh=True: ИИ заново отвечает по всем критериям; вердикты супервизора по неизменённым критериям сохраняются.
    previous = db.latest_evaluation(conv_id)
    ev = await evaluator.evaluate(conv, _checklist_for(checklist_id), samples, previous, reuse_answers=not fresh,
                                  precedents=db.get_setting("precedents", []))
    ev["reliability"] = evaluator.reliability(ev)
    return db.save_evaluation(ev)


async def _run_job(job_id: str, conv_ids: list[str], cl: dict, samples: int | None, reuse: bool = True):
    job = JOBS[job_id]

    async def one(cid: str):
        try:
            conv = db.get_conversation(cid)
            previous = db.latest_evaluation(cid) if reuse else None
            ev = await evaluator.evaluate(conv, cl, samples, previous, precedents=db.get_setting("precedents", []))
            ev["reliability"] = evaluator.reliability(ev)
            db.save_evaluation(ev)
            job["cost_usd"] += ev["cost_usd"]
            job["llm_calls"] += ev["llm_calls"]
            job["results"].append({"conversation_id": cid, "before": ev.get("previous_total"),
                                   "after": ev["final_score"]["total"], "ai_after": ev["original_score"]["total"]})
            llm.log(f"✔ {cid}: {ev['final_score']['total']} ({job['done'] + 1}/{job['total']}, {ev['duration_s']} с)")
        except Exception as e:  # noqa: BLE001
            job["failed"].append({"conversation_id": cid, "error": str(e)})
            llm.log(f"✘ {cid}: {str(e)[:300]}")
        finally:
            job["done"] += 1

    # Небольшими группами: каждый диалог сохраняется сразу после оценки, прогресс не теряется.
    # Общий лимит одновременных запросов к LLM держит семафор в llm.py.
    sem = asyncio.Semaphore(config.DIALOG_CONCURRENCY)

    async def limited(cid: str):
        async with sem:
            await one(cid)

    try:
        await asyncio.gather(*[limited(c) for c in conv_ids])
    finally:
        job["status"] = "finished"


@app.post("/api/evaluate")
async def evaluate_batch(req: EvalReq):
    if not llm.is_configured():
        raise HTTPException(400, "LLM не настроена: задайте OPENAI_API_KEY (или OPENAI_BASE_URL) в .env")
    cl = _checklist_for(req.checklist_id)
    ids = req.conversation_ids or [c["conversation_id"] for c in db.list_conversations(req.dataset)]
    if req.only_missing:
        latest = db.latest_evaluations_map()
        ids = [
            i for i in ids
            if not (latest.get(i) and latest[i]["checklist_id"] == cl["id"] and latest[i]["checklist_version"] == cl["version"])
        ]
    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = {"id": job_id, "status": "running", "total": len(ids), "done": 0, "failed": [], "cost_usd": 0.0,
                    "llm_calls": 0, "results": [], "checklist": f"{cl['name']} v{cl['version']}"}
    # Отдельная задача в цикле событий: ответ уходит сразу, опрос /api/jobs не ждёт окончания оценки.
    task = asyncio.create_task(_run_job(job_id, ids, cl, req.samples, req.reuse))
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)
    return JOBS[job_id]


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Задача не найдена")
    return JOBS[job_id]


@app.get("/api/evaluations/{ev_id}")
def evaluation(ev_id: int):
    ev = db.get_evaluation(ev_id)
    if not ev:
        raise HTTPException(404, "Оценка не найдена")
    return ev


# ---------------- апелляции ----------------
class AppealReq(BaseModel):
    evaluation_id: int
    criterion_id: str
    comment: str
    requested_answer: str | None = None
    author: str | None = "Оператор"


class DecisionReq(BaseModel):
    decision: str  # confirm | change
    new_answer: str | None = None
    reason: str
    reviewer: str | None = "Проверяющий"


@app.post("/api/appeals")
def create_appeal(req: AppealReq):
    ev = db.get_evaluation(req.evaluation_id)
    if not ev:
        raise HTTPException(404, "Оценка не найдена")
    item = next((c for c in ev["criteria"] if c["criterion_id"] == req.criterion_id), None)
    if not item:
        raise HTTPException(404, "Критерий не найден в оценке")
    if not req.comment.strip():
        raise HTTPException(400, "Добавьте комментарий")
    if any(a["status"] == "open" and a["criterion_id"] == req.criterion_id and a["evaluation_id"] == ev["id"]
           for a in db.list_appeals(conversation_id=ev["conversation_id"])):
        raise HTTPException(409, "По этому критерию уже есть открытая апелляция")
    data = {
        "comment": req.comment.strip(),
        "author": req.author,
        "requested_answer": req.requested_answer,
        "criterion_name": item["name"],
        "disputed_answer": ev["final_answers"].get(req.criterion_id),
        "disputed_answer_label": item.get("final_answer_label") or item["answer_label"],
        "total_before": ev["final_score"]["total"],
    }
    return db.create_appeal(ev["id"], ev["conversation_id"], req.criterion_id, data)


@app.get("/api/appeals")
def appeals(status: str | None = None):
    out = []
    for a in db.list_appeals(status=status):
        ev = db.get_evaluation(a["evaluation_id"])
        item = next((c for c in ev["criteria"] if c["criterion_id"] == a["criterion_id"]), None) if ev else None
        crit = next((c for c in ev["checklist_snapshot"]["criteria"] if c["id"] == a["criterion_id"]), None) if ev else None
        out.append({**a, "criterion": item, "criterion_def": crit,
                    "allowed_answers": scoring.allowed_answers(crit) if crit else []})
    return out


@app.post("/api/appeals/{appeal_id}/decision")
def decide(appeal_id: int, req: DecisionReq):
    a = db.get_appeal(appeal_id)
    if not a:
        raise HTTPException(404, "Апелляция не найдена")
    if a["status"] != "open":
        raise HTTPException(409, "Апелляция уже рассмотрена")
    ev = db.get_evaluation(a["evaluation_id"])
    current = ev["final_answers"].get(a["criterion_id"])
    new_answer = current if req.decision == "confirm" else req.new_answer
    if new_answer is None:
        raise HTTPException(400, "Выберите новую оценку")
    ev = _apply_review(ev, a["criterion_id"], new_answer, req.reason, req.reviewer, appeal_id)
    return {"appeal": db.get_appeal(appeal_id), "evaluation": ev}


def _apply_review(ev: dict, criterion_id: str, new_answer: str, reason: str, reviewer: str | None,
                  appeal_id: int | None = None) -> dict:
    """Вердикт супервизора: меняет/подтверждает ответ, пересчитывает итог, закрывает открытую апелляцию."""
    if not reason.strip():
        raise HTTPException(400, "Укажите причину решения")
    current = ev["final_answers"].get(criterion_id)
    before = ev["final_score"]["total"]
    try:
        ev = evaluator.apply_override(ev, criterion_id, new_answer)
    except (ValueError, StopIteration) as e:
        raise HTTPException(400, str(e))
    ev.setdefault("decisions", []).append(
        {"appeal_id": appeal_id, "criterion_id": criterion_id, "from": current, "to": new_answer,
         "reason": reason.strip(), "reviewer": reviewer or "Супервизор", "at": db.now(),
         "total_before": before, "total_after": ev["final_score"]["total"]}
    )
    db.update_evaluation(ev)
    if new_answer != current:
        _add_precedent(ev, criterion_id, new_answer, reason)
    for a in db.list_appeals(status="open", conversation_id=ev["conversation_id"]):
        if a["criterion_id"] == criterion_id and (appeal_id is None or a["id"] == appeal_id):
            data = {**a["data"], "decision": "change" if new_answer != current else "confirm", "new_answer": new_answer,
                    "reason": reason.strip(), "reviewer": reviewer, "total_after": ev["final_score"]["total"]}
            db.resolve_appeal(a["id"], "confirmed" if new_answer == current else "changed", data)
    return ev


def _add_precedent(ev: dict, criterion_id: str, decided: str, reason: str) -> None:
    """Решение супервизора становится прецедентом: ИИ учитывает его в следующих оценках других диалогов."""
    item = next((i for i in ev["criteria"] if i["criterion_id"] == criterion_id), {})
    quote = "; ".join(e.get("quote", "") for e in (item.get("evidence") or [])[:2])
    precs = [p for p in db.get_setting("precedents", [])
             if not (p["conversation_id"] == ev["conversation_id"] and p["criterion_id"] == criterion_id)]
    precs.append({"id": uuid.uuid4().hex[:8], "criterion_id": criterion_id, "criterion_name": item.get("name", criterion_id),
                  "conversation_id": ev["conversation_id"], "ai_answer": ev["original_answers"].get(criterion_id),
                  "decided": decided, "reason": reason.strip(), "quote": quote, "at": db.now()})
    db.set_setting("precedents", precs[-200:])


@app.get("/api/precedents")
def precedents():
    return list(reversed(db.get_setting("precedents", [])))


@app.delete("/api/precedents/{pid}")
def delete_precedent(pid: str):
    precs = db.get_setting("precedents", [])
    db.set_setting("precedents", [p for p in precs if p["id"] != pid])
    return {"deleted": pid}


@app.post("/api/evaluations/{ev_id}/coaching")
async def coaching(ev_id: int):
    ev = db.get_evaluation(ev_id)
    if not ev:
        raise HTTPException(404, "Оценка не найдена")
    conv = db.get_conversation(ev["conversation_id"])
    try:
        data, usage = await evaluator.coach(conv, ev)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"Не удалось получить рекомендации: {e}")
    ev["coaching"] = {**data, "at": db.now(), "cost_usd": round(llm.cost_usd(usage), 5),
                      "for_total": ev["final_score"]["total"]}
    db.update_evaluation(ev)
    return ev["coaching"]


class ReviewReq(BaseModel):
    criterion_id: str
    new_answer: str
    reason: str
    reviewer: str | None = "Супервизор"


@app.post("/api/evaluations/{ev_id}/review")
def review(ev_id: int, req: ReviewReq):
    ev = db.get_evaluation(ev_id)
    if not ev:
        raise HTTPException(404, "Оценка не найдена")
    return _apply_review(ev, req.criterion_id, req.new_answer, req.reason, req.reviewer)


# ---------------- грубые ошибки: все диалоги с флагом и объяснением ----------------
@app.get("/api/gross")
def gross(dataset: str | None = None):
    latest = db.latest_evaluations_map()
    rows = []
    for c in db.list_conversations(dataset):
        ev = latest.get(c["conversation_id"])
        if not ev or not ev["final_score"].get("gross_error"):
            continue
        flag = next((x for x in ev["checklist_snapshot"]["criteria"] if x["type"] == "flag"), None)
        if not flag:
            continue
        item = next((i for i in ev["criteria"] if i["criterion_id"] == flag["id"]), {})
        dec = [d for d in ev.get("decisions", []) if d["criterion_id"] == flag["id"]]
        by_sup = bool(dec and dec[-1]["to"] == "yes" and ev["original_answers"].get(flag["id"]) != "yes")
        rows.append({
            "conversation_id": c["conversation_id"], "language": c.get("language"), "channel": c.get("channel"),
            "total": ev["final_score"]["total"], "criterion_id": flag["id"],
            "rationale": dec[-1]["reason"] if by_sup else item.get("rationale", ""),
            "evidence": [{"msg_id": e.get("msg_id"), "quote": e.get("quote", "")} for e in item.get("evidence") or []],
            "by_supervisor": by_sup,
        })
    return rows


# ---------------- экспорт ----------------
@app.get("/api/export")
def export(dataset: str | None = None):
    latest = db.latest_evaluations_map()
    rows = []
    for c in db.list_conversations(dataset):
        ev = latest.get(c["conversation_id"])
        if not ev:
            continue
        rows.append({
            "conversation_id": c["conversation_id"],
            "checklist": f"{ev['checklist_id']} v{ev['checklist_version']}",
            "total": ev["final_score"]["total"],
            "original_total": ev["original_score"]["total"],
            "gross_error": ev["final_score"]["gross_error"],
            "answers": ev["final_answers"],
            "criteria": [{"id": i["criterion_id"], "name": i["name"], "answer": ev["final_answers"].get(i["criterion_id"]),
                          "rationale": i["rationale"], "evidence": [e["msg_id"] for e in i["evidence"]]}
                         for i in ev["criteria"]],
        })
    return rows


TEAM_NAME = __import__("os").getenv("CQA_TEAM_NAME", "CQA Team")


def _results_doc(ds: str | None, use: str) -> tuple[dict, list[str]]:
    convs = db.list_conversations(ds)
    full = [db.get_conversation(c["conversation_id"]) for c in convs]
    return results_export.build(full, db.latest_evaluations_map(), TEAM_NAME, use), [c["conversation_id"] for c in convs]


@app.get("/api/results")
def results(dataset: str | None = None, use: str = "ai"):
    """results.json в формате организаторов (JSON 1.0). use=ai — ответы ИИ, use=final — после апелляций."""
    doc, _ = _results_doc(dataset, use)
    return doc


@app.post("/api/results/save")
def results_save(dataset: str, use: str = "ai"):
    """Записывает results/dev|prod/results.json и проверяет формат."""
    doc, ids = _results_doc(dataset, use)
    if not ids:
        raise HTTPException(404, f"В наборе {dataset} нет диалогов")
    folder = "dev" if dataset.upper() == "DEV" else "prod"
    path = config.ROOT / "results" / folder / "results.json"
    results_export.write(doc, path)
    warnings: list[str] = []
    problems = results_export.validate(doc, ids, 10 if folder == "prod" else None, warnings)
    ok = sum(1 for r in doc["results"] if r["status"] == "ok")
    return {"path": str(path.relative_to(config.ROOT)), "total": len(ids), "ok": ok, "errors_count": len(ids) - ok,
            "problems": problems, "warnings": warnings}


# ---------------- эталонная разметка (для измерения точности) ----------------
GOLD_PATH = config.ROOT / "gold" / "gold.json"


def _load_gold() -> dict:
    if GOLD_PATH.exists():
        return json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    return {}


class GoldReq(BaseModel):
    answers: dict[str, str]


@app.get("/api/review-queue")
def review_queue_api(dataset: str | None = None, threshold: float = 70):
    """Очередь проверки по риску + честный замер по эталону: наша очередь vs случайная выборка vs сортировка по баллу."""
    convs = [db.get_conversation(c["conversation_id"]) for c in db.list_conversations(dataset)]
    latest = db.latest_evaluations_map()
    opens: dict[str, int] = {}
    for a in db.list_appeals(status="open"):
        opens[a["conversation_id"]] = opens.get(a["conversation_id"], 0) + 1
    rows = review_queue.build_queue(convs, latest, opens)
    metrics = review_queue.evaluate_queue(rows, _load_gold(), latest, threshold)
    return {"rows": rows, "metrics": metrics}


@app.get("/api/gold/report")
def gold_report():
    return gold_compare(_load_gold(), db.latest_evaluations_map())


@app.get("/api/gold/{conv_id}")
def gold_get(conv_id: str):
    return _load_gold().get(conv_id) or {"answers": {}}


@app.put("/api/gold/{conv_id}")
def gold_put(conv_id: str, req: GoldReq):
    gold = _load_gold()
    gold[conv_id] = {"answers": req.answers, "updated_at": db.now()}
    GOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLD_PATH.write_text(json.dumps(gold, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return gold[conv_id]


def gold_compare(gold: dict, latest: dict) -> dict:
    """Сравнивает исходные ответы ИИ (до апелляций) с эталоном."""
    per: dict[str, dict] = {}
    total_err, total_n, match_all, n_all, convs = 0.0, 0, 0, 0, 0
    for cid, g in gold.items():
        ev = latest.get(cid)
        if not ev or not g.get("answers"):
            continue
        convs += 1
        cl = ev["checklist_snapshot"]
        for c in cl["criteria"]:
            if c["id"] not in g["answers"]:
                continue
            pred, true = ev["original_answers"].get(c["id"]), g["answers"][c["id"]]
            r = per.setdefault(c["id"], {"id": c["id"], "name": c["name"], "n": 0, "match": 0, "mismatches": []})
            r["n"] += 1
            n_all += 1
            if pred == true:
                r["match"] += 1
                match_all += 1
            else:
                r["mismatches"].append({"conversation_id": cid, "pred": pred, "gold": true})
        if all(c["id"] in g["answers"] for c in cl["criteria"]):
            gt = scoring.compute(cl, g["answers"])["total"]
            pt = ev["original_score"]["total"]
            if gt is not None and pt is not None:
                total_err += abs(gt - pt)
                total_n += 1
    rows = list(per.values())
    for r in rows:
        r["accuracy"] = r["match"] / r["n"] if r["n"] else None
    return {
        "n": convs,
        "answer_accuracy": match_all / n_all if n_all else None,
        "total_mae": round(total_err / total_n, 2) if total_n else None,
        "per_criterion": rows,
    }


# ---------------- фронтенд ----------------
app.mount("/static", StaticFiles(directory=config.FRONTEND_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(config.FRONTEND_DIR / "index.html")
