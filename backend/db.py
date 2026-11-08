"""Хранилище на SQLite. Сущности хранятся как JSON-документы — так проще менять схему во время хакатона."""
import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

from .config import DB_PATH, DEFAULT_CHECKLIST

_lock = threading.Lock()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


# Своё подключение на каждый поток: FastAPI выполняет запросы параллельно в пуле потоков,
# а одно общее подключение SQLite при одновременном чтении путает результаты (сервер падал с 500).
_local = threading.local()


class _ConnProxy:
    def __getattr__(self, name):
        c = getattr(_local, "conn", None)
        if c is None:
            c = _local.conn = _connect()
        return getattr(c, name)


_conn = _ConnProxy()
_wal = _connect()
try:
    _wal.execute("PRAGMA journal_mode=WAL")  # чтение не блокируется записью
except sqlite3.DatabaseError:
    pass
_wal.close()


def init() -> None:
    with _lock:
        _conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY, dataset TEXT, data TEXT, created_at TEXT);
            CREATE TABLE IF NOT EXISTS checklists (
                pk INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT, version INTEGER,
                data TEXT, created_at TEXT, UNIQUE(id, version));
            CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT,
                checklist_id TEXT, checklist_version INTEGER, data TEXT, created_at TEXT);
            CREATE TABLE IF NOT EXISTS appeals (
                id INTEGER PRIMARY KEY AUTOINCREMENT, evaluation_id INTEGER,
                conversation_id TEXT, criterion_id TEXT, status TEXT, data TEXT,
                created_at TEXT, resolved_at TEXT);
            CREATE INDEX IF NOT EXISTS ix_eval_conv ON evaluations(conversation_id);
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, kind TEXT, ref TEXT,
                digest TEXT, payload TEXT, prev_hash TEXT, hash TEXT);
            CREATE INDEX IF NOT EXISTS ix_audit_ref ON audit_log(ref);
            """
        )
        _conn.commit()
        if not _conn.execute("SELECT 1 FROM audit_log LIMIT 1").fetchone():
            _audit_baseline()
    if not list_checklists():
        cl = json.loads(DEFAULT_CHECKLIST.read_text(encoding="utf-8"))
        save_checklist({**cl, "source": "default"})
        set_setting("active_checklist", cl["id"])
    else:
        _upgrade_default()


_META = ("version", "saved_at", "active", "versions", "source")


def _upgrade_default() -> None:
    """Обновлённый регламент из checklists/default.json применяется к существующей базе автоматически —
    но только если чек-лист не правили вручную (последняя версия — исходная)."""
    cl = json.loads(DEFAULT_CHECKLIST.read_text(encoding="utf-8"))
    latest = get_checklist(cl["id"])
    if not latest or latest.get("source") == "user":
        return
    strip = lambda c: {k: v for k, v in c.items() if k not in _META}  # noqa: E731
    if strip(latest) == strip(cl):
        return
    if latest.get("source") == "default" or latest.get("version") == 1:
        save_checklist({**cl, "source": "default"})


# ---------- settings ----------
def get_setting(key: str, default: Any = None) -> Any:
    row = _conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return json.loads(row["value"]) if row else default


def set_setting(key: str, value: Any) -> None:
    with _lock:
        _conn.execute(
            "INSERT INTO settings(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value, ensure_ascii=False)),
        )
        _conn.commit()


# ---------- conversations ----------
def upsert_conversations(records: list[dict], dataset: str) -> int:
    with _lock:
        for r in records:
            _conn.execute(
                "INSERT INTO conversations(id, dataset, data, created_at) VALUES(?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET dataset=excluded.dataset, data=excluded.data",
                (r["conversation_id"], dataset, json.dumps(r, ensure_ascii=False), now()),
            )
        _conn.commit()
    return len(records)


def get_conversation(conv_id: str) -> dict | None:
    row = _conn.execute("SELECT * FROM conversations WHERE id=?", (conv_id,)).fetchone()
    if not row:
        return None
    d = json.loads(row["data"])
    d["dataset"] = row["dataset"]
    return d


def list_conversations(dataset: str | None = None) -> list[dict]:
    q = "SELECT id, dataset, data FROM conversations"
    args: tuple = ()
    if dataset:
        q += " WHERE dataset=?"
        args = (dataset,)
    out = []
    for row in _conn.execute(q + " ORDER BY rowid", args):
        d = json.loads(row["data"])
        out.append(
            {
                "conversation_id": row["id"],
                "dataset": row["dataset"],
                "channel": d.get("channel"),
                "language": d.get("language"),
                "messages": len(d.get("transcript") or []),
                "has_audio": bool(d.get("audio_url")),
            }
        )
    return out


def list_datasets() -> list[str]:
    return [r["dataset"] for r in _conn.execute("SELECT DISTINCT dataset FROM conversations ORDER BY dataset")]


# ---------- checklists ----------
def save_checklist(cl: dict) -> dict:
    with _lock:
        row = _conn.execute("SELECT MAX(version) v FROM checklists WHERE id=?", (cl["id"],)).fetchone()
        version = (row["v"] or 0) + 1
        cl = {**cl, "version": version, "saved_at": now()}
        text = json.dumps(cl, ensure_ascii=False)
        _conn.execute(
            "INSERT INTO checklists(id, version, data, created_at) VALUES(?,?,?,?)",
            (cl["id"], version, text, cl["saved_at"]),
        )
        _audit("checklist_saved", f"checklist:{cl['id']}:v{version}", text,
               {"version": version, "criteria": len(cl.get("criteria", [])), "source": cl.get("source")})
        _conn.commit()
    return cl


def get_checklist(cl_id: str, version: int | None = None) -> dict | None:
    if version is None:
        row = _conn.execute(
            "SELECT data FROM checklists WHERE id=? ORDER BY version DESC LIMIT 1", (cl_id,)
        ).fetchone()
    else:
        row = _conn.execute("SELECT data FROM checklists WHERE id=? AND version=?", (cl_id, version)).fetchone()
    return json.loads(row["data"]) if row else None


def list_checklists() -> list[dict]:
    rows = _conn.execute(
        "SELECT c.data FROM checklists c JOIN (SELECT id, MAX(version) v FROM checklists GROUP BY id) m "
        "ON c.id=m.id AND c.version=m.v ORDER BY c.pk"
    ).fetchall()
    return [json.loads(r["data"]) for r in rows]


def checklist_versions(cl_id: str) -> list[dict]:
    rows = _conn.execute("SELECT version, created_at FROM checklists WHERE id=? ORDER BY version", (cl_id,))
    return [dict(r) for r in rows]


def active_checklist() -> dict:
    cl_id = get_setting("active_checklist")
    cl = get_checklist(cl_id) if cl_id else None
    return cl or list_checklists()[0]


# ---------- evaluations ----------
def save_evaluation(ev: dict) -> dict:
    with _lock:
        cur = _conn.execute(
            "INSERT INTO evaluations(conversation_id, checklist_id, checklist_version, data, created_at) "
            "VALUES(?,?,?,?,?)",
            (ev["conversation_id"], ev["checklist_id"], ev["checklist_version"],
             json.dumps(ev, ensure_ascii=False), now()),
        )
        _conn.commit()
        ev = {**ev, "id": cur.lastrowid}
        text = json.dumps(ev, ensure_ascii=False)
        _conn.execute("UPDATE evaluations SET data=? WHERE id=?", (text, ev["id"]))
        _audit("evaluation_saved", f"evaluation:{ev['id']}", text, _ev_summary(ev))
        _conn.commit()
    return ev


def update_evaluation(ev: dict) -> None:
    with _lock:
        text = json.dumps(ev, ensure_ascii=False)
        _conn.execute("UPDATE evaluations SET data=? WHERE id=?", (text, ev["id"]))
        last = (ev.get("decisions") or [None])[-1]
        _audit("evaluation_updated", f"evaluation:{ev['id']}", text,
               {**_ev_summary(ev), "decision": last})
        _conn.commit()


def get_evaluation(ev_id: int) -> dict | None:
    row = _conn.execute("SELECT data FROM evaluations WHERE id=?", (ev_id,)).fetchone()
    return json.loads(row["data"]) if row else None


def latest_evaluation(conv_id: str, checklist_id: str | None = None) -> dict | None:
    q = "SELECT data FROM evaluations WHERE conversation_id=?"
    args: list = [conv_id]
    if checklist_id:
        q += " AND checklist_id=?"
        args.append(checklist_id)
    row = _conn.execute(q + " ORDER BY id DESC LIMIT 1", args).fetchone()
    return json.loads(row["data"]) if row else None


def evaluation_history(conv_id: str) -> list[dict]:
    rows = _conn.execute(
        "SELECT id, checklist_id, checklist_version, created_at, data FROM evaluations WHERE conversation_id=? ORDER BY id DESC",
        (conv_id,),
    )
    out = []
    for r in rows:
        d = json.loads(r["data"])
        out.append({
            "id": r["id"], "checklist_id": r["checklist_id"], "checklist_version": r["checklist_version"],
            "created_at": r["created_at"], "ai_total": d["original_score"]["total"],
            "final_total": d["final_score"]["total"], "model": d.get("model"),
            "llm_calls": d.get("llm_calls"), "reused": d.get("reused_criteria", []),
            "previous_total": d.get("previous_total"),
        })
    return out


def latest_evaluations_map(checklist_id: str | None = None) -> dict[str, dict]:
    q = "SELECT conversation_id, data FROM evaluations"
    args: tuple = ()
    if checklist_id:
        q += " WHERE checklist_id=?"
        args = (checklist_id,)
    out: dict[str, dict] = {}
    for row in _conn.execute(q + " ORDER BY id", args):
        out[row["conversation_id"]] = json.loads(row["data"])
    return out


# ---------- appeals ----------
def create_appeal(evaluation_id: int, conversation_id: str, criterion_id: str, data: dict) -> dict:
    with _lock:
        cur = _conn.execute(
            "INSERT INTO appeals(evaluation_id, conversation_id, criterion_id, status, data, created_at) "
            "VALUES(?,?,?,?,?,?)",
            (evaluation_id, conversation_id, criterion_id, "open", json.dumps(data, ensure_ascii=False), now()),
        )
        _audit("appeal_created", f"appeal:{cur.lastrowid}", _appeal_text(cur.lastrowid),
               {"conversation_id": conversation_id, "criterion_id": criterion_id, "evaluation_id": evaluation_id})
        _conn.commit()
    return get_appeal(cur.lastrowid)


def _appeal_row(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["data"] = json.loads(d["data"])
    return d


def get_appeal(appeal_id: int) -> dict | None:
    row = _conn.execute("SELECT * FROM appeals WHERE id=?", (appeal_id,)).fetchone()
    return _appeal_row(row) if row else None


def list_appeals(status: str | None = None, conversation_id: str | None = None) -> list[dict]:
    q, args = "SELECT * FROM appeals WHERE 1=1", []
    if status:
        q += " AND status=?"
        args.append(status)
    if conversation_id:
        q += " AND conversation_id=?"
        args.append(conversation_id)
    return [_appeal_row(r) for r in _conn.execute(q + " ORDER BY id DESC", args)]


def resolve_appeal(appeal_id: int, status: str, data: dict) -> dict:
    with _lock:
        _conn.execute(
            "UPDATE appeals SET status=?, data=?, resolved_at=? WHERE id=?",
            (status, json.dumps(data, ensure_ascii=False), now(), appeal_id),
        )
        _audit("appeal_resolved", f"appeal:{appeal_id}", _appeal_text(appeal_id),
               {"status": status, "new_answer": data.get("new_answer"), "reviewer": data.get("reviewer")})
        _conn.commit()
    return get_appeal(appeal_id)


# ---------- журнал для регулятора: цепочка хэшей ----------
# Каждое событие хранит отпечаток (SHA-256) содержимого записи и хэш, сцепленный с предыдущим событием.
# Изменить оценку/апелляцию/чек-лист в обход системы или подменить запись журнала незаметно нельзя:
# verify_audit() пересчитывает цепочку и сверяет текущие данные с последним зафиксированным отпечатком.
GENESIS = "0" * 64


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _chain_hash(prev: str, ts: str, kind: str, ref: str, digest: str, payload: str) -> str:
    return _sha("|".join([prev, ts, kind, ref, digest, payload]))


def _ev_summary(ev: dict) -> dict:
    return {"conversation_id": ev.get("conversation_id"), "checklist_version": ev.get("checklist_version"),
            "model": ev.get("model"), "ai_total": (ev.get("original_score") or {}).get("total"),
            "final_total": (ev.get("final_score") or {}).get("total")}


def _appeal_text(appeal_id: int) -> str:
    row = _conn.execute("SELECT status, data FROM appeals WHERE id=?", (appeal_id,)).fetchone()
    return (row["status"] or "") + "|" + (row["data"] or "") if row else ""


def _audit(kind: str, ref: str, record_text: str, payload: dict) -> None:
    """Вызывать под _lock, до commit — запись и событие журнала попадают в базу одной транзакцией."""
    prev = _conn.execute("SELECT hash FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()
    prev_hash = prev["hash"] if prev else GENESIS
    ts, digest = now(), _sha(record_text)
    ptext = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    _conn.execute("INSERT INTO audit_log(ts, kind, ref, digest, payload, prev_hash, hash) VALUES(?,?,?,?,?,?,?)",
                  (ts, kind, ref, digest, ptext, prev_hash, _chain_hash(prev_hash, ts, kind, ref, digest, ptext)))


def _audit_baseline() -> None:
    """Первый запуск журнала: фиксируем всё, что уже есть в базе, как точку отсчёта."""
    for r in _conn.execute("SELECT id, version, data FROM checklists ORDER BY pk").fetchall():
        d = json.loads(r["data"])
        _audit("baseline", f"checklist:{d['id']}:v{r['version']}", r["data"], {"version": r["version"]})
    for r in _conn.execute("SELECT id, data FROM evaluations ORDER BY id").fetchall():
        _audit("baseline", f"evaluation:{r['id']}", r["data"], _ev_summary(json.loads(r["data"])))
    for r in _conn.execute("SELECT id FROM appeals ORDER BY id").fetchall():
        _audit("baseline", f"appeal:{r['id']}", _appeal_text(r["id"]), {})
    _conn.commit()


def list_audit(limit: int = 30) -> list[dict]:
    rows = _conn.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["payload"] = json.loads(d["payload"] or "{}")
        out.append(d)
    return out


def verify_audit() -> dict:
    """Пересчитать цепочку и сверить текущие записи с последним зафиксированным отпечатком."""
    rows = _conn.execute("SELECT * FROM audit_log ORDER BY id").fetchall()
    prev, broken = GENESIS, None
    last_digest: dict[str, tuple[str, int]] = {}
    for r in rows:
        if r["prev_hash"] != prev or _chain_hash(prev, r["ts"], r["kind"], r["ref"], r["digest"], r["payload"]) != r["hash"]:
            broken = {"id": r["id"], "ts": r["ts"], "ref": r["ref"]}
            break
        prev = r["hash"]
        last_digest[r["ref"]] = (r["digest"], r["id"])
    tampered, unlogged = [], []
    current: dict[str, str] = {}
    for r in _conn.execute("SELECT id, data FROM evaluations"):
        current[f"evaluation:{r['id']}"] = r["data"]
    for r in _conn.execute("SELECT id FROM appeals"):
        current[f"appeal:{r['id']}"] = _appeal_text(r["id"])
    for r in _conn.execute("SELECT version, data FROM checklists"):
        current[f"checklist:{json.loads(r['data'])['id']}:v{r['version']}"] = r["data"]
    for ref, text in current.items():
        if ref not in last_digest:
            unlogged.append(ref)
        elif _sha(text) != last_digest[ref][0]:
            tampered.append({"ref": ref, "logged_event": last_digest[ref][1]})
    missing = [ref for ref in last_digest if ref not in current]
    ok = broken is None and not tampered and not unlogged and not missing
    return {"ok": ok, "entries": len(rows), "head_hash": prev, "chain_broken_at": broken,
            "tampered": tampered, "unlogged": unlogged, "deleted": missing, "records_checked": len(current),
            "checked_at": now()}
