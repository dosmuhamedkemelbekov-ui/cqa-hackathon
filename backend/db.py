"""Хранилище на SQLite. Сущности хранятся как JSON-документы — так проще менять схему во время хакатона."""
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
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


_conn = _connect()


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
            """
        )
        _conn.commit()
    if not list_checklists():
        cl = json.loads(DEFAULT_CHECKLIST.read_text(encoding="utf-8"))
        save_checklist(cl)
        set_setting("active_checklist", cl["id"])


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
        _conn.execute(
            "INSERT INTO checklists(id, version, data, created_at) VALUES(?,?,?,?)",
            (cl["id"], version, json.dumps(cl, ensure_ascii=False), cl["saved_at"]),
        )
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
        _conn.execute("UPDATE evaluations SET data=? WHERE id=?", (json.dumps(ev, ensure_ascii=False), ev["id"]))
        _conn.commit()
    return ev


def update_evaluation(ev: dict) -> None:
    with _lock:
        _conn.execute("UPDATE evaluations SET data=? WHERE id=?", (json.dumps(ev, ensure_ascii=False), ev["id"]))
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
        "SELECT id, checklist_id, checklist_version, created_at FROM evaluations WHERE conversation_id=? ORDER BY id DESC",
        (conv_id,),
    )
    return [dict(r) for r in rows]


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
        _conn.commit()
    return get_appeal(appeal_id)
