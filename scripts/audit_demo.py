"""Демонстрация журнала для регулятора: «подделать» итог оценки напрямую в базе и откатить обратно.

    py scripts/audit_demo.py tamper p1-ru-002     # подменить итог на 100 в обход системы
    py scripts/audit_demo.py restore              # вернуть исходные данные

После tamper откройте «Аналитика → Журнал для регулятора → Проверить целостность»: система покажет,
какая запись изменена после фиксации. После restore проверка снова зелёная.
"""
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend import config  # noqa: E402

BACKUP = config.ROOT / "data" / ".audit_demo_backup.json"


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    c = sqlite3.connect(config.DB_PATH)
    if cmd == "tamper":
        conv = sys.argv[2] if len(sys.argv) > 2 else "p1-ru-002"
        row = c.execute("SELECT id, data FROM evaluations WHERE conversation_id=? ORDER BY id DESC LIMIT 1", (conv,)).fetchone()
        if not row:
            print("Нет оценки для", conv)
            return
        BACKUP.write_text(json.dumps({"id": row[0], "data": row[1]}, ensure_ascii=False), encoding="utf-8")
        d = json.loads(row[1])
        was = d["final_score"]["total"]
        d["final_score"]["total"] = 100.0
        c.execute("UPDATE evaluations SET data=? WHERE id=?", (json.dumps(d, ensure_ascii=False), row[0]))
        c.commit()
        print(f"Итог {conv} подменён в базе: {was} → 100 (оценка #{row[0]}). Проверьте целостность журнала.")
    elif cmd == "restore":
        if not BACKUP.exists():
            print("Нечего восстанавливать")
            return
        b = json.loads(BACKUP.read_text(encoding="utf-8"))
        c.execute("UPDATE evaluations SET data=? WHERE id=?", (b["data"], b["id"]))
        c.commit()
        BACKUP.unlink()
        print(f"Оценка #{b['id']} восстановлена. Проверка целостности снова пройдёт.")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
