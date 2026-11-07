"""Загружает набор в базу без запуска сервера: python scripts/load_dev.py [group_id|путь_к_json]"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend import dataset, db  # noqa: E402


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else "DEV"
    db.init()
    p = Path(arg)
    if p.exists():
        ds, records = dataset.parse_payload(json.loads(p.read_text(encoding="utf-8-sig")))
        ds = ds or p.stem
    else:
        ds, records = asyncio.run(dataset.fetch_group(arg))
    print(f"{ds}: загружено {db.upsert_conversations(records, ds)} диалогов")


if __name__ == "__main__":
    main()
