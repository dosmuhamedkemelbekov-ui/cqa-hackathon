"""Быстрая проверка ключа и модели из .env одним маленьким запросом.

    py scripts/check_llm.py

Выводит «OK» и ответ модели или полный текст ошибки провайдера (ключ, имя модели, лимит).
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend import config, llm  # noqa: E402

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer"],
    "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}},
}


async def main() -> int:
    info = llm.provider_info()
    print(f"Провайдер: {info['provider']} · модель: {info['model']} · адрес: {info['base_url']}")
    if not llm.is_configured():
        print("Ключ не задан: проверьте .env (OPENAI_API_KEY или ANTHROPIC_API_KEY).")
        return 1
    t0 = time.time()
    try:
        data, usage = await llm.complete_json(
            "Ответь строго JSON по схеме.", "Оператор поздоровался? Реплика: «Здравствуйте, меня зовут Алина». "
            "answer: yes или no.", SCHEMA, name="check")
    except Exception as e:  # noqa: BLE001
        print(f"ОШИБКА через {time.time() - t0:.1f} с:\n{e}")
        return 1
    print(f"OK за {time.time() - t0:.1f} с: {data} · токены {usage}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
