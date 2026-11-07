"""Проверка подключения к LLM из .env: список моделей и пробный запрос.

    python scripts/check_llm.py            # список моделей + пробный вызов
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend import config, llm  # noqa: E402


async def main() -> None:
    info = llm.provider_info()
    print("Провайдер:", info)
    if not llm.is_configured():
        print("Ключ не задан — заполните .env")
        return
    if config.LLM_PROVIDER == "openai":
        try:
            models = [m.id async for m in llm.client().models.list()]
            print(f"Доступно моделей: {len(models)}")
            for m in sorted(models):
                print("  ", m)
        except Exception as e:  # noqa: BLE001
            print("Список моделей недоступен:", e)
    schema = {"type": "object", "additionalProperties": False, "required": ["answer"],
              "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}}}
    try:
        data, usage = await llm.complete_json("Отвечай строго по схеме.", "Сәлеметсіз бе — это приветствие? answer: yes/no",
                                              schema, name="check")
        print("Пробный вызов OK:", data, usage)
    except Exception as e:  # noqa: BLE001
        print("Пробный вызов НЕ удался:", e)


if __name__ == "__main__":
    asyncio.run(main())
