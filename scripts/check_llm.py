"""Быстрая проверка ключа и модели из .env одним маленьким запросом.

    py scripts/check_llm.py            # проверить модель из .env
    py scripts/check_llm.py --models   # список моделей, доступных ключу

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
    if "--models" in sys.argv or config.LLM_PROVIDER != "anthropic":
        try:
            ids = sorted([m.id.removeprefix("models/") async for m in llm.client().models.list()])
        except Exception as e:  # noqa: BLE001
            ids = []
            print(f"Список моделей получить не удалось: {str(e)[:200]}")
        if "--models" in sys.argv:
            print("Доступные модели:")
            for i in ids:
                print("  ", i)
            return 0
        if ids and config.LLM_MODEL not in ids:
            near = [i for i in ids if "flash" in i or "lite" in i or "mini" in i][:15]
            print(f"Модели «{config.LLM_MODEL}» нет у провайдера. Похожие доступные:")
            for i in near:
                print("  ", i)
            print("Впишите одну из них в CQA_MODEL в .env (полный список: py scripts/check_llm.py --models).")
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
