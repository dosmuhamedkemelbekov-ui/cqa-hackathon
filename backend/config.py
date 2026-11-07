"""Настройки приложения. Читаются из переменных окружения или файла .env в корне проекта."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

DB_PATH = Path(os.getenv("CQA_DB_PATH", ROOT / "data" / "cqa.sqlite3"))
DEFAULT_CHECKLIST = ROOT / "checklists" / "default.json"
FRONTEND_DIR = ROOT / "frontend"

# API организаторов. group_id подставляется отдельно.
DATASET_API_URL = os.getenv(
    "CQA_DATASET_API_URL",
    "https://ec88f21f1accecf38e20095a9fe175.a5.environment.api.powerplatform.com:443/powerautomate/"
    "automations/direct/cu/11/workflows/b3d2a68006a24662be2b721454cb4e00/triggers/manual/paths/invoke"
    "?api-version=1&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=5Sog2MThgCRTlk3HObv70LRmRk6u7IOwzlwMpe804qo",
)

# LLM: любой OpenAI-совместимый API (OpenAI, Azure-совместимые прокси, Ollama, LM Studio, vLLM).
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_BASE_URL = os.getenv("OPENAI_BASE_URL") or None  # напр. http://localhost:11434/v1 для Ollama
LLM_MODEL = os.getenv("CQA_MODEL", "gpt-4.1")
LLM_CONCURRENCY = int(os.getenv("CQA_CONCURRENCY", "8"))
# Сколько независимых прогонов на критерий (голосование большинством / медиана). 1 — быстрее и дешевле.
LLM_SAMPLES = int(os.getenv("CQA_SAMPLES", "1"))
# Цена за 1M токенов (USD) — для расчёта стоимости одной оценки. Задайте под свою модель.
PRICE_INPUT_PER_M = float(os.getenv("CQA_PRICE_INPUT_PER_M", "2.0"))
PRICE_CACHED_INPUT_PER_M = float(os.getenv("CQA_PRICE_CACHED_INPUT_PER_M", "0.5"))
PRICE_OUTPUT_PER_M = float(os.getenv("CQA_PRICE_OUTPUT_PER_M", "8.0"))
