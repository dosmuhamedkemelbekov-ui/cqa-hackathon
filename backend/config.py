"""Настройки приложения. Читаются из переменных окружения или файла .env в корне проекта."""
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    values: dict[str, str] = {}
    for line in env.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if value[:1] in ('"', "'") and value[:1] in value[1:]:
            value = value[1:value.index(value[0], 1)]       # значение в кавычках
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()  # «значение   # комментарий»
        values[key.strip()] = value  # при повторе ключа в .env побеждает последняя строка
    for key, value in values.items():
        os.environ.setdefault(key, value)  # переменные окружения процесса важнее .env


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
LLM_CONCURRENCY = int(os.getenv("CQA_CONCURRENCY", "16"))
# Сколько диалогов оценивается одновременно в пакете (каждый сохраняется сразу после оценки).
DIALOG_CONCURRENCY = int(os.getenv("CQA_DIALOG_CONCURRENCY", "4"))
# Сколько независимых прогонов на критерий (голосование большинством / медиана). 1 — быстрее и дешевле.
LLM_SAMPLES = int(os.getenv("CQA_SAMPLES", "1"))
# Сколько раз ждать и повторять при поминутном лимите (429) бесплатных тарифов.
LLM_RATE_RETRIES = int(os.getenv("CQA_RATE_RETRIES", "8"))
# Самопроверка: рискованные ответы (Нет, N/A, неполный балл, грубая ошибка, низкая уверенность)
# перепроверяет независимый «контролёр» и исправляет ошибку. 0 — выключить.
LLM_VERIFY = os.getenv("CQA_VERIFY", "1") not in ("0", "false", "no")
# Сколько последних прецедентов (решений супервизора) по критерию подмешивать в промпт.
PRECEDENTS_PER_CRITERION = int(os.getenv("CQA_PRECEDENTS", "6"))
# Распознавание речи для загруженных аудиозаписей (OpenAI Whisper API или совместимый сервис).
# По умолчанию — тот же ключ и адрес, что у LLM.
STT_MODEL = os.getenv("CQA_STT_MODEL", "whisper-1")
STT_API_KEY = os.getenv("CQA_STT_API_KEY") or LLM_API_KEY
STT_BASE_URL = os.getenv("CQA_STT_BASE_URL") or LLM_BASE_URL
# ---- Режим оценки ----
# per_criterion — отдельный вызов на каждый критерий (точнее, 8+ вызовов на диалог);
# single — все критерии в одном вызове (1 вызов на диалог) — для бесплатных тарифов с дневным лимитом.
EVAL_MODE = os.getenv("CQA_EVAL_MODE", "per_criterion").strip().lower()
# Ограничение запросов в минуту к LLM (0 — без ограничения). Для бесплатных тарифов: 10–15.
LLM_RPM = float(os.getenv("CQA_RPM", "0"))
# ---- Провайдер LLM: "openai" (любой OpenAI-совместимый API) или "anthropic" (Claude) ----
LLM_PROVIDER = os.getenv("CQA_PROVIDER", "openai").strip().lower()
ANTHROPIC_API_KEY = os.getenv("CQA_ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY", "")
# Адрес API Anthropic задаётся явно: переменная окружения ANTHROPIC_BASE_URL может принадлежать другим программам.
ANTHROPIC_BASE_URL = os.getenv("CQA_ANTHROPIC_BASE_URL", "https://api.anthropic.com")
CLAUDE_MODEL = os.getenv("CQA_CLAUDE_MODEL", "claude-opus-5-5")
# Глубина рассуждения Claude: low | medium | high | xhigh | max
CLAUDE_EFFORT = os.getenv("CQA_CLAUDE_EFFORT", "medium")
if LLM_PROVIDER == "anthropic":
    LLM_MODEL = CLAUDE_MODEL

# Цены Claude за 1M токенов (USD): вход, чтение кэша, запись кэша, выход.
CLAUDE_PRICES = {
    "claude-fable-5-1": (10.0, 0.25, 12.5, 50.0),
    "claude-opus-5-5": (4.0, 0.20, 5.0, 20.0),
    "claude-opus-5": (5.0, 0.50, 6.25, 25.0),
    "claude-sonnet-5-5": (2.0, 0.20, 2.5, 10.0),
    "claude-haiku-5-5": (0.10, 0.01, 0.125, 0.50),
}

# Цена за 1M токенов (USD) для OpenAI-совместимого провайдера. Задайте под свою модель.
PRICE_INPUT_PER_M = float(os.getenv("CQA_PRICE_INPUT_PER_M", "2.0"))
PRICE_CACHED_INPUT_PER_M = float(os.getenv("CQA_PRICE_CACHED_INPUT_PER_M", "0.5"))
PRICE_OUTPUT_PER_M = float(os.getenv("CQA_PRICE_OUTPUT_PER_M", "8.0"))
