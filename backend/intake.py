"""Новый диалог вручную: текстовый транскрипт или аудиозапись (Whisper → роли говорящих через LLM)."""
import re
import uuid
from pathlib import Path

from openai import AsyncOpenAI

from . import config, llm

UPLOAD_DIR = config.ROOT / "data" / "uploads"
AUDIO_EXT = {"mp3", "wav", "m4a", "ogg", "oga", "mp4", "mpeg", "mpga", "webm", "flac"}
MAX_AUDIO = 25 * 1024 * 1024  # лимит Whisper API
LANG_CODE = {"RU": "ru", "KZ": "kk", "EN": "en"}
LANG_FROM_WHISPER = {"russian": "RU", "ru": "RU", "kazakh": "KZ", "kk": "KZ", "english": "EN", "en": "EN"}
SPEAKERS = ("operator", "client", "bot", "system")
UPLOAD_PREFIX = "upload:"

_stt: AsyncOpenAI | None = None


def stt_client() -> AsyncOpenAI:
    global _stt
    if _stt is None:
        _stt = AsyncOpenAI(api_key=config.STT_API_KEY or "local", base_url=config.STT_BASE_URL, max_retries=2)
    return _stt


def stt_configured() -> bool:
    return bool(config.STT_API_KEY or config.STT_BASE_URL)


def fmt_ts(sec: float) -> str:
    t = max(0, int(sec or 0))
    return f"{t // 60:02d}:{t % 60:02d}"


# ---------------- аудио ----------------
def save_upload(data: bytes, filename: str) -> str:
    ext = (filename.rsplit(".", 1)[-1] if "." in filename else "").lower()
    if ext not in AUDIO_EXT:
        raise ValueError(f"Неподдерживаемый формат «{filename}». Нужна аудиозапись: mp3, wav, m4a или ogg")
    if not data:
        raise ValueError("Файл пустой")
    if len(data) > MAX_AUDIO:
        raise ValueError(f"Файл весит {len(data) / 1048576:.1f} МБ — лимит Whisper API 25 МБ")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4().hex[:12]}.{ext}"
    (UPLOAD_DIR / name).write_bytes(data)
    return name


def upload_path(audio_url: str) -> Path | None:
    """audio_url вида upload:<файл> → путь к загруженной записи (без выхода за пределы каталога)."""
    if not (audio_url or "").startswith(UPLOAD_PREFIX):
        return None
    name = audio_url[len(UPLOAD_PREFIX):]
    if not re.fullmatch(r"[0-9a-f]{12}\.[a-z0-9]{2,5}", name):
        return None
    p = UPLOAD_DIR / name
    return p if p.exists() else None


async def transcribe(path: Path, language: str | None) -> dict:
    """Whisper: сегменты с таймкодами. Голоса Whisper не разделяет — роли размечаются отдельно."""
    kwargs: dict = {"model": config.STT_MODEL, "response_format": "verbose_json",
                    "timestamp_granularities": ["segment"], "prompt": "Freedom Broker"}
    if language in LANG_CODE:
        kwargs["language"] = LANG_CODE[language]
    with path.open("rb") as f:
        resp = await stt_client().audio.transcriptions.create(file=(path.name, f.read()), **kwargs)
    segs = [{"start": float(getattr(s, "start", 0) or 0), "end": float(getattr(s, "end", 0) or 0),
             "text": (getattr(s, "text", "") or "").strip()} for s in (getattr(resp, "segments", None) or [])]
    segs = [s for s in segs if s["text"]]
    duration = float(getattr(resp, "duration", 0) or 0)
    if not segs and (getattr(resp, "text", "") or "").strip():
        segs = [{"start": 0.0, "end": duration, "text": resp.text.strip()}]
    if not segs:
        raise ValueError("В записи не распознана речь (тишина, шум или неподдерживаемый язык)")
    return {"segments": segs, "language": getattr(resp, "language", "") or "", "duration": duration or segs[-1]["end"]}


def heuristic_roles(segs: list[dict]) -> list[str]:
    """Запасная разметка: смена говорящего на паузе ≥ 0.7 с, первым говорит оператор."""
    out: list[str] = []
    for i, s in enumerate(segs):
        if i == 0:
            out.append("operator")
        elif s["start"] - segs[i - 1]["end"] >= 0.7:
            out.append("client" if out[-1] == "operator" else "operator")
        else:
            out.append(out[-1])
    return out


async def diarize(segs: list[dict], lang: str, operator: str | None) -> tuple[list[str], int]:
    """Роли говорящих по смыслу и порядку реплик. Возвращает (роли, сколько исправлено по чередованию)."""
    n = len(segs)
    op_hint = f" (оператора зовут «{operator}»)" if operator else ""
    system = (
        f"Ты размечаешь роли говорящих в автоматически распознанном транскрипте разговора клиента с оператором "
        f"брокерской компании Freedom Broker (язык: {lang}). Речь распознана без разделения по голосам, поэтому "
        f"определи роль по смыслу и порядку реплик.\n\n"
        f"Роли: \"operator\" — сотрудник Freedom Broker{op_hint}: здоровается, называет компанию и своё имя, "
        f"консультирует, задаёт уточняющие и проверочные вопросы; \"client\" — клиент, обратившийся с вопросом; "
        f"\"bot\" — автоответчик, IVR или виртуальный помощник; \"system\" — технические объявления "
        f"(например, «звонок записывается»).\n\n"
        f"Верни массив speakers ровно из {n} элементов — по одному на каждый сегмент, в том же порядке."
    )
    user = "Сегменты (формат «номер | время | текст»):\n" + "\n".join(
        f"{i} | {fmt_ts(s['start'])} | {s['text']}" for i, s in enumerate(segs))
    schema = {"type": "object", "additionalProperties": False, "required": ["speakers"],
              "properties": {"speakers": {"type": "array", "items": {"type": "string", "enum": list(SPEAKERS)}}}}
    data, _ = await llm.complete_json(system, user, schema, "speaker_roles")
    raw = data.get("speakers") or []
    labels: list[str] = []
    fixed = 0
    for i in range(n):
        v = raw[i] if i < len(raw) else None
        if v in SPEAKERS:
            labels.append(v)
        else:
            fixed += 1
            labels.append("operator" if i == 0 or labels[-1] != "operator" else "client")
    return labels, fixed + abs(len(raw) - n)


def build_transcript(segs: list[dict], labels: list[str]) -> list[dict]:
    """Склеивает соседние сегменты одного говорящего (пауза ≤ 4 с) в одну реплику."""
    msgs: list[dict] = []
    for s, sp in zip(segs, labels):
        text = re.sub(r"\s+", " ", s["text"]).strip()
        if not text:
            continue
        last = msgs[-1] if msgs else None
        if last and last["speaker"] == sp and s["start"] - last["end"] <= 4:
            last["text"] += " " + text
            last["end"] = s["end"]
        else:
            msgs.append({"start": s["start"], "end": s["end"], "speaker": sp, "text": text})
    out, used = [], set()
    for m in msgs:
        sec = int(m["start"])
        while fmt_ts(sec) in used:  # таймкоды реплик должны быть уникальны
            sec += 1
        used.add(fmt_ts(sec))
        out.append({"timestamp": fmt_ts(sec), "speaker": m["speaker"], "text": m["text"]})
    return out


# ---------------- новый диалог ----------------
def make_record(conv_id: str, channel: str, language: str, transcript: list[dict], audio_file: str | None) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_.\-]{3,40}", conv_id or ""):
        raise ValueError("ID: 3–40 символов — латиница, цифры, «_», «.», «-»")
    lines = []
    for t in transcript:
        sp = str(t.get("speaker", "")).lower().strip()
        text = str(t.get("text", "")).strip()
        ts = str(t.get("timestamp", "")).strip()
        if sp not in SPEAKERS:
            raise ValueError(f"Неизвестный спикер «{sp}» (допустимо: {', '.join(SPEAKERS)})")
        if not re.fullmatch(r"\d{1,3}:\d{2}(:\d{2})?", ts):
            raise ValueError(f"Некорректное время «{ts}» — нужно MM:SS")
        if text:
            lines.append({"timestamp": ts, "speaker": sp, "text": text})
    if len(lines) < 2:
        raise ValueError("Нужно минимум 2 реплики")
    if not any(x["speaker"] == "operator" for x in lines) or not any(x["speaker"] == "client" for x in lines):
        raise ValueError("В транскрипте должны быть реплики и operator, и client")
    audio_url = None
    if audio_file:
        audio_url = UPLOAD_PREFIX + audio_file
        if not upload_path(audio_url):
            raise ValueError("Загруженная аудиозапись не найдена — загрузите файл заново")
    return {
        "conversation_id": conv_id,
        "channel": "voice call" if channel == "call" or audio_url else "agent chat",
        "language": (language or "RU").upper(),
        "transcript": lines,
        "audio_url": audio_url,
        "source": "audio" if audio_url else "manual",
    }
