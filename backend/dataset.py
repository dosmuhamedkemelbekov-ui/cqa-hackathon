"""Загрузка наборов диалогов из API организаторов и из файлов."""
import re
from urllib.parse import quote

import httpx

from .config import DATASET_API_URL

REQUIRED = {"conversation_id", "transcript"}


def validate_records(records: list[dict]) -> list[dict]:
    out = []
    for r in records:
        if not REQUIRED.issubset(r):
            raise ValueError(f"Запись без обязательных полей {REQUIRED}: {str(r)[:120]}")
        transcript = r["transcript"]
        if isinstance(transcript, str):  # на всякий случай: транскрипт строкой JSON
            import json

            transcript = json.loads(transcript)
        out.append(
            {
                "conversation_id": str(r["conversation_id"]),
                "channel": r.get("channel") or "agent chat",
                "language": (r.get("language") or "").upper() or None,
                "transcript": [
                    {"timestamp": str(t.get("timestamp", "")), "speaker": str(t.get("speaker", "")).lower(),
                     "text": str(t.get("text", ""))}
                    for t in transcript
                ],
                "audio_url": r.get("audio_url"),
            }
        )
    return out


def parse_payload(payload: dict | list) -> tuple[str | None, list[dict]]:
    """Принимает ответ API организаторов, объект data или просто список записей."""
    if isinstance(payload, list):
        return None, validate_records(payload)
    if "data" in payload and isinstance(payload["data"], dict):
        if payload.get("isSuccess") is False:
            raise ValueError(payload.get("error") or "API вернул isSuccess=false")
        payload = payload["data"]
    if "records" in payload:
        return payload.get("dataset_group_id"), validate_records(payload["records"])
    if "conversation_id" in payload:
        return None, validate_records([payload])
    raise ValueError("Не удалось распознать формат: ожидается ответ API, {records: [...]} или список диалогов")


async def fetch_group(group_id: str) -> tuple[str, list[dict]]:
    url = f"{DATASET_API_URL}&group_id={quote(group_id)}"
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        ds, records = parse_payload(resp.json())
    return ds or group_id, records


_AUDIO_SRC = re.compile(r'<source[^>]+src="([^"]+)"', re.I)


async def resolve_audio(audio_url: str) -> str:
    """audio_url организаторов отдаёт HTML-плеер; внутри — подписанная ссылка на mp3 (живёт ~1 час)."""
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        resp = await client.get(audio_url)
        resp.raise_for_status()
        if resp.headers.get("content-type", "").startswith("audio/"):
            return audio_url
        m = _AUDIO_SRC.search(resp.text)
        if not m:
            raise ValueError("В ответе плеера не найдена ссылка на аудио")
        return m.group(1).replace("&amp;", "&")
