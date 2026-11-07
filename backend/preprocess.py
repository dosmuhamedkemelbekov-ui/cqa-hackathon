"""Подготовка транскрипта для LLM: id реплик, время в секундах, факты, которые надёжнее посчитать кодом."""
import re
from datetime import datetime

ROLE_LABEL = {"client": "КЛИЕНТ", "operator": "ОПЕРАТОР", "bot": "БОТ", "system": "СИСТЕМА"}
LANG_NAME = {"RU": "русский", "KZ": "казахский", "EN": "английский"}

# Порог «длинной паузы» оператора (сек). Первая реакция оператора по условию задания не оценивается.
PAUSE_THRESHOLD = {"chat": 60, "voice": 30}

_REC_START = re.compile(r"(запись и транскрибирование (запущены|возобновлены)|жазу және транскрипция басталды|recording.*(started|resumed))", re.I)
_REC_PAUSE = re.compile(r"(запись и транскрибирование приостановлены|recording.*paused)", re.I)
_ESCALATION = re.compile(r"(эскалац|операторға жіберілді|transferred to an agent|агент присоединился|переведен от агента)", re.I)
_IDENTITY = re.compile(r"(личность клиента подтверждена|жеке басы расталды|identity verified)", re.I)
_END = re.compile(r"(закрыл чат|чатты жапты|closed the chat|покинул чат|завершил разговор|прервано|байланыс үзілді|connection .*lost|left the chat)", re.I)


def _parse_iso(ts: str) -> datetime | None:
    ts = ts.strip()
    m = re.match(r"^(.*T\d{2}:\d{2}:\d{2})(\.\d+)?(.*)$", ts)
    if not m:
        return None
    frac = (m.group(2) or "")[:7].ljust(7, "0")[:7] if m.group(2) else ""
    tz = m.group(3).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(m.group(1) + frac + tz)
    except ValueError:
        return None


def _to_seconds(ts: str, base: datetime | None) -> float | None:
    ts = (ts or "").strip()
    if re.fullmatch(r"\d{1,3}:\d{2}(:\d{2})?", ts):
        parts = [int(p) for p in ts.split(":")]
        return float(parts[0] * 60 + parts[1]) if len(parts) == 2 else float(parts[0] * 3600 + parts[1] * 60 + parts[2])
    dt = _parse_iso(ts)
    if dt and base:
        return (dt - base).total_seconds()
    return None


def fmt_clock(sec: float | None) -> str:
    if sec is None:
        return "--:--"
    sec = int(round(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def prepare(conv: dict) -> dict:
    """Возвращает {messages, facts, transcript_text}. Исходные метки времени сохраняются в поле ts."""
    transcript = conv.get("transcript") or []
    is_voice = "voice" in (conv.get("channel") or "").lower() or bool(conv.get("audio_url"))
    base = None
    for t in transcript:
        base = _parse_iso(t.get("timestamp", ""))
        if base:
            break

    messages = []
    for i, t in enumerate(transcript):
        sec = _to_seconds(t.get("timestamp", ""), base)
        messages.append(
            {
                "id": f"m{i}",
                "idx": i,
                "ts": t.get("timestamp", ""),
                "sec": sec,
                "clock": fmt_clock(sec),
                "speaker": t.get("speaker", ""),
                "text": t.get("text", ""),
            }
        )

    # Позиция в аудиозаписи: время с начала записи минус паузы записи (для перехода плеера к фрагменту).
    if is_voice:
        rec_on, audio_pos, last_sec = False, 0.0, None
        for m in messages:
            s = m["sec"]
            if rec_on and s is not None and last_sec is not None:
                audio_pos += max(0.0, s - last_sec)
            if s is not None:
                last_sec = s
            m["audio_sec"] = round(audio_pos, 1) if rec_on or audio_pos else None
            if m["speaker"] == "system" and _REC_START.search(m["text"]):
                rec_on = True
                m["audio_sec"] = round(audio_pos, 1)
            elif m["speaker"] == "system" and _REC_PAUSE.search(m["text"]):
                rec_on = False
        if not any(_REC_START.search(m["text"]) for m in messages if m["speaker"] == "system"):
            for m in messages:  # записи-маркеров нет — считаем от начала
                m["audio_sec"] = m["sec"]

    facts = compute_facts(conv, messages, is_voice)
    return {"messages": messages, "facts": facts, "is_voice": is_voice, "transcript_text": render(messages)}


def render(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        text = re.sub(r"\s+", " ", m["text"]).strip()
        lines.append(f"[{m['id']} | {m['clock']} | {ROLE_LABEL.get(m['speaker'], m['speaker'].upper())}] {text}")
    return "\n".join(lines)


def compute_facts(conv: dict, messages: list[dict], is_voice: bool) -> dict:
    ops = [m for m in messages if m["speaker"] == "operator"]
    facts: dict = {
        "channel": "звонок" if is_voice else "чат",
        "client_language": LANG_NAME.get((conv.get("language") or "").upper(), conv.get("language") or "не указан"),
        "operator_messages": len(ops),
        "first_operator_message": ops[0]["id"] if ops else None,
        "last_operator_message": ops[-1]["id"] if ops else None,
        "escalation_message": next(
            (m["id"] for m in messages if m["speaker"] == "system" and _ESCALATION.search(m["text"])), None
        ),
        "identity_verified_messages": [m["id"] for m in messages if m["speaker"] == "system" and _IDENTITY.search(m["text"])],
        "end_event": next(
            ({"id": m["id"], "text": m["text"]} for m in reversed(messages) if m["speaker"] == "system" and _END.search(m["text"])),
            None,
        ),
        "last_message": {"id": messages[-1]["id"], "speaker": messages[-1]["speaker"]} if messages else None,
    }

    # Паузы оператора: клиент написал, оператор ответил спустя > порога. Первая реакция оператора исключена.
    threshold = PAUSE_THRESHOLD["voice" if is_voice else "chat"]
    pauses = []
    first_op_idx = ops[0]["idx"] if ops else None
    pending_client = None
    for m in messages:
        if first_op_idx is None or m["idx"] <= first_op_idx:
            if m["speaker"] == "operator":
                pending_client = None
            continue
        if m["speaker"] == "client" and pending_client is None:
            pending_client = m
        elif m["speaker"] == "operator":
            if pending_client and pending_client["sec"] is not None and m["sec"] is not None:
                gap = m["sec"] - pending_client["sec"]
                if gap >= threshold:
                    pauses.append({"client_msg": pending_client["id"], "operator_reply": m["id"], "seconds": int(gap)})
            pending_client = None
    facts["long_operator_pauses"] = pauses
    unanswered = [m for m in messages if m["speaker"] == "client" and first_op_idx is not None and m["idx"] > (ops[-1]["idx"])]
    facts["client_messages_after_last_operator"] = [m["id"] for m in unanswered]
    return facts


def facts_text(facts: dict) -> str:
    lines = [
        f"- Канал: {facts['channel']}. Язык клиента по данным набора: {facts['client_language']}.",
        f"- Реплик оператора: {facts['operator_messages']}. Первая реплика оператора: {facts['first_operator_message'] or 'нет'}. "
        f"Последняя: {facts['last_operator_message'] or 'нет'}.",
        f"- Передача диалога оператору (системное событие): {facts['escalation_message'] or 'не найдено'}.",
        f"- Системное подтверждение личности клиента: {', '.join(facts['identity_verified_messages']) or 'нет'}.",
    ]
    if facts["end_event"]:
        lines.append(f"- Событие завершения: {facts['end_event']['id']} «{facts['end_event']['text']}».")
    else:
        lines.append("- Системного события завершения нет.")
    if facts["last_message"]:
        lines.append(f"- Последнее сообщение диалога: {facts['last_message']['id']} ({ROLE_LABEL.get(facts['last_message']['speaker'], '')}).")
    if facts["client_messages_after_last_operator"]:
        lines.append(f"- Сообщения клиента после последней реплики оператора: {', '.join(facts['client_messages_after_last_operator'])}.")
    if facts["long_operator_pauses"]:
        p = "; ".join(f"{x['client_msg']}→{x['operator_reply']} {x['seconds']} c" for x in facts["long_operator_pauses"])
        lines.append(f"- Долгие ответы оператора после первой реакции (клиент→ответ): {p}. Проверь, предупреждал ли оператор об ожидании до паузы.")
    else:
        lines.append("- Долгих пауз оператора после первой реакции не найдено.")
    return "\n".join(lines)
