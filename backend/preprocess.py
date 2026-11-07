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
        # В звонках таймкод транскрипта — КОНЕЦ фразы (проверено на DEV: так выходит ~2.2 слова/с, иначе ~5.8).
        # Начало фразы оцениваем по числу слов (~2.3 слова/с), не раньше конца предыдущей реплики:
        # плеер переходит к НАЧАЛУ фразы, а профиль разговора считает длительности честно.
        # Фразы собеседников в звонке могут перекрываться, поэтому начало ограничиваем не концом,
        # а НАЧАЛОМ предыдущей реплики — порядок реплик сохраняется.
        prev_start, prev_astart = None, None
        for m in messages:
            if m["sec"] is None:
                continue
            est = 0.0 if m["speaker"] == "system" else _words(m["text"]) / 2.3 + 0.3
            start = m["sec"] - est if prev_start is None else max(prev_start, m["sec"] - est)
            m["end_sec"] = m["sec"]
            m["start_sec"] = round(start, 2)
            if m.get("audio_sec") is not None:
                a_start = m["audio_sec"] - (m["sec"] - start)
                if prev_astart is not None:
                    a_start = max(prev_astart, a_start)
                m["audio_end"] = m["audio_sec"]
                m["audio_sec"] = round(max(0.0, a_start), 1)
                prev_astart = m["audio_sec"]
            prev_start = start
    else:
        for m in messages:
            m["start_sec"] = m["sec"]
            m["end_sec"] = m["sec"]

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
                gap = m.get("start_sec", m["sec"]) - pending_client["sec"]  # от конца фразы клиента до начала ответа
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
    ]
    if facts["identity_verified_messages"]:
        lines.append(f"- Системное подтверждение личности клиента: {', '.join(facts['identity_verified_messages'])}.")
    elif facts["channel"] == "звонок":
        lines.append("- Системного сообщения о верификации нет: в звонках идентификация проходит голосом "
                     "(клиент называет ФИО/данные/SMS-код, оператор подтверждает «всё верно», «идентификация пройдена»). "
                     "Оценивай по репликам; скрытые маской данные означают, что клиент их назвал.")
    else:
        lines.append("- Системного подтверждения личности в чате нет (проверь, не подтверждена ли личность в репликах).")
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


# ======================= ПРОФИЛЬ РАЗГОВОРА (по таймкодам, без ИИ) =======================
_REC_ANY = re.compile(r"(запись и транскрибирование|жазу және транскрипция|recording)", re.I)


def _words(t: str) -> int:
    return len(re.findall(r"\w+", t or "")) if "[" not in (t or "") else max(1, len(re.findall(r"\w+", re.sub(r"\[[^\]]*\]", "x", t))))


def compute_profile(messages: list[dict], is_voice: bool) -> dict:
    """Темп речи, доля речи, паузы, перебивания (звонки) и скорость ответов (чаты) — по таймкодам транскрипта.
    Длительность реплики в звонке оценивается: до начала следующей реплики, но не дольше, чем нужно на её слова."""
    ms = [m for m in messages if m.get("sec") is not None]
    ops = [m for m in ms if m["speaker"] == "operator"]
    if len(ms) < 2 or not ops:
        return {"available": False}
    start = ops[0]["idx"]
    part = [m for m in ms if m["idx"] >= start and m["speaker"] in ("client", "operator")]
    prof: dict = {"available": True, "kind": "call" if is_voice else "chat",
                  "duration_s": round(ms[-1]["sec"] - ms[0]["sec"]),
                  "operator_part_s": round(part[-1]["sec"] - part[0]["sec"]) if len(part) > 1 else 0,
                  "insights": []}
    ins = prof["insights"]

    if is_voice:
        # В звонках таймкод = конец фразы; начало оценено в prepare() (start_sec).
        # Доля речи — по числу слов (надёжно при любой нарезке распознавания).
        words = {"client": 0, "operator": 0}
        for m in part:
            words[m["speaker"]] += _words(m["text"])
        tw = words["client"] + words["operator"] or 1
        prof["talk_share"] = {"operator": round(words["operator"] / tw * 100), "client": round(words["client"] / tw * 100)}
        # Темп — медиана по фразам, длительность которых измерима: фраза идёт сразу за предыдущей
        # (от конца предыдущей до своего конца) и скорость в правдоподобных пределах 1.2–4.5 слова/с.
        rates: dict[str, list[float]] = {"client": [], "operator": []}
        for prev, m in zip(part, part[1:]):
            w = _words(m["text"])
            d = m["end_sec"] - prev["end_sec"]
            if w >= 6 and d > 0 and 1.2 <= w / d <= 4.5:
                rates[m["speaker"]].append(w / d)
        prof["wpm"] = {k: (round(sorted(v)[len(v) // 2] * 60) if len(v) >= 3 else None) for k, v in rates.items()}
        # Перебивание: оператор начал содержательную фразу (≥4 слов) минимум на 2 с раньше, чем клиент договорил.
        # Перекрытия >10 с — артефакты нарезки распознавания, не считаем.
        inter = []
        for a, b in zip(part, part[1:]):
            if a["speaker"] == "client" and b["speaker"] == "operator" and _words(b["text"]) >= 4                     and _words(a["text"]) >= 3:
                b_start_raw = b["end_sec"] - (_words(b["text"]) / 2.3 + 0.3)
                overlap = a["end_sec"] - b_start_raw
                if 2.0 <= overlap <= 10.0:
                    inter.append({"client_msg": a["id"], "operator_msg": b["id"], "overlap_s": round(overlap, 1)})
        prof["interruptions"] = inter
        # Тишина: клиент договорил — оператор долго не начинает отвечать (паузы записи не считаем).
        silences = []
        for a, b in zip(part, part[1:]):
            if a["speaker"] == "client" and b["speaker"] == "operator":
                gap = b["start_sec"] - a["end_sec"]
                between = [m for m in ms if a["idx"] < m["idx"] < b["idx"] and m["speaker"] == "system" and _REC_ANY.search(m["text"])]
                if gap >= 8 and not between:
                    silences.append({"after": a["id"], "reply": b["id"], "seconds": round(gap)})
        silences.sort(key=lambda x: -x["seconds"])
        prof["silences"] = silences[:5]
        op_share, wpm = prof["talk_share"]["operator"], prof["wpm"]["operator"]
        if op_share >= 70:
            ins.append({"level": "warn", "text": f"Оператор говорит {op_share}% времени — клиенту почти не дают слова"})
        elif op_share <= 25:
            ins.append({"level": "info", "text": f"Оператор говорит лишь {op_share}% времени — разговор ведёт клиент"})
        if wpm and wpm >= 170:
            ins.append({"level": "warn", "text": f"Быстрый темп речи оператора: ~{wpm} слов/мин (комфортно 110–150)"})
        elif wpm and wpm <= 85:
            ins.append({"level": "info", "text": f"Медленный темп речи оператора: ~{wpm} слов/мин"})
        if len(inter) >= 2:
            ins.append({"level": "warn", "text": f"Оператор, похоже, перебивал клиента: {len(inter)} раз(а)", "msg": inter[0]["operator_msg"]})
        elif inter:
            ins.append({"level": "info", "text": "Одно возможное перебивание клиента", "msg": inter[0]["operator_msg"]})
        if silences:
            s = silences[0]
            ins.append({"level": "warn" if s["seconds"] >= 15 else "info",
                        "text": f"Самая долгая тишина оператора после клиента: {s['seconds']} с", "msg": s["reply"]})
    else:
        # чат: как быстро оператор отвечает на сообщения клиента (первая реакция по заданию не оценивается)
        waits, pending = [], None
        for m in part:
            if m["speaker"] == "client" and pending is None:
                pending = m
            elif m["speaker"] == "operator" and pending is not None:
                waits.append({"after": pending["id"], "reply": m["id"], "seconds": round(m["sec"] - pending["sec"])})
                pending = None
        secs = sorted(w["seconds"] for w in waits)
        prof["replies"] = len(waits)
        prof["reply_median_s"] = secs[len(secs) // 2] if secs else None
        prof["reply_max"] = max(waits, key=lambda w: w["seconds"]) if waits else None
        op_words = [_words(m["text"]) for m in part if m["speaker"] == "operator"]
        prof["operator_avg_words"] = round(sum(op_words) / len(op_words)) if op_words else 0
        cl = [m for m in part if m["speaker"] == "client"]
        prof["talk_share"] = {"operator": round(len(op_words) / max(1, len(op_words) + len(cl)) * 100),
                              "client": round(len(cl) / max(1, len(op_words) + len(cl)) * 100)}
        if prof["reply_max"] and prof["reply_max"]["seconds"] >= 60:
            r = prof["reply_max"]
            ins.append({"level": "warn" if r["seconds"] >= 120 else "info",
                        "text": f"Самое долгое ожидание ответа оператора: {fmt_clock(r['seconds'])}", "msg": r["reply"]})
        if prof["operator_avg_words"] <= 4 and len(op_words) >= 2:
            ins.append({"level": "warn", "text": f"Оператор отвечает очень коротко: в среднем {prof['operator_avg_words']} слова"})
        unanswered = [m for m in part if m["speaker"] == "client" and m["idx"] > ops[-1]["idx"]]
        if unanswered:
            ins.append({"level": "info", "text": "Последнее сообщение клиента осталось без ответа оператора", "msg": unanswered[-1]["id"]})
    if not ins:
        ins.append({"level": "ok", "text": "По таймкодам отклонений не видно"})
    return prof
