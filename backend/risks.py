"""Сканер регуляторных рисков брокера — сверх чек-листа, на балл не влияет.

Ищет в репликах ОПЕРАТОРА (RU/KZ/EN) фразы, опасные для брокерской компании:
- обещание доходности / «без риска»;
- персональная инвестиционная рекомендация («купите эту бумагу»);
- уход от компании: деньги на личную карту, частная платная консультация, личные контакты;
- запрос у клиента пароля, CVV или PIN (SMS-код при идентификации — это процедура, не риск).
Поиск по шаблонам — мгновенно и бесплатно; по кнопке ИИ подтверждает или отклоняет найденное (1 вызов на диалог).
"""
import json
import re

from . import llm, preprocess

I = re.I

CATEGORIES = {
    "guarantee": {
        "name": "Обещание доходности",
        "why": "Брокер не вправе гарантировать доходность или отсутствие риска",
        "patterns": [
            r"гарантир\w*\s+(доход|прибыл|заработ|рост|выгод)", r"без\s+(всякого\s+)?риска", r"точно\s+(вырастет|заработа|подорожа)",
            r"с\s+выгодой", r"100\s*%\s*(доход|прибыл)",
            r"кепілдік\w*\s+(табыс|пайда)", r"тәуекелсіз", r"міндетті\s+түрде\s+(табыс|пайда)",
            r"guarantee\w*\s+(a\s+)?(return|profit|gain)", r"risk[- ]free", r"can'?t\s+lose", r"sure\s+profit",
        ],
    },
    "advice": {
        "name": "Персональная инвестиционная рекомендация",
        "why": "Оператор поддержки не даёт советов, что покупать или продавать",
        "patterns": [
            r"(советую|рекомендую|предлагаю)\s+(вам\s+)?(купить|вложить|взять|продать|приобрести)",
            r"(купите|покупайте|вложите|вкладывайте|продайте)\s+(эт|акци|облигац|бумаг|фонд)",
            r"какие\s+бумаги\s+(взять|купить)", r"что\s+(лучше\s+)?купить[^?]{0,30}(скажу|подскажу|объясню)",
            r"қандай\s+қағаз\s+алу\s+керегін", r"(сатып\s+алыңыз|салыңыз)\s+(осы|мына)",
            r"i\s+(recommend|suggest|advise)\s+(buying|selling|you\s+buy|investing)", r"you\s+should\s+(buy|sell|invest)",
            r"which\s+(bonds|stocks|shares)\s+to\s+buy",
        ],
    },
    "off_platform": {
        "name": "Уход от компании",
        "why": "Деньги, услуги или контакты в обход компании — прямой риск мошенничества",
        "patterns": [
            r"(на\s+(мою|личную)\s+карт|мне\s+на\s+карт|скиньте\s+(мне\s+)?деньги|переведите\s+(мне|деньги\s+мне)|я\s+сам\s+их\s+вложу)",
            r"через\s+компанию\s+(оформлять\s+)?не\s+надо", r"(частн\w+|мо[яю]\s+личн\w+)\s+консультац", r"за\s+плату",
            r"личный\s+номер", r"(whatsapp|ватсап|вотсап|телеграм)", r"созвонюсь\s+с\s+вами",
            r"жеке\s+(нөмір|карта)", r"ақылы\s+түрде",
            r"to\s+my\s+(personal\s+)?(card|account)", r"send\s+me\s+(the\s+)?money", r"private\s+advice", r"i\s+charge\s+for",
            r"personal\s+number", r"call\s+you\s+after\s+work",
        ],
    },
    "credentials": {
        "name": "Запрос пароля, CVV или PIN",
        "why": "Сотрудник никогда не запрашивает пароль, CVV или PIN карты",
        "patterns": [
            r"(назовите|продиктуйте|сообщите|скажите|напишите|отправьте)[^.?!]{0,25}(cvv|cvc|пароль|пин[- ]?код|\bpin\b)",
            r"(айтыңыз|жазыңыз)[^.?!]{0,25}(cvv|құпия\s*сөз|пин)",
            r"(tell|send|give|share)\s+(me\s+)?(your\s+)?(cvv|cvc|password|pin)\b",
        ],
    },
}
_COMPILED = {k: [re.compile(p, I) for p in v["patterns"]] for k, v in CATEGORIES.items()}


def scan(conv: dict) -> list[dict]:
    prepared = preprocess.prepare(conv)
    hits = []
    for m in prepared["messages"]:
        if m["speaker"] != "operator":
            continue
        for cat, pats in _COMPILED.items():
            mt = next((p.search(m["text"]) for p in pats if p.search(m["text"])), None)
            if mt:
                hits.append({"category": cat, "name": CATEGORIES[cat]["name"], "why": CATEGORIES[cat]["why"],
                             "msg_id": m["id"], "clock": m["clock"], "quote": m["text"][:300], "match": mt.group(0)})
    return hits


def scan_all(convs: list[dict]) -> dict:
    rows = []
    for c in convs:
        h = scan(c)
        if h:
            rows.append({"conversation_id": c["conversation_id"], "language": c.get("language"), "hits": h})
    by_cat: dict[str, int] = {}
    for r in rows:
        for h in r["hits"]:
            by_cat[h["category"]] = by_cat.get(h["category"], 0) + 1
    return {"dialogs_scanned": len(convs), "dialogs_with_risks": len(rows), "by_category": by_cat,
            "categories": {k: v["name"] for k, v in CATEGORIES.items()}, "rows": rows}


CONFIRM_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["items"],
    "properties": {"items": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["msg_id", "category", "real", "reason"],
        "properties": {"msg_id": {"type": "string"}, "category": {"type": "string"},
                       "real": {"type": "boolean"}, "reason": {"type": "string"}}}}},
}

CONFIRM_SYSTEM = """Ты — комплаенс-аналитик брокерской компании. Тебе дают диалог и места, где автоматический поиск
нашёл возможный регуляторный риск в словах оператора. Для каждого места реши: это реальный риск (real=true) или ложная
тревога (real=false) — например, оператор объясняет правило, цитирует клиента, отказывает или говорит о безопасной процедуре.
reason — одно предложение по-русски."""


async def confirm(conv: dict, hits: list[dict]) -> list[dict]:
    if not hits:
        return []
    prepared = preprocess.prepare(conv)
    lst = "\n".join(f"- {h['msg_id']} [{h['category']}: {h['name']}] «{h['quote'][:200]}»" for h in hits)
    # Только реальные номера реплик и категории находок — модель не сможет назвать их иначе.
    schema = json.loads(json.dumps(CONFIRM_SCHEMA))
    props = schema["properties"]["items"]["items"]["properties"]
    props["msg_id"]["enum"] = sorted({h["msg_id"] for h in hits})
    props["category"]["enum"] = sorted({h["category"] for h in hits})
    data, _ = await llm.complete_json(CONFIRM_SYSTEM, "НАЙДЕННЫЕ МЕСТА:\n" + lst, schema, name="risk_confirm",
                                      user_prefix="ТРАНСКРИПТ:\n" + prepared["transcript_text"] + "\n\n")
    items = [x for x in data.get("items", []) if isinstance(x, dict)]
    norm = lambda s: str(s or "").strip().lower()  # noqa: E731
    verdict = {(norm(x.get("msg_id")), norm(x.get("category"))): x for x in items}
    out = []
    for i, h in enumerate(hits):
        v = verdict.get((norm(h["msg_id"]), norm(h["category"])))
        if v is None and len(items) == len(hits):  # запасной вариант: ответы по порядку находок
            v = items[i]
        out.append({**h, "confirmed": v["real"] if v else None, "ai_reason": v["reason"] if v else "ИИ не дал ответа"})
    return out
