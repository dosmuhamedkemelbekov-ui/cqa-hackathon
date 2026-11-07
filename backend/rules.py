"""Детерминированные правила поверх ответов LLM.

Объективные, механически проверяемые факты (приветствие с именем, системное подтверждение личности,
кто оборвал чат) модель иногда пропускает. Правила ниже проверены на всех 65 диалогах DEV против эталона
(100% совпадения в своих случаях) и применяются только к критериям, формулировка которых совпадает
с исходным чек-листом: если критерий заменён или переписан, правило не вмешивается.
Каждое срабатывание записывается в оценку (item["rule"]) и видно в карточке.
"""
from __future__ import annotations

import json
import re

from . import config, preprocess

# Приветствие на языке клиента.
GREET = {
    "RU": re.compile(r"(здравствуй|добр(ый|ое|ого)\s+(день|дня|вечер|утро|утра|времени)|приветствую)", re.I),
    "KZ": re.compile(r"(сәлеметсіз|салеметсиз|сәлеметсіздер|қайырлы\s+(күн|кеш|таң)|армысыз)", re.I),
    "EN": re.compile(r"\b(hello|hi|hey|good\s+(morning|afternoon|evening|day)|greetings)\b", re.I),
}
# Имя оператора: «меня зовут», «я Лера», «Мен Айбекпін», «I'm Maya», маска [Имя скрыто] и т.п.
NAME = [
    re.compile(r"(меня\s+зовут|меня\s+\S+\s+зовут|с\s+вами\s+[А-ЯЁ]|на\s+связи\s+[А-ЯЁ]|\[имя\s+скрыто\]|"
               r"менің\s+(атым|есімім)|my\s+name\s*('s|is)|you'?re\s+speaking\s+with|this\s+is\s+[a-z]+)", re.I),
    re.compile(r"(?:^|[\s.,!—-])(?i:я|мен)\s*[—-]?\s*[А-ЯЁӘІҢҒҮҰҚӨҺ]"),
    re.compile(r"\b(?i:i'?m|i\s+am)\s+[A-Z]"),
]
END_CLIENT = re.compile(r"(клиент покинул|покинул чат|соединение с клиентом прервано|связь прервал|байланыс үзілді|"
                        r"connection .*lost|customer left|left the chat)", re.I)
END_OPERATOR = re.compile(r"(оператор закрыл чат|оператор чатты жапты|agent closed the chat|operator closed the chat)", re.I)

# Вежливые формулы и прощания: если последняя реплика оператора состоит только из них — итога нет.
POLITE = re.compile(r"(пожалуйста|рад[аы]?\s+помочь|отлично|хорошо|спасибо за обращение|спасибо за ожидание|спасибо|"
                    r"оқасы жоқ|көмегім тигеніне қуаныштымын|жақсы|рақмет|"
                    r"you'?re welcome|glad (you got in|to help|i could help)|happy to help|great|of course|"
                    r"thanks for (reaching out|contacting us|your patience)|thank you|and)", re.I)
BYE = re.compile(r"(до свидания|хорошего (дня|вечера)|всего (доброго|хорошего)|сау болыңыз|күніңіз сәтті өтсін|"
                 r"goodbye|good bye|bye|have a (good|nice|great) (day|evening)|take care)", re.I)
# Уточняющий вопрос оператора: знак вопроса или просьба уточнить.
CLARIFY = re.compile(r"(\?|уточните|укажите|подскажите|напишите,? (какой|какая|какие|как|где|когда)|"
                     r"нақтылаңыз|айтыңыз|жазыңыз|specify|could you tell|let me know which)", re.I)


def only_polite_bye(text: str) -> bool:
    """Реплика — только вежливая формула + прощание, без итога и предложения помощи."""
    if not BYE.search(text):
        return False
    rest = POLITE.sub(" ", BYE.sub(" ", text))
    return not re.sub(r"[\s.,!?—-]+", " ", rest).strip()


_DEFAULT: dict | None = None


def _default_rules() -> dict[str, str]:
    """Тексты правил исходного чек-листа по коду критерия — правило применяем, только если критерий не менялся."""
    global _DEFAULT
    if _DEFAULT is None:
        cl = json.loads(config.DEFAULT_CHECKLIST.read_text(encoding="utf-8"))
        _DEFAULT = {c.get("code") or c["id"]: c.get("rule", "") for c in cl["criteria"]}
    return _DEFAULT


def _code(c: dict) -> str:
    from .results_export import criterion_code
    return criterion_code(c)


def contact_check(conv: dict, messages: list[dict]) -> tuple[str, str, str | None]:
    """(ответ, пояснение, id реплики) по первой реплике оператора."""
    op = next((m for m in messages if m["speaker"] == "operator"), None)
    if not op:
        return "no", "Оператор не написал ни одной реплики.", None
    lang = (conv.get("language") or "RU").upper()
    greet = GREET.get(lang, GREET["RU"]).search(op["text"])
    name = any(p.search(op["text"]) for p in NAME)
    lang_name = {"RU": "русском", "KZ": "казахском", "EN": "английском"}.get(lang, lang)
    if greet and name:
        return "yes", f"В первой реплике оператор поздоровался на {lang_name} и назвал имя.", op["id"]
    miss = [x for x, ok in (("приветствия на " + lang_name + " языке клиента", greet), ("имени оператора", name)) if not ok]
    return "no", "В первой реплике оператора нет " + " и ".join(miss) + ".", op["id"]


def apply(conv: dict, checklist: dict, prepared: dict, results: dict[str, dict]) -> list[dict]:
    """Поправляет ответы LLM по правилам. Возвращает список срабатываний."""
    if not config.RULES_ENABLED:
        return []
    msgs = prepared["messages"]
    is_voice = prepared.get("is_voice")
    defaults = _default_rules()
    fired = []

    def set_answer(c: dict, answer: str, why: str, msg_id: str | None):
        r = results.get(c["id"])
        if not r or r.get("answer") is None:
            return
        before = r["answer"]
        prev = r.get("rule") or {}
        r["rule"] = {"from": before, "to": answer, "why": why, "changed": before != answer,
                     "llm_rationale": prev.get("llm_rationale", r.get("rationale") or "")}
        if before != answer:
            r["answer"] = answer
            # В объяснении — только причина правила: рассуждение модели вело к другому ответу и противоречило бы ему.
            # Оно сохраняется в rule.llm_rationale и показывается в карточке.
            r["rationale"] = f"{why} Основание — проверка по правилу."
            if msg_id and not any(e.get("msg_id") == msg_id for e in r.get("evidence") or []):
                m = next((x for x in msgs if x["id"] == msg_id), None)
                if m:
                    r.setdefault("evidence", []).insert(0, {"msg_id": msg_id, "quote": m["text"][:120],
                                                            "why": "правило", "verified": True})
            fired.append({"criterion_id": c["id"], "from": before, "to": answer, "why": why})

    for c in checklist["criteria"]:
        code = _code(c)
        if code not in defaults or (c.get("rule") or "") != defaults[code]:
            continue  # критерий заменён или переписан — правила не вмешиваются
        if code == "contact":
            ans, why, mid = contact_check(conv, msgs)
            set_answer(c, ans, why, mid)
        elif code == "identification":
            conf = next((m for m in msgs if m["speaker"] == "system" and preprocess._IDENTITY.search(m["text"])), None)
            r = results.get(c["id"]) or {}
            # Системное подтверждение личности: «N/A» модели исправляем на «Да». «Нет» не трогаем —
            # данные могли раскрыть до подтверждения.
            if conf and r.get("answer") == "na":
                set_answer(c, "yes", f"Личность подтверждена системой ({conf['id']}) — идентификация проводилась.", conf["id"])
        elif code == "closure":
            closed = next((m for m in msgs if m["speaker"] == "system" and END_OPERATOR.search(m["text"])), None)
            if closed:
                set_answer(c, "no", "Оператор сам закрыл чат — это не обрыв, завершение не выполнено.", closed["id"])
                continue
            if is_voice:
                continue
            ops = [m for m in msgs if m["speaker"] == "operator"]
            if ops and only_polite_bye(ops[-1]["text"]):
                set_answer(c, "no", f"Последняя реплика оператора ({ops[-1]['id']}) — только вежливая формула и прощание: "
                                    "нет ни итога, ни предложения дальнейшей помощи.", ops[-1]["id"])
                continue
            end = next((m for m in reversed(msgs) if m["speaker"] == "system" and END_CLIENT.search(m["text"])), None)
            if end:
                before = [m for m in msgs if m["idx"] < end["idx"] and m["speaker"] in ("client", "operator")]
                if before and before[-1]["speaker"] == "client":
                    set_answer(c, "na", "Чат оборвался сразу после реплики клиента — у оператора не было "
                                        "возможности завершить диалог.", end["id"])
            elif ops and not any(BYE.search(m["text"]) for m in ops):
                set_answer(c, "no", "Оператор так и не попрощался, а чат не оборвался (нет события завершения).",
                           ops[-1]["id"])
        elif code == "discovery":
            ops = [m for m in msgs if m["speaker"] == "operator"]
            r = results.get(c["id"]) or {}
            if r.get("answer") == "yes" and ops and not any(CLARIFY.search(m["text"]) for m in ops):
                set_answer(c, "na", "Оператор не задал ни одного уточняющего вопроса и ответил сразу — «Да» "
                                    "требует уточнения недостающего; данных хватало с начала.", None)
    # Согласованность: если клиент так и не сформулировал запрос (консультация N/A), выявлять было нечего.
    by_code = {_code(c): c for c in checklist["criteria"]}
    cons, disc = by_code.get("consultation"), by_code.get("discovery")
    if cons and disc and (results.get(cons["id"]) or {}).get("answer") == "na" \
            and (disc.get("rule") or "") == defaults.get("discovery"):
        set_answer(disc, "na", "Клиент не сформулировал запрос (консультация N/A) — уточнять было нечего.", None)
    return fired


def facts_lines(conv: dict, prepared: dict) -> list[str]:
    """Подсказки для промпта: результат проверки приветствия (модель видит то же, что правило)."""
    ans, why, mid = contact_check(conv, prepared["messages"])
    return [f"- Проверка первой реплики оператора{(' ' + mid) if mid else ''}: {why}"]
