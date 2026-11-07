"""Очередь проверки: в каком порядке супервизору смотреть диалоги.

Система никого не пропускает — она только сортирует: сначала самые рискованные, в конце самые безопасные.
Риск считается ТОЛЬКО по признакам, видимым до проверки человеком (оценка ИИ, её надёжность, профиль разговора).
Веса заданы заранее и не подбирались под эталон — иначе замер «топ-k против случайной выборки» был бы нечестным.
"""
from . import preprocess, scoring

# Критерии, «Нет» по которым особенно важно перепроверить (безопасность и суть консультации).
CRITICAL = {"c03": 12.0, "c04": 6.0}


def risk(ev: dict, profile: dict | None = None, open_appeals: int = 0) -> tuple[float, list[str]]:
    """Возвращает (балл риска, человеко-понятные причины)."""
    score, why = 0.0, []
    fs = ev["final_score"]
    total = fs.get("total")
    if fs.get("gross_error"):
        score += 50
        why.append("грубая ошибка по оценке ИИ")
    if open_appeals:
        score += 25
        why.append("открыта апелляция")
    if total is not None:
        score += (100 - total) * 0.4
        if total < 70:
            why.append(f"низкий итог {total:g}")
    answers = ev.get("final_answers", {})
    defs = {c["id"]: c for c in ev["checklist_snapshot"]["criteria"]}
    for cid, w in CRITICAL.items():
        c, a = defs.get(cid), answers.get(cid)
        if not c or a is None or a == "na":
            continue
        bad = a == "no" if c["type"] == "binary" else (c["type"] == "scale" and float(a) <= scoring.scale_max(c) / 2)
        if bad:
            score += w
            why.append(f"«{c['name']}» — {'Нет' if a == 'no' else a}")
    rel = ev.get("reliability") or {}
    items = ev.get("criteria", [])
    low = sum(1 for i in items if i.get("confidence") == "low")
    split = sum(1 for i in items if i.get("votes") and len(i["votes"]) > 1)
    fixed = sum(1 for i in items if (i.get("verification") or {}).get("verdict") == "changed")
    unver = rel.get("quotes_total", 0) - rel.get("quotes_verified", 0)
    errs = len(ev.get("errors") or [])
    if low:
        score += 8 * low
        why.append(f"ИИ не уверен: {low} крит.")
    if split:
        score += 8 * split
        why.append(f"прогоны ИИ разошлись: {split} крит.")
    if fixed:
        score += 5 * fixed
        why.append(f"самопроверка исправила ответ: {fixed}")
    if unver > 0:
        score += 6 * min(unver, 3)
        why.append(f"неподтверждённых цитат: {unver}")
    if errs:
        score += 20
        why.append("ИИ не смог оценить часть критериев")
    if profile and profile.get("available"):
        warns = [x["text"] for x in profile.get("insights", []) if x.get("level") == "warn"]
        if warns:
            score += 4 * len(warns)
            why.append("профиль: " + warns[0][0].lower() + warns[0][1:])
    return round(score, 1), why


def build_queue(convs: list[dict], latest: dict[str, dict], open_appeals: dict[str, int]) -> list[dict]:
    rows = []
    for c in convs:
        ev = latest.get(c["conversation_id"])
        if not ev:
            continue
        prep = preprocess.prepare(c)
        prof = preprocess.compute_profile(prep["messages"], prep["is_voice"])
        r, why = risk(ev, prof, open_appeals.get(c["conversation_id"], 0))
        rows.append({"conversation_id": c["conversation_id"], "risk": r, "reasons": why,
                     "total": ev["final_score"].get("total"), "ai_total": ev["original_score"].get("total"),
                     "gross_error": ev["final_score"].get("gross_error"), "language": c.get("language"),
                     "channel": c.get("channel")})
    rows.sort(key=lambda x: -x["risk"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def gold_problem(gold_answers: dict, checklist: dict, threshold: float) -> tuple[bool, str]:
    """Реальная проблема по эталону: грубая ошибка или итог ниже порога."""
    s = scoring.compute(checklist, gold_answers)
    if s.get("gross_error"):
        return True, "грубая ошибка"
    if s.get("total") is not None and s["total"] < threshold:
        return True, f"итог {s['total']:g}"
    return False, ""


def evaluate_queue(rows: list[dict], gold: dict, latest: dict, threshold: float = 70, ks=(5, 10)) -> dict:
    """Сравнение: наша очередь vs случайная выборка vs сортировка только по баллу ИИ."""
    labeled = [r for r in rows if r["conversation_id"] in gold and gold[r["conversation_id"]].get("answers")]
    if not labeled:
        return {"n": 0}
    truth = {}
    for r in labeled:
        cl = latest[r["conversation_id"]]["checklist_snapshot"]
        truth[r["conversation_id"]] = gold_problem(gold[r["conversation_id"]]["answers"], cl, threshold)
    n = len(labeled)
    pos = sum(1 for v in truth.values() if v[0])
    by_score = sorted(labeled, key=lambda r: (r["ai_total"] if r["ai_total"] is not None else 101))
    out = {"n": n, "problems": pos, "threshold": threshold, "prevalence": round(pos / n, 3), "k": []}
    for k in ks:
        k = min(k, n)
        ours = sum(1 for r in labeled[:k] if truth[r["conversation_id"]][0])
        score_only = sum(1 for r in by_score[:k] if truth[r["conversation_id"]][0])
        out["k"].append({"k": k, "ours": ours, "random": round(k * pos / n, 1), "score_only": score_only,
                         "recall": round(ours / pos, 3) if pos else None})
    # сколько диалогов нужно просмотреть, чтобы найти ВСЕ реальные проблемы
    def all_found(order: list[dict]) -> int | None:
        seen = 0
        for i, r in enumerate(order, 1):
            if truth[r["conversation_id"]][0]:
                seen += 1
                if seen == pos:
                    return i
        return None

    out["all_found_at"] = all_found(labeled)
    out["all_found_score_only"] = all_found(by_score)
    # при случайном порядке матожидание позиции последней из pos проблем среди n: pos*(n+1)/(pos+1)
    out["all_found_random"] = round(pos * (n + 1) / (pos + 1)) if pos else None
    out["truth"] = {cid: {"problem": v[0], "why": v[1]} for cid, v in truth.items()}
    return out
