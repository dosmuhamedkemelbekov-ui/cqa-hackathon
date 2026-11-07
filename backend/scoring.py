"""Детерминированный расчёт баллов по правилам чек-листа (стр. 2 PDF). LLM здесь не участвует."""
from decimal import ROUND_HALF_UP, Decimal


def _round(x: float | None, digits: int) -> float | None:
    if x is None:
        return None
    q = Decimal(1).scaleb(-digits)
    return float(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))


def scale_max(criterion: dict) -> float:
    return max(float(s["value"]) for s in criterion.get("scale") or [{"value": 1}])


def allowed_answers(criterion: dict) -> list[str]:
    t = criterion["type"]
    if t == "scale":
        vals = [str(int(s["value"]) if float(s["value"]).is_integer() else s["value"]) for s in criterion["scale"]]
    else:
        vals = ["yes", "no"]
    if criterion.get("na_allowed"):
        vals.append("na")
    return vals


def raw_points(criterion: dict, answer: str) -> float | None:
    """Баллы до перераспределения (в шкале веса критерия). None — критерий не участвует (N/A или флаг)."""
    if answer == "na" or criterion["type"] == "flag" or not criterion.get("weight"):
        return None
    w = float(criterion["weight"])
    if criterion["type"] == "binary":
        return w if answer == "yes" else 0.0
    if criterion["type"] == "scale":
        return float(answer) / scale_max(criterion) * w
    return None


def compute(checklist: dict, answers: dict[str, str]) -> dict:
    """answers: {criterion_id: 'yes'|'no'|'na'|'<значение шкалы>'}"""
    digits = int(checklist.get("rounding", 2))
    items = []
    applicable_weight = 0.0
    earned = 0.0
    gross = False
    for c in checklist["criteria"]:
        a = answers.get(c["id"])
        if a is None:
            continue
        if c["type"] == "flag":
            if a == "yes":
                gross = True
            items.append({"criterion_id": c["id"], "applicable": False, "flag": a == "yes"})
            continue
        pts = raw_points(c, a)
        if pts is None:
            items.append({"criterion_id": c["id"], "applicable": False, "weight": c["weight"]})
            continue
        applicable_weight += float(c["weight"])
        earned += pts
        items.append({"criterion_id": c["id"], "applicable": True, "weight": c["weight"], "raw_points": pts})

    factor = 100.0 / applicable_weight if applicable_weight else None
    for it in items:
        if it.get("applicable") and factor:
            it["effective_weight"] = _round(float(it["weight"]) * factor, digits)
            it["points"] = _round(it["raw_points"] * factor, digits)
            it["raw_points"] = _round(it["raw_points"], digits)
        elif "weight" in it:
            it["effective_weight"] = 0
            it["points"] = None

    total = _round(earned * factor, digits) if factor else None

    # Секции: доля набранного от применимого максимума внутри секции + вклад в итог.
    sections = []
    by_id = {c["id"]: c for c in checklist["criteria"]}
    for s in checklist.get("sections") or []:
        its = [it for it in items if by_id[it["criterion_id"]].get("section") == s["id"] and "weight" in it]
        if not its:
            continue
        app = [it for it in its if it["applicable"]]
        max_w = sum(float(it["weight"]) for it in app)
        got = sum(float(it["raw_points"]) for it in app)
        sections.append(
            {
                "id": s["id"],
                "name": s["name"],
                "percent": _round(got / max_w * 100, digits) if max_w else None,
                "points": _round(got * factor, digits) if factor and app else None,
                "max_points": _round(max_w * factor, digits) if factor and app else None,
            }
        )

    return {
        "total": total,
        "max_total": 100 if factor else None,
        "applicable_weight": applicable_weight,
        "gross_error": gross,
        "items": items,
        "sections": sections,
    }
