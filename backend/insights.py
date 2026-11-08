"""Аналитика без вызовов ИИ: предпросмотр изменения чек-листа и справедливость между языками."""
from . import evaluator, scoring

PROMPT_FIELDS = evaluator.PROMPT_FIELDS


def _same_prompt(a: dict, b: dict) -> bool:
    return all(a.get(f) == b.get(f) for f in PROMPT_FIELDS)


def preview_checklist(draft: dict, base: dict, latest: dict[str, dict], conv_ids: set[str] | None = None) -> dict:
    """«Что будет, если…»: итоги по готовым ответам с черновиком чек-листа, без вызовов модели.
    Сравнение с текущим активным чек-листом (base): изменились вес, N/A, секция — пересчёт точный;
    изменились формулировка или шкала, или критерий новый — его ответ исключается (заново оценит ИИ)."""
    rows, pending = [], {}
    base_defs = {c["id"]: c for c in base["criteria"]}
    for c in draft["criteria"]:
        if not (base_defs.get(c["id"]) and _same_prompt(c, base_defs[c["id"]])):
            pending[c["id"]] = c.get("name", c["id"])
    for cid, ev in latest.items():
        if conv_ids is not None and cid not in conv_ids:
            continue
        answers = {}
        for c in draft["criteria"]:
            a = ev["final_answers"].get(c["id"])
            if c["id"] not in pending and a is not None and a in scoring.allowed_answers(c):
                answers[c["id"]] = a
        before = ev["final_score"].get("total")
        after = scoring.compute(draft, answers).get("total")
        rows.append({"conversation_id": cid, "before": before, "after": after,
                     "delta": round(after - before, 2) if before is not None and after is not None else None})
    vals_b = [r["before"] for r in rows if r["before"] is not None]
    vals_a = [r["after"] for r in rows if r["after"] is not None]
    avg = lambda v: round(sum(v) / len(v), 2) if v else None  # noqa: E731
    moved = sorted([r for r in rows if r["delta"]], key=lambda r: -abs(r["delta"]))
    return {
        "n": len(rows),
        "avg_before": avg(vals_b), "avg_after": avg(vals_a),
        "below60_before": sum(1 for v in vals_b if v < 60), "below60_after": sum(1 for v in vals_a if v < 60),
        "changed": len(moved), "top": moved[:6],
        "pending": [{"id": k, "name": v} for k, v in pending.items()],
        "exact": not pending,
    }


def _bad(c: dict, a: str | None) -> bool | None:
    """Ответ «не в пользу оператора»: Нет, неполный балл шкалы, нарушение. N/A и пусто — не учитываются."""
    if a is None or a == "na":
        return None
    if c["type"] == "binary":
        return a == "no"
    if c["type"] == "flag":
        return a == "yes"
    if c["type"] == "scale":
        return float(a) < scoring.scale_max(c)
    return None


def fairness(latest: dict[str, dict], langs: dict[str, str], gold: dict | None = None, min_n: int = 5) -> dict:
    """Не оценивает ли ИИ один язык строже другого. Доля ответов «не в пользу оператора» по каждому
    критерию и языку; если есть эталон — сравнение со строгостью эталона на том же языке
    (перекос ИИ = доля ИИ − доля эталона: так видно, где ИИ строже людей именно на одном языке)."""
    by_lang: dict[str, list[str]] = {}
    for cid in latest:
        lang = langs.get(cid)
        if lang:
            by_lang.setdefault(lang, []).append(cid)
    languages = sorted(by_lang)
    ref = next(iter(latest.values()))["checklist_snapshot"] if latest else {"criteria": []}
    crit = []
    for c in ref["criteria"]:
        cells = {}
        for lang in languages:
            ai_bad = ai_n = g_bad = g_n = 0
            for cid in by_lang[lang]:
                ev = latest[cid]
                b = _bad(c, ev["original_answers"].get(c["id"]))
                if b is not None:
                    ai_n += 1
                    ai_bad += b
                if gold and cid in gold:
                    gb = _bad(c, (gold[cid].get("answers") or {}).get(c["id"]))
                    if gb is not None:
                        g_n += 1
                        g_bad += gb
            cells[lang] = {"n": ai_n, "ai_rate": round(ai_bad / ai_n, 3) if ai_n else None,
                           "gold_n": g_n, "gold_rate": round(g_bad / g_n, 3) if g_n else None}
            if cells[lang]["ai_rate"] is not None and cells[lang]["gold_rate"] is not None:
                cells[lang]["bias"] = round(cells[lang]["ai_rate"] - cells[lang]["gold_rate"], 3)
        # Главная метрика справедливости — доля ОШИБОК ИИ относительно эталона на каждом языке (по всем диалогам,
        # знаменатель — число диалогов, а не применимых ответов: устойчиво к малым выборкам).
        for lang in languages:
            err = tot = 0
            for cid in by_lang[lang]:
                g = ((gold or {}).get(cid) or {}).get("answers") or {}
                if c["id"] in g:
                    tot += 1
                    err += latest[cid]["original_answers"].get(c["id"]) != g[c["id"]]
            cells[lang]["err_n"] = tot
            cells[lang]["err_rate"] = round(err / tot, 3) if tot else None
        rates = [v["ai_rate"] for v in cells.values() if v["ai_rate"] is not None and v["n"] >= min_n]
        spread = round(max(rates) - min(rates), 3) if len(rates) >= 2 else 0
        errs = [v["err_rate"] for v in cells.values() if v.get("err_rate") is not None and v["err_n"] >= 10]
        err_spread = round(max(errs) - min(errs), 3) if len(errs) >= 2 else None
        biases = [v.get("bias") for v in cells.values() if v.get("bias") is not None and v["n"] >= min_n]
        bias_spread = round(max(biases) - min(biases), 3) if len(biases) >= 2 else None
        # Перекос = ИИ ошибается на одном языке заметно чаще, чем на другом (при эталоне);
        # без эталона — только сильный разброс доли «не в пользу оператора».
        flag = (err_spread is not None and err_spread >= 0.15) if gold else spread >= 0.3
        crit.append({"id": c["id"], "name": c["name"], "cells": cells, "spread": spread,
                     "bias_spread": bias_spread, "err_spread": err_spread, "flag": flag})
    totals = {}
    for lang in languages:
        t = [latest[cid]["original_score"]["total"] for cid in by_lang[lang] if latest[cid]["original_score"]["total"] is not None]
        acc_n = acc_ok = 0
        if gold:
            for cid in by_lang[lang]:
                g = (gold.get(cid) or {}).get("answers") or {}
                for k, v in g.items():
                    if k in latest[cid]["original_answers"]:
                        acc_n += 1
                        acc_ok += latest[cid]["original_answers"][k] == v
        totals[lang] = {"dialogs": len(by_lang[lang]), "avg_total": round(sum(t) / len(t), 2) if t else None,
                        "accuracy": round(acc_ok / acc_n, 3) if acc_n else None}
    return {"languages": languages, "totals": totals, "criteria": crit, "has_gold": bool(gold),
            "flags": sum(1 for x in crit if x["flag"])}
