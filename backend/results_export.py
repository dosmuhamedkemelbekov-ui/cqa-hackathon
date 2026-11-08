"""Выгрузка результатов в формат организаторов «Результаты оценки · JSON 1.0».

results/dev/results.json и results/prod/results.json: один объект на файл, в results — все диалоги набора.
Коды критериев, типы значений и нумерация реплик — строго по приложению к гайду.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from . import config, scoring

SCHEMA_VERSION = "1.0"

# Официальные коды критериев (приложение к гайду). Порядок — как в чек-листе.
OFFICIAL_CODES = ["contact", "discovery", "identification", "consultation", "empathy", "dialogue", "closure", "critical"]
# Старые id критериев (c01…c08) → официальные коды: оценки, сделанные до переименования, тоже выгружаются.
LEGACY_IDS = dict(zip([f"c0{i}" for i in range(1, 9)], OFFICIAL_CODES))

# Допустимые значения по коду (для валидации)
ALLOWED = {
    "contact": {True, False},
    "discovery": {True, False, None},
    "identification": {True, False, None},
    "consultation": {0, 5, 10, 15, 20, None},
    "empathy": {True, False},
    "dialogue": {0, 4, 7, 10},
    "closure": {True, False, None},
    "critical": {True, False},
}


def criterion_code(c: dict) -> str:
    """Код критерия для выгрузки: явное поле code → старый id c0N → сам id."""
    return c.get("code") or LEGACY_IDS.get(c["id"]) or c["id"]


def to_value(c: dict, answer: str | None):
    """Внутренний ответ ('yes'/'no'/'na'/'<ступень>') → значение JSON 1.0."""
    if answer is None:
        raise ValueError("нет ответа")
    if answer == "na":
        return None
    if c["type"] in ("binary", "flag"):
        return answer == "yes"
    v = float(answer)
    return int(v) if v.is_integer() else v


def evidence_turns(item: dict, idx_by_id: dict[str, int]) -> list[int]:
    """Номера реплик с 1 по исходному массиву транскрипта (m0 → 1). Если цитат нет — проверенный участок."""
    turns = {idx_by_id[e["msg_id"]] + 1 for e in item.get("evidence") or [] if e.get("msg_id") in idx_by_id}
    if not turns and item.get("checked_range"):
        a = idx_by_id.get(item["checked_range"].get("from_msg"))
        b = idx_by_id.get(item["checked_range"].get("to_msg"))
        if a is not None and b is not None:
            lo, hi = min(a, b), max(a, b)
            turns = set(range(lo + 1, hi + 2)) if hi - lo < 12 else {lo + 1, hi + 1}
    return sorted(turns)


def _models_from(conversations: list[dict], latest: dict[str, dict]) -> list[str]:
    """Модели, которые фактически оценивали диалоги набора (из сохранённых оценок), в порядке появления."""
    out: list[str] = []
    for c in conversations:
        ev = latest.get(c["conversation_id"])
        for m in str((ev or {}).get("model") or "").split(","):
            m = m.strip()
            if m and m not in out:
                out.append(m)
    return out


def model_info(used: list[str] | None = None) -> dict:
    base = (config.LLM_BASE_URL or "").lower()
    local = any(h in base for h in ("localhost", "127.0.0.1", "0.0.0.0", "ollama"))
    mode = os.getenv("CQA_MODEL_MODE") or ("local" if local else "external_api")
    # По правилам: если моделей несколько — перечислить все. Берём фактически использованные.
    name = os.getenv("CQA_MODEL_NAME") or (", ".join(used) if used else config.LLM_MODEL)
    return {"mode": mode, "name": name}


def conversation_result(conv: dict, ev: dict | None, use: str = "ai", fail_reason: str | None = None) -> dict:
    """use='ai' — исходные ответы ИИ; 'final' — с учётом решений супервизора по апелляциям."""
    cid = conv["conversation_id"]

    def error(msg: str) -> dict:
        return {"conversation_id": cid, "status": "error", "criteria": [], "score": None, "error": msg}

    if not ev:
        return error(f"Не удалось оценить диалог: {fail_reason[:300]}" if fail_reason else "Диалог не оценён системой.")
    cl = ev["checklist_snapshot"]
    answers = ev["original_answers"] if use == "ai" else ev["final_answers"]
    items = {i["criterion_id"]: i for i in ev["criteria"]}
    idx_by_id = {f"m{i}": i for i in range(len(conv.get("transcript") or []))}

    by_code: dict[str, dict] = {}
    for c in cl["criteria"]:
        code = criterion_code(c)
        if code not in OFFICIAL_CODES:
            continue
        item = items.get(c["id"], {})
        try:
            value = to_value(c, answers.get(c["id"]))
        except ValueError:
            return error(f"Критерий {code}: модель не вернула ответ ({item.get('error') or 'нет данных'}).")
        by_code[code] = {
            "criterion_id": code,
            "value": value,
            "rationale": (item.get("rationale") or "").strip() or "Обоснование не получено.",
            "evidence_turns": evidence_turns(item, idx_by_id),
        }
    missing = [k for k in OFFICIAL_CODES if k not in by_code]
    if missing:
        return error("В чек-листе оценки нет критериев: " + ", ".join(missing))

    score = scoring.compute(cl, answers)
    return {
        "conversation_id": cid,
        "status": "ok",
        "criteria": [by_code[k] for k in OFFICIAL_CODES],
        "score": {"final_score": score["total"], "critical_flag": bool(by_code["critical"]["value"])},
    }


def build(conversations: list[dict], latest: dict[str, dict], team_name: str, use: str = "ai",
          failures: dict[str, str] | None = None) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "team_name": team_name,
        "model": model_info(_models_from(conversations, latest)),
        "results": [conversation_result(c, latest.get(c["conversation_id"]), use, (failures or {}).get(c["conversation_id"]))
                    for c in conversations],
    }


# ---------------- проверка файла перед push ----------------
def validate(doc: dict, expected_ids: list[str] | None = None, expected_count: int | None = None,
             warnings: list[str] | None = None) -> list[str]:
    """Возвращает список проблем; пустой список — файл соответствует формату 1.0.
    В warnings (если передан) попадают расхождения итога с весами исходного чек-листа (например, после замены критерия)."""
    errs: list[str] = []
    warnings = warnings if warnings is not None else []
    if doc.get("schema_version") != SCHEMA_VERSION:
        errs.append('schema_version должен быть "1.0"')
    if not isinstance(doc.get("team_name"), str) or not doc["team_name"].strip():
        errs.append("team_name пустой")
    m = doc.get("model") or {}
    if m.get("mode") not in ("local", "external_api", "hybrid") or not m.get("name"):
        errs.append("model: нужны mode (local/external_api/hybrid) и name")
    res = doc.get("results")
    if not isinstance(res, list):
        return errs + ["results должен быть массивом"]
    ids = [r.get("conversation_id") for r in res]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        errs.append(f"Повторяющиеся conversation_id: {sorted(dup)}")
    if expected_ids is not None:
        miss, extra = set(expected_ids) - set(ids), set(ids) - set(expected_ids)
        if miss:
            errs.append(f"Нет записей для: {sorted(miss)}")
        if extra:
            errs.append(f"Лишние записи: {sorted(extra)}")
    if expected_count is not None and len(res) != expected_count:
        errs.append(f"Ожидалось {expected_count} записей, в файле {len(res)}")
    for r in res:
        cid = r.get("conversation_id")
        st = r.get("status")
        if st == "error":
            if r.get("criteria") != [] or r.get("score") is not None or not r.get("error"):
                errs.append(f"{cid}: для status=error нужны criteria=[], score=null и error")
            continue
        if st != "ok":
            errs.append(f"{cid}: status должен быть ok или error")
            continue
        crit = r.get("criteria") or []
        codes = [c.get("criterion_id") for c in crit]
        if sorted(codes) != sorted(OFFICIAL_CODES):
            errs.append(f"{cid}: нужны все 8 критериев ровно по разу, есть {codes}")
            continue
        vals = {}
        for c in crit:
            code, v = c["criterion_id"], c.get("value")
            ok = (v is None or isinstance(v, bool)) if ALLOWED[code] <= {True, False, None} else (
                v is None or (isinstance(v, int) and not isinstance(v, bool)))
            if not ok or v not in ALLOWED[code]:
                errs.append(f"{cid}: {code} — недопустимое значение {v!r}")
            if not isinstance(c.get("rationale"), str) or not c["rationale"].strip():
                errs.append(f"{cid}: {code} — пустой rationale")
            t = c.get("evidence_turns")
            if not isinstance(t, list) or not all(isinstance(x, int) and not isinstance(x, bool) and x >= 1 for x in t):
                errs.append(f"{cid}: {code} — evidence_turns должен быть списком номеров с 1")
            vals[code] = v
        sc = r.get("score") or {}
        fs = sc.get("final_score")
        if fs is not None and (not isinstance(fs, (int, float)) or isinstance(fs, bool) or not 0 <= fs <= 100
                               or round(fs, 2) != fs):
            errs.append(f"{cid}: final_score должен быть числом 0–100 с двумя знаками")
        if sc.get("critical_flag") is not vals.get("critical"):
            errs.append(f"{cid}: critical_flag должен совпадать с value критерия critical")
        exp = official_score(vals)
        if exp != fs:
            warnings.append(f"{cid}: final_score {fs} не совпадает с расчётом по правилам ({exp})")
    return errs


WEIGHTS = {"contact": 10, "discovery": 15, "identification": 20, "consultation": 20, "empathy": 15, "dialogue": 15,
           "closure": 5}
SCALE_MAX = {"consultation": 20, "dialogue": 10}


def official_score(vals: dict) -> float | None:
    """Независимый пересчёт итога по правилам чек-листа (для самопроверки выгрузки)."""
    got = app = 0.0
    for k, w in WEIGHTS.items():
        v = vals.get(k)
        if v is None:
            continue
        app += w
        got += (w if v else 0) if isinstance(v, bool) else v / SCALE_MAX[k] * w
    return scoring._round(got / app * 100, 2) if app else None


def write(doc: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
