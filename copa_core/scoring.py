"""Soft-preference scoring. Weights live in config/weights.json.

raw_score(c, t, r) is a weighted sum of simple terms. Historical data enters here and only
here: it can make an option more attractive, never make an infeasible option valid.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .model import Problem, parse_time, parse_days, room_key, building_of

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def weights() -> dict:
    return json.loads((ROOT / "config" / "weights.json").read_text())


@lru_cache(maxsize=1)
def history() -> dict:
    p = ROOT / "data" / "history_stats.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _hist_pattern_sim(hist_patterns, t) -> float:
    best = 0.0
    for hp in hist_patterns:
        same_days = parse_days(hp["days"]) == t.days
        dur_close = abs(hp["minutes"] - t.minutes) <= 10
        start_close = abs(parse_time(hp["start"]) - t.start) <= 60
        best = max(best, 0.4 * same_days + 0.4 * dur_close + 0.2 * start_close)
    return best


def terms(prob: Problem, cid: str, tid: str, rid: str) -> dict:
    """Individual (unweighted) preference terms, also used to write rationales."""
    w = weights()
    c, t, r = prob.courses[cid], prob.timeslots[tid], prob.rooms[rid]
    h = history().get(c.hist_key, {})
    hrooms, hbuild = h.get("rooms", {}), h.get("buildings", {})
    return {
        "course_preferred_time": float(tid in c.preferred_times),
        "course_avoid_time": float(tid in c.avoid_times),
        "faculty_preferred_time": float(any(tid in prob.faculty_pref.get(i, ()) for i in c.instructors)),
        "historical_pattern": _hist_pattern_sim(h.get("patterns", []), t),
        "historical_building": float(building_of(rid) in hbuild),
        "historical_room": float(room_key(rid) in hrooms),
        "early_morning": float(t.start < parse_time(w["early_morning_before"])),
        "late_evening": float(t.end > parse_time(w["late_evening_after"])),
        "friday": float("F" in t.days),
        "wasted_capacity": max(0.0, 1.0 - c.enrollment / r.capacity) if r.capacity else 0.0,
        "_has_history": bool(h),
    }


def raw_score(prob: Problem, cid: str, tid: str, rid: str) -> float:
    w, tm = weights(), terms(prob, cid, tid, rid)
    return sum(w[k] * v for k, v in tm.items() if not k.startswith("_"))


def preference_score(prob: Problem, cid: str, chosen: tuple, feasible: list[tuple]) -> float:
    """Where the chosen assignment sits between the worst (0) and best (1) feasible option
    for this course, holding the rest of the schedule fixed."""
    scores = [raw_score(prob, cid, t, r) for t, r in feasible] or [raw_score(prob, cid, *chosen)]
    lo, hi = min(scores), max(scores)
    s = raw_score(prob, cid, *chosen)
    return 1.0 if hi - lo < 1e-9 else round(max(0.0, min(1.0, (s - lo) / (hi - lo))), 3)


def absolute_pref(prob: Problem, cid: str, tid: str, rid: str) -> float:
    """Assignment quality in [0,1] across ALL statically valid options (used for impact scores)."""
    cache = prob.__dict__.setdefault("_abs_pref_range", {})
    if cid not in cache:
        vals = [raw_score(prob, cid, t, r) for t in prob.timeslots for r in prob.rooms if prob.static_ok(cid, t, r)]
        cache[cid] = (min(vals), max(vals)) if vals else (0.0, 0.0)
    lo, hi = cache[cid]
    s = raw_score(prob, cid, tid, rid)
    return 1.0 if hi - lo < 1e-9 else max(0.0, min(1.0, (s - lo) / (hi - lo)))


def rationale(prob: Problem, cid: str, tid: str, rid: str) -> list[str]:
    c, r = prob.courses[cid], prob.rooms[rid]
    tm = terms(prob, cid, tid, rid)
    out = [f"Room capacity {r.capacity} fits expected enrollment {c.enrollment}."]
    if c.required_features:
        out.append(f"Room provides required feature(s): {', '.join(sorted(c.required_features))}.")
    if tm["course_preferred_time"]:
        out.append("Time is listed in the course's preferred times.")
    if tm["faculty_preferred_time"]:
        out.append("Time is preferred by the instructor.")
    if tm["historical_pattern"] >= 0.8:
        out.append("Matches the meeting pattern this course has historically used.")
    if tm["historical_room"]:
        out.append("Same room the course has used before.")
    elif tm["historical_building"]:
        out.append("Same building the course has historically used.")
    if tm["wasted_capacity"] > 0.7:
        out.append("Room is much larger than needed; no better-fitting room was free at this time.")
    if tm["course_avoid_time"]:
        out.append("Uses a time the course asked to avoid; no conflict-free preferred time remained.")
    return out
