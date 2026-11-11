"""CP-SAT assignment model shared by generate_schedule and repair_schedule.

Decision variable x[c,t,r] = 1 if course c meets in timeslot t in room r. Only statically valid
(c,t,r) triples get a variable (capacity, features, faculty availability and blackouts are
filtered out up front). Overlap constraints use "clique points": two intervals overlap iff
one contains the other's start time, so for every (day, start time) we constrain the set of
timeslots active at that instant. That is exact even when the coordinator's timeslots overlap
each other irregularly (e.g. MW-1100 vs MWF-1100 vs MW-1100-1230).
"""
from __future__ import annotations

from ortools.sat.python import cp_model

from .model import Problem
from .scoring import raw_score, weights

SCHEDULE_BONUS = 100_000   # dominates everything: schedule as many courses as possible first


def _clique_points(prob: Problem) -> list[list[str]]:
    """Each returned list is a set of timeslots that all overlap at a single instant."""
    cliques, seen = [], set()
    days = {d for t in prob.timeslots.values() for d in t.days}
    for d in days:
        slots = [t for t in prob.timeslots.values() if d in t.days]
        for p in sorted({t.start for t in slots}):
            active = tuple(sorted(t.id for t in slots if t.start <= p < t.end))
            if len(active) and active not in seen:
                seen.add(active)
                cliques.append(list(active))
    # drop cliques strictly contained in another (redundant constraints)
    sets = [set(c) for c in cliques]
    return [c for i, c in enumerate(cliques) if not any(i != j and sets[i] < sets[j] for j in range(len(sets)))]


def solve(prob: Problem, current: dict | None = None, time_limit: float | None = None) -> tuple[dict, list[str], str]:
    """Return (schedule {cid: (tid, rid)}, unscheduled course ids, solver status).

    If `current` is given (repair mode), moving a course away from its current assignment is
    penalized so the solver keeps everything it can and only moves what it must.
    """
    w = weights()
    m = cp_model.CpModel()
    x: dict[tuple, cp_model.IntVar] = {}
    for cid in prob.courses:
        for tid in prob.timeslots:
            for rid in prob.rooms:
                if prob.static_ok(cid, tid, rid):
                    x[cid, tid, rid] = m.NewBoolVar(f"x|{cid}|{tid}|{rid}")

    by_course: dict[str, list] = {c: [] for c in prob.courses}
    by_course_slot: dict[tuple, list] = {}
    by_room_slot: dict[tuple, list] = {}
    for (c, t, r), var in x.items():
        by_course[c].append(var)
        by_course_slot.setdefault((c, t), []).append(var)
        by_room_slot.setdefault((r, t), []).append(var)

    for c, vars_ in by_course.items():
        if vars_:
            m.AddAtMostOne(vars_)

    cliques = _clique_points(prob)
    for clique in cliques:
        for rid in prob.rooms:
            vs = [v for t in clique for v in by_room_slot.get((rid, t), [])]
            if len(vs) > 1:
                m.AddAtMostOne(vs)
        for iid, cids in prob.courses_of_instructor.items():
            if len(cids) < 2:
                continue
            vs = [v for c in cids for t in clique for v in by_course_slot.get((c, t), [])]
            if len(vs) > 1:
                m.AddAtMostOne(vs)
        for pair in prob.conflict_pairs:
            vs = [v for c in pair for t in clique for v in by_course_slot.get((c, t), [])]
            if len(vs) > 1:
                m.AddAtMostOne(vs)

    obj = []
    for (c, t, r), var in x.items():
        coef = SCHEDULE_BONUS + int(round(100 * raw_score(prob, c, t, r)))
        if current and c in current:
            ct, cr = current[c]
            if t != ct:
                coef -= w["repair_time_change_cost"]
            elif r != cr:
                coef -= w["repair_room_change_cost"]
        obj.append(coef * var)
    m.Maximize(sum(obj))

    if current:
        for c, (t, r) in current.items():
            if (c, t, r) in x:
                m.AddHint(x[c, t, r], 1)

    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = float(time_limit or w["solver_time_limit_s"])
    s.parameters.num_workers = 8
    status = s.Solve(m)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {}, list(prob.courses), s.StatusName(status)
    sched = {c: (t, r) for (c, t, r), var in x.items() if s.Value(var)}
    unscheduled = [c for c in prob.courses if c not in sched]
    return sched, unscheduled, s.StatusName(status)


def diagnose_unscheduled(prob: Problem, cid: str) -> str:
    """Explain why a course could not be placed (most specific reason first)."""
    c = prob.courses[cid]
    fits = [r for r in prob.rooms.values() if r.capacity >= c.enrollment]
    if not fits:
        biggest = max((r.capacity for r in prob.rooms.values()), default=0)
        return f"No allocated room holds {c.enrollment} students (largest is {biggest})."
    feat = [r for r in fits if c.required_features <= r.features]
    if not feat:
        return (f"No allocated room with capacity >= {c.enrollment} has required feature(s) "
                f"{', '.join(sorted(c.required_features))}.")
    if not any(prob.static_ok(cid, t, r.id) for t in prob.timeslots for r in feat):
        return "Every supplied timeslot is blocked by instructor unavailability or a blackout period."
    return ("Every statically valid time/room is taken by other courses, instructor commitments, "
            "or courses it must not overlap.")
