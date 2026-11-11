"""Deterministic hard-constraint checker.

This is the single source of truth for validity. The solver is built to satisfy the same
rules, the LLM is never trusted on them, and every schedule is re-checked here before it
is returned to the coordinator.
"""
from __future__ import annotations

from itertools import combinations

from .model import Problem


def normalize_schedule(entries) -> tuple[dict, list[dict]]:
    """[{course_id, timeslot_id, room_id, ...}] -> {course_id: (timeslot_id, room_id)} plus structural violations."""
    sched, issues = {}, []
    for e in entries or []:
        cid, tid, rid = e.get("course_id"), e.get("timeslot_id"), e.get("room_id")
        if cid in sched:
            issues.append(dict(constraint="duplicate_assignment", course_id=cid,
                               detail=f"{cid} is assigned more than once"))
            continue
        sched[cid] = (tid, rid)
    return sched, issues


def check_schedule(prob: Problem, sched: dict) -> list[dict]:
    """Return every hard-constraint violation in `sched` ({course_id: (timeslot_id, room_id)})."""
    v = []
    valid = {}
    for cid, (tid, rid) in sched.items():
        bad = False
        if cid not in prob.courses:
            v.append(dict(constraint="unknown_course", course_id=cid, detail=f"{cid} is not in the course list")); bad = True
        if tid not in prob.timeslots:
            v.append(dict(constraint="unknown_timeslot", course_id=cid, detail=f"timeslot {tid} was not supplied by the coordinator")); bad = True
        if rid not in prob.rooms:
            v.append(dict(constraint="unknown_room", course_id=cid, detail=f"room {rid} is not allocated to the unit")); bad = True
        if not bad:
            v += prob.static_violations(cid, tid, rid)
            valid[cid] = (tid, rid)

    for (a, (ta, ra)), (b, (tb, rb)) in combinations(valid.items(), 2):
        if not prob.overlaps(ta, tb):
            continue
        if ra == rb:
            v.append(dict(constraint="room_double_booked", course_id=a, other_course_id=b,
                          detail=f"{a} ({ta}) and {b} ({tb}) overlap in {ra}"))
        shared = set(prob.courses[a].instructors) & set(prob.courses[b].instructors)
        if shared:
            v.append(dict(constraint="instructor_double_booked", course_id=a, other_course_id=b,
                          detail=f"instructor {', '.join(sorted(shared))} teaches {a} ({ta}) and {b} ({tb}) at overlapping times"))
        if frozenset((a, b)) in prob.conflict_pairs:
            v.append(dict(constraint="course_conflict", course_id=a, other_course_id=b,
                          detail=f"{a} ({ta}) and {b} ({tb}) must not overlap"))
    return v


def violations_for_course(prob: Problem, sched: dict, cid: str) -> list[dict]:
    return [x for x in check_schedule(prob, sched) if cid in (x.get("course_id"), x.get("other_course_id"))]


class Occupancy:
    """Fast incremental feasibility for one course against an otherwise-fixed schedule.
    Used to enumerate alternatives and to compute preference_score."""

    def __init__(self, prob: Problem, sched: dict):
        self.prob = prob
        self.room_slots: dict[str, list] = {}
        self.instr_slots: dict[str, list] = {}
        self.course_slot: dict[str, str] = {}
        for cid, (tid, rid) in sched.items():
            self.add(cid, tid, rid)

    def add(self, cid, tid, rid):
        self.room_slots.setdefault(rid, []).append((tid, cid))
        for i in self.prob.courses[cid].instructors:
            self.instr_slots.setdefault(i, []).append((tid, cid))
        self.course_slot[cid] = tid

    def dynamic_violations(self, cid, tid, rid, ignore=()) -> list[dict]:
        p, out = self.prob, []
        for t2, c2 in self.room_slots.get(rid, []):
            if c2 != cid and c2 not in ignore and p.overlaps(tid, t2):
                out.append(dict(constraint="room_double_booked", course_id=cid, other_course_id=c2,
                                detail=f"{rid} is already used by {c2} during {t2}"))
        for i in p.courses[cid].instructors:
            for t2, c2 in self.instr_slots.get(i, []):
                if c2 != cid and c2 not in ignore and p.overlaps(tid, t2):
                    out.append(dict(constraint="instructor_double_booked", course_id=cid, other_course_id=c2,
                                    detail=f"instructor {i} already teaches {c2} during {t2}"))
        for c2 in p.conflicts_of.get(cid, ()):
            t2 = self.course_slot.get(c2)
            if t2 and c2 not in ignore and p.overlaps(tid, t2):
                out.append(dict(constraint="course_conflict", course_id=cid, other_course_id=c2,
                                detail=f"{c2} (which must not overlap {cid}) meets during {t2}"))
        return out

    def options(self, cid, ignore=()):
        """All (timeslot, room) pairs where cid could go with everything else held fixed."""
        p = self.prob
        for tid in p.timeslots:
            for rid in p.rooms:
                if p.static_ok(cid, tid, rid) and not self.dynamic_violations(cid, tid, rid, ignore):
                    yield tid, rid
