"""The three coordinator operations. Each returns (result, warnings, errors, status)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from . import llm
from .checker import Occupancy, check_schedule, normalize_schedule, violations_for_course
from .model import DAY_ORDER, Problem, build_problem, parse_global_constraint
from .scoring import absolute_pref, preference_score, rationale, terms, weights
from .solver import diagnose_unscheduled, solve

STATE_DIR = Path(__file__).resolve().parent.parent / "data" / "state"
RESOURCE_KEYS = ("courses", "timeslots", "rooms", "faculty_constraints", "global_constraints")
_memory_cache: dict[str, dict] = {}


# ------------------------------------------------------------------ resource cache
# evaluate_change requests carry only current_schedule + proposed_change, so the agent
# remembers the last resource envelope it was given for each semester.

def remember_resources(semester: str, payload: dict):
    res = {k: payload.get(k) for k in RESOURCE_KEYS if payload.get(k) is not None}
    _memory_cache[semester] = res
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        (STATE_DIR / f"{_safe(semester)}.json").write_text(json.dumps(res))
    except OSError:
        pass


def recall_resources(semester: str) -> dict | None:
    if semester in _memory_cache:
        return _memory_cache[semester]
    p = STATE_DIR / f"{_safe(semester)}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _safe(s):
    return "".join(ch if ch.isalnum() else "_" for ch in str(s or "default"))


# ------------------------------------------------------------------ problem setup

def prepare_problem(payload: dict) -> Problem:
    """Build the Problem; hand any global constraints the parser didn't recognize to the LLM to
    translate into the structured blackout format, then validate what comes back."""
    prob = build_problem(payload)
    if not prob.unparsed_constraints:
        return prob
    reply = llm.ask_json("parse_constraints", {
        "constraints": prob.unparsed_constraints,
        "course_ids": list(prob.courses), "room_ids": list(prob.rooms),
        "instructor_ids": sorted(prob.courses_of_instructor),
    }) or {}
    accepted = []
    for gc in reply.get("constraints", []) if isinstance(reply.get("constraints"), list) else []:
        blocks, err = parse_global_constraint(gc, prob.timeslots)
        ids_ok = all(gc.get(k) is None or gc.get(k) in pool for k, pool in
                     (("course_id", prob.courses), ("room_id", prob.rooms), ("instructor_id", prob.courses_of_instructor)))
        if blocks and not err and ids_ok:
            accepted.append(gc)
    if accepted:
        p2 = dict(payload)
        known = [g for g in payload.get("global_constraints") or [] if g not in prob.unparsed_constraints]
        p2["global_constraints"] = known + accepted
        new = build_problem(p2)
        new.warnings.append(f"LLM translated {len(accepted)} free-form global constraint(s) into blackout rules: "
                            + json.dumps(accepted))
        prob, still = new, reply.get("unsupported") or []
    else:
        still = prob.unparsed_constraints
    for gc in still:
        prob.warnings.append(f"Could not interpret global constraint {gc!r}; it was NOT enforced.")
    return prob


def _entry(prob, sched, occ, cid):
    tid, rid = sched[cid]
    feas = list(occ.options(cid))
    return dict(course_id=cid, timeslot_id=tid, room_id=rid, rationale=rationale(prob, cid, tid, rid),
                preference_score=preference_score(prob, cid, (tid, rid), feas))


# ------------------------------------------------------------------ generate_schedule

def generate_schedule(payload: dict, semester: str):
    prob = prepare_problem(payload)
    remember_resources(semester, prob.raw_payload)   # includes any LLM-translated constraints
    sched, unscheduled, status = solve(prob)
    violations = check_schedule(prob, sched)
    if violations:     # should be impossible; never ship a violating schedule
        bad = {v["course_id"] for v in violations} | {v.get("other_course_id") for v in violations}
        for c in bad:
            sched.pop(c, None)
        unscheduled += [c for c in bad if c and c not in unscheduled]
        prob.warnings.append(f"Post-check removed {len(bad)} assignment(s) that failed validation.")

    occ = Occupancy(prob, sched)
    schedule = [_entry(prob, sched, occ, c) for c in prob.courses if c in sched]
    unresolved = [dict(course_id=c, reason=diagnose_unscheduled(prob, c)) for c in unscheduled]

    weakest = sorted(schedule, key=lambda e: e["preference_score"])[:5]
    summary = (f"Scheduled {len(schedule)} of {len(prob.courses)} courses with no hard-constraint violations."
               + (f" {len(unresolved)} course(s) could not be placed; see unresolved." if unresolved else ""))
    reply = llm.ask_json("schedule_summary", {
        "scheduled": len(schedule), "total": len(prob.courses), "unresolved": unresolved,
        "weakest_assignments": [dict(e, terms=terms(prob, e["course_id"], e["timeslot_id"], e["room_id"]),
                                     title=prob.courses[e["course_id"]].title) for e in weakest],
    })
    if reply and isinstance(reply.get("summary"), str):
        summary = reply["summary"]
        notes = reply.get("notes") if isinstance(reply.get("notes"), dict) else {}
        for e in schedule:
            if isinstance(notes.get(e["course_id"]), str):
                e["rationale"].append(notes[e["course_id"]])
    if status != "OPTIMAL" and schedule:
        prob.warnings.append(f"Solver stopped at time limit ({status}); schedule is valid but may not be preference-optimal.")

    result = dict(schedule=schedule, unresolved=unresolved, summary=summary)
    return result, prob.warnings, [], ("partial" if unresolved else "completed")


# ------------------------------------------------------------------ evaluate_change

def _impact(prob, cid, old, new) -> float:
    moved = 0.2 if new[0] != old[0] else (0.1 if new[1] != old[1] else 0.0)
    loss = max(0.0, absolute_pref(prob, cid, *old) - absolute_pref(prob, cid, *new))
    return round(min(1.0, moved + 0.8 * loss), 3)


LABELS = {"course_preferred_time": "the course's preferred time", "faculty_preferred_time": "the instructor's preferred time",
          "historical_pattern": "its historical meeting pattern", "historical_building": "its usual building",
          "historical_room": "its usual room", "course_avoid_time": "a time the course asked to avoid",
          "early_morning": "an early-morning start", "late_evening": "a late-evening end", "friday": "a Friday meeting",
          "wasted_capacity": "a poorly sized room"}


def _pref_diff(prob, cid, old, new):
    w, a, b = weights(), terms(prob, cid, *old), terms(prob, cid, *new)
    lost = [LABELS[k] for k in LABELS if w[k] > 0 and a[k] - b[k] > 0.3]
    added = [LABELS[k] for k in LABELS if w[k] < 0 and b[k] - a[k] > 0.3]
    return lost, added


def _alt_reason(prob, cid, old, tid, rid):
    lost, added = _pref_diff(prob, cid, old, (tid, rid))
    if not lost and not added:
        return "No hard conflicts and no preference penalty."
    parts = (["loses " + ", ".join(lost)] if lost else []) + (["adds " + ", ".join(added)] if added else [])
    return "No hard conflicts; " + "; ".join(parts) + "."


def alternatives(prob, sched, cid, proposed, n=None):
    """Feasible (time, room) options for cid with the rest of the schedule fixed, filtered to what
    the coordinator was trying to achieve, ranked by impact."""
    n = n or weights()["max_alternatives"]
    cur_t, cur_r = sched[cid]
    occ = Occupancy(prob, {c: a for c, a in sched.items() if c != cid})
    wants_new_time = proposed[0] is not None and proposed[0] != cur_t
    opts = []
    for t, r in occ.options(cid):
        if (t, r) in ((cur_t, cur_r), proposed):
            continue
        if wants_new_time and cur_t in prob.timeslots and prob.overlaps(t, cur_t):
            continue          # coordinator needs this course out of its current time window
        if not wants_new_time and r == cur_r:
            continue          # coordinator needs this course out of its current room
        opts.append((_impact(prob, cid, (cur_t, cur_r), (t, r)), t != proposed[0], t, r))
    opts.sort()
    return [dict(timeslot_id=t, room_id=r, impact_score=imp, reason=_alt_reason(prob, cid, (cur_t, cur_r), t, r))
            for imp, _, t, r in opts[:n]]


def evaluate_change(payload: dict, semester: str):
    warnings = []
    resources = {k: payload[k] for k in RESOURCE_KEYS if payload.get(k) is not None}
    if not {"courses", "timeslots", "rooms"} <= set(resources):
        cached = recall_resources(semester) or {}
        resources = {**cached, **resources}
        if cached:
            warnings.append("Resource envelope not included in request; using the last one supplied for this semester.")
    if not {"courses", "timeslots", "rooms"} <= set(resources):
        return None, warnings, ["evaluate_change needs courses/timeslots/rooms: include them in the payload or call generate_schedule first."], "failed"

    prob = prepare_problem(resources)
    warnings += prob.warnings
    sched, struct = normalize_schedule(payload.get("current_schedule"))
    pc = payload.get("proposed_change") or {}
    cid = pc.get("course_id")
    if cid not in sched or cid not in prob.courses:
        return dict(decision="reject", reason=f"Course {cid} is not part of this unit's current schedule.",
                    impact_score=1.0, alternatives=[]), warnings, [], "completed"

    old = sched[cid]
    new = (pc.get("new_timeslot_id") or old[0], pc.get("new_room_id") or old[1])
    hypo = dict(sched); hypo[cid] = new
    viol = violations_for_course(prob, hypo, cid)
    pre_existing = {(v["constraint"], v.get("other_course_id")) for v in violations_for_course(prob, sched, cid)}
    if pre_existing:
        warnings.append(f"{cid}'s current assignment already has violations: {sorted(pre_existing)}")

    if viol:
        alts = alternatives(prob, sched, cid, new) if old[0] in prob.timeslots else []
        reason = "Rejected: " + "; ".join(sorted({f"{v['constraint']}: {v['detail']}" for v in viol}))
        if not alts:
            reason += " No feasible alternative exists with the rest of the schedule held fixed."
        return dict(decision="reject", reason=reason, impact_score=1.0, alternatives=alts), warnings, [], "completed"

    impact = _impact(prob, cid, old, new)
    tm_old, tm_new = terms(prob, cid, *old), terms(prob, cid, *new)
    decision = "accept" if impact <= weights()["accept_impact_threshold"] else "reject"
    lost, added = _pref_diff(prob, cid, old, new)
    reason = ("Accepted: no hard-constraint violations" if decision == "accept" else
              "Rejected: feasible, but the preference cost is high") + \
             (f"; loses {', '.join(lost)}" if lost else "") + (f"; adds {', '.join(added)}" if added else "") + "."

    reply = llm.ask_json("evaluate_change", {
        "course": dict(id=cid, title=prob.courses[cid].title, enrollment=prob.courses[cid].enrollment),
        "current": dict(timeslot_id=old[0], room_id=old[1], preference_terms=tm_old),
        "proposed": dict(timeslot_id=new[0], room_id=new[1], preference_terms=tm_new),
        "impact_score": impact, "coordinator_reason": payload.get("reason", ""),
        "policy": "accept if feasible unless the preference cost is clearly unjustified",
    })
    if reply and reply.get("decision") in ("accept", "reject") and isinstance(reply.get("reason"), str):
        decision, reason = reply["decision"], reply["reason"]

    alts = []
    if decision == "reject":
        alts = [a for a in alternatives(prob, sched, cid, new) if a["impact_score"] < impact]
        if not alts:   # rejecting with nothing better to offer would just stall the coordinator
            decision = "accept"
            reason += " (Accepted anyway: no lower-impact alternative exists.)"
    return dict(decision=decision, reason=reason, impact_score=impact, alternatives=alts), warnings, [], "completed"


# ------------------------------------------------------------------ repair_schedule

def apply_disruptions(payload: dict, disruptions: list) -> tuple[dict, list[str]]:
    p = copy.deepcopy(payload)
    notes = []
    rooms = {r["room_id"]: r for r in p.get("rooms", [])}
    courses = {c["course_id"]: c for c in p.get("courses", [])}
    fac = {f["instructor_id"]: f for f in p.setdefault("faculty_constraints", []) or []}
    gcs = p.setdefault("global_constraints", []) or []
    p["global_constraints"] = gcs

    def times_of(d):
        out = []
        for k in ("unavailable_times", "timeslot_ids", "times"):
            out += d.get(k) or []
        if d.get("timeslot_id"):
            out.append(d["timeslot_id"])
        if {"days", "start", "end"} <= set(d):
            out.append({"days": d["days"], "start": d["start"], "end": d["end"]})
        return out

    for d in disruptions or []:
        typ = str(d.get("type", "")).lower()
        if typ in ("room_unavailable", "room_removed", "room_closed"):
            rid = d.get("room_id")
            if times_of(d):
                gcs.append({"type": "blackout", "room_id": rid, "times": times_of(d), "description": typ})
            else:
                p["rooms"] = [r for r in p["rooms"] if r["room_id"] != rid]
        elif "capacity" in typ:
            rid = d.get("room_id")
            cap = d.get("new_capacity", d.get("capacity"))
            if rid in rooms and cap is not None:
                rooms[rid]["capacity"] = cap
            else:
                notes.append(f"Disruption {d} could not be applied (unknown room or missing capacity).")
        elif typ in ("faculty_unavailable", "instructor_unavailable", "faculty_constraint_change"):
            iid = d.get("instructor_id")
            f = fac.setdefault(iid, {"instructor_id": iid, "unavailable_times": [], "preferred_times": []})
            if f not in p["faculty_constraints"]:
                p["faculty_constraints"].append(f)
            f["unavailable_times"] = list(f.get("unavailable_times") or []) + times_of(d)
        elif "enrollment" in typ:
            cid = d.get("course_id")
            val = d.get("new_enrollment", d.get("expected_enrollment", d.get("enrollment")))
            if cid in courses and val is not None:
                courses[cid]["expected_enrollment"] = val
            else:
                notes.append(f"Disruption {d} could not be applied (unknown course or missing enrollment).")
        elif typ in ("timeslot_unavailable", "timeslot_removed"):
            gone = set(times_of(d))
            p["timeslots"] = [t for t in p["timeslots"] if t["timeslot_id"] not in gone]
        elif "blackout" in typ:
            gcs.append({**d, "type": "blackout"})
        elif typ in ("course_must_move", "move_course", "course_move"):
            gcs.append({"type": "blackout", "course_id": d.get("course_id"),
                        "times": times_of(d) or [d.get("current_timeslot_id") or "__CURRENT__"],
                        "description": "course must move"})
        else:
            notes.append(f"Unrecognized disruption type '{d.get('type')}'; ignored.")
    return p, notes


def repair_schedule(payload: dict, semester: str):
    resources = {k: payload[k] for k in RESOURCE_KEYS if payload.get(k) is not None}
    if not {"courses", "timeslots", "rooms"} <= set(resources):
        cached = recall_resources(semester) or {}
        resources = {**cached, **resources}
    if not {"courses", "timeslots", "rooms"} <= set(resources):
        return None, [], ["repair_schedule needs courses/timeslots/rooms."], "failed"

    current, _ = normalize_schedule(payload.get("current_schedule"))
    # course_must_move with no explicit time: block the course's current slot
    disruptions = copy.deepcopy(payload.get("disruptions") or [])
    for d in disruptions:
        if str(d.get("type", "")).lower() in ("course_must_move", "move_course", "course_move") and d.get("course_id") in current:
            d.setdefault("current_timeslot_id", current[d["course_id"]][0])
    disrupted, notes = apply_disruptions(resources, disruptions)
    prob = prepare_problem(disrupted)
    warnings = prob.warnings + notes

    current = {c: a for c, a in current.items() if c in prob.courses}
    broken = {v["course_id"] for v in check_schedule(prob, current)} | \
             {v.get("other_course_id") for v in check_schedule(prob, current) if v.get("other_course_id")}
    sched, unscheduled, status = solve(prob, current=current)
    if check_schedule(prob, sched):
        return None, warnings, ["Internal error: repaired schedule failed validation."], "failed"

    occ = Occupancy(prob, sched)
    changes, reasons = [], {}
    for cid, (ot, orr) in current.items():
        if cid not in sched or sched[cid] == (ot, orr):
            continue
        nt, nr = sched[cid]
        why = ("Previous assignment became invalid after the disruption." if cid in broken
               else "Moved to free a slot needed by a course the disruption displaced.")
        changes.append(dict(course_id=cid, old_timeslot_id=ot, old_room_id=orr, new_timeslot_id=nt,
                            new_room_id=nr, reason=why))
    unresolved = [dict(course_id=c, reason=diagnose_unscheduled(prob, c)) for c in unscheduled]
    for c in prob.courses:
        if c not in current and c in sched:
            warnings.append(f"{c} was not in current_schedule; newly scheduled at {sched[c]}.")

    summary = (f"Repaired schedule: {len(changes)} course(s) changed"
               + (f", {len(unresolved)} could not be placed" if unresolved else "")
               + f"; {len(broken)} assignment(s) were invalidated by the disruption.")
    reply = llm.ask_json("repair_summary", {
        "disruptions": payload.get("disruptions"), "changes": changes, "unresolved": unresolved,
        "broken_by_disruption": sorted(b for b in broken if b),
    })
    if reply and isinstance(reply.get("summary"), str):
        summary = reply["summary"]
        rs = reply.get("reasons") if isinstance(reply.get("reasons"), dict) else {}
        for ch in changes:
            if isinstance(rs.get(ch["course_id"]), str):
                ch["reason"] = rs[ch["course_id"]]

    schedule = [_entry(prob, sched, occ, c) for c in prob.courses if c in sched]
    remember_resources(semester, disrupted)
    result = dict(schedule=schedule, changes=changes, change_count=len(changes), unresolved=unresolved, summary=summary)
    return result, warnings, [], ("partial" if unresolved else "completed")
