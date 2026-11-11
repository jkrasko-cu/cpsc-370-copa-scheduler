"""Problem model: parses a coordinator payload into normalized, indexed objects.

Everything downstream (checker, scorer, solver) works on a `Problem`, never on raw
payload dicts, so input quirks (day spellings, time formats, room-id formats) are
handled in exactly one place.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

DAY_ALIASES = {
    "M": "M", "MO": "M", "MON": "M", "MONDAY": "M",
    "T": "T", "TU": "T", "TUE": "T", "TUES": "T", "TUESDAY": "T",
    "W": "W", "WE": "W", "WED": "W", "WEDNESDAY": "W",
    "R": "R", "TH": "R", "THU": "R", "THUR": "R", "THURS": "R", "THURSDAY": "R",
    "F": "F", "FR": "F", "FRI": "F", "FRIDAY": "F",
    "S": "S", "SA": "S", "SAT": "S", "SATURDAY": "S",
    "U": "U", "SU": "U", "SUN": "U", "SUNDAY": "U",
}
DAY_ORDER = "MTWRFSU"


def norm_day(d) -> str:
    key = str(d).strip().upper()
    if key not in DAY_ALIASES:
        raise ValueError(f"unrecognized day '{d}'")
    return DAY_ALIASES[key]


def parse_days(days) -> frozenset:
    """Accepts ["M","W"], ["Tu","Th"], or a compact string like "MW" / "TR" / "TTh"."""
    if isinstance(days, str):
        tokens = re.findall(r"Th|TH|Su|SU|Sa|SA|[MTWRFSU]", days)
        days = tokens
    return frozenset(norm_day(d) for d in days)


def days_str(days) -> str:
    return "".join(d for d in DAY_ORDER if d in days)


def parse_time(s) -> int:
    """'10:30', '1030', '10:30am', '1:00 PM', 630 (int minutes passthrough) -> minutes after midnight."""
    if isinstance(s, (int, float)):
        return int(s)
    t = str(s).strip().lower().replace(" ", "")
    m = re.fullmatch(r"(\d{1,2}):?(\d{2})(am|pm)?", t)
    if not m:
        raise ValueError(f"unrecognized time '{s}'")
    h, mi, ap = int(m.group(1)), int(m.group(2)), m.group(3)
    if ap == "pm" and h != 12:
        h += 12
    if ap == "am" and h == 12:
        h = 0
    return h * 60 + mi


def fmt_time(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def room_key(room_id: str) -> str:
    """Normalize 'BH 109', 'BH-109', 'bh109' to the same key for history lookups."""
    return re.sub(r"[^A-Z0-9]", "", str(room_id).upper())


def building_of(room_id: str) -> str:
    """'BH 109' / 'BH-109' -> 'BH'; 'SWE150' -> 'SWE' (leading letters when there is no separator)."""
    s = str(room_id).strip().upper()
    head = re.split(r"[\s\-_]", s)[0]
    if head != s:
        return head
    m = re.match(r"[A-Z]+", s)
    return m.group(0) if m else s


def catalog_key(course_id: str) -> str:
    """'MUS-307-01' -> 'MUS-307' (the section-independent course key used by history)."""
    parts = str(course_id).split("-")
    return "-".join(parts[:2]).upper() if len(parts) >= 2 else str(course_id).upper()


@dataclass(frozen=True)
class Interval:
    days: frozenset
    start: int
    end: int

    def overlaps(self, other: "Interval") -> bool:
        return bool(self.days & other.days) and self.start < other.end and other.start < self.end


@dataclass
class Timeslot:
    id: str
    days: frozenset
    start: int
    end: int

    @property
    def interval(self) -> Interval:
        return Interval(self.days, self.start, self.end)

    @property
    def minutes(self) -> int:
        return self.end - self.start


@dataclass
class Room:
    id: str
    capacity: int
    features: set


@dataclass
class Course:
    id: str
    title: str
    instructors: list           # one or more instructor ids (team teaching supported)
    enrollment: int
    required_features: set
    preferred_times: set
    avoid_times: set
    conflicts_with: set
    raw: dict = field(default_factory=dict)

    @property
    def hist_key(self) -> str:
        return catalog_key(self.id)


@dataclass
class Block:
    """A time window in which something may not be scheduled.
    scope: 'global' | 'course' | 'room' | 'instructor'; target is the id (None for global)."""
    scope: str
    target: str | None
    interval: Interval
    source: str


class Problem:
    def __init__(self, courses, timeslots, rooms, faculty_unavail, faculty_pref, blocks, warnings):
        self.courses: dict[str, Course] = courses
        self.timeslots: dict[str, Timeslot] = timeslots
        self.rooms: dict[str, Room] = rooms
        self.faculty_unavail: dict[str, list[Interval]] = faculty_unavail
        self.faculty_pref: dict[str, set] = faculty_pref
        self.blocks: list[Block] = blocks
        self.warnings: list[str] = warnings
        self._overlap = {
            a: {b for b in timeslots if timeslots[a].interval.overlaps(timeslots[b].interval)}
            for a in timeslots
        }
        # symmetric conflict pairs; references to courses outside the unit are kept (they
        # can't be checked locally) but reported as warnings
        self.conflict_pairs: set[frozenset] = set()
        for c in courses.values():
            for other in c.conflicts_with:
                if other in courses:
                    self.conflict_pairs.add(frozenset((c.id, other)))
                else:
                    warnings.append(f"{c.id}: conflicts_with references '{other}', which is not in this unit's course list; cannot enforce locally")
        self.conflicts_of: dict[str, set] = {cid: set() for cid in courses}
        for pair in self.conflict_pairs:
            a, b = tuple(pair)
            self.conflicts_of[a].add(b)
            self.conflicts_of[b].add(a)
        self.courses_of_instructor: dict[str, list] = {}
        for c in courses.values():
            for i in c.instructors:
                self.courses_of_instructor.setdefault(i, []).append(c.id)

    def overlaps(self, t1: str, t2: str) -> bool:
        return t2 in self._overlap.get(t1, ())

    # ---- static (schedule-independent) feasibility of one assignment ----
    def static_violations(self, cid: str, tid: str, rid: str) -> list[dict]:
        out = []
        c, t, r = self.courses[cid], self.timeslots[tid], self.rooms[rid]
        if c.enrollment > r.capacity:
            out.append(dict(constraint="room_capacity", course_id=cid,
                            detail=f"expected enrollment {c.enrollment} exceeds capacity {r.capacity} of {rid}"))
        missing = c.required_features - r.features
        if missing:
            out.append(dict(constraint="room_features", course_id=cid,
                            detail=f"{rid} lacks required feature(s): {', '.join(sorted(missing))}"))
        for i in c.instructors:
            for iv in self.faculty_unavail.get(i, []):
                if iv.overlaps(t.interval):
                    out.append(dict(constraint="faculty_unavailable", course_id=cid,
                                    detail=f"instructor {i} is unavailable during {tid}"))
                    break
        for b in self.blocks:
            if not b.interval.overlaps(t.interval):
                continue
            if (b.scope == "global"
                    or (b.scope == "course" and b.target == cid)
                    or (b.scope == "room" and b.target == rid)
                    or (b.scope == "instructor" and b.target in c.instructors)):
                out.append(dict(constraint="blocked_time", course_id=cid,
                                detail=f"{tid} falls in a blocked period ({b.source})"))
        return out

    def static_ok(self, cid, tid, rid) -> bool:
        return not self.static_violations(cid, tid, rid)

    def to_payload_parts(self) -> dict:
        """Serialize resources back to payload form (used for the agent's resource cache)."""
        return dict(
            courses=[c.raw for c in self.courses.values()],
            timeslots=[dict(timeslot_id=t.id, days=sorted(t.days, key=DAY_ORDER.index),
                            start=fmt_time(t.start), end=fmt_time(t.end)) for t in self.timeslots.values()],
            rooms=[dict(room_id=r.id, capacity=r.capacity, features=sorted(r.features)) for r in self.rooms.values()],
        )


# ------------------------------------------------------------------ parsing

def _as_list(x):
    if x is None:
        return []
    return list(x) if isinstance(x, (list, tuple, set)) else [x]


def _interval_from_ref(ref, timeslots: dict[str, Timeslot]) -> Interval | None:
    """A time reference may be a timeslot id or a dict {days,start,end}."""
    if isinstance(ref, str):
        t = timeslots.get(ref)
        return t.interval if t else None
    if isinstance(ref, dict) and {"days", "start", "end"} <= set(ref):
        return Interval(parse_days(ref["days"]), parse_time(ref["start"]), parse_time(ref["end"]))
    if isinstance(ref, dict) and "timeslot_id" in ref:
        return _interval_from_ref(ref["timeslot_id"], timeslots)
    return None


def parse_global_constraint(gc, timeslots) -> tuple[list[Block], str | None]:
    """Recognize structured hard constraints. Returns (blocks, None) or ([], reason-unrecognized)."""
    if not isinstance(gc, dict):
        return [], f"unrecognized global constraint (not an object): {gc!r}"
    gtype = str(gc.get("type", "")).lower()
    scope, target = "global", None
    for key, sc in (("course_id", "course"), ("room_id", "room"), ("instructor_id", "instructor")):
        if gc.get(key):
            scope, target = sc, gc[key]
    refs = []
    if {"days", "start", "end"} <= set(gc):
        refs.append({"days": gc["days"], "start": gc["start"], "end": gc["end"]})
    refs += _as_list(gc.get("timeslot_id")) + _as_list(gc.get("timeslot_ids")) + _as_list(gc.get("times"))
    blockish = any(w in gtype for w in ("blackout", "unavailable", "block", "no_class", "closed", "forbid"))
    if blockish and refs:
        blocks = []
        for ref in refs:
            iv = _interval_from_ref(ref, timeslots)
            if iv is None:
                return [], f"global constraint references unknown time {ref!r}"
            blocks.append(Block(scope, target, iv, gc.get("description") or gtype or "global constraint"))
        return blocks, None
    if blockish and scope != "global" and not refs:   # e.g. room unavailable for the whole term
        return [Block(scope, target, Interval(frozenset(DAY_ORDER), 0, 24 * 60), gtype)], None
    return [], f"unrecognized global constraint: {gc!r}"


def build_problem(payload: dict) -> Problem:
    warnings: list[str] = []
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    for key in ("courses", "timeslots", "rooms"):
        if not isinstance(payload.get(key), list):
            raise ValueError(f"payload.{key} must be a list")

    timeslots = {}
    for ts in payload["timeslots"]:
        tid = ts["timeslot_id"]
        t = Timeslot(tid, parse_days(ts["days"]), parse_time(ts["start"]), parse_time(ts["end"]))
        if t.end <= t.start:
            raise ValueError(f"timeslot {tid} ends before it starts")
        if tid in timeslots:
            warnings.append(f"duplicate timeslot id {tid}; keeping the first")
            continue
        timeslots[tid] = t

    rooms = {}
    for r in payload["rooms"]:
        rid = r["room_id"]
        if rid in rooms:
            warnings.append(f"duplicate room id {rid}; keeping the first")
            continue
        rooms[rid] = Room(rid, int(r.get("capacity") or 0), {str(f).lower() for f in _as_list(r.get("features"))})

    courses = {}
    for c in payload["courses"]:
        cid = c["course_id"]
        if cid in courses:
            warnings.append(f"duplicate course id {cid}; keeping the first")
            continue
        instructors = [i for i in _as_list(c.get("instructor_id")) + _as_list(c.get("co_instructor_ids")) if i]
        for key in ("preferred_times", "avoid_times"):
            for ref in _as_list(c.get(key)):
                if isinstance(ref, str) and ref not in timeslots:
                    warnings.append(f"{cid}: {key} references unknown timeslot {ref}; ignored")
        courses[cid] = Course(
            id=cid,
            title=c.get("title", ""),
            instructors=list(dict.fromkeys(instructors)),
            enrollment=int(c.get("expected_enrollment") or 0),
            required_features={str(f).lower() for f in _as_list(c.get("required_features"))},
            preferred_times={x for x in _as_list(c.get("preferred_times")) if x in timeslots},
            avoid_times={x for x in _as_list(c.get("avoid_times")) if x in timeslots},
            conflicts_with=set(_as_list(c.get("conflicts_with"))),
            raw=c,
        )

    faculty_unavail, faculty_pref = {}, {}
    for fc in payload.get("faculty_constraints") or []:
        iid = fc.get("instructor_id")
        if not iid:
            continue
        for ref in _as_list(fc.get("unavailable_times")):
            iv = _interval_from_ref(ref, timeslots)
            if iv is None:
                warnings.append(f"faculty {iid}: unavailable time {ref!r} not understood; ignored")
            else:
                faculty_unavail.setdefault(iid, []).append(iv)
        faculty_pref.setdefault(iid, set()).update(x for x in _as_list(fc.get("preferred_times")) if x in timeslots)

    blocks, unparsed = [], []
    for gc in payload.get("global_constraints") or []:
        b, err = parse_global_constraint(gc, timeslots)
        if err:
            unparsed.append(gc)
        blocks += b

    prob = Problem(courses, timeslots, rooms, faculty_unavail, faculty_pref, blocks, warnings)
    prob.unparsed_constraints = unparsed
    prob.raw_payload = payload
    return prob
