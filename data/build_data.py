"""Turn the registrar CSV exports into (1) the agent's historical-preference table and
(2) realistic practice payloads in the coordinator's format.

Usage (from the project root):
    python data/build_data.py

Outputs:
    data/history_stats.json                    used by the agent at runtime (soft preferences only)
    tests/cases/practice_fall2026.json         a full generate_schedule request built from Fall 2026

What counts as a schedulable COPA section (everything else is out of scope; see agent_design.md):
  * Acad Group == COPA, Class Stat != Cancelled
  * has a real meeting time (not the 12:00am async/TBA placeholder) and an on-campus room
  * Private lessons / coaching / independent study have no room in the data (studio-based,
    arranged individually) and are excluded.
Rows that share the same days/time/room are one physical meeting (team teaching or combined
cross-listed sections) and are merged into one schedulable unit.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from copa_core.model import parse_time, fmt_time, room_key, building_of  # noqa: E402

DAY_COLS = [("M", "M"), ("T", "T"), ("W", "W"), ("Th", "R"), ("Fri", "F"), ("S", "S"), ("Su", "U")]

# ---- Room-feature ASSUMPTIONS (the CSV has no feature column) -----------------------------
# Derived from which subject uses each building and what is taught there. Verify with the
# department; the coordinator's own feature list always wins at runtime.
ROOM_FEATURES = {
    "building": {"CPH": ["dance_studio"]},
    "room": {
        "OH 103": ["rehearsal_hall", "piano"], "BH 100": ["rehearsal_hall", "piano", "stage"],
        "BH 109": ["rehearsal_hall", "piano"], "OH B01": ["music_tech", "computer"],
        "BH 223": ["piano_lab"], "MC 149": ["theatre_studio"], "MC 230": ["theatre_studio"],
        "ETC 103": ["theatre_studio"], "ETC 112": ["theatre_studio"],
    },
}
SPECIALIZED = {"dance_studio", "rehearsal_hall", "music_tech", "piano_lab", "theatre_studio"}

# Explicit course-pair conflicts (students take these together). Real data has none, so these
# are illustrative for the practice case; the coordinator supplies real ones.
CONFLICT_PAIRS = [("MUS-133", "MUS-135"), ("MUS-233", "MUS-235")]


def room_features(room: str) -> list[str]:
    return sorted(set(ROOM_FEATURES["room"].get(room, [])) | set(ROOM_FEATURES["building"].get(building_of(room), [])))


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="cp1252", dtype=str)
    df = df[(df["Acad Group"] == "COPA") & (df["Class Stat"] != "Cancelled")].copy()
    df["days"] = df.apply(lambda r: "".join(code for col, code in DAY_COLS if pd.notna(r.get(col))), axis=1)
    keep = (df["Bldng/Room"].notna() & (df["Bldng/Room"] != "OFF CAMPUS") & df["Start Time"].notna()
            & (df["Start Time"] != "12:00am") & (df["days"] != ""))
    df = df[keep].copy()
    df["start"] = df["Start Time"].map(parse_time)
    df["end"] = df["End Time"].map(parse_time)
    df["course_id"] = df["Subject"].str.strip() + "-" + df["Catalog"].str.strip() + "-" + df["Sec"].str.strip()
    return df


def meetings(df: pd.DataFrame) -> list[dict]:
    """Merge rows into physical meetings keyed by (days, start, end, room)."""
    out = []
    for (days, start, end, room), g in df.groupby(["days", "start", "end", "Bldng/Room"], sort=True):
        ids = sorted(g["course_id"].unique())
        per_section = g.drop_duplicates("course_id")
        out.append(dict(
            course_ids=ids, title=per_section.iloc[0]["Descr"].strip(), days=days, start=int(start), end=int(end),
            room=room, room_cap=int(float(g["Room Cap"].iloc[0])),
            enrollment=int(per_section["Enrl Tot"].astype(float).sum()),
            instr_ids=sorted({i for i in g["Instr ID"].dropna()}),
        ))
    return out


def history_stats(dfs: list[pd.DataFrame]) -> dict:
    stats = defaultdict(lambda: {"patterns": Counter(), "rooms": Counter(), "buildings": Counter()})
    for df in dfs:
        for m in meetings(df):
            for cid in m["course_ids"]:
                key = "-".join(cid.split("-")[:2]).upper()
                s = stats[key]
                s["patterns"][(m["days"], m["start"], m["end"] - m["start"])] += 1
                s["rooms"][room_key(m["room"])] += 1
                s["buildings"][building_of(m["room"])] += 1
    return {
        k: {
            "patterns": [dict(days=d, start=fmt_time(st), minutes=mins, count=n)
                         for (d, st, mins), n in v["patterns"].most_common()],
            "rooms": dict(v["rooms"].most_common()),
            "buildings": dict(v["buildings"].most_common()),
        } for k, v in sorted(stats.items())
    }


def slot_id(days: str, start: int, end: int, taken: set) -> str:
    base = f"{days}-{start // 60:02d}{start % 60:02d}"
    sid = base if base not in taken else f"{base}-{end // 60:02d}{end % 60:02d}"
    taken.add(sid)
    return sid


def practice_payload(fall: pd.DataFrame, spring: pd.DataFrame, min_pattern_count: int = 2) -> dict:
    ms = meetings(fall)

    # timeslot menu = meeting patterns COPA actually used at least twice
    pat_counts = Counter((m["days"], m["start"], m["end"]) for m in ms)
    taken, timeslots, slot_by_pat = set(), [], {}
    for (days, start, end), n in sorted(pat_counts.items(), key=lambda kv: ("MTWRFSU".index(kv[0][0][0]), kv[0][1], kv[0][0])):
        if n < min_pattern_count:
            continue
        sid = slot_id(days, start, end, taken)
        slot_by_pat[(days, start, end)] = sid
        timeslots.append(dict(timeslot_id=sid, days=list(days), start=fmt_time(start), end=fmt_time(end)))

    rooms = {}
    for m in ms:
        rooms.setdefault(m["room"], dict(room_id=m["room"], capacity=m["room_cap"], features=room_features(m["room"])))

    # anonymize instructor ids (the registrar's ids are internal; don't ship them to an LLM API)
    all_ids = sorted({i for m in ms for i in m["instr_ids"]} | set(spring["Instr ID"].dropna()))
    anon = {real: f"FAC-{n:03d}" for n, real in enumerate(all_ids, 1)}

    pair_lookup = defaultdict(set)
    for a, b in CONFLICT_PAIRS:
        pair_lookup[a].add(b)
        pair_lookup[b].add(a)

    courses, used_ids = [], set()
    for m in ms:
        cid = m["course_ids"][0]
        while cid in used_ids:            # same section meeting at two different times
            cid += "x"
        used_ids.add(cid)
        instr = [anon[i] for i in m["instr_ids"]]
        feats = sorted(set(room_features(m["room"])) & SPECIALIZED)
        c = dict(course_id=cid, title=m["title"], instructor_id=instr[0] if instr else None,
                 expected_enrollment=max(1, m["enrollment"]), required_features=feats,
                 preferred_times=[], avoid_times=[], conflicts_with=[])
        if len(instr) > 1:
            c["co_instructor_ids"] = instr[1:]
        if len(m["course_ids"]) > 1:
            c["combined_sections"] = m["course_ids"]
        courses.append(c)
    # resolve illustrative conflict pairs onto real section ids
    by_key = defaultdict(list)
    for c in courses:
        by_key["-".join(c["course_id"].split("-")[:2])].append(c["course_id"])
    for c in courses:
        key = "-".join(c["course_id"].split("-")[:2])
        for other_key in pair_lookup.get(key, ()):
            c["conflicts_with"] += by_key.get(other_key, [])

    # faculty preferences inferred from the instructor's Spring 2026 meeting patterns
    faculty = []
    spring_pref = defaultdict(set)
    for m in meetings(spring):
        for i in m["instr_ids"]:
            sid = slot_by_pat.get((m["days"], m["start"], m["end"]))
            if sid:
                spring_pref[anon.get(i)].add(sid)
    for fid in sorted({c["instructor_id"] for c in courses if c["instructor_id"]}):
        faculty.append(dict(instructor_id=fid, unavailable_times=[], preferred_times=sorted(spring_pref.get(fid, []))))

    return dict(courses=courses, timeslots=timeslots, rooms=sorted(rooms.values(), key=lambda r: r["room_id"]),
                faculty_constraints=faculty, global_constraints=[], context={"source": "Fall 2026 registrar export"})


def main():
    fall = load(ROOT / "data" / "Fall2026.csv")
    spring = load(ROOT / "data" / "Spring2026.csv")

    hist = history_stats([spring, fall])
    (ROOT / "data" / "history_stats.json").write_text(json.dumps(hist, indent=1))

    payload = practice_payload(fall, spring)
    req = {"protocol_version": "1.0", "request_id": "practice-fall2026", "operation": "generate_schedule",
           "unit": "College of Performing Arts", "semester": "Fall 2026", "payload": payload}
    out = ROOT / "tests" / "cases" / "practice_fall2026.json"
    case = {"name": "practice_fall2026", "operation": "generate_schedule", "categories": ["real_data"],
            "description": "All schedulable COPA sections from the Fall 2026 export, on COPA's own rooms and "
                           "common meeting patterns. Realistic-scale smoke test.",
            "request": req, "expect": {"status": "completed"}}
    out.write_text(json.dumps(case, indent=1))
    print(f"history: {len(hist)} courses | practice: {len(payload['courses'])} courses, "
          f"{len(payload['timeslots'])} timeslots, {len(payload['rooms'])} rooms, "
          f"{len(payload['faculty_constraints'])} instructors -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
