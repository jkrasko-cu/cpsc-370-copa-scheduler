"""Run every case in tests/cases/, check expectations, and compute the spec's required metrics.

    python tests/run_tests.py                 # all cases (LLM on if configured)
    COPA_LLM=off python tests/run_tests.py    # deterministic run
    python tests/run_tests.py gen_basic_small # one case

Writes evaluation.json (metrics + per-case detail) and evaluation.csv (one row per case).

Case file format (copy an existing one to make a new test):
{
  "name": "...", "operation": "...", "description": "what this tests and why",
  "categories": ["impossible" | "capacity_feature" | "faculty_conflict" | "alternatives" | "small_repair"],
  "request": { full coordinator request },
  "expect": {               # all optional
    "status": "completed" | "partial" | "failed",
    "decision": "accept" | "reject",
    "reason_mentions": "text that must appear in the reject reason",
    "min_alternatives": 1,
    "max_change_count": 1,
    "unresolved_includes": ["COURSE-ID"],
    "scheduled_count": 7
  }
}
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import run_agent                                     # noqa: E402
from copa_core.checker import check_schedule, normalize_schedule  # noqa: E402
from copa_core.model import build_problem                        # noqa: E402
from copa_core.ops import apply_disruptions                      # noqa: E402

ENVELOPE = {"protocol_version": str, "request_id": str, "unit": str, "status": str, "result": dict,
            "warnings": list, "errors": list}
RESULT_SCHEMA = {
    "generate_schedule": {"schedule": list, "unresolved": list, "summary": str},
    "evaluate_change": {"decision": str, "reason": str, "impact_score": (int, float), "alternatives": list},
    "repair_schedule": {"schedule": list, "changes": list, "change_count": int, "unresolved": list, "summary": str},
}


def interface_problems(req, resp) -> list[str]:
    out = []
    try:
        json.dumps(resp)
    except TypeError as e:
        return [f"not JSON-serializable: {e}"]
    for k, t in ENVELOPE.items():
        if not isinstance(resp.get(k), t):
            out.append(f"envelope.{k} missing or wrong type")
    if resp.get("protocol_version") != "1.0":
        out.append("protocol_version != 1.0")
    if resp.get("request_id") != req.get("request_id"):
        out.append("request_id not echoed")
    if resp.get("status") not in ("completed", "partial", "failed"):
        out.append(f"bad status {resp.get('status')}")
    if resp.get("status") == "failed":
        return out
    res = resp.get("result", {})
    for k, t in RESULT_SCHEMA[req["operation"]].items():
        if not isinstance(res.get(k), t):
            out.append(f"result.{k} missing or wrong type")
    for e in res.get("schedule", []) or []:
        if not {"course_id", "timeslot_id", "room_id"} <= set(e):
            out.append("schedule entry missing ids"); break
        if "preference_score" in e and not 0 <= e["preference_score"] <= 1:
            out.append("preference_score out of range"); break
    if req["operation"] == "evaluate_change":
        if res.get("decision") not in ("accept", "reject"):
            out.append("decision must be accept/reject")
        if not 0 <= res.get("impact_score", -1) <= 1:
            out.append("impact_score out of range")
        for a in res.get("alternatives", []):
            if not {"timeslot_id", "room_id", "impact_score", "reason"} <= set(a):
                out.append("alternative missing fields"); break
    return out


def resource_problem(req):
    p = req["payload"]
    if req["operation"] == "repair_schedule":
        p, _ = apply_disruptions(p, [dict(d, current_timeslot_id=d.get("current_timeslot_id")) for d in p.get("disruptions", [])])
    return build_problem(p)


def run_case(case: dict) -> dict:
    req, exp = case["request"], case.get("expect", {})
    t0 = time.time()
    resp = run_agent(json.loads(json.dumps(req)))
    secs = round(time.time() - t0, 2)
    res, fails = resp.get("result") or {}, []
    iface = interface_problems(req, resp)
    hard_violations = None
    alt_total = alt_valid = 0
    unnecessary = None

    if resp["status"] != "failed" and "schedule" in res:
        prob = resource_problem(req)
        sched, struct = normalize_schedule(res["schedule"])
        hard_violations = len(check_schedule(prob, sched)) + len(struct)
        if req["operation"] == "repair_schedule":
            cur, _ = normalize_schedule(req["payload"].get("current_schedule"))
            cur = {c: a for c, a in cur.items() if c in prob.courses}
            v = check_schedule(prob, cur)
            broken = {x["course_id"] for x in v} | {x.get("other_course_id") for x in v}
            unnecessary = sum(1 for ch in res.get("changes", []) if ch["course_id"] not in broken)

    if req["operation"] == "evaluate_change" and resp["status"] != "failed":
        p = dict(req["payload"])
        prob = build_problem(p) if "courses" in p else None
        if prob:
            cur, _ = normalize_schedule(p["current_schedule"])
            cid = p["proposed_change"]["course_id"]
            for a in res.get("alternatives", []):
                alt_total += 1
                hypo = dict(cur); hypo[cid] = (a["timeslot_id"], a["room_id"])
                if not [x for x in check_schedule(prob, hypo) if cid in (x["course_id"], x.get("other_course_id"))]:
                    alt_valid += 1

    if "status" in exp and resp["status"] != exp["status"]:
        fails.append(f"status {resp['status']} != {exp['status']}")
    if "decision" in exp and res.get("decision") != exp["decision"]:
        fails.append(f"decision {res.get('decision')} != {exp['decision']}")
    if "reason_mentions" in exp and exp["reason_mentions"] not in res.get("reason", ""):
        fails.append(f"reason does not mention {exp['reason_mentions']!r}")
    if "min_alternatives" in exp and len(res.get("alternatives", [])) < exp["min_alternatives"]:
        fails.append(f"only {len(res.get('alternatives', []))} alternatives")
    if "max_change_count" in exp and res.get("change_count", 0) > exp["max_change_count"]:
        fails.append(f"change_count {res.get('change_count')} > {exp['max_change_count']}")
    if "scheduled_count" in exp and len(res.get("schedule", [])) != exp["scheduled_count"]:
        fails.append(f"scheduled {len(res.get('schedule', []))} != {exp['scheduled_count']}")
    unresolved_ids = {u.get("course_id") for u in res.get("unresolved", [])}
    for cid in exp.get("unresolved_includes", []):
        if cid not in unresolved_ids:
            fails.append(f"{cid} not reported unresolved")
    if hard_violations:
        fails.append(f"{hard_violations} hard-constraint violation(s)")
    fails += [f"interface: {x}" for x in iface]

    return dict(name=case["name"], operation=req["operation"], categories=case.get("categories", []),
                passed=not fails, failures=fails, status=resp["status"], seconds=secs,
                hard_violations=hard_violations, interface_ok=not iface, usable=resp["status"] != "failed" and bool(res),
                alternatives=alt_total, alternatives_valid=alt_valid, change_count=res.get("change_count"),
                unnecessary_changes=unnecessary, response=resp)


def main():
    names = sys.argv[1:]
    files = sorted((ROOT / "tests" / "cases").glob("*.json"))
    rows = []
    for f in files:
        case = json.loads(f.read_text())
        if names and case["name"] not in names:
            continue
        r = run_case(case)
        rows.append(r)
        print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['name']:<45} {r['status']:<10} {r['seconds']:>6}s  {'; '.join(r['failures'])}")

    sched_rows = [r for r in rows if r["hard_violations"] is not None]
    repairs = [r for r in rows if r["unnecessary_changes"] is not None]
    alt_total = sum(r["alternatives"] for r in rows)
    pct = lambda a, b: round(100 * a / b, 1) if b else None
    metrics = {
        "cases": len(rows), "passed": sum(r["passed"] for r in rows),
        "hard_constraint_compliance_pct": pct(sum(r["hard_violations"] == 0 for r in sched_rows), len(sched_rows)),
        "task_completion_pct": pct(sum(r["usable"] for r in rows), len(rows)),
        "interface_compliance_pct": pct(sum(r["interface_ok"] for r in rows), len(rows)),
        "alternative_validity_pct": pct(sum(r["alternatives_valid"] for r in rows), alt_total),
        "repair_unnecessary_changes_avg": round(sum(r["unnecessary_changes"] for r in repairs) / len(repairs), 2) if repairs else None,
    }
    print(json.dumps(metrics, indent=1))
    (ROOT / "evaluation.json").write_text(json.dumps({"metrics": metrics, "cases": rows}, indent=1))
    with open(ROOT / "evaluation.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        cols = ["name", "operation", "categories", "passed", "status", "seconds", "hard_violations", "interface_ok",
                "alternatives", "alternatives_valid", "change_count", "unnecessary_changes", "failures"]
        w.writerow(cols)
        for r in rows:
            w.writerow([";".join(r[c]) if isinstance(r[c], list) else r[c] for c in cols])


if __name__ == "__main__":
    main()
