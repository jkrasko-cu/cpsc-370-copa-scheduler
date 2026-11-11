"""College of Performing Arts scheduling agent — coordinator entrypoint.

    from agent import run_agent
    response = run_agent(request_dict)

Architecture (see agent_design.md):
  parse payload -> CP-SAT solver (hard constraints by construction, soft prefs as objective)
  -> deterministic checker re-validates -> LLM explains / judges feasible trade-offs / parses
  free-form constraints -> checker validates anything the LLM touched -> JSON response.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from copa_core import ops  # noqa: E402

UNIT_NAME = "College of Performing Arts"
PROTOCOL = "1.0"
OPERATIONS = {
    "generate_schedule": ops.generate_schedule,
    "evaluate_change": ops.evaluate_change,
    "repair_schedule": ops.repair_schedule,
}


def _response(request, status, result=None, warnings=None, errors=None):
    req = request if isinstance(request, dict) else {}
    return {
        "protocol_version": PROTOCOL,
        "request_id": req.get("request_id"),
        "unit": req.get("unit") or UNIT_NAME,
        "status": status,
        "result": result if result is not None else {},
        "warnings": [str(w) for w in (warnings or [])],
        "errors": [str(e) for e in (errors or [])],
    }


def run_agent(request: dict) -> dict:
    """Process one coordinator request and return a JSON-compatible dictionary."""
    try:
        if not isinstance(request, dict):
            return _response(request, "failed", errors=["request must be a JSON object"])
        problems = []
        if request.get("protocol_version") != PROTOCOL:
            problems.append(f"unsupported protocol_version {request.get('protocol_version')!r}; expected '1.0'")
        if not request.get("request_id"):
            problems.append("missing request_id")
        op = request.get("operation")
        if op not in OPERATIONS:
            problems.append(f"unknown operation {op!r}; expected one of {sorted(OPERATIONS)}")
        if not isinstance(request.get("payload"), dict):
            problems.append("payload must be an object")
        if problems:
            return _response(request, "failed", errors=problems)

        semester = request.get("semester") or "default"
        if op == "generate_schedule":
            ops.remember_resources(semester, request["payload"])
        result, warnings, errors, status = OPERATIONS[op](request["payload"], semester)
        return _response(request, status, result, warnings, errors)
    except (KeyError, ValueError, TypeError) as e:
        return _response(request, "failed", errors=[f"malformed request: {type(e).__name__}: {e}"])
    except Exception as e:   # never crash the integration process
        return _response(request, "failed", errors=[f"internal error: {type(e).__name__}: {e}",
                                                    traceback.format_exc(limit=3)])


if __name__ == "__main__":
    import json
    req = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else json.load(sys.stdin)
    print(json.dumps(run_agent(req), indent=1))
