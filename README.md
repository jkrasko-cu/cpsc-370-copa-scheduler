# College of Performing Arts — Scheduling Agent (CPSC 370)

Academic-unit agent for COPA (Music, Dance, Theatre) in the Agentic Scheduling Challenge.

## Setup

```bash
pip install -r requirements.txt
python data/build_data.py            # rebuild history + practice case from the CSVs (already done)
COPA_LLM=off python tests/run_tests.py   # run all test cases, writes evaluation.json / evaluation.csv
python agent.py tests/some_request.json  # run one raw request
```

LLM: the agent auto-detects `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`, or the starter code can inject its
own client with `copa_core.llm.set_llm(fn)` where `fn(system, user) -> str`. With no LLM it still runs and
falls back to deterministic text, so the hard-constraint guarantees never depend on the LLM.

## Layout

| Path | What it is |
|---|---|
| `agent.py` | `run_agent(request) -> dict`: envelope validation, dispatch, never raises |
| `copa_core/model.py` | Parses payloads (day/time/room-id normalization, overlap math, blackouts) |
| `copa_core/checker.py` | **Deterministic hard-constraint checker**, the source of truth for validity |
| `copa_core/solver.py` | OR-Tools CP-SAT model for generate + repair (minimal-change objective) |
| `copa_core/scoring.py` | Soft-preference scoring, `preference_score`, rationale text |
| `copa_core/ops.py` | The three operations |
| `copa_core/llm.py` | Swappable LLM call with JSON parsing + fallback |
| `prompts/` | LLM prompts (editable without touching code) |
| `config/weights.json` | Soft-preference weights, repair costs, accept threshold |
| `data/build_data.py` | CSV → `history_stats.json` + practice payload |
| `tests/cases/` | One JSON file per test case (format documented in `tests/run_tests.py`) |

## How it works (short)

1. **Solver places, checker verifies.** CP-SAT only creates variables for statically valid
   (course, timeslot, room) triples and enforces room, instructor and conflict-pair overlaps exactly,
   even when the coordinator's timeslots overlap each other irregularly. Every result is re-checked by
   `checker.py` before it is returned.
2. **LLM does judgment, not validity.** It (a) translates free-text `global_constraints` into structured
   blackouts, which are then validated, (b) decides accept/reject on *feasible* proposed changes,
   (c) writes summaries and change explanations. An LLM rejection with no better alternative is overridden
   to accept, so the coordinator is never stalled.
3. **History is soft only.** `history_stats.json` (Spring + Fall 2026) gives per-course meeting patterns
   and rooms that add preference points, and never make an infeasible option valid.
4. **Repair = same model + change penalties.** Moving a course costs 1000 (room) / 1200 (time) points, so
   the solver only moves what the disruption forces.
5. **evaluate_change remembers resources.** The spec's evaluate payload has no rooms/timeslots, so the
   agent caches the last envelope it saw per semester (`data/state/`).

## Scope decisions / assumptions (verify these!)

- **Schedulable = has a room and a real meeting time.** About 420 Fall rows are private lessons and
  coaching with no room; these are excluded. Cancelled sections, `OFF CAMPUS`, and the 12:00am TBA
  placeholders are excluded too.
- Rows with the same days/time/room are merged into one meeting (team teaching and combined
  cross-listed sections, e.g. DANC 113-09 + DANC 313-09).
- **Room features are inferred**, since the CSV has none: `CPH` = dance_studio; `OH 103`, `BH 100` and
  `BH 109` = rehearsal_hall; `OH B01` = music_tech; `MC 149`, `MC 230`, `ETC 103` and `ETC 112` =
  theatre_studio; `BH 223` = piano_lab. See `ROOM_FEATURES` in `data/build_data.py`.
- Instructor IDs are anonymized to `FAC-###` in practice data before anything reaches an LLM.
- Theory I/Aural Skills I (and III) conflict pairs in the practice case are illustrative, not from data.
- Team teaching: optional `co_instructor_ids` on a course (a small extension; it is ignored if absent).

## Current results (`COPA_LLM=off`)

6 cases, 6 passing; 100% hard-constraint, interface and alternative validity; 0 unnecessary repair
changes. The full Fall 2026 practice case (217 meetings, 37 slots, 32 rooms) schedules 217/217 in about 25 s.

## TODO

- [ ] 7+ more test cases to reach the spec's mix (4 per op, 3 impossible, 2 capacity/feature,
      2 faculty conflict, 2 alternatives, 1 small repair). Copy an existing file in `tests/cases/`.
- [ ] Verify the room-feature map with someone who knows the buildings.
- [ ] Run with a real LLM and read the summaries and evaluate decisions.
- [ ] `agent_design.md` (2–4 pages) and the final reflection.
- [ ] Evaluate alternatives that move *one other* COPA course (currently others are held fixed).
- [ ] Solver hits its 20 s limit on the full case (valid but maybe not optimal); tune if integration has a timeout.
