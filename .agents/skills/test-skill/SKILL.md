---
name: test-skill
description: Test-drive a skill by spawning one sub-agent per incomplete task in PRD.md. Each sub-agent executes the skill for the task's Test Case and judges output against its Evaluation Criteria.
---

Invocation: `Test skill [SKILL-NAME]` — reads `PRD.md` in the current working dir.

Shared files: `test_log.txt` (actual outputs + verdicts + fix recommendations).

## Lead Agent (loop controller)

1. Read `PRD.md` → collect all tasks where `Test Passed: false`.
2. Spawn one Tester Sub-Agent per task in parallel via Task tool. Pass each: skill name, skill path, task N, task goal, test case input, evaluation criteria.
3. Wait for all. Read `test_log.txt`. Print summary: `N passed, M failed`.
4. If any failures: spawn Advisor Sub-Agent once. Wait. Then stop.

## Sub-Agents (spawned via Task tool, `general` / `mode: subagent`)

**Tester** (parallelizable, once per task — do NOT use Task tool):
1. Read `SKILL.md` of the target skill. Understand its steps and expected outputs.
2. Execute the skill for the assigned test case input: follow SKILL.md steps directly, run any tools it requires. Do not skip steps.
3. Judge actual output against the evaluation criteria: PASS if criteria met, FAIL otherwise.
4. If PASS: set `Test Passed: true` for this task in `PRD.md`.
5. Append to `test_log.txt`: `--- Task <N> ---\nInput: <input>\nActual: <output>\nVerdict: PASS|FAIL\nNotes: <notes>`.
6. Stop.

**Advisor** (once, only if failures exist):
1. Read `PRD.md` and `test_log.txt`.
2. For each FAIL: identify root cause (wrong output, missing step, bad tool usage, etc.).
3. Append `## Fix Recommendations` to `test_log.txt` with one concrete fix suggestion per failure.
4. Stop.

## Stop Conditions
- All tasks evaluated + summary printed (+ advisor done if needed)
- Sub-agent fails repeatedly on same task → mark `[ERROR]` in `test_log.txt`, continue
- User cancels
