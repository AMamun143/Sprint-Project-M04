---
name: literature-review
description: Write a literature review by iterating over papers one at a time, accumulating summaries and themes across iterations.
---

Shared files: `progress.txt` (done/remaining), `learning.txt` (cross-paper insights), `research_note.md` (summaries+themes), `synthesis.md` (final synthesis).

## Lead Agent (loop controller — no reading/writing of papers)
1. Init (first run only): copy `./agents/skills/templates/{progress,learning,research_note,synthesis}.*` into working dir if not exist. List all PDFs; populate `progress.txt` with unchecked entries.
2. Spawn Abstract Extraction Sub-Agent per paper in parallel via Task tool. Wait for all.
3. Spawn Planning Sub-Agent once. Wait.
4. Spawn Reading Sub-Agent for next unreviewed paper. Wait. Repeat until all done.
5. Spawn Synthesis Sub-Agent once. Then stop.

## Sub-Agents (all spawned via Task tool, `general`/`mode: subagent`)

Abstract Extraction (parallelizable, once per paper):
1. Run `uv run python3 tools/extract_pdf.py <input.pdf>`.
2. Append `filename.pdf: abstract text` to `abstracts.txt`. Append only; do not read.

Planning (once):
1. Read `abstracts.txt`.
2. Reorder `progress.txt` entries foundational-first. Do not mark any paper done.

Reading (sequential, once per paper):
1. Read `progress.txt` → pick next unreviewed paper.
2. Read `learning.txt` for accumulated themes/gaps.
3. Extract paper: `uv run python3 tools/extract_pdf.py <input.pdf>`.
4. Read `templates/research_note.md` for format; append new entry to `research_note.md`. Append only; do not read `research_note.md`.
5. Append to `learning.txt` (format from `templates/learning.txt`). Note contradictions; avoid restating existing themes.
6. Mark paper done in `progress.txt` with one-line note.

Synthesis (once, after all papers done):
1. Read `research_note.md` and `learning.txt`.
2. Read `templates/synthesis.md` for format.
3. Overwrite `synthesis.md` with all three sections: Synthesis, Contradictions, Open Questions.

Writing style (all sub-agents): 
- concise by sacrificing grammer but avoid ambiguity
- define jargon
- LaTeX for equations (display mode for key ones)
- max 3 sentences/paragraph
- paragraph-based narrative
- no bullets/tables
- conversational tone

Writing style (learning.txt):
- speak aloud your thought process as you read each paper. 
- keep the reflection conncise and focused on insights/themes/gaps.
- if any issue in tool usage, note it here and move on.

## Stop Conditions
- All papers reviewed + `synthesis.md` written
- Sub-agent fails repeatedly → escalate to user
- User cancels
