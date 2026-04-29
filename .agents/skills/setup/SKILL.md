---
name: setup
description: Check and install required dependencies for this repository.
---

1. Check Python: `uv run python3 --version`. Stop if this fails.

2. Check uv: `uv --version`. If missing, install:
   - Linux/macOS: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
   - Verify: `uv --version`. If still fails, skip uv and use pip.

3. Check pymupdf: `uv run python3 -c "import fitz"`. If missing:
   - uv: `uv sync` (uses `pyproject.toml` in repo root)
   - pip fallback: `python3 -m pip install pymupdf`
   - Verify: `uv run python3 -c "import fitz; print('ok')"`. Report result.
