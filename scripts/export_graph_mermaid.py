"""Export the compiled LangGraph MAS workflow as a Mermaid diagram source.

Renders ``img/nba_graph.mmd`` and prints the Mermaid text to stdout for the
README's MAS Workflow section.  Uses LangGraph's built-in Mermaid renderer
offline (``get_graph().draw_mermaid()``) — no network call, no third-party
service.

GitHub renders ```mermaid`` fenced blocks natively, so the diagram lives
directly in ``README.md`` and stays in sync with the graph.  The ``.mmd``
file under ``img/`` is a copy of the same source for local rendering
(e.g. via https://mermaid.live or the Mermaid CLI).

Run from the repo root:

    NBA_MOCK=1 python scripts/export_graph_png.py

``NBA_MOCK=1`` avoids constructing the LLM client just to print the graph.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Force mock mode BEFORE importing nba_app so no LLM client is constructed.
os.environ.setdefault("NBA_MOCK", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "app"))

import nba_app  # noqa: E402  (import after sys.path tweak)

OUT_PATH = REPO_ROOT / "img" / "nba_graph.mmd"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

mermaid_src = nba_app.get_graph().get_graph().draw_mermaid()
OUT_PATH.write_text(mermaid_src)
print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)}  ({len(mermaid_src):,} chars)\n")
print("---")
print(mermaid_src)
