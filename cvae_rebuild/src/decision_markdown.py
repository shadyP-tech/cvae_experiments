from __future__ import annotations

from pathlib import Path
from typing import Sequence


def write_decision_markdown(root: Path, lines: Sequence[str]) -> None:
    path = root / "reports" / "decision_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_decision_markdown_text(root: Path, text: str) -> None:
    path = root / "reports" / "decision_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
