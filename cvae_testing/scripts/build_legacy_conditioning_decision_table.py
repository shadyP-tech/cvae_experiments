#!/usr/bin/env python3
from __future__ import annotations

import sys


MESSAGE = """\
BLOCKED: legacy conditioning decision tables are quarantined.

These tables summarize legacy routed-CVAE runs that are no longer
thesis-facing. Use the learned utility v2 decision-table builders for
compatibility claims.

Original script:
  scripts/quarantined/legacy_or_diagnostic/build_legacy_conditioning_decision_table.py
"""


def main() -> int:
    sys.stderr.write(MESSAGE)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
