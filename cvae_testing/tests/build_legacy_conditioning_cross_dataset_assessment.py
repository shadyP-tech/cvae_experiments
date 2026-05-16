#!/usr/bin/env python3
from __future__ import annotations

import sys


MESSAGE = """\
BLOCKED: legacy conditioning cross-dataset assessment is quarantined.

These summaries depend on legacy routed-CVAE runs and are preserved only for
audit history. Use protocol-v2 learned utility, support-set calibration, or
domain-query oracle-gap artifacts for thesis-facing analysis.

Original script:
  scripts/quarantined/legacy_or_diagnostic/build_legacy_conditioning_cross_dataset_assessment.py
"""


def main() -> int:
    sys.stderr.write(MESSAGE)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
