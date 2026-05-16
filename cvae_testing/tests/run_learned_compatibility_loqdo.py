#!/usr/bin/env python3
from __future__ import annotations

import sys


MESSAGE = """\
BLOCKED: scripts/run_learned_compatibility_loqdo.py is quarantined.

This legacy runner emitted LOQDO rows with the held-out target expert still in
the candidate pool, so its compatibility estimates are not thesis-safe.

Use the learned utility v2 protocol instead:
  - experiment mode: learned_utility_routing
  - evaluator: src.eval.evaluators.learned_utility.evaluate_learned_utility_loqdo
  - protocol: learned_utility_loqdo_candidate_exclusion_v2

The original implementation is preserved for audit history at:
  scripts/quarantined/invalid_protocol/run_learned_compatibility_loqdo.py
"""


def main() -> int:
    sys.stderr.write(MESSAGE)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
