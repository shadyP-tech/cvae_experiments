#!/usr/bin/env bash
set -euo pipefail

printf '%s\n' \
  "BLOCKED: legacy routed-CVAE conditioning sweeps are quarantined and no longer thesis-facing." \
  "Use learned_utility_response_routing or support-set/domain-query oracle-gap protocols for new evidence." \
  "Original script: scripts/quarantined/legacy_or_diagnostic/run_legacy_conditioning_camelyon17_seed_sweep.sh" >&2
exit 2
