#!/usr/bin/env bash
set -euo pipefail

printf '%s\n' \
  "BLOCKED: metadata conditional-prior sweeps depend on the quarantined legacy routed mode." \
  "They are preserved only for audit history." \
  "Original script: scripts/quarantined/legacy_or_diagnostic/run_metadata_conditional_prior_seed_sweep.sh" >&2
exit 2
