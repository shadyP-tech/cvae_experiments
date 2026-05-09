#!/usr/bin/env bash
set -euo pipefail

printf '%s\n' \
  "BLOCKED: this Camelyon17 learned-compatibility sweep used the quarantined legacy LOQDO runner." \
  "Use configs/experiments/camelyon17/learned_utility_response_routing_v1.yaml." \
  "Original script: scripts/quarantined/invalid_protocol/run_learned_compatibility_camelyon17_seed_sweep.sh" >&2
exit 2
