# Protocol Status

This repository keeps invalid and diagnostic experiment code for audit history,
but only protocol-safe paths are thesis-facing.

## Thesis-Facing Paths

- `cvae_testing/src/eval/evaluators/learned_utility.py`
  - Protocol: `learned_utility_loqdo_candidate_exclusion_v2`
  - Experiment mode: `learned_utility_routing`
  - Required invariant: held-out target/query expert is excluded from every LOQDO candidate pool.
- `cvae_testing/src/eval/evaluators/support_set_calibration.py`
  - Protocol: target-local support NELBO estimates use support/evaluation splits that are disjoint.
- `cvae_testing/src/eval/evaluators/domain_query_oracle_gap.py`
  - Diagnostic protocol: target expert is excluded from candidate baselines; oracle values are reporting-only.

## Quarantined Paths

- `cvae_testing/scripts/quarantined/invalid_protocol/`
  - Preserves the legacy learned-compatibility LOQDO runner and seed sweeps.
  - Reason: the legacy runner allowed the held-out target expert in LOQDO candidate pools.
- `cvae_testing/scripts/quarantined/legacy_or_diagnostic/`
  - Preserves legacy routed-CVAE and conditioning scripts.
  - Reason: these modes are historical/diagnostic and are no longer thesis-facing.
- `cvae_testing/configs/quarantined/legacy_or_diagnostic/`
  - Preserves legacy routed and latent compatibility configs.
- `cvae_testing/results/quarantined/`
  - Preserves invalid or superseded tracked result artifacts, including old learned compatibility,
    response development, legacy routed, and legacy standard-deviation decision outputs.

## Blocked Public Interfaces

- `legacy_routed_cvae` and `latent_compatibility` are not registered experiment modes.
- `scripts/run_learned_compatibility_loqdo.py` is a fail-fast stub and exits with an invalid-protocol message.
- Legacy conditioning and metadata seed sweeps are fail-fast stubs and exit with quarantine messages.

## Required Artifact Invariants

Thesis-facing compatibility artifacts must satisfy:

- `protocol_version == learned_utility_loqdo_candidate_exclusion_v2` for learned utility outputs.
- `target_expert_excluded == 1`.
- `fold_query_domain` is absent from `candidate_experts`.
- `selected_expert` and `candidate_oracle_expert` are members of `candidate_experts`.
- Adoption-eligible methods have `routing_uses_eval_nelbo == 0`.
- Adoption-eligible methods have `routing_uses_eval_domain_statistics == 0`.
- Diagnostic/oracle methods are not adoption-eligible.

## Current Cleanup Decision

Invalid and diagnostic work is quarantined rather than deleted. Do not use
quarantined artifacts in thesis tables, manifests, or method-selection claims.
