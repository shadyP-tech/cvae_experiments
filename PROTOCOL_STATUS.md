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
- `cvae_support_routing/`
  - Owns direct support-NELBO experiment configs, run scripts, report builders, tests, and tracked comparison artifacts.
  - Protocol: moved assets preserve target-local support/evaluation separation and keep held-out NELBO utility as evaluation-only.
- `cvae_rebuild/src/experiments/support_selection/midogpp_support_nelbo_routing.py`
  - Owns the MIDOG++ routing-stage support-NELBO surface after schema/leakage/provenance validation.
  - Protocol: unlabeled target support selects heldout-excluded frozen source experts; held-out eval NELBO and oracle rows are diagnostic-only.
  - Artifact root: `cvae_rebuild/artifacts/midogpp/support_nelbo_routing_v1/`.
- `cvae_testing/src/eval/evaluators/domain_query_oracle_gap.py`
  - Diagnostic protocol: target expert is excluded from candidate baselines; oracle values are reporting-only.
- `sail/src/sail/midogpp_multiaxis.py` and `sail/src/sail/midogpp_signal_controls.py`
  - Protocol: MIDOG++ real Virchow2 train-cache diagnostics.
  - Claim boundary: real-feature learnability and preservation reference only; not CVAE preservation, metadata routing, or expert selection.
  - Required invariant: locked manifest, row-aligned feature cache, case/sample identity disjointness, fixed/fit-only preprocessing, and near-chance negative controls.
- `cvae_rebuild/src/midogpp_preservation_gate.py`
  - Protocol: MIDOG++ pca128 Virchow2-CVAE preservation gate under SAIL signal-control splits.
  - Claim boundary: pca128 `decode_mu` preservation mechanics only; not GMM composition, routing, expert selection, or controllable class-conditional generation.
  - Current synced decision: `PCA128_CVAE_DECODE_PRESERVATION_PASS`, `PCA128_CVAE_STRONG_PRESERVATION`, and `GMM_FEASIBILITY_ALLOWED_NEXT`, with `LATENT_CLASS_SIGNAL_DOMINATES_CONDITION_WARNING`.
  - Required invariant: corrected manifest/cache lineage, fit-only PCA/CVAE training, eval labels scoring-only, identity-overlap PASS, and real/decoded negative controls not above the locked threshold.

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
- MIDOG++ support-NELBO adoption rows have `support_labels_used == false`,
  `routing_uses_eval_nelbo == 0`, and contract-derived candidate counts.
- MIDOG++ diagnostic oracle rows have `adoption_eligible == false` and
  `routing_uses_eval_nelbo == 1`.
- Feature-cache learnability diagnostics require cache sanity before metric
  interpretation: row count/order alignment, feature provenance, crop/input
  validity, and no unexplained large duplicate/near-duplicate feature clusters.
- MIDOG++ CVAE preservation claims require an explicit generated-vs-real
  reference row. The current thesis-facing preservation claim is limited to
  pca128 `decode_mu` synthetic embeddings preserving tumor-balanced
  signal-control utility; latent-prior sampling, GMM composition, and routing
  require separate gates.

## Current Cleanup Decision

Invalid and diagnostic work is quarantined rather than deleted. Do not use
quarantined artifacts in thesis tables, manifests, or method-selection claims.
