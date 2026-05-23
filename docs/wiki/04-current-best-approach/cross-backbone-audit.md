# Cross-Backbone Audit

## Purpose

Define the role and claim boundary for cross-backbone aggregation after the SAIL extraction.

## Key Claims

- Cross-backbone aggregation may include Phikon, UNI, and Virchow2.
- It is audit-only and not part of the active SAIL Virchow2 implementation.
- It may estimate an ensemble ceiling.
- It cannot justify CVAE rebuild readiness.

## Evidence / Source Artifacts

- `../../../cvae_downstream_evaluation/legacy/superseded_by_sail/r12c/README.md`
- `../../../cvae_downstream_evaluation/legacy/superseded_by_sail/r12c/configs/r12c_virchow2_dense_config_aggregation.yaml`

## Interpretation

Cross-backbone success may come from heterogeneous feature-space diversity. A CVAE rebuild needs evidence for a single feature space, currently Virchow2.

## Implication For Thesis

Cross-backbone results should be reported as diagnostic ceiling evidence only. They must not be used as deployable or rebuild-readiness evidence.

## Limitations

No active SAIL cross-backbone output artifacts are expected. Any future cross-backbone run must be explicitly labeled audit-only.

## Next Checks

- If a future cross-backbone audit is added, confirm generated rows are marked `audit_only`.
- Confirm leakage reports exclude cross-backbone rows from rebuild labels.
