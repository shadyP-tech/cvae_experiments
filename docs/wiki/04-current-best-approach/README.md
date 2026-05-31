# Current Best Approach

## Purpose

Document the current empirical synthesis and active SAIL implementation.

## Pages

- [Current synthesis](current-synthesis.md)
- [Generated-embedding CVAE current synthesis](generative-cvae-current-synthesis.md)
- [R1.2c-V lineage](r12c-v-plan.md)
- [Virchow2-only rationale](virchow2-only-rationale.md)
- [Cross-backbone audit](cross-backbone-audit.md)
- [Rebuild gates](rebuild-gates.md)
- [Next experiment sequence](next-experiment-sequence.md)

## Key Claims

- SAIL is the active current method: Source-only Aggregation via Inner-domain Leaveout.
- Virchow2 is the current backbone instantiation, not the method name.
- SAIL tests whether source-only dense real-feature aggregation is stable enough to justify a later CVAE preservation test.
- SAIL does not prove CVAE generation or metadata routing.
- For generated embeddings, the current best diagnostic is centralized
  source-union K16; the cleanest decentralized dense aggregation result is the
  paired dense all4 reliability confirmation.
- Component-union/random mass-bag runs reach high mean BACC, but remain
  diagnostic unless matched controls are beaten and weak-tail failures improve.
- No D-series experiment currently supports a full PASS target-conditioned
  compatibility-routing claim.

## Evidence / Source Artifacts

- `../../context/current_experimental_state.md`
- `../../../sail/configs/sail_virchow2.yaml`
- `../../../sail/src/sail/`
- `../../../cvae_downstream_evaluation/legacy/superseded_by_sail/r12c/README.md`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_decentralized_reliability_weighted_gmm_prior_v1/tables/decentralized_reliability_summary.csv`
- `../../../cvae_rebuild/artifacts/virchow2_cvae_paired_dense_all4_reliability_confirmation_v1/reports/decision_summary.md`
- `../../../cvae_rebuild/artifacts/virchow2_cvae_decentralized_component_union_mass_bagged_v1/reports/decision_summary.md`
- `../../../cvae_rebuild/artifacts/virchow2_cvae_dense_reliability_tailshield_random_mass_bag_v1/reports/decision_summary.md`
