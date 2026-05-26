# Empirical Pivot

## Purpose

Explain the pivot from metadata-first routing to compatibility-driven routing and aggregation.

## Key Claims

- Earlier framing treated metadata as the likely primary routing signal.
- Current evidence supports treating metadata as a baseline, proxy, and interpretability signal.
- The current bottleneck has shifted from representation ceiling alone to source-selected stability and later CVAE preservation.
- SAIL is a diagnostic gate for Virchow2 real-feature transfer stability, not a final CVAE method.
- The latest generated-embedding experiments shift the CVAE bottleneck toward
  latent prior sampling and decentralized composition; the paired dense all4
  reliability confirmation is a dense aggregation PASS, while support-NELBO
  and source-inner transfer are not validated final signals.

## Evidence / Source Artifacts

- `../../context/pivot_statement.md`
- `../../context/current_experimental_state.md`
- `../../../cvae_downstream_evaluation/artifacts/reports/r12_decision_report.md`
- `../../../cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/reports/r12b_decision_report.md`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_decentralized_reliability_weighted_gmm_prior_v1/tables/decentralized_reliability_summary.csv`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_decentralized_source_inner_transfer_top3_gmm_prior_v1/tables/decentralized_source_inner_transfer_summary.csv`
- `../../../cvae_rebuild/artifacts/virchow2_cvae_paired_dense_all4_reliability_confirmation_v1/tables/paired_dense_all4_summary.csv`

## Interpretation

Pathology foundation embeddings raised the real-feature transfer ceiling. The
remaining real-feature failure is exact source-selected top-1 instability.

The generated-embedding artifacts show that CVAE utility can be partially
preserved. The deployable decentralized composition story is now sharper:
source-union K16 is strong but centralized, paired dense all4 reliability
weighting improves dense aggregation, and D1.3/D1.5 do not validate their
proposed sparse or target-conditioned compatibility signals.

## Implication For Thesis

The thesis-facing narrative should keep real-feature and generated-embedding
evidence separate. SAIL remains the real-feature gate; D-series results now
provide generated-embedding preservation/composition evidence and negative
selector evidence.

## Limitations

Provided synthesis says older DINOv2/PCA64 likely did not support 0.90 BACC. Where not verified against an artifact, label this: `Provided synthesis; verify against artifact if available.`

## Next Checks

- Verify SAIL output artifacts once generated or synced.
- Reuse paired generation/prediction invariants before any further D-series
  selector confirmation.
