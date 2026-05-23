# Foundation Embeddings

## Purpose

Document why pathology foundation embeddings are central to the current empirical direction.

## Key Claims

- Generic visual embeddings plus PCA may not preserve enough pathology-specific transfer utility.
- Pathology embeddings raised the real-feature ceiling in R1.2/R1.2b.
- Virchow2 is the current primary feature space for the next deployable diagnostic.

## Evidence / Source Artifacts

- `../../context/current_experimental_state.md`
- `../../../cvae_downstream_evaluation/artifacts/reports/r12_decision_report.md`
- `../../../cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/reports/r12b_decision_report.md`
- `../../../cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/tables/r12b_backbone_ranking.csv`

## Interpretation

R1.2b verified Virchow2 as the top source-inner-LODO selected backbone in the inspected artifact set, with mean BACC 0.9155 and 4/5 centers at or above 0.85. Weak-center and seed instability remain.

## Implication For Thesis

The immediate next step is not another broad backbone screen. It is SAIL: a backbone-agnostic source-only aggregation method currently instantiated with Virchow2.

## Limitations

Cross-backbone gains, if present, may reflect heterogeneous ensemble diversity and cannot justify a single-space CVAE rebuild.

## Next Checks

- Verify SAIL outputs once available.
- Keep any future cross-backbone outputs audit-only.
