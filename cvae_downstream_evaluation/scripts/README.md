# Scripts

Scripts in this directory should stay thin. Put reusable logic under `src/cvae_downstream_evaluation/`.

Planned entrypoints:

- `run_direct_support_nelbo_downstream.py`: run the locked second-stage experiment.
- `build_downstream_decision_report.py`: consolidate artifacts into thesis-facing tables.
- `run_z11_ceiling_audit.py`: run the Z1.1 current-setup real-feature/PCA/C6.3 ceiling audit.
- `build_allowed_feature_table.py`: join candidate, support, source-inner, and metadata CSVs into allowed pre-evaluation features.
- `train_source_inner_utility_estimator.py`: train a source-inner ridge-linear downstream utility estimator.
- `predict_learned_utility_features.py`: apply the learned estimator to allowed pre-evaluation features.
- `build_learned_utility_selection_report.py`: write adoption-eligible selections first, then join diagnostic downstream utility for final alignment reporting.
- `run_learned_utility_pipeline.py`: run the full allowed-features, estimator, prediction, selection, alignment, and leakage-report path.
- `build_selection_leakage_report.py`: write leakage/provenance flags from materialized artifacts.
- `normalize_c52_legacy_artifacts.py`: normalize legacy C5.2 router examples and downstream matrices into the learned-utility pipeline inputs.
- `run_c52_legacy_learned_utility_batch.py`: run the learned-utility pipeline across filtered legacy target/support contexts and write summary alignment/leakage manifests.

Scripts must write manifests before running expensive generation or classifier work.
