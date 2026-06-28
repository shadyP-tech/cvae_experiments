# CVAE Downstream Evaluation

This root package contains the second-stage evaluation surface for CVAE
selection/aggregation experiments.

It keeps the thesis pipeline explicit:

```text
target support set
-> allowed pre-evaluation candidate features
-> source-inner learned downstream utility / support-NELBO / metadata routing
-> selected or aggregated source candidates
-> synthetic embedding generation
-> downstream classifier / distributional evaluation
-> utility on held-out target evaluation data
```

The package is intentionally separate from `cvae_support_routing` and `cvae_testing`:

- `cvae_support_routing`: established support-NELBO selection and protocol artifacts.
- `cvae_testing`: existing training, routing, and evaluation runtime.
- `cvae_downstream_evaluation`: next-stage synthetic-embedding utility experiments.

## Thesis Placement

This package supports the thesis outline sections on:

- independently trained CVAE-style generative experts
- post-hoc routing without expert retraining or parameter sharing
- domain-specific and unseen-domain generalization
- generation quality and downstream classification performance
- comparison against metadata routing, random experts, naive ensembles, and oracle diagnostics

The current primary question is:

```text
Can source-only selection or aggregation improve held-out downstream BACC
without changing the CVAE generators?
```

Support NELBO remains an allowed pre-selection signal and baseline. Downstream
BACC is the primary utility for the learned downstream utility estimator and
final evaluation.

## Structure

```text
cvae_downstream_evaluation/
  configs/       Versioned experiment templates.
  docs/          Protocol, claim boundaries, thesis alignment, implementation order.
  scripts/       Thin CLI entrypoints for future runs and reports.
  src/           Implementation modules with explicit ownership boundaries.
  tests/         Protocol and artifact contract tests.
  artifacts/     Ignored generated manifests, reports, tables, and plots.
```

## First Experiment

Start with:

```text
direct_support_nelbo_selected_synthetic_downstream_v1
```

Locked v1 scope:

- Camelyon17 only.
- Synthetic-only downstream utility.
- Class-stratified reference-posterior resampling through frozen source experts.
- Single-expert downstream oracle diagnostic.
- Late probability ensemble baseline, reported outside the single-expert oracle.
- Lightweight distributional fidelity diagnostics.

Defer source-only augmentation, few-shot target augmentation, and corruption robustness until the primary downstream transfer question has a stable answer.

## Selection/Aggregation Extension

The approved extension adds:

- an all-candidate downstream utility matrix stored as diagnostic-only oracle evidence
- allowed pre-evaluation feature tables built from target support, source-inner folds, and source/provenance metadata
- source-inner learned downstream utility estimators
- top1, top-k uniform, and soft aggregation routing rules
- metadata, random, support-NELBO, budget-matched dense, full-budget dense, and oracle diagnostic comparisons

Diagnostic downstream matrices must be named `diagnostic_downstream_utility.*`
and must not be read by deployable feature extraction, estimator fitting,
normalization, routing, aggregation, generation settings, classifier settings,
or reportable method selection.

## Non-Negotiable Protocol

- Candidate expert pool excludes the held-out target/query expert under LOQDO/LODO.
- Target support samples are disjoint from target evaluation samples.
- Target support labels are not used for routing in the primary protocol.
- Direct target identity fields may appear in lineage and reports, but not in deployable predictive features.
- Target evaluation labels, target evaluation NELBO, downstream oracle expert labels, and target test metrics are forbidden before routing and generation decisions are locked.
- Source expert checkpoints are frozen; routing and downstream evaluation must not update them.
- Synthetic budget, class prior, classifier architecture, classifier hyperparameters, and metric set must be fixed before inspecting target evaluation results.
- Single-expert synthetic samples live in the selected expert's projected CVAE feature frame; target evaluation embeddings must be projected through the same expert head.
- The current CVAE is not treated as class-conditional. Labels enter the primary generation mode through labeled source reference pools.

## Expected Outputs

Each completed run should produce:

- protocol manifest
- split manifest
- expert provenance table
- support-NELBO selection table
- all-expert downstream utility matrix
- diagnostic downstream utility matrix, when using the selection/aggregation path
- selected-vs-oracle downstream gap table
- baseline comparison table
- support-size stratified downstream summary
- generation seed stability table
- classifier seed stability table
- fidelity diagnostics table
- leakage/provenance report
- frozen protocol/config snapshot
- allowed pre-evaluation feature table
- adoption-eligible prediction/selection table

Generated outputs belong under `artifacts/` during local development or under the existing repo-wide `outputs/` convention for full experiment runs.

## Artifact Builders

The selection/aggregation path is split into explicit artifact steps:

```bash
python cvae_downstream_evaluation/scripts/build_allowed_feature_table.py \
  --candidates <candidate_manifest.csv> \
  --support-features <support_features.csv> \
  --source-inner-features <source_inner_features.csv> \
  --metadata-features <metadata_features.csv> \
  --out <run>/features/allowed_pre_eval_features.csv

python cvae_downstream_evaluation/scripts/train_source_inner_utility_estimator.py \
  --source-inner <source_inner_training.csv> \
  --features support_nelbo,source_inner_stability \
  --model-out <run>/models/source_inner_utility_estimator.json \
  --diagnostics-out <run>/reports/source_inner_estimator_diagnostics.json

python cvae_downstream_evaluation/scripts/predict_learned_utility_features.py \
  --model <run>/models/source_inner_utility_estimator.json \
  --features <run>/features/allowed_pre_eval_features.csv \
  --out <run>/features/allowed_pre_eval_features_with_predictions.csv

python cvae_downstream_evaluation/scripts/build_learned_utility_selection_report.py \
  --features <run>/features/allowed_pre_eval_features_with_predictions.csv \
  --diagnostic-matrix <run>/matrices/diagnostic_downstream_utility.csv \
  --out-dir <run>

python cvae_downstream_evaluation/scripts/build_selection_leakage_report.py \
  --candidates <candidate_manifest.csv> \
  --features <run>/features/allowed_pre_eval_features.csv \
  --selections <run>/selections/adoption_eligible_predictions.csv \
  --generation-frozen \
  --classifier-frozen \
  --out <run>/reports/leakage_report.json
```

The learned-utility selection builder reads the diagnostic matrix only after
writing adoption-eligible selections, and only to build final alignment reports.

The legacy workstation runner can write the quarantined matrix directly:

```bash
python cvae_downstream_evaluation/scripts/run_direct_support_nelbo_downstream.py \
  --config cvae_downstream_evaluation/configs/experiments/direct_support_nelbo_selected_synthetic_downstream_v1.yaml \
  --build-matrix \
  --diagnostic-matrix
```

The full learned-utility artifact path can also be run as one command:

```bash
python cvae_downstream_evaluation/scripts/run_learned_utility_pipeline.py \
  --candidates <candidate_manifest.csv> \
  --source-inner-training <source_inner_training.csv> \
  --diagnostic-matrix <run>/tables/diagnostic_downstream_utility.csv \
  --support-features <support_features.csv> \
  --source-inner-features <source_inner_features.csv> \
  --features support_nelbo,source_inner_stability \
  --out-dir <run>
```
