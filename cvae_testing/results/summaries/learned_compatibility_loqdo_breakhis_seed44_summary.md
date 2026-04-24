# Learned Compatibility LOQDO Summary (BreakHis, Seed 44, Variant B)

## Scope

- Protocol: leave-one-query-domain-out (LOQDO)
- Dataset: BreakHis
- Backbones: resnet18, resnet50, dinov2_vitb14
- Variant evaluated: B
- Methods compared:
  - metadata_routing baseline
  - constant_mean baseline
  - expert_prior baseline
  - linear_regression
  - mlp_regression
- Feature sets:
  - A: metadata_distance
  - B: metadata_distance + embedding_distance + safe domain features + expert_domain_one_hot

## Sanity Checks

- LOQDO folds per backbone: 4
- Query domains observed: 40, 100, 200, 400
- Pair rows per run: 16
- Fold-level rows total: 108
- Aggregate rows total: 27

## Headline Table (Best Learned Method Per Backbone)

| backbone | best learned method | feature set | gap mean | top1 mean | spearman mean | baseline gap | baseline top1 | baseline spearman |
|---|---|---|---:|---:|---:|---:|---:|---:|
| resnet18 | linear_regression | A | 0.005643 | 0.500 | 0.387 | 0.005643 | 0.500 | 0.387 |
| resnet50 | mlp_regression | A | 0.003974 | 0.250 | -0.129 | 0.003042 | 0.500 | 0.258 |
| dinov2_vitb14 | linear_regression | A | 0.009483 | 0.000 | 0.000 | 0.009483 | 0.000 | 0.000 |

## Trivial Baseline Check (High-Value Comparator)

| backbone | constant_mean gap | expert_prior gap | best learned gap | best non-learned winner |
|---|---:|---:|---:|---|
| resnet18 | 0.006013 | 0.006570 | 0.005643 | learned/model tie with metadata baseline |
| resnet50 | 0.006108 | 0.001754 | 0.003974 | expert_prior (non-learned) |
| dinov2_vitb14 | 0.013117 | 0.010355 | 0.009483 | learned beats trivial gaps, but ranking remains weak |

## Interpretation Against Thesis Cases

- Case 1 (clear learned improvement): not supported in this seed-44 LOQDO pass.
- Case 2 (representation dependence): partially supported.
  - resnet18: learned model ties metadata baseline on all primary metrics.
  - resnet50: expert_prior is strongest on gap and top1; learned methods are not clearly better.
  - dinov2: learned model improves gap relative to trivial baselines, but top1 remains 0.0.
- Case 3 (signals insufficient): plausible for current feature design, especially under strict LOQDO generalization.

## Feature-Set A vs B Observation

- Feature set B did not consistently improve over A in this run.
- In several cells, B reduced ranking quality (top1/Spearman) even when gap was similar.
- This suggests current B features are not yet adding stable signal under query-domain holdout.

## Practical Conclusion (This Run)

- The new learned compatibility pipeline is functioning and leakage-safe.
- The result is currently not a strong positive claim that learned compatibility outperforms metadata routing.
- The trivial-baseline check was essential: on resnet50, expert_prior outperforms learned methods on gap.

## Suggested Next Steps

1. Run the same LOQDO protocol across seeds 42/43/44 and aggregate mean plus std before drawing final conclusions.
2. Keep linear regression as the primary method and use MLP as sensitivity only.
3. Refine feature set B with additional safe query/expert metadata attributes while preserving no query one-hot.
4. Add confidence intervals (bootstrap over folds and then over seeds) for gap, top1, and Spearman.

## Source Files

- Stats table: results/comparison_tables/learned_compatibility_loqdo_breakhis_stats.csv
- Raw table: results/comparison_tables/learned_compatibility_loqdo_breakhis_raw.csv
- Run summary metadata: results/comparison_tables/learned_compatibility_loqdo_breakhis_summary.json
