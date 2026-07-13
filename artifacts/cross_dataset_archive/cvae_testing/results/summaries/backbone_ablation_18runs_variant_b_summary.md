# Controlled Backbone Ablation under Hybrid Variant B

- Runs detected: 18/18
- Primary lens: Variant B at budget_1.0x (A/C retained for context)

## breakhis

| backbone | variant | gap mean | gap std | spearman mean | top1 mean | sharpness gap mean |
|---|---|---:|---:|---:|---:|---:|
| dinov2_vitb14 | A | 0.098946 | 0.003540 | -0.172133 | 0.166667 | -0.005089 |
| dinov2_vitb14 | B | 0.053233 | 0.000974 | -0.172133 | 0.166667 | 0.002428 |
| dinov2_vitb14 | C | 0.099408 | 0.002971 | -0.172133 | 0.166667 | -0.002346 |
| resnet18 | A | 0.090465 | 0.005978 | 0.000000 | 0.250000 | -0.002836 |
| resnet18 | B | 0.048092 | 0.002619 | -0.043033 | 0.250000 | -0.000663 |
| resnet18 | C | 0.089654 | 0.006557 | 0.043033 | 0.250000 | -0.001690 |
| resnet50 | A | 0.068830 | 0.003195 | -0.086066 | 0.083333 | -0.002120 |
| resnet50 | B | 0.037378 | 0.001658 | 0.043033 | 0.250000 | -0.000295 |
| resnet50 | C | 0.064912 | 0.002881 | 0.000000 | 0.250000 | -0.000273 |

## camelyon17

| backbone | variant | gap mean | gap std | spearman mean | top1 mean | sharpness gap mean |
|---|---|---:|---:|---:|---:|---:|
| dinov2_vitb14 | A | 0.006905 | 0.000208 | 0.117851 | 0.333333 | 0.000193 |
| dinov2_vitb14 | B | 0.009605 | 0.006393 | 0.471405 | 0.533333 | 0.058635 |
| dinov2_vitb14 | C | 0.012940 | 0.002836 | 0.306413 | 0.533333 | 0.057263 |
| resnet18 | A | 0.006795 | 0.000198 | 0.023570 | 0.133333 | 0.000051 |
| resnet18 | B | 0.002877 | 0.000095 | 0.353553 | 0.533333 | 0.001023 |
| resnet18 | C | 0.006909 | 0.000308 | 0.329983 | 0.400000 | 0.001406 |
| resnet50 | A | 0.005491 | 0.000127 | 0.023570 | 0.266667 | 0.000084 |
| resnet50 | B | 0.002144 | 0.000186 | 0.282843 | 0.333333 | 0.000378 |
| resnet50 | C | 0.005568 | 0.000063 | 0.117851 | 0.200000 | 0.000376 |

## Interpretation Layer

- Case 1 (B stable best across backbones): specialization invariant to representation choice.
- Case 2 (B weakens with stronger backbone): stronger pretrained features reduce specialization gain.
- Case 3 (B strengthens with stronger backbone): better representations amplify routing signal.
- Use metadata_to_oracle_gap, spearman/top1, and sharpness gap jointly for the final thesis claim.

- Raw table: results/comparison_tables/backbone_ablation_18runs_raw.csv
- Stats table: results/comparison_tables/backbone_ablation_18runs_stats.csv
