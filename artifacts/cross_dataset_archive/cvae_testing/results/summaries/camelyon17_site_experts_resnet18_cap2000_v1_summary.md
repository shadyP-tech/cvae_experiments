# Camelyon17 Cap2000 Follow-up Summary (Seed 42)

## Run and config

- Config: `configs/experiments/camelyon17/camelyon17_site_experts_resnet18_cap2000_v1.yaml`
- New run: `outputs/camelyon17/camelyon17_site_experts_resnet18_cap2000_v1/2026-03-11_1853_seed42/`
- Baseline run used for comparison: `outputs/camelyon17/routed_cvae_v1/seed42_with_plots/`

Key setting change in this follow-up:
- `max_samples_per_domain: 2000` (previous cap was 500)
- split unchanged at `70/15/15`

## Data increase per expert

| Run | Train/site | Val/site | Test/site |
|---|---:|---:|---:|
| baseline (cap500) | ~350-419 | mixed (some domains missing in val/test) | mixed |
| cap2000 | 1400 | 300 | 300 |

New cap2000 manifest is fully balanced across all 5 sites for all splits.

## 1. Expert-domain matrix (most important)

Source files:
- baseline: `outputs/camelyon17/routed_cvae_v1/seed42_with_plots/reports/expert_matrix.json`
- cap2000: `outputs/camelyon17/camelyon17_site_experts_resnet18_cap2000_v1/2026-03-11_1853_seed42/reports/expert_matrix.json`

Summary (lower reconstruction is better):
- Baseline matrix coverage was incomplete (3x4 observed entries).
- Cap2000 matrix is complete 5x5.
- Diagonal mean reconstruction:
  - baseline: `391.64`
  - cap2000: `281.14`
- Off-diagonal mean reconstruction:
  - baseline: `529.12`
  - cap2000: `350.86`
- Off-diagonal minus diagonal gap:
  - baseline: `+137.48`
  - cap2000: `+69.71`
- Rows where diagonal is row-min (best expert for its own domain):
  - baseline: `1/3`
  - cap2000: `2/5`

Interpretation:
- The cap2000 run improves overall reconstruction substantially and gives a complete per-site matrix.
- A strong, clean diagonal is still not consistent across all domains.

## 2. Global vs expert gap

Source files:
- baseline: `outputs/camelyon17/routed_cvae_v1/seed42_with_plots/reports/routing_results.json`
- cap2000: `outputs/camelyon17/camelyon17_site_experts_resnet18_cap2000_v1/2026-03-11_1853_seed42/reports/routing_results.json`

N-ELBO (lower is better):

| Run | Global | Oracle | Hard | Soft |
|---|---:|---:|---:|---:|
| baseline | 315.386 | 533.360 | 533.360 | 540.232 |
| cap2000 | 264.001 | 318.189 | 318.189 | 359.578 |

Gap vs global (positive means worse than global):
- Oracle - Global:
  - baseline: `+217.974`
  - cap2000: `+54.188`
- Hard - Global:
  - baseline: `+217.974`
  - cap2000: `+54.188`
- Soft - Global:
  - baseline: `+224.845`
  - cap2000: `+95.577`

Interpretation:
- This is a large and meaningful shrink in the global-vs-expert gap.
- More per-domain data clearly helped expert quality.

## 3. Routing usefulness

- Routing selection accuracy:
  - baseline: `0.455`
  - cap2000: `1.000`
- Hard equals oracle in cap2000 (exact metadata routing behavior is now clean).
- Even with perfect routing decisions, experts still underperform global by ~54 N-ELBO.

Interpretation:
- Sample count was a major bottleneck and was partially resolved.
- Remaining gap suggests sample count alone is not the full explanation (representation and/or expert modeling still limits specialization).

## Embedding-level context from the prior diagnostic

From `outputs/camelyon17/site_separability/site_separability_report.md`:
- Logistic site probe: `0.654` (chance `0.200`)
- Linear SVM site probe: `0.615`
- kNN same-site recall: `0.457`
- Verdict: moderate site structure remains in frozen ResNet18 embeddings.

Implication:
- Site signal exists but is not maximally separable, so routed experts can improve with more data but may still trail a pooled global model unless representation/expert capacity is strengthened.

## Bottom line

The cap2000 run is informative and successful as a diagnostic:
- it materially improved routed expert performance,
- reduced the global gap by about 4x,
- restored clean hard-routing behavior,
- but did not yet make experts beat global.

This supports: data scarcity was important, but not sufficient to explain the full routing underperformance.
