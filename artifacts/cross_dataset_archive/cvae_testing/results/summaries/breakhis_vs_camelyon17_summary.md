# BreakHis vs Camelyon17 Summary (Seed 42)

## Scope

This summary compares one leakage-safe run from each dataset:

- BreakHis: `outputs/breakhis/routed_cvae_v1/seed42/`
- Camelyon17: `outputs/camelyon17/routed_cvae_v1/seed42_sitewise_exact/`

Both runs use the same core pipeline design:
- embedding cache
- per-domain experts
- global baseline
- hard/soft routing and baseline comparisons

## Data and split integrity

| Dataset | Train | Val | Test | Patient overlap |
|---|---:|---:|---:|---|
| BreakHis | 1400 | 300 | 300 | none |
| Camelyon17 | 1750 | 375 | 375 | none |

Both runs are leakage-safe at the patient/group level.

## Main metric comparison (lower is better)

| Dataset | Hard NELBO | Soft NELBO | Global NELBO | Oracle NELBO | Routing Accuracy |
|---|---:|---:|---:|---:|---:|
| BreakHis | 472.89 | 514.57 | 329.08 | 472.89 | 1.00 |
| Camelyon17 | 480.58 | 525.27 | 316.43 | 480.58 | 1.00 |

## Reconstruction comparison (lower is better)

| Dataset | Hard Recon | Soft Recon | Global Recon | Oracle Recon |
|---|---:|---:|---:|---:|
| BreakHis | 435.85 | 477.28 | 290.61 | 435.85 |
| Camelyon17 | 441.57 | 487.93 | 277.62 | 441.57 |

## Interpretation

1. Routing behavior is technically correct in both runs.
- Hard routing equals oracle because exact metadata is used for domain-to-expert mapping.
- Confusion matrices are perfectly diagonal in both datasets.

2. Global model remains clearly stronger than routed experts in both datasets.
- BreakHis hard-global NELBO gap: `472.89 - 329.08 = 143.81`
- Camelyon17 hard-global NELBO gap: `480.58 - 316.43 = 164.14`

3. Camelyon17 does not currently improve the routing conclusion.
- Even with site-wise domains and leakage-safe grouped splitting, specialization+routing still underperforms pooled modeling.

## Thesis-facing conclusion

At this stage, evidence supports the same directional finding on both datasets:
- under the current embedding + CVAE-expert setup,
- metadata-based routing is stable,
- but does not outperform a single global CVAE.

This is a valid negative result and can be reported as such.

## Suggested immediate next step

Run a small 3-seed sweep for Camelyon17 (`seed42/43/44`) and summarize with:

```bash
python -m src.eval.summarize_seed_sweep \
  --experiment-dir outputs/camelyon17/routed_cvae_v1 \
  --run-ids seed42_sitewise_exact seed43_sitewise_exact seed44_sitewise_exact \
  --csv-out results/comparison_tables/camelyon17_routed_cvae_v1_seed_sweep.csv \
  --md-out results/summaries/camelyon17_routed_cvae_v1_seed_sweep.md
```
