# Hybrid Ablation Three-Run Summary (Seed 42)

## Scope

This summary consolidates three completed hybrid runs:

- Camelyon17 (exact site metadata routing):
  - `outputs/camelyon17/hybrid_ablation_v1/cam17_hybrid_full_seed42/`
- BreakHis (exact magnification metadata routing):
  - `outputs/breakhis/hybrid_ablation_v1/breakhis_hybrid_full_seed42/`
- BreakHis (ordinal magnification diagnostic routing):
  - `outputs/breakhis/hybrid_ablation_ordinal_diag_v1/breakhis_hybrid_ordinal_seed42/`

Source compact reports:

- `reports/hybrid_variant_comparison.csv` from each run
- Consolidated table written to:
  - `results/comparison_tables/hybrid_ablation_three_runs_seed42.csv`

## Protocol check (expected controls)

- Variant ablations present: `A`, `B`, `C`
- Synthetic budgets present: `1.0x` and `0.5x`
- Downstream comparisons present: real-only, random synthetic, pooled synthetic, routed synthetic
- Oracle gap metrics present: `metadata_to_oracle_gap`
- Routing-alignment metrics present: Spearman, top-1 agreement, mean rank

All required fields are present in all three runs.

## 1. Routing-alignment result (most important)

Across all three runs, **Variant B** is the strongest and most consistent on routing-alignment metrics.

Camelyon17 (exact):
- Variant B: `metadata_to_oracle_gap = 0.0027`, `spearman = 0.5657`, `top1 = 0.80`, `mean_rank = 1.40`
- Variant A: `gap = 0.0068`, `spearman = 0.0707`, `top1 = 0.20`, `mean_rank = 2.80`
- Variant C: `gap = 0.0071`, `spearman = 0.3536`, `top1 = 0.60`, `mean_rank = 2.00`

BreakHis (exact):
- Variant B: `gap = 0.0411`, `spearman = 0.6455`, `top1 = 0.75`, `mean_rank = 1.25`
- Variant A: `gap = 0.0755`, `spearman = 0.1291`, `top1 = 0.25`, `mean_rank = 2.25`
- Variant C: `gap = 0.0736`, `spearman = 0.2582`, `top1 = 0.50`, `mean_rank = 2.00`

BreakHis (ordinal diagnostic):
- Variant B: `gap = 0.0411`, `spearman = 0.6000`, `top1 = 0.75`, `mean_rank = 1.25`
- Variant A: `gap = 0.0755`, `spearman = 0.2500`, `top1 = 0.25`, `mean_rank = 2.25`
- Variant C: `gap = 0.0736`, `spearman = 0.0500`, `top1 = 0.50`, `mean_rank = 2.00`

Interpretation:
- The ablation where encoder specialization is present with a shared pooled CVAE (`B`) aligns metadata routing with compatibility best.
- This is consistent across both datasets and both BreakHis routing policies.

## 2. Exact vs ordinal metadata on BreakHis

BreakHis exact vs ordinal differences are modest in this single seed:

- Variant B remains best under both policies.
- Ordinal routing slightly decreases Spearman for B (`0.6455 -> 0.6000`) in this run.
- Variant-level ranking does not change.

Interpretation:
- For seed 42, ordinal similarity does not create a clear routing-alignment gain over exact metadata identity.
- This remains a diagnostic result and should be confirmed with multi-seed runs.

## 3. Downstream utility (AUROC deltas)

### Camelyon17

- `budget_1.0x`: routed AUROC underperforms for all variants relative to real-only.
- `budget_0.5x`: mixed behavior; A improves vs all comparators, B improves vs random/pooled but not real-only, C remains below real-only.

### BreakHis (exact)

- `budget_1.0x`: A improves vs real-only; B is near-neutral vs real-only but better than random and pooled; C is below.
- `budget_0.5x`: A slightly positive; B turns negative vs all; C remains negative.

### BreakHis (ordinal)

- Similar mixed pattern to exact.
- No robust advantage of ordinal policy in downstream AUROC for this seed.

Important caveat:
- Balanced accuracy deltas are `0.0` in all rows, indicating no discriminative movement under the current downstream setup for this seed. AUROC is therefore the more informative metric here.

## 4. Global baseline comparison values

Reported NELBO baselines from run compact reports:

- Camelyon17:
  - legacy global: `262.3260`
  - hybrid pooled global: `0.0111`
- BreakHis exact:
  - legacy global: `325.1107`
  - hybrid pooled global: `0.1785`
- BreakHis ordinal:
  - legacy global: `323.2401`
  - hybrid pooled global: `0.1836`

These are not directly on the same representation scale and should be interpreted as within-run references, not cross-family absolute comparisons.

## Thesis-facing conclusion (seed 42)

1. The hybrid ablation implementation worked as intended and produced all required controls.
2. Variant B is the most convincing configuration for routing-alignment quality across datasets.
3. Downstream utility gains are not yet stable across budgets and routing policies.
4. BreakHis ordinal diagnostic did not clearly outperform exact routing in this seed.

Current status: promising signal on **where specialization helps** (routing compatibility alignment), but **insufficient single-seed evidence** for stable downstream utility gains.

## Recommended immediate next step

Run a 3-seed repeat (`42/43/44`) for at least Variant B on:

- Camelyon17 exact
- BreakHis exact
- BreakHis ordinal diagnostic

Then summarize mean/std for:

- `metadata_to_oracle_gap`
- Spearman, top-1, mean rank
- AUROC deltas (routed vs real/random/pooled) at both budgets

This will determine whether the observed alignment advantage translates into reproducible utility gains.
