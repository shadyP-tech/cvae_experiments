# Compact Interpretation Summary (Leakage-Fixed Rerun)

## Run status
- Pipeline executed successfully after patient-level split fix.
- Data volume remains as configured: train `1400`, val `300`, test `300` (`2000` total).
- Leakage check is clean: no patient overlap across train/val/test.

## Key quantitative outcomes
From `artifacts/reports/routing_results.json`:
- `global_cvae_nelbo = 330.68` (best; lower is better)
- `hard_metadata_routing_nelbo = 473.88`
- `soft_metadata_routing_nelbo = 516.94`
- `oracle_expert_nelbo = 473.88`
- `random_expert_nelbo = 520.76`
- `uniform_sampling_nelbo = 533.55`
- `equal_weight_scoring_nelbo = 533.31`
- `routing_selection_accuracy = 1.00`

Reconstruction metric shows the same ranking:
- `global_cvae_recon = 293.16` (best)
- `hard_metadata_routing_recon = oracle_expert_recon = 435.39`
- soft/random/uniform/equal are worse.

## Interpretation
- Routing is technically correct: confusion matrix is perfectly diagonal and selection accuracy is `1.0`.
- Hard routing equals oracle because magnification metadata directly selects the matching expert.
- Even with leakage fixed, specialization+routing still underperforms the global CVAE by a large margin.
- Current evidence indicates pooled modeling is stronger than routed experts in this setup.

## Expert specialization diagnostic
From `artifacts/reports/expert_matrix.json` reconstruction matrix:
- `E40` is best on `40x`.
- `E200` is best on `200x`.
- `E400` is best on `400x`.
- `100x` is only weakly specialized (`E200` slightly better than `E100`).

Overall: specialization exists but is partial and not strong enough to beat the global model.

## Magnification signal checks (A/B/C + Case 3)
From `results/magnification_signal/magnification_signal_report.json`:
- Linear magnification classifier on embeddings (A): `0.67` test accuracy (moderate signal).
- PCA/UMAP visualization (B): generated; visual separation is limited.
- Inter-vs-intra distance check (C): inter/intra ratio is `1.032` (weak separation).
- Decision grid result: `mixed` (not strong Case 1, not pure Case 2).

From `results/magnification_signal/case3_raw_vs_embedding_report.json`:
- Embedding linear probe accuracy: `0.6633`.
- Raw-image linear probe accuracy (64x64): `0.4567`.
- Delta (`raw - embedding`) = `-0.2067`.
- Case 3 status: **not supported**.

Interpretation of Case 3:
- The representation issue is not that ResNet18 embeddings are worse than raw pixels for magnification.
- Embeddings retain more linearly usable magnification signal than raw pixels in this setup.
- The likely bottleneck remains weak/partial domain separation and limited expert advantage versus pooled modeling.

## Practical conclusion for thesis framing
- The methodological concern (patient leakage) is resolved.
- The negative result persists after the fix: routing does not beat global CVAE.
- Magnification signal exists but is only moderately separable; this weakens the potential gain from routing.
- It is reasonable to report that BreakHis may be a weak dataset for showcasing routing gains under this toy CVAE setup.

## Suggested next step before switching datasets
1. Keep this as a completed, leakage-correct baseline.
2. Optionally run 2-3 seeds for stability check.
3. If trend persists, move to another dataset for routing demonstration.
