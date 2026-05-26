# Generated-Embedding CVAE Current Synthesis

## Purpose

State the current best interpretation after the Virchow2 CVAE rebuild,
source-union GMM prior diagnostics, and D-series decentralized composition
experiments.

## Key Claims

- The current generated-embedding bottleneck is latent prior/composition, not
  simply Virchow2 feature quality.
- Centralized source-union K16 is the strongest CVAE prior-preservation
  diagnostic, but it is not deployable decentralized routing.
- The paired dense all4 reliability confirmation is the strongest
  protocol-clean dense generated-embedding aggregation result.
- D1.3/D1.3.1 do not validate support-NELBO as the final compatibility signal
  because shuffled-support controls remain competitive.
- D1.5 provides negative evidence for source-inner off-diagonal transfer as a
  drop-one source selector.
- No generated-embedding artifact yet proves sparse routing or target-specific
  expert selection.

## Evidence / Source Artifacts

See `../../context/current_experimental_state.md` for the full evidence list.

Primary synced artifact roots:

- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_preservation_repair_v1/`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_source_union_gmm_prior_v1/`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_decentralized_reliability_weighted_gmm_prior_v1/`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_decentralized_support_nelbo_reliability_gmm_prior_v1/`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_decentralized_source_inner_transfer_top3_gmm_prior_v1/`
- `../../../cvae_rebuild/artifacts/virchow2_cvae_paired_dense_all4_reliability_confirmation_v1/`

## Interpretation

The evidence separates three layers:

```text
real-feature transfer ceiling
CVAE decoder/source-pool preservation capacity
decentralized prior/composition/routing
```

The Virchow2 CVAE repair result shows that generated-embedding utility can be
high when latent inputs are favorable:

```text
decode(mu) mean BACC: 0.8572
```

The source-union K16 GMM diagnostic shows that prior sampling can recover much
of this utility:

```text
source_union_cc_diag_gmm_k16_prior_sample_diagnostic mean BACC: 0.8924
```

The decentralized experiments show that preserving this utility without pooled
source fitting is harder:

```text
D1.2 reliability-weighted adaptive summaries mean BACC: 0.8493
D1.3 support-NELBO x reliability mean BACC: 0.8495
D1.5 source-inner transfer top3 mean BACC: 0.8354
paired dense all4 reliability-weighted mean BACC: 0.8506
```

## Current Best Generated-Embedding Method

Best diagnostic upper bound:

```text
centralized source-union K16 GMM prior
```

Best protocol-clean decentralized dense aggregation method:

```text
adaptive source-local latent summaries
+ heldout-excluded source-local generation-preservation reliability weighting
+ paired dense all4 generated-embedding aggregation
```

This is a dense preservation/composition result, not sparse expert selection or
a target-specific compatibility router.

Key paired confirmation numbers:

```text
decision report primary method: paired_reliability_all4_shrink050_geom
best reliability method: paired_reliability_all4_weighted_geom
equal all4 center-equal BACC: 0.8235
full reliability-weighted center-equal BACC: 0.8506
delta vs equal all4: +0.0271
delta vs strongest negative control: +0.0416
pairing invariant audit: 420/420 PASS
positive paired cells: 9/14
centers improved: 4/5
gap vs real-feature dense reference: -0.0570
```

## What Failed

Support-NELBO:

- D1.3 and D1.3.1 show some alignment signal.
- D1.3.1 reached Spearman 0.3143 and top3 oracle containment 0.8571.
- Shuffled-support control remained competitive and beat the D1.3.1 primary by
  0.0096 BACC.

Conclusion:

```text
support-NELBO remains diagnostic; not validated as final routing evidence.
```

Source-inner transfer:

- D1.5 source-inner Spearman vs target subset utility: -0.1102.
- D1.5 underperformed equal all4 by 0.0165 BACC.
- D1.5 underperformed shuffled-score control by 0.0326 BACC.

Conclusion:

```text
source-inner off-diagonal transfer should not be extended directly as the next
selector.
```

Reliability-only sparse top3:

- D1.4 underperformed equal all4 by 0.0062 BACC.
- Random source-drop and shuffled-reliability controls were competitive.

Conclusion:

```text
source-local reliability is useful for weighting, not validated as sparse
drop-one selection.
```

## Implication For Thesis

The thesis can use the generated-embedding results as a disciplined progression:

```text
vanilla prior failed
-> decoder/source-pool capacity exists
-> K16 source-union prior diagnoses the prior bottleneck
-> paired dense all4 reliability weighting improves dense aggregation
-> target-conditioned support-NELBO and source-inner transfer are negative/mixed
```

This is a coherent thesis contribution: it identifies a protocol-clean dense
generated-embedding aggregation win, while also showing that sparse or
target-conditioned compatibility routing remains unresolved.

## Limitations

- The paired dense all4 confirmation is a PASS for dense aggregation only.
- K16 source-union is centralized.
- D1.5 found a paired-sampling audit issue; the paired dense all4 confirmation
  fixed that issue for dense all4 comparisons, but future sparse selectors must
  reuse the same invariant discipline.
- No formal differential privacy claim is supported.

## Next Checks

1. Reuse paired generation-cache invariants for any future selector comparison.
2. Keep equal all4 and paired reliability-weighted dense all4 as fixed
   generated-embedding baselines.
3. Treat support-NELBO and source-inner transfer as diagnostic until they beat
   matched controls.
4. Decide whether the thesis contribution should emphasize partial
   data-minimizing composition evidence rather than a final target-conditioned
   router.
