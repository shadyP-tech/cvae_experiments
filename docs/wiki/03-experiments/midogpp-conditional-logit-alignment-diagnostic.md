# MIDOG++ Conditional-Logit Alignment Diagnostic

RESULT INTERPRETATION:

## Purpose

This Stage-10 experiment asks whether a source-inner-selected
class-conditional centroid-contrast penalty improves a fixed pooled logistic
classifier on held-out MIDOG++ centers. It is a standalone real-feature
mechanism diagnostic, not a replacement for the matched real-feature reference
and not an input to Stage 20 or later selection.

Experiment identity:

```text
midogpp.real_feature.conditional_logit_alignment.v1
```

Final classification:

- run status: `COMPLETE`
- protocol status: `PASS`
- evidence labels: `NEGATIVE_RESULT`, `DIAGNOSTIC_ONLY`
- claim scope: `real_feature_transfer_only`
- adoption: forbidden

## Evidence And Reproducibility

Canonical relative artifact root:

```text
artifacts/midogpp/10_real_feature_reference/conditional_logit_alignment_v1/seed42/
```

Verified workstation root:

```text
/home/stud/spark/cvae_experiments/artifacts/midogpp/10_real_feature_reference/conditional_logit_alignment_v1/seed42/
```

The bundle was inspected directly on `xai-master`. It is not present in the
local checkout. Its artifact-catalog entry still has the lifecycle label
`TODO_VERIFY_ARTIFACT`; local sync and catalog metadata promotion remain TODOs.
That stale metadata does not make the complete workstation bundle adoptive.

Authoritative repository definitions:

```text
experiments/midogpp/registry.yaml
experiments/midogpp/artifact_catalog.yaml
experiments/midogpp/stages/10_real_feature_reference/configs/conditional_logit_alignment_v1.yaml
```

Primary inspected files:

- `config.resolved.yaml`
- `provenance/input_artifacts.json`
- `manifests/frozen_protocol_snapshot.json`
- `manifests/protocol_manifest.json`
- `manifests/content_index.json`
- `reports/leakage_provenance_report.json`
- `reports/decision_summary.json`
- `reports/decision_report.md`
- `reports/runtime_summary.json`
- `tables/source_inner_fold_scores.csv`
- `tables/source_inner_gamma_summary.csv`
- `tables/outer_results.csv`
- `tables/outer_predictions.csv`
- `tables/conditional_frame_audit.csv`
- `tables/solver_audit.csv`
- `tables/outer_comparison.csv`

Key content identities:

| Identity | Value |
| --- | --- |
| protocol hash | `3806cca63f914a09` |
| design hash | `03f06e799c24f625` |
| table-bundle hash | `4b19d9052da239f6` |
| decision hash | `f60a3c2455ba3f0c` |
| contract-manifest hash | `db661ac7e3dbafde8e283528de6706ab35f2c26629b389706c4504e458cc5869` |
| feature-cache hash | `f6608e513fb2d06671e3ec117b093a85d58530b77b1fae44a3be1680d9feabd2` |

Reproduce through the registered workspace entry:

```bash
conda run -n thesis python -m midogpp_thesis workspace prepare \
  midogpp.real_feature.conditional_logit_alignment.v1
conda run -n thesis python -m midogpp_thesis workspace run \
  midogpp.real_feature.conditional_logit_alignment.v1
```

The recorded runtime environment is single-threaded and version-matched:
NumPy `1.26.4`, SciPy `1.17.1`, and scikit-learn `1.8.0`.

## Design And Guardrails

The run uses the corrected real Virchow2 `xyxy` train cache with 2,560 feature
dimensions and the nine eligible centers `0,1,2,3,5,6,7,8,9`; quarantined
center `4` is excluded.

For each outer held-out center, eight source-inner center-LODO folds select one
gamma from:

```text
0, 0.0001, 0.001, 0.01, 0.1, 1, 10
```

Selection maximizes equal-center mean source-inner BACC and breaks exact ties
toward the smallest gamma. The outer fold scores only the selected gamma and a
matched `gamma=0` baseline. It does not score all gamma values on the held-out
center and does not compute an outer oracle gamma.

The protocol checks establish:

- the outer center is excluded from the scaler, conditional operator,
  classifier fit, and gamma selection;
- each inner pseudo-target is excluded from its inner-fold preprocessing,
  operator construction, and fit;
- sample, case, image-path, and feature-row overlap counts are zero;
- target evaluation labels are used for scoring only;
- no generated embeddings, CVAE checkpoint, expert bank, support labels,
  NELBO, router, expert selection, weighting, or aggregation are used;
- `may_feed_recipe_selection=false` and
  `may_feed_deployable_selection=false`.

## Primary Result

The predeclared primary contrast is selected CLA minus matched `gamma=0` under
equal-center mean BACC.

| Quantity | Selected CLA | `gamma=0` | Delta |
| --- | ---: | ---: | ---: |
| mean BACC | `0.7434042753` | `0.7434898099` | `-0.0000855346` |
| mean macro-F1 | `0.7414722419` | `0.7414225940` | `+0.0000496479` |
| worst-center BACC | `0.6913746631` | `0.6913746631` | `0.0000000000` |

The positive-mean-BACC gate fails. The nonworse-worst-center and at-least-five
nonnegative-center checks pass, but they cannot compensate for the failed
primary requirement.

Center-level BACC outcomes are:

- wins: `2/9`
- ties: `4/9`
- losses: `3/9`

Source-inner selection chooses a positive gamma for every outer center:

- `gamma=10`: seven centers
- `gamma=1`: two centers

This establishes that the mechanism was selected and active, but not that it
improved held-out utility.

## Mechanism And Prediction Audit

Across the 9,648 held-out predictions, selected CLA changes only 22 labels
relative to `gamma=0`:

- 9 flips correct a `gamma=0` error;
- 13 flips introduce an error.

The source-inner versus outer per-center improvement Spearman correlation is
`0.052`, so source-inner gains do not meaningfully rank outer gains in this
run. This correlation is a post-run mechanism diagnostic, not a deployable
selection score.

The conditional operator is deliberately low rank. In the outer folds it has
rank 14 within the 2,560-dimensional standardized feature frame; source-inner
operators have rank 12. The supplied artifact-backed causal audit synthesis
supports the narrow interpretation that the regularizer acts in a low-overlap
subspace relative to the fitted discriminative direction. Most probability
changes remain on the same side of the classifier decision threshold,
producing threshold plateaus.

The defensible conclusion is therefore **mechanism-active but utility-inert**.
This wording is a causal-mechanism audit interpretation, not evidence that CLA
causally removes domain shift. Probability changes and any post-hoc probability
metric are diagnostic only; none was part of the predeclared adoption gate and
none can overturn the negative BACC decision.

## Claim Classification

`NEGATIVE_RESULT` for the tested hypothesis that source-inner-selected CLA
improves the matched pooled real-feature classifier under the predeclared BACC
gate.

`DIAGNOSTIC_ONLY` for every use beyond that narrow negative result. The bundle
cannot:

- replace the matched Stage-10 real-feature denominator;
- adopt a classifier, gamma, regularizer, recipe, or deployable policy;
- feed Stage-20 CVAE preservation or any expert-bank, prior, generation,
  routing, composition, or downstream selection;
- support claims about CVAE preservation, expert compatibility, NELBO,
  synthetic generation, or downstream utility;
- support a causal domain-shift-removal claim.

## Limitations

- This is one experiment seed and one classifier seed on one corrected feature
  cache.
- The gamma grid is bounded at `10`, but the negative utility result does not
  justify expanding it post hoc.
- Exact ties and the very small number of prediction flips show that BACC is
  locally insensitive to many probability changes.
- The Spearman, flip decomposition, low-overlap interpretation, and any
  probability-based inspection are post-run diagnostics, not selection rules.
- The bundle remains workstation-only, and the catalog lifecycle metadata has
  not yet been promoted from `TODO_VERIFY_ARTIFACT`.

## Stop Recommendation

Do not adopt CLA and do not expand the gamma sweep. The method produced no mean
BACC benefit, did not improve worst-center BACC, and introduced more prediction
errors than it corrected.

Optional future work is justified only if it is predeclared as a bounded
mechanism diagnostic, with no authority to revise the current Stage-10
denominator or feed later evidence stages. The active thesis sequence remains:

```text
real-feature reference -> CVAE preservation -> routing/composition -> downstream utility
```

## Remaining TODOs

- Sync the complete canonical bundle from `xai-master` into the local canonical
  artifact root.
- After sync, independently revalidate hashes and all 16 required files.
- Promote the artifact catalog lifecycle label only after that sync and
  validation; do not edit `experiments/midogpp/artifact_catalog.yaml` from this
  documentation update.
