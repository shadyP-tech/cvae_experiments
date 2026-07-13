# MIDOG++ Real-Feature Source-Inner Classifier Reference

RESULT INTERPRETATION:

## Evidence Source

Canonical artifact target:

```text
artifacts/midogpp/10_real_feature_reference/real_feature_threshold_both_annotation_patch_xyxy_virchow2_seed42/
```

The validated bundle is present locally and on the workstation at this
canonical path. Its retired workstation source is absent; matching pre/post
manifests and the original source name are retained only in the stage-90
repository-migration audit.

Inspected files:

- `manifests/protocol_manifest.json`
- `reports/leakage_provenance_report.json`
- `tables/source_inner_classifier_tuning.csv`
- `tables/classifier_tuned_source_results.csv`
- `tables/classifier_tuned_predictions.csv`

Workstation validation:

```text
leakage/protocol inspection -> PASS
```

Run configuration:

- dataset: MIDOG++
- feature frame: real Virchow2 train cache
- eligible centers: `0,1,2,3,5,6,7,8,9`
- classifier grid: `C in {0.01,0.1,1,10,100}`, `penalty=l2`,
  `solver=lbfgs`, `class_weight in {none,balanced}`,
  `max_iter in {2000,5000}`
- classifier seed: `23`
- experiment seed: `42`
- threshold variants: fixed `0.5` and source-inner selected threshold
- threshold rule:
  `source_inner_macro_center_bacc_one_se_closest_0_5_v1`
- feature cache:
  `datasets/midogpp/derived/features/virchow2/annotation_patch_xyxy/seed42/embeddings/train.pt`

Canonical implementation and registered runner:

```text
src/midogpp_thesis/real_features/classifier_reference/
midogpp.real_feature.tuned_classifier.seed42
```

The corrected cache is present and hash-verified. Prepare with:

```bash
conda run -n thesis python -m midogpp_thesis workspace prepare \
  midogpp.real_feature.tuned_classifier.seed42
```

Protocol fields recorded in the artifact:

- `claim_scope=real_feature_transfer_only`
- `generated_embeddings_used=false`
- `cvae_checkpoint_used=false`
- `source_summary_manifest_used=false`
- `is_router=false`
- `selection_used_target_labels=false`
- target labels used for final held-out scoring only

## Thesis Question

Does source-inner classifier hyperparameter selection improve the MIDOG++
real-feature source-only transfer reference without target-label leakage?

This answers a real-feature reference-surface question. It does not answer
whether CVAE-generated embeddings preserve utility, whether NELBO compatibility
aligns with downstream utility, or whether a router can identify a target
compatible expert.

## Primary Metrics

The artifact does not contain downstream oracle hit, support-NELBO-to-downstream
Spearman, downstream oracle gap, top-1 expert hit, or NELBO utility deltas.
Those metrics are not applicable to this real-feature source-only classifier
reference.

Held-out target utility from real-feature classifiers:

| Method | Mean BACC | Mean Macro-F1 | Worst Center BACC | Best Center BACC |
| --- | ---: | ---: | ---: | ---: |
| source-inner tuned, fixed 0.5 | 0.740312 | 0.737205 | center 1 = 0.679245 | center 6 = 0.792350 |
| source-inner tuned, source-inner threshold | 0.740312 | 0.737205 | center 1 = 0.679245 | center 6 = 0.792350 |
| default untuned, fixed 0.5 | 0.665812 | 0.661730 | center 2 = 0.594595 | center 8 = 0.729338 |
| default untuned, source-inner threshold | 0.665812 | 0.661730 | center 2 = 0.594595 | center 8 = 0.729338 |

Tuned minus default:

- mean BACC delta at fixed `0.5`: `+0.074500`
- center wins: `9/9`
- largest gain: center `3`, `+0.117801`
- smallest gain: center `1`, `+0.032345`

Threshold result:

- source-inner threshold selection chose `0.5` for every held-out center
- fixed-0.5 and source-inner-threshold predictions are identical within each
  classifier variant
- threshold tuning gain over fixed `0.5`: `+0.000000` mean BACC
- raw source-inner argmax thresholds were not always `0.5`
  (`0.58,0.54,0.50,0.49,0.52,0.55,0.64,0.46,0.52` by held-out center), but
  their source-inner advantages over `0.5` were smaller than the one-SE margin
  used by the predeclared rule
- diagnostic-only target scoring with the raw argmax thresholds had mixed
  center effects and only about `+0.0003` mean BACC, so the protocol-clean
  conclusion remains that threshold tuning is inert here

Selected classifier pattern:

- all held-out centers selected `C=0.01`, `penalty=l2`, `solver=lbfgs`,
  `max_iter=2000`
- centers `0,1,2,9` selected `class_weight=balanced`
- centers `3,5,6,7,8` selected `class_weight=none`

## Baseline Comparison

The artifact includes an untuned default classifier baseline and the tuned
method beats it on all nine held-out centers.

Contextual comparison against the earlier real-feature source-inner reliability
artifact:

- previous uniform dense mean BACC: `0.689704`
- previous source-inner weighted dense mean BACC: `0.689295`
- current source-inner tuned mean BACC: `0.740312`

This contextual comparison is useful for planning, but it is not a formal
paired baseline row in the new artifact. Treat it as evidence that classifier
regularization/class-weight tuning is a strong next real-feature reference
surface, not as proof that a routing or weighting policy works.

## Claim Classification

`WEAK PASS` for the narrow real-feature source-only classifier-tuning question.

Reason: source-inner-only model selection improves balanced accuracy
substantially and consistently over the untuned default while preserving target
labels as final scoring only. The threshold branch is `DIAGNOSTIC ONLY` as a
method change because it selected the fixed `0.5` threshold everywhere and did
not change predictions. The result lacks seed stability, oracle/ranking metrics,
and a formal paired SAIL comparison, so it should not be promoted to a strong
thesis-facing PASS.

For routing, compatibility, CVAE preservation, synthetic downstream utility, or
generative quality claims, classify this artifact as `DIAGNOSTIC ONLY`.

## Thesis Text

Source-inner classifier selection on real Virchow2 MIDOG++ embeddings improved
the source-only transfer reference from 0.6658 to 0.7403 mean held-out-center
balanced accuracy, with gains on all nine eligible centers and no target-label
use during selection.

The source-inner thresholding follow-up selected the fixed `0.5` decision
threshold in every held-out-center fold; threshold tuning therefore did not
improve BACC beyond classifier hyperparameter/class-weight selection.

This result strengthens the real-feature reference surface for later MIDOG++
CVAE candidate-surface experiments, but it does not establish CVAE utility
preservation, NELBO compatibility, or deployable routing quality.

## Caveats

- Only one classifier seed was run.
- The artifact does not include oracle gap, top-1/top-k behavior, Spearman
  ranking, or per-expert utility matrices.
- The comparison to the previous uniform dense real-feature artifact is
  contextual rather than a formal paired row in this artifact.
- Probabilities are marked uncalibrated.
- Threshold selection was source-inner and protocol-clean, but the selected
  threshold was `0.5` in every held-out-center fold.
- No target labels may be used to choose thresholds, calibration, routing, or
  policy changes after this result.

## Next Evidence Needed

- Repeat the same source-inner classifier tuning over additional classifier
  seeds or a predeclared seed-stability axis.
- Compare against another canonical real-feature aggregation baseline using
  hash-matched inputs or a formal paired report.
- Do not spend another run on threshold tuning unless the threshold rule changes
  for a predeclared reason; this one-SE source-inner rule is empirically inert
  on the corrected `xyxy` cache.
- If calibration is explored, select it source-inner only and freeze it before
  target-center scoring.
- For CVAE claims, run generated-embedding downstream utility artifacts using
  this real-feature reference only as a comparison surface.
