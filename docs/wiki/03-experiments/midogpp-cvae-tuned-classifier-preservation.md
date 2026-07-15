# MIDOG++ CVAE Tuned-Classifier Preservation

## Purpose

This note tracks the MIDOG++ pca128 CVAE preservation surface that reuses the
source-inner tuned real-feature classifier reference as a frozen comparator and
classifier-spec source.

It answers a CVAE preservation question: when the per-held-out-center
classifier specs are imported from the real Virchow2 reference, how much
downstream class-discriminative signal is preserved by `decode_mu`,
`posterior_sample`, and `prior_sample` representations?

## Evidence Source

Config:

```text
experiments/midogpp/stages/20_cvae_preservation/configs/tuned_classifier_preservation_v1.yaml
```

Experiment ID:

```text
midogpp.cvae.tuned_classifier_preservation.v1
```

Implementation:

```text
src/midogpp_thesis/cvae/preservation/tuned_classifier.py
src/midogpp_thesis/cvae/preservation/tuned_reference.py
src/midogpp_thesis/cvae/preservation/cli.py
src/midogpp_thesis/workspace/runtime.py
```

Canonical verified artifact root:

```text
artifacts/midogpp/20_cvae_preservation/virchow2_cvae_midogpp_tuned_classifier_preservation_v1/seed42/
```

Current artifact status: `THESIS-FACING` for the narrow
`cvae_preservation_only` claim and `local_and_workstation` availability. The
retired workstation source is absent; matching pre/post manifests and its
original name remain only in the stage-90 repository-migration audit.

Canonical preparation command:

```bash
conda run -n thesis python -m midogpp_thesis workspace prepare \
  midogpp.cvae.tuned_classifier_preservation.v1
```

## Inputs

- MIDOG++ contract: `midogpp_dataset_contract_annotation_patch_v1`
- corrected real Virchow2 `xyxy` cache:
  `midogpp_virchow2_xyxy_feature_cache_seed42`
- real-feature tuned classifier reference:
  `midogpp_real_feature_threshold_both_xyxy_seed42`

The imported reference is required to pass schema, leakage, and protocol checks
before preservation scoring is interpreted. The canonical workspace resolves
these logical IDs, verifies cataloged expected file hashes, and records SHA-256
for authoritative inputs in `provenance/input_artifacts.json`.

## Protocol Guardrails

- The imported reference must have
  `schema_version=midogpp_real_feature_source_only_classifier_reference_v1`.
- The imported reference must keep `claim_scope=real_feature_transfer_only`.
- The imported reference leakage report must be `PASS`.
- The imported reference must have `selection_used_target_labels=false`,
  `fit_used_target_center=false`, `generated_embeddings_used=false`,
  `cvae_checkpoint_used=false`, `source_summary_manifest_used=false`, and
  `is_router=false`.
- The new preservation artifact must keep `claim_scope=cvae_preservation_only`.
- Target labels are final scoring only; they must not select classifier specs,
  thresholds, calibration, routing, or generation settings.
- This artifact has `may_feed_deployable_selection=false` and explicitly
  forbids reuse as routing, expert-selection, NELBO-compatibility, or synthetic
  downstream-utility evidence.

## Expected Outputs

```text
tables/tuned_preservation_metrics.csv
tables/imported_real_tuned_reference.csv
tables/reconstruction_diagnostics.csv
tables/training_diagnostics.csv
tables/identity_overlap_audit.csv
tables/predictions.csv
manifests/protocol_manifest.json
reports/leakage_report.json
reports/decision_report.md
```

The primary interpretation table is `tables/tuned_preservation_metrics.csv`.
It should include seed-level and summary rows for:

- `real_pca128_reference`
- `decode_mu_fit_to_real_eval`
- `posterior_sample_fit_to_real_eval`
- `prior_sample_fit_to_real_eval`

## Result

Comparator: imported tuned real-feature reference mean BACC `0.740312` and
macro-F1 `0.737205`.

| Representation | Mean BACC | Macro-F1 | Preservation ratio |
|---|---:|---:|---:|
| `decode_mu_fit_to_real_eval` | `0.719681` | `0.717766` | `0.919368` |
| `posterior_sample_fit_to_real_eval` | `0.716630` | `0.714110` | `0.910740` |
| `prior_sample_fit_to_real_eval` | `0.637563` | `0.630151` | `0.571675` |
| `real_pca128_reference` | `0.720533` | `0.718135` | `0.922785` |

The leakage report and identity-overlap audit are `PASS`; the identity audit
found zero overlap across `sample_id`, `case_id`, `image_path`, and
`feature_row_index` for the nine eligible centers.

## Claim Classification

`THESIS-FACING` for the narrow `cvae_preservation_only` claim. Decode and
posterior representations preserve almost all of the PCA128 real-feature
classifier surface, while the weaker prior-sampling result does not support a
broad unconditional-generation utility claim.

Do not promote this surface into a routing, expert-selection,
metadata-compatibility, NELBO-compatibility, controllable-generation,
GMM-composition, or downstream synthetic-utility claim.

## Next Evidence Needed

- Run a predeclared seed or variant stability check for the preservation
  surface.
- Treat prior generation as a separate bottleneck requiring its own
  generation and held-out downstream evidence.
- Build the provenance-clean independently trained source-expert bank before
  starting routing or composition claims.

## Relationship To The Prior-Recovery Surface

The new `midogpp.cvae.prior_recovery_source_inner.v1` and
`midogpp.cvae.prior_recovery_outer.v1` experiments are separate from this
validated v1 artifact. They use a new eligible-only Stage-10 matched reference,
fully nested source-inner `RecipeLock` selection, and a conditionally unlocked
A/B/C/D outer factorial. The Stage-10 reference and source-inner result are now
complete and validated on `xai-master`; the source-inner gate is
`NEGATIVE_GATE_COMPLETE` with two standard-normal fallback locks, so the outer
experiment remains blocked. This page's preservation metrics must not be
presented as prior-recovery or Task-Fisher results.

Validated source-inner locks may later feed a Stage-30 expert recipe. Outer
preservation metrics may never feed model or routing selection, and Stage 40
remains post-expert-bank generation validation.
