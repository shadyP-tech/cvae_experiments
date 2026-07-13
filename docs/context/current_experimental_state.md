# Current Experimental State

Last updated: 2026-07-12

This page records verified evidence and canonical availability after the
completed MIDOG++ repository migration. Active inputs and the two tuned
evidence bundles are present locally and on the workstation at canonical paths
with cataloged hash verification.

## Local Readiness Snapshot

- `conda run -n thesis python -m midogpp_thesis workspace validate` passes once
  the package is installed from this repository.
- The complete 22,569-patch contract is present locally and on the workstation
  under `datasets/midogpp/contract/annotation_patch_v1/`; `dataset-validate`
  passes with nine eligible domains.
- The corrected `xyxy` Virchow2 cache is present locally and on the workstation
  under
  `datasets/midogpp/derived/features/virchow2/annotation_patch_xyxy/seed42/`.
  Train, validation, and test counts align exactly with the contract, and the
  train tensor hash is
  `f6608e513fb2d06671e3ec117b093a85d58530b77b1fae44a3be1680d9feabd2`.
- The tuned real-feature reference and tuned CVAE preservation bundle are
  present locally and on the workstation at their canonical stage-10 and
  stage-20 paths and validate `PASS`.
- The approximately 65 GB raw source tree is workstation-only at
  `datasets/midogpp/raw/MIDOGpp/` and intentionally not synced to the Mac.
- Stages 30 through 70 have no active expert-bank, generation, routing, or
  frozen-policy downstream implementation. Their directories and protocol
  contracts are planning scaffolds, not experimental evidence.

## MIDOG++ Real-Feature Gate

Canonical target:

```text
artifacts/midogpp/10_real_feature_reference/midogpp_real_feature_gate_v1/
```

Availability: verified historical evidence, not currently present in the local
canonical target. The original verified source was the retired path
`midogpp_real_feature_gate/artifacts/midogpp_real_feature_gate_v1/`; that string
is provenance only and is not an active path.

Validation status:

- schema: `midogpp_real_feature_transfer_ceiling_v1`
- leakage/provenance report: `PASS`
- decision labels: `GO_REAL_FEATURE_GATE_PASSED` and
  `CLAIM_SCOPE_REAL_FEATURE_TRANSFER_ONLY`
- valid source-only held-out centers: 9/9 eligible centers
- center `4`: quarantine-only diagnostic row

Result summary:

- source-only mean BACC `0.668`, macro-F1 `0.662`, AUROC `0.728`, and PR-AUC
  `0.737`
- pooled diagnostic ceiling mean BACC `0.902`, macro-F1 `0.902`, AUROC `0.964`,
  and PR-AUC `0.969`

Interpretation: `USABLE WITH CAVEATS` for real-feature source-only signal and
headroom. It is not CVAE preservation, compatibility, routing, generation, or
synthetic downstream-utility evidence.

## MIDOG++ Source-Inner Reliability Dense Ensemble

Canonical local artifact:

```text
artifacts/midogpp/10_real_feature_reference/midogpp_source_inner_reliability_v1/
```

Validation status:

- protocol boundary: `PASS`
- evidence labels: `NEGATIVE_RESULT`, `DIAGNOSTIC_ONLY`, and
  `TODO_VERIFY_ARTIFACT` for missing bundle elements
- target expert excluded; target labels scoring-only
- source-inner normalization used target center: `false`
- target labels used for weights: `false`
- uniform dense baseline included
- no copied config snapshot or decision report is present in the retained
  bundle

Result summary:

- source-inner weighted mean BACC `0.689295`, macro-F1 `0.686507`, AUROC
  `0.777831`
- uniform dense mean BACC `0.689704`, macro-F1 `0.685938`, AUROC `0.786136`
- weighted selection wins on 5/9 centers but is slightly worse on mean and
  worst-center BACC

Interpretation: protocol-clean negative evidence for this exact source-inner
softmax dense weighting rule. Uniform dense remains the stronger baseline for
this run. This real-feature result does not establish CVAE or routing claims.

## MIDOG++ Tuned Real-Feature Classifier Reference

Canonical target:

```text
artifacts/midogpp/10_real_feature_reference/real_feature_threshold_both_annotation_patch_xyxy_virchow2_seed42/
```

Availability: `local_and_workstation`, migrated with byte-identical pre/post
manifests. The retired workstation source path is absent; it remains only in
the migration audit as provenance.

Validation status:

- workstation leakage/protocol inspection: `PASS`
- schema: `midogpp_real_feature_source_only_classifier_reference_v1`
- evidence label: `WEAK_PASS_REAL_FEATURE_TRANSFER_ONLY`
- claim scope: `real_feature_transfer_only`
- generated embeddings and CVAE checkpoint used: `false`
- router: `false`
- target labels used for selection: `false`
- target labels used for final scoring only: `true`
- manifest hash:
  `db661ac7e3dbafde8e283528de6706ab35f2c26629b389706c4504e458cc5869`
- feature-cache hash:
  `f6608e513fb2d06671e3ec117b093a85d58530b77b1fae44a3be1680d9feabd2`

Result summary:

- tuned fixed-0.5 mean BACC `0.740312`, macro-F1 `0.737205`
- untuned fixed-0.5 mean BACC `0.665812`, macro-F1 `0.661730`
- tuned-minus-default mean BACC `+0.074500`
- tuned classifier wins on all 9 eligible held-out centers
- worst tuned center: center `1`, BACC `0.679245`
- best tuned center: center `6`, BACC `0.792350`

Threshold conclusion: the source-inner one-SE rule selected `0.5` for every
fold, so threshold tuning changed no prediction and added `+0.000000` mean
BACC. The diagnostic raw argmax thresholds had mixed target effects and are not
an adoption rule.

Interpretation: `WEAK PASS` for source-inner real-feature classifier tuning.
This is the frozen comparator for stage-20 preservation; it does not establish
CVAE preservation, routing, or generated-embedding utility.

Missing evidence:

- classifier-seed stability beyond seed `23`
- formal paired comparison against another active real-feature aggregation
  baseline on hash-matched inputs

## MIDOG++ Tuned-Classifier CVAE Preservation

Canonical target:

```text
artifacts/midogpp/20_cvae_preservation/virchow2_cvae_midogpp_tuned_classifier_preservation_v1/seed42/
```

Availability: `local_and_workstation`, migrated with byte-identical pre/post
manifests. The retired workstation source path is absent; it remains only in
the migration audit as provenance.

Validation status:

- verdict: `THESIS-FACING` for `claim_scope=cvae_preservation_only`
- leakage report and identity-overlap audit: `PASS`
- zero overlap for `sample_id`, `case_id`, `image_path`, and
  `feature_row_index`
- generated embeddings and CVAE checkpoint used: `true`
- target labels used for selection: `false`
- target center used during fit: `false`
- target evaluation labels used for scoring only: `true`
- eligible held-out centers: `0,1,2,3,5,6,7,8,9`
- imported real-reference hash: `78c1d254019a2cc0`

Result summary:

| Representation | Mean BACC | Macro-F1 | Preservation ratio |
| --- | ---: | ---: | ---: |
| imported tuned real-feature reference | `0.740312` | `0.737205` | — |
| `decode_mu_fit_to_real_eval` | `0.719681` | `0.717766` | `0.919368` |
| `posterior_sample_fit_to_real_eval` | `0.716630` | `0.714110` | `0.910740` |
| `prior_sample_fit_to_real_eval` | `0.637563` | `0.630151` | `0.571675` |
| `real_pca128_reference` | `0.720533` | `0.718135` | `0.922785` |

Interpretation: the single `pca128_beta001` run supports the narrow claim that
decode and posterior representations preserve almost all of the PCA128
real-feature classifier surface over nine centers. Prior sampling is materially
weaker and does not support an unconditional-generation utility claim.

This artifact is explicitly forbidden as routing, expert-selection,
NELBO-compatibility, controllable-generation, GMM-composition, or downstream
synthetic-utility evidence. It has
`may_feed_deployable_selection=false`.

Next preservation-specific evidence:

- seed or variant stability for the same predeclared preservation protocol
- separate expert-bank, generation, routing, and held-out downstream stages
  before making any broader thesis claim

## Quarantine And Planned Work

Repository migration audit:

```text
artifacts/midogpp/90_oracles_and_diagnostics/repository_migration/2026-07-12_xai_master/
```

The audit is `AUDIT_ONLY` and present locally and on the workstation. Its
catalog-pinned pre/post manifests match for the complete contract, corrected
cache, tuned real-feature reference, and tuned preservation bundle. Raw-data
verification preserves stable relative-path/size metadata and hashes for six
critical files. All retired workstation source artifact/data locations are
absent.

- The stale `coco_xywh` config and cache lineage are `REJECTED` and live under
  `datasets/midogpp/configs/quarantine/` and
  `datasets/midogpp/derived/features/quarantine/`.
- Rejected legacy expert-bank, prior, and routing artifacts live under
  `artifacts/midogpp/90_oracles_and_diagnostics/rejected/`; they are audit-only
  and cannot seed new stages 30 through 70.
- The retained stage-50 phase-1 bundle is a historical post-hoc diagnostic. Its
  target utility and oracle rows cannot train or select a deployable router.
- BreakHis, Camelyon17, and generic historical material is outside the active
  registry under `artifacts/cross_dataset_archive/`.
- The immediate experimental bottlenecks are preservation seed or variant
  stability and a new provenance-clean independently trained MIDOG++
  source-expert bank. Routing is premature until that bank and a fresh
  protocol-clean utility surface exist.
