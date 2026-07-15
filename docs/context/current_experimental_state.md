# Current Experimental State

Last updated: 2026-07-15

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
- The Stage-10 matched-reference v2 and Stage-20 source-inner
  prior-recovery/Task-Fisher runs are complete on `xai-master` and validate
  `PASS`. These new workstation bundles have not yet been synced into this
  local checkout. The registered Stage-20 outer run remains blocked by the
  completed source-inner gate.

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

## Prior-Recovery And Task-Fisher Result

Status: the Stage-10 matched reference and Stage-20 source-inner bundle are
complete and directly validated on `xai-master`. The catalog still labels these
canonical output destinations `TODO_VERIFY_ARTIFACT`; catalog promotion and a
local artifact sync remain separate follow-up work.

| Stage | Experiment | Canonical output | Current evidence |
| --- | --- | --- | --- |
| 10 | `midogpp.real_feature.eligible_tuned_predict_reference.v2` | `artifacts/midogpp/10_real_feature_reference/eligible_tuned_real_reference_v2/seed42/` | validator `PASS` |
| 20 | `midogpp.cvae.prior_recovery_source_inner.v1` | `artifacts/midogpp/20_cvae_preservation/prior_recovery_source_inner_v1/seed42/` | `COMPLETE`; full validator `PASS`; `NEGATIVE_GATE_COMPLETE` |
| 20 | `midogpp.cvae.prior_recovery_source_inner_training_seed_stability.v1` | `artifacts/midogpp/20_cvae_preservation/prior_recovery_source_inner_training_seed_stability_v1/seeds17_42_101/` | `IMPLEMENTED AND REGISTERED`; not yet production-run; `TODO_VERIFY_ARTIFACT` |
| 20 | `midogpp.cvae.prior_recovery_outer.v1` | `artifacts/midogpp/20_cvae_preservation/prior_recovery_outer_v1/seeds17_42_101/` | blocked by the source-inner gate; no outer result |

The matched Stage-10 v2 reference uses the eligible nine centers, full
Virchow2 features, and sklearn `predict`. Its mean BACC is `0.740312` and mean
macro-F1 is `0.737205`; center `1` is worst at `0.679245` BACC and center `6`
is best at `0.792350`. Its protocol hash is `786589b799d61b14` and its bound
reference-bundle hash is `995aa193c82ee7ec`. This confirms the matched
denominator only; it remains a `real_feature_transfer_only` result.

The source-inner Stage-20 bundle contains nine valid `RecipeLock` files. Seven
locks select a conditional sampler: centers `0,6,7,8` select Task-Fisher plus
the full conditional sampler (`D`), centers `1,2` select isotropic plus full
conditional sampling (`C`), and center `3` selects isotropic plus diagonal
conditional sampling (`C`). Centers `5` and `9` retain isotropic standard-normal
sampling (`A`). The selection bundle hash is `1e929d05ff987ad9`; the protocol
hash is `dd7ca955d79fade4`.

The gate outcome is `NEGATIVE_GATE_COMPLETE` with
`factorial_triggered=false`. This is a protocol-complete negative gate, not a
numerical, leakage, identity, checkpoint, or sampler-realization failure.
Center `5` is borderline and generation-seed-sensitive: its best diagonal
sampler has mean
preservation-ratio delta `+0.109849` over `A` but wins only `5/8` strict inner
comparisons. Center `9` is less consistent: its best full sampler has mean
delta `+0.087841` but wins only `4/8`. Both miss the predeclared six-win gate.

The registered outer v1 requires all nine locks to select `C` or `D` and
requires `factorial_triggered=true`; it therefore must not be run against this
bundle. No outer preservation, routing, compatibility, or downstream-utility
claim follows from the source-inner result.

The bounded next Stage-20 check is training-seed stability with training seeds
`17,42,101` and the existing generation seeds `17,42,101`. This check is
`IMPLEMENTED AND REGISTERED, NOT YET PRODUCTION-RUN`. Its canonical experiment
ID is `midogpp.cvae.prior_recovery_source_inner_training_seed_stability.v1`,
and its output destination is
`artifacts/midogpp/20_cvae_preservation/prior_recovery_source_inner_training_seed_stability_v1/seeds17_42_101/`.
The fully crossed panel writes 27 training-seed locks and nine fold-level
consensus locks. Preparation is seed-free and shared per outer/inner fold
`(H,I)`, including one Task-Fisher state per `(H,I)` when needed. CVAE training
has distinct RNG identities for each training seed, while posterior and prior
generation noise is paired by generation seed; the recomputable identities are
persisted in `tables/rng_pairing_audit.csv`.

The predeclared rule keeps a unanimous conditional sampler
family, falls back from mixed `C/D` objectives to isotropic `C`, and uses the
conservative standard-normal `A` recipe when seeds or conditional sampler
families disagree. Any invalid child lock disables export. No stability
artifact or cross-seed result exists yet, so the catalog remains
`TODO_VERIFY_ARTIFACT`. `reports/publication_state.json` is the fail-closed
Stage-30 publication gate: only `PUBLISHED` is consumable. Stage 30 first
requires every consensus lock in the bundle to be valid and export-ready, then
loads lock `H` only for fold `H`. The scalar seed-42 source-inner result and the
blocked outer-v1 evidence boundary are unchanged; the stability bundle does not
retroactively unlock or replace the outer gate. After this one bounded check,
stop Stage-20 optimization and proceed to the planned Stage-30
provenance-clean expert bank.

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
- The immediate experimental sequence is one bounded source-inner
  training-seed stability check, followed by the new provenance-clean
  independently trained MIDOG++ source-expert bank. Routing remains premature
  until that bank and a fresh protocol-clean utility surface exist.
