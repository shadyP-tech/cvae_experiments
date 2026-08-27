# MIDOG++ Experiment File Workflow

Last updated: 2026-08-22

This is the operational path and command reference for the active MIDOG++
checkout. Result interpretation belongs in
`docs/context/current_experimental_state.md` and the experiment pages under
`docs/wiki/03-experiments/`.

## Canonical Ownership

The repository has one active Python package and one staged experiment
workspace:

```text
src/midogpp_thesis/        reusable dataset, real-feature, CVAE, and workspace code
tests/                     package and protocol tests
datasets/midogpp/raw/      workstation-local MIDOG++ source data
datasets/midogpp/configs/  active and quarantined dataset configs
datasets/midogpp/contract/ frozen dataset contract and patch payload
datasets/midogpp/derived/  derived feature caches
experiments/midogpp/       registry, catalog, protocol defaults, and staged configs
artifacts/midogpp/         canonical MIDOG++ evidence and run destinations
artifacts/cross_dataset_archive/
                           retired non-MIDOG++ and generic historical evidence
docs/                      thesis context and evidence-backed interpretation
```

The former top-level capability roots are deleted. They are not import paths,
active runners, or resolution fallbacks. An old path may remain only inside an
immutable historical manifest, relocation sidecar, or repository-migration
audit. Retired workstation source locations are absent.

## Environment And Entry Point

Install the package in the thesis environment once from the repository root:

```bash
conda run -n thesis python -m pip install -e '.[cache,dataset-full]'
```

All operational commands then use the package entry point:

```bash
conda run -n thesis python -m midogpp_thesis --help
conda run -n thesis python -m midogpp_thesis workspace validate
conda run -n thesis python -m midogpp_thesis workspace list
```

The module groups are:

- `dataset-build`, `dataset-validate`, `dataset-inspect`, and
  `dataset-physical-multiscale`;
- `real-features` for cache building and real-feature diagnostics;
- `real-feature-classifier` for the tuned and eligible matched references;
- `cvae-preservation` for preservation, source-inner recipe locking, and the
  conditional outer factorial;
- `cvae-expert-bank` for Stage-30 promotion, validation, and authorized expert
  loading;
- `cvae-generation` for Stage-40 GenerationLock materialization and validation;
- `cvae-routing` for Stage-60 policy-lock materialization and validation;
- `workspace` for registry validation, artifact resolution, preparation, and
  registered runs.

Do not restore package-specific `PYTHONPATH` launch commands.

## Evidence Stages

The registry orders evidence as follows:

| Stage | Surface | Current status |
| --- | --- | --- |
| 10 | real-feature reference | active and diagnostic entries |
| 20 | CVAE preservation | active and diagnostic entries |
| 30 | provenance-clean expert bank | active Uniform-B v2 bank; alternative consensus-recipe v1 path planned |
| 40 | prior and generation | active validated Uniform-B v2 GenerationLock |
| 50 | all-candidate utility matrix | planned for new runs; local historical diagnostic retained |
| 60 | routing and composition | active validated equal-union, metadata max-tie, source-inner utility, and utility/regret fallback locks |
| 70 | frozen-policy downstream utility | completed validated descriptive comparison on previously consumed test; fresh confirmation blocked |
| 90 | oracles, audits, and rejected lineages | diagnostic or rejected only |

Stage 30 has an active, validated routing-authorized bank. Stages 40 and 60
have validated generation, direct-control, compatibility-proxy, and comparison-
policy contracts. Stage 70 has a validated descriptive downstream comparison:
equal-union BACC is `0.774968`, metadata max-tie is `0.745099`, and the
utility/regret policy is exactly equal-union. Because the test surface was
previously consumed, the result does not promote a router or establish fresh
routing quality. A preservation result cannot be treated as an expert bank,
router input, or synthetic downstream-utility result without an explicit
registered promotion boundary.

## Dataset Contract

Active source-data and contract paths:

```text
datasets/midogpp/raw/MIDOGpp/
datasets/midogpp/configs/annotation_patch_v1.yaml
datasets/midogpp/contract/annotation_patch_v1/
```

The active geometry is `xyxy`. The rejected geometry config is retained only
at:

```text
datasets/midogpp/configs/quarantine/annotation_patch_v1_coco_xywh_stale.yaml
```

The frozen contract contains 22,569 manifest rows, a case-disjoint split, and a
stored `PASS` leakage report. The manifest hash remains
`db661ac7e3dbafde8e283528de6706ab35f2c26629b389706c4504e458cc5869`.
Its `image_path` values intentionally retain their original repository prefix
so the frozen manifest bytes and hash do not change. The audited sidecar
`datasets/midogpp/contract/annotation_patch_v1/path_relocation.json` maps only
that prefix to the canonical contract tree.

All 22,569 referenced patches are present locally and on the workstation.
Canonical validation passes with nine eligible domains. The approximately
65 GB raw source tree is intentionally workstation-only at
`datasets/midogpp/raw/MIDOGpp/`; the frozen contract and cache are sufficient
for the currently registered experiments.

Validate with:

```bash
conda run -n thesis python -m midogpp_thesis dataset-validate \
  --artifact-root datasets/midogpp/contract/annotation_patch_v1
```

Read-only contract/cache inspection uses:

```bash
conda run -n thesis python -m midogpp_thesis dataset-inspect \
  --artifact-root datasets/midogpp/contract/annotation_patch_v1 \
  --cache-report datasets/midogpp/derived/features/virchow2/annotation_patch_xyxy/seed42/reports/cache_builder_report.json
```

The active downstream axis is `center`; eligible centers are
`0,1,2,3,5,6,7,8,9`, center `4` is quarantined from deployable evaluation, and
held-out target labels are scoring-only.

## Virchow2 Feature Caches

The only active corrected cache location is:

```text
datasets/midogpp/derived/features/virchow2/annotation_patch_xyxy/seed42/
```

Its required train tensor SHA-256 is
`f6608e513fb2d06671e3ec117b093a85d58530b77b1fae44a3be1680d9feabd2`.
The cache is present locally and on the workstation. Its train, validation, and
test counts align exactly with the contract: `9,886`, `2,677`, and `10,006`.

Two other local lineages are deliberately non-active:

- `datasets/midogpp/derived/features/virchow2/historical_train_only/seed42/`
  is `DIAGNOSTIC ONLY` and has a different train tensor hash;
- `datasets/midogpp/derived/features/quarantine/coco_xywh/virchow2/` is
  `REJECTED`.

Neither may substitute for the corrected `xyxy` cache.

### Validated physical-multiscale cache lineage

V1 and v2 are immutable failed-audit lineages. The current v3 implementation
has a passing source audit and a hash-promoted workstation-only contract/cache
lineage:

```text
datasets/midogpp/configs/physical_multiscale_clipped_bbox_annotation_local_pooling_pilot_v3.yaml
datasets/midogpp/contract/physical_multiscale_clipped_bbox_annotation_local_pooling_pilot_v3/
datasets/midogpp/derived/features/virchow2/physical_multiscale_clipped_bbox_annotation_local_pooling_pilot_v3/seed42/
```

The contract and caches are reusable dataset inputs, so they remain under
`datasets/midogpp`; the Stage-10 evidence bundle remains under
`artifacts/midogpp`. Build order is `audit-v3`, `build-contract-v3`,
`build-cache-v3`, then `validate-v3` through the
`dataset-physical-multiscale` command. Raw TIFFs, tiled reading, and the exact
Virchow2 commit/checkpoint/runtime identity make these workstation-only
operations.

The 2026-07-23 `xai-master` `audit-v3` pass covers 9,648 rows and 216 TIFFs,
including 84 clipped bboxes, with no row exclusion, padding, or synthesized
pixels. The subsequent contract and atomic B/C build independently passes
`validate-v3`, including 28,944 pooling records, the canonical-A numeric
bridge, frozen task bridge, decoder/runtime identity, and content-index checks.
All required files are hash-promoted in the catalog. Do not run a later build
command against v1 or v2, and do not treat their failed audits as fallbacks.

## Registered Real-Feature Workflow

The retained tuned-classifier experiment is:

```text
midogpp.real_feature.tuned_classifier.seed42
```

It consumes the logical artifacts
`midogpp_dataset_contract_annotation_patch_v1` and
`midogpp_virchow2_xyxy_feature_cache_seed42`. It writes new output to:

```text
artifacts/midogpp/10_real_feature_reference/midogpp_real_feature_tuned_classifier/seed42/
```

Prepare and run through the registry:

```bash
conda run -n thesis python -m midogpp_thesis workspace prepare \
  midogpp.real_feature.tuned_classifier.seed42

conda run -n thesis python -m midogpp_thesis workspace run \
  midogpp.real_feature.tuned_classifier.seed42
```

The validated tuned reference is present locally and on the workstation at:

```text
artifacts/midogpp/10_real_feature_reference/real_feature_threshold_both_annotation_patch_xyxy_virchow2_seed42/
```

Its claim scope is `real_feature_transfer_only`. It is a frozen comparator for
CVAE preservation, not routing or generated-embedding evidence.

### Eligible-only matched reference v2

The prior-recovery surface requires a separate full-Virchow2, eligible-only,
predict-policy reference:

```text
midogpp.real_feature.eligible_tuned_predict_reference.v2
experiments/midogpp/stages/10_real_feature_reference/configs/eligible_tuned_real_reference_v2.yaml
artifacts/midogpp/10_real_feature_reference/eligible_tuned_real_reference_v2/seed42/
```

Run it with:

```bash
conda run -n thesis python -m midogpp_thesis workspace prepare \
  midogpp.real_feature.eligible_tuned_predict_reference.v2
conda run -n thesis python -m midogpp_thesis workspace run \
  midogpp.real_feature.eligible_tuned_predict_reference.v2
```

The completed workstation bundle contains `config.resolved.yaml`,
`provenance/input_artifacts.json`, protocol and leakage/provenance JSON, the
source-inner tuning table, held-out result table, and prediction table. It
validates `PASS` on `xai-master`, with protocol hash `786589b799d61b14` and
reference-bundle hash `995aa193c82ee7ec`. The matched reference reports mean
BACC `0.740312`, mean macro-F1 `0.737205`, worst-center BACC `0.679245` at
center `1`, and best-center BACC `0.792350` at center `6`. The bundle has not
yet been synced into the local checkout, and its catalog destination still
retains the lifecycle label `TODO_VERIFY_ARTIFACT`. The earlier failed root
remains non-evidence; `PASS` refers only to the repaired completed bundle.
Validation binds the three table contents through `reference_bundle_hash`, then
binds that content identity to the protocol, resolved config, registered
dataset/cache inputs, and their SHA-256 values. The Stage-20 outer run imports
that bound reference identity; it does not accept an unverified table copy.

### Fixed-C risk-weighting diagnostic v1

The registered non-adoptive weighting diagnostic is:

```text
midogpp.real_feature.fixed_c_risk_diagnostic.v1
experiments/midogpp/stages/10_real_feature_reference/configs/fixed_c_risk_diagnostic_v1.yaml
artifacts/midogpp/10_real_feature_reference/fixed_c_risk_diagnostic_v1/seed42/
```

It consumes only the frozen MIDOG++ contract and corrected Virchow2 `xyxy`
feature cache. Across the nine eligible held-out centers it runs the four
predeclared `pooled`, `global_class`, `domain`, and `domain_class` source-fit
weighting arms under the same fixed logistic-regression configuration. Raw
weights use `1`, `N/(2*n_y)`, `N/(D*n_d)`, and `N/(2*D*n_dy)`,
respectively, and each arm is normalized to sum to its source-fit row count.
This is a fixed `9 x 4 = 36` diagnostic; it performs no model, arm, recipe, or
policy selection.

Prepare and run it through the workspace:

```bash
conda run -n thesis python -m midogpp_thesis workspace prepare \
  midogpp.real_feature.fixed_c_risk_diagnostic.v1
conda run -n thesis python -m midogpp_thesis workspace run \
  midogpp.real_feature.fixed_c_risk_diagnostic.v1
```

Its maximum claim scope is `real_feature_transfer_only`, with held-out labels
used for scoring only. Catalog policy fixes
`may_feed_recipe_selection=false` and
`may_feed_deployable_selection=false`, and forbids consumption by every
current workspace purpose except Stage-90
`oracle_and_diagnostic_evidence`. It is not a Stage-20 input, a routing
signal, a deployable weighting rule, or evidence for CVAE preservation,
generation, prior quality, or synthetic downstream utility. The catalog
defines the required 12-file destination bundle, but this workflow page
records no result or metric for an unvalidated run.

### Conditional logit alignment diagnostic v1

The implemented source-inner regularization diagnostic is:

```text
midogpp.real_feature.conditional_logit_alignment.v1
experiments/midogpp/stages/10_real_feature_reference/configs/conditional_logit_alignment_v1.yaml
artifacts/midogpp/10_real_feature_reference/conditional_logit_alignment_v1/seed42/
```

It consumes only the frozen MIDOG++ contract and corrected Virchow2 `xyxy`
train cache. For each outer center it selects one gamma from
`0, 0.0001, 0.001, 0.01, 0.1, 1, 10` through eight source-inner center-LODO
folds. The outer center is absent from all preprocessing and selection; each
inner pseudo-target is absent from its fold's scaler, conditional-centroid
factor, and classifier fit. Only the selected gamma and matched `gamma=0`
baseline are scored on the outer center.

The registry status is `diagnostic`. Prepare or reproduce the canonical run
through the workspace command surface:

```bash
conda run -n thesis python -m midogpp_thesis workspace prepare \
  midogpp.real_feature.conditional_logit_alignment.v1
conda run -n thesis python -m midogpp_thesis workspace run \
  midogpp.real_feature.conditional_logit_alignment.v1
```

The catalog fixes `may_feed_recipe_selection=false` and
`may_feed_deployable_selection=false` and permits only Stage-90 diagnostic
consumption. This experiment cannot replace the matched Stage-10 denominator
or feed CVAE preservation, expert-bank, generation, routing, or downstream
selection.

The canonical workstation run is complete and protocol-clean. All 16 required
files are present on `xai-master`, the leakage/provenance report is `PASS`, and
the runtime, protocol, and content identities agree. The selected method has
mean BACC `0.7434042753` versus `0.7434898099` for matched `gamma=0`, a delta of
`-0.0000855346`; macro-F1 changes by `+0.0000496479`, and worst-center BACC is
unchanged at `0.6913746631`. Source-inner selection chooses positive gamma in
all nine folds (`gamma=10` seven times, `gamma=1` twice), but held-out BACC has
only two wins, four ties, and three losses. The result is therefore a
protocol-valid `NEGATIVE_RESULT` and remains `DIAGNOSTIC_ONLY`.

The bundle is currently workstation-only. Its catalog lifecycle label remains
`TODO_VERIFY_ARTIFACT` pending local sync and metadata promotion; do not edit or
reinterpret that catalog field by inference. Do not expand the gamma sweep or
adopt CLA. Any later study must be predeclared and mechanism-diagnostic only.
Post-hoc probability metrics cannot overturn the predeclared BACC decision.
Detailed artifact paths, causal audit interpretation, limitations, and the
stop recommendation are recorded in
`docs/wiki/03-experiments/midogpp-conditional-logit-alignment-diagnostic.md`.

### Physical multiscale clipped-bbox annotation-local pilot v3

The current registered pilot is:

```text
midogpp.real_feature.physical_multiscale_clipped_bbox_annotation_local_pooling_pilot.v3
experiments/midogpp/stages/10_real_feature_reference/configs/physical_multiscale_clipped_bbox_annotation_local_pooling_pilot_v3.yaml
artifacts/midogpp/10_real_feature_reference/physical_multiscale_clipped_bbox_annotation_local_pooling_pilot_v3/seed42/
```

Its status is `diagnostic`. The v3 contract and atomic B/C bundle exist on
`xai-master`, independently pass `validate-v3`, and have complete cataloged
expected hashes. `workspace run` is therefore authorized, but no Stage-10
output exists yet. The pilot freezes the 30-candidate hash
`2f651b2f8bd53c1a`, produces exactly 2,160 source-inner cells, writes all nine
per-center locks before outer evaluation, and requires exact A replay. Posthoc
target-scored candidates are non-adoptive and excluded from all lock hashes.
The catalog blocks the output from Stage 20 through Stage 70.

V1 and v2 stay registered only as audit-blocked historical plans; neither may
be activated, resumed, or repointed to the v3 artifacts.

## Registered CVAE Preservation Workflow

The validated tuned-classifier preservation experiment is:

```text
midogpp.cvae.tuned_classifier_preservation.v1
```

Canonical config:

```text
experiments/midogpp/stages/20_cvae_preservation/configs/tuned_classifier_preservation_v1.yaml
```

Logical inputs:

```text
midogpp_dataset_contract_annotation_patch_v1
midogpp_virchow2_xyxy_feature_cache_seed42
midogpp_real_feature_threshold_both_xyxy_seed42
```

Canonical output:

```text
artifacts/midogpp/20_cvae_preservation/virchow2_cvae_midogpp_tuned_classifier_preservation_v1/seed42/
```

All declared inputs are migrated and hash-verified. Prepare and run with:

```bash
conda run -n thesis python -m midogpp_thesis workspace prepare \
  midogpp.cvae.tuned_classifier_preservation.v1

conda run -n thesis python -m midogpp_thesis workspace run \
  midogpp.cvae.tuned_classifier_preservation.v1
```

The validated workstation result supports only
`claim_scope=cvae_preservation_only`. Decode and posterior preservation do not
establish prior quality, routing, expert selection, NELBO compatibility,
controllable generation, GMM composition, or held-out synthetic utility.

## Registered Prior-Recovery Workflow

The implementation separates source-inner recipe selection from held-out outer
evaluation.

Source-inner experiment and output:

```text
midogpp.cvae.prior_recovery_source_inner.v1
experiments/midogpp/stages/20_cvae_preservation/configs/prior_recovery_source_inner_v1.yaml
artifacts/midogpp/20_cvae_preservation/prior_recovery_source_inner_v1/seed42/
```

Run it with:

```bash
conda run -n thesis python -m midogpp_thesis workspace prepare \
  midogpp.cvae.prior_recovery_source_inner.v1
conda run -n thesis python -m midogpp_thesis workspace run \
  midogpp.cvae.prior_recovery_source_inner.v1
```

This run uses only the dataset contract and corrected cache. For every outer
center it removes that center completely, builds inner pseudo-target folds from
the remaining eligible centers, and records the distinct two-spec Stage-20
classifier grid (`C=0.01`, class weight none/balanced), nested classifier
tuning, nested real denominators, source-inner preservation metrics, sampler
realizations, identity audits, checkpoint, Task-Fisher, and feature-frame
indexes, hashed `RecipeLock` files, leakage/gate/run-state/runtime-summary
reports, `tables/runtime_timings.csv`, and
`tables/checkpoint_reuse_audit.csv`. `C` and PCA128 are fixed design choices,
not sweeps; only class weight is selected inside each deeper fold. The audit
requires `A/C` checkpoint reuse and,
when Task-Fisher is triggered, `B/D` reuse plus paired `A/B` initialization and
stochastic streams. Source-inner metric validation also requires equal
per-class generation budgets across compared arms.

The runner writes checkpoint and PCA cache entries incrementally. Reissuing the
same command resumes exact `prior_recovery_v2_resume` keys and overwrites only
diagnostic timing/cache-status rows. Protocol, row, recipe, classifier, cache,
code, or library drift produces a miss; a matching corrupt entry fails closed.
Pre-v2 partial checkpoints are not eligible for resume.

The completed seed-42 source-inner bundle on `xai-master` is `COMPLETE` and its
full validator passes. All nine recipe locks are valid, seven select a
conditional sampler, and centers `5` and `9` retain the standard-normal `A`
fallback. The gate status is `NEGATIVE_GATE_COMPLETE` with
`factorial_triggered=false`; its selection-bundle hash is
`1e929d05ff987ad9` and protocol hash is `dd7ca955d79fade4`. This is a valid
recipe-lock result and a complete negative gate, not an execution failure.

The outer experiment and output are:

```text
midogpp.cvae.prior_recovery_outer.v1
experiments/midogpp/stages/20_cvae_preservation/configs/prior_recovery_outer_v1.yaml
artifacts/midogpp/20_cvae_preservation/prior_recovery_outer_v1/seeds17_42_101/
```

It depends on the matched Stage-10 v2 bundle and the source-inner lock bundle.
The current source-inner bundle has `factorial_triggered=false` and two `A`
locks, so the registered outer v1 is blocked and must not be run. If a future
predeclared source-inner experiment validates with all conditional locks and
`factorial_triggered=true`, the outer command is:

```bash
conda run -n thesis python -m midogpp_thesis workspace prepare \
  midogpp.cvae.prior_recovery_outer.v1
conda run -n thesis python -m midogpp_thesis workspace run \
  midogpp.cvae.prior_recovery_outer.v1
```

The runner fails closed if the matched reference is not schema v2/predict
policy, the lock bundle is incomplete or tampered, any lock is not conditional,
identity overlap exists, a requested conditional sampler falls back, or full
factorial coverage is missing. Its bundle contains preservation metrics,
sampler realizations, paired deltas, equal-center aggregation, checkpoint-reuse
and identity audits, coverage/protocol/provenance manifests, decision/leakage
reports, content-addressed checkpoints, and Task-Fisher states.
`tables/sampler_realizations.csv` is required and validated against the outer
metric rows and recorded sampler states. The outer bundle also requires the
feature-frame index, runtime timings/summary, and run-state report.

Both source-inner and outer metrics use the same frozen classifier
specification/predict policy within each fold and the chance-corrected ratio
`(BACC_generated - 0.5) / (BACC_real - 0.5)`. Outer validation requires equal
per-class generation budgets across A/B/C/D and binds every metric row to the
validated source `RecipeLock`, its shared selection-bundle identity, and the
source protocol/selection-evidence file hashes.

A complete valid factorial keeps `claim_scope=cvae_preservation_only` whether
the decision is `POSITIVE_PRESERVATION` or `NEGATIVE_PRESERVATION`.
`claim_scope=diagnostic_only` and status
`INCOMPLETE_OR_INVALID_DIAGNOSTIC` are reserved for incomplete or invalid
executions.

Only validated fold-level consensus locks from the bounded training-seed
stability bundle may feed the registered planned Stage-30 expert recipe. The
Stage-30 loader requires `reports/publication_state.json` to be `PUBLISHED`,
validates the complete bundle, requires all consensus locks bundle-wide to be
valid and export-ready, and only then consumes lock `H` for fold `H`. A
`PENDING` or `FAILED` publication state blocks consumption. Scalar seed-42
locks remain source-inner evidence, while outer target labels and metrics are
evaluation-only and may never feed model or routing selection. The outer
experiment continues to consume the scalar source-inner bundle; the stability
bundle neither changes nor unlocks that evidence boundary. Stage 40 remains
the later post-expert-bank generation-validation stage.

Current status: the scalar source-inner seed-42 result and bounded
training-seed stability result are complete and validated on `xai-master`.
The stability publication is an eligible Stage-30 recipe input. The outer
output is absent because its separate fail-closed prerequisite is not
satisfied. There is no outer-preservation result.

### Completed bounded training-seed stability check

The bounded Stage-20 panel is limited to training seeds `17,42,101` with the
existing generation seeds `17,42,101`. It is registered as:

```text
midogpp.cvae.prior_recovery_source_inner_training_seed_stability.v1
experiments/midogpp/stages/20_cvae_preservation/configs/prior_recovery_source_inner_training_seed_stability_v1.yaml
artifacts/midogpp/20_cvae_preservation/prior_recovery_source_inner_training_seed_stability_v1/seeds17_42_101/
```

Reproduce or resume exactly the registered panel with:

```bash
conda run -n thesis python -m midogpp_thesis workspace prepare \
  midogpp.cvae.prior_recovery_source_inner_training_seed_stability.v1
conda run -n thesis python -m midogpp_thesis workspace run \
  midogpp.cvae.prior_recovery_source_inner_training_seed_stability.v1
```

The runner performs seed-free deterministic preparation once per outer/inner
fold `(H,I)`: nested classifier selection, real reference, identity audit, and
PCA frame are shared across training seeds. It also fits at most one shared
Task-Fisher state per `(H,I)`. CVAE training retains distinct initialization
and stochastic-stream identities for training seeds `17,42,101`, while
posterior and prior-generation noise is paired by generation seed across those
training arms. These recomputable identities are persisted in
`tables/rng_pairing_audit.csv`. The runner preserves the source class-count
budget without rebalancing and writes one wrapped lock per
`(training seed, outer center)` plus one consensus lock per outer center.
The frozen consensus rule exports unanimous `A`; exports `D` only when all
seeds select `D` with one conditional sampler family; falls back to `C` for a
shared conditional family with any `C`; and conservatively exports `A` for
cross-seed arm or conditional-family disagreement. Any invalid child disables
export. Structural bundle validity and Stage-30 recipe readiness are reported
separately.

Bundle writing begins with `reports/publication_state.json` set to `PENDING`.
Only after the complete bundle validates is it rewritten as `PUBLISHED`; a
validation error writes `FAILED`. The normal validator and the Stage-30 loader
accept only `PUBLISHED`. Stage 30 then requires all nine consensus locks to be
valid and `recipe_export_ready=true` before returning fold `H`'s lock to fold
`H`.

The canonical workstation bundle is now `COMPLETE` and `PUBLISHED`. Full
bundle, leakage, identity-overlap, and RNG validation passes; all `27/27`
training-seed child locks validate, all `9/9` consensus locks are export-ready,
and `stage30_recipe_ready=true`. The Stage-30 loader accepts the bundle. Its
protocol hash is `bbde3e5c5a1e3374`, and its selection-bundle hash is
`79cb9b614779c23b`. The artifact has not yet been synced locally, and the
catalog entry still has the stale lifecycle label `TODO_VERIFY_ARTIFACT`
pending catalog promotion.

The published fold recipes are:

| Outer center | Consensus recipe | Stability outcome |
| --- | --- | --- |
| `0,1,2,3,5,9` | `A`: isotropic objective, standard-normal sampler | cross-seed arm or conditional-family disagreement |
| `6,7` | `D`: Task-Fisher objective, full conditional sampler | exactly unanimous |
| `8` | `C`: isotropic objective, full conditional sampler | sampler family stable, objective unstable |

Thus centers `0,1,2,3,5,8,9` are unstable and only centers `6,7` are exactly
unanimous. This is a protocol-valid `NEGATIVE_RESULT` for broad training-seed
stability of source-inner recipe selection and an operational `PASS` for the
predeclared conservative publication gate. It remains
`claim_scope=cvae_recipe_lock_only`: no outer-preservation, routing,
generation-quality, or downstream-utility claim follows.

The consensus-lock publication remains the eligible input to the separate
planned `midogpp.expert_bank.provenance_clean.v1` path; that runner is still a
placeholder. It is not the input to the active Uniform-B v2 bank. The latter
uses its own reviewed aggregate-prior union v2 source evidence and completed
Stage-30 promotion. Neither path changes or unlocks outer v1, which still
consumes the scalar seed-42 lock bundle.

### Non-adoptive v2 source-inner mechanism studies

Two additional Stage-20 studies are registered independently of the current
recipe locks:

| Experiment ID | Config | Canonical output |
| --- | --- | --- |
| `midogpp.cvae.learned_conditional_prior_source_inner.v2` | `experiments/midogpp/stages/20_cvae_preservation/configs/learned_conditional_prior_source_inner_v2.yaml` | `artifacts/midogpp/20_cvae_preservation/learned_conditional_prior_source_inner_v2/seeds17_42_101/` |
| `midogpp.cvae.task_fisher_shrinkage_source_inner.v2` | `experiments/midogpp/stages/20_cvae_preservation/configs/task_fisher_shrinkage_source_inner_v2.yaml` | `artifacts/midogpp/20_cvae_preservation/task_fisher_shrinkage_source_inner_v2/seeds17_42_101/` |

Prepare and run them separately:

```bash
conda run -n thesis python -m midogpp_thesis workspace prepare \
  midogpp.cvae.learned_conditional_prior_source_inner.v2
conda run -n thesis python -m midogpp_thesis workspace run \
  midogpp.cvae.learned_conditional_prior_source_inner.v2

conda run -n thesis python -m midogpp_thesis workspace prepare \
  midogpp.cvae.task_fisher_shrinkage_source_inner.v2
conda run -n thesis python -m midogpp_thesis workspace run \
  midogpp.cvae.task_fisher_shrinkage_source_inner.v2
```

The underlying preservation CLI surfaces are
`source-inner-learned-conditional-prior-study` and
`source-inner-task-fisher-shrinkage-study`. The registered workspace commands
should be preferred because they resolve and record the exact dataset/cache
inputs and canonical output root.

Both runners use all nine outer centers, all eight remaining inner
pseudo-targets, training seeds `17,42,101`, and generation seeds
`17,42,101`. They persist exact-key checkpoints and final learned-prior or raw
Fisher state, pair initialization/training/evaluation RNG where declared, and
validate their complete bundle before marking `reports/run_state.json` as
`COMPLETE`. A matching interrupted run resumes from exact sidecars; changed
protocol, data, frame, objective, prior family, or code identity is a cache
miss.

Status: `IMPLEMENTED AND REGISTERED, NOT YET PRODUCTION-RUN`. Their output
scope is `cvae_source_inner_study_only`; the catalog forbids reuse as expert
bank, generation, routing, NELBO-compatibility, or downstream-utility evidence.
They have no publication state, no `RecipeLock`, and no Stage-30 consumption
edge. The current consensus locks and outer-v1 gate remain unchanged.

### Independent-source aggregate-prior v3 study

The bounded prior-mismatch exception is registered as:

```text
midogpp.cvae.aggregate_posterior_mixture_geco_source_inner.v3
```

Run only through the workspace so the resolved source contract and corrected
Virchow2 cache are snapshotted:

```bash
conda run -n thesis python -m midogpp_thesis workspace prepare \
  midogpp.cvae.aggregate_posterior_mixture_geco_source_inner.v3
conda run -n thesis python -m midogpp_thesis workspace run \
  midogpp.cvae.aggregate_posterior_mixture_geco_source_inner.v3
```

Each reusable training key is neutral to `H` and `I` and binds one source
center, source-only row/case hashes, source-local PCA frame, arm, and training
seed. Evaluation keys separately enforce `H != I` and `E not in {H,I}`.
Generated PCA128 samples are inverse-transformed into the common 2,560-d
Virchow2 frame before classifier fitting. Each prior row is paired with a
deterministic balanced source-posterior reconstruction reference. The
predeclared gate requires `KG` both to preserve that posterior path and to
reduce the posterior-minus-prior BACC gap relative to `SF`; posterior rows are
diagnostic ceilings, not an alternative generator or a consumable result.

The output is
`artifacts/midogpp/20_cvae_preservation/aggregate_posterior_mixture_geco_source_inner_v3/seeds17_42_101/`.
It is a non-consumable `cvae_source_inner_study_only` bundle. No outcome may
alter Stage 30 without its own implemented, validated, and explicitly
registered promotion. The completed Uniform-B v2 promotion described below
does not authorize this v3 artifact.

## Uniform-B V2 Routing-Authorized Expert Bank

The completed promotion is registered as:

```text
midogpp.expert_bank.uniform_b_v2_routing_promotion.v1
```

Run or revalidate it through the workstation venv and workspace boundary:

```bash
/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run \
  midogpp.expert_bank.uniform_b_v2_routing_promotion.v1
```

Canonical output:

```text
artifacts/midogpp/30_expert_bank/uniform_b_v2_routing_authorized_expert_bank_v1/
```

Current result: `COMPLETE`, validator `PASS`, decision
`PROMOTED_AS_ROUTING_AUTHORIZED_EXPERT_BANK`, and publication state
`ROUTING_AUTHORIZED`. The bank contains all nine source centers and training
seeds `17,42,101` (27 checkpoints) with no expert or seed selection. It is the
only current Stage-30 artifact allowed to feed deployable-selection work.

The loader must reject the Stage-20 source artifact as a direct routing input.
Source-inner labels were consumed for whole-bank adoption and may not be used
again to select an expert, seed, or router. Consumers must bind bank lock
`9972a41dcd4814cd` and control lock `cddbcc3b3343fe38`.

One explicitly scoped exception is registered for SCEPTRE v1: a separate
Stage-90 consumer-fenced alias and amendment may reuse the immutable
source-inner candidate-utility bytes plus the exact label-free pre-label
prediction packet (`candidate_predictions.npz`, its index, classifier-fit
inventory, and evaluation-row inventory) for adaptive, descriptive architecture
development only. For every outer target `H`, all rows with query `H` or
candidate `H` must be removed before any transform, normalization, fit, or
tuning operation; nested LODO repeats the same query/candidate exclusion for
its training side before recomputing training transforms, while the held
`q=K` validation rows are transformed separately after removing candidate
`K` and never influence training transforms. The nine
training/generation seed cells are nuisance replications, never independent
samples or selectable candidates. The complete nine-target router, proposal
policy, thresholds, case partition, and exact-B controls are frozen together.
Labeled support and calibration may only preserve the label-free G proposal or
abstain to exact B; no phase may switch to another expert or revive a rejected
proposal. The phase manager owns the exact calibration-uncertainty decision
and rejects any calibration record whose uncertainty hash, candidate, route,
or acceptance flag differs. After all 45 decisions are sealed, it emits one
canonical replayable route table and binds that table's hash into the terminal
capability. This exception does not mutate or reinterpret the original
policy-consumption lock, does not authorize consumed-test execution, and can
only yield `POST_HOC_CONSUMED_TEST_SENSITIVITY` evidence. Phase capabilities in
the development package record lineage only; the test-label reader, cache,
manifest, ledger, and real subprocess attestation remain future-only. The
result cannot support a new-center, confirmatory, promotion, deployment, or
downstream-consumer claim.

The canonical routing control is `uniform_b_v2_equal_union_ps`: exclude target
`H`, use all eight remaining sources, allocate 128 generated rows per source
and class for a 1,024-per-class total, cross training/generation seeds
`17,42,101`, and report the predeclared mean without seed selection. Future
router and control arms must use paired RNG, shuffles, classifier budgets,
candidate eligibility, and evaluation rows. The control needs fresh scoring;
the promotion's `0.770112` source-inner PS BACC came from seven-source tasks.

The output content is hash-locked, but its provenance records
`repository_dirty=true` at revision
`40221038ca714bf33fd21582857d21fa1db4e6f3`. Preserve the exact working-tree
diff or reproduce and validate the artifact from a clean commit before final
archival.

## Uniform-B V2 GenerationLock And Stage-60 Policy Locks

The completed Stage-40 and Stage-60 experiments are:

```text
midogpp.prior_and_generation.uniform_b_v2_generation_lock.v1
midogpp.routing_and_composition.uniform_b_v2_equal_union_policy_lock.v1
midogpp.routing_compatibility.uniform_b_v2_metadata_exact_match_lock.v1
midogpp.routing_and_composition.uniform_b_v2_metadata_tie_union_policy_lock.v1
midogpp.routing_and_composition.uniform_b_v2_source_inner_candidate_utility.v1
midogpp.routing_and_composition.uniform_b_v2_utility_regret_policy_lock.v1
```

Run them only in dependency order through `workspace run`. Their canonical
outputs are:

```text
artifacts/midogpp/40_prior_and_generation/uniform_b_v2_generation_lock/v1/
artifacts/midogpp/60_routing_and_composition/uniform_b_v2_equal_union_policy_lock/v1/
artifacts/midogpp/60_routing_and_composition/uniform_b_v2_metadata_exact_match_compatibility/v1/
artifacts/midogpp/60_routing_and_composition/uniform_b_v2_metadata_tie_union_policy_lock/v1/
artifacts/midogpp/60_routing_and_composition/uniform_b_v2_source_inner_candidate_utility/v1/
artifacts/midogpp/60_routing_and_composition/uniform_b_v2_utility_regret_policy_lock/v1/
```

Both workstation artifacts are `COMPLETE`; production validation and separate
validator reruns report `PASS`. Stage 40 binds GenerationLock
`34e551425710362e`. Stage 60 binds policy lock `4b9ea514308b084f`, policy-plan
hash `9ec24122d7d0cdf1`, and assignment-table hash `c85415c1b953c04e` over 81
replicates and 648 source assignments. The Stage-60 decision is
`FROZEN_AS_CANONICAL_EQUAL_UNION_ROUTING_CONTROL`, with publication state
`POLICY_FROZEN_FOR_STAGE70_EVALUATION`.

The Stage-60 bundle consumes only the validated bank and GenerationLock. It
uses no target samples, labels, support rows, compatibility score, ranking,
learned weighting, or individual seed selection. Target identity is structural
only: held-out-fold identity, target-expert exclusion, and the predeclared
label-blind within-class shuffle namespace. The bundle computes no BACC,
macro-F1, routing quality, or downstream utility.

The metadata compatibility command must run before the metadata policy command.
It consumes only the hash-pinned `domain_mapping.json`, sanitizes exactly the
three routing-time axes, and freezes all 72 ordered target-excluded exact-match
proxy scores. It performs no selection. The consuming policy retains all
maximum-score ties, uses 1,024/512/256 rows per selected source and class for a
fixed 1,024-per-class total, and reuses the equal-union Stage-40 streams and
shuffle seeds. It freezes nine selections, 81 replicates, and 153 assignments.

Compatibility lock `4b46b3d157b07781` and metadata policy lock
`27f16953b32c46cd` are independently validated. Every file in both 12-file
artifacts is SHA-256-pinned in the catalog. Neither lock reports routing quality
or downstream utility.

The substantive source-inner comparison is also complete. First, the direct
label-blind validation-cache surface builds and validates:

```text
datasets/midogpp/derived/features/virchow2/uniform_b_v2_routing_validation_cache_v1/seed42/
experiments/midogpp/stages/60_routing_and_composition/configs/uniform_b_v2_routing_validation_cache_v1.yaml
```

It contains 2,615 rows from 44 cases, 3,840 dimensions, and all nine eligible
centers; it persists no labels. Its content hash is `e1d281d44e47c7b2` and
build-protocol hash is `f57aad1bf7f7efed`. The exact dependency sequence from
the repository root on the workstation is:

```bash
/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis cvae-routing \
  uniform-b-v2-routing-validation-cache \
  --config experiments/midogpp/stages/60_routing_and_composition/configs/uniform_b_v2_routing_validation_cache_v1.yaml

/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run \
  midogpp.routing_and_composition.uniform_b_v2_source_inner_candidate_utility.v1

/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run \
  midogpp.routing_and_composition.uniform_b_v2_utility_regret_policy_lock.v1
```

Append `--validate-only` to the direct cache command for cache-only validation.
The registered utility and policy configs are
`experiments/midogpp/stages/60_routing_and_composition/configs/uniform_b_v2_source_inner_candidate_utility_v1.yaml`
and
`experiments/midogpp/stages/60_routing_and_composition/configs/uniform_b_v2_utility_regret_policy_lock_v1.yaml`.
They consume the promoted bank, GenerationLock, and, for the policy, the exact
equal-union control listed above.

The utility artifact is non-selecting source-inner policy-training evidence:
81 source-only synthetic classifier fits predict all rows before labels are
opened, then produce 648 `q != e` utility rows and 3,168 case-confusion rows
under utility lock `a787b24b8e62e203`. Its `selection_source` is source-inner
validation case-confusion utility, its `prior_method` is promoted aggregate
prior `PS`, and its `claim_role` is never target evidence. A per-query best
source is a non-deployable source-inner oracle reference used only to compute
regret, never a deployed choice or target result.

The policy removes both `q = H` and `e = H`, forms 4,536 regret cells and 72
candidate summaries, and freezes nine selections. No fold meets the unique
winner gate: best-source win probabilities range `0.392`--`0.786` (below
`0.80`) and all margin lower bounds are negative. Thus every `H` reuses the
exact frozen equal-union fallback without re-estimation. The decision is
`FROZEN_AS_SOURCE_INNER_UTILITY_REGRET_POLICY_WITH_EXACT_EQUAL_UNION_FALLBACK`,
publication is `POLICY_FROZEN_FOR_MATCHED_STAGE70_EVALUATION`, policy lock is
`d504ea0a07302acd`, plan hash is `cefe176313b1ea23`, and assignment hash is
`bd004f2bbb49228b`.

The matched Stage-70 descriptive comparison is complete at
`artifacts/midogpp/70_frozen_policy_downstream/uniform_b_v2_descriptive_frozen_policy_comparison/v1/`.
It preserves identical candidates, GenerationLock/classifier settings, seed
replicates, and evaluation rows, and exposes labels only for metrics after the
243 prediction cells are sealed. The utility/regret arm is exactly equivalent
to equal-union; metadata max-tie is worse by `-0.029868` mean BACC with
descriptive interval `[-0.050406, -0.008705]`.

The comparison is explicitly on a previously consumed test surface. It may be
used as descriptive downstream evidence only. A fresh routing-quality claim
still requires a separately authorized, genuinely unconsumed whole-case/
patient/slide-disjoint or external/new-center evaluation surface. Stage-90
diagnostics may not satisfy or feed that requirement.

## Registered Uniform-B Low-Noise Diagnostic

The bounded Variant-B training-stability audit is a two-step Stage-90 workflow:

```text
midogpp.oracle.uniform_b_paired_reparameterization_snapshot.v1
midogpp.oracle.uniform_b_paired_reparameterization_audit.v1
```

The snapshot consumes only the canonical MIDOG++ contract and the
workstation-only canonical-B cache. Historical pilot-v2 paths are inert
provenance strings and are never resolved as workspace inputs. The audit then
consumes only that hash-promoted snapshot and executes 36 declared cells: 12
legacy replay-only cells and 12 controlled pairs comparing fold-fixed
one-epsilon against fold-fixed antithetic reconstruction.

After syncing the reviewed implementation to `xai-master`, run from the remote
repository root in this order:

```bash
/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace validate
/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run \
  midogpp.oracle.uniform_b_paired_reparameterization_snapshot.v1
/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run \
  midogpp.oracle.uniform_b_paired_reparameterization_audit.v1
```

`workspace run` performs preparation before execution. Do not add `--force` on
the first run; a changed existing run snapshot must be investigated before any
forced replacement. The two canonical output roots are:

```text
artifacts/midogpp/90_oracles_and_diagnostics/inputs/uniform_b_paired_reparameterization_snapshot_v1/
artifacts/midogpp/90_oracles_and_diagnostics/uniform_b_paired_reparameterization_audit/v1/
```

Current status: `IMPLEMENTED AND REGISTERED; NOT RUN`. Both outputs are absent
locally and on `xai-master`. The workstation contract/cache hashes match the
catalog and both RTX A5000 devices are visible, but the new implementation is
not yet present in the remote checkout. Code readiness is not artifact
evidence. A completed audit remains `AUDIT_ONLY`, cannot export a recipe or
checkpoint for reuse, and cannot feed Stage 20 through Stage 70.

## Fail-Closed Preparation

`workspace prepare` resolves every `artifact://` and `output://` reference. It
rejects missing inputs, expected-hash mismatches, rejected evidence,
claim-incompatible reuse, undeclared inputs, path traversal, and outputs outside
`artifacts/midogpp/`.

After successful preparation it writes:

```text
config.resolved.yaml
provenance/input_artifacts.json
```

The provenance manifest records logical artifact IDs, canonical resolved paths,
claim scopes, evidence labels, semantic identities, computed SHA-256 values,
expected-hash verification, and git state. `--allow-missing-inputs` is for
read-only resolution diagnostics; it does not make a missing artifact eligible
for a real run.

Useful inspection commands:

```bash
conda run -n thesis python -m midogpp_thesis workspace show \
  midogpp.cvae.tuned_classifier_preservation.v1

conda run -n thesis python -m midogpp_thesis workspace resolve \
  midogpp_virchow2_xyxy_feature_cache_seed42
```

## Completed Workstation Migration

Migration evidence is present locally and on the workstation at:

```text
artifacts/midogpp/90_oracles_and_diagnostics/repository_migration/2026-07-12_xai_master/
```

It contains matching pre/post SHA-256 manifests for the complete contract,
corrected cache, tuned real-feature reference, and tuned preservation bundle.
Raw-data verification uses stable relative-path/size metadata and matching
hashes for six critical metadata/annotation files. The raw tree now lives only
at the canonical workstation path `datasets/midogpp/raw/MIDOGpp/` and is not
synced to the Mac. All retired workstation source artifact/data paths are
absent; their names survive only in immutable audit records and embedded
historical provenance.

## Historical And Rejected Evidence

Local MIDOG++ historical bundles are organized by meaning:

- stage 10 contains retained real-feature references and diagnostics;
- stage 50 contains a historical post-hoc all-candidate diagnostic only;
- stage 90 contains cache audits and rejected routing, expert-bank, and prior
  lineages that cannot feed stages 30 through 70.

BreakHis, Camelyon17, generic routing outputs, retired logs, and documentation
snapshots live under `artifacts/cross_dataset_archive/`. They are outside the
active MIDOG++ registry and cannot serve as MIDOG++ baselines or inputs.

Paths embedded inside files ending in `.historical.md`, provenance snapshots,
or immutable manifests describe the original run. They are not current run
instructions.

## Verification

Run the canonical workspace and test checks from the repository root:

```bash
conda run -n thesis python -m midogpp_thesis workspace validate
conda run -n thesis python -m pytest -q
```

Before any new experiment:

1. Choose the evidence stage and claim scope first.
2. Confirm the complete pixel payload if the run reads patches.
3. Resolve every logical input through the catalog.
4. Confirm the corrected `xyxy` cache hash and reject historical substitutes.
5. Exclude center `4` and the held-out target expert where applicable.
6. Keep target evaluation labels scoring-only.
7. Freeze all selection, generation, classifier, threshold, budget, and seed
   decisions before target scoring.
8. Preserve `config.resolved.yaml`, input hashes, protocol manifest, leakage
   report, and identity audits in the run bundle.
9. Update result documentation only after validating the produced artifact.
