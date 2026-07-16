# MIDOG++ Experiment File Workflow

Last updated: 2026-07-16

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
conda run -n thesis python -m pip install -e .
```

All operational commands then use the package entry point:

```bash
conda run -n thesis python -m midogpp_thesis --help
conda run -n thesis python -m midogpp_thesis workspace validate
conda run -n thesis python -m midogpp_thesis workspace list
```

The module groups are:

- `dataset-build`, `dataset-validate`, and `dataset-inspect`;
- `real-features` for cache building and real-feature diagnostics;
- `real-feature-classifier` for the tuned and eligible matched references;
- `cvae-preservation` for preservation, source-inner recipe locking, and the
  conditional outer factorial;
- `workspace` for registry validation, artifact resolution, preparation, and
  registered runs.

Do not restore package-specific `PYTHONPATH` launch commands.

## Evidence Stages

The registry orders evidence as follows:

| Stage | Surface | Current status |
| --- | --- | --- |
| 10 | real-feature reference | active and diagnostic entries |
| 20 | CVAE preservation | active and diagnostic entries |
| 30 | provenance-clean expert bank | planned |
| 40 | prior and generation | planned |
| 50 | all-candidate utility matrix | planned for new runs; local historical diagnostic retained |
| 60 | routing and composition | planned |
| 70 | frozen-policy downstream utility | planned |
| 90 | oracles, audits, and rejected lineages | diagnostic or rejected only |

Stages 30 through 70 have no active routing/generation/downstream stack. A
preservation result cannot be treated as an expert bank, router input, or
synthetic downstream-utility result.

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

Stop Stage-20 tuning. Stage 30 now has an eligible consensus-lock input, but
its registry entry remains a planned placeholder without a runnable expert-bank
implementation. Implement and protocol-review that runner before attempting
Stage 30. The stability bundle does not change or unlock outer v1, which still
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
