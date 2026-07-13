# MIDOG++ Experiment File Workflow

Last updated: 2026-07-12

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
- `real-feature-classifier` for the tuned source-inner reference;
- `cvae-preservation` for the four preservation surfaces;
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

The active classifier experiment is:

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

## Registered CVAE Preservation Workflow

The active preservation experiment is:

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
