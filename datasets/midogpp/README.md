# MIDOG++ Dataset Contracts

This directory owns MIDOG++ dataset-contract artifacts before any routing,
SAIL, Virchow2 cache, or C6.3 experiment consumes the dataset.

The v1 contract defines:

- sample: annotation-centered patch
- group: case/specimen/slide id (`case_id`)
- class: mitotic vs hard-negative/non-mitotic
- domain candidates: `scanner_model`, `tumor_type`,
  `tumor_type|lab_or_origin|scanner_model`

Generated patch images and feature caches are local artifacts and must not be
committed. The small CSV/JSON contract artifacts are intended to be tracked so
the dataset definition is auditable.

Tracked layout:

```text
datasets/midogpp/
  configs/annotation_patch_v1.yaml
  configs/deprecated/
  schemas/dataset_contract.schema.json
  src/midogpp_contract/
  scripts/
  tests/
  artifacts/.gitignore
```

The active annotation-patch config uses `bbox_format: xyxy`. The old
`coco_xywh` config has been moved to:

```text
datasets/midogpp/configs/deprecated/annotation_patch_v1_coco_xywh_stale.yaml
```

That deprecated config is provenance only. Do not use it for thesis-facing
cache builds or downstream experiments. It is associated with the stale
full-split cache lineage:

```text
sail/artifacts/pathology_embeddings_midogpp_annotation_patch_v1/virchow2/
```

The expected frozen artifact is:

```text
datasets/midogpp/artifacts/midogpp_annotation_patch_v1/
  manifest.csv
  domain_mapping.json
  split_manifest.csv
  domain_feasibility.csv
  class_balance_by_domain.csv
  leakage_report.json
  dataset_contract.json
  patches_224/
```

Current verified contract snapshot:

- manifest rows: `22569`
- train rows: `9886`
- val rows: `2677`
- test rows: `10006`
- class counts: `11937` mitotic positives and `10632`
  hard-negative/non-mitotic rows
- duplicate `sample_id` values: `0`
- case overlap across train/val/test splits: `0`
- leakage report status: `PASS`

The synced Virchow2 train cache at
`sail/artifacts/pathology_embeddings/midogpp/virchow2/seed42/embeddings/train.pt`
has `9886 x 2560` features and aligns rowwise to the train split with `0`
metadata mismatches across `sample_id`, `label`, `split`, `center`, and
`magnification`. This verifies the dataset/cache contract for real-feature
diagnostics only; it does not validate CVAE generation or routing claims.

For future full train/val/test Virchow2 cache builds, use a cache root that
names the corrected annotation geometry:

```text
sail/artifacts/pathology_embeddings_midogpp_annotation_patch_xyxy/virchow2/
```

Do not reuse `sail/artifacts/pathology_embeddings_midogpp_annotation_patch_v1/`
for new thesis-facing real-feature experiments unless it has been explicitly
rebuilt from the active `xyxy` config and revalidated against the manifest.

Build from the repository root:

```bash
PYTHONPATH=datasets/midogpp/src conda run -n thesis python \
  datasets/midogpp/scripts/build_annotation_patch_contract.py \
  --config datasets/midogpp/configs/annotation_patch_v1.yaml
```

Validate:

```bash
PYTHONPATH=datasets/midogpp/src conda run -n thesis python \
  datasets/midogpp/scripts/validate_contract.py \
  --artifact-root datasets/midogpp/artifacts/midogpp_annotation_patch_v1
```

Inspect eligible domains and downstream cache hints:

```bash
PYTHONPATH=datasets/midogpp/src conda run -n thesis python \
  datasets/midogpp/scripts/inspect_cache_and_domains.py \
  --artifact-root datasets/midogpp/artifacts/midogpp_annotation_patch_v1 \
  --cache-report sail/artifacts/pathology_embeddings/midogpp/virchow2/seed42/reports/cache_builder_report.json
```

Run tests:

```bash
PYTHONPATH=datasets/midogpp/src pytest datasets/midogpp/tests
```

After validation passes, run the SAIL/Virchow2 cache builder in dry-run mode:

```bash
PYTHONPATH=sail/src conda run -n thesis python -m sail.cli build-cache \
  --samples-manifest datasets/midogpp/artifacts/midogpp_annotation_patch_v1/manifest.csv \
  --experiment-seed 42 \
  --output-root sail/artifacts/pathology_embeddings_midogpp_annotation_patch_xyxy \
  --model-ref hf-hub:paige-ai/Virchow2 \
  --batch-size 16 \
  --device cuda \
  --dry-run
```

Only build the actual feature cache after the dry-run passes.
