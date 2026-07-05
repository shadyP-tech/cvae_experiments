# Manifest

## Copied Files

No files were copied verbatim. The extraction is a new minimal implementation
based on the inspected source behavior and artifacts listed below.

## Newly Created Files

| File | Source / evidence used | Reason included |
| --- | --- | --- |
| `README.md` | `docs/context/current_experimental_state.md`, `docs/wiki/04-current-best-approach/current-synthesis.md`, R1.2b reports | Reproducibility and claim-boundary documentation for the extracted method. |
| `MANIFEST.md` | User request and inspected repository structure | Records provenance, inclusions, exclusions, and outside dependencies. |
| `pyproject.toml` | Minimal runtime needs from downstream scripts/tests | Defines package metadata and dependencies. |
| `configs/camelyon17_virchow2_legacy/sail_virchow2.yaml` | Legacy R1.2c config now archived at `cvae_downstream_evaluation/legacy/superseded_by_sail/r12c/configs/r12c_virchow2_dense_config_aggregation.yaml`, `r12b_source_selector_pathology_screen.yaml` | Quarantined Camelyon17-center Virchow2 SAIL config; not a MIDOG++ config. |
| `configs/midogpp_virchow2_real_feature_multiaxis_baseline.yaml` | User-approved MIDOG++ multi-axis diagnostic plan | Locked config for train-only real Virchow2 LODO learnability across tumor, scanner, lab/origin, species, and composite axes. |
| `configs/midogpp_virchow2_real_feature_signal_controls.yaml` | User-approved MIDOG++ signal-control diagnostic plan | Locked config for pooled, tumor-class-balanced, within-tumor, and negative-control real-feature signal checks on the same train cache. |
| `artifacts/.gitignore` | User exclusion requirement | Prevents generated outputs/caches from being tracked inside the extraction. |
| `src/sail/__init__.py` | New | Package marker. |
| `src/sail/protocol.py` | `cvae_downstream_evaluation/src/cvae_downstream_evaluation/protocol.py`, protocol docs | Lightweight protocol errors, row roles, and primary-row leakage checks. |
| `src/sail/metrics.py` | `cvae_downstream_evaluation/src/cvae_downstream_evaluation/downstream.py` | Balanced accuracy, macro-F1, AUROC, and small numeric helpers. |
| `src/sail/config.py` | R1.2b/R1.2c config loaders, with R1.2c provenance archived under `cvae_downstream_evaluation/legacy/superseded_by_sail/r12c/` | Locked config validation for source-only aggregation. |
| `src/sail/splits.py` | `cvae_downstream_evaluation/src/cvae_downstream_evaluation/matrix.py`, `splits.py` | Target evaluation pool construction with support-sample exclusion. |
| `src/sail/features.py` | `pathology_cache_builder.py`, `pathology_embedding_screen.py` | Feature-cache loading plus optional Virchow2 cache creation. |
| `src/sail/pipeline.py` | `pathology_embedding_screen.py`, legacy `r12c_dense_config_aggregation.py` archived under `cvae_downstream_evaluation/legacy/superseded_by_sail/r12c/src/` | Source-inner LODO scoring, robust ranking, dense aggregation, final scoring, and reports. |
| `src/sail/midogpp_multiaxis.py` | User-approved MIDOG++ multi-axis diagnostic plan, MIDOG++ contract surfaces under `datasets/midogpp/`, and SAIL cache/metric helpers | Real-feature multi-axis train-only LODO baseline with strict cache alignment, fold validity gates, leakage reports, and axis-specific decision artifacts. |
| `src/sail/midogpp_signal_controls.py` | User-approved MIDOG++ signal-control diagnostic plan and prior MIDOG++ multiaxis artifact contract | Real-feature train-cache positive/negative controls with case-disjoint splits, tumor-class balancing, case-cluster CIs, and leakage/alignment checks. |
| `src/sail/cli.py` | R1.2b/R1.2c script entrypoints; old R1.2c script is now a fail-fast shim | Minimal CLI for config validation, cache building, and pipeline runs. |
| `tests/test_smoke.py` | Legacy R1.2c tests archived under `cvae_downstream_evaluation/legacy/superseded_by_sail/r12c/tests/` with a non-pytest-discoverable filename | Smoke coverage for imports, config loading, CLI help, dummy evaluation, and leakage firewall. |
| `tests/test_midogpp_multiaxis.py` | User-approved MIDOG++ multi-axis diagnostic plan | Synthetic coverage for axis mapping, class-minimum gates, overlap failures, MLP skip behavior, metrics, and CLI smoke path. |
| `tests/test_midogpp_signal_controls.py` | User-approved MIDOG++ signal-control diagnostic plan | Synthetic coverage for case-disjoint controls, tumor-class balancing, case-level support gates, case-cluster CIs, negative-control reporting, and CLI smoke path. |

## Inspected Evidence

Original extraction evidence recorded in this manifest:

- `docs/context/thesis_project_context.md` (referenced by the documentation
  workflow but absent in the local checkout during the MIDOG++ update)
- `docs/context/current_experimental_state.md` (referenced by the documentation
  workflow but absent in the local checkout during the MIDOG++ update)
- `docs/context/pivot_statement.md` (absent in the local checkout during the
  MIDOG++ update)
- `docs/wiki/04-current-best-approach/current-synthesis.md` (absent in the
  local checkout during the MIDOG++ update)
- `docs/wiki/04-current-best-approach/virchow2-only-rationale.md` (absent in
  the local checkout during the MIDOG++ update)
- `cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/reports/r12b_decision_report.md`
- `cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/reports/r12b_leakage_report.json`
- `cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/tables/r12b_backbone_ranking.csv`
- `cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/tables/r12b_selector_oracle_gap.csv`
- `cvae_downstream_evaluation/legacy/superseded_by_sail/r12c/configs/r12c_virchow2_dense_config_aggregation.yaml`
- `cvae_downstream_evaluation/legacy/superseded_by_sail/r12c/src/r12c_dense_config_aggregation.py`
- `cvae_downstream_evaluation/legacy/superseded_by_sail/r12c/tests/r12c_dense_config_aggregation_legacy_tests.py`

MIDOG++ real-feature signal-control evidence synced and verified after the
initial extraction:

- `datasets/midogpp/artifacts/midogpp_annotation_patch_v1/manifest.csv`
- `datasets/midogpp/artifacts/midogpp_annotation_patch_v1/dataset_contract.json`
- `datasets/midogpp/artifacts/midogpp_annotation_patch_v1/leakage_report.json`
- `sail/artifacts/pathology_embeddings/midogpp/virchow2/seed42/embeddings/train.pt`
- `sail/artifacts/pathology_embeddings/midogpp/virchow2/seed42/reports/cache_builder_report.json`
- `sail/artifacts/midogpp_virchow2_real_feature_signal_controls/tables/control_metrics.csv`
- `sail/artifacts/midogpp_virchow2_real_feature_signal_controls/tables/negative_control_metrics.csv`
- `sail/artifacts/midogpp_virchow2_real_feature_signal_controls/tables/split_manifest.csv`
- `sail/artifacts/midogpp_virchow2_real_feature_signal_controls/tables/predictions.csv`
- `sail/artifacts/midogpp_virchow2_real_feature_signal_controls/reports/leakage_report.json`

Verification result: the real Virchow2 train cache has shape `9886 x 2560`,
metadata length `9886`, and `0` rowwise mismatches against the train manifest
for `sample_id`, `label`, `split`, `center`, and `magnification`. The synced
signal-control summaries report pooled logistic BACC `0.6914`, pooled MLP BACC
`0.7146`, and tumor-balanced logistic BACC `0.6349`, while fit-label and
feature-row-shuffle negative controls remain near chance. This evidence is
diagnostic only and does not support CVAE preservation, routing, or synthetic
generation claims.

Annotation-patch cache provenance update:

- active config: `datasets/midogpp/configs/annotation_patch_v1.yaml`
- active bbox interpretation: `xyxy`
- deprecated config:
  `datasets/midogpp/configs/deprecated/annotation_patch_v1_coco_xywh_stale.yaml`
- rejected stale full-split cache lineage:
  `sail/artifacts/pathology_embeddings_midogpp_annotation_patch_v1/virchow2/`
- intended corrected rebuild root:
  `sail/artifacts/pathology_embeddings_midogpp_annotation_patch_xyxy/virchow2/`

The deprecated `coco_xywh` lineage must not be used for new thesis-facing
real-feature, threshold, or CVAE preservation experiments. It is retained only
to explain stale cache provenance.

## Dependencies Outside `sail/`

Runtime code does not import modules outside `sail/`.

Full real runs still depend on external input artifacts:

- frozen Virchow2 feature caches, or
- a samples manifest and local/workstation access to build Virchow2 caches

The MIDOG++ multi-axis diagnostic requires both a locked MIDOG++ manifest and a
row-aligned Virchow2 train cache. Cache construction is intentionally out of
scope for that diagnostic.

The MIDOG++ signal-control diagnostic uses the same input contract and also
requires `case_id` and `image_path` values for leakage auditing. It never builds
or realigns feature caches.

The verified MIDOG++ signal-control cache path is:

```text
sail/artifacts/pathology_embeddings/midogpp/virchow2/seed42/embeddings/train.pt
```

For new full train/val/test MIDOG++ Virchow2 cache builds, prefer:

```text
sail/artifacts/pathology_embeddings_midogpp_annotation_patch_xyxy/virchow2/seed42/embeddings/
```

The default config writes generated outputs under:

```text
sail/artifacts/
```

No cached embeddings, generated result tables, model weights, or raw data are
included in this extracted folder.
