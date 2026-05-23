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
| `configs/sail_virchow2.yaml` | Legacy R1.2c config now archived at `cvae_downstream_evaluation/legacy/superseded_by_sail/r12c/configs/r12c_virchow2_dense_config_aggregation.yaml`, `r12b_source_selector_pathology_screen.yaml` | Minimal locked config for the current Virchow2 instantiation of SAIL. |
| `artifacts/.gitignore` | User exclusion requirement | Prevents generated outputs/caches from being tracked inside the extraction. |
| `src/sail/__init__.py` | New | Package marker. |
| `src/sail/protocol.py` | `cvae_downstream_evaluation/src/cvae_downstream_evaluation/protocol.py`, protocol docs | Lightweight protocol errors, row roles, and primary-row leakage checks. |
| `src/sail/metrics.py` | `cvae_downstream_evaluation/src/cvae_downstream_evaluation/downstream.py` | Balanced accuracy, macro-F1, AUROC, and small numeric helpers. |
| `src/sail/config.py` | R1.2b/R1.2c config loaders, with R1.2c provenance archived under `cvae_downstream_evaluation/legacy/superseded_by_sail/r12c/` | Locked config validation for source-only aggregation. |
| `src/sail/splits.py` | `cvae_downstream_evaluation/src/cvae_downstream_evaluation/matrix.py`, `splits.py` | Target evaluation pool construction with support-sample exclusion. |
| `src/sail/features.py` | `pathology_cache_builder.py`, `pathology_embedding_screen.py` | Feature-cache loading plus optional Virchow2 cache creation. |
| `src/sail/pipeline.py` | `pathology_embedding_screen.py`, legacy `r12c_dense_config_aggregation.py` archived under `cvae_downstream_evaluation/legacy/superseded_by_sail/r12c/src/` | Source-inner LODO scoring, robust ranking, dense aggregation, final scoring, and reports. |
| `src/sail/cli.py` | R1.2b/R1.2c script entrypoints; old R1.2c script is now a fail-fast shim | Minimal CLI for config validation, cache building, and pipeline runs. |
| `tests/test_smoke.py` | Legacy R1.2c tests archived under `cvae_downstream_evaluation/legacy/superseded_by_sail/r12c/tests/` with a non-pytest-discoverable filename | Smoke coverage for imports, config loading, CLI help, dummy evaluation, and leakage firewall. |

## Inspected Evidence

Primary evidence:

- `docs/context/thesis_project_context.md`
- `docs/context/current_experimental_state.md`
- `docs/context/pivot_statement.md`
- `docs/wiki/04-current-best-approach/current-synthesis.md`
- `docs/wiki/04-current-best-approach/virchow2-only-rationale.md`
- `cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/reports/r12b_decision_report.md`
- `cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/reports/r12b_leakage_report.json`
- `cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/tables/r12b_backbone_ranking.csv`
- `cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/tables/r12b_selector_oracle_gap.csv`
- `cvae_downstream_evaluation/legacy/superseded_by_sail/r12c/configs/r12c_virchow2_dense_config_aggregation.yaml`
- `cvae_downstream_evaluation/legacy/superseded_by_sail/r12c/src/r12c_dense_config_aggregation.py`
- `cvae_downstream_evaluation/legacy/superseded_by_sail/r12c/tests/r12c_dense_config_aggregation_legacy_tests.py`

## Dependencies Outside `sail/`

Runtime code does not import modules outside `sail/`.

Full real runs still depend on external input artifacts:

- frozen Virchow2 feature caches, or
- a samples manifest and local/workstation access to build Virchow2 caches

The default config writes generated outputs under:

```text
sail/artifacts/
```

No cached embeddings, generated result tables, model weights, or raw data are
included in this extracted folder.
