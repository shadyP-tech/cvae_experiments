# SAIL: Source-only Aggregation via Inner-domain Leaveout

## What This Is

This folder is a clean extraction of **SAIL**:

```text
frozen feature caches
-> source-inner leave-one-domain/center-out config scoring
-> source-only top-k config selection
-> dense probability aggregation
-> held-out target-center balanced accuracy scoring
```

The current instantiation uses Virchow2 feature caches, but the method title is
backbone-agnostic. The pipeline is real-feature evaluation. It does not claim
that CVAE generation preserves Virchow2 utility. CVAE preservation is the next
diagnostic only after this real-feature gate is verified.

## Why This Is Current Best

Repository evidence in
`cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/`
shows Virchow2 is currently the strongest source-inner-LODO selected pathology
backbone:

- source-selected Virchow2 mean BACC: `0.9155`
- posthoc target-eval Virchow2 mean BACC: `0.9474` audit-only
- selector oracle top-3 containment: `15/15` in inspected primary rows
- sparse top-1 selection remains brittle

That motivates SAIL: keep selection source-only and stabilize sparse top-1
selection through dense top-k config aggregation. Virchow2 is the current
backbone used to instantiate SAIL.

## Expected Inputs

The runner expects frozen Virchow2 feature caches with this payload:

```python
{
  "embeddings": <2D tensor or array>,
  "metadata": [
    {"sample_id": "...", "center": "0", "label": 0, "split": "train", ...},
    ...
  ],
  "feature_extractor": {...}
}
```

Default cache paths:

```text
sail/artifacts/pathology_embeddings/virchow2/seed{seed}/embeddings/{train,test}.pt
```

The test helper also supports `.npz` caches, but full runs should use frozen
`.pt` caches built from the same samples manifest as the support/evaluation
split.

## Commands

Validate the config:

```bash
PYTHONPATH=sail/src conda run -n thesis python -m sail.cli validate-config \
  --config sail/configs/sail_virchow2.yaml
```

Dry-run Virchow2 cache creation:

```bash
PYTHONPATH=sail/src conda run -n thesis python -m sail.cli build-cache \
  --config sail/configs/sail_virchow2.yaml \
  --samples-manifest path/to/samples.csv \
  --experiment-seed 42 \
  --dry-run
```

Build Virchow2 caches on a workstation with `torch`, `timm`, and model access:

```bash
PYTHONPATH=sail/src conda run -n thesis python -m sail.cli build-cache \
  --config sail/configs/sail_virchow2.yaml \
  --samples-manifest path/to/samples.csv \
  --experiment-seed 42
```

Run the extracted pipeline:

```bash
PYTHONPATH=sail/src conda run -n thesis python -m sail.cli run \
  --config sail/configs/sail_virchow2.yaml
```

Run the smoke tests:

```bash
conda run -n thesis python -m pytest sail/tests
```

## Expected Outputs

The default run writes:

```text
sail/artifacts/virchow2_dense_source_selected/
  tables/source_lodo_selection_matrix.csv
  tables/source_k_selection_matrix.csv
  tables/dense_aggregation_matrix.csv
  tables/member_manifest.csv
  tables/center_summary.csv
  manifests/protocol_manifest.json
  reports/leakage_report.json
  reports/decision_report.md
```

Generated artifacts are ignored by `sail/artifacts/.gitignore`.

## Protocol Safety

Selection is source-only:

- source-inner LODO config scores use source training centers only
- held-out target center is excluded from config ranking and aggregation
- target support labels are not used
- target evaluation labels are used only after predictions for final scoring
- metadata is not a primary selector

Verify safety by checking:

```text
reports/leakage_report.json
```

Expected status:

```json
{"status": "PASS"}
```

Also inspect these columns:

- `selection_used_target_labels` must be `false`
- `fit_used_target_center` must be `false`
- `target_eval_labels_used_for_scoring` may be `true`

## What Is Intentionally Excluded

Excluded from this folder:

- metadata-first routing as the primary method
- DINOv2/PCA-only historical experiments
- cross-backbone aggregation audit rows
- source-temperature calibration audit rows
- CVAE training, CVAE generation, and CVAE preservation claims
- failed or quarantined legacy branches
- exploratory notebooks
- cached embeddings, model weights, generated tables, and large artifacts

## Difference From Metadata Routing And CVAE Experiments

Metadata routing treats clinical/domain metadata as a compatibility proxy. This
extraction uses source-only downstream utility in Virchow2 feature space as the
selection signal; metadata remains a baseline or interpretability layer.

CVAE experiments ask whether generated embeddings preserve utility. This
extraction does not generate embeddings. It tests whether the real Virchow2
feature-space method is stable enough to justify a later CVAE preservation
experiment.
