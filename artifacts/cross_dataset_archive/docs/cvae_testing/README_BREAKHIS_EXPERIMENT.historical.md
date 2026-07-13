# BreakHis Routed CVAE Experiment

This file is part of historical BreakHis documentation preserved in the archive and
is not active guidance for MIDOG++ evidence or the current `src/midogpp_thesis` workflow.

This project runs a leakage-aware, metadata-routed CVAE experiment on BreakHis and stores outputs per run.

## Current project state

- Active config is in `configs/experiments/breakhis/routed_cvae_v1.yaml`.
- Legacy `artifacts/` output flow has been removed.
- Outputs are run-scoped under `outputs/<dataset>/<experiment>/<run_id>/`.
- Existing confirmation runs are present under:
  - `outputs/breakhis/routed_cvae_v1/seed42/`
  - `outputs/breakhis/routed_cvae_v1/seed43/`
  - `outputs/breakhis/routed_cvae_v1/seed44/`

## Environment

Install required packages in your selected Python environment:

```bash
pip install pyyaml torch torchvision pillow matplotlib umap-learn scikit-learn
```

## Dataset location

Place BreakHis images under:

`data/BreakHis/`

The parser infers:
- label from path tokens (`benign`/`malignant` or `b`/`m` patterns)
- magnification from filename/folder (`40`, `100`, `200`, `400`, optionally with `x`)
- patient ID using filename heuristics

If patient ID parsing is incomplete and `require_patient_ids: false`, image-level fallback is used for those samples and logged.

### Camelyon17

Camelyon17 config is available at:

`configs/experiments/camelyon17/routed_cvae_v1.yaml`

Expected default dataset root:

`data/Camelyon17/`

Expected metadata file (default):

`data/Camelyon17/metadata.csv`

Current Camelyon17 domain definition:
- one domain = one site/center (`data.domain_field`, default `center`)

Current split behavior:
- site-wise grouped split (grouped by `patient+slide` when slide exists, else `patient`)
- leakage-safe across train/val/test for the grouping key

Supported metadata columns:
- label: one of `label`, `tumor`, `target`, `y`
- domain: `center` by default (override with `data.domain_field`)
- patient ID: one of `patient`, `patient_id`, `case_id`
- optional split: `split` or `fold` (`train/val/test` or `0/1/2`)
- image path: one of `image_path`, `filepath`, `path`, `file`, `filename`

If no path column exists, the loader also supports WILDS-style columns:
- `patient`, `node`, `x_coord`, `y_coord`

and reconstructs image paths as:
- `patches/patient_<id>_node_<node>/patch_patient_<id>_node_<node>_x_<x>_y_<y>.png`

## Run one experiment

From `codebase/cvae_testing`:

```bash
python -m src.run_experiment --config configs/experiments/breakhis/routed_cvae_v1.yaml
```

Optional seed override:

```bash
python -m src.run_experiment --config configs/experiments/breakhis/routed_cvae_v1.yaml --seed 43
```

Optional explicit run ID:

```bash
python -m src.run_experiment --config configs/experiments/breakhis/routed_cvae_v1.yaml --seed 43 --run-id seed43
```

Camelyon17 run:

```bash
python -m src.run_experiment --config configs/experiments/camelyon17/routed_cvae_v1.yaml
```

## Output structure

Each run is saved to:

`outputs/<dataset_name>/<experiment_name>/<run_id>/`

Per-run files:
- `config_resolved.yaml`
- `manifests/samples.csv`
- `embeddings/{train,val,test}.pt`
- `checkpoints/*.pt`
- `reports/leakage_report.json`
- `reports/cache_report.json`
- `reports/expert_matrix.json`
- `reports/reconstruction_vs_magnification.png`
- `reports/routing_results.json`
- `plots/*.png` (auto-generated after each run)

Latest run pointer:
- `outputs/<dataset_name>/<experiment_name>/latest.txt`

## Analysis scripts

### 1) Plot generation

Plots are generated automatically at the end of `src.run_experiment` and stored in the run's `plots/` folder.

Manual regeneration is still available.

Use latest run automatically:

```bash
python -m src.eval.make_plots
```

Or target a specific run:

```bash
python -m src.eval.make_plots --run-dir outputs/breakhis/routed_cvae_v1/seed44
```

### 2) Magnification signal checks (A/B/C)

```bash
python -m src.eval.check_magnification_signal --run-dir outputs/breakhis/routed_cvae_v1/seed44
```

### 3) Case-3 check (raw vs embedding linear probes)

```bash
python -m src.eval.check_case3_raw_vs_embeddings --run-dir outputs/breakhis/routed_cvae_v1/seed44
```

### 4) Seed sweep summary

```bash
python -m src.eval.summarize_seed_sweep \
  --experiment-dir outputs/breakhis/routed_cvae_v1 \
  --run-ids seed42 seed43 seed44 \
  --csv-out results/comparison_tables/breakhis_routed_cvae_v1_seed_sweep.csv \
  --md-out results/summaries/breakhis_routed_cvae_v1_seed_sweep.md
```

## Implemented baselines

- hard metadata routing
- soft metadata routing
- random expert (single fixed random expert per run)
- uniform expert sampling (new random expert per sample)
- equal-weight scoring ensemble
- global CVAE
- oracle expert (true magnification)

Routing diagnostics include:
- selection accuracy
- confusion matrix (`True Domain x Routed Expert`)
- expert confidence stats (mean/variance ELBO per expert per domain)

## Fixed setting

Routing similarity baseline uses fixed:
- `tau = 100`

No `tau` tuning is performed in the first iteration.

For Camelyon17, default routing baseline is:
- `strategy: categorical_exact` (exact site metadata match)

Optional soft-routing extension:
- `strategy: site_similarity_matrix` with a user-provided `routing.similarity_matrix` in config.
