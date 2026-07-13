# MIDOG++ Phase-1 Diagnostic Runbook

This runbook is a **historical** MIDOG++ all-candidate downstream diagnostic
matrix and a legacy-path migration artifact.
The commands and paths below (`cvae_downstream_evaluation/*`,
`cvae_rebuild/*`) are archival references and are not active run instructions
in the current `src/midogpp_thesis` checkout.

For execution in this repo, use the active documentation and provenance in
`docs/context/current_experimental_state.md` and `docs/context/midogpp_experiment_file_workflow.md`.

This runbook is diagnostic-only evidence: downstream target metrics may evaluate
candidates and baselines, but they must not select a deployable router,
generation setting, classifier setting, or feature table.

## Scope

- Machine: university workstation.
- Repo path: `<remote_repo>/cvae_experiments`.
- Config: `cvae_downstream_evaluation/configs/experiments/utility_matrix/virchow2_midogpp_all_candidates.yaml`.
- Dataset regime: MIDOG++ heldout center, eligible centers `0,1,2,3,5,6,7,8,9`; center `4` remains excluded.
- Support regime: no target support for this phase, using `support_size=0`, `support_seed=none`, `support_set_id=none`.
- Synthetic budget: `synthetic_per_class_total=128`, fixed before target evaluation.
- Classifier: locked logistic regression from `cvae_downstream_evaluation.downstream`.

## Real-Feature Gate Context

Before using this phase-1 matrix to motivate CVAE candidate-surface follow-up,
refer to the independent MIDOG++ real-feature gate provenance path (legacy form):

```text
midogpp_real_feature_gate/artifacts/midogpp_real_feature_gate_v1/
```

Verified synced status:

- decision labels: `GO_REAL_FEATURE_GATE_PASSED`,
  `CLAIM_SCOPE_REAL_FEATURE_TRANSFER_ONLY`
- leakage/provenance report: `PASS`
- eligible held-out centers: `0,1,2,3,5,6,7,8,9`
- source-only valid folds: `9/9`
- mean source-only BACC/AUROC: `0.668` / `0.728`
- worst eligible center: center `2`, BACC `0.587`, AUROC `0.629`
- pooled diagnostic ceiling mean BACC/AUROC: `0.902` / `0.964`

Interpretation for phase-1 work:

- Real Virchow2 MIDOG++ features have above-chance source-only held-out-center
  signal.
- The pooled diagnostic ceiling shows substantial headroom, so exploratory CVAE
  candidate-surface work is defensible.
- This gate does not validate CVAE preservation, compatibility routing,
  synthetic downstream utility, or generative quality.
- The current gate bundle is missing separate negative-control and
  uncertainty/seed-stability artifacts; future phase-1 conclusions should treat
  it as `USABLE WITH CAVEATS`.

## Required Inputs

- Source-summary manifest:
  `cvae_rebuild/artifacts/midogpp/virchow2_cvae_dense_late_all_sources_midogpp_v1/tables/exported_source_summary_manifest.csv`
- Source-summary `.npz` files referenced by the manifest.
- MIDOG++ target eval embedding cache, either:
  - `--test-cache-root <cache_root>` containing `seed42/embeddings/test.pt` or `seed42/embeddings/test.npz`, or
  - `--test-cache-path <single_test_cache.pt_or_npz>`.
- Optional locked baseline matrices:
  - `cvae_rebuild/artifacts/midogpp/virchow2_cvae_dense_late_all_sources_midogpp_v1/tables/dense_late_all_sources_downstream_matrix.csv`
  - `cvae_rebuild/artifacts/midogpp/virchow2_cvae_dense_late_all_sources_midogpp_v1/tables/real_feature_reference_matrix.csv`

## Stop Conditions

Stop before scoring if any condition is true:

- `git status --short` contains unreviewed protocol edits.
- The source-summary manifest points to missing `.npz` files.
- The eval cache is missing or does not contain binary labels for every requested heldout center.
- Any requested baseline method is missing for any requested heldout center.
- The config validator rejects the frozen MIDOG++ config.
- Domain `4` appears in a selection-eligible candidate pool.
- The heldout center appears as a selection-eligible candidate.

## Preflight

Run this on the workstation before scoring. The script computes
`config_hash`, `protocol_hash`, and `feature_frame_hash` after preflight,
writes `configs/frozen_protocol_snapshot.json`, and records the values in
`reports/run_hashes_report.json`. The run-hashes report also records
`summary_manifest_hash`, `source_summary_file_hashes`, and `cache_file_hashes`
so a rerun can detect changed input files at the same paths.

```bash
conda run -n thesis python cvae_downstream_evaluation/scripts/run_midogpp_source_summary_phase1.py \
  --summary-manifest cvae_rebuild/artifacts/midogpp/virchow2_cvae_dense_late_all_sources_midogpp_v1/tables/exported_source_summary_manifest.csv \
  --test-cache-root <cache_root> \
  --out-dir cvae_downstream_evaluation/artifacts/midogpp/phase1_virchow2_seed42 \
  --experiment-seed 42 \
  --replicate-seed 0 \
  --heldout-centers 0,1,2,3,5,6,7,8,9 \
  --synthetic-per-class-total 128 \
  --generation-seed 17 \
  --latent-sample-seed 17 \
  --classifier-seed 23 \
  --baseline-matrix cvae_rebuild/artifacts/midogpp/virchow2_cvae_dense_late_all_sources_midogpp_v1/tables/real_feature_reference_matrix.csv \
  --baseline-method real_source_embedding_classifier_dense_reference \
  --preflight-only
```

Expected preflight artifacts:

- `reports/source_summary_preflight_report.json`
- `reports/baseline_preflight_report.json` when baseline args are supplied
- `reports/run_hashes_report.json`
- `configs/frozen_protocol_snapshot.json`

Preflight reports must contain `"status": "PASS"`. The baseline preflight
report must include `baseline_matrix_hashes` and `baseline_row_hashes`. The
run-hashes report must contain `config_hash`, `protocol_hash`, and
`feature_frame_hash`. If source-summary or baseline preflight fails, the script
still writes the corresponding preflight report with `"status": "FAIL"` and an
`error_message`; do not continue to scoring from a failed report.

## Scoring

After preflight passes, rerun the same command without `--preflight-only`.

```bash
conda run -n thesis python cvae_downstream_evaluation/scripts/run_midogpp_source_summary_phase1.py \
  --summary-manifest cvae_rebuild/artifacts/midogpp/virchow2_cvae_dense_late_all_sources_midogpp_v1/tables/exported_source_summary_manifest.csv \
  --test-cache-root <cache_root> \
  --out-dir cvae_downstream_evaluation/artifacts/midogpp/phase1_virchow2_seed42 \
  --experiment-seed 42 \
  --replicate-seed 0 \
  --heldout-centers 0,1,2,3,5,6,7,8,9 \
  --synthetic-per-class-total 128 \
  --generation-seed 17 \
  --latent-sample-seed 17 \
  --classifier-seed 23 \
  --baseline-matrix cvae_rebuild/artifacts/midogpp/virchow2_cvae_dense_late_all_sources_midogpp_v1/tables/real_feature_reference_matrix.csv \
  --baseline-method real_source_embedding_classifier_dense_reference
```

Expected scoring artifacts:

- `tables/diagnostic_downstream_utility.csv`
- `tables/diagnostic_downstream_utility.schema.json`
- `tables/candidate_manifest.csv`
- `tables/candidate_oracle_summary.csv`
- `tables/baseline_comparison.csv`
- `reports/leakage_report.json`
- `reports/decision_summary.md`

## Post-Run Validation

Run this on the workstation immediately after scoring, and again locally after
syncing artifacts back to the MacBook.

```bash
conda run -n thesis python cvae_downstream_evaluation/scripts/validate_midogpp_phase1_artifacts.py \
  --artifacts-root cvae_downstream_evaluation/artifacts/midogpp/phase1_virchow2_seed42 \
  --expected-heldout-center 0 \
  --expected-heldout-center 1 \
  --expected-heldout-center 2 \
  --expected-heldout-center 3 \
  --expected-heldout-center 5 \
  --expected-heldout-center 6 \
  --expected-heldout-center 7 \
  --expected-heldout-center 8 \
  --expected-heldout-center 9 \
  --expected-baseline-method real_source_embedding_classifier_dense_reference \
  --require-preflight-reports
```

Expected validation artifact:

- `reports/phase1_validation_report.json` with `"status": "PASS"`.

If validation fails, the same path is written with `"status": "FAIL"` and an
`error_message`. A failed validation report is a stop condition for thesis-facing
interpretation and for local post-sync acceptance.

## Sync Back To MacBook

From a local Mac terminal:

```bash
rsync -avh --progress thesis-ws:<remote_repo>/cvae_experiments/cvae_downstream_evaluation/artifacts/midogpp/phase1_virchow2_seed42/ \
  /Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_downstream_evaluation/artifacts/midogpp/phase1_virchow2_seed42/
```

After sync, rerun the post-run validator locally. Do not interpret the result
as thesis-facing until validation passes locally.
