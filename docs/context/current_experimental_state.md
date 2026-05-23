# Current Experimental State

Last updated: 2026-05-23

## Purpose

This file records the changing experimental synthesis for the thesis
repository. It should be updated when new runs, synced artifacts, decision
reports, or implementation plans change the current interpretation.

Stable framing belongs in:

```text
docs/context/thesis_project_context.md
```

## Evidence Status

Labels used below:

- Verified artifact: read from a local report, table, config, source file, or
  test in this repository.
- Existing synthesis note: read from a local narrative note, but not necessarily
  independently revalidated against raw artifacts in this update.
- Provided synthesis: supplied in the current user prompt; verify against
  artifact if available.
- TODO: missing or unverified evidence.

## Current Readout

The current empirical path is:

```text
R1.2b source-only pathology embedding selector evidence
-> SAIL Virchow2 dense source-selected config aggregation
-> if stability gates pass, vanilla Virchow2 CVAE preservation test
```

Verified artifact: R1.2b shows Virchow2 is currently the strongest
source-inner-LODO selected pathology backbone, but sparse top-1 config
selection is brittle at the seed/center level. This motivates a Virchow2-only
top-k dense aggregation diagnostic rather than an immediate complex CVAE
rebuild.

Current implementation extraction:

```text
sail/
```

SAIL means Source-only Aggregation via Inner-domain Leaveout. The method name is
backbone-agnostic; `sail/configs/sail_virchow2.yaml` is the current Virchow2
instantiation.

## Verified Source Artifacts

Primary current artifacts:

- `cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/reports/r12b_decision_report.md`
- `cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/reports/r12b_leakage_report.json`
- `cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/tables/r12b_backbone_ranking.csv`
- `cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/tables/r12b_center_summary.csv`
- `cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/tables/r12b_selector_oracle_gap.csv`
- `sail/configs/sail_virchow2.yaml`
- `sail/src/sail/`
- `sail/tests/test_smoke.py`
- `PROTOCOL_STATUS.md`

Historical and contextual artifacts:

- `cvae_downstream_evaluation/artifacts/reports/r12_decision_report.md`
- `cvae_downstream_evaluation/artifacts/tables/r12_backbone_ranking.csv`
- `cvae_downstream_evaluation/artifacts/tables/r12_center_summary.csv`
- `cvae_downstream_evaluation/docs/c63_conceptual_synthesis.md`
- `cvae_testing/thesis_outline.txt`

## R1.2 Pathology Embedding Screen

Verified artifact:
`cvae_downstream_evaluation/artifacts/reports/r12_decision_report.md`

Summary:

- Best posthoc target-eval mean BACC: 0.9090.
- Best source-inner-LODO selected mean BACC: 0.8805.
- Mean delta vs Z1.1 DINOv2/PCA64: 0.0580.
- Top source-selected backbone: `phikon`.
- Top posthoc audit backbone: `phikon`.
- Weak-center behavior persisted.

Interpretation:

R1.2 supported the direction that pathology foundation embeddings improve the
real-feature ceiling compared with the earlier DINOv2/PCA64 reference, but it
did not resolve weak-center stability.

Claim boundary:

Posthoc target-eval rows are audit-only. Source-inner-LODO selected rows are the
protocol-clean representation-selection evidence for this benchmark.

## R1.2b Source-Only Compatibility Selector Audit

Verified artifact:
`cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/reports/r12b_decision_report.md`

Decision labels:

- `R12B_SOURCE_SELECTOR_SCREEN_COMPLETE`
- `R12B_SOURCE_SELECTED_090_SUPPORTED`
- `R12_BACKBONE_MEAN_IMPROVES_WEAK_CENTER_FAILS`
- `R12_WEAK_CENTER_PERSISTS`
- `R12_EVAL_CLASS_BALANCE_CAVEAT`

Verified summary:

- Best posthoc target-eval mean BACC: 0.9481.
- Best source-inner-LODO selected mean BACC: 0.9155.
- Mean delta vs Z1.1 DINOv2/PCA64: 0.0931.
- Top source-selected backbone: `virchow2` with mean BACC 0.9155.
- Top posthoc audit backbone: `virchow2` with mean BACC 0.9474.
- Mean selector oracle gap: 0.0326.
- Median selected rank under target utility: 47.0.

Verified from `r12b_backbone_ranking.csv`:

| Backbone | Selection regime | Mean BACC | Worst center BACC | Centers >= 0.85 | Eligibility |
| --- | --- | ---: | ---: | ---: | --- |
| virchow2 | source_inner_lodo_selected | 0.9155 | 0.8070 | 4 | deployable_diagnostic |
| virchow2 | posthoc_target_eval | 0.9474 | 0.8166 | 4 | audit_only |

Verified center means from `r12b_center_summary.csv`:

| Held-out center | Source-selected backbone | Source-selected BACC | Weak-center status |
| --- | --- | ---: | --- |
| 0 | virchow2 | 0.9950 | repaired |
| 1 | virchow2 | 0.9130 | repaired |
| 2 | uni/virchow2 | 0.9922 | repaired |
| 3 | virchow2 | 0.8070 | persists |
| 4 | virchow2 | 0.8703 | repaired |

Verified selector diagnostics from `r12b_selector_oracle_gap.csv`, primary
robust-penalty rows:

- Rows inspected: 15.
- Top-1 config match: 0/15.
- Top-3 contains oracle: 15/15.
- Mean selected target BACC: 0.9155.
- Mean oracle gap: 0.0326.
- Median selected rank under target utility: 47.0.
- Mean Spearman source-score vs target BACC: 0.4589.
- Seed mean BACC values: seed 42 = 0.8956, seed 43 = 0.8644, seed 44 = 0.9865.
- Seed mean-BACC sample std: 0.0635.

Observed sparse-selection failures:

| Seed | Held-out center | Source-selected BACC | Posthoc best BACC | Oracle gap |
| --- | --- | ---: | ---: | ---: |
| 42 | 3 | 0.5000 | 0.5000 | 0.0000 |
| 43 | 4 | 0.6250 | 1.0000 | 0.3750 |

Interpretation:

The selector appears to identify a useful neighborhood of high-utility configs,
because the oracle is always in the top 3 in the inspected primary rows. Exact
top-1 selection is unstable, because top-1 never matches the oracle and some
seed/center rows collapse below the SAIL rebuild-stability floor.

This is the direct motivation for SAIL dense top-k aggregation.

## R1.2b Leakage / Protocol Status

Verified artifact:
`cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/reports/r12b_leakage_report.json`

Status:

- `status`: `PASS`
- `target_eval_labels_for_deployable_selection`: false
- `target_eval_labels_for_scoring_only`: true
- `diagnostics_used_for_selection`: false
- `diagnostics_used_for_decision_labels`: false
- `cvae_experts_modified`: false
- `violations`: []

Interpretation:

R1.2b source-inner-LODO selected rows are protocol-clean diagnostic evidence for
source-only representation/config selection. They are not evidence that CVAE
generation preserves the same utility.

## SAIL Current Implementation

Verified artifacts:

- `sail/configs/sail_virchow2.yaml`
- `sail/src/sail/`
- `sail/tests/test_smoke.py`
- `cvae_downstream_evaluation/legacy/superseded_by_sail/r12c/README.md`

Current status:

```text
SAIL is the active implementation. R1.2c-V code/config/tests have been archived
as provenance and are no longer active runnable surfaces.
```

Missing expected artifact root:

```text
sail/artifacts/virchow2_dense_source_selected
```

TODO: run or sync SAIL artifacts and verify:

- `tables/source_lodo_selection_matrix.csv`
- `tables/source_k_selection_matrix.csv`
- `tables/dense_aggregation_matrix.csv`
- `tables/member_manifest.csv`
- `tables/center_summary.csv`
- `manifests/protocol_manifest.json`
- `reports/leakage_report.json`
- `reports/decision_report.md`

Primary row role:

```text
source_only_dense_virchow2
```

Primary candidate pool:

```text
backbone = virchow2 only
representation in {raw, PCA64, PCA128, PCA256}
classifier C in {0.01, 0.1, 1.0, 10.0}
class_weight in {none, balanced}
standardization and PCA fit on source training rows only
```

Source-only robust score:

```text
robust_score =
    mean_inner_bacc
    - 0.25 * std_inner_bacc
    - 0.50 * max(0, 0.85 - min_inner_bacc)
```

Purpose:

```text
penalize weak-center collapse instead of selecting configs by mean BACC alone
```

Primary aggregation:

- fixed k values: 1, 3, 5, 10
- primary k values: 3, 5, 10
- aggregation rules: geometric, arithmetic
- primary calibration rule: none
- audit calibration rules: none, source_temperature

Geometric aggregation:

```text
score_c = mean_i log(max(p_i(c), eps))
```

Arithmetic aggregation:

```text
score_c = mean_i p_i(c)
```

Selection rules:

- k, aggregation rule, and primary calibration are selected using source-inner
  LODO only.
- Target eval labels are scoring-only.
- Cross-backbone rows are not part of the active SAIL Virchow2 instantiation.
- Source-temperature calibration is not part of the active SAIL primary method.
- CVAE experts are not retrained or evaluated by SAIL.

Implementation guardrails verified in tests:

- Config loading works for `sail/configs/sail_virchow2.yaml`.
- CLI help works through `python -m sail.cli`.
- Synthetic evaluation path verifies source-only selection does not consume
  target-eval labels before final scoring.

## SAIL Rebuild Gate

Verified artifact:
`sail/configs/sail_virchow2.yaml`

The SAIL Virchow2 instantiation can justify a vanilla Virchow2 CVAE
preservation test only if Virchow2-only primary rows satisfy:

```text
mean BACC >= 0.92
worst center BACC >= 0.85
seed mean-BACC std <= 0.03
no seed has worst-center BACC < 0.75
delta vs R1.2b Virchow2 top-1 >= 0.005
```

Current gate status:

```text
TODO: verify against SAIL artifacts.
```

R1.2b top-1 sparse selection would not satisfy the seed-stability spirit of
this gate because inspected primary rows include seed/center BACC values below
0.75 and seed mean-BACC std of 0.0635.

## Cross-Backbone Aggregation

Archived provenance:
`cvae_downstream_evaluation/legacy/superseded_by_sail/r12c/configs/r12c_virchow2_dense_config_aggregation.yaml`

The archived R1.2c-X audit allowed:

```text
phikon + uni + virchow2
```

but it was audit-only and is not part of the active SAIL Virchow2
implementation.

Reason:

```text
Cross-backbone ensemble diversity may improve real-feature classifier
performance, but it does not prove that one clean generative feature space is
sufficient for CVAE expert modeling.
```

Therefore:

```text
cross-backbone dense success -/> CVAE rebuild readiness
```

Only Virchow2-only primary rows may justify the next CVAE preservation test.

## CVAE / C6.3 Context

Existing synthesis note:
`cvae_downstream_evaluation/docs/c63_conceptual_synthesis.md`

That note frames C6.3 as dense posthoc output aggregation over frozen CVAE-style
expert/mode classifiers, not as a new compatibility estimator. It reports an
approximate C6.3 full-context mean BACC of 0.814 with weak centers still present.

Evidence label:

```text
Existing synthesis note; verify against synced C6.3 decision artifacts before
using as a final thesis table.
```

Current interpretation:

- C6.3 supports the value of dense aggregation when sparse top-1 routing is
  brittle.
- C6.3 does not prove that a Virchow2 CVAE preserves the R1.2b real-feature
  utility.
- SAIL is a real-feature source-only aggregation method upstream of any new
  CVAE rebuild.

## Not Yet Supported

Current evidence does not yet support:

```text
Virchow2 CVAEs preserve the observed real-feature utility.
```

```text
Metadata similarity alone is sufficient for compatibility estimation.
```

```text
Cross-backbone classifier ensembling proves one clean generative feature space.
```

```text
Posthoc target-eval ranking is deployable.
```

## Current Next Sequence

1. Run or sync SAIL:
   Virchow2 dense source-selected config aggregation.

2. Verify SAIL protocol safety:
   target-eval labels are scoring-only and source-inner selection stays
   source-only.

3. If SAIL passes the Virchow2-only gate:
   run a vanilla Virchow2 CVAE preservation test.

4. If vanilla Virchow2 CVAE preserves most real-feature utility:
   keep the CVAE simple and focus the thesis contribution on compatibility and
   routing.

5. If vanilla Virchow2 CVAE loses too much utility:
   calibrated or composable Virchow2 CVAE work becomes a justified novel
   contribution.

The label `R1.3a Vanilla Virchow2 CVAE Rebuild` is provided synthesis at this
point. TODO: create or verify a corresponding config/artifact before treating
that name as an implemented experiment.

## Missing Artifacts / TODOs

- TODO: run or sync SAIL output artifacts under
  `sail/artifacts/virchow2_dense_source_selected`.
- TODO: verify whether SAIL primary rows pass the rebuild gate.
- TODO: create or verify the vanilla Virchow2 CVAE preservation-test config if
  SAIL passes.
- TODO: verify C6.3 numerical synthesis against raw/synced decision artifacts
  before using it as a final thesis result table.
