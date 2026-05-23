# Source-Only Selection

## Purpose

Define source-only selection for routing, configs, and aggregation rules.

## Key Claims

- Source-only selection means target evaluation labels do not choose the method, config, expert, k, aggregation rule, or calibration rule.
- R1.2b source-inner-LODO selected rows are protocol-clean diagnostic evidence.
- SAIL must select k and aggregation rule using source-inner LODO only.

## Evidence / Source Artifacts

- `../../../cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/reports/r12b_decision_report.md`
- `../../../cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/reports/r12b_leakage_report.json`
- `../../../sail/configs/sail_virchow2.yaml`

## Interpretation

Source-only does not mean blind to all target information. Some protocols allow unlabeled target support statistics. It means forbidden target evaluation outcomes do not control selection.

## Implication For Thesis

Deployable claims require source-only or allowed-support-only selection. Posthoc target-eval rows are feasibility audits.

## Limitations

SAIL output artifacts are missing locally, so final source-only compliance for that run remains TODO.

## Next Checks

- Verify `target_eval_labels_used_for_scoring_only` in the SAIL leakage report.
- Confirm primary rows use source-inner-only selection.
