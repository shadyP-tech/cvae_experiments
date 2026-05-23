# Decision Labels

## Purpose

Document how decision labels should be used.

## Key Claims

- Decision labels summarize protocol and result status.
- They are not a substitute for artifact inspection.
- Labels must preserve audit/deployable boundaries.

## Evidence / Source Artifacts

- `../../../cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/reports/r12b_decision_report.md`
- `../../../cvae_downstream_evaluation/artifacts/reports/r12_decision_report.md`
- `../../../sail/src/sail/`

## Interpretation

R1.2b labels include `R12B_SOURCE_SELECTED_090_SUPPORTED` and `R12_WEAK_CENTER_PERSISTS`, showing that mean support and weak-center failure can coexist.

## Implication For Thesis

Decision labels should be reported with the supporting metric rows and claim boundary.

## Limitations

Future SAIL labels are not available until outputs exist. TODO: verify against artifact.

## Next Checks

- Add SAIL decision labels after running/syncing the report.
