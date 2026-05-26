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
- `../../../cvae_rebuild/artifacts/virchow2_cvae_paired_dense_all4_reliability_confirmation_v1/reports/decision_summary.md`

## Interpretation

R1.2b labels include `R12B_SOURCE_SELECTED_090_SUPPORTED` and `R12_WEAK_CENTER_PERSISTS`, showing that mean support and weak-center failure can coexist.

D-series labels show the same discipline for generated embeddings:

| Label | Interpretation |
| --- | --- |
| `INELIGIBLE` | Primary method did not satisfy a predeclared component/split/eligibility gate, even if diagnostic BACC is high. |
| `D1_1_PARTIAL_EVIDENCE` | Adaptive source-local summaries restored some viability but missed centralized retention or stability. |
| `D1_2_PARTIAL_EVIDENCE` | Source-local reliability weighting improved decentralized composition but did not become target-specific routing. |
| `D1_3_PARTIAL_EVIDENCE` | Support-NELBO improved utility on matched support/eval subsets but alignment or controls blocked a full routing claim. |
| `D1_3_1_WEAK_PASS` | Locked diagnostic improved some gates but remained limited by competitive shuffled-support control. |
| `D1_4_DIAGNOSTIC_ONLY` | Reliability-only sparse source selection was not adoption-eligible. |
| `D1_5_FAIL` | Source-inner off-diagonal transfer did not improve heldout target utility and failed alignment/control gates. |
| `PAIRED_DENSE_ALL4_RELIABILITY_PASS` | Heldout-excluded source-local reliability improved dense all-source generated-embedding aggregation over equal all4 under paired generation/prediction invariants; not sparse routing. |

For `PAIRED_DENSE_ALL4_RELIABILITY_PASS`, the decision report lists
`paired_reliability_all4_shrink050_geom` as the primary method and
`paired_reliability_all4_weighted_geom` as the best reliability method. Use the
full reliability-weighted row for the reported best dense aggregation metrics.

## Implication For Thesis

Decision labels should be reported with the supporting metric rows and claim boundary.

## Limitations

Future SAIL labels are not available until outputs exist. TODO: verify against artifact.

D-series labels are verified from synced workstation artifacts under
`/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/`.

The paired dense all4 label is verified from the local synced artifact root
`cvae_rebuild/artifacts/virchow2_cvae_paired_dense_all4_reliability_confirmation_v1/`.

## Next Checks

- Add SAIL decision labels after running/syncing the report.
- Keep D-series PASS/PARTIAL/FAIL labels tied to their decision reports and do
  not promote partial or diagnostic labels to deployable claims.
