# Leakage Rules

## Purpose

State the leakage rules that protect thesis-facing claims.

## Key Claims

- Candidate expert pools must exclude the held-out target/query expert under LOQDO/LODO unless the row is diagnostic-only.
- Target support and target evaluation splits must be disjoint when support-set calibration is used.
- Target evaluation labels and metrics are final scoring only.
- Quarantined outputs must not be used for method-selection claims.

## Evidence / Source Artifacts

- `../../../PROTOCOL_STATUS.md`
- `../../../cvae_downstream_evaluation/docs/protocol.md`
- `../../../cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/reports/r12b_leakage_report.json`
- `../../../cvae_support_routing/README.md`

## Interpretation

The repository preserves invalid and diagnostic work for audit history, but thesis-facing claims must come from protocol-safe paths and valid leakage reports.

## Implication For Thesis

Every thesis-facing result should cite a leakage/provenance report or protocol status evidence.

## Limitations

Some old reports may predate current protocol labels. Treat them as historical unless validated.

## Next Checks

- Validate the SAIL leakage report when available.
- Keep quarantine references out of final method-selection tables.
