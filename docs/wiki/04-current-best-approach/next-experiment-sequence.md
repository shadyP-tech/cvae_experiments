# Next Experiment Sequence

## Purpose

Record the current ordered next steps after the SAIL extraction.

## Key Claims

1. SAIL: Virchow2 dense source-selected config aggregation.
2. Verify SAIL leakage and decision artifacts.
3. If SAIL passes: R1.3a Vanilla Virchow2 CVAE Rebuild.
4. If vanilla Virchow2 CVAE preserves most real-feature utility: keep CVAE simple and focus thesis contribution on compatibility/routing.
5. If vanilla Virchow2 CVAE loses too much utility: calibrated or composable Virchow2 CVAE becomes the justified novel contribution.

## Evidence / Source Artifacts

- `../../context/current_experimental_state.md`
- `../../../sail/configs/sail_virchow2.yaml`

## Interpretation

The sequence deliberately separates real-feature transfer stability from CVAE generation preservation.

## Implication For Thesis

The thesis should avoid treating SAIL as a CVAE result. Its job is to decide whether a Virchow2 CVAE preservation test is justified.

## Limitations

`R1.3a Vanilla Virchow2 CVAE Rebuild` is currently provided synthesis. TODO: create or verify a corresponding config/artifact after SAIL passes.

## Next Checks

- Run or sync SAIL.
- If pass, design the simplest vanilla Virchow2 CVAE preservation test before adding compositional complexity.
