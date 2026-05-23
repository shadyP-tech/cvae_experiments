# Current Best Approach

## Purpose

Document the current empirical synthesis and active SAIL implementation.

## Pages

- [Current synthesis](current-synthesis.md)
- [R1.2c-V lineage](r12c-v-plan.md)
- [Virchow2-only rationale](virchow2-only-rationale.md)
- [Cross-backbone audit](cross-backbone-audit.md)
- [Rebuild gates](rebuild-gates.md)
- [Next experiment sequence](next-experiment-sequence.md)

## Key Claims

- SAIL is the active current method: Source-only Aggregation via Inner-domain Leaveout.
- Virchow2 is the current backbone instantiation, not the method name.
- SAIL tests whether source-only dense real-feature aggregation is stable enough to justify a later CVAE preservation test.
- SAIL does not prove CVAE generation or metadata routing.

## Evidence / Source Artifacts

- `../../context/current_experimental_state.md`
- `../../../sail/configs/sail_virchow2.yaml`
- `../../../sail/src/sail/`
- `../../../cvae_downstream_evaluation/legacy/superseded_by_sail/r12c/README.md`
