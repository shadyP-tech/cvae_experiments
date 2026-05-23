# Pivot Statement

The thesis has pivoted from metadata-first routing to compatibility-driven routing and aggregation.

Metadata remains a baseline, proxy, and interpretability signal. The current empirical path is source-only utility selection in pathology foundation feature spaces, especially Virchow2, followed by dense config aggregation as a diagnostic gate.

SAIL (Source-only Aggregation via Inner-domain Leaveout) now names the current source-only dense aggregation method. Its active extracted implementation lives under `sail/`, with `sail/configs/sail_virchow2.yaml` as the current Virchow2 instantiation.

SAIL tests whether Virchow2 real-feature transfer is stable enough to justify CVAE rebuilding. R1.2c-V is now only the lineage name for the archived pre-extraction implementation. SAIL does not solve CVAE routing.

Fast-changing result details belong in [current_experimental_state.md](current_experimental_state.md). Stable framing belongs in [thesis_project_context.md](thesis_project_context.md).
