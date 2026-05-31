# Pivot Statement

The thesis has pivoted from metadata-first routing to compatibility-driven routing and aggregation.

Metadata remains a baseline, proxy, and interpretability signal. The current empirical path is source-only utility selection in pathology foundation feature spaces, especially Virchow2, followed by dense config aggregation as a diagnostic gate.

SAIL (Source-only Aggregation via Inner-domain Leaveout) now names the current source-only dense aggregation method. Its active extracted implementation lives under `sail/`, with `sail/configs/sail_virchow2.yaml` as the current Virchow2 instantiation.

SAIL tests whether Virchow2 real-feature transfer is stable enough to justify CVAE rebuilding. R1.2c-V is now only the lineage name for the archived pre-extraction implementation. SAIL does not solve CVAE routing.

The later Virchow2 CVAE rebuild and D-series results add a second pivot inside
the generative surface: the bottleneck is no longer just whether Virchow2 is a
useful embedding space, but whether latent priors and decentralized
source-local summaries can preserve that utility without pooled source fitting.
The current generated-embedding evidence supports heldout-excluded
source-local reliability weighting as a dense aggregation compatibility proxy
under paired generation and prediction invariants. This is a dense all-source
aggregation result, not sparse expert selection; support-NELBO and source-inner
transfer remain diagnostic or negative.

The newest component-union audits add a third refinement: generated-embedding
mean utility can be high under source-local component composition and random
mass-bagging, but source-mass allocation is underidentified because matched
controls are competitive. The active generated-embedding bottleneck is now
weak-center/tail robustness and harmful source interaction, not another
mean-BACC-only allocator.

Fast-changing result details belong in [current_experimental_state.md](current_experimental_state.md). Stable framing belongs in [thesis_project_context.md](thesis_project_context.md).
