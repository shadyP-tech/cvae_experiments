# Cross-Dataset Artifact Archive

This subtree preserves historical BreakHis, Camelyon17, and generic routing
outputs that previously lived inside capability-package directories. It is not
part of the active MIDOG++ experiment registry and must not be used as a
MIDOG++ reference, baseline, router input, or thesis decision.

The migration moved files without changing their bytes and checked file counts
and SHA-256 content fingerprints before and after each move. Original paths
inside historical manifests are retained as immutable provenance; active code
and configs resolve only canonical `datasets/midogpp/` and
`artifacts/midogpp/<stage>/` locations.

Large generated contents remain ignored by Git. Previously tracked reports and
tables remain tracked after their path move.

Historical BreakHis/Camelyon17 documentation snapshots are stored under
`artifacts/cross_dataset_archive/docs/`. They describe retired interfaces and
are retained for provenance, not as current run instructions.
