# Stage 90 Artifacts

New non-deployable oracle, fidelity, and audit diagnostics are written below
this directory.

`rejected/` contains byte-preserved historical MIDOG++ bundles whose cache,
checkpoint, dataset, or selection lineage is not eligible for current claims.
These bundles are retained for audit only. They must never feed stages 30–70,
and their original configs are stored only as provenance snapshots.

`cache_integrity/` contains provenance and geometry audits. A cache-integrity
report can validate lineage, but is not real-feature, preservation, routing, or
downstream-utility evidence by itself.

`repository_migration/2026-07-12_xai_master/` is `AUDIT_ONLY` provenance for
the canonical workstation migration. It contains matching pre/post SHA-256
manifests for the contract, corrected cache, tuned reference, and tuned
preservation bundle, plus stable raw-tree metadata and critical raw-file
hashes. It does not add a new experimental claim.
