# Stage 20: CVAE Preservation

This stage tests whether decoded or sampled CVAE representations preserve a
frozen real-feature classifier surface. A passing result supports only the
declared preservation claim.

Every run must identify the real comparator, CVAE checkpoint lineage, feature
frame, fit-only transformations, scoring-only label use, preservation ratio,
negative controls, and identity-overlap audit. Preservation is not routing,
prior quality, controllable generation, or downstream synthetic utility.

Reusable implementation lives in
`src/midogpp_thesis/cvae/preservation/`.

Canonical configs are:

- `configs/preservation_sanity_v1.yaml`
- `configs/preservation_gate_pca128_v1.yaml`
- `configs/preservation_condition_audit_v1.yaml`
- `configs/tuned_classifier_preservation_v1.yaml`

They use logical artifact IDs for the dataset contract, corrected `xyxy`
cache, and their declared real-feature inputs, and write only to stage-20
canonical artifact roots. Only the tuned-classifier experiment is registered
as active; the other three are diagnostic surfaces.
