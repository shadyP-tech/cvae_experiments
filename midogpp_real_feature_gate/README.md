# MIDOG++ Real Feature Gate

Independent, contract-first scaffold for the MIDOG++ real-feature discriminative
transfer ceiling/control matrix.

This package is intentionally not a SAIL runtime dependency. SAIL remains the
reference implementation for existing MIDOG++ real-feature diagnostics, and this
package must compare its split and artifact semantics against SAIL before any
artifact is treated as thesis-facing.

## Protocol Boundary

- Real extracted features only.
- No CVAE training, routing, MoErging, synthetic generation, or candidate-surface
  selection inside this package.
- Source-only rows must exclude the held-out target from fit, preprocessing,
  thresholding, calibration, normalization, and model selection.
- Pooled and oracle rows are diagnostic controls only.
- Target labels are scoring-only for held-out evaluation diagnostics.

## Required Gate

CVAE candidate-surface work may begin only after the
`midogpp_real_feature_transfer_ceiling_v1` artifact bundle passes validation and
is reviewed.

