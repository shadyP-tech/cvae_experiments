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

## Synced Gate Result

Latest verified artifact bundle:

```text
midogpp_real_feature_gate/artifacts/midogpp_real_feature_gate_v1/
```

Primary files:

- `tables/matrix.csv`
- `tables/predictions.csv`
- `tables/source_only_ranking_gap.csv`
- `tables/source_vs_pooled_delta.csv`
- `tables/worst_domain_summary.csv`
- `manifests/protocol_manifest.json`
- `reports/leakage_provenance_report.json`
- `reports/decision_report.md`

Validation status:

- schema: `midogpp_real_feature_transfer_ceiling_v1`
- leakage/provenance report: `PASS`
- source-only eligible held-out centers: `0,1,2,3,5,6,7,8,9`
- valid source-only rows: `9/9`
- center `4`: quarantine-only diagnostic row, not adoption-eligible
- decision report labels: `GO_REAL_FEATURE_GATE_PASSED`,
  `CLAIM_SCOPE_REAL_FEATURE_TRANSFER_ONLY`

Source-only real-feature transfer over eligible held-out centers:

- mean BACC: `0.668`
- mean macro-F1: `0.662`
- mean AUROC: `0.728`
- mean PR-AUC: `0.737`
- worst eligible center: center `2`, BACC `0.587`, AUROC `0.629`
- best eligible center: center `7`, BACC `0.735`, AUROC `0.809`

Pooled diagnostic ceiling over the same eligible centers:

- mean BACC: `0.902`
- mean macro-F1: `0.902`
- mean AUROC: `0.964`
- mean PR-AUC: `0.969`

Interpretation:

The gate supports a `WEAK PASS` for real-feature transfer feasibility: real
Virchow2 MIDOG++ features carry above-chance held-out-center signal across all
eligible centers, while the pooled diagnostic ceiling shows large remaining
headroom. This makes CVAE candidate-surface exploration defensible as
exploratory follow-up.

Claim boundary:

This result does not prove CVAE preservation, NELBO compatibility, routing
quality, synthetic downstream utility, or generative quality. Pooled rows are
diagnostic-only and may not select CVAE candidates, thresholds, generation
settings, classifier settings, or routing methods.

Missing evidence before stronger thesis-facing interpretation:

- separate negative-control table/report for this exact gate bundle
- bootstrap confidence intervals or seed-stability artifacts
- optional semantic comparison against the SAIL MIDOG++ real-feature diagnostic
  outputs

## Current SAIL Reference Status

The current reference implementation is the SAIL signal-control diagnostic under:

```text
sail/artifacts/midogpp_virchow2_real_feature_signal_controls/
```

After syncing the workstation artifacts, the MIDOG++ manifest and real Virchow2
train cache passed a direct row-level alignment audit:

- manifest: `datasets/midogpp/artifacts/midogpp_annotation_patch_v1/manifest.csv`
- cache: `sail/artifacts/pathology_embeddings/midogpp/virchow2/seed42/embeddings/train.pt`
- embedding shape: `9886 x 2560`
- rowwise metadata mismatches: `0`
- checked fields: `sample_id`, `label`, `split`, `center`, `magnification`
- train-label counts: `5364` mitotic positive rows and `4522`
  hard-negative/non-mitotic rows

The synced signal-control summaries show real Virchow2 class signal rather than
the earlier stale near-`0.51` BACC concern:

- pooled logistic BACC: `0.6914`
- pooled MLP BACC: `0.7146`
- tumor-balanced logistic BACC: `0.6349`
- negative controls: near chance (`0.4939` feature-label row shuffle, `0.4926`
  label permutation)

This remains diagnostic real-feature evidence. It does not by itself validate
CVAE preservation, GMM sampling, composition, routing, or controllable
class-conditional generation.
