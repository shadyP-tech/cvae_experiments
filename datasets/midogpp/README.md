# MIDOG++ Dataset Ownership

This subtree owns all MIDOG++ data locations before an experiment consumes
them:

```text
raw/MIDOGpp/                         local source data, ignored by Git
configs/annotation_patch_v1.yaml     active xyxy contract build
configs/physical_multiscale_center_pooling_pilot_v1.yaml
                                      audit-blocked historical v1
configs/physical_multiscale_annotation_local_pooling_pilot_v2.yaml
                                      audit-blocked historical v2
configs/physical_multiscale_clipped_bbox_annotation_local_pooling_pilot_v3.yaml
                                      current audited v3 build profile
configs/quarantine/                  rejected geometry/config provenance
contract/annotation_patch_v1/        frozen CSV/JSON contract and local patches
contract/physical_multiscale_clipped_bbox_annotation_local_pooling_pilot_v3/
                                      validated workstation-only v3 geometry contract
derived/features/                    Virchow2 feature caches
```

The frozen annotation-patch contract contains 22,569 rows, uses case-disjoint
train/validation/test splits, and has a `PASS` leakage report. Its hashed
manifest retains the original repository-relative patch prefix. The audited
`path_relocation.json` sidecar maps only that prefix to the canonical contract
tree; the manifest itself is not rewritten.

All 22,569 referenced patch images are present locally and on the workstation
at the canonical contract path. Local `dataset-validate` passes with 22,569
rows and nine eligible domains. The approximately 65 GB unmodified raw source
tree remains workstation-only at `raw/MIDOGpp/` and is intentionally not synced
to the Mac.

The active annotation geometry is `xyxy`. The quarantined `coco_xywh` config
documents a stale cache lineage and must not feed current experiments.

Two feature-cache lineages are deliberately separate:

- `derived/features/virchow2/annotation_patch_xyxy/seed42` is the active,
  corrected cache, present locally and on the workstation with train tensor
  SHA-256
  `f6608e513fb2d06671e3ec117b093a85d58530b77b1fae44a3be1680d9feabd2`.
- `derived/features/virchow2/historical_train_only/seed42` is diagnostic-only;
  its train tensor has a different SHA-256 and cannot be used as a fallback.

Canonical commands from the repository root after installing the package with
`conda run -n thesis python -m pip install -e .`:

```bash
conda run -n thesis python -m midogpp_thesis dataset-build \
  --config datasets/midogpp/configs/annotation_patch_v1.yaml

conda run -n thesis python -m midogpp_thesis dataset-validate \
  --artifact-root datasets/midogpp/contract/annotation_patch_v1

conda run -n thesis python -m midogpp_thesis dataset-inspect \
  --artifact-root datasets/midogpp/contract/annotation_patch_v1
```

The current physical-multiscale lineage has a separate dataset-owned v3
command surface:

```bash
conda run -n thesis python -m midogpp_thesis dataset-physical-multiscale audit-v3 \
  --config datasets/midogpp/configs/physical_multiscale_clipped_bbox_annotation_local_pooling_pilot_v3.yaml \
  --report-path artifacts/midogpp/90_oracles_and_diagnostics/physical_multiscale_geometry_audit/physical_multiscale_clipped_bbox_annotation_local_pooling_pilot_v3/2026-07-23_xai_master/source_geometry_audit.json

conda run -n thesis python -m midogpp_thesis dataset-physical-multiscale build-contract-v3 \
  --config datasets/midogpp/configs/physical_multiscale_clipped_bbox_annotation_local_pooling_pilot_v3.yaml

conda run -n thesis python -m midogpp_thesis dataset-physical-multiscale build-cache-v3 \
  --config datasets/midogpp/configs/physical_multiscale_clipped_bbox_annotation_local_pooling_pilot_v3.yaml

conda run -n thesis python -m midogpp_thesis dataset-physical-multiscale validate-v3 \
  --config datasets/midogpp/configs/physical_multiscale_clipped_bbox_annotation_local_pooling_pilot_v3.yaml
```

The 2026-07-23 `xai-master` v3 source audit passes for all 9,648 eligible-train
rows and 216 TIFFs. It records 84 mechanically clipped annotation boxes, a
minimum retained bbox-area fraction of `0.40` against the frozen `0.25` floor,
no row exclusion, no padding, and no synthesized pixels. The two original bbox
centroids outside the image are deterministically repaired to `(11.5, 2416)`
and `(4211, 10)`. The audit remains Stage-90 evidence only. The v3 contract and
atomic B/C cache bundle now exist on `xai-master`, independently pass
`validate-v3`, and have complete required-file SHA-256 coverage in the catalog.
The Stage-10 experiment is `diagnostic` and ready to run; no Stage-10 result
exists yet. V1 and v2 remain immutable failed-audit lineages and must not be
built, resumed, repointed, or used as fallbacks.

Raw data stays on the workstation. Patch pixels and feature tensors are present
locally and on the workstation but remain ignored by Git. Small contract
tables, schemas, configs, relocation metadata, and lineage documentation are
tracked.
