# MIDOG++ Dataset Ownership

This subtree owns all MIDOG++ data locations before an experiment consumes
them:

```text
raw/MIDOGpp/                         local source data, ignored by Git
configs/annotation_patch_v1.yaml     active xyxy contract build
configs/quarantine/                  rejected geometry/config provenance
contract/annotation_patch_v1/        frozen CSV/JSON contract and local patches
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

Raw data stays on the workstation. Patch pixels and feature tensors are present
locally and on the workstation but remain ignored by Git. Small contract
tables, schemas, configs, relocation metadata, and lineage documentation are
tracked.
