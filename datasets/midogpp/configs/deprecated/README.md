# Deprecated MIDOG++ Configs

This folder preserves rejected or historical MIDOG++ dataset configs for
provenance only. Do not use these configs for thesis-facing cache builds,
real-feature gates, CVAE preservation tests, or downstream classifier runs.

## `annotation_patch_v1_coco_xywh_stale.yaml`

This config is the old annotation-patch lineage with:

```yaml
bbox_format: coco_xywh
```

It is deprecated because the corrected MIDOG++ annotation-patch contract treats
annotation boxes as `xyxy`. The stale `coco_xywh` interpretation can preserve
row order and `sample_id` alignment while producing different patch centers and
different Virchow2 embeddings.

Known affected cache lineage:

```text
sail/artifacts/pathology_embeddings_midogpp_annotation_patch_v1/virchow2/
```

Use the active config instead:

```text
datasets/midogpp/configs/annotation_patch_v1.yaml
```

The intended rebuilt cache root is:

```text
sail/artifacts/pathology_embeddings_midogpp_annotation_patch_xyxy/virchow2/
```
