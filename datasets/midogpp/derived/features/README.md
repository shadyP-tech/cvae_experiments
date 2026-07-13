# MIDOG++ Derived Feature Caches

This is the canonical root for new feature caches derived from the frozen
MIDOG++ dataset contract. It replaces retired package-owned cache locations
such as `sail/artifacts/pathology_embeddings_*`; those legacy locations are
historical only and should not be used for active routing or preservation runs.

Use the hierarchy:

```text
<backbone>/<contract-frame>/<experiment-seed>/
```

For the current Virchow2 setup, the expected logical location is:

```text
virchow2/annotation_patch_xyxy/seed42/
```

The local historical train-only cache is isolated at:

```text
virchow2/historical_train_only/seed42/
```

Its `train.pt` SHA-256 is
`96afc1b68b001159a37ee5a5ddf8ad104447cc8c88081fe2acb39705f71e1260`;
it is not interchangeable with the corrected active `xyxy` cache hash
`f6608e513fb2d06671e3ec117b093a85d58530b77b1fae44a3be1680d9feabd2`.
Rejected `coco_xywh` lineage is isolated under `quarantine/`.

Every cache must include or reference a report containing the dataset contract
hash, manifest hash, backbone/model reference, feature dimension, split counts,
seed, row-alignment audit, and geometry-sensitive provenance. Cache validation
must reject the deprecated `coco_xywh` lineage.

Generated tensors are ignored by Git. The corrected active cache and local
historical caches were moved into this ownership tree with matching pre/post
content fingerprints. The active cache is present locally and on the
workstation; embedded original paths remain provenance strings only.
