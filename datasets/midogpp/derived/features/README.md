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

The current planned v3 representation pilot reserves one atomic B/C parent:

```text
virchow2/physical_multiscale_clipped_bbox_annotation_local_pooling_pilot_v3/seed42/
  b_3840/
  c_11520/
```

The first child is the 3,840-dimensional B bridge on the exact canonical JPEG
crop.
Its first 2,560 dimensions must numerically reproduce canonical A within the
frozen cosine and relative-L2 tolerances, while an exact canonical-A tensor
copy is retained per center for independent verification. The newly extracted
prefix must also replay the frozen matched-reference task with prediction
agreement at least `0.999` and absolute equal-center mean BACC delta at most
`0.001`, using the exact per-center reference classifiers without reselection.
The cache report binds SHA-256 values for the canonical-v2 protocol, result,
and prediction tables used by that gate. The second child is the
11,520-dimensional C frame from frozen 28, 56, and 112 micrometer raw-TIFF
crops, deterministic in-bounds translation, clipped-bbox centroids, and
annotation-relative 4x4 token windows. Both children must align exactly to the
v3 train-only contract, exclude center 4, validate together, and publish
through one parent-directory rename. Neither cache exists yet. The failed v1
and v2 roots are not eligible substitutes.

Every cache must include or reference a report containing the dataset contract
hash, manifest hash, backbone/model reference, feature dimension, split counts,
seed, row-alignment audit, and geometry-sensitive provenance. Cache validation
must reject the deprecated `coco_xywh` lineage.

Generated tensors are ignored by Git. The corrected active cache and local
historical caches were moved into this ownership tree with matching pre/post
content fingerprints. The active cache is present locally and on the
workstation; embedded original paths remain provenance strings only.
