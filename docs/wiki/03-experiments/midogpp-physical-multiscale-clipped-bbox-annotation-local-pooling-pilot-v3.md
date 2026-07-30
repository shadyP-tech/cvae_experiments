# MIDOG++ Clipped-Bbox Annotation-Local Pooling Pilot v3

## Status

Implementation, the label-blind workstation TIFF audit, immutable v3 contract,
atomic B/C cache bundle, independent validation, complete expected-file hash
promotion, and reviewed `diagnostic` activation are complete. The registered
Stage-10 experiment is runnable but has not run; no Stage-10 performance result
exists.

V1 and v2 are preserved as separate failed-audit lineages. V1 exceeded its
frozen padding cap on 986 of 9,648 rows. V2 eliminated padding but retained the
unclipped canonical bbox centroid; two such centroids were outside the source
image. Neither failed lineage was allowed to create a contract, cache, or
Stage-10 run.

## Frozen v3 representation

V3 recomputes the annotation anchor from the canonical continuous half-open
axis-aligned bbox:

```text
original bbox: [x0,x1) x [y0,y1)
clipped bbox:  [max(0,x0),min(W,x1)) x [max(0,y0),min(H,y1))
anchor:        centroid of the clipped bbox
```

All coordinates must be finite, both original and clipped areas must be
positive, and the clipped/original area ratio must be at least `0.25`.
Violation aborts the complete build; rows are never filtered. The anchor is a
bbox centroid, not an object-mask or foreground centroid.

For each `28`, `56`, and `112` micrometer field of view, the requested square
is translated to the nearest wholly in-bounds level-0 square. Crop size is not
reduced and no pixels are padded or synthesized. After a spatially identity
Virchow2 preprocessing path, the annotation-relative token start is:

```text
column = clamp(floor(16 * p_x - 2), 0, 12)
row    = clamp(floor(16 * p_y - 2), 0, 12)
```

The local 4x4 window excludes the four register tokens. C concatenates the
three ordered scale blocks; this is deterministic feature concatenation, not a
mixture, expert aggregation, router, likelihood, or generative model.
Production TIFF reads additionally pin `pyvips 3.1.1` with
`libvips 8.18.4`; those identities are validated and persisted with the
existing PyTorch, timm, Pillow, model, and preprocessing identities.

## Workstation audit evidence

The 2026-07-23 `xai-master` report is:

```text
artifacts/midogpp/90_oracles_and_diagnostics/physical_multiscale_geometry_audit/
  physical_multiscale_clipped_bbox_annotation_local_pooling_pilot_v3/
  2026-07-23_xai_master/source_geometry_audit.json
```

The report records:

- status `PASS`;
- 9,648/9,648 eligible-train rows and 216/216 TIFFs;
- 84 clipped bboxes with no geometry-driven row exclusion;
- clipped-area fraction minimum `0.40`, median `0.81`, mean `0.80`, and maximum
  `0.98`, against the frozen `0.25` floor;
- 40 clipped label-0 rows and 44 clipped label-1 rows, reported descriptively
  only;
- two original bbox centroids outside the image:
  `305__305__ann15363__y1` maps from `(-2, 2416)` to `(11.5, 2416)`, and
  `309__309__ann15728__y0` maps from `(4211, -5)` to `(4211, 10)`;
- maximum absolute anchor correction `15` pixels;
- 1,286 samples requiring at least one in-bounds crop shift;
- maximum axis shift `230` pixels and maximum side-relative shift
  `0.4773755656`;
- zero padding and zero synthesized pixels.

The center and label breakdowns are audit diagnostics only. They do not affect
inclusion, geometry, selection, activation, or claims.

## Stage-10 protocol

The immutable candidate pool is:

- canonical A, 2,560 dimensions;
- v3 canonical-JPEG B with fixed-center token pooling, 3,840 dimensions;
- complete v3 clipped-bbox-anchor, shifted-in-bounds, multiscale
  annotation-local C, 11,520 dimensions;
- the unchanged ten-spec classifier grid.

The frozen 30-candidate hash is `2f651b2f8bd53c1a`. For each outer center, the
other eight centers supply the source-inner selector. The selector metric is
equal-center mean BACC; AUROC remains descriptive. A non-A candidate must have
mean BACC delta at least `+0.02`, strict wins on at least six of eight inner
centers, and worst delta at least `-0.01`; otherwise the lock falls back to A.
All nine locks must be written before outer evaluation.

Any future result supports only the complete deterministic
representation-plus-classifier pipeline. It cannot isolate gains from bbox
clipping, crop translation, physical scale, token pooling, or the classifier.
It cannot support CVAE preservation, NELBO validity, calibration, routing,
expert selection, generation, deployment, new-center confirmation, or
downstream synthetic utility. The catalog forbids Stage-20-through-70 reuse.

## Execution handoff

The completed activation sequence was:

1. build and independently validate the immutable v3 contract;
2. build B and C under one staging parent and validate canonical-A replay,
   model/runtime identity, row alignment, token starts, and the content index;
3. promote expected hashes in the artifact catalog;
4. perform a reviewed transition from `planned` to `diagnostic`, with
   lifecycle-test and current-state documentation alignment.

Run the non-adoptive diagnostic once from the workstation repository root:

```bash
/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run \
  midogpp.real_feature.physical_multiscale_clipped_bbox_annotation_local_pooling_pilot.v3
```

Do not add `--force` for the first run. `workspace run` performs the frozen
prepare step itself. The resulting output remains diagnostic and cannot revise
the representation, frozen reference, gates, or any Stage-20-through-70
decision.
