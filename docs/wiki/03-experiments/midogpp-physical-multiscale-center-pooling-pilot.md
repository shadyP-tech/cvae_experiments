# MIDOG++ Physical Multiscale Center-Pooling Pilot

Status update, 2026-07-23: this v1 plan is audit-blocked and non-runnable.
The workstation source audit found 986 of 9,648 rows above its frozen padding
cap before any contract, cache, or Stage-10 output existed. V2 is also
audit-blocked after two canonical annotation centroids fell outside the image.
Neither lineage may be repointed. The current implementation is the distinct
v3 clipped-bbox annotation-local lineage documented in
`midogpp-physical-multiscale-clipped-bbox-annotation-local-pooling-pilot-v3.md`.

## Status

Implementation complete; production inputs not built; registry status
`planned`; no experimental result.

The pilot tests the only source-only representation change currently judged
worth the cache-lineage disruption: deterministic physical multiscale,
center-aware Virchow2 features. It does not modify or reinterpret the canonical
2,560-dimensional reference and does not authorize CVAE retraining.

## Frozen representations

- A: canonical corrected-xyxy JPEG `CLS + global patch mean`, 2,560 dimensions.
- B: the exact canonical JPEG transform plus central 4x4 patch-token mean,
  3,840 dimensions.
- C: 28, 56, and 112 micrometer level-0 TIFF crops, each pooled as
  `CLS + global patch mean + central 4x4 patch-token mean`, concatenated to
  11,520 dimensions.

Virchow2 register tokens are excluded. Every required TIFF must have explicit,
plausible, mutually consistent resolution metadata, top-left orientation, and
no more than 10% padding for any crop. The B extractor bridge must reproduce A
with minimum cosine `0.99999` and maximum relative L2 `0.001`. Its newly
extracted A-prime prefix must additionally replay the frozen canonical
classifiers without reselection, with prediction agreement at least `0.999`
and absolute equal-center mean BACC delta at most `0.001`. The task bridge
report records SHA-256 values for the canonical-v2 protocol manifest, source
result table, and `y_pred` table used as comparators.

## Nested selection

For outer center `H`, only the other eight center shards are available to the
selector. Each representation uses the canonical ordered ten-spec logistic
grid. Eight source-inner center-LODO scores choose one classifier per
representation. B or C may replace A only when:

- equal-center mean BACC delta is at least `+0.02`;
- at least six of eight pseudo-target centers are strict wins;
- the worst pseudo-target delta is at least `-0.01`.

Ties prefer stronger mean, stronger worst case, more wins, lower dimension,
then representation ID. If no candidate passes, the policy is exactly A.
Across nine outer centers, the complete selector has 2,160 cells and 270
candidate summaries.

Every per-`H` decision binds the resolved protocol, all input hashes, the
ordered 30-candidate hash, the complete selector-table hash, and all three
locked classifier specs. All nine locks and their bundle hash must exist before
any outer target shard is opened. Outer fits are fresh. A must replay the
frozen matched-reference classifier hash, predictions, BACC, and macro-F1
exactly.

The completed output remains pending until an independent validator
reconstructs the selector Cartesian product, source-row identities, per-center
classifier choices, representation gates, conditional bootstrap, summaries,
and complete content index. Only that validator may promote protocol and
leakage manifests to `PASS`.

## Artifact and claim firewall

Target evaluation labels are used only for locked outer scoring and explicitly
posthoc, non-adoptive candidate rows. Posthoc rows cannot create or modify a
decision lock and are excluded from every decision hash. Bootstrap intervals
are paired evaluation-case intervals conditional on fixed fits, observed
centers, and locked selections; they do not cover training, selection, or
new-center uncertainty.

The maximum allowed claim is diagnostic performance of the complete nested
adaptive pipeline. The contract, B/C caches, and output are catalog-blocked
from Stage 20 through Stage 70 and from recipe or deployable selection.

## Activation checklist

1. Resolve the Virchow2 model revision and expected checkpoint SHA-256.
2. Run the workstation TIFF audit.
3. Build the immutable physical contract.
4. Build the center-sharded B/C caches.
5. Independently validate contract order, A/B/C identity, dimensions, bridge,
   center-4 exclusion, and reports.
6. Promote expected file hashes in the catalog.
7. Perform only the reviewed registry status transition from `planned` to
   `diagnostic`.
8. Prepare and run through the workspace, then validate the complete bundle.

Until all eight steps pass, the correct result statement is: implemented,
blocked before production execution, no metric and no claim.
