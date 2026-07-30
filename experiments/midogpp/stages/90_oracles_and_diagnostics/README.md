# Stage 90: Oracles and Diagnostics

This stage contains post-hoc oracle upper bounds, fidelity analyses, rejected
lineage audits, and other non-deployable diagnostics.

Every oracle row must be marked non-deployable. Diagnostic results may explain
a failure or quantify headroom, but they must not tune or select a deployable
policy and must remain separate from held-out utility claims.

The rejected subtree also preserves evidence from the old limited-domain
MIDOG++ scanner support-routing surface. That surface did not cover the current
full dataset contract, so its top-1, rank, and oracle-gap values are not
thesis-facing and must be regenerated before any routing claim.

The registered Uniform-B training-stability diagnostic is implemented as two
ordered experiments:

```text
midogpp.oracle.uniform_b_paired_reparameterization_snapshot.v1
midogpp.oracle.uniform_b_paired_reparameterization_audit.v1
```

The first builds a portable, hash-promoted snapshot from the canonical
contract and canonical-B train cache. The second runs exactly 12 legacy replay
cells and 12 controlled schedule/epsilon pairs, comparing one-epsilon and
antithetic reconstruction estimates while retaining three distinct
initialization seeds. Both outputs are `AUDIT_ONLY`, are currently absent, and
cannot update a Variant-B recipe or feed any non-diagnostic stage.
