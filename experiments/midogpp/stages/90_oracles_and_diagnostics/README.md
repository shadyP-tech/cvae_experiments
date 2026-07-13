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
