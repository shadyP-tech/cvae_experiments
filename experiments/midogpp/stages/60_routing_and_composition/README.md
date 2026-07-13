# Stage 60: Routing and Composition

This stage selects or composes source experts using only allowed
pre-evaluation information. Unlabeled target support must be disjoint from
target evaluation, the target expert must be excluded, and routing must not
modify source checkpoints.

The complete routing policy is frozen before stage 70. Stage 50 utilities and
stage 90 oracle identities are forbidden inputs.

Status: `PLANNED`. No routing implementation is active in the canonical
package, and no preservation or oracle artifact may serve as a routing input.
