# Uniform-B canonical-reference adoption boundary

The reviewed Stage-10 Uniform-B reference is complete and independently
validated.

## Canonical identities

- Feature cache:
  `midogpp_virchow2_uniform_b_canonical_train_cache_seed42`
- Reference:
  `midogpp_output_uniform_b_canonical_reference_v1`
- Experiment:
  `midogpp.real_feature.uniform_b_canonical_reference.v1`
- Representation:
  `annotation_jpeg_fixed_center_b_v3`
- Feature dimension: `3840`

Canonical A remains registered and immutable. “Canonical B” means a new
versioned reference, not an in-place replacement.

## What the review authorized

The review authorized B as a real-feature representation based on the passing
Phase-B prospective within-center confirmation. It did not authorize a router,
CVAE recipe, generated-data claim, unseen-center claim, or external-dataset
claim.

The Phase-B test split is consumed for representation adoption. Later work may
use it for descriptive scoring of already locked systems, but may not describe
it as fresh representation-selection or confirmation evidence.

## Downstream migration checklist

A later experiment adopts B only when all of the following are explicit:

1. replace the A feature-cache artifact input with the canonical B cache ID;
2. replace the A tuned-reference artifact input with the canonical B reference
   ID;
3. change the expected feature dimension from `2560` to `3840`;
4. update any CVAE input/output layer dimensions and checkpoint identities;
5. create new versioned outputs rather than overwriting A-based artifacts;
6. retain the test-consumption statement in the frozen protocol;
7. rerun source-inner recipe selection where the representation affects the
   model or objective;
8. validate the new bundle before allowing Stage-30 or later consumption.

No existing downstream registry entry was changed by the promotion experiment.
