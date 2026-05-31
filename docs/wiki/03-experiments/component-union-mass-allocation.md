# Component-Union Mass Allocation Audits

## Purpose

Document the post-D1 component-union experiments that tested whether
source-local latent components, reliability shrinkage, support calibration,
random mass-bagging, or probability tail shields can close the generated-
embedding utility gap.

## Key Claims

- Component-level source-local GMM summaries expose a high-utility generated-
  embedding composition surface.
- High mean BACC is no longer enough for adoption, because random and shuffled
  source-mass controls often match the proposed primary methods.
- The dominant bottleneck has shifted from mean utility to source-mass
  underidentification, weak-center/tail robustness, and harmful source
  interactions.
- These experiments are dense composition and robustness audits. They do not
  prove learned routing, sparse expert selection, formal privacy, or causal
  reliability validation.

## Evidence / Source Artifacts

Verified local reports:

- `../../../cvae_rebuild/artifacts/virchow2_cvae_decentralized_component_union_reliability_shrink025_v2/reports/decision_summary.md`
- `../../../cvae_rebuild/artifacts/virchow2_cvae_source_inner_validated_dense_component_hybrid_v1/reports/decision_summary.md`
- `../../../cvae_rebuild/artifacts/virchow2_cvae_decentralized_component_union_mass_bagged_v1/reports/decision_summary.md`
- `../../../cvae_rebuild/artifacts/virchow2_cvae_decentralized_component_union_reliability_shrink050_confirmation_v1/reports/decision_summary.md`
- `../../../cvae_rebuild/artifacts/virchow2_cvae_support8_calibrated_component_union_prior_v1/reports/decision_summary.md`
- `../../../cvae_rebuild/artifacts/virchow2_cvae_component_union_tailrisk_anchored_mass_bagged_v1/reports/decision_summary.md`
- `../../../cvae_rebuild/artifacts/virchow2_cvae_dense_reliability_tailshield_random_mass_bag_v1/reports/decision_summary.md`

Implemented / running, not final evidence:

- `../../../cvae_rebuild/configs/virchow2_cvae_source_inner_harmful_source_suppression_random_mass_bag_v1.yaml`
- `../../../cvae_rebuild/src/cvae_rebuild/source_inner_harmful_source_suppression.py`

TODO: verify final harmful-source suppression reports after the workstation
rerun writes `reports/`, `tables/`, and `manifests/`.

## Result Matrix

| Artifact | Primary method | Verdict | Mean BACC | Min center | Seed std | Interpretation |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Component-union shrink025 v2 | `decentralized_component_union_reliability_shrink025` | `COMPONENT_UNION_FAIL` | 0.8892 | 0.8168 | 0.0501 | High utility, but matched shuffled reliability and source-ablation dominance block adoption. |
| Source-inner dense/component hybrid | `source_inner_validated_dense_component_binary_gate` | `HYBRID_FAIL` | 0.8103 | 0.7192 | 0.0424 | Source-inner pseudo-target gate fails to decide when component union is safe. |
| Mass-uncertainty bagged component union | `decentralized_component_union_mass_uncertainty_bagged_v1` | `MASS_BAGGED_COMPONENT_UNION_FAIL` | 0.8903 | 0.7931 | 0.0568 | High mean and K16 retention, but random mass-bag and negative controls are competitive. |
| Reliability shrink050 confirmation | `decentralized_component_union_reliability_shrink050` | `ANCHOR_MISMATCH` | 0.8800 | 0.8000 | 0.0527 | Reliability shrinkage does not separate cleanly from matched shuffled/random controls. |
| Support8 calibrated component union | `support8_calibrated_component_union_softmax_shrink050` | `SUPPORT_CALIBRATED_COMPONENT_UNION_FAIL` | 0.8727 | 0.7886 | 0.0369 | Unlabeled support-NELBO calibration does not beat shrink050/random mass-bag or shuffled-support null. |
| Shrink050/random mass tail-risk blend | `component_union_tailrisk_anchored_shrink050_random_mass_bag_blend050` | `TAILRISK_ANCHORED_COMPONENT_UNION_USEFUL_THESIS_SUCCESS` | 0.8957 | 0.8032 | 0.0510 | Useful robustness evidence, but anchor mismatch and bottom-tail limitations remain. |
| Dense reliability tail shield | `dense_reliability_tailshield_random_mass_bag_blend25_75` | `DENSE_TAILSHIELD_RANDOM_MASS_BAG_FAIL` | 0.8988 | 0.7896 | 0.0403 | High mean and bottom20 gain, but center3 and worst-cell failure remain unresolved. |

## Interpretation

The component-union line changed the project diagnosis. Earlier D-series
experiments suggested the generated-embedding bottleneck was preserving utility
after CVAE generation. The component-union and random mass-bag experiments show
that mean utility can often approach the centralized source-union K16 diagnostic
region.

The unresolved issue is not simply whether component-union has enough capacity.
It is whether any protocol-clean, pre-target-evaluation signal can allocate
source mass or suppress harmful sources better than arbitrary/randomized dense
mass variation.

The mass-bagged artifact is the clearest example:

```text
center-equal mean BACC: 0.8903
retention vs source-union K16: 0.9960
delta vs random mass-bag control: -0.0016
```

That is high utility but not method adoption. It supports a ceiling/diagnostic
claim: random mass uncertainty already occupies most of the headroom, so a new
method must improve weak-center or tail robustness, not merely mean BACC.

## Implication

The next generated-embedding question is:

```text
Can source-only diagnostics predict when a source poisons a target-like regime?
```

This is why the current implementation/run is source-inner harmful-source
suppression over random mass-bag component union. It tests source interaction
and poisoning, not another generic mass allocator.

## Limitations

- Random mass-bag performance is a control/diagnostic result, not evidence that
  random weights are meaningful compatibility estimates.
- The support8 calibrated component-union artifact used target support, so it
  is not source-only. It also failed against the matched shuffled-support null.
- Tail-risk and dense-tailshield probability blends are robustness aggregation
  audits, not learned routing.
- Harmful-source suppression is not yet documented as a result because the
  final reports are not verified locally.

## Next Checks

- Sync and validate final harmful-source suppression artifacts.
- If harmful-source suppression passes, document whether source-inner removal
  effects identify target-ablation-confirmed harmful sources.
- If it fails, distinguish source-inner signal failure from target-regime
  identification failure.
- Keep random mass-bag, shrink050, paired dense reliability, source-union K16,
  and real-feature dense references as fixed comparators.
