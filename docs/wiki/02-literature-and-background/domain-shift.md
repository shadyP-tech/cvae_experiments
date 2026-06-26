# Domain Shift

## Purpose

Summarize the domain-shift settings used to test routing and aggregation.

## Key Claims

- BreakHis stresses magnification-domain shift.
- Camelyon17 stresses center/site shift.
- MIDOG-style scanner/lab/stain shifts are useful stress tests when artifacts are available.
- Weak-center behavior is a primary current failure mode.

## Evidence / Source Artifacts

- `../../../cvae_testing/README_BREAKHIS_EXPERIMENT.md`
- `../../../cvae_support_routing/artifacts/comparison_tables/breakhis_support_estimated_utility_routing_v1.md`
- `../../../cvae_support_routing/artifacts/comparison_tables/camelyon17_support_estimated_utility_routing_v2.md`
- `../../../legacy/midogpp_limited_domain_support_routing/cvae_testing/results/comparison_tables/midogpp_scanner_patch_support_estimated_utility_routing_v1.md` (legacy; not thesis-facing)

## Interpretation

The thesis should avoid claiming robustness across all medical domain shifts from one dataset. BreakHis magnification shift is narrower than hospital, scanner, staining, lab, or patient-population shift.

## Implication For Thesis

Domain-specific failure analysis should remain part of the evidence, especially for weak-center behavior in Camelyon17 and metadata-rich stress tests.

## Limitations

This page indexes known artifact surfaces. The old MIDOG++ scanner support-routing tables are archived as legacy because they used a limited-domain, partial-sample setup and should not be used as thesis-facing evidence.

## Next Checks

- Add dataset-specific claim boundaries to experiment result pages.
- Verify weak-center behavior for each final selected method.
