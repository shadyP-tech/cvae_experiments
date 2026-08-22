# Thesis experiments v4 — supervisor dossier index

This directory decomposes the current thesis synthesis into twelve experiment-level dossiers. Each folder contains:

- `report.md`: detailed experimental purpose, protocol, results, interpretation, thesis-objective alignment, and claim boundary;
- `data/results.csv`: the compact data surface used for the principal result plot;
- `slides/*.pptx`: a four-slide, supervisor-facing experiment deck;
- `plots/result_overview.png`: the rendered principal result slide used by the Markdown report;
- `qa/`: rendered slides, montage, layout inspection output, and overflow-test output.

## Reading order

| # | Experiment | Evidence role | Primary conclusion |
| ---: | --- | --- | --- |
| 00 | [BreakHis v3 bridge](00_breakhis_v3_bridge/report.md) | Prior-meeting context | Dense composition was more reliable than sparse proxy routing. |
| 01 | [Real-feature classifier reference](01_real_feature_classifier_reference/report.md) | Stage 10 thesis-facing reference | Source-inner classifier selection raises the real-feature denominator to `0.740312` BACC. |
| 02 | [Uniform-B prospective representation confirmation](02_uniform_b_prospective_representation/report.md) | Stage 90, within-center confirmation | Representation B improves A on new cases in all nine observed centers. |
| 03 | [CVAE preservation](03_cvae_preservation/report.md) | Stage 20 thesis-facing mechanism result | Decode/posterior preserve useful structure; naive prior sampling is the loss point. |
| 04 | [Aggregate-posterior prior recovery](04_aggregate_posterior_prior_recovery/report.md) | Stage 20 source-inner mechanism result | The aggregate prior nearly closes the gap to posterior sampling. |
| 05 | [Routing-authorized expert bank](05_routing_authorized_expert_bank/report.md) | Stage 30 construction/promotion result | All 27 independent experts pass the bank-promotion gates. |
| 06 | [Generation and policy locks](06_generation_and_policy_locks/report.md) | Stages 40/60 readiness result | Policies are reproducibly frozen; the utility gate abstains in all nine folds. |
| 07 | [Frozen-policy downstream comparison](07_stage70_frozen_policy_comparison/report.md) | Stage 70 descriptive downstream result | Equal union beats metadata routing; the learned policy exactly falls back. |
| 08 | [Exact-tail router diagnostic](08_exact_tail_router_diagnostic/report.md) | Stage 90 terminal diagnosis | Oracle headroom exists, but the router does not rank actions reliably. |
| 09 | [PSSCUR simultaneous-shift diagnostic](09_psscur_simultaneous_shift_diagnostic/report.md) | Stage 90 terminal diagnosis | The calibrated envelope authorizes zero routes and the information gate fails. |
| 10 | [PCSI-RACR route-scoped diagnostic](10_pcsi_racr_route_scoped_diagnostic/report.md) | Stage 90 terminal sensitivity | Removing the envelope activates cases, but the protected primary remains zero-route. |
| 11 | [CBPUPR prefix diagnostic](11_cbpupr_prefix_diagnostic_failed_validation/report.md) | Stage 90 failed validation bundle | Provisional small gains are scientifically invalid because final lineage validation failed. |

## Evidence hierarchy

The folders intentionally do not flatten all results into one confidence level:

1. **Thesis-facing component evidence:** Stage 10 reference and Stage 20 preservation.
2. **Construction/readiness evidence:** Stage 30 bank and Stages 40/60 locks.
3. **Descriptive downstream evidence:** completed Stage 70 comparison on a previously consumed test surface.
4. **Terminal diagnostics:** Stage 90 experiments that diagnose routing mechanics but may not promote a policy or feed later experiments.
5. **Failed validation:** CBPUPR is documented for debugging and methodological learning only.

The central supervisor-facing synthesis is therefore: the representation, generator, bank, and dense composition are viable; proxy-to-utility transport is the unresolved bottleneck; no current result supports fresh routing superiority, new-center generalization, deployment, or formal differential privacy.

