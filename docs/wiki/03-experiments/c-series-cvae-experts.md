# C-Series CVAE Experts

## Purpose

Summarize CVAE generator and dense aggregation experiments.

## Key Claims

- C-series work investigates whether frozen CVAE expert/mode components can support downstream utility.
- C6.3 is the strongest current CVAE synthesis because it reduces sparse routing risk through dense late aggregation.
- C6.3 does not learn target compatibility and does not prove Virchow2 CVAE preservation.
- Later Virchow2 CVAE rebuild artifacts show that decoder/source-pool capacity
  can be useful, but prior sampling and decentralized composition remain the
  bottlenecks.
- The paired dense-all4 reliability confirmation is the strongest current
  generated-embedding dense aggregation result, but it is not sparse routing.

## Evidence / Source Artifacts

- `../../../cvae_downstream_evaluation/docs/c63_conceptual_synthesis.md`
- `../../../cvae_downstream_evaluation/docs/thesis_alignment.md`
- `../../../cvae_downstream_evaluation/docs/protocol.md`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_preservation_repair_v1/reports/decision_summary.md`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_source_union_gmm_prior_v1/tables/gmm_prior_gap_summary.csv`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_decentralized_reliability_weighted_gmm_prior_v1/tables/decentralized_reliability_summary.csv`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_paired_dense_all4_reliability_confirmation_v1/reports/decision_summary.md`

## Interpretation

C6.3 supports the general principle that dense late aggregation can reduce
routing regret when sparse top-1 expert/mode selection is brittle. That
principle informs SAIL, but the surfaces differ: C6.3 aggregates frozen CVAE
component classifiers, while SAIL aggregates real-feature Virchow2 classifier
configs.

The later Virchow2 CVAE rebuild sequence adds a generated-embedding lesson:

```text
vanilla prior failed
-> decode(mu) / source-pool capacity partially rescued utility
-> source-union K16 GMM diagnosed the prior bottleneck
-> decentralized reliability-weighted summaries partially preserved utility
-> paired dense-all4 reliability weighting passed as dense aggregation
```

The strongest source-union K16 row is a centralized diagnostic, not a
deployable decentralized method.

## Implication For Thesis

The thesis should distinguish dense aggregation as a strategy from the specific
generative claim. SAIL can justify a CVAE preservation test only if the
Virchow2 real-feature gate passes. D-series generated-embedding results can
support a narrower claim about data-minimizing source-local summary composition
and dense all-source reliability weighting, but not a final target-conditioned
sparse routing claim.

## Limitations

The C6.3 numeric synthesis is marked as an existing synthesis note. Verify
raw/synced decision artifacts before final thesis tables.

D-series artifacts are synced under this working repo's
`cvae_rebuild/artifacts/` directory.

## Next Checks

- Locate or sync the C6.3 final decision artifacts.
- Compare generated Virchow2 utility against SAIL real-feature utility if SAIL
  artifacts become available.
- Use [D-series decentralized Virchow2 CVAE composition](d-series-decentralized-cvae.md)
  for the current generated-embedding evidence record.
