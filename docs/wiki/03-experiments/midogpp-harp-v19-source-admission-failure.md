# HARP v19 source admission failure

SCOPE LIMITED: post-hoc analysis of saved source-development diagnostics from
the workstation. This is not target-test evidence or a newly selected policy.
The v19 implementation completed its physical work and nested fitting, but
did not demonstrate safe real-data routing. Synthetic construction tests did
not establish that scientific result.

## Verified terminal state

The workstation lease and failure report are `FAILED_EXHAUSTED`. All 81
classifier tasks completed. Source and target physical menus and bank
independence proofs were sealed before source-label access. The source router
fit finished and the admission gate returned `NO_NONZERO_SAFE_OOF_COVERAGE`:
0/216 OOF routes, `bounds=null`, and `bootstrap_performed=false`. Target routed
actions were not constructed and test evaluation labels were not opened.

Outer folds 0–3 disabled their policies because no prescribed threshold passed
their inner-OOF constraints. Fold 4 enabled threshold 0.95 based on two inner
routes in two centers; that policy produced zero routes on its outer-held
cases. The final full-source inner selection also disabled the policy.
Consequently all actual nested outer-OOF selections were exact B.

The final 0.95 threshold is not evidence that lowering it would fix the run:
for disabled folds it is the default accompanying `policy_enabled=false`.

## What the saved diagnostic frontier reveals

These are *enabled-policy diagnostic replays* at the predefined common
thresholds, not the actual admitted nested policy. Harm and gain below use the
declared equal-center weighting; gain is an all-case aligned BACC difference.

| Gate threshold | Potential routes | Routed harm | BACC difference |
|---:|---:|---:|---:|
| 0.00 | 91 | 50.95% | +0.001485 |
| 0.50 | 46 | 51.21% | -0.000996 |
| 0.75 | 21 | 63.80% | -0.003214 |
| 0.95 | 2 | 0.00% | +0.000273 |

The harm cap is 25%. The two routes at 0.95 occur in one center, far short of
the final requirement of 18 routed cases and six centers with two routes
each. Their diagnostic moment checks passing does not satisfy final
admission, nor authorize choosing a threshold after inspecting outer results.

Among 143 cases with a nonempty winner menu, the gate's predicted mean harm
was 52.42%, compared with 54.00% observed. That average calibration conceals
weak discrimination: harm ROC AUC was 0.535 across those winners, and 0.497
among the 91 positive-score winners eligible for routing. The latter included
46 harmful winners and only 35 strictly proper-loss-safe positive winners.
AUC here is unweighted across cases; the mean-risk comparisons are
equal-center weighted and conditioned on the stated population.

Stricter scores did not consistently select safer cases. On final-source
inner OOF, threshold 0.95 retained seven cases with 83.23% center-balanced harm
and failed gain, harm, Brier, and log-loss constraints. This is a limitation of
the fitted scores on held cases, not a bootstrap or GPU bottleneck.

## Available opportunities and where they were lost

The primitive source surface has 100/216 strictly proper-loss-safe positive
opportunity cases and +0.024117 equal-center oracle BACC headroom. The actual
outer-fold candidate menus retain 88 cases and +0.021833 headroom. These are
source diagnostic oracle quantities, never selection inputs.

Of those 88 cases, the candidate score chose a strictly safe-positive winner
in 53. Eighteen of these safe winners received nonpositive signed scores, so
only 35 remained eligible after the positive-score condition. There were
35 missed safe-opportunity cases, including 22 where the selected winner was
harmful despite an available safe alternative. Only 47 of the 88 opportunity
cases had any safe candidate assigned a positive score.

Thus the directional action menu is useful, but candidate identification,
the sign of the safe-benefit estimate, and winner-level harm discrimination
all remain limiting. Oracle headroom alone does not establish that the
permitted label-free features make these outcomes predictably separable.

## Implications for a future design

Do not change or rerun the exhausted v19 identity. Lowering its threshold or
loosening the gate is unsupported by these source diagnostics. Further
calibration alone is insufficient unless held-case harm discrimination
improves; average calibration already substantially matches the observed
harm prevalence.

A separately registered successor should first test whether its candidate
and winner features improve held-case safe/harm identification over simple
predeclared baselines. Inner policy selection also needs a declared evidence
requirement for routed cases and center coverage: the current
`failed_constraints()` default permits a two-case, two-center inner policy.
That is an evidence-strength weakness, although it did not cause the large
harm rates in the other folds. Stronger evidence requirements may correctly
retain exact B; they do not create predictive signal.

## Evidence

Workstation root:
`/home/stud/spark/cvae_experiments/artifacts/midogpp/90_oracles_and_diagnostics/uniform_b_v2_consumed_test_fixed_bank_harp_router/v19`.

The audit used `reports/candidate_frontier.json`,
`reports/source_candidate_winner_joins.json`,
`reports/source_headroom_diagnostics.json`, `reports/failure_report.json`,
`reports/source_policy_nonadmission.json`,
`manifests/source_policy_admission_seal.json`, and the persisted pooled policy
manifest. All three diagnostic report hashes were recomputed successfully.
No raw labels were opened and no model was refitted during this audit.

The compact [audit data](../../validation/harp_v19_failure_audit_2026-09-05.json)
contains report SHA-256 values, all aggregate threshold rows, fold choices,
winner statistics, and the exact admission/failure payloads.

Relevant implementation:
`routing/safe_winner_router_v19/crossfit.py:179` selects valid inner policies;
`routing/safe_winner_router_v19/frontier.py:34` defines the inner moment and
center checks; `diagnostics/fixed_bank_harp_router_v19/runner.py:233` enforces
the nonadmission stop. These paths are relative to
`src/midogpp_thesis/cvae/`.
