# HARP v20: patch evidence and risk-aligned action selection

The public entrypoint is `fit_source_router`. This dedicated successor retains
exact B/U/D01_ONLY/D10_ONLY/BOTH actions and adds case-excluded patch class
evidence, risk penalties before winner selection, and nested selection of the
penalty scale and winner-gate threshold. V17–v19 are exhausted historical
experiments, not runtime dependencies.

| Responsibility | Modules |
|---|---|
| Typed menus, immutable probabilities and branch recipes | `contracts.py`, `composition.py` |
| Fixed Virchow2 sketch, scoped class evidence and honest predictions | `patch_evidence.py` |
| Actual-composite descriptors and bounded raw-feature cache | `features.py` |
| Signed gain, harm and proper-loss outcome estimates | `outcome_model.py`, `outcome_targets.py`, `estimators.py` |
| Pairwise donor ranking and honest candidate stacking | `modeling.py`, `proposer.py` |
| Risk objective before choosing the winner | `risk_selection.py` |
| Complete winner selection, held-out gate fitting | `learning.py`, `winner_gate.py`, `winner_records.py` |
| Action decision and exact-B fallback | `candidate_prediction.py`, `decision_evidence.py`, `records.py` |
| Nested selection, class normalization and source admission | `crossfit.py`, `splitting.py`, `aligned_metrics.py`, `truth.py`, `admission.py` |
| Threshold frontiers, source-only controls and outcome joins | `frontier.py`, `frontier_joins.py` |
| Bounded fit reuse, public policy and model attestation | `fit_cache.py`, `policy.py`, `model_integrity.py` |

For each structurally eligible hard-changing action, select the largest

`J = predicted_gain - scale * [0.05*(predicted_harm-0.25) + (predicted_brier_delta-0.002) + 0.25*(predicted_logloss_delta-0.005)]`.

The scale is chosen from `{0.5, 1, 2}` using inner source OOF predictions.
An enabled policy routes the winner only when `J > 0` and the independently
fitted winner gate satisfies `1-P(winner harm) >= threshold`. A veto returns
exact B; it never silently promotes a runner-up. No score is a safety bound.
The final source policy still must pass the unchanged risk and coverage gate.

The sketch is a fixed signed projection of the canonical 3840-dimensional
Virchow2 rows into 64 coordinates, computed before source labels open. Every
learned patch standardizer/class model excludes its held case. The action
model sees ranker-OOF composites and patch-OOF evidence from the same fold.
Complete winner-gate folds exclude their held cases from both upstream fits.

The physical executor lives in `runtime/harp_v20_execution`; lifecycle,
preparation and authority live in `diagnostics/fixed_bank_harp_router_v20`.
Truth-bearing fits stay in one process with one BLAS thread. Bounded caches
reuse only identical menus, training scopes, label capabilities and fitting
settings. Ranker/proposer fits can be reused across decision-only penalty
settings; complete winner/gate fits cannot.

See `docs/wiki/03-experiments/midogpp-harp-v20-risk-aligned-router.md` for the
protocol, formulas, inspection commands and measured validation. This is a
planned terminal consumed-test diagnostic; implementation does not issue an
execution amendment or lease and does not establish useful real routing.
