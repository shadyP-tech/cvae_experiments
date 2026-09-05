# Safe winner router v19

Dedicated HARP v19 science: exact directional composites, safe-benefit-minus-
damage candidate selection, and an independently nested winner harm gate.
The experiment is planned and unauthorized; importing or testing this package
does not issue source-label access, an execution amendment, or a lease.
V17/v18 remain exhausted and are not runtime dependencies.

The public entrypoint is `fit_source_router`; `stacked_fitting.py` remains a
small facade. Responsibilities are separated as follows:

| Layer | Modules |
|---|---|
| Menus and exact executed recipes | `contracts.py`, `composition.py` |
| Features and coherent S/H/O outcome estimates | `features.py`, `estimators.py`, `outcome_model.py` |
| Pairwise ranking and candidate stacking | `modeling.py`, `proposer.py` |
| Complete winner nesting and gate | `winner_records.py`, `winner_gate.py`, `learning.py` |
| Winner, abstention, and signed decision transcripts | `candidate_prediction.py`, `decision_evidence.py`, `records.py` |
| Source crossfit and policy admission | `splitting.py`, `crossfit.py`, `aligned_metrics.py`, `truth.py`, `admission.py` |
| Detailed source evidence and public policy | `frontier.py`, `frontier_joins.py`, `policy.py` |
| Scoped performance and fresh attestation | `fit_cache.py`, `model_integrity.py` |

Choose the unique eligible hard-changing nonbaseline action with highest
`s = p_safe*m_safe - p_harm*m_harm`, using the fixed arm-ID tie rule. For an
admitted enabled policy, route it **if and only if** `s > 0` and
`1 - winner_harm_probability >= tau`; otherwise return exact B. A gate veto
never selects a runner-up. The score is a model estimate, not a lower
confidence bound.

`FitComplete(S)` trains its gate on unthresholded winners from
`FitProposer(S minus W)`. Each proposer separately stacks its donor ranker
before candidate-outcome fitting. Outer/inner folds rebuild the complete
procedure; full-scope class normalization retains empty-action cases. Raw
labels stay inside nonserializable source capabilities. Probability-only
candidates are excluded from routing-head fitting but retained in diagnostics.

Workstation physical generation/classification is owned by
`runtime/harp_v19_execution`; lifecycle authority is owned by
`diagnostics/fixed_bank_harp_router_v19`. Nested science uses one parent process
and one BLAS thread, with bounded execution-local raw-feature and exact-scope
fit caches. It adds no physical classifier or GPU fits. An easy synthetic
216-case Mac benchmark took 522.16 seconds and routed all cases; it is solely
construction/performance evidence, not a workstation or MIDOG++ result.

The full experiment document is
`docs/wiki/03-experiments/midogpp-harp-v19-safe-winner-router.md` at repository
root. It records formulas, estimand, approximate bootstrap boundaries,
replayable winner transcripts, safe inspection commands, and remaining
scientific failure conditions. All eventual consumed-test results remain
terminal diagnostics and cannot feed Stage 60/70 or another experiment.
