# MIDOG++ consumed-test fixed-bank decision audit

Experiment:
`midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_decision_audit.v1`

Status: implemented terminal Stage-90 diagnostic; no canonical workstation result
is claimed by this checkout until its complete closed-world bundle validates.

## Question

The earlier case-aware screen used strict `H/q/e` exclusion. That estimand asks
whether a model can generalize to an unseen candidate expert, even though the
deployed expert bank is frozen and all candidate identities are known. It also
allowed each candidate-specific fold to acquire a different intercept, so its
nominal null was not a tied no-information control.

This sibling asks a narrower retrospective question: can response history for a
known, immutable bank be combined with label-free local support features to
improve exact candidate choice over a faithful global source-quality prior?
It is not an unseen-expert-transfer experiment and it does not emit a policy.

## Frozen design

- Inputs are the frozen bank, GenerationLock, dedicated label-free/post-seal
  aliases over the already-consumed MIDOG++ test bytes, immutable metadata, the
  original ledger through an experiment-fenced alias, and a hash-chained ledger
  amendment authorizing exactly this additional terminal diagnostic.
- Eight whole support cases per center are fixed under seed `20260809`; the
  remaining 146 cases are evaluation-only. Support and evaluation case/row IDs
  are disjoint and support labels never open.
- All nine seed-cell probabilities are materialized. Label-free case-aware
  features and the complete prediction surface are sealed before evaluation
  labels open.
- Each held-out `(H,q)` uses one common 210-row training set. Any row containing
  `H` or `q` in its outer-target, pseudoquery, or candidate-source role is
  excluded. Legal history for the known candidate `e` is retained.
- The exact endpoint is the BACC change after averaging all nine probability
  vectors and thresholding once. Exact models compare a tied null, faithful
  source prior `G`, rich/shift/boundary `R` arms, and blocked permutations.
- The old pooled row-weighted shift is retained as a non-eligible control, so
  any apparent benefit from equal-case aggregation is measured rather than
  assumed.
- Metadata similarity is persisted for provenance and descriptive inspection,
  but it is deliberately absent from every exact and smooth model family.
- `case_balanced_rich_exact` is the single predeclared primary `R` arm. Shift
  and boundary arms are descriptive challengers and cannot replace it after
  looking at this consumed result.
- Smooth BACC is fit only in isolated descriptive models. It shares no
  coefficients, thresholds, family selection, gate inputs, abstention fields,
  or publication decision with the exact models.
- The primary report includes selected exact gain versus `B`, paired `R-G`,
  top-1/tie-aware accuracy, rank correlation, pairwise accuracy, normalized
  regret, regret relative to `G`, and source-selection concentration. Failure
  of any predeclared exact gate records exact-`B` abstention.

## Claim boundary

The output is always `EXPLORATORY_CONSUMED_DATA_ONLY`. A positive screen would
motivate a newly versioned model on genuinely new MIDOG++ cases; it cannot
authorize an action, update a policy, establish routing quality, feed Stage 60
or 70 (or another Stage-90 experiment), select a recipe, support promotion, or
support deployment. The fresh B/U/G/S confirmation path remains blocked until
new, case-disjoint evidence exists.

## Workstation profile

The run keeps the established bounded topology: one persistent source worker
on each A5000 (`cuda:0`, `cuda:1`), a CUDA-free parent, no AMP/TF32, followed by
four spawned classifier workers with three BLAS threads each. GPU and CPU phases
do not overlap. Float32 memmaps store generated/prediction arrays and float64 is
used for scientific reductions. Hash-validated checkpoints use
`/data/local/fixed_bank_decision_audit_v1` before canonical persistence.

Canonical command after workspace preparation:

```bash
python -m midogpp_thesis workspace run \
  midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_decision_audit.v1
```
