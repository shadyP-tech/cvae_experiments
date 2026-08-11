# Fixed-bank disagreement-regret prediction-only diagnostic

This Stage-90 adapter fits fixed-capacity `G`, `R`, and matched-permutation `P`
pairwise-regret models from post-hoc source-train OOF evidence, then emits
unscored label-free route suggestions for the complete consumed MIDOG++ test
cache. It is deliberately not a routing policy or routing-success experiment.

The prelabel classifier boundary contains two independently sealed banks:

- a strict source-OOF bank with 324 unordered `{H,q}` tasks, 5,184 physical
  classifier fits, and 10,368 oriented prediction cells; every composition
  excludes both outer target `H` and prediction query `q`;
- a target-compatible bank with 1,458 classifiers for `q=H` inference.

The strict source geometry restores the original per-class effective action
mass using label-independent logistic-fit weights (`B/U=8/7`, `A0=9/8`,
`A1=72/65`); the scaler remains unweighted. Exact unordered-pair reuse makes
the two orientations share classifier parameters without sharing predictions.

After the composite prediction seal, train-only labels may open through the
source capability. All 54 `(H, geometry, family)` model banks and their fixed
hyperparameters are sealed before the test cache is admitted. Test inference
uses all 9,928 rows, fits no classifier or regret model, and writes case-level
candidate contrasts plus `R_raw`/simultaneous-LCB `R_safe` diagnostics.

The adapter never exposes a test-label capability and cannot compute test
BACC, accuracy, regret, utility, oracle headroom, NELBO, or downstream utility.
The output is `EXPLORATORY_CONSUMED_DATA_ONLY`, cannot authorize routing or
promotion, and cannot feed another experiment. A routing claim still requires
new predeclared whole-case/patient/slide-disjoint evidence.

Canonical workstation launch:

```bash
/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_disagreement_regret_prediction_only.v1
```
