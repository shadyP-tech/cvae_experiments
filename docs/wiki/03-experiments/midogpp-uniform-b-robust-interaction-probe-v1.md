# MIDOG++ Uniform-B Robust Interaction Probe v1

## Status

`COMPLETE — NO_FAMILY_PASSES_ROBUST_BPLUS_GATE`

This Stage-90 diagnostic first audits the frozen standard-Nyström B+ error
exchange and then compares center×class-robust Nyström training with a
low-rank global–local bilinear model. Validation and test remain untouched.

## Regression audit

The completed paired audit contains 751 nonlinear rescues, 550 nonlinear
regressions, and 1,318 shared hard-core errors.

- 257 of the regressions are center-2 false positives.
- 24 are center-9 false negatives.
- 246/550 regressions are within probability margin 0.1 of the boundary.
- Only 2/550 are confidently wrong at probability confidence at least 0.9.
- C predictions are available for 309 regressions; C is correct for 251.
- Mean nonlinear margin is 0.138 for regressions, 0.176 for rescues, and
  0.238 for the shared hard core.

Annotation geometry is constant on the audited surface: bbox area 2,500,
aspect ratio 1, and annotation offset zero. It therefore cannot explain the
exchange. Source patch files are unavailable at the manifest-relative paths
on this workstation, so brightness/contrast/edge metrics are explicitly
missing rather than imputed.

## Robust Nyström

Source-inner selection compared:

- equal center×class weighting;
- group DRO with update rate 0.1;
- group DRO with update rate 0.5.

Selection maximized worst inner center×class recall before mean BACC. The
primary result retained most of standard B+:

| Quantity | Robust Nyström |
| --- | ---: |
| Equal-center BACC delta versus linear B | +0.020437 |
| Delta versus standard Nyström B+ | −0.002755 |
| Strict wins versus linear B | 8/9 |
| Worst-center BACC delta versus linear B | −0.004016 |

It substantially corrected center 2: specificity changed by only −3.76
points versus linear B, rather than standard B+'s −12.40 points. However,
center-9 recall fell by 12.72 points versus linear B. This violates the frozen
−5-point class-direction floor, so robust Nyström is not locked.

## Bilinear interaction model

The model splits B into global dimension 2,560 and local dimension 1,280 and
adds a rank-{4,8,16} interaction. Each fit starts from an equal-center×class
weighted linear model; the linear term is frozen while one strongly
regularized GPU epoch fits only the interaction.

The bilinear family fails clearly:

| Quantity | Bilinear |
| --- | ---: |
| Equal-center BACC delta versus linear B | −0.006267 |
| Delta versus standard Nyström B+ | −0.029458 |
| Strict wins versus linear B | 2/9 |
| Worst-center BACC delta versus linear B | −0.030120 |

Thus the Nyström gain cannot be reduced to this minimal low-rank bilinear
local–global interaction.

## Decision and implication

No final B+ classifier is frozen. Equal group weighting and group DRO move the
error burden rather than eliminating it: center-2 false positives improve,
but center-9 false negatives become the limiting group.

The appropriate next step is not another broad classifier search. Inspect the
near-boundary center-2/center-9 exchange and test a predeclared constrained
objective that places separate floors on source-inner sensitivity and
specificity. If that still merely moves the failure, retain standard Nyström
as diagnostic evidence and revisit the 1,318-case hard core before considering
B-spatial.

## Runtime and validation

The exact run completed in 465.09 seconds using four three-thread CPU workers
and both RTX A5000 GPUs. Independent validation passed for 216 robust selector
cells, 216 bilinear selector cells, 2,619 audited errors, 38,592 primary
predictions, and 38,592 stability predictions.

Canonical bundle:

```text
artifacts/midogpp/90_oracles_and_diagnostics/
  uniform_b_robust_interaction_probe_v1/seed42/
```
