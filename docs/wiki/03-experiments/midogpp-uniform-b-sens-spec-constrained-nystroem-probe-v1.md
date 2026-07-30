# MIDOG++ Uniform-B Sensitivity/Specificity-Constrained Nyström Probe v1

## Question

Can nonlinear B retain a useful average advantage while explicitly preventing
large source-inner sensitivity, specificity, and BACC regressions at the fixed
global threshold of `0.5`?

This is a third post-hoc Stage-90 diagnostic over the already inspected
9,648-row train surface. It is not independent confirmation and cannot
authorize validation or test scoring.

## Frozen protocol

For each outer center, the experiment inherits the already selected Nyström
kernel tuple and canonical linear-B classifier specification. Every ordered
source-inner fold excludes both the outer and inner centers from the scaler,
gamma estimate, landmarks, and classifier fits.

Four decision objectives are evaluated:

- canonical inherited class weighting;
- equal center×class weighting;
- group DRO with eta `0.1`;
- group DRO with eta `0.5`.

The global threshold remains fixed at `0.5`. Capacity is varied analytically
through native decision logits:

\[
z_\alpha=z_{\mathrm{linear}}+
\alpha(z_{\mathrm{Nyström}}-z_{\mathrm{linear}}),
\qquad
\alpha\in\{0.25,0.5,0.75,1.0\}.
\]

A nonlinear candidate must satisfy all of the following across all eight
source-inner centers:

- recall delta at least `−0.02` in every fold;
- specificity delta at least `−0.02` in every fold;
- BACC delta at least `−0.01` in every fold;
- positive equal-center mean recall delta;
- mean specificity delta at least `−0.01`;
- positive equal-center mean BACC delta.

If no nonlinear candidate is feasible, the fold fails closed to exact linear B
with `alpha=0`.

## Result

Decision: `NO_CONSTRAINED_BPLUS_CANDIDATE_PASSES`.

Only two outer recipes admitted nonlinear capacity:

| Center | Objective | Alpha | Primary BACC delta | Recall delta | Specificity delta |
|---|---|---:|---:|---:|---:|
| 6 | canonical class weight | 0.25 | +0.01093 | +0.02732 | −0.00546 |
| 9 | equal center×class | 0.25 | +0.01445 | −0.03468 | +0.06358 |

Centers `0,1,2,3,5,7,8` had no feasible nonlinear source-inner candidate and
therefore used exact linear B.

Across all nine outer centers:

- equal-center mean BACC delta: `+0.00282`;
- strict BACC wins: `2/9`;
- mean recall delta: `−0.00082`;
- mean specificity delta: `+0.00646`;
- worst recall delta: `−0.03468`;
- worst specificity delta: `−0.00546`.

The paired case bootstrap was supportive only and crossed zero:
`[−0.000027, +0.004990]`.

## Interpretation

The constraints do not preserve most of the original `+0.0232` Nyström gain.
Seven centers require complete shrinkage to linear B, and center 9 violates the
outer recall floor despite satisfying the source-inner constraints. The
unconstrained gain therefore depends materially on class-direction exchanges
that do not generalize uniformly across centers.

This does not show that B lacks nonlinear information. It shows that the
current uniform, fixed-threshold nonlinear decision family cannot convert that
information into a broadly feasible improvement under the frozen constraints.
Further objective reweighting or capacity interpolation is not currently
justified as a canonical B+ direction.

## Runtime and validation

The final run took `275.92` seconds (`4m36s`) on the 12-core/24-thread Xeon
workstation using four worker processes and three BLAS threads per worker.
The two RTX A5000 GPUs were intentionally unused.

The independent validator reconstructed:

- input and lineage hashes;
- resolved and frozen protocol bindings;
- both inherited endpoint replays;
- 1,152 blend metric cells and 144 candidate summaries;
- all nine source-only selection locks;
- 9,648 primary and 19,296 stability predictions;
- outer metrics, deltas, error exchange, bootstrap, feasibility, and progression
  decision.

Validation and test remain untouched.
