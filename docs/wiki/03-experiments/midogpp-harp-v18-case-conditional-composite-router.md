# HARP v18: case-conditional composite routing

V18 implements the successor to the scientifically exhausted v17 diagnostic.
It is registered as **planned, execution unauthorized**. Implementation and
synthetic tests are construction evidence; there is no v18 MIDOG++ result.
V17 remains exhausted and its outputs, caches, labels, amendment, lease,
learned state, and thresholds are not inputs to v18.

## What changes

V17 completed generation and all 81 classifier tasks but selected exact B in
all five source outer folds. Admission stopped at
`NO_NONZERO_SAFE_OOF_COVERAGE`: zero of 216 cases routed; neither bootstrap
bounds nor target actions were constructed. Primitive oracle opportunities
do not prove that a fitted policy can identify those opportunities safely.

V18 addresses the representation and observability failures:

| Bottleneck | Implemented behavior |
|---|---|
| One global action configuration | Select an action separately for each case from B, exact U, D01_ONLY, D10_ONLY, BOTH. |
| Best-direction score executes both directions | Candidate features and targets describe the actual executed composite; an ONLY action copies B exactly on the other branch. |
| One primitive score reused across K and lambda | Shared outcome heads use actual-composite probability changes, branch-gated selected donor features, family, K, and lambda. |
| Positive-only gain targets | Keep signed classwise and aligned gain, harm, Brier and log-loss outcomes, plus proper-loss-safe-positive event estimates. |
| One ineligible case discards an entire arm | Eligibility and exact-B fallback are case-local; byte duplicates and rejection causes remain in the frontier. |
| Inconsistent source and terminal case weighting | Normalize source gain to equal centers, equal classes, equal supporting cases, including single-class cases. |
| Missing rejection evidence | Persist every action/threshold frontier and actual proposed-menu oracle diagnostics before admission can abort. |

The menu has at most 38 configurations: B and U, plus three directional
families times K in `{1,2,4}` times lambda in `{0.25,0.5,0.75,1}`. K and lambda
are case-level action descriptors, not globally selected hyperparameters.
Ineligible configurations and byte duplicates cannot route. Exact U retains
the physical U vector; it is not reconstructed by averaging donors.
An action that changes only margins cannot route either: unchanged hard
predictions imply exactly zero BACC gain regardless of labels. Such examples
remain available to train the signed outcome heads.

## Learning and risk contract

`Fit(S)` builds four ranker folds wholly within S. Each ranker learns only
from its training cases, proposes the held cases' composite vectors, and seals
them before outcome scoring. After all ranker-OOF composites in S are sealed,
a training-scope normalizer supplies their signed targets to a shared ridge
outcome model. The donor ranker is then refitted on S. Every inner validation
split refits this **complete stack**, including feature normalization.

Five outer folds evaluate that learning procedure. Within each outer training
set, four inner folds tune only the abstention threshold. All outer decisions
are sealed before the joint outer-OOF class-support weights or metrics are
calculated. Final threshold selection repeats full-source inner CV and then
refits the complete stack. It does not take a modal action or median threshold
from the outer folds. Full-source primitive audit aggregates never train these
fold models.

For each case, eligible candidates with estimated positive signed gain pass a
fixed predictive screen: harm probability at most 0.25, Brier delta at most
0.002, and log-loss delta at most 0.005. The highest estimated gain above the
selected threshold wins, with deterministic ties; otherwise exact B wins.
All heads share candidate-row weights within a case, then balance cases within
centers. Ridge penalties are fixed at 1.0. Branch-gated mechanism features
allow the preferred direction to reverse with case context.

The model also reports a safe-positive probability and a pessimistic gain
score (estimated gain minus training residual RMSE). These are **model
estimates, not calibrated individual safety guarantees**. The pessimistic
score is not an action-level admission bound or hard veto.

Whole-policy admission needs at least 18 routed source OOF cases and at least
six centers with two or more routes each. Incidental centers with one route
remain in risk accounting without invalidating the six qualifying centers.
The approximate joint bootstrap bounds constrain aligned gain and
coverage-scaled harm/Brier/log-loss moments. It resamples whole cases within
the nine fixed centers, recomputing class-support denominators in each draw.
An unsupported sampled class gets conservative gain -1 and is counted in the
report. Centers and technical seed cells are not independent replicates of
new-center generalization. These bounds exclude refit uncertainty and are not
conformal or final-policy safety certificates.

Zero source OOF routing still fails closed before target actions. Other
nonadmission retains the declared exact-B fallback. If an admitted source
policy produces zero nonbaseline target actions, a new label-free coverage
gate stops before evaluation truth. No gate lowers thresholds or forces routes.

## Code responsibilities

| Package or module | Responsibility |
|---|---|
| `routing/case_conditional_composite_router_v18/contracts.py`, `composition.py` | Typed label-free menus and byte-exact candidate recipes. |
| `modeling.py`, `outcome_model.py` | Pairwise proposals and shared signed composite outcome heads. |
| `truth.py`, `aligned_metrics.py` | Nonserializable source capabilities and scope-specific outcome joins. |
| `splitting.py`, `stacked_fitting.py`, `crossfit.py` | Deterministic case splits and fully nested fitting. |
| `frontier.py`, `admission.py`, `policy.py` | Rejection evidence, policy risk, and label-free decisions. |
| `runtime/harp_v18_execution` | GPU/classifier work, physical probability stores, adapter, two fresh reconstruction processes. |
| `runtime/harp_v18_execution/branch_recipe.py` | Bind declared family and K to physical components; verify unused-branch B bytes. |
| `diagnostics/fixed_bank_harp_router_v18` | Independent preparation, source sealing, amendment/lease checks and phase ordering. |
| `execution/source_diagnostics.py` | Durable frontier/headroom reports before the admission gate. |
| `diagnostics/harp_v18_cli.py` | V18 lifecycle commands without expanding the shared dispatcher. |

Paths in this table are relative to `src/midogpp_thesis/cvae/` except the
execution module, which is within the v18 diagnostics package.

## Workstation execution

The frozen profile is Xeon W-2265 (12 cores/24 threads), 125 GB RAM and two
24 GB RTX A5000 GPUs. Two persistent GPU workers generate physical data;
four classifier workers use three BLAS threads each. Their queues are bounded
and phases do not overlap competing CPU pools. The 81 classifier tasks still
perform 810 physical fits, each predicting source and target role surfaces.
The larger action menu adds **zero generation or classifier fits**.

Nested CPU science keeps nonserializable truth capabilities in the parent.
An active `threadpoolctl` context enforces one thread even for already-loaded
BLAS libraries. Shared descriptors and probability decoding reuse label-free
values; candidate predictions are batched and reused across thresholds.
Bootstrap draws are processed in bounded batches. No learned model cache is
shared across distinct training scopes. The configured science worker pool is
capacity metadata and is not used to serialize truth-bearing fits.
Frontier threshold sweeps reuse aggregate observations from sealed candidates;
they do not rebuild and hash thousands of identical per-case records. A
reference-replay regression checks unchanged utility and risk summaries.

Physical probabilities use float32; means, shrinkage arithmetic, fitting and
metrics use float64 with declared float32 transport boundaries. Reconstruction
checks D01_ONLY/D10_ONLY/BOTH semantics, component identities, K, lambda, and
exact unused-branch bytes in two independent processes before test truth.

## Identity and use

The experiment is
`midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_harp_router.v18`.
The config is
`experiments/midogpp/stages/90_oracles_and_diagnostics/configs/uniform_b_v2_consumed_test_fixed_bank_harp_router_v18.yaml`.
Its output root ends in `uniform_b_v2_consumed_test_fixed_bank_harp_router/v18`;
scratch is `/data/local/fixed_bank_harp_router_v18`. New v18 aliases bind the
source/target cache, source label capability, test release, original-ledger
consumer fence, and execution amendment. Preparation derives these from the
catalog's original canonical inputs, never a predecessor's output. The
authorized bank and generation locks remain the direct canonical inputs.

The mutation-free inspection command is:

```sh
PYTHONPATH=src python -m midogpp_thesis cvae-diagnostics \
  fixed-bank-harp-router-v18 \
  --config experiments/midogpp/stages/90_oracles_and_diagnostics/configs/uniform_b_v2_consumed_test_fixed_bank_harp_router_v18.yaml \
  --inspect-plan
```

Preparation, activation and execution remain distinct. The checked-in planned
config grants none of their scientific permissions. Actual use needs prepared
v18 inputs, the reviewed source snapshot, an independently issued amendment,
and a new single-use lease.

The artifact and claim dataset are MIDOG++ Virchow2 with center as the domain
axis; source development uses 216 training cases/9,648 rows, terminal
evaluation 218 test cases/9,928 rows. Eligible centers are
`{0,1,2,3,5,6,7,8,9}`. Source q and target H menus exclude their own expert:
`C - {q}` and `C - {H}`. Source OOF is conditional on the already-frozen expert
bank, not an end-to-end refit of that bank. Test labels may only score globally
sealed, independently reconstructed routes.

Any eventual result remains `POST_HOC_CONSUMED_TEST_SENSITIVITY`,
`TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE`, and `fresh_evidence=false`.
It cannot establish deployment safety, unseen-center generalization,
significance or fresh thesis-confirmatory improvement, or feed Stage 60/70 or
another experiment. A correct v18 implementation can still fail scientific
admission if the available features cannot recover the oracle opportunities.

## Implementation validation (2026-09-05)

The saved record is `docs/validation/harp_v18_2026-09-05.json`.
All 80 v18 tests passed in 23.44 seconds, including complete nested fitting,
held-label poisoning invariance, audit-aggregate independence, opposing branch
preferences, the single-class metric correction, margin-only fallback,
frontier replay parity, and two independent process reconstructions. Ten
predecessor registration/lifecycle tests also passed. Workspace validation,
path-free inspect/dry-run dispatch, and source-closure exclusion checks passed.
The closure fingerprint covers 207 Python files up to the shared dispatch
boundary; this is a construction fingerprint, not an issued execution seal.

The standalone benchmark `benchmarks/harp_v18_synthetic.py` creates only
synthetic physical vectors and labels. With 216 source cases, nine centers,
48 rows per case, the complete 38-configuration menu and production 5/4/4 fold
counts, it completed 30 stacked fits/150 ranker fits in 162.44 seconds total
(161.15 seconds fitting), with peak RSS 739,115,008 bytes (about 705 MiB).
It produced 7,020 frontier rows and admitted 216/216 synthetic OOF routes.
This run used one BLAS thread on macOS arm64, not the Xeon/A5000 workstation;
it verifies CPU geometry and construction, not real routing utility or GPU
throughput. The benchmark contains no original data, expert generation,
classifier training, authority, lease or test-label access.

```sh
PYTHONPATH=src python benchmarks/harp_v18_synthetic.py \
  --output-dir /tmp/harp_v18_synthetic_benchmark
```

The benchmark refuses to overwrite an existing result and restricts its
outputs to a temporary subdirectory. It has no scientific execution identity.
