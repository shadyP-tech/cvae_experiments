# Disagreement-regret core

This is a non-runnable mathematical library for a future, fresh-evidence
router. It converts sealed, label-free action probabilities into sparse
threshold-disagreement features, constructs exact additive BACC gains from
synthetic or separately authorized source-OOF labels, fits fixed-capacity
known-bank (not unseen-transfer) pairwise models with strict `H/q/e` query
exclusion, and emits development-only raw/safe selection diagnostics.

Training and target inference use disjoint authority types. A
`DevelopmentContext` still rejects consumed/target data. After training, the
complete outer-target family can be frozen as a canonical
`PairwiseRegretModelBank`. A `LabelFreeInferenceContext` may then bind that
bank to one separately sealed consumed-target cache for terminal prediction
only. It requires exact cache content/order hashes, the target prediction
seal, and the complete action schema, and makes label access, fresh-evidence
claims, routing/promotion authorization, and downstream experiment reuse
unrepresentable.

It intentionally contains no config, registry entry, runner, file I/O,
artifact adapter, label loader, oracle evaluator, or promotion surface. The
already-consumed MIDOG++ test set is admissible only through the typed
label-free inference seam above; its labels remain forbidden and its outputs
cannot validate routing. A runnable fresh-evidence study still requires a new
predeclared whole-case/patient/slide-disjoint reservation and a separate
protocol-owning adapter.

An `AUTHORIZED_SOURCE_OOF` adapter must bind a predeclared donor-query
allowlist and the SHA-256 of the exact sorted sample-key list, and must enforce
one-time authorization outside this pure in-memory package. Merely asserting a
context in Python is not a substitute for that durable ledger.

`AUTHORIZED_POSTHOC_SOURCE_OOF` is a distinct, explicitly non-fresh regime for
previously available source-only outcomes. It requires
`authorization_unused=False` and `source_evidence_previously_consumed=True`;
its fitted bank remains descriptive and cannot authorize routing or promotion.

The CPU core is vectorized and designed for bounded candidate-wise execution.
The canonical external workstation budget is four workers with three threads
each on the Xeon CPU; GPU probability production remains upstream and outside
this package.
