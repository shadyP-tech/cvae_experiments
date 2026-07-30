# MIDOG++ Uniform Variant-B Diagnostics

## Purpose and status

This note records two completed, workstation-only Stage-90 diagnostics and one
implemented, not-yet-run paired audit for a canonical-B CVAE representation.
They ask narrow representation and training-stability questions on
deterministic train-case holdouts. They do not establish CVAE preservation for
the active protocol, expert-bank eligibility, prior/generation quality,
routing, composition, or downstream utility.

Both artifacts carry `claim_scope=diagnostic_only` and
`may_feed_expert_bank=false`; neither may export a `RecipeLock`. Consequently,
they cannot revise the published Stage-20 bounded training-seed consensus locks
or supply a Stage-30 expert-bank recipe.

## Remote namespaces and artifact locations

The complete bundles are available on `xai-master`, not in this local checkout.
Their remote repository root is `/home/stud/spark/cvae_experiments`.

| Experiment namespace | Registered CLI surface | Remote artifact root | Complete evidence |
| --- | --- | --- | --- |
| `midogpp.oracle.uniform_b_source_expert_adaptation_pilot.v2` | `cvae-expert-bank uniform-b-adaptation-pilot` | `/home/stud/spark/cvae_experiments/artifacts/midogpp/90_oracles_and_diagnostics/uniform_b_source_expert_adaptation_pilot_v2/` | `reports/run_state.json`, `reports/validation_report.json`, `reports/pilot_decision.json` |
| `midogpp.oracle.uniform_b_block_tail_average_stability_probe.v1` | `cvae-expert-bank uniform-b-block-tail-average-stability-probe` | `/home/stud/spark/cvae_experiments/artifacts/midogpp/90_oracles_and_diagnostics/uniform_b_block_tail_average_stability_probe_v1/` | `reports/run_state.json`, `reports/validation_report.json`, `reports/stability_decision.json` |

The recorded runner argv is `python -m midogpp_thesis cvae-expert-bank <surface>
--config {resolved_config} --artifact-root output://<artifact-id>`. These are
recorded namespaces for provenance, not a recommendation to rerun either
diagnostic. The remote registry/config/catalog entries should be synced and
independently reviewed before any future local execution.

## Pilot v2: B-block representation and conservative-prior screen

Artifact evidence (remote paths on `xai-master`):

- `artifacts/midogpp/90_oracles_and_diagnostics/uniform_b_source_expert_adaptation_pilot_v2/config.resolved.yaml`
- `artifacts/midogpp/90_oracles_and_diagnostics/uniform_b_source_expert_adaptation_pilot_v2/manifests/frozen_protocol.json`
- `artifacts/midogpp/90_oracles_and_diagnostics/uniform_b_source_expert_adaptation_pilot_v2/provenance/input_artifacts.json`
- `artifacts/midogpp/90_oracles_and_diagnostics/uniform_b_source_expert_adaptation_pilot_v2/reports/pilot_decision.json`
- `artifacts/midogpp/90_oracles_and_diagnostics/uniform_b_source_expert_adaptation_pilot_v2/reports/validation_report.json`

Status: `COMPLETE` (`36/36` jobs) and validation `PASS`. It compares
`a_global_pca128`, `b_joint_pca128`, and `b_block_pca96_32` across centers
`2,5,6,9` with training and generation seeds `17,42,101`. The B-block frame
allocates `96` global and `32` local PCA components (128 total).

| Diagnostic readout | A-global mean BACC | B-joint mean BACC | B-block mean BACC |
| --- | ---: | ---: | ---: |
| decoded mean | `0.645872` | `0.728308` | `0.745109` |
| standard-normal prior sample | `0.835395` | `0.730414` | `0.698459` |
| class-conditional diagonal-shrinkage prior sample | `0.779243` | `0.807888` | `0.801297` |

The table is diagnostic context, not a source-inner selection comparison. The
pilot decision is `B_ADAPTATION_NOT_FEASIBLE`: `b_adaptation_feasible=false`,
`block_aware_justified=false`, and `conservative_prior_viable=false`. Although
the conditional prior improves B-block sampling relative to its standard-normal
baseline, center-9 recall is unsafe for the declared viability criterion: mean
`0.531467`, minimum `0.468750`, against a `0.55` floor. The B-block decoded
seed mean-preservation range is `0.069490`, already above the later stability
diagnostic's `0.05` target.

The pilot consumes train-case holdouts only. Its leakage/provenance report is
`PASS`, while also recording that heldout labels are used for diagnostic
scoring/progression and the decoded reconstruction condition; this is exactly
why the artifact remains diagnostic-only rather than a deployable result.

## Tail-average stability probe v1

Artifact evidence (remote paths on `xai-master`):

- `artifacts/midogpp/90_oracles_and_diagnostics/uniform_b_block_tail_average_stability_probe_v1/config.resolved.yaml`
- `artifacts/midogpp/90_oracles_and_diagnostics/uniform_b_block_tail_average_stability_probe_v1/manifests/frozen_protocol.json`
- `artifacts/midogpp/90_oracles_and_diagnostics/uniform_b_block_tail_average_stability_probe_v1/reports/predecessor_audit.json`
- `artifacts/midogpp/90_oracles_and_diagnostics/uniform_b_block_tail_average_stability_probe_v1/reports/stability_decision.json`
- `artifacts/midogpp/90_oracles_and_diagnostics/uniform_b_block_tail_average_stability_probe_v1/reports/validation_report.json`

Status: `COMPLETE`, `12/12` jobs, zero failures; full validation `PASS`. This
is a fresh, exact v2 endpoint replay restricted to B-block center/seed jobs. It
averages uniform FP32 post-update parameter states at every step from `751`
through `1000` (250 states); optimizer state is not averaged. The predecessor
audit binds the complete v2 input hashes, and validation passes
`terminal_control_replay`, `tail_average_derivation`, `lineage_hashes`,
`metric_prediction_reconciliation`, `decision_recomputation`, and
`claim_firewall`.

| Endpoint readout | Mean BACC | Interpretation |
| --- | ---: | --- |
| terminal B-block | `0.745109` | frozen v2 comparator |
| tail average (751–1000) | `0.741535` | `-0.003574` BACC; mean performance is preserved but not improved |

The decision `TAIL_AVERAGING_INSUFFICIENT` is based on two failed stability
gates out of 22, despite 20 passing gates:

- Seed mean-preservation range is `0.082269`, exceeding the maximum `0.05`.
- Maximum center/class-direction seed range is `0.192771`, exceeding the
  maximum `0.15`; center-9 specificity drives this failure. Center-5 positive
  recall also remains variable at `0.166667`.

No prior, generation, validation, or test evaluation was performed in this
probe. Its `next_step_if_pass` field conditionally names a separately reviewed
prior-only replay, but the gate did not pass; that replay is not authorized by
this result.

## Provenance correction

The final completed run follows a pre-training correction to component hashing:
the historical expected target `cvae/models.py` did not match the packaged
model directory `cvae/models/`. The frozen tail-average manifest now binds
`dependency/models/__init__.py` and `dependency/models/cvae.py`, and its
lineage-hash validation passes. This is a reproducibility repair, not an
intervention whose metric result can rehabilitate an earlier failed run or
promote Variant B.

## Low-noise paired reparameterization audit v1

Status: `IMPLEMENTED AND REGISTERED; NOT RUN`. No snapshot, audit decision, or
metric from this study exists yet.

The study isolates the two gradient-noise sources that changed with
`training_seed` in pilot v2 while preserving the B-block representation,
objective, optimizer, KL weight, training length, batch size, PCA allocation,
classifier, and decoded comparator. It has exactly three candidates over
centers `2,5,6,9` and initialization seeds `17,42,101`:

| Candidate | Schedule | Posterior reconstruction estimate | Decision role |
| --- | --- | --- | --- |
| legacy v2 replay | seed-specific | one epsilon | exact replay validation only |
| controlled baseline | fold-fixed | one epsilon | controlled comparison |
| proposed low-noise arm | fold-fixed | paired epsilon and negative epsilon | controlled comparison |

The two controlled arms share one schedule and one stored epsilon trace per
center. Their model initialization differs across the three seeds but is
identical within each one-epsilon/antithetic pair. The antithetic arm decodes
both latent perturbations, averages their two reconstruction losses, applies
the analytic KL once, and performs one optimizer update. Legacy rows cannot
enter the paired comparison or decision.

The registered workflow first builds a hash-promoted snapshot from only the
canonical MIDOG++ contract and canonical-B cache, then runs exactly 36 keys:
12 legacy replay cells plus 24 controlled cells forming 12 pairs. Workspace
entrypoint checks reject unregistered paths before writes or GPU use; snapshot
and audit roots have crash-recoverable exclusive locks. Bundle validation
recomputes key coverage, optimizer/decoder accounting, epsilon consumption,
legacy hashes, decoded metrics, real-reference denominators, paired deltas,
eval-row inventory bindings, claim firewalls, and the content index.

Registered experiment IDs:

```text
midogpp.oracle.uniform_b_paired_reparameterization_snapshot.v1
midogpp.oracle.uniform_b_paired_reparameterization_audit.v1
```

Canonical outputs:

```text
artifacts/midogpp/90_oracles_and_diagnostics/inputs/uniform_b_paired_reparameterization_snapshot_v1/
artifacts/midogpp/90_oracles_and_diagnostics/uniform_b_paired_reparameterization_audit/v1/
```

Both outputs are currently absent. A future passing run would resolve only the
B training-stability question. The conditional prior still has a separate
center-9 direction-floor failure and requires its own source-inner decision.
This audit cannot promote the old Stage-90 bundles, export a `RecipeLock`, or
feed Stage 20 through Stage 70.

## Interpretation and next boundary

The B-block representation has a useful diagnostic signal: its decoded mean
BACC exceeds the A-global and B-joint comparators in this restricted pilot.
However, independent training runs remain materially different. Tail averaging
within each last-quarter trajectory does not solve that between-training-seed
variation, and slightly reduces average BACC.

The appropriate conclusion is `DIAGNOSTIC ONLY` and `NEGATIVE_RESULT` for
tail averaging as a Variant-B stabilization intervention. Do not promote B
checkpoints, priors, or this diagnostic recipe to Stage 30; do not use it for a
prior-only replay, routing, expert selection, generation, or downstream
utility claim. The active eligible input for the planned Stage-30 source-expert
bank remains the published bounded training-seed consensus `RecipeLock` bundle
documented in
[MIDOG++ Prior Recovery And Task-Fisher Preservation](midogpp-prior-recovery-task-fisher.md).

The next authorized Variant-B study is the registered low-noise paired audit
above. It tests fold-fixed schedules first and then the antithetic estimator,
without changing learning rate, training length, or batch size. If that exact
study fails, the next single amendment may test gradient accumulation to an
effective batch of `256`; it does not authorize an open-ended optimizer sweep.

## Remaining TODOs

- Sync the remote Stage-90 configs, registry/catalog entries, and complete
  bundles before treating local navigation as a complete offline record.
- Sync the implemented paired-audit code to `xai-master`, run the snapshot
  first and the audit second, and independently validate both emitted bundles.
- Keep any later conditional-prior decision separate from the B
  training-stability result.
