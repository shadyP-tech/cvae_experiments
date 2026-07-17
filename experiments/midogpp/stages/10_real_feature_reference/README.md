# Stage 10: Real-Feature Reference

This stage establishes source-only transfer signal and the real-feature
classifier surface used as a comparator for later CVAE work.

Configs must use a validated MIDOG++ contract and corrected Virchow2 feature
cache. Classifier, threshold, and aggregation selection is source-inner only;
held-out center labels are scoring-only. Results from this stage cannot support
CVAE preservation, routing, generation, or synthetic-utility claims.

Reusable implementation lives in
`src/midogpp_thesis/real_features/`.

Current canonical definitions:

- `configs/multiaxis_baseline_v1.yaml`
- `configs/virchow2_signal_controls_v1.yaml`
- `configs/eligible_tuned_real_reference_v2.yaml`
- `configs/fixed_c_risk_diagnostic_v1.yaml`
- registry entry `midogpp.real_feature.tuned_classifier.seed42`
- registry entry `midogpp.real_feature.eligible_tuned_predict_reference.v2`
- registry entry `midogpp.real_feature.fixed_c_risk_diagnostic.v1`
- registry entry `midogpp.real_feature.signal_controls.v1`
- registry entry `midogpp.real_feature.multiaxis.v1`

## Eligible Matched Reference v2

`eligible_tuned_real_reference_v2.yaml` defines the matched real-feature
denominator for the Stage-20 prior-recovery experiment. It uses the complete
Virchow2 feature frame for the nine eligible centers
`0,1,2,3,5,6,7,8,9`, excludes quarantined center `4`, freezes the exact
10-spec classifier grid with hash `5abd0897d02bdcaa`, and uses sklearn
`predict` rather than a separately tuned threshold policy.

For each held-out center, classifier selection is source-inner and the outer
center is absent from fitting and selection. Target-center labels are used only
for final real-feature scoring. This artifact has
`claim_scope=real_feature_transfer_only`, is not a router, and cannot establish
CVAE preservation or prior quality.

Run from an installed checkout:

```bash
conda run -n thesis python -m midogpp_thesis workspace prepare \
  midogpp.real_feature.eligible_tuned_predict_reference.v2

conda run -n thesis python -m midogpp_thesis workspace run \
  midogpp.real_feature.eligible_tuned_predict_reference.v2
```

Canonical output root:

```text
artifacts/midogpp/10_real_feature_reference/eligible_tuned_real_reference_v2/seed42/
```

Required output files are:

```text
config.resolved.yaml
provenance/input_artifacts.json
manifests/protocol_manifest.json
reports/leakage_provenance_report.json
tables/source_inner_classifier_tuning.csv
tables/classifier_tuned_source_results.csv
tables/classifier_tuned_predictions.csv
```

Status: `IMPLEMENTED, NOT RUN`. The catalog label remains
`TODO_VERIFY_ARTIFACT`; no v2 metric or claim exists until this bundle is
produced and validated.

## Fixed-C Risk-Weighting Diagnostic v1

`fixed_c_risk_diagnostic_v1.yaml` holds the classifier constant at
`C=0.01`, `penalty=l2`, `solver=lbfgs`, `max_iter=5000`,
`class_weight=none`, `random_state=23`, and sklearn `predict`. It evaluates
four predeclared source-fit weighting arms for each of the nine eligible
held-out centers:

| Arm | Raw source-fit weight |
| --- | --- |
| `pooled` | `1` |
| `global_class` | `N/(2*n_y)` |
| `domain` | `N/(D*n_d)` |
| `domain_class` | `N/(2*D*n_dy)` |

Here `N` is the number of source-fit rows, `D` is the number of source
domains, and `n_y`, `n_d`, and `n_dy` are the source-fit class, domain, and
domain-class cell counts. Each arm is normalized to sum to `N`; missing,
non-finite, or non-positive cells fail closed. The fixed design is therefore
`9 centers x 4 arms = 36` fits, with no classifier or arm selection.

Prepare and run the registered diagnostic with:

```bash
conda run -n thesis python -m midogpp_thesis workspace prepare \
  midogpp.real_feature.fixed_c_risk_diagnostic.v1

conda run -n thesis python -m midogpp_thesis workspace run \
  midogpp.real_feature.fixed_c_risk_diagnostic.v1
```

Canonical output root:

```text
artifacts/midogpp/10_real_feature_reference/fixed_c_risk_diagnostic_v1/seed42/
```

The required 12-file bundle is:

```text
config.resolved.yaml
provenance/input_artifacts.json
manifests/frozen_protocol_snapshot.json
manifests/protocol_manifest.json
reports/leakage_provenance_report.json
reports/diagnostic_summary.json
reports/diagnostic_report.md
reports/runtime_summary.json
tables/fixed_c_risk_results.csv
tables/fixed_c_risk_predictions.csv
tables/fixed_c_risk_weight_audit.csv
tables/fixed_c_risk_paired_comparison.csv
```

Claim boundary: this is a non-adoptive real-feature transfer diagnostic.
Held-out-center labels are final-scoring-only. No result may choose a
classifier, weighting rule, recipe, router, expert, generation setting, or
downstream policy. The output is catalog-blocked from every current reuse
purpose except Stage-90 oracle and diagnostic evidence, and it cannot support
CVAE preservation, prior, routing, or synthetic-utility claims. No metric or
result is claimed until a produced bundle passes its validator.
