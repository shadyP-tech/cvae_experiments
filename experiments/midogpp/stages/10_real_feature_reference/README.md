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
- registry entry `midogpp.real_feature.tuned_classifier.seed42`
- registry entry `midogpp.real_feature.eligible_tuned_predict_reference.v2`
- registry entry `midogpp.real_feature.signal_controls.v1`
- registry entry `midogpp.real_feature.multiaxis.v1`

## Eligible Matched Reference v2

`eligible_tuned_real_reference_v2.yaml` defines the matched real-feature
denominator for the Stage-20 prior-recovery experiment. It uses the complete
Virchow2 feature frame for the nine eligible centers
`0,1,2,3,5,6,7,8,9`, excludes quarantined center `4`, freezes the exact
20-spec classifier grid with hash `16a7a1183ea3f65b`, and uses sklearn
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
