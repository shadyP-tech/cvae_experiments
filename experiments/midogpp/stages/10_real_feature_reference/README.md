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
- registry entry `midogpp.real_feature.tuned_classifier.seed42`
- registry entry `midogpp.real_feature.signal_controls.v1`
- registry entry `midogpp.real_feature.multiaxis.v1`
