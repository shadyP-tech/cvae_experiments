# Cross-Dataset Assessment: LOQDO Utility-Compatible Learning

- Classification: non_actionable_ranking_only
- Rationale: Some ranking uplift appears, but utility-gap reduction does not hold across datasets.
- Transfer success: False
- BreakHis transfer gate pass: False
- Camelyon17 transfer gate pass: False
- Include diagnostic oracle methods: False
- Require adoption_gate_pass_proxy: True
- Allow derived fallback when missing: False
- BreakHis best method: static_embedding::linear_regression__static_embedding__probe_off__interact_off__arm_static_embedding
- Camelyon17 best method: static_embedding::linear_regression__static_embedding__probe_off__interact_off__arm_static_embedding
- BreakHis best (all candidates): response_oracle_diagnostic::oracle_eval_mean_cheat__response_oracle_diagnostic__probe_off__interact_off__arm_response_oracle_diagnostic
- Camelyon17 best (all candidates): response_oracle_diagnostic::oracle_eval_mean_cheat__response_oracle_diagnostic__probe_off__interact_off__arm_response_oracle_diagnostic

## breakhis_best
- method_key: static_embedding::linear_regression__static_embedding__probe_off__interact_off__arm_static_embedding
- tier: fail
- top1_uplift_vs_metadata_mean: 0.0
- spearman_uplift_vs_metadata_mean: 0.19207001931529108
- oracle_gap_reduction_vs_metadata_mean: 0.0
- normalized_oracle_gap_reduction_vs_metadata_mean: 0.0
- calibration_error_mean: 434.28361245194327
- improving_run_count: 0
- instability_breach: 0

## camelyon17_best
- method_key: static_embedding::linear_regression__static_embedding__probe_off__interact_off__arm_static_embedding
- tier: fail
- top1_uplift_vs_metadata_mean: 0.0
- spearman_uplift_vs_metadata_mean: 0.15693723010282123
- oracle_gap_reduction_vs_metadata_mean: 0.0
- normalized_oracle_gap_reduction_vs_metadata_mean: 0.0
- calibration_error_mean: 453.27976218463476
- improving_run_count: 0
- instability_breach: 0
