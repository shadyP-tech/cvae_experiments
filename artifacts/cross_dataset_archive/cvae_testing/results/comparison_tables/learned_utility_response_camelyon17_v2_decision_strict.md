# Compatibility Decision Table

- uplift_reference_method: metadata_routing
- overall_tier: fail
- selected_method: metadata_routing
- min_improving_seeds: 3
- instability: std_threshold=0.05 sign_inconsistency_min_count=2

| method | role | eligible | decision | tier | n_seeds | top1 | spearman | mean_oracle_gap_pct | top1_uplift_vs_metadata | spearman_uplift_vs_metadata | gap_pct_reduction_vs_metadata | improving_seed_count | raw_instability_breach | instability_gate_applied | instability_breach |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| candidate_oracle_routing | diagnostic | 0 | not_selected | reference_only | 3 | 1.0000 +- 0.0000 | 1.0000 +- 0.0000 | 0.0000 +- 0.0000 | 0.6753 +- 0.0773 | 1.0000 +- 0.0000 | 10.9376 +- 2.0915 | 3 | 1 | 0 | 0 |
| hybrid_alpha_0.0 | diagnostic | 0 | not_selected | reference_only | 3 | 0.3450 +- 0.0675 | 0.3325 +- 0.0703 | 11.6416 +- 2.4734 | 0.0203 +- 0.0099 | 0.3325 +- 0.0703 | -0.7040 +- 0.7499 | 1 | 1 | 0 | 0 |
| hybrid_alpha_0.0_minmax | diagnostic | 0 | not_selected | reference_only | 3 | 0.3450 +- 0.0675 | 0.3325 +- 0.0703 | 11.6416 +- 2.4734 | 0.0203 +- 0.0099 | 0.3325 +- 0.0703 | -0.7040 +- 0.7499 | 1 | 1 | 0 | 0 |
| hybrid_alpha_0.1 | diagnostic | 0 | not_selected | reference_only | 3 | 0.3450 +- 0.0675 | 0.3325 +- 0.0703 | 11.6416 +- 2.4734 | 0.0203 +- 0.0099 | 0.3325 +- 0.0703 | -0.7040 +- 0.7499 | 1 | 1 | 0 | 0 |
| hybrid_alpha_0.1_minmax | diagnostic | 0 | not_selected | reference_only | 3 | 0.3450 +- 0.0675 | 0.3325 +- 0.0703 | 11.6416 +- 2.4734 | 0.0203 +- 0.0099 | 0.3325 +- 0.0703 | -0.7040 +- 0.7499 | 1 | 1 | 0 | 0 |
| hybrid_alpha_0.2 | diagnostic | 0 | not_selected | reference_only | 3 | 0.3450 +- 0.0675 | 0.3325 +- 0.0703 | 11.6416 +- 2.4734 | 0.0203 +- 0.0099 | 0.3325 +- 0.0703 | -0.7040 +- 0.7499 | 1 | 1 | 0 | 0 |
| hybrid_alpha_0.2_minmax | diagnostic | 0 | not_selected | reference_only | 3 | 0.3450 +- 0.0675 | 0.3325 +- 0.0703 | 11.6416 +- 2.4734 | 0.0203 +- 0.0099 | 0.3325 +- 0.0703 | -0.7040 +- 0.7499 | 1 | 1 | 0 | 0 |
| hybrid_alpha_0.3 | diagnostic | 0 | not_selected | reference_only | 3 | 0.3450 +- 0.0675 | 0.3325 +- 0.0703 | 11.6416 +- 2.4734 | 0.0203 +- 0.0099 | 0.3325 +- 0.0703 | -0.7040 +- 0.7499 | 1 | 1 | 0 | 0 |
| hybrid_alpha_0.3_minmax | diagnostic | 0 | not_selected | reference_only | 3 | 0.3450 +- 0.0675 | 0.3325 +- 0.0703 | 11.6416 +- 2.4734 | 0.0203 +- 0.0099 | 0.3325 +- 0.0703 | -0.7040 +- 0.7499 | 1 | 1 | 0 | 0 |
| hybrid_alpha_0.4 | diagnostic | 0 | not_selected | reference_only | 3 | 0.3450 +- 0.0675 | 0.3325 +- 0.0703 | 11.6416 +- 2.4734 | 0.0203 +- 0.0099 | 0.3325 +- 0.0703 | -0.7040 +- 0.7499 | 1 | 1 | 0 | 0 |
| hybrid_alpha_0.4_minmax | diagnostic | 0 | not_selected | reference_only | 3 | 0.3450 +- 0.0675 | 0.3325 +- 0.0703 | 11.6416 +- 2.4734 | 0.0203 +- 0.0099 | 0.3325 +- 0.0703 | -0.7040 +- 0.7499 | 1 | 1 | 0 | 0 |
| hybrid_alpha_0.5 | diagnostic | 0 | not_selected | reference_only | 3 | 0.3450 +- 0.0675 | 0.3325 +- 0.0703 | 11.6416 +- 2.4734 | 0.0203 +- 0.0099 | 0.3325 +- 0.0703 | -0.7040 +- 0.7499 | 1 | 1 | 0 | 0 |
| hybrid_alpha_0.5_minmax | diagnostic | 0 | not_selected | reference_only | 3 | 0.3450 +- 0.0675 | 0.3325 +- 0.0703 | 11.6416 +- 2.4734 | 0.0203 +- 0.0099 | 0.3325 +- 0.0703 | -0.7040 +- 0.7499 | 1 | 1 | 0 | 0 |
| hybrid_alpha_0.6 | diagnostic | 0 | not_selected | reference_only | 3 | 0.3450 +- 0.0675 | 0.3325 +- 0.0703 | 11.6416 +- 2.4734 | 0.0203 +- 0.0099 | 0.3325 +- 0.0703 | -0.7040 +- 0.7499 | 1 | 1 | 0 | 0 |
| hybrid_alpha_0.6_minmax | diagnostic | 0 | not_selected | reference_only | 3 | 0.3450 +- 0.0675 | 0.3325 +- 0.0703 | 11.6416 +- 2.4734 | 0.0203 +- 0.0099 | 0.3325 +- 0.0703 | -0.7040 +- 0.7499 | 1 | 1 | 0 | 0 |
| hybrid_alpha_0.7 | diagnostic | 0 | not_selected | reference_only | 3 | 0.3450 +- 0.0675 | 0.3325 +- 0.0703 | 11.6416 +- 2.4734 | 0.0203 +- 0.0099 | 0.3325 +- 0.0703 | -0.7040 +- 0.7499 | 1 | 1 | 0 | 0 |
| hybrid_alpha_0.7_minmax | diagnostic | 0 | not_selected | reference_only | 3 | 0.3450 +- 0.0675 | 0.3325 +- 0.0703 | 11.6416 +- 2.4734 | 0.0203 +- 0.0099 | 0.3325 +- 0.0703 | -0.7040 +- 0.7499 | 1 | 1 | 0 | 0 |
| hybrid_alpha_0.8 | diagnostic | 0 | not_selected | reference_only | 3 | 0.3450 +- 0.0675 | 0.3325 +- 0.0703 | 11.6416 +- 2.4734 | 0.0203 +- 0.0099 | 0.3325 +- 0.0703 | -0.7040 +- 0.7499 | 1 | 1 | 0 | 0 |
| hybrid_alpha_0.8_minmax | diagnostic | 0 | not_selected | reference_only | 3 | 0.3450 +- 0.0675 | 0.3325 +- 0.0703 | 11.6416 +- 2.4734 | 0.0203 +- 0.0099 | 0.3325 +- 0.0703 | -0.7040 +- 0.7499 | 1 | 1 | 0 | 0 |
| hybrid_alpha_0.9 | diagnostic | 0 | not_selected | reference_only | 3 | 0.3450 +- 0.0675 | 0.3325 +- 0.0703 | 11.6416 +- 2.4734 | 0.0203 +- 0.0099 | 0.3325 +- 0.0703 | -0.7040 +- 0.7499 | 1 | 1 | 0 | 0 |
| hybrid_alpha_0.9_minmax | diagnostic | 0 | not_selected | reference_only | 3 | 0.3450 +- 0.0675 | 0.3325 +- 0.0703 | 11.6416 +- 2.4734 | 0.0203 +- 0.0099 | 0.3325 +- 0.0703 | -0.7040 +- 0.7499 | 1 | 1 | 0 | 0 |
| hybrid_alpha_1.0 | diagnostic | 0 | not_selected | reference_only | 3 | 0.3247 +- 0.0773 | 0.0000 +- 0.0000 | 10.9376 +- 2.0915 | 0.0000 +- 0.0000 | 0.0000 +- 0.0000 | 0.0000 +- 0.0000 | 0 | 0 | 0 | 0 |
| hybrid_alpha_1.0_minmax | diagnostic | 0 | not_selected | reference_only | 3 | 0.3247 +- 0.0773 | 0.0000 +- 0.0000 | 10.9376 +- 2.0915 | 0.0000 +- 0.0000 | 0.0000 +- 0.0000 | 0.0000 +- 0.0000 | 0 | 0 | 0 | 0 |
| latent_wasserstein_routing | diagnostic | 0 | not_selected | reference_only | 3 | 0.3450 +- 0.0675 | 0.3325 +- 0.0703 | 11.6416 +- 2.4734 | 0.0203 +- 0.0099 | 0.3325 +- 0.0703 | -0.7040 +- 0.7499 | 1 | 1 | 0 | 0 |
| linear_regressor | learned | 1 | not_selected | fail | 3 | 0.3123 +- 0.0669 | 0.0407 +- 0.1511 | 16.3585 +- 2.1929 | -0.0123 +- 0.0660 | 0.0407 +- 0.1511 | -5.4209 +- 1.6136 | 0 | 1 | 1 | 1 |
| metadata_routing | baseline | 0 | baseline_reference | baseline | 3 | 0.3247 +- 0.0773 | 0.0000 +- 0.0000 | 10.9376 +- 2.0915 | 0.0000 +- 0.0000 | 0.0000 +- 0.0000 | 0.0000 +- 0.0000 | 0 | 0 | 0 | 0 |
| mlp_regressor | learned | 1 | not_selected | fail | 3 | 0.4070 +- 0.0539 | 0.2987 +- 0.1064 | 10.8400 +- 2.4592 | 0.0823 +- 0.0368 | 0.2987 +- 0.1064 | 0.0976 +- 0.3703 | 2 | 1 | 1 | 1 |
| pairwise_ranker_combined | learned | 1 | not_selected | fail | 3 | 0.5337 +- 0.0196 | 0.5901 +- 0.0521 | 5.3827 +- 0.7449 | 0.2090 +- 0.0941 | 0.5901 +- 0.0521 | 5.5549 +- 2.7488 | 3 | 1 | 1 | 1 |
| pairwise_ranker_latent_only | learned | 1 | not_selected | fail | 3 | 0.5117 +- 0.0482 | 0.5951 +- 0.0490 | 5.8094 +- 0.9825 | 0.1870 +- 0.0323 | 0.5951 +- 0.0490 | 5.1282 +- 1.1144 | 3 | 1 | 1 | 1 |
| pairwise_ranker_metadata_only | learned | 1 | not_selected | fail | 3 | 0.3567 +- 0.0677 | 0.1784 +- 0.1475 | 13.3415 +- 3.9027 | 0.0320 +- 0.0110 | 0.1784 +- 0.1475 | -2.4039 +- 2.0677 | 1 | 1 | 1 | 1 |
| random_rank_floor | control | 0 | not_selected | reference_only | 3 | 0.2720 +- 0.0578 | -0.0275 +- 0.0605 | 18.1714 +- 2.5107 | -0.0527 +- 0.0984 | -0.0275 +- 0.0605 | -7.2338 +- 0.8997 | 0 | 1 | 0 | 0 |
| random_score_floor | control | 0 | not_selected | reference_only | 3 | 0.2513 +- 0.0045 | -0.0060 +- 0.0129 | 17.6636 +- 1.2400 | -0.0733 +- 0.0788 | -0.0060 +- 0.0129 | -6.7260 +- 1.5494 | 0 | 1 | 0 | 0 |
