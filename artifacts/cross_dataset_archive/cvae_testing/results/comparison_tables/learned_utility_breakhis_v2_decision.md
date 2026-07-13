# Compatibility Decision Table

- uplift_reference_method: metadata_routing
- overall_tier: fail
- selected_method: metadata_routing
- min_improving_seeds: 2
- instability: std_threshold=0.05 sign_inconsistency_min_count=2

| method | role | eligible | decision | tier | n_seeds | top1 | spearman | mean_oracle_gap_pct | top1_uplift_vs_metadata | spearman_uplift_vs_metadata | gap_pct_reduction_vs_metadata | improving_seed_count | raw_instability_breach | instability_gate_applied | instability_breach |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| candidate_oracle_routing | diagnostic | 0 | not_selected | reference_only | 3 | 1.0000 +- 0.0000 | 1.0000 +- 0.0000 | 0.0000 +- 0.0000 | 0.6867 +- 0.0109 | 1.0000 +- 0.0000 | 14.6738 +- 1.3758 | 3 | 1 | 0 | 0 |
| hybrid_alpha_0.0 | diagnostic | 0 | not_selected | reference_only | 3 | 0.6689 +- 0.0301 | 0.6661 +- 0.0209 | 3.8494 +- 0.2464 | 0.3556 +- 0.0275 | 0.6661 +- 0.0209 | 10.8244 +- 1.1469 | 3 | 1 | 0 | 0 |
| hybrid_alpha_0.0_minmax | diagnostic | 0 | not_selected | reference_only | 3 | 0.6689 +- 0.0301 | 0.6661 +- 0.0209 | 3.8494 +- 0.2464 | 0.3556 +- 0.0275 | 0.6661 +- 0.0209 | 10.8244 +- 1.1469 | 3 | 1 | 0 | 0 |
| hybrid_alpha_0.1 | diagnostic | 0 | not_selected | reference_only | 3 | 0.6689 +- 0.0301 | 0.6661 +- 0.0209 | 3.8494 +- 0.2464 | 0.3556 +- 0.0275 | 0.6661 +- 0.0209 | 10.8244 +- 1.1469 | 3 | 1 | 0 | 0 |
| hybrid_alpha_0.1_minmax | diagnostic | 0 | not_selected | reference_only | 3 | 0.6689 +- 0.0301 | 0.6661 +- 0.0209 | 3.8494 +- 0.2464 | 0.3556 +- 0.0275 | 0.6661 +- 0.0209 | 10.8244 +- 1.1469 | 3 | 1 | 0 | 0 |
| hybrid_alpha_0.2 | diagnostic | 0 | not_selected | reference_only | 3 | 0.6689 +- 0.0301 | 0.6661 +- 0.0209 | 3.8494 +- 0.2464 | 0.3556 +- 0.0275 | 0.6661 +- 0.0209 | 10.8244 +- 1.1469 | 3 | 1 | 0 | 0 |
| hybrid_alpha_0.2_minmax | diagnostic | 0 | not_selected | reference_only | 3 | 0.6689 +- 0.0301 | 0.6661 +- 0.0209 | 3.8494 +- 0.2464 | 0.3556 +- 0.0275 | 0.6661 +- 0.0209 | 10.8244 +- 1.1469 | 3 | 1 | 0 | 0 |
| hybrid_alpha_0.3 | diagnostic | 0 | not_selected | reference_only | 3 | 0.6689 +- 0.0301 | 0.6661 +- 0.0209 | 3.8494 +- 0.2464 | 0.3556 +- 0.0275 | 0.6661 +- 0.0209 | 10.8244 +- 1.1469 | 3 | 1 | 0 | 0 |
| hybrid_alpha_0.3_minmax | diagnostic | 0 | not_selected | reference_only | 3 | 0.6689 +- 0.0301 | 0.6661 +- 0.0209 | 3.8494 +- 0.2464 | 0.3556 +- 0.0275 | 0.6661 +- 0.0209 | 10.8244 +- 1.1469 | 3 | 1 | 0 | 0 |
| hybrid_alpha_0.4 | diagnostic | 0 | not_selected | reference_only | 3 | 0.6689 +- 0.0301 | 0.6661 +- 0.0209 | 3.8494 +- 0.2464 | 0.3556 +- 0.0275 | 0.6661 +- 0.0209 | 10.8244 +- 1.1469 | 3 | 1 | 0 | 0 |
| hybrid_alpha_0.4_minmax | diagnostic | 0 | not_selected | reference_only | 3 | 0.6689 +- 0.0301 | 0.6661 +- 0.0209 | 3.8494 +- 0.2464 | 0.3556 +- 0.0275 | 0.6661 +- 0.0209 | 10.8244 +- 1.1469 | 3 | 1 | 0 | 0 |
| hybrid_alpha_0.5 | diagnostic | 0 | not_selected | reference_only | 3 | 0.6689 +- 0.0301 | 0.6661 +- 0.0209 | 3.8494 +- 0.2464 | 0.3556 +- 0.0275 | 0.6661 +- 0.0209 | 10.8244 +- 1.1469 | 3 | 1 | 0 | 0 |
| hybrid_alpha_0.5_minmax | diagnostic | 0 | not_selected | reference_only | 3 | 0.6689 +- 0.0301 | 0.6661 +- 0.0209 | 3.8494 +- 0.2464 | 0.3556 +- 0.0275 | 0.6661 +- 0.0209 | 10.8244 +- 1.1469 | 3 | 1 | 0 | 0 |
| hybrid_alpha_0.6 | diagnostic | 0 | not_selected | reference_only | 3 | 0.6689 +- 0.0301 | 0.6661 +- 0.0209 | 3.8494 +- 0.2464 | 0.3556 +- 0.0275 | 0.6661 +- 0.0209 | 10.8244 +- 1.1469 | 3 | 1 | 0 | 0 |
| hybrid_alpha_0.6_minmax | diagnostic | 0 | not_selected | reference_only | 3 | 0.6689 +- 0.0301 | 0.6661 +- 0.0209 | 3.8494 +- 0.2464 | 0.3556 +- 0.0275 | 0.6661 +- 0.0209 | 10.8244 +- 1.1469 | 3 | 1 | 0 | 0 |
| hybrid_alpha_0.7 | diagnostic | 0 | not_selected | reference_only | 3 | 0.6689 +- 0.0301 | 0.6661 +- 0.0209 | 3.8494 +- 0.2464 | 0.3556 +- 0.0275 | 0.6661 +- 0.0209 | 10.8244 +- 1.1469 | 3 | 1 | 0 | 0 |
| hybrid_alpha_0.7_minmax | diagnostic | 0 | not_selected | reference_only | 3 | 0.6689 +- 0.0301 | 0.6661 +- 0.0209 | 3.8494 +- 0.2464 | 0.3556 +- 0.0275 | 0.6661 +- 0.0209 | 10.8244 +- 1.1469 | 3 | 1 | 0 | 0 |
| hybrid_alpha_0.8 | diagnostic | 0 | not_selected | reference_only | 3 | 0.6689 +- 0.0301 | 0.6661 +- 0.0209 | 3.8494 +- 0.2464 | 0.3556 +- 0.0275 | 0.6661 +- 0.0209 | 10.8244 +- 1.1469 | 3 | 1 | 0 | 0 |
| hybrid_alpha_0.8_minmax | diagnostic | 0 | not_selected | reference_only | 3 | 0.6689 +- 0.0301 | 0.6661 +- 0.0209 | 3.8494 +- 0.2464 | 0.3556 +- 0.0275 | 0.6661 +- 0.0209 | 10.8244 +- 1.1469 | 3 | 1 | 0 | 0 |
| hybrid_alpha_0.9 | diagnostic | 0 | not_selected | reference_only | 3 | 0.6689 +- 0.0301 | 0.6661 +- 0.0209 | 3.8494 +- 0.2464 | 0.3556 +- 0.0275 | 0.6661 +- 0.0209 | 10.8244 +- 1.1469 | 3 | 1 | 0 | 0 |
| hybrid_alpha_0.9_minmax | diagnostic | 0 | not_selected | reference_only | 3 | 0.6689 +- 0.0301 | 0.6661 +- 0.0209 | 3.8494 +- 0.2464 | 0.3556 +- 0.0275 | 0.6661 +- 0.0209 | 10.8244 +- 1.1469 | 3 | 1 | 0 | 0 |
| hybrid_alpha_1.0 | diagnostic | 0 | not_selected | reference_only | 3 | 0.3133 +- 0.0109 | 0.0000 +- 0.0000 | 14.6738 +- 1.3758 | 0.0000 +- 0.0000 | 0.0000 +- 0.0000 | 0.0000 +- 0.0000 | 0 | 0 | 0 | 0 |
| hybrid_alpha_1.0_minmax | diagnostic | 0 | not_selected | reference_only | 3 | 0.3133 +- 0.0109 | 0.0000 +- 0.0000 | 14.6738 +- 1.3758 | 0.0000 +- 0.0000 | 0.0000 +- 0.0000 | 0.0000 +- 0.0000 | 0 | 0 | 0 | 0 |
| latent_wasserstein_routing | diagnostic | 0 | not_selected | reference_only | 3 | 0.6689 +- 0.0301 | 0.6661 +- 0.0209 | 3.8494 +- 0.2464 | 0.3556 +- 0.0275 | 0.6661 +- 0.0209 | 10.8244 +- 1.1469 | 3 | 1 | 0 | 0 |
| linear_regressor | learned | 1 | not_selected | fail | 3 | 0.3133 +- 0.0237 | 0.2828 +- 0.0170 | 9.2144 +- 0.6558 | 0.0000 +- 0.0331 | 0.2828 +- 0.0170 | 5.4594 +- 0.7597 | 2 | 1 | 1 | 1 |
| metadata_routing | baseline | 0 | baseline_reference | baseline | 3 | 0.3133 +- 0.0109 | 0.0000 +- 0.0000 | 14.6738 +- 1.3758 | 0.0000 +- 0.0000 | 0.0000 +- 0.0000 | 0.0000 +- 0.0000 | 0 | 0 | 0 | 0 |
| mlp_regressor | learned | 1 | not_selected | fail | 3 | 0.4367 +- 0.0519 | 0.3806 +- 0.0604 | 7.1335 +- 0.4678 | 0.1233 +- 0.0536 | 0.3806 +- 0.0604 | 7.5403 +- 1.8430 | 3 | 1 | 1 | 1 |
| pairwise_ranker_combined | learned | 1 | not_selected | fail | 3 | 0.3756 +- 0.0796 | 0.4806 +- 0.0303 | 8.0281 +- 1.0964 | 0.0622 +- 0.0903 | 0.4806 +- 0.0303 | 6.6457 +- 1.6930 | 2 | 1 | 1 | 1 |
| pairwise_ranker_latent_only | learned | 1 | not_selected | fail | 3 | 0.3133 +- 0.0237 | 0.3933 +- 0.0200 | 9.2144 +- 0.6558 | 0.0000 +- 0.0331 | 0.3933 +- 0.0200 | 5.4594 +- 0.7597 | 2 | 1 | 1 | 1 |
| pairwise_ranker_metadata_only | learned | 1 | not_selected | fail | 3 | 0.5167 +- 0.0242 | 0.3600 +- 0.1276 | 5.9617 +- 0.3763 | 0.2033 +- 0.0347 | 0.3600 +- 0.1276 | 8.7121 +- 1.0645 | 3 | 1 | 1 | 1 |
| random_rank_floor | control | 0 | not_selected | reference_only | 3 | 0.4378 +- 0.1634 | 0.0033 +- 0.3448 | 9.7967 +- 5.7570 | 0.1244 +- 0.1711 | 0.0033 +- 0.3448 | 4.8772 +- 4.7799 | 2 | 1 | 0 | 0 |
| random_score_floor | control | 0 | not_selected | reference_only | 3 | 0.3267 +- 0.0245 | -0.0167 +- 0.0136 | 12.8035 +- 1.2459 | 0.0133 +- 0.0136 | -0.0167 +- 0.0136 | 1.8703 +- 0.1440 | 0 | 1 | 0 | 0 |
