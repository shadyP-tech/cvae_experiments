# Compatibility Decision Table

- uplift_reference_method: metadata_routing
- overall_tier: fail
- selected_method: metadata_routing
- min_improving_seeds: 2
- instability: std_threshold=0.05 sign_inconsistency_min_count=2

| method | role | eligible | decision | tier | n_seeds | top1 | spearman | mean_oracle_gap_pct | top1_uplift_vs_metadata | spearman_uplift_vs_metadata | gap_pct_reduction_vs_metadata | improving_seed_count | raw_instability_breach | instability_gate_applied | instability_breach |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| candidate_oracle_routing | diagnostic | 0 | not_selected | reference_only | 3 | 1.0000 +- 0.0000 | 1.0000 +- 0.0000 | 0.0000 +- 0.0000 | 0.6617 +- 0.0165 | 1.0000 +- 0.0000 | 32.1571 +- 1.9557 | 3 | 1 | 0 | 0 |
| hybrid_alpha_0.0 | diagnostic | 0 | not_selected | reference_only | 3 | 0.7542 +- 0.0409 | 0.7854 +- 0.0318 | 4.7262 +- 0.8333 | 0.4158 +- 0.0554 | 0.7854 +- 0.0318 | 27.4309 +- 1.2505 | 3 | 1 | 0 | 0 |
| hybrid_alpha_0.0_minmax | diagnostic | 0 | not_selected | reference_only | 3 | 0.7542 +- 0.0409 | 0.7854 +- 0.0318 | 4.7262 +- 0.8333 | 0.4158 +- 0.0554 | 0.7854 +- 0.0318 | 27.4309 +- 1.2505 | 3 | 1 | 0 | 0 |
| hybrid_alpha_0.1 | diagnostic | 0 | not_selected | reference_only | 3 | 0.7542 +- 0.0409 | 0.7854 +- 0.0318 | 4.7262 +- 0.8333 | 0.4158 +- 0.0554 | 0.7854 +- 0.0318 | 27.4309 +- 1.2505 | 3 | 1 | 0 | 0 |
| hybrid_alpha_0.1_minmax | diagnostic | 0 | not_selected | reference_only | 3 | 0.7542 +- 0.0409 | 0.7854 +- 0.0318 | 4.7262 +- 0.8333 | 0.4158 +- 0.0554 | 0.7854 +- 0.0318 | 27.4309 +- 1.2505 | 3 | 1 | 0 | 0 |
| hybrid_alpha_0.2 | diagnostic | 0 | not_selected | reference_only | 3 | 0.7542 +- 0.0409 | 0.7854 +- 0.0318 | 4.7262 +- 0.8333 | 0.4158 +- 0.0554 | 0.7854 +- 0.0318 | 27.4309 +- 1.2505 | 3 | 1 | 0 | 0 |
| hybrid_alpha_0.2_minmax | diagnostic | 0 | not_selected | reference_only | 3 | 0.7542 +- 0.0409 | 0.7854 +- 0.0318 | 4.7262 +- 0.8333 | 0.4158 +- 0.0554 | 0.7854 +- 0.0318 | 27.4309 +- 1.2505 | 3 | 1 | 0 | 0 |
| hybrid_alpha_0.3 | diagnostic | 0 | not_selected | reference_only | 3 | 0.7542 +- 0.0409 | 0.7854 +- 0.0318 | 4.7262 +- 0.8333 | 0.4158 +- 0.0554 | 0.7854 +- 0.0318 | 27.4309 +- 1.2505 | 3 | 1 | 0 | 0 |
| hybrid_alpha_0.3_minmax | diagnostic | 0 | not_selected | reference_only | 3 | 0.7542 +- 0.0409 | 0.7854 +- 0.0318 | 4.7262 +- 0.8333 | 0.4158 +- 0.0554 | 0.7854 +- 0.0318 | 27.4309 +- 1.2505 | 3 | 1 | 0 | 0 |
| hybrid_alpha_0.4 | diagnostic | 0 | not_selected | reference_only | 3 | 0.7542 +- 0.0409 | 0.7854 +- 0.0318 | 4.7262 +- 0.8333 | 0.4158 +- 0.0554 | 0.7854 +- 0.0318 | 27.4309 +- 1.2505 | 3 | 1 | 0 | 0 |
| hybrid_alpha_0.4_minmax | diagnostic | 0 | not_selected | reference_only | 3 | 0.7542 +- 0.0409 | 0.7854 +- 0.0318 | 4.7262 +- 0.8333 | 0.4158 +- 0.0554 | 0.7854 +- 0.0318 | 27.4309 +- 1.2505 | 3 | 1 | 0 | 0 |
| hybrid_alpha_0.5 | diagnostic | 0 | not_selected | reference_only | 3 | 0.7542 +- 0.0409 | 0.7854 +- 0.0318 | 4.7262 +- 0.8333 | 0.4158 +- 0.0554 | 0.7854 +- 0.0318 | 27.4309 +- 1.2505 | 3 | 1 | 0 | 0 |
| hybrid_alpha_0.5_minmax | diagnostic | 0 | not_selected | reference_only | 3 | 0.7542 +- 0.0409 | 0.7854 +- 0.0318 | 4.7262 +- 0.8333 | 0.4158 +- 0.0554 | 0.7854 +- 0.0318 | 27.4309 +- 1.2505 | 3 | 1 | 0 | 0 |
| hybrid_alpha_0.6 | diagnostic | 0 | not_selected | reference_only | 3 | 0.7542 +- 0.0409 | 0.7854 +- 0.0318 | 4.7262 +- 0.8333 | 0.4158 +- 0.0554 | 0.7854 +- 0.0318 | 27.4309 +- 1.2505 | 3 | 1 | 0 | 0 |
| hybrid_alpha_0.6_minmax | diagnostic | 0 | not_selected | reference_only | 3 | 0.7542 +- 0.0409 | 0.7854 +- 0.0318 | 4.7262 +- 0.8333 | 0.4158 +- 0.0554 | 0.7854 +- 0.0318 | 27.4309 +- 1.2505 | 3 | 1 | 0 | 0 |
| hybrid_alpha_0.7 | diagnostic | 0 | not_selected | reference_only | 3 | 0.7542 +- 0.0409 | 0.7854 +- 0.0318 | 4.7262 +- 0.8333 | 0.4158 +- 0.0554 | 0.7854 +- 0.0318 | 27.4309 +- 1.2505 | 3 | 1 | 0 | 0 |
| hybrid_alpha_0.7_minmax | diagnostic | 0 | not_selected | reference_only | 3 | 0.7542 +- 0.0409 | 0.7854 +- 0.0318 | 4.7262 +- 0.8333 | 0.4158 +- 0.0554 | 0.7854 +- 0.0318 | 27.4309 +- 1.2505 | 3 | 1 | 0 | 0 |
| hybrid_alpha_0.8 | diagnostic | 0 | not_selected | reference_only | 3 | 0.7542 +- 0.0409 | 0.7854 +- 0.0318 | 4.7262 +- 0.8333 | 0.4158 +- 0.0554 | 0.7854 +- 0.0318 | 27.4309 +- 1.2505 | 3 | 1 | 0 | 0 |
| hybrid_alpha_0.8_minmax | diagnostic | 0 | not_selected | reference_only | 3 | 0.7542 +- 0.0409 | 0.7854 +- 0.0318 | 4.7262 +- 0.8333 | 0.4158 +- 0.0554 | 0.7854 +- 0.0318 | 27.4309 +- 1.2505 | 3 | 1 | 0 | 0 |
| hybrid_alpha_0.9 | diagnostic | 0 | not_selected | reference_only | 3 | 0.7542 +- 0.0409 | 0.7854 +- 0.0318 | 4.7262 +- 0.8333 | 0.4158 +- 0.0554 | 0.7854 +- 0.0318 | 27.4309 +- 1.2505 | 3 | 1 | 0 | 0 |
| hybrid_alpha_0.9_minmax | diagnostic | 0 | not_selected | reference_only | 3 | 0.7542 +- 0.0409 | 0.7854 +- 0.0318 | 4.7262 +- 0.8333 | 0.4158 +- 0.0554 | 0.7854 +- 0.0318 | 27.4309 +- 1.2505 | 3 | 1 | 0 | 0 |
| hybrid_alpha_1.0 | diagnostic | 0 | not_selected | reference_only | 3 | 0.3383 +- 0.0165 | 0.0000 +- 0.0000 | 32.1571 +- 1.9557 | 0.0000 +- 0.0000 | 0.0000 +- 0.0000 | 0.0000 +- 0.0000 | 0 | 0 | 0 | 0 |
| hybrid_alpha_1.0_minmax | diagnostic | 0 | not_selected | reference_only | 3 | 0.3383 +- 0.0165 | 0.0000 +- 0.0000 | 32.1571 +- 1.9557 | 0.0000 +- 0.0000 | 0.0000 +- 0.0000 | 0.0000 +- 0.0000 | 0 | 0 | 0 | 0 |
| latent_wasserstein_routing | diagnostic | 0 | not_selected | reference_only | 3 | 0.7542 +- 0.0409 | 0.7854 +- 0.0318 | 4.7262 +- 0.8333 | 0.4158 +- 0.0554 | 0.7854 +- 0.0318 | 27.4309 +- 1.2505 | 3 | 1 | 0 | 0 |
| linear_regressor | learned | 1 | not_selected | fail | 3 | 0.2863 +- 0.0422 | 0.0700 +- 0.2343 | 16.1687 +- 1.1636 | -0.0521 +- 0.0552 | 0.0700 +- 0.2343 | 15.9884 +- 0.9594 | 0 | 1 | 1 | 1 |
| metadata_routing | baseline | 0 | baseline_reference | baseline | 3 | 0.3383 +- 0.0165 | 0.0000 +- 0.0000 | 32.1571 +- 1.9557 | 0.0000 +- 0.0000 | 0.0000 +- 0.0000 | 0.0000 +- 0.0000 | 0 | 0 | 0 | 0 |
| mlp_regressor | learned | 1 | not_selected | fail | 3 | 0.5067 +- 0.0438 | 0.5340 +- 0.0540 | 9.1688 +- 0.7658 | 0.1683 +- 0.0590 | 0.5340 +- 0.0540 | 22.9883 +- 1.3991 | 3 | 1 | 1 | 1 |
| pairwise_ranker_combined | learned | 1 | not_selected | fail | 3 | 0.6425 +- 0.1333 | 0.7544 +- 0.0781 | 6.6118 +- 2.4088 | 0.3042 +- 0.1463 | 0.7544 +- 0.0781 | 25.5453 +- 1.3801 | 3 | 1 | 1 | 1 |
| pairwise_ranker_latent_only | learned | 1 | not_selected | fail | 3 | 0.2879 +- 0.0525 | 0.5737 +- 0.0338 | 15.8687 +- 0.6164 | -0.0504 +- 0.0684 | 0.5737 +- 0.0338 | 16.2884 +- 1.3685 | 1 | 1 | 1 | 1 |
| pairwise_ranker_metadata_only | learned | 1 | not_selected | fail | 3 | 0.3642 +- 0.0731 | 0.2700 +- 0.0966 | 13.9504 +- 3.1795 | 0.0258 +- 0.0569 | 0.2700 +- 0.0966 | 18.2067 +- 3.9583 | 1 | 1 | 1 | 1 |
| random_rank_floor | control | 0 | not_selected | reference_only | 3 | 0.2979 +- 0.0958 | -0.0962 +- 0.3240 | 25.2816 +- 6.4677 | -0.0404 +- 0.1114 | -0.0962 +- 0.3240 | 6.8755 +- 4.9112 | 1 | 1 | 0 | 0 |
| random_score_floor | control | 0 | not_selected | reference_only | 3 | 0.3458 +- 0.0097 | 0.0335 +- 0.0158 | 24.7239 +- 0.9039 | 0.0075 +- 0.0239 | 0.0335 +- 0.0158 | 7.4332 +- 2.5500 | 2 | 1 | 0 | 0 |
