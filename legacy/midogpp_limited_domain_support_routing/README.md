# Legacy MIDOG++ Limited-Domain Support Routing

This folder archives the old MIDOG++ scanner support-NELBO experiment surface.

The archived runs are not thesis-facing because they used a limited-domain MIDOG++
setup and did not cover the full dataset. In particular, the top1, Spearman, and
oracle-gap values under `cvae_testing/results/comparison_tables/` came from that
legacy setup and should not be interpreted as evidence for the current full
MIDOG++ dataset contract.

Archived contents:

- `cvae_testing/configs/experiments/midogpp/`: old experiment configs.
- `cvae_support_routing/configs/experiments/midogpp/`: old support-routing config.
- `cvae_support_routing/scripts/preflight/`: old scanner preflight script.
- `cvae_support_routing/scripts/run/`: old MIDOG++ support-NELBO run script.
- `cvae_testing/scripts/`: old patch-manifest helper.
- `cvae_testing/archived_tests/`: tests for the archived helper/config surface.
- `cvae_testing/results/comparison_tables/`: old preflight and support-routing artifacts.

Current MIDOG++ work should use the dataset-contract surface under
`datasets/midogpp/` and should regenerate any routing evidence on the full intended
sample/domain protocol before making thesis claims.
