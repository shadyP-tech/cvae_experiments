from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .covariance_prior import load_covariance_prior_config, run_covariance_prior_confirmation
from .covariance_shrinkage import load_covariance_shrinkage_config, run_covariance_shrinkage_stability
from .covariance_viability import load_covariance_viability_config, run_covariance_prior_viability_audit
from .decentralized_k16_gmm_prior import (
    load_decentralized_k16_gmm_prior_config,
    run_decentralized_k16_gmm_prior,
)
from .decentralized_adaptive_gmm_prior import (
    load_decentralized_adaptive_gmm_prior_config,
    run_decentralized_adaptive_gmm_prior,
)
from .decentralized_component_union_prior import (
    load_decentralized_component_union_prior_config,
    run_decentralized_component_union_prior,
)
from .component_union_mass_bagged import (
    load_mass_bagged_component_union_config,
    run_mass_bagged_component_union,
)
from .decentralized_pruned_adaptive_equal_all4_prior import (
    load_pruned_adaptive_equal_all4_config,
    run_pruned_adaptive_equal_all4_confirmation,
)
from .decentralized_reliability_weighted_gmm_prior import (
    load_decentralized_reliability_weighted_gmm_prior_config,
    run_decentralized_reliability_weighted_gmm_prior,
)
from .decentralized_reliability_top3_gmm_prior import (
    load_decentralized_reliability_top3_gmm_prior_config,
    run_decentralized_reliability_top3_gmm_prior,
)
from .decentralized_source_inner_transfer_top3_gmm_prior import (
    load_decentralized_source_inner_transfer_top3_gmm_prior_config,
    run_decentralized_source_inner_transfer_top3_gmm_prior,
)
from .decentralized_support_nelbo_reliability_gmm_prior import (
    load_decentralized_support_nelbo_reliability_gmm_prior_config,
    run_decentralized_support_nelbo_reliability_gmm_prior,
)
from .decentralized_support8_top3_tau05_gmm_prior import (
    load_decentralized_support8_top3_tau05_gmm_prior_config,
    run_decentralized_support8_top3_tau05_gmm_prior,
)
from .support_calibrated_component_union_prior import (
    load_support_calibrated_component_union_config,
    run_support_calibrated_component_union_prior,
)
from .paired_dense_all4_reliability_confirmation import (
    load_paired_dense_all4_reliability_config,
    run_paired_dense_all4_reliability_confirmation,
)
from .paired_component_coverage_audit import (
    load_paired_component_coverage_audit_config,
    run_paired_component_coverage_audit,
)
from .pipeline import run_artifact_contract_smoke, run_real_cache_backed, run_synthetic_smoke
from .preservation import load_preservation_config, run_preservation_diagnosis
from .preservation_repair import load_repair_config, run_preservation_repair
from .preservation_sampling import load_sampling_config, run_preservation_sampling
from .prior_calibration import load_prior_calibration_config, run_prior_calibration
from .source_union_balanced_gmm_prior import (
    load_source_union_balanced_gmm_prior_config,
    run_source_union_balanced_gmm_prior,
)
from .source_union_gmm_prior import load_source_union_gmm_prior_config, run_source_union_gmm_prior
from .source_union_k24_gmm_prior import load_source_union_k24_gmm_prior_config, run_source_union_k24_gmm_prior
from .source_inner_validated_dense_component_hybrid import (
    load_source_inner_validated_hybrid_config,
    run_source_inner_validated_dense_component_hybrid,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Virchow2-CVAE rebuild runner.")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-config", help="Validate a locked rebuild config.")
    validate.add_argument("--config", required=True)

    smoke = sub.add_parser("smoke-artifacts", help="Write empty artifact-contract outputs.")
    smoke.add_argument("--config", required=True)
    smoke.add_argument("--artifact-root", default=None)

    run = sub.add_parser("run", help="Run the rebuild pipeline or a synthetic smoke run.")
    run.add_argument("--config", required=True)
    run.add_argument("--artifact-root", default=None)
    run.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="Run a tiny end-to-end synthetic train/routing/downstream smoke.",
    )

    diagnose = sub.add_parser("diagnose-preservation", help="Run the Virchow2-CVAE preservation diagnosis.")
    diagnose.add_argument("--config", required=True)
    diagnose.add_argument("--artifact-root", default=None)

    repair = sub.add_parser("diagnose-preservation-repair", help="Run the Virchow2-CVAE preservation repair diagnosis.")
    repair.add_argument("--config", required=True)
    repair.add_argument("--artifact-root", default=None)

    sampling = sub.add_parser("diagnose-preservation-sampling", help="Run the Virchow2-CVAE PCA64 sampling continuation.")
    sampling.add_argument("--config", required=True)
    sampling.add_argument("--artifact-root", default=None)

    prior = sub.add_parser("diagnose-latent-prior-calibration", help="Run the Virchow2-CVAE latent prior calibration diagnostic.")
    prior.add_argument("--config", required=True)
    prior.add_argument("--artifact-root", default=None)

    cov_prior = sub.add_parser(
        "diagnose-covariance-prior-confirmation",
        help="Run the Virchow2-CVAE covariance prior confirmation diagnostic.",
    )
    cov_prior.add_argument("--config", required=True)
    cov_prior.add_argument("--artifact-root", default=None)

    cov_viability = sub.add_parser(
        "diagnose-covariance-prior-viability",
        help="Run the Virchow2-CVAE covariance prior viability audit.",
    )
    cov_viability.add_argument("--config", required=True)
    cov_viability.add_argument("--artifact-root", default=None)

    cov_shrinkage = sub.add_parser(
        "diagnose-covariance-shrinkage-stability",
        help="Run the Virchow2-CVAE covariance shrinkage stability diagnostic.",
    )
    cov_shrinkage.add_argument("--config", required=True)
    cov_shrinkage.add_argument("--artifact-root", default=None)

    source_union_gmm = sub.add_parser(
        "diagnose-source-union-gmm-prior",
        help="Run the Virchow2-CVAE source-union GMM prior diagnostic.",
    )
    source_union_gmm.add_argument("--config", required=True)
    source_union_gmm.add_argument("--artifact-root", default=None)

    source_union_balanced_gmm = sub.add_parser(
        "diagnose-source-union-balanced-gmm-prior",
        help="Run the Virchow2-CVAE source-union center-balanced GMM prior diagnostic.",
    )
    source_union_balanced_gmm.add_argument("--config", required=True)
    source_union_balanced_gmm.add_argument("--artifact-root", default=None)

    source_union_k24_gmm = sub.add_parser(
        "diagnose-source-union-k24-gmm-prior",
        help="Run the Virchow2-CVAE source-union K24 GMM prior locked follow-up.",
    )
    source_union_k24_gmm.add_argument("--config", required=True)
    source_union_k24_gmm.add_argument("--artifact-root", default=None)

    decentralized_k16_gmm = sub.add_parser(
        "diagnose-decentralized-k16-gmm-prior",
        help="Run the Virchow2-CVAE decentralized K16 summary-composition preservation test.",
    )
    decentralized_k16_gmm.add_argument("--config", required=True)
    decentralized_k16_gmm.add_argument("--artifact-root", default=None)

    decentralized_adaptive_gmm = sub.add_parser(
        "diagnose-decentralized-adaptive-gmm-prior",
        help="Run the Virchow2-CVAE adaptive source-local latent summary preservation test.",
    )
    decentralized_adaptive_gmm.add_argument("--config", required=True)
    decentralized_adaptive_gmm.add_argument("--artifact-root", default=None)

    decentralized_component_union = sub.add_parser(
        "diagnose-decentralized-component-union-prior",
        help="Run the Virchow2-CVAE decentralized component-level prior composition audit.",
    )
    decentralized_component_union.add_argument("--config", required=True)
    decentralized_component_union.add_argument("--artifact-root", default=None)

    decentralized_component_union_shrink050 = sub.add_parser(
        "diagnose-decentralized-component-union-reliability-shrink050",
        help="Run the Virchow2-CVAE component-union reliability shrink050 confirmation audit.",
    )
    decentralized_component_union_shrink050.add_argument("--config", required=True)
    decentralized_component_union_shrink050.add_argument("--artifact-root", default=None)

    decentralized_component_union_mass_bagged = sub.add_parser(
        "diagnose-decentralized-component-union-mass-bagged",
        help="Run the Virchow2-CVAE mass-uncertainty bagged component-union audit.",
    )
    decentralized_component_union_mass_bagged.add_argument("--config", required=True)
    decentralized_component_union_mass_bagged.add_argument("--artifact-root", default=None)

    decentralized_pruned_equal_all4 = sub.add_parser(
        "diagnose-decentralized-pruned-adaptive-equal-all4-prior",
        help="Run the pruned adaptive equal-all4 late-geometric confirmation test.",
    )
    decentralized_pruned_equal_all4.add_argument("--config", required=True)
    decentralized_pruned_equal_all4.add_argument("--artifact-root", default=None)

    decentralized_reliability_gmm = sub.add_parser(
        "diagnose-decentralized-reliability-weighted-gmm-prior",
        help="Run the Virchow2-CVAE source-local reliability-weighted decentralized composition test.",
    )
    decentralized_reliability_gmm.add_argument("--config", required=True)
    decentralized_reliability_gmm.add_argument("--artifact-root", default=None)

    decentralized_reliability_top3_gmm = sub.add_parser(
        "diagnose-decentralized-reliability-top3-gmm-prior",
        help="Run the locked D1.4 source-local reliability top-3 decentralized composition test.",
    )
    decentralized_reliability_top3_gmm.add_argument("--config", required=True)
    decentralized_reliability_top3_gmm.add_argument("--artifact-root", default=None)

    decentralized_source_inner_transfer_gmm = sub.add_parser(
        "diagnose-decentralized-source-inner-transfer-top3-gmm-prior",
        help="Run the locked D1.5 source-inner off-diagonal transfer drop-one confirmation test.",
    )
    decentralized_source_inner_transfer_gmm.add_argument("--config", required=True)
    decentralized_source_inner_transfer_gmm.add_argument("--artifact-root", default=None)

    decentralized_support_nelbo_reliability_gmm = sub.add_parser(
        "diagnose-decentralized-support-nelbo-reliability-gmm-prior",
        help="Run the Virchow2-CVAE support-NELBO x reliability decentralized composition test.",
    )
    decentralized_support_nelbo_reliability_gmm.add_argument("--config", required=True)
    decentralized_support_nelbo_reliability_gmm.add_argument("--artifact-root", default=None)

    decentralized_support8_top3_tau05_gmm = sub.add_parser(
        "diagnose-decentralized-support8-top3-tau05-gmm-prior",
        help="Run the locked D1.3.1 support-size-8 top-3 tau-0.5 confirmation test.",
    )
    decentralized_support8_top3_tau05_gmm.add_argument("--config", required=True)
    decentralized_support8_top3_tau05_gmm.add_argument("--artifact-root", default=None)

    support_calibrated_component_union = sub.add_parser(
        "diagnose-support-calibrated-component-union-prior",
        help="Run the support-calibrated component-union prior audit.",
    )
    support_calibrated_component_union.add_argument("--config", required=True)
    support_calibrated_component_union.add_argument("--artifact-root", default=None)

    paired_dense_all4_reliability = sub.add_parser(
        "diagnose-paired-dense-all4-reliability",
        help="Run the paired dense-all4 source-only reliability confirmation audit.",
    )
    paired_dense_all4_reliability.add_argument("--config", required=True)
    paired_dense_all4_reliability.add_argument("--artifact-root", default=None)

    paired_component_coverage = sub.add_parser(
        "diagnose-paired-component-coverage-audit",
        help="Run the paired dense-all4 component coverage sampling-fidelity audit.",
    )
    paired_component_coverage.add_argument("--config", required=True)
    paired_component_coverage.add_argument("--artifact-root", default=None)

    source_inner_hybrid = sub.add_parser(
        "diagnose-source-inner-validated-dense-component-hybrid",
        help="Run the source-inner validated dense/component binary-gate confirmation audit.",
    )
    source_inner_hybrid.add_argument("--config", required=True)
    source_inner_hybrid.add_argument("--artifact-root", default=None)

    args = parser.parse_args(argv)
    if args.command == "diagnose-source-inner-validated-dense-component-hybrid":
        cfg = load_source_inner_validated_hybrid_config(args.config)
        root = run_source_inner_validated_dense_component_hybrid(
            cfg,
            artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        )
        print(root)
        return 0
    if args.command == "diagnose-paired-component-coverage-audit":
        cfg = load_paired_component_coverage_audit_config(args.config)
        root = run_paired_component_coverage_audit(
            cfg,
            artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        )
        print(root)
        return 0
    if args.command == "diagnose-paired-dense-all4-reliability":
        cfg = load_paired_dense_all4_reliability_config(args.config)
        root = run_paired_dense_all4_reliability_confirmation(
            cfg,
            artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        )
        print(root)
        return 0
    if args.command == "diagnose-decentralized-support8-top3-tau05-gmm-prior":
        cfg = load_decentralized_support8_top3_tau05_gmm_prior_config(args.config)
        root = run_decentralized_support8_top3_tau05_gmm_prior(
            cfg,
            artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        )
        print(root)
        return 0
    if args.command == "diagnose-support-calibrated-component-union-prior":
        cfg = load_support_calibrated_component_union_config(args.config)
        root = run_support_calibrated_component_union_prior(
            cfg,
            artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        )
        print(root)
        return 0
    if args.command == "diagnose-decentralized-source-inner-transfer-top3-gmm-prior":
        cfg = load_decentralized_source_inner_transfer_top3_gmm_prior_config(args.config)
        root = run_decentralized_source_inner_transfer_top3_gmm_prior(
            cfg,
            artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        )
        print(root)
        return 0
    if args.command == "diagnose-decentralized-reliability-top3-gmm-prior":
        cfg = load_decentralized_reliability_top3_gmm_prior_config(args.config)
        root = run_decentralized_reliability_top3_gmm_prior(
            cfg,
            artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        )
        print(root)
        return 0
    if args.command == "diagnose-decentralized-support-nelbo-reliability-gmm-prior":
        cfg = load_decentralized_support_nelbo_reliability_gmm_prior_config(args.config)
        root = run_decentralized_support_nelbo_reliability_gmm_prior(
            cfg,
            artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        )
        print(root)
        return 0
    if args.command == "diagnose-decentralized-reliability-weighted-gmm-prior":
        cfg = load_decentralized_reliability_weighted_gmm_prior_config(args.config)
        root = run_decentralized_reliability_weighted_gmm_prior(
            cfg,
            artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        )
        print(root)
        return 0
    if args.command == "diagnose-decentralized-component-union-prior":
        cfg = load_decentralized_component_union_prior_config(args.config)
        root = run_decentralized_component_union_prior(
            cfg,
            artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        )
        print(root)
        return 0
    if args.command == "diagnose-decentralized-component-union-reliability-shrink050":
        cfg = load_decentralized_component_union_prior_config(args.config)
        root = run_decentralized_component_union_prior(
            cfg,
            artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        )
        print(root)
        return 0
    if args.command == "diagnose-decentralized-component-union-mass-bagged":
        cfg = load_mass_bagged_component_union_config(args.config)
        root = run_mass_bagged_component_union(
            cfg,
            artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        )
        print(root)
        return 0
    if args.command == "diagnose-decentralized-pruned-adaptive-equal-all4-prior":
        cfg = load_pruned_adaptive_equal_all4_config(args.config)
        root = run_pruned_adaptive_equal_all4_confirmation(
            cfg,
            artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        )
        print(root)
        return 0
    if args.command == "diagnose-decentralized-adaptive-gmm-prior":
        cfg = load_decentralized_adaptive_gmm_prior_config(args.config)
        root = run_decentralized_adaptive_gmm_prior(
            cfg,
            artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        )
        print(root)
        return 0
    if args.command == "diagnose-decentralized-k16-gmm-prior":
        cfg = load_decentralized_k16_gmm_prior_config(args.config)
        root = run_decentralized_k16_gmm_prior(
            cfg,
            artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        )
        print(root)
        return 0
    if args.command == "diagnose-source-union-k24-gmm-prior":
        cfg = load_source_union_k24_gmm_prior_config(args.config)
        root = run_source_union_k24_gmm_prior(
            cfg,
            artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        )
        print(root)
        return 0
    if args.command == "diagnose-source-union-balanced-gmm-prior":
        cfg = load_source_union_balanced_gmm_prior_config(args.config)
        root = run_source_union_balanced_gmm_prior(
            cfg,
            artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        )
        print(root)
        return 0
    if args.command == "diagnose-source-union-gmm-prior":
        cfg = load_source_union_gmm_prior_config(args.config)
        root = run_source_union_gmm_prior(
            cfg,
            artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        )
        print(root)
        return 0
    if args.command == "diagnose-covariance-shrinkage-stability":
        cfg = load_covariance_shrinkage_config(args.config)
        root = run_covariance_shrinkage_stability(
            cfg,
            artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        )
        print(root)
        return 0
    if args.command == "diagnose-covariance-prior-viability":
        cfg = load_covariance_viability_config(args.config)
        root = run_covariance_prior_viability_audit(
            cfg,
            artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        )
        print(root)
        return 0
    if args.command == "diagnose-covariance-prior-confirmation":
        cfg = load_covariance_prior_config(args.config)
        root = run_covariance_prior_confirmation(
            cfg,
            artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        )
        print(root)
        return 0
    if args.command == "diagnose-latent-prior-calibration":
        cfg = load_prior_calibration_config(args.config)
        root = run_prior_calibration(
            cfg,
            artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        )
        print(root)
        return 0
    if args.command == "diagnose-preservation-sampling":
        cfg = load_sampling_config(args.config)
        root = run_preservation_sampling(
            cfg,
            artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        )
        print(root)
        return 0
    if args.command == "diagnose-preservation-repair":
        cfg = load_repair_config(args.config)
        root = run_preservation_repair(
            cfg,
            artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        )
        print(root)
        return 0
    if args.command == "diagnose-preservation":
        cfg = load_preservation_config(args.config)
        root = run_preservation_diagnosis(
            cfg,
            artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        )
        print(root)
        return 0
    cfg = load_config(args.config)
    if args.command == "validate-config":
        print(f"OK: {cfg.name}")
        return 0
    if args.command == "smoke-artifacts":
        root = run_artifact_contract_smoke(
            cfg,
            artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        )
        print(root)
        return 0
    if args.command == "run":
        if args.synthetic_smoke:
            root = run_synthetic_smoke(
                cfg,
                artifact_root=Path(args.artifact_root) if args.artifact_root else None,
            )
            print(root)
            return 0
        root = run_real_cache_backed(
            cfg,
            artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        )
        print(root)
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
