"""CLI for Stage-70 reservation, cache, authorization, and descriptive scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..protocol import ProtocolError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="surface", required=True)

    reservation = sub.add_parser(
        "reserve-consumed-test",
        help="Freeze the label-sealed consumed-test reservation before extraction.",
    )
    reservation.add_argument("--config", required=True)
    reservation.add_argument("--artifact-root", default=None)

    cache = sub.add_parser(
        "build-descriptive-test-cache",
        help="Build the label-blind Virchow2 test cache from the reservation.",
    )
    cache.add_argument("--config", required=True)
    cache.add_argument("--validate-only", action="store_true")

    authorization = sub.add_parser(
        "authorize-prediction",
        help="Bind the completed cache and authorize frozen-policy prediction only.",
    )
    authorization.add_argument("--config", required=True)
    authorization.add_argument("--artifact-root", default=None)

    evaluation = sub.add_parser(
        "evaluate-descriptively",
        help="Seal all frozen-policy predictions before scoring consumed test labels.",
    )
    evaluation.add_argument("--config", required=True)

    validation = sub.add_parser(
        "validate-descriptive-bundle",
        help="Independently validate a completed Stage-70 descriptive bundle.",
    )
    validation.add_argument("--artifact-root", required=True)

    residual_topup_fresh = sub.add_parser(
        "evaluate-residual-topup-fresh",
        help=(
            "Run the fresh fixed B/U/G/S residual-top-up evaluation after "
            "all policy, reservation, cache, and manifest gates are active."
        ),
    )
    residual_topup_fresh.add_argument("--config", required=True)
    residual_topup_fresh.add_argument(
        "--enable-local-scratch",
        action="store_true",
        help=(
            "Use the predeclared workstation-local scratch for resumable source "
            "blocks; canonical outputs remain hash-validated workspace artifacts."
        ),
    )
    utility_aligned_fresh = sub.add_parser(
        "evaluate-utility-aligned-residual-fresh",
        help=(
            "Evaluate the frozen utility-aligned residual-tail policy on its "
            "fresh, case-disjoint MIDOG++ target surface."
        ),
    )
    utility_aligned_fresh.add_argument("--config", required=True)
    utility_aligned_fresh.add_argument(
        "--enable-local-scratch",
        action="store_true",
        help=(
            "Use the predeclared workstation-local scratch for resumable caches; "
            "canonical outputs remain hash-validated workspace artifacts."
        ),
    )
    harp_fresh = sub.add_parser(
        "evaluate-harp-fresh",
        help=(
            "Evaluate the fully frozen HARP action policy on its fresh, "
            "case-disjoint MIDOG++ target surface."
        ),
    )
    harp_fresh.add_argument("--config", required=True)
    harp_fresh.add_argument(
        "--enable-local-scratch",
        action="store_true",
        help=(
            "Use the predeclared workstation-local scratch for resumable "
            "label-free caches; canonical outputs remain hash-validated "
            "workspace artifacts."
        ),
    )

    args = parser.parse_args(argv)
    if args.surface == "reserve-consumed-test":
        from .authorization import (
            load_reservation_config,
            run_target_evaluation_reservation,
        )

        config = load_reservation_config(args.config)
        requested = Path(args.artifact_root) if args.artifact_root else config.artifact_root
        _require_canonical_output(requested, config.artifact_root, "reservation")
        print(run_target_evaluation_reservation(config))
        return 0
    if args.surface == "build-descriptive-test-cache":
        from ...data.features.stage70_test_cache import (
            build_stage70_test_cache,
            load_stage70_test_cache_config,
            validate_stage70_test_cache,
        )

        config = load_stage70_test_cache_config(args.config)
        if args.validate_only:
            checks = validate_stage70_test_cache(config.cache_root, expected_config=config)
            print(json.dumps(checks, indent=2, sort_keys=True))
            return 0
        print(build_stage70_test_cache(config))
        return 0
    if args.surface == "authorize-prediction":
        from .authorization import (
            load_final_authorization_config,
            run_final_prediction_authorization,
        )

        config = load_final_authorization_config(args.config)
        requested = Path(args.artifact_root) if args.artifact_root else config.artifact_root
        _require_canonical_output(requested, config.artifact_root, "final authorization")
        print(run_final_prediction_authorization(config))
        return 0
    if args.surface == "evaluate-descriptively":
        from .config import load_frozen_policy_downstream_config
        from .runner import run_frozen_policy_downstream

        config = load_frozen_policy_downstream_config(args.config)
        print(run_frozen_policy_downstream(config))
        return 0
    if args.surface == "validate-descriptive-bundle":
        from .validation import validate_frozen_policy_downstream_bundle

        checks = validate_frozen_policy_downstream_bundle(args.artifact_root)
        print(json.dumps(checks, indent=2, sort_keys=True))
        return 0
    if args.surface == "evaluate-residual-topup-fresh":
        from .residual_topup_fresh.config import (
            load_residual_topup_fresh_config,
        )
        from .residual_topup_fresh.runner import run_residual_topup_fresh

        config = load_residual_topup_fresh_config(args.config)
        print(
            run_residual_topup_fresh(
                config,
                enable_optional_local_scratch=args.enable_local_scratch,
            )
        )
        return 0
    if args.surface == "evaluate-utility-aligned-residual-fresh":
        from .utility_aligned_residual_fresh import (
            load_utility_aligned_residual_fresh_config,
            run_utility_aligned_residual_fresh,
        )

        config = load_utility_aligned_residual_fresh_config(args.config)
        print(
            run_utility_aligned_residual_fresh(
                config,
                enable_optional_local_scratch=args.enable_local_scratch,
            )
        )
        return 0
    if args.surface == "evaluate-harp-fresh":
        from .harp_fresh import (
            load_harp_fresh_stage70_config,
            run_harp_fresh_stage70,
        )

        config = load_harp_fresh_stage70_config(args.config)
        print(
            run_harp_fresh_stage70(
                config,
                enable_optional_local_scratch=args.enable_local_scratch,
            )
        )
        return 0
    raise AssertionError(f"Unknown Stage-70 surface: {args.surface}")


def _require_canonical_output(requested: Path, expected: Path, role: str) -> None:
    if requested.resolve() != expected.resolve():
        raise ProtocolError(f"Stage-70 {role} must remain at its canonical workspace path.")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
