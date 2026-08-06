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
    raise AssertionError(f"Unknown Stage-70 surface: {args.surface}")


def _require_canonical_output(requested: Path, expected: Path, role: str) -> None:
    if requested.resolve() != expected.resolve():
        raise ProtocolError(f"Stage-70 {role} must remain at its canonical workspace path.")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
