"""CLI for frozen CVAE routing and composition policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..protocol import ProtocolError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="surface", required=True)
    policy = sub.add_parser(
        "uniform-b-v2-equal-union-policy-lock",
        help="Freeze the target-excluded equal-union policy for Stage-70 scoring.",
    )
    policy.add_argument("--config", required=True)
    policy.add_argument("--artifact-root", default=None)
    metadata = sub.add_parser(
        "uniform-b-v2-metadata-exact-match-compatibility",
        help="Freeze the label-free MIDOG++ metadata compatibility proxy.",
    )
    metadata.add_argument("--config", required=True)
    metadata.add_argument("--artifact-root", default=None)
    tie_union = sub.add_parser(
        "uniform-b-v2-metadata-tie-union-policy-lock",
        help="Freeze the tied-maximum metadata comparison policy.",
    )
    tie_union.add_argument("--config", required=True)
    tie_union.add_argument("--artifact-root", default=None)
    validation_cache = sub.add_parser(
        "uniform-b-v2-routing-validation-cache",
        help=(
            "Build the immutable label-blind Uniform-B validation cache used by "
            "the predeclared source-inner utility study."
        ),
    )
    validation_cache.add_argument("--config", required=True)
    validation_cache.add_argument("--validate-only", action="store_true")
    utility = sub.add_parser(
        "uniform-b-v2-source-inner-candidate-utility",
        help="Materialize the frozen non-selecting source-inner utility surface.",
    )
    utility.add_argument("--config", required=True)
    utility.add_argument("--artifact-root", default=None)
    regret_policy = sub.add_parser(
        "uniform-b-v2-utility-regret-policy-lock",
        help="Freeze the uncertainty-gated utility/regret routing policy.",
    )
    regret_policy.add_argument("--config", required=True)
    regret_policy.add_argument("--artifact-root", default=None)
    args = parser.parse_args(argv)
    if args.surface == "uniform-b-v2-equal-union-policy-lock":
        from .config import load_equal_union_policy_config
        from .runner import run_equal_union_policy_lock
        from .workspace_binding import validate_production_workspace_binding

        config = load_equal_union_policy_config(args.config)
        validate_production_workspace_binding(config)
        requested = Path(args.artifact_root) if args.artifact_root else config.artifact_root
        if requested.resolve() != config.artifact_root.resolve():
            raise ProtocolError(
                "Equal-union policy output must remain at its canonical workspace path."
            )
        print(run_equal_union_policy_lock(config, artifact_root=requested))
        return 0
    if args.surface == "uniform-b-v2-metadata-exact-match-compatibility":
        from .metadata_compatibility.config import load_metadata_compatibility_config
        from .metadata_compatibility.runner import run_metadata_compatibility_lock
        from .metadata_compatibility.workspace_binding import (
            validate_production_workspace_binding,
        )

        config = load_metadata_compatibility_config(args.config)
        validate_production_workspace_binding(config)
        requested = Path(args.artifact_root) if args.artifact_root else config.artifact_root
        if requested.resolve() != config.artifact_root.resolve():
            raise ProtocolError(
                "Metadata compatibility output must remain at its canonical workspace path."
            )
        print(run_metadata_compatibility_lock(config, artifact_root=requested))
        return 0
    if args.surface == "uniform-b-v2-metadata-tie-union-policy-lock":
        from .metadata_tie_union.config import load_metadata_tie_union_policy_config
        from .metadata_tie_union.runner import run_metadata_tie_union_policy_lock
        from .metadata_tie_union.workspace_binding import (
            validate_production_workspace_binding,
        )

        config = load_metadata_tie_union_policy_config(args.config)
        validate_production_workspace_binding(config)
        requested = Path(args.artifact_root) if args.artifact_root else config.artifact_root
        if requested.resolve() != config.artifact_root.resolve():
            raise ProtocolError(
                "Metadata tie-union output must remain at its canonical workspace path."
            )
        print(run_metadata_tie_union_policy_lock(config, artifact_root=requested))
        return 0
    if args.surface == "uniform-b-v2-routing-validation-cache":
        from ...data.features.uniform_b_routing_validation import (
            build_uniform_b_routing_validation_cache,
            load_routing_validation_cache_config,
            resolve_routing_validation_cache_config,
            validate_uniform_b_routing_validation_cache,
        )

        config = load_routing_validation_cache_config(args.config)
        if args.validate_only:
            resolved = resolve_routing_validation_cache_config(config)
            checks = validate_uniform_b_routing_validation_cache(
                resolved.cache_root,
                expected_config=resolved,
            )
            print(json.dumps(checks, indent=2, sort_keys=True))
            return 0
        print(build_uniform_b_routing_validation_cache(config))
        return 0
    if args.surface == "uniform-b-v2-source-inner-candidate-utility":
        from .source_inner_utility.config import load_source_inner_utility_config
        from .source_inner_utility.runner import run_source_inner_candidate_utility
        from .source_inner_utility.workspace_binding import (
            validate_production_workspace_binding,
        )

        config = load_source_inner_utility_config(args.config)
        validate_production_workspace_binding(config)
        requested = Path(args.artifact_root) if args.artifact_root else config.artifact_root
        if requested.resolve() != config.artifact_root.resolve():
            raise ProtocolError(
                "Source-inner utility output must remain at its canonical workspace path."
            )
        print(run_source_inner_candidate_utility(config, artifact_root=requested))
        return 0
    if args.surface == "uniform-b-v2-utility-regret-policy-lock":
        from .utility_regret_policy.config import load_utility_regret_policy_config
        from .utility_regret_policy.runner import run_utility_regret_policy_lock
        from .utility_regret_policy.workspace_binding import (
            validate_production_workspace_binding,
        )

        config = load_utility_regret_policy_config(args.config)
        validate_production_workspace_binding(config)
        requested = Path(args.artifact_root) if args.artifact_root else config.artifact_root
        if requested.resolve() != config.artifact_root.resolve():
            raise ProtocolError(
                "Utility/regret policy output must remain at its canonical workspace path."
            )
        print(run_utility_regret_policy_lock(config, artifact_root=requested))
        return 0
    raise AssertionError(f"Unknown routing surface: {args.surface}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
