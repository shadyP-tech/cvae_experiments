"""Dataset-owned physical multiscale audit, contract, cache, and validation CLI."""

from __future__ import annotations

import argparse
import json

from midogpp_thesis.data.features.virchow2 import resolve_virchow2_identity

from .cache_builder import build_physical_multiscale_caches
from .config import load_build_config
from .contract import (
    audit_physical_multiscale_sources,
    build_physical_multiscale_contract,
)
from .validation import (
    validate_cache_bundle,
    validate_cache_pair,
    validate_contract_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in (
        "resolve-model",
        "audit",
        "build-contract",
        "build-cache",
        "validate",
        "resolve-model-v2",
        "audit-v2",
        "build-contract-v2",
        "build-cache-v2",
        "validate-v2",
        "resolve-model-v3",
        "audit-v3",
        "build-contract-v3",
        "build-cache-v3",
        "validate-v3",
    ):
        child = sub.add_parser(name)
        child.add_argument("--config", required=True)
        if name in {"audit-v2", "audit-v3"}:
            child.add_argument("--report-path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command.endswith("-v3"):
        return _main_v3(args)
    if args.command.endswith("-v2"):
        return _main_v2(args)
    config = load_build_config(
        args.config,
        require_inputs=args.command != "resolve-model",
    )
    if args.command == "resolve-model":
        identity = resolve_virchow2_identity(
            model_ref=config.model_ref,
            model_revision=config.model_revision,
            device=config.device,
        )
        print(json.dumps(identity, indent=2, sort_keys=True))
        return 0
    if args.command == "audit":
        report = audit_physical_multiscale_sources(config)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.command == "build-contract":
        output = build_physical_multiscale_contract(config)
        validation = validate_contract_bundle(output)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "contract_root": str(output),
                    "validation": validation,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "build-cache":
        b_root, c_root = build_physical_multiscale_caches(config)
        print(
            json.dumps(
                {"status": "PASS", "b_cache_root": str(b_root), "c_cache_root": str(c_root)},
                indent=2,
            )
        )
        return 0
    report = {
        "contract": validate_contract_bundle(config.contract_root),
        "b_cache": validate_cache_bundle(
            config.b_cache_root,
            expected_dim=3840,
            config=config,
        ),
        "c_cache": validate_cache_bundle(
            config.c_cache_root,
            expected_dim=11520,
            config=config,
        ),
        "pair": validate_cache_pair(
            config.b_cache_root,
            config.c_cache_root,
            contract_root=config.contract_root,
            canonical_cache_path=config.canonical_cache_path,
            canonical_reference_root=config.canonical_reference_root,
            config=config,
        ),
    }
    print(json.dumps({"status": "PASS", **report}, indent=2, sort_keys=True))
    return 0


def _main_v2(args: argparse.Namespace) -> int:
    from .cache_builder_v2 import build_physical_multiscale_caches_v2
    from .cache_validation_v2 import validate_cache_bundle_v2
    from .config_v2 import load_build_config_v2
    from .contract_v2 import (
        audit_physical_multiscale_sources_v2,
        build_physical_multiscale_contract_v2,
    )
    from .contract_validation import validate_contract_bundle_v2

    command = str(args.command).removesuffix("-v2")
    config = load_build_config_v2(
        args.config,
        require_inputs=command != "resolve-model",
    )
    if command == "resolve-model":
        identity = resolve_virchow2_identity(
            model_ref=config.model_ref,
            model_revision=config.model_revision,
            device=config.device,
        )
        print(json.dumps(identity, indent=2, sort_keys=True))
        return 0
    if command == "audit":
        report = audit_physical_multiscale_sources_v2(
            config,
            report_path=args.report_path,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if command == "build-contract":
        output = build_physical_multiscale_contract_v2(config)
        validation = validate_contract_bundle_v2(
            output,
            verify_raw_files=True,
            expected_config=config,
        )
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "contract_root": str(output),
                    "validation": validation,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if command == "build-cache":
        output = build_physical_multiscale_caches_v2(config)
        print(
            json.dumps(
                {"status": "PASS", "cache_bundle_root": str(output)},
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    report = {
        "contract": validate_contract_bundle_v2(
            config.contract_root,
            verify_raw_files=True,
            expected_config=config,
        ),
        "cache_bundle": validate_cache_bundle_v2(
            config.cache_bundle_root,
            contract_root=config.contract_root,
            canonical_cache_path=config.canonical_cache_path,
            canonical_reference_root=config.canonical_reference_root,
            expected_config=config,
        ),
    }
    print(json.dumps({"status": "PASS", **report}, indent=2, sort_keys=True))
    return 0


def _main_v3(args: argparse.Namespace) -> int:
    from .cache_builder_v3 import build_physical_multiscale_caches_v3
    from .cache_validation_v3 import validate_cache_bundle_v3
    from .config_v3 import load_build_config_v3
    from .contract_v3 import (
        audit_physical_multiscale_sources_v3,
        build_physical_multiscale_contract_v3,
    )
    from .contract_validation_v3 import validate_contract_bundle_v3

    command = str(args.command).removesuffix("-v3")
    config = load_build_config_v3(
        args.config,
        require_inputs=command != "resolve-model",
    )
    if command == "resolve-model":
        identity = resolve_virchow2_identity(
            model_ref=config.model_ref,
            model_revision=config.model_revision,
            device=config.device,
        )
        print(json.dumps(identity, indent=2, sort_keys=True))
        return 0
    if command == "audit":
        report = audit_physical_multiscale_sources_v3(
            config,
            report_path=args.report_path,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if command == "build-contract":
        output = build_physical_multiscale_contract_v3(config)
        validation = validate_contract_bundle_v3(
            output,
            verify_raw_files=True,
            expected_config=config,
        )
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "contract_root": str(output),
                    "validation": validation,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if command == "build-cache":
        output = build_physical_multiscale_caches_v3(config)
        print(
            json.dumps(
                {"status": "PASS", "cache_bundle_root": str(output)},
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    report = {
        "contract": validate_contract_bundle_v3(
            config.contract_root,
            verify_raw_files=True,
            expected_config=config,
        ),
        "cache_bundle": validate_cache_bundle_v3(
            config.cache_bundle_root,
            contract_root=config.contract_root,
            canonical_cache_path=config.canonical_cache_path,
            canonical_reference_root=config.canonical_reference_root,
            expected_config=config,
        ),
    }
    print(json.dumps({"status": "PASS", **report}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
