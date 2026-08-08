"""Orchestrate the fresh, fixed B/U/G/S residual-topup Stage-60 lock."""

from __future__ import annotations

from pathlib import Path

from ...reporting import write_csv_rows, write_json
from .artifact_io import (
    assert_closed_world,
    read_json,
    write_content_index,
    write_state,
)
from .artifact_schema import (
    POLICY_DECISION,
    PUBLICATION_STATE,
    action_table_rows,
    ballot_table_rows,
    build_leakage_report,
    build_policy_decision,
    build_protocol_manifest,
    build_protocol_report,
    rank_table_rows,
)
from .config import ResidualTopupPolicyLockConfig
from .io import load_validated_fresh_proxy_inputs
from .products import (
    PolicyProducts,
    build_policy_lock_payload,
    build_policy_products,
)
from .workspace_binding import validate_launch_workspace_files


def run_residual_topup_policy_lock(
    config: ResidualTopupPolicyLockConfig,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    """Write and independently validate the frozen Stage-60 policy artifact."""

    root = Path(artifact_root or config.artifact_root)
    workspace_binding = validate_launch_workspace_files(config, artifact_root=root)
    inputs = load_validated_fresh_proxy_inputs(config)
    products = build_policy_products(config, inputs)
    assert_closed_world(root)
    for relative in ("manifests", "reports", "tables", "provenance"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    state_path = root / "reports/run_state.json"
    if state_path.is_file() and read_json(state_path).get("status") == "COMPLETE":
        from .validation import validate_residual_topup_policy_bundle

        validate_residual_topup_policy_bundle(root, config=config)
        return root
    write_state(root, "RUNNING")
    try:
        policy_lock = build_policy_lock_payload(
            config,
            inputs,
            products,
            workspace_binding=workspace_binding,
        )
        action_library = products.action_library.to_payload()
        protocol = build_protocol_manifest(config, inputs, products, policy_lock)
        write_json(
            root / "manifests/fresh_surface_attestation.json",
            inputs.attestation.to_payload(),
        )
        write_json(root / "manifests/policy_lock.json", policy_lock)
        write_json(root / "manifests/action_library.json", action_library)
        write_json(root / "manifests/protocol_manifest.json", protocol)
        write_csv_rows(root / "tables/proxy_ballots.csv", ballot_table_rows(products))
        write_csv_rows(root / "tables/proxy_ranks.csv", rank_table_rows(products))
        write_csv_rows(root / "tables/policy_actions.csv", action_table_rows(products))
        write_json(
            root / "reports/protocol_report.json",
            build_protocol_report(policy_lock, protocol),
        )
        write_json(root / "reports/leakage_report.json", build_leakage_report())
        write_json(
            root / "reports/policy_decision.json",
            build_policy_decision(products, policy_lock),
        )
        write_content_index(root)
        write_state(root, "COMPLETE")

        from .validation import validate_residual_topup_policy_bundle

        checks = validate_residual_topup_policy_bundle(
            root,
            config=config,
            allow_pending=True,
            _inputs=inputs,
            _products=products,
        )
        write_json(
            root / "reports/validation_report.json",
            {
                "schema_version": "midogpp_residual_topup_b_u_g_s_validation_v1",
                "status": "PASS",
                "validator": "validate_residual_topup_policy_bundle",
                "checks": checks,
            },
        )
        validate_residual_topup_policy_bundle(
            root,
            config=config,
            _inputs=inputs,
            _products=products,
        )
    except Exception:
        write_state(root, "FAILED")
        raise
    return root


__all__ = (
    "POLICY_DECISION",
    "PUBLICATION_STATE",
    "PolicyProducts",
    "action_table_rows",
    "ballot_table_rows",
    "build_policy_lock_payload",
    "build_policy_products",
    "build_protocol_manifest",
    "rank_table_rows",
    "run_residual_topup_policy_lock",
)
