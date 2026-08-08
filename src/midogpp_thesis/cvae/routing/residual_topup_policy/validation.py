"""Independent reconstructive validation for the fresh Stage-60 policy lock."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ..residual_topup.hashing import canonical_sha256
from .artifact_schema import (
    action_table_rows,
    ballot_table_rows,
    build_leakage_report,
    build_policy_decision,
    build_protocol_manifest,
    build_protocol_report,
    rank_table_rows,
)
from .bundle import CONTENT_INDEX_MEMBERS, REQUIRED_FILES
from .config import CLAIM_SCOPE, ResidualTopupPolicyLockConfig
from .io import ValidatedFreshProxyInputs, load_validated_fresh_proxy_inputs
from .products import (
    PolicyProducts,
    build_policy_lock_payload,
    build_policy_products,
)
from .workspace_binding import validate_launch_workspace_files


def validate_residual_topup_policy_bundle(
    root: str | Path,
    *,
    config: ResidualTopupPolicyLockConfig,
    allow_pending: bool = False,
    _inputs: ValidatedFreshProxyInputs | None = None,
    _products: PolicyProducts | None = None,
) -> dict[str, object]:
    path = Path(root)
    required = set(REQUIRED_FILES)
    if allow_pending:
        required.remove("reports/validation_report.json")
    missing = sorted(relative for relative in required if not (path / relative).is_file())
    if missing:
        raise ProtocolError(f"Residual-topup policy bundle is incomplete: {missing}.")
    _validate_closed_world(path)
    workspace_binding = validate_launch_workspace_files(config, artifact_root=path)
    inputs = _inputs or load_validated_fresh_proxy_inputs(config)
    products = _products or build_policy_products(config, inputs)
    expected_library = products.action_library.to_payload()
    expected_lock = build_policy_lock_payload(
        config,
        inputs,
        products,
        workspace_binding=workspace_binding,
    )
    expected_protocol = build_protocol_manifest(
        config,
        inputs,
        products,
        expected_lock,
    )
    if _json(path / "manifests/fresh_surface_attestation.json") != inputs.attestation.to_payload():
        raise ProtocolError("Residual-topup bundled fresh-surface attestation drifted.")
    if _json(path / "manifests/action_library.json") != expected_library:
        raise ProtocolError("Residual-topup frozen action library drifted.")
    if _json(path / "manifests/policy_lock.json") != expected_lock:
        raise ProtocolError("Residual-topup policy lock drifted from fresh inputs.")
    if _json(path / "manifests/protocol_manifest.json") != expected_protocol:
        raise ProtocolError("Residual-topup protocol manifest drifted.")
    _validate_csv(
        path / "tables/proxy_ballots.csv",
        ballot_table_rows(products),
        label="proxy ballot",
    )
    _validate_csv(
        path / "tables/proxy_ranks.csv",
        rank_table_rows(products),
        label="proxy rank",
    )
    _validate_csv(
        path / "tables/policy_actions.csv",
        action_table_rows(products),
        label="policy action",
    )
    expected_protocol_report = build_protocol_report(
        expected_lock,
        expected_protocol,
    )
    if _json(path / "reports/protocol_report.json") != expected_protocol_report:
        raise ProtocolError("Residual-topup protocol report drifted.")
    expected_leakage = build_leakage_report()
    if _json(path / "reports/leakage_report.json") != expected_leakage:
        raise ProtocolError("Residual-topup leakage report drifted.")
    expected_decision = build_policy_decision(products, expected_lock)
    if _json(path / "reports/policy_decision.json") != expected_decision:
        raise ProtocolError("Residual-topup policy decision drifted.")
    expected_state = {
        "schema_version": "midogpp_residual_topup_b_u_g_s_run_state_v1",
        "status": "COMPLETE",
        "claim_scope": CLAIM_SCOPE,
    }
    if _json(path / "reports/run_state.json") != expected_state:
        raise ProtocolError("Residual-topup run state drifted.")
    _validate_content_index(path)
    checks: dict[str, object] = {
        "status": "PASS",
        "config_contract_hash": config.contract_hash,
        "fresh_surface_attestation_hash": inputs.attestation.attestation_hash,
        "proxy_score_table_sha256": inputs.proxy_score_table_sha256,
        "rank_summary_hash": products.rank_summary_hash,
        "action_library_hash": products.action_library.action_library_hash,
        "policy_lock_hash": expected_lock["policy_lock_hash"],
        "target_count": len(config.centers),
        "action_count": products.action_library.action_count,
        "ballot_table_row_count": len(ballot_table_rows(products)),
        "rank_table_row_count": len(rank_table_rows(products)),
        "action_table_row_count": len(action_table_rows(products)),
        "all_actions_frozen_before_stage70": True,
        "proxy_only": True,
        "labels_consumed": False,
        "target_evaluation_used": False,
        "source_experts_updated": False,
        "routing_quality_claimed": False,
        "downstream_outcome_computed": False,
    }
    if not allow_pending:
        expected_report = {
            "schema_version": "midogpp_residual_topup_b_u_g_s_validation_v1",
            "status": "PASS",
            "validator": "validate_residual_topup_policy_bundle",
            "checks": checks,
        }
        if _json(path / "reports/validation_report.json") != expected_report:
            raise ProtocolError("Residual-topup validation report drifted.")
    return checks


def _validate_csv(
    path: Path,
    expected_rows: Sequence[Mapping[str, object]],
    *,
    label: str,
) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        observed = [dict(row) for row in reader]
    if not expected_rows:
        raise ProtocolError(f"Residual-topup expected {label} table is empty.")
    columns = tuple(expected_rows[0])
    if tuple(reader.fieldnames or ()) != columns or len(observed) != len(expected_rows):
        raise ProtocolError(f"Residual-topup {label} table geometry drifted.")
    for actual, expected in zip(observed, expected_rows, strict=True):
        if any(
            actual.get(key, "") != ("" if expected[key] is None else str(expected[key]))
            for key in columns
        ):
            raise ProtocolError(f"Residual-topup {label} table content drifted.")


def _validate_content_index(root: Path) -> None:
    payload = _json(root / "manifests/content_index.json")
    if set(payload) != {"schema_version", "records", "content_hash"}:
        raise ProtocolError("Residual-topup content-index fields drifted.")
    unhashed = {key: value for key, value in payload.items() if key != "content_hash"}
    if (
        payload.get("schema_version")
        != "midogpp_residual_topup_b_u_g_s_content_index_v1"
        or payload.get("content_hash") != canonical_sha256(unhashed)
        or not isinstance(payload.get("records"), list)
    ):
        raise ProtocolError("Residual-topup content-index hash is invalid.")
    observed: list[str] = []
    for raw in payload["records"]:  # type: ignore[index]
        if not isinstance(raw, Mapping) or set(raw) != {
            "relative_path",
            "sha256",
            "size_bytes",
        }:
            raise ProtocolError("Residual-topup content-index row is invalid.")
        relative = str(raw.get("relative_path", ""))
        member = _safe_member(root, relative)
        if (
            not member.is_file()
            or member.stat().st_size != raw.get("size_bytes")
            or _sha256_file(member) != raw.get("sha256")
        ):
            raise ProtocolError(f"Residual-topup content member drifted: {relative}.")
        observed.append(relative)
    if tuple(observed) != CONTENT_INDEX_MEMBERS:
        raise ProtocolError("Residual-topup content-index coverage drifted.")


def _validate_closed_world(root: Path) -> None:
    actual = {
        member.relative_to(root).as_posix()
        for member in root.rglob("*")
        if member.is_file()
    }
    unexpected = sorted(actual.difference(REQUIRED_FILES))
    if unexpected:
        raise ProtocolError(f"Residual-topup artifact contains unexpected files: {unexpected}.")


def _json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Residual-topup JSON is unreadable: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Residual-topup JSON must be an object: {path}.")
    return payload


def _safe_member(root: Path, relative: str) -> Path:
    resolved_root = root.resolve()
    member = (resolved_root / relative).resolve()
    if member == resolved_root or not member.is_relative_to(resolved_root):
        raise ProtocolError("Residual-topup content path escapes its artifact root.")
    return member


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ("validate_residual_topup_policy_bundle",)
