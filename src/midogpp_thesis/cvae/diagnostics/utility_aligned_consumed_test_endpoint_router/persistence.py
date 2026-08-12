"""Atomic persistence for endpoint-router scientific phase boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import canonical_sha256
from .artifact_io import persist_or_validate_csv, persist_or_validate_json
from .experiment_contracts import DEVELOPMENT_RESPONSE_COUNT
from .reports import protocol_manifest_payload


def persist_initial_surfaces(
    root: str | Path,
    *,
    config: object,
    protocol: object,
    provenance: Mapping[str, Mapping[str, object]],
    cache_binding_hash: str,
    manifest_admission_hash: str,
    firewall: Mapping[str, object],
    support_partition: object,
    action_library: object,
) -> None:
    base = Path(root)
    input_ids = tuple(getattr(config, "input_artifact_ids"))
    if len(input_ids) != 6 or set(provenance) != set(input_ids):
        raise ProtocolError("Endpoint-router provenance must bind exactly six inputs.")
    input_hashes = {
        artifact_id: canonical_sha256(provenance[artifact_id])
        for artifact_id in input_ids
    }
    persist_or_validate_json(
        base / "manifests/protocol_manifest.json",
        protocol_manifest_payload(
            config,
            protocol=protocol,
            input_artifact_hashes=input_hashes,
            cache_binding_hash=cache_binding_hash,
            manifest_admission_hash=manifest_admission_hash,
            firewall=firewall,
        ),
    )
    partition_lock = getattr(support_partition, "lock_payload", None)
    partition_payload = (
        dict(partition_lock)
        if isinstance(partition_lock, Mapping)
        else _payload(support_partition)
    )
    action_payload = _payload(action_library)
    persist_or_validate_json(
        base / "manifests/support_partition_lock.json", partition_payload
    )
    persist_or_validate_json(base / "manifests/action_library.json", action_payload)
    surface_rows = getattr(support_partition, "table_rows", None)
    partition_rows = (
        [dict(row) for row in surface_rows]
        if isinstance(surface_rows, (list, tuple)) and surface_rows
        else _rows_from_partition(partition_payload)
    )
    persist_or_validate_csv(base / "tables/support_partitions.csv", partition_rows)


def persist_development_surfaces(
    root: str | Path,
    *,
    response_rows: Sequence[Mapping[str, object]],
    source_inner_feature_rows: Sequence[Mapping[str, object]],
    response_seal: Mapping[str, object],
    feature_surface_set: Mapping[str, object],
    development_label_access_report: Mapping[str, object],
) -> None:
    if len(response_rows) != DEVELOPMENT_RESPONSE_COUNT:
        raise ProtocolError("Endpoint-router requires exactly 504 development responses.")
    base = Path(root)
    persist_or_validate_csv(
        base / "tables/development_endpoint_responses.csv", response_rows
    )
    persist_or_validate_csv(
        base / "tables/source_inner_feature_rows.csv", source_inner_feature_rows
    )
    persist_or_validate_json(
        base / "manifests/development_endpoint_response_seal.json", response_seal
    )
    persist_or_validate_json(
        base / "manifests/feature_surface_set.json", feature_surface_set
    )
    persist_or_validate_json(
        base / "reports/development_label_access_report.json",
        development_label_access_report,
    )


def persist_model_and_plan_surfaces(
    root: str | Path,
    *,
    model_index: Mapping[str, object],
    model_rows: Sequence[Mapping[str, object]],
    cardinality_transfer_seal: Mapping[str, object],
    target_feature_rows: Sequence[Mapping[str, object]],
    target_policy_plans: Mapping[str, object],
    target_policy_plan_rows: Sequence[Mapping[str, object]],
    frozen_actions: Mapping[str, object],
    frozen_action_rows: Sequence[Mapping[str, object]],
    global_prelabel_seal: Mapping[str, object],
) -> None:
    base = Path(root)
    objects = (
        ("manifests/model_index.json", model_index),
        ("manifests/cardinality_transfer_seal.json", cardinality_transfer_seal),
        ("manifests/target_policy_plans.json", target_policy_plans),
        ("manifests/frozen_actions.json", frozen_actions),
        ("manifests/global_prelabel_seal.json", global_prelabel_seal),
    )
    for member, payload in objects:
        persist_or_validate_json(base / member, payload)
    tables = (
        ("tables/model_index.csv", model_rows),
        ("tables/target_feature_rows.csv", target_feature_rows),
        ("tables/target_policy_plans.csv", target_policy_plan_rows),
        ("tables/frozen_actions.csv", frozen_action_rows),
    )
    for member, rows in tables:
        persist_or_validate_csv(base / member, rows)


def persist_terminal_surfaces(
    root: str | Path,
    *,
    endpoint_rows: Sequence[Mapping[str, object]],
    contrast_rows: Sequence[Mapping[str, object]],
    aggregate_contrast_rows: Sequence[Mapping[str, object]],
    oracle_rows: Sequence[Mapping[str, object]],
    sealed_terminal_evaluation: Mapping[str, object],
    label_capability_report: Mapping[str, object],
    leakage_report: Mapping[str, object],
    runtime_summary: Mapping[str, object],
    publication_decision: Mapping[str, object],
) -> None:
    base = Path(root)
    for member, rows in (
        ("tables/terminal_endpoint_scores.csv", endpoint_rows),
        ("tables/center_contrasts.csv", contrast_rows),
        ("tables/aggregate_contrasts.csv", aggregate_contrast_rows),
        ("tables/oracle_rank_diagnostics.csv", oracle_rows),
    ):
        persist_or_validate_csv(base / member, rows)
    for member, payload in (
        ("manifests/sealed_terminal_evaluation.json", sealed_terminal_evaluation),
        ("reports/label_capability_report.json", label_capability_report),
        ("reports/leakage_report.json", leakage_report),
        ("reports/runtime_summary.json", runtime_summary),
        ("reports/publication_decision.json", publication_decision),
    ):
        persist_or_validate_json(base / member, payload)


def persist_validation_report(
    root: str | Path, report: Mapping[str, object]
) -> None:
    persist_or_validate_json(Path(root) / "reports/validation_report.json", report)


def _payload(value: object) -> dict[str, object]:
    rendered = value.to_payload() if hasattr(value, "to_payload") else value
    if not isinstance(rendered, Mapping):
        raise ProtocolError("Endpoint-router persisted object is not a mapping.")
    return dict(rendered)


def _rows_from_partition(payload: Mapping[str, object]) -> list[dict[str, object]]:
    rows = payload.get("rows")
    if isinstance(rows, list) and rows and all(isinstance(row, Mapping) for row in rows):
        return [dict(row) for row in rows]
    targets = payload.get("targets")
    if not isinstance(targets, list):
        raise ProtocolError("Endpoint-router support partition has no reconstructive rows.")
    result: list[dict[str, object]] = []
    for target in targets:
        if not isinstance(target, Mapping):
            raise ProtocolError("Endpoint-router support partition target is malformed.")
        target_id = str(target.get("target_center", target.get("target_id", "")))
        for role, key in (("support", "support_case_ids"), ("evaluation", "evaluation_case_ids")):
            cases = target.get(key)
            if not isinstance(cases, list):
                raise ProtocolError("Endpoint-router support partition case list is malformed.")
            result.extend(
                {"target_center": target_id, "case_id": str(case), "role": role}
                for case in cases
            )
    if not result:
        raise ProtocolError("Endpoint-router support partition is empty.")
    return result


__all__ = (
    "persist_development_surfaces",
    "persist_initial_surfaces",
    "persist_model_and_plan_surfaces",
    "persist_terminal_surfaces",
    "persist_validation_report",
)
