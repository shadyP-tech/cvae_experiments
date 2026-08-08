"""Workspace and persisted-provenance binding for the fresh Stage-60 lock."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from ....workspace.runtime import MidogppWorkspace
from ...protocol import ProtocolError
from .config import (
    CLAIM_SCOPE,
    EXPERIMENT_ID,
    INPUT_ARTIFACT_IDS,
    OUTPUT_ARTIFACT_ID,
    PROXY_SURFACE_ARTIFACT_ID,
    STAGE_ID,
    ResidualTopupPolicyLockConfig,
)


PROXY_REQUIRED_FILES = (
    "tables/proxy_scores.csv",
    "manifests/fresh_surface_attestation.json",
)
CONFIG_PATH = (
    "experiments/midogpp/stages/60_routing_and_composition/configs/"
    "uniform_b_v2_residual_topup_b_u_g_s_policy_lock_v1.yaml"
)
OUTPUT_CANONICAL_PATH = (
    "artifacts/midogpp/60_routing_and_composition/"
    "uniform_b_v2_residual_topup_b_u_g_s_policy_lock/v1"
)
PROXY_CANONICAL_PATH = (
    "artifacts/midogpp/60_routing_and_composition/"
    "residual_topup_fresh_proxy_surface/v1"
)


def validate_planned_workspace_contract(
    config: ResidualTopupPolicyLockConfig,
    *,
    _workspace: MidogppWorkspace | None = None,
) -> None:
    """Validate the deliberately planned registry/catalog/config topology."""

    workspace = _workspace or MidogppWorkspace.load()
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    proxy = workspace.artifacts[PROXY_SURFACE_ARTIFACT_ID]
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]
    from .bundle import REQUIRED_FILES

    if (
        experiment.status != "planned"
        or experiment.stage != STAGE_ID
        or experiment.claim_scope != CLAIM_SCOPE
        or experiment.config_path != CONFIG_PATH
        or experiment.output_artifact_id != OUTPUT_ARTIFACT_ID
        or experiment.input_artifact_ids != INPUT_ARTIFACT_IDS
        or proxy.availability != "planned"
        or proxy.canonical_path != PROXY_CANONICAL_PATH
        or proxy.required_files != PROXY_REQUIRED_FILES
        or proxy.claim_scope != "routing_compatibility_only"
        or proxy.may_feed_deployable_selection is not True
        or output.availability != "planned"
        or output.canonical_path != OUTPUT_CANONICAL_PATH
        or output.required_files != REQUIRED_FILES
        or output.claim_scope != CLAIM_SCOPE
        or output.may_feed_deployable_selection is not True
    ):
        raise ProtocolError("Fresh residual-topup planned workspace contract drifted.")
    protocol = config.protocol
    actions = config.actions
    if (
        protocol.get("dataset_family") != "MIDOG++"
        or protocol.get("feature_frame") != "annotation_jpeg_fixed_center_b_v3"
        or protocol.get("target_labels_used") is not False
        or protocol.get("target_evaluation_used") is not False
        or protocol.get("source_experts_updated") is not False
        or actions.get("freeze_all_H_by_e_single_source_tail_actions") is not True
        or actions.get("freeze_permutation_control") is not True
    ):
        raise ProtocolError("Fresh residual-topup planned config binding drifted.")


def validate_production_workspace_binding(
    config: ResidualTopupPolicyLockConfig,
    *,
    _workspace: MidogppWorkspace | None = None,
) -> None:
    """Require an explicit planned-to-active registry transition before launch."""

    workspace = _workspace or MidogppWorkspace.load()
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    if experiment.status != "active":
        raise ProtocolError(
            "Fresh residual-topup Stage-60 experiment remains status='planned'; "
            "fresh inputs and registry activation are required."
        )
    if (
        experiment.stage != STAGE_ID
        or experiment.claim_scope != CLAIM_SCOPE
        or experiment.output_artifact_id != OUTPUT_ARTIFACT_ID
        or experiment.input_artifact_ids != INPUT_ARTIFACT_IDS
    ):
        raise ProtocolError("Fresh residual-topup production workspace binding drifted.")
    expected = {
        "artifact_root": workspace.resolve_artifact(
            OUTPUT_ARTIFACT_ID, for_output=True, require_exists=False
        ),
        "expert_bank_root": workspace.resolve_artifact(INPUT_ARTIFACT_IDS[0]),
        "generation_lock_root": workspace.resolve_artifact(INPUT_ARTIFACT_IDS[1]),
        "equal_union_policy_root": workspace.resolve_artifact(INPUT_ARTIFACT_IDS[2]),
        "proxy_surface_root": workspace.resolve_artifact(INPUT_ARTIFACT_IDS[3]),
        "proxy_score_table_path": workspace.resolve_artifact(INPUT_ARTIFACT_IDS[3])
        / PROXY_REQUIRED_FILES[0],
        "proxy_attestation_path": workspace.resolve_artifact(INPUT_ARTIFACT_IDS[3])
        / PROXY_REQUIRED_FILES[1],
    }
    for field, expected_path in expected.items():
        if Path(getattr(config, field)).resolve() != Path(expected_path).resolve():
            raise ProtocolError(f"Fresh residual-topup workspace path drifted: {field}.")


def validate_launch_workspace_files(
    config: ResidualTopupPolicyLockConfig,
    *,
    artifact_root: str | Path,
) -> dict[str, object]:
    """Validate persisted config/provenance and both fresh proxy file hashes."""

    root = Path(artifact_root)
    resolved_config = root / "config.resolved.yaml"
    provenance_path = root / "provenance/input_artifacts.json"
    if not resolved_config.is_file() or not provenance_path.is_file():
        raise ProtocolError(
            "Fresh residual-topup policy must be launched through the MIDOG++ workspace."
        )
    from .config import load_residual_topup_policy_lock_config

    if load_residual_topup_policy_lock_config(resolved_config) != config:
        raise ProtocolError("Fresh residual-topup resolved config drifted.")
    provenance = _json(provenance_path)
    required = {
        "schema_version": "midogpp_input_artifacts_v2",
        "dataset_id": "midogpp",
        "experiment_id": EXPERIMENT_ID,
        "stage": STAGE_ID,
        "claim_scope": CLAIM_SCOPE,
        "selection_used_target_eval_artifacts": False,
    }
    if any(provenance.get(key) != value for key, value in required.items()):
        raise ProtocolError("Fresh residual-topup input provenance header drifted.")
    raw_rows = provenance.get("input_artifacts")
    if not isinstance(raw_rows, list):
        raise ProtocolError("Fresh residual-topup input provenance rows are missing.")
    rows: dict[str, Mapping[str, object]] = {}
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise ProtocolError("Fresh residual-topup input provenance row is invalid.")
        artifact_id = str(raw.get("artifact_id", ""))
        if artifact_id in rows:
            raise ProtocolError("Fresh residual-topup provenance artifact is duplicated.")
        rows[artifact_id] = raw
    if set(rows) != set(INPUT_ARTIFACT_IDS):
        raise ProtocolError("Fresh residual-topup input provenance grid drifted.")
    if any(row.get("exists") is not True for row in rows.values()):
        raise ProtocolError("Fresh residual-topup provenance contains an absent input.")
    proxy_row = rows[PROXY_SURFACE_ARTIFACT_ID]
    if Path(str(proxy_row.get("resolved_path", ""))).resolve() != config.proxy_surface_root.resolve():
        raise ProtocolError("Fresh proxy-surface provenance root drifted.")
    integrity = proxy_row.get("file_integrity")
    if not isinstance(integrity, Mapping) or not isinstance(integrity.get("files"), list):
        raise ProtocolError("Fresh proxy-surface file integrity is missing.")
    files: dict[str, Mapping[str, object]] = {}
    for raw in integrity["files"]:  # type: ignore[index]
        if not isinstance(raw, Mapping):
            raise ProtocolError("Fresh proxy-surface integrity row is invalid.")
        relative = str(raw.get("path", ""))
        if relative in files:
            raise ProtocolError("Fresh proxy-surface integrity row is duplicated.")
        files[relative] = raw
    if set(files) != set(PROXY_REQUIRED_FILES):
        raise ProtocolError("Fresh proxy-surface required-file grid drifted.")
    expected_paths = {
        PROXY_REQUIRED_FILES[0]: config.proxy_score_table_path,
        PROXY_REQUIRED_FILES[1]: config.proxy_attestation_path,
    }
    bound_hashes: dict[str, str] = {}
    for relative, expected_path in expected_paths.items():
        row = files[relative]
        computed = row.get("computed")
        if (
            row.get("exists") is not True
            or Path(str(row.get("resolved_path", ""))).resolve()
            != expected_path.resolve()
            or not isinstance(computed, Mapping)
            or computed.get("sha256") != _sha256_file(expected_path)
        ):
            raise ProtocolError(f"Fresh proxy-surface provenance hash drifted: {relative}.")
        bound_hashes[relative] = str(computed["sha256"])
    return {
        "config_resolved_sha256": _sha256_file(resolved_config),
        "input_provenance_sha256": _sha256_file(provenance_path),
        "proxy_required_file_sha256": bound_hashes,
    }


def _json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Fresh residual-topup JSON is unreadable: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Fresh residual-topup JSON must be an object: {path}.")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = (
    "CONFIG_PATH",
    "OUTPUT_CANONICAL_PATH",
    "PROXY_CANONICAL_PATH",
    "PROXY_REQUIRED_FILES",
    "validate_launch_workspace_files",
    "validate_planned_workspace_contract",
    "validate_production_workspace_binding",
)
