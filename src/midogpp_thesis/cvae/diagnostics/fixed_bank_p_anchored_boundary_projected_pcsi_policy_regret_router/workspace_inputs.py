"""Workspace binding and exact-six provenance validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from ....workspace.runtime import MidogppWorkspace
from ...protocol import ProtocolError
from .constants import (
    EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT,
    EXPECTED_POLICY_REPLAY_COUNT,
    EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT,
    EXPECTED_UTILITY_MODEL_FIT_COUNT,
)
from .experiment_contracts import (
    CLAIM_SCOPE,
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    LEDGER_AMENDMENT_ARTIFACT_ID,
    STAGE_ID,
    TEST_CACHE_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
)
from .protocol import build_frozen_protocol


def validate_active_workspace_binding(config: object) -> Mapping[str, object]:
    try:
        workspace = MidogppWorkspace.load()
        workspace.validate()
        experiment = workspace.get_experiment(str(getattr(config, "experiment_id")))
        output = workspace.artifacts[str(getattr(config, "output_artifact_id"))]
    except (KeyError, ValueError, OSError) as exc:
        raise ProtocolError("PCSI-PARC workspace binding failed.") from exc
    if (
        experiment.status != "planned"
        or experiment.stage != STAGE_ID
        or experiment.claim_scope != CLAIM_SCOPE
        or experiment.input_artifact_ids != tuple(getattr(config, "input_artifact_ids"))
        or output.stage != STAGE_ID
        or output.claim_scope != CLAIM_SCOPE
        or output.semantic_identities.get("config_contract_hash")
        != str(getattr(config, "contract_hash"))
        or output.semantic_identities.get("protocol_contract_hash")
        != build_frozen_protocol().protocol_hash
        or output.semantic_identities.get("physical_probability_cell_count") != "810"
        or output.semantic_identities.get("outer_endpoint_model_fit_count")
        != str(EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT)
        or output.semantic_identities.get("target_local_posterior_model_fit_count")
        != str(EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT)
        or output.semantic_identities.get("utility_model_fit_count")
        != str(EXPECTED_UTILITY_MODEL_FIT_COUNT)
        or output.semantic_identities.get("whole_policy_pseudo_target_replay_count")
        != str(EXPECTED_POLICY_REPLAY_COUNT)
        or output.semantic_identities.get("transport_semantics")
        != "support_conditioned_endpoint_reconstructed_P_B_I_R"
        or output.semantic_identities.get("transport_endpoint_support_scope")
        != "endpoint_target_T_minus_held_case_c"
        or output.semantic_identities.get("transport_actual_source_prior_scope")
        != "q_not_in_endpoint_target_T_or_source_e"
        or output.semantic_identities.get("transport_donor_source_prior_scope")
        != "q_not_in_outer_H_or_endpoint_target_T_or_source_e"
        or output.semantic_identities.get(
            "transport_source_prior_labels_used_upstream"
        )
        != "true"
        or output.semantic_identities.get(
            "transport_route_local_support_labels_used_upstream"
        )
        != "true"
        or output.semantic_identities.get(
            "transport_held_case_evaluation_capability_used_directly"
        )
        != "false"
        or output.semantic_identities.get(
            "transport_pseudo_evaluation_capability_used_directly"
        )
        != "false"
        or output.semantic_identities.get(
            "transport_terminal_evaluation_capability_used_directly"
        )
        != "false"
        or output.semantic_identities.get("transport_label_free_claim") != "false"
        or output.semantic_identities.get(
            "transport_uses_pre_equivalence_endpoint_crossing_rates"
        )
        != "true"
        or output.semantic_identities.get(
            "transport_screens_sealed_before_pseudo_evaluation_capability_open"
        )
        != "true"
        or output.semantic_identities.get(
            "transport_screens_sealed_before_terminal_evaluation_capability_open"
        )
        != "true"
        or output.semantic_identities.get(
            "transport_identity_level_route_noninterference_required"
        )
        != "true"
        or output.semantic_identities.get(
            "transport_identity_level_route_noninterference_proven"
        )
        != "false"
        or output.semantic_identities.get("transport_authorization_valid") != "false"
        or output.semantic_identities.get("transport_protocol_status")
        != "BLOCKED_IDENTITY_LEVEL_ROUTE_FEEDBACK"
        or output.semantic_identities.get("execution_authorized") != "false"
        or output.semantic_identities.get("fresh_evidence") != "false"
        or output.semantic_identities.get("may_feed_another_experiment") != "false"
        or "oracle_and_diagnostic_evidence" not in output.forbidden_reuse
    ):
        raise ProtocolError("PCSI-PARC workspace catalog drifted.")
    raise ProtocolError(
        "PCSI-PARC execution is blocked: identity-level route noninterference "
        "is unproved for support-conditioned transport. A route-scoped redesign "
        "and poison validation are required."
    )


def validate_workspace_provenance(
    root: Path, config: object
) -> dict[str, Mapping[str, object]]:
    payload = _read_object(root / "provenance/input_artifacts.json")
    if (
        payload.get("schema_version") != "midogpp_input_artifacts_v2"
        or payload.get("dataset_id") != "midogpp"
        or payload.get("experiment_id") != getattr(config, "experiment_id")
        or payload.get("stage") != STAGE_ID
        or payload.get("claim_scope") != CLAIM_SCOPE
    ):
        raise ProtocolError("PCSI-PARC provenance header drifted.")
    raw_rows = payload.get("input_artifacts")
    if not isinstance(raw_rows, list) or not all(
        isinstance(row, Mapping) for row in raw_rows
    ):
        raise ProtocolError("PCSI-PARC provenance rows are malformed.")
    expected_ids = tuple(sorted(getattr(config, "input_artifact_ids")))
    actual_ids = tuple(str(row.get("artifact_id")) for row in raw_rows)
    if actual_ids != expected_ids or len(actual_ids) != 6 or len(set(actual_ids)) != 6:
        raise ProtocolError("PCSI-PARC provenance coverage drifted.")
    rows = {str(row["artifact_id"]): row for row in raw_rows}
    expected_paths = {
        EXPERT_BANK_ARTIFACT_ID: getattr(config, "expert_bank_root"),
        GENERATION_LOCK_ARTIFACT_ID: getattr(config, "generation_lock_root"),
        TEST_CACHE_ARTIFACT_ID: getattr(config, "test_cache_root"),
        TEST_MANIFEST_ARTIFACT_ID: Path(getattr(config, "test_manifest_path")).parent,
        TEST_CONSUMPTION_LEDGER_ARTIFACT_ID: Path(
            getattr(config, "test_consumption_ledger_path")
        ).parent.parent,
        LEDGER_AMENDMENT_ARTIFACT_ID: Path(
            getattr(config, "ledger_amendment_path")
        ).parent,
    }
    for artifact_id, expected_path in expected_paths.items():
        row = rows.get(artifact_id)
        if (
            not isinstance(row, Mapping)
            or Path(str(row.get("resolved_path", ""))).resolve()
            != Path(expected_path).resolve()
            or row.get("exists") is not True
            or not isinstance(row.get("semantic_identities"), Mapping)
            or not isinstance(row.get("file_integrity"), Mapping)
        ):
            raise ProtocolError(f"PCSI-PARC provenance drifted: {artifact_id}.")
    _replay_workspace_manifest(payload, config)
    return {
        artifact_id: rows[artifact_id]
        for artifact_id in getattr(config, "input_artifact_ids")
    }


def _replay_workspace_manifest(payload: Mapping[str, object], config: object) -> None:
    try:
        workspace = MidogppWorkspace.load()
        workspace.validate()
        rendered = workspace._render_run(  # noqa: SLF001 - deliberate audit seam
            str(getattr(config, "experiment_id")),
            require_inputs=True,
            validate_workspace=False,
            include_all_declared_inputs=True,
        )
    except (KeyError, ValueError, OSError) as exc:
        raise ProtocolError("PCSI-PARC provenance replay failed.") from exc
    expected = rendered.input_manifest
    header = (
        "schema_version",
        "dataset_id",
        "experiment_id",
        "stage",
        "claim_scope",
        "selection_used_target_eval_artifacts",
    )
    if any(payload.get(key) != expected.get(key) for key in header) or payload.get(
        "input_artifacts"
    ) != expected.get("input_artifacts"):
        raise ProtocolError("PCSI-PARC provenance replay differs.")


def _read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read PCSI-PARC JSON: {path}.") from exc
    if not isinstance(value, dict):
        raise ProtocolError("PCSI-PARC JSON must be an object.")
    return value


__all__ = ("validate_active_workspace_binding", "validate_workspace_provenance")
