"""Independent closed-world validation for Stage-70 authorization bundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from ....workspace.runtime import MidogppWorkspace
from ....data.contract.stage70_target_evaluation.contracts import (
    EXPECTED_TEST_ROWS_BY_CENTER,
    evaluation_row_id,
    reservation_protocol_payload,
    semantic_sha256,
)
from ...protocol import ProtocolError
from .bundle import (
    FINAL_REQUIRED_FILES,
    RESERVATION_REQUIRED_FILES,
    assert_embedded_hash,
    authorization_decision,
    final_authorization_token,
    final_evaluation_plan,
    final_identity_lock,
    final_protocol_manifest,
    input_provenance,
    leakage_report,
    projected_reservation_payload,
    read_json,
    read_target_identity,
    reservation_evaluation_plan,
    reservation_identity_lock,
    reservation_protocol_manifest,
    run_state,
    sha256_file,
)
from .config import (
    FinalAuthorizationConfig,
    ReservationConfig,
    load_final_authorization_config,
    load_reservation_config,
    validate_final_authorization_config,
    validate_reservation_config,
)
from .contracts import (
    AuthorizationValidationInputs,
    CacheBinding,
    EXPECTED_CENTER_COUNT,
    EXPECTED_EVALUATION_PLAN_ROWS,
    EXPECTED_TEST_ROWS,
    FINAL_AUTHORIZATION_PHASE,
    FINAL_DESCRIPTIVE_STATUS,
    FRESH_CONFIRMATORY_STATUS,
    FinalAuthorizationToken,
    POLICY_ARMS,
    RESERVATION_DESCRIPTIVE_STATUS,
    RESERVATION_PHASE,
    RUN_COMPLETE,
)
from .inputs import (
    load_validated_authorization_inputs,
    load_validated_cache_binding,
)
from .projection import Projector, project_and_validate_target_identity
from .workspace_binding import (
    COMMON_INPUT_IDS,
    DATASET_CONTRACT_ARTIFACT_ID,
    FINAL_INPUT_IDS,
    RESERVATION_INPUT_IDS,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    TEST_SCORING_MANIFEST_ARTIFACT_ID,
    validate_final_production_workspace_binding,
    validate_reservation_production_workspace_binding,
)


def validate_target_evaluation_reservation(
    root: str | Path,
    *,
    config: ReservationConfig,
    validation_inputs: AuthorizationValidationInputs | None = None,
    projected_reservation: object | None = None,
    projector: Projector | None = None,
    allow_pending: bool = False,
) -> dict[str, object]:
    """Reconstruct and validate the cache-extraction-only reservation."""

    validate_reservation_config(config)
    if projected_reservation is not None or projector is not None:
        _require_test_injection(config)
    inputs = _resolve_inputs(config, validation_inputs)
    projected_object = projected_reservation or project_and_validate_target_identity(
        config,
        projector=projector,
    )
    return _validate_reservation_with_projected_identity(
        Path(root),
        config=config,
        inputs=inputs,
        projected_object=projected_object,
        allow_pending=allow_pending,
    )


def _validate_reservation_with_projected_identity(
    path: Path,
    *,
    config: ReservationConfig,
    inputs: AuthorizationValidationInputs,
    projected_object: object,
    allow_pending: bool,
) -> dict[str, object]:
    validate_reservation_config(config)
    required = tuple(
        member
        for member in RESERVATION_REQUIRED_FILES
        if not (allow_pending and member == "reports/validation_report.json")
    )
    _assert_closed_world(path, required)
    projected = projected_reservation_payload(projected_object)
    manifest_hash = sha256_file(config.scoring_manifest_path)
    if (
        manifest_hash != config.expected_scoring_manifest_sha256
        or projected["manifest_sha256"] != manifest_hash
    ):
        raise ProtocolError("Stage-70 reservation scoring-manifest identity drifted.")

    _require_config_snapshot(path, config)
    identity = reservation_identity_lock(config, projected)
    plan = reservation_evaluation_plan(config, identity)
    protocol = reservation_protocol_manifest(config, inputs, identity, plan)
    expected_binding = input_provenance(
        config=config,
        inputs=inputs,
        scoring_manifest_sha256=manifest_hash,
    )
    _require_exact_json(
        path / "manifests/input_binding.json",
        expected_binding,
        "reservation input binding",
    )
    _require_provenance(
        path / "provenance/input_artifacts.json",
        config=config,
        expected_test_binding=expected_binding,
    )
    _require_exact_json(
        path / "manifests/identity_lock.json", identity, "reservation identity lock"
    )
    _require_exact_json(
        path / "manifests/evaluation_plan.json", plan, "reservation evaluation plan"
    )
    _require_exact_json(
        path / "manifests/protocol_manifest.json", protocol, "reservation protocol"
    )
    observed_rows = read_target_identity(path / "tables/target_identity.csv")
    expected_rows = projected["rows"]
    if observed_rows != expected_rows:
        raise ProtocolError("Stage-70 target-identity table drifted from the projector.")
    _require_exact_json(
        path / "reports/authorization_decision.json",
        authorization_decision(final=False),
        "reservation decision",
    )
    _require_exact_json(
        path / "reports/leakage_report.json",
        leakage_report(final=False),
        "reservation leakage report",
    )
    _require_exact_json(
        path / "reports/run_state.json",
        run_state(final=False, status=RUN_COMPLETE),
        "reservation run state",
    )
    content = _validate_content_index(path, RESERVATION_REQUIRED_FILES)
    checks = {
        "status": "PASS",
        "phase": RESERVATION_PHASE,
        "fresh_confirmatory_status": FRESH_CONFIRMATORY_STATUS,
        "descriptive_status": RESERVATION_DESCRIPTIVE_STATUS,
        "row_count": EXPECTED_TEST_ROWS,
        "center_count": EXPECTED_CENTER_COUNT,
        "target_evaluation_reservation_id": projected["reservation_id"],
        "target_evaluation_reservation_protocol_hash": projected["protocol_hash"],
        "target_identity_table_hash": projected["target_identity_table_hash"],
        "authorization_protocol_hash": protocol["protocol_hash"],
        "content_hash": content["content_hash"],
        "prediction_performed": False,
        "metric_scoring_performed": False,
        "target_labels_opened": False,
    }
    if not allow_pending:
        _require_validation_report(
            path / "reports/validation_report.json",
            schema="midogpp_stage70_target_evaluation_reservation_validation_v1",
            validator="validate_target_evaluation_reservation",
            checks=checks,
        )
    return checks


def validate_final_prediction_authorization(
    root: str | Path,
    *,
    config: FinalAuthorizationConfig,
    validation_inputs: AuthorizationValidationInputs | None = None,
    projected_reservation: object | None = None,
    projector: Projector | None = None,
    cache_binding: CacheBinding | None = None,
    allow_pending: bool = False,
) -> dict[str, object]:
    """Reconstruct and validate the prediction-only final authorization."""

    validate_final_authorization_config(config)
    if projected_reservation is not None or projector is not None:
        _require_test_injection(config)
    path = Path(root)
    required = tuple(
        member
        for member in FINAL_REQUIRED_FILES
        if not (allow_pending and member == "reports/validation_report.json")
    )
    _assert_closed_world(path, required)
    inputs = _resolve_inputs(config, validation_inputs)
    reservation_checks = _validate_reservation_for_final_authorization(
        config,
        inputs=inputs,
        projected_reservation=projected_reservation,
        projector=projector,
    )
    reservation_identity = read_json(
        config.reservation_root / "manifests/identity_lock.json"
    )
    reservation_content = read_json(
        config.reservation_root / "manifests/content_index.json"
    )
    cache = _resolve_cache(config, cache_binding)
    _validate_cache_reservation_binding(config, reservation_identity, cache)
    manifest_hash = sha256_file(config.scoring_manifest_path)
    if manifest_hash != config.expected_scoring_manifest_sha256:
        raise ProtocolError("Stage-70 final scoring-manifest identity drifted.")

    _require_config_snapshot(path, config)
    identity = final_identity_lock(
        config,
        reservation_identity,
        str(reservation_content.get("content_hash", "")),
        cache,
    )
    plan = final_evaluation_plan(config, inputs, identity, cache)
    protocol = final_protocol_manifest(config, inputs, identity, plan, cache)
    token = final_authorization_token(config, inputs, identity, plan, protocol, cache)
    expected_binding = input_provenance(
        config=config,
        inputs=inputs,
        scoring_manifest_sha256=manifest_hash,
        reservation_content_hash=str(reservation_content.get("content_hash", "")),
        cache=cache,
    )
    _require_exact_json(
        path / "manifests/input_binding.json",
        expected_binding,
        "final authorization input binding",
    )
    _require_provenance(
        path / "provenance/input_artifacts.json",
        config=config,
        expected_test_binding=expected_binding,
    )
    _require_exact_json(
        path / "manifests/identity_lock.json", identity, "final identity lock"
    )
    _require_exact_json(
        path / "manifests/evaluation_plan.json", plan, "final evaluation plan"
    )
    _require_exact_json(
        path / "manifests/protocol_manifest.json", protocol, "final protocol"
    )
    observed_token = read_final_authorization_token(path)
    if observed_token.to_payload() != token.to_payload():
        raise ProtocolError("Stage-70 final authorization token drifted.")
    _require_exact_json(
        path / "reports/authorization_decision.json",
        authorization_decision(final=True),
        "final authorization decision",
    )
    _require_exact_json(
        path / "reports/leakage_report.json",
        leakage_report(final=True),
        "final authorization leakage report",
    )
    _require_exact_json(
        path / "reports/run_state.json",
        run_state(final=True, status=RUN_COMPLETE),
        "final authorization run state",
    )
    content = _validate_content_index(path, FINAL_REQUIRED_FILES)
    checks = {
        "status": "PASS",
        "phase": FINAL_AUTHORIZATION_PHASE,
        "fresh_confirmatory_status": FRESH_CONFIRMATORY_STATUS,
        "descriptive_status": FINAL_DESCRIPTIVE_STATUS,
        "reservation_status": reservation_checks["status"],
        "row_count": cache.row_count,
        "center_count": len(cache.rows_by_center),
        "policy_count": len(POLICY_ARMS),
        "evaluation_plan_rows": EXPECTED_EVALUATION_PLAN_ROWS,
        "synthetic_rows_per_class": 1024,
        "authorization_token_hash": token.authorization_token_hash,
        "authorization_protocol_hash": protocol["protocol_hash"],
        "content_hash": content["content_hash"],
        "prediction_performed": False,
        "metric_scoring_performed": False,
        "target_labels_opened": False,
    }
    if not allow_pending:
        _require_validation_report(
            path / "reports/validation_report.json",
            schema="midogpp_stage70_final_prediction_authorization_validation_v1",
            validator="validate_final_prediction_authorization",
            checks=checks,
        )
    return checks


def read_final_authorization_token(root: str | Path) -> FinalAuthorizationToken:
    path = Path(root) / "manifests/authorization_token.json"
    return FinalAuthorizationToken(read_json(path))


def _validate_reservation_for_final_authorization(
    config: FinalAuthorizationConfig,
    *,
    inputs: AuthorizationValidationInputs,
    projected_reservation: object | None,
    projector: Projector | None,
) -> dict[str, object]:
    if config.production_workspace_binding is not True:
        return validate_target_evaluation_reservation(
            config.reservation_root,
            config=reservation_config_for_final(config),
            validation_inputs=inputs,
            projected_reservation=projected_reservation,
            projector=projector,
        )
    reservation_config = load_reservation_config(
        config.reservation_root / "config.resolved.yaml"
    )
    if reservation_config != reservation_config_for_final(config):
        raise ProtocolError("Stage-70 final/reservation configuration binding drifted.")
    return _validate_reservation_with_projected_identity(
        config.reservation_root,
        config=reservation_config,
        inputs=inputs,
        projected_object=_sealed_reservation_projection(
            config.reservation_root,
            manifest_sha256=config.expected_scoring_manifest_sha256,
        ),
        allow_pending=False,
    )


def _sealed_reservation_projection(
    root: Path,
    *,
    manifest_sha256: str,
) -> dict[str, object]:
    """Reconstruct safe reservation identity without parsing scoring rows."""

    identity = read_json(root / "manifests/identity_lock.json")
    observed_rows = read_target_identity(root / "tables/target_identity.csv")
    expected_counts = dict(EXPECTED_TEST_ROWS_BY_CENTER)
    indices = [int(row["contract_row_index"]) for row in observed_rows]
    if indices != sorted(indices) or len(indices) != len(set(indices)):
        raise ProtocolError("Stage-70 sealed reservation row order drifted.")
    rows: list[dict[str, object]] = []
    for row in observed_rows:
        index = int(row["contract_row_index"])
        expected_row_id = evaluation_row_id(manifest_sha256, index)
        if row["evaluation_row_id"] != expected_row_id:
            raise ProtocolError("Stage-70 sealed evaluation-row identity drifted.")
        rows.append(
            {
                "evaluation_row_id": expected_row_id,
                "contract_row_index": index,
                "center": str(row["target_center"]),
                "split": str(row["split"]),
            }
        )
    protocol_hash = semantic_sha256(
        reservation_protocol_payload(
            manifest_sha256=manifest_sha256,
            expected_rows_by_center=expected_counts,
            coverage_scope="canonical",
        )
    )
    reservation_id = "reservation_" + semantic_sha256(
        {
            "protocol_hash": protocol_hash,
            "evaluation_row_ids": [row["evaluation_row_id"] for row in rows],
        }
    )
    if (
        identity.get("scoring_manifest_sha256") != manifest_sha256
        or identity.get("target_evaluation_reservation_protocol_hash")
        != protocol_hash
        or identity.get("target_evaluation_reservation_id") != reservation_id
    ):
        raise ProtocolError("Stage-70 sealed reservation identity drifted.")
    return {
        "manifest_sha256": manifest_sha256,
        "reservation_id": reservation_id,
        "protocol_hash": protocol_hash,
        "rows": rows,
        "rows_by_center": expected_counts,
    }


def reservation_config_for_final(config: FinalAuthorizationConfig) -> ReservationConfig:
    return ReservationConfig(
        artifact_root=config.reservation_root,
        canonical_reference_root=config.canonical_reference_root,
        bank_root=config.bank_root,
        generation_lock_root=config.generation_lock_root,
        equal_union_policy_root=config.equal_union_policy_root,
        metadata_policy_root=config.metadata_policy_root,
        utility_policy_root=config.utility_policy_root,
        scoring_manifest_path=config.scoring_manifest_path,
        test_consumption_ledger_path=(
            config.canonical_reference_root / "reports/test_consumption_ledger.json"
        ),
        prospective_cache_root=config.cache_root,
        expected_scoring_manifest_sha256=config.expected_scoring_manifest_sha256,
        expected_cache_extractor_protocol_hash=(
            config.expected_cache_extractor_protocol_hash
        ),
        cache_experiment_id=config.cache_experiment_id,
        cache_artifact_id=config.cache_artifact_id,
        consumer_experiment_id=config.consumer_experiment_id,
        purpose=config.purpose,
        claim_scope=config.claim_scope,
        expected_test_rows=config.expected_test_rows,
        production_workspace_binding=config.production_workspace_binding,
        allow_test_validation_injection=config.allow_test_validation_injection,
    )


def _resolve_inputs(
    config: ReservationConfig | FinalAuthorizationConfig,
    injected: AuthorizationValidationInputs | None,
) -> AuthorizationValidationInputs:
    if injected is not None:
        _require_test_injection(config)
        return injected
    if config.production_workspace_binding is not True:
        raise ProtocolError("Stage-70 production validation requires workspace binding.")
    if isinstance(config, ReservationConfig):
        validate_reservation_production_workspace_binding(config)
    else:
        validate_final_production_workspace_binding(config)
    return load_validated_authorization_inputs(config)


def _resolve_cache(
    config: FinalAuthorizationConfig,
    injected: CacheBinding | None,
) -> CacheBinding:
    if injected is not None:
        _require_test_injection(config)
        return injected
    return load_validated_cache_binding(config)


def _require_test_injection(
    config: ReservationConfig | FinalAuthorizationConfig,
) -> None:
    if (
        config.allow_test_validation_injection is not True
        or config.production_workspace_binding is not False
    ):
        raise ProtocolError(
            "Stage-70 injected validation evidence is allowed only by explicit "
            "non-production test configuration."
        )


def _validate_cache_reservation_binding(
    config: FinalAuthorizationConfig,
    identity: Mapping[str, object],
    cache: CacheBinding,
) -> None:
    exact = {
        "artifact_id": (cache.artifact_id, config.cache_artifact_id),
        "manifest_sha256": (
            cache.manifest_sha256,
            config.expected_scoring_manifest_sha256,
        ),
        "reservation_id": (
            cache.target_evaluation_reservation_id,
            identity.get("target_evaluation_reservation_id"),
        ),
        "reservation_protocol_hash": (
            cache.target_evaluation_reservation_protocol_hash,
            identity.get("target_evaluation_reservation_protocol_hash"),
        ),
        "cache_extractor_protocol_hash": (
            cache.cache_extractor_protocol_hash,
            config.expected_cache_extractor_protocol_hash,
        ),
        "row_count": (cache.row_count, identity.get("row_count")),
        "rows_by_center": (dict(cache.rows_by_center), identity.get("rows_by_center")),
    }
    mismatch = [key for key, values in exact.items() if values[0] != values[1]]
    if mismatch:
        raise ProtocolError(
            f"Stage-70 cache/reservation binding drifted: {mismatch}."
        )


def _require_config_snapshot(
    root: Path,
    expected: ReservationConfig | FinalAuthorizationConfig,
) -> None:
    snapshot = root / "config.resolved.yaml"
    if expected.production_workspace_binding is True:
        loaded = (
            load_reservation_config(snapshot)
            if isinstance(expected, ReservationConfig)
            else load_final_authorization_config(snapshot)
        )
        if loaded != expected or loaded.contract_hash != expected.contract_hash:
            raise ProtocolError("Stage-70 resolved config snapshot drifted.")
        return
    try:
        payload = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolError("Cannot read Stage-70 resolved config snapshot.") from exc
    if payload != expected.to_payload():
        raise ProtocolError("Stage-70 resolved config snapshot drifted.")


def _require_provenance(
    path: Path,
    *,
    config: ReservationConfig | FinalAuthorizationConfig,
    expected_test_binding: Mapping[str, object],
) -> None:
    payload = read_json(path)
    if payload == dict(expected_test_binding):
        _require_test_injection(config)
        return
    top_fields = {
        "schema_version",
        "dataset_id",
        "experiment_id",
        "stage",
        "claim_scope",
        "selection_used_target_eval_artifacts",
        "input_artifacts",
        "repository_revision",
        "repository_dirty",
        "repository_status_hash",
    }
    if set(payload) != top_fields or any(
        payload.get(key) != value
        for key, value in {
            "schema_version": "midogpp_input_artifacts_v2",
            "dataset_id": "midogpp",
            "experiment_id": config.experiment_id,
            "stage": "70_frozen_policy_downstream",
            "claim_scope": config.claim_scope,
            "selection_used_target_eval_artifacts": False,
        }.items()
    ):
        raise ProtocolError("Stage-70 workspace provenance identity drifted.")
    if (
        not _is_hex(payload.get("repository_revision"), 40)
        or not isinstance(payload.get("repository_dirty"), bool)
        or not _is_hex(payload.get("repository_status_hash"), 64)
    ):
        raise ProtocolError("Stage-70 workspace repository provenance is malformed.")

    expected_ids = (
        RESERVATION_INPUT_IDS
        if isinstance(config, ReservationConfig)
        else FINAL_INPUT_IDS
    )
    raw_rows = payload.get("input_artifacts")
    if not isinstance(raw_rows, list) or not all(
        isinstance(row, Mapping) for row in raw_rows
    ):
        raise ProtocolError("Stage-70 workspace input provenance is malformed.")
    observed_ids = [str(row.get("artifact_id", "")) for row in raw_rows]
    if observed_ids != sorted(expected_ids) or len(set(observed_ids)) != len(
        observed_ids
    ):
        raise ProtocolError("Stage-70 workspace provenance input coverage drifted.")

    expected_roots = _provenance_input_roots(config)
    if set(expected_roots) != set(expected_ids):
        raise ProtocolError("Stage-70 internal provenance path coverage drifted.")
    workspace = None
    if config.production_workspace_binding is True:
        workspace = MidogppWorkspace.load()
        workspace.validate()
    for row in raw_rows:
        artifact_id = str(row.get("artifact_id", ""))
        _validate_workspace_provenance_row(
            row,
            artifact_id=artifact_id,
            expected_root=expected_roots[artifact_id],
            workspace=workspace,
        )


def _provenance_input_roots(
    config: ReservationConfig | FinalAuthorizationConfig,
) -> dict[str, Path]:
    (
        reference_id,
        bank_id,
        generation_id,
        equal_id,
        metadata_id,
        utility_id,
    ) = COMMON_INPUT_IDS
    roots = {
        reference_id: config.canonical_reference_root,
        bank_id: config.bank_root,
        generation_id: config.generation_lock_root,
        equal_id: config.equal_union_policy_root,
        metadata_id: config.metadata_policy_root,
        utility_id: config.utility_policy_root,
    }
    if isinstance(config, ReservationConfig):
        roots.update(
            {
                DATASET_CONTRACT_ARTIFACT_ID: config.scoring_manifest_path.parent,
                TEST_CONSUMPTION_LEDGER_ARTIFACT_ID: (
                    config.test_consumption_ledger_path.parent.parent
                ),
            }
        )
    else:
        roots.update(
            {
                TEST_SCORING_MANIFEST_ARTIFACT_ID: (
                    config.scoring_manifest_path.parent
                ),
                config.reservation_artifact_id: config.reservation_root,
                config.cache_artifact_id: config.cache_root,
            }
        )
    return roots


def _validate_workspace_provenance_row(
    row: Mapping[str, object],
    *,
    artifact_id: str,
    expected_root: Path,
    workspace: MidogppWorkspace | None,
) -> None:
    row_fields = {
        "artifact_id",
        "resolved_path",
        "stage",
        "evidence_label",
        "claim_scope",
        "semantic_identities",
        "semantic_identities_are_file_hashes",
        "file_integrity",
        "exists",
    }
    identities = row.get("semantic_identities")
    if (
        set(row) != row_fields
        or Path(str(row.get("resolved_path", ""))).resolve()
        != expected_root.resolve()
        or row.get("exists") is not True
        or row.get("semantic_identities_are_file_hashes") is not False
        or not isinstance(row.get("stage"), str)
        or not str(row.get("stage", ""))
        or not isinstance(row.get("evidence_label"), str)
        or not str(row.get("evidence_label", ""))
        or not isinstance(row.get("claim_scope"), str)
        or not str(row.get("claim_scope", ""))
        or not isinstance(identities, Mapping)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in identities.items()
        )
    ):
        raise ProtocolError(f"Stage-70 workspace provenance drifted: {artifact_id}.")

    expected_inventory: set[str] | None = None
    has_catalog_expectations: bool | None = None
    if workspace is not None:
        artifact = workspace.artifacts.get(artifact_id)
        if artifact is None or any(
            row.get(key) != value
            for key, value in {
                "stage": artifact.stage,
                "evidence_label": artifact.evidence_label,
                "claim_scope": artifact.claim_scope,
                "semantic_identities": dict(artifact.semantic_identities),
            }.items()
        ):
            raise ProtocolError(
                f"Stage-70 workspace catalog provenance drifted: {artifact_id}."
            )
        expected_inventory = set(artifact.provenance_files)
        has_catalog_expectations = bool(artifact.expected_file_hashes)

    integrity = row.get("file_integrity")
    if not isinstance(integrity, Mapping) or set(integrity) != {
        "status",
        "default_recording_algorithm",
        "files",
    } or integrity.get("default_recording_algorithm") != "sha256":
        raise ProtocolError(
            f"Stage-70 workspace file integrity drifted: {artifact_id}."
        )
    files = integrity.get("files")
    if not isinstance(files, list) or not all(
        isinstance(item, Mapping) for item in files
    ):
        raise ProtocolError(
            f"Stage-70 workspace file inventory is malformed: {artifact_id}."
        )
    inventory = [str(item.get("path", "")) for item in files]
    if len(set(inventory)) != len(inventory) or (
        expected_inventory is not None and set(inventory) != expected_inventory
    ):
        raise ProtocolError(
            f"Stage-70 workspace file coverage drifted: {artifact_id}."
        )
    has_expectations = False
    for item in files:
        has_expectations = (
            _validate_workspace_file_row(
                item,
                artifact_id=artifact_id,
                expected_root=expected_root,
            )
            or has_expectations
        )
    if has_catalog_expectations is not None:
        has_expectations = has_catalog_expectations
    expected_status = (
        "EXPECTED_FILE_HASHES_MATCH"
        if has_expectations
        else (
            "HASHES_RECORDED_NO_EXPECTATIONS"
            if files
            else "NO_PROVENANCE_FILES_DECLARED"
        )
    )
    if integrity.get("status") != expected_status:
        raise ProtocolError(
            f"Stage-70 workspace integrity status drifted: {artifact_id}."
        )


def _validate_workspace_file_row(
    item: Mapping[str, object],
    *,
    artifact_id: str,
    expected_root: Path,
) -> bool:
    fields = {
        "path",
        "resolved_path",
        "exists",
        "expected",
        "size_bytes",
        "computed",
        "verification",
    }
    relative = str(item.get("path", ""))
    relative_path = Path(relative)
    member = (expected_root.resolve() / relative_path).resolve()
    computed = item.get("computed")
    if (
        set(item) != fields
        or not relative
        or relative_path.is_absolute()
        or ".." in relative_path.parts
        or not member.is_relative_to(expected_root.resolve())
        or Path(str(item.get("resolved_path", ""))).resolve() != member
        or item.get("exists") is not True
        or not member.is_file()
        or item.get("size_bytes") != member.stat().st_size
        or not isinstance(computed, Mapping)
        or "sha256" not in computed
        or computed.get("sha256") != sha256_file(member)
    ):
        raise ProtocolError(
            f"Stage-70 workspace input member drifted: {artifact_id}:{relative}."
        )
    expected = item.get("expected")
    if expected is None:
        if set(computed) != {"sha256"} or item.get("verification") != (
            "RECORDED_NO_EXPECTATION"
        ):
            raise ProtocolError(
                f"Stage-70 workspace input verification drifted: {artifact_id}:{relative}."
            )
        return False
    if (
        not isinstance(expected, Mapping)
        or set(expected) != {"algorithm", "digest"}
        or not isinstance(expected.get("algorithm"), str)
        or not isinstance(expected.get("digest"), str)
    ):
        raise ProtocolError(
            f"Stage-70 workspace expected hash is malformed: {artifact_id}:{relative}."
        )
    algorithm = str(expected["algorithm"])
    digest = str(expected["digest"])
    wanted_computed = {"sha256", algorithm}
    if (
        set(computed) != wanted_computed
        or not _is_hex(digest, hashlib.new(algorithm).digest_size * 2)
        or computed.get(algorithm) != _hash_file(member, algorithm)
        or digest != computed.get(algorithm)
        or item.get("verification") != "MATCH"
    ):
        raise ProtocolError(
            f"Stage-70 workspace expected hash failed: {artifact_id}:{relative}."
        )
    return True


def _hash_file(path: Path, algorithm: str) -> str:
    try:
        digest = hashlib.new(algorithm)
    except ValueError as exc:
        raise ProtocolError(f"Unsupported Stage-70 provenance hash: {algorithm}.") from exc
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ProtocolError(f"Cannot hash Stage-70 provenance member: {path}.") from exc
    return digest.hexdigest()


def _is_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_exact_json(
    path: Path,
    expected: Mapping[str, object],
    label: str,
) -> None:
    if read_json(path) != dict(expected):
        raise ProtocolError(f"Stage-70 {label} drifted.")


def _validate_content_index(
    root: Path,
    required_files: Sequence[str],
) -> Mapping[str, object]:
    payload = read_json(root / "manifests/content_index.json")
    assert_embedded_hash(payload, "content_hash")
    records = payload.get("records")
    if (
        payload.get("schema_version")
        != "midogpp_stage70_authorization_content_index_v1"
        or not isinstance(records, list)
    ):
        raise ProtocolError("Stage-70 content-index schema drifted.")
    excluded = {
        "manifests/content_index.json",
        "reports/run_state.json",
        "reports/validation_report.json",
    }
    expected = set(required_files) - excluded
    observed: set[str] = set()
    for row in records:
        if not isinstance(row, Mapping) or set(row) != {
            "relative_path",
            "sha256",
            "size_bytes",
        }:
            raise ProtocolError("Stage-70 content-index row is malformed.")
        relative = str(row.get("relative_path", ""))
        relative_path = Path(relative)
        member = root / relative_path
        if (
            not relative
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative in observed
            or not member.is_file()
            or member.is_symlink()
            or row.get("sha256") != sha256_file(member)
            or row.get("size_bytes") != member.stat().st_size
        ):
            raise ProtocolError("Stage-70 content-index member drifted.")
        observed.add(relative)
    if observed != expected:
        raise ProtocolError("Stage-70 content-index coverage drifted.")
    return payload


def _require_validation_report(
    path: Path,
    *,
    schema: str,
    validator: str,
    checks: Mapping[str, object],
) -> None:
    expected = {
        "schema_version": schema,
        "status": "PASS",
        "validator": validator,
        "checks": dict(checks),
    }
    _require_exact_json(path, expected, "validation report")


def _assert_closed_world(root: Path, required_files: Sequence[str]) -> None:
    if not root.is_dir() or root.is_symlink():
        raise ProtocolError("Stage-70 authorization root is absent or a symlink.")
    symlinks = sorted(
        member.relative_to(root).as_posix()
        for member in root.rglob("*")
        if member.is_symlink()
    )
    if symlinks:
        raise ProtocolError(f"Stage-70 authorization contains symlinks: {symlinks}.")
    actual = {
        member.relative_to(root).as_posix()
        for member in root.rglob("*")
        if member.is_file()
    }
    expected = set(required_files)
    if actual != expected:
        raise ProtocolError(
            "Stage-70 authorization closed-world coverage drifted: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}."
        )


__all__ = (
    "read_final_authorization_token",
    "reservation_config_for_final",
    "validate_final_prediction_authorization",
    "validate_target_evaluation_reservation",
)
