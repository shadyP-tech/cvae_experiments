"""Staged runners for Stage-70 reservation and final prediction authorization."""

from __future__ import annotations

from pathlib import Path

from ....common.staged_directory import staged_directory, staged_existing_directory
from ...protocol import ProtocolError
from .bundle import (
    FINAL_REQUIRED_FILES,
    RESERVATION_REQUIRED_FILES,
    authorization_decision,
    final_authorization_token,
    final_evaluation_plan,
    final_identity_lock,
    final_protocol_manifest,
    input_provenance,
    leakage_report,
    projected_reservation_payload,
    read_json,
    reservation_evaluation_plan,
    reservation_identity_lock,
    reservation_protocol_manifest,
    run_state,
    sha256_file,
    write_content_index,
    write_json,
    write_resolved_config,
    write_target_identity,
)
from .config import (
    FinalAuthorizationConfig,
    ReservationConfig,
    validate_final_authorization_config,
    validate_reservation_config,
)
from .contracts import AuthorizationValidationInputs, CacheBinding
from .inputs import (
    load_validated_authorization_inputs,
    load_validated_cache_binding,
)
from .projection import (
    Projector,
    project_and_validate_target_identity,
    validate_cache_extractor_protocol_hash,
)
from .validation import (
    _validate_reservation_for_final_authorization,
    validate_final_prediction_authorization,
    validate_target_evaluation_reservation,
)
from .workspace_binding import (
    validate_final_production_workspace_binding,
    validate_reservation_production_workspace_binding,
)


def run_target_evaluation_reservation(
    config: ReservationConfig,
    *,
    validation_inputs: AuthorizationValidationInputs | None = None,
    projector: Projector | None = None,
) -> Path:
    """Reserve the consumed test rows for label-blind cache extraction only."""

    validate_reservation_config(config)
    final_root = Path(config.artifact_root)
    prepared = _is_exact_prepared_root(final_root)
    if final_root.exists() and not prepared:
        validate_target_evaluation_reservation(
            final_root,
            config=config,
            validation_inputs=validation_inputs,
            projector=projector,
        )
        return final_root
    if config.production_workspace_binding is True and not prepared:
        raise ProtocolError(
            "Stage-70 production reservation must start from an exact workspace "
            "preparation containing the resolved config and input provenance."
        )
    inputs = _runner_inputs(config, validation_inputs)
    injected = validation_inputs is not None or projector is not None
    if projector is not None:
        _require_test_injection(config)
    validate_cache_extractor_protocol_hash(
        config,
        skip_public_check=injected,
    )
    scoring_manifest_sha256 = sha256_file(config.scoring_manifest_path)
    if scoring_manifest_sha256 != config.expected_scoring_manifest_sha256:
        raise ProtocolError("Stage-70 reservation scoring-manifest hash drifted.")
    projected_object = project_and_validate_target_identity(config, projector=projector)
    projected = projected_reservation_payload(projected_object)
    if projected["manifest_sha256"] != scoring_manifest_sha256:
        raise ProtocolError("Stage-70 projector/scoring-manifest binding drifted.")

    transaction = staged_existing_directory if prepared else staged_directory
    with transaction(final_root) as root:
        write_json(root / "reports/run_state.json", run_state(final=False, status="RUNNING"))
        try:
            if not prepared:
                write_resolved_config(root / "config.resolved.yaml", config)
            identity = reservation_identity_lock(config, projected)
            plan = reservation_evaluation_plan(config, identity)
            protocol = reservation_protocol_manifest(config, inputs, identity, plan)
            binding = input_provenance(
                config=config,
                inputs=inputs,
                scoring_manifest_sha256=scoring_manifest_sha256,
            )
            if not prepared:
                write_json(
                    root / "provenance/input_artifacts.json",
                    binding,
                )
            write_json(root / "manifests/input_binding.json", binding)
            write_json(root / "manifests/identity_lock.json", identity)
            write_json(root / "manifests/evaluation_plan.json", plan)
            write_json(root / "manifests/protocol_manifest.json", protocol)
            write_target_identity(
                root / "tables/target_identity.csv",
                projected["rows"],  # type: ignore[arg-type]
            )
            write_json(
                root / "reports/authorization_decision.json",
                authorization_decision(final=False),
            )
            write_json(
                root / "reports/leakage_report.json",
                leakage_report(final=False),
            )
            write_content_index(root, RESERVATION_REQUIRED_FILES)
            write_json(
                root / "reports/run_state.json",
                run_state(final=False, status="COMPLETE"),
            )
            checks = validate_target_evaluation_reservation(
                root,
                config=config,
                validation_inputs=(inputs if injected else None),
                projected_reservation=(projected_object if injected else None),
                allow_pending=True,
            )
            write_json(
                root / "reports/validation_report.json",
                {
                    "schema_version": (
                        "midogpp_stage70_target_evaluation_reservation_validation_v1"
                    ),
                    "status": "PASS",
                    "validator": "validate_target_evaluation_reservation",
                    "checks": checks,
                },
            )
            validate_target_evaluation_reservation(
                root,
                config=config,
                validation_inputs=(inputs if injected else None),
                projected_reservation=(projected_object if injected else None),
            )
        except Exception:
            write_json(
                root / "reports/run_state.json",
                run_state(final=False, status="FAILED"),
            )
            raise
    return final_root


def run_final_prediction_authorization(
    config: FinalAuthorizationConfig,
    *,
    validation_inputs: AuthorizationValidationInputs | None = None,
    projected_reservation: object | None = None,
    projector: Projector | None = None,
    cache_binding: CacheBinding | None = None,
) -> Path:
    """Authorize only frozen locked-policy predictions; scoring stays sealed."""

    validate_final_authorization_config(config)
    final_root = Path(config.artifact_root)
    prepared = _is_exact_prepared_root(final_root)
    if final_root.exists() and not prepared:
        validate_final_prediction_authorization(
            final_root,
            config=config,
            validation_inputs=validation_inputs,
            projected_reservation=projected_reservation,
            projector=projector,
            cache_binding=cache_binding,
        )
        return final_root
    if config.production_workspace_binding is True and not prepared:
        raise ProtocolError(
            "Stage-70 production final authorization must start from an exact "
            "workspace preparation containing the resolved config and input provenance."
        )
    injected = any(
        item is not None
        for item in (
            validation_inputs,
            projected_reservation,
            projector,
            cache_binding,
        )
    )
    if injected:
        _require_test_injection(config)
    inputs = _runner_inputs(config, validation_inputs)
    cache = _runner_cache(config, cache_binding)
    reservation_checks = _validate_reservation_for_final_authorization(
        config,
        inputs=inputs,
        projected_reservation=projected_reservation,
        projector=projector,
    )
    if reservation_checks.get("status") != "PASS":
        raise ProtocolError("Stage-70 reservation did not validate PASS.")
    reservation_identity = read_json(
        config.reservation_root / "manifests/identity_lock.json"
    )
    reservation_content = read_json(
        config.reservation_root / "manifests/content_index.json"
    )
    _require_cache_binding(config, reservation_identity, cache)
    scoring_manifest_sha256 = sha256_file(config.scoring_manifest_path)
    if scoring_manifest_sha256 != config.expected_scoring_manifest_sha256:
        raise ProtocolError("Stage-70 final scoring-manifest hash drifted.")

    transaction = staged_existing_directory if prepared else staged_directory
    with transaction(final_root) as root:
        write_json(root / "reports/run_state.json", run_state(final=True, status="RUNNING"))
        try:
            if not prepared:
                write_resolved_config(root / "config.resolved.yaml", config)
            identity = final_identity_lock(
                config,
                reservation_identity,
                str(reservation_content.get("content_hash", "")),
                cache,
            )
            plan = final_evaluation_plan(config, inputs, identity, cache)
            protocol = final_protocol_manifest(config, inputs, identity, plan, cache)
            token = final_authorization_token(
                config,
                inputs,
                identity,
                plan,
                protocol,
                cache,
            )
            binding = input_provenance(
                config=config,
                inputs=inputs,
                scoring_manifest_sha256=scoring_manifest_sha256,
                reservation_content_hash=str(
                    reservation_content.get("content_hash", "")
                ),
                cache=cache,
            )
            if not prepared:
                write_json(
                    root / "provenance/input_artifacts.json",
                    binding,
                )
            write_json(root / "manifests/input_binding.json", binding)
            write_json(root / "manifests/identity_lock.json", identity)
            write_json(root / "manifests/evaluation_plan.json", plan)
            write_json(root / "manifests/protocol_manifest.json", protocol)
            write_json(root / "manifests/authorization_token.json", token.to_payload())
            write_json(
                root / "reports/authorization_decision.json",
                authorization_decision(final=True),
            )
            write_json(root / "reports/leakage_report.json", leakage_report(final=True))
            write_content_index(root, FINAL_REQUIRED_FILES)
            write_json(
                root / "reports/run_state.json",
                run_state(final=True, status="COMPLETE"),
            )
            checks = validate_final_prediction_authorization(
                root,
                config=config,
                validation_inputs=(inputs if injected else None),
                projected_reservation=projected_reservation,
                projector=projector,
                cache_binding=(cache if injected else None),
                allow_pending=True,
            )
            write_json(
                root / "reports/validation_report.json",
                {
                    "schema_version": (
                        "midogpp_stage70_final_prediction_authorization_validation_v1"
                    ),
                    "status": "PASS",
                    "validator": "validate_final_prediction_authorization",
                    "checks": checks,
                },
            )
            validate_final_prediction_authorization(
                root,
                config=config,
                validation_inputs=(inputs if injected else None),
                projected_reservation=projected_reservation,
                projector=projector,
                cache_binding=(cache if injected else None),
            )
        except Exception:
            write_json(
                root / "reports/run_state.json",
                run_state(final=True, status="FAILED"),
            )
            raise
    return final_root


def _runner_inputs(
    config: ReservationConfig | FinalAuthorizationConfig,
    injected: AuthorizationValidationInputs | None,
) -> AuthorizationValidationInputs:
    if injected is not None:
        _require_test_injection(config)
        return injected
    if config.production_workspace_binding is not True:
        raise ProtocolError("Stage-70 production runner requires workspace binding.")
    if isinstance(config, ReservationConfig):
        validate_reservation_production_workspace_binding(config)
    else:
        validate_final_production_workspace_binding(config)
    return load_validated_authorization_inputs(config)


def _runner_cache(
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
            "Stage-70 test injection requires an explicit non-production config."
        )


def _require_cache_binding(
    config: FinalAuthorizationConfig,
    identity: dict[str, object],
    cache: CacheBinding,
) -> None:
    checks = {
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
    mismatch = [key for key, values in checks.items() if values[0] != values[1]]
    if mismatch:
        raise ProtocolError(f"Stage-70 cache/reservation binding drifted: {mismatch}.")


def _is_exact_prepared_root(root: Path) -> bool:
    if not root.exists():
        return False
    if not root.is_dir() or root.is_symlink():
        raise ProtocolError("Stage-70 prepared output root is unsafe.")
    symlinks = [member for member in root.rglob("*") if member.is_symlink()]
    if symlinks:
        raise ProtocolError("Stage-70 prepared output contains symlinks.")
    files = {
        member.relative_to(root).as_posix()
        for member in root.rglob("*")
        if member.is_file()
    }
    prepared = {
        "config.resolved.yaml",
        "provenance/input_artifacts.json",
    }
    if files == prepared:
        return True
    required_sets = {frozenset(RESERVATION_REQUIRED_FILES), frozenset(FINAL_REQUIRED_FILES)}
    if frozenset(files) in required_sets:
        return False
    raise ProtocolError(
        "Stage-70 existing output is neither an exact workspace preparation nor "
        "a complete closed-world artifact."
    )


__all__ = (
    "run_final_prediction_authorization",
    "run_target_evaluation_reservation",
)
