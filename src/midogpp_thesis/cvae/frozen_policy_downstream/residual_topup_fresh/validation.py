"""Independent reconstruction and closed-world validation for fresh Stage-70."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .bundle import (
    CENTER_CONTRAST_COLUMNS,
    CONTENT_INDEX_EXCLUSIONS,
    ENSEMBLE_METRIC_COLUMNS,
    INFERENCE_COLUMNS,
    ORACLE_COLUMNS,
    REQUIRED_FILES,
    SEED_METRIC_COLUMNS,
    write_validation_report,
)
from .config import ResidualTopupFreshConfig, load_residual_topup_fresh_config
from .contracts import EXPECTED_ENSEMBLE_METRIC_COUNT, EXPECTED_PLAN_CELL_COUNT
from .execution import (
    PREDICTION_INDEX_COLUMNS,
    load_frozen_policy_actions,
    load_prediction_cache,
)
from .inference import evaluate_sealed_predictions
from .label_access import open_scoring_labels_after_prediction_seal
from .planning import build_evaluation_plan
from .prediction_seal import seal_predictions, validate_prediction_seal
from .source_cache import load_source_cache, load_validated_generation_lock
from .target_cache import (
    load_fresh_target_surface,
)


def validate_residual_topup_fresh_bundle(
    root: str | Path,
    *,
    config: ResidualTopupFreshConfig,
    allow_pending: bool = False,
) -> dict[str, object]:
    """Rebuild the action menu, prediction seal, metrics, and file inventory."""

    output = Path(root)
    required = set(REQUIRED_FILES)
    if allow_pending:
        required.remove("reports/validation_report.json")
    missing = sorted(member for member in required if not (output / member).is_file())
    if missing:
        raise ProtocolError(f"Fresh Stage-70 bundle is incomplete: {missing}.")

    resolved = load_residual_topup_fresh_config(output / "config.resolved.yaml")
    if resolved.contract_hash != config.contract_hash:
        raise ProtocolError("Fresh Stage-70 resolved config contract drifted.")
    policy = load_frozen_policy_actions(config)
    target = load_fresh_target_surface(config)
    generation_lock = load_validated_generation_lock(config)
    source = load_source_cache(output / "checkpoints/source")
    if (
        source.generation_lock_hash != generation_lock.generation_lock_hash
        or source.bank_lock_hash != generation_lock.bank_lock_hash
    ):
        raise ProtocolError("Fresh Stage-70 source-cache upstream binding drifted.")
    plan = build_evaluation_plan(
        policy.actions_by_target,
        evaluation_row_ids_by_target=target.evaluation_row_ids_by_target,
    )
    prediction = load_prediction_cache(
        output / "checkpoints/predictions",
        plan=plan,
        config=config,
        policy=policy,
        source_cache=source,
        target_surface=target,
        generation_lock_hash=generation_lock.generation_lock_hash,
    )
    prediction_lock = _json(
        output / "checkpoints/predictions/prediction_cache.json"
    )
    if (
        prediction.source_cache_hash != source.cache_hash
        or prediction.generation_lock_hash != generation_lock.generation_lock_hash
        or prediction_lock.get("bank_lock_hash") != generation_lock.bank_lock_hash
        or prediction_lock.get("target_cache_content_hash")
        != target.cache_content_hash
        or prediction_lock.get("target_cache_protocol_hash")
        != target.cache_protocol_hash
        or prediction_lock.get("reservation_hash")
        != target.reservation.reservation_hash
        or prediction_lock.get("target_frame_sha256_by_center")
        != {
            center: target.frames_by_center[center].file_sha256
            for center in target.frames_by_center
        }
        or any(
            row.get("classifier_converged") is not True
            for row in prediction.index_rows
        )
    ):
        raise ProtocolError("Fresh Stage-70 prediction-cache binding/convergence failed.")

    capability = seal_predictions(plan, prediction.predictions)
    seal_summary = validate_prediction_seal(capability, expected_plan=plan)
    labels = open_scoring_labels_after_prediction_seal(target, capability)
    reconstructed = evaluate_sealed_predictions(capability, labels)
    _validate_manifests_and_reports(
        output,
        config=config,
        policy=policy,
        target=target,
        source=source,
        prediction=prediction,
        plan=plan,
        seal_hash=seal_summary.seal_hash,
        report=reconstructed,
    )
    _validate_csv(
        output / "tables/prediction_index.csv",
        prediction.index_rows,
        PREDICTION_INDEX_COLUMNS,
        label="prediction index",
    )
    _validate_dataclass_csv(
        output / "tables/seed_cell_metrics.csv",
        reconstructed.scored.seed_cell_metrics,
        SEED_METRIC_COLUMNS,
        label="seed-cell metrics",
    )
    _validate_dataclass_csv(
        output / "tables/ensemble_metrics.csv",
        reconstructed.scored.ensemble_metrics,
        ENSEMBLE_METRIC_COLUMNS,
        label="ensemble metrics",
    )
    _validate_dataclass_csv(
        output / "tables/center_contrasts.csv",
        reconstructed.center_contrasts,
        CENTER_CONTRAST_COLUMNS,
        label="center contrasts",
    )
    _validate_dataclass_csv(
        output / "tables/contrast_inference.csv",
        reconstructed.contrast_inference,
        INFERENCE_COLUMNS,
        label="contrast inference",
    )
    _validate_dataclass_csv(
        output / "tables/oracle_diagnostics.csv",
        reconstructed.oracle_diagnostics,
        ORACLE_COLUMNS,
        label="aggregate oracle diagnostics",
    )
    _validate_closed_world(
        output,
        source=source,
        prediction=prediction,
        allow_pending=allow_pending,
    )
    checks: dict[str, object] = {
        "status": "PASS",
        "config_contract_hash": config.contract_hash,
        "policy_lock_hash": policy.policy_lock_hash,
        "action_library_hash": policy.action_library_hash,
        "reservation_id": target.reservation.reservation_id,
        "reservation_hash": target.reservation.reservation_hash,
        "target_cache_content_hash": target.cache_content_hash,
        "source_cache_hash": source.cache_hash,
        "prediction_cache_hash": prediction.cache_hash,
        "evaluation_plan_hash": plan.plan_hash,
        "prediction_seal_hash": seal_summary.seal_hash,
        "prediction_task_count": len(prediction.records),
        "prediction_cell_count": len(prediction.predictions),
        "ensemble_metric_count": len(reconstructed.scored.ensemble_metrics),
        "all_classifier_fits_converged": True,
        "all_predictions_sealed_before_labels": True,
        "labels_used_for_scoring_only": True,
        "support_evaluation_cases_globally_disjoint": True,
        "consumed_stage70_used": False,
        "consumed_stage90_used": False,
        "policy_update_emitted": False,
        "oracle_action_exported": False,
    }
    if not allow_pending:
        expected = {
            "schema_version": "midogpp_residual_topup_fresh_validation_v1",
            "status": "PASS",
            "validator": "validate_residual_topup_fresh_bundle",
            "checks": checks,
        }
        if _json(output / "reports/validation_report.json") != expected:
            raise ProtocolError("Fresh Stage-70 validation report drifted.")
    return checks


def validate_and_write_residual_topup_fresh_bundle(
    root: str | Path,
    *,
    config: ResidualTopupFreshConfig,
) -> dict[str, object]:
    checks = validate_residual_topup_fresh_bundle(
        root,
        config=config,
        allow_pending=True,
    )
    write_validation_report(root, checks)
    validate_residual_topup_fresh_bundle(root, config=config)
    return checks


def _validate_manifests_and_reports(
    root: Path,
    *,
    config: ResidualTopupFreshConfig,
    policy: object,
    target: object,
    source: object,
    prediction: object,
    plan: object,
    seal_hash: str,
    report: object,
) -> None:
    policy_manifest = _json(root / "manifests/policy_binding.json")
    plan_manifest = _json(root / "manifests/evaluation_plan.json")
    seal_manifest = _json(root / "manifests/prediction_seal.json")
    protocol = _json(root / "manifests/protocol_manifest.json")
    provenance = _json(root / "provenance/input_artifacts.json")
    leakage = _json(root / "reports/leakage_report.json")
    access = _json(root / "reports/label_access_report.json")
    state = _json(root / "reports/run_state.json")
    publication = _json(root / "reports/publication_decision.json")
    if protocol.get("protocol_hash") != stable_hash(
        {key: value for key, value in protocol.items() if key != "protocol_hash"}
    ):
        raise ProtocolError("Fresh Stage-70 protocol manifest hash drifted.")
    provenance_rows = provenance.get("input_artifacts")
    if (
        protocol.get("input_binding_hash") != stable_hash(provenance)
        or provenance.get("experiment_id") != config.experiment_id
        or provenance.get("selection_used_target_eval_artifacts") is not False
        or not isinstance(provenance_rows, list)
        or len(provenance_rows) != len(config.input_artifact_ids)
        or {
            str(row.get("artifact_id", ""))
            for row in provenance_rows
            if isinstance(row, Mapping)
        }
        != set(config.input_artifact_ids)
    ):
        raise ProtocolError("Fresh Stage-70 provenance binding drifted.")
    if policy_manifest.get("policy_binding_hash") != stable_hash(
        {
            key: value
            for key, value in policy_manifest.items()
            if key != "policy_binding_hash"
        }
    ):
        raise ProtocolError("Fresh Stage-70 policy binding hash drifted.")
    if (
        protocol.get("config_contract_hash") != config.contract_hash
        or protocol.get("policy_lock_hash") != policy.policy_lock_hash
        or protocol.get("action_library_hash") != policy.action_library_hash
        or protocol.get("reservation_id") != target.reservation.reservation_id
        or protocol.get("source_cache_hash") != source.cache_hash
        or protocol.get("prediction_cache_hash") != prediction.cache_hash
        or protocol.get("evaluation_plan_hash") != plan.plan_hash
        or protocol.get("prediction_seal_hash") != seal_hash
        or protocol.get("policy_update_emitted") is not False
        or protocol.get("oracle_action_exported") is not False
        or plan_manifest.get("plan_hash") != plan.plan_hash
        or plan_manifest.get("labels_available_to_planning") is not False
        or seal_manifest.get("prediction_seal_hash") != seal_hash
        or seal_manifest.get("prediction_cell_count") != EXPECTED_PLAN_CELL_COUNT
        or seal_manifest.get("all_predictions_sealed_before_labels") is not True
        or seal_manifest.get("labels_opened_at_seal_time") is not False
        or access.get("prediction_seal_hash") != seal_hash
        or access.get("labels_opened_only_after_global_prediction_seal") is not True
        or access.get("labels_available_to_fit_or_predict") is not False
        or leakage.get("status") != "PASS"
        or leakage.get("consumed_stage70_used") is not False
        or leakage.get("consumed_stage90_used") is not False
        or leakage.get("policy_update_emitted") is not False
        or leakage.get("oracle_action_exported") is not False
        or state.get("status") != "COMPLETE"
        or state.get("policy_update_emitted") is not False
        or publication.get("policy_update_emitted") is not False
        or publication.get("oracle_action_exported") is not False
        or report.policy_update_emitted is not False
    ):
        raise ProtocolError("Fresh Stage-70 manifests/reports drifted.")


def _validate_closed_world(
    root: Path,
    *,
    source: object,
    prediction: object,
    allow_pending: bool,
) -> None:
    expected = set(REQUIRED_FILES)
    if allow_pending:
        expected.remove("reports/validation_report.json")
    for record in source.records:
        expected.add(f"checkpoints/source/{record.relative_path}")
        expected.add(f"checkpoints/source/metadata/{record.stream_id}.json")
    for record in prediction.records:
        expected.add(f"checkpoints/predictions/{record.metadata_member}")
        expected.add(f"checkpoints/predictions/{record.probability_member}")
        expected.add(f"checkpoints/predictions/{record.prediction_member}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual != expected or any(path.is_symlink() for path in root.rglob("*")):
        raise ProtocolError("Fresh Stage-70 closed-world file inventory drifted.")
    content = _json(root / "manifests/content_index.json")
    observed_hash = content.get("content_hash")
    unhashed = {key: value for key, value in content.items() if key != "content_hash"}
    records = content.get("files")
    indexed_members = expected.difference(CONTENT_INDEX_EXCLUSIONS)
    if (
        observed_hash != stable_hash(unhashed)
        or content.get("status") != "COMPLETE"
        or content.get("file_count") != len(indexed_members)
        or content.get("scratch_authoritative") is not False
        or not isinstance(records, list)
    ):
        raise ProtocolError("Fresh Stage-70 content index drifted.")
    seen: set[str] = set()
    for raw in records:
        if not isinstance(raw, Mapping):
            raise ProtocolError("Fresh Stage-70 content row is malformed.")
        member = str(raw.get("path", ""))
        path = _safe_member(root, member)
        if (
            member in seen
            or member not in indexed_members
            or raw.get("sha256") != _sha256_file(path)
            or raw.get("size_bytes") != path.stat().st_size
        ):
            raise ProtocolError("Fresh Stage-70 content member drifted.")
        seen.add(member)
    if seen != indexed_members:
        raise ProtocolError("Fresh Stage-70 content coverage drifted.")


def _validate_csv(
    path: Path,
    expected_rows: Sequence[Mapping[str, object]],
    columns: Sequence[str],
    *,
    label: str,
) -> None:
    observed = _read_csv(path, columns)
    expected = tuple(
        {column: _render(row[column]) for column in columns}
        for row in expected_rows
    )
    if observed != expected:
        raise ProtocolError(f"Fresh Stage-70 {label} table drifted.")


def _validate_dataclass_csv(
    path: Path,
    expected_rows: Sequence[object],
    columns: Sequence[str],
    *,
    label: str,
) -> None:
    observed = _read_csv(path, columns)
    expected = tuple(
        {column: _render(getattr(row, column)) for column in columns}
        for row in expected_rows
    )
    if observed != expected:
        raise ProtocolError(f"Fresh Stage-70 {label} table drifted.")


def _read_csv(path: Path, columns: Sequence[str]) -> tuple[dict[str, str], ...]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != tuple(columns):
                raise ProtocolError("Fresh Stage-70 CSV columns drifted.")
            return tuple(dict(row) for row in reader)
    except OSError as exc:
        raise ProtocolError("Cannot read fresh Stage-70 CSV.") from exc


def _render(value: object) -> str:
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return str(value)


def _json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read fresh Stage-70 JSON: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("Fresh Stage-70 JSON must be a mapping.")
    return payload


def _safe_member(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ProtocolError("Fresh Stage-70 content member is unsafe or absent.")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = (
    "validate_and_write_residual_topup_fresh_bundle",
    "validate_residual_topup_fresh_bundle",
)
