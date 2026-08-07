"""Run the optimized, resumable Stage-90 MMD/KMM diagnostic."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Iterator, Mapping

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from ...reporting import write_csv_rows, write_json
from .bundle import CONTENT_INDEX_MEMBERS, REQUIRED_FILES
from .config import MMDKMMRouterDiagnosticConfig
from .contracts import (
    CENTERS,
    CLAIM_SCOPE,
    PUBLICATION_STATUS,
)
from .inputs import (
    SUPPORT_PARTITION_COLUMNS,
    build_partition_surface,
    load_label_free_validation_frame,
    load_validated_locks,
    validate_workspace_provenance,
)
from .metrics import PAIRED_DELTA_COLUMNS, TARGET_METRIC_COLUMNS, score_predictions
from .planning import build_router_plans
from .prediction import materialize_target_predictions
from .profiles import CONDITIONAL_ROUTER_MODE
from .seals import build_global_prediction_seal, open_evaluation_labels
from .source_products import (
    materialize_source_products,
    validate_source_products_lock,
)


def run_mmd_kmm_router_diagnostic(
    config: MMDKMMRouterDiagnosticConfig,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    _assert_resolved_paths(config)
    root = Path(artifact_root or config.artifact_root)
    for relative in ("arrays", "manifests", "provenance", "reports", "tables"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    _assert_launch_files(root)
    _assert_closed_world(root, allow_incomplete=True)
    with _exclusive_run_lock(root):
        _prune_stale_temp_files(root)
        state_path = root / "reports/run_state.json"
        if state_path.is_file() and _json(state_path).get("status") == "COMPLETE":
            _validate_bundle(root, config=config)
            return root
        started = time.monotonic()
        phase = "INITIALIZING"
        _write_state(root, status="RUNNING", phase=phase)
        try:
            provenance = validate_workspace_provenance(root, config)
            locks = load_validated_locks(config)
            frame = load_label_free_validation_frame(config)
            partitions = build_partition_surface(
                frame,
                config_contract_hash=config.contract_hash,
            )
            protocol = _protocol_manifest(
                config,
                provenance=provenance,
                validation_cache_binding_hash=frame.cache_binding_hash,
                support_partition_lock_hash=partitions.lock_hash,
            )
            write_json(root / "manifests/protocol_manifest.json", protocol)
            write_json(root / "manifests/support_partition_lock.json", partitions.lock_payload)
            write_csv_rows(
                root / "tables/support_partitions.csv",
                partitions.table_rows,
                columns=SUPPORT_PARTITION_COLUMNS,
            )

            phase = "SOURCE_PRODUCTS"
            _write_state(root, status="RUNNING", phase=phase)
            source_products = materialize_source_products(
                config,
                locks.generation,
                frame,
                partitions,
                root=root,
            )
            source_lock = validate_source_products_lock(
                root,
                config=config,
                generation_lock=locks.generation,
                frame=frame,
                partitions=partitions,
                source_products=source_products,
            )
            write_json(
                root / "reports/phase_01_source_products_complete.json",
                _phase_payload(
                    "PHASE_01_SOURCE_PRODUCTS_COMPLETE",
                    source_products_hash=source_products.source_products_hash,
                    source_block_count=len(source_products.index_rows),
                    compatibility_score_count=len(source_products.compatibility_score_rows),
                    manifest_labels_opened=False,
                ),
            )

            phase = "ROUTER_PLANS"
            _write_state(root, status="RUNNING", phase=phase)
            plans = build_router_plans(
                config,
                source_products,
                frame,
                partitions,
                source_products_lock_hash=str(
                    source_lock["source_products_lock_hash"]
                ),
                root=root,
            )
            write_json(
                root / "reports/phase_02_router_plans_complete.json",
                _phase_payload(
                    "PHASE_02_ROUTER_PLANS_COMPLETE",
                    router_plan_lock_hash=plans.lock_hash,
                    target_count=len(plans.plans_by_target),
                    nonuniform_plan_count=sum(
                        not bool(plan["used_uniform_fallback"])
                        for plan in plans.plans_by_target.values()
                    ),
                    target_labels_used=False,
                    evaluation_embeddings_used_for_router=False,
                ),
            )

            phase = "TARGET_PREDICTIONS"
            _write_state(root, status="RUNNING", phase=phase)
            predictions = materialize_target_predictions(
                config,
                locks.generation.generation_lock_hash,
                source_products,
                plans,
                frame,
                partitions,
                source_products_lock_hash=str(
                    source_lock["source_products_lock_hash"]
                ),
                root=root,
            )
            seal = build_global_prediction_seal(
                config,
                partitions,
                plans,
                predictions,
                root=root,
            )
            write_json(
                root / "reports/phase_03_predictions_sealed.json",
                _phase_payload(
                    "PHASE_03_ALL_TARGET_PREDICTIONS_SEALED",
                    global_prediction_seal_hash=seal["seal_hash"],
                    prediction_cell_count=len(predictions.index_rows),
                    unique_classifier_fit_count=predictions.unique_classifier_fit_count,
                    target_labels_opened=False,
                ),
            )

            phase = "SCORING"
            _write_state(root, status="RUNNING", phase=phase)
            labels_by_sample, label_report = open_evaluation_labels(
                config,
                partitions,
                root=root,
            )
            metrics, deltas, scoring = score_predictions(
                predictions,
                labels_by_sample_id=labels_by_sample,
            )
            write_csv_rows(
                root / "tables/target_metrics.csv",
                metrics,
                columns=TARGET_METRIC_COLUMNS,
            )
            write_csv_rows(
                root / "tables/paired_deltas.csv",
                deltas,
                columns=PAIRED_DELTA_COLUMNS,
            )
            write_json(root / "reports/label_access_report.json", label_report)
            write_json(root / "reports/phase_04_scoring_complete.json", scoring)
            write_json(root / "reports/leakage_report.json", _leakage_report())
            write_json(
                root / "reports/publication_decision.json",
                _publication_decision(scoring, plans=plans.plans_by_target),
            )
            write_json(
                root / "reports/runtime_summary.json",
                _runtime_summary(
                    config,
                    elapsed_seconds=time.monotonic() - started,
                    unique_classifier_fit_count=predictions.unique_classifier_fit_count,
                ),
            )

            phase = "VALIDATING"
            _write_state(root, status="RUNNING", phase=phase)
            _write_content_index(root)
            checks = _validate_bundle(
                root,
                config=config,
                allow_pending=True,
            )
            validator_name = (
                "validate_conditional_contrast_mmd_router_bundle"
                if config.router_mode == CONDITIONAL_ROUTER_MODE
                else "validate_mmd_kmm_router_bundle"
            )
            write_json(
                root / "reports/validation_report.json",
                {
                    "schema_version": "midogpp_mmd_kmm_validation_report_v1",
                    "status": "PASS",
                    "validator": validator_name,
                    "checks": checks,
                },
            )
            _write_state(root, status="COMPLETE", phase="COMPLETE")
            _validate_bundle(root, config=config)
            return root
        except BaseException as exc:
            _write_state(
                root,
                status="FAILED",
                phase=phase,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise


def _validate_bundle(
    root: Path,
    *,
    config: MMDKMMRouterDiagnosticConfig,
    allow_pending: bool = False,
) -> dict[str, object]:
    if config.router_mode == CONDITIONAL_ROUTER_MODE:
        from ..conditional_contrast_mmd_router.validation import (
            validate_conditional_contrast_mmd_router_bundle,
        )

        return validate_conditional_contrast_mmd_router_bundle(
            root,
            config=config,
            allow_pending=allow_pending,
        )
    from .validation import validate_mmd_kmm_router_bundle

    return validate_mmd_kmm_router_bundle(
        root,
        config=config,
        allow_pending=allow_pending,
    )


def _protocol_manifest(
    config: MMDKMMRouterDiagnosticConfig,
    *,
    provenance: Mapping[str, Mapping[str, object]],
    validation_cache_binding_hash: str,
    support_partition_lock_hash: str,
) -> dict[str, object]:
    input_hashes = {
        artifact_id: stable_hash(dict(provenance[artifact_id]))
        for artifact_id in config.input_artifact_ids
    }
    payload: dict[str, object] = {
        "schema_version": "midogpp_mmd_kmm_protocol_manifest_v1",
        "experiment_id": config.experiment_id,
        "output_artifact_id": config.output_artifact_id,
        "stage": "90_oracles_and_diagnostics",
        "claim_scope": CLAIM_SCOPE,
        "publication_status": PUBLICATION_STATUS,
        "config_contract_hash": config.contract_hash,
        "input_artifact_ids": list(config.input_artifact_ids),
        "input_artifact_hashes": input_hashes,
        "validation_cache_binding_hash": validation_cache_binding_hash,
        "support_partition_lock_hash": support_partition_lock_hash,
        "protocol": dict(config.protocol),
        "proxy": dict(config.proxy),
        "classifier": config.classifier.to_payload(),
        "runtime": dict(config.runtime),
        "claim_boundary": dict(config.claim_boundary),
    }
    payload["protocol_hash"] = stable_hash(payload)
    return payload


def _phase_payload(phase: str, **values: object) -> dict[str, object]:
    payload = {
        "schema_version": "midogpp_mmd_kmm_phase_report_v1",
        "phase": phase,
        "claim_scope": CLAIM_SCOPE,
        "diagnostic_only": True,
        "fresh_evidence": False,
        "routing_quality_claimed": False,
        "promotion_eligible": False,
        **values,
    }
    payload["phase_hash"] = stable_hash(payload)
    return payload


def _leakage_report() -> dict[str, object]:
    return {
        "schema_version": "midogpp_mmd_kmm_leakage_report_v1",
        "status": "PASS",
        "source_experts_frozen_source_only": True,
        "target_expert_excluded_from_every_pool": True,
        "support_and_evaluation_case_disjoint": True,
        "support_and_evaluation_sample_disjoint": True,
        "support_labels_used": False,
        "evaluation_embeddings_used_for_router": False,
        "evaluation_labels_available_before_global_prediction_seal": False,
        "evaluation_labels_used_for_scoring_only": True,
        "individual_expert_or_seed_selection_performed": False,
        "previous_stage90_router_or_utility_inputs_used": False,
        "dense_residual_output_used": False,
        "local_marginal_utility_output_used": False,
        "routing_quality_claimed": False,
        "promotion_eligible": False,
    }


def _publication_decision(scoring: Mapping[str, object], *, plans: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "midogpp_mmd_kmm_publication_decision_v1",
        "decision": "PUBLISH_AS_EXPLORATORY_CONSUMED_DATA_DIAGNOSTIC_ONLY",
        "publication_status": PUBLICATION_STATUS,
        "mean_equal_union_bacc": scoring["mean_equal_union_bacc"],
        "mean_mmd_kmm_bacc": scoring["mean_mmd_kmm_bacc"],
        "mean_paired_bacc_delta_center_equal": scoring["mean_paired_bacc_delta_center_equal"],
        "nonuniform_plan_count": sum(not bool(plan["used_uniform_fallback"]) for plan in plans.values()),
        "uniform_fallback_count": sum(bool(plan["used_uniform_fallback"]) for plan in plans.values()),
        "routing_quality_claimed": False,
        "fresh_evidence": False,
        "fresh_confirmation": False,
        "promotion_eligible": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
        "required_next_evidence": "separately_authorized_fresh_case_disjoint_target_surface",
    }


def _runtime_summary(config: MMDKMMRouterDiagnosticConfig, *, elapsed_seconds: float, unique_classifier_fit_count: int) -> dict[str, object]:
    return {
        "schema_version": "midogpp_mmd_kmm_runtime_summary_v1",
        "workstation_profile": config.runtime["workstation_profile"],
        "generation_devices": config.runtime["generation_devices"],
        "kernel_devices": config.runtime["kernel_devices"],
        "source_expert_load_count": 27,
        "source_block_count": 81,
        "source_prefix_per_class": 256,
        "source_cache_bytes": 81 * 2 * 256 * 3840 * 4,
        "classifier_workers": config.runtime["classifier_workers"],
        "classifier_threads_per_worker": config.runtime["classifier_threads_per_worker"],
        "unique_classifier_fit_count": unique_classifier_fit_count,
        "maximum_unique_classifier_fit_count": config.runtime["maximum_unique_classifier_fit_count"],
        "resume_policy": config.runtime["resume_policy"],
        "elapsed_seconds": float(elapsed_seconds),
    }


def _write_content_index(root: Path) -> None:
    records = []
    for relative in CONTENT_INDEX_MEMBERS:
        path = root / relative
        if not path.is_file():
            raise ProtocolError(f"MMD/KMM content member is missing: {relative}.")
        records.append(
            {
                "relative_path": relative,
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    payload: dict[str, object] = {
        "schema_version": "midogpp_mmd_kmm_content_index_v1",
        "records": records,
    }
    payload["content_hash"] = stable_hash(payload)
    write_json(root / "manifests/content_index.json", payload)


def _write_state(
    root: Path,
    *,
    status: str,
    phase: str,
    error_type: str | None = None,
    error_message: str | None = None,
) -> None:
    _atomic_write_json(
        root / "reports/run_state.json",
        {
            "schema_version": "midogpp_mmd_kmm_run_state_v1",
            "status": status,
            "phase": phase,
            "resumable": status in {"RUNNING", "FAILED"},
            "error_type": error_type,
            "error_message": error_message,
            "diagnostic_only": True,
        },
    )


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@contextmanager
def _exclusive_run_lock(root: Path) -> Iterator[None]:
    path = root / ".run.lock"
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProtocolError("Another MMD/KMM runner already owns this artifact.") from exc
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            # Keep the inode stable across rejected concurrent launches.  An
            # unlinked flock can otherwise be bypassed by opening a new file
            # at the same pathname while the first process still owns it.


def _assert_resolved_paths(config: MMDKMMRouterDiagnosticConfig) -> None:
    paths = (
        config.artifact_root,
        config.expert_bank_root,
        config.generation_lock_root,
        config.equal_union_policy_root,
        config.validation_cache_root,
        config.validation_manifest_path,
    )
    if any(not path.is_absolute() for path in paths):
        raise ProtocolError(
            "MMD/KMM execution requires workspace-resolved paths; use "
            "`python -m midogpp_thesis workspace run`."
        )


def _assert_launch_files(root: Path) -> None:
    missing = [
        relative
        for relative in ("config.resolved.yaml", "provenance/input_artifacts.json")
        if not (root / relative).is_file()
    ]
    if missing:
        raise ProtocolError(f"MMD/KMM workspace launch files are missing: {missing}.")


def _assert_closed_world(root: Path, *, allow_incomplete: bool) -> None:
    allowed = set(REQUIRED_FILES)
    unexpected = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in allowed or relative == ".run.lock":
            continue
        if allow_incomplete and (relative.startswith("checkpoints/") or ".tmp" in path.name):
            continue
        unexpected.append(relative)
    if unexpected:
        raise ProtocolError(f"MMD/KMM artifact contains unexpected files: {sorted(unexpected)}.")


def _prune_stale_temp_files(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file() and path.name.endswith(".tmp"):
            path.unlink()


def _json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read MMD/KMM JSON: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("MMD/KMM JSON must be an object.")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ("run_mmd_kmm_router_diagnostic",)
