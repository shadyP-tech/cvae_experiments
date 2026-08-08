"""Concrete workstation adapter for the exact-tail Stage-60 producer."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ..utility_aligned import CandidateFeatureRow
from .bundle import (
    CONTENT_INDEX_MEMBERS,
    build_surface_lock,
    leakage_report_payload,
    sha256_file,
    validate_surface_bundle,
)
from .config import ExactTailUtilitySurfaceConfig, FreshInputAttestation
from .contracts import CENTERS, DevelopmentPartition
from .features import materialize_candidate_feature_rows
from .prediction_execution import (
    CoarsePredictionRecord,
    materialize_exact_tail_predictions,
)
from .production_inputs import (
    DevelopmentReservation,
    load_development_reservation,
    prepare_development_cache_arrays,
)
from .runner import PreparedPredictionCapability
from .runtime import WorkstationSnapshot
from .scoring import ScoredExactTailUtilityRow
from .source_generation import (
    GeneratedDevelopmentCache,
    materialize_generated_development_cache,
)


class ProductionExactTailAdapter:
    """Default execution adapter used by the workspace CLI on the workstation."""

    def __init__(self) -> None:
        self._reservation: DevelopmentReservation | None = None
        self._generated: GeneratedDevelopmentCache | None = None
        self._task_records: tuple[CoarsePredictionRecord, ...] = ()

    def collect_workstation_snapshot(
        self, config: ExactTailUtilitySurfaceConfig
    ) -> WorkstationSnapshot:
        total_memory = _ram_gib()
        gpu_names, gpu_total, gpu_free = _nvidia_snapshot()
        torch_module = sys.modules.get("torch")
        parent_cuda_initialized = bool(
            torch_module is not None
            and getattr(torch_module, "cuda", None) is not None
            and torch_module.cuda.is_initialized()
        )
        disk_probe = _nearest_existing_parent(config.artifact_root)
        return WorkstationSnapshot(
            logical_cpu_count=int(os.cpu_count() or 0),
            ram_gib=total_memory,
            gpu_names=gpu_names,
            gpu_total_memory_mib=gpu_total,
            gpu_free_memory_mib=gpu_free,
            artifact_disk_free_gib=float(shutil.disk_usage(disk_probe).free)
            / (1024.0**3),
            parent_cuda_initialized=parent_cuda_initialized,
        )

    def materialize_label_free_predictions(
        self,
        config: ExactTailUtilitySurfaceConfig,
        attestation: FreshInputAttestation,
    ) -> PreparedPredictionCapability:
        config.artifact_root.mkdir(parents=True, exist_ok=True)
        reservation = load_development_reservation(config, attestation)
        execution_root = _select_execution_root(config)
        prepared = prepare_development_cache_arrays(
            config, reservation, output_root=execution_root
        )
        generated_root = execution_root / "generated_source_cache"
        generated = materialize_generated_development_cache(
            config,
            prepared,
            root=generated_root,
            scratch_root=execution_root / "generation_scratch",
        )
        features = materialize_candidate_feature_rows(prepared, generated)
        execution = materialize_exact_tail_predictions(
            config,
            attestation,
            prepared,
            generated,
            root=config.artifact_root,
            checkpoint_root=execution_root / "prediction_checkpoints",
        )
        self._reservation = reservation
        self._generated = generated
        self._task_records = execution.task_records
        return PreparedPredictionCapability(
            partitions=reservation.partitions,
            predictions=execution.predictions,
            seal=execution.seal,
            seal_path=execution.seal_path,
            prediction_index_path=execution.prediction_index_path,
            prediction_arrays_path=execution.prediction_arrays_path,
            feature_rows=features,
        )

    def persist_scored_bundle(
        self,
        config: ExactTailUtilitySurfaceConfig,
        capability: PreparedPredictionCapability,
        rows: Sequence[ScoredExactTailUtilityRow],
    ) -> Path:
        if self._reservation is None or self._generated is None or not self._task_records:
            raise ProtocolError("Exact-tail production adapter lacks its sealed execution state.")
        root = config.artifact_root
        _require_workspace_prelude(root)
        _atomic_json(
            root / "manifests/development_reservation.json",
            dict(self._reservation.raw_payload),
        )
        _atomic_json(
            root / "manifests/protocol_manifest.json",
            _protocol_manifest(
                config,
                capability,
                self._reservation,
                self._generated,
            ),
        )
        _write_csv(
            root / "tables/source_streams.csv",
            [record.to_payload() for record in self._generated.source_records],
        )
        _write_csv(
            root / "tables/coarse_prediction_tasks.csv",
            [_task_row(record) for record in self._task_records],
        )
        _write_csv(
            root / "tables/evaluation_rows.csv",
            [
                {
                    "schema_version": "midogpp_exact_tail_evaluation_row_v1",
                    **row.identity_payload(),
                    "label_present": False,
                }
                for center in CENTERS
                for row in capability.partitions[center].evaluation_rows
            ],
        )
        _write_csv(
            root / "tables/candidate_features.csv",
            [
                {
                    **feature.to_payload(),
                    "row_hash": feature.row_hash,
                    "distribution_mmd_semantics": "linear_kernel_mmd_squared",
                }
                for feature in capability.feature_rows
            ],
        )
        utility_rows = tuple(rows)
        _write_csv(
            root / "tables/exact_tail_utility.csv",
            [row.to_payload() for row in utility_rows],
        )

        utility_sha = sha256_file(root / "tables/exact_tail_utility.csv")
        feature_sha = sha256_file(root / "tables/candidate_features.csv")
        member_sha = {
            member: sha256_file(root / member) for member in CONTENT_INDEX_MEMBERS
        }
        lock = build_surface_lock(
            seal=capability.seal,
            rows=utility_rows,
            feature_rows=capability.feature_rows,
            utility_table_sha256=utility_sha,
            feature_table_sha256=feature_sha,
            member_sha256=member_sha,
        )
        _atomic_json(
            root / "manifests/exact_tail_utility_surface_lock.json",
            lock.to_payload(),
        )
        _atomic_json(
            root / "manifests/content_index.json",
            {
                "schema_version": "midogpp_exact_tail_content_index_v1",
                "member_sha256": member_sha,
                "surface_lock_hash": lock.surface_lock_hash,
            },
        )
        _atomic_json(root / "reports/leakage_report.json", leakage_report_payload(lock))
        _atomic_json(
            root / "reports/run_state.json",
            {
                "schema_version": "midogpp_exact_tail_run_state_v1",
                "status": "COMPLETE",
                "surface_lock_hash": lock.surface_lock_hash,
                "source_stream_count": len(self._generated.source_records),
                "coarse_task_count": len(self._task_records),
                "prediction_cell_count": len(capability.seal.cells),
                "utility_row_count": len(utility_rows),
                "feature_row_count": len(capability.feature_rows),
                "resumed_checkpoints_hash_validated": True,
            },
        )
        # A PASS authorization exists only after independent array/index/seal
        # reconstruction and fresh-manifest rescoring have succeeded.
        validate_surface_bundle(
            root,
            config=config,
            _allow_pending_validation_report=True,
        )
        from .surface_validation import validation_report_payload

        _atomic_json(
            root / "reports/validation_report.json",
            validation_report_payload(lock.surface_lock_hash),
        )
        validate_surface_bundle(root, config=config)
        return root


def _protocol_manifest(
    config: ExactTailUtilitySurfaceConfig,
    capability: PreparedPredictionCapability,
    reservation: DevelopmentReservation,
    generated: GeneratedDevelopmentCache,
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_exact_tail_protocol_manifest_v1",
        "experiment_id": config.experiment_id,
        "output_artifact_id": config.output_artifact_id,
        "config_contract_hash": config.contract_hash,
        "reservation_hash": reservation.reservation_hash,
        "generated_source_cache_hash": generated.cache_hash,
        "generation_lock_hash": generated.generation_lock_hash,
        "bank_lock_hash": generated.bank_lock_hash,
        "metadata_profile_sha256": reservation.raw_payload[
            "metadata_profile_sha256"
        ],
        "prediction_seal_hash": capability.seal.seal_hash,
        "inner_geometry": "seven_by_144_base_plus_126_single_source_tail",
        "distribution_mmd_semantics": "linear_kernel_mmd_squared",
        "minimum_independent_support_cases_per_query": 8,
        "uncertainty_units": ["query_cluster", "case_cluster"],
        "seed_cells_are_uncertainty_units": False,
        "all_predictions_sealed_before_development_labels": True,
        "dedicated_scoring_manifest_contains_exactly_sealed_rows": True,
        "target_support_labels_used": False,
        "target_evaluation_labels_used": False,
        "source_experts_updated": False,
        "seed_selection_performed": False,
    }


def _task_row(record: CoarsePredictionRecord) -> dict[str, object]:
    task = record.task
    return {
        "schema_version": "midogpp_exact_tail_coarse_prediction_task_v1",
        "outer_target": task.outer_target,
        "pseudo_query": task.pseudo_query,
        "training_seed": task.training_seed,
        "generation_seed": task.generation_seed,
        "candidate_sources_json": _compact_json(list(task.candidate_sources)),
        "action_ids_json": _compact_json(list(task.action_ids)),
        "task_hash": task.task_hash,
        "checkpoint_member": record.checkpoint_relative_path,
        "checkpoint_file_sha256": record.checkpoint_file_sha256,
        "checkpoint_hash": record.checkpoint_hash,
        "evaluation_row_count": record.evaluation_row_count,
        "action_composition_sha256_json": _compact_json(
            dict(record.action_composition_sha256)
        ),
        "action_scaler_state_hash_json": _compact_json(
            dict(record.action_scaler_state_hash)
        ),
        "all_eight_actions_materialized": True,
    }


def _select_execution_root(config: ExactTailUtilitySurfaceConfig) -> Path:
    local = Path("/data/local")
    if local.is_dir() and os.access(local, os.W_OK):
        selected = local / "midogpp_exact_tail" / config.contract_hash
        selected.mkdir(parents=True, exist_ok=True)
        return selected
    # Preserve hash-valid resume on the repository/NFS filesystem without
    # placing transient scientific bytes inside the closed-world artifact.
    selected = (
        config.artifact_root.parent
        / ".midogpp_checkpoints"
        / "exact_tail_utility_surface"
        / config.contract_hash
    )
    selected.mkdir(parents=True, exist_ok=True)
    try:
        selected.resolve().relative_to(config.artifact_root.resolve())
    except ValueError:
        return selected
    raise ProtocolError("Exact-tail execution root must remain outside the artifact bundle.")


def _require_workspace_prelude(root: Path) -> None:
    for member in ("config.resolved.yaml", "provenance/input_artifacts.json"):
        if not (root / member).is_file():
            raise ProtocolError(
                f"Exact-tail workspace did not prepare required provenance member: {member}."
            )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    materialized = tuple(dict(row) for row in rows)
    if not materialized:
        raise ProtocolError(f"Exact-tail refuses to persist an empty table: {path.name}.")
    columns = tuple(materialized[0])
    if any(tuple(row) != columns for row in materialized):
        raise ProtocolError(f"Exact-tail table rows have inconsistent schema: {path.name}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)
        handle.flush()
    temporary.replace(path)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _compact_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _ram_gib() -> float:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        page_count = int(os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError) as exc:
        raise ProtocolError("Cannot determine workstation RAM capacity.") from exc
    return float(page_size * page_count) / (1024.0**3)


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path.resolve(strict=False)
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise ProtocolError("Cannot locate the exact-tail artifact filesystem.")
        candidate = parent
    return candidate


def _nvidia_snapshot() -> tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProtocolError("Cannot query workstation GPUs with nvidia-smi.") from exc
    names: list[str] = []
    total: list[int] = []
    free: list[int] = []
    for line in completed.stdout.splitlines():
        parts = tuple(value.strip() for value in line.split(","))
        if len(parts) != 3:
            raise ProtocolError("nvidia-smi output schema drifted.")
        names.append(parts[0])
        total.append(int(parts[1]))
        free.append(int(parts[2]))
    return tuple(names), tuple(total), tuple(free)


__all__ = ("ProductionExactTailAdapter",)
