"""Adapters into neutral GPU/CPU workstation runtimes."""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import tempfile
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.artifact_io import atomic_json, read_json
from ...runtime.fixed_bank_a1_action_predictions import (
    GlobalPredictionSeal,
    load_global_prediction_seal,
    materialize_fixed_bank_a1_action_predictions,
)
from ...runtime.frozen_source_streams import (
    FrozenSourceStreamCache,
    load_frozen_source_streams,
    materialize_frozen_source_streams,
    stage_frozen_source_streams,
)
from ...runtime.preflight import run_label_free_workstation_preflight as _preflight
from ...runtime.preflight import REQUIRED_DISTRIBUTIONS, REQUIRED_THREAD_ENVIRONMENT
from .actions import action_library_by_target
from .constants import CENTERS, OOF_FOLD_SEED, SCRATCH_ROOT
from .partitions import CaseIdentityRow, ThreeRolePartition, build_three_role_partition


LOCAL_SOURCE_DIRECTORY = "source_cache"
LOCAL_GENERATION_DIRECTORY = "source_generation"
LOCAL_PREDICTION_DIRECTORY = "prediction_cache"


def build_case_partition(frame: object, *, config: object) -> ThreeRolePartition:
    seed = int(getattr(config, "protocol")["partition_seed"])
    if seed != OOF_FOLD_SEED:
        raise ProtocolError("Flip-router partition seed drifted.")
    identities = tuple(
        CaseIdentityRow(str(row.center), str(row.case_id), str(row.evaluation_row_id))
        for row in getattr(frame, "rows")
    )
    return build_three_role_partition(identities, partition_seed=seed)


def run_workstation_preflight(
    root: Path, *, runtime: Mapping[str, object]
) -> Mapping[str, object]:
    """Freshly probe resources for every compute-path launch or replay."""

    _assert_preflight_runtime(runtime)
    report_path = root / "reports/workstation_preflight.json"
    if report_path.is_symlink() or (report_path.exists() and not report_path.is_file()):
        raise ProtocolError("Flip-router workstation preflight path is unsafe.")
    fixed = _preflight_fixed_payload()
    shared = dict(runtime)
    shared.update({
        "cuda_visible_devices": "0,1", "launch_blas_threads": 1,
        "minimum_logical_cpu_count": 12,
        "minimum_physical_ram_bytes": 107_374_182_400,
        "minimum_artifact_disk_free_bytes": 12_884_901_888,
        "minimum_gpu_free_mib_per_device": 18_000,
        "source_job_count": 27, "source_stream_count": 81,
        "target_action_identity_count": 90,
        "target_probability_cell_count": 810,
        "target_unique_classifier_fit_count": 810,
        "maximum_total_classifier_fit_count": 810,
        "resume_policy": "hash_validated_atomic_phase_and_task_checkpoints",
    })
    with tempfile.TemporaryDirectory(prefix=".flip-router-preflight-", dir=root.parent) as probe:
        payload = dict(
            _preflight(
                Path(probe),
                runtime=shared,
                expected_scratch_root=SCRATCH_ROOT,
                expected_target_action_identity_count=90,
                expected_target_probability_cell_count=810,
                expected_unique_classifier_fit_count=810,
            )
        )
    payload["disk_probe_path"] = str(root.resolve())
    payload.update(fixed)
    _validate_preflight_payload(payload, root=root, fixed=fixed)
    atomic_json(report_path, payload)
    return payload


def load_validated_workstation_preflight(
    root: Path, *, runtime: Mapping[str, object]
) -> Mapping[str, object]:
    """Validate the launch probe without probing hardware during validation."""

    _assert_preflight_runtime(runtime)
    report_path = root / "reports/workstation_preflight.json"
    if report_path.is_symlink() or not report_path.is_file():
        raise ProtocolError("Flip-router persisted workstation preflight is unsafe.")
    payload = read_json(report_path)
    _validate_preflight_payload(
        payload,
        root=root,
        fixed=_preflight_fixed_payload(),
    )
    return payload


def _assert_preflight_runtime(runtime: Mapping[str, object]) -> None:
    if (
        int(runtime.get("physical_actions_per_target_task", -1)) != 10
        or int(runtime.get("target_probability_cell_count", -1)) != 810
        or int(runtime.get("classifier_workers", -1)) != 4
        or int(runtime.get("classifier_threads_per_worker", -1)) != 3
        or runtime.get("multiprocessing_start_method") != "spawn"
        or tuple(runtime.get("scratch_preference", ())) != (SCRATCH_ROOT, "artifact_parent")
    ):
        raise ProtocolError("Flip-router workstation topology drifted.")


def _preflight_fixed_payload() -> Mapping[str, object]:
    return {
        "source_generation_devices": ["cuda:0", "cuda:1"],
        "physical_actions_per_target_task": 10,
        "target_action_identity_count": 90,
        "target_probability_cell_count": 810,
        "target_unique_classifier_fit_count": 810,
        "scratch_root": SCRATCH_ROOT,
        "persistent_a5000_worker_count": 2,
        "cpu_classifier_worker_count": 4,
        "blas_threads_per_classifier_worker": 3,
    }


def _validate_preflight_payload(
    payload: Mapping[str, object], *, root: Path, fixed: Mapping[str, object]
) -> None:
    exact = {
        "schema_version": "midogpp_label_free_workstation_preflight_v1",
        "status": "PASS",
        "generation_devices": ["cuda:0", "cuda:1"],
        "persistent_gpu_workers": 2,
        "classifier_workers": 4,
        "blas_threads_per_classifier_worker": 3,
        "target_action_identity_count": 90,
        "target_probability_cell_count": 810,
        "target_unique_classifier_fit_count": 810,
        "maximum_total_classifier_fit_count": 810,
        "gpu_then_cpu_phase_order": True,
        "phase_disjoint_gpu_and_cpu_pools": True,
        "parent_cuda_initialized": False,
        "tf32_enabled": False,
        "amp_enabled": False,
        "scratch_preference": [SCRATCH_ROOT, "artifact_parent"],
        "thread_environment": dict(REQUIRED_THREAD_ENVIRONMENT),
        "cuda_visible_devices": "0,1",
        **fixed,
    }
    if any(payload.get(key) != value for key, value in exact.items()):
        raise ProtocolError("Persisted flip-router preflight topology drifted.")
    versions = payload.get("package_versions")
    gpus = payload.get("gpus")
    if (
        payload.get("disk_probe_path") != str(root.resolve())
        or int(payload.get("available_cpu_affinity_count", -1)) < 12
        or int(payload.get("physical_ram_bytes", -1)) < 107_374_182_400
        or int(payload.get("disk_free_bytes_at_launch", -1)) < 12_884_901_888
        or not isinstance(versions, Mapping)
        or set(versions) != set(REQUIRED_DISTRIBUTIONS)
        or any(not isinstance(value, str) or not value for value in versions.values())
        or not isinstance(gpus, list)
        or len(gpus) != 2
    ):
        raise ProtocolError("Persisted flip-router preflight observations drifted.")
    for expected_index, row in enumerate(gpus):
        if (
            not isinstance(row, Mapping)
            or row.get("index") != expected_index
            or "RTX A5000" not in str(row.get("name"))
            or int(row.get("memory_total_mib", -1)) < 23_000
            or int(row.get("memory_free_mib", -1)) < 18_000
        ):
            raise ProtocolError("Persisted flip-router GPU preflight drifted.")


def materialize_sources(config: object, generation_lock: object, *, root: Path) -> FrozenSourceStreamCache:
    base = Path(SCRATCH_ROOT)
    if base.is_symlink():
        raise ProtocolError("Flip-router scratch root is a symlink.")
    base.mkdir(parents=True, exist_ok=True)
    _require_owned_scratch_base(base)
    local_root = base / LOCAL_GENERATION_DIRECTORY
    if local_root.is_symlink():
        raise ProtocolError("Flip-router source-generation scratch is a symlink.")
    local_root.mkdir(parents=True, exist_ok=True)
    _require_owned_generation_inventory(local_root)
    local = materialize_frozen_source_streams(config, generation_lock, root=local_root)
    _require_exact_source_inventory(local_root)
    stage_frozen_source_streams(
        local,
        scratch_root=root.parent,
        canonical_root=local_root,
        local_directory=root.name,
    )
    canonical = load_frozen_source_streams(
        root,
        expected_config_hash=str(getattr(config, "contract_hash")),
        expected_generation_lock_hash=str(getattr(generation_lock, "generation_lock_hash")),
    )
    if dict(canonical.lock_payload) != dict(local.lock_payload):
        raise ProtocolError("Canonical source cache differs from local generation cache.")
    return canonical


def stage_sources_for_cpu(cache: FrozenSourceStreamCache, *, config: object, root: Path) -> FrozenSourceStreamCache:
    if tuple(getattr(config, "runtime")["scratch_preference"]) != (SCRATCH_ROOT, "artifact_parent"):
        raise ProtocolError("Flip-router scratch preference drifted.")
    base = Path(SCRATCH_ROOT)
    _require_owned_scratch_base(base)
    destination = base / LOCAL_SOURCE_DIRECTORY
    if destination.exists() and any(destination.iterdir()):
        _require_exact_source_inventory(destination)
        existing = load_frozen_source_streams(
            destination,
            expected_config_hash=str(getattr(config, "contract_hash")),
            expected_generation_lock_hash=str(
                cache.lock_payload["generation_lock_hash"]
            ),
        )
        if dict(existing.lock_payload) != dict(cache.lock_payload):
            raise ProtocolError(
                "Existing staged source cache differs from canonical bytes."
            )
    staged = stage_frozen_source_streams(
        cache,
        scratch_root=base,
        canonical_root=root,
        local_directory=LOCAL_SOURCE_DIRECTORY,
    )
    if staged.root.resolve() != root.resolve():
        _require_exact_source_inventory(staged.root)
    return staged


def materialize_probabilities(
    config: object,
    source_cache: FrozenSourceStreamCache,
    frame: object,
    partition: ThreeRolePartition,
    *,
    root: Path,
) -> GlobalPredictionSeal:
    base = Path(SCRATCH_ROOT)
    _require_owned_scratch_base(base)
    prediction_scratch = base / LOCAL_PREDICTION_DIRECTORY
    if prediction_scratch.is_symlink():
        raise ProtocolError("Flip-router prediction scratch is a symlink.")
    prediction_scratch.mkdir(parents=True, exist_ok=True)
    return materialize_fixed_bank_a1_action_predictions(
        config,
        source_cache,
        frame,
        partition_hash=partition.partition_hash,
        action_library=action_library_by_target(),
        root=root,
        scratch_root=prediction_scratch,
    )


def runtime_summary_payload(
    *,
    source_cache: FrozenSourceStreamCache,
    prediction: GlobalPredictionSeal,
    preflight: Mapping[str, object],
    local_staging: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_labeled_support_flip_runtime_summary_v1",
        "status": "PASS",
        "source_stream_lock_hash": source_cache.lock_hash,
        "global_prediction_seal_hash": prediction.seal_hash,
        "source_stream_count": len(source_cache.records),
        "classifier_cell_count": len(prediction.store.cells),
        "unique_classifier_fit_count": len(prediction.store.cells),
        "physical_action_count_per_target": 10,
        "local_source_staging": dict(local_staging),
        "workstation_preflight": dict(preflight),
        "scratch_root": SCRATCH_ROOT,
        "previous_stage90_output_prediction_or_scratch_reused": False,
        "recomputed_from_original_six_inputs": True,
        "gpu_source_phase_completed_before_cuda_free_cpu_fit_phase": True,
        "persistent_a5000_gpu_worker_count": 2,
        "cpu_classifier_worker_count": 4,
        "blas_threads_per_classifier_worker": 3,
        "multiprocessing_start_method": "spawn",
        "float32_source_and_probability_store": True,
        "float64_scientific_reductions": True,
        "resume_checkpoints_hash_validated": True,
    }


def cleanup_validated_local_stage(
    config: object,
    *,
    canonical_source: FrozenSourceStreamCache | None = None,
) -> None:
    if tuple(getattr(config, "runtime")["scratch_preference"]) != (SCRATCH_ROOT, "artifact_parent"):
        raise ProtocolError("Refusing to clean noncanonical flip-router scratch.")
    base = Path(SCRATCH_ROOT)
    if base.is_symlink():
        raise ProtocolError("Refusing to clean a symlinked flip-router scratch root.")
    _require_owned_scratch_base(base)
    source_path = base / LOCAL_SOURCE_DIRECTORY
    generation_path = base / LOCAL_GENERATION_DIRECTORY
    prediction_path = base / LOCAL_PREDICTION_DIRECTORY
    if source_path.exists():
        if source_path.is_symlink() or not source_path.is_dir():
            raise ProtocolError("Refusing to clean unsafe staged source cache.")
        canonical = canonical_source or load_frozen_source_streams(Path(getattr(config, "artifact_root")), expected_config_hash=str(getattr(config, "contract_hash")))
        staged = load_frozen_source_streams(source_path, expected_config_hash=str(getattr(config, "contract_hash")))
        if dict(staged.lock_payload) != dict(canonical.lock_payload):
            raise ProtocolError("Staged source cache differs from canonical bytes.")
        _require_exact_source_inventory(source_path)
        shutil.rmtree(source_path)
    if prediction_path.exists():
        if prediction_path.is_symlink() or not prediction_path.is_dir() or any(prediction_path.iterdir()):
            raise ProtocolError("Refusing to clean nonempty/unsafe prediction scratch root.")
        prediction_path.rmdir()
    if generation_path.exists():
        if generation_path.is_symlink() or not generation_path.is_dir():
            raise ProtocolError("Refusing to clean unsafe source-generation scratch.")
        local = load_frozen_source_streams(
            generation_path,
            expected_config_hash=str(getattr(config, "contract_hash")),
        )
        canonical = canonical_source or load_frozen_source_streams(
            Path(getattr(config, "artifact_root")),
            expected_config_hash=str(getattr(config, "contract_hash")),
        )
        if dict(local.lock_payload) != dict(canonical.lock_payload):
            raise ProtocolError("Local generation cache differs from canonical source cache.")
        _require_exact_source_inventory(generation_path)
        shutil.rmtree(generation_path)
    if base.exists() and not base.is_symlink() and base.is_dir() and not any(base.iterdir()):
        base.rmdir()


def _require_owned_scratch_base(base: Path) -> None:
    """Reject foreign or linked members in the experiment-owned local root."""

    if not base.exists():
        return
    if base.is_symlink() or not base.is_dir():
        raise ProtocolError("Flip-router scratch root is unsafe.")
    allowed = {
        LOCAL_SOURCE_DIRECTORY,
        LOCAL_GENERATION_DIRECTORY,
        LOCAL_PREDICTION_DIRECTORY,
    }
    observed = set()
    for member in base.iterdir():
        if member.is_symlink():
            raise ProtocolError("Flip-router scratch root contains a symlink.")
        if not member.is_dir():
            raise ProtocolError("Flip-router scratch root contains a foreign file.")
        observed.add(member.name)
    if not observed <= allowed:
        raise ProtocolError(
            f"Flip-router scratch root contains foreign directories: "
            f"{sorted(observed - allowed)}."
        )


def _require_owned_generation_inventory(path: Path) -> None:
    """Validate the resumable local source tree before it may be mutated."""

    members = tuple(path.rglob("*"))
    if any(member.is_symlink() for member in members):
        raise ProtocolError("Flip-router source-generation tree contains a symlink.")
    allowed_directories = {
        "arrays",
        "manifests",
        "checkpoints",
        "checkpoints/frozen_source_streams",
    }
    directories = {
        member.relative_to(path).as_posix()
        for member in members
        if member.is_dir()
    }
    if not directories <= allowed_directories:
        raise ProtocolError("Flip-router source-generation tree has foreign directories.")
    centers = r"(?:0|1|2|3|5|6|7|8|9)"
    seeds = r"(?:17|42|101)"
    allowed_files = (
        r"arrays/frozen_source_streams\.npy",
        r"manifests/frozen_source_stream_index\.json",
        r"manifests/frozen_source_stream_lock\.json",
        rf"checkpoints/frozen_source_streams/source_{centers}_train_{seeds}\.(?:json|npy)",
    )
    for member in members:
        if not member.is_file():
            continue
        relative = member.relative_to(path).as_posix()
        base = re.sub(r"\.[1-9][0-9]*\.tmp$", "", relative)
        if not any(re.fullmatch(pattern, base) for pattern in allowed_files):
            raise ProtocolError(
                "Flip-router source-generation tree contains a foreign file."
            )


def _require_exact_source_inventory(path: Path) -> None:
    from ...runtime.frozen_source_streams import (
        SOURCE_ARRAY_MEMBER,
        SOURCE_INDEX_MEMBER,
        SOURCE_LOCK_MEMBER,
    )

    expected = {SOURCE_ARRAY_MEMBER, SOURCE_INDEX_MEMBER, SOURCE_LOCK_MEMBER}
    members = tuple(path.rglob("*"))
    observed = {member.relative_to(path).as_posix() for member in members if member.is_file()}
    directories = {member.relative_to(path).as_posix() for member in members if member.is_dir()}
    if (
        any(member.is_symlink() for member in members)
        or observed != expected
        or directories != {"arrays", "manifests"}
    ):
        raise ProtocolError(
            f"Flip-router local source inventory drifted: "
            f"missing={sorted(expected - observed)}, extras={sorted(observed - expected)}."
        )


__all__ = (
    "GlobalPredictionSeal", "build_case_partition", "cleanup_validated_local_stage",
    "load_frozen_source_streams", "load_global_prediction_seal", "materialize_probabilities",
    "load_validated_workstation_preflight", "materialize_sources", "run_workstation_preflight",
    "runtime_summary_payload", "stage_sources_for_cpu",
)
