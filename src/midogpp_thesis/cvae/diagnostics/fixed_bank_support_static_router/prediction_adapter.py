"""Adapters into the neutral frozen-source and exact-A1 prediction runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import tempfile
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
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
from ...runtime.preflight import (
    REQUIRED_DISTRIBUTIONS,
    REQUIRED_THREAD_ENVIRONMENT,
    run_label_free_workstation_preflight,
)
from .actions import actions_for_target
from .experiment_contracts import (
    ACTION_COUNT_PER_TARGET,
    CENTERS,
    SCRATCH_ROOT,
    TARGET_PROBABILITY_CELL_COUNT,
    TARGET_TASK_COUNT,
)


LOCAL_GENERATION_DIRECTORY = "source_generation"
LOCAL_SOURCE_DIRECTORY = "source_cache"
LOCAL_PREDICTION_DIRECTORY = "prediction_cache"


def run_workstation_preflight(
    root: Path, *, runtime: Mapping[str, object]
) -> Mapping[str, object]:
    _assert_runtime(runtime)
    report = Path(root) / "reports/workstation_preflight.json"
    if report.is_symlink():
        raise ProtocolError("S4 workstation preflight path is a symlink.")
    if report.exists():
        if not report.is_file():
            raise ProtocolError("S4 workstation preflight path is not a file.")
        # Volatile GPU/disk observations are evidence from admission.  A retry
        # validates and reuses those exact bytes; it never silently replaces
        # the evidence with a later workstation snapshot.
        admitted = load_validated_workstation_preflight(root, runtime=runtime)
        with tempfile.TemporaryDirectory(
            prefix=".midogpp-s4-preflight-reprobe-",
            dir=Path(root).parent,
        ) as temporary:
            current = _run_workstation_probe(
                Path(temporary) / "probe_artifact", runtime=runtime
            )
        _validate_workstation_preflight_payload(current)
        return admitted
    current = _run_workstation_probe(root, runtime=runtime)
    _validate_workstation_preflight_payload(current)
    return load_validated_workstation_preflight(root, runtime=runtime)


def _run_workstation_probe(
    root: Path, *, runtime: Mapping[str, object]
) -> Mapping[str, object]:
    shared = _neutral_runtime(runtime)
    shared.update(
        {
            "cuda_visible_devices": "0,1",
            "launch_blas_threads": 1,
            "minimum_logical_cpu_count": 12,
            "minimum_physical_ram_bytes": 107_374_182_400,
            "minimum_artifact_disk_free_bytes": 12_884_901_888,
            "minimum_gpu_free_mib_per_device": 18_000,
            "source_job_count": 27,
            "source_stream_count": 81,
            "target_action_identity_count": 90,
            "target_unique_classifier_fit_count": 810,
            "resume_policy": "hash_validated_atomic_phase_and_task_checkpoints",
        }
    )
    return run_label_free_workstation_preflight(
        root,
        runtime=shared,
        expected_scratch_root=SCRATCH_ROOT,
        expected_target_action_identity_count=len(CENTERS) * ACTION_COUNT_PER_TARGET,
        expected_target_probability_cell_count=TARGET_PROBABILITY_CELL_COUNT,
        expected_unique_classifier_fit_count=TARGET_PROBABILITY_CELL_COUNT,
    )


def load_validated_workstation_preflight(
    root: Path, *, runtime: Mapping[str, object]
) -> Mapping[str, object]:
    _assert_runtime(runtime)
    path = root / "reports/workstation_preflight.json"
    if path.is_symlink() or not path.is_file():
        raise ProtocolError("S4 workstation preflight is absent or unsafe.")
    payload = read_json(path)
    _validate_workstation_preflight_payload(payload)
    return payload


def _validate_workstation_preflight_payload(
    payload: Mapping[str, object],
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
    }
    if any(payload.get(key) != value for key, value in exact.items()):
        raise ProtocolError("S4 workstation preflight topology drifted.")
    versions = payload.get("package_versions")
    gpus = payload.get("gpus")
    if (
        not isinstance(versions, Mapping)
        or set(versions) != set(REQUIRED_DISTRIBUTIONS)
        or not isinstance(gpus, list)
        or len(gpus) != 2
    ):
        raise ProtocolError("S4 workstation preflight observations drifted.")
    for expected_index, row in enumerate(gpus):
        if (
            not isinstance(row, Mapping)
            or row.get("index") != expected_index
            or "RTX A5000" not in str(row.get("name"))
            or int(row.get("memory_total_mib", -1)) < 23_000
            or int(row.get("memory_free_mib", -1)) < 18_000
        ):
            raise ProtocolError("S4 GPU preflight drifted.")


def materialize_sources(
    config: object, generation_lock: object, *, root: Path
) -> FrozenSourceStreamCache:
    existing = _load_existing_canonical_source_or_require_safe_partial(
        config, generation_lock, root=root
    )
    if existing is not None:
        return existing
    base = _owned_scratch_base(create=True)
    local_root = base / LOCAL_GENERATION_DIRECTORY
    _plain_directory(local_root)
    _assert_owned_scratch_inventory(base)
    local = materialize_frozen_source_streams(
        _neutral_config(config), generation_lock, root=local_root
    )
    _stage_canonical_source_nonrepairing(local, destination=root)
    canonical = load_frozen_source_streams(
        root,
        expected_config_hash=str(getattr(config, "contract_hash")),
        expected_generation_lock_hash=str(
            getattr(generation_lock, "generation_lock_hash")
        ),
    )
    if dict(canonical.lock_payload) != dict(local.lock_payload):
        raise ProtocolError("S4 canonical source cache differs from scratch bytes.")
    return canonical


def _load_existing_canonical_source_or_require_safe_partial(
    config: object, generation_lock: object, *, root: Path
) -> FrozenSourceStreamCache | None:
    from ...runtime.frozen_source_streams import (
        SOURCE_ARRAY_MEMBER,
        SOURCE_INDEX_MEMBER,
        SOURCE_LOCK_MEMBER,
    )

    members = (SOURCE_ARRAY_MEMBER, SOURCE_INDEX_MEMBER, SOURCE_LOCK_MEMBER)
    paths = tuple(root / member for member in members)
    _assert_plain_source_parent_chains(root, paths)
    if any(path.is_symlink() for path in paths):
        raise ProtocolError("S4 canonical source member is a symlink.")
    present = tuple(path.is_file() for path in paths)
    if any(path.exists() and not path.is_file() for path in paths) or present not in {
        (False, False, False),
        (True, False, False),
        (True, True, False),
        (True, True, True),
    }:
        raise ProtocolError("S4 canonical source trio is an unsafe partial state.")
    if not all(present):
        return None
    return load_frozen_source_streams(
        root,
        expected_config_hash=str(getattr(config, "contract_hash")),
        expected_generation_lock_hash=str(
            getattr(generation_lock, "generation_lock_hash")
        ),
    )


def _stage_canonical_source_nonrepairing(
    cache: FrozenSourceStreamCache, *, destination: Path
) -> FrozenSourceStreamCache:
    """Publish only missing ordered source members; never repair canonical bytes."""

    from ...runtime.artifact_io import sha256_file
    from ...runtime.frozen_source_streams import (
        SOURCE_ARRAY_MEMBER,
        SOURCE_INDEX_MEMBER,
        SOURCE_LOCK_MEMBER,
    )

    members = (SOURCE_ARRAY_MEMBER, SOURCE_INDEX_MEMBER, SOURCE_LOCK_MEMBER)
    paths = tuple(destination / member for member in members)
    _assert_plain_source_parent_chains(destination, paths)
    if any(path.is_symlink() for path in paths):
        raise ProtocolError("S4 canonical source member is a symlink.")
    present = tuple(path.is_file() for path in paths)
    if any(path.exists() and not path.is_file() for path in paths) or present not in {
        (False, False, False),
        (True, False, False),
        (True, True, False),
        (True, True, True),
    }:
        raise ProtocolError("S4 canonical source trio is an unsafe partial state.")
    expected = {
        SOURCE_ARRAY_MEMBER: str(cache.lock_payload["source_array_sha256"]),
        SOURCE_INDEX_MEMBER: str(cache.lock_payload["source_stream_index_sha256"]),
        SOURCE_LOCK_MEMBER: sha256_file(cache.root / SOURCE_LOCK_MEMBER),
    }
    for member, path, exists in zip(members, paths, present, strict=True):
        if exists and sha256_file(path) != expected[member]:
            raise ProtocolError(
                "Existing S4 canonical source member differs; refusing repair."
            )
    if all(present):
        canonical = load_frozen_source_streams(
            destination,
            expected_config_hash=str(cache.lock_payload["config_contract_hash"]),
            expected_generation_lock_hash=str(
                cache.lock_payload["generation_lock_hash"]
            ),
        )
        if dict(canonical.lock_payload) != dict(cache.lock_payload):
            raise ProtocolError("S4 canonical source cache differs from scratch bytes.")
        return canonical
    staged = stage_frozen_source_streams(
        cache,
        scratch_root=destination.parent,
        canonical_root=cache.root,
        local_directory=destination.name,
    )
    if dict(staged.lock_payload) != dict(cache.lock_payload):
        raise ProtocolError("S4 canonical source cache differs from scratch bytes.")
    return staged


def _assert_plain_source_parent_chains(
    root: Path, member_paths: tuple[Path, ...]
) -> None:
    base = Path(root)
    if base.exists() and (base.is_symlink() or not base.is_dir()):
        raise ProtocolError("S4 canonical source root is unsafe.")
    for member in member_paths:
        current = member.parent
        try:
            current.relative_to(base)
        except ValueError as exc:
            raise ProtocolError("S4 canonical source member escapes its root.") from exc
        while current != base:
            if current.exists() and (current.is_symlink() or not current.is_dir()):
                raise ProtocolError("S4 canonical source parent is unsafe.")
            current = current.parent


def stage_sources_for_cpu(
    cache: FrozenSourceStreamCache, *, config: object, root: Path
) -> FrozenSourceStreamCache:
    if tuple(getattr(config, "runtime")["scratch_preference"]) != (
        SCRATCH_ROOT,
        "artifact_parent",
    ):
        raise ProtocolError("S4 scratch preference drifted.")
    base = _owned_scratch_base(create=True)
    destination = base / LOCAL_SOURCE_DIRECTORY
    _plain_directory(destination)
    staged = stage_frozen_source_streams(
        cache,
        scratch_root=base,
        canonical_root=root,
        local_directory=LOCAL_SOURCE_DIRECTORY,
    )
    if dict(staged.lock_payload) != dict(cache.lock_payload):
        raise ProtocolError("S4 CPU-staged source cache differs from canonical.")
    return staged


def materialize_probabilities(
    config: object,
    source_cache: FrozenSourceStreamCache,
    frame: object,
    partition: object,
    *,
    root: Path,
) -> GlobalPredictionSeal:
    base = _owned_scratch_base(create=True)
    prediction_scratch = base / LOCAL_PREDICTION_DIRECTORY
    _plain_directory(prediction_scratch)
    action_library = {
        target: tuple(actions_for_target(target)) for target in CENTERS
    }
    return materialize_fixed_bank_a1_action_predictions(
        _neutral_config(config),
        source_cache,
        frame,
        partition_hash=str(getattr(partition, "partition_hash")),
        action_library=action_library,
        root=root,
        scratch_root=prediction_scratch,
    )


def enter_cuda_free_cpu_phase() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"


def runtime_summary_payload(
    *,
    source_cache: FrozenSourceStreamCache,
    prediction: GlobalPredictionSeal,
    preflight: Mapping[str, object],
    staged_source: FrozenSourceStreamCache,
    artifact_root: Path,
) -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_support_static_router_runtime_summary_v1",
        "status": "PASS",
        "source_stream_lock_hash": source_cache.lock_hash,
        "global_prediction_seal_hash": prediction.seal_hash,
        "source_stream_count": len(source_cache.records),
        "classifier_cell_count": len(prediction.store.cells),
        "unique_classifier_fit_count": len(prediction.store.cells),
        "physical_action_count_per_target": ACTION_COUNT_PER_TARGET,
        "workstation_preflight": dict(preflight),
        "scratch_root": SCRATCH_ROOT,
        "local_cpu_source_stage_used": staged_source.root.resolve()
        != artifact_root.resolve(),
        "previous_stage90_output_prediction_or_scratch_reused": False,
        "recomputed_from_original_six_inputs": True,
        "gpu_source_phase_completed_before_cuda_free_cpu_fit_phase": True,
        "persistent_a5000_gpu_worker_count": 2,
        "cpu_classifier_worker_count": 4,
        "blas_threads_per_classifier_worker": 3,
        "multiprocessing_start_method": "spawn",
        "float32_source_and_probability_store": True,
        "float64_scientific_reductions": True,
        "donor_model_fit_count": 0,
        "target_calibration_fit_count": 0,
        "replayed_phase_checkpoints_hash_validated": True,
        "terminal_checkpoint_recovery_supported": False,
        "terminal_checkpoint_is_atomicity_boundary_only": True,
    }


def cleanup_validated_local_stage(
    config: object, *, canonical_source: FrozenSourceStreamCache | None = None
) -> None:
    if tuple(getattr(config, "runtime")["scratch_preference"]) != (
        SCRATCH_ROOT,
        "artifact_parent",
    ):
        raise ProtocolError("Refusing to clean a noncanonical S4 scratch root.")
    base = Path(SCRATCH_ROOT)
    if not base.exists():
        return
    _assert_owned_scratch_inventory(base)
    canonical = canonical_source
    if canonical is None:
        canonical = load_frozen_source_streams(
            Path(getattr(config, "artifact_root")),
            expected_config_hash=str(getattr(config, "contract_hash")),
        )
    for name in (LOCAL_SOURCE_DIRECTORY, LOCAL_GENERATION_DIRECTORY):
        path = base / name
        if not path.exists():
            continue
        local = load_frozen_source_streams(
            path, expected_config_hash=str(getattr(config, "contract_hash"))
        )
        if dict(local.lock_payload) != dict(canonical.lock_payload):
            raise ProtocolError("Refusing to clean changed S4 source scratch.")
        _remove_empty_checkpoint_parent(path)
        shutil.rmtree(path)
    prediction = base / LOCAL_PREDICTION_DIRECTORY
    if prediction.exists():
        if prediction.is_symlink() or not prediction.is_dir():
            raise ProtocolError("Refusing to clean unsafe S4 prediction scratch.")
        _remove_empty_checkpoint_parent(prediction)
        if any(prediction.iterdir()):
            raise ProtocolError("Refusing to clean nonempty S4 prediction scratch.")
        prediction.rmdir()
    if not any(base.iterdir()):
        base.rmdir()


def _assert_runtime(runtime: Mapping[str, object]) -> None:
    if (
        tuple(runtime.get("generation_devices", ())) != ("cuda:0", "cuda:1")
        or int(runtime.get("classifier_workers", -1)) != 4
        or int(runtime.get("classifier_threads_per_worker", -1)) != 3
        or runtime.get("multiprocessing_start_method") != "spawn"
        or runtime.get("phase_disjoint_gpu_and_cpu_pools") is not True
        or runtime.get("generated_cache_format") != "float32_npy_memmap"
        or runtime.get("scientific_reductions_dtype") != "float64"
        or int(runtime.get("target_task_count", -1)) != TARGET_TASK_COUNT
        or int(runtime.get("target_probability_cell_count", -1))
        != TARGET_PROBABILITY_CELL_COUNT
        or runtime.get("resume_policy")
        != "deterministic_restart_from_admission_with_nonrepairing_hash_validation"
        or tuple(runtime.get("scratch_preference", ()))
        != (SCRATCH_ROOT, "artifact_parent")
    ):
        raise ProtocolError("S4 workstation runtime contract drifted.")


@dataclass(frozen=True)
class _NeutralRuntimeConfig:
    contract_hash: str
    expert_bank_root: Path
    classifier: object
    runtime: Mapping[str, object]


def _neutral_config(config: object) -> _NeutralRuntimeConfig:
    return _NeutralRuntimeConfig(
        contract_hash=str(getattr(config, "contract_hash")),
        expert_bank_root=Path(getattr(config, "expert_bank_root")),
        classifier=getattr(config, "classifier"),
        runtime=_neutral_runtime(getattr(config, "runtime")),
    )


def _neutral_runtime(runtime: Mapping[str, object]) -> dict[str, object]:
    """Adapt naming only; no scientific or resource setting is relaxed."""

    result = dict(runtime)
    result.update(
        {
            "parent_cuda_context_forbidden": True,
            "source_prefix_rows_per_class": 270,
            "source_job_count": 27,
            "source_stream_count": 81,
            # The shared neutral runtime predates this terminal diagnostic's
            # stricter public restart wording.  This adapter preserves its
            # internal checkpoint contract without widening S4's claim.
            "resume_policy": "hash_validated_atomic_phase_and_task_checkpoints",
        }
    )
    return result


def _owned_scratch_base(*, create: bool) -> Path:
    base = Path(SCRATCH_ROOT)
    if base.is_symlink():
        raise ProtocolError("S4 scratch root is a symlink.")
    if create:
        base.mkdir(parents=True, exist_ok=True)
    if not base.is_dir() or base.is_symlink():
        raise ProtocolError("S4 scratch root is unsafe.")
    _assert_owned_scratch_inventory(base)
    return base


def _assert_owned_scratch_inventory(base: Path) -> None:
    allowed = {
        LOCAL_GENERATION_DIRECTORY,
        LOCAL_SOURCE_DIRECTORY,
        LOCAL_PREDICTION_DIRECTORY,
    }
    for member in base.iterdir():
        if member.is_symlink() or not member.is_dir() or member.name not in allowed:
            raise ProtocolError("S4 scratch root contains a foreign member.")


def _plain_directory(path: Path) -> None:
    if path.is_symlink():
        raise ProtocolError("S4 scratch subdirectory is a symlink.")
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir() or path.is_symlink():
        raise ProtocolError("S4 scratch subdirectory is unsafe.")


def _remove_empty_checkpoint_parent(path: Path) -> None:
    checkpoint = path / "checkpoints"
    if not checkpoint.exists():
        return
    if checkpoint.is_symlink() or not checkpoint.is_dir() or any(checkpoint.iterdir()):
        raise ProtocolError("S4 scratch checkpoint parent is not empty and safe.")
    checkpoint.rmdir()


__all__ = (
    "GlobalPredictionSeal",
    "FrozenSourceStreamCache",
    "cleanup_validated_local_stage",
    "enter_cuda_free_cpu_phase",
    "load_frozen_source_streams",
    "load_global_prediction_seal",
    "load_validated_workstation_preflight",
    "materialize_probabilities",
    "materialize_sources",
    "run_workstation_preflight",
    "runtime_summary_payload",
    "stage_sources_for_cpu",
)
