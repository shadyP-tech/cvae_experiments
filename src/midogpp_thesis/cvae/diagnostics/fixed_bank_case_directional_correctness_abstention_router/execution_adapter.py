"""Isolated adapters to the neutral frozen-stream and fixed-bank A1 runtime."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...runtime.artifact_io import atomic_json, read_json, sha256_array
from ...runtime.fixed_bank_a1_action_predictions import (
    GlobalPredictionSeal,
    materialize_fixed_bank_a1_action_predictions,
)
from ...runtime.frozen_source_streams import (
    FrozenSourceStreamCache,
    load_frozen_source_streams,
    materialize_frozen_source_streams,
    stage_frozen_source_streams,
)
from ...runtime.preflight import run_label_free_workstation_preflight as _preflight
from .constants import (
    ACTION_COUNT_PER_TARGET,
    CENTERS,
    EXPECTED_TEST_ROW_COUNT,
    TARGET_PROBABILITY_CELL_COUNT,
)
from .experiment_contracts import EXPECTED_GENERATION_LOCK_HASH, SCRATCH_ROOT
from .hashing import canonical_hash, json_native


@dataclass(frozen=True)
class ProbabilityIndexRow:
    """Compact binding of one target/action mean to its nine physical cells."""

    target_center: str
    action_id: str
    row_count: int
    source_cell_probability_sha256: tuple[str, ...]
    sample_identity_hash: str
    case_identity_hash: str
    exact_nine_probability_sha256: str
    storage_dtype: str = "float32"
    reduction_dtype: str = "float64"

    def to_payload(self) -> dict[str, object]:
        return json_native(self)  # type: ignore[return-value]


@dataclass(frozen=True)
class RuntimeProbabilitySurface:
    """Science-facing exact-nine rows plus their compact persisted index."""

    rows: tuple[object, ...]
    index_rows: tuple[ProbabilityIndexRow, ...]
    store_hash: str
    surface_hash: str


@dataclass(frozen=True)
class MaterializedSourceCaches:
    """Canonical bundle cache plus same-seal local classifier cache."""

    canonical: FrozenSourceStreamCache
    local: FrozenSourceStreamCache

    @property
    def lock_hash(self) -> str:
        return self.canonical.lock_hash

    @property
    def records(self) -> tuple[object, ...]:
        return self.canonical.records


LOCAL_GENERATION_DIRECTORY = "source_generation"
LOCAL_PREDICTION_DIRECTORY = "prediction_cache"


def run_label_free_workstation_preflight(
    root: Path, *, runtime: Mapping[str, object]
) -> Mapping[str, object]:
    """Probe the exact topology while preserving the stricter DCSE resume rule."""

    if (
        runtime.get("resume_policy")
        != "no_cross_run_recovery_intra_launch_atomic_task_checkpoints_only"
        or runtime.get("owned_task_checkpoint_replay_allowed") is not False
        or runtime.get("foreign_checkpoint_reuse_forbidden") is not True
        or runtime.get("cross_run_recovery_allowed") is not False
        or runtime.get("terminal_recovery_allowed") is not False
        or runtime.get(
            "successful_phase_checkpoint_cleanup_after_validated_global_seal"
        )
        is not True
        or tuple(runtime.get("scratch_preference", ()))
        != (SCRATCH_ROOT, "artifact_parent")
        or runtime.get("probability_storage_dtype") != "float32"
        or runtime.get("confusion_count_dtype") != "int64"
    ):
        raise ProtocolError("Case-directional runtime recovery contract drifted.")
    neutral = dict(runtime)
    neutral["resume_policy"] = "hash_validated_atomic_phase_and_task_checkpoints"
    # The neutral probe requires the declared scratch identity; the actual
    # dedicated tree is created fresh only when the generation phase starts.
    neutral["scratch_preference"] = [SCRATCH_ROOT, "artifact_parent"]
    with tempfile.TemporaryDirectory(
        prefix=".cdca-preflight-", dir=root.parent
    ) as probe:
        probed = dict(
            _preflight(
            Path(probe),
            runtime=neutral,
            expected_scratch_root=SCRATCH_ROOT,
            expected_target_action_identity_count=90,
            expected_target_probability_cell_count=TARGET_PROBABILITY_CELL_COUNT,
            expected_unique_classifier_fit_count=TARGET_PROBABILITY_CELL_COUNT,
            )
        )
    # No persisted report may expose raw filesystem paths.  The neutral probe
    # can return local diagnostic path fields; retain only their measurements.
    probed = {
        key: value
        for key, value in probed.items()
        if not str(key).casefold().endswith("_path")
    }
    scratch_checks = _probe_dedicated_scratch(runtime)
    additions = {
        "schema_version": "fixed_bank_cdca_workstation_preflight_v1",
        "resume_policy": runtime["resume_policy"],
        "scratch_preference": [SCRATCH_ROOT, "artifact_parent"],
        "owned_task_checkpoint_replay_allowed": False,
        "task_checkpoints_are_intra_launch_atomicity_only": True,
        "foreign_checkpoint_reuse_forbidden": True,
        "cross_run_recovery_allowed": False,
        "terminal_recovery_allowed": False,
        "cuda_disabled_after_generation": True,
        "probability_storage_dtype": "float32",
        "confusion_count_dtype": "int64",
        "scientific_reductions_dtype": "float64",
        **scratch_checks,
    }
    probed.update(additions)
    atomic_json(root / "reports/workstation_preflight.json", probed)
    return probed


def _probe_dedicated_scratch(runtime: Mapping[str, object]) -> dict[str, object]:
    """Fail before GPU work unless the literal scratch parent is writable."""

    scratch = _literal_scratch_root()
    parent = scratch.parent
    if scratch.exists() or scratch.is_symlink():
        raise ProtocolError(
            "Case-directional prior-run or foreign scratch is forbidden."
        )
    if parent.is_symlink() or not parent.is_dir():
        raise ProtocolError("Case-directional scratch parent is absent or unsafe.")
    free_bytes = int(shutil.disk_usage(parent).free)
    if free_bytes < int(runtime["minimum_artifact_disk_free_bytes"]):
        raise ProtocolError("Case-directional scratch filesystem reserve is too low.")
    try:
        with tempfile.TemporaryDirectory(
            prefix=".cdca-write-probe-", dir=parent
        ) as probe:
            marker = Path(probe) / "write_probe"
            marker.write_bytes(b"cdca-scratch-write-probe\n")
            with marker.open("r+b") as handle:
                os.fsync(handle.fileno())
    except OSError as exc:
        raise ProtocolError(
            "Case-directional scratch parent is not writable."
        ) from exc
    return {
        "dedicated_scratch_absent_at_launch": True,
        "dedicated_scratch_parent_writable": True,
        "dedicated_scratch_free_bytes_at_launch": free_bytes,
    }


def load_validated_workstation_preflight(
    root: Path, *, runtime: Mapping[str, object]
) -> Mapping[str, object]:
    payload = read_json(root / "reports/workstation_preflight.json")
    if (
        payload.get("schema_version") != "fixed_bank_cdca_workstation_preflight_v1"
        or payload.get("status") != "PASS"
        or payload.get("resume_policy") != runtime.get("resume_policy")
        or payload.get("scratch_preference") != [SCRATCH_ROOT, "artifact_parent"]
        or payload.get("owned_task_checkpoint_replay_allowed") is not False
        or payload.get("task_checkpoints_are_intra_launch_atomicity_only") is not True
        or payload.get("foreign_checkpoint_reuse_forbidden") is not True
        or payload.get("cross_run_recovery_allowed") is not False
        or payload.get("terminal_recovery_allowed") is not False
        or payload.get("generation_devices") != ["cuda:0", "cuda:1"]
        or payload.get("persistent_gpu_workers") != 2
        or payload.get("classifier_workers") != 4
        or payload.get("blas_threads_per_classifier_worker") != 3
        or payload.get("target_probability_cell_count")
        != TARGET_PROBABILITY_CELL_COUNT
        or payload.get("probability_storage_dtype") != "float32"
        or payload.get("confusion_count_dtype") != "int64"
        or payload.get("scientific_reductions_dtype") != "float64"
    ):
        raise ProtocolError("Case-directional persisted preflight drifted.")
    return payload


def materialize_sources(
    config: object, generation_lock: object, *, root: Path
) -> MaterializedSourceCaches:
    """Materialize only under this artifact; no historical/local scratch input."""

    base = _fresh_scratch_base()
    local_root = base / LOCAL_GENERATION_DIRECTORY
    local_root.mkdir(parents=True, exist_ok=False)
    local = materialize_frozen_source_streams(config, generation_lock, root=local_root)
    stage_frozen_source_streams(
        local,
        scratch_root=root.parent,
        canonical_root=local_root,
        local_directory=root.name,
    )
    canonical = load_frozen_source_streams(
        root,
        expected_config_hash=str(getattr(config, "contract_hash")),
        expected_generation_lock_hash=str(
            getattr(generation_lock, "generation_lock_hash")
        ),
    )
    if dict(canonical.lock_payload) != dict(local.lock_payload):
        raise ProtocolError("Case-directional staged source bytes drifted.")
    return MaterializedSourceCaches(canonical=canonical, local=local)


def materialize_probabilities(
    config: object,
    source_cache: FrozenSourceStreamCache,
    frame: object,
    *,
    partition_hash: str,
    action_library: Mapping[str, Sequence[object]],
    root: Path,
) -> GlobalPredictionSeal:
    return materialize_fixed_bank_a1_action_predictions(
        config,
        source_cache,
        frame,
        partition_hash=partition_hash,
        action_library=action_library,
        root=root,
        scratch_root=_prediction_scratch(),
    )


def physical_partition_hash(frame: object) -> str:
    """Label-blind global row/case topology binding required by neutral runtime."""

    rows = tuple(getattr(frame, "rows"))
    return canonical_hash(
        {
            "schema_version": "fixed_bank_cdca_global_physical_plan_v1",
            "rows": [
                {
                    "target_center": str(getattr(row, "center")),
                    "case_id": str(getattr(row, "case_id")),
                    "sample_id": str(getattr(row, "sample_id")),
                }
                for row in rows
            ],
            "row_count": len(rows),
            "case_count": len(
                {
                    (str(getattr(row, "center")), str(getattr(row, "case_id")))
                    for row in rows
                }
            ),
            "labels_used": False,
            "arbitrary_folds_used": False,
        }
    )


def build_exact_nine_surface(
    prediction: GlobalPredictionSeal,
    *,
    row_factory: object | None = None,
    surface_factory: object | None = None,
) -> RuntimeProbabilitySurface | object:
    """Reconstruct exact-nine means in float64 from the 810 float32 cells."""

    if row_factory is None or surface_factory is None:
        try:
            from .probability_surfaces import (  # type: ignore[attr-defined]
                ExactNineProbabilityRow,
                ExactNineProbabilitySurface,
            )
        except ImportError:
            ExactNineProbabilityRow = None  # type: ignore[assignment,misc]
            ExactNineProbabilitySurface = None  # type: ignore[assignment,misc]
        row_factory = row_factory or ExactNineProbabilityRow
        surface_factory = surface_factory or ExactNineProbabilitySurface
    store = prediction.store
    rows: list[object] = []
    index: list[ProbabilityIndexRow] = []
    for target in CENTERS:
        target_cells = [cell for cell in store.cells if cell.target_center == target]
        action_ids = tuple(dict.fromkeys(cell.action_id for cell in target_cells))
        if len(action_ids) != ACTION_COUNT_PER_TARGET:
            raise ProtocolError("Case-directional action coverage drifted.")
        sample_ids = tuple(store.rows_by_center[target])
        case_ids = tuple(store.case_ids_by_center[target])
        for action in action_ids:
            cells = tuple(cell for cell in target_cells if cell.action_id == action)
            if len(cells) != 9:
                raise ProtocolError("Case-directional seed-pair coverage drifted.")
            matrix = np.stack([cell.probabilities for cell in cells]).astype(
                np.float64, copy=False
            )
            means = np.mean(matrix, axis=0, dtype=np.float64)
            if len(means) != len(sample_ids):
                raise ProtocolError("Case-directional target row coverage drifted.")
            index.append(
                ProbabilityIndexRow(
                    target,
                    action,
                    len(means),
                    tuple(cell.probability_sha256 for cell in cells),
                    canonical_hash(list(sample_ids)),
                    canonical_hash(list(case_ids)),
                    sha256_array(means),
                )
            )
            if row_factory is not None:
                for ordinal, (sample_id, case_id, probability) in enumerate(
                    zip(sample_ids, case_ids, means, strict=True)
                ):
                    seed_values = tuple(float(value) for value in matrix[:, ordinal])
                    rows.append(
                        row_factory(
                            target,
                            case_id,
                            sample_id,
                            action,
                            seed_values,
                        )
                    )
    if len(index) != len(CENTERS) * ACTION_COUNT_PER_TARGET:
        raise ProtocolError("Case-directional compact probability index drifted.")
    if row_factory is None or surface_factory is None:
        surface_hash = canonical_hash(
            {
                "schema_version": "fixed_bank_cdca_exact_nine_surface_v1",
                "store_hash": store.store_hash,
                "index": [row.to_payload() for row in index],
            }
        )
        return RuntimeProbabilitySurface(
            tuple(rows), tuple(index), store.store_hash, surface_hash
        )
    science_surface = surface_factory(tuple(rows), store.store_hash)
    return science_surface


def probability_index_rows(
    prediction: GlobalPredictionSeal,
) -> tuple[ProbabilityIndexRow, ...]:
    """Reconstruct the compact index independently of science serialization."""

    store = prediction.store
    rows: list[ProbabilityIndexRow] = []
    for target in CENTERS:
        sample_ids = tuple(store.rows_by_center[target])
        case_ids = tuple(store.case_ids_by_center[target])
        actions = tuple(
            dict.fromkeys(cell.action_id for cell in store.cells if cell.target_center == target)
        )
        for action in actions:
            cells = tuple(
                cell
                for cell in store.cells
                if cell.target_center == target and cell.action_id == action
            )
            values = np.mean(
                np.stack([cell.probabilities for cell in cells]).astype(np.float64),
                axis=0,
                dtype=np.float64,
            )
            rows.append(
                ProbabilityIndexRow(
                    target,
                    action,
                    len(values),
                    tuple(cell.probability_sha256 for cell in cells),
                    canonical_hash(list(sample_ids)),
                    canonical_hash(list(case_ids)),
                    sha256_array(values),
                )
            )
    return tuple(rows)


def runtime_summary_payload(
    *,
    source_cache: FrozenSourceStreamCache | MaterializedSourceCaches,
    prediction: GlobalPredictionSeal,
    preflight: Mapping[str, object],
    runtime: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_cdca_runtime_summary_v1",
        "status": "PASS",
        "source_stream_lock_hash": source_cache.lock_hash,
        "global_prediction_seal_hash": prediction.seal_hash,
        "source_stream_count": len(source_cache.records),
        "classifier_cell_count": len(prediction.store.cells),
        "unique_classifier_fit_count": len(prediction.store.cells),
        "workstation_preflight": dict(preflight),
        "source_generation_devices": ["cuda:0", "cuda:1"],
        "persistent_generation_worker_count": 2,
        "gpu_generation_completed_before_cpu_phase": True,
        "cuda_visible_devices_during_cpu_phase": "",
        "classifier_workers": int(runtime["classifier_workers"]),
        "classifier_threads_per_worker": int(
            runtime["classifier_threads_per_worker"]
        ),
        "multiprocessing_start_method": runtime["multiprocessing_start_method"],
        "source_storage_dtype": "float32",
        "probability_storage_dtype": "float32",
        "confusion_count_dtype": "int64",
        "scientific_reductions_dtype": "float64",
        "resume_policy": runtime["resume_policy"],
        "task_checkpoints_are_intra_launch_atomicity_only": True,
        "terminal_or_cross_run_recovery_used": False,
        "dedicated_local_scratch_used_for_throughput": True,
        "classifier_source_cache_role": "dedicated_intra_launch_scratch",
        "canonical_source_cache_role": "current_artifact_root",
        "scratch_root_id": "fixed_bank_case_directional_correctness_abstention_router_v1",
        "local_and_canonical_source_lock_identical": (
            not isinstance(source_cache, MaterializedSourceCaches)
            or dict(source_cache.local.lock_payload)
            == dict(source_cache.canonical.lock_payload)
        ),
        "prior_run_scratch_used_as_evidence": False,
        "previous_stage90_artifact_checkpoint_or_scratch_reused": False,
        "recomputed_from_original_six_inputs": True,
    }


def cleanup_validated_scratch(config: object) -> None:
    """Delete only this experiment's known sealed scratch after validation PASS."""

    if tuple(getattr(config, "runtime")["scratch_preference"]) != (
        SCRATCH_ROOT,
        "artifact_parent",
    ):
        raise ProtocolError("Refusing to clean noncanonical CDCA scratch.")
    base = _literal_scratch_root()
    _validate_scratch_tree(
        base,
        allow_complete_generation=True,
        require_complete_prediction=True,
    )
    local = load_frozen_source_streams(
        base / LOCAL_GENERATION_DIRECTORY,
        expected_config_hash=str(getattr(config, "contract_hash")),
        expected_generation_lock_hash=EXPECTED_GENERATION_LOCK_HASH,
    )
    canonical = load_frozen_source_streams(
        Path(getattr(config, "artifact_root")),
        expected_config_hash=str(getattr(config, "contract_hash")),
        expected_generation_lock_hash=EXPECTED_GENERATION_LOCK_HASH,
    )
    if dict(local.lock_payload) != dict(canonical.lock_payload):
        raise ProtocolError(
            "Case-directional scratch/canonical source seals differ."
        )
    if base.exists():
        shutil.rmtree(base)


def _fresh_scratch_base() -> Path:
    base = _literal_scratch_root()
    if base.exists() or base.is_symlink():
        raise ProtocolError(
            "Case-directional prior-run or foreign scratch is forbidden."
        )
    base.mkdir(parents=True, exist_ok=False)
    return base


def _prediction_scratch() -> Path:
    base = _literal_scratch_root()
    _validate_scratch_tree(base, allow_complete_generation=True)
    destination = base / LOCAL_PREDICTION_DIRECTORY
    if destination.exists() or destination.is_symlink():
        raise ProtocolError("Case-directional prediction scratch is pre-existing.")
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def _validate_scratch_tree(
    base: Path,
    *,
    allow_complete_generation: bool,
    require_complete_prediction: bool = False,
) -> None:
    if base != _literal_scratch_root():
        raise ProtocolError("Case-directional scratch root identity drifted.")
    if not base.exists():
        if base.is_symlink():
            raise ProtocolError("Case-directional scratch is a dangling symlink.")
        return
    if base.is_symlink() or not base.is_dir():
        raise ProtocolError("Case-directional scratch root is unsafe.")
    allowed_roots = {LOCAL_GENERATION_DIRECTORY, LOCAL_PREDICTION_DIRECTORY}
    for member in base.iterdir():
        if member.name not in allowed_roots or member.is_symlink() or not member.is_dir():
            raise ProtocolError("Case-directional scratch contains foreign state.")
    generation = base / LOCAL_GENERATION_DIRECTORY
    if allow_complete_generation and not generation.is_dir():
        raise ProtocolError(
            "Case-directional generation scratch is absent."
        )
    if generation.exists() and allow_complete_generation:
        allowed = {
            "arrays/frozen_source_streams.npy",
            "manifests/frozen_source_stream_index.json",
            "manifests/frozen_source_stream_lock.json",
        }
        observed = {
            path.relative_to(generation).as_posix()
            for path in generation.rglob("*")
            if path.is_file()
        }
        directories = {
            path.relative_to(generation).as_posix()
            for path in generation.rglob("*")
            if path.is_dir()
        }
        if (
            observed != allowed
            or directories
            not in (
                {"arrays", "manifests"},
                {"arrays", "manifests", "checkpoints"},
            )
            or any(path.is_symlink() for path in generation.rglob("*"))
        ):
            raise ProtocolError("Case-directional generation scratch is not sealed.")
    prediction = base / LOCAL_PREDICTION_DIRECTORY
    if require_complete_prediction and not prediction.is_dir():
        raise ProtocolError(
            "Case-directional completed prediction scratch is absent."
        )
    if prediction.exists():
        observed_files = {
            path.relative_to(prediction).as_posix()
            for path in prediction.rglob("*")
            if path.is_file()
        }
        observed_directories = {
            path.relative_to(prediction).as_posix()
            for path in prediction.rglob("*")
            if path.is_dir()
        }
        if (
            observed_files
            or observed_directories not in (set(), {"checkpoints"})
            or any(path.is_symlink() for path in prediction.rglob("*"))
        ):
            raise ProtocolError(
                "Case-directional prediction scratch is not a sealed empty tree."
            )


def _literal_scratch_root() -> Path:
    root = Path(SCRATCH_ROOT)
    if (
        not root.is_absolute()
        or str(root) != SCRATCH_ROOT
        or root.resolve(strict=False) != root
    ):
        raise ProtocolError("Case-directional scratch root is not literal/resolved.")
    return root


__all__ = (
    "GlobalPredictionSeal",
    "MaterializedSourceCaches",
    "ProbabilityIndexRow",
    "RuntimeProbabilitySurface",
    "build_exact_nine_surface",
    "cleanup_validated_scratch",
    "load_frozen_source_streams",
    "load_validated_workstation_preflight",
    "materialize_probabilities",
    "materialize_sources",
    "physical_partition_hash",
    "probability_index_rows",
    "run_label_free_workstation_preflight",
    "runtime_summary_payload",
)
