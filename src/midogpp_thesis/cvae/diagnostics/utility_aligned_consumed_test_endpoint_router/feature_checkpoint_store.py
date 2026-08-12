"""Hash-reconstructive checkpoints for label-free feature components."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import canonical_sha256
from .artifact_io import atomic_json, read_json, sha256_file
from .array_io import atomic_save_npz
from .feature_runtime_contracts import FeatureComponentRecord, FeatureTask


def publish_feature_checkpoint(
    task: FeatureTask,
    *,
    arrays: Mapping[str, np.ndarray],
    component_payloads: Sequence[Mapping[str, object]],
) -> tuple[FeatureComponentRecord, ...]:
    """Atomically publish one expert task and return its validated components."""

    npz_path = Path(task.checkpoint_npz_path)
    json_path = Path(task.checkpoint_json_path)
    if npz_path.exists() or json_path.exists():
        raise ProtocolError("Endpoint-router feature checkpoint is partially pre-existing.")
    expected_names = _expected_array_names(task)
    normalized = {
        str(name): np.ascontiguousarray(value, dtype=np.float64)
        for name, value in arrays.items()
    }
    if tuple(normalized) != expected_names:
        raise ProtocolError("Endpoint-router feature checkpoint array schema drifted.")
    atomic_save_npz(npz_path, normalized)
    digest = sha256_file(npz_path)
    records = _records_from_worker_payloads(
        task,
        component_payloads,
        relative_npz_path=_relative_checkpoint_path(task, npz_path),
        npz_sha256=digest,
    )
    unhashed = {
        "schema_version": "midogpp_endpoint_router_feature_task_checkpoint_v1",
        "task_hash": task.task_hash,
        "source_center": task.source_center,
        "training_seed": task.training_seed,
        "device": task.device,
        "relative_npz_path": _relative_checkpoint_path(task, npz_path),
        "npz_sha256": digest,
        "array_names": list(expected_names),
        "components": [record.to_payload() for record in records],
        "component_count": len(records),
        "labels_used": False,
        "evaluation_embeddings_used": False,
    }
    atomic_json(
        json_path,
        {**unhashed, "checkpoint_hash": canonical_sha256(unhashed)},
    )
    return load_feature_checkpoint(task, required=True) or ()


def load_feature_checkpoint(
    task: FeatureTask, *, required: bool = False
) -> tuple[FeatureComponentRecord, ...] | None:
    """Load and reconstruct every task member; any byte drift fails closed."""

    npz_path = Path(task.checkpoint_npz_path)
    json_path = Path(task.checkpoint_json_path)
    if not npz_path.exists() and not json_path.exists():
        if required:
            raise ProtocolError("Endpoint-router feature checkpoint is absent.")
        return None
    if not npz_path.is_file() or not json_path.is_file():
        existing = tuple(path for path in (npz_path, json_path) if path.exists())
        if (
            len(existing) == 1
            and existing[0].is_file()
            and not existing[0].is_symlink()
            and _task_paths_are_owned(task, npz_path=npz_path, json_path=json_path)
        ):
            existing[0].unlink()
            if required:
                raise ProtocolError("Endpoint-router feature checkpoint is absent.")
            return None
        raise ProtocolError("Endpoint-router feature checkpoint is incomplete.")
    raw = read_json(json_path)
    unhashed = {key: value for key, value in raw.items() if key != "checkpoint_hash"}
    component_rows = raw.get("components")
    expected_names = _expected_array_names(task)
    if (
        raw.get("schema_version")
        != "midogpp_endpoint_router_feature_task_checkpoint_v1"
        or raw.get("checkpoint_hash") != canonical_sha256(unhashed)
        or raw.get("task_hash") != task.task_hash
        or raw.get("source_center") != task.source_center
        or raw.get("training_seed") != task.training_seed
        or raw.get("device") != task.device
        or raw.get("relative_npz_path") != _relative_checkpoint_path(task, npz_path)
        or raw.get("npz_sha256") != sha256_file(npz_path)
        or raw.get("array_names") != list(expected_names)
        or raw.get("component_count") != len(task.support_slices)
        or raw.get("labels_used") is not False
        or raw.get("evaluation_embeddings_used") is not False
        or not isinstance(component_rows, list)
    ):
        raise ProtocolError("Endpoint-router feature checkpoint header drifted.")
    records = tuple(_record_from_payload(row) for row in component_rows)
    if (
        tuple(record.query_center for record in records)
        != tuple(item.query_center for item in task.support_slices)
        or any(
            record.candidate_source != task.source_center
            or record.training_seed != task.training_seed
            or record.task_hash != task.task_hash
            or record.relative_npz_path != raw["relative_npz_path"]
            or record.npz_sha256 != raw["npz_sha256"]
            for record in records
        )
    ):
        raise ProtocolError("Endpoint-router feature checkpoint component binding drifted.")
    with np.load(npz_path, allow_pickle=False) as payload:
        if tuple(payload.files) != expected_names:
            raise ProtocolError("Endpoint-router feature checkpoint NPZ schema drifted.")
        for record in records:
            expected_shape = (record.support_row_count,)
            for suffix in ("reconstruction_0", "reconstruction_1", "kl_0", "kl_1"):
                values = np.asarray(payload[f"{record.array_prefix}_{suffix}"])
                if (
                    values.shape != expected_shape
                    or values.dtype != np.float64
                    or not np.isfinite(values).all()
                    or np.any(values < 0.0)
                ):
                    raise ProtocolError(
                        "Endpoint-router feature checkpoint component array drifted."
                    )
    return records


def _task_paths_are_owned(
    task: FeatureTask, *, npz_path: Path, json_path: Path
) -> bool:
    expected_stem = f"feature_e{task.source_center}_train{task.training_seed}"
    return (
        npz_path.is_absolute()
        and json_path.is_absolute()
        and npz_path.parent == json_path.parent
        and npz_path.parent.name == "feature_runtime"
        and npz_path.name == f"{expected_stem}.npz"
        and json_path.name == f"{expected_stem}.json"
        and ".." not in npz_path.parts
        and ".." not in json_path.parts
    )


def load_component_arrays(
    root: Path, record: FeatureComponentRecord
) -> tuple[Mapping[int, np.ndarray], Mapping[int, np.ndarray]]:
    path = _safe_member(root, record.relative_npz_path)
    if not path.is_file() or sha256_file(path) != record.npz_sha256:
        raise ProtocolError("Endpoint-router feature component file drifted.")
    with np.load(path, allow_pickle=False) as payload:
        reconstruction = {
            label: np.asarray(
                payload[f"{record.array_prefix}_reconstruction_{label}"],
                dtype=np.float64,
            )
            for label in (0, 1)
        }
        kl = {
            label: np.asarray(
                payload[f"{record.array_prefix}_kl_{label}"], dtype=np.float64
            )
            for label in (0, 1)
        }
    if any(
        value.shape != (record.support_row_count,)
        or not np.isfinite(value).all()
        or np.any(value < 0.0)
        for value in (*reconstruction.values(), *kl.values())
    ):
        raise ProtocolError("Endpoint-router feature component geometry drifted.")
    return reconstruction, kl


def _records_from_worker_payloads(
    task: FeatureTask,
    payloads: Sequence[Mapping[str, object]],
    *,
    relative_npz_path: str,
    npz_sha256: str,
) -> tuple[FeatureComponentRecord, ...]:
    rows = tuple(payloads)
    if len(rows) != len(task.support_slices):
        raise ProtocolError("Endpoint-router feature worker component coverage drifted.")
    records: list[FeatureComponentRecord] = []
    for support, raw in zip(task.support_slices, rows, strict=True):
        mmd_raw = raw.get("linear_kernel_mmd2_by_generation_seed")
        if not isinstance(mmd_raw, Mapping):
            raise ProtocolError("Endpoint-router feature worker MMD payload is malformed.")
        mmd = {int(key): float(value) for key, value in mmd_raw.items()}
        provisional = {
            "schema_version": "midogpp_endpoint_router_feature_component_v1",
            "query_center": support.query_center,
            "candidate_source": task.source_center,
            "training_seed": task.training_seed,
            "relative_npz_path": relative_npz_path,
            "npz_sha256": npz_sha256,
            "array_prefix": f"q{support.query_center}",
            "support_row_count": support.row_count,
            "support_case_count": len(support.support_case_ids),
            "support_partition_hash": support.feature_support_partition_hash,
            "support_row_identity_hash": support.row_identity_hash,
            "center_partition_hash": support.center_partition_hash,
            "case_equal_energy": float(raw["case_equal_energy"]),
            "linear_kernel_mmd2_by_generation_seed": {
                str(key): float(mmd[key]) for key in sorted(mmd)
            },
            "task_hash": task.task_hash,
            "labels_used": False,
            "evaluation_embeddings_used": False,
        }
        records.append(
            FeatureComponentRecord(
                query_center=support.query_center,
                candidate_source=task.source_center,
                training_seed=task.training_seed,
                relative_npz_path=relative_npz_path,
                npz_sha256=npz_sha256,
                array_prefix=f"q{support.query_center}",
                support_row_count=support.row_count,
                support_case_count=len(support.support_case_ids),
                support_partition_hash=support.feature_support_partition_hash,
                support_row_identity_hash=support.row_identity_hash,
                center_partition_hash=support.center_partition_hash,
                case_equal_energy=float(raw["case_equal_energy"]),
                linear_kernel_mmd2_by_generation_seed=mmd,
                task_hash=task.task_hash,
                component_hash=canonical_sha256(provisional),
            )
        )
    return tuple(records)


def _record_from_payload(raw: object) -> FeatureComponentRecord:
    if not isinstance(raw, Mapping):
        raise ProtocolError("Endpoint-router feature component payload is malformed.")
    mmd_raw = raw.get("linear_kernel_mmd2_by_generation_seed")
    if not isinstance(mmd_raw, Mapping):
        raise ProtocolError("Endpoint-router feature component MMD payload is malformed.")
    return FeatureComponentRecord(
        query_center=str(raw["query_center"]),
        candidate_source=str(raw["candidate_source"]),
        training_seed=int(raw["training_seed"]),
        relative_npz_path=str(raw["relative_npz_path"]),
        npz_sha256=str(raw["npz_sha256"]),
        array_prefix=str(raw["array_prefix"]),
        support_row_count=int(raw["support_row_count"]),
        support_case_count=int(raw["support_case_count"]),
        support_partition_hash=str(raw["support_partition_hash"]),
        support_row_identity_hash=str(raw["support_row_identity_hash"]),
        center_partition_hash=str(raw["center_partition_hash"]),
        case_equal_energy=float(raw["case_equal_energy"]),
        linear_kernel_mmd2_by_generation_seed={
            int(key): float(value) for key, value in mmd_raw.items()
        },
        task_hash=str(raw["task_hash"]),
        component_hash=str(raw["component_hash"]),
    )


def _expected_array_names(task: FeatureTask) -> tuple[str, ...]:
    return tuple(
        f"q{support.query_center}_{suffix}"
        for support in task.support_slices
        for suffix in ("reconstruction_0", "reconstruction_1", "kl_0", "kl_1")
    )


def _relative_checkpoint_path(task: FeatureTask, path: Path) -> str:
    try:
        return path.resolve().relative_to(Path(task.support_root).resolve()).as_posix()
    except ValueError as exc:
        raise ProtocolError("Endpoint-router feature checkpoint escaped its root.") from exc


def _safe_member(root: Path, relative: str) -> Path:
    member = (root.resolve() / relative).resolve()
    try:
        member.relative_to(root.resolve())
    except ValueError as exc:
        raise ProtocolError("Endpoint-router feature component escaped its root.") from exc
    return member


__all__ = (
    "load_component_arrays",
    "load_feature_checkpoint",
    "publish_feature_checkpoint",
)
