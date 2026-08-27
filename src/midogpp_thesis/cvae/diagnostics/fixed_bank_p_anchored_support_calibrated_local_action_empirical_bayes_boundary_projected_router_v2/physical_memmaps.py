"""Pack the 810 label-free probability cells into read-only worker maps."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .artifacts.io import atomic_json
from .execution import MemmapReference
from .hashing import canonical_hash
from .identity import (
    CENTERS,
    EXPECTED_PHYSICAL_CELL_COUNT,
    EXPECTED_TEST_CACHE_CONTENT_HASH,
    EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
    GovernanceError,
)
from .physical import ExactNineActionView
from .physical.contracts import (
    GENERATION_SEEDS,
    TRAINING_SEEDS,
    array_sha256,
    physical_action_ids,
)


INDEX_SCHEMA = "scale_bp_v2_physical_memmap_index_v1"
_ROLE_PREFIX = "physical_probability_center_"


@dataclass(frozen=True, slots=True)
class PhysicalMemmapBundle:
    index_path: Path
    references: tuple[MemmapReference, ...]
    index_hash: str
    bundle_hash: str = field(init=False)

    def __post_init__(self) -> None:
        references = tuple(self.references)
        if (
            not self.index_path.is_absolute()
            or len(references) != len(CENTERS)
            or tuple(row.semantic_role for row in references)
            != tuple(f"{_ROLE_PREFIX}{center}" for center in CENTERS)
            or not self.index_hash
        ):
            raise GovernanceError("SCALE-BP v2 physical memmap bundle drifted.")
        object.__setattr__(self, "references", references)
        object.__setattr__(
            self,
            "bundle_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_v2_physical_memmap_bundle_v1",
                    "index_path": str(self.index_path),
                    "index_hash": self.index_hash,
                    "reference_hashes": tuple(row.reference_hash for row in references),
                    "physical_cell_count": EXPECTED_PHYSICAL_CELL_COUNT,
                    "labels_used": False,
                }
            ),
        )


@dataclass(frozen=True)
class MappedPhysicalStore:
    """Duck-compatible physical adapter backed only by worker read maps."""

    arrays: Mapping[str, np.memmap]
    rows_by_center: Mapping[str, tuple[str, ...]]
    case_ids_by_center: Mapping[str, tuple[str, ...]]
    index_hash: str
    adapter_hash: str = field(init=False)

    def __post_init__(self) -> None:
        arrays = {str(key): value for key, value in self.arrays.items()}
        rows = {str(key): tuple(str(item) for item in value) for key, value in self.rows_by_center.items()}
        cases = {
            str(key): tuple(str(item) for item in value)
            for key, value in self.case_ids_by_center.items()
        }
        for center in CENTERS:
            role = f"{_ROLE_PREFIX}{center}"
            expected_shape = (
                len(physical_action_ids(center)),
                len(TRAINING_SEEDS),
                len(GENERATION_SEEDS),
                len(rows.get(center, ())),
            )
            values = arrays.get(role)
            if (
                not isinstance(values, np.memmap)
                or values.dtype != np.dtype("float32")
                or values.shape != expected_shape
                or values.flags.writeable
                or len(rows[center]) != len(cases[center])
                or len(set(rows[center])) != len(rows[center])
            ):
                raise GovernanceError("SCALE-BP v2 mapped physical store drifted.")
        object.__setattr__(self, "arrays", MappingProxyType(arrays))
        object.__setattr__(self, "rows_by_center", MappingProxyType(rows))
        object.__setattr__(self, "case_ids_by_center", MappingProxyType(cases))
        object.__setattr__(
            self,
            "adapter_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_v2_mapped_physical_adapter_v1",
                    "index_hash": self.index_hash,
                    "centers": CENTERS,
                    "physical_cell_count": EXPECTED_PHYSICAL_CELL_COUNT,
                    "read_only": True,
                    "labels_used": False,
                }
            ),
        )

    @property
    def case_indices_by_center(self) -> Mapping[str, Mapping[str, tuple[int, ...]]]:
        result: dict[str, Mapping[str, tuple[int, ...]]] = {}
        for center in CENTERS:
            grouped: dict[str, list[int]] = {}
            for index, case_id in enumerate(self.case_ids_by_center[center]):
                grouped.setdefault(case_id, []).append(index)
            result[center] = MappingProxyType(
                {case: tuple(indices) for case, indices in grouped.items()}
            )
        return MappingProxyType(result)

    def case_ids(self, target_center: object) -> tuple[str, ...]:
        target = str(target_center)
        try:
            return tuple(self.case_indices_by_center[target])
        except KeyError as exc:
            raise GovernanceError("SCALE-BP v2 mapped target center is absent.") from exc

    def case_indices(self, target_center: object, case_id: object) -> tuple[int, ...]:
        try:
            return self.case_indices_by_center[str(target_center)][str(case_id)]
        except KeyError as exc:
            raise GovernanceError("SCALE-BP v2 mapped case is absent.") from exc

    def exact_nine_view(
        self,
        target_center: object,
        action_id: object,
        *,
        case_id: object | None = None,
    ) -> ExactNineActionView:
        target, action = str(target_center), str(action_id)
        actions = physical_action_ids(target)
        if action not in actions:
            raise GovernanceError("SCALE-BP v2 mapped action is absent.")
        indices = (
            tuple(range(len(self.rows_by_center[target])))
            if case_id is None
            else self.case_indices(target, case_id)
        )
        raw = self.arrays[f"{_ROLE_PREFIX}{target}"][actions.index(action)]
        seeds = np.ascontiguousarray(
            raw[:, :, np.asarray(indices, dtype=np.int64)].reshape(
                len(TRAINING_SEEDS) * len(GENERATION_SEEDS), len(indices)
            ),
            dtype=np.float64,
        )
        mean = np.mean(seeds, axis=0, dtype=np.float64)
        standard_deviation = np.std(seeds, axis=0, ddof=0, dtype=np.float64)
        votes = np.mean(seeds >= 0.5, axis=0, dtype=np.float64)
        sample_ids = tuple(self.rows_by_center[target][index] for index in indices)
        case_ids = tuple(self.case_ids_by_center[target][index] for index in indices)
        payload = {
            "schema_version": "scale_bp_v2_exact_nine_action_view_v1",
            "target_center": target,
            "action_id": action,
            "sample_ids": sample_ids,
            "case_ids": case_ids,
            "seed_probability_sha256": array_sha256(seeds),
            "mean_probability_sha256": array_sha256(mean),
            "seed_standard_deviation_sha256": array_sha256(standard_deviation),
            "positive_vote_fraction_sha256": array_sha256(votes),
            "label_free": True,
        }
        return ExactNineActionView(
            target,
            action,
            sample_ids,
            case_ids,
            seeds,
            mean,
            standard_deviation,
            votes,
            canonical_hash(payload),
        )


def persist_physical_memmaps(
    store: object,
    *,
    root: str | Path,
) -> PhysicalMemmapBundle:
    """Write each center's exact 90 cells once as one atomic float32 map."""

    adapter = store
    neutral = getattr(adapter, "store", None)
    if neutral is None or len(tuple(getattr(neutral, "cells", ()))) != EXPECTED_PHYSICAL_CELL_COUNT:
        raise GovernanceError("SCALE-BP v2 cannot pack a foreign physical store.")
    output = Path(root).resolve(strict=False)
    if output.exists() or output.is_symlink():
        raise GovernanceError("SCALE-BP v2 physical memmap root already exists.")
    output.mkdir(parents=True, exist_ok=False)
    references: list[MemmapReference] = []
    center_rows: dict[str, dict[str, object]] = {}
    for center in CENTERS:
        actions = physical_action_ids(center)
        packed = np.stack(
            [
                np.asarray(
                    neutral.probabilities(center, action, training, generation),
                    dtype=np.float32,
                )
                for action in actions
                for training in TRAINING_SEEDS
                for generation in GENERATION_SEEDS
            ],
            axis=0,
        ).reshape(
            len(actions),
            len(TRAINING_SEEDS),
            len(GENERATION_SEEDS),
            len(neutral.rows_by_center[center]),
        )
        packed = np.ascontiguousarray(packed, dtype=np.float32)
        path = output / f"center_{center}.float32.bin"
        _atomic_array_bytes(path, packed)
        digest = _sha256_file(path)
        row_hash = canonical_hash(
            {
                "schema_version": "scale_bp_v2_physical_center_rows_v1",
                "center": center,
                "sample_ids": tuple(neutral.rows_by_center[center]),
                "case_ids": tuple(neutral.case_ids_by_center[center]),
            }
        )
        reference = MemmapReference(
            path=str(path),
            dtype="float32",
            shape=tuple(int(value) for value in packed.shape),
            offset_bytes=0,
            byte_length=int(packed.nbytes),
            sha256=digest,
            semantic_role=f"{_ROLE_PREFIX}{center}",
            row_index_hash=row_hash,
            cache_content_hash=EXPECTED_TEST_CACHE_CONTENT_HASH,
            row_order_hash=EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
        )
        references.append(reference)
        center_rows[center] = {
            "actions": list(actions),
            "sample_ids": list(neutral.rows_by_center[center]),
            "case_ids": list(neutral.case_ids_by_center[center]),
            "reference": reference.to_payload(),
        }
    body = {
        "schema_version": INDEX_SCHEMA,
        "centers": list(CENTERS),
        "training_seeds": list(TRAINING_SEEDS),
        "generation_seeds": list(GENERATION_SEEDS),
        "physical_cell_count": EXPECTED_PHYSICAL_CELL_COUNT,
        "cache_content_hash": EXPECTED_TEST_CACHE_CONTENT_HASH,
        "row_order_hash": EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
        "center_rows": center_rows,
        "read_only_worker_maps": True,
        "labels_used": False,
    }
    payload = {**body, "index_hash": canonical_hash(body)}
    index_path = output / "physical_memmap_index.json"
    atomic_json(index_path, payload)
    return PhysicalMemmapBundle(
        index_path.resolve(strict=True), tuple(references), str(payload["index_hash"])
    )


def open_mapped_physical_store(
    arrays: Mapping[str, np.memmap],
    *,
    index_path: str | Path,
    expected_index_hash: object,
) -> MappedPhysicalStore:
    path = Path(index_path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise GovernanceError("SCALE-BP v2 physical memmap index is unsafe.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GovernanceError("SCALE-BP v2 physical memmap index is unreadable.") from exc
    if not isinstance(payload, dict):
        raise GovernanceError("SCALE-BP v2 physical memmap index is malformed.")
    body = {key: value for key, value in payload.items() if key != "index_hash"}
    index_hash = canonical_hash(body)
    center_rows = payload.get("center_rows")
    if (
        payload.get("schema_version") != INDEX_SCHEMA
        or payload.get("index_hash") != index_hash
        or index_hash != str(expected_index_hash)
        or payload.get("physical_cell_count") != EXPECTED_PHYSICAL_CELL_COUNT
        or not isinstance(center_rows, dict)
    ):
        raise GovernanceError("SCALE-BP v2 physical memmap index drifted.")
    rows = {
        center: tuple(str(value) for value in center_rows[center]["sample_ids"])
        for center in CENTERS
    }
    cases = {
        center: tuple(str(value) for value in center_rows[center]["case_ids"])
        for center in CENTERS
    }
    return MappedPhysicalStore(arrays, rows, cases, index_hash)


def _atomic_array_bytes(path: Path, values: np.ndarray) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(memoryview(values).cast("B"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = (
    "MappedPhysicalStore",
    "PhysicalMemmapBundle",
    "open_mapped_physical_store",
    "persist_physical_memmaps",
)
