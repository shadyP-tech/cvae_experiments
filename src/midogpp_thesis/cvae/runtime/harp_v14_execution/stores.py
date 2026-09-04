"""Deterministic compact NPZ stores with canonical manifests and chunk hashes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
from types import MappingProxyType
import zipfile

import numpy as np

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_bytes, canonical_hash
from ..artifact_io import atomic_json, read_json, sha256_file
from .contracts import (
    ActionKind,
    ArtifactValue,
    LabelFreeActionBlock,
    LabelFreeOuterMenu,
    PrelabelRouteSet,
    RoutedCase,
)


_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True, slots=True)
class CompactStoreReceipt:
    root: Path
    manifest_path: Path
    npz_path: Path
    manifest_hash: str
    manifest_sha256: str
    npz_sha256: str
    chunk_hashes: Mapping[str, str]


def _canonical_array(value: object, *, name: str) -> np.ndarray:
    if type(name) is not str or not name or name.strip() != name or "/" in name:
        raise ProtocolError("HARP v14 compact array name is unsafe.")
    raw = np.asarray(value)
    if raw.dtype.hasobject or not np.isfinite(raw).all():
        raise ProtocolError("HARP v14 compact store rejects object/nonfinite arrays.")
    if raw.dtype.byteorder == ">":
        raw = raw.byteswap().view(raw.dtype.newbyteorder("<"))
    elif raw.dtype.byteorder == "=" and not np.little_endian:
        raw = raw.byteswap().view(raw.dtype.newbyteorder("<"))
    return np.ascontiguousarray(raw)


def _npy_bytes(values: np.ndarray) -> bytes:
    handle = io.BytesIO()
    np.lib.format.write_array(handle, values, allow_pickle=False)
    return handle.getvalue()


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    chunks: dict[str, str] = {}
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for name in sorted(arrays):
            values = _canonical_array(arrays[name], name=name)
            payload = _npy_bytes(values)
            chunks[name] = hashlib.sha256(payload).hexdigest()
            info = zipfile.ZipInfo(filename=f"{name}.npy", date_time=_FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            info.create_system = 3
            archive.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    os.replace(temporary, path)
    return chunks


def _read_arrays(path: Path, *, expected: Mapping[str, str]) -> dict[str, np.ndarray]:
    if not path.is_file() or path.is_symlink():
        raise ProtocolError("HARP v14 compact NPZ store is absent or unsafe.")
    observed: dict[str, str] = {}
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        if names != [f"{name}.npy" for name in sorted(expected)]:
            raise ProtocolError("HARP v14 compact NPZ member inventory drifted.")
        for member in names:
            key = member[:-4]
            payload = archive.read(member)
            observed[key] = hashlib.sha256(payload).hexdigest()
    if observed != dict(expected):
        raise ProtocolError("HARP v14 compact NPZ chunk identity drifted.")
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {
                name: np.ascontiguousarray(archive[name]) for name in sorted(expected)
            }
    except (OSError, ValueError, KeyError) as exc:
        raise ProtocolError("HARP v14 compact NPZ could not be loaded.") from exc
    if any(values.dtype.hasobject or not np.isfinite(values).all() for values in arrays.values()):
        raise ProtocolError("HARP v14 compact NPZ contains invalid values.")
    return arrays


def _write_store(
    root: Path,
    *,
    schema_version: str,
    payload: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
) -> CompactStoreReceipt:
    root.mkdir(parents=True, exist_ok=True)
    npz_path = root / "arrays.npz"
    chunk_hashes = _write_deterministic_npz(npz_path, arrays)
    body = {
        "schema_version": schema_version,
        **dict(payload),
        "npz_member": "arrays.npz",
        "npz_sha256": sha256_file(npz_path),
        "chunk_hashes": dict(sorted(chunk_hashes.items())),
        "array_names": sorted(arrays),
        "compact_npz": True,
        "pickle_allowed": False,
    }
    manifest = {**body, "manifest_hash": canonical_hash(body)}
    manifest_path = root / "manifest.json"
    atomic_json(manifest_path, manifest)
    # JSON has one canonical sequence representation: an in-memory tuple is
    # restored as a list.  Compare canonical bytes rather than Python container
    # types so the durability check validates semantic JSON identity while
    # still failing on any value, key, ordering, or hash drift.
    if canonical_bytes(read_json(manifest_path)) != canonical_bytes(manifest):
        raise ProtocolError("HARP v14 compact manifest failed a durable round trip.")
    return CompactStoreReceipt(
        root=root,
        manifest_path=manifest_path,
        npz_path=npz_path,
        manifest_hash=str(manifest["manifest_hash"]),
        manifest_sha256=sha256_file(manifest_path),
        npz_sha256=str(body["npz_sha256"]),
        chunk_hashes=MappingProxyType(dict(chunk_hashes)),
    )


def _read_store(root: Path, *, schema_version: str) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    root = Path(root)
    manifest_path = root / "manifest.json"
    if (
        not root.is_dir()
        or root.is_symlink()
        or not manifest_path.is_file()
        or manifest_path.is_symlink()
    ):
        raise ProtocolError("HARP v14 compact store root or manifest is unsafe.")
    manifest = read_json(manifest_path)
    stored_hash = manifest.get("manifest_hash")
    body = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if manifest.get("schema_version") != schema_version or stored_hash != canonical_hash(body):
        raise ProtocolError("HARP v14 compact store manifest identity drifted.")
    expected_chunks = manifest.get("chunk_hashes")
    if not isinstance(expected_chunks, Mapping) or any(
        type(key) is not str or type(value) is not str
        for key, value in expected_chunks.items()
    ):
        raise ProtocolError("HARP v14 compact store chunk manifest is malformed.")
    # The member is deliberately not interpreted as a path.  Exact equality
    # prevents an otherwise correctly re-hashed manifest from escaping the
    # store root through an absolute path or ``..`` traversal.
    if manifest.get("npz_member") != "arrays.npz":
        raise ProtocolError("HARP v14 compact store NPZ member binding drifted.")
    npz_path = root / "arrays.npz"
    if not npz_path.is_file() or npz_path.is_symlink():
        raise ProtocolError("HARP v14 compact NPZ store is absent or unsafe.")
    if sha256_file(npz_path) != manifest.get("npz_sha256"):
        raise ProtocolError("HARP v14 compact NPZ file identity drifted.")
    arrays = _read_arrays(npz_path, expected=expected_chunks)  # type: ignore[arg-type]
    return manifest, arrays


def write_label_free_outer_menu(root: Path, menu: LabelFreeOuterMenu) -> CompactStoreReceipt:
    arrays = {
        name: values
        for index, block in enumerate(menu.blocks)
        for name, values in (
            (f"p_{index:03d}", block.probabilities),
            (f"d_{index:03d}", block.seed_dispersion),
        )
    }
    blocks = [
        {
            "surface_role": block.surface_role,
            "outer_target_id": block.outer_target_id,
            "query_center_id": block.query_center_id,
            "action_kind": block.action_kind.value,
            "selected_source_id": block.selected_source_id,
            "sample_ids": list(block.sample_ids),
            "case_ids": list(block.case_ids),
            "array_name": f"p_{index:03d}",
            "dispersion_array_name": f"d_{index:03d}",
            "block_hash": block.block_hash,
        }
        for index, block in enumerate(menu.blocks)
    ]
    return _write_store(
        root,
        schema_version="midogpp_harp_v14_outer_menu_compact_store_v2",
        payload={
            "outer_target_id": menu.outer_target_id,
            "menu_hash": menu.menu_hash,
            "blocks": blocks,
            "lineage": dict(menu.lineage),
            "labels_consumed": False,
            "physical_expert_weight": 1.0,
            "exact_nine_seed_dispersion_persisted": True,
        },
        arrays=arrays,
    )


def read_label_free_outer_menu(root: Path) -> LabelFreeOuterMenu:
    manifest, arrays = _read_store(
        root, schema_version="midogpp_harp_v14_outer_menu_compact_store_v2"
    )
    raw_blocks = manifest.get("blocks")
    if not isinstance(raw_blocks, list):
        raise ProtocolError("HARP v14 outer-menu block manifest is absent.")
    blocks: list[LabelFreeActionBlock] = []
    for raw in raw_blocks:
        if not isinstance(raw, Mapping):
            raise ProtocolError("HARP v14 outer-menu block manifest is malformed.")
        array_name = raw.get("array_name")
        dispersion_name = raw.get("dispersion_array_name")
        if (
            type(array_name) is not str
            or array_name not in arrays
            or type(dispersion_name) is not str
            or dispersion_name not in arrays
        ):
            raise ProtocolError("HARP v14 outer-menu array binding is absent.")
        block = LabelFreeActionBlock(
            surface_role=str(raw.get("surface_role")),
            outer_target_id=str(raw.get("outer_target_id")),
            query_center_id=str(raw.get("query_center_id")),
            action_kind=ActionKind(str(raw.get("action_kind"))),
            selected_source_id=(
                None if raw.get("selected_source_id") is None else str(raw["selected_source_id"])
            ),
            sample_ids=tuple(str(value) for value in raw.get("sample_ids", ())),
            case_ids=tuple(str(value) for value in raw.get("case_ids", ())),
            probabilities=np.asarray(arrays[array_name], dtype=np.float32),
            seed_dispersion=np.asarray(arrays[dispersion_name], dtype=np.float32),
        )
        if block.block_hash != raw.get("block_hash"):
            raise ProtocolError("HARP v14 outer-menu block identity drifted.")
        blocks.append(block)
    menu = LabelFreeOuterMenu(
        outer_target_id=str(manifest.get("outer_target_id")),
        blocks=tuple(blocks),
        lineage=dict(manifest.get("lineage", {})),
    )
    if menu.menu_hash != manifest.get("menu_hash"):
        raise ProtocolError("HARP v14 outer-menu identity drifted.")
    return menu


def write_artifact_value(root: Path, value: ArtifactValue, *, role: str) -> CompactStoreReceipt:
    return _write_store(
        root,
        schema_version="midogpp_harp_v14_opaque_artifact_store_v1",
        payload={
            "artifact_role": role,
            "scientific_manifest": dict(value.manifest),
        },
        arrays=value.arrays,
    )


def read_artifact_value(root: Path, *, role: str) -> ArtifactValue:
    manifest, arrays = _read_store(
        root, schema_version="midogpp_harp_v14_opaque_artifact_store_v1"
    )
    if manifest.get("artifact_role") != role or not isinstance(
        manifest.get("scientific_manifest"), Mapping
    ):
        raise ProtocolError("HARP v14 opaque artifact role drifted.")
    return ArtifactValue(state=None, manifest=manifest["scientific_manifest"], arrays=arrays)


def write_prelabel_routes(root: Path, routes: PrelabelRouteSet) -> CompactStoreReceipt:
    offsets = [0]
    baseline: list[np.ndarray] = []
    uniform: list[np.ndarray] = []
    selected: list[np.ndarray] = []
    routed: list[np.ndarray] = []
    components: list[np.ndarray] = []
    component_offsets = [0]
    case_component_offsets = [0]
    cases = []
    for case in routes.cases:
        baseline.append(case.baseline_probabilities)
        uniform.append(case.uniform_probabilities)
        selected.append(case.selected_probabilities)
        routed.append(case.routed_probabilities)
        for component in case.component_probabilities:
            components.append(component)
            component_offsets.append(component_offsets[-1] + len(component))
        case_component_offsets.append(len(components))
        offsets.append(offsets[-1] + len(case.sample_ids))
        cases.append(
            {
                "outer_target_id": case.outer_target_id,
                "case_id": case.case_id,
                "sample_ids": list(case.sample_ids),
                "selected_kind": case.selected_kind.value,
                "selected_source_id": case.selected_source_id,
                "direction": case.direction,
                "shrinkage": case.shrinkage,
                "component_action_ids": list(case.component_action_ids),
                "component_weights": list(case.component_weights),
                "reason": case.reason,
                "decision_payload": dict(case.decision_payload),
                "decision_hash": case.decision_hash,
            }
        )
    arrays = {
        "case_offsets": np.asarray(offsets, dtype=np.int64),
        "baseline": np.ascontiguousarray(np.concatenate(baseline), dtype=np.float32),
        "uniform": np.ascontiguousarray(np.concatenate(uniform), dtype=np.float32),
        "selected": np.ascontiguousarray(np.concatenate(selected), dtype=np.float32),
        "routed": np.ascontiguousarray(np.concatenate(routed), dtype=np.float32),
        "component_offsets": np.asarray(component_offsets, dtype=np.int64),
        "case_component_offsets": np.asarray(case_component_offsets, dtype=np.int64),
        "components": (
            np.ascontiguousarray(np.concatenate(components), dtype=np.float32)
            if components
            else np.empty((0,), dtype=np.float32)
        ),
    }
    return _write_store(
        root,
        schema_version="midogpp_harp_v14_prelabel_route_compact_store_v1",
        payload={
            "route_hash": routes.route_hash,
            "ordered_case_identity_hash": routes.ordered_case_identity_hash,
            "ordered_sample_identity_hash": routes.ordered_sample_identity_hash,
            "policy_hash": routes.policy_hash,
            "model_hash": routes.model_hash,
            "target_action_hash": routes.target_action_hash,
            "cases": cases,
            "case_count": len(cases),
            "row_count": offsets[-1],
            "case_consistent": True,
            "evaluation_labels_opened": False,
        },
        arrays=arrays,
    )


def read_prelabel_routes(root: Path) -> PrelabelRouteSet:
    manifest, arrays = _read_store(
        root, schema_version="midogpp_harp_v14_prelabel_route_compact_store_v1"
    )
    expected_arrays = {
        "case_offsets",
        "baseline",
        "uniform",
        "selected",
        "routed",
        "component_offsets",
        "case_component_offsets",
        "components",
    }
    if set(arrays) != expected_arrays:
        raise ProtocolError("HARP v14 route array inventory drifted.")
    offsets = np.asarray(arrays["case_offsets"])
    component_offsets = np.asarray(arrays["component_offsets"])
    case_component_offsets = np.asarray(arrays["case_component_offsets"])
    raw_cases = manifest.get("cases")
    if (
        offsets.dtype != np.dtype("int64")
        or offsets.ndim != 1
        or not isinstance(raw_cases, list)
        or len(offsets) != len(raw_cases) + 1
        or offsets[0] != 0
        or np.any(np.diff(offsets) <= 0)
        or component_offsets.dtype != np.dtype("int64")
        or component_offsets.ndim != 1
        or not len(component_offsets)
        or component_offsets[0] != 0
        or np.any(np.diff(component_offsets) <= 0)
        or case_component_offsets.dtype != np.dtype("int64")
        or case_component_offsets.shape != (len(raw_cases) + 1,)
        or case_component_offsets[0] != 0
        or np.any(np.diff(case_component_offsets) < 0)
        or case_component_offsets[-1] != len(component_offsets) - 1
    ):
        raise ProtocolError("HARP v14 route case offsets drifted.")
    row_count = int(offsets[-1])
    if (
        type(manifest.get("row_count")) is not int
        or manifest.get("row_count") != row_count
        or type(manifest.get("case_count")) is not int
        or manifest.get("case_count") != len(raw_cases)
    ):
        raise ProtocolError("HARP v14 route manifest dimensions drifted.")
    for name in ("baseline", "uniform", "selected", "routed"):
        values = arrays[name]
        if values.dtype != np.dtype("float32") or values.ndim != 1 or len(values) != row_count:
            raise ProtocolError(
                "HARP v14 route probability arrays do not end at the final case offset."
            )
    component_values = arrays["components"]
    if (
        component_values.dtype != np.dtype("float32")
        or component_values.ndim != 1
        or len(component_values) != int(component_offsets[-1])
    ):
        raise ProtocolError("HARP v14 route component array geometry drifted.")
    cases: list[RoutedCase] = []
    for ordinal, raw in enumerate(raw_cases):
        if not isinstance(raw, Mapping):
            raise ProtocolError("HARP v14 route case manifest is malformed.")
        start, stop = int(offsets[ordinal]), int(offsets[ordinal + 1])
        raw_sample_ids = raw.get("sample_ids")
        if not isinstance(raw_sample_ids, list) or len(raw_sample_ids) != stop - start:
            raise ProtocolError("HARP v14 route case identities and offsets are misaligned.")
        component_start = int(case_component_offsets[ordinal])
        component_stop = int(case_component_offsets[ordinal + 1])
        component_rows = tuple(
            np.asarray(
                component_values[
                    int(component_offsets[index]) : int(component_offsets[index + 1])
                ],
                dtype=np.float32,
            )
            for index in range(component_start, component_stop)
        )
        raw_action_ids = raw.get("component_action_ids")
        raw_weights = raw.get("component_weights")
        if (
            not isinstance(raw_action_ids, list)
            or not isinstance(raw_weights, list)
            or len(raw_action_ids) != len(component_rows)
            or len(raw_weights) != len(component_rows)
            or any(len(value) != stop - start for value in component_rows)
        ):
            raise ProtocolError("HARP v14 route component manifest is misaligned.")
        case = RoutedCase(
            outer_target_id=str(raw.get("outer_target_id")),
            case_id=str(raw.get("case_id")),
            sample_ids=tuple(str(value) for value in raw_sample_ids),
            selected_kind=ActionKind(str(raw.get("selected_kind"))),
            selected_source_id=(
                None if raw.get("selected_source_id") is None else str(raw["selected_source_id"])
            ),
            reason=str(raw.get("reason")),
            baseline_probabilities=np.asarray(arrays["baseline"][start:stop], dtype=np.float32),
            uniform_probabilities=np.asarray(arrays["uniform"][start:stop], dtype=np.float32),
            selected_probabilities=np.asarray(arrays["selected"][start:stop], dtype=np.float32),
            routed_probabilities=np.asarray(arrays["routed"][start:stop], dtype=np.float32),
            direction=(
                None if raw.get("direction") is None else str(raw.get("direction"))
            ),
            shrinkage=float(raw.get("shrinkage", 0.0)),
            component_action_ids=tuple(str(value) for value in raw_action_ids),
            component_weights=tuple(float(value) for value in raw_weights),
            component_probabilities=component_rows,
            decision_payload=dict(raw.get("decision_payload", {})),
        )
        if case.decision_hash != raw.get("decision_hash"):
            raise ProtocolError("HARP v14 route decision identity drifted.")
        cases.append(case)
    routes = PrelabelRouteSet(
        cases=tuple(cases),
        policy_hash=str(manifest.get("policy_hash")),
        model_hash=str(manifest.get("model_hash")),
        target_action_hash=str(manifest.get("target_action_hash")),
    )
    if routes.route_hash != manifest.get("route_hash"):
        raise ProtocolError("HARP v14 prelabel route identity drifted.")
    if (
        routes.ordered_case_identity_hash
        != manifest.get("ordered_case_identity_hash")
        or routes.ordered_sample_identity_hash
        != manifest.get("ordered_sample_identity_hash")
    ):
        raise ProtocolError("HARP v14 prelabel target identity inventory drifted.")
    return routes


__all__ = (
    "CompactStoreReceipt",
    "read_artifact_value",
    "read_label_free_outer_menu",
    "read_prelabel_routes",
    "write_artifact_value",
    "write_label_free_outer_menu",
    "write_prelabel_routes",
)
