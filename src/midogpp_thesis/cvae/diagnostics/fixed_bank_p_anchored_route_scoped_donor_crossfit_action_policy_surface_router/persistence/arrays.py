"""Tamper-evident dense array storage without pickle."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from ....protocol import ProtocolError
from ....runtime.artifact_io import atomic_json, atomic_npz, read_json, sha256_array, sha256_file
from ..identity import canonical_hash, require_sha256
from .safety import reject_forbidden_persisted_values


def persist_dense_arrays(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    *,
    schema_version: str,
    lineage_hashes: Mapping[str, str],
) -> dict[str, object]:
    target = Path(path)
    metadata_path = target.with_suffix(target.suffix + ".manifest.json")
    canonical = {
        str(key): np.ascontiguousarray(value)
        for key, value in sorted(arrays.items())
    }
    if not canonical or any(value.dtype == object for value in canonical.values()):
        raise ProtocolError("P-DCAPS dense array store is empty or object-backed.")
    reject_forbidden_persisted_values(lineage_hashes)
    for role, digest in lineage_hashes.items():
        require_sha256(digest, f"dense-array {role} lineage hash")
    manifest_base = {
        "schema_version": str(schema_version),
        "arrays": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": sha256_array(value),
            }
            for key, value in canonical.items()
        },
        "lineage_hashes": dict(sorted(lineage_hashes.items())),
        "allow_pickle": False,
    }
    if target.exists() or metadata_path.exists():
        if not target.is_file() or not metadata_path.is_file():
            raise ProtocolError("P-DCAPS dense array store is an unsafe partial state.")
        observed = load_dense_arrays(target)
        expected_base = {**manifest_base, "file_sha256": sha256_file(target)}
        expected_manifest = {
            **expected_base,
            "manifest_hash": canonical_hash(expected_base),
        }
        if (
            set(observed[0]) != set(canonical)
            or any(
                not np.array_equal(observed[0][key], canonical[key])
                for key in canonical
            )
            or observed[1] != expected_manifest
        ):
            raise ProtocolError("P-DCAPS refuses to repair a different dense store.")
        return expected_manifest
    atomic_npz(target, **canonical)
    manifest_base_with_file = {**manifest_base, "file_sha256": sha256_file(target)}
    manifest = {
        **manifest_base_with_file,
        "manifest_hash": canonical_hash(manifest_base_with_file),
    }
    atomic_json(metadata_path, manifest)
    return manifest


def load_dense_arrays(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    target = Path(path)
    manifest = read_json(target.with_suffix(target.suffix + ".manifest.json"))
    if manifest.get("file_sha256") != sha256_file(target):
        raise ProtocolError("P-DCAPS dense store bytes drifted.")
    unhashed = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if manifest.get("manifest_hash") != canonical_hash(unhashed):
        raise ProtocolError("P-DCAPS dense store manifest hash drifted.")
    try:
        with np.load(target, allow_pickle=False) as store:
            arrays = {key: np.ascontiguousarray(store[key]) for key in store.files}
    except (OSError, ValueError) as exc:
        raise ProtocolError("Cannot load P-DCAPS dense store.") from exc
    expected = manifest.get("arrays")
    if not isinstance(expected, dict) or set(arrays) != set(expected):
        raise ProtocolError("P-DCAPS dense store inventory drifted.")
    for key, value in arrays.items():
        row = expected[key]
        if (
            not isinstance(row, dict)
            or row.get("shape") != list(value.shape)
            or row.get("dtype") != str(value.dtype)
            or row.get("sha256") != sha256_array(value)
        ):
            raise ProtocolError("P-DCAPS dense array content drifted.")
    return arrays, manifest


__all__ = ("load_dense_arrays", "persist_dense_arrays")
