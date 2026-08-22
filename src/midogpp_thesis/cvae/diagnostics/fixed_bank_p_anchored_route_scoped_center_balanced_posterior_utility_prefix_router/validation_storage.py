"""Shared persisted-table and dense-array structural validation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json, sha256_array, sha256_file
from .hashing import canonical_hash


def load_table(root: Path, name: str) -> list[Mapping[str, object]]:
    payload = read_json(root / "tables" / f"{name}.json")
    rows = payload.get("rows")
    if not isinstance(rows, list) or payload.get("row_count") != len(rows) or any(
        not isinstance(row, Mapping) for row in rows
    ):
        raise ProtocolError(f"CBPUPR table drifted: {name}.")
    return rows


def validate_npz_manifest(
    root: Path, manifest_name: str, array_name: str
) -> None:
    manifest = read_json(root / "manifests" / f"{manifest_name}.json")
    npz_path = root / "arrays" / f"{array_name}.npz"
    rows = manifest.get("arrays")
    unhashed = {
        key: value for key, value in manifest.items() if key != "manifest_hash"
    }
    expected_roles = {
        "route_endpoint_probabilities": "route_endpoint_probabilities",
        "pseudo_route_endpoint_probabilities": (
            "pseudo_route_endpoint_probabilities"
        ),
        "target_local_posterior_probabilities": (
            "target_local_posterior_probabilities"
        ),
        "candidate_probabilities": "candidate_probabilities",
        "composed_probabilities": "endpoint_and_composed_probabilities",
    }
    if (
        manifest.get("schema_version")
        != "fixed_bank_cbpupr_dense_array_manifest_v1"
        or manifest.get("role") != expected_roles[array_name]
        or manifest.get("member") != npz_path.name
        or manifest.get("store_sha256") != sha256_file(npz_path)
        or not isinstance(rows, list)
        or any(not isinstance(row, Mapping) for row in rows)
        or len({str(row.get("key")) for row in rows}) != len(rows)
        or manifest.get("raw_labels_persisted") is not False
        or manifest.get("sample_or_image_paths_persisted") is not False
        or manifest.get("manifest_hash") != canonical_hash(unhashed)
    ):
        raise ProtocolError("CBPUPR dense-array manifest drifted.")
    with np.load(npz_path, allow_pickle=False) as store:
        if tuple(store.files) != tuple(row.get("key") for row in rows):
            raise ProtocolError("CBPUPR dense-array key order drifted.")
        for row in rows:
            array = np.asarray(store[str(row["key"])])
            if (
                list(array.shape) != row.get("shape")
                or str(array.dtype) != row.get("dtype")
                or sha256_array(array) != row.get("array_sha256")
            ):
                raise ProtocolError("CBPUPR dense-array member drifted.")


__all__ = ("load_table", "validate_npz_manifest")
