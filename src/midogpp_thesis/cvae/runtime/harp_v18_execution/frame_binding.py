"""Durable, label-free frame-store identity for HARP v18 classifier tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from ..artifact_io import atomic_json, read_json, sha256_file
from .hash_contracts import require_sha256, require_stable_hash


_PROVENANCE_KEYS = frozenset(
    {
        "schema_version",
        "cache_index_hash",
        "cache_content_sha256",
        "config_hash",
        "protocol_hash",
        "physical_input_receipt_hash",
        "representation_id",
        "feature_backbone",
        "roles",
        "centers",
        "contexts",
        "ordered_row_identity_hash",
        "ordered_sample_identity_hash",
        "ordered_case_identity_hash",
        "row_count",
        "output_dim",
        "dtype",
        "labels_stored",
    }
)


@dataclass(frozen=True, slots=True)
class FrameBinding:
    """Independent semantic and byte identities for one immutable frame store."""

    array_sha256: str
    provenance_hash: str
    receipt_hash: str
    receipt_sha256: str


def persist_or_validate_frame_binding(
    *,
    array_path: Path,
    receipt_path: Path,
    shape: Sequence[int],
    provenance: Mapping[str, object],
    receipt_creation_authorized: bool = False,
) -> FrameBinding:
    """Seal one frame array to its exact cache, row order, and run protocol.

    A receipt may be created only in the call that materialized the array.  A
    later caller therefore cannot bless a same-shaped foreign scratch array by
    supplying new provenance after the fact.
    """

    if (
        not array_path.is_absolute()
        or not receipt_path.is_absolute()
        or array_path.is_symlink()
        or receipt_path.is_symlink()
        or not array_path.is_file()
    ):
        raise ProtocolError("HARP v18 frame binding paths are unsafe.")
    if type(receipt_creation_authorized) is not bool:
        raise ProtocolError("HARP v18 frame receipt creation flag is malformed.")
    if (
        len(shape) != 2
        or any(type(value) is not int or value <= 0 for value in shape)
    ):
        raise ProtocolError("HARP v18 frame shape is malformed.")
    shape_values = (int(shape[0]), int(shape[1]))
    provenance_payload = _validate_provenance(provenance, shape=shape_values)
    provenance_hash = require_sha256(
        canonical_hash(provenance_payload), name="frame-provenance hash"
    )
    array_sha256 = require_sha256(
        sha256_file(array_path), name="frame-array hash"
    )
    body = {
        "schema_version": "midogpp_harp_v18_scratch_frame_receipt_v2",
        "array_sha256": array_sha256,
        "shape": list(shape_values),
        "dtype": "float32",
        "labels_stored": False,
        "provenance": provenance_payload,
        "provenance_hash": provenance_hash,
    }
    receipt_hash = require_stable_hash(
        stable_hash(body), name="frame-receipt hash"
    )
    expected = {**body, "frame_receipt_hash": receipt_hash}
    if receipt_path.exists():
        if not receipt_path.is_file() or read_json(receipt_path) != expected:
            raise ProtocolError("HARP v18 existing frame receipt drifted.")
    else:
        if not receipt_creation_authorized:
            raise ProtocolError(
                "HARP v18 cannot seal a pre-existing frame without its "
                "authenticated provenance receipt."
            )
        atomic_json(receipt_path, expected)
    if read_json(receipt_path) != expected:
        raise ProtocolError("HARP v18 frame receipt failed a durable round trip.")
    return FrameBinding(
        array_sha256=array_sha256,
        provenance_hash=provenance_hash,
        receipt_hash=receipt_hash,
        receipt_sha256=require_sha256(
            sha256_file(receipt_path), name="frame-receipt SHA-256"
        ),
    )


def _validate_provenance(
    value: Mapping[str, object], *, shape: Sequence[int]
) -> dict[str, object]:
    """Validate the closed provenance surface before it reaches JSON IO."""

    if not isinstance(value, Mapping) or set(value) != _PROVENANCE_KEYS:
        raise ProtocolError("HARP v18 frame provenance schema drifted.")
    payload = dict(value)
    for key in (
        "cache_index_hash",
        "cache_content_sha256",
        "config_hash",
        "protocol_hash",
        "physical_input_receipt_hash",
        "ordered_row_identity_hash",
        "ordered_sample_identity_hash",
        "ordered_case_identity_hash",
    ):
        require_sha256(payload.get(key), name=f"frame-provenance {key}")
    contexts = payload.get("contexts")
    roles = payload.get("roles")
    centers = payload.get("centers")
    if (
        payload.get("schema_version")
        != "midogpp_harp_v18_scratch_frame_provenance_v1"
        or payload.get("representation_id")
        != "midogpp_virchow2_common_3840_float32_v1"
        or payload.get("feature_backbone") != "Virchow2_3840"
        or not isinstance(roles, list)
        or not roles
        or any(type(item) is not str or not item for item in roles)
        or len(set(roles)) != len(roles)
        or not isinstance(centers, list)
        or not centers
        or any(type(item) is not str or not item for item in centers)
        or len(set(centers)) != len(centers)
        or not isinstance(contexts, list)
        or len(contexts) != len(roles) * len(centers)
        or type(payload.get("row_count")) is not int
        or int(payload["row_count"]) != int(shape[0])
        or type(payload.get("output_dim")) is not int
        or int(payload["output_dim"]) != int(shape[1])
        or payload.get("dtype") != "float32"
        or payload.get("labels_stored") is not False
    ):
        raise ProtocolError("HARP v18 frame provenance semantics drifted.")
    seen: set[tuple[str, str]] = set()
    observed_pairs: list[tuple[str, str]] = []
    total = 0
    for raw in contexts:
        if not isinstance(raw, Mapping) or set(raw) != {
            "role",
            "center",
            "frame_start",
            "frame_stop",
            "row_count",
            "row_identity_hash",
            "sample_ids_hash",
            "case_ids_hash",
        }:
            raise ProtocolError("HARP v18 frame context provenance drifted.")
        role = raw.get("role")
        center = raw.get("center")
        start = raw.get("frame_start")
        stop = raw.get("frame_stop")
        row_count = raw.get("row_count")
        if (
            type(role) is not str
            or role not in roles
            or type(center) is not str
            or center not in centers
            or (role, center) in seen
            or type(start) is not int
            or type(stop) is not int
            or type(row_count) is not int
            or start != total
            or stop <= start
            or row_count != stop - start
        ):
            raise ProtocolError("HARP v18 frame context geometry drifted.")
        for key in ("row_identity_hash", "sample_ids_hash", "case_ids_hash"):
            require_sha256(raw.get(key), name=f"frame-context {key}")
        seen.add((role, center))
        observed_pairs.append((role, center))
        total = stop
    expected_pairs = [(role, center) for role in roles for center in centers]
    if total != int(shape[0]) or observed_pairs != expected_pairs:
        raise ProtocolError("HARP v18 frame context coverage drifted.")
    # The canonical encoder is deliberately strict and rejects accidental
    # dataclasses, mapping proxies with non-string keys, NaNs, or path objects.
    canonical_hash(payload)
    return payload


__all__ = ("FrameBinding", "persist_or_validate_frame_binding")
