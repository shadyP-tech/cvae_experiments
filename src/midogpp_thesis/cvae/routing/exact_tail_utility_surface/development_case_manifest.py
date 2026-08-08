"""Typed case-identity view of the sealed exact-tail development reservation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from ..residual_topup.hashing import canonical_sha256
from .contracts import CENTERS
from .production_inputs import parse_development_partition


@dataclass(frozen=True)
class DevelopmentCaseManifest:
    reservation_hash: str
    support_case_ids_by_center: Mapping[str, tuple[str, ...]]
    evaluation_case_ids_by_center: Mapping[str, tuple[str, ...]]
    target_evaluation_case_ids_by_center: Mapping[str, tuple[str, ...]]
    partition_hashes_by_center: Mapping[str, str]
    case_manifest_hash: str


def load_development_case_manifest(root: str | Path) -> DevelopmentCaseManifest:
    path = Path(root) / "manifests/development_reservation.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("Cannot read exact-tail development case manifest.") from exc
    required = {
        "schema_version", "status", "dataset_family", "center_universe",
        "partitions", "metadata_similarity_by_query_source",
        "metadata_profile_sha256", "reservation_cache_and_index_contain_labels",
        "whole_case_support_evaluation_disjoint",
        "development_target_evaluation_disjoint", "reservation_hash",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ProtocolError("Exact-tail development case manifest schema drifted.")
    unhashed = {key: value for key, value in raw.items() if key != "reservation_hash"}
    values = raw.get("partitions")
    if (
        raw.get("schema_version")
        != "midogpp_utility_aligned_development_reservation_v1"
        or raw.get("status") != "READY"
        or raw.get("dataset_family") != "MIDOG++"
        or raw.get("center_universe") != list(CENTERS)
        or raw.get("reservation_cache_and_index_contain_labels") is not False
        or raw.get("whole_case_support_evaluation_disjoint") is not True
        or raw.get("development_target_evaluation_disjoint") is not True
        or raw.get("reservation_hash") != stable_hash(unhashed)
        or not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
    ):
        raise ProtocolError("Exact-tail development case manifest identity drifted.")
    partitions = {}
    for value in values:
        partition = parse_development_partition(value)
        if partition.center in partitions:
            raise ProtocolError("Exact-tail development case partition is duplicated.")
        partitions[partition.center] = partition
    if tuple(partitions) != CENTERS:
        raise ProtocolError("Exact-tail development case coverage drifted.")
    support = {
        center: partitions[center].support_case_ids for center in CENTERS
    }
    evaluation = {
        center: tuple(sorted({row.case_id for row in partitions[center].evaluation_rows}))
        for center in CENTERS
    }
    target_evaluation = {
        center: partitions[center].target_evaluation_case_ids for center in CENTERS
    }
    _globally_disjoint(support, evaluation, target_evaluation)
    partition_hashes = {
        center: partitions[center].reservation_hash for center in CENTERS
    }
    payload = {
        "schema_version": "midogpp_exact_tail_development_case_manifest_v1",
        "reservation_hash": raw["reservation_hash"],
        "support_case_ids_by_center": {key: list(value) for key, value in support.items()},
        "evaluation_case_ids_by_center": {key: list(value) for key, value in evaluation.items()},
        "target_evaluation_case_ids_by_center": {
            key: list(value) for key, value in target_evaluation.items()
        },
        "partition_hashes_by_center": partition_hashes,
    }
    return DevelopmentCaseManifest(
        reservation_hash=str(raw["reservation_hash"]),
        support_case_ids_by_center=MappingProxyType(support),
        evaluation_case_ids_by_center=MappingProxyType(evaluation),
        target_evaluation_case_ids_by_center=MappingProxyType(target_evaluation),
        partition_hashes_by_center=MappingProxyType(partition_hashes),
        case_manifest_hash=canonical_sha256(payload),
    )


def _globally_disjoint(*mappings: Mapping[str, tuple[str, ...]]) -> None:
    role_sets = []
    for mapping in mappings:
        observed: set[str] = set()
        for center in CENTERS:
            values = set(mapping[center])
            if observed & values:
                raise ProtocolError("Exact-tail case IDs repeat across centers.")
            observed.update(values)
        role_sets.append(observed)
    for left_index, left in enumerate(role_sets):
        if any(left & right for right in role_sets[left_index + 1 :]):
            raise ProtocolError("Exact-tail case roles overlap globally.")


__all__ = ("DevelopmentCaseManifest", "load_development_case_manifest")
