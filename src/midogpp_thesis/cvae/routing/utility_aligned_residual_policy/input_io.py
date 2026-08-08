"""Closed schema parsers shared by policy input families."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ..utility_aligned import CandidateFeatureRow


def read_csv(path: Path) -> tuple[Mapping[str, object], ...]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ProtocolError("Utility-aligned CSV header drifted.")
            return tuple(MappingProxyType(dict(row)) for row in reader)
    except OSError as exc:
        raise ProtocolError(f"Cannot read utility-aligned CSV: {path}.") from exc


def read_json(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read utility-aligned JSON: {path}.") from exc
    if not isinstance(raw, dict):
        raise ProtocolError("Utility-aligned JSON must be an object.")
    return raw


def mapping_sequence(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ProtocolError("Utility-aligned row sequence is malformed.")
    result = tuple(value)
    if any(not isinstance(item, Mapping) for item in result):
        raise ProtocolError("Utility-aligned row sequence contains a non-object.")
    return tuple(MappingProxyType(dict(item)) for item in result)  # type: ignore[arg-type]


def parse_feature_row(raw: Mapping[str, object]) -> CandidateFeatureRow:
    expected = {
        "schema_version", "role", "outer_target_id", "query_id", "candidate_source",
        "training_seed", "generation_seed", "replicate_id", "candidate_source_count",
        "support_partition_hash", "support_case_count", "reconstruction_mean",
        "reconstruction_std", "reconstruction_q25", "reconstruction_q50",
        "reconstruction_q75", "kl_mean", "kl_std", "kl_q25", "kl_q50", "kl_q75",
        "replica_disagreement", "distribution_mmd", "metadata_similarity",
        "feature_semantics", "row_hash",
    }
    values = dict(raw)
    if "distribution_mmd_semantics" in values:
        if values.pop("distribution_mmd_semantics") != "linear_kernel_mmd_squared":
            raise ProtocolError("Candidate feature MMD semantics drifted.")
    if set(values) != expected:
        raise ProtocolError("Candidate feature row schema drifted.")
    try:
        row = CandidateFeatureRow(
            role=str(values["role"]), outer_target_id=str(values["outer_target_id"]),
            query_id=str(values["query_id"]), candidate_source=str(values["candidate_source"]),
            training_seed=int(values["training_seed"]), generation_seed=int(values["generation_seed"]),
            candidate_source_count=int(values["candidate_source_count"]),
            support_partition_hash=str(values["support_partition_hash"]),
            support_case_count=int(values["support_case_count"]),
            reconstruction_mean=float(values["reconstruction_mean"]),
            reconstruction_std=float(values["reconstruction_std"]),
            reconstruction_q25=float(values["reconstruction_q25"]),
            reconstruction_q50=float(values["reconstruction_q50"]),
            reconstruction_q75=float(values["reconstruction_q75"]),
            kl_mean=float(values["kl_mean"]), kl_std=float(values["kl_std"]),
            kl_q25=float(values["kl_q25"]), kl_q50=float(values["kl_q50"]),
            kl_q75=float(values["kl_q75"]),
            replica_disagreement=float(values["replica_disagreement"]),
            distribution_mmd=float(values["distribution_mmd"]),
            metadata_similarity=float(values["metadata_similarity"]),
            feature_semantics=str(values["feature_semantics"]),
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolError("Candidate feature values are malformed.") from exc
    if values["schema_version"] != "midogpp_utility_aligned_candidate_feature_row_v1" or values["replicate_id"] != row.replicate_id or values["row_hash"] != row.row_hash:
        raise ProtocolError("Candidate feature row hash/identity drifted.")
    return row


__all__ = ("mapping_sequence", "parse_feature_row", "read_csv", "read_json")
