"""Partition and development-response reconstruction for science validation."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import array_sha256, canonical_sha256
from ...routing.utility_aligned import (
    build_case_bootstrap_plan,
    scored_ensemble_utility_response_from_payload,
)
from .artifact_io import read_json
from .contracts import BASE_ACTION_ID, CENTERS, h_x_e_action_id
from .endpoint_adapter import validate_development_endpoint_responses
from .prediction_contracts import DEVELOPMENT_ROLE
from .prediction_store import load_prediction_store
from .validation_science_common import (
    boolean,
    floating,
    integer,
    json_value,
    nullable_text,
    read_csv,
    require_fields,
    validate_prediction_seal,
)
from .validation_science_contracts import (
    DevelopmentScienceValidation,
    RESPONSE_FIELDS,
    ScientificPartitionContext,
)


def validate_partition_context(root: str | Path) -> ScientificPartitionContext:
    base = Path(root)
    lock = read_json(base / "manifests/support_partition_lock.json")
    rows = read_csv(base / "tables/support_partitions.csv")
    support_cases: dict[str, set[str]] = {center: set() for center in CENTERS}
    support_rows: dict[str, list[dict[str, object]]] = {
        center: [] for center in CENTERS
    }
    evaluation_rows: dict[str, list[dict[str, object]]] = {
        center: [] for center in CENTERS
    }
    partition_hashes = lock.get("partition_hashes_by_center")
    if not isinstance(partition_hashes, Mapping):
        raise ProtocolError("Scientific partition hashes are absent.")
    for raw in rows:
        center = raw.get("center", "")
        role = raw.get("partition_role", "")
        if center not in CENTERS or role not in {"support", "evaluation"}:
            raise ProtocolError("Scientific partition row is malformed.")
        identity = {
            "row_ordinal": integer(raw["row_ordinal"], "row_ordinal"),
            "manifest_row_index": integer(
                raw["manifest_row_index"], "manifest_row_index"
            ),
            "evaluation_row_id": raw["evaluation_row_id"],
            "case_id": raw["case_id"],
            "center": center,
            "split": raw["split"],
            "partition_role": role,
        }
        (support_rows if role == "support" else evaluation_rows)[center].append(
            identity
        )
        if role == "support":
            support_cases[center].add(raw["case_id"])
    case_ids = {
        center: tuple(sorted(support_cases[center])) for center in CENTERS
    }
    if any(len(case_ids[center]) != 8 for center in CENTERS):
        raise ProtocolError("Scientific support-case geometry drifted.")
    support_feature_hashes = {
        center: build_case_bootstrap_plan(
            target_id=center, support_case_ids=case_ids[center]
        ).support_partition_hash
        for center in CENTERS
    }
    evaluation_hashes = {
        center: canonical_sha256(evaluation_rows[center]) for center in CENTERS
    }
    support_identity_hashes = {
        center: canonical_sha256(support_rows[center]) for center in CENTERS
    }
    lock_hash = str(lock.get("support_partition_lock_hash", ""))
    if not lock_hash:
        raise ProtocolError("Scientific partition lock hash is absent.")
    return ScientificPartitionContext(
        support_case_ids_by_center=MappingProxyType(case_ids),
        support_row_identity_hash_by_center=MappingProxyType(
            support_identity_hashes
        ),
        support_feature_hash_by_center=MappingProxyType(support_feature_hashes),
        evaluation_identity_hash_by_center=MappingProxyType(evaluation_hashes),
        partition_hash_by_center=MappingProxyType(
            {center: str(partition_hashes[center]) for center in CENTERS}
        ),
        support_partition_lock_hash=lock_hash,
    )


def validate_development_science(
    root: str | Path, partitions: ScientificPartitionContext
) -> DevelopmentScienceValidation:
    base = Path(root)
    rows = []
    for raw in read_csv(base / "tables/development_endpoint_responses.csv"):
        payload = decode_response(raw)
        persisted_hash = str(payload.pop("row_hash"))
        row = scored_ensemble_utility_response_from_payload(payload)
        if persisted_hash != row.row_hash or payload != row.to_payload():
            raise ProtocolError("Development response row hash drifted.")
        if (
            row.support_partition_hash
            != partitions.support_row_identity_hash_by_center[row.query_id]
            or row.evaluation_partition_hash
            != partitions.evaluation_identity_hash_by_center[row.query_id]
        ):
            raise ProtocolError("Development response partition binding drifted.")
        rows.append(row)

    prediction_seal = read_json(
        base / "manifests/development_prediction_seal.json"
    )
    seal_hash = str(prediction_seal.get("development_prediction_seal_hash", ""))
    # Validate the strict H/q/e geometry before opening large prediction arrays.
    response_set = validate_development_endpoint_responses(
        rows, development_prediction_seal_hash=seal_hash
    )
    response_seal = read_json(
        base / "manifests/development_endpoint_response_seal.json"
    )
    if response_seal != response_set.to_payload():
        raise ProtocolError("Development response seal is not reconstructive.")

    store = load_prediction_store(base, phase=DEVELOPMENT_ROLE)
    validate_prediction_seal(
        base,
        prediction_seal,
        hash_field="development_prediction_seal_hash",
        expected_arrays_member="arrays/development_probabilities.npz",
        expected_index_member="manifests/development_prediction_index.json",
        store=store,
    )
    if store.partition_lock_hash != partitions.support_partition_lock_hash:
        raise ProtocolError("Development prediction partition lock drifted.")
    for row in response_set.rows:
        base_vectors = store.vectors(
            outer_target=row.outer_target_id,
            query_center=row.query_id,
            action_id=BASE_ACTION_ID,
            role="evaluation",
        )
        tail_vectors = store.vectors(
            outer_target=row.outer_target_id,
            query_center=row.query_id,
            action_id=h_x_e_action_id(row.candidate_source),
            role="evaluation",
        )
        validate_response_probability_lineage(row, base_vectors, tail_vectors)
    return DevelopmentScienceValidation(
        response_count=len(response_set.rows),
        response_set_hash=response_set.response_set_hash,
        prediction_seal_hash=seal_hash,
        binding_hash_by_target=MappingProxyType(
            {
                target: response_set.binding_hash_for_outer_target(target)
                for target in CENTERS
            }
        ),
        response_set=response_set,
    )


def decode_response(raw: Mapping[str, str]) -> dict[str, object]:
    require_fields(raw, RESPONSE_FIELDS, "development response")
    payload: dict[str, object] = dict(raw)
    payload["candidate_source_count"] = integer(
        raw["candidate_source_count"], "candidate_source_count"
    )
    for name in ("base_bacc", "tail_bacc", "utility_delta"):
        payload[name] = floating(raw[name], name)
    for name in (
        "support_eval_disjoint", "predictions_sealed_before_labels",
        "source_expert_frozen", "target_labels_used_for_routing",
    ):
        payload[name] = boolean(raw[name], name)
    for name in ("base_component_vector_hashes", "tail_component_vector_hashes"):
        payload[name] = json_value(raw[name], name, list)
    for name in (
        "evaluation_label_hash", "source_response_hash", "source_endpoint_row_hash"
    ):
        payload[name] = nullable_text(raw[name])
    return payload


def validate_response_probability_lineage(
    row: object,
    base_vectors: Sequence[object],
    tail_vectors: Sequence[object],
) -> None:
    base_hashes = tuple(str(getattr(vector, "vector_hash")) for vector in base_vectors)
    tail_hashes = tuple(str(getattr(vector, "vector_hash")) for vector in tail_vectors)
    base_mean = np.mean(
        np.stack(
            [getattr(vector, "positive_class_probabilities") for vector in base_vectors]
        ),
        axis=0,
        dtype=np.float64,
    )
    tail_mean = np.mean(
        np.stack(
            [getattr(vector, "positive_class_probabilities") for vector in tail_vectors]
        ),
        axis=0,
        dtype=np.float64,
    )
    if (
        getattr(row, "base_component_vector_hashes") != base_hashes
        or getattr(row, "tail_component_vector_hashes") != tail_hashes
        or getattr(row, "base_probability_cell_hashes_hash")
        != canonical_sha256(list(base_hashes))
        or getattr(row, "tail_probability_cell_hashes_hash")
        != canonical_sha256(list(tail_hashes))
        or getattr(row, "base_ensemble_probability_hash")
        != array_sha256(base_mean)
        or getattr(row, "tail_ensemble_probability_hash")
        != array_sha256(tail_mean)
        or getattr(row, "base_ensemble_prediction_hash")
        != array_sha256((base_mean >= 0.5).astype(np.uint8))
        or getattr(row, "tail_ensemble_prediction_hash")
        != array_sha256((tail_mean >= 0.5).astype(np.uint8))
    ):
        raise ProtocolError("Development response probability lineage drifted.")


__all__ = (
    "decode_response", "validate_development_science",
    "validate_partition_context", "validate_response_probability_lineage",
)
