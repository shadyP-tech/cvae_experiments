"""Source-stream and prediction-store reconstruction for SCEPTRE v4."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import CENTERS

from ....protocol import ProtocolError
from ....runtime.artifact_io import sha256_file
from ...fixed_bank_sceptre_router.hashing import canonical_hash, require_sha256
from ...fixed_bank_sceptre_router.partitions import ThreeRolePartition
from ..experiment_contracts import EXPECTED_TEST_ROWS_BY_CENTER
from .validation_io import read_validation_object


PREDICTION_RECEIPT_MEMBER = (
    "prediction_store/manifests/sceptre_v4_prediction_receipt.json"
)
PREDICTION_INDEX_MEMBER = (
    "prediction_store/manifests/sceptre_v4_prediction_index.json"
)
CANDIDATE_ARRAY_MEMBER = (
    "prediction_store/arrays/sceptre_v4_candidate_probabilities.npy"
)
EXACT_B_ARRAY_MEMBER = "prediction_store/arrays/sceptre_v4_exact_b_probabilities.npy"


def validate_physical_graph(
    destination: Path,
    *,
    index: Mapping[str, object],
    bundle: Mapping[str, object],
    input_binding: Mapping[str, object],
    source_store: Mapping[str, object],
    prediction_store: Mapping[str, object],
    phases: Mapping[str, object],
    prediction_hashes: Mapping[str, object],
    partition: ThreeRolePartition,
) -> None:
    """Authenticate physical members, semantic indices, arrays, and exclusions."""

    for name, digest in prediction_hashes.items():
        path = destination / str(name)
        if (
            path.is_symlink()
            or not path.is_file()
            or sha256_file(path) != require_sha256(digest, f"prediction member {name}")
        ):
            raise ProtocolError("SCEPTRE v4 prediction member drifted.")

    source_store_body = {
        key: value for key, value in source_store.items() if key != "source_store_hash"
    }
    source_receipt = source_store.get("source_receipt")
    source_receipt_body = (
        {
            key: value
            for key, value in source_receipt.items()
            if key != "receipt_sha256"
        }
        if isinstance(source_receipt, Mapping)
        else {}
    )
    prediction_members = prediction_store.get("member_hashes")
    prediction_receipt = read_validation_object(
        destination / PREDICTION_RECEIPT_MEMBER
    )
    prediction_index = read_validation_object(destination / PREDICTION_INDEX_MEMBER)
    prediction_receipt_body = {
        key: value
        for key, value in prediction_receipt.items()
        if key != "receipt_sha256"
    }
    lease = bundle.get("authorization_lease")
    if not isinstance(lease, Mapping):
        raise ProtocolError("SCEPTRE v4 physical authorization lineage is malformed.")
    if (
        source_store.get("schema_version")
        != "sceptre_v4_source_store_binding_v1"
        or source_store.get("source_store_hash") != canonical_hash(source_store_body)
        or index.get("source_store_hash") != source_store.get("source_store_hash")
        or not isinstance(source_receipt, Mapping)
        or source_receipt.get("receipt_sha256")
        != canonical_hash(source_receipt_body)
        or source_store.get("receipt_hash") != source_receipt.get("receipt_sha256")
        or source_store.get("stream_count") != 81
        or len(source_store.get("record_hashes", ())) != 81
        or source_store.get("labels_opened") is not False
        or source_store.get("source_array_file_sha256")
        != source_receipt.get("source_array_sha256")
        or source_store.get("source_index_file_sha256")
        != source_receipt.get("source_index_file_sha256")
        or source_store.get("attempt_id") != lease.get("lease_hash")
        or prediction_store.get("schema_version")
        != "sceptre_v4_prediction_store_binding_v1"
        or not isinstance(prediction_members, Mapping)
        or dict(prediction_members) != dict(prediction_hashes)
        or prediction_store.get("receipt_hash") != phases.get("prediction_store_hash")
        or prediction_store.get("receipt_hash")
        != prediction_receipt.get("receipt_sha256")
        or prediction_receipt.get("receipt_sha256")
        != canonical_hash(prediction_receipt_body)
        or prediction_receipt.get("source_receipt_sha256")
        != source_store.get("receipt_hash")
        or prediction_receipt.get("attempt_id") != source_store.get("attempt_id")
        or prediction_store.get("candidate_shape") != [9, 9, 9928]
        or prediction_store.get("exact_b_shape") != [9, 9928]
        or prediction_store.get("candidate_source_order") != list(CENTERS)
        or prediction_store.get("read_only_memmap") is not True
        or prediction_store.get("labels_opened") is not False
    ):
        raise ProtocolError("SCEPTRE v4 preterminal physical graph drifted.")
    _validate_prediction_surface(
        destination,
        bundle=bundle,
        input_binding=input_binding,
        source_store=source_store,
        prediction_receipt=prediction_receipt,
        prediction_index=prediction_index,
        partition=partition,
    )


def _validate_prediction_surface(
    destination: Path,
    *,
    bundle: Mapping[str, object],
    input_binding: Mapping[str, object],
    source_store: Mapping[str, object],
    prediction_receipt: Mapping[str, object],
    prediction_index: Mapping[str, object],
    partition: ThreeRolePartition,
) -> None:
    index_body = {
        key: value for key, value in prediction_index.items() if key != "index_sha256"
    }
    receipt_body = {
        key: value
        for key, value in prediction_receipt.items()
        if key != "receipt_sha256"
    }
    row_ids = prediction_index.get("row_ids")
    row_centers = prediction_index.get("row_centers")
    fit_rows = prediction_index.get("fit_rows")
    lease = bundle.get("authorization_lease")
    if not isinstance(lease, Mapping) or not all(
        isinstance(value, list) for value in (row_ids, row_centers, fit_rows)
    ):
        raise ProtocolError("SCEPTRE v4 prediction semantic index is malformed.")
    expected_row_identity = canonical_hash(
        [
            {"row_ordinal": ordinal, "row_id": str(row_id), "center": str(center)}
            for ordinal, (row_id, center) in enumerate(
                zip(row_ids, row_centers, strict=True)
            )
        ]
    )
    observed_rows = tuple(
        (str(row_id), str(center))
        for row_id, center in zip(row_ids, row_centers, strict=True)
    )
    expected_rows = {
        (identity.sample_id, identity.target_center)
        for identity in partition.identities
    }
    center_counts = {
        center: sum(str(value) == center for value in row_centers)
        for center in CENTERS
    }
    if (
        prediction_index.get("index_sha256") != canonical_hash(index_body)
        or prediction_receipt.get("receipt_sha256") != canonical_hash(receipt_body)
        or prediction_receipt.get("prediction_index_sha256")
        != prediction_index.get("index_sha256")
        or prediction_index.get("config_hash") != bundle.get("config_hash")
        or prediction_receipt.get("config_hash") != bundle.get("config_hash")
        or prediction_index.get("attempt_id") != lease.get("lease_hash")
        or prediction_receipt.get("attempt_id") != lease.get("lease_hash")
        or prediction_index.get("source_receipt_sha256")
        != source_store.get("receipt_hash")
        or prediction_receipt.get("source_receipt_sha256")
        != source_store.get("receipt_hash")
        or prediction_index.get("cache_binding_hash")
        != input_binding.get("cache_binding_hash")
        or prediction_receipt.get("cache_binding_hash")
        != input_binding.get("cache_binding_hash")
        or len(row_ids) != 9928
        or len(row_centers) != 9928
        or len(set(map(str, row_ids))) != 9928
        or set(observed_rows) != expected_rows
        or len(observed_rows) != len(expected_rows)
        or center_counts != EXPECTED_TEST_ROWS_BY_CENTER
        or prediction_index.get("row_identity_sha256") != expected_row_identity
        or prediction_receipt.get("row_identity_sha256") != expected_row_identity
        or len(fit_rows) != 162
        or prediction_index.get("fit_count") != 162
        or prediction_receipt.get("fit_count") != 162
        or prediction_index.get("fit_index_sha256") != canonical_hash(fit_rows)
        or prediction_receipt.get("fit_index_sha256")
        != prediction_index.get("fit_index_sha256")
        or prediction_index.get("candidate_source_order") != list(CENTERS)
        or prediction_index.get("seed_selection_performed") is not False
        or prediction_receipt.get("seed_selection_performed") is not False
        or prediction_index.get("manifest_opened") is not False
        or prediction_receipt.get("manifest_opened") is not False
    ):
        raise ProtocolError("SCEPTRE v4 prediction semantic lineage drifted.")
    family_counts = {"single_source": 0, "exact_B": 0}
    for raw in fit_rows:
        if not isinstance(raw, Mapping):
            raise ProtocolError("SCEPTRE v4 prediction fit row is malformed.")
        body = {key: value for key, value in raw.items() if key != "fit_sha256"}
        family = str(raw.get("family"))
        if family not in family_counts:
            raise ProtocolError("SCEPTRE v4 prediction fit family drifted.")
        family_counts[family] += 1
        if (
            raw.get("fit_sha256") != canonical_hash(body)
            or raw.get("converged") is not True
            or raw.get("evaluated_row_count") != 9928
            or (
                family == "single_source"
                and (
                    raw.get("excluded_evaluation_center")
                    != raw.get("source_center")
                    or not isinstance(raw.get("masked_row_count"), int)
                    or int(raw["masked_row_count"]) <= 0
                )
            )
            or (
                family == "exact_B"
                and (
                    raw.get("target_center") in tuple(raw.get("source_centers", ()))
                    or raw.get("target_expert_excluded") is not True
                )
            )
        ):
            raise ProtocolError("SCEPTRE v4 prediction fit semantics drifted.")
    if family_counts != {"single_source": 81, "exact_B": 81}:
        raise ProtocolError("SCEPTRE v4 prediction fit coverage drifted.")

    try:
        candidate = np.load(
            destination / CANDIDATE_ARRAY_MEMBER,
            mmap_mode="r",
            allow_pickle=False,
        )
        exact_b = np.load(
            destination / EXACT_B_ARRAY_MEMBER,
            mmap_mode="r",
            allow_pickle=False,
        )
    except (OSError, ValueError) as exc:
        raise ProtocolError("SCEPTRE v4 prediction arrays cannot be opened safely.") from exc
    if (
        candidate.shape != (9, 9, 9928)
        or exact_b.shape != (9, 9928)
        or candidate.dtype != np.float32
        or exact_b.dtype != np.float32
        or candidate.flags.writeable
        or exact_b.flags.writeable
        or not np.isfinite(candidate).all()
        or not np.isfinite(exact_b).all()
        or np.any((exact_b < 0.0) | (exact_b > 1.0))
    ):
        raise ProtocolError("SCEPTRE v4 prediction array geometry drifted.")
    centers_array = np.asarray(row_centers, dtype=str)
    for source_ordinal, center in enumerate(CENTERS):
        masked = centers_array == center
        if (
            not masked.any()
            or not np.all(candidate[:, source_ordinal, masked] == np.float32(-1.0))
            or np.any(candidate[:, source_ordinal, ~masked] < 0.0)
            or np.any(candidate[:, source_ordinal, ~masked] > 1.0)
        ):
            raise ProtocolError("SCEPTRE v4 candidate exclusion semantics drifted.")


__all__ = ("validate_physical_graph",)
