"""Closed-world admission of the candidate-level ensemble endpoint table.

The legacy per-seed utility table remains a descriptive Stage-60 member.  It
must never cross this boundary into a routing fit: this module admits only the
predeclared 504-row all-nine probability-ensemble endpoint table and its
separate lock.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from ..exact_tail_utility_surface.ensemble_scoring import (
    ENSEMBLE_AGGREGATION,
    ENSEMBLE_ENDPOINT_LOCK_MEMBER,
    ENSEMBLE_ENDPOINT_ROLE,
    ENSEMBLE_ENDPOINT_TABLE_MEMBER,
    ENSEMBLE_RESPONSE_SEMANTICS,
    ENSEMBLE_SEED_PAIR_COUNT,
    ENSEMBLE_THRESHOLD,
    ExactTailEnsembleEndpointLock,
    ScoredExactTailEnsembleEndpointRow,
    load_ensemble_endpoint_lock,
)
from ..residual_topup.hashing import canonical_sha256
from ..utility_aligned import (
    ENSEMBLE_SEED_KEYS,
    EnsembleUtilitySurface,
    SUPPORT_ACTION_PROBABILITY_SHIFT_NAME,
    SUPPORT_ACTION_PROBABILITY_SHIFT_SEMANTICS,
    SupportActionProbabilityShift,
    validate_ensemble_utility_responses,
)
from ..exact_tail_utility_surface.contracts import (
    expected_ensemble_endpoint_keys,
    expected_utility_keys,
)
from ..exact_tail_utility_surface.support_shift_surface import (
    SUPPORT_SHIFT_LOCK_MEMBER, SUPPORT_SHIFT_TABLE_MEMBER,
    SUPPORT_SHIFT_ROW_SCALAR_SEMANTICS,
    ExactTailSupportActionShiftLock,
    ExactTailSupportActionShiftRow,
    load_support_action_shift_lock,
)
from .input_io import read_csv


@dataclass(frozen=True)
class EnsembleEndpointInputs:
    """Immutable operational response surface plus endpoint provenance."""

    endpoint_lock: ExactTailEnsembleEndpointLock
    support_shift_lock: ExactTailSupportActionShiftLock
    rows: tuple[ScoredExactTailEnsembleEndpointRow, ...]
    rows_by_key: Mapping[tuple[str, str, str], ScoredExactTailEnsembleEndpointRow]
    utility_surface: EnsembleUtilitySurface
    support_shifts_by_outer: Mapping[
        str, Mapping[tuple[str, str, str], SupportActionProbabilityShift]
    ]
    endpoint_response_hash: str
    endpoint_bindings: Mapping[str, object]


def load_ensemble_endpoint_inputs(root: Path) -> EnsembleEndpointInputs:
    """Load exactly the locked ensemble endpoint, rejecting the legacy table."""

    table = root / ENSEMBLE_ENDPOINT_TABLE_MEMBER
    lock_path = root / ENSEMBLE_ENDPOINT_LOCK_MEMBER
    if not table.is_file() or not lock_path.is_file():
        raise ProtocolError("Utility policy requires the locked ensemble endpoint table.")
    lock = load_ensemble_endpoint_lock(lock_path)
    rows = tuple(_parse_row(raw) for raw in read_csv(table))
    support_shifts, support_shift_lock = _load_support_shifts(
        root, endpoint_lock=lock
    )
    by_key = {row.row_key: row for row in rows}
    if (
        len(by_key) != len(rows)
        or len(rows) != lock.endpoint_row_count
        or tuple(row.row_key for row in rows) != expected_ensemble_endpoint_keys()
    ):
        raise ProtocolError("Ensemble endpoint table key coverage drifted.")
    if _sha256_file(table) != lock.endpoint_table_sha256:
        raise ProtocolError("Ensemble endpoint table bytes escaped their lock.")
    if stable_hash([row.endpoint_row_hash for row in rows]) != lock.endpoint_row_hashes_hash:
        raise ProtocolError("Ensemble endpoint response hashes escaped their lock.")
    response_hash = canonical_sha256(
        {
            "endpoint_lock_hash": lock.endpoint_lock_hash,
            "endpoint_row_hashes": [row.endpoint_row_hash for row in rows],
        }
    )
    bindings: dict[str, object] = {
        "ensemble_endpoint_id": "all_nine_seed_probability_ensemble_bacc_delta_v1",
        "ensemble_endpoint_lock_hash": lock.endpoint_lock_hash,
        "ensemble_endpoint_table_sha256": lock.endpoint_table_sha256,
        "ensemble_endpoint_response_hash": response_hash,
        "ensemble_endpoint_row_hashes_hash": lock.endpoint_row_hashes_hash,
        "ensemble_probability_cell_surface_hash": lock.probability_cell_surface_hash,
        "ensemble_prediction_arrays_sha256": lock.prediction_arrays_sha256,
        "ensemble_seed_pair_count": ENSEMBLE_SEED_PAIR_COUNT,
        "ensemble_threshold": ENSEMBLE_THRESHOLD,
        "ensemble_aggregation_semantics": ENSEMBLE_AGGREGATION,
        "ensemble_response_semantics": ENSEMBLE_RESPONSE_SEMANTICS,
        "ensemble_endpoint_role": ENSEMBLE_ENDPOINT_ROLE,
        "source_inner_action_shift_lock_hash": support_shift_lock.shift_lock_hash,
        "source_inner_action_shift_table_sha256": (
            support_shift_lock.shift_table_sha256
        ),
        "source_inner_action_shift_row_hashes_hash": (
            support_shift_lock.shift_row_hashes_hash
        ),
        "source_inner_action_shift_row_count": support_shift_lock.shift_row_count,
        "source_inner_action_shift_scalar_name": (
            SUPPORT_ACTION_PROBABILITY_SHIFT_NAME
        ),
        "source_inner_action_shift_row_semantics": (
            SUPPORT_SHIFT_ROW_SCALAR_SEMANTICS
        ),
        "source_inner_action_shift_aggregate_semantics": (
            SUPPORT_ACTION_PROBABILITY_SHIFT_SEMANTICS
        ),
        "source_inner_action_shift_descriptive_seed_values_may_feed_model": False,
    }
    return EnsembleEndpointInputs(
        endpoint_lock=lock,
        support_shift_lock=support_shift_lock,
        rows=rows,
        rows_by_key=MappingProxyType(by_key),
        utility_surface=validate_ensemble_utility_responses(
            tuple(row.to_payload() for row in rows)
        ),
        support_shifts_by_outer=MappingProxyType(support_shifts),
        endpoint_response_hash=response_hash,
        endpoint_bindings=MappingProxyType(bindings),
    )


def _parse_row(raw: Mapping[str, object]) -> ScoredExactTailEnsembleEndpointRow:
    expected = {
        "schema_version", "outer_target", "pseudo_query", "candidate_source",
        "base_bacc", "tail_bacc", "delta_bacc", "evaluation_row_count",
        "evaluation_case_count", "evaluation_row_hash", "support_partition_hash",
        "prediction_seal_hash", "evaluation_label_sha256",
        "base_probability_cell_hashes_hash", "tail_probability_cell_hashes_hash",
        "base_ensemble_probability_sha256", "tail_ensemble_probability_sha256",
        "base_ensemble_prediction_sha256", "tail_ensemble_prediction_sha256",
        "base_endpoint_hash", "tail_endpoint_hash", "ensemble_utility_response_hash",
        "seed_pair_count", "seed_pairs_hash", "threshold", "primary_metric",
        "primary_utility_endpoint", "aggregation_semantics", "response_semantics",
        "endpoint_role", "development_labels_used_for_scoring_only",
        "technical_seed_repeats_are_not_independent_units",
        "target_support_labels_used", "target_evaluation_labels_used",
        "seed_selection_performed", "endpoint_row_hash",
    }
    if set(raw) != expected:
        raise ProtocolError("Ensemble endpoint CSV schema drifted.")
    try:
        row = ScoredExactTailEnsembleEndpointRow(
            outer_target=str(raw["outer_target"]),
            pseudo_query=str(raw["pseudo_query"]),
            candidate_source=str(raw["candidate_source"]),
            base_bacc=float(raw["base_bacc"]), tail_bacc=float(raw["tail_bacc"]),
            delta_bacc=float(raw["delta_bacc"]),
            evaluation_row_count=int(raw["evaluation_row_count"]),
            evaluation_case_count=int(raw["evaluation_case_count"]),
            evaluation_row_hash=str(raw["evaluation_row_hash"]),
            support_partition_hash=str(raw["support_partition_hash"]),
            prediction_seal_hash=str(raw["prediction_seal_hash"]),
            evaluation_label_sha256=str(raw["evaluation_label_sha256"]),
            base_probability_cell_hashes_hash=str(raw["base_probability_cell_hashes_hash"]),
            tail_probability_cell_hashes_hash=str(raw["tail_probability_cell_hashes_hash"]),
            base_ensemble_probability_sha256=str(raw["base_ensemble_probability_sha256"]),
            tail_ensemble_probability_sha256=str(raw["tail_ensemble_probability_sha256"]),
            base_ensemble_prediction_sha256=str(raw["base_ensemble_prediction_sha256"]),
            tail_ensemble_prediction_sha256=str(raw["tail_ensemble_prediction_sha256"]),
            base_endpoint_hash=str(raw["base_endpoint_hash"]),
            tail_endpoint_hash=str(raw["tail_endpoint_hash"]),
            ensemble_utility_response_hash=str(raw["ensemble_utility_response_hash"]),
            endpoint_row_hash=str(raw["endpoint_row_hash"]),
            seed_pair_count=int(raw["seed_pair_count"]), threshold=float(raw["threshold"]),
            primary_metric=str(raw["primary_metric"]),
            aggregation_semantics=str(raw["aggregation_semantics"]),
            response_semantics=str(raw["response_semantics"]),
            endpoint_role=str(raw["endpoint_role"]),
            target_support_labels_used=_bool(raw["target_support_labels_used"]),
            target_evaluation_labels_used=_bool(raw["target_evaluation_labels_used"]),
            seed_selection_performed=_bool(raw["seed_selection_performed"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("Ensemble endpoint row is malformed.") from exc
    canonical = row.to_payload()
    if (
        raw["schema_version"] != canonical["schema_version"]
        or raw["seed_pairs_hash"] != canonical["seed_pairs_hash"]
        or raw["primary_utility_endpoint"]
        != "all_nine_seed_probability_ensemble_bacc_delta"
        or raw["development_labels_used_for_scoring_only"] != "True"
        or raw["technical_seed_repeats_are_not_independent_units"] != "True"
    ):
        raise ProtocolError("Ensemble endpoint row metadata drifted.")
    return row


def _bool(value: object) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError("Expected canonical CSV boolean.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_support_shifts(
    root: Path,
    *,
    endpoint_lock: ExactTailEnsembleEndpointLock,
) -> tuple[
    dict[str, Mapping[tuple[str, str, str], SupportActionProbabilityShift]],
    ExactTailSupportActionShiftLock,
]:
    table = root / SUPPORT_SHIFT_TABLE_MEMBER
    lock_path = root / SUPPORT_SHIFT_LOCK_MEMBER
    if not table.is_file() or not lock_path.is_file():
        raise ProtocolError("Utility policy requires the exact-tail support-shift table/lock.")
    lock = load_support_action_shift_lock(lock_path)
    lock_payload = lock.to_payload()
    if (
        lock.config_contract_hash != endpoint_lock.config_contract_hash
        or lock.prediction_seal_hash != endpoint_lock.prediction_seal_hash
        or lock.prediction_index_sha256 != endpoint_lock.prediction_index_sha256
        or lock.prediction_arrays_sha256 != endpoint_lock.prediction_arrays_sha256
        or lock_payload.get("scalar_name")
        != SUPPORT_ACTION_PROBABILITY_SHIFT_NAME
        or lock_payload.get("scalar_semantics")
        != SUPPORT_SHIFT_ROW_SCALAR_SEMANTICS
        or lock_payload.get("candidate_aggregate_semantics")
        != SUPPORT_ACTION_PROBABILITY_SHIFT_SEMANTICS
        or lock_payload.get("descriptive_seed_values_may_feed_model") is not False
    ):
        raise ProtocolError("Endpoint/support-shift prediction bindings drifted.")
    if _sha256_file(table) != lock.shift_table_sha256:
        raise ProtocolError("Exact-tail support-shift table bytes escaped their lock.")
    rows = tuple(_parse_support_shift_row(raw) for raw in read_csv(table))
    if (
        len(rows) != lock.shift_row_count
        or tuple(row.row_key for row in rows) != expected_utility_keys()
        or stable_hash([row.shift_row_hash for row in rows])
        != lock.shift_row_hashes_hash
    ):
        raise ProtocolError("Exact-tail support-shift row hashes escaped their lock.")
    grouped: dict[tuple[str, str, str], list[ExactTailSupportActionShiftRow]] = {}
    for row in rows:
        grouped.setdefault((row.outer_target, row.pseudo_query, row.candidate_source), []).append(row)
    by_outer: dict[str, dict[tuple[str, str, str], SupportActionProbabilityShift]] = {}
    for key, values in grouped.items():
        ordered = tuple(sorted(values, key=lambda row: (row.training_seed, row.generation_seed)))
        if (
            tuple((row.training_seed, row.generation_seed) for row in ordered)
            != ENSEMBLE_SEED_KEYS
            or len({row.candidate_aggregate_shift_hash for row in ordered}) != 1
            or len({row.support_row_hash for row in ordered}) != 1
            or len({row.support_row_count for row in ordered}) != 1
            or len({row.support_case_count for row in ordered}) != 1
            or len({row.support_partition_hash for row in ordered}) != 1
            or len({row.prediction_seal_hash for row in ordered}) != 1
            or len(
                {row.candidate_ensemble_mean_absolute_shift for row in ordered}
            )
            != 1
            or len(
                {
                    row.candidate_base_ensemble_probability_sha256
                    for row in ordered
                }
            )
            != 1
            or len(
                {
                    row.candidate_tail_ensemble_probability_sha256
                    for row in ordered
                }
            )
            != 1
            or len(
                {
                    row.candidate_ensemble_absolute_difference_sha256
                    for row in ordered
                }
            )
            != 1
        ):
            raise ProtocolError("Exact-tail support-shift candidate aggregation drifted.")
        if ordered[0].prediction_seal_hash != lock.prediction_seal_hash:
            raise ProtocolError("Exact-tail support-shift row escaped its prediction seal.")
        vector = tuple(
            float(row.descriptive_seed_mean_absolute_shift) for row in ordered
        )
        base_vector_hashes = tuple(
            row.base_component_vector_hash for row in ordered
        )
        tail_vector_hashes = tuple(
            row.tail_component_vector_hash for row in ordered
        )
        # The producer persists the exact typed vector hashes because its raw
        # probability SHA binds float32 checkpoint bytes while SeedProbabilityVector
        # normalizes values to float64.  A raw SHA must never be substituted for
        # the typed component hash.
        if any(
            component == raw_sha
            for component, raw_sha in zip(
                (*base_vector_hashes, *tail_vector_hashes),
                (
                    *(row.base_support_probability_sha256 for row in ordered),
                    *(row.tail_support_probability_sha256 for row in ordered),
                ),
                strict=True,
            )
        ):
            raise ProtocolError(
                "Exact-tail support-shift substituted a raw probability SHA for a typed vector hash."
            )
        values_array = np.asarray(vector, dtype=np.float64)
        standard_deviation = float(np.std(values_array, ddof=0, dtype=np.float64))
        minimum = float(np.min(values_array))
        maximum = float(np.max(values_array))
        try:
            shift = SupportActionProbabilityShift(
                row_identity_hash=ordered[0].support_row_hash,
                seed_keys=ENSEMBLE_SEED_KEYS,
                base_component_vector_hashes=base_vector_hashes,
                tail_component_vector_hashes=tail_vector_hashes,
                per_seed_mean_absolute_shifts=vector,
                base_ensemble_probability_hash=(
                    ordered[0].candidate_base_ensemble_probability_sha256
                ),
                tail_ensemble_probability_hash=(
                    ordered[0].candidate_tail_ensemble_probability_sha256
                ),
                ensemble_absolute_difference_hash=(
                    ordered[0].candidate_ensemble_absolute_difference_sha256
                ),
                value=ordered[0].candidate_ensemble_mean_absolute_shift,
                seed_standard_deviation=standard_deviation,
                seed_minimum=minimum,
                seed_maximum=maximum,
                seed_range=maximum - minimum,
                shift_hash=ordered[0].candidate_aggregate_shift_hash,
            )
        except ProtocolError as exc:
            raise ProtocolError(
                "Exact-tail support-shift aggregate hash cannot be reconstructed."
            ) from exc
        by_outer.setdefault(key[0], {})[key] = shift
    expected_outer = {outer for outer, _, _ in expected_ensemble_endpoint_keys()}
    if set(by_outer) != expected_outer or any(
        set(values) != {
            key for key in expected_ensemble_endpoint_keys() if key[0] == outer
        }
        for outer, values in by_outer.items()
    ):
        raise ProtocolError("Exact-tail support-shift candidate coverage drifted.")
    return (
        {outer: MappingProxyType(values) for outer, values in by_outer.items()},
        lock,
    )


def _parse_support_shift_row(raw: Mapping[str, object]) -> ExactTailSupportActionShiftRow:
    expected = {
        "schema_version", "outer_target", "pseudo_query", "candidate_source",
        "training_seed", "generation_seed",
        "descriptive_seed_mean_absolute_shift",
        "candidate_ensemble_mean_absolute_shift", "support_row_count",
        "support_case_count", "support_row_hash", "support_partition_hash",
        "prediction_seal_hash", "base_support_probability_sha256",
        "tail_support_probability_sha256", "base_component_vector_hash",
        "tail_component_vector_hash",
        "candidate_base_ensemble_probability_sha256",
        "candidate_tail_ensemble_probability_sha256",
        "candidate_ensemble_absolute_difference_sha256",
        "candidate_aggregate_shift_hash",
        "scalar_name", "scalar_semantics", "row_role",
        "descriptive_seed_value_may_feed_model", "labels_used",
        "support_labels_available", "target_labels_used",
        "seed_selection_performed", "shift_row_hash",
    }
    if (
        set(raw) != expected
        or raw.get("descriptive_seed_value_may_feed_model") not in (False, "False")
    ):
        raise ProtocolError("Exact-tail support-shift CSV schema drifted.")
    try:
        row = ExactTailSupportActionShiftRow(
            outer_target=str(raw["outer_target"]), pseudo_query=str(raw["pseudo_query"]),
            candidate_source=str(raw["candidate_source"]),
            training_seed=int(raw["training_seed"]), generation_seed=int(raw["generation_seed"]),
            descriptive_seed_mean_absolute_shift=float(
                raw["descriptive_seed_mean_absolute_shift"]
            ),
            candidate_ensemble_mean_absolute_shift=float(
                raw["candidate_ensemble_mean_absolute_shift"]
            ),
            support_row_count=int(raw["support_row_count"]),
            support_case_count=int(raw["support_case_count"]), support_row_hash=str(raw["support_row_hash"]),
            support_partition_hash=str(raw["support_partition_hash"]), prediction_seal_hash=str(raw["prediction_seal_hash"]),
            base_support_probability_sha256=str(raw["base_support_probability_sha256"]),
            tail_support_probability_sha256=str(raw["tail_support_probability_sha256"]),
            base_component_vector_hash=str(raw["base_component_vector_hash"]),
            tail_component_vector_hash=str(raw["tail_component_vector_hash"]),
            candidate_base_ensemble_probability_sha256=str(
                raw["candidate_base_ensemble_probability_sha256"]
            ),
            candidate_tail_ensemble_probability_sha256=str(
                raw["candidate_tail_ensemble_probability_sha256"]
            ),
            candidate_ensemble_absolute_difference_sha256=str(
                raw["candidate_ensemble_absolute_difference_sha256"]
            ),
            candidate_aggregate_shift_hash=str(raw["candidate_aggregate_shift_hash"]),
            shift_row_hash=str(raw["shift_row_hash"]),
            scalar_name=str(raw["scalar_name"]), scalar_semantics=str(raw["scalar_semantics"]),
            row_role=str(raw["row_role"]), labels_used=_bool(raw["labels_used"]),
            support_labels_available=_bool(raw["support_labels_available"]),
            target_labels_used=_bool(raw["target_labels_used"]),
            seed_selection_performed=_bool(raw["seed_selection_performed"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("Exact-tail support-shift row is malformed.") from exc
    if raw["schema_version"] != row.to_payload()["schema_version"]:
        raise ProtocolError("Exact-tail support-shift row schema version drifted.")
    return row


__all__ = ("EnsembleEndpointInputs", "load_ensemble_endpoint_inputs")
