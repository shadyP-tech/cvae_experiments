"""Label-free support action-probability shifts from already fitted cells."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from ..utility_aligned.ensemble_contracts import (
    SUPPORT_ACTION_PROBABILITY_SHIFT_NAME,
    SUPPORT_ACTION_PROBABILITY_SHIFT_SEMANTICS,
    SUPPORT_ACTION_TECHNICAL_SEED_SPREAD_SEMANTICS,
    SeedProbabilityVector,
    SupportActionProbabilityShift,
)
from ..utility_aligned.ensemble_endpoint import support_action_probability_shift
from .contracts import (
    BASE_ACTION_ID,
    EXPECTED_UTILITY_ROW_COUNT,
    DevelopmentPartition,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
    expected_ensemble_endpoint_keys,
    expected_utility_keys,
    legal_sources,
    tail_action_id,
)
from .scoring import SealedPredictionSurface
from .seals import GlobalPredictionSeal


SUPPORT_SHIFT_TABLE_MEMBER = "tables/exact_tail_support_action_shifts.csv"
SUPPORT_SHIFT_LOCK_MEMBER = "manifests/exact_tail_support_action_shifts_lock.json"
SUPPORT_SHIFT_ROW_SCHEMA = (
    "midogpp_exact_tail_support_action_probability_shift_row_v2"
)
SUPPORT_SHIFT_LOCK_SCHEMA = "midogpp_exact_tail_support_action_shifts_lock_v2"
SUPPORT_SHIFT_ROW_ROLE = (
    "label_free_descriptive_technical_seed_cell_with_bound_ensemble_first_"
    "candidate_scalar"
)
SUPPORT_SHIFT_ROW_SCALAR_SEMANTICS = (
    SUPPORT_ACTION_TECHNICAL_SEED_SPREAD_SEMANTICS
)


@dataclass(frozen=True)
class ExactTailSupportActionShiftRow:
    outer_target: str
    pseudo_query: str
    candidate_source: str
    training_seed: int
    generation_seed: int
    descriptive_seed_mean_absolute_shift: float
    candidate_ensemble_mean_absolute_shift: float
    support_row_count: int
    support_case_count: int
    support_row_hash: str
    support_partition_hash: str
    prediction_seal_hash: str
    base_support_probability_sha256: str
    tail_support_probability_sha256: str
    base_component_vector_hash: str
    tail_component_vector_hash: str
    candidate_base_ensemble_probability_sha256: str
    candidate_tail_ensemble_probability_sha256: str
    candidate_ensemble_absolute_difference_sha256: str
    candidate_aggregate_shift_hash: str
    shift_row_hash: str
    scalar_name: str = SUPPORT_ACTION_PROBABILITY_SHIFT_NAME
    scalar_semantics: str = SUPPORT_SHIFT_ROW_SCALAR_SEMANTICS
    row_role: str = SUPPORT_SHIFT_ROW_ROLE
    labels_used: bool = False
    support_labels_available: bool = False
    target_labels_used: bool = False
    seed_selection_performed: bool = False

    def __post_init__(self) -> None:
        if (
            self.pseudo_query == self.outer_target
            or self.candidate_source
            not in legal_sources(
                outer_target=self.outer_target, pseudo_query=self.pseudo_query
            )
            or self.training_seed not in TRAINING_SEEDS
            or self.generation_seed not in GENERATION_SEEDS
        ):
            raise ProtocolError("Exact-tail support shift violates H/q/e/seed geometry.")
        for value in (
            self.descriptive_seed_mean_absolute_shift,
            self.candidate_ensemble_mean_absolute_shift,
        ):
            if not math.isfinite(float(value)) or not 0.0 <= value <= 1.0:
                raise ProtocolError("Exact-tail support action shift is outside [0,1].")
        if self.support_row_count <= 0 or self.support_case_count <= 0:
            raise ProtocolError("Exact-tail support action shift has empty support rows.")
        for value, role, lengths in (
            (self.support_row_hash, "support-row hash", {16}),
            (self.support_partition_hash, "support-partition hash", {16}),
            (self.prediction_seal_hash, "prediction-seal hash", {16}),
            (
                self.base_support_probability_sha256,
                "base support-probability SHA-256",
                {64},
            ),
            (
                self.tail_support_probability_sha256,
                "tail support-probability SHA-256",
                {64},
            ),
            (
                self.base_component_vector_hash,
                "base component-vector hash",
                {64},
            ),
            (
                self.tail_component_vector_hash,
                "tail component-vector hash",
                {64},
            ),
            (
                self.candidate_base_ensemble_probability_sha256,
                "candidate base-ensemble SHA-256",
                {64},
            ),
            (
                self.candidate_tail_ensemble_probability_sha256,
                "candidate tail-ensemble SHA-256",
                {64},
            ),
            (
                self.candidate_ensemble_absolute_difference_sha256,
                "candidate ensemble absolute-difference SHA-256",
                {64},
            ),
            (
                self.candidate_aggregate_shift_hash,
                "candidate aggregate-shift hash",
                {64},
            ),
            (self.shift_row_hash, "support shift-row hash", {16}),
        ):
            _require_hash(value, role, lengths)
        if (
            self.scalar_name != SUPPORT_ACTION_PROBABILITY_SHIFT_NAME
            or self.scalar_semantics != SUPPORT_SHIFT_ROW_SCALAR_SEMANTICS
            or self.row_role != SUPPORT_SHIFT_ROW_ROLE
            or self.labels_used is not False
            or self.support_labels_available is not False
            or self.target_labels_used is not False
            or self.seed_selection_performed is not False
        ):
            raise ProtocolError("Exact-tail support action-shift contract drifted.")
        if self.shift_row_hash != stable_hash(self._unhashed_payload()):
            raise ProtocolError("Exact-tail support action-shift row hash drifted.")

    @property
    def row_key(self) -> tuple[str, str, str, int, int]:
        return (
            self.outer_target,
            self.pseudo_query,
            self.candidate_source,
            self.training_seed,
            self.generation_seed,
        )

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": SUPPORT_SHIFT_ROW_SCHEMA,
            "outer_target": self.outer_target,
            "pseudo_query": self.pseudo_query,
            "candidate_source": self.candidate_source,
            "training_seed": self.training_seed,
            "generation_seed": self.generation_seed,
            "descriptive_seed_mean_absolute_shift": (
                self.descriptive_seed_mean_absolute_shift
            ),
            "candidate_ensemble_mean_absolute_shift": (
                self.candidate_ensemble_mean_absolute_shift
            ),
            "support_row_count": self.support_row_count,
            "support_case_count": self.support_case_count,
            "support_row_hash": self.support_row_hash,
            "support_partition_hash": self.support_partition_hash,
            "prediction_seal_hash": self.prediction_seal_hash,
            "base_support_probability_sha256": (
                self.base_support_probability_sha256
            ),
            "tail_support_probability_sha256": (
                self.tail_support_probability_sha256
            ),
            "base_component_vector_hash": self.base_component_vector_hash,
            "tail_component_vector_hash": self.tail_component_vector_hash,
            "candidate_base_ensemble_probability_sha256": (
                self.candidate_base_ensemble_probability_sha256
            ),
            "candidate_tail_ensemble_probability_sha256": (
                self.candidate_tail_ensemble_probability_sha256
            ),
            "candidate_ensemble_absolute_difference_sha256": (
                self.candidate_ensemble_absolute_difference_sha256
            ),
            "candidate_aggregate_shift_hash": self.candidate_aggregate_shift_hash,
            "scalar_name": self.scalar_name,
            "scalar_semantics": self.scalar_semantics,
            "row_role": self.row_role,
            "descriptive_seed_value_may_feed_model": False,
            "labels_used": self.labels_used,
            "support_labels_available": self.support_labels_available,
            "target_labels_used": self.target_labels_used,
            "seed_selection_performed": self.seed_selection_performed,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "shift_row_hash": self.shift_row_hash}


@dataclass(frozen=True)
class ExactTailSupportActionShiftLock:
    config_contract_hash: str
    prediction_seal_hash: str
    prediction_index_sha256: str
    prediction_arrays_sha256: str
    support_probability_cell_surface_hash: str
    shift_keys_hash: str
    shift_row_hashes_hash: str
    shift_table_sha256: str
    shift_row_count: int
    shift_lock_hash: str
    schema_version: str = SUPPORT_SHIFT_LOCK_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != SUPPORT_SHIFT_LOCK_SCHEMA:
            raise ProtocolError("Exact-tail support shift-lock schema drifted.")
        if self.shift_row_count != EXPECTED_UTILITY_ROW_COUNT:
            raise ProtocolError("Exact-tail support shift-lock count drifted.")
        for value, role, lengths in (
            (self.config_contract_hash, "config hash", {16, 64}),
            (self.prediction_seal_hash, "prediction-seal hash", {16}),
            (self.prediction_index_sha256, "prediction-index SHA-256", {64}),
            (self.prediction_arrays_sha256, "prediction-array SHA-256", {64}),
            (
                self.support_probability_cell_surface_hash,
                "support-probability surface hash",
                {16},
            ),
            (self.shift_keys_hash, "support shift-key hash", {16}),
            (self.shift_row_hashes_hash, "support shift-row hash", {16}),
            (self.shift_table_sha256, "support shift-table SHA-256", {64}),
            (self.shift_lock_hash, "support shift-lock hash", {16}),
        ):
            _require_hash(value, role, lengths)
        if self.shift_keys_hash != _shift_keys_hash():
            raise ProtocolError("Exact-tail support shift identity grid escaped its lock.")
        if self.shift_lock_hash != stable_hash(self._unhashed_payload()):
            raise ProtocolError("Exact-tail support shift-lock hash drifted.")

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "config_contract_hash": self.config_contract_hash,
            "prediction_seal_hash": self.prediction_seal_hash,
            "prediction_index_sha256": self.prediction_index_sha256,
            "prediction_arrays_sha256": self.prediction_arrays_sha256,
            "support_probability_cell_surface_hash": (
                self.support_probability_cell_surface_hash
            ),
            "shift_keys_hash": self.shift_keys_hash,
            "shift_row_hashes_hash": self.shift_row_hashes_hash,
            "shift_table_sha256": self.shift_table_sha256,
            "shift_row_count": self.shift_row_count,
            "scalar_name": SUPPORT_ACTION_PROBABILITY_SHIFT_NAME,
            "scalar_semantics": SUPPORT_SHIFT_ROW_SCALAR_SEMANTICS,
            "candidate_aggregate_semantics": (
                SUPPORT_ACTION_PROBABILITY_SHIFT_SEMANTICS
            ),
            "descriptive_seed_values_may_feed_model": False,
            "one_row_per_H_q_e_training_seed_generation_seed": True,
            "support_probabilities_sealed_before_development_labels": True,
            "labels_used": False,
            "support_labels_available": False,
            "target_labels_used": False,
            "seed_selection_performed": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "shift_lock_hash": self.shift_lock_hash}


def build_support_action_shift_rows(
    predictions: SealedPredictionSurface,
    partitions: Mapping[str, DevelopmentPartition],
) -> tuple[ExactTailSupportActionShiftRow, ...]:
    """Derive all 4,536 label-free rows from already sealed support outputs."""

    seal = predictions.seal
    seal.verify_complete()
    normalized = {str(center): partition for center, partition in partitions.items()}
    cell_by_key = {cell.key: cell for cell in seal.cells}
    aggregates: dict[tuple[str, str, str], object] = {}
    for outer, query, source in expected_ensemble_endpoint_keys():
        partition = normalized.get(query)
        if (
            partition is None
            or partition.reservation_hash != seal.partition_hash_by_center[query]
        ):
            raise ProtocolError("Exact-tail support shift partition escaped the seal.")
        base_keys = _action_seed_keys(outer, query, BASE_ACTION_ID)
        tail_keys = _action_seed_keys(outer, query, tail_action_id(source))
        base_vectors = _support_vectors(predictions, base_keys, cell_by_key)
        tail_vectors = _support_vectors(predictions, tail_keys, cell_by_key)
        aggregates[(outer, query, source)] = support_action_probability_shift(
            base_vectors, tail_vectors
        )
    rows: list[ExactTailSupportActionShiftRow] = []
    seed_ordinal = {
        (training_seed, generation_seed): ordinal
        for ordinal, (training_seed, generation_seed) in enumerate(
            _seed_pairs()
        )
    }
    for outer, query, source, training_seed, generation_seed in expected_utility_keys():
        aggregate = aggregates[(outer, query, source)]
        ordinal = seed_ordinal[(training_seed, generation_seed)]
        base_key = (outer, query, BASE_ACTION_ID, training_seed, generation_seed)
        tail_key = (
            outer,
            query,
            tail_action_id(source),
            training_seed,
            generation_seed,
        )
        partition = normalized[query]
        values: dict[str, object] = {
            "outer_target": outer,
            "pseudo_query": query,
            "candidate_source": source,
            "training_seed": training_seed,
            "generation_seed": generation_seed,
            "descriptive_seed_mean_absolute_shift": (
                aggregate.per_seed_mean_absolute_shifts[ordinal]
            ),
            "candidate_ensemble_mean_absolute_shift": aggregate.value,
            "support_row_count": len(partition.support_rows),
            "support_case_count": len(partition.support_case_ids),
            "support_row_hash": seal.support_row_hash_by_center[query],
            "support_partition_hash": partition.reservation_hash,
            "prediction_seal_hash": seal.seal_hash,
            "base_support_probability_sha256": cell_by_key[
                base_key
            ].support_probability_sha256,
            "tail_support_probability_sha256": cell_by_key[
                tail_key
            ].support_probability_sha256,
            "base_component_vector_hash": (
                aggregate.base_component_vector_hashes[ordinal]
            ),
            "tail_component_vector_hash": (
                aggregate.tail_component_vector_hashes[ordinal]
            ),
            "candidate_base_ensemble_probability_sha256": (
                aggregate.base_ensemble_probability_hash
            ),
            "candidate_tail_ensemble_probability_sha256": (
                aggregate.tail_ensemble_probability_hash
            ),
            "candidate_ensemble_absolute_difference_sha256": (
                aggregate.ensemble_absolute_difference_hash
            ),
            "candidate_aggregate_shift_hash": aggregate.shift_hash,
            "shift_row_hash": "",
        }
        provisional = ExactTailSupportActionShiftRow.__new__(
            ExactTailSupportActionShiftRow
        )
        for key, value in values.items():
            object.__setattr__(provisional, key, value)
        for key, value in _row_defaults().items():
            object.__setattr__(provisional, key, value)
        values["shift_row_hash"] = stable_hash(provisional._unhashed_payload())
        rows.append(ExactTailSupportActionShiftRow(**values))  # type: ignore[arg-type]
    if (
        len(rows) != EXPECTED_UTILITY_ROW_COUNT
        or tuple(row.row_key for row in rows) != expected_utility_keys()
    ):
        raise ProtocolError("Exact-tail support action-shift coverage drifted.")
    return tuple(rows)


def build_support_action_shift_lock(
    *,
    seal: GlobalPredictionSeal,
    rows: Sequence[ExactTailSupportActionShiftRow],
    shift_table_sha256: str,
) -> ExactTailSupportActionShiftLock:
    shift_rows = tuple(rows)
    if tuple(row.row_key for row in shift_rows) != expected_utility_keys():
        raise ProtocolError("Exact-tail support action-shift rows are not canonical.")
    if any(row.prediction_seal_hash != seal.seal_hash for row in shift_rows):
        raise ProtocolError("Exact-tail support action-shift rows mix prediction seals.")
    validate_support_shift_group_bindings(shift_rows)
    values: dict[str, object] = {
        "config_contract_hash": seal.config_contract_hash,
        "prediction_seal_hash": seal.seal_hash,
        "prediction_index_sha256": seal.prediction_index_sha256,
        "prediction_arrays_sha256": seal.prediction_arrays_sha256,
        "support_probability_cell_surface_hash": stable_hash(
            [cell.support_probability_sha256 for cell in seal.cells]
        ),
        "shift_keys_hash": _shift_keys_hash(),
        "shift_row_hashes_hash": stable_hash(
            [row.shift_row_hash for row in shift_rows]
        ),
        "shift_table_sha256": str(shift_table_sha256),
        "shift_row_count": len(shift_rows),
        "shift_lock_hash": "",
        "schema_version": SUPPORT_SHIFT_LOCK_SCHEMA,
    }
    provisional = ExactTailSupportActionShiftLock.__new__(
        ExactTailSupportActionShiftLock
    )
    for key, value in values.items():
        object.__setattr__(provisional, key, value)
    values["shift_lock_hash"] = stable_hash(provisional._unhashed_payload())
    return ExactTailSupportActionShiftLock(**values)  # type: ignore[arg-type]


def load_support_action_shift_lock(
    path: str | Path,
) -> ExactTailSupportActionShiftLock:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("Cannot load exact-tail support action-shift lock.") from exc
    required = set(ExactTailSupportActionShiftLock.__dataclass_fields__)
    required.update(
        {
            "scalar_name",
            "scalar_semantics",
            "candidate_aggregate_semantics",
            "descriptive_seed_values_may_feed_model",
            "one_row_per_H_q_e_training_seed_generation_seed",
            "support_probabilities_sealed_before_development_labels",
            "labels_used",
            "support_labels_available",
            "target_labels_used",
            "seed_selection_performed",
        }
    )
    if set(raw) != required or any(
        (
            raw.get("schema_version") != SUPPORT_SHIFT_LOCK_SCHEMA,
            raw.get("scalar_name") != SUPPORT_ACTION_PROBABILITY_SHIFT_NAME,
            raw.get("scalar_semantics") != SUPPORT_SHIFT_ROW_SCALAR_SEMANTICS,
            raw.get("candidate_aggregate_semantics")
            != SUPPORT_ACTION_PROBABILITY_SHIFT_SEMANTICS,
            raw.get("descriptive_seed_values_may_feed_model") is not False,
            raw.get("one_row_per_H_q_e_training_seed_generation_seed") is not True,
            raw.get("support_probabilities_sealed_before_development_labels")
            is not True,
            raw.get("labels_used") is not False,
            raw.get("support_labels_available") is not False,
            raw.get("target_labels_used") is not False,
            raw.get("seed_selection_performed") is not False,
        )
    ):
        raise ProtocolError("Exact-tail support action-shift lock violates its contract.")
    return ExactTailSupportActionShiftLock(
        config_contract_hash=str(raw["config_contract_hash"]),
        prediction_seal_hash=str(raw["prediction_seal_hash"]),
        prediction_index_sha256=str(raw["prediction_index_sha256"]),
        prediction_arrays_sha256=str(raw["prediction_arrays_sha256"]),
        support_probability_cell_surface_hash=str(
            raw["support_probability_cell_surface_hash"]
        ),
        shift_keys_hash=str(raw["shift_keys_hash"]),
        shift_row_hashes_hash=str(raw["shift_row_hashes_hash"]),
        shift_table_sha256=str(raw["shift_table_sha256"]),
        shift_row_count=int(raw["shift_row_count"]),
        shift_lock_hash=str(raw["shift_lock_hash"]),
        schema_version=str(raw["schema_version"]),
    )


def _support_vectors(
    predictions: SealedPredictionSurface,
    keys: Sequence[tuple[str, str, str, int, int]],
    cell_by_key: Mapping[tuple[str, str, str, int, int], object],
) -> tuple[SeedProbabilityVector, ...]:
    try:
        return tuple(
            SeedProbabilityVector(
                training_seed=key[3],
                generation_seed=key[4],
                row_identity_hash=str(
                    getattr(cell_by_key[key], "support_row_identity_hash")
                ),
                prediction_provenance_hash=str(
                    getattr(cell_by_key[key], "support_probability_sha256")
                ),
                positive_class_probabilities=(
                    predictions.support_probabilities_by_key[key]
                ),
            )
            for key in keys
        )
    except (KeyError, AttributeError) as exc:
        raise ProtocolError("Exact-tail sealed support probability cell is missing.") from exc


def validate_support_shift_group_bindings(
    rows: Sequence[ExactTailSupportActionShiftRow],
) -> None:
    """Require one hash-bound v2 aggregate plus nine descriptive seed rows."""

    grouped: dict[tuple[str, str, str], list[ExactTailSupportActionShiftRow]] = {}
    for row in rows:
        grouped.setdefault(
            (row.outer_target, row.pseudo_query, row.candidate_source), []
        ).append(row)
    for key, group in grouped.items():
        ordered = tuple(group)
        if tuple(
            (row.training_seed, row.generation_seed) for row in ordered
        ) != _seed_pairs():
            raise ProtocolError(
                f"Exact-tail support shift group {key!r} is not canonical exact-nine."
            )
        bound_values = {
            (
                row.candidate_ensemble_mean_absolute_shift,
                row.candidate_base_ensemble_probability_sha256,
                row.candidate_tail_ensemble_probability_sha256,
                row.candidate_ensemble_absolute_difference_sha256,
                row.candidate_aggregate_shift_hash,
            )
            for row in ordered
        }
        if len(bound_values) != 1:
            raise ProtocolError(
                "Exact-tail descriptive seed rows disagree on their ensemble aggregate."
            )
        aggregate_value = ordered[0].candidate_ensemble_mean_absolute_shift
        descriptive_mean = float(
            sum(row.descriptive_seed_mean_absolute_shift for row in ordered)
            / len(ordered)
        )
        if aggregate_value > descriptive_mean + 1.0e-12:
            raise ProtocolError(
                "Exact-tail ensemble-first shift exceeds its technical-seed bound."
            )
        descriptive = np.asarray(
            [row.descriptive_seed_mean_absolute_shift for row in ordered],
            dtype=np.float64,
        )
        SupportActionProbabilityShift(
            row_identity_hash=ordered[0].support_row_hash,
            seed_keys=_seed_pairs(),
            base_component_vector_hashes=tuple(
                row.base_component_vector_hash for row in ordered
            ),
            tail_component_vector_hashes=tuple(
                row.tail_component_vector_hash for row in ordered
            ),
            per_seed_mean_absolute_shifts=tuple(
                float(value) for value in descriptive
            ),
            base_ensemble_probability_hash=(
                ordered[0].candidate_base_ensemble_probability_sha256
            ),
            tail_ensemble_probability_hash=(
                ordered[0].candidate_tail_ensemble_probability_sha256
            ),
            ensemble_absolute_difference_hash=(
                ordered[0].candidate_ensemble_absolute_difference_sha256
            ),
            value=aggregate_value,
            seed_standard_deviation=float(
                np.std(descriptive, ddof=0, dtype=np.float64)
            ),
            seed_minimum=float(np.min(descriptive)),
            seed_maximum=float(np.max(descriptive)),
            seed_range=float(np.max(descriptive) - np.min(descriptive)),
            shift_hash=ordered[0].candidate_aggregate_shift_hash,
        )


def _action_seed_keys(
    outer: str, query: str, action_id: str
) -> tuple[tuple[str, str, str, int, int], ...]:
    return tuple(
        (outer, query, action_id, training_seed, generation_seed)
        for training_seed, generation_seed in _seed_pairs()
    )


def _seed_pairs() -> tuple[tuple[int, int], ...]:
    return tuple(
        (training_seed, generation_seed)
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
    )


def _row_defaults() -> dict[str, object]:
    return {
        "scalar_name": SUPPORT_ACTION_PROBABILITY_SHIFT_NAME,
        "scalar_semantics": SUPPORT_SHIFT_ROW_SCALAR_SEMANTICS,
        "row_role": SUPPORT_SHIFT_ROW_ROLE,
        "labels_used": False,
        "support_labels_available": False,
        "target_labels_used": False,
        "seed_selection_performed": False,
    }


def _shift_keys_hash() -> str:
    return stable_hash([list(key) for key in expected_utility_keys()])


def _require_hash(value: object, role: str, lengths: set[int]) -> None:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProtocolError(f"Exact-tail {role} is malformed.")


__all__ = (
    "SUPPORT_SHIFT_LOCK_MEMBER",
    "SUPPORT_SHIFT_LOCK_SCHEMA",
    "SUPPORT_SHIFT_ROW_ROLE",
    "SUPPORT_SHIFT_ROW_SCALAR_SEMANTICS",
    "SUPPORT_SHIFT_ROW_SCHEMA",
    "SUPPORT_SHIFT_TABLE_MEMBER",
    "ExactTailSupportActionShiftLock",
    "ExactTailSupportActionShiftRow",
    "build_support_action_shift_lock",
    "build_support_action_shift_rows",
    "load_support_action_shift_lock",
    "validate_support_shift_group_bindings",
)
