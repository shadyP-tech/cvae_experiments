"""Global pre-label capability for the local marginal-utility surface."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .contracts import (
    BOOST_ARM_ROLE,
    CENTERS,
    CONTROL_ACTION_ID,
    CONTROL_ARM_ROLE,
    EXPECTED_DEVELOPMENT_CLASSIFIER_FIT_COUNT,
    GENERATION_SEEDS,
    PERTURBATION_LIBRARY_HASH,
    TRAINING_SEEDS,
    ValidationRowIdentity,
    action_ids,
    boost_action_id,
    development_queries,
    legal_sources,
    perturbation_library_for,
    row_identity_hash,
)


GLOBAL_DEVELOPMENT_SEAL_STATUS = (
    "SEALED_GLOBAL_DEVELOPMENT_UTILITY_PREDICTIONS_BEFORE_LABEL_ACCESS"
)


def expected_prediction_keys() -> tuple[tuple[str, str, str, int, int], ...]:
    return tuple(
        (outer, query, action_id, training_seed, generation_seed)
        for outer in CENTERS
        for query in development_queries(outer)
        for action_id in action_ids(outer_target=outer, query_center=query)
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
    )


@dataclass(frozen=True)
class PredictionCellSeal:
    """One label-free classifier-output cell inside the global seal."""

    outer_target: str
    query_center: str
    action_id: str
    arm_role: str
    boosted_source: str | None
    candidate_sources: tuple[str, ...]
    training_seed: int
    generation_seed: int
    evaluation_row_ids: tuple[str, ...]
    evaluation_row_identity_hash: str
    perturbation_hash: str
    prediction_sha256: str
    probability_sha256: str
    composition_hash: str
    classifier_config_hash: str
    phase: str = "development_utility_surface"

    def __post_init__(self) -> None:
        if self.phase != "development_utility_surface":
            raise ProtocolError("Local-utility prediction cell phase drifted.")
        expected_sources = legal_sources(
            outer_target=self.outer_target,
            query_center=self.query_center,
        )
        if self.candidate_sources != expected_sources:
            raise ProtocolError(
                "Local-utility prediction cell violates H/q source exclusions."
            )
        if self.action_id == CONTROL_ACTION_ID:
            if self.arm_role != CONTROL_ARM_ROLE or self.boosted_source is not None:
                raise ProtocolError("Local-utility control cell identity drifted.")
        elif (
            self.arm_role != BOOST_ARM_ROLE
            or self.boosted_source not in expected_sources
            or self.action_id != boost_action_id(str(self.boosted_source))
        ):
            raise ProtocolError("Local-utility boost cell identity drifted.")
        if (
            isinstance(self.training_seed, bool)
            or not isinstance(self.training_seed, int)
            or self.training_seed not in TRAINING_SEEDS
        ):
            raise ProtocolError("Local-utility cell training seed drifted.")
        if (
            isinstance(self.generation_seed, bool)
            or not isinstance(self.generation_seed, int)
            or self.generation_seed not in GENERATION_SEEDS
        ):
            raise ProtocolError("Local-utility cell generation seed drifted.")
        if not self.evaluation_row_ids or len(self.evaluation_row_ids) != len(
            set(self.evaluation_row_ids)
        ):
            raise ProtocolError("Local-utility cell rows are empty or duplicated.")
        action_by_id = {
            action.action_id: action
            for action in perturbation_library_for(
                outer_target=self.outer_target,
                query_center=self.query_center,
            )
        }
        expected_action = action_by_id.get(self.action_id)
        if expected_action is None or self.perturbation_hash != stable_hash(
            expected_action.to_payload()
        ):
            raise ProtocolError("Local-utility cell perturbation binding drifted.")
        for role, value in (
            ("evaluation row-identity hash", self.evaluation_row_identity_hash),
            ("perturbation hash", self.perturbation_hash),
            ("prediction SHA-256", self.prediction_sha256),
            ("probability SHA-256", self.probability_sha256),
            ("composition hash", self.composition_hash),
            ("classifier hash", self.classifier_config_hash),
        ):
            _require_hash(value, role)

    @property
    def key(self) -> tuple[str, str, str, int, int]:
        return (
            self.outer_target,
            self.query_center,
            self.action_id,
            self.training_seed,
            self.generation_seed,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "outer_target": self.outer_target,
            "query_center": self.query_center,
            "action_id": self.action_id,
            "arm_role": self.arm_role,
            "boosted_source": self.boosted_source,
            "candidate_sources": list(self.candidate_sources),
            "training_seed": self.training_seed,
            "generation_seed": self.generation_seed,
            "evaluation_row_count": len(self.evaluation_row_ids),
            "evaluation_row_identity_hash": self.evaluation_row_identity_hash,
            "perturbation_hash": self.perturbation_hash,
            "prediction_sha256": self.prediction_sha256,
            "probability_sha256": self.probability_sha256,
            "composition_hash": self.composition_hash,
            "classifier_config_hash": self.classifier_config_hash,
        }


@dataclass(frozen=True)
class GlobalDevelopmentPredictionSeal:
    """Complete 5,184-cell capability required before any labels open."""

    config_contract_hash: str
    perturbation_library_hash: str
    support_partition_lock_hash: str
    compatibility_index_hash: str
    validation_cache_binding_hash: str
    validation_manifest_sha256: str
    prediction_index_sha256: str
    prediction_arrays_sha256: str
    evaluation_row_ids_by_query: Mapping[str, tuple[str, ...]]
    evaluation_row_identity_hash_by_query: Mapping[str, str]
    cells: tuple[PredictionCellSeal, ...]
    seal_hash: str
    status: str = GLOBAL_DEVELOPMENT_SEAL_STATUS

    def __post_init__(self) -> None:
        rows = {
            str(query): tuple(str(value) for value in values)
            for query, values in self.evaluation_row_ids_by_query.items()
        }
        hashes = {
            str(query): str(value)
            for query, value in self.evaluation_row_identity_hash_by_query.items()
        }
        object.__setattr__(self, "evaluation_row_ids_by_query", MappingProxyType(rows))
        object.__setattr__(
            self,
            "evaluation_row_identity_hash_by_query",
            MappingProxyType(hashes),
        )
        object.__setattr__(self, "cells", tuple(self.cells))
        self.verify_complete()

    @property
    def row_count(self) -> int:
        return sum(len(rows) for rows in self.evaluation_row_ids_by_query.values())

    @property
    def cell_count(self) -> int:
        return len(self.cells)

    def verify_complete(self) -> None:
        if self.status != GLOBAL_DEVELOPMENT_SEAL_STATUS:
            raise ProtocolError("Global local-utility prediction seal is incomplete.")
        if self.perturbation_library_hash != PERTURBATION_LIBRARY_HASH:
            raise ProtocolError("Global local-utility perturbation library drifted.")
        for role, value in (
            ("config contract hash", self.config_contract_hash),
            ("support partition-lock hash", self.support_partition_lock_hash),
            ("compatibility-index hash", self.compatibility_index_hash),
            ("validation-cache binding hash", self.validation_cache_binding_hash),
            ("validation manifest SHA-256", self.validation_manifest_sha256),
            ("prediction-index SHA-256", self.prediction_index_sha256),
            ("prediction-array SHA-256", self.prediction_arrays_sha256),
        ):
            _require_hash(value, role)
        if tuple(self.evaluation_row_ids_by_query) != CENTERS or tuple(
            self.evaluation_row_identity_hash_by_query
        ) != CENTERS:
            raise ProtocolError(
                "Global local-utility seal lacks canonical query-center coverage."
            )
        all_ids: list[str] = []
        for query in CENTERS:
            row_ids = self.evaluation_row_ids_by_query[query]
            if not row_ids or len(row_ids) != len(set(row_ids)):
                raise ProtocolError(
                    "Global local-utility query rows are empty or duplicated."
                )
            _require_hash(
                self.evaluation_row_identity_hash_by_query[query],
                f"query {query} row-identity hash",
            )
            all_ids.extend(row_ids)
        if len(all_ids) != len(set(all_ids)):
            raise ProtocolError(
                "Global local-utility evaluation rows cross query centers."
            )

        expected_keys = expected_prediction_keys()
        if len(expected_keys) != EXPECTED_DEVELOPMENT_CLASSIFIER_FIT_COUNT:
            raise ProtocolError("Local-utility expected-cell arithmetic drifted.")
        observed_keys: list[tuple[str, str, str, int, int]] = []
        for cell in self.cells:
            if not isinstance(cell, PredictionCellSeal):
                raise ProtocolError(
                    "Global local-utility seal contains a non-prediction cell."
                )
            query = cell.query_center
            if (
                cell.evaluation_row_ids
                != self.evaluation_row_ids_by_query.get(query)
                or cell.evaluation_row_identity_hash
                != self.evaluation_row_identity_hash_by_query.get(query)
            ):
                raise ProtocolError(
                    "Local-utility cell escaped its sealed query rows."
                )
            observed_keys.append(cell.key)
        if tuple(observed_keys) != expected_keys:
            raise ProtocolError(
                "Global local-utility seal lacks complete canonical 5,184-cell "
                "coverage."
            )
        if self.seal_hash != stable_hash(self._unhashed_payload()):
            raise ProtocolError("Global local-utility prediction seal hash drifted.")

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": (
                "midogpp_local_marginal_utility_global_prediction_seal_v1"
            ),
            "status": self.status,
            "config_contract_hash": self.config_contract_hash,
            "perturbation_library_hash": self.perturbation_library_hash,
            "support_partition_lock_hash": self.support_partition_lock_hash,
            "compatibility_index_hash": self.compatibility_index_hash,
            "validation_cache_binding_hash": self.validation_cache_binding_hash,
            "validation_manifest_sha256": self.validation_manifest_sha256,
            "prediction_index_sha256": self.prediction_index_sha256,
            "prediction_arrays_sha256": self.prediction_arrays_sha256,
            "evaluation_row_ids_by_query": {
                query: list(self.evaluation_row_ids_by_query[query])
                for query in CENTERS
            },
            "evaluation_row_identity_hash_by_query": {
                query: self.evaluation_row_identity_hash_by_query[query]
                for query in CENTERS
            },
            "outer_target_count": len(CENTERS),
            "query_count_per_outer_target": len(CENTERS) - 1,
            "action_count_per_outer_query": len(CENTERS) - 1,
            "seed_cell_count_per_action": len(TRAINING_SEEDS)
            * len(GENERATION_SEEDS),
            "cell_count": len(self.cells),
            "cells": [cell.to_payload() for cell in self.cells],
            "all_global_development_predictions_materialized": True,
            "development_labels_opened": False,
            "target_labels_opened": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "seal_hash": self.seal_hash}


def build_global_development_prediction_seal(
    *,
    config_contract_hash: str,
    support_partition_lock_hash: str,
    compatibility_index_hash: str,
    validation_cache_binding_hash: str,
    validation_manifest_sha256: str,
    prediction_index_sha256: str,
    prediction_arrays_sha256: str,
    evaluation_rows_by_query: Mapping[str, Sequence[ValidationRowIdentity]],
    cells: Sequence[PredictionCellSeal],
) -> GlobalDevelopmentPredictionSeal:
    normalized_rows = {
        str(query): tuple(rows) for query, rows in evaluation_rows_by_query.items()
    }
    if tuple(normalized_rows) != CENTERS:
        raise ProtocolError(
            "Global local-utility seal requires rows for all nine query centers."
        )
    if any(
        not rows
        or any(
            row.partition_role != "evaluation" or row.center != query
            for row in rows
        )
        for query, rows in normalized_rows.items()
    ):
        raise ProtocolError(
            "Global local-utility seal may bind evaluation rows for q only."
        )
    rows_by_query = {
        query: tuple(row.sample_id for row in normalized_rows[query])
        for query in CENTERS
    }
    identity_hashes = {
        query: row_identity_hash(normalized_rows[query]) for query in CENTERS
    }
    by_key: dict[tuple[str, str, str, int, int], PredictionCellSeal] = {}
    for cell in cells:
        if not isinstance(cell, PredictionCellSeal):
            raise ProtocolError(
                "Global local-utility seal contains a non-prediction cell."
            )
        if cell.key in by_key:
            raise ProtocolError("Global local-utility seal duplicates a cell.")
        by_key[cell.key] = cell
    expected_keys = expected_prediction_keys()
    if set(by_key) != set(expected_keys):
        raise ProtocolError(
            "Global local-utility prediction seal lacks complete cell coverage."
        )
    ordered_cells = tuple(by_key[key] for key in expected_keys)
    values: dict[str, object] = {
        "config_contract_hash": str(config_contract_hash),
        "perturbation_library_hash": PERTURBATION_LIBRARY_HASH,
        "support_partition_lock_hash": str(support_partition_lock_hash),
        "compatibility_index_hash": str(compatibility_index_hash),
        "validation_cache_binding_hash": str(validation_cache_binding_hash),
        "validation_manifest_sha256": str(validation_manifest_sha256),
        "prediction_index_sha256": str(prediction_index_sha256),
        "prediction_arrays_sha256": str(prediction_arrays_sha256),
        "evaluation_row_ids_by_query": rows_by_query,
        "evaluation_row_identity_hash_by_query": identity_hashes,
        "cells": ordered_cells,
        "seal_hash": "",
        "status": GLOBAL_DEVELOPMENT_SEAL_STATUS,
    }
    provisional = GlobalDevelopmentPredictionSeal.__new__(
        GlobalDevelopmentPredictionSeal
    )
    for field, value in values.items():
        object.__setattr__(provisional, field, value)
    values["seal_hash"] = stable_hash(provisional._unhashed_payload())
    return GlobalDevelopmentPredictionSeal(**values)  # type: ignore[arg-type]


def _require_hash(value: str, role: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) not in {16, 64}
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProtocolError(f"Local marginal-utility {role} is malformed.")


__all__ = (
    "GLOBAL_DEVELOPMENT_SEAL_STATUS",
    "GlobalDevelopmentPredictionSeal",
    "PredictionCellSeal",
    "build_global_development_prediction_seal",
    "expected_prediction_keys",
)
