"""Global pre-label prediction capability for exact additive-tail utility."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .contracts import (
    CENTERS,
    EXPECTED_PREDICTION_CELL_COUNT,
    DevelopmentPartition,
    action_library_for,
    expected_prediction_keys,
    row_identity_hash,
)


GLOBAL_SEAL_STATUS = "SEALED_ALL_EXACT_TAIL_PREDICTIONS_BEFORE_DEVELOPMENT_LABELS"


@dataclass(frozen=True)
class PredictionCellSeal:
    outer_target: str
    pseudo_query: str
    action_id: str
    training_seed: int
    generation_seed: int
    action_hash: str
    evaluation_row_identity_hash: str
    support_row_identity_hash: str
    prediction_sha256: str
    probability_sha256: str
    support_probability_sha256: str
    composition_sha256: str
    classifier_config_hash: str

    def __post_init__(self) -> None:
        actions = {
            action.action_id: action
            for action in action_library_for(
                outer_target=self.outer_target, pseudo_query=self.pseudo_query
            )
        }
        expected = actions.get(self.action_id)
        if expected is None or expected.action_hash != self.action_hash:
            raise ProtocolError("Exact-tail prediction cell action binding drifted.")
        for value, role, lengths in (
            (self.evaluation_row_identity_hash, "evaluation-row hash", {16}),
            (self.support_row_identity_hash, "support-row hash", {16}),
            (self.action_hash, "action hash", {16}),
            (self.prediction_sha256, "prediction SHA-256", {64}),
            (self.probability_sha256, "probability SHA-256", {64}),
            (
                self.support_probability_sha256,
                "support-probability SHA-256",
                {64},
            ),
            (self.composition_sha256, "composition SHA-256", {64}),
            (self.classifier_config_hash, "classifier hash", {16, 64}),
        ):
            _require_hash(value, role, lengths)

    @property
    def key(self) -> tuple[str, str, str, int, int]:
        return (
            self.outer_target,
            self.pseudo_query,
            self.action_id,
            self.training_seed,
            self.generation_seed,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "outer_target": self.outer_target,
            "pseudo_query": self.pseudo_query,
            "action_id": self.action_id,
            "training_seed": self.training_seed,
            "generation_seed": self.generation_seed,
            "action_hash": self.action_hash,
            "evaluation_row_identity_hash": self.evaluation_row_identity_hash,
            "support_row_identity_hash": self.support_row_identity_hash,
            "prediction_sha256": self.prediction_sha256,
            "probability_sha256": self.probability_sha256,
            "support_probability_sha256": self.support_probability_sha256,
            "composition_sha256": self.composition_sha256,
            "classifier_config_hash": self.classifier_config_hash,
            "evaluation_labels_available_to_fit_or_predict": False,
            "support_labels_used": False,
            "support_probabilities_are_label_free": True,
            "target_labels_used": False,
            "seed_selection_performed": False,
        }


@dataclass(frozen=True)
class GlobalPredictionSeal:
    config_contract_hash: str
    reservation_index_hash: str
    development_cache_binding_hash: str
    development_manifest_sha256: str
    target_evaluation_binding_hash: str
    prediction_index_sha256: str
    prediction_arrays_sha256: str
    partition_hash_by_center: Mapping[str, str]
    evaluation_row_hash_by_center: Mapping[str, str]
    support_row_hash_by_center: Mapping[str, str]
    cells: tuple[PredictionCellSeal, ...]
    seal_hash: str
    status: str = GLOBAL_SEAL_STATUS

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "partition_hash_by_center",
            MappingProxyType(dict(self.partition_hash_by_center)),
        )
        object.__setattr__(
            self,
            "evaluation_row_hash_by_center",
            MappingProxyType(dict(self.evaluation_row_hash_by_center)),
        )
        object.__setattr__(
            self,
            "support_row_hash_by_center",
            MappingProxyType(dict(self.support_row_hash_by_center)),
        )
        object.__setattr__(self, "cells", tuple(self.cells))
        self.verify_complete()

    def verify_complete(self) -> None:
        if self.status != GLOBAL_SEAL_STATUS:
            raise ProtocolError("Exact-tail global prediction seal is not complete.")
        for value, role, lengths in (
            (self.config_contract_hash, "config hash", {16, 64}),
            (self.reservation_index_hash, "reservation hash", {16, 64}),
            (self.development_cache_binding_hash, "cache hash", {16, 64}),
            (self.development_manifest_sha256, "manifest SHA-256", {64}),
            (self.target_evaluation_binding_hash, "target-eval binding hash", {16, 64}),
            (self.prediction_index_sha256, "prediction-index SHA-256", {64}),
            (self.prediction_arrays_sha256, "prediction-array SHA-256", {64}),
        ):
            _require_hash(value, role, lengths)
        if tuple(self.partition_hash_by_center) != CENTERS or tuple(
            self.evaluation_row_hash_by_center
        ) != CENTERS or tuple(self.support_row_hash_by_center) != CENTERS:
            raise ProtocolError("Exact-tail global seal lacks canonical center coverage.")
        for value in (
            *self.partition_hash_by_center.values(),
            *self.evaluation_row_hash_by_center.values(),
            *self.support_row_hash_by_center.values(),
        ):
            _require_hash(value, "partition identity", {16})
        expected = expected_prediction_keys()
        if len(expected) != EXPECTED_PREDICTION_CELL_COUNT:
            raise ProtocolError("Exact-tail expected prediction arithmetic drifted.")
        if len(self.cells) != len(expected) or tuple(cell.key for cell in self.cells) != expected:
            raise ProtocolError(
                "Exact-tail global seal lacks complete canonical prediction coverage."
            )
        if len({cell.key for cell in self.cells}) != len(self.cells):
            raise ProtocolError("Exact-tail global seal duplicates prediction cells.")
        for cell in self.cells:
            if cell.evaluation_row_identity_hash != self.evaluation_row_hash_by_center[
                cell.pseudo_query
            ]:
                raise ProtocolError("Exact-tail cell escaped its sealed query rows.")
            if cell.support_row_identity_hash != self.support_row_hash_by_center[
                cell.pseudo_query
            ]:
                raise ProtocolError("Exact-tail cell escaped its sealed support rows.")
        if self.seal_hash != stable_hash(self._unhashed_payload()):
            raise ProtocolError("Exact-tail global prediction seal hash drifted.")

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_exact_tail_global_prediction_seal_v2",
            "status": self.status,
            "config_contract_hash": self.config_contract_hash,
            "reservation_index_hash": self.reservation_index_hash,
            "development_cache_binding_hash": self.development_cache_binding_hash,
            "development_manifest_sha256": self.development_manifest_sha256,
            "target_evaluation_binding_hash": self.target_evaluation_binding_hash,
            "prediction_index_sha256": self.prediction_index_sha256,
            "prediction_arrays_sha256": self.prediction_arrays_sha256,
            "partition_hash_by_center": dict(self.partition_hash_by_center),
            "evaluation_row_hash_by_center": dict(self.evaluation_row_hash_by_center),
            "support_row_hash_by_center": dict(self.support_row_hash_by_center),
            "prediction_cell_count": len(self.cells),
            "cells": [cell.to_payload() for cell in self.cells],
            "all_predictions_materialized": True,
            "development_evaluation_labels_opened": False,
            "target_support_labels_opened": False,
            "target_evaluation_labels_opened": False,
            "label_free_support_probabilities_materialized": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "seal_hash": self.seal_hash}


def build_global_prediction_seal(
    *,
    config_contract_hash: str,
    reservation_index_hash: str,
    development_cache_binding_hash: str,
    development_manifest_sha256: str,
    target_evaluation_binding_hash: str,
    prediction_index_sha256: str,
    prediction_arrays_sha256: str,
    partitions: Mapping[str, DevelopmentPartition],
    cells: Sequence[PredictionCellSeal],
) -> GlobalPredictionSeal:
    normalized = {str(center): partition for center, partition in partitions.items()}
    if tuple(normalized) != CENTERS or any(
        not isinstance(partition, DevelopmentPartition)
        or partition.center != center
        for center, partition in normalized.items()
    ):
        raise ProtocolError("Exact-tail seal requires all development partitions.")
    support_cases: set[str] = set()
    evaluation_cases: set[str] = set()
    target_cases: set[str] = set()
    for partition in normalized.values():
        for values, seen, role in (
            (partition.support_case_ids, support_cases, "support"),
            (
                tuple({row.case_id for row in partition.evaluation_rows}),
                evaluation_cases,
                "development evaluation",
            ),
            (partition.target_evaluation_case_ids, target_cases, "target evaluation"),
        ):
            if set(values) & seen:
                raise ProtocolError(f"Exact-tail {role} cases cross center partitions.")
            seen.update(values)
    if support_cases & evaluation_cases or (
        support_cases | evaluation_cases
    ) & target_cases:
        raise ProtocolError("Exact-tail support/development/target cases overlap.")

    by_key: dict[tuple[str, str, str, int, int], PredictionCellSeal] = {}
    for cell in cells:
        if not isinstance(cell, PredictionCellSeal) or cell.key in by_key:
            raise ProtocolError("Exact-tail seal cells are invalid or duplicated.")
        by_key[cell.key] = cell
    expected = expected_prediction_keys()
    if set(by_key) != set(expected):
        raise ProtocolError("Exact-tail global prediction cells are incomplete.")
    ordered = tuple(by_key[key] for key in expected)
    values: dict[str, object] = {
        "config_contract_hash": str(config_contract_hash),
        "reservation_index_hash": str(reservation_index_hash),
        "development_cache_binding_hash": str(development_cache_binding_hash),
        "development_manifest_sha256": str(development_manifest_sha256),
        "target_evaluation_binding_hash": str(target_evaluation_binding_hash),
        "prediction_index_sha256": str(prediction_index_sha256),
        "prediction_arrays_sha256": str(prediction_arrays_sha256),
        "partition_hash_by_center": {
            center: normalized[center].reservation_hash for center in CENTERS
        },
        "evaluation_row_hash_by_center": {
            center: row_identity_hash(normalized[center].evaluation_rows)
            for center in CENTERS
        },
        "support_row_hash_by_center": {
            center: row_identity_hash(normalized[center].support_rows)
            for center in CENTERS
        },
        "cells": ordered,
        "seal_hash": "",
        "status": GLOBAL_SEAL_STATUS,
    }
    provisional = GlobalPredictionSeal.__new__(GlobalPredictionSeal)
    for field, value in values.items():
        object.__setattr__(provisional, field, value)
    values["seal_hash"] = stable_hash(provisional._unhashed_payload())
    return GlobalPredictionSeal(**values)  # type: ignore[arg-type]


def _require_hash(value: object, role: str, lengths: set[int]) -> None:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProtocolError(f"Exact-tail {role} is malformed.")


__all__ = (
    "GLOBAL_SEAL_STATUS",
    "GlobalPredictionSeal",
    "PredictionCellSeal",
    "build_global_prediction_seal",
)
