"""Durable prediction and decision capabilities for phased label access."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .contracts import (
    ACTION_IDS,
    ACTION_LIBRARY_HASH,
    CONTROL_ACTION_ID,
    CENTERS,
    GENERATION_SEEDS,
    NONUNIFORM_PASS_RULE,
    SELECTION_OBJECTIVE,
    TRAINING_SEEDS,
    ValidationRowIdentity,
    development_queries,
    legal_sources,
    row_identity_hash,
    target_sources,
)


DEVELOPMENT_SEAL_STATUS = "SEALED_ALL_ACTION_DEVELOPMENT_PREDICTIONS"
ALL_ACTION_TARGET_SEAL_STATUS = (
    "SEALED_ALL_TARGET_ACTION_PREDICTIONS_BEFORE_ANY_LABEL_ACCESS"
)
DECISION_SEAL_STATUS = "SEALED_DIAGNOSTIC_ACTION_SELECTION"
TARGET_SEAL_STATUS = "SEALED_SELECTED_AND_CONTROL_TARGET_PREDICTIONS"


@dataclass(frozen=True)
class PredictionCellSeal:
    """One hash-bound prediction array inside a phase seal."""

    phase: str
    outer_target: str
    query_center: str
    action_id: str
    arm_role: str
    candidate_sources: tuple[str, ...]
    training_seed: int
    generation_seed: int
    evaluation_row_ids: tuple[str, ...]
    evaluation_row_identity_hash: str
    prediction_sha256: str
    probability_sha256: str
    composition_hash: str
    classifier_config_hash: str

    def __post_init__(self) -> None:
        if self.phase not in {"development", "target"}:
            raise ProtocolError("Prediction cell phase is invalid.")
        if self.outer_target not in CENTERS or self.query_center not in CENTERS:
            raise ProtocolError("Prediction cell contains an unknown center.")
        if self.action_id not in ACTION_IDS:
            raise ProtocolError("Prediction cell contains an unknown action.")
        allowed_roles = (
            {"development_action"}
            if self.phase == "development"
            else {"selected", "control"}
        )
        if self.arm_role not in allowed_roles:
            raise ProtocolError("Prediction cell arm role is invalid for its phase.")
        expected_sources = (
            legal_sources(
                outer_target=self.outer_target,
                query_center=self.query_center,
            )
            if self.phase == "development"
            else target_sources(self.outer_target)
        )
        if self.candidate_sources != expected_sources:
            raise ProtocolError(
                "Prediction cell candidate pool violates outer/query exclusions."
            )
        if (
            isinstance(self.training_seed, bool)
            or not isinstance(self.training_seed, int)
            or self.training_seed not in TRAINING_SEEDS
        ):
            raise ProtocolError("Prediction cell training seed drifted.")
        if (
            isinstance(self.generation_seed, bool)
            or not isinstance(self.generation_seed, int)
            or self.generation_seed not in GENERATION_SEEDS
        ):
            raise ProtocolError("Prediction cell generation seed drifted.")
        if not self.evaluation_row_ids or len(self.evaluation_row_ids) != len(
            set(self.evaluation_row_ids)
        ):
            raise ProtocolError("Prediction cell row identities are empty or duplicated.")
        for role, value in (
            ("evaluation row-identity hash", self.evaluation_row_identity_hash),
            ("prediction SHA-256", self.prediction_sha256),
            ("probability SHA-256", self.probability_sha256),
            ("composition hash", self.composition_hash),
            ("classifier hash", self.classifier_config_hash),
        ):
            _require_hash(value, role)

    @property
    def development_key(self) -> tuple[str, str, int, int]:
        return (
            self.action_id,
            self.query_center,
            self.training_seed,
            self.generation_seed,
        )

    @property
    def target_key(self) -> tuple[str, int, int]:
        return self.arm_role, self.training_seed, self.generation_seed

    @property
    def all_action_target_key(self) -> tuple[str, str, str, int, int]:
        return (
            self.outer_target,
            self.arm_role,
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
            "candidate_sources": list(self.candidate_sources),
            "training_seed": self.training_seed,
            "generation_seed": self.generation_seed,
            "row_count": len(self.evaluation_row_ids),
            "evaluation_row_ids": list(self.evaluation_row_ids),
            "evaluation_row_identity_hash": self.evaluation_row_identity_hash,
            "prediction_sha256": self.prediction_sha256,
            "probability_sha256": self.probability_sha256,
            "composition_hash": self.composition_hash,
            "classifier_config_hash": self.classifier_config_hash,
        }


@dataclass(frozen=True)
class DevelopmentPredictionSeal:
    outer_target: str
    config_contract_hash: str
    action_library_hash: str
    support_partition_lock_hash: str
    validation_cache_binding_hash: str
    validation_manifest_sha256: str
    prediction_index_sha256: str
    prediction_arrays_sha256: str
    evaluation_row_ids_by_query: Mapping[str, tuple[str, ...]]
    evaluation_row_identity_hash_by_query: Mapping[str, str]
    cells: tuple[PredictionCellSeal, ...]
    seal_hash: str
    status: str = DEVELOPMENT_SEAL_STATUS

    def __post_init__(self) -> None:
        if self.outer_target not in CENTERS:
            raise ProtocolError("Development seal contains an unknown outer target.")
        if self.status != DEVELOPMENT_SEAL_STATUS:
            raise ProtocolError("Development prediction seal is not complete.")
        if self.action_library_hash != ACTION_LIBRARY_HASH:
            raise ProtocolError("Development seal action library drifted.")
        for role, value in (
            ("config contract hash", self.config_contract_hash),
            ("support partition-lock hash", self.support_partition_lock_hash),
            ("validation-cache binding hash", self.validation_cache_binding_hash),
            ("validation manifest SHA-256", self.validation_manifest_sha256),
            ("development prediction-index SHA-256", self.prediction_index_sha256),
            ("development prediction-array SHA-256", self.prediction_arrays_sha256),
        ):
            _require_hash(value, role)

        expected_queries = development_queries(self.outer_target)
        rows_by_query = {
            str(query): tuple(str(value) for value in rows)
            for query, rows in self.evaluation_row_ids_by_query.items()
        }
        hashes_by_query = {
            str(query): str(value)
            for query, value in self.evaluation_row_identity_hash_by_query.items()
        }
        if set(rows_by_query) != set(expected_queries) or set(hashes_by_query) != set(
            expected_queries
        ):
            raise ProtocolError("Development seal query coverage drifted.")
        all_ids: list[str] = []
        for query in expected_queries:
            row_ids = rows_by_query[query]
            if not row_ids or len(row_ids) != len(set(row_ids)):
                raise ProtocolError("Development seal query rows are empty or duplicated.")
            _require_hash(
                hashes_by_query[query],
                f"development query {query} row-identity hash",
            )
            all_ids.extend(row_ids)
        if len(all_ids) != len(set(all_ids)):
            raise ProtocolError("Development seal reuses rows across pseudo-targets.")

        expected_keys = {
            (action, query, train_seed, generation_seed)
            for action in ACTION_IDS
            for query in expected_queries
            for train_seed in TRAINING_SEEDS
            for generation_seed in GENERATION_SEEDS
        }
        observed: dict[tuple[str, str, int, int], PredictionCellSeal] = {}
        for cell in self.cells:
            if (
                cell.phase != "development"
                or cell.arm_role != "development_action"
                or cell.outer_target != self.outer_target
                or cell.query_center == self.outer_target
                or cell.evaluation_row_ids != rows_by_query.get(cell.query_center)
                or cell.evaluation_row_identity_hash
                != hashes_by_query.get(cell.query_center)
            ):
                raise ProtocolError("Development prediction cell escaped its seal geometry.")
            if cell.development_key in observed:
                raise ProtocolError("Development prediction seal duplicates a cell.")
            observed[cell.development_key] = cell
        if set(observed) != expected_keys:
            raise ProtocolError(
                "Development prediction seal lacks complete all-action coverage."
            )

        object.__setattr__(self, "evaluation_row_ids_by_query", MappingProxyType(rows_by_query))
        object.__setattr__(
            self,
            "evaluation_row_identity_hash_by_query",
            MappingProxyType(hashes_by_query),
        )
        if self.seal_hash != stable_hash(self._unhashed_payload()):
            raise ProtocolError("Development prediction seal hash drifted.")

    @property
    def row_count(self) -> int:
        return sum(len(rows) for rows in self.evaluation_row_ids_by_query.values())

    @property
    def cell_count(self) -> int:
        return len(self.cells)

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_dense_residual_development_prediction_seal_v1",
            "status": self.status,
            "outer_target": self.outer_target,
            "config_contract_hash": self.config_contract_hash,
            "action_library_hash": self.action_library_hash,
            "support_partition_lock_hash": self.support_partition_lock_hash,
            "validation_cache_binding_hash": self.validation_cache_binding_hash,
            "validation_manifest_sha256": self.validation_manifest_sha256,
            "prediction_index_sha256": self.prediction_index_sha256,
            "prediction_arrays_sha256": self.prediction_arrays_sha256,
            "evaluation_row_ids_by_query": {
                query: list(self.evaluation_row_ids_by_query[query])
                for query in development_queries(self.outer_target)
            },
            "evaluation_row_identity_hash_by_query": {
                query: self.evaluation_row_identity_hash_by_query[query]
                for query in development_queries(self.outer_target)
            },
            "cell_count": len(self.cells),
            "cells": [cell.to_payload() for cell in self.cells],
            "development_labels_opened": False,
            "target_labels_opened": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "seal_hash": self.seal_hash}


@dataclass(frozen=True)
class AllActionTargetPredictionSeal:
    """One global pre-label capability over every target action and seed cell."""

    config_contract_hash: str
    action_library_hash: str
    support_partition_lock_hash: str
    compatibility_index_hash: str
    validation_cache_binding_hash: str
    validation_manifest_sha256: str
    prediction_index_sha256: str
    prediction_arrays_sha256: str
    evaluation_row_ids_by_target: Mapping[str, tuple[str, ...]]
    evaluation_row_identity_hash_by_target: Mapping[str, str]
    cells: tuple[PredictionCellSeal, ...]
    seal_hash: str
    status: str = ALL_ACTION_TARGET_SEAL_STATUS

    def __post_init__(self) -> None:
        rows_by_target = {
            str(target): tuple(str(value) for value in rows)
            for target, rows in self.evaluation_row_ids_by_target.items()
        }
        hashes_by_target = {
            str(target): str(value)
            for target, value in self.evaluation_row_identity_hash_by_target.items()
        }
        object.__setattr__(
            self, "evaluation_row_ids_by_target", MappingProxyType(rows_by_target)
        )
        object.__setattr__(
            self,
            "evaluation_row_identity_hash_by_target",
            MappingProxyType(hashes_by_target),
        )
        object.__setattr__(self, "cells", tuple(self.cells))
        self.verify_complete()

    @property
    def row_count(self) -> int:
        return sum(len(rows) for rows in self.evaluation_row_ids_by_target.values())

    @property
    def cell_count(self) -> int:
        return len(self.cells)

    def verify_complete(self) -> None:
        """Revalidate completeness and hash after any illicit in-memory mutation."""

        if self.status != ALL_ACTION_TARGET_SEAL_STATUS:
            raise ProtocolError("All-target-action prediction seal is not complete.")
        if self.action_library_hash != ACTION_LIBRARY_HASH:
            raise ProtocolError("All-target-action seal action library drifted.")
        for role, value in (
            ("config contract hash", self.config_contract_hash),
            ("support partition-lock hash", self.support_partition_lock_hash),
            ("compatibility-index hash", self.compatibility_index_hash),
            ("validation-cache binding hash", self.validation_cache_binding_hash),
            ("validation manifest SHA-256", self.validation_manifest_sha256),
            ("target prediction-index SHA-256", self.prediction_index_sha256),
            ("target prediction-array SHA-256", self.prediction_arrays_sha256),
        ):
            _require_hash(value, role)

        rows_by_target = self.evaluation_row_ids_by_target
        hashes_by_target = self.evaluation_row_identity_hash_by_target
        if tuple(rows_by_target) != CENTERS or tuple(hashes_by_target) != CENTERS:
            raise ProtocolError(
                "All-target-action seal lacks canonical nine-target identity coverage."
            )
        all_row_ids: list[str] = []
        for target in CENTERS:
            row_ids = rows_by_target[target]
            if not row_ids or len(row_ids) != len(set(row_ids)):
                raise ProtocolError(
                    "All-target-action seal target rows are empty or duplicated."
                )
            _require_hash(
                hashes_by_target[target],
                f"target {target} evaluation row-identity hash",
            )
            all_row_ids.extend(row_ids)
        if len(all_row_ids) != len(set(all_row_ids)):
            raise ProtocolError(
                "All-target-action seal reuses evaluation rows across targets."
            )

        expected_keys = _all_action_target_keys()
        observed_keys: list[tuple[str, str, str, int, int]] = []
        for cell in self.cells:
            if not isinstance(cell, PredictionCellSeal):
                raise ProtocolError(
                    "All-target-action seal contains a non-prediction cell."
                )
            target = cell.outer_target
            if (
                cell.phase != "target"
                or cell.query_center != target
                or cell.arm_role not in {"selected", "control"}
                or (
                    cell.arm_role == "selected"
                    and cell.action_id not in ACTION_IDS
                )
                or (
                    cell.arm_role == "control"
                    and cell.action_id != CONTROL_ACTION_ID
                )
                or cell.evaluation_row_ids != rows_by_target.get(target)
                or cell.evaluation_row_identity_hash
                != hashes_by_target.get(target)
            ):
                raise ProtocolError(
                    "All-target-action prediction cell escaped its global geometry."
                )
            observed_keys.append(cell.all_action_target_key)
        if tuple(observed_keys) != expected_keys:
            raise ProtocolError(
                "All-target-action prediction seal lacks complete all-target-action "
                "coverage."
            )
        if self.seal_hash != stable_hash(self._unhashed_payload()):
            raise ProtocolError("All-target-action prediction seal hash drifted.")

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": (
                "midogpp_dense_residual_all_action_target_prediction_seal_v1"
            ),
            "status": self.status,
            "config_contract_hash": self.config_contract_hash,
            "action_library_hash": self.action_library_hash,
            "support_partition_lock_hash": self.support_partition_lock_hash,
            "compatibility_index_hash": self.compatibility_index_hash,
            "validation_cache_binding_hash": self.validation_cache_binding_hash,
            "validation_manifest_sha256": self.validation_manifest_sha256,
            "prediction_index_sha256": self.prediction_index_sha256,
            "prediction_arrays_sha256": self.prediction_arrays_sha256,
            "evaluation_row_ids_by_target": {
                target: list(self.evaluation_row_ids_by_target[target])
                for target in CENTERS
            },
            "evaluation_row_identity_hash_by_target": {
                target: self.evaluation_row_identity_hash_by_target[target]
                for target in CENTERS
            },
            "target_count": len(CENTERS),
            "selected_action_count_per_target": len(ACTION_IDS),
            "separate_control_count_per_target": 1,
            "seed_cell_count_per_arm": len(TRAINING_SEEDS)
            * len(GENERATION_SEEDS),
            "cell_count": len(self.cells),
            "cells": [cell.to_payload() for cell in self.cells],
            "development_labels_opened": False,
            "target_labels_opened": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "seal_hash": self.seal_hash}


@dataclass(frozen=True)
class DiagnosticDecisionSeal:
    outer_target: str
    config_contract_hash: str
    development_prediction_seal_hash: str
    development_label_vector_hash: str
    development_metrics_sha256: str
    action_summaries_sha256: str
    selected_action_id: str
    selected_rho: float
    selected_mean_paired_bacc_delta_vs_control: float
    fallback_applied: bool
    fallback_reason: str
    decision_hash: str
    status: str = DECISION_SEAL_STATUS

    def __post_init__(self) -> None:
        if self.outer_target not in CENTERS or self.status != DECISION_SEAL_STATUS:
            raise ProtocolError("Diagnostic decision seal identity/status drifted.")
        if self.selected_action_id not in ACTION_IDS:
            raise ProtocolError("Diagnostic decision seal selected an unknown action.")
        expected_rho = float(self.selected_action_id.removeprefix("rho_"))
        if self.selected_rho != expected_rho:
            raise ProtocolError("Diagnostic decision action/rho binding drifted.")
        for role, value in (
            ("config contract hash", self.config_contract_hash),
            ("development seal hash", self.development_prediction_seal_hash),
            ("development label-vector hash", self.development_label_vector_hash),
            ("development metrics SHA-256", self.development_metrics_sha256),
            ("action summaries SHA-256", self.action_summaries_sha256),
        ):
            _require_hash(value, role)
        delta = float(self.selected_mean_paired_bacc_delta_vs_control)
        if not math.isfinite(delta):
            raise ProtocolError("Diagnostic decision paired delta must be finite.")
        if self.selected_action_id == CONTROL_ACTION_ID:
            if delta != 0.0:
                raise ProtocolError("Control decision must report zero paired delta.")
        elif delta <= 0.0 or self.fallback_applied:
            raise ProtocolError(
                "Nonuniform diagnostic action failed its positive-mean-delta gate."
            )
        if self.fallback_applied and self.selected_action_id != CONTROL_ACTION_ID:
            raise ProtocolError("Diagnostic fallback must select exact rho0 control.")
        if self.fallback_applied and not self.fallback_reason:
            raise ProtocolError("Diagnostic fallback requires a recorded reason.")
        if not self.fallback_applied and self.fallback_reason:
            raise ProtocolError("Diagnostic non-fallback decision has a fallback reason.")
        if self.decision_hash != stable_hash(self._unhashed_payload()):
            raise ProtocolError("Diagnostic decision seal hash drifted.")

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_dense_residual_diagnostic_decision_seal_v1",
            "status": self.status,
            "outer_target": self.outer_target,
            "config_contract_hash": self.config_contract_hash,
            "development_prediction_seal_hash": self.development_prediction_seal_hash,
            "development_label_vector_hash": self.development_label_vector_hash,
            "development_metrics_sha256": self.development_metrics_sha256,
            "action_summaries_sha256": self.action_summaries_sha256,
            "selection_objective": SELECTION_OBJECTIVE,
            "nonuniform_pass_rule": NONUNIFORM_PASS_RULE,
            "selected_action_id": self.selected_action_id,
            "selected_rho": self.selected_rho,
            "selected_mean_paired_bacc_delta_vs_control": (
                self.selected_mean_paired_bacc_delta_vs_control
            ),
            "fallback_applied": self.fallback_applied,
            "fallback_reason": self.fallback_reason,
            "target_labels_opened": False,
            "diagnostic_only": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "decision_hash": self.decision_hash}


@dataclass(frozen=True)
class TargetPredictionSeal:
    outer_target: str
    config_contract_hash: str
    diagnostic_decision_hash: str
    selected_action_id: str
    validation_cache_binding_hash: str
    validation_manifest_sha256: str
    prediction_index_sha256: str
    prediction_arrays_sha256: str
    evaluation_row_ids: tuple[str, ...]
    evaluation_row_identity_hash: str
    cells: tuple[PredictionCellSeal, ...]
    seal_hash: str
    status: str = TARGET_SEAL_STATUS

    def __post_init__(self) -> None:
        if self.outer_target not in CENTERS or self.status != TARGET_SEAL_STATUS:
            raise ProtocolError("Target prediction seal identity/status drifted.")
        if self.selected_action_id not in ACTION_IDS:
            raise ProtocolError("Target prediction seal contains an unknown action.")
        for role, value in (
            ("config contract hash", self.config_contract_hash),
            ("diagnostic decision hash", self.diagnostic_decision_hash),
            ("validation-cache binding hash", self.validation_cache_binding_hash),
            ("validation manifest SHA-256", self.validation_manifest_sha256),
            ("target prediction-index SHA-256", self.prediction_index_sha256),
            ("target prediction-array SHA-256", self.prediction_arrays_sha256),
        ):
            _require_hash(value, role)
        if not self.evaluation_row_ids or len(self.evaluation_row_ids) != len(
            set(self.evaluation_row_ids)
        ):
            raise ProtocolError("Target prediction seal rows are empty or duplicated.")
        _require_hash(
            self.evaluation_row_identity_hash,
            "target evaluation row-identity hash",
        )

        expected_keys = {
            (role, training_seed, generation_seed)
            for role in ("selected", "control")
            for training_seed in TRAINING_SEEDS
            for generation_seed in GENERATION_SEEDS
        }
        observed: dict[tuple[str, int, int], PredictionCellSeal] = {}
        for cell in self.cells:
            expected_action = (
                self.selected_action_id if cell.arm_role == "selected" else CONTROL_ACTION_ID
            )
            if (
                cell.phase != "target"
                or cell.outer_target != self.outer_target
                or cell.query_center != self.outer_target
                or cell.action_id != expected_action
                or cell.evaluation_row_ids != self.evaluation_row_ids
                or cell.evaluation_row_identity_hash != self.evaluation_row_identity_hash
            ):
                raise ProtocolError("Target prediction cell escaped its seal geometry.")
            if cell.target_key in observed:
                raise ProtocolError("Target prediction seal duplicates a logical arm cell.")
            observed[cell.target_key] = cell
        if set(observed) != expected_keys:
            raise ProtocolError(
                "Target seal lacks selected-plus-control coverage for all seed cells."
            )
        if self.seal_hash != stable_hash(self._unhashed_payload()):
            raise ProtocolError("Target prediction seal hash drifted.")

    @property
    def row_count(self) -> int:
        return len(self.evaluation_row_ids)

    @property
    def cell_count(self) -> int:
        return len(self.cells)

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_dense_residual_target_prediction_seal_v1",
            "status": self.status,
            "outer_target": self.outer_target,
            "config_contract_hash": self.config_contract_hash,
            "diagnostic_decision_hash": self.diagnostic_decision_hash,
            "selected_action_id": self.selected_action_id,
            "control_action_id": CONTROL_ACTION_ID,
            "validation_cache_binding_hash": self.validation_cache_binding_hash,
            "validation_manifest_sha256": self.validation_manifest_sha256,
            "prediction_index_sha256": self.prediction_index_sha256,
            "prediction_arrays_sha256": self.prediction_arrays_sha256,
            "evaluation_row_ids": list(self.evaluation_row_ids),
            "evaluation_row_identity_hash": self.evaluation_row_identity_hash,
            "cell_count": len(self.cells),
            "cells": [cell.to_payload() for cell in self.cells],
            "target_labels_opened": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "seal_hash": self.seal_hash}


def build_development_prediction_seal(
    *,
    outer_target: str,
    config_contract_hash: str,
    support_partition_lock_hash: str,
    validation_cache_binding_hash: str,
    validation_manifest_sha256: str,
    prediction_index_sha256: str,
    prediction_arrays_sha256: str,
    evaluation_rows_by_query: Mapping[str, Sequence[ValidationRowIdentity]],
    cells: Sequence[PredictionCellSeal],
) -> DevelopmentPredictionSeal:
    rows_by_query = {
        str(query): tuple(row.sample_id for row in rows)
        for query, rows in evaluation_rows_by_query.items()
    }
    identity_hashes = {
        str(query): row_identity_hash(tuple(rows))
        for query, rows in evaluation_rows_by_query.items()
    }
    normalized_cells = tuple(cells)
    provisional = DevelopmentPredictionSeal.__new__(DevelopmentPredictionSeal)
    values = {
        "outer_target": str(outer_target),
        "config_contract_hash": str(config_contract_hash),
        "action_library_hash": ACTION_LIBRARY_HASH,
        "support_partition_lock_hash": str(support_partition_lock_hash),
        "validation_cache_binding_hash": str(validation_cache_binding_hash),
        "validation_manifest_sha256": str(validation_manifest_sha256),
        "prediction_index_sha256": str(prediction_index_sha256),
        "prediction_arrays_sha256": str(prediction_arrays_sha256),
        "evaluation_row_ids_by_query": rows_by_query,
        "evaluation_row_identity_hash_by_query": identity_hashes,
        "cells": normalized_cells,
        "seal_hash": "",
        "status": DEVELOPMENT_SEAL_STATUS,
    }
    for field, value in values.items():
        object.__setattr__(provisional, field, value)
    if any(
        row.partition_role != "evaluation" or row.center != query
        for query, rows in evaluation_rows_by_query.items()
        for row in rows
    ):
        raise ProtocolError("Development seal may bind evaluation rows for q only.")
    if len(identity_hashes) != len(evaluation_rows_by_query):
        raise ProtocolError("Development row identity coverage drifted.")
    values["seal_hash"] = stable_hash(provisional._unhashed_payload())
    return DevelopmentPredictionSeal(**values)  # type: ignore[arg-type]


def build_all_action_target_prediction_seal(
    *,
    config_contract_hash: str,
    support_partition_lock_hash: str,
    compatibility_index_hash: str,
    validation_cache_binding_hash: str,
    validation_manifest_sha256: str,
    prediction_index_sha256: str,
    prediction_arrays_sha256: str,
    evaluation_rows_by_target: Mapping[
        str, Sequence[ValidationRowIdentity]
    ],
    cells: Sequence[PredictionCellSeal],
) -> AllActionTargetPredictionSeal:
    """Build the global target prediction capability before any label access."""

    normalized_rows = {
        str(target): tuple(rows)
        for target, rows in evaluation_rows_by_target.items()
    }
    if tuple(normalized_rows) != CENTERS:
        raise ProtocolError(
            "All-target-action seal requires canonical rows for all nine targets."
        )
    if any(
        not rows
        or any(
            row.partition_role != "evaluation" or row.center != target
            for row in rows
        )
        for target, rows in normalized_rows.items()
    ):
        raise ProtocolError(
            "All-target-action seal may bind target evaluation rows only."
        )
    rows_by_target = {
        target: tuple(row.sample_id for row in normalized_rows[target])
        for target in CENTERS
    }
    identity_hashes = {
        target: row_identity_hash(normalized_rows[target]) for target in CENTERS
    }
    cell_by_key: dict[
        tuple[str, str, str, int, int], PredictionCellSeal
    ] = {}
    for cell in cells:
        if not isinstance(cell, PredictionCellSeal):
            raise ProtocolError(
                "All-target-action seal contains a non-prediction cell."
            )
        key = cell.all_action_target_key
        if key in cell_by_key:
            raise ProtocolError("All-target-action seal duplicates a target cell.")
        cell_by_key[key] = cell
    expected_keys = _all_action_target_keys()
    if set(cell_by_key) != set(expected_keys):
        raise ProtocolError(
            "All-target-action prediction seal lacks complete all-target-action "
            "coverage."
        )
    normalized_cells = tuple(cell_by_key[key] for key in expected_keys)
    values: dict[str, object] = {
        "config_contract_hash": str(config_contract_hash),
        "action_library_hash": ACTION_LIBRARY_HASH,
        "support_partition_lock_hash": str(support_partition_lock_hash),
        "compatibility_index_hash": str(compatibility_index_hash),
        "validation_cache_binding_hash": str(validation_cache_binding_hash),
        "validation_manifest_sha256": str(validation_manifest_sha256),
        "prediction_index_sha256": str(prediction_index_sha256),
        "prediction_arrays_sha256": str(prediction_arrays_sha256),
        "evaluation_row_ids_by_target": rows_by_target,
        "evaluation_row_identity_hash_by_target": identity_hashes,
        "cells": normalized_cells,
        "seal_hash": "",
        "status": ALL_ACTION_TARGET_SEAL_STATUS,
    }
    provisional = AllActionTargetPredictionSeal.__new__(
        AllActionTargetPredictionSeal
    )
    for field, value in values.items():
        object.__setattr__(provisional, field, value)
    values["seal_hash"] = stable_hash(provisional._unhashed_payload())
    return AllActionTargetPredictionSeal(**values)  # type: ignore[arg-type]


def build_diagnostic_decision_seal(
    *,
    outer_target: str,
    config_contract_hash: str,
    development_prediction_seal_hash: str,
    development_label_vector_hash: str,
    development_metrics_sha256: str,
    action_summaries_sha256: str,
    selected_action_id: str,
    selected_mean_paired_bacc_delta_vs_control: float,
    fallback_applied: bool,
    fallback_reason: str = "",
) -> DiagnosticDecisionSeal:
    action_id = str(selected_action_id)
    rho = float(action_id.removeprefix("rho_")) if action_id in ACTION_IDS else -1.0
    values: dict[str, object] = {
        "outer_target": str(outer_target),
        "config_contract_hash": str(config_contract_hash),
        "development_prediction_seal_hash": str(
            development_prediction_seal_hash
        ),
        "development_label_vector_hash": str(development_label_vector_hash),
        "development_metrics_sha256": str(development_metrics_sha256),
        "action_summaries_sha256": str(action_summaries_sha256),
        "selected_action_id": action_id,
        "selected_rho": rho,
        "selected_mean_paired_bacc_delta_vs_control": float(
            selected_mean_paired_bacc_delta_vs_control
        ),
        "fallback_applied": bool(fallback_applied),
        "fallback_reason": str(fallback_reason),
        "decision_hash": "",
        "status": DECISION_SEAL_STATUS,
    }
    provisional = DiagnosticDecisionSeal.__new__(DiagnosticDecisionSeal)
    for field, value in values.items():
        object.__setattr__(provisional, field, value)
    values["decision_hash"] = stable_hash(provisional._unhashed_payload())
    return DiagnosticDecisionSeal(**values)  # type: ignore[arg-type]


def build_target_prediction_seal(
    *,
    outer_target: str,
    config_contract_hash: str,
    diagnostic_decision_hash: str,
    selected_action_id: str,
    validation_cache_binding_hash: str,
    validation_manifest_sha256: str,
    prediction_index_sha256: str,
    prediction_arrays_sha256: str,
    evaluation_rows: Sequence[ValidationRowIdentity],
    cells: Sequence[PredictionCellSeal],
) -> TargetPredictionSeal:
    rows = tuple(evaluation_rows)
    if any(
        row.partition_role != "evaluation" or row.center != str(outer_target)
        for row in rows
    ):
        raise ProtocolError("Target seal may bind target-H evaluation rows only.")
    row_ids = tuple(row.sample_id for row in rows)
    values: dict[str, object] = {
        "outer_target": str(outer_target),
        "config_contract_hash": str(config_contract_hash),
        "diagnostic_decision_hash": str(diagnostic_decision_hash),
        "selected_action_id": str(selected_action_id),
        "validation_cache_binding_hash": str(validation_cache_binding_hash),
        "validation_manifest_sha256": str(validation_manifest_sha256),
        "prediction_index_sha256": str(prediction_index_sha256),
        "prediction_arrays_sha256": str(prediction_arrays_sha256),
        "evaluation_row_ids": row_ids,
        "evaluation_row_identity_hash": row_identity_hash(rows),
        "cells": tuple(cells),
        "seal_hash": "",
        "status": TARGET_SEAL_STATUS,
    }
    provisional = TargetPredictionSeal.__new__(TargetPredictionSeal)
    for field, value in values.items():
        object.__setattr__(provisional, field, value)
    values["seal_hash"] = stable_hash(provisional._unhashed_payload())
    return TargetPredictionSeal(**values)  # type: ignore[arg-type]


def _require_hash(value: str, role: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) not in {16, 64}
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProtocolError(f"Dense residual {role} is malformed.")


def _all_action_target_keys() -> tuple[tuple[str, str, str, int, int], ...]:
    arms = tuple(("selected", action_id) for action_id in ACTION_IDS) + (
        ("control", CONTROL_ACTION_ID),
    )
    return tuple(
        (target, arm_role, action_id, training_seed, generation_seed)
        for target in CENTERS
        for arm_role, action_id in arms
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
    )


__all__ = (
    "ALL_ACTION_TARGET_SEAL_STATUS",
    "DECISION_SEAL_STATUS",
    "DEVELOPMENT_SEAL_STATUS",
    "TARGET_SEAL_STATUS",
    "AllActionTargetPredictionSeal",
    "DevelopmentPredictionSeal",
    "DiagnosticDecisionSeal",
    "PredictionCellSeal",
    "TargetPredictionSeal",
    "build_all_action_target_prediction_seal",
    "build_development_prediction_seal",
    "build_diagnostic_decision_seal",
    "build_target_prediction_seal",
)
