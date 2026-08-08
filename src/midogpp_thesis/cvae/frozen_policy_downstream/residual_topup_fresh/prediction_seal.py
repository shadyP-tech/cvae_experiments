"""Opaque capability issued only after complete label-free prediction coverage."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .contracts import (
    CENTERS,
    EXPECTED_PLAN_CELL_COUNT,
    EvaluationPlan,
    PredictionCell,
)


_ISSUER = object()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class PredictionSealSummary:
    seal_hash: str
    plan_hash: str
    prediction_cell_count: int
    target_count: int
    action_seed_coverage_complete: bool
    row_coverage_complete: bool
    labels_opened: bool = False


@dataclass(frozen=True)
class _SealState:
    plan: EvaluationPlan
    predictions: tuple[PredictionCell, ...]
    row_ids_by_target: Mapping[str, tuple[str, ...]]
    seal_hash: str


class PredictionSealCapability:
    """Non-serializable authority to join labels to a complete prediction menu.

    The constructor is intentionally unavailable to callers.  Only
    :func:`seal_predictions` can issue a valid instance, and scoring rehashes
    the complete state before accepting it.
    """

    __slots__ = ("__state", "__issuer")

    def __init__(self, state: _SealState, issuer: object) -> None:
        if issuer is not _ISSUER or not isinstance(state, _SealState):
            raise TypeError(
                "PredictionSealCapability is issued only by seal_predictions()."
            )
        self.__state = state
        self.__issuer = issuer

    @property
    def seal_hash(self) -> str:
        return self.__state.seal_hash

    @property
    def plan_hash(self) -> str:
        return self.__state.plan.plan_hash

    @property
    def prediction_cell_count(self) -> int:
        return len(self.__state.predictions)

    def __reduce__(self) -> object:
        raise TypeError("Prediction seal capabilities are intentionally opaque.")


def seal_predictions(
    plan: EvaluationPlan,
    predictions: Sequence[PredictionCell | Mapping[object, object] | object],
) -> PredictionSealCapability:
    """Validate and seal every B/U/G/S/P/Hxe prediction before label access."""

    if not isinstance(plan, EvaluationPlan):
        raise ProtocolError("Fresh Stage-70 prediction plan is invalid.")
    normalized = tuple(_normalize_prediction(value) for value in predictions)
    if len(normalized) != EXPECTED_PLAN_CELL_COUNT:
        raise ProtocolError(
            "Fresh Stage-70 must predict every target/action/seed cell before labels."
        )
    by_key: dict[tuple[str, int, int, str], PredictionCell] = {}
    for cell in normalized:
        if cell.key in by_key:
            raise ProtocolError("Fresh Stage-70 prediction cells duplicate.")
        by_key[cell.key] = cell
    expected_by_key = {cell.key: cell for cell in plan.cells}
    if set(by_key) != set(expected_by_key):
        raise ProtocolError("Fresh Stage-70 prediction action/seed coverage drifted.")

    ordered: list[PredictionCell] = []
    rows_by_target: dict[str, tuple[str, ...]] = {}
    globally_seen_rows: set[str] = set()
    for planned in plan.cells:
        observed = by_key[planned.key]
        if observed.action_hash != planned.action_hash:
            raise ProtocolError("Fresh Stage-70 prediction action hash drifted.")
        expected_rows = plan.evaluation_row_ids_by_target.get(
            observed.target_center
        )
        if expected_rows and observed.evaluation_row_ids != expected_rows:
            raise ProtocolError("Fresh Stage-70 prediction row plan drifted.")
        established = rows_by_target.setdefault(
            observed.target_center, observed.evaluation_row_ids
        )
        if observed.evaluation_row_ids != established:
            raise ProtocolError(
                "Fresh Stage-70 row order differs across actions or seeds."
            )
        # Copy once more at the seal boundary so a caller-owned array cannot
        # be mutated after the capability is issued.
        probabilities = np.ascontiguousarray(
            observed.probabilities.copy(), dtype=np.float64
        )
        probabilities.setflags(write=False)
        ordered.append(
            PredictionCell(
                target_center=observed.target_center,
                training_seed=observed.training_seed,
                generation_seed=observed.generation_seed,
                action_id=observed.action_id,
                action_hash=observed.action_hash,
                evaluation_row_ids=observed.evaluation_row_ids,
                probabilities=probabilities,
            )
        )
    if set(rows_by_target) != set(CENTERS):
        raise ProtocolError("Fresh Stage-70 prediction rows omit a target.")
    for target in CENTERS:
        rows = rows_by_target[target]
        if globally_seen_rows.intersection(rows):
            raise ProtocolError("Fresh Stage-70 target prediction rows overlap.")
        globally_seen_rows.update(rows)

    row_mapping = MappingProxyType(dict(rows_by_target))
    ordered_tuple = tuple(ordered)
    seal_hash = _state_hash(plan, ordered_tuple, row_mapping)
    state = _SealState(
        plan=plan,
        predictions=ordered_tuple,
        row_ids_by_target=row_mapping,
        seal_hash=seal_hash,
    )
    return PredictionSealCapability(state, _ISSUER)


seal_all_predictions = seal_predictions


def validate_prediction_seal(
    capability: PredictionSealCapability,
    *,
    expected_plan: EvaluationPlan | None = None,
) -> PredictionSealSummary:
    state = _open_prediction_seal(capability)
    if expected_plan is not None and (
        not isinstance(expected_plan, EvaluationPlan)
        or expected_plan.plan_hash != state.plan.plan_hash
    ):
        raise ProtocolError("Fresh Stage-70 prediction seal/plan binding drifted.")
    return PredictionSealSummary(
        seal_hash=state.seal_hash,
        plan_hash=state.plan.plan_hash,
        prediction_cell_count=len(state.predictions),
        target_count=len(state.row_ids_by_target),
        action_seed_coverage_complete=True,
        row_coverage_complete=True,
    )


def _open_prediction_seal(capability: PredictionSealCapability) -> _SealState:
    if not isinstance(capability, PredictionSealCapability):
        raise ProtocolError(
            "Fresh Stage-70 scoring requires an issued prediction seal capability."
        )
    try:
        issuer = capability._PredictionSealCapability__issuer
        state = capability._PredictionSealCapability__state
    except AttributeError as exc:
        raise ProtocolError("Fresh Stage-70 prediction seal is malformed.") from exc
    if (
        issuer is not _ISSUER
        or not isinstance(state, _SealState)
        or state.seal_hash
        != _state_hash(state.plan, state.predictions, state.row_ids_by_target)
    ):
        raise ProtocolError("Fresh Stage-70 prediction seal integrity failed.")
    return state


def read_sealed_prediction_snapshot(
    capability: PredictionSealCapability,
) -> _SealState:
    """Package-internal validated reader for an issued immutable seal."""

    return _open_prediction_seal(capability)


def _state_hash(
    plan: EvaluationPlan,
    predictions: Sequence[PredictionCell],
    rows_by_target: Mapping[str, tuple[str, ...]],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"midogpp_residual_topup_fresh_prediction_seal_v1\0")
    digest.update(plan.plan_hash.encode("utf-8"))
    for target in CENTERS:
        digest.update(target.encode("utf-8"))
        digest.update(
            json.dumps(list(rows_by_target[target]), separators=(",", ":")).encode(
                "utf-8"
            )
        )
    for cell in predictions:
        identity = (
            cell.target_center,
            cell.training_seed,
            cell.generation_seed,
            cell.action_id,
            cell.action_hash,
        )
        digest.update(
            json.dumps(identity, separators=(",", ":")).encode("utf-8")
        )
        digest.update(_array_sha256(cell.probabilities).encode("ascii"))
    return digest.hexdigest()


def _normalize_prediction(raw: object) -> PredictionCell:
    if isinstance(raw, PredictionCell):
        return raw
    if isinstance(raw, Mapping):
        payload = raw
    else:
        fields = (
            "target_center",
            "outer_target",
            "training_seed",
            "generation_seed",
            "action_id",
            "action_hash",
            "evaluation_row_ids",
            "row_ids",
            "probabilities",
            "positive_class_probabilities",
            "labels",
            "y_true",
        )
        payload = {
            field: getattr(raw, field)
            for field in fields
            if hasattr(raw, field)
        }
    if "labels" in payload or "y_true" in payload or "target_labels" in payload:
        raise ProtocolError("Prediction sealing cannot accept target labels.")
    probabilities = payload.get(
        "probabilities", payload.get("positive_class_probabilities")
    )
    rows = payload.get("evaluation_row_ids", payload.get("row_ids"))
    if probabilities is None or rows is None:
        raise ProtocolError("Fresh Stage-70 prediction payload is incomplete.")
    try:
        training_seed = int(payload.get("training_seed"))  # type: ignore[arg-type]
        generation_seed = int(payload.get("generation_seed"))  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ProtocolError("Fresh Stage-70 prediction seed is invalid.") from exc
    if isinstance(payload.get("training_seed"), bool) or isinstance(
        payload.get("generation_seed"), bool
    ):
        raise ProtocolError("Fresh Stage-70 prediction seed is invalid.")
    return PredictionCell(
        target_center=str(
            payload.get("target_center", payload.get("outer_target", ""))
        ),
        training_seed=training_seed,
        generation_seed=generation_seed,
        action_id=str(payload.get("action_id", "")),
        action_hash=str(payload.get("action_hash", "")),
        evaluation_row_ids=tuple(rows),  # type: ignore[arg-type]
        probabilities=np.asarray(probabilities),
    )


__all__ = (
    "PredictionSealCapability",
    "PredictionSealSummary",
    "seal_all_predictions",
    "seal_predictions",
    "validate_prediction_seal",
    "read_sealed_prediction_snapshot",
)
