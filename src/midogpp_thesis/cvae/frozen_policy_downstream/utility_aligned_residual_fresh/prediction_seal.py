"""Opaque authority issued only after all logical predictions are complete."""

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
    EXPECTED_LOGICAL_PREDICTION_COUNT,
    EvaluationPlan,
    PredictionCell,
)


_ISSUER = object()


@dataclass(frozen=True)
class PredictionSealSummary:
    seal_hash: str
    plan_hash: str
    logical_prediction_count: int
    unique_composition_count: int
    target_count: int
    logical_action_coverage_complete: bool
    row_coverage_complete: bool
    labels_opened: bool = False


@dataclass(frozen=True)
class _SealState:
    plan: EvaluationPlan
    predictions: tuple[PredictionCell, ...]
    row_ids_by_target: Mapping[str, tuple[str, ...]]
    seal_hash: str


class PredictionSealCapability:
    """Non-serializable authority to cross the target-label boundary."""

    __slots__ = ("__state", "__issuer")

    def __init__(self, state: _SealState, issuer: object) -> None:
        if issuer is not _ISSUER or not isinstance(state, _SealState):
            raise TypeError("Prediction seals are issued only by seal_predictions().")
        self.__state = state
        self.__issuer = issuer

    @property
    def seal_hash(self) -> str:
        return self.__state.seal_hash

    @property
    def plan_hash(self) -> str:
        return self.__state.plan.plan_hash

    def __reduce__(self) -> object:
        raise TypeError("Prediction seal capabilities are intentionally opaque.")


def seal_predictions(
    plan: EvaluationPlan,
    predictions: Sequence[PredictionCell | Mapping[object, object] | object],
) -> PredictionSealCapability:
    """Seal B/U/G_delta/R/P/Hxe as distinct logical predictions."""

    if not isinstance(plan, EvaluationPlan):
        raise ProtocolError("Utility-aligned prediction plan is invalid.")
    normalized = tuple(_normalize_prediction(value) for value in predictions)
    if len(normalized) != EXPECTED_LOGICAL_PREDICTION_COUNT:
        raise ProtocolError(
            "Every utility-aligned action/target/seed prediction must exist before labels."
        )
    by_key: dict[tuple[str, int, int, str], PredictionCell] = {}
    for cell in normalized:
        if cell.key in by_key:
            raise ProtocolError("Utility-aligned logical predictions duplicate.")
        by_key[cell.key] = cell
    expected = {cell.key: cell for cell in plan.logical_cells}
    if set(by_key) != set(expected):
        raise ProtocolError("Utility-aligned logical prediction coverage drifted.")

    ordered: list[PredictionCell] = []
    rows_by_target: dict[str, tuple[str, ...]] = {}
    for planned in plan.logical_cells:
        observed = by_key[planned.key]
        if (
            observed.action_hash != planned.action_hash
            or observed.composition_hash != planned.composition_hash
        ):
            raise ProtocolError("Utility-aligned prediction/action binding drifted.")
        expected_rows = plan.evaluation_row_ids_by_target.get(observed.target_center)
        if expected_rows and observed.evaluation_row_ids != expected_rows:
            raise ProtocolError("Utility-aligned prediction row plan drifted.")
        established = rows_by_target.setdefault(
            observed.target_center, observed.evaluation_row_ids
        )
        if established != observed.evaluation_row_ids:
            raise ProtocolError("Utility-aligned row order differs across actions.")
        probabilities = np.ascontiguousarray(
            observed.probabilities.copy(), dtype=np.float32
        )
        probabilities.setflags(write=False)
        ordered.append(
            PredictionCell(
                target_center=observed.target_center,
                training_seed=observed.training_seed,
                generation_seed=observed.generation_seed,
                action_id=observed.action_id,
                action_hash=observed.action_hash,
                composition_hash=observed.composition_hash,
                evaluation_row_ids=observed.evaluation_row_ids,
                probabilities=probabilities,
            )
        )
    if set(rows_by_target) != set(CENTERS):
        raise ProtocolError("Utility-aligned predictions omit a target.")
    seen_rows: set[str] = set()
    for target in CENTERS:
        rows = rows_by_target[target]
        if seen_rows.intersection(rows):
            raise ProtocolError("Utility-aligned target prediction rows overlap.")
        seen_rows.update(rows)
    row_mapping = MappingProxyType(rows_by_target)
    ordered_tuple = tuple(ordered)
    state = _SealState(
        plan=plan,
        predictions=ordered_tuple,
        row_ids_by_target=row_mapping,
        seal_hash=_state_hash(plan, ordered_tuple, row_mapping),
    )
    return PredictionSealCapability(state, _ISSUER)


seal_all_predictions = seal_predictions


def validate_prediction_seal(
    capability: PredictionSealCapability,
    *,
    expected_plan: EvaluationPlan | None = None,
) -> PredictionSealSummary:
    state = read_sealed_prediction_snapshot(capability)
    if expected_plan is not None and expected_plan.plan_hash != state.plan.plan_hash:
        raise ProtocolError("Utility-aligned prediction seal/plan binding drifted.")
    return PredictionSealSummary(
        seal_hash=state.seal_hash,
        plan_hash=state.plan.plan_hash,
        logical_prediction_count=len(state.predictions),
        unique_composition_count=len(state.plan.composition_cells),
        target_count=len(state.row_ids_by_target),
        logical_action_coverage_complete=True,
        row_coverage_complete=True,
    )


def read_sealed_prediction_snapshot(
    capability: PredictionSealCapability,
) -> _SealState:
    if not isinstance(capability, PredictionSealCapability):
        raise ProtocolError("Scoring requires an issued utility-aligned prediction seal.")
    try:
        issuer = capability._PredictionSealCapability__issuer
        state = capability._PredictionSealCapability__state
    except AttributeError as exc:
        raise ProtocolError("Utility-aligned prediction seal is malformed.") from exc
    if (
        issuer is not _ISSUER
        or not isinstance(state, _SealState)
        or state.seal_hash
        != _state_hash(state.plan, state.predictions, state.row_ids_by_target)
    ):
        raise ProtocolError("Utility-aligned prediction seal integrity failed.")
    return state


def _state_hash(
    plan: EvaluationPlan,
    predictions: Sequence[PredictionCell],
    rows_by_target: Mapping[str, tuple[str, ...]],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"midogpp_utility_aligned_prediction_seal_v1\0")
    digest.update(plan.plan_hash.encode("utf-8"))
    for target in CENTERS:
        digest.update(target.encode("utf-8"))
        digest.update(json.dumps(list(rows_by_target[target])).encode("utf-8"))
    for cell in predictions:
        digest.update(
            json.dumps(
                (
                    cell.target_center,
                    cell.training_seed,
                    cell.generation_seed,
                    cell.action_id,
                    cell.action_hash,
                    cell.composition_hash,
                ),
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(_array_sha256(cell.probabilities).encode("ascii"))
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _normalize_prediction(raw: object) -> PredictionCell:
    if isinstance(raw, PredictionCell):
        return raw
    if isinstance(raw, Mapping):
        payload = raw
    else:
        payload = {
            key: getattr(raw, key)
            for key in (
                "target_center",
                "training_seed",
                "generation_seed",
                "action_id",
                "action_hash",
                "composition_hash",
                "evaluation_row_ids",
                "probabilities",
                "labels",
                "target_labels",
                "y_true",
            )
            if hasattr(raw, key)
        }
    if any(key in payload for key in ("labels", "target_labels", "y_true")):
        raise ProtocolError("Prediction sealing cannot accept target labels.")
    try:
        training_seed = int(payload.get("training_seed"))  # type: ignore[arg-type]
        generation_seed = int(payload.get("generation_seed"))  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ProtocolError("Utility-aligned prediction seeds are invalid.") from exc
    return PredictionCell(
        target_center=str(payload.get("target_center", "")),
        training_seed=training_seed,
        generation_seed=generation_seed,
        action_id=str(payload.get("action_id", "")),
        action_hash=str(payload.get("action_hash", "")),
        composition_hash=str(payload.get("composition_hash", "")),
        evaluation_row_ids=tuple(payload.get("evaluation_row_ids", ())),  # type: ignore[arg-type]
        probabilities=np.asarray(payload.get("probabilities")),
    )


__all__ = (
    "PredictionSealCapability",
    "PredictionSealSummary",
    "read_sealed_prediction_snapshot",
    "seal_all_predictions",
    "seal_predictions",
    "validate_prediction_seal",
)
