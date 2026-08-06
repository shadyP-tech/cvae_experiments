"""Label-blind classifier fitting and prediction for frozen Stage-70 arms."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from ...real_features.classifier_reference.classifiers import (
    ClassifierSpec,
    FittedClassifierResult,
    fit_logistic_classifier,
)
from ..protocol import ProtocolError
from .composition import compose_policy_replicate
from .contracts import (
    CONTROL_ARM,
    EXPECTED_METRIC_ROWS,
    UTILITY_ARM,
    PolicyReplicate,
    PredictionCell,
    SyntheticComposition,
    TargetFrame,
    array_sha256,
)


@dataclass(frozen=True)
class CompositionRecord:
    policy_id: str
    target_center: str
    training_seed: int
    generation_seed: int
    replicate_id: str
    policy_lock_hash: str
    assignment_table_hash: str
    composition_manifest_hash: str
    train_content_sha256: str
    pre_shuffle_sha256_by_label: Mapping[str, str]
    post_shuffle_sha256_by_label: Mapping[str, str]


@dataclass(frozen=True)
class FrozenPolicyPredictionPass:
    """Complete label-free predictions for all frozen arm/seed cells."""

    cells: tuple[PredictionCell, ...]
    compositions: tuple[CompositionRecord, ...]
    classifier_fit_count: int
    prediction_reuse_count: int
    phase: str = "PREDICTIONS_COMPUTED_LABELS_SEALED"

    def __post_init__(self) -> None:
        if len(self.cells) != EXPECTED_METRIC_ROWS or len(self.compositions) != EXPECTED_METRIC_ROWS:
            raise ProtocolError("Stage-70 prediction pass must cover all 243 arm cells.")
        keys = {
            (cell.policy_id, cell.target_center, cell.training_seed, cell.generation_seed)
            for cell in self.cells
        }
        if len(keys) != EXPECTED_METRIC_ROWS:
            raise ProtocolError("Stage-70 prediction cells are incomplete or duplicated.")
        if self.classifier_fit_count + self.prediction_reuse_count != EXPECTED_METRIC_ROWS:
            raise ProtocolError("Stage-70 fit/reuse accounting drifted.")


@dataclass(frozen=True)
class PersistedPredictionPass:
    """Immutable capability naming the exact durable prediction transaction.

    It intentionally carries no ``FrozenPolicyPredictionPass`` and therefore no
    mutable ndarray that could be substituted after disk verification.
    """

    artifact_root: Path
    authorization_binding_hash: str
    phase_01_sha256: str
    prediction_index_sha256: str
    prediction_arrays_sha256: str
    prediction_seal_sha256: str
    phase_02_sha256: str
    phase: str = "PREDICTIONS_PERSISTED"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.artifact_root, Path)
            or self.artifact_root.is_symlink()
            or len(self.authorization_binding_hash) != 16
            or any(
                len(value) != 64
                for value in (
                    self.phase_01_sha256,
                    self.prediction_index_sha256,
                    self.prediction_arrays_sha256,
                    self.prediction_seal_sha256,
                    self.phase_02_sha256,
                )
            )
            or self.phase != "PREDICTIONS_PERSISTED"
        ):
            raise ProtocolError("Stage-70 prediction seal lacks SHA-256 identities.")


def run_label_free_prediction_pass(
    *,
    replicates: Sequence[PolicyReplicate],
    source_blocks: Mapping[str, object],
    target_frames: Mapping[str, TargetFrame],
    classifier_spec: ClassifierSpec,
    classifier_fit: Callable[..., FittedClassifierResult] = fit_logistic_classifier,
) -> FrozenPolicyPredictionPass:
    """Fit/predict every retained cell without accepting a label object."""

    ordered = sorted(
        replicates,
        key=lambda row: (
            (CONTROL_ARM, "metadata_max_tie_union", UTILITY_ARM).index(row.policy_id),
            row.training_seed,
            row.generation_seed,
            row.target_center,
        ),
    )
    compositions: list[CompositionRecord] = []
    cells: list[PredictionCell] = []
    reusable: dict[tuple[str, str, str, str, int, int], PredictionCell] = {}
    fit_count = 0
    reuse_count = 0
    for replicate in ordered:
        try:
            target = target_frames[replicate.target_center]
        except KeyError as exc:
            raise ProtocolError(
                f"Stage-70 target frame is missing center {replicate.target_center}."
            ) from exc
        composition = compose_policy_replicate(replicate, source_blocks)  # type: ignore[arg-type]
        compositions.append(
            CompositionRecord(
                policy_id=replicate.policy_id,
                target_center=replicate.target_center,
                training_seed=replicate.training_seed,
                generation_seed=replicate.generation_seed,
                replicate_id=replicate.replicate_id,
                policy_lock_hash=replicate.policy_lock_hash,
                assignment_table_hash=replicate.assignment_table_hash,
                composition_manifest_hash=composition.composition_manifest_hash,
                train_content_sha256=composition.train_content_sha256,
                pre_shuffle_sha256_by_label=dict(
                    composition.pre_shuffle_sha256_by_label
                ),
                post_shuffle_sha256_by_label=dict(
                    composition.post_shuffle_sha256_by_label
                ),
            )
        )
        signature = (
            composition.train_content_sha256,
            classifier_spec.config_hash,
            target.content_hash,
            replicate.target_center,
            replicate.training_seed,
            replicate.generation_seed,
        )
        reused = reusable.get(signature)
        may_reuse = replicate.policy_id == UTILITY_ARM and all(
            assignment.exact_equal_union_fallback
            and assignment.assignment_id == assignment.equal_union_assignment_id
            for assignment in replicate.assignments
        )
        if reused is not None and may_reuse:
            if reused.policy_id != CONTROL_ARM:
                raise ProtocolError("Utility fallback may reuse only equal-union predictions.")
            predictions = np.array(reused.predictions, copy=True)
            probabilities = np.array(reused.probabilities, copy=True)
            classifier_hash = reused.classifier_config_hash
            scaler_hash = reused.scaler_state_hash
            reused_from = CONTROL_ARM
            reuse_count += 1
        else:
            fitted = classifier_fit(
                composition.embeddings,
                composition.labels,
                target.embeddings,
                spec=classifier_spec,
            )
            if tuple(int(value) for value in fitted.classes) != (0, 1):
                raise ProtocolError("Stage-70 classifier probability columns drifted.")
            predictions, probabilities = _validated_prediction_outputs(
                fitted.predictions,
                fitted.probabilities,
                expected_rows=len(target.evaluation_row_ids),
            )
            classifier_hash = str(fitted.classifier_config_hash)
            scaler_hash = str(fitted.scaler_state_hash)
            reused_from = ""
            fit_count += 1
        cell = PredictionCell(
            policy_id=replicate.policy_id,
            target_center=replicate.target_center,
            training_seed=replicate.training_seed,
            generation_seed=replicate.generation_seed,
            replicate_id=replicate.replicate_id,
            evaluation_row_ids=target.evaluation_row_ids,
            contract_row_indices=target.contract_row_indices,
            case_ids=target.case_ids,
            predictions=predictions,
            probabilities=probabilities,
            composition_manifest_hash=composition.composition_manifest_hash,
            train_content_sha256=composition.train_content_sha256,
            classifier_config_hash=classifier_hash,
            scaler_state_hash=scaler_hash,
            target_row_order_hash=target.row_order_hash,
            prediction_sha256=array_sha256(predictions),
            probability_sha256=array_sha256(probabilities),
            reused_from_policy_id=reused_from,
        )
        if replicate.policy_id == CONTROL_ARM:
            reusable[signature] = cell
        if reused_from:
            _assert_reuse_identity(cell, reused)  # type: ignore[arg-type]
        cells.append(cell)
    return FrozenPolicyPredictionPass(
        cells=tuple(cells),
        compositions=tuple(compositions),
        classifier_fit_count=fit_count,
        prediction_reuse_count=reuse_count,
    )


def _assert_reuse_identity(observed: PredictionCell, control: PredictionCell) -> None:
    if (
        observed.train_content_sha256 != control.train_content_sha256
        or observed.classifier_config_hash != control.classifier_config_hash
        or observed.scaler_state_hash != control.scaler_state_hash
        or observed.target_row_order_hash != control.target_row_order_hash
        or observed.evaluation_row_ids != control.evaluation_row_ids
        or observed.prediction_sha256 != control.prediction_sha256
        or observed.probability_sha256 != control.probability_sha256
        or not np.array_equal(observed.predictions, control.predictions)
        or not np.array_equal(observed.probabilities, control.probabilities)
    ):
        raise ProtocolError("Utility fallback prediction reuse is not bitwise exact.")


def _validated_prediction_outputs(
    predictions: object,
    probabilities: object,
    *,
    expected_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    observed_predictions = np.asarray(predictions, dtype=np.int64)
    observed_probabilities = np.asarray(probabilities, dtype=np.float64)
    if (
        observed_predictions.shape != (expected_rows,)
        or observed_probabilities.shape != (expected_rows, 2)
        or set(int(value) for value in np.unique(observed_predictions)) - {0, 1}
        or not np.isfinite(observed_probabilities).all()
        or np.any(observed_probabilities < 0.0)
        or np.any(observed_probabilities > 1.0)
        or not np.allclose(
            observed_probabilities.sum(axis=1),
            1.0,
            rtol=0.0,
            atol=1.0e-12,
        )
        or not np.array_equal(
            np.argmax(observed_probabilities, axis=1),
            observed_predictions,
        )
    ):
        raise ProtocolError("Stage-70 classifier prediction/probability geometry drifted.")
    return (
        np.array(observed_predictions, dtype=np.int64, copy=True),
        np.array(observed_probabilities, dtype=np.float64, copy=True),
    )


__all__ = (
    "CompositionRecord",
    "FrozenPolicyPredictionPass",
    "PersistedPredictionPass",
    "run_label_free_prediction_pass",
)
