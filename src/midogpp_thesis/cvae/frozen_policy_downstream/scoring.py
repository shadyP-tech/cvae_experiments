"""Metric-only scoring after the Stage-70 prediction seal exists."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ...data.contract.stage70_target_evaluation.contracts import EXPECTED_TEST_ROWS
from ..metrics import balanced_accuracy, macro_f1
from ..protocol import ProtocolError
from .contracts import EXPECTED_METRIC_ROWS
from .prediction import PersistedPredictionPass
from .prediction_seal import validate_persisted_prediction_pass
from .target_loader import _OpenedScoringLabels


@dataclass(frozen=True)
class TargetMetricRow:
    policy_id: str
    target_center: str
    training_seed: int
    generation_seed: int
    replicate_id: str
    n_eval: int
    n_cases: int
    bacc: float
    macro_f1: float
    prediction_sha256: str
    probability_sha256: str
    prediction_cell_hash: str
    target_identity_hash: str
    composition_manifest_hash: str
    train_content_sha256: str
    classifier_config_hash: str
    scaler_state_hash: str
    target_row_order_hash: str
    label_manifest_sha256: str
    reused_from_policy_id: str
    authorization_binding_hash: str
    final_authorization_hash: str
    authorization_protocol_hash: str
    identity_lock_hash: str
    evaluation_plan_hash: str
    reservation_content_hash: str
    target_evaluation_reservation_id: str
    target_evaluation_reservation_protocol_hash: str
    target_cache_artifact_id: str
    target_cache_content_hash: str
    target_cache_row_order_hash: str
    target_cache_shard_sha256: str
    phase_01_sha256: str
    prediction_index_sha256: str
    prediction_arrays_sha256: str
    prediction_seal_sha256: str
    phase_02_sha256: str

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_stage70_target_metric_v1",
            "claim_scope": (
                "descriptive_frozen_policy_comparison_on_previously_consumed_test"
            ),
            "claim_role": "descriptive_locked_policy_comparison",
            "row_role": "target_evaluation_metric",
            "policy_id": self.policy_id,
            "target_center": self.target_center,
            "training_seed": self.training_seed,
            "generation_seed": self.generation_seed,
            "replicate_id": self.replicate_id,
            "n_eval": self.n_eval,
            "n_cases": self.n_cases,
            "bacc": self.bacc,
            "macro_f1": self.macro_f1,
            "macro_f1_role": "secondary_descriptive_only",
            "prediction_sha256": self.prediction_sha256,
            "probability_sha256": self.probability_sha256,
            "prediction_cell_hash": self.prediction_cell_hash,
            "target_identity_hash": self.target_identity_hash,
            "composition_manifest_hash": self.composition_manifest_hash,
            "train_content_sha256": self.train_content_sha256,
            "classifier_config_hash": self.classifier_config_hash,
            "scaler_state_hash": self.scaler_state_hash,
            "target_row_order_hash": self.target_row_order_hash,
            "label_manifest_sha256": self.label_manifest_sha256,
            "reused_from_policy_id": self.reused_from_policy_id,
            "authorization_binding_hash": self.authorization_binding_hash,
            "final_authorization_hash": self.final_authorization_hash,
            "authorization_protocol_hash": self.authorization_protocol_hash,
            "identity_lock_hash": self.identity_lock_hash,
            "evaluation_plan_hash": self.evaluation_plan_hash,
            "reservation_content_hash": self.reservation_content_hash,
            "target_evaluation_reservation_id": self.target_evaluation_reservation_id,
            "target_evaluation_reservation_protocol_hash": (
                self.target_evaluation_reservation_protocol_hash
            ),
            "target_cache_artifact_id": self.target_cache_artifact_id,
            "target_cache_content_hash": self.target_cache_content_hash,
            "target_cache_row_order_hash": self.target_cache_row_order_hash,
            "target_cache_shard_sha256": self.target_cache_shard_sha256,
            "phase_01_sha256": self.phase_01_sha256,
            "prediction_index_sha256": self.prediction_index_sha256,
            "prediction_arrays_sha256": self.prediction_arrays_sha256,
            "prediction_seal_sha256": self.prediction_seal_sha256,
            "phase_02_sha256": self.phase_02_sha256,
            "target_labels_used_for_scoring_only": True,
            "fresh_confirmatory_evidence": False,
            "policy_or_seed_selection_performed": False,
        }


@dataclass(frozen=True)
class CaseConfusionRow:
    policy_id: str
    target_center: str
    training_seed: int
    generation_seed: int
    case_id: str
    tn: int
    fp: int
    fn: int
    tp: int
    replicate_id: str = ""
    prediction_sha256: str = ""
    target_identity_hash: str = ""
    label_manifest_sha256: str = ""
    authorization_binding_hash: str = ""
    prediction_index_sha256: str = ""
    prediction_arrays_sha256: str = ""

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_stage70_case_confusion_v1",
            "policy_id": self.policy_id,
            "target_center": self.target_center,
            "training_seed": self.training_seed,
            "generation_seed": self.generation_seed,
            "case_id": self.case_id,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "tp": self.tp,
            "replicate_id": self.replicate_id,
            "prediction_sha256": self.prediction_sha256,
            "target_identity_hash": self.target_identity_hash,
            "label_manifest_sha256": self.label_manifest_sha256,
            "authorization_binding_hash": self.authorization_binding_hash,
            "prediction_index_sha256": self.prediction_index_sha256,
            "prediction_arrays_sha256": self.prediction_arrays_sha256,
            "target_labels_used_for_scoring_only": True,
        }


@dataclass(frozen=True)
class ScoredFrozenPolicies:
    metrics: tuple[TargetMetricRow, ...]
    case_confusions: tuple[CaseConfusionRow, ...]
    prediction_index_sha256: str
    prediction_arrays_sha256: str
    prediction_seal_sha256: str
    phase_01_sha256: str
    phase_02_sha256: str
    authorization_binding_hash: str
    final_authorization_hash: str
    target_cache_content_hash: str
    label_manifest_sha256: str
    phase: str = "SCORING_COMPLETE"

    def __post_init__(self) -> None:
        if len(self.metrics) != EXPECTED_METRIC_ROWS:
            raise ProtocolError("Stage-70 scoring must emit exactly 243 metric rows.")


def score_persisted_predictions(
    sealed: PersistedPredictionPass,
    labels: _OpenedScoringLabels,
) -> ScoredFrozenPolicies:
    """Join labels only after a durable prediction index has been sealed."""

    verified = validate_persisted_prediction_pass(sealed)
    phase = verified.phase_01_binding
    if (
        not isinstance(labels, _OpenedScoringLabels)
        or labels.label_manifest_sha256 != phase.get("scoring_manifest_sha256")
        or labels.authorization_binding_hash != verified.authorization_binding_hash
        or labels.phase_01_sha256 != verified.phase_01_sha256
        or labels.prediction_index_sha256 != verified.prediction_index_sha256
        or labels.prediction_arrays_sha256 != verified.prediction_arrays_sha256
        or labels.prediction_seal_sha256 != verified.prediction_seal_sha256
        or labels.phase_02_sha256 != verified.phase_02_sha256
        or len(labels.evaluation_row_ids) != EXPECTED_TEST_ROWS
        or len(set(labels.evaluation_row_ids)) != EXPECTED_TEST_ROWS
    ):
        raise ProtocolError("Stage-70 scoring-label provenance drifted from phase 01.")
    label_by_id = {
        row_id: int(label)
        for row_id, label in zip(labels.evaluation_row_ids, labels.labels, strict=True)
    }
    metrics: list[TargetMetricRow] = []
    confusions: list[CaseConfusionRow] = []
    shard_hashes = phase.get("target_cache_shard_sha256_by_center")
    if not isinstance(shard_hashes, Mapping):
        raise ProtocolError("Stage-70 phase-01 cache-shard provenance is malformed.")
    for cell in verified.cells:
        try:
            truth = np.asarray(
                [label_by_id[row_id] for row_id in cell.evaluation_row_ids],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise ProtocolError("Stage-70 scoring labels do not cover prediction rows.") from exc
        predictions = np.asarray(cell.predictions, dtype=np.int64)
        if set(int(value) for value in np.unique(truth)) != {0, 1}:
            raise ProtocolError("Stage-70 target center lacks both scoring classes.")
        metrics.append(
            TargetMetricRow(
                policy_id=cell.policy_id,
                target_center=cell.target_center,
                training_seed=cell.training_seed,
                generation_seed=cell.generation_seed,
                replicate_id=cell.replicate_id,
                n_eval=len(truth),
                n_cases=len(set(cell.case_ids)),
                bacc=balanced_accuracy(truth.tolist(), predictions.tolist()),
                macro_f1=macro_f1(truth.tolist(), predictions.tolist()),
                prediction_sha256=cell.prediction_sha256,
                probability_sha256=cell.probability_sha256,
                prediction_cell_hash=cell.prediction_cell_hash,
                target_identity_hash=cell.target_identity_hash,
                composition_manifest_hash=cell.composition_manifest_hash,
                train_content_sha256=cell.train_content_sha256,
                classifier_config_hash=cell.classifier_config_hash,
                scaler_state_hash=cell.scaler_state_hash,
                target_row_order_hash=cell.target_row_order_hash,
                label_manifest_sha256=labels.label_manifest_sha256,
                reused_from_policy_id=cell.reused_from_policy_id,
                authorization_binding_hash=verified.authorization_binding_hash,
                final_authorization_hash=str(phase["final_authorization_hash"]),
                authorization_protocol_hash=str(phase["authorization_protocol_hash"]),
                identity_lock_hash=str(phase["identity_lock_hash"]),
                evaluation_plan_hash=str(phase["evaluation_plan_hash"]),
                reservation_content_hash=str(phase["reservation_content_hash"]),
                target_evaluation_reservation_id=str(
                    phase["target_evaluation_reservation_id"]
                ),
                target_evaluation_reservation_protocol_hash=str(
                    phase["target_evaluation_reservation_protocol_hash"]
                ),
                target_cache_artifact_id=str(phase["target_cache_artifact_id"]),
                target_cache_content_hash=str(phase["target_cache_content_hash"]),
                target_cache_row_order_hash=str(phase["target_cache_row_order_hash"]),
                target_cache_shard_sha256=str(shard_hashes[cell.target_center]),
                phase_01_sha256=verified.phase_01_sha256,
                prediction_index_sha256=verified.prediction_index_sha256,
                prediction_arrays_sha256=verified.prediction_arrays_sha256,
                prediction_seal_sha256=verified.prediction_seal_sha256,
                phase_02_sha256=verified.phase_02_sha256,
            )
        )
        by_case: dict[str, list[int]] = {}
        for index, case_id in enumerate(cell.case_ids):
            by_case.setdefault(case_id, []).append(index)
        for case_id in sorted(by_case):
            indices = np.asarray(by_case[case_id], dtype=np.int64)
            y = truth[indices]
            p = predictions[indices]
            confusions.append(
                CaseConfusionRow(
                    policy_id=cell.policy_id,
                    target_center=cell.target_center,
                    training_seed=cell.training_seed,
                    generation_seed=cell.generation_seed,
                    case_id=case_id,
                    tn=int(np.sum((y == 0) & (p == 0))),
                    fp=int(np.sum((y == 0) & (p == 1))),
                    fn=int(np.sum((y == 1) & (p == 0))),
                    tp=int(np.sum((y == 1) & (p == 1))),
                    replicate_id=cell.replicate_id,
                    prediction_sha256=cell.prediction_sha256,
                    target_identity_hash=cell.target_identity_hash,
                    label_manifest_sha256=labels.label_manifest_sha256,
                    authorization_binding_hash=verified.authorization_binding_hash,
                    prediction_index_sha256=verified.prediction_index_sha256,
                    prediction_arrays_sha256=verified.prediction_arrays_sha256,
                )
            )
    _validate_confusion_reconstruction(metrics, confusions)
    return ScoredFrozenPolicies(
        metrics=tuple(metrics),
        case_confusions=tuple(confusions),
        prediction_index_sha256=verified.prediction_index_sha256,
        prediction_arrays_sha256=verified.prediction_arrays_sha256,
        prediction_seal_sha256=verified.prediction_seal_sha256,
        phase_01_sha256=verified.phase_01_sha256,
        phase_02_sha256=verified.phase_02_sha256,
        authorization_binding_hash=verified.authorization_binding_hash,
        final_authorization_hash=str(phase["final_authorization_hash"]),
        target_cache_content_hash=str(phase["target_cache_content_hash"]),
        label_manifest_sha256=labels.label_manifest_sha256,
    )


def _validate_confusion_reconstruction(
    metrics: list[TargetMetricRow],
    rows: list[CaseConfusionRow],
) -> None:
    grouped: dict[tuple[str, str, int, int], list[CaseConfusionRow]] = {}
    for row in rows:
        grouped.setdefault(
            (row.policy_id, row.target_center, row.training_seed, row.generation_seed),
            [],
        ).append(row)
    for metric in metrics:
        key = (
            metric.policy_id,
            metric.target_center,
            metric.training_seed,
            metric.generation_seed,
        )
        group = grouped.get(key, [])
        tn = sum(row.tn for row in group)
        fp = sum(row.fp for row in group)
        fn = sum(row.fn for row in group)
        tp = sum(row.tp for row in group)
        observed = _bacc_from_counts(tn=tn, fp=fp, fn=fn, tp=tp)
        if abs(observed - metric.bacc) > 1.0e-12:
            raise ProtocolError("Stage-70 case-confusion reconstruction drifted.")


def _bacc_from_counts(*, tn: int, fp: int, fn: int, tp: int) -> float:
    negative = tn + fp
    positive = tp + fn
    if negative <= 0 or positive <= 0:
        raise ProtocolError("Balanced accuracy requires both class denominators.")
    return 0.5 * ((tn / negative) + (tp / positive))


__all__ = (
    "CaseConfusionRow",
    "ScoredFrozenPolicies",
    "TargetMetricRow",
    "score_persisted_predictions",
)
