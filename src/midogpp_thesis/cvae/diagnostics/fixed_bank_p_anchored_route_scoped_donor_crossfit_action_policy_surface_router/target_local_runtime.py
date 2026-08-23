"""Target-local, whole-case-excluded posterior fitting for P-DCAPS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .identity import canonical_hash, require_sha256
from .label_firewall import (
    SupportLabelCapability,
    require_support_label_capability,
)
from .physical_adapter import CenterPhysicalSurface


POSTERIOR_CONTROL_IDS = ("IDENTITY", "WITHIN_CASE_CYCLIC_SHIFT")
POSTERIOR_RIDGE_ALPHA = 1.0
POSTERIOR_CLIP = 1.0e-12
DISCRIMINATIVE_MODEL_KIND = (
    "discriminative_target_label_probability_not_cvae_posterior"
)


@dataclass(frozen=True)
class FingerprintSurface:
    center: str
    sample_ids: tuple[str, ...]
    case_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_values: np.ndarray
    physical_surface_hash: str
    center_surface_hash: str
    control_id: str
    fingerprint_hash: str = field(init=False)

    def __post_init__(self) -> None:
        values = np.ascontiguousarray(self.feature_values, dtype=np.float64)
        if (
            self.control_id not in POSTERIOR_CONTROL_IDS
            or values.shape != (len(self.sample_ids), len(self.feature_names))
            or len(self.sample_ids) != len(self.case_ids)
            or not np.isfinite(values).all()
        ):
            raise ProtocolError("P-DCAPS fingerprint surface drifted.")
        values.setflags(write=False)
        object.__setattr__(self, "feature_values", values)
        object.__setattr__(
            self,
            "fingerprint_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_fingerprint_surface_v1",
                    "center": self.center,
                    "sample_ids": self.sample_ids,
                    "case_ids": self.case_ids,
                    "feature_names": self.feature_names,
                    "feature_values": values,
                    "physical_surface_hash": self.physical_surface_hash,
                    "center_surface_hash": self.center_surface_hash,
                    "control_id": self.control_id,
                    "labels_used": False,
                }
            ),
        )


@dataclass(frozen=True)
class TargetPosteriorModel:
    target_center: str
    held_case_id: str
    control_id: str
    training_case_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    training_row_count: int
    training_n_positive: int
    training_n_negative: int
    fingerprint_hash: str
    training_identity_hash: str
    support_capability_hash: str
    iterations: int
    converged: bool
    model_kind: str = DISCRIMINATIVE_MODEL_KIND
    model_hash: str = field(init=False)

    def __post_init__(self) -> None:
        width = len(self.feature_names)
        values = np.asarray(
            [
                *self.feature_mean,
                *self.feature_scale,
                *self.coefficients,
                self.intercept,
            ],
            dtype=np.float64,
        )
        if (
            self.control_id not in POSTERIOR_CONTROL_IDS
            or self.held_case_id in self.training_case_ids
            or width == 0
            or len(self.feature_mean) != width
            or len(self.feature_scale) != width
            or len(self.coefficients) != width
            or np.any(np.asarray(self.feature_scale) <= 0.0)
            or not np.isfinite(values).all()
            or self.training_row_count
            != self.training_n_positive + self.training_n_negative
            or min(self.training_n_positive, self.training_n_negative) <= 0
            or self.model_kind != DISCRIMINATIVE_MODEL_KIND
        ):
            raise ProtocolError("P-DCAPS target posterior model drifted.")
        require_sha256(self.support_capability_hash, "support capability")
        object.__setattr__(
            self,
            "model_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_target_posterior_model_v1",
                    "target_center": self.target_center,
                    "held_case_id": self.held_case_id,
                    "control_id": self.control_id,
                    "training_case_ids": self.training_case_ids,
                    "feature_names": self.feature_names,
                    "feature_mean": self.feature_mean,
                    "feature_scale": self.feature_scale,
                    "coefficients": self.coefficients,
                    "intercept": self.intercept,
                    "training_row_count": self.training_row_count,
                    "training_n_positive": self.training_n_positive,
                    "training_n_negative": self.training_n_negative,
                    "fingerprint_hash": self.fingerprint_hash,
                    "training_identity_hash": self.training_identity_hash,
                    "support_capability_hash": self.support_capability_hash,
                    "iterations": self.iterations,
                    "converged": self.converged,
                    "model_kind": self.model_kind,
                    "nelbo_or_cvae_posterior_claimed": False,
                    "whole_case_excluded": True,
                }
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "target_center": self.target_center,
            "held_case_id": self.held_case_id,
            "control_id": self.control_id,
            "training_case_ids": list(self.training_case_ids),
            "feature_names": list(self.feature_names),
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "training_row_count": self.training_row_count,
            "training_n_positive": self.training_n_positive,
            "training_n_negative": self.training_n_negative,
            "fingerprint_hash": self.fingerprint_hash,
            "training_identity_hash": self.training_identity_hash,
            "support_capability_hash": self.support_capability_hash,
            "iterations": self.iterations,
            "converged": self.converged,
            "model_kind": self.model_kind,
            "nelbo_or_cvae_posterior_claimed": False,
            "whole_case_excluded": True,
            "model_hash": self.model_hash,
        }


@dataclass(frozen=True)
class CasePosteriorPrediction:
    target_center: str
    held_case_id: str
    control_id: str
    sample_ids: tuple[str, ...]
    natural_probabilities: np.ndarray
    model_hash: str
    fingerprint_hash: str
    prediction_hash: str = field(init=False)

    def __post_init__(self) -> None:
        eta = np.ascontiguousarray(self.natural_probabilities, dtype=np.float64)
        if (
            eta.shape != (len(self.sample_ids),)
            or not np.isfinite(eta).all()
            or np.any((eta <= 0.0) | (eta >= 1.0))
        ):
            raise ProtocolError("P-DCAPS target posterior prediction drifted.")
        eta.setflags(write=False)
        object.__setattr__(self, "natural_probabilities", eta)
        object.__setattr__(
            self,
            "prediction_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_case_posterior_prediction_v1",
                    "target_center": self.target_center,
                    "held_case_id": self.held_case_id,
                    "control_id": self.control_id,
                    "sample_ids": self.sample_ids,
                    "natural_probabilities": eta,
                    "model_hash": self.model_hash,
                    "fingerprint_hash": self.fingerprint_hash,
                    "held_case_labels_used": False,
                }
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "target_center": self.target_center,
            "held_case_id": self.held_case_id,
            "control_id": self.control_id,
            "sample_ids": list(self.sample_ids),
            "natural_probabilities": self.natural_probabilities.tolist(),
            "model_hash": self.model_hash,
            "fingerprint_hash": self.fingerprint_hash,
            "held_case_labels_used": False,
            "prediction_hash": self.prediction_hash,
        }


@dataclass(frozen=True)
class PseudoPosteriorReference:
    outer_center: str
    scored_center: str
    held_case_id: str
    prediction_hash: str
    model_hash: str
    excluded_outer_center: str
    excluded_scored_center: str
    reference_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.outer_center == self.scored_center
            or self.excluded_outer_center != self.outer_center
            or self.excluded_scored_center != self.scored_center
        ):
            raise ProtocolError("P-DCAPS pseudo posterior H/J binding drifted.")
        object.__setattr__(
            self,
            "reference_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_pseudo_posterior_reference_v1",
                    "outer_center": self.outer_center,
                    "scored_center": self.scored_center,
                    "held_case_id": self.held_case_id,
                    "prediction_hash": self.prediction_hash,
                    "model_hash": self.model_hash,
                    "excluded_outer_center": self.outer_center,
                    "excluded_scored_center": self.scored_center,
                    "posterior_refit": False,
                }
            ),
        )


def build_fingerprint_surface(
    surface: CenterPhysicalSurface,
    *,
    physical_surface_hash: str,
    control_id: str,
) -> FingerprintSurface:
    if control_id not in POSTERIOR_CONTROL_IDS:
        raise ProtocolError("P-DCAPS posterior fingerprint control drifted.")
    feature_names: list[str] = []
    columns: list[np.ndarray] = []
    for action_id, seed_values in surface.seed_probabilities:
        seed = seed_values.astype(np.float64, copy=False)
        mean = np.mean(seed, axis=0, dtype=np.float64)
        standard_deviation = np.std(seed, axis=0, ddof=0, dtype=np.float64)
        vote = np.mean(seed >= 0.5, axis=0, dtype=np.float64)
        feature_names.extend(
            (
                f"{action_id}::exact_nine_mean",
                f"{action_id}::seed_pair_sd",
                f"{action_id}::positive_vote_fraction",
            )
        )
        columns.extend((mean, standard_deviation, vote))
    values = np.column_stack(columns)
    if control_id == POSTERIOR_CONTROL_IDS[1]:
        shifted = np.empty_like(values)
        for case in sorted(set(surface.case_ids)):
            positions = surface.positions(case)
            shifted[positions] = np.roll(values[positions], shift=1, axis=0)
        values = shifted
    return FingerprintSurface(
        surface.center,
        surface.sample_ids,
        surface.case_ids,
        tuple(feature_names),
        values,
        physical_surface_hash,
        surface.center_surface_hash,
        control_id,
    )


def fit_route_posterior(
    fingerprint: FingerprintSurface,
    *,
    held_case_id: str,
    support_capability: SupportLabelCapability,
    maximum_iterations: int = 100,
) -> tuple[TargetPosteriorModel, CasePosteriorPrediction]:
    held = str(held_case_id)
    support_positions = np.flatnonzero(np.asarray(fingerprint.case_ids) != held)
    held_positions = np.flatnonzero(np.asarray(fingerprint.case_ids) == held)
    expected_keys = {
        (
            fingerprint.center,
            fingerprint.case_ids[position],
            fingerprint.sample_ids[position],
        )
        for position in support_positions
    }
    expected_key_order = tuple(
        (
            fingerprint.center,
            fingerprint.case_ids[position],
            fingerprint.sample_ids[position],
        )
        for position in support_positions
    )
    rows = require_support_label_capability(
        support_capability,
        center=fingerprint.center,
        held_case_id=held,
        expected_keys=expected_key_order,
    )
    label_map = {row.key: row.value for row in rows}
    if (
        not len(held_positions)
        or len(label_map) != len(rows)
        or set(label_map) != expected_keys
        or any(row.case_id == held for row in rows)
    ):
        raise ProtocolError("P-DCAPS target posterior support scope drifted.")
    x = fingerprint.feature_values[support_positions]
    y = np.asarray(
        [
            label_map[
                (
                    fingerprint.center,
                    fingerprint.case_ids[position],
                    fingerprint.sample_ids[position],
                )
            ]
            for position in support_positions
        ],
        dtype=np.float64,
    )
    mean = np.mean(x, axis=0, dtype=np.float64)
    scale = np.std(x, axis=0, ddof=0, dtype=np.float64)
    scale = np.where(scale > 0.0, scale, 1.0)
    design = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    beta = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.diag(np.asarray([0.0, *([POSTERIOR_RIDGE_ALPHA] * x.shape[1])]))
    converged = False
    iteration = 0
    for iteration in range(1, int(maximum_iterations) + 1):
        eta = np.clip(design @ beta, -30.0, 30.0)
        probability = np.clip(1.0 / (1.0 + np.exp(-eta)), POSTERIOR_CLIP, 1.0 - POSTERIOR_CLIP)
        gradient = design.T @ (y - probability) - penalty @ beta
        information = design.T @ ((probability * (1.0 - probability))[:, None] * design) + penalty
        try:
            update = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            update = np.linalg.pinv(information, rcond=1.0e-12) @ gradient
        if not np.isfinite(update).all():
            break
        beta += update
        if float(np.max(np.abs(update))) <= 1.0e-10:
            converged = True
            break
    training_cases = tuple(sorted({fingerprint.case_ids[position] for position in support_positions}))
    model = TargetPosteriorModel(
        fingerprint.center,
        held,
        fingerprint.control_id,
        training_cases,
        fingerprint.feature_names,
        tuple(float(value) for value in mean),
        tuple(float(value) for value in scale),
        tuple(float(value) for value in beta[1:]),
        float(beta[0]),
        len(y),
        int(np.sum(y == 1.0)),
        int(np.sum(y == 0.0)),
        fingerprint.fingerprint_hash,
        canonical_hash(
            [
                [
                    fingerprint.center,
                    fingerprint.case_ids[position],
                    fingerprint.sample_ids[position],
                ]
                for position in support_positions
            ]
        ),
        support_capability.capability_hash,
        iteration,
        converged,
    )
    held_design = (
        fingerprint.feature_values[held_positions] - np.asarray(model.feature_mean)
    ) / np.asarray(model.feature_scale)
    logits = model.intercept + held_design @ np.asarray(model.coefficients)
    eta = np.clip(1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0))), POSTERIOR_CLIP, 1.0 - POSTERIOR_CLIP)
    prediction = CasePosteriorPrediction(
        fingerprint.center,
        held,
        fingerprint.control_id,
        tuple(fingerprint.sample_ids[position] for position in held_positions),
        eta,
        model.model_hash,
        fingerprint.fingerprint_hash,
    )
    return model, prediction


def bind_pseudo_reference(
    prediction: CasePosteriorPrediction,
    *,
    outer_center: str,
) -> PseudoPosteriorReference:
    return PseudoPosteriorReference(
        str(outer_center),
        prediction.target_center,
        prediction.held_case_id,
        prediction.prediction_hash,
        prediction.model_hash,
        str(outer_center),
        prediction.target_center,
    )


__all__ = (
    "CasePosteriorPrediction",
    "DISCRIMINATIVE_MODEL_KIND",
    "FingerprintSurface",
    "POSTERIOR_CONTROL_IDS",
    "PseudoPosteriorReference",
    "TargetPosteriorModel",
    "bind_pseudo_reference",
    "build_fingerprint_surface",
    "fit_route_posterior",
)
