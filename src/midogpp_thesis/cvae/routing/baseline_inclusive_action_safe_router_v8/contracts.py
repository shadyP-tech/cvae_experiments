"""Typed contracts for baseline-inclusive, action-safe HARP v8 routing.

Only ``SourceActionOutcome`` can carry development endpoints.  Target-facing
predictions contain label-free estimates and source-OOF residual certificates;
there is no target/evaluation outcome field in their constructors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import struct
from typing import Sequence

from ...protocol import ProtocolError
from .hashing import canonical_hash


_FORBIDDEN_FEATURE_TOKENS = (
    "label",
    "truth",
    "outcome",
    "oracle",
    "bacc",
    "brier",
    "log_loss",
    "evaluation",
)


class Direction(str, Enum):
    D01 = "D01"
    D10 = "D10"


def canonical_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ProtocolError(f"{name} must be a canonical nonempty string.")
    return value


def finite(value: object, *, name: str) -> float:
    if type(value) not in (int, float):
        raise ProtocolError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError(f"{name} must be finite.")
    return 0.0 if result == 0.0 else result


def float32_probability_hex(values: Sequence[float]) -> tuple[str, ...]:
    output: list[str] = []
    for raw in values:
        value = finite(raw, name="probability")
        if not 0.0 <= value <= 1.0:
            raise ProtocolError("HARP v8 probability cells must lie in [0,1].")
        output.append(struct.pack("<f", value).hex())
    if not output:
        raise ProtocolError("HARP v8 probability vector cannot be empty.")
    return tuple(output)


def canonical_probability_hex(values: Sequence[str]) -> tuple[str, ...]:
    output: list[str] = []
    for raw in values:
        if type(raw) is not str or len(raw) != 8:
            raise ProtocolError("HARP v8 probability cells must be float32 hex.")
        try:
            packed = bytes.fromhex(raw)
        except ValueError as exc:
            raise ProtocolError("HARP v8 probability cells must be hexadecimal.") from exc
        value = struct.unpack("<f", packed)[0]
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ProtocolError("HARP v8 probability cells must lie in [0,1].")
        output.append(raw.lower())
    if not output:
        raise ProtocolError("HARP v8 probability vector cannot be empty.")
    return tuple(output)


def probability_bytes_to_hex(values: Sequence[bytes]) -> tuple[str, ...]:
    output: list[str] = []
    for raw in values:
        if type(raw) is not bytes or len(raw) != 4:
            raise ProtocolError("HARP v8 probability byte cells must be float32.")
        output.append(raw.hex())
    return canonical_probability_hex(output)


def probability_hex_to_bytes(values: Sequence[str]) -> tuple[bytes, ...]:
    return tuple(bytes.fromhex(raw) for raw in canonical_probability_hex(values))


@dataclass(frozen=True, slots=True)
class LabelFreeAction:
    outer_target_id: str
    query_center_id: str
    case_id: str
    action_id: str
    action_kind: str
    direction: Direction
    candidate_source_id: str | None
    feature_names: tuple[str, ...]
    feature_values: tuple[float, ...]
    baseline_probability_hex: tuple[str, ...]
    action_probability_hex: tuple[str, ...]
    action_hash: str = field(init=False)

    def __post_init__(self) -> None:
        h = canonical_text(self.outer_target_id, name="outer target H")
        q = canonical_text(self.query_center_id, name="query center q")
        case = canonical_text(self.case_id, name="case id")
        action = canonical_text(self.action_id, name="action id")
        kind = canonical_text(self.action_kind, name="action kind")
        if kind not in {"U", "HXE"} or not isinstance(self.direction, Direction):
            raise ProtocolError("HARP v8 actions require U/HXE and D01/D10.")
        source = self.candidate_source_id
        if kind == "HXE":
            source = canonical_text(source, name="candidate source")
            if source in {h, q}:
                raise ProtocolError("HARP v8 HXE candidate crossed H/q exclusion.")
        elif source is not None:
            raise ProtocolError("HARP v8 uniform action cannot claim a source.")
        names = tuple(canonical_text(name, name="feature name") for name in self.feature_names)
        values = tuple(finite(value, name="feature value") for value in self.feature_values)
        lowered = tuple(name.lower() for name in names)
        if (
            not names
            or len(names) != len(values)
            or len(set(names)) != len(names)
            or any(token in name for name in lowered for token in _FORBIDDEN_FEATURE_TOKENS)
        ):
            raise ProtocolError("HARP v8 label-free feature schema is outcome-bearing or invalid.")
        baseline = canonical_probability_hex(self.baseline_probability_hex)
        probability = canonical_probability_hex(self.action_probability_hex)
        if len(baseline) != len(probability):
            raise ProtocolError("HARP v8 baseline/action probability vectors are misaligned.")
        object.__setattr__(self, "outer_target_id", h)
        object.__setattr__(self, "query_center_id", q)
        object.__setattr__(self, "case_id", case)
        object.__setattr__(self, "action_id", action)
        object.__setattr__(self, "action_kind", kind)
        object.__setattr__(self, "candidate_source_id", source)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "feature_values", values)
        object.__setattr__(self, "baseline_probability_hex", baseline)
        object.__setattr__(self, "action_probability_hex", probability)
        object.__setattr__(
            self,
            "action_hash",
            canonical_hash(
                {
                    "schema_version": "baseline_inclusive_label_free_action_v8",
                    "outer_target_id": h,
                    "query_center_id": q,
                    "case_id": case,
                    "action_id": action,
                    "action_kind": kind,
                    "direction": self.direction.value,
                    "candidate_source_id": source,
                    "feature_names": names,
                    "feature_values": values,
                    "baseline_probability_hex": baseline,
                    "action_probability_hex": probability,
                    "target_labels_consumed": False,
                }
            ),
        )

    @property
    def is_active(self) -> bool:
        return self.action_probability_hex != self.baseline_probability_hex

    @property
    def physical_output_hash(self) -> str:
        return canonical_hash(self.action_probability_hex)


@dataclass(frozen=True, slots=True)
class SourceActionOutcome:
    action: LabelFreeAction
    bacc_gain: float
    brier_delta: float
    log_delta: float
    split_role: str = "SOURCE_DEVELOPMENT"
    outcome_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.action, LabelFreeAction):
            raise ProtocolError("HARP v8 source outcome requires a label-free action.")
        if self.action.query_center_id == self.action.outer_target_id:
            raise ProtocolError("Target evaluation labels cannot enter HARP v8 fitting.")
        if self.split_role != "SOURCE_DEVELOPMENT":
            raise ProtocolError("Only SOURCE_DEVELOPMENT outcomes may fit HARP v8.")
        bacc = finite(self.bacc_gain, name="BACC gain")
        brier = finite(self.brier_delta, name="Brier delta")
        log_delta = finite(self.log_delta, name="log-loss delta")
        object.__setattr__(self, "bacc_gain", bacc)
        object.__setattr__(self, "brier_delta", brier)
        object.__setattr__(self, "log_delta", log_delta)
        object.__setattr__(
            self,
            "outcome_hash",
            canonical_hash(
                {
                    "schema_version": "baseline_inclusive_source_outcome_v8",
                    "action_hash": self.action.action_hash,
                    "bacc_gain": bacc,
                    "brier_delta": brier,
                    "log_delta": log_delta,
                    "split_role": self.split_role,
                }
            ),
        )


def action_group(action: LabelFreeAction) -> str:
    """Return a candidate-identity-free physical action calibration cell."""

    if not isinstance(action, LabelFreeAction):
        raise ProtocolError("HARP v8 action grouping requires a label-free action.")
    return f"{action.action_kind}:{action.direction.value}"


@dataclass(frozen=True, slots=True)
class ActionEstimate:
    """Four baseline-relative predictions for one physical action."""

    action_id: str
    action_hash: str
    action_group: str
    direction: Direction
    predicted_bacc_gain: float
    predicted_harm_probability: float
    predicted_brier_delta: float
    predicted_log_delta: float
    model_available: bool
    estimate_hash: str = field(init=False)

    def __post_init__(self) -> None:
        action_id = canonical_text(self.action_id, name="action id")
        group = canonical_text(self.action_group, name="action group")
        if type(self.action_hash) is not str or len(self.action_hash) != 64:
            raise ProtocolError("HARP v8 action-estimate hash is malformed.")
        if not isinstance(self.direction, Direction):
            raise ProtocolError("HARP v8 action-estimate direction is malformed.")
        gain = finite(self.predicted_bacc_gain, name="predicted BACC gain")
        harm = finite(self.predicted_harm_probability, name="predicted harm probability")
        brier = finite(self.predicted_brier_delta, name="predicted Brier delta")
        log_delta = finite(self.predicted_log_delta, name="predicted log-loss delta")
        if not 0.0 <= harm <= 1.0:
            raise ProtocolError("HARP v8 harm probability must lie in [0,1].")
        object.__setattr__(self, "action_id", action_id)
        object.__setattr__(self, "action_group", group)
        object.__setattr__(self, "predicted_bacc_gain", gain)
        object.__setattr__(self, "predicted_harm_probability", harm)
        object.__setattr__(self, "predicted_brier_delta", brier)
        object.__setattr__(self, "predicted_log_delta", log_delta)
        object.__setattr__(
            self,
            "estimate_hash",
            canonical_hash(
                {
                    "schema_version": "baseline_inclusive_action_estimate_v8",
                    "action_id": action_id,
                    "action_hash": self.action_hash,
                    "action_group": group,
                    "direction": self.direction.value,
                    "predicted_bacc_gain": gain,
                    "predicted_harm_probability": harm,
                    "predicted_brier_delta": brier,
                    "predicted_log_delta": log_delta,
                    "model_available": bool(self.model_available),
                    "evaluation_labels_used": False,
                }
            ),
        )

    @property
    def score(self) -> float:
        """Compatibility alias: v8 ranks by predicted signed gain."""

        return self.predicted_bacc_gain


@dataclass(frozen=True, slots=True)
class ActionCertificate:
    """Source-OOF, center-group-calibrated safety evidence for one action."""

    estimate: ActionEstimate
    gain_lcb: float
    harm_probability_ucb: float
    brier_delta_ucb: float
    log_delta_ucb: float
    harm_brier_risk: float
    harm_log_loss_risk: float
    calibration_cell_hash: str
    safe: bool
    failed_gates: tuple[str, ...]
    certificate_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.estimate, ActionEstimate):
            raise ProtocolError("HARP v8 certificate requires an action estimate.")
        values = tuple(
            finite(value, name="certificate value")
            for value in (
                self.gain_lcb,
                self.harm_probability_ucb,
                self.brier_delta_ucb,
                self.log_delta_ucb,
                self.harm_brier_risk,
                self.harm_log_loss_risk,
            )
        )
        if not 0.0 <= values[1] <= 1.0 or values[4] < 0.0 or values[5] < 0.0:
            raise ProtocolError("HARP v8 certificate probability/risk is malformed.")
        if type(self.calibration_cell_hash) is not str or len(self.calibration_cell_hash) != 64:
            raise ProtocolError("HARP v8 calibration-cell hash is malformed.")
        gates = tuple(sorted(canonical_text(value, name="failed gate") for value in self.failed_gates))
        if bool(self.safe) != (not gates):
            raise ProtocolError("HARP v8 action safety and failed gates disagree.")
        for name, value in zip(
            (
                "gain_lcb",
                "harm_probability_ucb",
                "brier_delta_ucb",
                "log_delta_ucb",
                "harm_brier_risk",
                "harm_log_loss_risk",
            ),
            values,
            strict=True,
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "failed_gates", gates)
        object.__setattr__(
            self,
            "certificate_hash",
            canonical_hash(
                {
                    "schema_version": "baseline_inclusive_action_certificate_v8",
                    "estimate_hash": self.estimate.estimate_hash,
                    "gain_lcb": values[0],
                    "harm_probability_ucb": values[1],
                    "brier_delta_ucb": values[2],
                    "log_delta_ucb": values[3],
                    "harm_brier_risk": values[4],
                    "harm_log_loss_risk": values[5],
                    "calibration_cell_hash": self.calibration_cell_hash,
                    "safe": bool(self.safe),
                    "failed_gates": gates,
                    "target_labels_used": False,
                }
            ),
        )

    @property
    def action_id(self) -> str:
        return self.estimate.action_id

    @property
    def action_hash(self) -> str:
        return self.estimate.action_hash

    @property
    def direction(self) -> Direction:
        return self.estimate.direction

    @property
    def score(self) -> float:
        return self.estimate.predicted_bacc_gain


@dataclass(frozen=True, slots=True)
class CasePrediction:
    """A baseline-inclusive target/source-OOF prediction for a complete case."""

    outer_target_id: str
    query_center_id: str
    case_id: str
    action_certificates: tuple[ActionCertificate, ...]
    model_hash: str
    training_center_ids: tuple[str, ...]
    training_candidate_ids: tuple[str, ...]
    excluded_center_ids: tuple[str, ...]
    menu_hash: str
    prediction_hash: str = field(init=False)

    def __post_init__(self) -> None:
        h = canonical_text(self.outer_target_id, name="outer target H")
        q = canonical_text(self.query_center_id, name="query center q")
        case = canonical_text(self.case_id, name="case id")
        certificates = tuple(sorted(self.action_certificates, key=lambda row: row.action_id))
        if len({row.action_id for row in certificates}) != len(certificates):
            raise ProtocolError("HARP v8 case prediction contains duplicate actions.")
        training = tuple(sorted(canonical_text(value, name="training center") for value in self.training_center_ids))
        candidates = tuple(sorted(canonical_text(value, name="training candidate") for value in self.training_candidate_ids))
        excluded = tuple(sorted(canonical_text(value, name="excluded center") for value in self.excluded_center_ids))
        if (
            len(set(training)) != len(training)
            or len(set(candidates)) != len(candidates)
            or len(set(excluded)) != len(excluded)
            or h not in excluded
            or q in training
            or q in candidates
            or set(excluded) & set(training)
            or set(excluded) & set(candidates)
        ):
            raise ProtocolError("HARP v8 prediction crossed outer/held-center exclusion.")
        for name in ("model_hash", "menu_hash"):
            value = getattr(self, name)
            if type(value) is not str or len(value) != 64:
                raise ProtocolError(f"HARP v8 {name} is malformed.")
        object.__setattr__(self, "outer_target_id", h)
        object.__setattr__(self, "query_center_id", q)
        object.__setattr__(self, "case_id", case)
        object.__setattr__(self, "action_certificates", certificates)
        object.__setattr__(self, "training_center_ids", training)
        object.__setattr__(self, "training_candidate_ids", candidates)
        object.__setattr__(self, "excluded_center_ids", excluded)
        object.__setattr__(
            self,
            "prediction_hash",
            canonical_hash(
                {
                    "schema_version": "baseline_inclusive_case_prediction_v8",
                    "outer_target_id": h,
                    "query_center_id": q,
                    "case_id": case,
                    "certificate_hashes": tuple(row.certificate_hash for row in certificates),
                    "model_hash": self.model_hash,
                    "training_center_ids": training,
                    "training_candidate_ids": candidates,
                    "excluded_center_ids": excluded,
                    "menu_hash": self.menu_hash,
                    "baseline_B_explicit": True,
                    "outcomes_consumed": False,
                }
            ),
        )

    @property
    def safe_action_ids(self) -> tuple[str, ...]:
        return tuple(row.action_id for row in self.action_certificates if row.safe)

    @property
    def raw_top_action_id(self) -> str | None:
        if not self.action_certificates:
            return None
        return min(
            self.action_certificates,
            key=lambda row: (-row.estimate.predicted_bacc_gain, row.action_id),
        ).action_id

    @property
    def top_action_id(self) -> str | None:
        safe = tuple(row for row in self.action_certificates if row.safe)
        if not safe:
            return None
        return min(safe, key=lambda row: (-row.estimate.predicted_bacc_gain, row.action_id)).action_id

    @property
    def certificate_confidence_diagnostic(self) -> float:
        """Largest action confidence diagnostic, never a utility sign test."""

        if not self.action_certificates:
            return 0.0
        return max(1.0 - row.estimate.predicted_harm_probability for row in self.action_certificates)

    @property
    def rank_margin(self) -> float:
        safe_scores = sorted(
            (row.estimate.predicted_bacc_gain for row in self.action_certificates if row.safe),
            reverse=True,
        )
        if len(safe_scores) < 2:
            return 0.0
        return safe_scores[0] - safe_scores[1]

    def passes_rank_margin(self, threshold: float) -> bool:
        value = finite(threshold, name="rank-margin threshold")
        if value < 0.0:
            raise ProtocolError("Rank-margin threshold must be nonnegative.")
        return len(self.safe_action_ids) == 1 or self.rank_margin >= value


__all__ = (
    "ActionCertificate",
    "ActionEstimate",
    "CasePrediction",
    "Direction",
    "LabelFreeAction",
    "SourceActionOutcome",
    "action_group",
    "canonical_probability_hex",
    "canonical_text",
    "finite",
    "float32_probability_hex",
    "probability_bytes_to_hex",
    "probability_hex_to_bytes",
)
