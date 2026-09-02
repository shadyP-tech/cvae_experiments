"""Leakage-resistant typed contracts for policy-calibrated HARP v9.

Only :class:`SourceActionOutcome` can carry development outcomes.  Target
predictions deliberately expose scores and source-trained risk estimates but
have no evaluation outcome field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import struct
from typing import Sequence

from ...protocol import ProtocolError
from .hashing import canonical_hash, require_sha256


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
            raise ProtocolError("HARP v9 probability cells must lie in [0,1].")
        output.append(struct.pack("<f", value).hex())
    if not output:
        raise ProtocolError("HARP v9 probability vector cannot be empty.")
    return tuple(output)


def canonical_probability_hex(values: Sequence[str]) -> tuple[str, ...]:
    output: list[str] = []
    for raw in values:
        if type(raw) is not str or len(raw) != 8:
            raise ProtocolError("HARP v9 probability cells must be float32 hex.")
        try:
            packed = bytes.fromhex(raw)
        except ValueError as exc:
            raise ProtocolError("HARP v9 probability cells must be hexadecimal.") from exc
        value = struct.unpack("<f", packed)[0]
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ProtocolError("HARP v9 probability cells must lie in [0,1].")
        output.append(raw.lower())
    if not output:
        raise ProtocolError("HARP v9 probability vector cannot be empty.")
    return tuple(output)


def probability_bytes_to_hex(values: Sequence[bytes]) -> tuple[str, ...]:
    output: list[str] = []
    for raw in values:
        if type(raw) is not bytes or len(raw) != 4:
            raise ProtocolError("HARP v9 probability byte cells must be float32.")
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
            raise ProtocolError("HARP v9 actions require U/HXE and D01/D10.")
        source = self.candidate_source_id
        if kind == "HXE":
            source = canonical_text(source, name="candidate source")
            if source in {h, q}:
                raise ProtocolError("HARP v9 HXE candidate crossed H/q exclusion.")
        elif source is not None:
            raise ProtocolError("HARP v9 uniform action cannot claim a source.")
        names = tuple(canonical_text(name, name="feature name") for name in self.feature_names)
        values = tuple(finite(value, name="feature value") for value in self.feature_values)
        lowered = tuple(name.lower() for name in names)
        if (
            not names
            or len(names) != len(values)
            or len(set(names)) != len(names)
            or any(token in name for name in lowered for token in _FORBIDDEN_FEATURE_TOKENS)
        ):
            raise ProtocolError("HARP v9 label-free feature schema is invalid or outcome-bearing.")
        baseline = canonical_probability_hex(self.baseline_probability_hex)
        probability = canonical_probability_hex(self.action_probability_hex)
        if len(baseline) != len(probability):
            raise ProtocolError("HARP v9 baseline/action probability vectors are misaligned.")
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
                    "schema_version": "policy_calibrated_label_free_action_v9",
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
            raise ProtocolError("HARP v9 source outcome requires a label-free action.")
        if self.action.query_center_id == self.action.outer_target_id:
            raise ProtocolError("Target evaluation labels cannot enter HARP v9 fitting.")
        if self.split_role != "SOURCE_DEVELOPMENT":
            raise ProtocolError("Only SOURCE_DEVELOPMENT outcomes may fit HARP v9.")
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
                    "schema_version": "policy_calibrated_source_outcome_v9",
                    "action_hash": self.action.action_hash,
                    "bacc_gain": bacc,
                    "brier_delta": brier,
                    "log_delta": log_delta,
                    "split_role": self.split_role,
                }
            ),
        )


def action_group(action: LabelFreeAction) -> str:
    if not isinstance(action, LabelFreeAction):
        raise ProtocolError("HARP v9 action grouping requires a label-free action.")
    return f"{action.action_kind}:{action.direction.value}"


@dataclass(frozen=True, slots=True)
class ActionScore:
    action_id: str
    action_hash: str
    action_group: str
    direction: Direction
    pairwise_score: float
    predicted_budget_gain: float
    predicted_allocation_gain: float
    predicted_total_gain: float
    predicted_harm_probability: float
    predicted_brier_delta: float
    predicted_log_delta: float
    acceptance_probability: float
    model_available: bool
    score_hash: str = field(init=False)

    def __post_init__(self) -> None:
        action_id = canonical_text(self.action_id, name="action id")
        action_hash = require_sha256(self.action_hash, name="action hash")
        group = canonical_text(self.action_group, name="action group")
        if not isinstance(self.direction, Direction):
            raise ProtocolError("HARP v9 action score direction is malformed.")
        names = (
            "pairwise_score",
            "predicted_budget_gain",
            "predicted_allocation_gain",
            "predicted_total_gain",
            "predicted_harm_probability",
            "predicted_brier_delta",
            "predicted_log_delta",
            "acceptance_probability",
        )
        values = tuple(finite(getattr(self, name), name=name) for name in names)
        if not 0.0 <= values[4] <= 1.0 or not 0.0 <= values[7] <= 1.0:
            raise ProtocolError("HARP v9 action probabilities must lie in [0,1].")
        if not math.isclose(values[1] + values[2], values[3], abs_tol=1e-10):
            raise ProtocolError("HARP v9 budget/allocation decomposition does not sum.")
        object.__setattr__(self, "action_id", action_id)
        object.__setattr__(self, "action_hash", action_hash)
        object.__setattr__(self, "action_group", group)
        for name, value in zip(names, values, strict=True):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "score_hash",
            canonical_hash(
                {
                    "schema_version": "policy_calibrated_action_score_v9",
                    "action_id": action_id,
                    "action_hash": action_hash,
                    "action_group": group,
                    "direction": self.direction.value,
                    **dict(zip(names, values, strict=True)),
                    "model_available": bool(self.model_available),
                    "evaluation_labels_used": False,
                }
            ),
        )

    @property
    def score(self) -> float:
        return self.pairwise_score

    def public_payload(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "action_hash": self.action_hash,
            "action_group": self.action_group,
            "direction": self.direction.value,
            "pairwise_score": self.pairwise_score,
            "predicted_budget_gain": self.predicted_budget_gain,
            "predicted_allocation_gain": self.predicted_allocation_gain,
            "predicted_total_gain": self.predicted_total_gain,
            "predicted_harm_probability": self.predicted_harm_probability,
            "predicted_brier_delta": self.predicted_brier_delta,
            "predicted_log_delta": self.predicted_log_delta,
            "acceptance_probability": self.acceptance_probability,
            "model_available": self.model_available,
            "score_hash": self.score_hash,
        }


@dataclass(frozen=True, slots=True)
class CasePrediction:
    outer_target_id: str
    query_center_id: str
    case_id: str
    action_scores: tuple[ActionScore, ...]
    raw_top_action_id: str
    top_action_id: str
    acceptance_probability: float
    rank_margin: float
    model_hash: str
    ranker_hash: str
    acceptor_hash: str
    training_center_ids: tuple[str, ...]
    training_candidate_ids: tuple[str, ...]
    excluded_center_ids: tuple[str, ...]
    menu_hash: str
    prediction_hash: str = field(init=False)

    def __post_init__(self) -> None:
        h = canonical_text(self.outer_target_id, name="outer target H")
        q = canonical_text(self.query_center_id, name="query center q")
        case = canonical_text(self.case_id, name="case id")
        scores = tuple(sorted(self.action_scores, key=lambda row: row.action_id))
        if len({row.action_id for row in scores}) != len(scores):
            raise ProtocolError("HARP v9 case prediction contains duplicate actions.")
        action_ids = {row.action_id for row in scores}
        if self.raw_top_action_id not in action_ids | {"B"} or self.top_action_id != self.raw_top_action_id:
            raise ProtocolError("HARP v9 raw/top action identities are malformed.")
        probability = finite(self.acceptance_probability, name="acceptance probability")
        margin = finite(self.rank_margin, name="rank margin")
        if not 0.0 <= probability <= 1.0 or margin < 0.0:
            raise ProtocolError("HARP v9 case confidence values are malformed.")
        if self.top_action_id == "B" and probability != 0.0:
            raise ProtocolError("HARP v9 virtual baseline cannot carry acceptance mass.")
        training = tuple(sorted(canonical_text(value, name="training center") for value in self.training_center_ids))
        candidates = tuple(sorted(canonical_text(value, name="training candidate") for value in self.training_candidate_ids))
        excluded = tuple(sorted(canonical_text(value, name="excluded center") for value in self.excluded_center_ids))
        if (
            h not in excluded
            or q not in excluded
            or q in training
            or q in candidates
            or set(excluded) & set(training)
            or set(excluded) & set(candidates)
            or len(set(training)) != len(training)
            or len(set(candidates)) != len(candidates)
        ):
            raise ProtocolError("HARP v9 prediction crossed outer/held-center exclusion.")
        for name in ("model_hash", "ranker_hash", "acceptor_hash", "menu_hash"):
            object.__setattr__(self, name, require_sha256(getattr(self, name), name=name))
        object.__setattr__(self, "outer_target_id", h)
        object.__setattr__(self, "query_center_id", q)
        object.__setattr__(self, "case_id", case)
        object.__setattr__(self, "action_scores", scores)
        object.__setattr__(self, "acceptance_probability", probability)
        object.__setattr__(self, "rank_margin", margin)
        object.__setattr__(self, "training_center_ids", training)
        object.__setattr__(self, "training_candidate_ids", candidates)
        object.__setattr__(self, "excluded_center_ids", excluded)
        object.__setattr__(
            self,
            "prediction_hash",
            canonical_hash(
                {
                    "schema_version": "policy_calibrated_case_prediction_v9",
                    "outer_target_id": h,
                    "query_center_id": q,
                    "case_id": case,
                    "action_score_hashes": tuple(row.score_hash for row in scores),
                    "raw_top_action_id": self.raw_top_action_id,
                    "top_action_id": self.top_action_id,
                    "acceptance_probability": probability,
                    "rank_margin": margin,
                    "model_hash": self.model_hash,
                    "ranker_hash": self.ranker_hash,
                    "acceptor_hash": self.acceptor_hash,
                    "training_center_ids": training,
                    "training_candidate_ids": candidates,
                    "excluded_center_ids": excluded,
                    "menu_hash": self.menu_hash,
                    "virtual_B_in_rank_set": True,
                    "outcomes_consumed": False,
                }
            ),
        )

    @property
    def ranked_action_ids(self) -> tuple[str, ...]:
        ranked = sorted(self.action_scores, key=lambda row: (-row.pairwise_score, row.action_id))
        return tuple(row.action_id for row in ranked)

    def score_for(self, action_id: str) -> ActionScore | None:
        return next((row for row in self.action_scores if row.action_id == action_id), None)

    def public_payload(self) -> dict[str, object]:
        return {
            "outer_target_id": self.outer_target_id,
            "query_center_id": self.query_center_id,
            "case_id": self.case_id,
            "raw_top_action_id": self.raw_top_action_id,
            "top_action_id": self.top_action_id,
            "acceptance_probability": self.acceptance_probability,
            "rank_margin": self.rank_margin,
            "ranked_action_ids": list(self.ranked_action_ids),
            "model_hash": self.model_hash,
            "ranker_hash": self.ranker_hash,
            "acceptor_hash": self.acceptor_hash,
            "training_center_ids": list(self.training_center_ids),
            "training_candidate_ids": list(self.training_candidate_ids),
            "excluded_center_ids": list(self.excluded_center_ids),
            "menu_hash": self.menu_hash,
            "prediction_hash": self.prediction_hash,
            "action_scores": [row.public_payload() for row in self.action_scores],
        }


__all__ = (
    "ActionScore",
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
