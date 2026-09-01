"""Outcome-separated contracts for the physical, case-level HARP v4 router.

Only :class:`CaseTrainingObservation` can represent source-development
responses.  :class:`CaseTargetAction` deliberately has no truth, label, or
outcome member, which keeps target evaluation labels outside the routing API.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import struct

from ...protocol import ProtocolError


class ActionKind(str, Enum):
    B = "B"
    U = "U"
    HXE = "HXE"


class Comparison(str, Enum):
    U_VS_B = "U_VS_B"
    HXE_VS_B = "HXE_VS_B"
    HXE_VS_U = "HXE_VS_U"


OUTCOME_NAMES = (
    "case_equal_bacc_contribution_gain",
    "brier_delta",
    "log_loss_delta",
)


def canonical_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ProtocolError(f"{name} must be a canonical nonempty string.")
    return value


def canonical_features(
    names: tuple[str, ...], values: tuple[float, ...]
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    normalized_names = tuple(canonical_text(value, name="feature name") for value in names)
    normalized_values = tuple(float(value) for value in values)
    if (
        not normalized_names
        or len(normalized_names) != len(normalized_values)
        or len(set(normalized_names)) != len(normalized_names)
        or any(not math.isfinite(value) for value in normalized_values)
    ):
        raise ProtocolError("HARP v4 features must be finite, unique, and aligned.")
    return normalized_names, normalized_values


def _probability_bytes(values: tuple[bytes, ...]) -> tuple[bytes, ...]:
    if not values:
        raise ProtocolError("A case action requires at least one probability.")
    normalized: list[bytes] = []
    for raw in values:
        if type(raw) is not bytes or len(raw) != 4:
            raise ProtocolError("Case probabilities must retain exact little-endian float32 bytes.")
        value = struct.unpack("<f", raw)[0]
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ProtocolError("Case probabilities must lie in [0, 1].")
        normalized.append(raw)
    return tuple(normalized)


@dataclass(frozen=True)
class EffectVector:
    """Case-local effects in the exact terminal estimand units."""

    case_equal_bacc_contribution_gain: float
    brier_delta: float
    log_loss_delta: float

    def __post_init__(self) -> None:
        for name in OUTCOME_NAMES:
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ProtocolError("HARP v4 effects must be finite.")
            object.__setattr__(self, name, value)

    def as_tuple(self) -> tuple[float, float, float]:
        return (
            self.case_equal_bacc_contribution_gain,
            self.brier_delta,
            self.log_loss_delta,
        )


@dataclass(frozen=True, kw_only=True)
class CaseTrainingObservation:
    """One source-development case comparison after strict outer exclusion."""

    outer_target_id: str
    pseudo_query_id: str
    candidate_source_id: str | None
    case_id: str
    comparison: Comparison
    feature_names: tuple[str, ...]
    feature_values: tuple[float, ...]
    effects: EffectVector
    class_counts: tuple[int, int]
    pseudo_query_case_count: int
    pseudo_query_class_support_case_counts: tuple[int, int]

    def __post_init__(self) -> None:
        for name in ("outer_target_id", "pseudo_query_id", "case_id"):
            object.__setattr__(self, name, canonical_text(getattr(self, name), name=name))
        try:
            comparison = Comparison(self.comparison)
        except (TypeError, ValueError) as exc:
            raise ProtocolError("Unknown HARP v4 comparison.") from exc
        object.__setattr__(self, "comparison", comparison)
        candidate = self.candidate_source_id
        if comparison is Comparison.U_VS_B:
            if candidate is not None:
                raise ProtocolError("U-vs-B source rows cannot name an expert candidate.")
        else:
            candidate = canonical_text(candidate, name="candidate_source_id")
            object.__setattr__(self, "candidate_source_id", candidate)
            if candidate == self.pseudo_query_id:
                raise ProtocolError("A pseudo-query cannot use its own expert candidate.")
        if self.outer_target_id == self.pseudo_query_id or candidate == self.outer_target_id:
            raise ProtocolError("Outer H must be excluded from query and candidate roles.")
        names, values = canonical_features(self.feature_names, self.feature_values)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "feature_values", values)
        if not isinstance(self.effects, EffectVector):
            raise ProtocolError("Source-development rows require typed effects.")
        if (
            type(self.class_counts) is not tuple
            or len(self.class_counts) != 2
            or any(type(value) is not int or value < 0 for value in self.class_counts)
            or sum(self.class_counts) <= 0
        ):
            raise ProtocolError("Source case class counts must be nonnegative and nonempty.")
        support = self.pseudo_query_class_support_case_counts
        if (
            type(self.pseudo_query_case_count) is not int
            or self.pseudo_query_case_count < 1
            or type(support) is not tuple
            or len(support) != 2
            or any(type(value) is not int for value in support)
            or any(
                value < 1 or value > self.pseudo_query_case_count
                for value in support
            )
        ):
            raise ProtocolError("Source case BACC normalization is malformed.")

    @property
    def row_key(self) -> tuple[str, str, str, str, str]:
        return (
            self.outer_target_id,
            self.pseudo_query_id,
            self.case_id,
            self.comparison.value,
            self.candidate_source_id or "",
        )


@dataclass(frozen=True, kw_only=True)
class CaseTargetAction:
    """One sealed, label-free action for an entire target case.

    ``HXE`` denotes a physical expert endpoint and is structurally fixed to
    expert weight one.  Probability blends cannot cross this boundary.
    """

    outer_target_id: str
    target_query_id: str
    case_id: str
    action_kind: ActionKind
    candidate_source_id: str | None
    feature_names: tuple[str, ...]
    feature_values: tuple[float, ...]
    sample_ids: tuple[str, ...]
    probability_bytes: tuple[bytes, ...]
    prediction_seal_hash: str
    expert_weight: float

    def __post_init__(self) -> None:
        for name in ("outer_target_id", "target_query_id", "case_id", "prediction_seal_hash"):
            object.__setattr__(self, name, canonical_text(getattr(self, name), name=name))
        if self.target_query_id != self.outer_target_id:
            raise ProtocolError("A target action must belong to its held-out outer H.")
        try:
            kind = ActionKind(self.action_kind)
        except (TypeError, ValueError) as exc:
            raise ProtocolError("Unknown HARP v4 action kind.") from exc
        object.__setattr__(self, "action_kind", kind)
        candidate = self.candidate_source_id
        weight = float(self.expert_weight)
        if kind is ActionKind.HXE:
            candidate = canonical_text(candidate, name="candidate_source_id")
            if candidate == self.outer_target_id:
                raise ProtocolError("The held-out target expert cannot be routed.")
            if weight != 1.0:
                raise ProtocolError("Physical Hxe actions require lambda=1 exactly.")
            object.__setattr__(self, "candidate_source_id", candidate)
        elif candidate is not None or weight != 0.0:
            raise ProtocolError("B and U actions cannot carry an expert or expert weight.")
        object.__setattr__(self, "expert_weight", weight)
        names, values = canonical_features(self.feature_names, self.feature_values)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "feature_values", values)
        sample_ids = tuple(canonical_text(value, name="sample_id") for value in self.sample_ids)
        if not sample_ids or len(set(sample_ids)) != len(sample_ids):
            raise ProtocolError("Case sample identities must be nonempty and unique.")
        probabilities = _probability_bytes(self.probability_bytes)
        if len(sample_ids) != len(probabilities):
            raise ProtocolError("Case sample identities and probabilities must align.")
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "probability_bytes", probabilities)

    @property
    def case_key(self) -> tuple[str, str]:
        return (self.outer_target_id, self.case_id)

    @property
    def action_id(self) -> str:
        if self.action_kind is ActionKind.HXE:
            assert self.candidate_source_id is not None
            return f"HXE:{self.candidate_source_id}"
        return self.action_kind.value


@dataclass(frozen=True, kw_only=True)
class SupportSummary:
    comparison: Comparison
    candidate_source_id: str | None
    donor_ids: tuple[str, ...]
    paired_case_count: int
    class_counts: tuple[int, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "comparison", Comparison(self.comparison))
        candidate = self.candidate_source_id
        if self.comparison is Comparison.U_VS_B:
            if candidate is not None:
                raise ProtocolError("U-vs-B support cannot name an expert.")
        else:
            object.__setattr__(self, "candidate_source_id", canonical_text(candidate, name="candidate_source_id"))
        donors = tuple(sorted({canonical_text(value, name="donor_id") for value in self.donor_ids}))
        if donors != self.donor_ids:
            raise ProtocolError("Support donor identities must be sorted and unique.")
        if type(self.paired_case_count) is not int or self.paired_case_count < 0:
            raise ProtocolError("Paired-case support must be a nonnegative integer.")
        if (
            type(self.class_counts) is not tuple
            or len(self.class_counts) != 2
            or any(type(value) is not int or value < 0 for value in self.class_counts)
        ):
            raise ProtocolError("Support class counts are malformed.")

    @property
    def donor_count(self) -> int:
        return len(self.donor_ids)


@dataclass(frozen=True, kw_only=True)
class PolicyConfig:
    case_equal_bacc_contribution_gain_threshold: float = 0.0
    brier_noninferiority_margin: float = 0.0
    log_loss_noninferiority_margin: float = 0.0
    max_calibrated_geometry_ratio: float = 1.0
    min_compatibility_shrinkage: float = 0.25
    min_donor_count: int = 4
    min_paired_case_count: int = 16

    def __post_init__(self) -> None:
        for name in (
            "case_equal_bacc_contribution_gain_threshold",
            "brier_noninferiority_margin",
            "log_loss_noninferiority_margin",
            "max_calibrated_geometry_ratio",
            "min_compatibility_shrinkage",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ProtocolError(f"{name} must be finite.")
            object.__setattr__(self, name, value)
        if self.max_calibrated_geometry_ratio <= 0 or not 0.0 <= self.min_compatibility_shrinkage <= 1.0:
            raise ProtocolError("HARP v4 geometry gates are invalid.")
        if (
            type(self.min_donor_count) is not int
            or self.min_donor_count < 1
            or type(self.min_paired_case_count) is not int
            or self.min_paired_case_count < 1
        ):
            raise ProtocolError("HARP v4 support gates must be positive integers.")


__all__ = (
    "OUTCOME_NAMES",
    "ActionKind",
    "CaseTargetAction",
    "CaseTrainingObservation",
    "Comparison",
    "EffectVector",
    "PolicyConfig",
    "SupportSummary",
    "canonical_features",
    "canonical_text",
)
