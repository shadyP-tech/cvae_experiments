"""Immutable scoped-label count and exact directional-gain products."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction

from ...protocol import ProtocolError
from .constants import CENTERS, DIRECTION_IDS, candidate_sources, physical_action_ids
from .hashing import canonical_hash, require_sha256


@dataclass(frozen=True, order=True)
class BinaryLabel:
    """Ephemeral scoped label. It intentionally has no persistence API."""

    target_center: str
    case_id: str
    sample_id: str
    value: int
    label_scope: str

    def __post_init__(self) -> None:
        value = int(self.value)
        if (
            self.target_center not in CENTERS
            or not self.case_id
            or not self.sample_id
            or not self.label_scope
            or isinstance(self.value, bool)
            or value not in (0, 1)
        ):
            raise ProtocolError("OGDE scoped binary label drifted.")
        object.__setattr__(self, "value", value)

    @property
    def key(self) -> tuple[str, str, str]:
        return self.target_center, self.case_id, self.sample_id


@dataclass(frozen=True, order=True)
class CaseActionConfusion:
    target_center: str
    case_id: str
    action_id: str
    n_positive: int
    true_positive: int
    n_negative: int
    true_negative: int
    flip_0to1_positive: int
    flip_0to1_negative: int
    flip_1to0_positive: int
    flip_1to0_negative: int
    label_scope: str
    confusion_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        counts = tuple(
            int(value)
            for value in (
                self.n_positive,
                self.true_positive,
                self.n_negative,
                self.true_negative,
                self.flip_0to1_positive,
                self.flip_0to1_negative,
                self.flip_1to0_positive,
                self.flip_1to0_negative,
            )
        )
        if (
            self.target_center not in CENTERS
            or not self.case_id
            or self.action_id not in physical_action_ids(self.target_center)
            or not self.label_scope
            or any(value < 0 for value in counts)
            or counts[1] > counts[0]
            or counts[3] > counts[2]
            or counts[4] > counts[0]
            or counts[5] > counts[2]
            or counts[6] > counts[0]
            or counts[7] > counts[2]
        ):
            raise ProtocolError("OGDE case/action confusion drifted.")
        for name, value in zip(
            (
                "n_positive",
                "true_positive",
                "n_negative",
                "true_negative",
                "flip_0to1_positive",
                "flip_0to1_negative",
                "flip_1to0_positive",
                "flip_1to0_negative",
            ),
            counts,
            strict=True,
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "confusion_hash", canonical_hash(self._unhashed()))

    @property
    def case_key(self) -> tuple[str, str]:
        return self.target_center, self.case_id

    @property
    def key(self) -> tuple[str, str, str]:
        return self.target_center, self.case_id, self.action_id

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_ogde_case_action_confusion_v1",
            "target_center": self.target_center,
            "case_id": self.case_id,
            "action_id": self.action_id,
            "n_positive": self.n_positive,
            "true_positive": self.true_positive,
            "n_negative": self.n_negative,
            "true_negative": self.true_negative,
            "flip_0to1_positive": self.flip_0to1_positive,
            "flip_0to1_negative": self.flip_0to1_negative,
            "flip_1to0_positive": self.flip_1to0_positive,
            "flip_1to0_negative": self.flip_1to0_negative,
            "label_scope": self.label_scope,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "confusion_hash": self.confusion_hash}

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "CaseActionConfusion":
        row = cls(
            str(payload["target_center"]), str(payload["case_id"]), str(payload["action_id"]),
            int(payload["n_positive"]), int(payload["true_positive"]),
            int(payload["n_negative"]), int(payload["true_negative"]),
            int(payload["flip_0to1_positive"]), int(payload["flip_0to1_negative"]),
            int(payload["flip_1to0_positive"]), int(payload["flip_1to0_negative"]),
            str(payload["label_scope"]),
        )
        if require_sha256(payload["confusion_hash"], "confusion_hash") != row.confusion_hash:
            raise ProtocolError("OGDE case/action confusion hash drifted after reload.")
        return row


@dataclass(frozen=True, order=True)
class CaseActionSufficientStat:
    """Scope-independent persisted statistic derived from capability-scoped rows."""

    target_center: str
    case_id: str
    action_id: str
    n_positive: int
    true_positive: int
    n_negative: int
    true_negative: int
    flip_0to1_positive: int
    flip_0to1_negative: int
    flip_1to0_positive: int
    flip_1to0_negative: int
    stat_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        # Reuse the scoped validator with a non-persisted synthetic capability.
        checked = CaseActionConfusion(
            self.target_center, self.case_id, self.action_id,
            self.n_positive, self.true_positive, self.n_negative, self.true_negative,
            self.flip_0to1_positive, self.flip_0to1_negative,
            self.flip_1to0_positive, self.flip_1to0_negative,
            "scope_independent_sufficient_stat_validation",
        )
        for name in (
            "n_positive", "true_positive", "n_negative", "true_negative",
            "flip_0to1_positive", "flip_0to1_negative",
            "flip_1to0_positive", "flip_1to0_negative",
        ):
            object.__setattr__(self, name, getattr(checked, name))
        object.__setattr__(self, "stat_hash", canonical_hash(self._unhashed()))

    @property
    def key(self) -> tuple[str, str, str]:
        return self.target_center, self.case_id, self.action_id

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_ogde_case_action_sufficient_stat_v1",
            "target_center": self.target_center,
            "case_id": self.case_id,
            "action_id": self.action_id,
            "n_positive": self.n_positive,
            "true_positive": self.true_positive,
            "n_negative": self.n_negative,
            "true_negative": self.true_negative,
            "flip_0to1_positive": self.flip_0to1_positive,
            "flip_0to1_negative": self.flip_0to1_negative,
            "flip_1to0_positive": self.flip_1to0_positive,
            "flip_1to0_negative": self.flip_1to0_negative,
            "scope_independent_after_capability_scoped_scoring": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "stat_hash": self.stat_hash}

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "CaseActionSufficientStat":
        row = cls(
            str(payload["target_center"]), str(payload["case_id"]), str(payload["action_id"]),
            int(payload["n_positive"]), int(payload["true_positive"]),
            int(payload["n_negative"]), int(payload["true_negative"]),
            int(payload["flip_0to1_positive"]), int(payload["flip_0to1_negative"]),
            int(payload["flip_1to0_positive"]), int(payload["flip_1to0_negative"]),
        )
        if require_sha256(payload["stat_hash"], "stat_hash") != row.stat_hash:
            raise ProtocolError("OGDE case/action sufficient-stat hash drifted after reload.")
        return row


def sufficient_stat_from_confusion(row: CaseActionConfusion) -> CaseActionSufficientStat:
    return CaseActionSufficientStat(
        row.target_center, row.case_id, row.action_id,
        row.n_positive, row.true_positive, row.n_negative, row.true_negative,
        row.flip_0to1_positive, row.flip_0to1_negative,
        row.flip_1to0_positive, row.flip_1to0_negative,
    )


def deduplicate_sufficient_stats(
    rows: tuple[CaseActionConfusion, ...] | list[CaseActionConfusion],
) -> tuple[CaseActionSufficientStat, ...]:
    result: dict[tuple[str, str, str], CaseActionSufficientStat] = {}
    for row in rows:
        stat = sufficient_stat_from_confusion(row)
        previous = result.setdefault(stat.key, stat)
        if previous != stat:
            raise ProtocolError("OGDE capability-scoped counts disagree for one physical case/action.")
    return tuple(result[key] for key in sorted(result))


@dataclass(frozen=True, order=True)
class DirectionalGain:
    query_center: str
    excluded_case_id: str | None
    source: str
    direction: str
    n_positive: int
    n_negative: int
    favorable_count: int
    adverse_count: int
    contributing_case_ids: tuple[str, ...]
    label_scope: str
    numerator: int = field(init=False, compare=True)
    denominator: int = field(init=False, compare=True)
    value: float = field(init=False, compare=True)
    gain_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        cases = tuple(sorted(str(value) for value in self.contributing_case_ids))
        positive, negative = int(self.n_positive), int(self.n_negative)
        favorable, adverse = int(self.favorable_count), int(self.adverse_count)
        if (
            self.query_center not in CENTERS
            or self.source not in candidate_sources(self.query_center)
            or self.direction not in DIRECTION_IDS
            or positive <= 0 or negative <= 0 or favorable < 0 or adverse < 0
            or not cases or len(cases) != len(set(cases))
            or (self.excluded_case_id is not None and self.excluded_case_id in cases)
            or not self.label_scope
        ):
            raise ProtocolError("OGDE directional gain drifted.")
        favorable_denominator = positive if self.direction == "zero_to_one" else negative
        adverse_denominator = negative if self.direction == "zero_to_one" else positive
        exact = Fraction(favorable, 2 * favorable_denominator) - Fraction(adverse, 2 * adverse_denominator)
        object.__setattr__(self, "contributing_case_ids", cases)
        object.__setattr__(self, "n_positive", positive)
        object.__setattr__(self, "n_negative", negative)
        object.__setattr__(self, "favorable_count", favorable)
        object.__setattr__(self, "adverse_count", adverse)
        object.__setattr__(self, "numerator", exact.numerator)
        object.__setattr__(self, "denominator", exact.denominator)
        object.__setattr__(self, "value", float(exact))
        object.__setattr__(self, "gain_hash", canonical_hash(self._unhashed()))

    @property
    def exact(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_ogde_directional_gain_v1",
            "query_center": self.query_center,
            "excluded_case_id": self.excluded_case_id,
            "source": self.source,
            "direction": self.direction,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "favorable_count": self.favorable_count,
            "adverse_count": self.adverse_count,
            "contributing_case_ids": list(self.contributing_case_ids),
            "label_scope": self.label_scope,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "value": self.value,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "gain_hash": self.gain_hash}


__all__ = (
    "BinaryLabel",
    "CaseActionConfusion",
    "CaseActionSufficientStat",
    "DirectionalGain",
    "deduplicate_sufficient_stats",
    "sufficient_stat_from_confusion",
)
