"""Immutable v2 DTOs for probabilities and pooled sufficient statistics.

No per-case balanced accuracy is represented here.  A case may contain only
one class; balanced accuracy becomes defined only after sufficient statistics
are pooled over a scope containing both classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from .core_hashing import canonical_hash, finite, nonnegative_int, require_sha256
from .scientific_constants import (
    BASELINE_ACTION_ID,
    EXPECTED_SEED_PAIR_COUNT,
    MIDOGPP_CENTERS,
    action_ids,
)


def _text(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ProtocolError(f"{name} must be a non-empty string.")
    return value


@dataclass(frozen=True, order=True)
class CaseIdentityRow:
    target_center: str
    case_id: str
    sample_id: str

    def __post_init__(self) -> None:
        target = _text(self.target_center, "target_center")
        if target not in MIDOGPP_CENTERS:
            raise ProtocolError("Case identity uses an unknown MIDOG++ center.")
        _text(self.case_id, "case_id")
        _text(self.sample_id, "sample_id")

    @property
    def case_key(self) -> tuple[str, str]:
        return (self.target_center, self.case_id)

    def to_payload(self) -> dict[str, object]:
        return {
            "target_center": self.target_center,
            "case_id": self.case_id,
            "sample_id": self.sample_id,
        }


@dataclass(frozen=True, order=True)
class BinaryLabelRow:
    target_center: str
    case_id: str
    sample_id: str
    label: int

    def __post_init__(self) -> None:
        CaseIdentityRow(self.target_center, self.case_id, self.sample_id)
        if isinstance(self.label, bool) or self.label not in (0, 1):
            raise ProtocolError("Binary labels must be integer zero or one.")

    @property
    def case_key(self) -> tuple[str, str]:
        return (self.target_center, self.case_id)


@dataclass(frozen=True, order=True)
class SeedProbabilityRow:
    target_center: str
    case_id: str
    sample_id: str
    action_id: str
    seed_pair_ordinal: int
    probability: float
    probability_store_hash: str
    predictions_globally_sealed_before_labels: bool = True

    def __post_init__(self) -> None:
        CaseIdentityRow(self.target_center, self.case_id, self.sample_id)
        if self.action_id not in action_ids(self.target_center):
            if self.action_id == self.target_center:
                raise ProtocolError("The held-out target expert cannot be an action.")
            raise ProtocolError("Probability row contains an unknown action.")
        ordinal = self.seed_pair_ordinal
        if (
            isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or ordinal < 0
            or ordinal >= EXPECTED_SEED_PAIR_COUNT
        ):
            raise ProtocolError("Seed-pair ordinal is outside the exact-nine lock.")
        probability = finite(self.probability, "probability")
        if probability < 0.0 or probability > 1.0:
            raise ProtocolError("Probability must lie in [0, 1].")
        require_sha256(self.probability_store_hash, "probability_store_hash")
        if self.predictions_globally_sealed_before_labels is not True:
            raise ProtocolError("All action probabilities must be sealed before labels.")
        object.__setattr__(self, "probability", probability)

    def to_payload(self) -> dict[str, object]:
        return {
            "target_center": self.target_center,
            "case_id": self.case_id,
            "sample_id": self.sample_id,
            "action_id": self.action_id,
            "seed_pair_ordinal": self.seed_pair_ordinal,
            "probability": self.probability,
            "probability_store_hash": self.probability_store_hash,
            "predictions_globally_sealed_before_labels": True,
            "target_expert_used": False,
            "support_labels_used": False,
            "evaluation_labels_used": False,
        }


@dataclass(frozen=True, order=True)
class AggregatedProbabilityRow:
    target_center: str
    case_id: str
    sample_id: str
    action_id: str
    probability_mean: float
    probability_sd: float
    seed_pair_count: int
    seed_probability_hash: str
    row_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        CaseIdentityRow(self.target_center, self.case_id, self.sample_id)
        if self.action_id not in action_ids(self.target_center):
            raise ProtocolError("Aggregated probability uses an invalid action.")
        mean = finite(self.probability_mean, "probability_mean")
        sd = finite(self.probability_sd, "probability_sd")
        if mean < 0.0 or mean > 1.0 or sd < 0.0:
            raise ProtocolError("Aggregated probability moments are invalid.")
        if self.seed_pair_count != EXPECTED_SEED_PAIR_COUNT:
            raise ProtocolError("Aggregated probability is not exact-nine.")
        require_sha256(self.seed_probability_hash, "seed_probability_hash")
        object.__setattr__(self, "probability_mean", mean)
        object.__setattr__(self, "probability_sd", sd)
        object.__setattr__(self, "row_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pooled_bacc_aggregated_probability_row_v2",
            "target_center": self.target_center,
            "case_id": self.case_id,
            "sample_id": self.sample_id,
            "action_id": self.action_id,
            "probability_mean": self.probability_mean,
            "probability_sd": self.probability_sd,
            "seed_pair_count": self.seed_pair_count,
            "seed_probability_hash": self.seed_probability_hash,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "row_hash": self.row_hash}


@dataclass(frozen=True)
class SealedProbabilitySurface:
    rows: tuple[AggregatedProbabilityRow, ...]
    probability_store_hash: str
    surface_hash: str
    predictions_globally_sealed_before_labels: bool = True
    labels_readable_during_materialization: bool = False

    def __post_init__(self) -> None:
        rows = tuple(self.rows)
        require_sha256(self.probability_store_hash, "probability_store_hash")
        require_sha256(self.surface_hash, "surface_hash")
        if not rows or tuple(sorted(rows, key=_probability_sort_key)) != rows:
            raise ProtocolError("Sealed probability rows must be non-empty and canonical.")
        keys = {(r.target_center, r.case_id, r.sample_id, r.action_id) for r in rows}
        if len(keys) != len(rows):
            raise ProtocolError("Sealed probability surface contains duplicate action rows.")
        for identity in {
            (row.target_center, row.case_id, row.sample_id) for row in rows
        }:
            observed = {key[3] for key in keys if key[:3] == identity}
            if observed != set(action_ids(identity[0])):
                raise ProtocolError("Every probability row needs B and eight legal sources.")
        if (
            self.predictions_globally_sealed_before_labels is not True
            or self.labels_readable_during_materialization is not False
        ):
            raise ProtocolError("Probability surface violated global pre-label sealing.")
        expected = canonical_hash(_probability_surface_payload(rows, self.probability_store_hash))
        if expected != self.surface_hash:
            raise ProtocolError("Sealed probability surface hash drifted.")
        object.__setattr__(self, "rows", rows)

    @property
    def identities(self) -> tuple[CaseIdentityRow, ...]:
        return tuple(
            CaseIdentityRow(*key)
            for key in sorted(
                {(r.target_center, r.case_id, r.sample_id) for r in self.rows}
            )
        )

    def probabilities(self) -> Mapping[tuple[str, str, str, str], float]:
        return {
            (row.target_center, row.case_id, row.sample_id, row.action_id): row.probability_mean
            for row in self.rows
        }

    def to_payload(self) -> dict[str, object]:
        return {
            **_probability_surface_payload(self.rows, self.probability_store_hash),
            "surface_hash": self.surface_hash,
        }


@dataclass(frozen=True, order=True)
class CaseActionSufficientStatistics:
    """Hard-prediction class counts for one whole case and one action."""

    target_center: str
    case_id: str
    action_id: str
    n_positive: int
    true_positive: int
    n_negative: int
    true_negative: int
    statistic_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        if self.target_center not in MIDOGPP_CENTERS:
            raise ProtocolError("Sufficient statistic uses an unknown target center.")
        _text(self.case_id, "case_id")
        if self.action_id not in action_ids(self.target_center):
            raise ProtocolError("Sufficient statistic uses an invalid action.")
        for name in ("n_positive", "true_positive", "n_negative", "true_negative"):
            nonnegative_int(getattr(self, name), name)
        if self.n_positive + self.n_negative <= 0:
            raise ProtocolError("A case statistic cannot be empty.")
        if self.true_positive > self.n_positive or self.true_negative > self.n_negative:
            raise ProtocolError("Correct-prediction counts exceed their class totals.")
        object.__setattr__(self, "statistic_hash", canonical_hash(self._unhashed()))

    @property
    def case_key(self) -> tuple[str, str]:
        return (self.target_center, self.case_id)

    @property
    def row_count(self) -> int:
        return self.n_positive + self.n_negative

    @property
    def class_counts(self) -> tuple[int, int]:
        return (self.n_positive, self.n_negative)

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pooled_bacc_case_action_statistics_v2",
            "target_center": self.target_center,
            "case_id": self.case_id,
            "action_id": self.action_id,
            "n_positive": self.n_positive,
            "true_positive": self.true_positive,
            "n_negative": self.n_negative,
            "true_negative": self.true_negative,
            "single_class_case_allowed": True,
            "per_case_bacc_stored": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "statistic_hash": self.statistic_hash}


@dataclass(frozen=True)
class SufficientStatisticSurface:
    rows: tuple[CaseActionSufficientStatistics, ...]
    allowed_case_keys: tuple[tuple[str, str], ...]
    label_scope: str
    prerequisite_seal_hash: str
    statistics_surface_hash: str
    hard_threshold: float = 0.5

    def __post_init__(self) -> None:
        rows = tuple(self.rows)
        allowed = tuple(sorted((str(center), str(case)) for center, case in self.allowed_case_keys))
        if not rows or tuple(sorted(rows, key=_statistic_sort_key)) != rows:
            raise ProtocolError("Sufficient-statistic rows must be non-empty and canonical.")
        if not allowed or len(allowed) != len(set(allowed)):
            raise ProtocolError("Statistic label capability must contain unique whole cases.")
        if set(row.case_key for row in rows) != set(allowed):
            raise ProtocolError("Sufficient-statistic rows escaped their label capability.")
        keys = {(row.target_center, row.case_id, row.action_id) for row in rows}
        if len(keys) != len(rows):
            raise ProtocolError("Sufficient-statistic surface contains duplicate actions.")
        by_case: dict[tuple[str, str], list[CaseActionSufficientStatistics]] = {}
        for row in rows:
            by_case.setdefault(row.case_key, []).append(row)
        for (center, _case), case_rows in by_case.items():
            if {row.action_id for row in case_rows} != set(action_ids(center)):
                raise ProtocolError("Each case must contain every legal action statistic.")
            if len({row.class_counts for row in case_rows}) != 1:
                raise ProtocolError("Class counts drifted across actions within a case.")
        require_sha256(self.prerequisite_seal_hash, "prerequisite_seal_hash")
        require_sha256(self.statistics_surface_hash, "statistics_surface_hash")
        threshold = finite(self.hard_threshold, "hard_threshold")
        if threshold != 0.5:
            raise ProtocolError("Only the frozen 0.5 hard threshold is allowed.")
        expected = canonical_hash(
            _statistics_surface_payload(
                rows,
                allowed,
                label_scope=self.label_scope,
                prerequisite_seal_hash=self.prerequisite_seal_hash,
            )
        )
        if expected != self.statistics_surface_hash:
            raise ProtocolError("Sufficient-statistic surface hash drifted.")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "allowed_case_keys", allowed)
        object.__setattr__(self, "hard_threshold", threshold)

    def by_key(self) -> Mapping[tuple[str, str, str], CaseActionSufficientStatistics]:
        return {(r.target_center, r.case_id, r.action_id): r for r in self.rows}

    def rows_for_action(
        self, action_id: str, *, target_center: str | None = None
    ) -> tuple[CaseActionSufficientStatistics, ...]:
        return tuple(
            row
            for row in self.rows
            if row.action_id == str(action_id)
            and (target_center is None or row.target_center == str(target_center))
        )

    def to_payload(self) -> dict[str, object]:
        return {
            **_statistics_surface_payload(
                self.rows,
                self.allowed_case_keys,
                label_scope=self.label_scope,
                prerequisite_seal_hash=self.prerequisite_seal_hash,
            ),
            "statistics_surface_hash": self.statistics_surface_hash,
        }


def _action_sort_key(action_id: str) -> tuple[int, str]:
    return (0, "") if action_id == BASELINE_ACTION_ID else (1, action_id)


def _probability_sort_key(row: AggregatedProbabilityRow) -> tuple[object, ...]:
    return (
        MIDOGPP_CENTERS.index(row.target_center),
        row.case_id,
        row.sample_id,
        _action_sort_key(row.action_id),
    )


def _statistic_sort_key(row: CaseActionSufficientStatistics) -> tuple[object, ...]:
    return (
        MIDOGPP_CENTERS.index(row.target_center),
        row.case_id,
        _action_sort_key(row.action_id),
    )


def canonical_probability_rows(
    rows: Sequence[AggregatedProbabilityRow],
) -> tuple[AggregatedProbabilityRow, ...]:
    return tuple(sorted(tuple(rows), key=_probability_sort_key))


def canonical_statistic_rows(
    rows: Sequence[CaseActionSufficientStatistics],
) -> tuple[CaseActionSufficientStatistics, ...]:
    return tuple(sorted(tuple(rows), key=_statistic_sort_key))


def make_statistics_surface(
    rows: Sequence[CaseActionSufficientStatistics],
    *,
    allowed_case_keys: Sequence[tuple[str, str]],
    label_scope: str,
    prerequisite_seal_hash: str,
) -> SufficientStatisticSurface:
    canonical = canonical_statistic_rows(rows)
    allowed = tuple(sorted((str(center), str(case)) for center, case in allowed_case_keys))
    payload = _statistics_surface_payload(
        canonical,
        allowed,
        label_scope=label_scope,
        prerequisite_seal_hash=prerequisite_seal_hash,
    )
    return SufficientStatisticSurface(
        rows=canonical,
        allowed_case_keys=allowed,
        label_scope=label_scope,
        prerequisite_seal_hash=prerequisite_seal_hash,
        statistics_surface_hash=canonical_hash(payload),
    )


def _probability_surface_payload(
    rows: Sequence[AggregatedProbabilityRow], probability_store_hash: str
) -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_pooled_bacc_probability_surface_v2",
        "probability_store_hash": probability_store_hash,
        "rows": [row.to_payload() for row in rows],
        "predictions_globally_sealed_before_labels": True,
        "labels_readable_during_materialization": False,
        "target_expert_used": False,
    }


def _statistics_surface_payload(
    rows: Sequence[CaseActionSufficientStatistics],
    allowed_case_keys: Sequence[tuple[str, str]],
    *,
    label_scope: str,
    prerequisite_seal_hash: str,
) -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_pooled_bacc_statistics_surface_v2",
        "label_scope": _text(label_scope, "label_scope"),
        "prerequisite_seal_hash": prerequisite_seal_hash,
        "allowed_case_keys": [list(key) for key in allowed_case_keys],
        "rows": [row.to_payload() for row in rows],
        "hard_threshold": 0.5,
        "single_class_case_allowed": True,
        "per_case_bacc_computed": False,
        "scope_bacc_requires_both_classes": True,
    }


__all__ = (
    "AggregatedProbabilityRow",
    "BinaryLabelRow",
    "CaseActionSufficientStatistics",
    "CaseIdentityRow",
    "SealedProbabilitySurface",
    "SeedProbabilityRow",
    "SufficientStatisticSurface",
    "canonical_probability_rows",
    "canonical_statistic_rows",
    "make_statistics_surface",
)
