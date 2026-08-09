"""Small immutable DTOs shared by the scientific ceiling modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from .core_hashing import canonical_hash, finite, require_sha256
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
        action = _text(self.action_id, "action_id")
        if action not in action_ids(self.target_center):
            if action == self.target_center:
                raise ProtocolError("The held-out target expert cannot be an action.")
            raise ProtocolError("Probability row contains an unknown action.")
        if (
            isinstance(self.seed_pair_ordinal, bool)
            or not isinstance(self.seed_pair_ordinal, int)
            or self.seed_pair_ordinal < 0
            or self.seed_pair_ordinal >= EXPECTED_SEED_PAIR_COUNT
        ):
            raise ProtocolError("Seed-pair ordinal is outside the exact-nine lock.")
        probability = finite(self.probability, "probability")
        if probability < 0.0 or probability > 1.0:
            raise ProtocolError("Probability must lie in [0, 1].")
        object.__setattr__(self, "probability", probability)
        require_sha256(self.probability_store_hash, "probability_store_hash")
        if self.predictions_globally_sealed_before_labels is not True:
            raise ProtocolError("All action probabilities must be globally sealed before labels.")

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
        if len({(r.target_center, r.case_id, r.sample_id, r.action_id) for r in rows}) != len(rows):
            raise ProtocolError("Sealed probability surface contains duplicate action rows.")
        if (
            self.predictions_globally_sealed_before_labels is not True
            or self.labels_readable_during_materialization is not False
        ):
            raise ProtocolError("Probability surface violated global pre-label sealing.")
        expected = canonical_hash(
            {
                "schema_version": "fixed_bank_label_aware_probability_surface_v1",
                "probability_store_hash": self.probability_store_hash,
                "rows": [row.to_payload() for row in rows],
                "predictions_globally_sealed_before_labels": True,
                "labels_readable_during_materialization": False,
            }
        )
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


@dataclass(frozen=True, order=True)
class CaseActionUtility:
    target_center: str
    case_id: str
    action_id: str
    sample_count: int
    exact_bacc: float
    smooth_bacc: float
    exact_gain_vs_b: float
    utility_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        if self.target_center not in MIDOGPP_CENTERS or self.action_id not in action_ids(self.target_center):
            raise ProtocolError("Case utility uses an invalid center/action pair.")
        _text(self.case_id, "case_id")
        if isinstance(self.sample_count, bool) or self.sample_count <= 0:
            raise ProtocolError("Case utility sample_count must be positive.")
        for name in ("exact_bacc", "smooth_bacc"):
            value = finite(getattr(self, name), name)
            if value < 0.0 or value > 1.0:
                raise ProtocolError(f"{name} must lie in [0, 1].")
            object.__setattr__(self, name, value)
        gain = finite(self.exact_gain_vs_b, "exact_gain_vs_b")
        if gain < -1.0 or gain > 1.0:
            raise ProtocolError("exact_gain_vs_b must lie in [-1, 1].")
        if self.action_id == BASELINE_ACTION_ID and abs(gain) > 1.0e-12:
            raise ProtocolError("Baseline case utility must have zero gain versus B.")
        object.__setattr__(self, "exact_gain_vs_b", gain)
        object.__setattr__(self, "utility_hash", canonical_hash(self._unhashed()))

    @property
    def case_key(self) -> tuple[str, str]:
        return (self.target_center, self.case_id)

    def _unhashed(self) -> dict[str, object]:
        return {
            "target_center": self.target_center,
            "case_id": self.case_id,
            "action_id": self.action_id,
            "sample_count": self.sample_count,
            "exact_bacc": self.exact_bacc,
            "smooth_bacc": self.smooth_bacc,
            "exact_gain_vs_b": self.exact_gain_vs_b,
        }

    def exact_payload(self) -> dict[str, object]:
        """Decision identity deliberately excludes descriptive smooth utility."""

        return {
            "target_center": self.target_center,
            "case_id": self.case_id,
            "action_id": self.action_id,
            "sample_count": self.sample_count,
            "exact_bacc": self.exact_bacc,
            "exact_gain_vs_b": self.exact_gain_vs_b,
        }


@dataclass(frozen=True)
class CaseUtilitySurface:
    rows: tuple[CaseActionUtility, ...]
    allowed_case_keys: tuple[tuple[str, str], ...]
    label_scope: str
    prerequisite_seal_hash: str
    exact_surface_hash: str
    descriptive_surface_hash: str

    def __post_init__(self) -> None:
        rows = tuple(self.rows)
        keys = tuple(sorted((str(center), str(case)) for center, case in self.allowed_case_keys))
        if not rows or tuple(sorted(rows, key=_utility_sort_key)) != rows:
            raise ProtocolError("Case utilities must be non-empty and canonical.")
        if set(row.case_key for row in rows) != set(keys):
            raise ProtocolError("Case utility rows escaped their label capability.")
        require_sha256(self.prerequisite_seal_hash, "prerequisite_seal_hash")
        require_sha256(self.exact_surface_hash, "exact_surface_hash")
        require_sha256(self.descriptive_surface_hash, "descriptive_surface_hash")
        exact = canonical_hash(
            {
                "schema_version": "fixed_bank_label_aware_case_utility_exact_v1",
                "label_scope": self.label_scope,
                "prerequisite_seal_hash": self.prerequisite_seal_hash,
                "allowed_case_keys": [list(key) for key in keys],
                "rows": [row.exact_payload() for row in rows],
            }
        )
        descriptive = canonical_hash(
            {
                "exact_surface_hash": exact,
                "smooth_bacc": [row.smooth_bacc for row in rows],
            }
        )
        if exact != self.exact_surface_hash or descriptive != self.descriptive_surface_hash:
            raise ProtocolError("Case utility surface hash drifted.")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "allowed_case_keys", keys)

    def by_key(self) -> Mapping[tuple[str, str, str], CaseActionUtility]:
        return {(r.target_center, r.case_id, r.action_id): r for r in self.rows}


def _action_sort_key(action_id: str) -> tuple[int, str]:
    return (0, "") if action_id == BASELINE_ACTION_ID else (1, action_id)


def _probability_sort_key(row: AggregatedProbabilityRow) -> tuple[object, ...]:
    return (
        MIDOGPP_CENTERS.index(row.target_center),
        row.case_id,
        row.sample_id,
        _action_sort_key(row.action_id),
    )


def _utility_sort_key(row: CaseActionUtility) -> tuple[object, ...]:
    return (
        MIDOGPP_CENTERS.index(row.target_center),
        row.case_id,
        _action_sort_key(row.action_id),
    )


def canonical_probability_rows(rows: Sequence[AggregatedProbabilityRow]) -> tuple[AggregatedProbabilityRow, ...]:
    return tuple(sorted(tuple(rows), key=_probability_sort_key))


def canonical_utility_rows(rows: Sequence[CaseActionUtility]) -> tuple[CaseActionUtility, ...]:
    return tuple(sorted(tuple(rows), key=_utility_sort_key))


__all__ = (
    "AggregatedProbabilityRow",
    "BinaryLabelRow",
    "CaseActionUtility",
    "CaseIdentityRow",
    "CaseUtilitySurface",
    "SealedProbabilitySurface",
    "SeedProbabilityRow",
    "canonical_probability_rows",
    "canonical_utility_rows",
)
