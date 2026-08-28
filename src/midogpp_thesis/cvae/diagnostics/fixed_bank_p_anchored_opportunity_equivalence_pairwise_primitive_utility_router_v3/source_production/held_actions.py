"""Frozen held-pair B/U/A1 menu for direct-input-#3 production.

Every source pseudo-target block holds out both the eventual outer center H and
the observed source center q.  The normalization below makes the seven-source
held library comparable to the eight-source final library without allowing
source outcomes to reach the scaler or action compiler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from types import MappingProxyType
from typing import Mapping, Sequence

from ....protocol import ProtocolError
from ..hashing import canonical_hash, require_sha256
from ..identity import CENTERS


B_ACTION_ID = "B"
U_ACTION_ID = "U"
A1_ACTION_PREFIX = "A1::source="
TRAINING_SEEDS = (17, 42, 101)
GENERATION_SEEDS = (17, 42, 101)

B_ROWS_PER_SOURCE_CLASS = 128
U_ROWS_PER_SOURCE_CLASS = 144
A1_SELECTED_ROWS_PER_CLASS = 256
A1_OTHER_ROWS_PER_CLASS = 128

B_U_HELD_TO_FINAL_NORMALIZATION = 8.0 / 7.0
A1_HELD_TO_FINAL_NORMALIZATION = 72.0 / 65.0
A1_SELECTED_BASE_WEIGHT = 23.0 / 16.0
A1_OTHER_BASE_WEIGHT = 7.0 / 8.0


def _pair(values: Sequence[object]) -> tuple[str, str]:
    pair = tuple(sorted(str(value) for value in values))
    if len(pair) != 2 or len(set(pair)) != 2 or any(value not in CENTERS for value in pair):
        raise ProtocolError("OE-PPUR v3 held action requires two known distinct centers.")
    return pair[0], pair[1]


def held_candidate_sources(excluded_centers: Sequence[object]) -> tuple[str, ...]:
    excluded = set(_pair(excluded_centers))
    return tuple(center for center in CENTERS if center not in excluded)


def a1_action_id(source_center: object) -> str:
    source = str(source_center)
    if source not in CENTERS:
        raise ProtocolError("OE-PPUR v3 held A1 source center is unknown.")
    return f"{A1_ACTION_PREFIX}{source}"


@dataclass(frozen=True, slots=True)
class HeldMassPolicyReceipt:
    """Explicit numeric mass/scaler policy shared by every held pair."""

    b_u_normalization: float = B_U_HELD_TO_FINAL_NORMALIZATION
    a1_normalization: float = A1_HELD_TO_FINAL_NORMALIZATION
    scaler_uses_sample_weight: bool = False
    logistic_uses_sample_weight: bool = True
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            float(self.b_u_normalization) != 8.0 / 7.0
            or float(self.a1_normalization) != 72.0 / 65.0
            or self.scaler_uses_sample_weight is not False
            or self.logistic_uses_sample_weight is not True
        ):
            raise ProtocolError("OE-PPUR v3 held mass policy drifted.")
        object.__setattr__(self, "b_u_normalization", 8.0 / 7.0)
        object.__setattr__(self, "a1_normalization", 72.0 / 65.0)
        object.__setattr__(
            self,
            "receipt_hash",
            canonical_hash(
                {
                    "schema_version": "oe_ppur_v3_held_mass_policy_v1",
                    "held_candidate_count": 7,
                    "final_candidate_count": 8,
                    "B_U_normalization": "8/7",
                    "A1_normalization": "72/65",
                    "B_effective_mass_per_class": 1024.0,
                    "U_effective_mass_per_class": 1152.0,
                    "A1_effective_mass_per_class": 1152.0,
                    "scaler_fit": "unweighted_synthetic_rows_only",
                    "sample_weight_scope": "logistic_regression_fit_only",
                    "labels_used": False,
                }
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v3_held_mass_policy_v1",
            "B_U_normalization": "8/7",
            "A1_normalization": "72/65",
            "scaler_fit": "unweighted_synthetic_rows_only",
            "sample_weight_scope": "logistic_regression_fit_only",
            "receipt_hash": self.receipt_hash,
        }


@dataclass(frozen=True, slots=True)
class HeldActionSpec:
    excluded_centers: tuple[str, str]
    action_id: str
    selected_source: str | None
    counts_by_class: Mapping[str, Mapping[str, int]]
    sample_weight_by_source: Mapping[str, float]
    mass_policy_receipt_hash: str
    action_hash: str = field(init=False)

    def __post_init__(self) -> None:
        excluded = _pair(self.excluded_centers)
        sources = held_candidate_sources(excluded)
        selected = None if self.selected_source is None else str(self.selected_source)
        policy_hash = require_sha256(
            self.mass_policy_receipt_hash, "held mass policy receipt hash"
        )
        try:
            counts = {
                label: {source: int(self.counts_by_class[label][source]) for source in sources}
                for label in ("0", "1")
            }
            weights = {source: float(self.sample_weight_by_source[source]) for source in sources}
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("OE-PPUR v3 held action mapping drifted.") from exc
        if self.action_id == B_ACTION_ID:
            expected_counts = {source: B_ROWS_PER_SOURCE_CLASS for source in sources}
            expected_weights = {source: 8.0 / 7.0 for source in sources}
            valid = selected is None
        elif self.action_id == U_ACTION_ID:
            expected_counts = {source: U_ROWS_PER_SOURCE_CLASS for source in sources}
            expected_weights = {source: 8.0 / 7.0 for source in sources}
            valid = selected is None
        else:
            expected_counts = {
                source: A1_SELECTED_ROWS_PER_CLASS if source == selected else A1_OTHER_ROWS_PER_CLASS
                for source in sources
            }
            expected_weights = {
                source: (
                    A1_SELECTED_BASE_WEIGHT if source == selected else A1_OTHER_BASE_WEIGHT
                )
                * A1_HELD_TO_FINAL_NORMALIZATION
                for source in sources
            }
            valid = selected in sources and self.action_id == a1_action_id(selected)
        if (
            not valid
            or any(counts[label] != expected_counts for label in ("0", "1"))
            or weights != expected_weights
        ):
            raise ProtocolError("OE-PPUR v3 held action numeric policy drifted.")
        body = {
            "schema_version": "oe_ppur_v3_held_action_v1",
            "excluded_centers": excluded,
            "candidate_sources": sources,
            "action_id": self.action_id,
            "selected_source": selected,
            "counts_by_class": counts,
            "sample_weight_by_source": weights,
            "mass_policy_receipt_hash": policy_hash,
            "scaler_fit_used_sample_weight": False,
            "sample_weight_scope": "logistic_regression_fit_only",
            "labels_used": False,
        }
        object.__setattr__(self, "excluded_centers", excluded)
        object.__setattr__(self, "selected_source", selected)
        object.__setattr__(self, "mass_policy_receipt_hash", policy_hash)
        object.__setattr__(
            self,
            "counts_by_class",
            MappingProxyType({label: MappingProxyType(counts[label]) for label in ("0", "1")}),
        )
        object.__setattr__(self, "sample_weight_by_source", MappingProxyType(weights))
        object.__setattr__(self, "action_hash", canonical_hash(body))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v3_held_action_v1",
            "excluded_centers": list(self.excluded_centers),
            "candidate_sources": list(held_candidate_sources(self.excluded_centers)),
            "action_id": self.action_id,
            "selected_source": self.selected_source,
            "counts_by_class": {label: dict(self.counts_by_class[label]) for label in ("0", "1")},
            "sample_weight_by_source": dict(self.sample_weight_by_source),
            "mass_policy_receipt_hash": self.mass_policy_receipt_hash,
            "scaler_fit_used_sample_weight": False,
            "sample_weight_scope": "logistic_regression_fit_only",
            "labels_used": False,
            "action_hash": self.action_hash,
        }


@dataclass(frozen=True, slots=True)
class HeldActionLibraryReceipt:
    mass_policy: HeldMassPolicyReceipt
    pair_action_hashes: tuple[tuple[tuple[str, str], tuple[str, ...]], ...]
    library_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.mass_policy, HeldMassPolicyReceipt):
            raise ProtocolError("OE-PPUR v3 held library lacks its mass policy.")
        expected_pairs = tuple(
            (CENTERS[i], CENTERS[j])
            for i in range(len(CENTERS))
            for j in range(i + 1, len(CENTERS))
        )
        rows = tuple(self.pair_action_hashes)
        if (
            tuple(pair for pair, _ in rows) != expected_pairs
            or any(len(hashes) != 9 for _, hashes in rows)
            or any(require_sha256(value, "held action hash") != value for _, hashes in rows for value in hashes)
        ):
            raise ProtocolError("OE-PPUR v3 held action-library coverage drifted.")
        object.__setattr__(self, "pair_action_hashes", rows)
        object.__setattr__(
            self,
            "library_hash",
            canonical_hash(
                {
                    "schema_version": "oe_ppur_v3_held_action_library_v1",
                    "mass_policy_receipt_hash": self.mass_policy.receipt_hash,
                    "pair_action_hashes": rows,
                    "unordered_held_pair_count": 36,
                    "action_count_per_pair": 9,
                    "labels_used": False,
                }
            ),
        )


def _build_action(pair: tuple[str, str], action_id: str, selected: str | None, policy: HeldMassPolicyReceipt) -> HeldActionSpec:
    sources = held_candidate_sources(pair)
    counts = {
        label: {
            source: (
                B_ROWS_PER_SOURCE_CLASS
                if action_id == B_ACTION_ID
                else U_ROWS_PER_SOURCE_CLASS
                if action_id == U_ACTION_ID
                else A1_SELECTED_ROWS_PER_CLASS
                if source == selected
                else A1_OTHER_ROWS_PER_CLASS
            )
            for source in sources
        }
        for label in ("0", "1")
    }
    weights = {
        source: (
            B_U_HELD_TO_FINAL_NORMALIZATION
            if selected is None
            else (A1_SELECTED_BASE_WEIGHT if source == selected else A1_OTHER_BASE_WEIGHT)
            * A1_HELD_TO_FINAL_NORMALIZATION
        )
        for source in sources
    }
    return HeldActionSpec(pair, action_id, selected, counts, weights, policy.receipt_hash)


@lru_cache(maxsize=36)
def actions_for_held_pair(first: object, second: object) -> tuple[HeldActionSpec, ...]:
    pair = _pair((first, second))
    policy = HeldMassPolicyReceipt()
    sources = held_candidate_sources(pair)
    return (
        _build_action(pair, B_ACTION_ID, None, policy),
        _build_action(pair, U_ACTION_ID, None, policy),
        *(_build_action(pair, a1_action_id(source), source, policy) for source in sources),
    )


@lru_cache(maxsize=1)
def canonical_held_action_library() -> HeldActionLibraryReceipt:
    policy = HeldMassPolicyReceipt()
    rows = []
    for i, first in enumerate(CENTERS):
        for second in CENTERS[i + 1 :]:
            rows.append(((first, second), tuple(action.action_hash for action in actions_for_held_pair(first, second))))
    return HeldActionLibraryReceipt(policy, tuple(rows))


__all__ = (
    "B_ACTION_ID",
    "GENERATION_SEEDS",
    "HeldActionLibraryReceipt",
    "HeldActionSpec",
    "HeldMassPolicyReceipt",
    "TRAINING_SEEDS",
    "U_ACTION_ID",
    "a1_action_id",
    "actions_for_held_pair",
    "canonical_held_action_library",
    "held_candidate_sources",
)
