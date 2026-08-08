"""Closed scientific contracts for the fresh exact additive-tail surface.

The development action is deliberately the same action family used at the
target: a seven-source equal-union base with 144 rows per source and class,
optionally followed by a 126-row-per-class tail from exactly one legal source.
It is not a fixed-total perturbation and it is not full-source utility.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from types import MappingProxyType
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ...expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from ...protocol import ProtocolError
from ..residual_topup import build_single_source_tail_action
from ..residual_topup.contracts import (
    INNER_BASE_PER_SOURCE,
    INNER_SOURCE_COUNT,
    INNER_TOPUP_TOTAL_PER_CLASS,
    TopupGeometry,
)


EXPERIMENT_ID = (
    "midogpp.routing_and_composition."
    "uniform_b_v2_exact_tail_utility_surface.v1"
)
OUTPUT_ARTIFACT_ID = "midogpp_output_uniform_b_v2_exact_tail_utility_surface_v1"
CLAIM_SCOPE = "routing_and_composition"

DEVELOPMENT_RESERVATION_ARTIFACT_ID = (
    "midogpp_utility_aligned_router_development_reservation_v1"
)
DEVELOPMENT_CACHE_ARTIFACT_ID = "midogpp_utility_aligned_router_development_cache_v1"
DEVELOPMENT_MANIFEST_ARTIFACT_ID = (
    "midogpp_utility_aligned_router_development_manifest_v1"
)
TARGET_SUPPORT_SURFACE_ARTIFACT_ID = (
    "midogpp_utility_aligned_target_support_surface_v1"
)

BASE_ACTION_ID = "inner_base_equal_union"
TAIL_ACTION_PREFIX = "inner_single_source_tail::"
PRIMARY_METRIC = "balanced_accuracy"
RESPONSE_SEMANTICS = "bacc_exact_additive_tail_minus_bacc_exact_base"
SURFACE_SCHEMA_VERSION = "midogpp_exact_additive_tail_utility_surface_v1"
MINIMUM_SUPPORT_CASE_COUNT = 8

SOURCE_PREFIX_ROWS_PER_CLASS = (
    INNER_BASE_PER_SOURCE + INNER_TOPUP_TOTAL_PER_CLASS
)
EXPECTED_SOURCE_STREAM_COUNT = len(CENTERS) * len(TRAINING_SEEDS) * len(
    GENERATION_SEEDS
)
EXPECTED_COARSE_TASK_COUNT = (
    len(CENTERS)
    * (len(CENTERS) - 1)
    * len(TRAINING_SEEDS)
    * len(GENERATION_SEEDS)
)
EXPECTED_PREDICTION_CELL_COUNT = EXPECTED_COARSE_TASK_COUNT * (
    1 + INNER_SOURCE_COUNT
)
EXPECTED_UTILITY_ROW_COUNT = EXPECTED_COARSE_TASK_COUNT * INNER_SOURCE_COUNT


def development_queries(outer_target: object) -> tuple[str, ...]:
    outer = _center(outer_target, "outer target H")
    return tuple(center for center in CENTERS if center != outer)


def legal_sources(*, outer_target: object, pseudo_query: object) -> tuple[str, ...]:
    outer = _center(outer_target, "outer target H")
    query = _center(pseudo_query, "pseudo-query q")
    if outer == query:
        raise ProtocolError("Exact-tail pseudo-query q must differ from outer H.")
    sources = tuple(center for center in CENTERS if center not in {outer, query})
    if len(sources) != INNER_SOURCE_COUNT:
        raise ProtocolError("Exact-tail inner candidate cardinality drifted from seven.")
    return sources


def target_sources(target_center: object) -> tuple[str, ...]:
    target = _center(target_center, "target H")
    return tuple(center for center in CENTERS if center != target)


def tail_action_id(source_center: object) -> str:
    return f"{TAIL_ACTION_PREFIX}{_center(source_center, 'tail source e')}"


def tail_source(action_id: object) -> str | None:
    action = str(action_id)
    if not action.startswith(TAIL_ACTION_PREFIX):
        return None
    source = action.removeprefix(TAIL_ACTION_PREFIX)
    return _center(source, "tail source e")


@dataclass(frozen=True)
class ExactTailActionSpec:
    """One exact inner base or exact additive one-source tail composition."""

    outer_target: str
    pseudo_query: str
    action_id: str
    selected_source: str | None
    source_order: tuple[str, ...]
    counts_per_class: Mapping[str, int]
    total_per_class: int
    action_hash: str

    def __post_init__(self) -> None:
        expected_sources = legal_sources(
            outer_target=self.outer_target, pseudo_query=self.pseudo_query
        )
        if self.source_order != expected_sources:
            raise ProtocolError("Exact-tail action candidate order drifted.")
        counts = {str(key): int(value) for key, value in self.counts_per_class.items()}
        if tuple(counts) != expected_sources or sum(counts.values()) != int(
            self.total_per_class
        ):
            raise ProtocolError("Exact-tail action counts do not cover its geometry.")
        if self.action_id == BASE_ACTION_ID:
            if self.selected_source is not None or any(
                value != INNER_BASE_PER_SOURCE for value in counts.values()
            ):
                raise ProtocolError("Exact-tail base action is not exact equal union.")
        else:
            selected = tail_source(self.action_id)
            if selected != self.selected_source or selected not in expected_sources:
                raise ProtocolError("Exact-tail selected-source identity drifted.")
            for source, value in counts.items():
                expected = INNER_BASE_PER_SOURCE + (
                    INNER_TOPUP_TOTAL_PER_CLASS if source == selected else 0
                )
                if value != expected:
                    raise ProtocolError("Exact-tail action is not an additive tail.")
        expected_hash = stable_hash(self._unhashed_payload())
        if self.action_hash != expected_hash:
            raise ProtocolError("Exact-tail action hash drifted.")
        object.__setattr__(self, "counts_per_class", MappingProxyType(counts))

    @property
    def is_base(self) -> bool:
        return self.action_id == BASE_ACTION_ID

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_exact_additive_tail_action_v1",
            "outer_target": self.outer_target,
            "pseudo_query": self.pseudo_query,
            "action_id": self.action_id,
            "selected_source": self.selected_source,
            "source_order": list(self.source_order),
            "counts_per_class": dict(self.counts_per_class),
            "total_per_class": int(self.total_per_class),
            "base_per_source": INNER_BASE_PER_SOURCE,
            "topup_total_per_class": (
                0 if self.is_base else INNER_TOPUP_TOTAL_PER_CLASS
            ),
            "response_role": "control" if self.is_base else "exact_additive_tail",
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "action_hash": self.action_hash}


def action_library_for(
    *, outer_target: object, pseudo_query: object
) -> tuple[ExactTailActionSpec, ...]:
    outer = _center(outer_target, "outer target H")
    query = _center(pseudo_query, "pseudo-query q")
    sources = legal_sources(outer_target=outer, pseudo_query=query)
    base_counts = {source: INNER_BASE_PER_SOURCE for source in sources}
    base_payload = {
        "schema_version": "midogpp_exact_additive_tail_action_v1",
        "outer_target": outer,
        "pseudo_query": query,
        "action_id": BASE_ACTION_ID,
        "selected_source": None,
        "source_order": list(sources),
        "counts_per_class": base_counts,
        "total_per_class": INNER_BASE_PER_SOURCE * INNER_SOURCE_COUNT,
        "base_per_source": INNER_BASE_PER_SOURCE,
        "topup_total_per_class": 0,
        "response_role": "control",
    }
    actions = [
        ExactTailActionSpec(
            outer_target=outer,
            pseudo_query=query,
            action_id=BASE_ACTION_ID,
            selected_source=None,
            source_order=sources,
            counts_per_class=base_counts,
            total_per_class=INNER_BASE_PER_SOURCE * INNER_SOURCE_COUNT,
            action_hash=stable_hash(base_payload),
        )
    ]
    geometry = TopupGeometry(
        source_order=sources,
        base_per_source=INNER_BASE_PER_SOURCE,
        topup_total_per_class=INNER_TOPUP_TOTAL_PER_CLASS,
    )
    for source in sources:
        topup = build_single_source_tail_action(source, geometry=geometry)
        counts = dict(topup.final_counts_by_class[0])
        payload = {
            "schema_version": "midogpp_exact_additive_tail_action_v1",
            "outer_target": outer,
            "pseudo_query": query,
            "action_id": tail_action_id(source),
            "selected_source": source,
            "source_order": list(sources),
            "counts_per_class": counts,
            "total_per_class": geometry.final_total_per_class,
            "base_per_source": INNER_BASE_PER_SOURCE,
            "topup_total_per_class": INNER_TOPUP_TOTAL_PER_CLASS,
            "response_role": "exact_additive_tail",
        }
        actions.append(
            ExactTailActionSpec(
                outer_target=outer,
                pseudo_query=query,
                action_id=tail_action_id(source),
                selected_source=source,
                source_order=sources,
                counts_per_class=counts,
                total_per_class=geometry.final_total_per_class,
                action_hash=stable_hash(payload),
            )
        )
    return tuple(actions)


def expected_coarse_task_keys() -> tuple[tuple[str, str, int, int], ...]:
    return tuple(
        (outer, query, training_seed, generation_seed)
        for outer in CENTERS
        for query in development_queries(outer)
        for training_seed, generation_seed in product(
            TRAINING_SEEDS, GENERATION_SEEDS
        )
    )


def expected_prediction_keys() -> tuple[tuple[str, str, str, int, int], ...]:
    return tuple(
        (outer, query, action.action_id, training_seed, generation_seed)
        for outer, query, training_seed, generation_seed in expected_coarse_task_keys()
        for action in action_library_for(outer_target=outer, pseudo_query=query)
    )


def expected_utility_keys() -> tuple[tuple[str, str, str, int, int], ...]:
    return tuple(
        (outer, query, source, training_seed, generation_seed)
        for outer, query, training_seed, generation_seed in expected_coarse_task_keys()
        for source in legal_sources(outer_target=outer, pseudo_query=query)
    )


@dataclass(frozen=True)
class EvaluationRowIdentity:
    """Label-free identity of one whole-case development evaluation row."""

    row_ordinal: int
    manifest_row_index: int
    sample_id: str
    case_id: str
    center: str
    split: str
    cache_shard_path: str
    cache_row_index: int
    partition_role: str = "development_evaluation"

    def __post_init__(self) -> None:
        if (
            isinstance(self.row_ordinal, bool)
            or isinstance(self.manifest_row_index, bool)
            or not isinstance(self.row_ordinal, int)
            or not isinstance(self.manifest_row_index, int)
            or self.row_ordinal < 0
            or self.manifest_row_index < 0
            or isinstance(self.cache_row_index, bool)
            or not isinstance(self.cache_row_index, int)
            or self.cache_row_index < 0
        ):
            raise ProtocolError("Exact-tail row indices must be nonnegative integers.")
        if not self.sample_id or not self.case_id or not self.cache_shard_path:
            raise ProtocolError("Exact-tail rows require sample and case identities.")
        _center(self.center, "evaluation center")
        if not self.split or self.partition_role not in {
            "development_support",
            "development_evaluation",
        }:
            raise ProtocolError("Exact-tail row partition role drifted.")

    def identity_payload(self) -> dict[str, object]:
        return {
            "row_ordinal": self.row_ordinal,
            "manifest_row_index": self.manifest_row_index,
            "sample_id": self.sample_id,
            "case_id": self.case_id,
            "center": self.center,
            "split": self.split,
            "cache_shard_path": self.cache_shard_path,
            "cache_row_index": self.cache_row_index,
            "partition_role": self.partition_role,
        }


def row_identity_hash(rows: Sequence[EvaluationRowIdentity]) -> str:
    return stable_hash([row.identity_payload() for row in rows])


@dataclass(frozen=True)
class DevelopmentPartition:
    """Case-disjoint support/evaluation reservation for one pseudo-domain."""

    center: str
    support_case_ids: tuple[str, ...]
    support_rows: tuple[EvaluationRowIdentity, ...]
    evaluation_rows: tuple[EvaluationRowIdentity, ...]
    target_evaluation_case_ids: tuple[str, ...]
    reservation_hash: str
    labels_present: bool = False

    def __post_init__(self) -> None:
        _center(self.center, "development center")
        support = _canonical_ids(self.support_case_ids, "support case")
        if len(support) < MINIMUM_SUPPORT_CASE_COUNT:
            raise ProtocolError(
                "Exact-tail development support requires at least eight independent cases."
            )
        support_rows = tuple(self.support_rows)
        evaluation = tuple(self.evaluation_rows)
        if (
            not support_rows
            or any(
                row.center != self.center
                or row.partition_role != "development_support"
                for row in support_rows
            )
            or {row.case_id for row in support_rows} != set(support)
        ):
            raise ProtocolError("Development support rows do not match support cases.")
        if not evaluation or any(
            row.center != self.center
            or row.partition_role != "development_evaluation"
            for row in evaluation
        ):
            raise ProtocolError("Development evaluation rows cross pseudo-domains.")
        evaluation_cases = tuple(sorted({row.case_id for row in evaluation}))
        target_eval = _canonical_ids(
            self.target_evaluation_case_ids, "target-evaluation case"
        )
        if set(support) & set(evaluation_cases):
            raise ProtocolError("Development support overlaps development evaluation.")
        if (set(support) | set(evaluation_cases)) & set(target_eval):
            raise ProtocolError("Development rows overlap the sealed target evaluation.")
        if self.labels_present is not False:
            raise ProtocolError("Development partition manifests must be label-free.")
        expected = stable_hash(self._unhashed_payload())
        if self.reservation_hash != expected:
            raise ProtocolError("Development partition reservation hash drifted.")
        object.__setattr__(self, "support_case_ids", support)
        object.__setattr__(self, "support_rows", support_rows)
        object.__setattr__(self, "evaluation_rows", evaluation)
        object.__setattr__(self, "target_evaluation_case_ids", target_eval)

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_exact_tail_development_partition_v1",
            "center": self.center,
            "support_case_ids": list(self.support_case_ids),
            "support_rows": [row.identity_payload() for row in self.support_rows],
            "evaluation_rows": [row.identity_payload() for row in self.evaluation_rows],
            "target_evaluation_case_ids": list(self.target_evaluation_case_ids),
            "labels_present": self.labels_present,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "reservation_hash": self.reservation_hash}


def build_development_partition(
    *,
    center: str,
    support_case_ids: Sequence[str],
    support_rows: Sequence[EvaluationRowIdentity],
    evaluation_rows: Sequence[EvaluationRowIdentity],
    target_evaluation_case_ids: Sequence[str],
) -> DevelopmentPartition:
    values = {
        "center": str(center),
        "support_case_ids": tuple(support_case_ids),
        "support_rows": tuple(support_rows),
        "evaluation_rows": tuple(evaluation_rows),
        "target_evaluation_case_ids": tuple(target_evaluation_case_ids),
        "reservation_hash": "",
        "labels_present": False,
    }
    provisional = DevelopmentPartition.__new__(DevelopmentPartition)
    for key, value in values.items():
        object.__setattr__(provisional, key, value)
    values["reservation_hash"] = stable_hash(provisional._unhashed_payload())
    return DevelopmentPartition(**values)  # type: ignore[arg-type]


def _center(value: object, role: str) -> str:
    rendered = str(value)
    if rendered not in CENTERS:
        raise ProtocolError(f"Exact-tail {role} is outside the frozen center universe.")
    return rendered


def _canonical_ids(values: Sequence[str], role: str) -> tuple[str, ...]:
    normalized = tuple(sorted(str(value) for value in values))
    if not normalized or any(not value for value in normalized) or len(
        set(normalized)
    ) != len(normalized):
        raise ProtocolError(f"Exact-tail {role} IDs are empty or duplicated.")
    return normalized


__all__ = (
    "BASE_ACTION_ID",
    "CENTERS",
    "CLAIM_SCOPE",
    "DEVELOPMENT_CACHE_ARTIFACT_ID",
    "DEVELOPMENT_MANIFEST_ARTIFACT_ID",
    "DEVELOPMENT_RESERVATION_ARTIFACT_ID",
    "EXPECTED_COARSE_TASK_COUNT",
    "EXPECTED_PREDICTION_CELL_COUNT",
    "EXPECTED_SOURCE_STREAM_COUNT",
    "EXPECTED_UTILITY_ROW_COUNT",
    "EXPERIMENT_ID",
    "EvaluationRowIdentity",
    "ExactTailActionSpec",
    "GENERATION_SEEDS",
    "MINIMUM_SUPPORT_CASE_COUNT",
    "OUTPUT_ARTIFACT_ID",
    "PRIMARY_METRIC",
    "RESPONSE_SEMANTICS",
    "SOURCE_PREFIX_ROWS_PER_CLASS",
    "SURFACE_SCHEMA_VERSION",
    "TAIL_ACTION_PREFIX",
    "TARGET_SUPPORT_SURFACE_ARTIFACT_ID",
    "TRAINING_SEEDS",
    "DevelopmentPartition",
    "action_library_for",
    "build_development_partition",
    "development_queries",
    "expected_coarse_task_keys",
    "expected_prediction_keys",
    "expected_utility_keys",
    "legal_sources",
    "row_identity_hash",
    "tail_action_id",
    "tail_source",
    "target_sources",
)
