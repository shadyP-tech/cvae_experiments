"""Deterministic, label-independent case inventories and split scopes."""
from __future__ import annotations
from collections import Counter, defaultdict
from typing import Sequence
from ...protocol import ProtocolError
from .contracts import LabelFreeCaseMenu, RouterFitConfig, SupportActionOutcome, SupportCaseClassProfile, SurfaceRole
from .hashing import canonical_hash
from .truth import SupportTruthCapability

CaseKey = tuple[str, str]

def center_stratified_folds(
    case_keys: Sequence[CaseKey],
    *,
    fold_count: int,
    namespace: str,
) -> tuple[tuple[CaseKey, ...], ...]:
    keys = tuple(sorted(case_keys))
    if (
        type(fold_count) is not int
        or fold_count < 2
        or len(keys) != len(set(keys))
        or len(keys) < fold_count
    ):
        raise ProtocolError("HARP v19 center-stratified fold inventory is malformed.")
    grouped: dict[str, list[CaseKey]] = defaultdict(list)
    for key in keys:
        grouped[key[0]].append(key)
    folds: list[list[CaseKey]] = [[] for _ in range(fold_count)]
    offset = 0
    for center, rows in sorted(grouped.items()):
        ordered = sorted(
            rows,
            key=lambda key: (
                canonical_hash(
                    {
                        "schema_version": "pooled_pairwise_center_stratified_fold_key_v19",
                        "namespace": namespace,
                        "center_id": center,
                        "case_id": key[1],
                    }
                ),
                key[1],
            ),
        )
        for ordinal, key in enumerate(ordered):
            folds[(offset + ordinal) % fold_count].append(key)
        offset = (offset + len(ordered)) % fold_count
    output = tuple(tuple(sorted(rows)) for rows in folds)
    if any(not rows for rows in output) or set(key for rows in output for key in rows) != set(keys):
        raise ProtocolError("HARP v19 center-stratified folds are incomplete.")
    return output


def validate_source_inventory(
    menus: Sequence[LabelFreeCaseMenu],
    capability: SupportTruthCapability,
    *,
    config: RouterFitConfig,
) -> tuple[LabelFreeCaseMenu, ...]:
    rows = tuple(sorted(menus, key=lambda row: (row.center_id, row.case_id)))
    keys = tuple((row.center_id, row.case_id) for row in rows)
    center_counts = Counter(center for center, _ in keys)
    required_per_center = max(
        config.minimum_cases_per_center,
        config.outer_folds,
        config.inner_folds + 1,
    )
    if (
        not rows
        or any(not isinstance(row, LabelFreeCaseMenu) for row in rows)
        or any(row.surface_role is not SurfaceRole.SOURCE_TRAIN_DEVELOPMENT for row in rows)
        or len(keys) != len(set(keys))
        or keys != capability.case_keys
        or min(center_counts.values(), default=0) < required_per_center
        or (
            config.required_source_case_count is not None
            and len(rows) != config.required_source_case_count
        )
        or (
            config.required_source_center_count is not None
            and len(center_counts) != config.required_source_center_count
        )
    ):
        raise ProtocolError(
            "HARP v19 requires the exact center-stratified source-train case inventory."
        )
    return rows


def _subset_menus(rows: Sequence[LabelFreeCaseMenu], keys: set[CaseKey]) -> tuple[LabelFreeCaseMenu, ...]:
    return tuple(row for row in rows if (row.center_id, row.case_id) in keys)


def _subset_profiles(
    rows: Sequence[SupportCaseClassProfile], keys: set[CaseKey]
) -> tuple[SupportCaseClassProfile, ...]:
    return tuple(row for row in rows if (row.center_id, row.case_id) in keys)


def _subset_outcomes(
    rows: Sequence[SupportActionOutcome], keys: set[CaseKey]
) -> tuple[SupportActionOutcome, ...]:
    return tuple(
        row for row in rows if (row.action.center_id, row.action.case_id) in keys
    )

