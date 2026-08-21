"""Shared fail-closed route topology helpers for replay validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ...protocol import ProtocolError
from .constants import (
    CENTERS,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    EXPECTED_PSEUDO_ROUTE_COUNT,
)
from .hashing import require_sha256
from .posterior_contracts import CONTROL_IDS


Row = Mapping[str, object]
RouteKey = tuple[str, str, str, str]
CaseKey = tuple[str, str]


def index_pseudo_candidates(
    rows: Sequence[Row],
) -> tuple[dict[RouteKey, Row], dict[str, tuple[str, ...]]]:
    indexed: dict[RouteKey, Row] = {}
    for row in rows:
        require_mapping(row, "pseudo candidate")
        key = (
            str(row.get("outer_center", "")),
            str(row.get("center", "")),
            str(row.get("case_id", "")),
            str(row.get("control_id", "")),
        )
        outer, donor, case, control = key
        candidate_hashes = strings(row, "candidate_hashes", allow_empty=True)
        selected = row.get("selected_candidate_hash")
        excluded = strings(row, "source_excluded_centers")
        if (
            key in indexed
            or outer not in CENTERS
            or donor not in CENTERS
            or outer == donor
            or not case
            or control not in CONTROL_IDS
            or excluded != tuple(sorted((outer, donor)))
            or len(candidate_hashes) != len(set(candidate_hashes))
            or (
                selected is not None
                and str(selected) not in set(candidate_hashes)
            )
        ):
            fail("pseudo candidate route topology")
        for digest in candidate_hashes:
            require_sha256(digest, "pseudo candidate action hash")
        if selected is not None:
            require_sha256(selected, "selected pseudo candidate hash")
        require_sha256(row.get("runtime_hash"), "pseudo candidate runtime hash")
        require_sha256(
            row.get("endpoint_lineage_hash"), "pseudo endpoint lineage hash"
        )
        indexed[key] = row

    cases_by_center = {
        donor: tuple(
            sorted(
                {
                    case
                    for (_outer, observed, case, _control) in indexed
                    if observed == donor
                }
            )
        )
        for donor in CENTERS
    }
    if {
        center: len(cases) for center, cases in cases_by_center.items()
    } != dict(EXPECTED_CASE_COUNTS_BY_CENTER):
        fail("pseudo candidate case rectangle")
    expected = {
        (outer, donor, case, control)
        for outer in CENTERS
        for donor in CENTERS
        if donor != outer
        for case in cases_by_center[donor]
        for control in CONTROL_IDS
    }
    if (
        len(indexed) != 2 * EXPECTED_PSEUDO_ROUTE_COUNT
        or set(indexed) != expected
    ):
        fail("pseudo candidate H/J/d/control rectangle")
    return indexed, cases_by_center


def validate_case_sample_counts(
    rows: Mapping[CaseKey, int], cases: Mapping[str, tuple[str, ...]]
) -> dict[CaseKey, int]:
    expected = {
        (center, case) for center in CENTERS for case in cases[center]
    }
    observed = {
        (str(center), str(case)): int(value)
        for (center, case), value in rows.items()
    }
    if set(observed) != expected or any(value <= 0 for value in observed.values()):
        fail("outer-plan case sample counts")
    return observed


def strings(row: Row, key: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    values = list_value(row, key)
    if not values and not allow_empty:
        fail(f"{key} list")
    result = tuple(str(value) for value in values)
    if any(not value for value in result):
        fail(f"{key} identities")
    return result


def list_value(row: Row, key: str) -> list[object]:
    value = row.get(key)
    if not isinstance(value, list):
        fail(f"{key} list")
    return value


def mapping_value(row: Row, key: str) -> Row:
    value = row.get(key)
    if not isinstance(value, Mapping):
        fail(f"{key} mapping")
    return value


def require_mapping(value: object, role: str) -> None:
    if not isinstance(value, Mapping):
        fail(role)


def fail(role: str) -> None:
    raise ProtocolError(f"CBPUPR persisted {role} drifted.")


__all__ = (
    "CaseKey",
    "RouteKey",
    "Row",
    "fail",
    "index_pseudo_candidates",
    "list_value",
    "mapping_value",
    "require_mapping",
    "strings",
    "validate_case_sample_counts",
)
