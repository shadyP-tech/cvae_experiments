"""Exact label-capability grant topology and phase-order validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from ...runtime.artifact_io import read_json
from .constants import CENTERS, EXPECTED_PSEUDO_ROUTE_COUNT, EXPECTED_TOTAL_CASE_COUNT
from .hashing import canonical_hash
from .validation_plans import PlanPosteriorTopology
from .validation_shared import Row, fail, index_rows, string_list, support_identities


def validate_capability_topology(
    root: Path, *, topology: PlanPosteriorTopology, capability: Row
) -> None:
    events = capability.get("events")
    if not isinstance(events, list):
        fail("capability events")
    by_role = index_rows(events, ("role",), "capability roles")
    expected: dict[str, dict[str, object]] = {}
    plans = topology.plans

    for target in CENTERS:
        for source in CENTERS:
            if source == target:
                continue
            role = f"source_prior::target={target}::source={source}"
            expected[role] = _event_payload(
                role,
                plans,
                excluded_centers=(target, source),
                outer=None,
                target=target,
                case=None,
                excluded_cases=(),
            )
    for outer in CENTERS:
        for pseudo in CENTERS:
            if pseudo == outer:
                continue
            for source in CENTERS:
                if source in {outer, pseudo}:
                    continue
                role = (
                    f"source_prior::outer_H={outer}::J={pseudo}::source={source}"
                )
                expected[role] = _event_payload(
                    role,
                    plans,
                    excluded_centers=(outer, pseudo, source),
                    outer=outer,
                    target=pseudo,
                    case=None,
                    excluded_cases=(),
                )
    for center, case in plans:
        role = f"outer_support::H={center}::excluded_c={case}"
        expected[role] = _event_from_identities(
            role,
            support_identities(plans, center, case),
            outer=center,
            target=center,
            case=case,
            excluded_centers=(),
            excluded_cases=(case,),
        )
    for outer in CENTERS:
        for pseudo in CENTERS:
            if pseudo == outer:
                continue
            for case in topology.cases_by_center[pseudo]:
                role = (
                    f"PSEUDO_EVALUATION::H={outer}::J={pseudo}::excluded_d={case}"
                )
                identities = tuple(
                    (pseudo, case, sample)
                    for sample in string_list(
                        plans[(pseudo, case)], "evaluation_sample_ids"
                    )
                )
                expected[role] = _event_from_identities(
                    role,
                    identities,
                    outer=outer,
                    target=pseudo,
                    case=case,
                    excluded_centers=(outer,),
                    excluded_cases=(),
                )
    terminal_role = "target_terminal_after_aggregate_seal"
    expected[terminal_role] = _event_payload(
        terminal_role,
        plans,
        excluded_centers=(),
        outer=None,
        target=None,
        case=None,
        excluded_cases=(),
    )

    if set(key[0] for key in by_role) != set(expected):
        fail("capability grant rectangle")
    if [str(row.get("role")) for row in events] != list(expected):
        fail("capability phase order")
    for (role,), row in by_role.items():
        if dict(row) != expected[role]:
            fail("capability grant lineage")

    plan_seal = read_json(root / "manifests/outer_plan_seal.json")
    if (
        capability.get("event_count") != len(events)
        or capability.get("audit_hash") != canonical_hash(events)
        or capability.get("plan_seal_hash") != plan_seal.get("seal_hash")
        or capability.get("target_candidate_seal_complete") is not True
        or capability.get("pre_evaluation_seal_complete") is not True
        or capability.get("pseudo_evaluation_route_count")
        != EXPECTED_PSEUDO_ROUTE_COUNT
        or capability.get("calibration_seal_complete") is not True
        or capability.get("decision_count") != 4 * EXPECTED_TOTAL_CASE_COUNT
        or capability.get("aggregate_seal_complete") is not True
        or capability.get("terminal_opened") is not True
        or capability.get("raw_labels_persisted") is not False
    ):
        fail("capability audit summary")


def _event_payload(
    role: str,
    plans: Mapping[tuple[str, str], Row],
    *,
    excluded_centers: Sequence[str],
    outer: str | None,
    target: str | None,
    case: str | None,
    excluded_cases: Sequence[str],
) -> dict[str, object]:
    excluded = set(excluded_centers)
    identities = tuple(
        sorted(
            (center, case_id, sample)
            for (center, case_id), row in plans.items()
            if center not in excluded
            for sample in string_list(row, "evaluation_sample_ids")
        )
    )
    return _event_from_identities(
        role,
        identities,
        outer=outer,
        target=target,
        case=case,
        excluded_centers=tuple(sorted(excluded)),
        excluded_cases=excluded_cases,
    )


def _event_from_identities(
    role: str,
    identities: Sequence[tuple[str, str, str]],
    *,
    outer: str | None,
    target: str | None,
    case: str | None,
    excluded_centers: Sequence[str],
    excluded_cases: Sequence[str],
) -> dict[str, object]:
    rows = tuple(sorted(identities))
    return {
        "role": role,
        "outer_target_center": outer,
        "target_center": target,
        "case_id": case,
        "excluded_centers": list(excluded_centers),
        "excluded_case_ids": list(excluded_cases),
        "row_count": len(rows),
        "case_count": len(
            {(center, case_id) for center, case_id, _sample in rows}
        ),
        "identity_hash": canonical_hash([list(value) for value in rows]),
        "raw_labels_persisted": False,
    }


__all__ = ("validate_capability_topology",)
