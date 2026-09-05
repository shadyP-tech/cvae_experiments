"""Durable source-only rejection evidence, committed before admission gates."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from ....protocol import ProtocolError
from ....routing.harp_protocol import canonical_hash
from ....runtime.artifact_io import atomic_json
from ....runtime.harp_v18_execution.durability import durable_barrier


def write_source_diagnostics(
    root: Path, *, fitted: object, source_surface: object, config_hash: str
) -> tuple[Path, ...]:
    """Persist aggregate frontiers; never retain a source truth capability."""

    policy = fitted.state.policy
    crossfit = policy.crossfit.public_payload()
    rows = crossfit.get("frontier_rows")
    oracles = crossfit.get("actual_menu_oracle_diagnostics")
    if not isinstance(rows, list) or not rows or not isinstance(oracles, list) or not oracles:
        raise ProtocolError("HARP v18 cannot admit a policy without its candidate frontier.")
    for row in rows:
        if not isinstance(row, Mapping):
            raise ProtocolError("HARP v18 candidate frontier row is malformed.")
        body = dict(row)
        digest = body.pop("frontier_row_hash", None)
        if canonical_hash(body) != digest:
            raise ProtocolError("HARP v18 candidate frontier hash drifted.")

    case_gains: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_family: dict[str, set[tuple[str, str]]] = defaultdict(set)
    positive: set[tuple[str, str]] = set()
    for _center, outcomes in source_surface.state.outcomes_by_outer:
        for outcome in outcomes:
            key = (outcome.action.center_id, outcome.action.case_id)
            case_gains[key]  # Include cases with no safe-positive action.
            if outcome.bacc_gain > 0.0:
                positive.add(key)
                if outcome.brier_delta <= 0.0 and outcome.log_loss_delta <= 0.0:
                    case_gains[key].append(outcome.bacc_gain)
                    by_family[outcome.action.direction.value].add(key)
    by_center: dict[str, list[float]] = defaultdict(list)
    for (center, _case), gains in case_gains.items():
        by_center[center].append(max((0.0, *gains)))
    identity = {
        "config_hash": config_hash,
        "model_hash": policy.model_hash,
        "policy_hash": policy.policy_hash,
        "crossfit_hash": crossfit["result_hash"],
        "all_outer_prediction_seal_hash": crossfit["all_outer_prediction_seal_hash"],
        "raw_labels_persisted": False,
        "evaluation_labels_opened": False,
        "diagnostic_only": True,
    }
    frontier = {
        **identity, "schema_version": "midogpp_harp_v18_candidate_frontier_v1",
        "row_count": len(rows), "rows": rows,
    }
    headroom = {
        **identity, "schema_version": "midogpp_harp_v18_source_headroom_v1",
        "estimand": "equal_centers_equal_classes_equal_supporting_cases",
        "primitive_case_count": len(case_gains),
        "primitive_positive_case_count": len(positive),
        "primitive_proper_loss_safe_positive_case_count": sum(bool(v) for v in case_gains.values()),
        "primitive_proper_loss_safe_positive_cases_by_family": {k: len(v) for k, v in sorted(by_family.items())},
        "primitive_safe_oracle_gain": float(np.mean([np.mean(v) for v in by_center.values()])),
        "actual_ranker_proposed_menu_oracles": oracles,
        "oracle_used_for_selection": False,
    }
    paths = (root / "reports/candidate_frontier.json", root / "reports/source_headroom_diagnostics.json")
    for path, body in zip(paths, (frontier, headroom), strict=True):
        atomic_json(path, {**body, "report_hash": canonical_hash(body)})
    durable_barrier(paths)
    return paths


def enforce_admitted_target_coverage(root: Path, *, routes: object, policy_admission: Mapping[str, object]) -> None:
    """Avoid spending a terminal release on an admitted policy with zero actions."""

    cases = tuple(routes.cases)
    routed = sum(case.selected_kind.value != "B" for case in cases)
    body = {
        "schema_version": "midogpp_harp_v18_label_free_target_coverage_v1",
        "target_case_count": len(cases), "nonbaseline_case_count": routed,
        "evaluation_labels_opened": False,
        "threshold_or_policy_changed": False,
    }
    path = root / "reports/label_free_target_coverage.json"
    atomic_json(path, {**body, "report_hash": canonical_hash(body)})
    durable_barrier((path,))
    admission = policy_admission.get("source_only_admission", {})
    if isinstance(admission, Mapping) and admission.get("admitted") is True and routed == 0:
        raise ProtocolError("HARP v18 admitted source policy produced zero target actions; evaluation truth remains closed.")
