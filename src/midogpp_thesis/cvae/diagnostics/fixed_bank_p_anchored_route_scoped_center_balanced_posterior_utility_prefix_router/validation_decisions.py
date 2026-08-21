"""Deterministic prefix semantics and route-decision lineage validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from .constants import (
    BLOCKED_CONTROL_METHOD_ID,
    BLOCKED_FINGERPRINT_CONTROL_ID,
    CENTERS,
    PRIMARY_FINGERPRINT_CONTROL_ID,
    PRIMARY_METHOD_ID,
    UTILITY_ZERO_TOLERANCE,
)
from .hashing import canonical_hash
from .posterior_expected_utility import FavorableUtility
from .validation_candidates import CandidateTopology
from .validation_plans import PlanPosteriorTopology
from .validation_shared import (
    Row,
    fail,
    index_rows,
    mapping_field,
    string_list,
    table_rows,
)


def validate_route_decisions(
    root: Path,
    *,
    plan_topology: PlanPosteriorTopology,
    candidate_topology: CandidateTopology,
    rows: Sequence[Row],
    calibrations: Mapping[tuple[str, str], Mapping[str, object]],
) -> dict[tuple[str, str], Row]:
    indexed = index_rows(rows, ("center", "method_id"), "route decisions")
    expected = {
        (center, method)
        for center in CENTERS
        for method in (PRIMARY_METHOD_ID, BLOCKED_CONTROL_METHOD_ID)
    }
    if set(indexed) != expected:
        fail("route decision rectangle")

    prefix_rows = index_rows(
        table_rows(root, "prefix_decisions"),
        ("center", "method_id"),
        "prefix decision table",
    )
    if set(prefix_rows) != expected:
        fail("prefix decision rectangle")

    for (center, method), row in indexed.items():
        control = (
            PRIMARY_FINGERPRINT_CONTROL_ID
            if method == PRIMARY_METHOD_ID
            else BLOCKED_FINGERPRINT_CONTROL_ID
        )
        runtimes = tuple(
            candidate_topology.targets[(center, case, control)]
            for case in plan_topology.cases_by_center[center]
        )
        expected_runtime_hashes = tuple(
            sorted(str(runtime["runtime_hash"]) for runtime in runtimes)
        )
        if string_list(row, "candidate_runtime_hashes") != expected_runtime_hashes:
            fail("decision candidate-runtime lineage")

        selection = mapping_field(row, "prefix_selection")
        expected_prefix_row = {
            "center": center,
            "method_id": method,
            **dict(selection),
        }
        if dict(prefix_rows[(center, method)]) != expected_prefix_row:
            fail("prefix decision cross-table lineage")
        composition = mapping_field(row, "composition")
        selected_k = int(selection["selected_k"])

        structural = mapping_field(row, "structural_transport")
        structural_payload = {
            "schema_version": "cbpupr_structural_transport_gate_v1",
            "target_center": center,
            "probability_lineage_match": structural.get(
                "probability_lineage_match"
            ),
            "plan_lineage_match": structural.get("plan_lineage_match"),
            "target_excluded_from_fit": structural.get("target_excluded_from_fit"),
            "own_route_noninterference": structural.get(
                "own_route_noninterference"
            ),
            "finite_inputs": structural.get("finite_inputs"),
            "reason_codes": structural.get("reason_codes"),
        }
        structural_checks = (
            ("PROBABILITY_LINEAGE_MISMATCH", structural.get("probability_lineage_match")),
            ("PLAN_LINEAGE_MISMATCH", structural.get("plan_lineage_match")),
            ("TARGET_NOT_EXCLUDED", structural.get("target_excluded_from_fit")),
            ("OWN_ROUTE_INTERFERENCE", structural.get("own_route_noninterference")),
            ("NONFINITE_INPUT", structural.get("finite_inputs")),
        )
        if any(type(value) is not bool for _reason, value in structural_checks):
            fail("structural transport booleans")
        failures = [reason for reason, value in structural_checks if not value]
        expected_structural_reasons = (
            ["STRUCTURAL_TRANSPORT_PASS"] if not failures else failures
        )
        passed = not failures
        calibration = calibrations.get((center, control))
        if not isinstance(calibration, Mapping):
            fail("route decision calibration key")
        calibration_supported = calibration.get("supported") is True
        if (
            type(calibration.get("supported")) is not bool
            or type(calibration.get("forces_exact_P")) is not bool
            or calibration.get("forces_exact_P") != (not calibration_supported)
            or (
                calibration_supported
                and not isinstance(calibration.get("bias"), Mapping)
            )
            or (
                calibration_supported
                and calibration.get("reason_code") is not None
            )
            or (
                not calibration_supported
                and (
                    calibration.get("bias") is not None
                    or not isinstance(calibration.get("reason_code"), str)
                    or not calibration.get("reason_code")
                )
            )
            or row.get("utility_calibration_hash")
            != calibration.get("calibration_hash")
        ):
            fail("route decision calibration lineage")
        _validate_prefix_selection(
            center=center,
            control=control,
            runtimes=runtimes,
            selection=selection,
            composition=composition,
            candidate_topology=candidate_topology,
            calibration=calibration,
            structural_passed=passed,
        )
        routed = calibration_supported and passed and selected_k > 0
        if not calibration_supported:
            expected_decision_reasons = (str(calibration.get("reason_code")),)
        elif not passed:
            expected_decision_reasons = tuple(expected_structural_reasons)
        elif selected_k > 0:
            expected_decision_reasons = (
                "STRUCTURAL_AND_AGGREGATE_UTILITY_PASS",
            )
        else:
            expected_decision_reasons = ("NO_FEASIBLE_AGGREGATE_PREFIX",)
        if (
            structural.get("target_center") != center
            or structural.get("reason_codes") != expected_structural_reasons
            or structural.get("passed") != passed
            or structural.get("gate_hash") != canonical_hash(structural_payload)
            or row.get("policy_replay_bias_used") is not False
            or string_list(row, "reason_codes") != expected_decision_reasons
            or (row.get("action") == "ROUTE_SELECTED_PREFIX") != routed
            or (row.get("action") == "ABSTAIN_TO_EXACT_P") != (not routed)
            or (not passed and selected_k != 0)
            or (
                not passed
                and string_list(row, "reason_codes")
                != tuple(expected_structural_reasons)
            )
            or (not calibration_supported and selected_k != 0)
        ):
            fail("structural transport/route action lineage")

        decision_payload = {
            "schema_version": "cbpupr_route_decision_v1",
            "center": center,
            "method_id": method,
            "action": row.get("action"),
            "reason_codes": list(string_list(row, "reason_codes")),
            "prefix_selection_hash": selection.get("selection_hash"),
            "composition_hash": composition.get("composition_hash"),
            "structural_transport_hash": structural.get("gate_hash"),
            "utility_calibration_hash": row.get("utility_calibration_hash"),
            "candidate_runtime_hashes": list(expected_runtime_hashes),
            "policy_replay_bias_used": False,
        }
        if row.get("decision_hash") != canonical_hash(decision_payload):
            fail("route decision hash")
    return indexed


def _validate_prefix_selection(
    *,
    center: str,
    control: str,
    runtimes: Sequence[Row],
    selection: Row,
    composition: Row,
    candidate_topology: CandidateTopology,
    calibration: Mapping[str, object],
    structural_passed: bool,
) -> None:
    ranked = selection.get("ranked_candidates")
    evaluations = selection.get("evaluations")
    if not isinstance(ranked, list) or not isinstance(evaluations, list):
        fail("prefix selection rows")
    expected_ranked = _expected_ranked_candidates(
        center=center,
        control=control,
        runtimes=runtimes,
        candidate_topology=candidate_topology,
        calibration=calibration,
        structural_passed=structural_passed,
    )
    if ranked != expected_ranked:
        fail("prefix one-bias candidate reconstruction")
    selected_runtime_hashes = {
        str(runtime["selected_candidate_hash"])
        for runtime in runtimes
        if runtime.get("selected_candidate_hash") is not None
    }
    ranked_hashes: list[str] = []
    ranked_cases: list[str] = []
    policy_hashes: list[str] = []
    for candidate in ranked:
        if not isinstance(candidate, Mapping):
            fail("ranked prefix candidate")
        action_hash = str(candidate.get("action_hash"))
        case_id = str(candidate.get("case_id"))
        payload = {
            "schema_version": "cbpupr_prefix_candidate_v1",
            "center": center,
            "case_id": case_id,
            "action_hash": action_hash,
            "corrected_utility": candidate.get("corrected_utility"),
            "calibration_hash": candidate.get("calibration_hash"),
        }
        if (
            candidate.get("center") != center
            or candidate.get("control_id") != control
            or action_hash not in selected_runtime_hashes
            or candidate.get("policy_hash") != canonical_hash(payload)
        ):
            fail("ranked prefix candidate lineage")
        ranked_hashes.append(action_hash)
        ranked_cases.append(case_id)
        policy_hashes.append(str(candidate["policy_hash"]))
    if len(ranked_cases) != len(set(ranked_cases)):
        fail("prefix case uniqueness")
    expected_rank = sorted(
        ranked,
        key=lambda candidate: (
            -float(mapping_field(candidate, "corrected_utility")["bacc_gain"]),
            str(candidate["case_id"]),
            str(candidate["policy_hash"]),
        ),
    )
    if list(ranked) != expected_rank:
        fail("prefix deterministic ranking")

    evaluation_hashes: list[str] = []
    cumulative = {"bacc_gain": 0.0, "brier_gain": 0.0, "log_gain": 0.0}
    for index, evaluation in enumerate(evaluations):
        if not isinstance(evaluation, Mapping):
            fail("prefix evaluation")
        candidate_hashes = string_list(
            evaluation, "candidate_hashes", allow_empty=True
        )
        if index:
            corrected = mapping_field(ranked[index - 1], "corrected_utility")
            cumulative = {
                key: cumulative[key] + float(corrected[key]) for key in cumulative
            }
            reasons: list[str] = []
            if cumulative["bacc_gain"] <= UTILITY_ZERO_TOLERANCE:
                reasons.append("NONPOSITIVE_AGGREGATE_BACC")
            if cumulative["brier_gain"] < -UTILITY_ZERO_TOLERANCE:
                reasons.append("NEGATIVE_AGGREGATE_BRIER_GAIN")
            if cumulative["log_gain"] < -UTILITY_ZERO_TOLERANCE:
                reasons.append("NEGATIVE_AGGREGATE_LOG_GAIN")
            expected_reasons = (
                ("AGGREGATE_UTILITY_PASS",) if not reasons else tuple(reasons)
            )
            expected_feasible = not reasons
        else:
            expected_reasons = ("EXACT_P_BASELINE",)
            expected_feasible = True
        payload = {
            "schema_version": "cbpupr_prefix_evaluation_v1",
            "k": index,
            "candidate_hashes": list(candidate_hashes),
            "aggregate_utility": dict(cumulative),
            "feasible": expected_feasible,
            "reason_codes": list(expected_reasons),
        }
        if (
            evaluation.get("k") != index
            or candidate_hashes != tuple(ranked_hashes[:index])
            or mapping_field(evaluation, "aggregate_utility") != cumulative
            or evaluation.get("feasible") is not expected_feasible
            or string_list(evaluation, "reason_codes") != expected_reasons
            or evaluation.get("prefix_hash") != canonical_hash(payload)
        ):
            fail("prefix evaluation semantics")
        evaluation_hashes.append(str(evaluation["prefix_hash"]))

    selected_k = selection.get("selected_k")
    if not isinstance(selected_k, int) or not 0 <= selected_k < len(evaluations):
        fail("prefix selected k")
    feasible = [
        evaluation
        for evaluation in evaluations
        if isinstance(evaluation, Mapping) and evaluation.get("feasible") is True
    ]
    maximum = max(
        float(mapping_field(evaluation, "aggregate_utility")["bacc_gain"])
        for evaluation in feasible
    )
    tied = [
        evaluation
        for evaluation in feasible
        if abs(
            float(mapping_field(evaluation, "aggregate_utility")["bacc_gain"])
            - maximum
        )
        <= UTILITY_ZERO_TOLERANCE
    ]
    expected_selected = min(
        tied,
        key=lambda evaluation: (
            int(evaluation["k"]),
            str(evaluation["prefix_hash"]),
        ),
    )
    selection_payload = {
        "schema_version": "cbpupr_prefix_selection_v1",
        "ranked_policy_hashes": policy_hashes,
        "evaluation_hashes": evaluation_hashes,
        "selected_k": selected_k,
        "selected_prefix_hash": selection.get("selected_prefix_hash"),
    }
    selected_cases = tuple(ranked_cases[:selected_k])
    selected_hashes = tuple(ranked_hashes[:selected_k])
    if (
        selected_k != expected_selected.get("k")
        or selection.get("selected_prefix_hash") != evaluation_hashes[selected_k]
        or selection.get("selection_hash") != canonical_hash(selection_payload)
        or string_list(selection, "selected_case_ids", allow_empty=True)
        != selected_cases
        or string_list(selection, "selected_candidate_hashes", allow_empty=True)
        != selected_hashes
        or selection.get("dense_probabilities_persisted") is not False
        or string_list(composition, "selected_case_ids", allow_empty=True)
        != selected_cases
        or string_list(composition, "selected_candidate_hashes", allow_empty=True)
        != selected_hashes
    ):
        fail("prefix selection/composition topology")
    composition_payload = {
        "schema_version": "cbpupr_composition_v1",
        "selected_case_ids": list(selected_cases),
        "selected_candidate_hashes": list(selected_hashes),
        "probability_sha256": composition.get("probability_sha256"),
        "changed_probability_count": composition.get("changed_probability_count"),
        "exact_p": composition.get("exact_p"),
    }
    if (
        composition.get("composition_hash") != canonical_hash(composition_payload)
        or composition.get("exact_p") != (selected_k == 0)
        or composition.get("dense_probabilities_persisted") is not False
    ):
        fail("composition hash/topology")


def _expected_ranked_candidates(
    *,
    center: str,
    control: str,
    runtimes: Sequence[Row],
    candidate_topology: CandidateTopology,
    calibration: Mapping[str, object],
    structural_passed: bool,
) -> list[dict[str, object]]:
    if calibration.get("supported") is not True or not structural_passed:
        return []
    bias_payload = calibration.get("bias")
    if not isinstance(bias_payload, Mapping):
        fail("supported calibration bias")
    bias = FavorableUtility.from_payload(bias_payload)
    calibration_hash = str(calibration.get("calibration_hash"))
    ranked: list[dict[str, object]] = []
    for runtime in runtimes:
        case = str(runtime.get("case_id"))
        record = candidate_topology.selected_action_by_runtime.get(
            (center, center, case, control)
        )
        if (
            runtime.get("selected_candidate_hash")
            != (None if record is None else record.action_hash)
        ):
            fail("prefix selected runtime action")
        if record is None:
            continue
        corrected = record.action.estimate.utility - bias
        payload = {
            "schema_version": "cbpupr_prefix_candidate_v1",
            "center": center,
            "case_id": case,
            "action_hash": record.action_hash,
            "corrected_utility": corrected.to_payload(),
            "calibration_hash": calibration_hash,
        }
        ranked.append(
            {
                "center": center,
                "case_id": case,
                "control_id": control,
                "action_hash": record.action_hash,
                "policy_hash": canonical_hash(payload),
                "corrected_utility": corrected.to_payload(),
                "calibration_hash": calibration_hash,
            }
        )
    ranked.sort(
        key=lambda candidate: (
            -float(mapping_field(candidate, "corrected_utility")["bacc_gain"]),
            str(candidate["case_id"]),
            str(candidate["policy_hash"]),
        )
    )
    return ranked


__all__ = ("validate_route_decisions",)
