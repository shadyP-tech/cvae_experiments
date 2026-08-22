"""Cross-store dense probability and aggregate-composition validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from ...runtime.artifact_io import read_json
from .canonical_probabilities import probability_sha256, require_byte_exact_p
from .constants import (
    BLOCKED_CONTROL_METHOD_ID,
    CANDIDATE_ONLY_METHOD_ID,
    CENTERS,
    COMPOSED_POLICY_IDS,
    ENDPOINT_METHOD_IDS,
    OBSERVED_MAX_CONTROL_METHOD_ID,
    PORTFOLIO_METHOD_ID,
    PRIMARY_FINGERPRINT_CONTROL_ID,
    PRIMARY_METHOD_ID,
)
from .hashing import canonical_hash
from .validation_candidates import CandidateTopology
from .validation_controls import reconstruct_control_policies
from .validation_plans import PlanPosteriorTopology
from .validation_shared import Row, fail, index_rows, mapping_field, string_list


def validate_composed_probability_stores(
    root: Path,
    *,
    plan_topology: PlanPosteriorTopology,
    candidate_topology: CandidateTopology,
    decisions: Mapping[tuple[str, str], Row],
    rows: Sequence[Row],
    donor_case_replay_rows: Sequence[Row],
) -> None:
    indexed = index_rows(rows, ("method_id", "target_center"), "composed predictions")
    methods = (*ENDPOINT_METHOD_IDS, *COMPOSED_POLICY_IDS)
    expected = {(method, center) for method in methods for center in CENTERS}
    if set(indexed) != expected:
        fail("composed probability rectangle")
    control_policies = reconstruct_control_policies(
        plan_topology=plan_topology,
        candidate_topology=candidate_topology,
        decisions=decisions,
        donor_case_replay_rows=donor_case_replay_rows,
    )

    manifest = read_json(root / "manifests/composed_probability_index.json")
    arrays = manifest.get("arrays")
    if not isinstance(arrays, list) or {
        str(row.get("key")) for row in arrays
    } != {f"{method}__{center}" for method, center in expected}:
        fail("composed array key rectangle")

    endpoint_manifest = read_json(
        root / "manifests/route_endpoint_probability_index.json"
    )
    endpoint_rows = endpoint_manifest.get("index_rows")
    if not isinstance(endpoint_rows, list):
        fail("endpoint index for composed validation")
    endpoint_index = index_rows(
        endpoint_rows,
        ("target_center", "case_id", "method_id"),
        "composed endpoint source",
    )

    aggregate_probabilities: dict[str, dict[str, dict[str, str]]] = {
        method: {center: {} for center in CENTERS}
        for method in (PORTFOLIO_METHOD_ID, *COMPOSED_POLICY_IDS)
    }
    decision_seals: list[list[object]] = []
    with (
        np.load(root / "arrays/composed_probabilities.npz", allow_pickle=False) as store,
        np.load(
            root / "arrays/route_endpoint_probabilities.npz", allow_pickle=False
        ) as endpoint_store,
        np.load(
            root / "arrays/candidate_probabilities.npz", allow_pickle=False
        ) as candidate_store,
    ):
        for center in CENTERS:
            p = np.asarray(
                store[f"{PORTFOLIO_METHOD_ID}__{center}"], dtype=np.float32
            )
            case_order: tuple[str, ...] | None = None
            for method in methods:
                row = indexed[(method, center)]
                observed_cases = string_list(row, "case_ids")
                if case_order is None:
                    case_order = observed_cases
                selected = string_list(row, "selected_case_ids", allow_empty=True)
                if (
                    observed_cases != case_order
                    or observed_cases != plan_topology.cases_by_center[center]
                    or row.get("case_count")
                    != len(plan_topology.cases_by_center[center])
                    or row.get("case_identity_hash")
                    != canonical_hash(list(observed_cases))
                    or row.get("array_key") != f"{method}__{center}"
                    or len(selected) != len(set(selected))
                    or not set(selected)
                    <= set(plan_topology.cases_by_center[center])
                ):
                    fail("composed case topology")

                selected_hashes = _validate_policy_row(
                    row=row,
                    method=method,
                    center=center,
                    selected=selected,
                    decisions=decisions,
                    candidate_topology=candidate_topology,
                    cases=plan_topology.cases_by_center[center],
                    control_policies=control_policies,
                )
                if len(selected_hashes) != len(selected):
                    fail("composed selected candidate/case arity")
                selected_by_case = dict(
                    zip(selected, selected_hashes, strict=True)
                )
                actual = np.asarray(store[f"{method}__{center}"], dtype=np.float32)
                if actual.shape != p.shape:
                    fail("composed center array shape")
                if method in ENDPOINT_METHOD_IDS:
                    expected_endpoint = np.concatenate(
                        [
                            np.asarray(
                                endpoint_store[
                                    str(
                                        endpoint_index[(center, case, method)][
                                            "array_key"
                                        ]
                                    )
                                ],
                                dtype=np.float32,
                            )
                            for case in observed_cases
                        ]
                    )
                    if actual.tobytes(order="C") != expected_endpoint.tobytes(
                        order="C"
                    ):
                        fail("composed endpoint concatenation")

                selected_set = set(selected)
                offset = 0
                for case in observed_cases:
                    length = len(
                        string_list(
                            plan_topology.plans[(center, case)],
                            "evaluation_sample_ids",
                        )
                    )
                    case_values = actual[offset : offset + length]
                    if method in COMPOSED_POLICY_IDS and case not in selected_set:
                        require_byte_exact_p(
                            case_values, p[offset : offset + length]
                        )
                    elif method in COMPOSED_POLICY_IDS:
                        candidate_values = np.asarray(
                            candidate_store[selected_by_case[case]],
                            dtype=np.float32,
                        )
                        if (
                            candidate_values.shape != case_values.shape
                            or candidate_values.tobytes(order="C")
                            != case_values.tobytes(order="C")
                        ):
                            fail("composed selected candidate bytes")
                    if method in aggregate_probabilities:
                        aggregate_probabilities[method][center][case] = canonical_hash(
                            list(case_values)
                        )
                    if method in COMPOSED_POLICY_IDS:
                        decision_seals.append(
                            [
                                method,
                                center,
                                case,
                                canonical_hash(
                                    [
                                        row.get("policy_hash"),
                                        center,
                                        case,
                                        case in selected_set,
                                    ]
                                ),
                            ]
                        )
                    offset += length
                if offset != len(actual):
                    fail("composed case slice coverage")
                if method in {PRIMARY_METHOD_ID, BLOCKED_CONTROL_METHOD_ID}:
                    composition = mapping_field(
                        decisions[(center, method)], "composition"
                    )
                    changed = int(
                        np.count_nonzero(
                            actual.view(np.uint32) != p.view(np.uint32)
                        )
                    )
                    if (
                        composition.get("probability_sha256")
                        != probability_sha256(actual)
                        or composition.get("changed_probability_count") != changed
                    ):
                        fail("decision/composed probability hash")

    decision_barrier = read_json(root / "manifests/decision_barrier.json")
    aggregate = read_json(root / "manifests/preterminal_aggregate_seal.json")
    expected_aggregate_seal = canonical_hash(
        [
            decision_barrier.get("replay_calibration_seal_hash"),
            sorted(decision_seals),
            canonical_hash(aggregate_probabilities),
        ]
    )
    if aggregate.get("aggregate_seal_hash") != expected_aggregate_seal:
        fail("aggregate composed-probability seal")


def _validate_policy_row(
    *,
    row: Row,
    method: str,
    center: str,
    selected: tuple[str, ...],
    decisions: Mapping[tuple[str, str], Row],
    candidate_topology: CandidateTopology,
    cases: tuple[str, ...],
    control_policies: Mapping[tuple[str, str], object],
) -> tuple[str, ...]:
    if method in ENDPOINT_METHOD_IDS:
        if (
            selected
            or row.get("policy_hash") is not None
            or row.get("control_policy") is not None
        ):
            fail("endpoint policy lineage")
        return ()
    if method in {PRIMARY_METHOD_ID, BLOCKED_CONTROL_METHOD_ID}:
        expected_selected = string_list(
            mapping_field(decisions[(center, method)], "composition"),
            "selected_case_ids",
            allow_empty=True,
        )
        expected_hashes = string_list(
            mapping_field(decisions[(center, method)], "composition"),
            "selected_candidate_hashes",
            allow_empty=True,
        )
        if (
            selected != expected_selected
            or row.get("policy_hash")
            != decisions[(center, method)].get("decision_hash")
            or row.get("control_policy") is not None
        ):
            fail("route decision/composed policy lineage")
        return expected_hashes
    if method not in {
        CANDIDATE_ONLY_METHOD_ID,
        OBSERVED_MAX_CONTROL_METHOD_ID,
    }:
        fail("unknown composed policy")

    policy = row.get("control_policy")
    expected_policy = control_policies.get((method, center))
    if expected_policy is None or not isinstance(policy, Mapping):
        fail("control selection candidate lineage")
    expected_payload = expected_policy.to_payload()  # type: ignore[union-attr]
    if dict(policy) != expected_payload:
        fail("control exact scientific reconstruction")
    selected_hashes = string_list(
        policy, "selected_candidate_hashes", allow_empty=True
    )
    case_by_hash = {
        str(
            candidate_topology.targets[
                (center, case, PRIMARY_FINGERPRINT_CONTROL_ID)
            ]["selected_candidate_hash"]
        ): case
        for case in cases
        if candidate_topology.targets[
            (center, case, PRIMARY_FINGERPRINT_CONTROL_ID)
        ].get("selected_candidate_hash")
        is not None
    }
    if (
        policy.get("method_id") != method
        or tuple(case_by_hash.get(value, "") for value in selected_hashes)
        != selected
        or policy.get("authorized") != bool(selected_hashes)
    ):
        fail("control selected-case lineage")
    if (
        row.get("policy_hash") != policy.get("policy_hash")
    ):
        fail("control policy hash lineage")
    return selected_hashes


__all__ = ("validate_composed_probability_stores",)
