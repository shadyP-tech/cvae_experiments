"""Exact endpoint and candidate-runtime lineage validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...runtime.artifact_io import read_json
from .constants import CENTERS, ENDPOINT_METHOD_IDS
from .hashing import canonical_hash, require_sha256
from .posterior_contracts import CONTROL_IDS
from .validation_candidate_semantics import (
    CandidateActionRecord,
    SOURCE_EXCLUSION_ROLE,
    validate_candidate_semantics,
)
from .validation_endpoint_evidence import EndpointEvidenceTopology
from .validation_plans import PlanPosteriorTopology
from .validation_shared import Row, fail, index_rows, string_list, support_identities


@dataclass(frozen=True)
class CandidateTopology:
    targets: Mapping[tuple[str, str, str], Row]
    pseudos: Mapping[tuple[str, str, str, str], Row]
    action_records: Mapping[str, CandidateActionRecord]
    runtime_actions: Mapping[
        tuple[str, str, str, str], tuple[CandidateActionRecord, ...]
    ]
    selected_action_by_runtime: Mapping[
        tuple[str, str, str, str], CandidateActionRecord | None
    ]


def validate_candidate_topology(
    root: Path,
    *,
    topology: PlanPosteriorTopology,
    posterior_probabilities: Mapping[tuple[str, str, str], np.ndarray],
    endpoint_topology: EndpointEvidenceTopology,
    target_rows: Sequence[Row],
    pseudo_rows: Sequence[Row],
) -> CandidateTopology:
    _validate_candidate_table_envelopes(root)
    expected_target_order = tuple(
        (center, case, control)
        for center in CENTERS
        for case in topology.cases_by_center[center]
        for control in CONTROL_IDS
    )
    expected_pseudo_order = tuple(
        (outer, pseudo, case, control)
        for outer in CENTERS
        for pseudo in CENTERS
        if pseudo != outer
        for case in topology.cases_by_center[pseudo]
        for control in CONTROL_IDS
    )
    if (
        tuple(
            (
                str(row.get("center")),
                str(row.get("case_id")),
                str(row.get("control_id")),
            )
            for row in target_rows
        )
        != expected_target_order
        or tuple(
            (
                str(row.get("outer_center")),
                str(row.get("center")),
                str(row.get("case_id")),
                str(row.get("control_id")),
            )
            for row in pseudo_rows
        )
        != expected_pseudo_order
    ):
        fail("candidate runtime canonical row order")
    targets = index_rows(
        target_rows,
        ("center", "case_id", "control_id"),
        "target candidates",
    )
    pseudos = index_rows(
        pseudo_rows,
        ("outer_center", "center", "case_id", "control_id"),
        "pseudo candidates",
    )
    expected_target = {
        (center, case, control)
        for center, case in topology.plans
        for control in {key[2] for key in topology.models}
    }
    if (
        set(targets) != expected_target
        or set(pseudos) != set(topology.pseudo_references)
    ):
        fail("candidate runtime rectangle")

    endpoint_hashes, endpoint_probabilities = _validate_endpoint_store(
        root, topology.plans
    )
    _require_probability_maps_equal(
        endpoint_probabilities,
        endpoint_topology.target_probabilities,
        role="target endpoint replay",
    )
    for (center, case, control), row in targets.items():
        support_event = topology.support[(center, case)]
        identities = support_identities(topology.plans, center, case)
        support_hash = canonical_hash(
            {
                "schema_version": "fixed_bank_cbpupr_support_reference_v1",
                "scope": support_event["role"],
                "identities": [list(value) for value in identities],
                "values_persisted": False,
            }
        )
        if (
            row.get("outer_center") != center
            or string_list(row, "source_excluded_centers") != (center,)
            or row.get("source_excluded_centers_role") != SOURCE_EXCLUSION_ROLE
            or row.get("posterior_model_hash")
            != topology.models[(center, case, control)].get("model_hash")
            or row.get("support_capability_hash") != support_hash
            or row.get("endpoint_lineage_hash") != endpoint_hashes[(center, case)]
            or row.get("posterior_model_reference_count") != 1
            or row.get("posterior_fit_increment") != 0
            or row.get("posterior_refit") is not False
            or row.get("posterior_refit_performed_in_candidate_runtime") is not False
            or row.get("sealed_posterior_reference_reused") is not False
        ):
            fail("target candidate lineage")
        _validate_candidate_summary(row)

    pseudo_endpoint_hashes: dict[tuple[str, str, str], str] = {}
    for (outer, pseudo, case, control), row in pseudos.items():
        key = (outer, pseudo, case)
        observed = str(row.get("endpoint_lineage_hash"))
        if key in pseudo_endpoint_hashes and pseudo_endpoint_hashes[key] != observed:
            fail("pseudo endpoint control lineage")
        pseudo_endpoint_hashes[key] = observed
        reference = topology.pseudo_references[(outer, pseudo, case, control)]
        if (
            outer == pseudo
            or string_list(row, "source_excluded_centers")
            != tuple(sorted((outer, pseudo)))
            or row.get("source_excluded_centers_role") != SOURCE_EXCLUSION_ROLE
            or row.get("posterior_model_hash")
            != topology.models[(pseudo, case, control)].get("model_hash")
            or row.get("support_capability_hash") != reference.get("reference_hash")
            or row.get("posterior_model_reference_count") != 1
            or row.get("posterior_fit_increment") != 0
            or row.get("posterior_refit") is not False
            or row.get("posterior_refit_performed_in_candidate_runtime") is not False
            or row.get("sealed_posterior_reference_reused") is not True
            or observed != endpoint_topology.pseudo_evidence_hashes[key]
        ):
            fail("pseudo candidate H/J lineage")
        require_sha256(observed, "pseudo endpoint lineage hash")
        _validate_candidate_summary(row)

    manifest = read_json(root / "manifests/candidate_probability_index.json")
    if manifest.get("runtime_rows") != [*target_rows, *pseudo_rows]:
        fail("candidate manifest/table lineage")
    array_rows = manifest.get("arrays")
    if not isinstance(array_rows, list):
        fail("candidate array index")
    expected_array_keys = {
        digest
        for row in (*target_rows, *pseudo_rows)
        for digest in string_list(row, "candidate_hashes", allow_empty=True)
    }
    if {str(row.get("key")) for row in array_rows} != expected_array_keys:
        fail("candidate dense-array key lineage")
    actions, runtime_actions, selected = validate_candidate_semantics(
        root,
        topology=topology,
        runtime_rows=(*target_rows, *pseudo_rows),
        posterior_probabilities=posterior_probabilities,
        endpoint_topology=endpoint_topology,
    )
    return CandidateTopology(
        targets,
        pseudos,
        actions,
        runtime_actions,
        selected,
    )


def _validate_candidate_table_envelopes(root: Path) -> None:
    expected = {
        "expected_utility_predictions": (
            "fixed_bank_cbpupr_expected_utility_predictions_v1"
        ),
        "candidate_eligibility": "fixed_bank_cbpupr_candidate_eligibility_v1",
        "target_candidate_policies": (
            "fixed_bank_cbpupr_target_candidate_policies_v1"
        ),
        "pseudo_candidate_policies": (
            "fixed_bank_cbpupr_pseudo_candidate_policies_v1"
        ),
    }
    for name, schema in expected.items():
        payload = read_json(root / "tables" / f"{name}.json")
        rows = payload.get("rows")
        if (
            payload.get("schema_version") != schema
            or not isinstance(rows, list)
            or payload.get("row_count") != len(rows)
            or any(not isinstance(row, Mapping) for row in rows)
        ):
            fail(f"{name} table envelope")


def _validate_endpoint_store(
    root: Path, plans: Mapping[tuple[str, str], Row]
) -> tuple[
    dict[tuple[str, str], str],
    dict[tuple[str, str, str], np.ndarray],
]:
    manifest = read_json(root / "manifests/route_endpoint_probability_index.json")
    endpoint_rows = manifest.get("index_rows")
    if not isinstance(endpoint_rows, list):
        fail("endpoint index rows")
    endpoint_index = index_rows(
        endpoint_rows,
        ("target_center", "case_id", "method_id"),
        "endpoint predictions",
    )
    expected = {
        (center, case, method)
        for center, case in plans
        for method in ENDPOINT_METHOD_IDS
    }
    if set(endpoint_index) != expected:
        fail("endpoint prediction rectangle")

    hashes: dict[tuple[str, str], str] = {}
    arrays: dict[tuple[str, str, str], np.ndarray] = {}
    with np.load(
        root / "arrays/route_endpoint_probabilities.npz", allow_pickle=False
    ) as store:
        for center, case in plans:
            route_rows = {
                method: endpoint_index[(center, case, method)]
                for method in ENDPOINT_METHOD_IDS
            }
            prediction_hashes = {
                str(row.get("prediction_hash")) for row in route_rows.values()
            }
            state_hashes = {str(row.get("state_hash")) for row in route_rows.values()}
            sample_orders = {
                string_list(row, "sample_ids") for row in route_rows.values()
            }
            if (
                len(prediction_hashes) != 1
                or len(state_hashes) != 1
                or len(sample_orders) != 1
            ):
                fail("endpoint method lineage")
            prediction_hash = prediction_hashes.pop()
            state_hash = state_hashes.pop()
            samples = sample_orders.pop()
            expected_samples = string_list(
                plans[(center, case)], "evaluation_sample_ids"
            )
            if samples != expected_samples:
                fail("endpoint sample lineage")
            probabilities: dict[str, list[float]] = {}
            for method, row in route_rows.items():
                expected_key = f"{prediction_hash}__{method}"
                values = np.asarray(store[str(row.get("array_key"))], dtype=np.float32)
                if row.get("array_key") != expected_key or values.shape != (len(samples),):
                    fail("endpoint dense-array shape/key")
                probabilities[method] = [float(value) for value in values]
                arrays[(center, case, method)] = np.ascontiguousarray(values).copy()
            payload = {
                "schema_version": "fixed_bank_cbpupr_endpoint_prediction_v1",
                "center": center,
                "case_id": case,
                "sample_ids": list(samples),
                "probabilities": probabilities,
                "state_hash": state_hash,
            }
            if prediction_hash != canonical_hash(payload):
                fail("endpoint prediction hash")
            hashes[(center, case)] = prediction_hash
    return hashes, arrays


def _validate_candidate_summary(row: Row) -> None:
    hashes = string_list(row, "candidate_hashes", allow_empty=True)
    selected = row.get("selected_candidate_hash")
    if (
        len(hashes) != len(set(hashes))
        or (selected is not None and str(selected) not in hashes)
        or row.get("descriptor_count") != 6
        or row.get("no_crossing_count", -1) + len(hashes) != 6
    ):
        fail("candidate summary")
    for field in (
        "runtime_hash",
        "posterior_model_hash",
        "support_capability_hash",
        "endpoint_lineage_hash",
    ):
        require_sha256(row.get(field), field)


def _require_probability_maps_equal(
    observed: Mapping[tuple[str, str, str], np.ndarray],
    expected: Mapping[tuple[str, str, str], np.ndarray],
    *,
    role: str,
) -> None:
    if set(observed) != set(expected):
        fail(f"{role} rectangle")
    for key, values in observed.items():
        left = np.ascontiguousarray(values, dtype=np.float32)
        right = np.ascontiguousarray(expected[key], dtype=np.float32)
        if left.shape != right.shape or left.tobytes(order="C") != right.tobytes(
            order="C"
        ):
            fail(f"{role} bytes")


__all__ = ("CandidateTopology", "validate_candidate_topology")
