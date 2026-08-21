"""Structural transport-gate replay from persisted independent lineages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path

import numpy as np

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from .canonical_probabilities import CanonicalProbabilityVector
from .constants import ALTERNATIVE_METHOD_IDS, CENTERS, ENDPOINT_METHOD_IDS
from .hashing import canonical_hash, require_sha256
from .posterior_contracts import CONTROL_IDS
from .posterior_expected_utility import PosteriorUtilityEstimate
from .transport_geometry import StructuralTransportGate
from .validation_candidates import CandidateTopology
from .validation_plans import PlanPosteriorTopology
from .validation_shared import Row, fail, index_rows, string_list, table_rows


_STRUCTURAL_FIELDS = frozenset(
    "target_center probability_lineage_match plan_lineage_match "
    "target_excluded_from_fit own_route_noninterference finite_inputs "
    "reason_codes passed gate_hash".split()
)


def validate_structural_transport_rows(
    root: Path,
    *,
    rows: Sequence[Row],
    topology: PlanPosteriorTopology,
    candidate_topology: CandidateTopology,
) -> dict[str, StructuralTransportGate]:
    """Recompute all five structural booleans, reasons, status, and hashes."""

    observed = tuple(rows)
    if (
        len(observed) != len(CENTERS)
        or tuple(str(row.get("target_center", "")) for row in observed) != CENTERS
    ):
        fail("structural transport row rectangle")
    expected = _reconstruct_gates(
        topology=topology,
        candidate_topology=candidate_topology,
        endpoint_hashes=_endpoint_hashes(root, topology),
        finite_by_center=_target_input_finiteness(root, candidate_topology),
    )
    result: dict[str, StructuralTransportGate] = {}
    for center, persisted in zip(CENTERS, observed, strict=True):
        parsed = _parse_row(persisted, center=center)
        if parsed.to_payload() != expected[center].to_payload():
            fail(f"structural transport lineage for center {center}")
        result[center] = parsed
    return result


def _parse_row(row: Row, *, center: str) -> StructuralTransportGate:
    if set(row) != _STRUCTURAL_FIELDS or row.get("target_center") != center:
        fail("structural transport row schema")
    for key in (
        "probability_lineage_match",
        "plan_lineage_match",
        "target_excluded_from_fit",
        "own_route_noninterference",
        "finite_inputs",
        "passed",
    ):
        if type(row.get(key)) is not bool:
            fail(f"structural transport {key}")
    reasons = row.get("reason_codes")
    if (
        not isinstance(reasons, list)
        or not reasons
        or any(not isinstance(value, str) or not value for value in reasons)
    ):
        fail("structural transport reason codes")
    require_sha256(row.get("gate_hash"), "persisted structural transport hash")
    parsed = StructuralTransportGate(
        center,
        row["probability_lineage_match"],  # type: ignore[arg-type]
        row["plan_lineage_match"],  # type: ignore[arg-type]
        row["target_excluded_from_fit"],  # type: ignore[arg-type]
        row["own_route_noninterference"],  # type: ignore[arg-type]
        row["finite_inputs"],  # type: ignore[arg-type]
    )
    if dict(row) != parsed.to_payload():
        fail("structural transport hash/status/reasons")
    return parsed


def _reconstruct_gates(
    *,
    topology: PlanPosteriorTopology,
    candidate_topology: CandidateTopology,
    endpoint_hashes: Mapping[tuple[str, str], str],
    finite_by_center: Mapping[str, bool],
) -> dict[str, StructuralTransportGate]:
    plans = topology.plans
    models = topology.models
    targets = candidate_topology.targets
    result: dict[str, StructuralTransportGate] = {}
    for center in CENTERS:
        center_rows = tuple(
            row
            for (observed, _case, _control), row in targets.items()
            if observed == center
        )
        expected_routes = {
            (case, control)
            for observed, case in plans
            if observed == center
            for control in CONTROL_IDS
        }
        observed_routes = {
            (str(row.get("case_id", "")), str(row.get("control_id", "")))
            for row in center_rows
        }
        exact_topology = (
            len(center_rows) == len(expected_routes)
            and observed_routes == expected_routes
        )
        model_topology = {
            (case, control)
            for observed, case, control in models
            if observed == center
        } == expected_routes
        probability_lineage = all(
            row.get("endpoint_lineage_hash")
            == endpoint_hashes.get((center, str(row.get("case_id", ""))))
            for row in center_rows
        )
        plan_lineage = exact_topology and all(
            (center, str(row.get("case_id", ""))) in plans for row in center_rows
        )
        target_excluded = model_topology and all(
            string_list(models[_target_key(center, row)], "training_case_ids")
            == string_list(plans[_target_key(center, row)[:2]], "support_case_ids")
            for row in center_rows
        )
        noninterference = exact_topology and model_topology and all(
            row.get("outer_center") == center
            and set(string_list(row, "source_excluded_centers")) == {center}
            and models[_target_key(center, row)].get("held_case_id")
            == row.get("case_id")
            and row.get("posterior_model_hash")
            == models[_target_key(center, row)].get("model_hash")
            for row in center_rows
        )
        result[center] = StructuralTransportGate(
            center,
            probability_lineage,
            plan_lineage,
            target_excluded,
            noninterference,
            finite_by_center[center],
        )
    return result


def _target_key(center: str, row: Row) -> tuple[str, str, str]:
    return center, str(row.get("case_id", "")), str(row.get("control_id", ""))


def _endpoint_hashes(
    root: Path, topology: PlanPosteriorTopology
) -> dict[tuple[str, str], str]:
    manifest = read_json(root / "manifests/route_endpoint_probability_index.json")
    raw_rows = manifest.get("index_rows")
    if not isinstance(raw_rows, list):
        fail("transport endpoint lineage rows")
    indexed = index_rows(
        raw_rows,
        ("target_center", "case_id", "method_id"),
        "transport endpoint lineage",
    )
    expected = {
        (center, case, method)
        for center, case in topology.plans
        for method in ENDPOINT_METHOD_IDS
    }
    if set(indexed) != expected:
        fail("transport endpoint lineage rectangle")
    result: dict[tuple[str, str], str] = {}
    for center, case in topology.plans:
        hashes = {
            require_sha256(
                indexed[(center, case, method)].get("prediction_hash"),
                "transport endpoint prediction hash",
            )
            for method in ENDPOINT_METHOD_IDS
        }
        if len(hashes) != 1:
            fail("transport endpoint method lineage")
        result[(center, case)] = hashes.pop()
    return result


def _target_input_finiteness(
    root: Path, candidate_topology: CandidateTopology
) -> dict[str, bool]:
    estimates_by_runtime: dict[
        tuple[str, str, str], list[PosteriorUtilityEstimate]
    ] = {}
    for raw in table_rows(root, "expected_utility_predictions"):
        try:
            estimate = PosteriorUtilityEstimate.from_payload(raw)
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError(
                "CBPUPR persisted transport utility input drifted."
            ) from exc
        if dict(raw) != estimate.to_payload():
            fail("transport utility payload/hash")
        key = estimate.center, estimate.case_id, estimate.control_id
        estimates_by_runtime.setdefault(key, []).append(estimate)

    result = {center: True for center in CENTERS}
    try:
        with np.load(
            root / "arrays/candidate_probabilities.npz", allow_pickle=False
        ) as store:
            available = set(store.files)
            for key, runtime in candidate_topology.targets.items():
                center, case, control = key
                for action_hash in string_list(
                    runtime, "candidate_hashes", allow_empty=True
                ):
                    require_sha256(action_hash, "transport target action hash")
                    if action_hash not in available:
                        fail("transport target candidate array lineage")
                    probability = CanonicalProbabilityVector.from_array(
                        np.asarray(store[action_hash], dtype=np.float32)
                    )
                    estimates = estimates_by_runtime.get((center, case, control), [])
                    matching = {
                        estimate.estimate_hash
                        for estimate in estimates
                        if _action_hash(probability, estimate) == action_hash
                    }
                    if len(matching) != 1:
                        fail("transport target candidate utility lineage")
                    result[center] = result[center] and bool(
                        np.isfinite(probability.as_array()).all()
                        and all(
                            math.isfinite(value)
                            for estimate in estimates
                            if estimate.estimate_hash in matching
                            for value in estimate.utility.as_tuple()
                        )
                    )
    except (OSError, ValueError) as exc:
        raise ProtocolError(
            "CBPUPR persisted transport candidate array input drifted."
        ) from exc
    return result


def _action_hash(
    probability: CanonicalProbabilityVector,
    estimate: PosteriorUtilityEstimate,
) -> str:
    suffix = f"::{estimate.direction}"
    alternative = (
        estimate.action_id[: -len(suffix)]
        if estimate.action_id.endswith(suffix)
        else ""
    )
    if alternative not in ALTERNATIVE_METHOD_IDS:
        fail("transport action identity")
    return canonical_hash(
        {
            "schema_version": "cbpupr_action_candidate_v1",
            "center": estimate.center,
            "case_id": estimate.case_id,
            "alternative_id": alternative,
            "direction": estimate.direction,
            "control_id": estimate.control_id,
            "probability_sha256": probability.sha256,
            "estimate_hash": estimate.estimate_hash,
        }
    )


__all__ = ("validate_structural_transport_rows",)
