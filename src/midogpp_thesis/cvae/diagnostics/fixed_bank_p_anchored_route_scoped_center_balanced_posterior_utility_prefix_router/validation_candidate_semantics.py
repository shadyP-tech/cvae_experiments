"""Reconstruct persisted candidate actions, utilities, eligibility, and hashes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...runtime.artifact_io import read_json, sha256_array
from .candidate_runtime import build_case_candidates
from .canonical_probabilities import CanonicalProbabilityVector, probability_sha256
from .constants import ALTERNATIVE_METHOD_IDS, PORTFOLIO_METHOD_ID
from .eligibility import (
    ActionCandidate,
    EligibilityDecision,
    assess_action,
    select_best_eligible_action,
)
from .hashing import canonical_hash
from .posterior_expected_utility import PosteriorUtilityEstimate
from .validation_endpoint_evidence import EndpointEvidenceTopology
from .validation_plans import PlanPosteriorTopology
from .validation_shared import Row, fail, index_rows, string_list, table_rows


SOURCE_EXCLUSION_ROLE = (
    "actionable_endpoint_source_selection_only_not_posterior_"
    "fingerprint_covariates"
)


@dataclass(frozen=True)
class CandidateActionRecord:
    action: ActionCandidate
    eligibility: EligibilityDecision
    array_key: str
    array_sha256: str

    @property
    def action_hash(self) -> str:
        return self.action.action_hash

    @property
    def probabilities(self) -> np.ndarray:
        return self.action.probabilities.as_array()


def validate_candidate_semantics(
    root: Path,
    *,
    topology: PlanPosteriorTopology,
    runtime_rows: Sequence[Row],
    posterior_probabilities: Mapping[tuple[str, str, str], np.ndarray],
    endpoint_topology: EndpointEvidenceTopology,
) -> tuple[
    dict[str, CandidateActionRecord],
    dict[tuple[str, str, str, str], tuple[CandidateActionRecord, ...]],
    dict[tuple[str, str, str, str], CandidateActionRecord | None],
]:
    """Rebuild every persisted action without reopening any label capability."""

    estimate_payloads = table_rows(root, "expected_utility_predictions")
    estimates_by_route: dict[
        tuple[str, str, str], list[PosteriorUtilityEstimate]
    ] = {}
    for payload in estimate_payloads:
        try:
            estimate = PosteriorUtilityEstimate.from_payload(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise _malformed("expected utility estimate") from exc
        if dict(payload) != estimate.to_payload():
            fail("expected utility estimate hash")
        estimates_by_route.setdefault(
            (estimate.center, estimate.case_id, estimate.control_id), []
        ).append(estimate)

    eligibility_payloads = table_rows(root, "candidate_eligibility")
    eligibility_index = index_rows(
        eligibility_payloads,
        ("outer_center", "center", "case_id", "control_id", "candidate_hash"),
        "candidate eligibility",
    )
    manifest = read_json(root / "manifests/candidate_probability_index.json")
    array_rows = manifest.get("arrays")
    if not isinstance(array_rows, list):
        fail("candidate dense-array manifest")
    array_index = index_rows(array_rows, ("key",), "candidate dense arrays")
    expected_array_keys = {
        digest
        for runtime in runtime_rows
        for digest in string_list(runtime, "candidate_hashes", allow_empty=True)
    }
    if {key[0] for key in array_index} != expected_array_keys:
        fail("candidate dense-array rectangle")

    if set(posterior_probabilities) != set(topology.posteriors):
        fail("candidate posterior replay rectangle")

    actions: dict[str, CandidateActionRecord] = {}
    runtime_actions: dict[
        tuple[str, str, str, str], tuple[CandidateActionRecord, ...]
    ] = {}
    selected: dict[
        tuple[str, str, str, str], CandidateActionRecord | None
    ] = {}
    expected_estimate_rows: list[dict[str, object]] = []
    expected_eligibility_rows: list[dict[str, object]] = []

    with np.load(root / "arrays/candidate_probabilities.npz", allow_pickle=False) as store:
        for runtime in runtime_rows:
            outer = str(runtime.get("outer_center"))
            center = str(runtime.get("center"))
            case = str(runtime.get("case_id"))
            control = str(runtime.get("control_id"))
            runtime_key = (outer, center, case, control)
            if runtime_key in runtime_actions:
                fail("candidate runtime semantic key")
            records: list[CandidateActionRecord] = []
            for action_hash in string_list(
                runtime, "candidate_hashes", allow_empty=True
            ):
                values = np.ascontiguousarray(
                    np.asarray(store[action_hash], dtype=np.float32)
                )
                if values.ndim != 1:
                    fail("candidate probability shape")
                record = _match_action(
                    action_hash=action_hash,
                    center=center,
                    case=case,
                    control=control,
                    values=values,
                    estimates=estimates_by_route.get(
                        (center, case, control), ()
                    ),
                    array_row=array_index[(action_hash,)],
                )
                posterior_eta = posterior_probabilities[(center, case, control)]
                _validate_posterior_lineage(
                    record.action,
                    topology=topology,
                    posterior_eta=posterior_eta,
                )
                eligibility = assess_action(record.action)
                record = CandidateActionRecord(
                    record.action,
                    eligibility,
                    record.array_key,
                    record.array_sha256,
                )
                eligibility_key = (*runtime_key, action_hash)
                expected_eligibility = {
                    "outer_center": outer,
                    "center": center,
                    "case_id": case,
                    "control_id": control,
                    **eligibility.to_payload(),
                }
                if dict(eligibility_index.get(eligibility_key, {})) != expected_eligibility:
                    fail("candidate eligibility semantics")
                if action_hash in actions and actions[action_hash].action != record.action:
                    fail("candidate hash collision")
                actions.setdefault(action_hash, record)
                records.append(record)
                expected_estimate_rows.append(record.action.estimate.to_payload())
                expected_eligibility_rows.append(expected_eligibility)

            ordered_actions = tuple(record.action for record in records)
            _validate_runtime_rectangle(
                runtime,
                tuple(records),
                topology=topology,
                posterior_eta=posterior_probabilities[(center, case, control)],
                endpoint_topology=endpoint_topology,
            )
            expected_selected = select_best_eligible_action(ordered_actions)
            observed_selected = runtime.get("selected_candidate_hash")
            if observed_selected != (
                None if expected_selected is None else expected_selected.action_hash
            ):
                fail("candidate selected action semantics")
            _validate_runtime_hash(runtime, tuple(records), expected_selected)
            runtime_actions[runtime_key] = tuple(records)
            selected[runtime_key] = (
                None
                if expected_selected is None
                else actions[expected_selected.action_hash]
            )

    if (
        list(estimate_payloads) != expected_estimate_rows
        or list(eligibility_payloads) != expected_eligibility_rows
        or set(eligibility_index)
        != {
            (*runtime_key, record.action_hash)
            for runtime_key, records in runtime_actions.items()
            for record in records
        }
    ):
        fail("candidate utility/eligibility table order or coverage")
    return actions, runtime_actions, selected


def _validate_runtime_rectangle(
    runtime: Row,
    records: tuple[CandidateActionRecord, ...],
    *,
    topology: PlanPosteriorTopology,
    posterior_eta: np.ndarray,
    endpoint_topology: EndpointEvidenceTopology,
) -> None:
    outer = str(runtime["outer_center"])
    center = str(runtime["center"])
    case = str(runtime["case_id"])
    control = str(runtime["control_id"])
    model = topology.models[(center, case, control)]
    if outer == center:
        portfolio = endpoint_topology.target_probabilities[
            (center, case, PORTFOLIO_METHOD_ID)
        ]
        alternatives = {
            method: endpoint_topology.target_probabilities[(center, case, method)]
            for method in ALTERNATIVE_METHOD_IDS
        }
        excluded = (center,)
    else:
        evidence_key = (outer, center, case)
        if runtime.get("endpoint_lineage_hash") != (
            endpoint_topology.pseudo_evidence_hashes[evidence_key]
        ):
            fail("pseudo candidate endpoint evidence lineage")
        portfolio = endpoint_topology.pseudo_probabilities[
            (*evidence_key, PORTFOLIO_METHOD_ID)
        ]
        alternatives = {
            method: endpoint_topology.pseudo_probabilities[
                (*evidence_key, method)
            ]
            for method in ALTERNATIVE_METHOD_IDS
        }
        excluded = tuple(sorted((outer, center)))
    expected = build_case_candidates(
        center=center,
        case_id=case,
        portfolio_probabilities=portfolio,
        alternative_probabilities=alternatives,
        posterior_eta=posterior_eta,
        control_id=control,
        support_n_positive=float(model["training_n_positive"]),
        support_n_negative=float(model["training_n_negative"]),
        support_row_count=int(model["training_row_count"]),
        posterior_model_hash=str(runtime["posterior_model_hash"]),
        support_capability_hash=str(runtime["support_capability_hash"]),
        outer_center=outer,
        source_excluded_centers=excluded,
        endpoint_lineage_hash=str(runtime["endpoint_lineage_hash"]),
    )
    if (
        tuple(record.action for record in records) != expected.candidates
        or tuple(record.eligibility for record in records) != expected.eligibility
        or runtime.get("no_crossing_count") != expected.no_crossing_count
        or runtime.get("descriptor_count") != expected.descriptor_count
        or runtime.get("selected_candidate_hash")
        != (
            None
            if expected.selected_candidate is None
            else expected.selected_candidate.action_hash
        )
        or runtime.get("runtime_hash") != expected.runtime_hash
    ):
        fail("candidate complete six-descriptor scientific reconstruction")


def _match_action(
    *,
    action_hash: str,
    center: str,
    case: str,
    control: str,
    values: np.ndarray,
    estimates: Sequence[PosteriorUtilityEstimate],
    array_row: Row,
) -> CandidateActionRecord:
    matches: list[ActionCandidate] = []
    vector = CanonicalProbabilityVector.from_array(values)
    for estimate in estimates:
        if (
            estimate.center != center
            or estimate.case_id != case
            or estimate.control_id != control
        ):
            continue
        parts = estimate.action_id.rsplit("::", 1)
        if len(parts) != 2 or parts[1] != estimate.direction:
            fail("candidate action identity")
        try:
            action = ActionCandidate(
                center,
                case,
                parts[0],
                estimate.direction,
                control,
                vector,
                estimate,
            )
        except (TypeError, ValueError) as exc:
            raise _malformed("candidate action") from exc
        if action.action_hash == action_hash:
            matches.append(action)
    unique = {row.estimate.estimate_hash: row for row in matches}
    if len(unique) != 1:
        fail("candidate action/estimate/array binding")
    action = next(iter(unique.values()))
    if (
        array_row.get("shape") != [len(values)]
        or array_row.get("dtype") != "float32"
        or array_row.get("array_sha256") != sha256_array(values)
        or action.probabilities.sha256 != probability_sha256(values)
    ):
        fail("candidate dense-array hash")
    return CandidateActionRecord(
        action,
        assess_action(action),
        action_hash,
        str(array_row["array_sha256"]),
    )


def _validate_posterior_lineage(
    action: ActionCandidate,
    *,
    topology: PlanPosteriorTopology,
    posterior_eta: np.ndarray,
) -> None:
    model = topology.models[(action.center, action.case_id, action.control_id)]
    eta = np.asarray(posterior_eta, dtype=np.float64)
    posterior_hash = canonical_hash(
        {
            "schema_version": "cbpupr_singleton_route_posterior_v1",
            "center": action.center,
            "case_id": action.case_id,
            "control_id": action.control_id,
            "eta": eta.tolist(),
            "support_n_positive": float(model["training_n_positive"]),
            "support_n_negative": float(model["training_n_negative"]),
            "support_row_count": int(model["training_row_count"]),
            "whole_case_excluded": True,
            "inner_crossfit_used": False,
        }
    )
    if (
        action.estimate.posterior_hash != posterior_hash
        or len(action.probabilities.values) != len(eta)
    ):
        fail("candidate posterior lineage")


def _validate_runtime_hash(
    runtime: Row,
    records: tuple[CandidateActionRecord, ...],
    selected: ActionCandidate | None,
) -> None:
    payload = {
        "schema_version": "cbpupr_candidate_runtime_v1",
        "outer_center": runtime.get("outer_center"),
        "center": runtime.get("center"),
        "case_id": runtime.get("case_id"),
        "control_id": runtime.get("control_id"),
        "descriptor_count": runtime.get("descriptor_count"),
        "no_crossing_count": runtime.get("no_crossing_count"),
        "candidate_hashes": [row.action_hash for row in records],
        "eligibility": [row.eligibility.to_payload() for row in records],
        "selected_candidate_hash": (
            None if selected is None else selected.action_hash
        ),
        "posterior_model_reference_count": runtime.get(
            "posterior_model_reference_count"
        ),
        "posterior_fit_increment": 0,
        "posterior_refit": False,
        "posterior_model_hash": runtime.get("posterior_model_hash"),
        "support_capability_hash": runtime.get("support_capability_hash"),
        "source_excluded_centers": list(
            string_list(runtime, "source_excluded_centers")
        ),
        "source_excluded_centers_role": SOURCE_EXCLUSION_ROLE,
        "endpoint_lineage_hash": runtime.get("endpoint_lineage_hash"),
        "support_labels_used_indirectly": True,
        "held_case_label_used": False,
        "terminal_evaluation_labels_used": False,
    }
    if (
        runtime.get("source_excluded_centers_role") != SOURCE_EXCLUSION_ROLE
        or runtime.get("runtime_hash") != canonical_hash(payload)
    ):
        fail("candidate runtime semantic hash")


def _malformed(role: str) -> Exception:
    from ...protocol import ProtocolError

    return ProtocolError(f"CBPUPR persisted {role} is malformed.")


__all__ = ("CandidateActionRecord", "validate_candidate_semantics")
