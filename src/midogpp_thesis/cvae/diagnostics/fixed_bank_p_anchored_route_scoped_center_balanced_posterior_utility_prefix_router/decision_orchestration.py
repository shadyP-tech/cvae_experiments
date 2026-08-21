"""Post-seal pseudo replay, calibration, decisions and aggregate composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ...protocol import ProtocolError
from .candidate_orchestration import SealedCandidateProducts
from .candidate_runtime import CandidateRuntimeResult
from .canonical_probabilities import require_byte_exact_p
from .calibration import UnsupportedCalibration, supported_donor_centers
from .composition import CompositionResult, compose_center_probabilities, compose_exact_p
from .constants import (
    ALTERNATIVE_METHOD_IDS,
    BLOCKED_CONTROL_METHOD_ID,
    BLOCKED_FINGERPRINT_CONTROL_ID,
    CANDIDATE_ONLY_METHOD_ID,
    CENTERS,
    COMPOSED_POLICY_IDS,
    ENDPOINT_METHOD_IDS,
    FIXED_CONTROL_MENU,
    OBSERVED_MAX_CONTROL_METHOD_ID,
    PORTFOLIO_METHOD_ID,
    PRIMARY_FINGERPRINT_CONTROL_ID,
    PRIMARY_METHOD_ID,
)
from .controls import (
    ControlPolicy,
    candidate_only_control,
    cyclic_control_policy,
    observed_maximum_prefix_control,
)
from .decision import ABSTAIN_TO_P, RouteDecision, make_route_decision
from .donor_replay_runtime import DonorReplayResult, replay_candidate
from .hashing import canonical_hash
from .policy_replay_runtime import PolicyReplayRuntimeResult, replay_pseudo_policy
from .policy_prefixes import select_prefix
from .posterior_expected_utility import FavorableUtility
from .transport_geometry import (
    NumericTransportAudit,
    StructuralTransportGate,
    audit_numeric_transport,
)
from .utility_calibration import (
    CenterBalancedUtilityCalibration,
    UtilityReplay,
    build_center_balanced_utility_calibration,
)


@dataclass(frozen=True)
class SealedDecisionProducts:
    donor_replays: tuple[DonorReplayResult, ...]
    utility_calibrations: tuple[object, ...]
    policy_replays: tuple[PolicyReplayRuntimeResult, ...]
    policy_replay_diagnostics: tuple[Mapping[str, object], ...]
    route_decisions: tuple[RouteDecision, ...]
    control_policies: tuple[tuple[str, str, ControlPolicy], ...]
    transport_audits: tuple[NumericTransportAudit, ...]
    structural_gates: tuple[StructuralTransportGate, ...]
    probabilities: Mapping[
        str, Mapping[str, Mapping[str, tuple[float, ...]]]
    ]
    sample_ids: Mapping[str, Mapping[str, tuple[str, ...]]]
    replay_calibration_seal_hash: str
    aggregate_seal_hash: str


def build_sealed_decisions(
    products: SealedCandidateProducts,
) -> SealedDecisionProducts:
    """Open pseudo labels only after candidates seal, then seal all outputs."""

    firewall = products.firewall
    prepared = dict(products.prepared_centers)
    pseudo_candidates = {
        (row.outer_center, row.center, row.case_id, row.control_id): row
        for row in products.pseudo_candidates
    }
    pseudo_portfolios = {
        (outer, donor, case): values
        for outer, donor, case, values in products.pseudo_portfolios
    }

    labels_by_route = {}
    denominators: dict[tuple[str, str], tuple[int, int, int]] = {}
    for outer in CENTERS:
        for donor in CENTERS:
            if donor == outer:
                continue
            center_rows = []
            for case in prepared[donor].cases:
                rows = firewall.open_pseudo_evaluation_labels(outer, donor, case)
                labels_by_route[(outer, donor, case)] = rows
                center_rows.extend(rows)
            values = tuple(row.value for row in center_rows)
            positive = values.count(1)
            negative = values.count(0)
            if not positive or not negative:
                raise ProtocolError("CBPUPR pseudo donor lacks both classes.")
            denominators[(outer, donor)] = (positive, negative, len(values))

    replay_results: list[DonorReplayResult] = []
    realized_by_route: dict[
        tuple[str, str, str, str], Mapping[str, object]
    ] = {}
    for key, runtime in pseudo_candidates.items():
        outer, donor, case, control = key
        selected = runtime.selected_candidate
        if selected is None:
            realized_by_route[key] = {}
            continue
        label_rows = labels_by_route[(outer, donor, case)]
        label_map = {row.sample_id: row.value for row in label_rows}
        positions = prepared[donor].case_positions[case]
        ordered_samples = tuple(
            prepared[donor].surface.sample_ids[position] for position in positions
        )
        values = tuple(label_map[sample] for sample in ordered_samples)
        positive, negative, count = denominators[(outer, donor)]
        result = replay_candidate(
            selected,
            portfolio_probabilities=pseudo_portfolios[(outer, donor, case)],
            labels=values,
            outer_center=outer,
            donor_center=donor,
            center_n_positive=positive,
            center_n_negative=negative,
            center_row_count=count,
            label_scope=label_rows[0].scope,
            source_excluded_centers=(outer, donor),
            endpoint_lineage_hash=runtime.endpoint_lineage_hash,
        )
        replay_results.append(result)
        realized_by_route[key] = {
            selected.action_hash: result.replay.realized_utility
        }

    replays = tuple(row.replay for row in replay_results)
    utility_calibrations: dict[tuple[str, str], object] = {}
    for outer in CENTERS:
        for control in (
            PRIMARY_FINGERPRINT_CONTROL_ID,
            BLOCKED_FINGERPRINT_CONTROL_ID,
        ):
            calibration_rows = tuple(
                row
                for row in replays
                if row.outer_center == outer and row.control_id == control
            )
            utility_calibrations[(outer, control)] = _utility_calibration_or_unsupported(
                calibration_rows, outer=outer, excluded=(outer,)
            )

    policy_replays: list[PolicyReplayRuntimeResult] = []
    for outer in CENTERS:
        for donor in CENTERS:
            if donor == outer:
                continue
            leave_j_rows = tuple(
                row
                for row in replays
                if row.outer_center == outer
                and row.control_id == PRIMARY_FINGERPRINT_CONTROL_ID
                and row.donor_center != donor
            )
            leave_j = _utility_calibration_or_unsupported(
                leave_j_rows, outer=outer, excluded=(outer, donor)
            )
            runtimes = tuple(
                pseudo_candidates[(outer, donor, case, PRIMARY_FINGERPRINT_CONTROL_ID)]
                for case in prepared[donor].cases
            )
            realized = {
                digest: value
                for case in prepared[donor].cases
                for digest, value in realized_by_route[
                    (outer, donor, case, PRIMARY_FINGERPRINT_CONTROL_ID)
                ].items()
            }
            if isinstance(leave_j, CenterBalancedUtilityCalibration):
                policy_replays.append(
                    replay_pseudo_policy(
                        runtimes,
                        realized,
                        outer_center=outer,
                        donor_center=donor,
                        leave_j_candidate_calibration=leave_j,
                    )
                )
    policy_replay_diagnostics: list[Mapping[str, object]] = []
    for outer in CENTERS:
        rows = tuple(
            row.replay for row in policy_replays if row.replay.outer_center == outer
        )
        payload = {
            "schema_version": "fixed_bank_cbpupr_policy_replay_diagnostic_v1",
            "outer_center": outer,
            "donor_centers": sorted({row.donor_center for row in rows}),
            "replay_hashes": sorted(row.replay_hash for row in rows),
            "policy_replay_bias_used": False,
            "authorization_gate": False,
        }
        policy_replay_diagnostics.append(
            {**payload, "diagnostic_hash": canonical_hash(payload)}
        )

    replay_hash = canonical_hash(sorted(row.result_hash for row in replay_results))
    calibration_hash = canonical_hash(
        sorted(row.calibration_hash for row in utility_calibrations.values())
    )
    policy_replay_hash = canonical_hash(
        sorted(row.runtime_hash for row in policy_replays)
    )
    calibration_seal = firewall.seal_replays_and_calibrations(
        replay_hash, calibration_hash, policy_replay_hash
    )

    target_candidates = {
        (row.center, row.case_id, row.control_id): row
        for row in products.target_candidates
    }
    target_portfolios = {
        (center, case): values for center, case, values in products.target_portfolios
    }
    structural = _structural_gates(products)
    structural_by_center = {row.target_center: row for row in structural}
    audits = _transport_audits(products)

    route_decisions: list[RouteDecision] = []
    controls: list[tuple[str, str, ControlPolicy]] = []
    compositions: dict[tuple[str, str], CompositionResult] = {}
    for center in CENTERS:
        case_ids = prepared[center].surface.case_ids
        portfolio = _assemble_center_portfolio(
            prepared[center], target_portfolios, center
        )
        identity = tuple(
            target_candidates[(center, case, PRIMARY_FINGERPRINT_CONTROL_ID)]
            for case in prepared[center].cases
        )
        cyclic = tuple(
            target_candidates[(center, case, BLOCKED_FINGERPRINT_CONTROL_ID)]
            for case in prepared[center].cases
        )
        primary = _make_decision_or_exact_p(
            center=center,
            portfolio_probabilities=portfolio,
            sample_case_ids=case_ids,
            candidate_results=identity,
            utility_calibration=utility_calibrations[
                (center, PRIMARY_FINGERPRINT_CONTROL_ID)
            ],
            structural_transport=structural_by_center[center],
            method_id=PRIMARY_METHOD_ID,
        )
        cyclic_decision = _make_decision_or_exact_p(
            center=center,
            portfolio_probabilities=portfolio,
            sample_case_ids=case_ids,
            candidate_results=cyclic,
            utility_calibration=utility_calibrations[
                (center, BLOCKED_FINGERPRINT_CONTROL_ID)
            ],
            structural_transport=structural_by_center[center],
            method_id=BLOCKED_CONTROL_METHOD_ID,
        )
        route_decisions.extend((primary, cyclic_decision))
        compositions[(PRIMARY_METHOD_ID, center)] = primary.composition
        compositions[(BLOCKED_CONTROL_METHOD_ID, center)] = cyclic_decision.composition

        selected_identity = tuple(
            row.selected_candidate
            for row in identity
            if row.selected_candidate is not None
        )
        structural_passed = structural_by_center[center].passed
        if structural_passed:
            candidate_only = candidate_only_control(selected_identity)
        else:
            candidate_only = ControlPolicy(
                CANDIDATE_ONLY_METHOD_ID,
                (),
                FavorableUtility.zeros(),
                False,
                (structural_by_center[center].gate_hash,),
            )
        controls.append((CANDIDATE_ONLY_METHOD_ID, center, candidate_only))
        compositions[(CANDIDATE_ONLY_METHOD_ID, center)] = (
            compose_center_probabilities(portfolio, case_ids, selected_identity)
            if structural_passed and candidate_only.authorized
            else compose_exact_p(portfolio)
        )
        observed_rows = tuple(
            row
            for row in replays
            if row.outer_center == center
            and row.control_id == PRIMARY_FINGERPRINT_CONTROL_ID
        )
        if structural_passed and observed_rows:
            observed_selection, observed = observed_maximum_prefix_control(
                selected_identity, observed_rows
            )
            del observed_selection
        else:
            source_hash = canonical_hash(
                [
                    "CBPUPR_OBSERVED_MAX_UNSUPPORTED_OR_STRUCTURAL_FAILURE",
                    center,
                    structural_by_center[center].gate_hash,
                ]
            )
            observed = ControlPolicy(
                OBSERVED_MAX_CONTROL_METHOD_ID,
                (),
                FavorableUtility.zeros(),
                False,
                (source_hash,),
            )
        controls.append((OBSERVED_MAX_CONTROL_METHOD_ID, center, observed))
        observed_candidates = tuple(
            row
            for row in selected_identity
            if row.action_hash in set(observed.selected_candidate_hashes)
        )
        compositions[(OBSERVED_MAX_CONTROL_METHOD_ID, center)] = (
            compose_center_probabilities(portfolio, case_ids, observed_candidates)
            if structural_passed and observed.authorized
            else compose_exact_p(portfolio)
        )
        controls.append(
            (
                BLOCKED_CONTROL_METHOD_ID,
                center,
                cyclic_control_policy(cyclic_decision.prefix_selection),
            )
        )

    sample_ids = {
        center: {
            case: tuple(
                prepared[center].surface.sample_ids[position]
                for position in prepared[center].case_positions[case]
            )
            for case in prepared[center].cases
        }
        for center in CENTERS
    }
    probabilities = _build_probability_rectangle(
        products, prepared, compositions
    )
    for method in COMPOSED_POLICY_IDS:
        for center in CENTERS:
            decision_hash = _policy_hash(method, center, route_decisions, controls)
            selected_cases = set(compositions[(method, center)].selected_case_ids)
            for case in prepared[center].cases:
                firewall.seal_target_decision(
                    method,
                    center,
                    case,
                    canonical_hash(
                        [decision_hash, center, case, case in selected_cases]
                    ),
                )
    aggregate_prediction_hash = canonical_hash(
        {
            method: {
                center: {
                    case: canonical_hash(list(values))
                    for case, values in probabilities[method][center].items()
                }
                for center in CENTERS
            }
            for method in FIXED_CONTROL_MENU
        }
    )
    aggregate_seal = firewall.seal_aggregate(aggregate_prediction_hash)
    return SealedDecisionProducts(
        tuple(replay_results),
        tuple(utility_calibrations.values()),
        tuple(policy_replays),
        tuple(policy_replay_diagnostics),
        tuple(route_decisions),
        tuple(controls),
        audits,
        structural,
        probabilities,
        sample_ids,
        calibration_seal,
        aggregate_seal,
    )


def _assemble_center_portfolio(
    prepared: object,
    target_portfolios: Mapping[tuple[str, str], tuple[float, ...]],
    center: str,
) -> np.ndarray:
    result = np.empty(len(prepared.surface.sample_ids), dtype=np.float32)
    for case in prepared.cases:
        positions = prepared.case_positions[case]
        values = np.asarray(target_portfolios[(center, case)], dtype=np.float32)
        if len(values) != len(positions):
            raise ProtocolError("CBPUPR target portfolio alignment drifted.")
        result[positions] = values
    result.setflags(write=False)
    return result


def _utility_calibration_or_unsupported(
    rows: tuple[UtilityReplay, ...], *, outer: str, excluded: tuple[str, ...]
) -> object:
    donors = supported_donor_centers(rows, excluded_centers=excluded)
    if len(donors) < 6:
        return UnsupportedCalibration(
            "candidate_utility", outer, excluded, donors
        )
    return build_center_balanced_utility_calibration(
        rows,
        outer_center=outer,
        calibration_excluded_centers=excluded,
    )


def _make_decision_or_exact_p(
    *,
    center: str,
    portfolio_probabilities: object,
    sample_case_ids: tuple[str, ...],
    candidate_results: tuple[CandidateRuntimeResult, ...],
    utility_calibration: object,
    structural_transport: StructuralTransportGate,
    method_id: str,
) -> RouteDecision:
    if isinstance(utility_calibration, CenterBalancedUtilityCalibration):
        return make_route_decision(
            center=center,
            portfolio_probabilities=portfolio_probabilities,
            sample_case_ids=sample_case_ids,
            candidate_results=candidate_results,
            utility_calibration=utility_calibration,
            structural_transport=structural_transport,
            method_id=method_id,
        )
    if not isinstance(utility_calibration, UnsupportedCalibration):
        raise ProtocolError("CBPUPR calibration outcome type drifted.")
    return RouteDecision(
        center,
        method_id,
        ABSTAIN_TO_P,
        (utility_calibration.reason_code,),
        select_prefix(()),
        compose_exact_p(portfolio_probabilities),
        structural_transport,
        utility_calibration.calibration_hash,
        tuple(sorted(row.runtime_hash for row in candidate_results)),
    )


def _transport_audits(
    products: SealedCandidateProducts,
) -> tuple[NumericTransportAudit, ...]:
    surfaces = {row.center: row for row in products.primary_fingerprints}
    names = tuple(f"physical_fingerprint_dimension_{index}" for index in range(30))
    return tuple(
        audit_numeric_transport(
            target_center=center,
            target_vector=np.mean(
                surfaces[center].feature_values, axis=0, dtype=np.float64
            ),
            reference_vectors_by_center={
                other: surfaces[other].feature_values
                for other in CENTERS
                if other != center
            },
            feature_names=names,
        )
        for center in CENTERS
    )


def _structural_gates(
    products: SealedCandidateProducts,
) -> tuple[StructuralTransportGate, ...]:
    plans = {row.key: row for row in products.plan_seal.outer_plans}
    plan_keys = set(plans)
    models = {
        (row.target_center, row.held_case_id, row.control_id): row
        for row in products.posterior_models
    }
    candidates = tuple(products.target_candidates)
    endpoints = {
        (product.target_center, prediction.case_id): prediction.prediction_hash
        for product in products.endpoint_products
        for prediction in product.predictions
    }
    result = []
    for center in CENTERS:
        center_rows = tuple(row for row in candidates if row.center == center)
        expected_route_controls = {
            (case, control)
            for observed, case in plan_keys
            if observed == center
            for control in (
                PRIMARY_FINGERPRINT_CONTROL_ID,
                BLOCKED_FINGERPRINT_CONTROL_ID,
            )
        }
        observed_route_controls = {
            (row.case_id, row.control_id) for row in center_rows
        }
        exact_topology = (
            len(center_rows) == len(expected_route_controls)
            and observed_route_controls == expected_route_controls
        )
        model_topology = (
            {
                (case, control)
                for observed, case, control in models
                if observed == center
            }
            == expected_route_controls
        )
        probability_lineage = all(
            row.endpoint_lineage_hash == endpoints.get((center, row.case_id))
            for row in center_rows
        )
        plan_lineage = exact_topology and all(
            (center, row.case_id) in plan_keys for row in center_rows
        )
        target_excluded = model_topology and all(
            tuple(models[(center, row.case_id, row.control_id)].training_case_ids)
            == tuple(plans[(center, row.case_id)].support_case_ids)
            for row in center_rows
        )
        noninterference = exact_topology and model_topology and all(
            row.outer_center == center
            and set(row.source_excluded_centers) == {center}
            and models[(center, row.case_id, row.control_id)].held_case_id
            == row.case_id
            and row.posterior_model_hash
            == models[(center, row.case_id, row.control_id)].model_hash
            for row in center_rows
        )
        finite = all(
            np.isfinite(candidate.probabilities.as_array()).all()
            and np.isfinite(candidate.estimate.utility.as_tuple()).all()
            for row in center_rows
            for candidate in row.candidates
        )
        result.append(
            StructuralTransportGate(
                center,
                probability_lineage,
                plan_lineage,
                target_excluded,
                noninterference,
                finite,
            )
        )
    return tuple(result)


def _build_probability_rectangle(
    products: SealedCandidateProducts,
    prepared: Mapping[str, object],
    compositions: Mapping[tuple[str, str], CompositionResult],
) -> dict[str, dict[str, dict[str, tuple[float, ...]]]]:
    endpoint = {
        method: {center: {} for center in CENTERS} for method in ENDPOINT_METHOD_IDS
    }
    for center_product in products.endpoint_products:
        for prediction in center_product.predictions:
            for method in ENDPOINT_METHOD_IDS:
                endpoint[method][center_product.target_center][prediction.case_id] = tuple(
                    prediction.probabilities[method]
                )
    output = dict(endpoint)
    for method in COMPOSED_POLICY_IDS:
        output[method] = {}
        for center in CENTERS:
            composition = compositions[(method, center)]
            array = composition.probabilities.as_array()
            output[method][center] = {
                case: tuple(float(value) for value in array[prepared[center].case_positions[case]])
                for case in prepared[center].cases
            }
            selected = set(composition.selected_case_ids)
            for case in prepared[center].cases:
                if case not in selected:
                    require_byte_exact_p(
                        output[method][center][case],
                        endpoint[PORTFOLIO_METHOD_ID][center][case],
                    )
    return output


def _policy_hash(
    method: str,
    center: str,
    decisions: list[RouteDecision],
    controls: list[tuple[str, str, ControlPolicy]],
) -> str:
    if method in {PRIMARY_METHOD_ID, BLOCKED_CONTROL_METHOD_ID}:
        return next(
            row.decision_hash
            for row in decisions
            if row.center == center and row.method_id == method
        )
    return next(
        row.policy_hash
        for observed_method, observed_center, row in controls
        if observed_method == method and observed_center == center
    )


__all__ = ("SealedDecisionProducts", "build_sealed_decisions")
