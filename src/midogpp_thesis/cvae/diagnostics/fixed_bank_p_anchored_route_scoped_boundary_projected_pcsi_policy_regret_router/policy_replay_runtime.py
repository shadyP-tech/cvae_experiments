"""Route-scoped selection, pseudo replay, observed envelope, and decisions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .calibration import (
    DescriptorMatchedAnnotation,
    DonorCaseEnvelope,
    RouteCalibration,
    build_route_calibration,
)
from .case_regret import PseudoCaseReplay, build_pseudo_case_replay
from .constants import (
    CENTERS,
    COMPOSED_POLICY_IDS,
    DIRECTION_IDS,
    EXPECTED_FINAL_CASE_PREDICTION_COUNT,
    EXPECTED_POLICY_REPLAY_COUNT,
    EXPECTED_PRIMARY_GEOMETRY_DECISION_COUNT,
    EXPECTED_PROJECTED_NO_ENVELOPE_DECISION_COUNT,
    EXPECTED_ROLE_BOUND_TRANSPORT_DESCRIPTOR_COUNT,
    EXPECTED_TARGET_DECISION_COUNT_PER_GEOMETRY,
    PORTFOLIO_METHOD_ID,
    PRIMARY_FINGERPRINT_CONTROL_ID,
    PRIMARY_METHOD_ID,
    PROJECTION_GEOMETRY_ID,
    PROJECTED_NO_ENVELOPE_METHOD_ID,
    RAW_OBSERVED_MAX_METHOD_ID,
    UNPROJECTED_GEOMETRY_ID,
)
from .contracts import (
    EndpointCasePrediction,
    PseudoReferenceKey,
    PseudoRouteKey,
    TargetReferenceKey,
    TargetRouteKey,
)
from .decision import RouteDiagnosticDecision, make_route_decision
from .donor_runtime import DonorRuntimeResult, RACR_GEOMETRIES
from .endpoint_reconstruction import (
    PreparedCenter,
    reconstruct_case_endpoint_seed_probabilities,
)
from .hashing import canonical_hash
from .label_capabilities import PCSIRACRLabelFirewall
from .outer_endpoint_runtime import OuterEndpointProducts
from .policy_selection import (
    CaseCandidatePolicy,
    FinalCasePolicyPrediction,
    finalize_case_policy,
    select_and_compose_case_policy,
)
from .sample_influence_contracts import (
    InfluencePrediction,
    TargetLocalPosteriorModel,
    TargetLocalPosteriorPrediction,
)
from .transport import (
    RouteTransportLineage,
    RouteTransportScreen,
    SupportConditionedCaseDescriptor,
    TransportReferenceBlockSummary,
    TransportRuntimeSeal,
    build_case_transport_descriptor,
    build_reference_block_summary,
    evaluate_transport_screen,
    seal_transport_runtime,
)


GEOMETRY_POLICY = MappingProxyType(
    {
        PROJECTION_GEOMETRY_ID: PRIMARY_METHOD_ID,
        UNPROJECTED_GEOMETRY_ID: RAW_OBSERVED_MAX_METHOD_ID,
    }
)


@dataclass(frozen=True)
class PolicyReplayRuntimeResult:
    transport_descriptors_by_outer_candidate: Mapping[
        object, SupportConditionedCaseDescriptor
    ]
    transport_reference_blocks: Mapping[object, TransportReferenceBlockSummary]
    transport_screens: Mapping[object, RouteTransportScreen]
    descriptor_matches: Mapping[tuple[str, str, str], str]
    descriptor_match_hash: str
    target_influences_by_policy_center: Mapping[
        tuple[str, str], tuple[InfluencePrediction, ...]
    ]
    target_candidate_policies: Mapping[
        tuple[str, str, str], CaseCandidatePolicy
    ]
    pseudo_candidate_policies: Mapping[
        tuple[str, str, str, str], CaseCandidatePolicy
    ]
    replays: Mapping[tuple[str, str, str, str], PseudoCaseReplay]
    donor_envelopes: Mapping[tuple[str, str, str], DonorCaseEnvelope]
    calibrations: Mapping[tuple[str, str], RouteCalibration]
    matched_annotations: Mapping[tuple[str, str], DescriptorMatchedAnnotation]
    decisions: Mapping[tuple[str, str, str], RouteDiagnosticDecision]
    authorizations: Mapping[tuple[str, str, str], RouteDiagnosticDecision]
    final_predictions_by_policy: Mapping[
        str, tuple[FinalCasePolicyPrediction, ...]
    ]
    policy_menu_seal: Mapping[str, object]
    transport_seal: TransportRuntimeSeal
    transport_hash: str
    runtime_hash: str

    def summary_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_racr_policy_replay_runtime_v1",
            "transport_descriptor_count": len(
                self.transport_descriptors_by_outer_candidate
            ),
            "transport_numeric_leaf_count": self.transport_seal.numeric_leaf_count,
            "transport_reference_summary_count": len(
                self.transport_reference_blocks
            ),
            "transport_screen_count": len(self.transport_screens),
            "descriptor_match_count": len(self.descriptor_matches),
            "target_candidate_policy_count": len(self.target_candidate_policies),
            "pseudo_candidate_policy_count": len(self.pseudo_candidate_policies),
            "policy_replay_count": len(self.replays),
            "donor_envelope_count": len(self.donor_envelopes),
            "route_calibration_count": len(self.calibrations),
            "primary_geometry_decision_count": sum(
                key[0] in {PRIMARY_METHOD_ID, RAW_OBSERVED_MAX_METHOD_ID}
                for key in self.decisions
            ),
            "projected_no_envelope_decision_count": sum(
                key[0] == PROJECTED_NO_ENVELOPE_METHOD_ID
                for key in self.decisions
            ),
            "final_case_prediction_count_including_P": (
                sum(len(rows) for rows in self.final_predictions_by_policy.values())
                + sum(len(rows) for rows in self.final_predictions_by_policy.values())
                // len(COMPOSED_POLICY_IDS)
            ),
            "transport_hash": self.transport_hash,
            "transport_protocol_status": "ROUTE_SCOPED_OWN_CASE_NONINTERFERENCE",
            "transport_authorization_valid": True,
            "policy_menu_seal_hash": self.policy_menu_seal[
                "policy_menu_seal_hash"
            ],
            "runtime_hash": self.runtime_hash,
        }


def build_policy_replay_runtime(
    *,
    predictions_by_center: Mapping[str, Sequence[EndpointCasePrediction]],
    endpoint_products: Sequence[OuterEndpointProducts],
    donor_endpoint_products: Mapping[tuple[str, str], OuterEndpointProducts],
    prepared_by_center: Mapping[str, PreparedCenter],
    donor_runtime: DonorRuntimeResult,
    target_posterior_models_by_control: Mapping[
        str, Sequence[TargetLocalPosteriorModel]
    ],
    target_posterior_predictions_by_control: Mapping[
        str, Sequence[TargetLocalPosteriorPrediction]
    ],
    label_firewall: PCSIRACRLabelFirewall,
    strict_canonical_topology: bool = True,
) -> PolicyReplayRuntimeResult:
    expected_pairs = {
        (outer, donor)
        for outer in CENTERS
        for donor in CENTERS
        if donor != outer
    }
    endpoint_by_center = {row.target_center: row for row in endpoint_products}
    if (
        set(predictions_by_center) != set(CENTERS)
        or set(endpoint_by_center) != set(CENTERS)
        or set(donor_endpoint_products) != expected_pairs
        or set(prepared_by_center) != set(CENTERS)
        or PRIMARY_FINGERPRINT_CONTROL_ID not in target_posterior_models_by_control
        or PRIMARY_FINGERPRINT_CONTROL_ID
        not in target_posterior_predictions_by_control
    ):
        raise ProtocolError("PCSI-RACR policy runtime input matrix drifted.")

    model_index = {
        (row.target_center, row.held_case_id): row
        for row in target_posterior_models_by_control[
            PRIMARY_FINGERPRINT_CONTROL_ID
        ]
    }
    posterior_index = {
        (row.target_center, row.case_id): row
        for row in target_posterior_predictions_by_control[
            PRIMARY_FINGERPRINT_CONTROL_ID
        ]
    }
    (
        descriptors,
        reference_blocks,
        transport_screens,
        transport_seal,
    ) = _build_route_transport_runtime(
        endpoint_by_center=endpoint_by_center,
        donor_endpoint_products=donor_endpoint_products,
        donor_runtime=donor_runtime,
        prepared_by_center=prepared_by_center,
        strict_canonical_topology=strict_canonical_topology,
    )
    matches, match_hash = _build_descriptor_matches(
        descriptors, transport_screens
    )

    influences: dict[tuple[str, str], tuple[InfluencePrediction, ...]] = {}
    target_candidates: dict[tuple[str, str, str], CaseCandidatePolicy] = {}
    for geometry_id, policy_id in GEOMETRY_POLICY.items():
        geometry = donor_runtime.geometry_results[geometry_id]
        for center in CENTERS:
            actions_by_case = _rows_by_case(
                geometry.target_actions_by_center[center]
            )
            descriptors_by_case = _rows_by_case(
                geometry.target_descriptors_by_center[center]
            )
            utilities_by_case = _prediction_rows_by_case(
                geometry.target_descriptors_by_center[center],
                geometry.target_predictions_by_center[center],
            )
            center_influences: list[InfluencePrediction] = []
            for endpoint in predictions_by_center[center]:
                route = center, endpoint.case_id
                case_influences = _score_influences(
                    descriptors_by_case[endpoint.case_id],
                    posterior=posterior_index[route],
                    model=model_index[route],
                )
                center_influences.extend(case_influences)
                candidate = select_and_compose_case_policy(
                    endpoint,
                    actions_by_case[endpoint.case_id],
                    descriptors_by_case[endpoint.case_id],
                    case_influences,
                    utilities_by_case[endpoint.case_id],
                    policy_id=policy_id,
                    require_positive_bacc_prediction=False,
                )
                target_candidates[(policy_id, center, endpoint.case_id)] = candidate
                label_firewall.record_target_case_policy_seal(
                    center, endpoint.case_id, policy_id, candidate.policy_hash
                )
                if geometry_id == PROJECTION_GEOMETRY_ID:
                    no_envelope = select_and_compose_case_policy(
                        endpoint,
                        actions_by_case[endpoint.case_id],
                        descriptors_by_case[endpoint.case_id],
                        case_influences,
                        utilities_by_case[endpoint.case_id],
                        policy_id=PROJECTED_NO_ENVELOPE_METHOD_ID,
                        require_positive_bacc_prediction=False,
                    )
                    if (
                        no_envelope.probabilities != candidate.probabilities
                        or tuple(
                            row.selected_representative
                            for row in no_envelope.decisions
                        )
                        != tuple(
                            row.selected_representative
                            for row in candidate.decisions
                        )
                    ):
                        raise ProtocolError(
                            "PCSI-RACR q=0 control changed projected selection."
                        )
                    target_candidates[
                        (PROJECTED_NO_ENVELOPE_METHOD_ID, center, endpoint.case_id)
                    ] = no_envelope
                    label_firewall.record_target_case_policy_seal(
                        center,
                        endpoint.case_id,
                        PROJECTED_NO_ENVELOPE_METHOD_ID,
                        no_envelope.policy_hash,
                    )
            influences[(policy_id, center)] = tuple(center_influences)
            if geometry_id == PROJECTION_GEOMETRY_ID:
                influences[
                    (PROJECTED_NO_ENVELOPE_METHOD_ID, center)
                ] = tuple(center_influences)

    pseudo_candidates: dict[
        tuple[str, str, str, str], CaseCandidatePolicy
    ] = {}
    for geometry_id, policy_id in GEOMETRY_POLICY.items():
        geometry = donor_runtime.geometry_results[geometry_id]
        for outer in CENTERS:
            for donor in CENTERS:
                if donor == outer:
                    continue
                pair = outer, donor
                actions_by_case = _rows_by_case(
                    geometry.pseudo_actions_by_pair[pair]
                )
                descriptors_by_case = _rows_by_case(
                    geometry.pseudo_descriptors_by_pair[pair]
                )
                utilities_by_case = _prediction_rows_by_case(
                    geometry.pseudo_descriptors_by_pair[pair],
                    geometry.pseudo_predictions_by_pair[pair],
                )
                for endpoint in donor_endpoint_products[pair].predictions:
                    posterior_route = donor, endpoint.case_id
                    case_influences = _score_influences(
                        descriptors_by_case[endpoint.case_id],
                        posterior=posterior_index[posterior_route],
                        model=model_index[posterior_route],
                    )
                    candidate = select_and_compose_case_policy(
                        endpoint,
                        actions_by_case[endpoint.case_id],
                        descriptors_by_case[endpoint.case_id],
                        case_influences,
                        utilities_by_case[endpoint.case_id],
                        policy_id=policy_id,
                        require_positive_bacc_prediction=False,
                    )
                    key = geometry_id, outer, donor, endpoint.case_id
                    pseudo_candidates[key] = candidate
                    label_firewall.record_pseudo_case_policy_seal(
                        outer,
                        donor,
                        geometry_id,
                        endpoint.case_id,
                        candidate.policy_hash,
                    )

    pre_evaluation_payload = {
        "schema_version": "fixed_bank_pcsi_racr_pre_evaluation_seal_v1",
        "transport_hash": transport_seal.transport_hash,
        "descriptor_match_hash": match_hash,
        "target_candidate_hash": canonical_hash(
            [
                target_candidates[key].policy_hash
                for key in sorted(target_candidates)
            ]
        ),
        "pseudo_candidate_hash": canonical_hash(
            [
                pseudo_candidates[key].policy_hash
                for key in sorted(pseudo_candidates)
            ]
        ),
        "pseudo_evaluation_labels_used": False,
    }
    pre_evaluation_hash = canonical_hash(pre_evaluation_payload)
    label_firewall.record_pre_evaluation_seal(
        pre_evaluation_hash,
        transport_hash=transport_seal.transport_hash,
        match_hash=match_hash,
    )

    replays: dict[tuple[str, str, str, str], PseudoCaseReplay] = {}
    for geometry_id in RACR_GEOMETRIES:
        for outer in CENTERS:
            for donor in CENTERS:
                if donor == outer:
                    continue
                endpoints = {
                    row.case_id: row for row in donor_endpoint_products[(outer, donor)].predictions
                }
                labels_by_case = {
                    case_id: label_firewall.open_pseudo_evaluation_labels(
                        outer,
                        donor,
                        geometry_id,
                        case_id,
                        policy_seal_hash=pseudo_candidates[
                            (geometry_id, outer, donor, case_id)
                        ].policy_hash,
                        pre_evaluation_seal_hash=pre_evaluation_hash,
                    )
                    for case_id in sorted(endpoints)
                }
                all_labels = tuple(
                    row
                    for case_id in sorted(labels_by_case)
                    for row in labels_by_case[case_id]
                )
                n_positive = sum(row.value == 1 for row in all_labels)
                n_negative = sum(row.value == 0 for row in all_labels)
                if n_positive <= 0 or n_negative <= 0:
                    raise ProtocolError(
                        "PCSI-RACR pseudo donor center lacks both classes."
                    )
                for case_id in sorted(endpoints):
                    key = geometry_id, outer, donor, case_id
                    replay = build_pseudo_case_replay(
                        outer_center=outer,
                        candidate=pseudo_candidates[key],
                        endpoint=endpoints[case_id],
                        case_labels=labels_by_case[case_id],
                        center_n_positive=n_positive,
                        center_n_negative=n_negative,
                    )
                    label_firewall.record_pseudo_replay_seal(
                        outer,
                        donor,
                        geometry_id,
                        case_id,
                        replay.replay_hash,
                    )
                    replays[key] = replay

    if strict_canonical_topology and len(replays) != EXPECTED_POLICY_REPLAY_COUNT:
        raise ProtocolError("PCSI-RACR pseudo replay workload drifted.")

    case_ids_by_center = {
        center: tuple(
            row.case_id for row in predictions_by_center[center]
        )
        for center in CENTERS
    }
    calibrations: dict[tuple[str, str], RouteCalibration] = {}
    donor_envelopes: dict[tuple[str, str, str], DonorCaseEnvelope] = {}
    for geometry_id in RACR_GEOMETRIES:
        for outer in CENTERS:
            calibration = build_route_calibration(
                tuple(
                    replay
                    for key, replay in replays.items()
                    if key[0] == geometry_id and key[1] == outer
                ),
                geometry_id=geometry_id,
                outer_center=outer,
                expected_case_ids_by_center=case_ids_by_center,
            )
            calibrations[(geometry_id, outer)] = calibration
            for envelope in calibration.donor_envelopes:
                donor_envelopes[
                    (geometry_id, outer, envelope.donor_center)
                ] = envelope
            label_firewall.record_calibration_seal(
                geometry_id, outer, calibration.calibration_hash
            )

    matched_annotations = _build_matched_annotations(
        matches=matches,
        match_hash=match_hash,
        replays=replays,
    )

    decisions: dict[tuple[str, str, str], RouteDiagnosticDecision] = {}
    for geometry_id, policy_id in GEOMETRY_POLICY.items():
        for center in CENTERS:
            calibration = calibrations[(geometry_id, center)]
            for endpoint in predictions_by_center[center]:
                key = policy_id, center, endpoint.case_id
                decision = make_route_decision(
                    target_candidates[key],
                    transport_screens[TargetRouteKey(center, endpoint.case_id)],
                    calibration,
                    policy_id=policy_id,
                )
                decisions[key] = decision
                label_firewall.record_target_decision_seal(
                    policy_id, center, endpoint.case_id, decision.decision_hash
                )
                if geometry_id == PROJECTION_GEOMETRY_ID:
                    q0_key = (
                        PROJECTED_NO_ENVELOPE_METHOD_ID,
                        center,
                        endpoint.case_id,
                    )
                    q0 = make_route_decision(
                        target_candidates[q0_key],
                        transport_screens[
                            TargetRouteKey(center, endpoint.case_id)
                        ],
                        calibration,
                        policy_id=PROJECTED_NO_ENVELOPE_METHOD_ID,
                        no_envelope=True,
                    )
                    decisions[q0_key] = q0
                    label_firewall.record_target_decision_seal(
                        PROJECTED_NO_ENVELOPE_METHOD_ID,
                        center,
                        endpoint.case_id,
                        q0.decision_hash,
                    )

    if strict_canonical_topology:
        primary_count = sum(
            policy in {PRIMARY_METHOD_ID, RAW_OBSERVED_MAX_METHOD_ID}
            for policy, _center, _case in decisions
        )
        q0_count = sum(
            policy == PROJECTED_NO_ENVELOPE_METHOD_ID
            for policy, _center, _case in decisions
        )
        if (
            primary_count != EXPECTED_PRIMARY_GEOMETRY_DECISION_COUNT
            or q0_count != EXPECTED_PROJECTED_NO_ENVELOPE_DECISION_COUNT
        ):
            raise ProtocolError("PCSI-RACR target decision workload drifted.")

    final: dict[str, tuple[FinalCasePolicyPrediction, ...]] = {}
    endpoint_index = {
        (center, row.case_id): row
        for center in CENTERS
        for row in predictions_by_center[center]
    }
    for policy_id in COMPOSED_POLICY_IDS:
        rows = []
        for center in CENTERS:
            for case_id in case_ids_by_center[center]:
                key = policy_id, center, case_id
                decision = decisions[key]
                rows.append(
                    finalize_case_policy(
                        target_candidates[key],
                        endpoint_index[(center, case_id)],
                        authorized=decision.changed,
                        authorization_hash=decision.decision_hash,
                    )
                )
        final[policy_id] = tuple(rows)

    barrier = label_firewall.decision_barrier_payload()
    menu_payload = {
        "schema_version": "fixed_bank_pcsi_racr_policy_menu_seal_v1",
        "policy_ids": [
            PORTFOLIO_METHOD_ID,
            PRIMARY_METHOD_ID,
            RAW_OBSERVED_MAX_METHOD_ID,
            PROJECTED_NO_ENVELOPE_METHOD_ID,
        ],
        "decision_barrier_hash": barrier["decision_barrier_hash"],
        "decision_hash": canonical_hash(
            [decisions[key].decision_hash for key in sorted(decisions)]
        ),
        "terminal_labels_used": False,
    }
    menu = {
        **menu_payload,
        "policy_menu_seal_hash": canonical_hash(menu_payload),
    }
    final_count = sum(len(rows) for rows in final.values()) + sum(
        len(rows) for rows in predictions_by_center.values()
    )
    if strict_canonical_topology and final_count != EXPECTED_FINAL_CASE_PREDICTION_COUNT:
        raise ProtocolError("PCSI-RACR four-surface prediction workload drifted.")
    runtime_payload = {
        "schema_version": "fixed_bank_pcsi_racr_policy_replay_runtime_v1",
        "transport_hash": transport_seal.transport_hash,
        "descriptor_match_hash": match_hash,
        "pre_evaluation_seal_hash": pre_evaluation_hash,
        "replay_hash": canonical_hash(
            [replays[key].replay_hash for key in sorted(replays)]
        ),
        "calibration_hash": canonical_hash(
            [calibrations[key].calibration_hash for key in sorted(calibrations)]
        ),
        "decision_hash": canonical_hash(
            [decisions[key].decision_hash for key in sorted(decisions)]
        ),
        "final_prediction_hash": canonical_hash(
            [
                row.prediction_hash
                for policy in COMPOSED_POLICY_IDS
                for row in final[policy]
            ]
        ),
        "own_route_noninterference_proven": True,
        "pseudo_transport_audit_only": True,
        "consumed_test": True,
        "may_feed": False,
    }
    return PolicyReplayRuntimeResult(
        MappingProxyType(descriptors),
        MappingProxyType(reference_blocks),
        MappingProxyType(transport_screens),
        MappingProxyType(matches),
        match_hash,
        MappingProxyType(influences),
        MappingProxyType(target_candidates),
        MappingProxyType(pseudo_candidates),
        MappingProxyType(replays),
        MappingProxyType(donor_envelopes),
        MappingProxyType(calibrations),
        MappingProxyType(matched_annotations),
        MappingProxyType(decisions),
        MappingProxyType(decisions),
        MappingProxyType(final),
        MappingProxyType(menu),
        transport_seal,
        transport_seal.transport_hash,
        canonical_hash(runtime_payload),
    )


def _build_route_transport_runtime(
    *,
    endpoint_by_center: Mapping[str, OuterEndpointProducts],
    donor_endpoint_products: Mapping[tuple[str, str], OuterEndpointProducts],
    donor_runtime: DonorRuntimeResult,
    prepared_by_center: Mapping[str, PreparedCenter],
    strict_canonical_topology: bool,
) -> tuple[
    dict[object, SupportConditionedCaseDescriptor],
    dict[object, TransportReferenceBlockSummary],
    dict[object, RouteTransportScreen],
    TransportRuntimeSeal,
]:
    descriptors: dict[object, SupportConditionedCaseDescriptor] = {}
    blocks: dict[object, TransportReferenceBlockSummary] = {}

    def add_descriptor(
        key: object,
        products: OuterEndpointProducts,
        *,
        outer: str,
        endpoint_center: str,
        case_id: str,
        role: str,
        excluded_centers: tuple[str, ...],
    ) -> None:
        predictions = {row.case_id: row for row in products.predictions}
        states = dict(products.states)
        state = states[case_id]
        seeds = reconstruct_case_endpoint_seed_probabilities(
            prepared_by_center[endpoint_center],
            state,
            evaluation_case_id=case_id,
        )[PORTFOLIO_METHOD_ID]
        descriptors[key] = build_case_transport_descriptor(
            predictions[case_id],
            portfolio_seed_probabilities=seeds,
            lineage=RouteTransportLineage(
                outer,
                endpoint_center,
                case_id,
                role,
                tuple(state.support_case_ids),
                excluded_centers,
                state.state_hash,
            ),
        )

    for outer in CENTERS:
        target_products = endpoint_by_center[outer]
        for prediction in target_products.predictions:
            add_descriptor(
                TargetRouteKey(outer, prediction.case_id),
                target_products,
                outer=outer,
                endpoint_center=outer,
                case_id=prediction.case_id,
                role="target_candidate",
                excluded_centers=(outer,),
            )
        for reference in CENTERS:
            if reference == outer:
                continue
            products = donor_endpoint_products[(outer, reference)]
            keys = []
            for prediction in products.predictions:
                key = TargetReferenceKey(outer, reference, prediction.case_id)
                add_descriptor(
                    key,
                    products,
                    outer=outer,
                    endpoint_center=reference,
                    case_id=prediction.case_id,
                    role="target_reference",
                    excluded_centers=(outer, reference),
                )
                keys.append(key)
            blocks[("target", outer, reference)] = build_reference_block_summary(
                tuple(descriptors[key] for key in keys),
                outer_center=outer,
                donor_center=None,
                reference_center=reference,
            )
            for prediction in products.predictions:
                key = PseudoRouteKey(outer, reference, prediction.case_id)
                add_descriptor(
                    key,
                    products,
                    outer=outer,
                    endpoint_center=reference,
                    case_id=prediction.case_id,
                    role="pseudo_candidate",
                    excluded_centers=(outer, reference),
                )

        for donor in CENTERS:
            if donor == outer:
                continue
            for reference in CENTERS:
                if reference in {outer, donor}:
                    continue
                products = donor_runtime.pseudo_donor_endpoint_products[
                    (outer, donor, reference)
                ]
                keys = []
                for prediction in products.predictions:
                    key = PseudoReferenceKey(
                        outer, donor, reference, prediction.case_id
                    )
                    add_descriptor(
                        key,
                        products,
                        outer=outer,
                        endpoint_center=reference,
                        case_id=prediction.case_id,
                        role="pseudo_reference",
                        excluded_centers=(outer, donor, reference),
                    )
                    keys.append(key)
                blocks[("pseudo", outer, donor, reference)] = (
                    build_reference_block_summary(
                        tuple(descriptors[key] for key in keys),
                        outer_center=outer,
                        donor_center=donor,
                        reference_center=reference,
                    )
                )

    screens: dict[object, RouteTransportScreen] = {}
    for outer in CENTERS:
        target_refs = tuple(
            descriptors[key]
            for key in descriptors
            if isinstance(key, TargetReferenceKey)
            and key.outer_center == outer
        )
        target_blocks = tuple(
            blocks[("target", outer, reference)]
            for reference in CENTERS
            if reference != outer
        )
        for key in tuple(
            observed
            for observed in descriptors
            if isinstance(observed, TargetRouteKey)
            and observed.outer_center == outer
        ):
            _summary, screen = evaluate_transport_screen(
                descriptors[key],
                target_refs,
                role="target",
                reference_blocks=target_blocks,
            )
            screens[key] = screen
        for donor in CENTERS:
            if donor == outer:
                continue
            pseudo_refs = tuple(
                descriptors[key]
                for key in descriptors
                if isinstance(key, PseudoReferenceKey)
                and key.outer_center == outer
                and key.donor_center == donor
            )
            pseudo_blocks = tuple(
                blocks[("pseudo", outer, donor, reference)]
                for reference in CENTERS
                if reference not in {outer, donor}
            )
            for key in tuple(
                observed
                for observed in descriptors
                if isinstance(observed, PseudoRouteKey)
                and observed.outer_center == outer
                and observed.donor_center == donor
            ):
                _summary, screen = evaluate_transport_screen(
                    descriptors[key],
                    pseudo_refs,
                    role="pseudo_audit",
                    reference_blocks=pseudo_blocks,
                )
                screens[key] = screen

    seal = seal_transport_runtime(
        descriptors,
        blocks,
        screens,
        strict_canonical_topology=strict_canonical_topology,
    )
    if strict_canonical_topology and len(descriptors) != EXPECTED_ROLE_BOUND_TRANSPORT_DESCRIPTOR_COUNT:
        raise ProtocolError("PCSI-RACR role-bound descriptor count drifted.")
    return descriptors, blocks, screens, seal


def _build_descriptor_matches(
    descriptors: Mapping[object, SupportConditionedCaseDescriptor],
    screens: Mapping[object, RouteTransportScreen],
) -> tuple[dict[tuple[str, str, str], str], str]:
    matches: dict[tuple[str, str, str], str] = {}
    for outer in CENTERS:
        target_keys = tuple(
            key
            for key in descriptors
            if isinstance(key, TargetRouteKey) and key.outer_center == outer
        )
        for target_key in target_keys:
            target = descriptors[target_key]
            for donor in CENTERS:
                if donor == outer:
                    continue
                pseudo_keys = tuple(
                    key
                    for key in descriptors
                    if isinstance(key, PseudoRouteKey)
                    and key.outer_center == outer
                    and key.donor_center == donor
                )
                ranked = []
                for pseudo_key in pseudo_keys:
                    pseudo = descriptors[pseudo_key]
                    scale = np.asarray(
                        screens[pseudo_key].scale, dtype=np.float64
                    )
                    distance = float(
                        np.max(
                            np.abs(
                                np.asarray(target.feature_values, dtype=np.float64)
                                - np.asarray(pseudo.feature_values, dtype=np.float64)
                            )
                            / scale
                        )
                    )
                    ranked.append(
                        (
                            distance,
                            pseudo_key.case_id,
                            pseudo.descriptor_hash,
                        )
                    )
                matches[(outer, target_key.case_id, donor)] = min(ranked)[1]
    payload = {
        "schema_version": "fixed_bank_pcsi_racr_descriptor_match_table_v1",
        "rows": [
            [outer, case_id, donor, matches[(outer, case_id, donor)]]
            for outer, case_id, donor in sorted(matches)
        ],
        "sealed_before_pseudo_labels": True,
        "unscored_annotation_only": True,
    }
    return matches, canonical_hash(payload)


def _build_matched_annotations(
    *,
    matches: Mapping[tuple[str, str, str], str],
    match_hash: str,
    replays: Mapping[tuple[str, str, str, str], PseudoCaseReplay],
) -> dict[tuple[str, str], DescriptorMatchedAnnotation]:
    output = {}
    for outer in CENTERS:
        target_cases = tuple(
            sorted(
                case_id
                for observed_outer, case_id, _donor in matches
                if observed_outer == outer
            )
        )
        target_cases = tuple(dict.fromkeys(target_cases))
        for case_id in target_cases:
            selected = tuple(
                (
                    donor,
                    matches[(outer, case_id, donor)],
                )
                for donor in CENTERS
                if donor != outer
            )
            residuals = tuple(
                replays[
                    (
                        PROJECTION_GEOMETRY_ID,
                        outer,
                        donor,
                        matched_case,
                    )
                ].overprediction_residual
                for donor, matched_case in selected
            )
            margin = tuple(
                max(0.0, max(row[index] for row in residuals))
                for index in range(3)
            )
            output[(outer, case_id)] = DescriptorMatchedAnnotation(
                PROJECTION_GEOMETRY_ID,
                outer,
                case_id,
                selected,
                margin,
                match_hash,
            )
    return output


def _score_influences(
    descriptors: Sequence[object],
    *,
    posterior: TargetLocalPosteriorPrediction,
    model: TargetLocalPosteriorModel,
) -> tuple[InfluencePrediction, ...]:
    rows = tuple(descriptors)
    if (
        not rows
        or {str(getattr(row, "direction")) for row in rows} != set(DIRECTION_IDS)
        or any(
            str(getattr(row, "target_center")) != posterior.target_center
            or str(getattr(row, "case_id")) != posterior.case_id
            for row in rows
        )
        or model.model_hash != posterior.model_hash
        or model.target_center != posterior.target_center
        or model.held_case_id != posterior.case_id
    ):
        raise ProtocolError("PCSI-RACR influence surface/model binding drifted.")
    eta = dict(zip(posterior.sample_ids, posterior.natural_probabilities, strict=True))
    output = []
    for descriptor in rows:
        crossing_ids = tuple(getattr(descriptor, "crossing_sample_ids"))
        if any(sample_id not in eta for sample_id in crossing_ids):
            raise ProtocolError("PCSI-RACR influence crossing escaped held-case rows.")
        direction = str(getattr(descriptor, "direction"))
        representative = str(
            getattr(
                descriptor,
                "representative",
                getattr(descriptor, "alternative", ""),
            )
        )
        sign = 1.0 if direction == DIRECTION_IDS[0] else -1.0
        score = 0.5 * sum(
            sign
            * (
                eta[sample_id] / model.support_n_positive
                - (1.0 - eta[sample_id]) / model.support_n_negative
            )
            for sample_id in crossing_ids
        )
        output.append(
            InfluencePrediction(
                str(getattr(descriptor, "descriptor_hash")),
                posterior.target_center,
                posterior.case_id,
                representative,
                direction,
                len(crossing_ids),
                float(score),
                posterior.prediction_hash,
            )
        )
    return tuple(output)


def _rows_by_case(rows: Sequence[object]) -> Mapping[str, tuple[object, ...]]:
    cases = tuple(dict.fromkeys(str(getattr(row, "case_id")) for row in rows))
    return MappingProxyType(
        {
            case: tuple(
                row for row in rows if str(getattr(row, "case_id")) == case
            )
            for case in cases
        }
    )


def _prediction_rows_by_case(
    descriptors: Sequence[object], predictions: Sequence[object]
) -> Mapping[str, tuple[object, ...]]:
    prediction_by_hash = {
        str(getattr(row, "descriptor_hash")): row for row in predictions
    }
    if set(prediction_by_hash) != {
        str(getattr(row, "descriptor_hash")) for row in descriptors
    }:
        raise ProtocolError("PCSI-RACR descriptor/prediction surface drifted.")
    grouped = _rows_by_case(descriptors)
    return MappingProxyType(
        {
            case: tuple(
                prediction_by_hash[str(getattr(row, "descriptor_hash"))]
                for row in rows
            )
            for case, rows in grouped.items()
        }
    )


__all__ = ("PolicyReplayRuntimeResult", "build_policy_replay_runtime")
