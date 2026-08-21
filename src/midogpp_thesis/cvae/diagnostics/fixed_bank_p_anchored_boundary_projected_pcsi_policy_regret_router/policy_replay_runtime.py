"""Selection, sealed pseudo replay, and whole-policy authorization runtime."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    BLOCKED_CONTROL_METHOD_ID,
    BLOCKED_FINGERPRINT_CONTROL_ID,
    CENTERS,
    COMPOSED_POLICY_IDS,
    DIRECTION_IDS,
    EXPECTED_POLICY_REPLAY_COUNT,
    LEGACY_CONTROL_METHOD_ID,
    LEGACY_GEOMETRY_ID,
    PORTFOLIO_METHOD_ID,
    PRIMARY_FINGERPRINT_CONTROL_ID,
    PRIMARY_METHOD_ID,
    PROJECTION_GEOMETRY_ID,
    PROJECTED_NO_PARC_METHOD_ID,
    UNPROJECTED_GEOMETRY_ID,
    UNPROJECTED_PARC_METHOD_ID,
)
from .contracts import EndpointCasePrediction
from .donor_runtime import DonorRuntimeResult, PARC_GEOMETRIES
from .endpoint_reconstruction import (
    PreparedCenter,
    reconstruct_case_endpoint_seed_probabilities,
)
from .hashing import canonical_hash
from .label_capabilities import PARC_GEOMETRY_POLICY, PCSIPARCLabelFirewall
from .outer_endpoint_runtime import OuterEndpointProducts
from .policy_regret import (
    CenterCandidatePolicy,
    PolicyAuthorization,
    WholePolicyReplay,
    authorize_center_policy,
    build_center_candidate_policy,
    build_whole_policy_replay,
)
from .policy_selection import (
    CaseCandidatePolicy,
    DirectionalClassDecision,
    FinalCasePolicyPrediction,
    finalize_case_policy,
    select_and_compose_case_policy,
)
from .projection import ActionEquivalenceClass, build_action_equivalence_classes
from .projection_lattice import as_binary32, canonical_bytes
from .sample_influence_contracts import (
    InfluencePrediction,
    TargetLocalPosteriorModel,
    TargetLocalPosteriorPrediction,
)
from .sealing import seal_policy_menu
from .selection import select_directional_actions
from .transport import (
    CenterTransportDescriptor,
    TransportEndpointLineage,
    TransportRuntimeSeal,
    TransportScreen,
    build_center_transport_descriptor,
    evaluate_transport_screen,
    seal_transport_runtime,
)
from .utility_contracts import UtilityDescriptor, UtilityPrediction


@dataclass(frozen=True)
class PolicyReplayRuntimeResult:
    transport_descriptors_by_outer_candidate: Mapping[
        tuple[str, str], CenterTransportDescriptor
    ]
    transport_screens: Mapping[tuple[str, str | None], TransportScreen]
    target_influences_by_policy_center: Mapping[
        tuple[str, str], tuple[InfluencePrediction, ...]
    ]
    target_candidate_policies: Mapping[
        tuple[str, str], CenterCandidatePolicy
    ]
    pseudo_candidate_policies: Mapping[
        tuple[str, str, str], CenterCandidatePolicy
    ]
    replays: Mapping[tuple[str, str, str], WholePolicyReplay]
    authorizations: Mapping[tuple[str, str], PolicyAuthorization]
    final_predictions_by_policy: Mapping[
        str, tuple[FinalCasePolicyPrediction, ...]
    ]
    policy_menu_seal: Mapping[str, object]
    transport_seal: TransportRuntimeSeal
    transport_hash: str
    runtime_hash: str

    def summary_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_parc_policy_replay_runtime_v2",
            "transport_descriptor_count": len(
                self.transport_descriptors_by_outer_candidate
            ),
            "transport_screen_count": len(self.transport_screens),
            "target_center_policy_count": len(self.target_candidate_policies),
            "double_exclusion_policy_count": len(self.pseudo_candidate_policies),
            "policy_replay_count": len(self.replays),
            "authorization_count": len(self.authorizations),
            "final_case_prediction_count": sum(
                len(rows) for rows in self.final_predictions_by_policy.values()
            ),
            "transport_hash": self.transport_hash,
            "transport_seal": self.transport_seal.to_payload(),
            "transport_protocol_status": (
                "BLOCKED_IDENTITY_LEVEL_ROUTE_FEEDBACK"
            ),
            "transport_authorization_valid": False,
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
    label_firewall: PCSIPARCLabelFirewall,
    strict_canonical_topology: bool = True,
) -> PolicyReplayRuntimeResult:
    """Replay all 144 H/J policies after their case and aggregate seals."""

    expected_pairs = {
        (outer, pseudo)
        for outer in CENTERS
        for pseudo in CENTERS
        if pseudo != outer
    }
    endpoint_by_center = {row.target_center: row for row in endpoint_products}
    if (
        set(predictions_by_center) != set(CENTERS)
        or set(endpoint_by_center) != set(CENTERS)
        or set(donor_endpoint_products) != expected_pairs
        or set(prepared_by_center) != set(CENTERS)
        or set(target_posterior_models_by_control)
        != {PRIMARY_FINGERPRINT_CONTROL_ID, BLOCKED_FINGERPRINT_CONTROL_ID}
        or set(target_posterior_predictions_by_control)
        != {PRIMARY_FINGERPRINT_CONTROL_ID, BLOCKED_FINGERPRINT_CONTROL_ID}
    ):
        raise ProtocolError("PCSI-PARC policy runtime input matrix drifted.")

    model_index = {
        control: {
            (row.target_center, row.held_case_id): row
            for row in target_posterior_models_by_control[control]
        }
        for control in target_posterior_models_by_control
    }
    posterior_index = {
        control: {
            (row.target_center, row.case_id): row
            for row in target_posterior_predictions_by_control[control]
        }
        for control in target_posterior_predictions_by_control
    }

    transport_descriptors, transport_screens, transport_seal = (
        _build_transport_runtime(
            endpoint_by_center=endpoint_by_center,
            donor_endpoint_products=donor_endpoint_products,
            prepared_by_center=prepared_by_center,
        )
    )
    transport_hash = transport_seal.transport_hash

    influences: dict[tuple[str, str], tuple[InfluencePrediction, ...]] = {}
    target_cases: dict[tuple[str, str], tuple[CaseCandidatePolicy, ...]] = {}

    for center in CENTERS:
        projected = donor_runtime.geometry_results[PROJECTION_GEOMETRY_ID]
        raw = donor_runtime.geometry_results[UNPROJECTED_GEOMETRY_ID]
        for policy_id, geometry, control in (
            (
                PRIMARY_METHOD_ID,
                projected,
                PRIMARY_FINGERPRINT_CONTROL_ID,
            ),
            (
                PROJECTED_NO_PARC_METHOD_ID,
                projected,
                PRIMARY_FINGERPRINT_CONTROL_ID,
            ),
            (
                BLOCKED_CONTROL_METHOD_ID,
                projected,
                BLOCKED_FINGERPRINT_CONTROL_ID,
            ),
            (
                UNPROJECTED_PARC_METHOD_ID,
                raw,
                PRIMARY_FINGERPRINT_CONTROL_ID,
            ),
        ):
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
            policy_influences: list[InfluencePrediction] = []
            policies: list[CaseCandidatePolicy] = []
            for endpoint in predictions_by_center[center]:
                route = center, endpoint.case_id
                case_influences = _score_influences(
                    descriptors_by_case[endpoint.case_id],
                    posterior=posterior_index[control][route],
                    model=model_index[control][route],
                )
                policy_influences.extend(case_influences)
                policies.append(
                    select_and_compose_case_policy(
                        endpoint,
                        actions_by_case[endpoint.case_id],
                        descriptors_by_case[endpoint.case_id],
                        case_influences,
                        utilities_by_case[endpoint.case_id],
                        policy_id=policy_id,
                        require_positive_bacc_prediction=False,
                    )
                )
            influences[(policy_id, center)] = tuple(policy_influences)
            target_cases[(policy_id, center)] = tuple(policies)

        legacy_descriptors = donor_runtime.legacy.descriptors_by_center[center]
        legacy_descriptors_by_case = _rows_by_case(legacy_descriptors)
        legacy_utilities_by_case = _prediction_rows_by_case(
            legacy_descriptors,
            donor_runtime.legacy.predictions_by_center[center],
        )
        legacy_influences: list[InfluencePrediction] = []
        legacy_cases: list[CaseCandidatePolicy] = []
        for endpoint in predictions_by_center[center]:
            route = center, endpoint.case_id
            case_descriptors = legacy_descriptors_by_case[endpoint.case_id]
            case_influences = _score_influences(
                case_descriptors,
                posterior=posterior_index[PRIMARY_FINGERPRINT_CONTROL_ID][route],
                model=model_index[PRIMARY_FINGERPRINT_CONTROL_ID][route],
            )
            legacy_influences.extend(case_influences)
            legacy_cases.append(
                _build_legacy_case_policy(
                    endpoint,
                    case_descriptors,
                    case_influences,
                    legacy_utilities_by_case[endpoint.case_id],
                )
            )
        influences[(LEGACY_CONTROL_METHOD_ID, center)] = tuple(legacy_influences)
        target_cases[(LEGACY_CONTROL_METHOD_ID, center)] = tuple(legacy_cases)

    target_policies: dict[tuple[str, str], CenterCandidatePolicy] = {}
    for policy_id in COMPOSED_POLICY_IDS:
        for center in CENTERS:
            cases = target_cases[(policy_id, center)]
            for case in cases:
                label_firewall.record_target_case_policy_seal(
                    center, case.case_id, policy_id, case.policy_hash
                )
            target = build_center_candidate_policy(cases)
            label_firewall.record_target_center_policy_seal(
                center, policy_id, target.policy_seal_hash
            )
            target_policies[(policy_id, center)] = target

    pseudo_policies: dict[tuple[str, str, str], CenterCandidatePolicy] = {}
    # First seal the complete 2 x 72 pseudo-policy matrix.  No pseudo evaluation
    # label may be decoded while any later H/J decision remains mutable.
    for geometry_id in PARC_GEOMETRIES:
        geometry = donor_runtime.geometry_results[geometry_id]
        policy_id = PARC_GEOMETRY_POLICY[geometry_id]
        for outer in CENTERS:
            for pseudo in CENTERS:
                if pseudo == outer:
                    continue
                pair = outer, pseudo
                descriptors_by_case = _rows_by_case(
                    geometry.pseudo_descriptors_by_pair[pair]
                )
                actions_by_case = _rows_by_case(
                    geometry.pseudo_actions_by_pair[pair]
                )
                utilities_by_case = _prediction_rows_by_case(
                    geometry.pseudo_descriptors_by_pair[pair],
                    geometry.pseudo_predictions_by_pair[pair],
                )
                cases: list[CaseCandidatePolicy] = []
                for endpoint in donor_endpoint_products[pair].predictions:
                    route = pseudo, endpoint.case_id
                    case_influences = _score_influences(
                        descriptors_by_case[endpoint.case_id],
                        posterior=posterior_index[
                            PRIMARY_FINGERPRINT_CONTROL_ID
                        ][route],
                        model=model_index[PRIMARY_FINGERPRINT_CONTROL_ID][route],
                    )
                    case = select_and_compose_case_policy(
                        endpoint,
                        actions_by_case[endpoint.case_id],
                        descriptors_by_case[endpoint.case_id],
                        case_influences,
                        utilities_by_case[endpoint.case_id],
                        policy_id=policy_id,
                        require_positive_bacc_prediction=False,
                    )
                    label_firewall.record_pseudo_case_policy_seal(
                        outer,
                        pseudo,
                        geometry_id,
                        case.case_id,
                        case.policy_hash,
                    )
                    cases.append(case)
                pseudo_policy = build_center_candidate_policy(cases)
                label_firewall.record_pseudo_policy_seal(
                    outer,
                    pseudo,
                    geometry_id,
                    pseudo_policy.policy_seal_hash,
                )
                key = geometry_id, outer, pseudo
                pseudo_policies[key] = pseudo_policy

    # Re-seal the complete immutable transport matrix immediately before the
    # first pseudo-evaluation capability can open.  Terminal labels remain
    # unavailable until the later aggregate preterminal seal.
    if (
        seal_transport_runtime(transport_descriptors, transport_screens)
        != transport_seal
    ):
        raise ProtocolError("PCSI-PARC transport changed before pseudo replay.")

    replays: dict[tuple[str, str, str], WholePolicyReplay] = {}
    for geometry_id in PARC_GEOMETRIES:
        for outer in CENTERS:
            for pseudo in CENTERS:
                if pseudo == outer:
                    continue
                pair = outer, pseudo
                key = geometry_id, outer, pseudo
                pseudo_policy = pseudo_policies[key]
                evaluation_labels = label_firewall.open_pseudo_evaluation_labels(
                    outer,
                    pseudo,
                    geometry_id,
                    policy_seal_hash=pseudo_policy.policy_seal_hash,
                )
                replay = build_whole_policy_replay(
                    outer_target_center=outer,
                    pseudo_policy=pseudo_policy,
                    endpoints=donor_endpoint_products[pair].predictions,
                    evaluation_labels=evaluation_labels,
                    transport_screen=transport_screens[pair],
                )
                label_firewall.record_policy_replay_seal(
                    outer, pseudo, geometry_id, replay.replay_hash
                )
                replays[key] = replay

    if strict_canonical_topology and len(replays) != EXPECTED_POLICY_REPLAY_COUNT:
        raise ProtocolError("PCSI-PARC whole-policy replay workload drifted.")

    authorizations: dict[tuple[str, str], PolicyAuthorization] = {}
    for geometry_id, policy_id in PARC_GEOMETRY_POLICY.items():
        for center in CENTERS:
            rows = tuple(
                replays[(geometry_id, center, pseudo)]
                for pseudo in CENTERS
                if pseudo != center
            )
            authorizations[(policy_id, center)] = authorize_center_policy(
                target_policies[(policy_id, center)],
                rows,
                target_transport_screen=transport_screens[(center, None)],
                pseudo_transport_screens={
                    pseudo: transport_screens[(center, pseudo)]
                    for pseudo in CENTERS
                    if pseudo != center
                },
            )

    final: dict[str, tuple[FinalCasePolicyPrediction, ...]] = {}
    for policy_id in COMPOSED_POLICY_IDS:
        predictions: list[FinalCasePolicyPrediction] = []
        for center in CENTERS:
            target = target_policies[(policy_id, center)]
            if (policy_id, center) in authorizations:
                authorization = authorizations[(policy_id, center)]
                authorized = authorization.authorized
                authorization_hash = authorization.authorization_hash
            else:
                authorized = True
                authorization_hash = canonical_hash(
                    {
                        "schema_version": "fixed_bank_pcsi_parc_control_authorization_v1",
                        "policy_id": policy_id,
                        "target_center": center,
                        "authorized": True,
                        "uses_policy_regret": False,
                        "terminal_labels_used": False,
                    }
                )
            endpoint_by_case = {
                row.case_id: row for row in predictions_by_center[center]
            }
            predictions.extend(
                finalize_case_policy(
                    case,
                    endpoint_by_case[case.case_id],
                    authorized=authorized,
                    authorization_hash=authorization_hash,
                )
                for case in target.cases
            )
        final[policy_id] = tuple(predictions)

    barrier = label_firewall.decision_barrier_payload()
    menu = seal_policy_menu(
        tuple(
            target_policies[(policy, center)]
            for policy in COMPOSED_POLICY_IDS
            for center in CENTERS
        ),
        decision_barrier_hash=str(barrier["decision_barrier_hash"]),
    )
    payload = {
        "schema_version": "fixed_bank_pcsi_parc_policy_replay_runtime_v2",
        "transport_hash": transport_hash,
        "transport_seal_hash": transport_seal.transport_hash,
        "transport_protocol_status": "BLOCKED_IDENTITY_LEVEL_ROUTE_FEEDBACK",
        "transport_authorization_valid": False,
        "target_policy_hash": canonical_hash(
            [
                target_policies[(policy, center)].policy_seal_hash
                for policy in COMPOSED_POLICY_IDS
                for center in CENTERS
            ]
        ),
        "pseudo_policy_hash": canonical_hash(
            [
                pseudo_policies[(geometry, outer, pseudo)].policy_seal_hash
                for geometry in PARC_GEOMETRIES
                for outer in CENTERS
                for pseudo in CENTERS
                if pseudo != outer
            ]
        ),
        "policy_replay_hash": canonical_hash(
            [
                replays[(geometry, outer, pseudo)].replay_hash
                for geometry in PARC_GEOMETRIES
                for outer in CENTERS
                for pseudo in CENTERS
                if pseudo != outer
            ]
        ),
        "authorization_hash": canonical_hash(
            [
                authorizations[(policy, center)].authorization_hash
                for policy in (PRIMARY_METHOD_ID, UNPROJECTED_PARC_METHOD_ID)
                for center in CENTERS
            ]
        ),
        "final_prediction_hash": canonical_hash(
            [
                row.prediction_hash
                for policy in COMPOSED_POLICY_IDS
                for row in final[policy]
            ]
        ),
        "policy_menu_seal_hash": menu["policy_menu_seal_hash"],
        "terminal_labels_used": False,
    }
    return PolicyReplayRuntimeResult(
        MappingProxyType(transport_descriptors),
        MappingProxyType(transport_screens),
        MappingProxyType(influences),
        MappingProxyType(target_policies),
        MappingProxyType(pseudo_policies),
        MappingProxyType(replays),
        MappingProxyType(authorizations),
        MappingProxyType(final),
        MappingProxyType(dict(menu)),
        transport_seal,
        transport_hash,
        canonical_hash(payload),
    )


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
        raise ProtocolError("PCSI-PARC influence surface/model binding drifted.")
    eta = dict(zip(posterior.sample_ids, posterior.natural_probabilities, strict=True))
    output: list[InfluencePrediction] = []
    for descriptor in rows:
        crossing_ids = tuple(getattr(descriptor, "crossing_sample_ids"))
        if any(sample_id not in eta for sample_id in crossing_ids):
            raise ProtocolError("PCSI-PARC influence crossing escaped held-case rows.")
        direction = str(getattr(descriptor, "direction"))
        representative = str(
            getattr(descriptor, "representative", getattr(descriptor, "alternative", ""))
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


def _build_legacy_case_policy(
    endpoint: EndpointCasePrediction,
    descriptors: Sequence[UtilityDescriptor],
    influences: Sequence[InfluencePrediction],
    utilities: Sequence[UtilityPrediction],
) -> CaseCandidatePolicy:
    decisions = select_directional_actions(
        descriptors,
        influences,
        utilities,
        policy_id=LEGACY_CONTROL_METHOD_ID,
    )
    actions = build_action_equivalence_classes(
        endpoint,
        geometry_id=LEGACY_GEOMETRY_ID,
        collapse_equivalent=False,
    )
    action_by_key = {(row.representative, row.direction): row for row in actions}
    descriptor_by_key = {(row.alternative, row.direction): row for row in descriptors}
    influence_by_hash = {row.descriptor_hash: row for row in influences}
    utility_by_hash = {row.descriptor_hash: row for row in utilities}
    output = as_binary32(endpoint.probabilities[PORTFOLIO_METHOD_ID], name="legacy P").copy()
    policy_decisions: list[DirectionalClassDecision] = []
    total = np.zeros(3, dtype=np.float64)
    for decision in decisions:
        if decision.selected_alternative == PORTFOLIO_METHOD_ID:
            selected_hash = None
            vector = (0.0, 0.0, 0.0)
        else:
            action = action_by_key[
                (decision.selected_alternative, decision.direction)
            ]
            selected_hash = action.action_hash
            emitted = as_binary32(action.probabilities, name="legacy emitted action")
            portfolio = as_binary32(
                endpoint.probabilities[PORTFOLIO_METHOD_ID], name="legacy P"
            )
            mask = emitted.view(np.uint32) != portfolio.view(np.uint32)
            output[mask] = emitted[mask]
            descriptor = descriptor_by_key[
                (decision.selected_alternative, decision.direction)
            ]
            utility = utility_by_hash[descriptor.descriptor_hash]
            vector = (
                utility.robust("bacc_contribution_delta"),
                -utility.robust("brier_contribution_delta"),
                -utility.robust("log_loss_contribution_delta"),
            )
            total += np.asarray(vector, dtype=np.float64)
        direction_rows = tuple(
            sorted(
                (row for row in descriptors if row.direction == decision.direction),
                key=lambda row: row.alternative,
            )
        )
        policy_decisions.append(
            DirectionalClassDecision(
                endpoint.center,
                endpoint.case_id,
                LEGACY_CONTROL_METHOD_ID,
                decision.direction,
                selected_hash,
                decision.selected_alternative,
                decision.selected_score,
                tuple(float(value) for value in vector),
                tuple(
                    digest
                    for row in direction_rows
                    for digest in (
                        row.descriptor_hash,
                        influence_by_hash[row.descriptor_hash].influence_hash,
                        utility_by_hash[row.descriptor_hash].prediction_hash,
                    )
                ),
            )
        )
    return CaseCandidatePolicy(
        endpoint.center,
        endpoint.case_id,
        LEGACY_CONTROL_METHOD_ID,
        LEGACY_GEOMETRY_ID,
        endpoint.sample_ids,
        tuple(float(value) for value in output),
        tuple(policy_decisions),
        tuple(float(value) for value in total),
        endpoint.prediction_hash,
        hashlib.sha256(canonical_bytes(output)).hexdigest(),
    )


def _build_transport_runtime(
    *,
    endpoint_by_center: Mapping[str, OuterEndpointProducts],
    donor_endpoint_products: Mapping[tuple[str, str], OuterEndpointProducts],
    prepared_by_center: Mapping[str, PreparedCenter],
) -> tuple[
    dict[tuple[str, str], CenterTransportDescriptor],
    dict[tuple[str, str | None], TransportScreen],
    TransportRuntimeSeal,
]:
    descriptors: dict[tuple[str, str], CenterTransportDescriptor] = {}
    for outer in CENTERS:
        for candidate in CENTERS:
            products = (
                endpoint_by_center[outer]
                if candidate == outer
                else donor_endpoint_products[(outer, candidate)]
            )
            states = dict(products.states)
            seeds = {
                case_id: reconstruct_case_endpoint_seed_probabilities(
                    prepared_by_center[candidate],
                    states[case_id],
                    evaluation_case_id=case_id,
                )[PORTFOLIO_METHOD_ID]
                for case_id in states
            }
            descriptors[(outer, candidate)] = build_center_transport_descriptor(
                products.predictions,
                portfolio_seed_probabilities_by_case=seeds,
                lineage=TransportEndpointLineage(
                    outer,
                    candidate,
                    canonical_hash(
                        [list(row) for row in products.state_hashes]
                    ),
                ),
            )
    screens: dict[tuple[str, str | None], TransportScreen] = {}
    for outer in CENTERS:
        screens[(outer, None)] = evaluate_transport_screen(
            descriptors[(outer, outer)],
            tuple(
                descriptors[(outer, center)]
                for center in CENTERS
                if center != outer
            ),
        )
        for pseudo in CENTERS:
            if pseudo == outer:
                continue
            screens[(outer, pseudo)] = evaluate_transport_screen(
                descriptors[(outer, pseudo)],
                tuple(
                    descriptors[(outer, center)]
                    for center in CENTERS
                    if center not in {outer, pseudo}
                ),
            )
    return descriptors, screens, seal_transport_runtime(descriptors, screens)


def _rows_by_case(rows: Sequence[object]) -> Mapping[str, tuple[object, ...]]:
    cases = tuple(dict.fromkeys(str(getattr(row, "case_id")) for row in rows))
    return MappingProxyType(
        {
            case: tuple(row for row in rows if str(getattr(row, "case_id")) == case)
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
        raise ProtocolError("PCSI-PARC descriptor/prediction surface drifted.")
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
