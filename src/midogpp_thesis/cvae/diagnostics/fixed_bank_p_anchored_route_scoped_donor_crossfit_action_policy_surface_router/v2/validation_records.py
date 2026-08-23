"""Semantic reconstruction of persisted P-DCAPS preterminal DTO records."""

from __future__ import annotations

from typing import Mapping, TypeVar

from ....expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ....protocol import ProtocolError
from ..action_surface import (
    ActionCalibrationFamilies,
    ActionCalibrationModel,
    ActionKey,
    ActionStratumReliability,
    CalibratedActionSelection,
)
from ..contracts import FavorableUtility, RouteKey
from ..engine import OuterActionPolicyResult, RouteActionDecision
from ..identity import (
    CYCLIC_METHOD_ID,
    LEGACY_METHOD_ID,
    METHOD_MENU,
    PRIMARY_METHOD_ID,
    canonical_hash,
    require_sha256,
)
from ..policy_surface import (
    CalibratedPrefixCell,
    NestedPolicyCalibration,
    PolicyAction,
    PolicyCalibration,
    PolicyCalibrationFamilies,
    PolicyEnvelope,
    PolicyOOFResidual,
    PolicyRidgeModel,
    PolicySelection,
    PolicySurfaceProvenance,
    PrefixCell,
    PrefixSurface,
)
from ..preterminal import PreterminalOutputHashes
from ..target_local_runtime import POSTERIOR_CONTROL_IDS


_T = TypeVar("_T")


def validate_persisted_preterminal_records(
    science: Mapping[str, object],
    output: PreterminalOutputHashes,
) -> dict[str, object]:
    """Rebuild every persisted top-level science seal without refitting."""

    surface = _mapping(science, "surface_set")
    physical_hash = require_sha256(
        surface.get("physical_surface_hash"), "persisted physical surface"
    )
    surface_seals = {
        str(control): require_sha256(value, "persisted action surface")
        for control, value in _pairs(surface, "control_surface_seals")
    }
    if tuple(surface_seals) != POSTERIOR_CONTROL_IDS:
        raise ProtocolError("P-DCAPS v2 persisted control surfaces drifted.")

    expected_results = {
        (control, center): digest
        for control, center, digest in output.control_result_hashes
    }
    (
        identity_cases,
        identity_target_count,
        identity_pseudo_count,
    ) = _validate_outer_results(
        _mapping_rows(science, "identity_results"),
        control_id=POSTERIOR_CONTROL_IDS[0],
        centers=output.centers,
        expected_results=expected_results,
        expected_surface_seal=surface_seals[POSTERIOR_CONTROL_IDS[0]],
        expected_physical_hash=physical_hash,
    )
    (
        cyclic_cases,
        cyclic_target_count,
        cyclic_pseudo_count,
    ) = _validate_outer_results(
        _mapping_rows(science, "cyclic_results"),
        control_id=POSTERIOR_CONTROL_IDS[1],
        centers=output.centers,
        expected_results=expected_results,
        expected_surface_seal=surface_seals[POSTERIOR_CONTROL_IDS[1]],
        expected_physical_hash=physical_hash,
    )
    if cyclic_cases != identity_cases:
        raise ProtocolError("P-DCAPS v2 control route inventories drifted.")

    expected_legacy = {
        (control, center): digest
        for control, center, digest in output.legacy_control_seal_hashes
    }
    identity_legacy = _validate_legacy_controls(
        _mapping_rows(science, "identity_legacy_controls"),
        control_id=POSTERIOR_CONTROL_IDS[0],
        centers=output.centers,
        expected_results=expected_results,
        expected_legacy=expected_legacy,
        expected_surface_seal=surface_seals[POSTERIOR_CONTROL_IDS[0]],
        expected_physical_hash=physical_hash,
    )
    cyclic_legacy = _validate_legacy_controls(
        _mapping_rows(science, "cyclic_legacy_controls"),
        control_id=POSTERIOR_CONTROL_IDS[1],
        centers=output.centers,
        expected_results=expected_results,
        expected_legacy=expected_legacy,
        expected_surface_seal=surface_seals[POSTERIOR_CONTROL_IDS[1]],
        expected_physical_hash=physical_hash,
    )

    decisions = _mapping_rows(science, "method_decisions")
    expected_decisions = {
        (center, method): digest
        for center, method, digest in output.method_decision_hashes
    }
    expected_keys = tuple(
        (center, method) for center in output.centers for method in METHOD_MENU
    )
    if tuple(
        (str(row.get("outer_center")), str(row.get("method_id")))
        for row in decisions
    ) != expected_keys:
        raise ProtocolError("P-DCAPS v2 persisted method decision order drifted.")
    for row in decisions:
        center = str(row["outer_center"])
        method = str(row["method_id"])
        source_control = (
            POSTERIOR_CONTROL_IDS[1]
            if method == CYCLIC_METHOD_ID
            else POSTERIOR_CONTROL_IDS[0]
        )
        expected_legacy_hash = (
            expected_legacy[(source_control, center)]
            if method in {PRIMARY_METHOD_ID, LEGACY_METHOD_ID, CYCLIC_METHOD_ID}
            else None
        )
        expected_joint = (
            surface.get("surface_set_seal_hash")
            if method == CYCLIC_METHOD_ID
            else None
        )
        base = {key: value for key, value in row.items() if key != "decision_hash"}
        if (
            row.get("schema_version") != "pdcaps_method_control_decision_v1"
            or row.get("decision_hash") != canonical_hash(base)
            or row.get("decision_hash") != expected_decisions[(center, method)]
            or row.get("identity_result_hash")
            != expected_results[(POSTERIOR_CONTROL_IDS[0], center)]
            or row.get("identity_action_surface_seal_hash")
            != surface_seals[POSTERIOR_CONTROL_IDS[0]]
            or row.get("source_result_hash")
            != expected_results[(source_control, center)]
            or row.get("source_action_surface_seal_hash")
            != surface_seals[source_control]
            or row.get("physical_surface_hash") != physical_hash
            or row.get("posterior_control_id") != source_control
            or row.get("legacy_control_seal_hash") != expected_legacy_hash
            or row.get("joint_surface_set_seal_hash") != expected_joint
            or row.get("routing_authorized") is not False
            or row.get("promotion_allowed") is not False
            or row.get("target_labels_used") is not False
        ):
            raise ProtocolError("P-DCAPS v2 persisted method decision drifted.")

    compositions = _mapping_rows(science, "method_compositions")
    if len(compositions) != len(decisions) or any(
        _mapping(composition, "decision") != decision
        for composition, decision in zip(compositions, decisions, strict=True)
    ):
        raise ProtocolError("P-DCAPS v2 persisted composition decision drifted.")
    return {
        "identity_result_count": len(identity_cases),
        "cyclic_result_count": len(cyclic_cases),
        "legacy_control_count": identity_legacy + cyclic_legacy,
        "method_decision_count": len(decisions),
        "target_decision_counts_by_control": {
            POSTERIOR_CONTROL_IDS[0]: identity_target_count,
            POSTERIOR_CONTROL_IDS[1]: cyclic_target_count,
        },
        "pseudo_decision_counts_by_control": {
            POSTERIOR_CONTROL_IDS[0]: identity_pseudo_count,
            POSTERIOR_CONTROL_IDS[1]: cyclic_pseudo_count,
        },
        "case_ids_by_center": {
            center: list(identity_cases[center]) for center in output.centers
        },
        "semantic_record_reconstruction_without_refit": True,
    }


def _validate_outer_results(
    rows: tuple[dict[str, object], ...],
    *,
    control_id: str,
    centers: tuple[str, ...],
    expected_results: Mapping[tuple[str, str], str],
    expected_surface_seal: str,
    expected_physical_hash: str,
) -> tuple[dict[str, tuple[str, ...]], int, int]:
    if tuple(str(row.get("outer_center")) for row in rows) != centers:
        raise ProtocolError("P-DCAPS v2 persisted outer-result order drifted.")
    case_ids: dict[str, tuple[str, ...]] = {}
    pseudo_keys_by_outer: dict[str, tuple[tuple[str, str], ...]] = {}
    target_count = 0
    pseudo_count = 0
    for row in rows:
        center = str(row["outer_center"])
        target_decisions = _mapping_rows(row, "target_action_decisions")
        pseudo_decisions = _mapping_rows(row, "pseudo_action_decisions")
        target_cases = tuple(
            _validate_route_action_decision(value, role="target", outer=center)
            for value in target_decisions
        )
        for value in pseudo_decisions:
            _validate_route_action_decision(value, role="pseudo", outer=center)
        reconstructed = _reconstruct_outer_result(row)
        result_hash = reconstructed.result_hash
        pseudo_keys = tuple(
            (
                decision.route_key.route_center,
                decision.route_key.held_case_id,
            )
            for decision in reconstructed.pseudo_action_decisions
        )
        if (
            len(target_cases) != len(set(target_cases))
            or tuple(sorted(target_cases)) != target_cases
            or row.get("schema_version")
            != "pdcaps_outer_action_policy_result_v3"
            or row.get("posterior_control_id") != control_id
            or row.get("action_surface_seal_hash") != expected_surface_seal
            or row.get("physical_surface_hash") != expected_physical_hash
            or row.get("result_hash") != result_hash
            or row.get("result_hash") != expected_results[(control_id, center)]
            or row.get("target_labels_used") is not False
        ):
            raise ProtocolError("P-DCAPS v2 persisted outer result drifted.")
        case_ids[center] = target_cases
        pseudo_keys_by_outer[center] = pseudo_keys
        target_count += len(target_decisions)
        pseudo_count += len(pseudo_decisions)
    for outer in centers:
        observed_pseudo_keys = pseudo_keys_by_outer[outer]
        donors = tuple(center for center in CENTERS if center != outer)
        observed_donors = tuple(dict.fromkeys(center for center, _case in observed_pseudo_keys))
        ordered_pseudo_keys = tuple(
            sorted(
                observed_pseudo_keys,
                key=lambda value: (CENTERS.index(value[0]), value[1]),
            )
        )
        expected_pseudo_keys = (
            tuple(
                (donor, case_id)
                for donor in CENTERS
                if donor != outer
                for case_id in case_ids[donor]
            )
            if centers == CENTERS
            else observed_pseudo_keys
        )
        if (
            observed_donors != donors
            or len(observed_pseudo_keys) != len(set(observed_pseudo_keys))
            or observed_pseudo_keys != ordered_pseudo_keys
            or observed_pseudo_keys != expected_pseudo_keys
        ):
            raise ProtocolError(
                "P-DCAPS v2 persisted pseudo route topology/order drifted."
            )
    return case_ids, target_count, pseudo_count


def _outer_result_hash(row: Mapping[str, object]) -> str:
    try:
        calibration = _mapping(row, "calibration_families")
        target_reliability = _mapping_rows(row, "target_reliabilities")
        pseudo_reliability = tuple(
            (
                str(center),
                tuple(str(value["reliability_hash"]) for value in values),
            )
            for center, values in _center_rows(
                row, "pseudo_reliabilities_by_center"
            )
        )
        target_decisions = _mapping_rows(row, "target_action_decisions")
        pseudo_decisions = _mapping_rows(row, "pseudo_action_decisions")
        pseudo_surfaces = _mapping_rows(
            row, "pseudo_policy_response_surfaces"
        )
        policy_calibration = _mapping(row, "policy_calibration_families")
        pseudo_selections = tuple(
            (str(center), str(value["selection_hash"]))
            for center, value in _center_mappings(
                row, "pseudo_policy_selections_by_center"
            )
        )
        nested = _mapping(row, "nested_policy_calibration")
        target_surface = _mapping(row, "target_policy_surface")
        target_selection = _mapping(row, "target_policy_selection")
        base = {
            "schema_version": "pdcaps_outer_action_policy_result_v3",
            "outer_center": row["outer_center"],
            "action_surface_seal_hash": row["action_surface_seal_hash"],
            "physical_surface_hash": row["physical_surface_hash"],
            "posterior_control_id": row["posterior_control_id"],
            "action_calibration_plan_hash": calibration["plan_hash"],
            "target_reliability_hashes": tuple(
                value["reliability_hash"] for value in target_reliability
            ),
            "pseudo_reliability_hashes": pseudo_reliability,
            "target_action_decision_hashes": tuple(
                value["decision_hash"] for value in target_decisions
            ),
            "pseudo_action_decision_hashes": tuple(
                value["decision_hash"] for value in pseudo_decisions
            ),
            "pseudo_policy_response_surface_hashes": tuple(
                value["response_surface_hash"] for value in pseudo_surfaces
            ),
            "policy_calibration_plan_hash": policy_calibration["plan_hash"],
            "pseudo_policy_selection_hashes": pseudo_selections,
            "nested_policy_calibration_hash": nested["nested_hash"],
            "target_policy_surface_hash": target_surface["surface_hash"],
            "target_policy_selection_hash": target_selection["selection_hash"],
            "target_labels_used": False,
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("P-DCAPS v2 persisted outer result is malformed.") from exc
    return canonical_hash(base)


def _reconstruct_outer_result(
    row: Mapping[str, object],
) -> OuterActionPolicyResult:
    """Round-trip every nested DTO before accepting the outer-result seal."""

    try:
        calibration = _reconstruct_action_calibration_families(
            _mapping(row, "calibration_families")
        )
        target_reliabilities = tuple(
            _reconstruct_action_reliability(value)
            for value in _mapping_rows(row, "target_reliabilities")
        )
        pseudo_reliabilities = tuple(
            (
                str(center),
                tuple(
                    _reconstruct_action_reliability(value) for value in values
                ),
            )
            for center, values in _center_rows(
                row, "pseudo_reliabilities_by_center"
            )
        )
        target_decisions = tuple(
            _reconstruct_route_action_decision(value)
            for value in _mapping_rows(row, "target_action_decisions")
        )
        pseudo_decisions = tuple(
            _reconstruct_route_action_decision(value)
            for value in _mapping_rows(row, "pseudo_action_decisions")
        )
        pseudo_surfaces = tuple(
            _reconstruct_prefix_surface(value)
            for value in _mapping_rows(row, "pseudo_policy_response_surfaces")
        )
        policy_calibration = _reconstruct_policy_calibration_families(
            _mapping(row, "policy_calibration_families")
        )
        pseudo_selections = tuple(
            (str(center), _reconstruct_policy_selection(value))
            for center, value in _center_mappings(
                row, "pseudo_policy_selections_by_center"
            )
        )
        nested = _reconstruct_nested_policy_calibration(
            _mapping(row, "nested_policy_calibration")
        )
        target_surface = _reconstruct_prefix_surface(
            _mapping(row, "target_policy_surface")
        )
        target_selection = _reconstruct_policy_selection(
            _mapping(row, "target_policy_selection")
        )
        reconstructed = OuterActionPolicyResult(
            outer_center=str(row["outer_center"]),
            action_surface_seal_hash=str(row["action_surface_seal_hash"]),
            physical_surface_hash=str(row["physical_surface_hash"]),
            posterior_control_id=str(row["posterior_control_id"]),
            calibration_families=calibration,
            target_reliabilities=target_reliabilities,
            pseudo_reliabilities_by_center=pseudo_reliabilities,
            target_action_decisions=target_decisions,
            pseudo_action_decisions=pseudo_decisions,
            pseudo_policy_response_surfaces=pseudo_surfaces,
            policy_calibration_families=policy_calibration,
            pseudo_policy_selections_by_center=pseudo_selections,
            nested_policy_calibration=nested,
            target_policy_surface=target_surface,
            target_policy_selection=target_selection,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError(
            "P-DCAPS v2 persisted nested outer result is malformed."
        ) from exc

    target_hashes = tuple(row.reliability_hash for row in target_reliabilities)
    pseudo_hashes = {
        center: tuple(row.reliability_hash for row in values)
        for center, values in pseudo_reliabilities
    }
    if (
        any(
            decision.reliability_hashes != target_hashes
            for decision in target_decisions
        )
        or any(
            decision.reliability_hashes
            != pseudo_hashes.get(decision.route_key.route_center)
            for decision in pseudo_decisions
        )
    ):
        raise ProtocolError(
            "P-DCAPS v2 persisted decision/reliability lineage drifted."
        )
    return _require_roundtrip(row, reconstructed, "outer action-policy result")


def _reconstruct_action_calibration_families(
    row: Mapping[str, object],
) -> ActionCalibrationFamilies:
    target_models = tuple(
        _reconstruct_action_calibration_model(value)
        for value in _mapping_rows(row, "target_models")
    )
    pseudo_models = tuple(
        (
            str(center),
            tuple(
                _reconstruct_action_calibration_model(value) for value in values
            ),
        )
        for center, values in _center_rows(row, "pseudo_models_by_center")
    )
    contexts = []
    raw_contexts = row.get("pseudo_reliability_oof_by_context")
    if not isinstance(raw_contexts, list):
        raise ProtocolError(
            "P-DCAPS v2 persisted action calibration contexts are absent."
        )
    for context in raw_contexts:
        if (
            not isinstance(context, list)
            or len(context) != 2
            or not isinstance(context[1], list)
        ):
            raise ProtocolError(
                "P-DCAPS v2 persisted action calibration context drifted."
            )
        scored_rows = []
        for scored in context[1]:
            if (
                not isinstance(scored, list)
                or len(scored) != 2
                or not isinstance(scored[1], list)
                or not all(isinstance(value, Mapping) for value in scored[1])
            ):
                raise ProtocolError(
                    "P-DCAPS v2 persisted action calibration context drifted."
                )
            scored_rows.append(
                (
                    str(scored[0]),
                    tuple(
                        _reconstruct_action_calibration_model(value)
                        for value in scored[1]
                    ),
                )
            )
        contexts.append((str(context[0]), tuple(scored_rows)))
    reconstructed = ActionCalibrationFamilies(
        outer_center=str(row["outer_center"]),
        target_models=target_models,
        pseudo_models_by_center=pseudo_models,
        pseudo_reliability_oof_by_context=tuple(contexts),
        numerical_metric_fit_count=int(row["numerical_metric_fit_count"]),
        serialized_model_count=int(row["serialized_model_count"]),
    )
    return _require_roundtrip(row, reconstructed, "action calibration families")


def _reconstruct_action_calibration_model(
    row: Mapping[str, object],
) -> ActionCalibrationModel:
    reconstructed = ActionCalibrationModel(
        metric=str(row["metric"]),
        excluded_outer_center=str(row["excluded_outer_center"]),
        excluded_scored_center=(
            None
            if row.get("excluded_scored_center") is None
            else str(row["excluded_scored_center"])
        ),
        training_centers=tuple(str(value) for value in row["training_centers"]),
        feature_names=tuple(str(value) for value in row["feature_names"]),
        feature_mean=tuple(float(value) for value in row["feature_mean"]),
        feature_scale=tuple(float(value) for value in row["feature_scale"]),
        intercept=float(row["intercept"]),
        coefficients=tuple(float(value) for value in row["coefficients"]),
        ridge_alpha=float(row["ridge_alpha"]),
        training_row_count=int(row["training_row_count"]),
        training_response_hash=str(row["training_response_hash"]),
        weight_audit_hash=str(row["weight_audit_hash"]),
        solver=str(row["solver"]),
    )
    return _require_roundtrip(row, reconstructed, "action calibration model")


def _reconstruct_action_reliability(
    row: Mapping[str, object],
) -> ActionStratumReliability:
    reconstructed = ActionStratumReliability(
        excluded_outer_center=str(row["excluded_outer_center"]),
        excluded_scored_center=(
            None
            if row.get("excluded_scored_center") is None
            else str(row["excluded_scored_center"])
        ),
        family=str(row["family"]),
        direction=str(row["direction"]),
        represented_centers=tuple(
            str(value) for value in row["represented_centers"]
        ),
        center_metric_means=tuple(
            (
                str(value[0]),
                float(value[1]),
                float(value[2]),
                float(value[3]),
            )
            for value in row["center_metric_means"]
        ),
        equal_center_utility=_reconstruct_utility(
            _mapping(row, "equal_center_utility")
        ),
        bacc_spearman=(
            None
            if row.get("bacc_spearman") is None
            else float(row["bacc_spearman"])
        ),
        bacc_spearman_defined=bool(row["bacc_spearman_defined"]),
        positive_bacc_center_count=int(row["positive_bacc_center_count"]),
        minimum_center_count=int(row["minimum_center_count"]),
        bank_viable=bool(row["bank_viable"]),
        oof_row_count=int(row["oof_row_count"]),
        evidence_hash=str(row["evidence_hash"]),
    )
    return _require_roundtrip(row, reconstructed, "action reliability")


def _reconstruct_route_action_decision(
    row: Mapping[str, object],
) -> RouteActionDecision:
    reconstructed = RouteActionDecision(
        route_key=_reconstruct_route_key(_mapping(row, "route_key")),
        reliability_hashes=tuple(
            str(value) for value in row["reliability_hashes"]
        ),
        selection=_reconstruct_action_selection(_mapping(row, "selection")),
    )
    return _require_roundtrip(row, reconstructed, "route action decision")


def _reconstruct_action_selection(
    row: Mapping[str, object],
) -> CalibratedActionSelection:
    selected_payload = row.get("selected_action_key")
    if selected_payload is not None and not isinstance(selected_payload, Mapping):
        raise ProtocolError("P-DCAPS v2 persisted selected action is malformed.")
    reconstructed = CalibratedActionSelection(
        route_key=_reconstruct_route_key(_mapping(row, "route_key")),
        calibrated_action_hashes=tuple(
            str(value) for value in row["calibrated_action_hashes"]
        ),
        quarantined_action_hashes=tuple(
            str(value) for value in row["quarantined_action_hashes"]
        ),
        selected_action_key=(
            None
            if selected_payload is None
            else _reconstruct_action_key(selected_payload)
        ),
        selected_utility=_reconstruct_utility(_mapping(row, "selected_utility")),
        exact_p_fallback=bool(row["exact_p_fallback"]),
        reason=str(row["reason"]),
    )
    return _require_roundtrip(row, reconstructed, "calibrated action selection")


def _reconstruct_action_key(row: Mapping[str, object]) -> ActionKey:
    reconstructed = ActionKey(
        route_key=_reconstruct_route_key(_mapping(row, "route_key")),
        family=str(row["family"]),
        direction=str(row["direction"]),
        action_id=str(row["action_id"]),
        probability_hash=str(row["probability_hash"]),
        action_surface_seal_hash=str(row["action_surface_seal_hash"]),
    )
    return _require_roundtrip(row, reconstructed, "action key")


def _reconstruct_route_key(row: Mapping[str, object]) -> RouteKey:
    reconstructed = RouteKey(
        surface_role=str(row["surface_role"]),
        outer_center=str(row["outer_center"]),
        route_center=str(row["route_center"]),
        held_case_id=str(row["held_case_id"]),
        excluded_outer_center=str(row["excluded_outer_center"]),
        excluded_scored_center=(
            None
            if row.get("excluded_scored_center") is None
            else str(row["excluded_scored_center"])
        ),
        fit_scope_hash=str(row["fit_scope_hash"]),
    )
    return _require_roundtrip(row, reconstructed, "route key")


def _reconstruct_policy_calibration_families(
    row: Mapping[str, object],
) -> PolicyCalibrationFamilies:
    reconstructed = PolicyCalibrationFamilies(
        outer_center=str(row["outer_center"]),
        target=_reconstruct_nested_policy_calibration(_mapping(row, "target")),
        pseudo_calibrations_by_center=tuple(
            (str(center), _reconstruct_policy_calibration(value))
            for center, value in _center_mappings(
                row, "pseudo_calibrations_by_center"
            )
        ),
        pseudo_envelopes_by_center=tuple(
            (str(center), _reconstruct_policy_envelope(value))
            for center, value in _center_mappings(
                row, "pseudo_envelopes_by_center"
            )
        ),
        numerical_metric_fit_count=int(row["numerical_metric_fit_count"]),
        serialized_model_count=int(row["serialized_model_count"]),
    )
    return _require_roundtrip(row, reconstructed, "policy calibration families")


def _reconstruct_nested_policy_calibration(
    row: Mapping[str, object],
) -> NestedPolicyCalibration:
    reconstructed = NestedPolicyCalibration(
        outer_center=str(row["outer_center"]),
        final_calibration=_reconstruct_policy_calibration(
            _mapping(row, "final_calibration")
        ),
        oof_calibrations=tuple(
            _reconstruct_policy_calibration(value)
            for value in _mapping_rows(row, "oof_calibrations")
        ),
        oof_residuals=tuple(
            _reconstruct_policy_residual(value)
            for value in _mapping_rows(row, "oof_residuals")
        ),
        envelope=_reconstruct_policy_envelope(_mapping(row, "envelope")),
    )
    return _require_roundtrip(row, reconstructed, "nested policy calibration")


def _reconstruct_policy_calibration(
    row: Mapping[str, object],
) -> PolicyCalibration:
    reconstructed = PolicyCalibration(
        outer_center=str(row["outer_center"]),
        scored_center=(
            None if row.get("scored_center") is None else str(row["scored_center"])
        ),
        excluded_centers=tuple(str(value) for value in row["excluded_centers"]),
        supported_centers=tuple(str(value) for value in row["supported_centers"]),
        models=tuple(
            _reconstruct_policy_ridge_model(value)
            for value in _mapping_rows(row, "models")
        ),
        observation_hashes=tuple(
            str(value) for value in row["observation_hashes"]
        ),
        observation_weights=tuple(
            (str(value[0]), float(value[1]))
            for value in row["observation_weights"]
        ),
        additional_excluded_centers=tuple(
            str(value) for value in row["additional_excluded_centers"]
        ),
    )
    return _require_roundtrip(row, reconstructed, "policy calibration")


def _reconstruct_policy_ridge_model(
    row: Mapping[str, object],
) -> PolicyRidgeModel:
    reconstructed = PolicyRidgeModel(
        metric=str(row["metric"]),
        alpha=float(row["alpha"]),
        feature_names=tuple(str(value) for value in row["feature_names"]),
        feature_means=tuple(float(value) for value in row["feature_means"]),
        feature_scales=tuple(float(value) for value in row["feature_scales"]),
        intercept=float(row["intercept"]),
        coefficients=tuple(float(value) for value in row["coefficients"]),
    )
    return _require_roundtrip(row, reconstructed, "policy ridge model")


def _reconstruct_policy_residual(
    row: Mapping[str, object],
) -> PolicyOOFResidual:
    reconstructed = PolicyOOFResidual(
        outer_center=str(row["outer_center"]),
        scored_center=str(row["scored_center"]),
        route_hash=str(row["route_hash"]),
        cell_hash=str(row["cell_hash"]),
        predicted_utility=_reconstruct_utility(
            _mapping(row, "predicted_utility")
        ),
        realized_utility=_reconstruct_utility(
            _mapping(row, "realized_utility")
        ),
        calibration_hash=str(row["calibration_hash"]),
        calibration_excluded_centers=tuple(
            str(value) for value in row["calibration_excluded_centers"]
        ),
    )
    return _require_roundtrip(row, reconstructed, "policy OOF residual")


def _reconstruct_policy_envelope(row: Mapping[str, object]) -> PolicyEnvelope:
    reconstructed = PolicyEnvelope(
        outer_center=str(row["outer_center"]),
        center_means=tuple(
            (str(value[0]), _reconstruct_utility(_as_mapping(value[1])))
            for value in row["center_means"]
        ),
        full_equal_center_mean=_reconstruct_utility(
            _mapping(row, "full_equal_center_mean")
        ),
        leave_one_center_means=tuple(
            (str(value[0]), _reconstruct_utility(_as_mapping(value[1])))
            for value in row["leave_one_center_means"]
        ),
        correction=_reconstruct_utility(_mapping(row, "correction")),
        residual_hashes=tuple(str(value) for value in row["residual_hashes"]),
        excluded_scored_center=(
            None
            if row.get("excluded_scored_center") is None
            else str(row["excluded_scored_center"])
        ),
    )
    return _require_roundtrip(row, reconstructed, "policy envelope")


def _reconstruct_prefix_surface(row: Mapping[str, object]) -> PrefixSurface:
    reconstructed = PrefixSurface(
        provenance=_reconstruct_policy_provenance(_mapping(row, "provenance")),
        ranked_actions=tuple(
            _reconstruct_policy_action(value)
            for value in _mapping_rows(row, "ranked_actions")
        ),
        cells=tuple(
            _reconstruct_prefix_cell(value)
            for value in _mapping_rows(row, "cells")
        ),
    )
    return _require_roundtrip(row, reconstructed, "prefix surface")


def _reconstruct_policy_action(row: Mapping[str, object]) -> PolicyAction:
    reconstructed = PolicyAction(
        route_key=_reconstruct_route_key(_mapping(row, "route_key")),
        case_id=str(row["case_id"]),
        action_hash=str(row["action_hash"]),
        family=str(row["family"]),
        direction=str(row["direction"]),
        predicted_utility=_reconstruct_utility(
            _mapping(row, "predicted_utility")
        ),
        action_calibration_hash=str(row["action_calibration_hash"]),
    )
    return _require_roundtrip(row, reconstructed, "policy action")


def _reconstruct_policy_provenance(
    row: Mapping[str, object],
) -> PolicySurfaceProvenance:
    reconstructed = PolicySurfaceProvenance(
        surface_role=str(row["surface_role"]),
        outer_center=str(row["outer_center"]),
        route_center=str(row["route_center"]),
        excluded_outer_center=str(row["excluded_outer_center"]),
        excluded_scored_center=(
            None
            if row.get("excluded_scored_center") is None
            else str(row["excluded_scored_center"])
        ),
        action_surface_seal_hash=str(row["action_surface_seal_hash"]),
        action_exclusion_hashes=tuple(
            str(value) for value in row["action_exclusion_hashes"]
        ),
        action_fit_scope_hashes=tuple(
            str(value) for value in row["action_fit_scope_hashes"]
        ),
    )
    return _require_roundtrip(row, reconstructed, "policy surface provenance")


def _reconstruct_prefix_cell(row: Mapping[str, object]) -> PrefixCell:
    realized = row.get("realized_utility")
    if realized is not None and not isinstance(realized, Mapping):
        raise ProtocolError("P-DCAPS v2 persisted realized utility is malformed.")
    reconstructed = PrefixCell(
        provenance=_reconstruct_policy_provenance(_mapping(row, "provenance")),
        k=int(row["k"]),
        total_candidate_count=int(row["total_candidate_count"]),
        ordered_action_hashes=tuple(
            str(value) for value in row["ordered_action_hashes"]
        ),
        predicted_utility=_reconstruct_utility(
            _mapping(row, "predicted_utility")
        ),
        normalized_depth=float(row["normalized_depth"]),
        max_positive_candidate_share=float(row["max_positive_candidate_share"]),
        stratum_proportions=tuple(
            float(value) for value in row["stratum_proportions"]
        ),
        realized_utility=(
            None if realized is None else _reconstruct_utility(realized)
        ),
    )
    return _require_roundtrip(row, reconstructed, "prefix cell")


def _reconstruct_policy_selection(row: Mapping[str, object]) -> PolicySelection:
    reconstructed = PolicySelection(
        surface_hash=str(row["surface_hash"]),
        calibrated_cells=tuple(
            _reconstruct_calibrated_prefix_cell(value)
            for value in _mapping_rows(row, "calibrated_cells")
        ),
        selected_k=int(row["selected_k"]),
        selected_calibrated_cell_hash=str(
            row["selected_calibrated_cell_hash"]
        ),
    )
    return _require_roundtrip(row, reconstructed, "policy selection")


def _reconstruct_calibrated_prefix_cell(
    row: Mapping[str, object],
) -> CalibratedPrefixCell:
    reconstructed = CalibratedPrefixCell(
        cell=_reconstruct_prefix_cell(_mapping(row, "cell")),
        model_predicted_utility=_reconstruct_utility(
            _mapping(row, "model_predicted_utility")
        ),
        envelope_correction=_reconstruct_utility(
            _mapping(row, "envelope_correction")
        ),
        corrected_utility=_reconstruct_utility(
            _mapping(row, "corrected_utility")
        ),
        policy_calibration_hash=str(row["policy_calibration_hash"]),
        policy_envelope_hash=str(row["policy_envelope_hash"]),
        correction_applied_count=int(row["correction_applied_count"]),
    )
    return _require_roundtrip(row, reconstructed, "calibrated prefix cell")


def _reconstruct_utility(row: Mapping[str, object]) -> FavorableUtility:
    reconstructed = FavorableUtility(
        float(row["bacc_gain"]),
        float(row["brier_gain"]),
        float(row["log_gain"]),
    )
    return _require_roundtrip(row, reconstructed, "favorable utility")


def _require_roundtrip(
    row: Mapping[str, object], value: _T, role: str
) -> _T:
    to_payload = getattr(value, "to_payload", None)
    if not callable(to_payload) or to_payload() != dict(row):
        raise ProtocolError(f"P-DCAPS v2 persisted {role} drifted.")
    return value


def _as_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ProtocolError("P-DCAPS v2 persisted nested mapping is malformed.")
    return dict(value)


def _validate_route_action_decision(
    row: Mapping[str, object], *, role: str, outer: str
) -> str:
    route = _mapping(row, "route_key")
    selection = _mapping(row, "selection")
    route_base = {
        key: value for key, value in route.items() if key != "exclusion_hash"
    }
    held = str(route.get("held_case_id", ""))
    if (
        role not in {"target", "pseudo"}
        or route.get("surface_role") != role
        or route.get("outer_center") != outer
        or route.get("excluded_outer_center") != outer
        or not held
        or route.get("exclusion_hash") != canonical_hash(route_base)
        or (
            role == "target"
            and (
                route.get("route_center") != outer
                or route.get("excluded_scored_center") is not None
            )
        )
        or (
            role == "pseudo"
            and (
                route.get("route_center") == outer
                or route.get("excluded_scored_center")
                != route.get("route_center")
            )
        )
        or row.get("schema_version") != "pdcaps_route_action_decision_v1"
        or row.get("held_case_response_used") is not False
        or row.get("decision_hash")
        != canonical_hash(
            {
                "schema_version": "pdcaps_route_action_decision_v1",
                "route_key": route,
                "reliability_hashes": tuple(row.get("reliability_hashes", ())),
                "selection_hash": selection.get("selection_hash"),
                "held_case_response_used": False,
            }
        )
    ):
        raise ProtocolError("P-DCAPS v2 persisted H/J/d decision drifted.")
    return held


def _validate_legacy_controls(
    rows: tuple[dict[str, object], ...],
    *,
    control_id: str,
    centers: tuple[str, ...],
    expected_results: Mapping[tuple[str, str], str],
    expected_legacy: Mapping[tuple[str, str], str],
    expected_surface_seal: str,
    expected_physical_hash: str,
) -> int:
    if tuple(
        str(_mapping(row, "surface").get("outer_center")) for row in rows
    ) != centers:
        raise ProtocolError("P-DCAPS v2 persisted legacy-control order drifted.")
    for row in rows:
        surface = _mapping(row, "surface")
        center = str(surface["outer_center"])
        decisions = _mapping_rows(surface, "decisions")
        target = _mapping(surface, "target_decision")
        for decision in decisions:
            _validate_self_hash(decision, "decision_hash")
        _validate_self_hash(target, "decision_hash")
        control_hash = canonical_hash(
            {
                "schema_version": "pdcaps_legacy_control_surface_v1",
                "outer_center": center,
                "outer_result_hash": surface["outer_result_hash"],
                "physical_surface_hash": surface["physical_surface_hash"],
                "action_surface_seal_hash": surface[
                    "action_surface_seal_hash"
                ],
                "pseudo_response_surface_hashes": tuple(
                    tuple(value)
                    for value in surface["pseudo_response_surface_hashes"]
                ),
                "legacy_decision_hashes": tuple(
                    value["decision_hash"] for value in decisions
                ),
                "legacy_target_decision_hash": target["decision_hash"],
                "complete_pseudo_surfaces_required": True,
                "target_labels_used": False,
            }
        )
        seal_hash = canonical_hash(
            {
                "schema_version": "pdcaps_legacy_control_seal_v1",
                "outer_center": center,
                "outer_result_hash": surface["outer_result_hash"],
                "physical_surface_hash": surface["physical_surface_hash"],
                "action_surface_seal_hash": surface[
                    "action_surface_seal_hash"
                ],
                "pseudo_response_surface_hashes": tuple(
                    tuple(value)
                    for value in surface["pseudo_response_surface_hashes"]
                ),
                "legacy_decision_hashes": tuple(
                    value["decision_hash"] for value in decisions
                ),
                "legacy_target_decision_hash": target["decision_hash"],
                "control_surface_hash": control_hash,
                "same_run_control": True,
                "pseudo_only": True,
                "target_labels_used": False,
            }
        )
        references = _mapping_rows(row, "references")
        if (
            len(references) != len(CENTERS) - 1
            or surface.get("schema_version")
            != "pdcaps_legacy_control_surface_v1"
            or surface.get("outer_result_hash")
            != expected_results[(control_id, center)]
            or surface.get("physical_surface_hash") != expected_physical_hash
            or surface.get("action_surface_seal_hash") != expected_surface_seal
            or surface.get("control_surface_hash") != control_hash
            or row.get("schema_version") != "pdcaps_legacy_control_seal_v1"
            or row.get("legacy_control_seal_hash") != seal_hash
            or row.get("legacy_control_seal_hash")
            != expected_legacy[(control_id, center)]
            or row.get("same_run_control") is not True
            or row.get("pseudo_only") is not True
            or row.get("target_labels_used") is not False
        ):
            raise ProtocolError("P-DCAPS v2 persisted legacy control drifted.")
        for reference, decision in zip(references, decisions, strict=True):
            nested_target = _mapping(reference, "target_decision")
            nested_decision = _mapping(reference, "decision")
            reference_hash = canonical_hash(
                {
                    "schema_version": "pdcaps_legacy_pseudo_reference_v3",
                    "outer_result_hash": reference["outer_result_hash"],
                    "physical_surface_hash": reference["physical_surface_hash"],
                    "action_surface_seal_hash": reference[
                        "action_surface_seal_hash"
                    ],
                    "control_surface_hash": reference["control_surface_hash"],
                    "legacy_control_seal_hash": reference[
                        "legacy_control_seal_hash"
                    ],
                    "legacy_target_decision_hash": nested_target[
                        "decision_hash"
                    ],
                    "decision_hash": nested_decision["decision_hash"],
                    "same_run_control": True,
                    "target_labels_used": False,
                }
            )
            if (
                nested_target != target
                or nested_decision != decision
                or reference.get("outer_result_hash")
                != surface["outer_result_hash"]
                or reference.get("physical_surface_hash")
                != expected_physical_hash
                or reference.get("action_surface_seal_hash")
                != expected_surface_seal
                or reference.get("control_surface_hash") != control_hash
                or reference.get("legacy_control_seal_hash") != seal_hash
                or reference.get("reference_hash") != reference_hash
                or reference.get("same_run_control") is not True
                or reference.get("target_labels_used") is not False
            ):
                raise ProtocolError("P-DCAPS v2 persisted legacy reference drifted.")
    return len(rows)


def _validate_self_hash(row: Mapping[str, object], hash_key: str) -> None:
    base = {key: value for key, value in row.items() if key != hash_key}
    if row.get(hash_key) != canonical_hash(base):
        raise ProtocolError("P-DCAPS v2 persisted nested record hash drifted.")


def _mapping(payload: Mapping[str, object], key: str) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"P-DCAPS v2 persisted mapping is absent: {key}.")
    return dict(value)


def _mapping_rows(
    payload: Mapping[str, object], key: str
) -> tuple[dict[str, object], ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(row, Mapping) for row in value):
        raise ProtocolError(f"P-DCAPS v2 persisted rows are absent: {key}.")
    return tuple(dict(row) for row in value)


def _pairs(
    payload: Mapping[str, object], key: str
) -> tuple[tuple[object, object], ...]:
    value = payload.get(key)
    if (
        not isinstance(value, list)
        or not all(isinstance(row, list) and len(row) == 2 for row in value)
    ):
        raise ProtocolError(f"P-DCAPS v2 persisted pairs are absent: {key}.")
    return tuple((row[0], row[1]) for row in value)


def _center_rows(
    payload: Mapping[str, object], key: str
) -> tuple[tuple[object, tuple[dict[str, object], ...]], ...]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ProtocolError(f"P-DCAPS v2 persisted center rows are absent: {key}.")
    output = []
    for row in value:
        if (
            not isinstance(row, list)
            or len(row) != 2
            or not isinstance(row[1], list)
            or not all(isinstance(item, Mapping) for item in row[1])
        ):
            raise ProtocolError(
                f"P-DCAPS v2 persisted center rows are malformed: {key}."
            )
        output.append((row[0], tuple(dict(item) for item in row[1])))
    return tuple(output)


def _center_mappings(
    payload: Mapping[str, object], key: str
) -> tuple[tuple[object, dict[str, object]], ...]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ProtocolError(
            f"P-DCAPS v2 persisted center mappings are absent: {key}."
        )
    output = []
    for row in value:
        if (
            not isinstance(row, list)
            or len(row) != 2
            or not isinstance(row[1], Mapping)
        ):
            raise ProtocolError(
                f"P-DCAPS v2 persisted center mappings are malformed: {key}."
            )
        output.append((row[0], dict(row[1])))
    return tuple(output)


__all__ = ("validate_persisted_preterminal_records",)
