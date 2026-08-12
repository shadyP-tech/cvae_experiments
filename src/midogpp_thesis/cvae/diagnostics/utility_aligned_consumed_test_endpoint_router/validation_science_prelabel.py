"""Model, target-policy, frozen-action and prelabel-seal reconstruction."""

from __future__ import annotations

import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from ...protocol import ProtocolError
from ...routing.residual_topup import (
    build_single_source_tail_action,
    build_uniform_topup_action,
    target_topup_geometry,
)
from ...routing.residual_topup.hashing import canonical_sha256
from .actions import FrozenEndpointAction
from .artifact_io import read_json
from .contracts import (
    BASE_ACTION_ID,
    CENTERS,
    GLOBAL_ACTION_ID,
    ORACLE_ACTION_ROLE,
    PERMUTATION_ACTION_ID,
    ROUTED_ACTION_ID,
    TARGET_ACTION_ROLE,
    UNIFORM_ACTION_ID,
    candidate_sources,
    expected_target_action_ids,
    h_x_e_action_id,
    h_x_e_source,
)
from .models import fit_endpoint_router_model_set
from .prediction_contracts import TARGET_ROLE as TARGET_PREDICTION_ROLE
from .prediction_store import load_prediction_store
from .validation_science_common import (
    decode_hashed_row,
    mapping,
    mapping_field,
    nested_float_mapping,
    nullable_text,
    read_csv,
    require_payload_hash,
    validate_prediction_seal,
)
from .validation_science_contracts import (
    ACTION_FIELDS,
    MODEL_FIELDS,
    POLICY_FIELDS,
    DevelopmentScienceValidation,
    FeatureScienceValidation,
    PrelabelScienceValidation,
    ScientificPartitionContext,
)


def validate_prelabel_science(
    root: str | Path,
    *,
    partitions: ScientificPartitionContext,
    development: DevelopmentScienceValidation,
    features: FeatureScienceValidation,
) -> PrelabelScienceValidation:
    base = Path(root)
    model_rows = tuple(
        decode_hashed_row(raw, MODEL_FIELDS, "model_hash", "model")
        for raw in read_csv(base / "tables/model_index.csv")
    )
    if tuple(row.get("outer_target_id") for row in model_rows) != CENTERS:
        raise ProtocolError("Model target geometry drifted.")
    for row in model_rows:
        target = str(row["outer_target_id"])
        if (
            int(row.get("training_response_count", -1)) != 56
            or set(mapping(row.get("model_hashes_by_role"), "model hashes"))
            != {GLOBAL_ACTION_ID, ROUTED_ACTION_ID, PERMUTATION_ACTION_ID}
            or row.get("source_feature_surface_hash")
            != features.source_surface_hash_by_target[target]
            or row.get("development_response_set_hash")
            != development.binding_hash_by_target[target]
            or row.get("strict_H_q_e_exclusion") is not True
            or row.get("same_outer_H_evaluation_labels_used_for_fit") is not False
            or row.get("support_labels_used_for_fit") is not False
            or row.get("target_features_used_for_fit") is not False
        ):
            raise ProtocolError("Model scientific binding drifted.")
    model_index = read_json(base / "manifests/model_index.json")
    model_hashes = {
        str(row["outer_target_id"]): str(row["model_hash"])
        for row in model_rows
    }
    transfer_hashes = {
        str(row["outer_target_id"]): str(row["cardinality_transfer_hash"])
        for row in model_rows
    }
    expected_model_unhashed = {
        "schema_version": "midogpp_consumed_test_endpoint_router_model_set_v1",
        "centers": list(CENTERS),
        "model_hashes_by_target": model_hashes,
        "cardinality_transfer_hashes_by_target": transfer_hashes,
        "source_feature_surface_set_hash": features.source_surface_set_hash,
        "development_response_set_hash": development.response_set_hash,
        "strict_H_q_e_exclusion": True,
        "support_labels_used": False,
        "same_outer_H_evaluation_labels_used_for_model_H": False,
    }
    expected_model_index = {
        **expected_model_unhashed,
        "model_set_hash": canonical_sha256(expected_model_unhashed),
    }
    refitted_models = fit_endpoint_router_model_set(
        features.source_surface_set,
        development.response_set,
    )
    if (
        model_index != expected_model_index
        or model_index != refitted_models.to_payload()
        or any(
            row
            != refitted_models.by_target[
                str(row["outer_target_id"])
            ].to_payload()
            for row in model_rows
        )
    ):
        raise ProtocolError("Model-set manifest drifted.")

    transfer_seal = read_json(base / "manifests/cardinality_transfer_seal.json")
    transfers = mapping_field(transfer_seal, "transfer_by_target")
    for target in CENTERS:
        payload = mapping(transfers[target], "cardinality transfer")
        require_payload_hash(payload, "transfer_hash", "cardinality transfer")
        if (
            payload.get("outer_target_id") != target
            or payload.get("transfer_hash") != transfer_hashes[target]
            or int(payload.get("independent_query_count", -1)) != 8
            or int(payload.get("source_inner_candidate_count", -1)) != 7
            or int(payload.get("deployment_candidate_count", -1)) != 8
            or payload.get("claim_role")
            != "source_inner_ensemble_eligibility_only_not_target_utility"
            or not isinstance(payload.get("authorization_failures"), list)
            or (payload.get("authorized_for_target_policy") is True)
            != (payload.get("authorization_failures") == [])
        ):
            raise ProtocolError("Cardinality transfer/model binding drifted.")
    require_payload_hash(
        transfer_seal, "cardinality_transfer_seal_hash", "transfer seal"
    )
    if transfer_seal.get("transfer_hashes_by_target") != transfer_hashes:
        raise ProtocolError("Cardinality transfer seal geometry drifted.")

    plans_manifest = read_json(base / "manifests/target_policy_plans.json")
    policy_rows = tuple(
        decode_hashed_row(raw, POLICY_FIELDS, "policy_hash", "policy")
        for raw in read_csv(base / "tables/target_policy_plans.csv")
    )
    if tuple(row.get("target_id") for row in policy_rows) != CENTERS:
        raise ProtocolError("Target policy geometry drifted.")
    core_payloads = mapping_field(plans_manifest, "core_policies_by_target")
    policy_hashes: dict[str, str] = {}
    routed_candidates: dict[str, str] = {}
    routed_executed: dict[str, str | None] = {}
    selected_actions: dict[str, str] = {}
    routed_predictions: dict[str, Mapping[str, float]] = {}
    for row in policy_rows:
        target = str(row["target_id"])
        core = mapping(core_payloads[target], "core policy")
        require_core_policy_hash(core)
        predictions = nested_float_mapping(
            mapping_field(core, "role_prediction_by_source"), "role prediction"
        )
        routed = predictions.get(ROUTED_ACTION_ID)
        sources = set(candidate_sources(target))
        if (
            set(predictions)
            != {GLOBAL_ACTION_ID, ROUTED_ACTION_ID, PERMUTATION_ACTION_ID}
            or routed is None
            or any(set(values) != sources for values in predictions.values())
        ):
            raise ProtocolError("Routed policy prediction geometry drifted.")
        proposal = min(routed, key=lambda source: (-routed[source], source))
        executed = nullable_text(row.get("executed_routed_source"))
        production = mapping(
            mapping_field(
                plans_manifest, "target_feature_productions_by_target"
            )[target],
            "target production",
        )
        role_sources = mapping_field(core, "role_selected_source")
        role_actions = mapping_field(core, "role_selected_action")
        if set(role_sources) != set(predictions) or set(role_actions) != set(predictions):
            raise ProtocolError("Core policy role-selection geometry drifted.")
        validate_core_policy_formulas(
            core,
            predictions,
            sources,
            routed_transfer_authorized=(
                mapping(transfers[target], "cardinality transfer").get(
                    "authorized_for_target_policy"
                )
                is True
            ),
        )
        if (
            row.get("core_policy_hash") != core.get("policy_hash")
            or row.get("model_hash") != model_hashes[target]
            or row.get("target_feature_hash")
            != features.target_feature_hash_by_target[target]
            or row.get("support_partition_lock_hash")
            != partitions.support_partition_lock_hash
            or row.get("routed_candidate_source") != proposal
            or core.get("target_id") != target
            or core.get("selected_action_role") != row.get("selected_action_id")
            or nullable_text(core.get("selected_source")) != executed
            or core.get("exact_b_fallback") != row.get("exact_B_fallback")
            or role_actions.get(ROUTED_ACTION_ID) != row.get("selected_action_id")
            or nullable_text(role_sources.get(ROUTED_ACTION_ID)) != executed
            or core.get("cardinality_transfer_hash") != transfer_hashes[target]
            or core.get("point_feature_surface_hash")
            != production.get("point_surface_hash")
            or core.get("bootstrap_feature_surface_hashes")
            != production.get("bootstrap_surface_hashes")
            or row.get("selected_action_id")
            not in {BASE_ACTION_ID, ROUTED_ACTION_ID}
            or row.get("selected_action_role") != row.get("selected_action_id")
            or row.get("target_static") is not True
            or row.get("case_router_used") is not False
            or row.get("support_labels_used") is not False
            or row.get("same_outer_H_evaluation_labels_used") is not False
            or row.get("target_utility_used") is not False
            or row.get("may_update_from_terminal_scores") is not False
            or row.get("diagnostic_only") is not True
        ):
            raise ProtocolError("Target policy scientific binding drifted.")
        policy_hashes[target] = str(row["policy_hash"])
        routed_candidates[target] = proposal
        routed_executed[target] = executed
        selected_actions[target] = str(row["selected_action_id"])
        routed_predictions[target] = MappingProxyType(dict(routed))
    policy_set_unhashed = {
        "schema_version": "midogpp_consumed_test_frozen_target_policy_set_v1",
        "centers": list(CENTERS),
        "policy_hashes_by_target": policy_hashes,
        "model_set_hash": expected_model_index["model_set_hash"],
        "one_static_action_per_target": True,
        "support_labels_used": False,
        "same_outer_H_evaluation_labels_used": False,
        "terminal_scores_may_update_policy": False,
    }
    policy_set_hash = canonical_sha256(policy_set_unhashed)
    for key, value in {
        **policy_set_unhashed,
        "policy_set_hash": policy_set_hash,
        "target_plan_count": len(CENTERS),
    }.items():
        if plans_manifest.get(key) != value:
            raise ProtocolError("Target policy-set manifest drifted.")

    action_rows = tuple(
        decode_hashed_row(raw, ACTION_FIELDS, "action_hash", "action")
        for raw in read_csv(base / "tables/frozen_actions.csv")
    )
    expected_action_keys = tuple(
        (target, action_id)
        for target in CENTERS
        for action_id in expected_target_action_ids(target)
    )
    if tuple(
        (row.get("outer_target_id"), row.get("action_id"))
        for row in action_rows
    ) != expected_action_keys:
        raise ProtocolError("Frozen action key geometry drifted.")
    action_hashes: dict[tuple[str, str], str] = {}
    effective_actions: dict[tuple[str, str], str] = {}
    action_set_hashes: dict[str, str] = {}
    for target in CENTERS:
        selected_rows = tuple(
            row for row in action_rows if row["outer_target_id"] == target
        )
        expected_typed = tuple(
            rebuild_action(row, policy_hashes[target], core_payloads[target])
            for row in selected_rows
        )
        for row, typed in zip(selected_rows, expected_typed, strict=True):
            if row != typed.to_payload():
                raise ProtocolError("Frozen action reconstruction drifted.")
            key = (target, typed.action_id)
            action_hashes[key] = typed.action_hash
            effective_actions[key] = (
                BASE_ACTION_ID
                if typed.core_action is None
                else UNIFORM_ACTION_ID
                if typed.action_id == UNIFORM_ACTION_ID
                else h_x_e_action_id(typed.selected_source)
            )
        action_set_unhashed = {
            "schema_version": "midogpp_consumed_test_frozen_target_action_set_v1",
            "target_id": target,
            "action_ids": [row.action_id for row in expected_typed],
            "action_hashes": [row.action_hash for row in expected_typed],
            "policy_hash": policy_hashes[target],
            "one_static_action_per_target_geometry": True,
            "terminal_scores_may_update_actions": False,
            "diagnostic_only": True,
        }
        action_set_hashes[target] = canonical_sha256(action_set_unhashed)
    frozen_manifest = read_json(base / "manifests/frozen_actions.json")
    action_library_unhashed = {
        "schema_version": "midogpp_consumed_test_frozen_target_action_library_v1",
        "centers": list(CENTERS),
        "action_set_hashes_by_target": action_set_hashes,
        "policy_set_hash": policy_set_hash,
        "target_count": len(CENTERS),
        "reported_action_count": len(action_rows),
        "physical_action_count": 90,
        "terminal_scores_may_update_actions": False,
        "diagnostic_only": True,
    }
    action_library_hash = canonical_sha256(action_library_unhashed)
    if frozen_manifest != {
        **action_library_unhashed,
        "action_library_hash": action_library_hash,
    }:
        raise ProtocolError("Frozen action-library manifest drifted.")

    target_seal = read_json(base / "manifests/target_prediction_seal.json")
    target_store = load_prediction_store(base, phase=TARGET_PREDICTION_ROLE)
    validate_prediction_seal(
        base,
        target_seal,
        hash_field="target_prediction_seal_hash",
        expected_arrays_member="arrays/target_action_probabilities.npz",
        expected_index_member="manifests/target_prediction_index.json",
        store=target_store,
    )
    if target_store.partition_lock_hash != partitions.support_partition_lock_hash:
        raise ProtocolError("Target prediction partition lock drifted.")
    global_seal = read_json(base / "manifests/global_prelabel_seal.json")
    target_seal_hash = str(target_seal.get("target_prediction_seal_hash", ""))
    global_hash = str(global_seal.get("global_prelabel_seal_hash", ""))
    if (
        target_seal.get("target_policy_plan_hashes_by_center") != policy_hashes
        or target_seal.get("target_policy_plan_set_hash") != policy_set_hash
        or target_seal.get("frozen_action_set_hash") != action_library_hash
        or target_seal.get("global_prelabel_seal_hash") != global_hash
        or global_seal.get("target_policy_plan_set_hash") != policy_set_hash
        or global_seal.get("frozen_action_library_hash") != action_library_hash
        or global_seal.get("global_target_prediction_seal_hash") != target_seal_hash
        or global_seal.get("support_labels_used") is not False
        or global_seal.get(
            "same_outer_H_evaluation_labels_used_for_plan_H"
        ) is not False
    ):
        raise ProtocolError("Global prelabel seal lineage drifted.")
    return PrelabelScienceValidation(
        model_set_hash=str(expected_model_index["model_set_hash"]),
        policy_set_hash=policy_set_hash,
        action_library_hash=action_library_hash,
        policy_hash_by_target=MappingProxyType(policy_hashes),
        action_hash_by_key=MappingProxyType(action_hashes),
        effective_action_by_key=MappingProxyType(effective_actions),
        routed_candidate_by_target=MappingProxyType(routed_candidates),
        routed_executed_source_by_target=MappingProxyType(routed_executed),
        selected_action_by_target=MappingProxyType(selected_actions),
        routed_prediction_by_target=MappingProxyType(routed_predictions),
        global_target_prediction_seal_hash=target_seal_hash,
        global_prelabel_seal_hash=global_hash,
    )


def rebuild_action(
    row: Mapping[str, object], policy_hash: str, core_payload: object
) -> FrozenEndpointAction:
    target = str(row["outer_target_id"])
    action_id = str(row["action_id"])
    geometry = target_topup_geometry(candidate_sources(target))
    selected = nullable_text(row.get("selected_source"))
    effective = str(row["effective_action_id"])
    core = mapping(core_payload, "core policy")
    if action_id == BASE_ACTION_ID:
        expected_selected, expected_effective = None, BASE_ACTION_ID
    elif action_id == UNIFORM_ACTION_ID:
        expected_selected, expected_effective = None, UNIFORM_ACTION_ID
    elif h_x_e_source(action_id) is not None:
        expected_selected, expected_effective = h_x_e_source(action_id), action_id
    else:
        expected_selected = nullable_text(
            mapping_field(core, "role_selected_source").get(action_id)
        )
        expected_effective = str(
            mapping_field(core, "role_selected_action").get(action_id, "")
        )
    expected_diagnostic = action_id != ROUTED_ACTION_ID
    if (
        selected != expected_selected
        or effective != expected_effective
        or row.get("diagnostic_control") is not expected_diagnostic
    ):
        raise ProtocolError("Frozen action/core-policy realization drifted.")
    if effective == BASE_ACTION_ID:
        core_action = None
    elif action_id == UNIFORM_ACTION_ID:
        core_action = build_uniform_topup_action(geometry)
    else:
        if selected is None:
            raise ProtocolError("Frozen non-base action has no selected source.")
        core_action = build_single_source_tail_action(selected, geometry=geometry)
    oracle = h_x_e_source(action_id)
    expected_role = ORACLE_ACTION_ROLE if oracle is not None else TARGET_ACTION_ROLE
    if (
        row.get("query_id") != target
        or row.get("action_role") != expected_role
        or row.get("policy_hash") != policy_hash
        or row.get("target_static") is not True
        or row.get("case_router_used") is not False
        or row.get("labels_used_to_build") is not False
        or row.get("terminal_scores_used_to_build") is not False
        or row.get("diagnostic_only") is not True
    ):
        raise ProtocolError("Frozen target action semantics drifted.")
    return FrozenEndpointAction(
        outer_target_id=target,
        query_id=target,
        action_id=action_id,
        action_role=expected_role,
        geometry=geometry,
        selected_source=selected,
        effective_action_id=effective,
        core_action=core_action,
        policy_hash=policy_hash,
        diagnostic_control=bool(row["diagnostic_control"]),
        action_hash=str(row["action_hash"]),
    )


def require_core_policy_hash(payload: Mapping[str, object]) -> None:
    observed = payload.get("policy_hash")
    unhashed = {
        key: value for key, value in payload.items() if key != "policy_hash"
    }
    unhashed.update(
        {
            "target_labels_used": False,
            "target_utility_used": False,
            "seed_rows_are_independent_observations": False,
        }
    )
    if observed != canonical_sha256(unhashed):
        raise ProtocolError("Core target policy hash drifted.")


def validate_core_policy_formulas(
    payload: Mapping[str, object],
    predictions: Mapping[str, Mapping[str, float]],
    sources: set[str],
    *,
    routed_transfer_authorized: bool,
) -> None:
    roles = {GLOBAL_ACTION_ID, ROUTED_ACTION_ID, PERMUTATION_ACTION_ID}
    model_se = nested_float_mapping(
        mapping_field(payload, "role_model_standard_error_by_source"),
        "model standard error",
    )
    bootstrap_sd = nested_float_mapping(
        mapping_field(payload, "role_bootstrap_standard_deviation_by_source"),
        "bootstrap standard deviation",
    )
    combined = nested_float_mapping(
        mapping_field(payload, "role_combined_standard_error_by_source"),
        "combined standard error",
    )
    lower = nested_float_mapping(
        mapping_field(payload, "role_lower_confidence_bound_by_source"),
        "lower confidence bound",
    )
    scalar_spread = nested_float_mapping(
        mapping_field(
            payload, "role_target_scalar_seed_standard_deviation_by_source"
        ),
        "target scalar seed spread",
    )
    nested = (predictions, model_se, bootstrap_sd, combined, lower, scalar_spread)
    if any(
        set(values) != roles
        or any(set(by_source) != sources for by_source in values.values())
        for values in nested
    ):
        raise ProtocolError("Core policy uncertainty geometry drifted.")
    if (
        float(payload.get("gain_lcb_multiplier", math.nan)) != 1.96
        or payload.get(
            "bootstrap_dispersion_divided_by_seed_repeat_sqrt"
        ) is not False
        or payload.get(
            "target_scalar_seed_spread_enters_combined_standard_error"
        ) is not False
        or payload.get("target_scalar_seed_spread_role")
        != "descriptive_only_non_decision"
    ):
        raise ProtocolError("Core policy uncertainty semantics drifted.")
    for role in roles:
        for source in sources:
            expected_se = math.sqrt(
                model_se[role][source] ** 2 + bootstrap_sd[role][source] ** 2
            )
            expected_lcb = predictions[role][source] - 1.96 * expected_se
            if (
                not math.isclose(
                    combined[role][source],
                    expected_se,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                or not math.isclose(
                    lower[role][source],
                    expected_lcb,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            ):
                raise ProtocolError("Core policy LCB formula drifted.")
    role_sources = mapping_field(payload, "role_selected_source")
    role_actions = mapping_field(payload, "role_selected_action")
    for role in roles:
        candidate = min(
            predictions[role],
            key=lambda source: (-predictions[role][source], source),
        )
        selected_action = role_actions.get(role)
        selected_source = nullable_text(role_sources.get(role))
        expected_selected = lower[role][candidate] > 0.0 and (
            role != ROUTED_ACTION_ID or routed_transfer_authorized
        )
        expected_action = role if expected_selected else BASE_ACTION_ID
        expected_source = candidate if expected_selected else None
        if selected_action not in {BASE_ACTION_ID, role}:
            raise ProtocolError("Core policy selected-action role drifted.")
        if selected_action != expected_action or selected_source != expected_source:
            raise ProtocolError("Core policy positive-LCB selection drifted.")
    routed_candidate = min(
        predictions[ROUTED_ACTION_ID],
        key=lambda source: (-predictions[ROUTED_ACTION_ID][source], source),
    )
    expected_fallback_reason = (
        "source_inner_cardinality_or_capacity_gate_failed"
        if not routed_transfer_authorized
        else "routed_selected_gain_lcb_not_positive"
        if lower[ROUTED_ACTION_ID][routed_candidate] <= 0.0
        else None
    )
    if nullable_text(payload.get("fallback_reason")) != expected_fallback_reason:
        raise ProtocolError("Core policy fallback reason drifted.")


__all__ = (
    "rebuild_action", "require_core_policy_hash",
    "validate_core_policy_formulas", "validate_prelabel_science",
)
