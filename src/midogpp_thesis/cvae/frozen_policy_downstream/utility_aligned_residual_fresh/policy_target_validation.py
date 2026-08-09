"""Reconstructive validation of typed target features and frozen core policies."""

from __future__ import annotations

import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import canonical_sha256
from ...routing.utility_aligned.ensemble_policy_contracts import (
    ENSEMBLE_AUTHORIZATION_UNCERTAINTY_COMPONENTS,
    ENSEMBLE_TARGET_SEED_SPREAD_ROLE,
)
from ...routing.utility_aligned.contracts import (
    BASE_ACTION_ID as CORE_BASE_ACTION_ID,
    GLOBAL_ACTION_ID as CORE_GLOBAL_ACTION_ID,
    PERMUTATION_ACTION_ID as CORE_PERMUTATION_ACTION_ID,
    ROUTED_ACTION_ID as CORE_ROUTED_ACTION_ID,
    build_case_bootstrap_plan,
)
from .contracts import (
    CENTERS,
    GENERATION_SEEDS,
    GLOBAL_ACTION_ID as FRESH_GLOBAL_ACTION_ID,
    PERMUTATION_ACTION_ID as FRESH_PERMUTATION_ACTION_ID,
    ROUTED_ACTION_ID as FRESH_ROUTED_ACTION_ID,
    TRAINING_SEEDS,
    legal_sources,
)
from .policy_io import read_json, require_hash, sha256_like
from .policy_schema import (
    ENSEMBLE_POLICY_FAMILY,
    ENSEMBLE_TARGET_POLICY_SCHEMA,
    TARGET_FEATURE_LOCK_KEYS,
    TARGET_POLICY_KEYS,
    TARGET_POLICY_LOCK_SCHEMA,
    TARGET_POLICY_SHARED_KEYS,
)


def validate_target_policy_lock(
    root: Path,
    *,
    policy: Mapping[str, object],
    support: Mapping[str, tuple[str, ...]],
    evaluation: Mapping[str, tuple[str, ...]],
    frozen_actions: Mapping[tuple[str, str], Mapping[str, object]],
) -> None:
    target_lock = read_json(root / "manifests/target_policy_lock.json")
    require_hash(target_lock, "target_policy_lock_hash", "target policy lock")
    if (
        set(target_lock) != set(TARGET_POLICY_KEYS)
        or target_lock.get("schema_version") != TARGET_POLICY_LOCK_SCHEMA
        or any(target_lock.get(key) != policy.get(key) for key in TARGET_POLICY_SHARED_KEYS)
        or target_lock.get("target_support_case_ids_by_target")
        != {target: list(support[target]) for target in CENTERS}
        or target_lock.get("target_evaluation_case_ids_by_target")
        != {target: list(evaluation[target]) for target in CENTERS}
        or target_lock.get(
            "target_support_action_shift_case_ensemble_group_count"
        )
        != sum(len(support[target]) * len(legal_sources(target)) for target in CENTERS)
        or target_lock.get("target_support_action_shift_row_count")
        != 9
        * int(
            target_lock.get(
                "target_support_action_shift_case_ensemble_group_count", -1
            )
        )
    ):
        raise ProtocolError("Utility-aligned target-policy lock identity drifted.")
    if policy.get("policy_family") != ENSEMBLE_POLICY_FAMILY:
        raise ProtocolError(
            "Stage-70 utility-aligned residual fresh accepts only the ensemble policy family."
        )
    feature_locks = _validate_feature_locks(
        target_lock.get("target_feature_locks"),
        support=support,
    )
    _validate_policy_grid(
        target_lock.get("policies"),
        feature_locks=feature_locks,
        frozen_actions=frozen_actions,
    )


def _validate_feature_locks(
    raw_locks: object,
    *,
    support: Mapping[str, tuple[str, ...]],
) -> Mapping[str, Mapping[str, object]]:
    if (
        not isinstance(raw_locks, Sequence)
        or isinstance(raw_locks, (str, bytes))
        or len(raw_locks) != len(CENTERS)
    ):
        raise ProtocolError("Utility-aligned target-feature locks are incomplete.")
    feature_locks: dict[str, Mapping[str, object]] = {}
    for target, raw_lock in zip(CENTERS, raw_locks, strict=True):
        if (
            not isinstance(raw_lock, Mapping)
            or set(raw_lock) != set(TARGET_FEATURE_LOCK_KEYS)
            or raw_lock.get("target_id") != target
        ):
            raise ProtocolError("Utility-aligned target-feature lock fields drifted.")
        lock = {str(key): value for key, value in raw_lock.items()}
        require_hash(lock, "target_feature_lock_hash", f"target feature lock {target}")
        raw_plan = lock.get("case_bootstrap_plan")
        if not isinstance(raw_plan, Mapping):
            raise ProtocolError("Utility-aligned case-bootstrap plan is absent.")
        try:
            plan = build_case_bootstrap_plan(
                target_id=target,
                support_case_ids=tuple(raw_plan["support_case_ids"]),
                bootstrap_seed=int(raw_plan["bootstrap_seed"]),
                replicate_count=int(raw_plan["replicate_count"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ProtocolError("Utility-aligned case-bootstrap plan is malformed.") from exc
        if (
            dict(raw_plan) != plan.to_payload()
            or plan.support_case_ids != support[target]
            or plan.replicate_count < 32
        ):
            raise ProtocolError("Utility-aligned case-bootstrap plan failed reconstruction.")
        hashes = _hash_sequence(lock.get("bootstrap_surface_hashes"), "bootstrap feature surfaces")
        if (
            len(hashes) != plan.replicate_count
            or len(set(hashes)) != len(hashes)
            or lock.get("bootstrap_surface_hashes_hash") != canonical_sha256(list(hashes))
            or not sha256_like(lock.get("target_feature_surface_hash"))
            or lock.get("target_feature_row_count") != 8
            or lock.get("support_case_count") != len(plan.support_case_ids)
            or int(lock["support_case_count"]) < 8
            or lock.get("candidate_sources") != list(legal_sources(target))
            or lock.get("training_seeds") != list(TRAINING_SEEDS)
            or lock.get("generation_seeds") != list(GENERATION_SEEDS)
            or lock.get("case_level_resampling") is not True
            or lock.get("labels_used") is not False
        ):
            raise ProtocolError("Utility-aligned typed target-feature geometry drifted.")
        feature_locks[target] = MappingProxyType(lock)
    return MappingProxyType(feature_locks)


def _validate_policy_grid(
    raw_policies: object,
    *,
    feature_locks: Mapping[str, Mapping[str, object]],
    frozen_actions: Mapping[tuple[str, str], Mapping[str, object]],
) -> None:
    if (
        not isinstance(raw_policies, Sequence)
        or isinstance(raw_policies, (str, bytes))
        or len(raw_policies) != len(CENTERS) * 3
    ):
        raise ProtocolError("Utility-aligned target policies are incomplete.")
    fresh_id = {
        CORE_GLOBAL_ACTION_ID: FRESH_GLOBAL_ACTION_ID,
        CORE_ROUTED_ACTION_ID: FRESH_ROUTED_ACTION_ID,
        CORE_PERMUTATION_ACTION_ID: FRESH_PERMUTATION_ACTION_ID,
    }
    expected = tuple(
        (target, action)
        for target in CENTERS
        for action in (
            CORE_GLOBAL_ACTION_ID,
            CORE_ROUTED_ACTION_ID,
            CORE_PERMUTATION_ACTION_ID,
        )
    )
    for (target, proposed), raw in zip(expected, raw_policies, strict=True):
        if not isinstance(raw, Mapping):
            raise ProtocolError("Utility-aligned target policy is malformed.")
        schema = raw.get("schema_version")
        if schema != ENSEMBLE_TARGET_POLICY_SCHEMA:
            raise ProtocolError(
                "Stage-70 utility-aligned residual fresh accepts only the ensemble target-policy schema."
            )
        _validate_ensemble_policy(
            {str(key): value for key, value in raw.items()},
            target=target, proposed=proposed, feature_lock=feature_locks[target],
            frozen_action=frozen_actions[(target, fresh_id[proposed])],
        )


def _validate_ensemble_policy(
    payload: Mapping[str, object], *, target: str, proposed: str,
    feature_lock: Mapping[str, object], frozen_action: Mapping[str, object],
) -> None:
    required = {
        "schema_version", "target_id", "role", "candidate_sources",
        "proposed_action_id", "action_id", "proposed_source", "selected_source",
        "predicted_gain", "standard_error", "lower_confidence_bound",
        "support_case_count", "support_bootstrap_replicates",
        "used_exact_base_fallback", "fallback_reason", "model_hash",
        "feature_surface_hash", "cardinality_eligibility_hash", "policy_hash",
        "ensemble_policy",
    }
    role = {CORE_GLOBAL_ACTION_ID: "G", CORE_ROUTED_ACTION_ID: "R", CORE_PERMUTATION_ACTION_ID: "P"}[proposed]
    candidates = list(legal_sources(target))
    fallback = payload.get("used_exact_base_fallback") is True
    nested = payload.get("ensemble_policy")
    if not isinstance(nested, Mapping):
        raise ProtocolError("Utility-aligned ensemble policy contract is absent.")
    role_predictions = _nested_role_mapping(
        nested.get("role_prediction_by_source"), role=role, candidates=candidates
    )
    role_standard_errors = _nested_role_mapping(
        nested.get("role_combined_standard_error_by_source"),
        role=role,
        candidates=candidates,
    )
    role_model_standard_errors = _nested_role_mapping(
        nested.get("role_model_standard_error_by_source"),
        role=role,
        candidates=candidates,
    )
    role_bootstrap_standard_deviations = _nested_role_mapping(
        nested.get("role_bootstrap_standard_deviation_by_source"),
        role=role,
        candidates=candidates,
    )
    role_seed_standard_deviations = _nested_role_mapping(
        nested.get("role_target_scalar_seed_standard_deviation_by_source"),
        role=role,
        candidates=candidates,
    )
    role_lower_bounds = _nested_role_mapping(
        nested.get("role_lower_confidence_bound_by_source"),
        role=role,
        candidates=candidates,
    )
    proposed_source = min(
        candidates, key=lambda source: (-role_predictions[source], source)
    )
    selected_action_by_role = nested.get("role_selected_action")
    selected_source_by_role = nested.get("role_selected_source")
    expected_role_action = "B" if fallback else role
    if (
        set(payload) != required
        or payload.get("target_id") != target or payload.get("role") != role
        or payload.get("candidate_sources") != candidates
        or payload.get("feature_surface_hash") != feature_lock.get("target_feature_surface_hash")
        or not isinstance(payload.get("support_case_count"), int)
        or int(payload["support_case_count"]) < 8
        or payload.get("support_case_count") != feature_lock.get("support_case_count")
        or not isinstance(payload.get("support_bootstrap_replicates"), int)
        or int(payload["support_bootstrap_replicates"]) < 32
        or not all(sha256_like(payload.get(key)) for key in ("model_hash", "feature_surface_hash", "cardinality_eligibility_hash", "policy_hash"))
        or not all(_finite(payload.get(key)) for key in ("predicted_gain", "standard_error", "lower_confidence_bound"))
        or payload.get("proposed_action_id") != proposed
        or payload.get("action_id") != (CORE_BASE_ACTION_ID if fallback else proposed)
        or payload.get("proposed_source") != proposed_source
        or payload.get("predicted_gain") != role_predictions[proposed_source]
        or payload.get("standard_error") != role_standard_errors[proposed_source]
        or payload.get("lower_confidence_bound") != role_lower_bounds[proposed_source]
        or float(payload["standard_error"]) < 0.0
        or any(
            value < 0.0
            for values in (
                role_model_standard_errors,
                role_bootstrap_standard_deviations,
                role_seed_standard_deviations,
            )
            for value in values.values()
        )
        or any(
            abs(
                role_standard_errors[source] ** 2
                - role_model_standard_errors[source] ** 2
                - role_bootstrap_standard_deviations[source] ** 2
            )
            > 1e-12
            for source in candidates
        )
        or (
            abs(
                float(payload["lower_confidence_bound"])
                - (
                    float(payload["predicted_gain"])
                    - 1.96 * float(payload["standard_error"])
                )
            )
            > 1e-12
        )
        or (fallback and (payload.get("selected_source") is not None or not isinstance(payload.get("fallback_reason"), str) or not payload.get("fallback_reason")))
        or (not fallback and (payload.get("selected_source") != proposed_source or payload.get("fallback_reason") is not None))
        or (not fallback and role_lower_bounds[proposed_source] <= 0.0)
        or not isinstance(selected_action_by_role, Mapping)
        or selected_action_by_role.get(role) != expected_role_action
        or not isinstance(selected_source_by_role, Mapping)
        or selected_source_by_role.get(role) != payload.get("selected_source")
        or nested.get("schema_version") != ENSEMBLE_TARGET_POLICY_SCHEMA
        or nested.get("authorization_uncertainty_components")
        != list(ENSEMBLE_AUTHORIZATION_UNCERTAINTY_COMPONENTS)
        or nested.get("target_scalar_seed_spread_role")
        != ENSEMBLE_TARGET_SEED_SPREAD_ROLE
        or nested.get("target_scalar_seed_spread_enters_combined_standard_error")
        is not False
        or nested.get("target_id") != target
        or nested.get("point_feature_surface_hash")
        != feature_lock.get("target_feature_surface_hash")
        or nested.get("bootstrap_feature_surface_hashes")
        != feature_lock.get("bootstrap_surface_hashes")
        or nested.get("cardinality_transfer_hash")
        != payload.get("cardinality_eligibility_hash")
        or nested.get("policy_hash") != payload.get("policy_hash")
        or frozen_action.get("selected_source") != payload.get("selected_source")
        or frozen_action.get("abstained_to_base") is not fallback
        or frozen_action.get("fallback_reason") != payload.get("fallback_reason")
    ):
        raise ProtocolError("Utility-aligned ensemble target policy drifted.")


def _nested_role_mapping(
    raw: object,
    *,
    role: str,
    candidates: Sequence[str],
) -> Mapping[str, float]:
    if not isinstance(raw, Mapping) or set(str(key) for key in raw) != {"G", "R", "P"}:
        raise ProtocolError("Utility-aligned ensemble role mapping drifted.")
    values = raw.get(role)
    if (
        not isinstance(values, Mapping)
        or set(str(key) for key in values) != set(candidates)
        or any(not _finite(value) for value in values.values())
    ):
        raise ProtocolError("Utility-aligned ensemble candidate mapping drifted.")
    return MappingProxyType({str(key): float(value) for key, value in values.items()})


def _hash_sequence(value: object, role: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ProtocolError(f"Utility-aligned {role} are malformed.")
    output = tuple(str(item) for item in value)
    if not output or any(not sha256_like(item) for item in output):
        raise ProtocolError(f"Utility-aligned {role} are malformed.")
    return output


def _finite(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


__all__ = ("validate_target_policy_lock",)
