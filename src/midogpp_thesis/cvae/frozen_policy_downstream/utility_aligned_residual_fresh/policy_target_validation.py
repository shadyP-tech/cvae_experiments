"""Reconstructive validation of typed target features and frozen core policies."""

from __future__ import annotations

import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import canonical_sha256
from ...routing.utility_aligned.contracts import (
    ABSTENTION_SEMANTICS,
    BASE_ACTION_ID as CORE_BASE_ACTION_ID,
    GLOBAL_ACTION_ID as CORE_GLOBAL_ACTION_ID,
    PERMUTATION_ACTION_ID as CORE_PERMUTATION_ACTION_ID,
    ROUTED_ACTION_ID as CORE_ROUTED_ACTION_ID,
    CaseBootstrapPlan,
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
    TARGET_FEATURE_LOCK_KEYS,
    TARGET_POLICY_KEYS,
    TARGET_POLICY_LOCK_SCHEMA,
    TARGET_POLICY_SHARED_KEYS,
    UTILITY_POLICY_KEYS,
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
    ):
        raise ProtocolError("Utility-aligned target-policy lock identity drifted.")
    feature_locks, plans = _validate_feature_locks(
        target_lock.get("target_feature_locks"), support=support
    )
    _validate_policy_grid(
        target_lock.get("policies"),
        feature_locks=feature_locks,
        plans=plans,
        frozen_actions=frozen_actions,
    )


def _validate_feature_locks(
    raw_locks: object,
    *,
    support: Mapping[str, tuple[str, ...]],
) -> tuple[Mapping[str, Mapping[str, object]], Mapping[str, CaseBootstrapPlan]]:
    if (
        not isinstance(raw_locks, Sequence)
        or isinstance(raw_locks, (str, bytes))
        or len(raw_locks) != len(CENTERS)
    ):
        raise ProtocolError("Utility-aligned target-feature locks are incomplete.")
    feature_locks: dict[str, Mapping[str, object]] = {}
    plans: dict[str, CaseBootstrapPlan] = {}
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
            or lock.get("target_feature_row_count") != 72
            or lock.get("candidate_sources") != list(legal_sources(target))
            or lock.get("training_seeds") != list(TRAINING_SEEDS)
            or lock.get("generation_seeds") != list(GENERATION_SEEDS)
            or lock.get("case_level_resampling") is not True
            or lock.get("labels_used") is not False
        ):
            raise ProtocolError("Utility-aligned typed target-feature geometry drifted.")
        feature_locks[target] = MappingProxyType(lock)
        plans[target] = plan
    return MappingProxyType(feature_locks), MappingProxyType(plans)


def _validate_policy_grid(
    raw_policies: object,
    *,
    feature_locks: Mapping[str, Mapping[str, object]],
    plans: Mapping[str, CaseBootstrapPlan],
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
        _validate_policy(
            {str(key): value for key, value in raw.items()},
            target=target,
            proposed=proposed,
            feature_lock=feature_locks[target],
            plan=plans[target],
            frozen_action=frozen_actions[(target, fresh_id[proposed])],
        )


def _validate_policy(
    payload: Mapping[str, object],
    *,
    target: str,
    proposed: str,
    feature_lock: Mapping[str, object],
    plan: CaseBootstrapPlan,
    frozen_action: Mapping[str, object],
) -> None:
    if set(payload) != set(UTILITY_POLICY_KEYS):
        raise ProtocolError("Utility-aligned core policy fields drifted.")
    require_hash(payload, "policy_hash", "core target policy")
    candidates = tuple(legal_sources(target))
    role, global_only = {
        CORE_GLOBAL_ACTION_ID: ("global_source_quality_only", True),
        CORE_ROUTED_ACTION_ID: ("target_source_interaction", False),
        CORE_PERMUTATION_ACTION_ID: ("cyclic_feature_permutation_control", False),
    }[proposed]
    selected = payload.get("selected_source")
    chosen = payload.get("action_id")
    fallback = payload.get("used_exact_base_fallback")
    reason = payload.get("fallback_reason")
    if (
        payload.get("schema_version") != "midogpp_utility_aligned_policy_v1"
        or payload.get("target_id") != target
        or payload.get("candidate_sources") != list(candidates)
        or payload.get("router_kind") != role
        or payload.get("proposed_action_id") != proposed
        or payload.get("proposed_source") not in candidates
        or chosen not in {CORE_BASE_ACTION_ID, proposed}
        or payload.get("global_only") is not global_only
        or payload.get("target_support_labels_used") is not False
        or payload.get("target_evaluation_used") is not False
        or payload.get("seed_selection_performed") is not False
        or payload.get("abstention_semantics") != ABSTENTION_SEMANTICS
        or payload.get("support_case_count") != len(plan.support_case_ids)
        or payload.get("minimum_support_case_count") != 8
        or payload.get("seed_pair_count") != 9
        or not all(sha256_like(payload.get(key)) for key in (
            "model_hash", "feature_surface_hash", "cardinality_eligibility_hash"
        ))
        or not all(_finite(payload.get(key)) for key in (
            "predicted_gain", "standard_error", "lower_confidence_bound",
            "confidence_multiplier", "minimum_gain", "replicate_standard_deviation",
            "support_bootstrap_standard_deviation",
        ))
    ):
        raise ProtocolError("Utility-aligned core policy identity drifted.")
    if (
        (chosen == CORE_BASE_ACTION_ID) != (fallback is True)
        or (fallback is True and (selected is not None or not isinstance(reason, str) or not reason))
        or (fallback is False and (selected not in candidates or reason is not None))
        or frozen_action.get("selected_source") != selected
        or frozen_action.get("abstained_to_base") is not (fallback is True)
        or frozen_action.get("fallback_reason") != reason
    ):
        raise ProtocolError("Utility-aligned action/core-policy binding drifted.")
    permutation_seed = payload.get("permutation_seed")
    if (
        proposed == CORE_PERMUTATION_ACTION_ID
        and (isinstance(permutation_seed, bool) or not isinstance(permutation_seed, int))
    ) or (proposed != CORE_PERMUTATION_ACTION_ID and permutation_seed is not None):
        raise ProtocolError("Utility-aligned permutation policy drifted.")
    if proposed in {CORE_GLOBAL_ACTION_ID, CORE_ROUTED_ACTION_ID} and (
        payload.get("feature_surface_hash") != feature_lock["target_feature_surface_hash"]
    ):
        raise ProtocolError("Utility-aligned point feature-surface binding drifted.")
    if proposed == CORE_GLOBAL_ACTION_ID:
        if (
            payload.get("support_bootstrap_replicates") != 0
            or payload.get("support_bootstrap_surface_hashes") != []
            or payload.get("case_bootstrap_replicate_hashes") != []
            or payload.get("case_bootstrap_plan_hash") is not None
        ):
            raise ProtocolError("Global ablation consumed target bootstrap features.")
        return
    surfaces = _hash_sequence(payload.get("support_bootstrap_surface_hashes"), "policy bootstrap surfaces")
    replicates = _hash_sequence(payload.get("case_bootstrap_replicate_hashes"), "policy bootstrap replicates")
    if (
        payload.get("support_bootstrap_replicates") != plan.replicate_count
        or payload.get("minimum_support_bootstrap_replicates") != 32
        or sorted(surfaces) != sorted(feature_lock["bootstrap_surface_hashes"])
        or sorted(replicates) != sorted(item.replicate_hash for item in plan.replicates)
        or payload.get("case_bootstrap_plan_hash") != plan.plan_hash
    ):
        raise ProtocolError("Utility-aligned core policy bootstrap binding drifted.")


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
