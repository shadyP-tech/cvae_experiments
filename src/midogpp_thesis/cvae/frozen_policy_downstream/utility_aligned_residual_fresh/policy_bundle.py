"""Admission orchestration for a completed frozen Stage-60 policy bundle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import canonical_sha256
from .config import POLICY_ARTIFACT_ID, RESERVATION_ARTIFACT_ID, UtilityAlignedResidualFreshConfig
from .contracts import CENTERS, FrozenActionPayload, expected_action_ids, legal_sources
from .planning import build_evaluation_plan
from .policy_io import (
    case_mapping,
    read_json,
    require_disjoint_cases,
    require_hash,
    sha256_like,
    upstream_hash_like,
)
from .policy_schema import (
    ACTION_KEYS,
    ACTION_LIBRARY_SCHEMA,
    LIBRARY_KEYS,
    POLICY_EXPERIMENT_ID,
    POLICY_KEYS,
    POLICY_LOCK_SCHEMA,
    SHARED_BINDING_KEYS,
    UPSTREAM_ARTIFACT_HASH_KEYS,
)
from .policy_target_validation import validate_target_policy_lock


TARGET_SUPPORT_RESERVATION_ARTIFACT_ID = (
    "midogpp_utility_aligned_target_support_reservation_v1"
)


@dataclass(frozen=True)
class FrozenUtilityAlignedPolicySurface:
    policy_lock_hash: str
    action_library_hash: str
    exact_tail_utility_surface_lock_hash: str
    reservation_id: str
    reservation_hash: str
    support_case_ids_by_target: Mapping[str, tuple[str, ...]]
    evaluation_case_ids_by_target: Mapping[str, tuple[str, ...]]
    actions_by_target: Mapping[str, tuple[FrozenActionPayload, ...]]
    raw_actions_by_key: Mapping[tuple[str, str], Mapping[str, object]]
    policy_payload: Mapping[str, object]


def load_frozen_utility_aligned_policy(
    config: UtilityAlignedResidualFreshConfig,
) -> FrozenUtilityAlignedPolicySurface:
    root = config.policy_root
    library = read_json(root / "manifests/action_library.json")
    require_hash(library, "action_library_hash", "action library")
    _validate_library_identity(library)
    parsed, raw_by_key = _load_action_grid(library)
    build_evaluation_plan(parsed)

    policy = read_json(root / "manifests/policy_lock.json")
    require_hash(policy, "policy_lock_hash", "policy lock")
    _validate_final_policy(policy, library)
    support = case_mapping(
        policy.get("target_support_case_ids_by_target"), role="policy support"
    )
    evaluation = case_mapping(
        policy.get("target_evaluation_case_ids_by_target"), role="policy evaluation"
    )
    require_disjoint_cases(support, evaluation)
    _validate_development_case_manifest(
        policy,
        support=support,
        evaluation=evaluation,
    )
    validate_target_policy_lock(
        root,
        policy=policy,
        support=support,
        evaluation=evaluation,
        frozen_actions=raw_by_key,
    )
    _validate_completion(root, policy_lock_hash=str(policy["policy_lock_hash"]))
    return FrozenUtilityAlignedPolicySurface(
        policy_lock_hash=str(policy["policy_lock_hash"]),
        action_library_hash=str(library["action_library_hash"]),
        exact_tail_utility_surface_lock_hash=str(library["exact_tail_surface_lock_hash"]),
        reservation_id=RESERVATION_ARTIFACT_ID,
        reservation_hash=str(policy["target_reservation_hash"]),
        support_case_ids_by_target=support,
        evaluation_case_ids_by_target=evaluation,
        actions_by_target=MappingProxyType(parsed),
        raw_actions_by_key=MappingProxyType(raw_by_key),
        policy_payload=MappingProxyType(dict(policy)),
    )


def _validate_library_identity(library: Mapping[str, object]) -> None:
    if (
        set(library) != set(LIBRARY_KEYS)
        or library.get("schema_version") != ACTION_LIBRARY_SCHEMA
        or library.get("experiment_id") != POLICY_EXPERIMENT_ID
        or library.get("output_artifact_id") != POLICY_ARTIFACT_ID
        or not upstream_hash_like(library.get("exact_tail_surface_lock_hash"))
        or library.get("target_support_parent_reservation_artifact_id")
        != TARGET_SUPPORT_RESERVATION_ARTIFACT_ID
        or library.get("target_reservation_artifact_id") != RESERVATION_ARTIFACT_ID
    ):
        raise ProtocolError("Utility-aligned action-library identity drifted.")
    for key in LIBRARY_KEYS:
        if (key.endswith("_hash") or key.endswith("_sha256")) and key != "action_library_hash":
            valid = (
                upstream_hash_like(library.get(key))
                if key in UPSTREAM_ARTIFACT_HASH_KEYS
                else sha256_like(library.get(key))
            )
            if not valid:
                raise ProtocolError(f"Utility-aligned action-library hash drifted: {key}.")


def _validate_development_case_manifest(
    policy: Mapping[str, object],
    *,
    support: Mapping[str, tuple[str, ...]],
    evaluation: Mapping[str, tuple[str, ...]],
) -> None:
    development_support = case_mapping(
        policy.get("development_support_case_ids_by_query"),
        role="development support",
        minimum_count=1,
    )
    development_evaluation = case_mapping(
        policy.get("development_evaluation_case_ids_by_query"),
        role="development evaluation",
        minimum_count=1,
    )
    development_target_evaluation = case_mapping(
        policy.get("development_target_evaluation_case_ids_by_target"),
        role="development target evaluation",
        minimum_count=1,
    )
    require_disjoint_cases(development_support, development_evaluation)
    if dict(development_target_evaluation) != dict(evaluation):
        raise ProtocolError(
            "Exact-tail excluded target-evaluation cases differ from Stage-70 policy cases."
        )
    development_opened = {
        case
        for mapping in (development_support, development_evaluation)
        for target in CENTERS
        for case in mapping[target]
    }
    fresh_target = {
        case
        for mapping in (support, evaluation)
        for target in CENTERS
        for case in mapping[target]
    }
    if development_opened.intersection(fresh_target):
        raise ProtocolError(
            "Exact-tail development cases overlap fresh target support/evaluation."
        )
    partition_hashes = policy.get("development_partition_hashes_by_query")
    if (
        not isinstance(partition_hashes, Mapping)
        or {str(key) for key in partition_hashes} != set(CENTERS)
        or any(
            not upstream_hash_like(partition_hashes.get(target))
            for target in CENTERS
        )
    ):
        raise ProtocolError("Exact-tail development partition hashes drifted.")
    manifest_payload = {
        "schema_version": "midogpp_exact_tail_development_case_manifest_v1",
        "reservation_hash": policy.get("development_reservation_hash"),
        "support_case_ids_by_center": {
            target: list(development_support[target]) for target in CENTERS
        },
        "evaluation_case_ids_by_center": {
            target: list(development_evaluation[target]) for target in CENTERS
        },
        "target_evaluation_case_ids_by_center": {
            target: list(development_target_evaluation[target]) for target in CENTERS
        },
        "partition_hashes_by_center": {
            target: partition_hashes[target] for target in CENTERS
        },
    }
    if policy.get("development_case_manifest_hash") != canonical_sha256(
        manifest_payload
    ):
        raise ProtocolError("Exact-tail development case manifest hash drifted.")


def _load_action_grid(
    library: Mapping[str, object],
) -> tuple[
    dict[str, tuple[FrozenActionPayload, ...]],
    dict[tuple[str, str], Mapping[str, object]],
]:
    raw_actions = library.get("actions")
    if not isinstance(raw_actions, Sequence) or isinstance(raw_actions, (str, bytes)):
        raise ProtocolError("Utility-aligned action library has no target menu.")
    expected = tuple(
        (target, action_id)
        for target in CENTERS
        for action_id in expected_action_ids(target)
    )
    if (
        len(raw_actions) != len(expected)
        or library.get("action_count") != len(expected)
        or library.get("action_ids")
        != [str(raw["action_id"]) for raw in raw_actions if isinstance(raw, Mapping)]
    ):
        raise ProtocolError("Utility-aligned action-library coverage drifted.")
    accumulated: dict[str, list[FrozenActionPayload]] = {target: [] for target in CENTERS}
    raw_by_key: dict[tuple[str, str], Mapping[str, object]] = {}
    for (target, expected_id), raw_action in zip(expected, raw_actions, strict=True):
        if not isinstance(raw_action, Mapping) or set(raw_action) != set(ACTION_KEYS):
            raise ProtocolError("Utility-aligned frozen action fields drifted.")
        raw = {str(key): value for key, value in raw_action.items()}
        if (
            raw.get("target_center") != target
            or raw.get("action_id") != expected_id
            or raw.get("target_labels_used") is not False
            or raw.get("support_labels_used") is not False
            or list(raw.get("source_order", ())) != list(legal_sources(target))
        ):
            raise ProtocolError("Utility-aligned frozen action identity drifted.")
        topup_hash = raw.get("topup_action_hash")
        if not sha256_like(raw.get("decision_hash")) or (
            topup_hash is not None and not sha256_like(topup_hash)
        ):
            raise ProtocolError("Utility-aligned frozen action hash drifted.")
        action = FrozenActionPayload(
            target_center=target,
            action_id=expected_id,
            action_role=str(raw.get("action_role", "")),
            source_counts_by_class=_class_counts(raw.get("counts_per_class")),
            action_hash=str(raw.get("decision_hash", "")),
            selected_source=(None if raw.get("selected_source") is None else str(raw["selected_source"])),
            abstained_to_base=raw.get("abstained_to_base") is True,
            fallback_reason=(None if raw.get("fallback_reason") is None else str(raw["fallback_reason"])),
        )
        if (
            raw.get("total_per_class") != action.budget_per_class
            or (topup_hash is not None) is not (action.budget_per_class == 1152)
        ):
            raise ProtocolError("Utility-aligned frozen action budget drifted.")
        accumulated[target].append(action)
        raw_by_key[(target, expected_id)] = MappingProxyType(raw)
    return (
        {target: tuple(accumulated[target]) for target in CENTERS},
        raw_by_key,
    )


def _class_counts(value: object) -> Mapping[object, Mapping[object, object]]:
    if not isinstance(value, Mapping):
        raise ProtocolError("Utility-aligned counts_per_class is malformed.")
    if set(str(key) for key in value) == {"0", "1"} and all(
        isinstance(item, Mapping) for item in value.values()
    ):
        return value  # type: ignore[return-value]
    raise ProtocolError("Utility-aligned counts_per_class is malformed.")


def _validate_final_policy(
    policy: Mapping[str, object], library: Mapping[str, object]
) -> None:
    if (
        set(policy) != set(POLICY_KEYS)
        or policy.get("schema_version") != POLICY_LOCK_SCHEMA
        or policy.get("experiment_id") != POLICY_EXPERIMENT_ID
        or policy.get("output_artifact_id") != POLICY_ARTIFACT_ID
        or policy.get("action_library_hash") != library["action_library_hash"]
        or any(policy.get(key) != library.get(key) for key in SHARED_BINDING_KEYS)
        or policy.get("candidate_centers") != list(CENTERS)
        or policy.get("primary_contrasts") != ["R-B", "R-G_delta", "R-U"]
        or policy.get("permutation_contrast") != "R-P"
        or policy.get("success_requires_positive_one_sided_lcb")
        != ["R-B", "R-G_delta", "R-U", "R-P"]
        or not isinstance(policy.get("policy_family"), str)
        or not policy.get("policy_family")
        or policy.get("fallback_policy") != "exact_B"
        or policy.get("outer_target_excluded_from_fit") is not True
        or policy.get("target_support_labels_used") is not False
        or policy.get("target_evaluation_labels_used") is not False
        or policy.get("seed_selection_performed") is not False
        or policy.get("minimum_independent_support_cases_per_target") != 8
        or not isinstance(policy.get("support_bootstrap_count"), int)
        or int(policy["support_bootstrap_count"]) < 32
    ):
        raise ProtocolError("Utility-aligned Stage-60 policy-lock identity drifted.")


def _validate_completion(root: Path, *, policy_lock_hash: str) -> None:
    state = read_json(root / "reports/run_state.json")
    validation = read_json(root / "reports/validation_report.json")
    checks = validation.get("checks")
    if (
        state.get("status") != "COMPLETE"
        or validation.get("status") != "PASS"
        or not isinstance(checks, Mapping)
        or checks.get("status") != "PASS"
        or checks.get("policy_lock_hash") != policy_lock_hash
    ):
        raise ProtocolError("Utility-aligned policy validation authorization drifted.")


__all__ = ("FrozenUtilityAlignedPolicySurface", "load_frozen_utility_aligned_policy")
