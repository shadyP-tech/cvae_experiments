"""Independent Stage-60 policy admission and pure-action reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from ...protocol import ProtocolError
from ...routing.residual_topup import (
    ResidualTopupAction,
    build_borda_directed_topup_action,
    build_single_source_tail_action,
    build_uniform_topup_action,
    target_topup_geometry,
)
from ...routing.residual_topup.hashing import canonical_sha256
from .config import ResidualTopupFreshConfig
from .contracts import (
    BASE_ACTION_ID,
    CENTERS,
    EXPECTED_ACTION_COUNT_PER_TARGET,
    FrozenActionPayload,
    GLOBAL_ACTION_ID,
    PERMUTATION_ACTION_ID,
    SUPPORT_ACTION_ID,
    UNIFORM_ACTION_ID,
    expected_action_ids,
    legal_sources,
    tail_source,
)


ACTION_LIBRARY_SCHEMA = "midogpp_residual_topup_frozen_action_library_v1"
ACTION_SCHEMA = "midogpp_residual_topup_frozen_policy_action_v1"
POLICY_EXPERIMENT_ID = (
    "midogpp.routing_and_composition."
    "uniform_b_v2_residual_topup_b_u_g_s_policy_lock.v1"
)

_ACTION_KEYS = frozenset(
    {
        "schema_version",
        "outer_target",
        "action_id",
        "policy_id",
        "action_kind",
        "action_semantics",
        "source_order",
        "base_per_source_per_class",
        "topup_total_per_class",
        "final_total_per_class",
        "mean_normalized_midrank_by_source",
        "source_identity_permutation",
        "selected_source",
        "direction_weights_by_source",
        "topup_counts_by_source",
        "final_counts_by_class",
        "core_action_kind",
        "core_action_hash",
        "diagnostic_control",
        "action_hash",
    }
)
_POLICY_KEYS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "output_artifact_id",
        "claim_scope",
        "config_contract_hash",
        "input_artifact_ids",
        "bank_lock_hash",
        "generation_lock_hash",
        "equal_union_policy_lock_hash",
        "proxy_surface_artifact_id",
        "fresh_surface_reservation_id",
        "fresh_surface_attestation_hash",
        "fresh_surface_attestation_file_sha256",
        "proxy_score_table_sha256",
        "pseudoquery_case_ids_by_center",
        "support_case_ids_by_target",
        "evaluation_case_ids_by_target",
        "workspace_binding",
        "rank_summary_hash",
        "action_library_hash",
        "action_count",
        "actions_by_target",
        "policy_frozen_before_stage70",
        "all_main_and_control_actions_frozen",
        "all_H_by_e_actions_frozen",
        "proxy_only",
        "labels_consumed",
        "target_evaluation_used",
        "source_experts_updated",
        "consumed_stage70_used",
        "consumed_stage90_used",
        "hyperparameters_tuned",
        "routing_quality_claimed",
        "downstream_outcome_computed",
        "may_feed_stage70_only_after_validation_pass",
        "policy_lock_hash",
    }
)


@dataclass(frozen=True)
class FrozenPolicySurface:
    policy_lock_hash: str
    action_library_hash: str
    actions_by_target: Mapping[str, tuple[FrozenActionPayload, ...]]
    raw_actions_by_key: Mapping[tuple[str, str], Mapping[str, object]]
    policy_payload: Mapping[str, object]


def load_frozen_policy_actions(
    config: ResidualTopupFreshConfig,
) -> FrozenPolicySurface:
    """Validate the Stage-60 lock/library without importing Stage-60 code."""

    root = config.policy_root
    library = _json(root / "manifests/action_library.json")
    if set(library) != {
        "schema_version",
        "centers",
        "action_count",
        "actions_by_target",
        "policy_frozen_before_stage70",
        "action_library_hash",
    }:
        raise ProtocolError("Fresh Stage-70 action-library fields drifted.")
    _require_canonical_hash(library, "action_library_hash", "action library")
    raw_by_target = library.get("actions_by_target")
    if (
        library.get("schema_version") != ACTION_LIBRARY_SCHEMA
        or library.get("centers") != list(CENTERS)
        or library.get("action_count")
        != len(CENTERS) * EXPECTED_ACTION_COUNT_PER_TARGET
        or library.get("policy_frozen_before_stage70") is not True
        or not isinstance(raw_by_target, Mapping)
        or tuple(str(key) for key in raw_by_target) != CENTERS
    ):
        raise ProtocolError("Fresh Stage-70 action-library identity drifted.")

    actions_by_target: dict[str, tuple[FrozenActionPayload, ...]] = {}
    raw_by_key: dict[tuple[str, str], Mapping[str, object]] = {}
    normalized_targets = {str(key): value for key, value in raw_by_target.items()}
    for target in CENTERS:
        raw_actions = normalized_targets[target]
        if (
            not isinstance(raw_actions, list)
            or len(raw_actions) != EXPECTED_ACTION_COUNT_PER_TARGET
        ):
            raise ProtocolError("Fresh Stage-70 target action library is incomplete.")
        expected_ids = expected_action_ids(target)
        parsed: list[FrozenActionPayload] = []
        for expected_id, raw in zip(expected_ids, raw_actions, strict=True):
            if not isinstance(raw, Mapping) or set(raw) != _ACTION_KEYS:
                raise ProtocolError("Fresh Stage-70 frozen action fields drifted.")
            action = {str(key): value for key, value in raw.items()}
            _require_canonical_hash(action, "action_hash", "frozen action")
            if (
                action.get("schema_version") != ACTION_SCHEMA
                or action.get("outer_target") != target
                or action.get("action_id") != expected_id
                or action.get("source_order") != list(legal_sources(target))
                or action.get("base_per_source_per_class") != 128
                or action.get("final_total_per_class")
                != (1024 if expected_id == BASE_ACTION_ID else 1152)
            ):
                raise ProtocolError("Fresh Stage-70 frozen action identity drifted.")
            payload = FrozenActionPayload(
                target_center=target,
                action_id=expected_id,
                source_counts_by_class=action["final_counts_by_class"],  # type: ignore[arg-type]
                action_hash=str(action["action_hash"]),
                mean_normalized_midrank_by_source=action[
                    "mean_normalized_midrank_by_source"
                ],  # type: ignore[arg-type]
                source_identity_permutation=action[
                    "source_identity_permutation"
                ],  # type: ignore[arg-type]
            )
            rebuild_and_validate_core_action(payload, action)
            parsed.append(payload)
            raw_by_key[(target, expected_id)] = MappingProxyType(dict(action))
        actions_by_target[target] = tuple(parsed)

    policy = _json(root / "manifests/policy_lock.json")
    if set(policy) != _POLICY_KEYS:
        raise ProtocolError("Fresh Stage-70 policy-lock fields drifted.")
    _require_canonical_hash(policy, "policy_lock_hash", "policy lock")
    if (
        policy.get("schema_version")
        != "midogpp_residual_topup_b_u_g_s_policy_lock_v1"
        or policy.get("experiment_id") != POLICY_EXPERIMENT_ID
        or policy.get("claim_scope") != "routing_and_composition"
        or policy.get("bank_lock_hash") != config.expected_bank_lock_hash
        or policy.get("generation_lock_hash")
        != config.expected_generation_lock_hash
        or policy.get("action_library_hash") != library["action_library_hash"]
        or policy.get("actions_by_target") != library["actions_by_target"]
        or policy.get("action_count")
        != len(CENTERS) * EXPECTED_ACTION_COUNT_PER_TARGET
        or policy.get("policy_frozen_before_stage70") is not True
        or policy.get("all_main_and_control_actions_frozen") is not True
        or policy.get("all_H_by_e_actions_frozen") is not True
        or policy.get("proxy_only") is not True
        or policy.get("may_feed_stage70_only_after_validation_pass") is not True
    ):
        raise ProtocolError("Fresh Stage-70 policy-lock identity drifted.")
    _require_negative_policy_attestations(policy)
    _validate_policy_artifact_completion(
        root,
        policy_lock_hash=str(policy["policy_lock_hash"]),
        action_library_hash=str(library["action_library_hash"]),
    )
    return FrozenPolicySurface(
        policy_lock_hash=str(policy["policy_lock_hash"]),
        action_library_hash=str(library["action_library_hash"]),
        actions_by_target=MappingProxyType(actions_by_target),
        raw_actions_by_key=MappingProxyType(raw_by_key),
        policy_payload=MappingProxyType(dict(policy)),
    )


def rebuild_and_validate_core_action(
    frozen: FrozenActionPayload,
    raw: Mapping[str, object],
) -> ResidualTopupAction | None:
    """Rebuild U/G/S/P/Hxe from pure residual-topup primitives."""

    sources = legal_sources(frozen.target_center)
    geometry = target_topup_geometry(sources)
    if frozen.action_id == BASE_ACTION_ID:
        core = None
        if (
            raw.get("core_action_kind") is not None
            or raw.get("core_action_hash") is not None
        ):
            raise ProtocolError("Fresh base action unexpectedly carries a top-up core.")
    elif frozen.action_id == UNIFORM_ACTION_ID:
        core = build_uniform_topup_action(geometry)
    elif frozen.action_id in {
        GLOBAL_ACTION_ID,
        SUPPORT_ACTION_ID,
        PERMUTATION_ACTION_ID,
    }:
        core = build_borda_directed_topup_action(
            frozen.mean_normalized_midrank_by_source,
            geometry=geometry,
        )
    else:
        selected = tail_source(frozen.action_id)
        if selected is None:
            raise ProtocolError("Fresh frozen action identity is unknown.")
        core = build_single_source_tail_action(selected, geometry=geometry)
        if raw.get("selected_source") != selected:
            raise ProtocolError("Fresh single-source-tail identity drifted.")
    expected_topup = (
        {source: 0 for source in sources}
        if core is None
        else dict(core.topup_counts)
    )
    expected_final = {
        str(label): {
            source: 128 + expected_topup[source] for source in sources
        }
        for label in (0, 1)
    }
    if (
        raw.get("topup_counts_by_source") != expected_topup
        or raw.get("final_counts_by_class") != expected_final
        or (
            core is not None
            and (
                raw.get("core_action_kind") != core.action_kind
                or raw.get("core_action_hash") != core.action_hash
                or raw.get("direction_weights_by_source")
                != dict(core.direction_weights)
            )
        )
    ):
        raise ProtocolError("Fresh frozen action failed pure-core reconstruction.")
    return core


def _require_negative_policy_attestations(payload: Mapping[str, object]) -> None:
    required_false = (
        "labels_consumed",
        "target_evaluation_used",
        "source_experts_updated",
        "consumed_stage70_used",
        "consumed_stage90_used",
        "hyperparameters_tuned",
        "routing_quality_claimed",
        "downstream_outcome_computed",
    )
    for key in required_false:
        if payload.get(key) is not False:
            raise ProtocolError("Fresh policy-lock negative attestation failed.")
    rendered = json.dumps(payload, sort_keys=True).lower()
    if "oracle_action" in rendered or "utility_by_source" in rendered:
        raise ProtocolError("Fresh policy lock contains an oracle/utility payload.")


def _validate_policy_artifact_completion(
    root: Path,
    *,
    policy_lock_hash: str,
    action_library_hash: str,
) -> None:
    required = {
        "config.resolved.yaml",
        "provenance/input_artifacts.json",
        "manifests/fresh_surface_attestation.json",
        "manifests/protocol_manifest.json",
        "manifests/policy_lock.json",
        "manifests/action_library.json",
        "manifests/content_index.json",
        "reports/protocol_report.json",
        "reports/leakage_report.json",
        "reports/policy_decision.json",
        "reports/run_state.json",
        "reports/validation_report.json",
        "tables/proxy_ballots.csv",
        "tables/proxy_ranks.csv",
        "tables/policy_actions.csv",
    }
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual != required or any(path.is_symlink() for path in root.rglob("*")):
        raise ProtocolError(
            "Fresh Stage-70 policy artifact is not closed-world complete."
        )
    state = _json(root / "reports/run_state.json")
    validation = _json(root / "reports/validation_report.json")
    checks = validation.get("checks")
    if (
        state
        != {
            "schema_version": "midogpp_residual_topup_b_u_g_s_run_state_v1",
            "status": "COMPLETE",
            "claim_scope": "routing_and_composition",
        }
        or validation.get("schema_version")
        != "midogpp_residual_topup_b_u_g_s_validation_v1"
        or validation.get("status") != "PASS"
        or validation.get("validator")
        != "validate_residual_topup_policy_bundle"
        or not isinstance(checks, Mapping)
        or checks.get("status") != "PASS"
        or checks.get("policy_lock_hash") != policy_lock_hash
        or checks.get("action_library_hash") != action_library_hash
        or checks.get("labels_consumed") is not False
        or checks.get("target_evaluation_used") is not False
        or checks.get("source_experts_updated") is not False
    ):
        raise ProtocolError(
            "Fresh Stage-70 policy validation authorization drifted."
        )
    content = _json(root / "manifests/content_index.json")
    _require_canonical_hash(content, "content_hash", "policy content index")
    records = content.get("records")
    expected_members = required.difference(
        {
            "manifests/content_index.json",
            "reports/run_state.json",
            "reports/validation_report.json",
        }
    )
    if not isinstance(records, list) or len(records) != len(expected_members):
        raise ProtocolError("Fresh Stage-70 policy content index is incomplete.")
    observed_members: set[str] = set()
    for raw in records:
        if not isinstance(raw, Mapping):
            raise ProtocolError("Fresh Stage-70 policy content row is malformed.")
        relative = str(raw.get("relative_path", ""))
        path = _safe_member(root, relative)
        if (
            relative in observed_members
            or not path.is_file()
            or raw.get("sha256") != _sha256_file(path)
            or raw.get("size_bytes") != path.stat().st_size
        ):
            raise ProtocolError("Fresh Stage-70 policy content member drifted.")
        observed_members.add(relative)
    if observed_members != expected_members:
        raise ProtocolError("Fresh Stage-70 policy content members drifted.")


def _require_canonical_hash(
    payload: Mapping[str, object], key: str, role: str
) -> None:
    observed = payload.get(key)
    unhashed = {name: value for name, value in payload.items() if name != key}
    if observed != canonical_sha256(unhashed):
        raise ProtocolError(f"Fresh Stage-70 {role} hash drifted.")


def _json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read fresh Stage-70 JSON: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("Fresh Stage-70 JSON must be a mapping.")
    return payload


def _safe_member(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ProtocolError("Fresh policy content member escapes its root.")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = (
    "ACTION_LIBRARY_SCHEMA",
    "ACTION_SCHEMA",
    "FrozenPolicySurface",
    "load_frozen_policy_actions",
    "rebuild_and_validate_core_action",
)
