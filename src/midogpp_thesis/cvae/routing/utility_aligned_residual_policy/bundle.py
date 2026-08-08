"""Persistence and closed-world validation for the locked policy bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from ...protocol import ProtocolError
from ..residual_topup.hashing import canonical_sha256
from .contracts import (
    ACTION_LIBRARY_SCHEMA,
    EXPECTED_ACTION_COUNT,
    EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID,
    POLICY_LOCK_SCHEMA,
    TARGET_POLICY_LOCK_SCHEMA,
)
from .config import (
    UtilityAlignedResidualPolicyConfig,
    load_utility_aligned_residual_policy_config,
)
from .inputs import load_policy_inputs
from .policy_building import BuiltPolicyBundle, build_policy_bundle


REQUIRED_FILES = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "manifests/model_lock.json",
    "manifests/global_ablation_lock.json",
    "manifests/cardinality_transfer_lock.json",
    "manifests/target_policy_lock.json",
    "manifests/action_library.json",
    "manifests/policy_lock.json",
    "manifests/content_index.json",
    "reports/leakage_report.json",
    "reports/run_state.json",
    "reports/validation_report.json",
)


def persist_policy_bundle(
    root: Path,
    bundle: BuiltPolicyBundle,
    *,
    config: UtilityAlignedResidualPolicyConfig,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for member in ("config.resolved.yaml", "provenance/input_artifacts.json"):
        if not (root / member).is_file():
            raise ProtocolError(
                f"Utility-aligned workspace did not prepare required member: {member}."
            )
    for member, payload in (
        ("manifests/model_lock.json", bundle.model_lock),
        ("manifests/global_ablation_lock.json", bundle.global_ablation_lock),
        ("manifests/cardinality_transfer_lock.json", bundle.cardinality_transfer_lock),
        ("manifests/target_policy_lock.json", bundle.target_policy_lock),
        ("manifests/action_library.json", bundle.action_library),
        ("manifests/policy_lock.json", bundle.policy_lock),
    ):
        _atomic_json(root / member, payload)
    policy_hash = str(bundle.policy_lock["policy_lock_hash"])
    _atomic_json(
        root / "reports/leakage_report.json",
        {
            "schema_version": "midogpp_utility_aligned_policy_leakage_report_v1",
            "status": "PASS",
            "policy_lock_hash": policy_hash,
            "outer_target_excluded_from_fit": True,
            "target_support_labels_used": False,
            "target_evaluation_labels_used": False,
            "query_and_case_clusters_are_uncertainty_units": True,
            "seed_selection_performed": False,
        },
    )
    _atomic_json(
        root / "reports/run_state.json",
        {
            "schema_version": "midogpp_utility_aligned_policy_run_state_v1",
            "status": "COMPLETE",
            "policy_lock_hash": policy_hash,
            "target_policy_count": 27,
            "action_count": EXPECTED_ACTION_COUNT,
        },
    )
    # Re-open every fresh upstream and independently rebuild every scientific
    # lock before a PASS report exists.  A coherently rehashed local edit is
    # therefore not self-authorizing.
    _validate_reconstruction(root, config=config)
    _atomic_json(root / "reports/validation_report.json", _validation_payload(policy_hash))
    members = {
        member: _sha256_file(root / member)
        for member in REQUIRED_FILES
        if member != "manifests/content_index.json"
    }
    content_payload = {
        "schema_version": "midogpp_utility_aligned_policy_content_index_v1",
        "member_sha256": members,
        "policy_lock_hash": policy_hash,
    }
    content_payload["content_index_hash"] = canonical_sha256(content_payload)
    _atomic_json(root / "manifests/content_index.json", content_payload)
    return root


def validate_policy_bundle(
    root: str | Path,
    *,
    config: UtilityAlignedResidualPolicyConfig | None = None,
) -> Mapping[str, object]:
    path = Path(root)
    discovered = tuple(path.rglob("*")) if path.exists() else ()
    if any(member.is_symlink() for member in discovered):
        raise ProtocolError("Utility-aligned policy bundle forbids symbolic links.")
    actual = {
        str(member.relative_to(path)) for member in discovered if member.is_file()
    }
    if actual != set(REQUIRED_FILES):
        raise ProtocolError(
            "Utility-aligned policy bundle is not closed-world complete: "
            f"missing={sorted(set(REQUIRED_FILES)-actual)}, "
            f"extra={sorted(actual-set(REQUIRED_FILES))}."
        )
    content = _json(path / "manifests/content_index.json")
    expected_content_keys = {
        "schema_version",
        "member_sha256",
        "policy_lock_hash",
        "content_index_hash",
    }
    if set(content) != expected_content_keys:
        raise ProtocolError("Utility-aligned content-index schema drifted.")
    unhashed = {key: value for key, value in content.items() if key != "content_index_hash"}
    if (
        content.get("schema_version")
        != "midogpp_utility_aligned_policy_content_index_v1"
        or content.get("content_index_hash") != canonical_sha256(unhashed)
    ):
        raise ProtocolError("Utility-aligned content-index hash drifted.")
    member_sha = content.get("member_sha256")
    if not isinstance(member_sha, Mapping) or set(member_sha) != set(REQUIRED_FILES) - {
        "manifests/content_index.json"
    }:
        raise ProtocolError("Utility-aligned content-index coverage drifted.")
    for member, digest in member_sha.items():
        if not _is_sha256(digest) or _sha256_file(path / str(member)) != digest:
            raise ProtocolError("Utility-aligned policy member bytes drifted.")
    action = _json(path / "manifests/action_library.json")
    policy = _json(path / "manifests/policy_lock.json")
    target = _json(path / "manifests/target_policy_lock.json")
    for payload, key, schema in (
        (action, "action_library_hash", ACTION_LIBRARY_SCHEMA),
        (policy, "policy_lock_hash", POLICY_LOCK_SCHEMA),
        (target, "target_policy_lock_hash", TARGET_POLICY_LOCK_SCHEMA),
    ):
        unhashed_payload = {name: value for name, value in payload.items() if name != key}
        if payload.get("schema_version") != schema or payload.get(key) != canonical_sha256(
            unhashed_payload
        ):
            raise ProtocolError("Utility-aligned lock semantic hash drifted.")
    if (
        policy.get("experiment_id") != EXPERIMENT_ID
        or policy.get("output_artifact_id") != OUTPUT_ARTIFACT_ID
        or policy.get("action_library_hash") != action.get("action_library_hash")
        or policy.get("target_policy_lock_hash") != target.get("target_policy_lock_hash")
        or content.get("policy_lock_hash") != policy.get("policy_lock_hash")
        or action.get("action_count") != EXPECTED_ACTION_COUNT
    ):
        raise ProtocolError("Utility-aligned policy bundle identity drifted.")
    resolved = load_utility_aligned_residual_policy_config(
        path / "config.resolved.yaml"
    )
    effective = resolved if config is None else config
    if config is not None and resolved != config:
        raise ProtocolError("Utility-aligned resolved config drifted from the run config.")
    _validate_reconstruction(path, config=effective)
    if _json(path / "reports/validation_report.json") != _validation_payload(
        str(policy["policy_lock_hash"])
    ):
        raise ProtocolError("Utility-aligned reconstructive validation report drifted.")
    return policy


def _validate_reconstruction(
    root: Path, *, config: UtilityAlignedResidualPolicyConfig
) -> None:
    # Validation deliberately avoids a second ProcessPool: the completed-run
    # fast path still reconstructs every model/lock, serially and deterministically.
    rebuilt = build_policy_bundle(
        config, load_policy_inputs(config), spawn_workers=False
    )
    expected_by_member = {
        "manifests/model_lock.json": rebuilt.model_lock,
        "manifests/global_ablation_lock.json": rebuilt.global_ablation_lock,
        "manifests/cardinality_transfer_lock.json": rebuilt.cardinality_transfer_lock,
        "manifests/target_policy_lock.json": rebuilt.target_policy_lock,
        "manifests/action_library.json": rebuilt.action_library,
        "manifests/policy_lock.json": rebuilt.policy_lock,
    }
    for member, expected in expected_by_member.items():
        if _json(root / member) != dict(expected):
            raise ProtocolError(
                f"Utility-aligned {member} drifted from fresh-input reconstruction."
            )


def _validation_payload(policy_hash: str) -> dict[str, object]:
    return {
        "schema_version": "midogpp_utility_aligned_policy_validation_report_v1",
        "status": "PASS",
        "checks": {
            "status": "PASS",
            "policy_lock_hash": policy_hash,
            "fresh_inputs_reloaded": True,
            "models_reconstructed": True,
            "cardinality_transfer_reconstructed": True,
            "typed_case_bootstrap_plan_validated": True,
            "target_feature_geometry_validated": True,
            "bootstrap_surfaces_validated": True,
            "target_policies_reconstructed": True,
            "action_library_reconstructed": True,
            "canonical_equal_union_bound": True,
        },
    }


def _json(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read utility-aligned policy JSON: {path}.") from exc
    if not isinstance(raw, dict):
        raise ProtocolError("Utility-aligned policy JSON must be an object.")
    return raw


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    rendered = str(value or "")
    return len(rendered) == 64 and all(char in "0123456789abcdef" for char in rendered)


__all__ = ("REQUIRED_FILES", "persist_policy_bundle", "validate_policy_bundle")
