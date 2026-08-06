"""Materialize the frozen Uniform-B v2 equal-union Stage-60 policy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ...common.hashing import stable_hash
from ..protocol import ProtocolError
from ..reporting import write_csv_rows, write_json
from .bundle import CONTENT_INDEX_MEMBERS, REQUIRED_FILES
from .config import UniformBV2EqualUnionPolicyConfig
from .contracts import (
    CLAIM_SCOPE,
    EXPERIMENT_ID,
    POLICY_DECISION,
    PUBLICATION_STATE,
)
from .inputs import load_validated_inputs
from .policy import (
    assignment_rows,
    assignment_table_hash,
    build_policy_lock,
    build_policy_plan_payload,
)


def run_equal_union_policy_lock(
    config: UniformBV2EqualUnionPolicyConfig,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    """Create a lock-only routing/composition artifact; never touch target data."""

    root = Path(artifact_root or config.artifact_root)
    for relative in ("manifests", "reports", "tables", "provenance"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    _assert_closed_world(root)
    if not (root / "config.resolved.yaml").is_file() or not (
        root / "provenance/input_artifacts.json"
    ).is_file():
        raise ProtocolError(
            "Equal-union policy lock must be launched through the MIDOG++ workspace."
        )
    state_path = root / "reports/run_state.json"
    if state_path.is_file() and _json(state_path).get("status") == "COMPLETE":
        from .validation import validate_equal_union_policy_bundle

        validate_equal_union_policy_bundle(root, config=config)
        return root

    _write_state(root, "RUNNING")
    try:
        generation_lock = load_validated_inputs(config)
        plan = build_policy_plan_payload(generation_lock, config)
        assignments = assignment_rows(generation_lock, config)
        policy_lock = build_policy_lock(config, generation_lock)
        write_json(root / "manifests/policy_lock.json", policy_lock.to_payload())
        write_json(root / "manifests/equal_union_policy_plan.json", plan)

        protocol: dict[str, object] = {
            "schema_version": "midogpp_uniform_b_v2_equal_union_policy_protocol_v1",
            "experiment_id": EXPERIMENT_ID,
            "claim_scope": CLAIM_SCOPE,
            "config_contract_hash": config.contract_hash,
            "input_artifact_ids": [
                config.bank_artifact_id,
                config.generation_lock_artifact_id,
            ],
            "bank_lock_hash": config.expected_bank_lock_hash,
            "generation_lock_hash": config.expected_generation_lock_hash,
            "policy_lock_hash": policy_lock.policy_lock_hash,
            "policy_plan_hash": plan["plan_hash"],
            "assignment_table_hash": assignment_table_hash(assignments),
            "policy_frozen_before_target_evaluation": True,
            "target_identity_role": (
                "fold_identity_candidate_exclusion_and_label_blind_shuffle_seeding_only"
            ),
            "all_eligible_non_target_sources_retained": True,
            "full_training_x_generation_seed_cartesian_product": True,
            "target_data_used": False,
            "target_support_used": False,
            "target_labels_used": False,
            "expert_or_seed_selection_performed": False,
            "routing_quality_claimed": False,
            "downstream_utility_computed": False,
            "may_feed_deployable_selection": True,
        }
        protocol["protocol_hash"] = stable_hash(protocol)
        write_json(root / "manifests/protocol_manifest.json", protocol)

        write_csv_rows(
            root / "tables/policy_assignments.csv",
            [row.to_payload() for row in assignments],
        )
        write_json(
            root / "reports/policy_decision.json",
            {
                "schema_version": "midogpp_uniform_b_v2_equal_union_policy_decision_v1",
                "decision": POLICY_DECISION,
                "publication_state": PUBLICATION_STATE,
                "claim_scope": CLAIM_SCOPE,
                "policy_lock_hash": policy_lock.policy_lock_hash,
                "canonical_control": True,
                "deployable_selection_input": True,
                "routing_quality_claimed": False,
                "downstream_utility_claimed": False,
                "next_required_evidence": (
                    "freeze_stage60_comparison_policies_then_run_fresh_matched_"
                    "stage70_downstream_scoring"
                ),
            },
        )
        write_json(
            root / "reports/leakage_report.json",
            {
                "schema_version": "midogpp_uniform_b_v2_equal_union_policy_leakage_v1",
                "status": "PASS",
                "source_only_frozen_state": True,
                "target_identity_fold_candidate_exclusion_and_label_blind_shuffle_only": True,
                "target_identity_as_predictive_feature": False,
                "target_data_used": False,
                "target_support_used": False,
                "target_labels_used": False,
                "target_evaluation_labels_used": False,
                "target_metadata_used": False,
                "target_expert_excluded_in_every_replicate": True,
                "all_eligible_sources_retained": True,
                "individual_expert_or_seed_selection_performed": False,
                "source_ranking_performed": False,
                "source_weighting_learned": False,
                "compatibility_scores_computed": False,
                "nelbo_computed": False,
                "generation_performed": False,
                "classifier_fit_performed": False,
                "bacc_computed": False,
                "macro_f1_computed": False,
                "downstream_utility_computed": False,
            },
        )
        _write_content_index(root)
        _write_state(root, "COMPLETE")

        from .validation import validate_equal_union_policy_bundle

        checks = validate_equal_union_policy_bundle(
            root,
            config=config,
            allow_pending=True,
            _validated_generation_lock=generation_lock,
        )
        write_json(
            root / "reports/validation_report.json",
            {
                "schema_version": "midogpp_uniform_b_v2_equal_union_policy_validation_v1",
                "status": "PASS",
                "validator": "validate_equal_union_policy_bundle",
                "checks": checks,
            },
        )
        validate_equal_union_policy_bundle(
            root,
            config=config,
            _validated_generation_lock=generation_lock,
        )
    except Exception:
        _write_state(root, "FAILED")
        raise
    return root


def _write_content_index(root: Path) -> None:
    records = []
    for relative in CONTENT_INDEX_MEMBERS:
        member = root / relative
        if not member.is_file():
            raise ProtocolError(f"Equal-union policy content member is missing: {relative}.")
        records.append(
            {
                "relative_path": relative,
                "sha256": _sha256_file(member),
                "size_bytes": member.stat().st_size,
            }
        )
    payload: dict[str, object] = {
        "schema_version": "midogpp_uniform_b_v2_equal_union_policy_content_v1",
        "records": records,
    }
    payload["content_hash"] = stable_hash(payload)
    write_json(root / "manifests/content_index.json", payload)


def _assert_closed_world(root: Path) -> None:
    actual = {
        member.relative_to(root).as_posix()
        for member in root.rglob("*")
        if member.is_file()
    }
    unexpected = sorted(actual.difference(REQUIRED_FILES))
    if unexpected:
        raise ProtocolError(
            f"Equal-union policy artifact contains unexpected files: {unexpected}."
        )


def _write_state(root: Path, status: str) -> None:
    write_json(
        root / "reports/run_state.json",
        {
            "schema_version": "midogpp_uniform_b_v2_equal_union_policy_run_state_v1",
            "status": status,
            "claim_scope": CLAIM_SCOPE,
        },
    )


def _json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError(f"Equal-union policy JSON must be an object: {path}.")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ("run_equal_union_policy_lock",)
