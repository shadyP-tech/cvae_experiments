"""Materialize the frozen metadata exact-match tie-union policy artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from ...reporting import write_csv_rows, write_json
from .bundle import CONTENT_INDEX_MEMBERS, REQUIRED_FILES
from .config import UniformBV2MetadataTieUnionPolicyConfig
from .contracts import (
    CLAIM_SCOPE,
    EXPERIMENT_ID,
    POLICY_DECISION,
    PUBLICATION_STATE,
)
from .inputs import ValidatedTieUnionInputs, load_validated_inputs
from .policy import (
    assignment_rows,
    assignment_table_hash,
    build_policy_lock,
    build_policy_plan_payload,
    build_policy_selections,
    selection_table_hash,
)


def run_metadata_tie_union_policy_lock(
    config: UniformBV2MetadataTieUnionPolicyConfig,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    """Create a lock-only comparison policy without target samples or metrics."""

    root = Path(artifact_root or config.artifact_root)
    for relative in ("manifests", "reports", "tables", "provenance"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    state_path = root / "reports/run_state.json"
    try:
        state_status = _json(state_path).get("status") if state_path.is_file() else None
    except Exception:
        _write_state(root, "FAILED")
        raise
    if state_status == "COMPLETE":
        from .validation import validate_metadata_tie_union_policy_bundle

        try:
            _assert_closed_world(root)
            _assert_workspace_launch_files(root)
            validate_metadata_tie_union_policy_bundle(root, config=config)
        except Exception:
            _write_state(root, "FAILED")
            raise
        return root

    _assert_closed_world(root)
    _assert_workspace_launch_files(root)
    _write_state(root, "RUNNING")
    try:
        inputs = load_validated_inputs(config)
        selections = build_policy_selections(inputs.compatibility_scores, config)
        assignments = assignment_rows(
            inputs.generation_lock,
            inputs.compatibility_scores,
            config,
        )
        plan = build_policy_plan_payload(
            inputs.generation_lock,
            inputs.compatibility_scores,
            config,
        )
        policy_lock = build_policy_lock(
            config,
            inputs.generation_lock,
            inputs.equal_union_policy_lock,
            inputs.compatibility_lock,
            inputs.compatibility_scores,
        )
        selections_hash = selection_table_hash(selections)
        assignments_hash = assignment_table_hash(assignments)
        write_json(root / "manifests/policy_lock.json", policy_lock.to_payload())
        write_json(root / "manifests/metadata_tie_union_policy_plan.json", plan)
        write_csv_rows(
            root / "tables/policy_selections.csv",
            [row.to_payload() for row in selections],
        )
        write_csv_rows(
            root / "tables/policy_assignments.csv",
            [row.to_payload() for row in assignments],
        )

        protocol: dict[str, object] = {
            "schema_version": "midogpp_uniform_b_v2_metadata_tie_union_policy_protocol_v1",
            "experiment_id": EXPERIMENT_ID,
            "claim_scope": CLAIM_SCOPE,
            "config_contract_hash": config.contract_hash,
            "input_artifact_ids": [
                config.bank_artifact_id,
                config.generation_lock_artifact_id,
                config.equal_union_policy_artifact_id,
                config.metadata_compatibility_artifact_id,
            ],
            "bank_lock_hash": config.expected_bank_lock_hash,
            "generation_lock_hash": config.expected_generation_lock_hash,
            "equal_union_policy_lock_hash": (
                config.expected_equal_union_policy_lock_hash
            ),
            "equal_union_policy_plan_hash": (
                config.expected_equal_union_policy_plan_hash
            ),
            "equal_union_assignment_table_hash": (
                config.expected_equal_union_assignment_table_hash
            ),
            "compatibility_lock_hash": config.expected_compatibility_lock_hash,
            "compatibility_score_table_hash": (
                config.expected_compatibility_score_table_hash
            ),
            "policy_lock_hash": policy_lock.policy_lock_hash,
            "policy_plan_hash": plan["plan_hash"],
            "selection_table_hash": selections_hash,
            "assignment_table_hash": assignments_hash,
            "policy_frozen_before_target_evaluation": True,
            "selection_rule": "retain_all_sources_tied_at_maximum_exact_match_score",
            "canonical_candidate_order_role": "ordering_only_never_tie_break",
            "all_maximum_ties_retained": True,
            "fixed_total_per_class": 1024,
            "full_training_x_generation_seed_cartesian_product": True,
            "stage40_source_stream_ids_reused": True,
            "stage40_class_shuffle_seeds_reused": True,
            "target_samples_used": False,
            "target_support_used": False,
            "target_labels_used": False,
            "seed_selection_performed": False,
            "routing_quality_claimed": False,
            "downstream_utility_computed": False,
            "may_feed_deployable_selection": True,
        }
        protocol["protocol_hash"] = stable_hash(protocol)
        write_json(root / "manifests/protocol_manifest.json", protocol)

        write_json(
            root / "reports/policy_decision.json",
            {
                "schema_version": (
                    "midogpp_uniform_b_v2_metadata_tie_union_policy_decision_v1"
                ),
                "decision": POLICY_DECISION,
                "publication_state": PUBLICATION_STATE,
                "claim_scope": CLAIM_SCOPE,
                "policy_lock_hash": policy_lock.policy_lock_hash,
                "comparison_policy": True,
                "canonical_control": False,
                "deployable_selection_input": True,
                "metadata_score_is_proxy_only": True,
                "routing_quality_claimed": False,
                "downstream_utility_claimed": False,
                "next_required_evidence": (
                    "run_fresh_matched_stage70_downstream_scoring_against_the_frozen_"
                    "equal_union_control"
                ),
            },
        )
        write_json(
            root / "reports/leakage_report.json",
            {
                "schema_version": (
                    "midogpp_uniform_b_v2_metadata_tie_union_policy_leakage_v1"
                ),
                "status": "PASS",
                "source_only_frozen_state": True,
                "target_identity_fold_exclusion_profile_binding_and_shuffle_only": True,
                "target_identity_as_predictive_feature": False,
                "target_samples_used": False,
                "target_support_used": False,
                "target_labels_used": False,
                "target_evaluation_labels_used": False,
                "sanitized_target_metadata_profile_used": True,
                "target_expert_excluded_in_every_replicate": True,
                "compatibility_scores_consumed": True,
                "compatibility_scores_computed_in_policy": False,
                "metadata_proxy_selection_performed": True,
                "all_maximum_ties_retained": True,
                "tie_break_applied": False,
                "seed_selection_performed": False,
                "source_weighting_learned": False,
                "stage20_scores_reused": False,
                "stage50_artifacts_used": False,
                "stage90_artifacts_used": False,
                "nelbo_computed": False,
                "generation_performed": False,
                "classifier_fit_performed": False,
                "bacc_computed": False,
                "macro_f1_computed": False,
                "routing_quality_claimed": False,
                "downstream_utility_computed": False,
            },
        )
        _write_content_index(root)
        _write_state(root, "COMPLETE")

        from .validation import validate_metadata_tie_union_policy_bundle

        checks = validate_metadata_tie_union_policy_bundle(
            root,
            config=config,
            allow_pending=True,
            _validated_inputs=inputs,
        )
        write_json(
            root / "reports/validation_report.json",
            {
                "schema_version": (
                    "midogpp_uniform_b_v2_metadata_tie_union_policy_validation_v1"
                ),
                "status": "PASS",
                "validator": "validate_metadata_tie_union_policy_bundle",
                "checks": checks,
            },
        )
        validate_metadata_tie_union_policy_bundle(
            root,
            config=config,
            _validated_inputs=inputs,
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
            raise ProtocolError(f"Metadata tie-union content member is missing: {relative}.")
        records.append(
            {
                "relative_path": relative,
                "sha256": _sha256_file(member),
                "size_bytes": member.stat().st_size,
            }
        )
    payload: dict[str, object] = {
        "schema_version": "midogpp_uniform_b_v2_metadata_tie_union_policy_content_v1",
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
            f"Metadata tie-union artifact contains unexpected files: {unexpected}."
        )


def _assert_workspace_launch_files(root: Path) -> None:
    if not (root / "config.resolved.yaml").is_file() or not (
        root / "provenance/input_artifacts.json"
    ).is_file():
        raise ProtocolError(
            "Metadata tie-union policy must be launched through the MIDOG++ workspace."
        )


def _write_state(root: Path, status: str) -> None:
    write_json(
        root / "reports/run_state.json",
        {
            "schema_version": (
                "midogpp_uniform_b_v2_metadata_tie_union_policy_run_state_v1"
            ),
            "status": status,
            "claim_scope": CLAIM_SCOPE,
        },
    )


def _json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError(f"Metadata tie-union JSON must be an object: {path}.")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ("run_metadata_tie_union_policy_lock",)
