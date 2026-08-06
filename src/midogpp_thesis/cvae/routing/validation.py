"""Independent validation for the Stage-60 equal-union policy bundle."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping

from ...common.hashing import stable_hash
from ..expert_bank.uniform_b_v2_promotion.validation import (
    REQUIRED_FILES as BANK_REQUIRED_FILES,
)
from ..generation.validation import REQUIRED_FILES as GENERATION_REQUIRED_FILES
from ..generation.contracts import EXPECTED_CONTROL_LOCK_HASH, GenerationLock
from ..protocol import ProtocolError
from .bundle import CONTENT_INDEX_MEMBERS, REQUIRED_FILES
from .config import UniformBV2EqualUnionPolicyConfig, load_equal_union_policy_config
from .contracts import (
    CLAIM_SCOPE,
    EXPECTED_ASSIGNMENT_COUNT,
    EXPECTED_REPLICATE_COUNT,
    EXPERIMENT_ID,
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    POLICY_DECISION,
    PUBLICATION_STATE,
)
from .inputs import load_validated_inputs
from .policy import (
    assignment_rows,
    assignment_table_hash,
    build_policy_lock,
    build_policy_plan_payload,
    read_policy_lock,
)


def validate_equal_union_policy_bundle(
    root: str | Path,
    *,
    config: UniformBV2EqualUnionPolicyConfig,
    allow_pending: bool = False,
    _validated_generation_lock: GenerationLock | None = None,
) -> dict[str, object]:
    path = Path(root)
    required = set(REQUIRED_FILES)
    if allow_pending:
        required.remove("reports/validation_report.json")
    missing = sorted(relative for relative in required if not (path / relative).is_file())
    if missing:
        raise ProtocolError(f"Uniform-B v2 equal-union policy is incomplete: {missing}.")
    _validate_closed_world(path)
    if load_equal_union_policy_config(path / "config.resolved.yaml") != config:
        raise ProtocolError("Equal-union resolved config drifted from the running config.")
    validate_policy_provenance(path, config=config)

    generation_lock = _validated_generation_lock or load_validated_inputs(config)
    expected_lock = build_policy_lock(config, generation_lock)
    lock = read_policy_lock(path / "manifests/policy_lock.json")
    if lock.to_payload() != expected_lock.to_payload():
        raise ProtocolError("Equal-union policy lock drifted from its frozen upstreams.")

    expected_plan = build_policy_plan_payload(generation_lock, config)
    plan = _json(path / "manifests/equal_union_policy_plan.json")
    _assert_hash(plan, "plan_hash")
    if plan != expected_plan:
        raise ProtocolError("Equal-union policy plan drifted.")
    expected_assignments = assignment_rows(generation_lock, config)
    _validate_assignments(path / "tables/policy_assignments.csv", expected_assignments)
    if lock.to_payload().get("assignment_table_hash") != assignment_table_hash(
        expected_assignments
    ):
        raise ProtocolError("Equal-union policy assignment hash drifted.")

    protocol = _json(path / "manifests/protocol_manifest.json")
    _assert_hash(protocol, "protocol_hash")
    _require_only(
        protocol,
        {
            "schema_version",
            "experiment_id",
            "claim_scope",
            "config_contract_hash",
            "input_artifact_ids",
            "bank_lock_hash",
            "generation_lock_hash",
            "policy_lock_hash",
            "policy_plan_hash",
            "assignment_table_hash",
            "policy_frozen_before_target_evaluation",
            "target_identity_role",
            "all_eligible_non_target_sources_retained",
            "full_training_x_generation_seed_cartesian_product",
            "target_data_used",
            "target_support_used",
            "target_labels_used",
            "expert_or_seed_selection_performed",
            "routing_quality_claimed",
            "downstream_utility_computed",
            "may_feed_deployable_selection",
            "protocol_hash",
        },
        "protocol",
    )
    _require(
        protocol,
        {
            "schema_version": "midogpp_uniform_b_v2_equal_union_policy_protocol_v1",
            "experiment_id": EXPERIMENT_ID,
            "claim_scope": CLAIM_SCOPE,
            "config_contract_hash": config.contract_hash,
            "input_artifact_ids": [EXPERT_BANK_ARTIFACT_ID, GENERATION_LOCK_ARTIFACT_ID],
            "bank_lock_hash": config.expected_bank_lock_hash,
            "generation_lock_hash": config.expected_generation_lock_hash,
            "policy_lock_hash": lock.policy_lock_hash,
            "policy_plan_hash": plan["plan_hash"],
            "assignment_table_hash": assignment_table_hash(expected_assignments),
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
        },
        "protocol",
    )
    decision = _json(path / "reports/policy_decision.json")
    _require_only(
        decision,
        {
            "schema_version",
            "decision",
            "publication_state",
            "claim_scope",
            "policy_lock_hash",
            "canonical_control",
            "deployable_selection_input",
            "routing_quality_claimed",
            "downstream_utility_claimed",
            "next_required_evidence",
        },
        "policy decision",
    )
    _require(
        decision,
        {
            "schema_version": "midogpp_uniform_b_v2_equal_union_policy_decision_v1",
            "decision": POLICY_DECISION,
            "publication_state": PUBLICATION_STATE,
            "claim_scope": CLAIM_SCOPE,
            "policy_lock_hash": lock.policy_lock_hash,
            "canonical_control": True,
            "deployable_selection_input": True,
            "routing_quality_claimed": False,
            "downstream_utility_claimed": False,
            "next_required_evidence": (
                "freeze_stage60_comparison_policies_then_run_fresh_matched_"
                "stage70_downstream_scoring"
            ),
        },
        "policy decision",
    )
    leakage = _json(path / "reports/leakage_report.json")
    _require_only(
        leakage,
        {
            "schema_version",
            "status",
            "source_only_frozen_state",
            "target_identity_fold_candidate_exclusion_and_label_blind_shuffle_only",
            "target_identity_as_predictive_feature",
            "target_data_used",
            "target_support_used",
            "target_labels_used",
            "target_evaluation_labels_used",
            "target_metadata_used",
            "target_expert_excluded_in_every_replicate",
            "all_eligible_sources_retained",
            "individual_expert_or_seed_selection_performed",
            "source_ranking_performed",
            "source_weighting_learned",
            "compatibility_scores_computed",
            "nelbo_computed",
            "generation_performed",
            "classifier_fit_performed",
            "bacc_computed",
            "macro_f1_computed",
            "downstream_utility_computed",
        },
        "leakage report",
    )
    _require(
        leakage,
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
        "leakage report",
    )
    state = _json(path / "reports/run_state.json")
    _require_only(state, {"schema_version", "status", "claim_scope"}, "run state")
    _require(
        state,
        {
            "schema_version": "midogpp_uniform_b_v2_equal_union_policy_run_state_v1",
            "status": "COMPLETE",
            "claim_scope": CLAIM_SCOPE,
        },
        "run state",
    )
    _validate_content_index(path)

    checks = {
        "status": "PASS",
        "bank_lock_hash": config.expected_bank_lock_hash,
        "generation_lock_hash": config.expected_generation_lock_hash,
        "policy_lock_hash": lock.policy_lock_hash,
        "target_replicate_count": EXPECTED_REPLICATE_COUNT,
        "assignment_count": EXPECTED_ASSIGNMENT_COUNT,
        "target_expert_excluded": True,
        "all_eligible_sources_retained": True,
        "individual_expert_or_seed_selection": False,
        "target_data_used": False,
        "routing_quality_claimed": False,
        "downstream_utility_computed": False,
        "may_feed_deployable_selection": True,
    }
    if not allow_pending:
        report = _json(path / "reports/validation_report.json")
        _require_only(
            report,
            {"schema_version", "status", "validator", "checks"},
            "validation report",
        )
        _require(
            report,
            {
                "schema_version": "midogpp_uniform_b_v2_equal_union_policy_validation_v1",
                "status": "PASS",
                "validator": "validate_equal_union_policy_bundle",
                "checks": checks,
            },
            "validation report",
        )
    return checks


def validate_policy_provenance(
    root: str | Path, *, config: UniformBV2EqualUnionPolicyConfig
) -> None:
    output_root = Path(root)
    manifest = _json(output_root / "provenance/input_artifacts.json")
    _require_only(
        manifest,
        {
            "schema_version",
            "dataset_id",
            "experiment_id",
            "stage",
            "claim_scope",
            "selection_used_target_eval_artifacts",
            "input_artifacts",
            "repository_revision",
            "repository_dirty",
            "repository_status_hash",
        },
        "workspace provenance",
    )
    _require(
        manifest,
        {
            "schema_version": "midogpp_input_artifacts_v2",
            "dataset_id": "midogpp",
            "experiment_id": EXPERIMENT_ID,
            "stage": "60_routing_and_composition",
            "claim_scope": CLAIM_SCOPE,
            "selection_used_target_eval_artifacts": False,
        },
        "workspace provenance",
    )
    if (
        not _is_hex(manifest.get("repository_revision"), length=40)
        or not isinstance(manifest.get("repository_dirty"), bool)
        or not _is_hex(manifest.get("repository_status_hash"), length=64)
    ):
        raise ProtocolError("Equal-union policy repository provenance is malformed.")
    raw_rows = manifest.get("input_artifacts")
    if not isinstance(raw_rows, list) or not all(isinstance(row, Mapping) for row in raw_rows):
        raise ProtocolError("Equal-union policy workspace provenance is malformed.")
    rows = {str(row.get("artifact_id", "")): row for row in raw_rows}
    if len(raw_rows) != 2 or set(rows) != {
        EXPERT_BANK_ARTIFACT_ID,
        GENERATION_LOCK_ARTIFACT_ID,
    }:
        raise ProtocolError("Equal-union policy may consume only Stage 30 and Stage 40 locks.")
    specs = {
        EXPERT_BANK_ARTIFACT_ID: (
            config.bank_root,
            "30_expert_bank",
            "expert_bank_construction_only",
            "ROUTING_AUTHORIZED_AFTER_VALIDATION",
            set(BANK_REQUIRED_FILES) | {"reports/validation_report.json"},
            {},
        ),
        GENERATION_LOCK_ARTIFACT_ID: (
            config.generation_lock_root,
            "40_prior_and_generation",
            "generation_settings_and_frame_lock",
            "GENERATION_SETTINGS_LOCKED_AFTER_VALIDATION",
            set(GENERATION_REQUIRED_FILES) | {"reports/validation_report.json"},
            {
                "generation_lock_contract": "midogpp_uniform_b_v2_generation_lock_v1",
                "generation_lock_hash": config.expected_generation_lock_hash,
                "expert_bank_lock_hash": config.expected_bank_lock_hash,
                "equal_union_control_lock_hash": EXPECTED_CONTROL_LOCK_HASH,
            },
        ),
    }
    for artifact_id, (
        expected_root,
        stage,
        scope,
        evidence,
        required,
        semantic_identities,
    ) in specs.items():
        row = rows[artifact_id]
        _require_only(
            row,
            {
                "artifact_id",
                "resolved_path",
                "stage",
                "evidence_label",
                "claim_scope",
                "semantic_identities",
                "semantic_identities_are_file_hashes",
                "file_integrity",
                "exists",
            },
            f"workspace provenance row {artifact_id}",
        )
        if (
            Path(str(row.get("resolved_path", ""))).resolve() != expected_root.resolve()
            or row.get("exists") is not True
            or row.get("stage") != stage
            or row.get("claim_scope") != scope
            or row.get("evidence_label") != evidence
            or row.get("semantic_identities") != semantic_identities
            or row.get("semantic_identities_are_file_hashes") is not False
        ):
            raise ProtocolError(f"Equal-union policy provenance drifted: {artifact_id}.")
        integrity = row.get("file_integrity")
        if not isinstance(integrity, Mapping):
            raise ProtocolError(f"Equal-union policy input lacks integrity: {artifact_id}.")
        _require_only(
            integrity,
            {"status", "default_recording_algorithm", "files"},
            f"workspace integrity {artifact_id}",
        )
        if (
            integrity.get("status") != "HASHES_RECORDED_NO_EXPECTATIONS"
            or integrity.get("default_recording_algorithm") != "sha256"
        ):
            raise ProtocolError(f"Equal-union policy input lacks integrity: {artifact_id}.")
        files = integrity.get("files")
        if not isinstance(files, list) or not all(isinstance(item, Mapping) for item in files):
            raise ProtocolError(f"Equal-union policy file inventory is invalid: {artifact_id}.")
        inventory = {str(item.get("path", "")): item for item in files}
        if len(inventory) != len(files) or set(inventory) != required:
            raise ProtocolError(f"Equal-union policy file coverage drifted: {artifact_id}.")
        for relative, item in inventory.items():
            member = _safe_member(expected_root, relative)
            _require_only(
                item,
                {
                    "path",
                    "resolved_path",
                    "exists",
                    "expected",
                    "size_bytes",
                    "computed",
                    "verification",
                },
                f"workspace file row {artifact_id}:{relative}",
            )
            computed = item.get("computed")
            if (
                Path(str(item.get("resolved_path", ""))).resolve() != member
                or item.get("exists") is not True
                or item.get("expected") is not None
                or item.get("verification") != "RECORDED_NO_EXPECTATION"
                or not member.is_file()
                or not isinstance(computed, Mapping)
                or set(computed) != {"sha256"}
                or computed.get("sha256") != _sha256_file(member)
                or item.get("size_bytes") != member.stat().st_size
            ):
                raise ProtocolError(f"Equal-union policy input member drifted: {relative}.")


def _validate_assignments(path: Path, expected: tuple[object, ...]) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    expected_payloads = [row.to_payload() for row in expected]  # type: ignore[attr-defined]
    if len(rows) != EXPECTED_ASSIGNMENT_COUNT:
        raise ProtocolError("Equal-union assignment table row count drifted.")
    expected_columns = tuple(expected_payloads[0])
    if tuple(reader.fieldnames or ()) != expected_columns:
        raise ProtocolError("Equal-union assignment table columns drifted.")
    for observed, wanted in zip(rows, expected_payloads, strict=True):
        if any(
            observed.get(key, "") != ("" if wanted[key] is None else str(wanted[key]))
            for key in expected_columns
        ):
            raise ProtocolError("Equal-union assignment table content drifted.")


def _validate_content_index(root: Path) -> None:
    payload = _json(root / "manifests/content_index.json")
    _assert_hash(payload, "content_hash")
    _require_only(
        payload,
        {"schema_version", "records", "content_hash"},
        "content index",
    )
    _require(
        payload,
        {"schema_version": "midogpp_uniform_b_v2_equal_union_policy_content_v1"},
        "content index",
    )
    rows = payload.get("records")
    if not isinstance(rows, list):
        raise ProtocolError("Equal-union policy content index is invalid.")
    expected = tuple(CONTENT_INDEX_MEMBERS)
    observed: list[str] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ProtocolError("Equal-union policy content-index row is invalid.")
        _require_only(
            raw,
            {"relative_path", "sha256", "size_bytes"},
            "content-index row",
        )
        relative = str(raw.get("relative_path", ""))
        member = _safe_member(root, relative)
        if (
            not member.is_file()
            or member.stat().st_size != int(raw.get("size_bytes", -1))
            or _sha256_file(member) != raw.get("sha256")
        ):
            raise ProtocolError(f"Equal-union policy content member drifted: {relative}.")
        observed.append(relative)
    if tuple(observed) != expected:
        raise ProtocolError("Equal-union policy content-index coverage drifted.")


def _validate_closed_world(root: Path) -> None:
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


def _assert_hash(payload: Mapping[str, object], field: str) -> None:
    if stable_hash({key: value for key, value in payload.items() if key != field}) != payload.get(field):
        raise ProtocolError(f"Equal-union policy semantic hash drifted: {field}.")


def _require(observed: Mapping[str, object], expected: Mapping[str, object], label: str) -> None:
    mismatch = [key for key, value in expected.items() if observed.get(key) != value]
    if mismatch:
        raise ProtocolError(f"Equal-union policy {label} drifted: {mismatch}.")


def _require_only(observed: Mapping[str, object], allowed: set[str], label: str) -> None:
    actual = {str(key) for key in observed}
    if actual != allowed:
        missing = sorted(allowed.difference(actual))
        extra = sorted(actual.difference(allowed))
        raise ProtocolError(
            f"Equal-union policy {label} fields drifted: "
            f"missing={missing}, extra={extra}."
        )


def _json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read equal-union policy JSON: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Equal-union policy JSON must be an object: {path}.")
    return payload


def _safe_member(root: Path, relative: str) -> Path:
    resolved_root = root.resolve()
    member = (resolved_root / relative).resolve()
    if member == resolved_root or not member.is_relative_to(resolved_root):
        raise ProtocolError("Equal-union policy content path escapes its artifact root.")
    return member


def _is_hex(value: object, *, length: int) -> bool:
    rendered = str(value or "")
    return len(rendered) == length and all(
        character in "0123456789abcdef" for character in rendered
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = (
    "REQUIRED_FILES",
    "validate_equal_union_policy_bundle",
    "validate_policy_provenance",
)
