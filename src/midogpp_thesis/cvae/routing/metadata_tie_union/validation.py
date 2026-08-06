"""Exact reconstructive validation for the metadata tie-union policy bundle."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ...expert_bank.uniform_b_v2_promotion.validation import (
    REQUIRED_FILES as BANK_REQUIRED_FILES,
)
from ...generation.contracts import EXPECTED_CONTROL_LOCK_HASH
from ...generation.validation import REQUIRED_FILES as GENERATION_REQUIRED_FILES
from ...protocol import ProtocolError
from ..bundle import REQUIRED_FILES as EQUAL_UNION_REQUIRED_FILES
from ..contracts import (
    EXPECTED_CONFIG_CONTRACT_HASH as EXPECTED_EQUAL_UNION_CONFIG_CONTRACT_HASH,
)
from .bundle import CONTENT_INDEX_MEMBERS, REQUIRED_FILES
from .config import (
    UniformBV2MetadataTieUnionPolicyConfig,
    load_metadata_tie_union_policy_config,
)
from .contracts import (
    CLAIM_SCOPE,
    COMPATIBILITY_ARTIFACT_ID,
    EQUAL_UNION_POLICY_ARTIFACT_ID,
    EXPECTED_ASSIGNMENT_COUNT,
    EXPECTED_REPLICATE_COUNT,
    EXPECTED_SELECTION_COUNT,
    EXPERIMENT_ID,
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    POLICY_DECISION,
    PUBLICATION_STATE,
    SELECTED_SOURCES_BY_TARGET,
    SOURCE_BUDGET_BY_TIE_COUNT,
)
from .inputs import ValidatedTieUnionInputs, load_validated_inputs
from .policy import (
    assignment_rows,
    assignment_table_hash,
    build_policy_lock,
    build_policy_plan_payload,
    build_policy_selections,
    read_policy_lock,
    selection_table_hash,
)


def validate_metadata_tie_union_policy_bundle(
    root: str | Path,
    *,
    config: UniformBV2MetadataTieUnionPolicyConfig,
    allow_pending: bool = False,
    _validated_inputs: ValidatedTieUnionInputs | None = None,
) -> dict[str, object]:
    path = Path(root)
    required = set(REQUIRED_FILES)
    if allow_pending:
        required.remove("reports/validation_report.json")
    missing = sorted(relative for relative in required if not (path / relative).is_file())
    if missing:
        raise ProtocolError(f"Metadata tie-union policy is incomplete: {missing}.")
    _validate_closed_world(path)
    if load_metadata_tie_union_policy_config(path / "config.resolved.yaml") != config:
        raise ProtocolError("Metadata tie-union resolved config drifted from running config.")
    validate_policy_provenance(path, config=config)

    inputs = _validated_inputs or load_validated_inputs(config)
    expected_selections = build_policy_selections(inputs.compatibility_scores, config)
    expected_assignments = assignment_rows(
        inputs.generation_lock,
        inputs.compatibility_scores,
        config,
    )
    expected_plan = build_policy_plan_payload(
        inputs.generation_lock,
        inputs.compatibility_scores,
        config,
    )
    expected_lock = build_policy_lock(
        config,
        inputs.generation_lock,
        inputs.equal_union_policy_lock,
        inputs.compatibility_lock,
        inputs.compatibility_scores,
    )
    lock = read_policy_lock(path / "manifests/policy_lock.json")
    if lock.to_payload() != expected_lock.to_payload():
        raise ProtocolError("Metadata tie-union policy lock drifted from its upstreams.")
    plan = _json(path / "manifests/metadata_tie_union_policy_plan.json")
    _assert_hash(plan, "plan_hash")
    if plan != expected_plan:
        raise ProtocolError("Metadata tie-union policy plan drifted.")
    _validate_csv(
        path / "tables/policy_selections.csv",
        expected_selections,
        expected_count=EXPECTED_SELECTION_COUNT,
        label="selection",
    )
    _validate_csv(
        path / "tables/policy_assignments.csv",
        expected_assignments,
        expected_count=EXPECTED_ASSIGNMENT_COUNT,
        label="assignment",
    )
    lock_payload = lock.to_payload()
    if (
        lock_payload.get("selection_table_hash")
        != selection_table_hash(expected_selections)
        or lock_payload.get("assignment_table_hash")
        != assignment_table_hash(expected_assignments)
    ):
        raise ProtocolError("Metadata tie-union policy table hash drifted.")

    protocol = _json(path / "manifests/protocol_manifest.json")
    _assert_hash(protocol, "protocol_hash")
    expected_protocol = {
        "schema_version": "midogpp_uniform_b_v2_metadata_tie_union_policy_protocol_v1",
        "experiment_id": EXPERIMENT_ID,
        "claim_scope": CLAIM_SCOPE,
        "config_contract_hash": config.contract_hash,
        "input_artifact_ids": [
            EXPERT_BANK_ARTIFACT_ID,
            GENERATION_LOCK_ARTIFACT_ID,
            EQUAL_UNION_POLICY_ARTIFACT_ID,
            COMPATIBILITY_ARTIFACT_ID,
        ],
        "bank_lock_hash": config.expected_bank_lock_hash,
        "generation_lock_hash": config.expected_generation_lock_hash,
        "equal_union_policy_lock_hash": config.expected_equal_union_policy_lock_hash,
        "equal_union_policy_plan_hash": config.expected_equal_union_policy_plan_hash,
        "equal_union_assignment_table_hash": (
            config.expected_equal_union_assignment_table_hash
        ),
        "compatibility_lock_hash": config.expected_compatibility_lock_hash,
        "compatibility_score_table_hash": (
            config.expected_compatibility_score_table_hash
        ),
        "policy_lock_hash": lock.policy_lock_hash,
        "policy_plan_hash": plan["plan_hash"],
        "selection_table_hash": selection_table_hash(expected_selections),
        "assignment_table_hash": assignment_table_hash(expected_assignments),
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
    _require_only(protocol, set(expected_protocol) | {"protocol_hash"}, "protocol")
    _require(protocol, expected_protocol, "protocol")

    decision = _json(path / "reports/policy_decision.json")
    expected_decision = {
        "schema_version": "midogpp_uniform_b_v2_metadata_tie_union_policy_decision_v1",
        "decision": POLICY_DECISION,
        "publication_state": PUBLICATION_STATE,
        "claim_scope": CLAIM_SCOPE,
        "policy_lock_hash": lock.policy_lock_hash,
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
    }
    _require_only(decision, set(expected_decision), "policy decision")
    _require(decision, expected_decision, "policy decision")

    leakage = _json(path / "reports/leakage_report.json")
    expected_leakage = {
        "schema_version": "midogpp_uniform_b_v2_metadata_tie_union_policy_leakage_v1",
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
    }
    _require_only(leakage, set(expected_leakage), "leakage report")
    _require(leakage, expected_leakage, "leakage report")
    state = _json(path / "reports/run_state.json")
    expected_state = {
        "schema_version": "midogpp_uniform_b_v2_metadata_tie_union_policy_run_state_v1",
        "status": "COMPLETE",
        "claim_scope": CLAIM_SCOPE,
    }
    _require_only(state, set(expected_state), "run state")
    _require(state, expected_state, "run state")
    _validate_content_index(path)

    tie_counts = {
        row.target_center: row.tie_count for row in expected_selections
    }
    checks = {
        "status": "PASS",
        "bank_lock_hash": config.expected_bank_lock_hash,
        "generation_lock_hash": config.expected_generation_lock_hash,
        "equal_union_policy_lock_hash": config.expected_equal_union_policy_lock_hash,
        "compatibility_lock_hash": config.expected_compatibility_lock_hash,
        "compatibility_score_table_hash": (
            config.expected_compatibility_score_table_hash
        ),
        "policy_lock_hash": lock.policy_lock_hash,
        "selection_count": EXPECTED_SELECTION_COUNT,
        "target_replicate_count": EXPECTED_REPLICATE_COUNT,
        "assignment_count": EXPECTED_ASSIGNMENT_COUNT,
        "selected_sources_by_target": {
            target: list(SELECTED_SOURCES_BY_TARGET[target])
            for target in SELECTED_SOURCES_BY_TARGET
        },
        "source_budget_by_tie_count": {
            str(count): budget for count, budget in SOURCE_BUDGET_BY_TIE_COUNT.items()
        },
        "observed_tie_count_by_target": tie_counts,
        "all_maximum_ties_retained": True,
        "target_expert_excluded": True,
        "seed_selection_performed": False,
        "target_samples_used": False,
        "routing_quality_claimed": False,
        "downstream_utility_computed": False,
        "may_feed_deployable_selection": True,
    }
    if not allow_pending:
        report = _json(path / "reports/validation_report.json")
        expected_report = {
            "schema_version": (
                "midogpp_uniform_b_v2_metadata_tie_union_policy_validation_v1"
            ),
            "status": "PASS",
            "validator": "validate_metadata_tie_union_policy_bundle",
            "checks": checks,
        }
        _require_only(report, set(expected_report), "validation report")
        _require(report, expected_report, "validation report")
    return checks


def validate_policy_provenance(
    root: str | Path,
    *,
    config: UniformBV2MetadataTieUnionPolicyConfig,
) -> None:
    """Accept only the four frozen bank/generation/control/proxy inputs."""

    from ..metadata_compatibility.bundle import (
        REQUIRED_FILES as COMPATIBILITY_REQUIRED_FILES,
    )
    from ..metadata_compatibility.contracts import OUTPUT_SEMANTIC_IDENTITIES

    output_root = Path(root)
    manifest = _json(output_root / "provenance/input_artifacts.json")
    expected_manifest = {
        "schema_version": "midogpp_input_artifacts_v2",
        "dataset_id": "midogpp",
        "experiment_id": EXPERIMENT_ID,
        "stage": "60_routing_and_composition",
        "claim_scope": CLAIM_SCOPE,
        "selection_used_target_eval_artifacts": False,
    }
    _require_only(
        manifest,
        set(expected_manifest)
        | {
            "input_artifacts",
            "repository_revision",
            "repository_dirty",
            "repository_status_hash",
        },
        "workspace provenance",
    )
    _require(manifest, expected_manifest, "workspace provenance")
    if (
        not _is_hex(manifest.get("repository_revision"), length=40)
        or not isinstance(manifest.get("repository_dirty"), bool)
        or not _is_hex(manifest.get("repository_status_hash"), length=64)
    ):
        raise ProtocolError("Metadata tie-union repository provenance is malformed.")
    raw_rows = manifest.get("input_artifacts")
    if not isinstance(raw_rows, list) or not all(
        isinstance(row, Mapping) for row in raw_rows
    ):
        raise ProtocolError("Metadata tie-union workspace provenance is malformed.")
    rows = {str(row.get("artifact_id", "")): row for row in raw_rows}
    expected_ids = {
        EXPERT_BANK_ARTIFACT_ID,
        GENERATION_LOCK_ARTIFACT_ID,
        EQUAL_UNION_POLICY_ARTIFACT_ID,
        COMPATIBILITY_ARTIFACT_ID,
    }
    if len(raw_rows) != 4 or set(rows) != expected_ids:
        raise ProtocolError("Metadata tie-union may consume only its four frozen inputs.")
    equal_union_semantics = {
        "policy_lock_contract": "midogpp_uniform_b_v2_equal_union_policy_lock_v1",
        "config_contract_hash": EXPECTED_EQUAL_UNION_CONFIG_CONTRACT_HASH,
        "policy_lock_hash": config.expected_equal_union_policy_lock_hash,
        "policy_plan_hash": config.expected_equal_union_policy_plan_hash,
        "assignment_table_hash": config.expected_equal_union_assignment_table_hash,
        "generation_lock_hash": config.expected_generation_lock_hash,
        "expert_bank_lock_hash": config.expected_bank_lock_hash,
    }
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
        EQUAL_UNION_POLICY_ARTIFACT_ID: (
            config.equal_union_policy_root,
            "60_routing_and_composition",
            "routing_and_composition",
            "ROUTING_POLICY_FROZEN_AFTER_VALIDATION",
            set(EQUAL_UNION_REQUIRED_FILES),
            equal_union_semantics,
        ),
        COMPATIBILITY_ARTIFACT_ID: (
            config.metadata_compatibility_root,
            "60_routing_and_composition",
            "routing_compatibility_only",
            "ROUTING_COMPATIBILITY_PROXY_FROZEN_AFTER_VALIDATION",
            set(COMPATIBILITY_REQUIRED_FILES),
            dict(OUTPUT_SEMANTIC_IDENTITIES),
        ),
    }
    for artifact_id, (
        expected_root,
        stage,
        scope,
        evidence,
        required_files,
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
            Path(str(row.get("resolved_path", ""))).resolve()
            != expected_root.resolve()
            or row.get("exists") is not True
            or row.get("stage") != stage
            or row.get("claim_scope") != scope
            or row.get("evidence_label") != evidence
            or row.get("semantic_identities") != semantic_identities
            or row.get("semantic_identities_are_file_hashes") is not False
        ):
            raise ProtocolError(f"Metadata tie-union provenance drifted: {artifact_id}.")
        integrity = row.get("file_integrity")
        if not isinstance(integrity, Mapping):
            raise ProtocolError(f"Metadata tie-union input lacks integrity: {artifact_id}.")
        _require_only(
            integrity,
            {"status", "default_recording_algorithm", "files"},
            f"workspace integrity {artifact_id}",
        )
        integrity_status = integrity.get("status")
        if (
            integrity.get("default_recording_algorithm") != "sha256"
            or integrity_status
            not in {
                "HASHES_RECORDED_NO_EXPECTATIONS",
                "EXPECTED_FILE_HASHES_MATCH",
            }
        ):
            raise ProtocolError(f"Metadata tie-union input lacks integrity: {artifact_id}.")
        files = integrity.get("files")
        if not isinstance(files, list) or not all(
            isinstance(item, Mapping) for item in files
        ):
            raise ProtocolError(f"Metadata tie-union file inventory is invalid: {artifact_id}.")
        inventory = {str(item.get("path", "")): item for item in files}
        if len(inventory) != len(files) or set(inventory) != required_files:
            raise ProtocolError(f"Metadata tie-union file coverage drifted: {artifact_id}.")
        has_expectations = False
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
                or not member.is_file()
                or not isinstance(computed, Mapping)
                or set(computed) != {"sha256"}
                or computed.get("sha256") != _sha256_file(member)
                or item.get("size_bytes") != member.stat().st_size
            ):
                raise ProtocolError(f"Metadata tie-union input member drifted: {relative}.")
            expected = item.get("expected")
            if expected is None:
                if item.get("verification") != "RECORDED_NO_EXPECTATION":
                    raise ProtocolError(
                        f"Metadata tie-union input verification drifted: {relative}."
                    )
            elif (
                not isinstance(expected, Mapping)
                or set(expected) != {"algorithm", "digest"}
                or expected.get("algorithm") != "sha256"
                or expected.get("digest") != computed.get("sha256")
                or item.get("verification") != "MATCH"
            ):
                raise ProtocolError(
                    f"Metadata tie-union expected input hash failed: {relative}."
                )
            else:
                has_expectations = True
        expected_integrity_status = (
            "EXPECTED_FILE_HASHES_MATCH"
            if has_expectations
            else "HASHES_RECORDED_NO_EXPECTATIONS"
        )
        if integrity_status != expected_integrity_status:
            raise ProtocolError(
                f"Metadata tie-union input integrity status drifted: {artifact_id}."
            )


def _validate_csv(
    path: Path,
    expected: Sequence[object],
    *,
    expected_count: int,
    label: str,
) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    payloads = [row.to_payload() for row in expected]  # type: ignore[attr-defined]
    if len(rows) != expected_count:
        raise ProtocolError(f"Metadata tie-union {label} table row count drifted.")
    expected_columns = tuple(payloads[0])
    if tuple(reader.fieldnames or ()) != expected_columns:
        raise ProtocolError(f"Metadata tie-union {label} table columns drifted.")
    for observed, wanted in zip(rows, payloads, strict=True):
        if any(
            observed.get(key, "") != ("" if wanted[key] is None else str(wanted[key]))
            for key in expected_columns
        ):
            raise ProtocolError(f"Metadata tie-union {label} table content drifted.")


def _validate_content_index(root: Path) -> None:
    payload = _json(root / "manifests/content_index.json")
    _assert_hash(payload, "content_hash")
    _require_only(payload, {"schema_version", "records", "content_hash"}, "content index")
    _require(
        payload,
        {"schema_version": "midogpp_uniform_b_v2_metadata_tie_union_policy_content_v1"},
        "content index",
    )
    rows = payload.get("records")
    if not isinstance(rows, list):
        raise ProtocolError("Metadata tie-union content index is invalid.")
    observed: list[str] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ProtocolError("Metadata tie-union content-index row is invalid.")
        _require_only(raw, {"relative_path", "sha256", "size_bytes"}, "content-index row")
        relative = str(raw.get("relative_path", ""))
        member = _safe_member(root, relative)
        if (
            not member.is_file()
            or member.stat().st_size != int(raw.get("size_bytes", -1))
            or _sha256_file(member) != raw.get("sha256")
        ):
            raise ProtocolError(f"Metadata tie-union content member drifted: {relative}.")
        observed.append(relative)
    if tuple(observed) != CONTENT_INDEX_MEMBERS:
        raise ProtocolError("Metadata tie-union content-index coverage drifted.")


def _validate_closed_world(root: Path) -> None:
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


def _assert_hash(payload: Mapping[str, object], field: str) -> None:
    unhashed = {key: value for key, value in payload.items() if key != field}
    if stable_hash(unhashed) != payload.get(field):
        raise ProtocolError(f"Metadata tie-union semantic hash drifted: {field}.")


def _require(
    observed: Mapping[str, object], expected: Mapping[str, object], label: str
) -> None:
    mismatch = [key for key, value in expected.items() if observed.get(key) != value]
    if mismatch:
        raise ProtocolError(f"Metadata tie-union {label} drifted: {mismatch}.")


def _require_only(observed: Mapping[str, object], allowed: set[str], label: str) -> None:
    actual = {str(key) for key in observed}
    if actual != allowed:
        missing = sorted(allowed.difference(actual))
        extra = sorted(actual.difference(allowed))
        raise ProtocolError(
            f"Metadata tie-union {label} fields drifted: missing={missing}, extra={extra}."
        )


def _json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read metadata tie-union JSON: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Metadata tie-union JSON must be an object: {path}.")
    return payload


def _safe_member(root: Path, relative: str) -> Path:
    resolved_root = root.resolve()
    member = (resolved_root / relative).resolve()
    if member == resolved_root or not member.is_relative_to(resolved_root):
        raise ProtocolError("Metadata tie-union content path escapes its artifact root.")
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
    "validate_metadata_tie_union_policy_bundle",
    "validate_policy_provenance",
)
