"""Pure workspace projection for HARP v20 activation.

This module renders and validates bytes only.  It intentionally owns no file
or transaction operations so activation planning remains mutation-free.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass, replace
from types import MappingProxyType

import yaml

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from . import authorization
from .amendment_publisher import HarpV20AmendmentDraft
from .config import HarpStage90V20Config, INPUT_ARTIFACT_IDS
from .identity import EXECUTION_REVISION, EXPERIMENT_ID, OUTPUT_ARTIFACT_ID, claim_boundary_payload
from .input_surfaces import CACHE_INDEX, CONTENT_INDEX
from .preparation import PREPARATION_RECEIPT
from .workspace_paths import SOURCE_TRAIN_REQUIRED_OUTPUT_MEMBERS


@dataclass(frozen=True, slots=True)
class RenderedActivationWorkspace:
    final_config_bytes: bytes
    final_registry_bytes: bytes
    final_catalog_bytes: bytes
    authorized_config: HarpStage90V20Config


def render_activation_workspace(
    *,
    original_config_bytes: bytes,
    original_registry_bytes: bytes,
    original_catalog_bytes: bytes,
    draft: HarpV20AmendmentDraft,
) -> RenderedActivationWorkspace:
    config_mapping = yaml_mapping(original_config_bytes, label="config")
    final_config_mapping = _render_config_mapping(config_mapping, draft)
    final_config_bytes = yaml_bytes(final_config_mapping)
    authorized_config = replace(
        draft.authorized_config,
        expected_hashes={
            **dict(draft.authorized_config.expected_hashes),
            "execution_amendment_sha256": draft.amendment_sha256,
        },
        claim_boundary=MappingProxyType(
            dict(claim_boundary_payload(execution_authorized=True))
        ),
        config_hash=canonical_hash(final_config_mapping),
    )
    authorization.validate_execution_amendment_payload(
        draft.amendment_payload,
        authorized_config,
        repo_root=draft.repository_root,
    )

    registry_mapping = yaml_mapping(original_registry_bytes, label="registry")
    final_registry_mapping = _render_registry_mapping(registry_mapping)
    final_registry_bytes = yaml_bytes(final_registry_mapping)
    catalog_mapping = yaml_mapping(
        original_catalog_bytes, label="artifact catalog"
    )
    final_catalog_mapping = _render_catalog_mapping(catalog_mapping, draft)
    final_catalog_bytes = yaml_bytes(final_catalog_mapping)
    validate_rendered_workspace(final_registry_mapping, final_catalog_mapping)
    return RenderedActivationWorkspace(
        final_config_bytes=final_config_bytes,
        final_registry_bytes=final_registry_bytes,
        final_catalog_bytes=final_catalog_bytes,
        authorized_config=authorized_config,
    )


def _render_config_mapping(
    original: Mapping[str, object], draft: HarpV20AmendmentDraft
) -> dict[str, object]:
    rendered = copy.deepcopy(dict(original))
    experiment = mapping_member(rendered, "experiment", "config experiment")
    inputs = mapping_member(rendered, "inputs", "config inputs")
    if experiment.get("status") != "planned" or experiment.get(
        "execution_authorized"
    ) is not False:
        raise ProtocolError("HARP v20 config is not in the planned activation state.")
    if any(
        inputs.get(role) is not None
        for role in (
            "test_cache_content_sha256",
            "development_manifest_sha256",
            "evaluation_manifest_sha256",
            "parent_ledger_sha256",
            "execution_amendment_sha256",
        )
    ):
        raise ProtocolError("HARP v20 planned config already binds execution hashes.")
    experiment["status"] = "diagnostic"
    experiment["execution_authorized"] = True
    inputs.update(dict(draft.computed_hashes))
    inputs["execution_amendment_sha256"] = draft.amendment_sha256
    rendered["claim_boundary"] = claim_boundary_payload(execution_authorized=True)
    return rendered


def _render_registry_mapping(original: Mapping[str, object]) -> dict[str, object]:
    rendered = copy.deepcopy(dict(original))
    experiments = sequence_member(rendered, "experiments", "registry experiments")
    matches = [
        row
        for row in experiments
        if isinstance(row, dict) and row.get("experiment_id") == EXPERIMENT_ID
    ]
    if len(matches) != 1 or matches[0].get("status") != "planned":
        raise ProtocolError("HARP v20 planned registry entry is absent or ambiguous.")
    matches[0]["status"] = "diagnostic"
    notes = matches[0].get("notes")
    if isinstance(notes, list) and notes:
        notes[0] = (
            "SCOPE LIMITED - v20 is separately authorized for one terminal "
            "consumed-test diagnostic; its distinct amendment remains unclaimed "
            "until launch and no result may be promoted."
        )
    return rendered


def _render_catalog_mapping(
    original: Mapping[str, object], draft: HarpV20AmendmentDraft
) -> dict[str, object]:
    rendered = copy.deepcopy(dict(original))
    artifacts = sequence_member(rendered, "artifacts", "catalog artifacts")
    rows = {
        str(row.get("artifact_id")): row
        for row in artifacts
        if isinstance(row, dict)
        and row.get("artifact_id") in {*INPUT_ARTIFACT_IDS[2:], OUTPUT_ARTIFACT_ID}
    }
    expected_ids = {*INPUT_ARTIFACT_IDS[2:], OUTPUT_ARTIFACT_ID}
    if set(rows) != expected_ids:
        raise ProtocolError("HARP v20 catalog activation inventory is incomplete.")
    for row in rows.values():
        semantics = row.get("semantic_identities")
        if not isinstance(semantics, dict):
            raise ProtocolError("HARP v20 catalog semantic identities are malformed.")
        if semantics.get("execution_authorized") != "false":
            raise ProtocolError("HARP v20 catalog is not in its planned state.")
        semantics["execution_authorized"] = "true"
        semantics["consumed_test_reuse_authorized"] = "true"

    cache = rows[INPUT_ARTIFACT_IDS[2]]
    cache["availability"] = "workstation_only"
    cache["evidence_label"] = (
        "SCOPE_LIMITED_HARP_V20_AUTHORIZED_LABEL_FREE_SOURCE_TRAIN_FULL_TEST_CACHE"
    )
    cache_semantics = cache["semantic_identities"]
    cache_semantics.update(
        {
            "composite_cache_content_sha256": draft.computed_hashes[
                "test_cache_content_sha256"
            ],
            "partition_hash": draft.partition_hash,
            "preparation_receipt_hash": draft.preparation_receipt_hash,
            "partition_namespace_is_predeclared_and_unchanged": "true",
        }
    )
    cache["expected_file_hashes"] = {
        CACHE_INDEX.as_posix(): _hash_entry(draft.cache_index_sha256),
        CONTENT_INDEX.as_posix(): _hash_entry(draft.content_index_sha256),
        PREPARATION_RECEIPT.as_posix(): _hash_entry(
            draft.preparation_receipt_sha256
        ),
    }
    cache["notes"] = [
        "This v20-only composite was rebuilt from separately authenticated canonical "
        "train and test feature caches. It stores all 216 source-train cases "
        "and all 218 test cases in physically separate label-free shards, and reuses "
        "no v1-v19 HARP cache, outcome, model, threshold, route, or output."
    ]

    development = rows[INPUT_ARTIFACT_IDS[3]]
    development["availability"] = "workstation_only"
    development["evidence_label"] = (
        "SCOPE_LIMITED_HARP_V20_AUTHORIZED_SOURCE_TRAIN_LABEL_CAPABILITY"
    )
    development["semantic_identities"].update(
        {
            "manifest_sha256": draft.computed_hashes[
                "development_manifest_sha256"
            ],
            "mixed_patch_labels_within_case_allowed": "true",
            "center_sharded_label_capability": "true",
            "label_capability_shard_count": "9",
            "source_fit_label_scope": (
                "ALL_216_TRAIN_CASES_POOLED_ACROSS_NINE_KNOWN_CENTERS"
            ),
            "source_capability_state": (
                "SOURCE_TRAIN_CENTER_SCOPED_OPEN_AFTER_ALL_SOURCE_AND_TARGET_MENU_SEALS_AND_BANK_ATTESTATIONS"
            ),
            "all_eighteen_menu_seals_required_before_any_source_truth": "true",
            "source_candidate_pool": "C_MINUS_q",
            "exact_one_capability_per_q": "true",
            "exact_center_coverage_required": "true",
        }
    )
    development["expected_file_hashes"] = {
        relative: _hash_entry(digest)
        for relative, digest in draft.development_member_sha256.items()
    }

    evaluation = rows[INPUT_ARTIFACT_IDS[4]]
    evaluation["availability"] = "workstation_only"
    evaluation["evidence_label"] = (
        "SCOPE_LIMITED_HARP_V20_AUTHORIZED_SEALED_EVALUATION_RELEASE"
    )
    evaluation["semantic_identities"].update(
        {
            "evaluation_release_descriptor_sha256": draft.computed_hashes[
                "evaluation_manifest_sha256"
            ],
            "evaluation_truth_rows_stored": "false",
            "canonical_truth_reopened_terminal_only": "true",
            "frozen_route_receipt_required": "true",
        }
    )
    evaluation["expected_file_hashes"] = {
        "release.json": _hash_entry(
            draft.computed_hashes["evaluation_manifest_sha256"]
        )
    }

    parent = rows[INPUT_ARTIFACT_IDS[5]]
    parent["evidence_label"] = (
        "SCOPE_LIMITED_HARP_V20_AUTHORIZED_CONSUMED_TEST_PARENT_BINDING"
    )
    parent["notes"] = [
        "This v20 consumer fence binds only the immutable original consumption "
        "ledger; no v1-v19 HARP authority, output, "
        "or state is reused."
    ]

    amendment = rows[INPUT_ARTIFACT_IDS[6]]
    amendment["availability"] = "local_and_workstation"
    amendment["evidence_label"] = (
        "SCOPE_LIMITED_HARP_V20_SINGLE_USE_EXECUTION_AMENDMENT_AUTHORIZED_UNCLAIMED"
    )
    amendment_semantics = amendment["semantic_identities"]
    source = draft.amendment_payload["source_snapshot_identity"]
    binding = draft.amendment_payload["authorized_input_binding"]
    if not isinstance(source, Mapping) or not isinstance(binding, Mapping):
        raise ProtocolError("HARP v20 amendment source/input binding is malformed.")
    amendment_semantics.update(
        {
            "authorization_basis": draft.amendment_payload["authorization_basis"],
            "authorization_date": draft.amendment_payload["authorization_date"],
            "authorization_scope": draft.amendment_payload["authorization_scope"],
            "execution_revision": draft.amendment_payload["execution_revision"],
            "authorization_exhausted": "false",
            "amendment_status": "AUTHORIZED_SINGLE_USE_NOT_CONSUMED",
            "amendment_schema_version": draft.amendment_payload["schema_version"],
            "amendment_sha256": draft.amendment_sha256,
            "amendment_hash": draft.amendment_payload["amendment_hash"],
            "authorized_input_binding_hash": binding["input_binding_hash"],
            "scientific_contract_hash": draft.amendment_payload[
                "scientific_contract_hash"
            ],
            "workspace_registration_execution_contract_hash": draft.amendment_payload[
                "workspace_registration_execution_contract_hash"
            ],
            "source_snapshot_schema": source["source_snapshot_schema"],
            "source_snapshot_manifest_sha256": source[
                "source_snapshot_manifest_sha256"
            ],
            "source_snapshot_tree_sha256": source["source_snapshot_tree_sha256"],
            "source_snapshot_member_count": str(source["source_snapshot_member_count"]),
            "preparation_receipt_hash": draft.preparation_receipt_hash,
            "physical_input_receipt_hash": draft.physical_input_receipt_hash,
        }
    )
    amendment["expected_file_hashes"] = {
        authorization.EXECUTION_AMENDMENT_FILENAME: _hash_entry(
            draft.amendment_sha256
        )
    }
    amendment["notes"] = [
        "SCOPE LIMITED - this v20-only amendment is bound to the exact prepared "
        "inputs and source snapshot and remains unclaimed until one launch.",
        "Activation opens no labels, creates no output, and reuses no v1-v19 "
        "HARP amendment, lease, output, cache, route, or scratch state.",
    ]

    output = rows[OUTPUT_ARTIFACT_ID]
    output["evidence_label"] = (
        "SCOPE_LIMITED_HARP_V20_TERMINAL_CONSUMED_TEST_AUTHORIZED_UNCLAIMED"
    )
    output["semantic_identities"].update(
        {
            "authorization_exhausted": "false",
            "amendment_status": "AUTHORIZED_SINGLE_USE_NOT_CONSUMED",
            "execution_amendment_sha256": draft.amendment_sha256,
            "authorized_input_binding_hash": binding["input_binding_hash"],
            "scientific_contract_hash": draft.amendment_payload[
                "scientific_contract_hash"
            ],
            "source_snapshot_manifest_sha256": source[
                "source_snapshot_manifest_sha256"
            ],
            "source_snapshot_tree_sha256": source["source_snapshot_tree_sha256"],
        }
    )
    return rendered


def validate_rendered_workspace(
    registry: Mapping[str, object], catalog: Mapping[str, object]
) -> None:
    experiments = sequence_member(registry, "experiments", "registry experiments")
    match = [
        row
        for row in experiments
        if isinstance(row, Mapping) and row.get("experiment_id") == EXPERIMENT_ID
    ]
    if len(match) != 1:
        raise ProtocolError("HARP v20 rendered registry entry is not unique.")
    experiment = match[0]
    runner = experiment.get("runner")
    if not isinstance(runner, Mapping):
        raise ProtocolError("HARP v20 rendered runner is malformed.")
    artifacts = sequence_member(catalog, "artifacts", "catalog artifacts")
    outputs = [
        row
        for row in artifacts
        if isinstance(row, Mapping) and row.get("artifact_id") == OUTPUT_ARTIFACT_ID
    ]
    if len(outputs) != 1:
        raise ProtocolError("HARP v20 rendered output entry is not unique.")
    required_files = outputs[0].get("required_files")
    if (
        not isinstance(required_files, list)
        or any(type(value) is not str for value in required_files)
        or not set(SOURCE_TRAIN_REQUIRED_OUTPUT_MEMBERS).issubset(required_files)
    ):
        raise ProtocolError(
            "HARP v20 rendered source-train output inventory is incomplete."
        )
    projection = {
        "experiment_id": experiment.get("experiment_id"),
        "stage": experiment.get("stage"),
        "status": experiment.get("status"),
        "claim_scope": experiment.get("claim_scope"),
        "config_path": experiment.get("config_path"),
        "output_artifact_id": experiment.get("output_artifact_id"),
        "output_canonical_path": outputs[0].get("canonical_path"),
        "input_artifact_ids": experiment.get("input_artifact_ids"),
        "preparation_authority_gate": runner.get("preparation_authority_gate"),
        "run_recovery_strategy": runner.get("run_recovery_strategy"),
        "runner_argv": runner.get("argv"),
        "runner_environment": runner.get("environment"),
    }
    authorization.validate_workspace_registration_execution_projection(projection)
    by_artifact_id = {
        str(row.get("artifact_id")): row
        for row in artifacts
        if isinstance(row, Mapping)
    }
    cache_semantics = by_artifact_id.get(INPUT_ARTIFACT_IDS[2], {}).get(
        "semantic_identities"
    )
    development_semantics = by_artifact_id.get(INPUT_ARTIFACT_IDS[3], {}).get(
        "semantic_identities"
    )
    evaluation_semantics = by_artifact_id.get(INPUT_ARTIFACT_IDS[4], {}).get(
        "semantic_identities"
    )
    output_semantics = outputs[0].get("semantic_identities")
    if (
        not isinstance(cache_semantics, Mapping)
        or cache_semantics.get("source_train_row_count") != "9648"
        or cache_semantics.get("source_train_case_count") != "216"
        or cache_semantics.get("target_evaluation_row_count") != "9928"
        or cache_semantics.get("target_evaluation_case_count") != "218"
        or cache_semantics.get("consumed_test_development_case_count") != "0"
        or cache_semantics.get("source_train_and_target_shards_physically_separate")
        != "true"
        or not isinstance(development_semantics, Mapping)
        or development_semantics.get("center_sharded_label_capability") != "true"
        or development_semantics.get("label_capability_shard_count") != "9"
        or development_semantics.get("source_fit_label_scope")
        != "ALL_216_TRAIN_CASES_POOLED_ACROSS_NINE_KNOWN_CENTERS"
        or development_semantics.get("source_capability_state")
        != "SOURCE_TRAIN_CENTER_SCOPED_OPEN_AFTER_ALL_SOURCE_AND_TARGET_MENU_SEALS_AND_BANK_ATTESTATIONS"
        or development_semantics.get(
            "all_eighteen_menu_seals_required_before_any_source_truth"
        )
        != "true"
        or development_semantics.get("source_candidate_pool") != "C_MINUS_q"
        or development_semantics.get("exact_one_capability_per_q") != "true"
        or development_semantics.get("exact_center_coverage_required") != "true"
        or not isinstance(evaluation_semantics, Mapping)
        or evaluation_semantics.get("row_count") != "9928"
        or evaluation_semantics.get("case_count") != "218"
        or evaluation_semantics.get("contains_development_rows") != "false"
        or not isinstance(output_semantics, Mapping)
        or output_semantics.get("routing_regime")
        != "KNOWN_CENTER_SOURCE_TRAIN_DEVELOPMENT_TO_FULL_TEST"
        or output_semantics.get("source_train_context_count") != "9"
        or output_semantics.get("target_evaluation_context_count") != "9"
        or output_semantics.get("total_physical_context_count") != "18"
        or output_semantics.get("source_train_target_classifier_task_count")
        != "81"
        or output_semantics.get("total_classifier_fit_count") != "810"
        or output_semantics.get("H_q_r_seven_expert_folds_used") != "false"
        or output_semantics.get("policy_family")
        != "patch_evidence_risk_aligned_action_selection_with_crossfitted_winner_harm_gate"
        or output_semantics.get("execution_revision") != EXECUTION_REVISION
        or output_semantics.get("patch_features_target_fit") != "false"
        or output_semantics.get("risk_selection_nested") != "true"
        or output_semantics.get("risk_penalty_scales") != "0.5_1.0_2.0"
        or output_semantics.get("predecessor_v1_through_v19_state_reused") != "false"
        or output_semantics.get("case_local_action_eligibility") != "true"
        or output_semantics.get("signed_action_outcomes") != "true"
        or output_semantics.get("source_frontier_required") != "true"
        or output_semantics.get("nested_stack_folds") != "4"
        or output_semantics.get("source_oof_selection_estimand")
        != "EQUAL_CENTERS_EQUAL_CLASSES_EQUAL_SUPPORTING_CASES"
        or output_semantics.get("nested_center_stratified_outer_folds") != "5"
        or output_semantics.get("nested_center_stratified_inner_folds") != "4"
        or output_semantics.get("minimum_routed_oof_cases") != "18"
        or output_semantics.get("minimum_routed_oof_centers") != "6"
        or output_semantics.get("minimum_routed_oof_cases_per_center") != "2"
        or output_semantics.get("target_case_features_may_fit_or_calibrate_router")
        != "false"
    ):
        raise ProtocolError("HARP v20 rendered train/full-test protocol drifted.")
    for artifact_id in (*INPUT_ARTIFACT_IDS[2:], OUTPUT_ARTIFACT_ID):
        matching_rows = [
            row
            for row in artifacts
            if isinstance(row, Mapping) and row.get("artifact_id") == artifact_id
        ]
        if len(matching_rows) != 1:
            raise ProtocolError("HARP v20 rendered catalog identity is ambiguous.")
        semantics = matching_rows[0].get("semantic_identities")
        if (
            not isinstance(semantics, Mapping)
            or semantics.get("execution_authorized") != "true"
            or semantics.get("consumed_test_reuse_authorized") != "true"
            or semantics.get("fresh_evidence") != "false"
            or semantics.get("may_feed_another_experiment") != "false"
        ):
            raise ProtocolError("HARP v20 rendered catalog claim boundary drifted.")


def yaml_bytes(value: Mapping[str, object]) -> bytes:
    raw = yaml.safe_dump(
        dict(value),
        sort_keys=False,
        allow_unicode=False,
        default_flow_style=False,
        width=1000,
    ).encode("utf-8")
    if yaml_mapping(raw, label="rendered YAML") != dict(value):
        raise ProtocolError("HARP v20 deterministic YAML renderer changed semantics.")
    return raw


def yaml_mapping(raw: bytes, *, label: str) -> dict[str, object]:
    try:
        value = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ProtocolError(f"HARP v20 {label} is not readable YAML.") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"HARP v20 {label} must be a mapping.")
    return value


def mapping_member(
    value: Mapping[str, object], key: str, label: str
) -> dict[str, object]:
    member = value.get(key)
    if not isinstance(member, dict):
        raise ProtocolError(f"HARP v20 {label} is malformed.")
    return member


def sequence_member(
    value: Mapping[str, object], key: str, label: str
) -> list[object]:
    member = value.get(key)
    if not isinstance(member, list):
        raise ProtocolError(f"HARP v20 {label} is malformed.")
    return member


def _hash_entry(digest: str) -> dict[str, str]:
    return {"algorithm": "sha256", "digest": digest}


__all__ = (
    "RenderedActivationWorkspace",
    "render_activation_workspace",
    "validate_rendered_workspace",
    "yaml_bytes",
    "yaml_mapping",
)
