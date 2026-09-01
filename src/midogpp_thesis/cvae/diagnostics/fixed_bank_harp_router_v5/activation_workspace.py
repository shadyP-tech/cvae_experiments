"""Pure workspace projection for HARP v5 activation.

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
from .amendment_publisher import HarpV5AmendmentDraft
from .config import HarpStage90V5Config, INPUT_ARTIFACT_IDS
from .identity import EXPERIMENT_ID, OUTPUT_ARTIFACT_ID, claim_boundary_payload
from .input_surfaces import CACHE_INDEX, CONTENT_INDEX
from .preparation import PREPARATION_RECEIPT


@dataclass(frozen=True, slots=True)
class RenderedActivationWorkspace:
    final_config_bytes: bytes
    final_registry_bytes: bytes
    final_catalog_bytes: bytes
    authorized_config: HarpStage90V5Config


def render_activation_workspace(
    *,
    original_config_bytes: bytes,
    original_registry_bytes: bytes,
    original_catalog_bytes: bytes,
    draft: HarpV5AmendmentDraft,
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
    original: Mapping[str, object], draft: HarpV5AmendmentDraft
) -> dict[str, object]:
    rendered = copy.deepcopy(dict(original))
    experiment = mapping_member(rendered, "experiment", "config experiment")
    inputs = mapping_member(rendered, "inputs", "config inputs")
    if experiment.get("status") != "planned" or experiment.get(
        "execution_authorized"
    ) is not False:
        raise ProtocolError("HARP v5 config is not in the planned activation state.")
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
        raise ProtocolError("HARP v5 planned config already binds execution hashes.")
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
        raise ProtocolError("HARP v5 planned registry entry is absent or ambiguous.")
    matches[0]["status"] = "diagnostic"
    notes = matches[0].get("notes")
    if isinstance(notes, list) and notes:
        notes[0] = (
            "SCOPE LIMITED - v5 is separately authorized for one terminal "
            "consumed-test diagnostic; its distinct amendment remains unclaimed "
            "until launch and no result may be promoted."
        )
    return rendered


def _render_catalog_mapping(
    original: Mapping[str, object], draft: HarpV5AmendmentDraft
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
        raise ProtocolError("HARP v5 catalog activation inventory is incomplete.")
    for row in rows.values():
        semantics = row.get("semantic_identities")
        if not isinstance(semantics, dict):
            raise ProtocolError("HARP v5 catalog semantic identities are malformed.")
        if semantics.get("execution_authorized") != "false":
            raise ProtocolError("HARP v5 catalog is not in its planned state.")
        semantics["execution_authorized"] = "true"
        semantics["consumed_test_reuse_authorized"] = "true"

    cache = rows[INPUT_ARTIFACT_IDS[2]]
    cache["availability"] = "workstation_only"
    cache["evidence_label"] = (
        "SCOPE_LIMITED_HARP_V5_AUTHORIZED_LABEL_FREE_CONSUMED_TEST_CACHE"
    )
    cache_semantics = cache["semantic_identities"]
    cache_semantics.update(
        {
            "test_cache_content_sha256": draft.computed_hashes[
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
        "This v5-only cache was rebuilt label-blind from the immutable canonical "
        "consumed-test cache and does not reuse any v1/v2 HARP cache or output."
    ]

    development = rows[INPUT_ARTIFACT_IDS[3]]
    development["availability"] = "workstation_only"
    development["evidence_label"] = (
        "SCOPE_LIMITED_HARP_V5_AUTHORIZED_DEVELOPMENT_LABEL_CAPABILITY"
    )
    development["semantic_identities"].update(
        {
            "manifest_sha256": draft.computed_hashes[
                "development_manifest_sha256"
            ],
            "mixed_patch_labels_within_case_allowed": "true",
        }
    )
    development["expected_file_hashes"] = {
        "manifest.csv": _hash_entry(
            draft.computed_hashes["development_manifest_sha256"]
        )
    }

    evaluation = rows[INPUT_ARTIFACT_IDS[4]]
    evaluation["availability"] = "workstation_only"
    evaluation["evidence_label"] = (
        "SCOPE_LIMITED_HARP_V5_AUTHORIZED_EVALUATION_LABEL_CAPABILITY"
    )
    evaluation["semantic_identities"].update(
        {
            "manifest_sha256": draft.computed_hashes[
                "evaluation_manifest_sha256"
            ],
            "mixed_patch_labels_within_case_allowed": "true",
        }
    )
    evaluation["expected_file_hashes"] = {
        "manifest.csv": _hash_entry(
            draft.computed_hashes["evaluation_manifest_sha256"]
        )
    }

    parent = rows[INPUT_ARTIFACT_IDS[5]]
    parent["evidence_label"] = (
        "SCOPE_LIMITED_HARP_V5_AUTHORIZED_CONSUMED_TEST_PARENT_BINDING"
    )
    parent["notes"] = [
        "This v5 consumer fence binds only the immutable original consumption "
        "ledger; no v1/v2/v3/v4 HARP authority, output, or state is reused."
    ]

    amendment = rows[INPUT_ARTIFACT_IDS[6]]
    amendment["availability"] = "local_and_workstation"
    amendment["evidence_label"] = (
        "SCOPE_LIMITED_HARP_V5_SINGLE_USE_EXECUTION_AMENDMENT_AUTHORIZED_UNCLAIMED"
    )
    amendment_semantics = amendment["semantic_identities"]
    source = draft.amendment_payload["source_snapshot_identity"]
    binding = draft.amendment_payload["authorized_input_binding"]
    if not isinstance(source, Mapping) or not isinstance(binding, Mapping):
        raise ProtocolError("HARP v5 amendment source/input binding is malformed.")
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
        "SCOPE LIMITED - this v5-only amendment is bound to the exact prepared "
        "inputs and source snapshot and remains unclaimed until one launch.",
        "Activation opens no labels, creates no output, and reuses no v1/v2/v3/v4 "
        "HARP amendment, lease, output, cache, route, or scratch state.",
    ]

    output = rows[OUTPUT_ARTIFACT_ID]
    output["evidence_label"] = (
        "SCOPE_LIMITED_HARP_V5_TERMINAL_CONSUMED_TEST_AUTHORIZED_UNCLAIMED"
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
        raise ProtocolError("HARP v5 rendered registry entry is not unique.")
    experiment = match[0]
    runner = experiment.get("runner")
    if not isinstance(runner, Mapping):
        raise ProtocolError("HARP v5 rendered runner is malformed.")
    artifacts = sequence_member(catalog, "artifacts", "catalog artifacts")
    outputs = [
        row
        for row in artifacts
        if isinstance(row, Mapping) and row.get("artifact_id") == OUTPUT_ARTIFACT_ID
    ]
    if len(outputs) != 1:
        raise ProtocolError("HARP v5 rendered output entry is not unique.")
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
    for artifact_id in (*INPUT_ARTIFACT_IDS[2:], OUTPUT_ARTIFACT_ID):
        matching_rows = [
            row
            for row in artifacts
            if isinstance(row, Mapping) and row.get("artifact_id") == artifact_id
        ]
        if len(matching_rows) != 1:
            raise ProtocolError("HARP v5 rendered catalog identity is ambiguous.")
        semantics = matching_rows[0].get("semantic_identities")
        if (
            not isinstance(semantics, Mapping)
            or semantics.get("execution_authorized") != "true"
            or semantics.get("consumed_test_reuse_authorized") != "true"
            or semantics.get("fresh_evidence") != "false"
            or semantics.get("may_feed_another_experiment") != "false"
        ):
            raise ProtocolError("HARP v5 rendered catalog claim boundary drifted.")


def yaml_bytes(value: Mapping[str, object]) -> bytes:
    raw = yaml.safe_dump(
        dict(value),
        sort_keys=False,
        allow_unicode=False,
        default_flow_style=False,
        width=1000,
    ).encode("utf-8")
    if yaml_mapping(raw, label="rendered YAML") != dict(value):
        raise ProtocolError("HARP v5 deterministic YAML renderer changed semantics.")
    return raw


def yaml_mapping(raw: bytes, *, label: str) -> dict[str, object]:
    try:
        value = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ProtocolError(f"HARP v5 {label} is not readable YAML.") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"HARP v5 {label} must be a mapping.")
    return value


def mapping_member(
    value: Mapping[str, object], key: str, label: str
) -> dict[str, object]:
    member = value.get(key)
    if not isinstance(member, dict):
        raise ProtocolError(f"HARP v5 {label} is malformed.")
    return member


def sequence_member(
    value: Mapping[str, object], key: str, label: str
) -> list[object]:
    member = value.get(key)
    if not isinstance(member, list):
        raise ProtocolError(f"HARP v5 {label} is malformed.")
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
