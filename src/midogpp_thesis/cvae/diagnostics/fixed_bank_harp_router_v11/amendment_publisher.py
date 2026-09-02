"""Mutation-free one-shot publication after separate HARP v11 activation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import hashlib
import os
from pathlib import Path
from types import MappingProxyType

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_bytes, canonical_hash
from ...runtime.artifact_io import read_json, sha256_file
from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from .input_lineage import validate_physical_inputs
from .preparation import (
    CANONICAL_CACHE_CONTENT_HASH,
    CANONICAL_CACHE_ROW_ORDER_HASH,
    CANONICAL_MANIFEST_SHA256,
    CANONICAL_SOURCE_TRAIN_TENSOR_SHA256,
)
from . import authorization
from .activation_paths import RepositoryBoundary
from .config import HarpStage90V11Config, load_config
from .identity import EXPERIMENT_ID, claim_boundary_payload
from .input_surfaces import (
    CONTENT_INDEX,
    DEVELOPMENT_ROLE,
    SOURCE_LABEL_INDEX_SCHEMA,
    V11_CACHE_IDENTITY,
    HarpConsumedCacheIndex,
    _read_evaluation_release_descriptor,
    load_cache_index,
)
from .preparation import (
    CASE_PARTITION,
    LABEL_FREE_BARRIER,
    LABEL_FREE_CONTENT_INDEX,
    PREPARATION_RECEIPT,
)
from .preparation_contracts import (
    EXPECTED_SOURCE_TRAIN_CASE_COUNT,
    EXPECTED_SOURCE_TRAIN_ROW_COUNT,
)
from .workspace_paths import resolve_harp_v11_workspace_paths


AUTHORIZATION_BASIS = authorization.AUTHORIZATION_BASIS
AMENDMENT_FILENAME = authorization.EXECUTION_AMENDMENT_FILENAME

_PREPARATION_RECEIPT_KEYS = {
    "schema_version",
    "experiment_id",
    "status",
    "canonical_source_train_tensor_sha256",
    "canonical_source_train_row_order_hash",
    "canonical_target_test_cache_content_hash",
    "canonical_target_test_row_order_hash",
    "canonical_test_manifest_sha256",
    "parent_ledger_sha256",
    "partition_hash",
    "label_free_barrier_sha256",
    "label_free_content_index_sha256",
    "pre_manifest_cache_content_sha256",
    "prepared_cache_index_hash",
    "prepared_row_count",
    "development_manifest_sha256",
    "evaluation_manifest_sha256",
    "source_label_capability_artifact_kind",
    "source_label_capability_shard_count",
    "source_label_capability_index_contains_labels",
    "source_label_capability_requires_fold_physical_surface_seal",
    "source_label_q_access_requires_prelabel_prediction_seal",
    "source_label_fit_scope_excludes_outer_H_and_heldout_q",
    "evaluation_artifact_kind",
    "evaluation_truth_rows_published_during_preparation",
    "evaluation_release_requires_frozen_route_receipt",
    "source_train_and_target_test_cases_disjoint",
    "source_development_row_count",
    "source_development_case_count",
    "target_evaluation_row_count",
    "target_evaluation_case_count",
    "test_development_case_count",
    "mixed_patch_labels_within_case_supported",
    "partition_selected_without_labels",
    "cache_fsynced_and_independently_validated_before_manifest_open",
    "execution_amendment_created",
    "execution_authorized",
    "publication_status",
    "terminal_decision",
    "fresh_evidence",
    "may_feed_stage60_or_stage70",
    "may_feed_another_experiment",
    "receipt_hash",
}


@dataclass(frozen=True, slots=True)
class HarpV11AmendmentPublicationReceipt:
    amendment_path: Path
    amendment_sha256: str
    amendment_hash: str
    input_binding_hash: str
    scientific_contract_hash: str
    workspace_registration_execution_contract_hash: str
    source_snapshot_manifest_sha256: str
    source_snapshot_tree_sha256: str
    preparation_receipt_hash: str
    physical_input_receipt_hash: str
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        path = Path(self.amendment_path)
        if not path.is_absolute() or not path.is_file() or path.is_symlink() or sha256_file(path) != self.amendment_sha256:
            raise ProtocolError("HARP v11 amendment publication receipt path is unsafe.")
        object.__setattr__(self, "receipt_hash", canonical_hash(self._base_payload()))

    def _base_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_harp_stage90_amendment_publication_receipt_v11",
            "experiment_id": EXPERIMENT_ID,
            "amendment_path": self.amendment_path.as_posix(),
            "amendment_sha256": self.amendment_sha256,
            "amendment_hash": self.amendment_hash,
            "input_binding_hash": self.input_binding_hash,
            "scientific_contract_hash": self.scientific_contract_hash,
            "workspace_registration_execution_contract_hash": self.workspace_registration_execution_contract_hash,
            "source_snapshot_manifest_sha256": self.source_snapshot_manifest_sha256,
            "source_snapshot_tree_sha256": self.source_snapshot_tree_sha256,
            "preparation_receipt_hash": self.preparation_receipt_hash,
            "physical_input_receipt_hash": self.physical_input_receipt_hash,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._base_payload(), "receipt_hash": self.receipt_hash}


@dataclass(frozen=True, slots=True)
class HarpV11AmendmentDraft:
    """Read-only, fully validated bytes for the one-shot v11 amendment."""

    repository_root: Path
    amendment_path: Path
    amendment_raw: bytes
    amendment_sha256: str
    amendment_payload: Mapping[str, object]
    authorized_config: HarpStage90V11Config
    computed_hashes: Mapping[str, str]
    cache_index_sha256: str
    content_index_sha256: str
    preparation_receipt_sha256: str
    development_member_sha256: Mapping[str, str]
    partition_hash: str
    preparation_receipt_hash: str
    physical_input_receipt_hash: str

    def __post_init__(self) -> None:
        if hashlib.sha256(self.amendment_raw).hexdigest() != self.amendment_sha256:
            raise ProtocolError("HARP v11 amendment draft byte identity drifted.")
        if self.amendment_raw != canonical_bytes(self.amendment_payload) + b"\n":
            raise ProtocolError("HARP v11 amendment draft is not canonical JSON.")

    def to_payload(self) -> dict[str, object]:
        source = self.amendment_payload.get("source_snapshot_identity")
        binding = self.amendment_payload.get("authorized_input_binding")
        if not isinstance(source, Mapping) or not isinstance(binding, Mapping):
            raise ProtocolError("HARP v11 amendment draft metadata is malformed.")
        return {
            "schema_version": "midogpp_harp_stage90_amendment_draft_v11",
            "experiment_id": EXPERIMENT_ID,
            "amendment_sha256": self.amendment_sha256,
            "amendment_hash": self.amendment_payload["amendment_hash"],
            "input_binding_hash": binding["input_binding_hash"],
            "scientific_contract_hash": self.amendment_payload[
                "scientific_contract_hash"
            ],
            "workspace_registration_execution_contract_hash": (
                self.amendment_payload[
                    "workspace_registration_execution_contract_hash"
                ]
            ),
            "source_snapshot_manifest_sha256": source[
                "source_snapshot_manifest_sha256"
            ],
            "source_snapshot_tree_sha256": source[
                "source_snapshot_tree_sha256"
            ],
            "preparation_receipt_hash": self.preparation_receipt_hash,
            "physical_input_receipt_hash": self.physical_input_receipt_hash,
            "filesystem_mutations": 0,
            "labels_opened": False,
            "output_created": False,
        }


def publish_harp_v11_execution_amendment(
    config: HarpStage90V11Config,
    *,
    expert_bank_root: str | Path,
    generation_lock_root: str | Path,
    prepared_cache_root: str | Path,
    development_manifest_path: str | Path,
    evaluation_manifest_path: str | Path,
    parent_ledger_path: str | Path,
    amendment_path: str | Path,
    authorization_basis: str,
    authorization_date: str,
    repository_root: str | Path,
) -> HarpV11AmendmentPublicationReceipt:
    """Reject the retired unjournaled publisher before any input access."""

    raise ProtocolError(
        "HARP v11 direct amendment publication is disabled; use the durable "
        "activate-fixed-bank-harp-router-v11 transaction."
    )


def build_harp_v11_execution_amendment_draft(
    config: HarpStage90V11Config,
    *,
    expert_bank_root: str | Path,
    generation_lock_root: str | Path,
    prepared_cache_root: str | Path,
    development_manifest_path: str | Path,
    evaluation_manifest_path: str | Path,
    parent_ledger_path: str | Path,
    amendment_path: str | Path,
    authorization_basis: str,
    authorization_date: str,
    repository_root: str | Path,
) -> HarpV11AmendmentDraft:
    """Authenticate planned inputs and render amendment bytes without writing."""

    if type(config) is not HarpStage90V11Config or config.execution_authorized:
        raise ProtocolError("HARP v11 amendment drafting requires the planned config.")
    authorization.validate_activation_metadata(authorization_basis, authorization_date)
    if config.expected_execution_amendment_sha256 is not None:
        raise ProtocolError("HARP v11 amendment is already bound in configuration.")
    boundary = RepositoryBoundary.open(repository_root)
    repository = boundary.resolved_root
    registered_config = boundary.member(
        authorization.WORKSPACE_CONFIG_RELATIVE_PATH,
        label="registered config",
        kind="file",
    )
    if config.source_path != registered_config:
        raise ProtocolError("HARP v11 publisher config is not the registered source.")
    if load_config(config.source_path) != config:
        raise ProtocolError("HARP v11 publisher config changed after load.")
    try:
        publication_path = boundary.path(
            amendment_path,
            label="execution amendment",
            kind="optional",
        )
    except ProtocolError as exc:
        raise ProtocolError(
            "HARP v11 amendment publication path drifted from its catalog identity."
        ) from exc
    expected_path = boundary.member(
        authorization.WORKSPACE_AMENDMENT_RELATIVE_PATH,
        label="registered execution amendment",
        kind="optional",
    )
    if publication_path != expected_path:
        raise ProtocolError("HARP v11 amendment publication path drifted from its catalog identity.")
    lease = boundary.path(
        authorization.lease_path(repository),
        label="authorization lease",
        kind="absent",
    )
    if os.path.lexists(lease):
        raise ProtocolError("HARP v11 authorization lease already exists.")

    bank = boundary.path(expert_bank_root, label="expert bank", kind="directory")
    generation = boundary.path(
        generation_lock_root, label="generation lock", kind="directory"
    )
    cache_root = boundary.path(
        prepared_cache_root, label="prepared cache", kind="directory"
    )
    development = boundary.path(
        development_manifest_path, label="development manifest", kind="file"
    )
    evaluation = boundary.path(
        evaluation_manifest_path, label="evaluation release descriptor", kind="file"
    )
    parent = boundary.path(parent_ledger_path, label="parent ledger", kind="file")
    paths = {bank, generation, cache_root, development, evaluation, parent, publication_path}
    if len(paths) != 7:
        raise ProtocolError("HARP v11 amendment input paths overlap.")
    catalog_paths = resolve_harp_v11_workspace_paths(
        repository,
        require_prepared=True,
    )
    expected_paths = {
        "expert bank": catalog_paths.expert_bank_root,
        "generation lock": catalog_paths.generation_lock_root,
        "prepared cache": catalog_paths.prepared_cache_root,
        "development manifest": catalog_paths.development_manifest_path,
        "evaluation release descriptor": catalog_paths.evaluation_manifest_path,
        "parent ledger": catalog_paths.parent_ledger_path,
        "execution amendment": catalog_paths.amendment_path,
    }
    observed_paths = {
        "expert bank": bank,
        "generation lock": generation,
        "prepared cache": cache_root,
        "development manifest": development,
        "evaluation release descriptor": evaluation,
        "parent ledger": parent,
        "execution amendment": publication_path,
    }
    drifted = [
        label
        for label, expected in expected_paths.items()
        if observed_paths[label] != expected
    ]
    if drifted:
        raise ProtocolError(
            "HARP v11 activation inputs drifted from catalog identities: "
            + ", ".join(drifted)
            + "."
        )
    content = read_json(cache_root / CONTENT_INDEX)
    content_base = {key: value for key, value in content.items() if key != "content_index_hash"}
    cache_content = content.get("content_index_hash")
    if type(cache_content) is not str or cache_content != canonical_hash(content_base):
        raise ProtocolError("HARP v11 prepared cache content identity drifted.")
    computed = {
        "test_cache_content_sha256": cache_content,
        "development_manifest_sha256": sha256_file(development),
        "evaluation_manifest_sha256": sha256_file(evaluation),
        "parent_ledger_sha256": sha256_file(parent),
    }
    read_json(parent)
    if any(config.expected_hashes.get(role) is not None for role in computed):
        raise ProtocolError("HARP v11 planned config pre-binds prepared-input hashes.")
    bound = replace(
        config,
        input_locations={
            **dict(config.input_locations), "expert_bank_root": bank.as_posix(),
            "generation_lock_root": generation.as_posix(), "test_cache_root": cache_root.as_posix(),
            "development_manifest_path": development.as_posix(),
            "evaluation_manifest_path": evaluation.as_posix(), "parent_ledger_path": parent.as_posix(),
            "execution_amendment_path": publication_path.as_posix(),
        },
        expected_hashes={
            **dict(config.expected_hashes),
            **computed,
            "execution_amendment_sha256": None,
        },
    )
    cache = load_cache_index(bound)
    _read_evaluation_release_descriptor(
        evaluation,
        expected_sha256=computed["evaluation_manifest_sha256"],
        cache=cache,
    )
    preparation_hash = _validate_preparation_receipt(cache, computed)
    physical = validate_physical_inputs(bound, cache)
    authorized = replace(
        config,
        expected_hashes={
            **dict(config.expected_hashes),
            **computed,
            "execution_amendment_sha256": None,
        },
        execution_authorized=True,
        claim_boundary=claim_boundary_payload(execution_authorized=True),
    )
    payload = authorization.canonical_execution_amendment_payload(
        authorized,
        authorization_basis=authorization_basis,
        authorization_date=authorization_date,
        repo_root=repository,
    )
    authorization.validate_execution_amendment_payload(
        payload, authorized, repo_root=repository
    )
    raw = canonical_bytes(payload) + b"\n"
    preparation = read_json(cache.root / PREPARATION_RECEIPT)
    partition_hash = preparation.get("partition_hash")
    if type(partition_hash) is not str:
        raise ProtocolError("HARP v11 preparation partition hash is absent.")
    return HarpV11AmendmentDraft(
        repository_root=repository,
        amendment_path=publication_path,
        amendment_raw=raw,
        amendment_sha256=hashlib.sha256(raw).hexdigest(),
        amendment_payload=MappingProxyType(dict(payload)),
        authorized_config=authorized,
        computed_hashes=MappingProxyType(dict(computed)),
        cache_index_sha256=sha256_file(cache.root / "manifests/cache_index.json"),
        content_index_sha256=sha256_file(cache.root / CONTENT_INDEX),
        preparation_receipt_sha256=sha256_file(cache.root / PREPARATION_RECEIPT),
        development_member_sha256=MappingProxyType(
            _source_label_member_sha256(
                development,
                cache_index_hash=cache.cache_hash,
                pre_manifest_cache_content_sha256=str(
                    preparation["pre_manifest_cache_content_sha256"]
                ),
            )
        ),
        partition_hash=partition_hash,
        preparation_receipt_hash=preparation_hash,
        physical_input_receipt_hash=str(physical.receipt_hash),
    )


def publish_harp_v11_amendment_draft_exclusive(
    draft: HarpV11AmendmentDraft,
) -> HarpV11AmendmentPublicationReceipt:
    """Reject direct amendment mutation outside the durable transaction."""

    if type(draft) is not HarpV11AmendmentDraft:
        raise ProtocolError("HARP v11 exclusive publisher requires a typed draft.")
    raise ProtocolError(
        "HARP v11 direct amendment publication is disabled; use the durable "
        "activate-fixed-bank-harp-router-v11 transaction."
    )


def _validate_preparation_receipt(
    cache: HarpConsumedCacheIndex, computed: Mapping[str, str],
) -> str:
    path = cache.root / PREPARATION_RECEIPT
    receipt_member = PREPARATION_RECEIPT.as_posix()
    label_free_member = LABEL_FREE_CONTENT_INDEX.as_posix()
    if (
        cache.content_sha256 != computed.get("test_cache_content_sha256")
        or cache.member_sha256.get(receipt_member) != sha256_file(path)
    ):
        raise ProtocolError("HARP v11 preparation receipt is not cache-bound.")
    receipt = read_json(path)
    base = {key: value for key, value in receipt.items() if key != "receipt_hash"}
    partition = read_json(cache.root / CASE_PARTITION)
    barrier = read_json(cache.root / LABEL_FREE_BARRIER)
    barrier_base = {
        key: value for key, value in barrier.items() if key != "barrier_hash"
    }
    label_free_path = cache.root / LABEL_FREE_CONTENT_INDEX
    label_free_sha256 = sha256_file(label_free_path)
    if cache.member_sha256.get(label_free_member) != label_free_sha256:
        raise ProtocolError("HARP v11 label-free content index is not cache-bound.")
    label_free = read_json(label_free_path)
    label_free_base = {
        key: value
        for key, value in label_free.items()
        if key != "content_index_hash"
    }

    # The final content index contains the preparation receipt itself.  The
    # pre-manifest identity was sealed before that receipt existed, so rebuild
    # it from the final closed-world inventory minus exactly that one member.
    pre_manifest_members = dict(cache.member_sha256)
    if pre_manifest_members.pop(receipt_member, None) is None:
        raise ProtocolError("HARP v11 final cache lacks its preparation receipt.")
    pre_manifest_base = {
        "schema_version": V11_CACHE_IDENTITY.content_schema,
        "members": dict(sorted(pre_manifest_members.items())),
    }
    label_free_members = label_free.get("members")
    expected_label_free_members = {
        key: value
        for key, value in pre_manifest_members.items()
        if key != label_free_member
    }
    fixed_values = {
        "schema_version": "midogpp_harp_source_train_full_test_preparation_receipt_v11",
        "experiment_id": EXPERIMENT_ID,
        "status": "PREPARED_INPUTS_NO_EXECUTION_AUTHORITY",
        "canonical_source_train_tensor_sha256": CANONICAL_SOURCE_TRAIN_TENSOR_SHA256,
        "canonical_target_test_cache_content_hash": CANONICAL_CACHE_CONTENT_HASH,
        "canonical_target_test_row_order_hash": CANONICAL_CACHE_ROW_ORDER_HASH,
        "canonical_test_manifest_sha256": CANONICAL_MANIFEST_SHA256,
        "parent_ledger_sha256": computed["parent_ledger_sha256"],
        "prepared_cache_index_hash": cache.cache_hash,
        "prepared_row_count": len(cache.rows),
        "development_manifest_sha256": computed["development_manifest_sha256"],
        "evaluation_manifest_sha256": computed["evaluation_manifest_sha256"],
        "source_label_capability_artifact_kind": (
            "center_sharded_source_label_capability"
        ),
        "source_label_capability_shard_count": 9,
        "source_label_capability_index_contains_labels": False,
        "source_label_capability_requires_fold_physical_surface_seal": True,
        "source_label_q_access_requires_prelabel_prediction_seal": True,
        "source_label_fit_scope_excludes_outer_H_and_heldout_q": True,
        "evaluation_artifact_kind": "sealed_label_free_release_descriptor",
        "evaluation_truth_rows_published_during_preparation": False,
        "evaluation_release_requires_frozen_route_receipt": True,
        "source_train_and_target_test_cases_disjoint": True,
        "source_development_row_count": 9648,
        "source_development_case_count": 216,
        "target_evaluation_row_count": 9928,
        "target_evaluation_case_count": 218,
        "test_development_case_count": 0,
        "mixed_patch_labels_within_case_supported": True,
        "partition_selected_without_labels": True,
        "cache_fsynced_and_independently_validated_before_manifest_open": True,
        "execution_amendment_created": False,
        "execution_authorized": False,
        "publication_status": "POST_HOC_CONSUMED_TEST_SENSITIVITY",
        "terminal_decision": "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE",
        "fresh_evidence": False,
        "may_feed_stage60_or_stage70": False,
        "may_feed_another_experiment": False,
    }
    if (
        set(receipt) != _PREPARATION_RECEIPT_KEYS
        or receipt.get("receipt_hash") != canonical_hash(base)
        or any(receipt.get(key) != value for key, value in fixed_values.items())
        or not _is_sha256(receipt.get("canonical_source_train_row_order_hash"))
        or receipt.get("partition_hash") != canonical_hash(partition)
        or receipt.get("label_free_barrier_sha256")
        != sha256_file(cache.root / LABEL_FREE_BARRIER)
        or receipt.get("label_free_content_index_sha256") != label_free_sha256
        or receipt.get("pre_manifest_cache_content_sha256")
        != canonical_hash(pre_manifest_base)
        or barrier.get("barrier_hash") != canonical_hash(barrier_base)
        or barrier.get("partition_hash") != receipt.get("partition_hash")
        or barrier.get("canonical_scoring_manifest_opened") is not False
        or label_free.get("content_index_hash") != canonical_hash(label_free_base)
        or not isinstance(label_free_members, Mapping)
        or dict(label_free_members) != expected_label_free_members
    ):
        raise ProtocolError("HARP v11 preparation receipt drifted.")
    return str(receipt["receipt_hash"])


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _source_label_member_sha256(
    index_path: Path,
    *,
    cache_index_hash: str,
    pre_manifest_cache_content_sha256: str,
) -> dict[str, str]:
    index = read_json(index_path)
    base = {key: value for key, value in index.items() if key != "index_hash"}
    raw_shards = index.get("shards")
    expected_keys = {
        "schema_version",
        "experiment_id",
        "artifact_role",
        "split_role",
        "cache_index_hash",
        "pre_manifest_cache_content_sha256",
        "source_train_tensor_sha256",
        "shards",
        "row_count",
        "case_count",
        "labels_stored_in_index",
        "capability_state",
        "publication_status",
        "terminal_decision",
        "fresh_evidence",
        "may_feed_stage60_or_stage70",
        "may_feed_another_experiment",
        "index_hash",
    }
    if (
        index_path.name != "index.json"
        or set(index) != expected_keys
        or index.get("schema_version") != SOURCE_LABEL_INDEX_SCHEMA
        or index.get("experiment_id") != EXPERIMENT_ID
        or index.get("artifact_role")
        != "center_sharded_source_label_capability"
        or index.get("split_role") != DEVELOPMENT_ROLE
        or index.get("cache_index_hash") != cache_index_hash
        or index.get("pre_manifest_cache_content_sha256")
        != pre_manifest_cache_content_sha256
        or index.get("source_train_tensor_sha256")
        != CANONICAL_SOURCE_TRAIN_TENSOR_SHA256
        or not isinstance(raw_shards, list)
        or len(raw_shards) != len(CENTERS)
        or index.get("row_count") != EXPECTED_SOURCE_TRAIN_ROW_COUNT
        or index.get("case_count") != EXPECTED_SOURCE_TRAIN_CASE_COUNT
        or index.get("labels_stored_in_index") is not False
        or index.get("capability_state")
        != "CENTER_SCOPED_OPEN_AFTER_FOLD_PHYSICAL_SURFACE_SEAL"
        or index.get("publication_status")
        != "POST_HOC_CONSUMED_TEST_SENSITIVITY"
        or index.get("terminal_decision")
        != "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"
        or index.get("fresh_evidence") is not False
        or index.get("may_feed_stage60_or_stage70") is not False
        or index.get("may_feed_another_experiment") is not False
        or index.get("index_hash") != canonical_hash(base)
    ):
        raise ProtocolError("HARP v11 source-label index is malformed.")
    output = {"index.json": sha256_file(index_path)}
    observed_centers: list[str] = []
    total_rows = 0
    total_cases = 0
    for raw in raw_shards:
        if not isinstance(raw, Mapping) or set(raw) != {
            "center",
            "relative_path",
            "sha256",
            "row_count",
            "case_count",
            "ordered_key_hash",
        }:
            raise ProtocolError("HARP v11 source-label shard index is malformed.")
        center = str(raw.get("center", ""))
        relative = str(raw.get("relative_path", ""))
        digest = raw.get("sha256")
        row_count = raw.get("row_count")
        case_count = raw.get("case_count")
        relative_path = Path(relative)
        if (
            center not in CENTERS
            or center in observed_centers
            or relative != f"by_center/center_{center}.csv"
            or relative in output
            or not relative
            or relative_path.is_absolute()
            or any(part in {"", ".", ".."} for part in relative_path.parts)
            or not _is_sha256(digest)
            or type(row_count) is not int
            or row_count < 1
            or type(case_count) is not int
            or case_count < 1
            or not _is_sha256(raw.get("ordered_key_hash"))
        ):
            raise ProtocolError("HARP v11 source-label shard identity drifted.")
        candidate = index_path.parent / relative_path
        if candidate.is_symlink():
            raise ProtocolError("HARP v11 source-label shard bytes drifted.")
        member = candidate.resolve()
        if (
            not member.is_relative_to(index_path.parent.resolve())
            or not member.is_file()
            or sha256_file(member) != digest
        ):
            raise ProtocolError("HARP v11 source-label shard bytes drifted.")
        output[relative] = str(digest)
        observed_centers.append(center)
        total_rows += row_count
        total_cases += case_count
    if (
        tuple(observed_centers) != CENTERS
        or total_rows != EXPECTED_SOURCE_TRAIN_ROW_COUNT
        or total_cases != EXPECTED_SOURCE_TRAIN_CASE_COUNT
    ):
        raise ProtocolError("HARP v11 source-label shard coverage drifted.")
    return output


__all__ = (
    "AMENDMENT_FILENAME", "AUTHORIZATION_BASIS",
    "HarpV11AmendmentDraft", "HarpV11AmendmentPublicationReceipt",
    "build_harp_v11_execution_amendment_draft",
    "publish_harp_v11_amendment_draft_exclusive",
    "publish_harp_v11_execution_amendment",
)
