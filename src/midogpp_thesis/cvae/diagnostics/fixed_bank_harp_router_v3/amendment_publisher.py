"""Mutation-free one-shot publication after separate HARP v3 activation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import hashlib
import json
import os
from pathlib import Path

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_bytes, canonical_hash
from ...runtime.artifact_io import read_json, sha256_file
from .input_lineage import validate_physical_inputs
from .preparation import (
    CANONICAL_CACHE_CONTENT_HASH,
    CANONICAL_CACHE_ROW_ORDER_HASH,
    CANONICAL_MANIFEST_SHA256,
)
from . import authorization
from .config import HarpStage90V3Config, load_config
from .identity import EXPERIMENT_ID
from .input_surfaces import (
    CONTENT_INDEX,
    V3_CACHE_IDENTITY,
    HarpConsumedCacheIndex,
    load_cache_index,
)
from .preparation import (
    CASE_PARTITION,
    LABEL_FREE_BARRIER,
    LABEL_FREE_CONTENT_INDEX,
    PREPARATION_RECEIPT,
)


AUTHORIZATION_BASIS = authorization.AUTHORIZATION_BASIS
AMENDMENT_FILENAME = authorization.EXECUTION_AMENDMENT_FILENAME

_PREPARATION_RECEIPT_KEYS = {
    "schema_version",
    "experiment_id",
    "status",
    "canonical_cache_content_hash",
    "canonical_cache_row_order_hash",
    "canonical_manifest_sha256",
    "parent_ledger_sha256",
    "partition_hash",
    "label_free_barrier_sha256",
    "label_free_content_index_sha256",
    "pre_manifest_cache_content_sha256",
    "prepared_cache_index_hash",
    "prepared_row_count",
    "development_manifest_sha256",
    "evaluation_manifest_sha256",
    "development_and_evaluation_cases_disjoint",
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
class HarpV3AmendmentPublicationReceipt:
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
            raise ProtocolError("HARP v3 amendment publication receipt path is unsafe.")
        object.__setattr__(self, "receipt_hash", canonical_hash(self._base_payload()))

    def _base_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_harp_stage90_amendment_publication_receipt_v3",
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


def publish_harp_v3_execution_amendment(
    config: HarpStage90V3Config,
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
) -> HarpV3AmendmentPublicationReceipt:
    if type(config) is not HarpStage90V3Config or not config.execution_authorized:
        raise ProtocolError("HARP v3 amendment publication requires an explicitly activated config.")
    authorization.validate_activation_metadata(authorization_basis, authorization_date)
    if config.expected_execution_amendment_sha256 is not None:
        raise ProtocolError("HARP v3 amendment is already bound in configuration.")
    repository = _existing_directory(repository_root)
    if config.source_path != (repository / authorization.WORKSPACE_CONFIG_RELATIVE_PATH).resolve():
        raise ProtocolError("HARP v3 publisher config is not the registered source.")
    if load_config(config.source_path) != config:
        raise ProtocolError("HARP v3 publisher config changed after load.")
    publication_path = Path(amendment_path).resolve()
    expected_path = (repository / authorization.WORKSPACE_AMENDMENT_RELATIVE_PATH).resolve()
    if publication_path != expected_path or not publication_path.parent.is_dir() or publication_path.parent.is_symlink():
        raise ProtocolError("HARP v3 amendment publication path drifted from its catalog identity.")
    if os.path.lexists(publication_path) or os.path.lexists(authorization.lease_path(repository)):
        raise ProtocolError("HARP v3 amendment or authorization lease already exists.")

    bank = _existing_directory(expert_bank_root)
    generation = _existing_directory(generation_lock_root)
    cache_root = _existing_directory(prepared_cache_root)
    development = _existing_file(development_manifest_path)
    evaluation = _existing_file(evaluation_manifest_path)
    parent = _existing_file(parent_ledger_path)
    paths = {bank, generation, cache_root, development, evaluation, parent, publication_path}
    if len(paths) != 7:
        raise ProtocolError("HARP v3 amendment input paths overlap.")
    content = read_json(cache_root / CONTENT_INDEX)
    content_base = {key: value for key, value in content.items() if key != "content_index_hash"}
    cache_content = content.get("content_index_hash")
    if type(cache_content) is not str or cache_content != canonical_hash(content_base):
        raise ProtocolError("HARP v3 prepared cache content identity drifted.")
    computed = {
        "test_cache_content_sha256": cache_content,
        "development_manifest_sha256": sha256_file(development),
        "evaluation_manifest_sha256": sha256_file(evaluation),
        "parent_ledger_sha256": sha256_file(parent),
    }
    read_json(parent)
    if any(
        config.expected_hashes.get(role) != value for role, value in computed.items()
    ):
        raise ProtocolError("HARP v3 activated config and prepared-input receipt disagree.")
    bound = replace(
        config,
        input_locations={
            **dict(config.input_locations), "expert_bank_root": bank.as_posix(),
            "generation_lock_root": generation.as_posix(), "test_cache_root": cache_root.as_posix(),
            "development_manifest_path": development.as_posix(),
            "evaluation_manifest_path": evaluation.as_posix(), "parent_ledger_path": parent.as_posix(),
            "execution_amendment_path": publication_path.as_posix(),
        },
    )
    cache = load_cache_index(bound)
    preparation_hash = _validate_preparation_receipt(cache, computed)
    physical = validate_physical_inputs(bound, cache)
    payload = authorization.canonical_execution_amendment_payload(
        bound,
        authorization_basis=authorization_basis,
        authorization_date=authorization_date,
        repo_root=repository,
    )
    authorization.validate_execution_amendment_payload(payload, bound, repo_root=repository)
    raw = canonical_bytes(payload) + b"\n"
    _write_exclusive(publication_path, raw)
    if publication_path.read_bytes() != raw:
        raise ProtocolError("HARP v3 amendment bytes changed after publication.")
    observed = json.loads(raw.decode("utf-8"))
    validated = authorization.validate_execution_amendment_payload(
        observed, bound, repo_root=repository
    )
    source = observed["source_snapshot_identity"]
    binding = observed["authorized_input_binding"]
    return HarpV3AmendmentPublicationReceipt(
        amendment_path=publication_path,
        amendment_sha256=hashlib.sha256(raw).hexdigest(),
        amendment_hash=validated.amendment_hash,
        input_binding_hash=str(binding["input_binding_hash"]),
        scientific_contract_hash=validated.scientific_contract_hash,
        workspace_registration_execution_contract_hash=validated.workspace_registration_execution_contract_hash,
        source_snapshot_manifest_sha256=str(source["source_snapshot_manifest_sha256"]),
        source_snapshot_tree_sha256=str(source["source_snapshot_tree_sha256"]),
        preparation_receipt_hash=preparation_hash,
        physical_input_receipt_hash=str(physical.receipt_hash),
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
        raise ProtocolError("HARP v3 preparation receipt is not cache-bound.")
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
        raise ProtocolError("HARP v3 label-free content index is not cache-bound.")
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
        raise ProtocolError("HARP v3 final cache lacks its preparation receipt.")
    pre_manifest_base = {
        "schema_version": V3_CACHE_IDENTITY.content_schema,
        "members": dict(sorted(pre_manifest_members.items())),
    }
    label_free_members = label_free.get("members")
    expected_label_free_members = {
        key: value
        for key, value in pre_manifest_members.items()
        if key != label_free_member
    }
    fixed_values = {
        "schema_version": "midogpp_harp_consumed_test_preparation_receipt_v3",
        "experiment_id": EXPERIMENT_ID,
        "status": "PREPARED_INPUTS_NO_EXECUTION_AUTHORITY",
        "canonical_cache_content_hash": CANONICAL_CACHE_CONTENT_HASH,
        "canonical_cache_row_order_hash": CANONICAL_CACHE_ROW_ORDER_HASH,
        "canonical_manifest_sha256": CANONICAL_MANIFEST_SHA256,
        "parent_ledger_sha256": computed["parent_ledger_sha256"],
        "prepared_cache_index_hash": cache.cache_hash,
        "prepared_row_count": len(cache.rows),
        "development_manifest_sha256": computed["development_manifest_sha256"],
        "evaluation_manifest_sha256": computed["evaluation_manifest_sha256"],
        "development_and_evaluation_cases_disjoint": True,
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
        raise ProtocolError("HARP v3 preparation receipt drifted.")
    return str(receipt["receipt_hash"])


def _existing_directory(value: str | Path) -> Path:
    path = Path(value).resolve()
    if not path.is_dir() or path.is_symlink():
        raise ProtocolError("HARP v3 required directory is absent or unsafe.")
    return path


def _existing_file(value: str | Path) -> Path:
    path = Path(value).resolve()
    if not path.is_file() or path.is_symlink():
        raise ProtocolError("HARP v3 required file is absent or unsafe.")
    return path


def _write_exclusive(path: Path, raw: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise


__all__ = (
    "AMENDMENT_FILENAME", "AUTHORIZATION_BASIS",
    "HarpV3AmendmentPublicationReceipt", "publish_harp_v3_execution_amendment",
)
