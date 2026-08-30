"""One-shot publication of the HARP v1 terminal execution amendment.

This module has one mutation edge: exclusive creation of the amendment file.
It does not edit configuration or workspace registration, create an output
artifact, claim the execution lease, or launch the experiment.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import hashlib
import json
import os
from pathlib import Path
import stat

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_bytes, canonical_hash
from ...runtime.artifact_io import read_json, sha256_file
from . import authorization
from .config import HarpStage90Config, load_config
from .input_surfaces import (
    CONTENT_INDEX,
    HarpConsumedCacheIndex,
    load_cache_index,
)
from .identity import claim_boundary_payload
from .physical_menu import HarpPhysicalInputReceipt, validate_physical_inputs
from .preparation import (
    CANONICAL_CACHE_CONTENT_HASH,
    CANONICAL_CACHE_ROW_ORDER_HASH,
    CANONICAL_MANIFEST_SHA256,
    CANONICAL_PARENT_LEDGER_SHA256,
    CASE_PARTITION,
    LABEL_FREE_BARRIER,
    LABEL_FREE_CONTENT_INDEX,
    PREPARATION_RECEIPT,
)


AUTHORIZATION_BASIS = authorization.AUTHORIZATION_BASIS
AUTHORIZATION_DATE = authorization.AUTHORIZATION_DATE
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
class _ValidatedPublisherInputs:
    config: HarpStage90Config
    cache: HarpConsumedCacheIndex
    preparation_receipt_hash: str
    physical_input_receipt_hash: str


@dataclass(frozen=True, slots=True)
class HarpAmendmentPublicationReceipt:
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
        if (
            not path.is_absolute()
            or not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != self.amendment_sha256
        ):
            raise ProtocolError("HARP amendment publication receipt path is unsafe.")
        for role in (
            "amendment_sha256",
            "amendment_hash",
            "input_binding_hash",
            "scientific_contract_hash",
            "workspace_registration_execution_contract_hash",
            "source_snapshot_manifest_sha256",
            "source_snapshot_tree_sha256",
            "preparation_receipt_hash",
            "physical_input_receipt_hash",
        ):
            value = getattr(self, role)
            if (
                type(value) is not str
                or len(value) != 64
                or value != value.lower()
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ProtocolError(f"HARP amendment publication {role} drifted.")
        object.__setattr__(self, "receipt_hash", canonical_hash(self._base_payload()))

    def _base_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_harp_stage90_amendment_publication_receipt_v1",
            "amendment_path": self.amendment_path.as_posix(),
            "amendment_sha256": self.amendment_sha256,
            "amendment_hash": self.amendment_hash,
            "input_binding_hash": self.input_binding_hash,
            "scientific_contract_hash": self.scientific_contract_hash,
            "workspace_registration_execution_contract_hash": (
                self.workspace_registration_execution_contract_hash
            ),
            "source_snapshot_manifest_sha256": self.source_snapshot_manifest_sha256,
            "source_snapshot_tree_sha256": self.source_snapshot_tree_sha256,
            "preparation_receipt_hash": self.preparation_receipt_hash,
            "physical_input_receipt_hash": self.physical_input_receipt_hash,
            "authorization_basis": AUTHORIZATION_BASIS,
            "authorization_date": AUTHORIZATION_DATE,
            "published_no_overwrite": True,
            "only_amendment_file_created": True,
            "configuration_or_registry_activated": False,
            "authorization_lease_claimed": False,
            "output_artifact_created": False,
            "experiment_launched": False,
            "label_values_opened": False,
            "fresh_evidence": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._base_payload(), "receipt_hash": self.receipt_hash}


def publish_harp_execution_amendment(
    config: HarpStage90Config,
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
) -> HarpAmendmentPublicationReceipt:
    """Validate exact inputs and issue one path-independent amendment once."""

    if type(config) is not HarpStage90Config or config.execution_authorized:
        raise ProtocolError("HARP amendment publication requires the planned config.")
    if authorization_basis != AUTHORIZATION_BASIS:
        raise ProtocolError("HARP amendment authorization basis is absent or drifted.")
    if authorization_date != AUTHORIZATION_DATE:
        raise ProtocolError("HARP amendment authorization date is absent or drifted.")
    if config.expected_execution_amendment_sha256 is not None:
        raise ProtocolError("HARP amendment has already been bound in configuration.")

    repository = _existing_directory(repository_root)
    _validate_registered_config_source(config, repository_root=repository)
    publication_path = _publication_path(
        amendment_path,
        repository_root=repository,
    )
    existing_lease = authorization.lease_path(repository)
    if os.path.lexists(existing_lease):
        raise ProtocolError("HARP authorization lease already exists; issuance is forbidden.")
    validated = _validate_inputs(
        config,
        expert_bank_root=expert_bank_root,
        generation_lock_root=generation_lock_root,
        prepared_cache_root=prepared_cache_root,
        development_manifest_path=development_manifest_path,
        evaluation_manifest_path=evaluation_manifest_path,
        parent_ledger_path=parent_ledger_path,
        amendment_path=publication_path,
    )
    # The authorized projection exists only in memory.  Publication does not
    # rewrite or activate the planned config.
    authorization_config = replace(
        validated.config,
        execution_authorized=True,
        claim_boundary=claim_boundary_payload(execution_authorized=True),
    )
    payload = authorization.canonical_execution_amendment_payload(
        authorization_config,
        repo_root=repository,
    )
    if (
        payload.get("authorization_basis") != authorization_basis
        or payload.get("authorization_date") != authorization_date
    ):
        raise ProtocolError("HARP canonical amendment authorization identity drifted.")
    authorization.validate_execution_amendment_payload(
        payload,
        authorization_config,
        repo_root=repository,
    )
    raw = canonical_bytes(payload) + b"\n"
    if os.path.lexists(authorization.lease_path(repository)):
        raise ProtocolError("HARP authorization lease appeared during publication.")

    # This is the publisher's only mutation.  A partial file is intentionally
    # retained after any late failure so the O_EXCL identity cannot be retried.
    _write_exclusive(publication_path, raw)
    observed = _read_unique_regular_file(publication_path)
    if observed != raw:
        raise ProtocolError("HARP amendment bytes changed after publication.")
    observed_payload = _decode_payload(observed)
    authorization.validate_execution_amendment_payload(
        observed_payload,
        authorization_config,
        repo_root=repository,
    )
    digest = hashlib.sha256(observed).hexdigest()
    binding = observed_payload.get("authorized_input_binding")
    source = observed_payload.get("source_snapshot_identity")
    if not isinstance(binding, Mapping) or not isinstance(source, Mapping):
        raise ProtocolError("HARP published amendment identities are malformed.")
    return HarpAmendmentPublicationReceipt(
        amendment_path=publication_path,
        amendment_sha256=digest,
        amendment_hash=str(observed_payload["amendment_hash"]),
        input_binding_hash=str(binding["input_binding_hash"]),
        scientific_contract_hash=str(observed_payload["scientific_contract_hash"]),
        workspace_registration_execution_contract_hash=str(
            observed_payload["workspace_registration_execution_contract_hash"]
        ),
        source_snapshot_manifest_sha256=str(
            source["source_snapshot_manifest_sha256"]
        ),
        source_snapshot_tree_sha256=str(source["source_snapshot_tree_sha256"]),
        preparation_receipt_hash=validated.preparation_receipt_hash,
        physical_input_receipt_hash=validated.physical_input_receipt_hash,
    )


def _validate_inputs(
    config: HarpStage90Config,
    *,
    expert_bank_root: str | Path,
    generation_lock_root: str | Path,
    prepared_cache_root: str | Path,
    development_manifest_path: str | Path,
    evaluation_manifest_path: str | Path,
    parent_ledger_path: str | Path,
    amendment_path: Path,
) -> _ValidatedPublisherInputs:
    bank = _existing_directory(expert_bank_root)
    generation = _existing_directory(generation_lock_root)
    cache_root = _existing_directory(prepared_cache_root)
    development = _existing_regular_file(development_manifest_path)
    evaluation = _existing_regular_file(evaluation_manifest_path)
    parent = _existing_regular_file(parent_ledger_path)
    exact_paths = {bank, generation, cache_root, development, evaluation, parent}
    if len(exact_paths) != 6 or amendment_path in exact_paths:
        raise ProtocolError("HARP amendment input paths overlap.")

    parent_sha256 = sha256_file(parent)
    if parent_sha256 != CANONICAL_PARENT_LEDGER_SHA256:
        raise ProtocolError("HARP amendment parent ledger drifted.")
    read_json(parent)
    development_sha256 = sha256_file(development)
    evaluation_sha256 = sha256_file(evaluation)
    content = read_json(cache_root / CONTENT_INDEX)
    content_base = {
        key: value for key, value in content.items() if key != "content_index_hash"
    }
    cache_content_sha256 = content.get("content_index_hash")
    if (
        not isinstance(cache_content_sha256, str)
        or cache_content_sha256 != canonical_hash(content_base)
    ):
        raise ProtocolError("HARP amendment cache content identity drifted.")

    computed = {
        "test_cache_content_sha256": cache_content_sha256,
        "development_manifest_sha256": development_sha256,
        "evaluation_manifest_sha256": evaluation_sha256,
        "parent_ledger_sha256": parent_sha256,
    }
    for role, value in computed.items():
        configured = config.expected_hashes.get(role)
        if configured is not None and configured != value:
            raise ProtocolError(f"HARP planned config {role} drifted from prepared input.")

    bound_config = replace(
        config,
        input_locations={
            **dict(config.input_locations),
            "expert_bank_root": bank.as_posix(),
            "generation_lock_root": generation.as_posix(),
            "test_cache_root": cache_root.as_posix(),
            "development_manifest_path": development.as_posix(),
            "evaluation_manifest_path": evaluation.as_posix(),
            "parent_ledger_path": parent.as_posix(),
            "execution_amendment_path": amendment_path.as_posix(),
        },
        expected_hashes={
            **dict(config.expected_hashes),
            **computed,
            "execution_amendment_sha256": None,
        },
    )
    cache = load_cache_index(bound_config)
    preparation_receipt_hash = _validate_preparation_receipt(
        cache,
        development_sha256=development_sha256,
        evaluation_sha256=evaluation_sha256,
        parent_ledger_sha256=parent_sha256,
    )
    # Hashing authenticates the externally stored capability bytes without
    # parsing or materializing either label column.  Role purity and exact row
    # coverage were sealed by the content-bound preparation receipt; the runner
    # remains the only semantic label opener after its phase barriers.
    physical = validate_physical_inputs(bound_config, cache)
    if type(physical) is not HarpPhysicalInputReceipt:
        raise ProtocolError("HARP amendment physical input validation is untyped.")
    return _ValidatedPublisherInputs(
        config=bound_config,
        cache=cache,
        preparation_receipt_hash=preparation_receipt_hash,
        physical_input_receipt_hash=physical.receipt_hash,
    )


def _validate_preparation_receipt(
    cache: HarpConsumedCacheIndex,
    *,
    development_sha256: str,
    evaluation_sha256: str,
    parent_ledger_sha256: str,
) -> str:
    receipt_path = cache.root / PREPARATION_RECEIPT
    if cache.member_sha256.get(PREPARATION_RECEIPT.as_posix()) != sha256_file(
        receipt_path
    ):
        raise ProtocolError("HARP preparation receipt is not content-bound.")
    receipt = read_json(receipt_path)
    receipt_base = {key: value for key, value in receipt.items() if key != "receipt_hash"}
    partition = read_json(cache.root / CASE_PARTITION)
    barrier = read_json(cache.root / LABEL_FREE_BARRIER)
    barrier_base = {key: value for key, value in barrier.items() if key != "barrier_hash"}
    label_free = read_json(cache.root / LABEL_FREE_CONTENT_INDEX)
    label_free_base = {
        key: value for key, value in label_free.items() if key != "content_index_hash"
    }
    pre_members = dict(cache.member_sha256)
    pre_members.pop(PREPARATION_RECEIPT.as_posix(), None)
    pre_content_base = {
        "schema_version": "midogpp_harp_consumed_test_content_index_v1",
        "members": dict(sorted(pre_members.items())),
    }
    label_free_expected = set(pre_members) - {LABEL_FREE_CONTENT_INDEX.as_posix()}

    fixed_values = {
        "schema_version": "midogpp_harp_consumed_test_preparation_receipt_v1",
        "experiment_id": config_experiment_id(),
        "status": "PREPARED_INPUTS_NO_EXECUTION_AUTHORITY",
        "canonical_cache_content_hash": CANONICAL_CACHE_CONTENT_HASH,
        "canonical_cache_row_order_hash": CANONICAL_CACHE_ROW_ORDER_HASH,
        "canonical_manifest_sha256": CANONICAL_MANIFEST_SHA256,
        "parent_ledger_sha256": parent_ledger_sha256,
        "prepared_cache_index_hash": cache.cache_hash,
        "prepared_row_count": len(cache.rows),
        "development_manifest_sha256": development_sha256,
        "evaluation_manifest_sha256": evaluation_sha256,
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
        or receipt.get("receipt_hash") != canonical_hash(receipt_base)
        or any(receipt.get(key) != value for key, value in fixed_values.items())
        or receipt.get("partition_hash") != canonical_hash(partition)
        or receipt.get("label_free_barrier_sha256")
        != sha256_file(cache.root / LABEL_FREE_BARRIER)
        or receipt.get("label_free_content_index_sha256")
        != sha256_file(cache.root / LABEL_FREE_CONTENT_INDEX)
        or receipt.get("pre_manifest_cache_content_sha256")
        != canonical_hash(pre_content_base)
        or barrier.get("barrier_hash") != canonical_hash(barrier_base)
        or barrier.get("partition_hash") != receipt.get("partition_hash")
        or barrier.get("canonical_scoring_manifest_opened") is not False
        or label_free.get("content_index_hash") != canonical_hash(label_free_base)
        or set(label_free.get("members", {})) != label_free_expected
    ):
        raise ProtocolError("HARP preparation receipt/content validation failed.")
    return str(receipt["receipt_hash"])


def config_experiment_id() -> str:
    # Kept local to make the receipt validator independent of config path data.
    from .identity import EXPERIMENT_ID

    return EXPERIMENT_ID


def _publication_path(value: str | Path, *, repository_root: Path) -> Path:
    raw = Path(value).expanduser()
    path = raw if raw.is_absolute() else Path.cwd() / raw
    path = Path(os.path.abspath(path))
    expected = (
        repository_root
        / "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts"
        / "harp_router_v1"
        / AMENDMENT_FILENAME
    )
    if path != expected:
        raise ProtocolError("HARP amendment path is not the registered contract member.")
    if os.path.lexists(path):
        raise ProtocolError("HARP amendment path already exists; overwrite is forbidden.")
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ProtocolError("HARP amendment publication parent is absent or unsafe.")
    return path


def _validate_registered_config_source(
    config: HarpStage90Config,
    *,
    repository_root: Path,
) -> None:
    expected = (
        repository_root
        / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
        / "uniform_b_v2_consumed_test_fixed_bank_harp_router_v1.yaml"
    )
    observed = _existing_regular_file(config.source_path)
    if observed != expected:
        raise ProtocolError("HARP publisher config is not the registered HARP v1 member.")
    reconstructed = load_config(observed)
    if config != reconstructed:
        raise ProtocolError(
            "HARP publisher config object drifted from the registered YAML bytes."
        )


def _existing_directory(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    if raw.is_symlink():
        raise ProtocolError("HARP amendment input directory is a symlink.")
    try:
        path = raw.resolve(strict=True)
    except OSError as exc:
        raise ProtocolError("HARP amendment input directory is absent.") from exc
    if not path.is_dir() or path.is_symlink():
        raise ProtocolError("HARP amendment input directory is unsafe.")
    return path


def _existing_regular_file(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    if raw.is_symlink():
        raise ProtocolError("HARP amendment input file is a symlink.")
    try:
        path = raw.resolve(strict=True)
    except OSError as exc:
        raise ProtocolError("HARP amendment input file is absent.") from exc
    try:
        mode = path.stat().st_mode
    except OSError as exc:
        raise ProtocolError("HARP amendment input file is unreadable.") from exc
    if not stat.S_ISREG(mode) or path.is_symlink():
        raise ProtocolError("HARP amendment input file is unsafe.")
    return path


def _write_exclusive(path: Path, raw: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
        try:
            view = memoryview(raw)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short exclusive write")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(path.parent)
    except FileExistsError as exc:
        raise ProtocolError("HARP amendment overwrite is forbidden.") from exc
    except OSError as exc:
        raise ProtocolError(
            "HARP amendment publication failed closed; any partial path is terminal."
        ) from exc


def _read_unique_regular_file(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ProtocolError("HARP amendment published member is not unique.")
            with os.fdopen(os.dup(descriptor), "rb") as handle:
                raw = handle.read(1024 * 1024 + 1)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise ProtocolError("HARP amendment published member is unreadable.") from exc
    if len(raw) > 1024 * 1024:
        raise ProtocolError("HARP amendment published member is oversized.")
    return raw


def _decode_payload(raw: bytes) -> dict[str, object]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("HARP amendment published bytes are malformed.") from exc
    if not isinstance(value, dict):
        raise ProtocolError("HARP amendment published payload is not an object.")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = (
    "AMENDMENT_FILENAME",
    "AUTHORIZATION_BASIS",
    "AUTHORIZATION_DATE",
    "HarpAmendmentPublicationReceipt",
    "publish_harp_execution_amendment",
)
