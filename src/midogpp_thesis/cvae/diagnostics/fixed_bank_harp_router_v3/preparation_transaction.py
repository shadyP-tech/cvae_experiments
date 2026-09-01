"""Crash-recoverable publication of prepared HARP v3 workstation inputs.

All expensive conversion and label partitioning happens below an owned staging
root.  A canonical journal authenticating the complete staged inventory is
fsynced before the first catalog destination appears.  Directory renames then
commit cache, development capability, and evaluation capability in that order.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import shutil
from typing import Protocol

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_bytes, canonical_hash
from ...runtime.artifact_io import read_json, sha256_file
from .identity import EXPERIMENT_ID, PUBLICATION_STATUS, TERMINAL_DECISION
from .input_surfaces import CONTENT_INDEX
from .preparation import PREPARATION_RECEIPT, HarpV3PreparedInputs
from .workspace_paths import HarpV3WorkspacePaths


TRANSACTION_SCHEMA = "midogpp_harp_v3_workstation_preparation_transaction_v1"
FaultInjector = Callable[[str], None]


class PreparationPlanLike(Protocol):
    paths: HarpV3WorkspacePaths
    preparation_plan_hash: str


@dataclass(frozen=True, slots=True)
class PreparationInventory:
    cache_members: Mapping[str, str]
    development_members: Mapping[str, str]
    evaluation_members: Mapping[str, str]

    def to_payload(self) -> dict[str, object]:
        return {
            "cache": dict(sorted(self.cache_members.items())),
            "development": dict(sorted(self.development_members.items())),
            "evaluation": dict(sorted(self.evaluation_members.items())),
        }


@dataclass(frozen=True, slots=True)
class PreparationJournal:
    repository_root: Path
    preparation_plan_hash: str
    paths: HarpV3WorkspacePaths
    prepared_payload: Mapping[str, object]
    inventory: PreparationInventory
    journal_hash: str

    def payload_without_hash(self) -> dict[str, object]:
        root = self.repository_root
        return {
            "schema_version": TRANSACTION_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "preparation_plan_hash": self.preparation_plan_hash,
            "paths": {
                "staging_root": _relative(root, self.paths.staging_root),
                "prepared_cache_root": _relative(
                    root, self.paths.prepared_cache_root
                ),
                "development_root": _relative(
                    root, self.paths.development_manifest_path.parent
                ),
                "evaluation_root": _relative(
                    root, self.paths.evaluation_manifest_path.parent
                ),
                "transaction": _relative(root, self.paths.transaction_path),
            },
            "prepared_payload": dict(self.prepared_payload),
            "inventory": self.inventory.to_payload(),
            "commit_order": ["cache", "development", "evaluation"],
            "journal_durable_before_publication": True,
            "progress_inferred_from_exact_inventory": True,
            "execution_amendment_created": False,
            "authorization_lease_claimed": False,
            "output_created": False,
        }

    def to_bytes(self) -> bytes:
        return canonical_bytes(
            {**self.payload_without_hash(), "journal_hash": self.journal_hash}
        ) + b"\n"


def build_preparation_journal(
    plan: PreparationPlanLike,
    prepared: HarpV3PreparedInputs,
) -> PreparationJournal:
    paths = plan.paths
    stage_cache, stage_development, stage_evaluation = staging_destinations(paths)
    if (
        prepared.cache_root != stage_cache
        or prepared.development_manifest_path != stage_development / "manifest.csv"
        or prepared.evaluation_manifest_path != stage_evaluation / "manifest.csv"
    ):
        raise ProtocolError("HARP v3 staged preparation paths drifted.")
    _validate_ready_receipt(prepared)
    inventory = PreparationInventory(
        cache_members=inventory_tree(stage_cache),
        development_members=inventory_tree(stage_development),
        evaluation_members=inventory_tree(stage_evaluation),
    )
    prepared_payload = _journal_prepared_payload(prepared)
    provisional = PreparationJournal(
        repository_root=paths.repository_root,
        preparation_plan_hash=plan.preparation_plan_hash,
        paths=paths,
        prepared_payload=prepared_payload,
        inventory=inventory,
        journal_hash="",
    )
    return PreparationJournal(
        repository_root=provisional.repository_root,
        preparation_plan_hash=provisional.preparation_plan_hash,
        paths=provisional.paths,
        prepared_payload=provisional.prepared_payload,
        inventory=provisional.inventory,
        journal_hash=canonical_hash(provisional.payload_without_hash()),
    )


def commit_prepared_inputs(
    journal: PreparationJournal,
    *,
    fault_injector: FaultInjector | None = None,
    recovered: bool = False,
) -> HarpV3PreparedInputs:
    """Publish one authenticated journal, rolling back ordinary failures."""

    paths = journal.paths
    with preparation_lock(paths):
        _require_no_authority_surface(paths)
        try:
            _install_or_validate_journal(journal)
            _inject(fault_injector, "journal_durable")
            result = _resume(journal, fault_injector=fault_injector)
            _inject(fault_injector, "all_inputs_committed")
        except Exception as exc:
            cleanup_errors = _rollback_and_cleanup(journal)
            if cleanup_errors:
                raise ProtocolError(
                    "HARP v3 preparation failed and exact cleanup was incomplete; "
                    "the durable journal was retained for recovery."
                ) from exc
            raise ProtocolError(
                "HARP v3 preparation failed closed and removed its exact staged "
                "and partially committed outputs."
            ) from exc
        _cleanup_completed_state(journal)
        return result


def recover_prepared_inputs(
    paths: HarpV3WorkspacePaths,
    *,
    expected_plan_hash: str,
    fault_injector: FaultInjector | None = None,
) -> HarpV3PreparedInputs:
    """Resume an interrupted exact-prefix commit from its durable journal."""

    with preparation_lock(paths):
        _require_no_authority_surface(paths)
        journal = load_preparation_journal(paths)
        if journal.preparation_plan_hash != expected_plan_hash:
            raise ProtocolError("HARP v3 preparation recovery plan drifted.")
        try:
            result = _resume(journal, fault_injector=fault_injector)
            _inject(fault_injector, "all_inputs_committed")
        except Exception as exc:
            cleanup_errors = _rollback_and_cleanup(journal)
            if cleanup_errors:
                raise ProtocolError(
                    "HARP v3 preparation recovery failed and exact cleanup was "
                    "incomplete; the durable journal was retained."
                ) from exc
            raise ProtocolError(
                "HARP v3 preparation recovery failed closed and cleaned its exact "
                "partial publication."
            ) from exc
        _cleanup_completed_state(journal)
        return result


def load_preparation_journal(paths: HarpV3WorkspacePaths) -> PreparationJournal:
    path = paths.transaction_path
    if not path.is_file() or path.is_symlink():
        raise ProtocolError("HARP v3 preparation journal is absent or unsafe.")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("HARP v3 preparation journal is unreadable.") from exc
    if not isinstance(raw, dict):
        raise ProtocolError("HARP v3 preparation journal is malformed.")
    expected_keys = {
        "schema_version",
        "experiment_id",
        "preparation_plan_hash",
        "paths",
        "prepared_payload",
        "inventory",
        "commit_order",
        "journal_durable_before_publication",
        "progress_inferred_from_exact_inventory",
        "execution_amendment_created",
        "authorization_lease_claimed",
        "output_created",
        "journal_hash",
    }
    base = {key: value for key, value in raw.items() if key != "journal_hash"}
    expected_paths = {
        "staging_root": _relative(paths.repository_root, paths.staging_root),
        "prepared_cache_root": _relative(
            paths.repository_root, paths.prepared_cache_root
        ),
        "development_root": _relative(
            paths.repository_root, paths.development_manifest_path.parent
        ),
        "evaluation_root": _relative(
            paths.repository_root, paths.evaluation_manifest_path.parent
        ),
        "transaction": _relative(paths.repository_root, paths.transaction_path),
    }
    if (
        set(raw) != expected_keys
        or raw.get("schema_version") != TRANSACTION_SCHEMA
        or raw.get("experiment_id") != EXPERIMENT_ID
        or raw.get("journal_hash") != canonical_hash(base)
        or raw.get("paths") != expected_paths
        or raw.get("commit_order") != ["cache", "development", "evaluation"]
        or raw.get("journal_durable_before_publication") is not True
        or raw.get("progress_inferred_from_exact_inventory") is not True
        or raw.get("execution_amendment_created") is not False
        or raw.get("authorization_lease_claimed") is not False
        or raw.get("output_created") is not False
    ):
        raise ProtocolError("HARP v3 preparation journal authentication failed.")
    prepared = _mapping(raw.get("prepared_payload"), label="prepared payload")
    inventory = _mapping(raw.get("inventory"), label="inventory")
    cache = _digest_mapping(inventory.get("cache"), label="cache inventory")
    development = _digest_mapping(
        inventory.get("development"), label="development inventory"
    )
    evaluation = _digest_mapping(
        inventory.get("evaluation"), label="evaluation inventory"
    )
    if set(inventory) != {"cache", "development", "evaluation"}:
        raise ProtocolError("HARP v3 preparation journal inventory drifted.")
    journal = PreparationJournal(
        repository_root=paths.repository_root,
        preparation_plan_hash=str(raw.get("preparation_plan_hash")),
        paths=paths,
        prepared_payload=dict(prepared),
        inventory=PreparationInventory(cache, development, evaluation),
        journal_hash=str(raw["journal_hash"]),
    )
    if journal.to_bytes() != path.read_bytes():
        raise ProtocolError("HARP v3 preparation journal bytes are not canonical.")
    _prepared_from_payload(paths, journal.prepared_payload)
    return journal


def inspect_preparation_recovery(
    paths: HarpV3WorkspacePaths,
) -> dict[str, object] | None:
    if not os.path.lexists(paths.transaction_path):
        return None
    journal = load_preparation_journal(paths)
    states = _states(journal)
    complete = all(states[name] == "final" for name in states)
    return {
        "schema_version": "midogpp_harp_v3_workstation_preparation_recovery_v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "PREPARED" if complete else "RECOVERY_REQUIRED",
        "preparation_plan_hash": journal.preparation_plan_hash,
        "journal_hash": journal.journal_hash,
        "observed_states": states,
        "confirmation_required": "PREPARE_HARP_V3_CONSUMED_TEST_INPUTS",
        "filesystem_mutations": 0,
        "execution_amendment_created": False,
        "authorization_lease_claimed": False,
        "output_created": False,
    }


def staging_destinations(paths: HarpV3WorkspacePaths) -> tuple[Path, Path, Path]:
    return (
        paths.staging_root / "cache",
        paths.staging_root / "development",
        paths.staging_root / "evaluation",
    )


def inventory_tree(root: Path) -> dict[str, str]:
    """Hash one closed-world tree with one recursive inventory traversal."""

    if not root.is_dir() or root.is_symlink():
        raise ProtocolError("HARP v3 preparation inventory root is unsafe.")
    entries = tuple(root.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise ProtocolError("HARP v3 preparation inventory contains a symlink.")
    files = tuple(path for path in entries if path.is_file())
    if any(not path.is_relative_to(root) for path in files):  # pragma: no cover
        raise ProtocolError("HARP v3 preparation inventory escaped its root.")
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(files)
    }


def preparation_lock(paths: HarpV3WorkspacePaths):
    return _preparation_lock(paths)


@contextmanager
def _preparation_lock(paths: HarpV3WorkspacePaths):
    descriptor = os.open(
        paths.lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600
    )
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProtocolError("HARP v3 preparation is already in progress.") from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _resume(
    journal: PreparationJournal,
    *,
    fault_injector: FaultInjector | None,
) -> HarpV3PreparedInputs:
    states = _states(journal)
    names = ("cache", "development", "evaluation")
    observed = tuple(states[name] for name in names)
    # Only an exact commit prefix is recoverable.  In particular, evaluation
    # can never appear before development, and labels can never publish before
    # the complete label-free cache directory.
    valid = {
        ("staged", "staged", "staged"),
        ("final", "staged", "staged"),
        ("final", "final", "staged"),
        ("final", "final", "final"),
    }
    if observed not in valid:
        raise ProtocolError("HARP v3 preparation commit prefix is unsafe.")
    for name in names:
        if states[name] == "staged":
            stage, final, _inventory = _locations(journal, name)
            final.parent.mkdir(parents=True, exist_ok=True)
            _fsync_directories((final.parent,))
            os.replace(stage, final)
            _fsync_directories((stage.parent, final.parent))
            # The source inventory was authenticated immediately above.  A
            # same-filesystem directory rename preserves those inodes and
            # bytes, so rehashing the multi-GB cache after every one of the
            # three commit points would add no integrity information.
        _inject(fault_injector, f"{name}_committed")
    return _validate_complete(journal)


def _states(journal: PreparationJournal) -> dict[str, str]:
    states: dict[str, str] = {}
    for name in ("cache", "development", "evaluation"):
        stage, final, expected = _locations(journal, name)
        stage_exists = os.path.lexists(stage)
        final_exists = os.path.lexists(final)
        if stage_exists and final_exists:
            states[name] = "duplicate"
        elif stage_exists:
            states[name] = (
                "staged" if inventory_tree(stage) == dict(expected) else "drifted"
            )
        elif final_exists:
            states[name] = (
                "final" if inventory_tree(final) == dict(expected) else "drifted"
            )
        else:
            states[name] = "missing"
    return states


def _locations(
    journal: PreparationJournal, name: str
) -> tuple[Path, Path, Mapping[str, str]]:
    stage_cache, stage_dev, stage_eval = staging_destinations(journal.paths)
    if name == "cache":
        return stage_cache, journal.paths.prepared_cache_root, journal.inventory.cache_members
    if name == "development":
        return (
            stage_dev,
            journal.paths.development_manifest_path.parent,
            journal.inventory.development_members,
        )
    if name == "evaluation":
        return (
            stage_eval,
            journal.paths.evaluation_manifest_path.parent,
            journal.inventory.evaluation_members,
        )
    raise ProtocolError("HARP v3 preparation transaction role is unknown.")


def _validate_complete(journal: PreparationJournal) -> HarpV3PreparedInputs:
    for name in ("cache", "development", "evaluation"):
        stage, final, _expected = _locations(journal, name)
        if os.path.lexists(stage) or not final.is_dir() or final.is_symlink():
            raise ProtocolError("HARP v3 preparation publication is incomplete.")
    result = _prepared_from_payload(journal.paths, journal.prepared_payload)
    if (
        sha256_file(result.development_manifest_path)
        != result.development_manifest_sha256
        or sha256_file(result.evaluation_manifest_path)
        != result.evaluation_manifest_sha256
    ):
        raise ProtocolError("HARP v3 prepared manifest bytes drifted.")
    return result


def _install_or_validate_journal(journal: PreparationJournal) -> None:
    path = journal.paths.transaction_path
    expected = journal.to_bytes()
    if os.path.lexists(path):
        observed = load_preparation_journal(journal.paths)
        if observed.to_bytes() != expected:
            raise ProtocolError("HARP v3 existing preparation journal differs.")
        return
    _write_exclusive(path, expected)
    _fsync_directories((path.parent,))


def _rollback_and_cleanup(
    journal: PreparationJournal,
) -> tuple[BaseException, ...]:
    errors: list[BaseException] = []
    for name in ("evaluation", "development", "cache"):
        try:
            stage, final, expected = _locations(journal, name)
            if os.path.lexists(final):
                if inventory_tree(final) != dict(expected) or os.path.lexists(stage):
                    raise ProtocolError(
                        f"HARP v3 cannot clean drifted {name} publication."
                    )
                stage.parent.mkdir(parents=True, exist_ok=True)
                os.replace(final, stage)
                _fsync_directories((final.parent, stage.parent))
            if os.path.lexists(stage):
                if inventory_tree(stage) != dict(expected):
                    raise ProtocolError(
                        f"HARP v3 cannot clean drifted {name} staging."
                    )
                shutil.rmtree(stage)
        except BaseException as exc:
            errors.append(exc)
    if not errors:
        try:
            _remove_empty_staging_root(journal.paths.staging_root)
            _unlink_exact_journal(journal)
            _fsync_directories((journal.paths.transaction_path.parent,))
        except BaseException as exc:
            errors.append(exc)
    return tuple(errors)


def _cleanup_completed_state(journal: PreparationJournal) -> None:
    for name in ("cache", "development", "evaluation"):
        stage, final, _expected = _locations(journal, name)
        if os.path.lexists(stage) or not final.is_dir() or final.is_symlink():
            raise ProtocolError("HARP v3 cannot clean an incomplete preparation state.")
    _remove_empty_staging_root(journal.paths.staging_root)
    _unlink_exact_journal(journal)
    _fsync_directories((journal.paths.transaction_path.parent,))


def _remove_empty_staging_root(root: Path) -> None:
    if not os.path.lexists(root):
        return
    if not root.is_dir() or root.is_symlink():
        raise ProtocolError("HARP v3 preparation staging root is unsafe.")
    leftovers = tuple(root.iterdir())
    if leftovers:
        raise ProtocolError("HARP v3 preparation staging root is not empty.")
    root.rmdir()
    _fsync_directories((root.parent,))


def _unlink_exact_journal(journal: PreparationJournal) -> None:
    path = journal.paths.transaction_path
    if not os.path.lexists(path):
        return
    if (
        not path.is_file()
        or path.is_symlink()
        or path.read_bytes() != journal.to_bytes()
    ):
        raise ProtocolError("HARP v3 refuses to remove a drifted preparation journal.")
    path.unlink()


def _require_no_authority_surface(paths: HarpV3WorkspacePaths) -> None:
    if os.path.lexists(paths.amendment_path):
        raise ProtocolError("HARP v3 preparation cannot coexist with an amendment.")
    if os.path.lexists(paths.output_root):
        raise ProtocolError("HARP v3 preparation cannot overwrite an output.")
    # Import lazily to avoid turning preparation path discovery into authority
    # module initialization.  Computing this fixed path does not create it.
    from .authorization import lease_path

    if os.path.lexists(lease_path(paths.repository_root)):
        raise ProtocolError("HARP v3 preparation cannot coexist with a lease.")


def _journal_prepared_payload(prepared: HarpV3PreparedInputs) -> dict[str, object]:
    return {
        "schema_version": "midogpp_harp_v3_staged_prepared_inputs_v1",
        "test_cache_content_sha256": prepared.cache_content_sha256,
        "development_manifest_sha256": prepared.development_manifest_sha256,
        "evaluation_manifest_sha256": prepared.evaluation_manifest_sha256,
        "parent_ledger_sha256": prepared.parent_ledger_sha256,
        "partition_hash": prepared.partition_hash,
        "preparation_receipt_hash": prepared.preparation_receipt_hash,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "fresh_evidence": False,
    }


def _validate_ready_receipt(prepared: HarpV3PreparedInputs) -> None:
    """Authenticate the staged completion receipt before any publication."""

    receipt = read_json(prepared.cache_root / PREPARATION_RECEIPT)
    receipt_base = {
        key: value for key, value in receipt.items() if key != "receipt_hash"
    }
    content = read_json(prepared.cache_root / CONTENT_INDEX)
    content_base = {
        key: value for key, value in content.items() if key != "content_index_hash"
    }
    if (
        receipt.get("receipt_hash") != canonical_hash(receipt_base)
        or receipt.get("receipt_hash") != prepared.preparation_receipt_hash
        or receipt.get("status") != "PREPARED_INPUTS_NO_EXECUTION_AUTHORITY"
        or receipt.get("execution_amendment_created") is not False
        or receipt.get("execution_authorized") is not False
        or receipt.get(
            "cache_fsynced_and_independently_validated_before_manifest_open"
        )
        is not True
        or content.get("content_index_hash") != canonical_hash(content_base)
        or content.get("content_index_hash") != prepared.cache_content_sha256
        or sha256_file(prepared.development_manifest_path)
        != prepared.development_manifest_sha256
        or sha256_file(prepared.evaluation_manifest_path)
        != prepared.evaluation_manifest_sha256
    ):
        raise ProtocolError("HARP v3 staged preparation ready receipt drifted.")


def _prepared_from_payload(
    paths: HarpV3WorkspacePaths,
    payload: Mapping[str, object],
) -> HarpV3PreparedInputs:
    expected_keys = {
        "schema_version",
        "test_cache_content_sha256",
        "development_manifest_sha256",
        "evaluation_manifest_sha256",
        "parent_ledger_sha256",
        "partition_hash",
        "preparation_receipt_hash",
        "publication_status",
        "terminal_decision",
        "fresh_evidence",
    }
    if (
        set(payload) != expected_keys
        or payload.get("schema_version")
        != "midogpp_harp_v3_staged_prepared_inputs_v1"
        or payload.get("publication_status") != PUBLICATION_STATUS
        or payload.get("terminal_decision") != TERMINAL_DECISION
        or payload.get("fresh_evidence") is not False
    ):
        raise ProtocolError("HARP v3 prepared journal payload drifted.")
    digests = (
        "test_cache_content_sha256",
        "development_manifest_sha256",
        "evaluation_manifest_sha256",
        "parent_ledger_sha256",
        "partition_hash",
        "preparation_receipt_hash",
    )
    if any(not _is_sha256(payload.get(key)) for key in digests):
        raise ProtocolError("HARP v3 prepared journal hashes are malformed.")
    return HarpV3PreparedInputs(
        cache_root=paths.prepared_cache_root,
        development_manifest_path=paths.development_manifest_path,
        evaluation_manifest_path=paths.evaluation_manifest_path,
        cache_content_sha256=str(payload["test_cache_content_sha256"]),
        development_manifest_sha256=str(payload["development_manifest_sha256"]),
        evaluation_manifest_sha256=str(payload["evaluation_manifest_sha256"]),
        parent_ledger_sha256=str(payload["parent_ledger_sha256"]),
        partition_hash=str(payload["partition_hash"]),
        preparation_receipt_hash=str(payload["preparation_receipt_hash"]),
    )


def _digest_mapping(value: object, *, label: str) -> dict[str, str]:
    mapping = _mapping(value, label=label)
    result: dict[str, str] = {}
    for relative, digest in mapping.items():
        if (
            type(relative) is not str
            or not relative
            or Path(relative).is_absolute()
            or any(part in {"", ".", ".."} for part in Path(relative).parts)
            or not _is_sha256(digest)
        ):
            raise ProtocolError(f"HARP v3 {label} is malformed.")
        result[relative] = str(digest)
    if not result:
        raise ProtocolError(f"HARP v3 {label} is empty.")
    return result


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"HARP v3 {label} is malformed.")
    return value


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _relative(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:  # pragma: no cover - resolver already enforces this
        raise ProtocolError("HARP v3 transaction path escaped the repository.") from exc


def _write_exclusive(path: Path, raw: bytes) -> None:
    try:
        descriptor = os.open(
            path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
        )
    except OSError as exc:
        raise ProtocolError("HARP v3 preparation exclusive write failed.") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _fsync_directories(paths: Sequence[Path]) -> None:
    for path in dict.fromkeys(paths):
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _inject(fault_injector: FaultInjector | None, point: str) -> None:
    if fault_injector is not None:
        fault_injector(point)


__all__ = (
    "PreparationInventory",
    "PreparationJournal",
    "build_preparation_journal",
    "commit_prepared_inputs",
    "inspect_preparation_recovery",
    "inventory_tree",
    "load_preparation_journal",
    "recover_prepared_inputs",
    "staging_destinations",
)
