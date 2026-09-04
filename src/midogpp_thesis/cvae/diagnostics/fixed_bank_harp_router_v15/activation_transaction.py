"""Durable, restartable HARP v15 activation transaction.

The immutable journal contains the exact original and final bytes for every
workspace member plus the exact amendment bytes.  It is fsynced before any of
those protected members change.  Recovery infers progress from byte identity,
so no mutable phase marker can become a second source of truth.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Protocol

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_bytes, canonical_hash
from ...runtime.artifact_io import sha256_file
from ....workspace.runtime import MidogppWorkspace, WorkspaceError
from . import authorization
from .activation_lock import LOCK_RELATIVE_PATH, activation_lock
from .activation_paths import RepositoryBoundary
from .activation_workspace import validate_rendered_workspace, yaml_mapping
from .config import HarpStage90V15Config, load_config
from .identity import (
    EXPERIMENT_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
)


ACTIVATION_RECEIPT_SCHEMA = "midogpp_harp_stage90_activation_receipt_v15"
TRANSACTION_SCHEMA = "midogpp_harp_stage90_activation_transaction_v15"
TRANSACTION_RELATIVE_PATH = (
    "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts/"
    "harp_router_v15/.harp_v15_activation_transaction.json"
)
FaultInjector = Callable[[str], None]


class ActivationPlanLike(Protocol):
    repository_root: Path
    config_path: Path
    registry_path: Path
    catalog_path: Path
    original_config_bytes: bytes
    original_registry_bytes: bytes
    original_catalog_bytes: bytes
    final_config_bytes: bytes
    final_registry_bytes: bytes
    final_catalog_bytes: bytes
    authorized_config: HarpStage90V15Config
    amendment_draft: object
    activation_plan_hash: str


@dataclass(frozen=True, slots=True)
class HarpV15ActivationReceipt:
    activation_plan_hash: str
    amendment_sha256: str
    config_sha256: str
    registry_sha256: str
    catalog_sha256: str
    workspace_registration_execution_contract_hash: str
    recovered_from_journal: bool = False
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "receipt_hash", canonical_hash(self._base_payload()))

    def _base_payload(self) -> dict[str, object]:
        return {
            "schema_version": ACTIVATION_RECEIPT_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "status": "ACTIVATED_AUTHORIZED_SINGLE_USE_NOT_CONSUMED",
            "activation_plan_hash": self.activation_plan_hash,
            "amendment_sha256": self.amendment_sha256,
            "config_sha256": self.config_sha256,
            "registry_sha256": self.registry_sha256,
            "catalog_sha256": self.catalog_sha256,
            "workspace_registration_execution_contract_hash": (
                self.workspace_registration_execution_contract_hash
            ),
            "registry_was_last_commit_point": True,
            "durable_recovery_journal_retained": True,
            "recovered_from_journal": self.recovered_from_journal,
            "authorization_lease_claimed": False,
            "labels_opened": False,
            "output_created": False,
            "publication_status": PUBLICATION_STATUS,
            "terminal_decision": TERMINAL_DECISION,
            "fresh_evidence": False,
            "may_feed_stage60_or_stage70": False,
            "may_feed_another_experiment": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._base_payload(), "receipt_hash": self.receipt_hash}


@dataclass(frozen=True, slots=True)
class ActivationJournal:
    repository_root: Path
    activation_plan_hash: str
    config_path: Path
    registry_path: Path
    catalog_path: Path
    amendment_path: Path
    original_config_bytes: bytes = field(repr=False)
    original_registry_bytes: bytes = field(repr=False)
    original_catalog_bytes: bytes = field(repr=False)
    final_config_bytes: bytes = field(repr=False)
    final_registry_bytes: bytes = field(repr=False)
    final_catalog_bytes: bytes = field(repr=False)
    amendment_bytes: bytes = field(repr=False)
    amendment_sha256: str
    journal_hash: str

    def payload_without_hash(self) -> dict[str, object]:
        root = self.repository_root
        return {
            "schema_version": TRANSACTION_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "activation_plan_hash": self.activation_plan_hash,
            "paths": {
                "config": self.config_path.relative_to(root).as_posix(),
                "registry": self.registry_path.relative_to(root).as_posix(),
                "catalog": self.catalog_path.relative_to(root).as_posix(),
                "amendment": self.amendment_path.relative_to(root).as_posix(),
            },
            "original_bytes_base64": {
                "config": _encode(self.original_config_bytes),
                "registry": _encode(self.original_registry_bytes),
                "catalog": _encode(self.original_catalog_bytes),
            },
            "final_bytes_base64": {
                "config": _encode(self.final_config_bytes),
                "registry": _encode(self.final_registry_bytes),
                "catalog": _encode(self.final_catalog_bytes),
            },
            "amendment_bytes_base64": _encode(self.amendment_bytes),
            "amendment_sha256": self.amendment_sha256,
            "commit_order": [
                "durable_journal",
                "exclusive_amendment",
                "authorized_config",
                "authorized_catalog",
                "diagnostic_registry_commit",
            ],
            "registry_is_last_commit_point": True,
            "journal_is_immutable": True,
            "progress_is_inferred_from_exact_bytes": True,
        }

    def to_bytes(self) -> bytes:
        payload = {**self.payload_without_hash(), "journal_hash": self.journal_hash}
        return canonical_bytes(payload) + b"\n"


def journal_path(repository_root: str | Path) -> Path:
    boundary = RepositoryBoundary.open(repository_root)
    return boundary.member(
        TRANSACTION_RELATIVE_PATH,
        label="activation transaction journal",
        kind="optional",
    )


def build_journal(plan: ActivationPlanLike) -> ActivationJournal:
    boundary = RepositoryBoundary.open(plan.repository_root)
    draft = plan.amendment_draft
    amendment_path = Path(getattr(draft, "amendment_path"))
    amendment_bytes = bytes(getattr(draft, "amendment_raw"))
    amendment_sha256 = str(getattr(draft, "amendment_sha256"))
    values = {
        "repository_root": boundary.resolved_root,
        "activation_plan_hash": plan.activation_plan_hash,
        "config_path": boundary.path(plan.config_path, label="config", kind="file"),
        "registry_path": boundary.path(plan.registry_path, label="registry", kind="file"),
        "catalog_path": boundary.path(plan.catalog_path, label="catalog", kind="file"),
        "amendment_path": boundary.path(
            amendment_path,
            label="execution amendment",
            kind="optional",
        ),
        "original_config_bytes": plan.original_config_bytes,
        "original_registry_bytes": plan.original_registry_bytes,
        "original_catalog_bytes": plan.original_catalog_bytes,
        "final_config_bytes": plan.final_config_bytes,
        "final_registry_bytes": plan.final_registry_bytes,
        "final_catalog_bytes": plan.final_catalog_bytes,
        "amendment_bytes": amendment_bytes,
        "amendment_sha256": amendment_sha256,
    }
    provisional = ActivationJournal(**values, journal_hash="")
    return ActivationJournal(
        **values,
        journal_hash=canonical_hash(provisional.payload_without_hash()),
    )


def inspect_activation_recovery(
    repository_root: str | Path,
) -> Mapping[str, object] | None:
    """Inspect a durable activation transaction without changing any byte."""

    boundary = RepositoryBoundary.open(repository_root)
    path = boundary.member(
        TRANSACTION_RELATIVE_PATH,
        label="activation transaction journal",
        kind="optional",
    )
    if not path.exists():
        return None
    journal = load_journal(boundary)
    states = _states(journal)
    complete = (
        states["amendment"] == "exact"
        and all(states[name] == "final" for name in ("config", "catalog", "registry"))
    )
    return {
        "schema_version": "midogpp_harp_stage90_activation_recovery_inspection_v15",
        "experiment_id": EXPERIMENT_ID,
        "status": (
            "ACTIVATED_AUTHORIZED_SINGLE_USE_NOT_CONSUMED"
            if complete
            else "RECOVERY_REQUIRED_EXACT_CONFIRMATION"
        ),
        "activation_plan_hash": journal.activation_plan_hash,
        "journal_hash": journal.journal_hash,
        "observed_states": states,
        "confirmation_required": (
            "ACTIVATE_HARP_V15_TERMINAL_CONSUMED_TEST_DIAGNOSTIC"
        ),
        "filesystem_mutations": 0,
        "registry_is_last_commit_point": True,
        "labels_opened": False,
        "output_created": False,
    }


def commit_activation(
    plan: ActivationPlanLike,
    *,
    confirmation: str,
    expected_confirmation: str,
    fault_injector: FaultInjector | None = None,
) -> HarpV15ActivationReceipt:
    if confirmation != expected_confirmation:
        raise ProtocolError("HARP v15 activation confirmation is absent or drifted.")
    journal = build_journal(plan)
    boundary = RepositoryBoundary.open(plan.repository_root)
    with activation_lock(boundary):
        _install_or_validate_journal(boundary, journal)
        _inject(fault_injector, "journal_durable")
        return _resume(
            boundary,
            journal,
            fault_injector=fault_injector,
            recovered=False,
        )


def recover_activation(
    repository_root: str | Path,
    *,
    confirmation: str,
    expected_confirmation: str,
    fault_injector: FaultInjector | None = None,
) -> HarpV15ActivationReceipt:
    if confirmation != expected_confirmation:
        raise ProtocolError("HARP v15 activation confirmation is absent or drifted.")
    boundary = RepositoryBoundary.open(repository_root)
    with activation_lock(boundary):
        journal = load_journal(boundary)
        return _resume(
            boundary,
            journal,
            fault_injector=fault_injector,
            recovered=True,
        )


def load_journal(boundary: RepositoryBoundary) -> ActivationJournal:
    path = boundary.member(
        TRANSACTION_RELATIVE_PATH,
        label="activation transaction journal",
        kind="file",
    )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("HARP v15 activation journal is unreadable.") from exc
    if not isinstance(raw, dict):
        raise ProtocolError("HARP v15 activation journal is malformed.")
    expected_keys = {
        "schema_version",
        "experiment_id",
        "activation_plan_hash",
        "paths",
        "original_bytes_base64",
        "final_bytes_base64",
        "amendment_bytes_base64",
        "amendment_sha256",
        "commit_order",
        "registry_is_last_commit_point",
        "journal_is_immutable",
        "progress_is_inferred_from_exact_bytes",
        "journal_hash",
    }
    base = {key: value for key, value in raw.items() if key != "journal_hash"}
    if (
        set(raw) != expected_keys
        or raw.get("schema_version") != TRANSACTION_SCHEMA
        or raw.get("experiment_id") != EXPERIMENT_ID
        or raw.get("journal_hash") != canonical_hash(base)
        or raw.get("commit_order")
        != [
            "durable_journal",
            "exclusive_amendment",
            "authorized_config",
            "authorized_catalog",
            "diagnostic_registry_commit",
        ]
        or raw.get("registry_is_last_commit_point") is not True
        or raw.get("journal_is_immutable") is not True
        or raw.get("progress_is_inferred_from_exact_bytes") is not True
    ):
        raise ProtocolError("HARP v15 activation journal authentication failed.")
    paths = _mapping(raw.get("paths"), label="journal paths")
    originals = _mapping(
        raw.get("original_bytes_base64"), label="journal original bytes"
    )
    finals = _mapping(raw.get("final_bytes_base64"), label="journal final bytes")
    expected_paths = {
        "config": authorization.WORKSPACE_CONFIG_RELATIVE_PATH,
        "registry": authorization.WORKSPACE_REGISTRY_RELATIVE_PATH,
        "catalog": authorization.WORKSPACE_ARTIFACT_CATALOG_RELATIVE_PATH,
        "amendment": authorization.WORKSPACE_AMENDMENT_RELATIVE_PATH,
    }
    if dict(paths) != expected_paths or set(originals) != {
        "config",
        "registry",
        "catalog",
    } or set(finals) != {"config", "registry", "catalog"}:
        raise ProtocolError("HARP v15 activation journal path binding drifted.")
    amendment = _decode(raw.get("amendment_bytes_base64"), label="amendment")
    amendment_sha256 = str(raw.get("amendment_sha256"))
    if hashlib.sha256(amendment).hexdigest() != amendment_sha256:
        raise ProtocolError("HARP v15 activation journal amendment hash drifted.")
    journal = ActivationJournal(
        repository_root=boundary.resolved_root,
        activation_plan_hash=str(raw.get("activation_plan_hash")),
        config_path=boundary.member(
            expected_paths["config"], label="config", kind="file"
        ),
        registry_path=boundary.member(
            expected_paths["registry"], label="registry", kind="file"
        ),
        catalog_path=boundary.member(
            expected_paths["catalog"], label="catalog", kind="file"
        ),
        amendment_path=boundary.member(
            expected_paths["amendment"], label="execution amendment", kind="optional"
        ),
        original_config_bytes=_decode(originals["config"], label="original config"),
        original_registry_bytes=_decode(
            originals["registry"], label="original registry"
        ),
        original_catalog_bytes=_decode(
            originals["catalog"], label="original catalog"
        ),
        final_config_bytes=_decode(finals["config"], label="final config"),
        final_registry_bytes=_decode(finals["registry"], label="final registry"),
        final_catalog_bytes=_decode(finals["catalog"], label="final catalog"),
        amendment_bytes=amendment,
        amendment_sha256=amendment_sha256,
        journal_hash=str(raw["journal_hash"]),
    )
    if journal.to_bytes() != path.read_bytes():
        raise ProtocolError("HARP v15 activation journal bytes are not canonical.")
    return journal


def _resume(
    boundary: RepositoryBoundary,
    journal: ActivationJournal,
    *,
    fault_injector: FaultInjector | None,
    recovered: bool,
) -> HarpV15ActivationReceipt:
    _validate_unconsumed_surface(boundary)
    states = _states(journal)
    if states["amendment"] not in {"absent", "exact"} or any(
        states[name] not in {"original", "final"}
        for name in ("config", "catalog", "registry")
    ):
        raise ProtocolError("HARP v15 activation recovery observed non-journal bytes.")

    # A runnable registry must never coexist with an incomplete prerequisite.
    if states["registry"] == "final" and (
        states["config"] != "final" or states["catalog"] != "final"
    ):
        _atomic_replace(
            journal.registry_path,
            journal.original_registry_bytes,
            journal.activation_plan_hash + ".close-gate",
        )
        states["registry"] = "original"

    try:
        if states["amendment"] == "absent":
            _write_exclusive(journal.amendment_path, journal.amendment_bytes)
            _fsync_directories((journal.amendment_path.parent,))
        _inject(fault_injector, "amendment_committed")

        if states["config"] == "original":
            _atomic_replace(
                journal.config_path,
                journal.final_config_bytes,
                journal.activation_plan_hash + ".config",
            )
        _inject(fault_injector, "config_committed")
        _validate_config_and_amendment(journal)

        if states["catalog"] == "original":
            _atomic_replace(
                journal.catalog_path,
                journal.final_catalog_bytes,
                journal.activation_plan_hash + ".catalog",
            )
        _inject(fault_injector, "catalog_committed")
        validate_rendered_workspace(
            yaml_mapping(journal.final_registry_bytes, label="journal registry"),
            yaml_mapping(journal.final_catalog_bytes, label="journal catalog"),
        )

        # Registry is deliberately the only runnable-gate commit point.
        if states["registry"] == "original":
            _atomic_replace(
                journal.registry_path,
                journal.final_registry_bytes,
                journal.activation_plan_hash + ".registry",
            )
        _inject(fault_injector, "registry_committed")
        return _validate_committed_workspace(journal, recovered=recovered)
    except Exception as exc:
        rollback_errors = _rollback(journal, fault_injector=fault_injector)
        if rollback_errors:
            raise ProtocolError(
                "HARP v15 activation failed and rollback was incomplete; the "
                "durable exact-byte recovery journal was retained."
            ) from exc
        raise ProtocolError(
            "HARP v15 activation failed closed; workspace metadata was restored "
            "to planned bytes and the exact-byte recovery journal was retained."
        ) from exc


def _validate_unconsumed_surface(boundary: RepositoryBoundary) -> None:
    boundary.path(
        authorization.lease_path(boundary.resolved_root),
        label="authorization lease",
        kind="absent",
    )
    boundary.member(
        authorization.WORKSPACE_OUTPUT_CANONICAL_PATH,
        label="output identity",
        kind="future",
    )


def _validate_config_and_amendment(journal: ActivationJournal) -> None:
    if journal.config_path.read_bytes() != journal.final_config_bytes:
        raise ProtocolError("HARP v15 committed config bytes drifted.")
    if journal.amendment_path.read_bytes() != journal.amendment_bytes:
        raise ProtocolError("HARP v15 committed amendment bytes drifted.")
    config = load_config(journal.config_path)
    try:
        amendment = json.loads(journal.amendment_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("HARP v15 journal amendment is not JSON.") from exc
    authorization.validate_execution_amendment_payload(
        amendment,
        config,
        repo_root=journal.repository_root,
    )


def _validate_committed_workspace(
    journal: ActivationJournal,
    *,
    recovered: bool,
) -> HarpV15ActivationReceipt:
    if (
        journal.registry_path.read_bytes() != journal.final_registry_bytes
        or journal.catalog_path.read_bytes() != journal.final_catalog_bytes
    ):
        raise ProtocolError("HARP v15 committed workspace bytes drifted.")
    try:
        authority = MidogppWorkspace.load(
            journal.repository_root
        ).validate_preparation_authority(EXPERIMENT_ID)
    except WorkspaceError as exc:
        raise ProtocolError(
            "HARP v15 committed workspace authority failed catalog-backed validation."
        ) from exc
    if authority is None or authority.authority_sha256 != journal.amendment_sha256:
        raise ProtocolError("HARP v15 committed workspace authority is absent or drifted.")
    return HarpV15ActivationReceipt(
        activation_plan_hash=journal.activation_plan_hash,
        amendment_sha256=journal.amendment_sha256,
        config_sha256=sha256_file(journal.config_path),
        registry_sha256=sha256_file(journal.registry_path),
        catalog_sha256=sha256_file(journal.catalog_path),
        workspace_registration_execution_contract_hash=str(
            authority.workspace_registration_contract_hash
        ),
        recovered_from_journal=recovered,
    )


def _rollback(
    journal: ActivationJournal,
    *,
    fault_injector: FaultInjector | None,
) -> tuple[BaseException, ...]:
    errors: list[BaseException] = []
    for name, path, original, final in (
        (
            "registry",
            journal.registry_path,
            journal.original_registry_bytes,
            journal.final_registry_bytes,
        ),
        (
            "catalog",
            journal.catalog_path,
            journal.original_catalog_bytes,
            journal.final_catalog_bytes,
        ),
        (
            "config",
            journal.config_path,
            journal.original_config_bytes,
            journal.final_config_bytes,
        ),
    ):
        try:
            current = path.read_bytes()
            if current == final:
                _inject(fault_injector, f"rollback_{name}_before_commit")
                _atomic_replace(
                    path,
                    original,
                    journal.activation_plan_hash + f".rollback-{name}",
                )
            elif current != original:
                raise ProtocolError(
                    f"HARP v15 rollback {name} encountered non-journal bytes."
                )
            _inject(fault_injector, f"rollback_{name}_committed")
        except BaseException as exc:
            errors.append(exc)
    return tuple(errors)


def _states(journal: ActivationJournal) -> dict[str, str]:
    states = {
        "config": _byte_state(
            journal.config_path,
            original=journal.original_config_bytes,
            final=journal.final_config_bytes,
        ),
        "catalog": _byte_state(
            journal.catalog_path,
            original=journal.original_catalog_bytes,
            final=journal.final_catalog_bytes,
        ),
        "registry": _byte_state(
            journal.registry_path,
            original=journal.original_registry_bytes,
            final=journal.final_registry_bytes,
        ),
    }
    if not os.path.lexists(journal.amendment_path):
        states["amendment"] = "absent"
    elif (
        journal.amendment_path.is_file()
        and not journal.amendment_path.is_symlink()
        and journal.amendment_path.read_bytes() == journal.amendment_bytes
    ):
        states["amendment"] = "exact"
    else:
        states["amendment"] = "drifted"
    return states


def _byte_state(path: Path, *, original: bytes, final: bytes) -> str:
    if not path.is_file() or path.is_symlink():
        return "drifted"
    raw = path.read_bytes()
    if raw == original:
        return "original"
    if raw == final:
        return "final"
    return "drifted"


def _install_or_validate_journal(
    boundary: RepositoryBoundary,
    journal: ActivationJournal,
) -> None:
    path = boundary.member(
        TRANSACTION_RELATIVE_PATH,
        label="activation transaction journal",
        kind="optional",
    )
    expected = journal.to_bytes()
    if path.exists():
        observed = load_journal(boundary)
        if observed.to_bytes() != expected:
            raise ProtocolError("HARP v15 existing activation journal does not match plan.")
        return
    temporary = path.parent / (
        f".{path.name}.pending.{os.getpid()}.{time.time_ns()}"
    )
    _write_exclusive(temporary, expected)
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            observed = load_journal(boundary)
            if observed.to_bytes() != expected:
                raise ProtocolError(
                    "HARP v15 concurrent activation journal differs from plan."
                )
        _fsync_directories((path.parent,))
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_replace(path: Path, raw: bytes, token: str) -> None:
    temporary = path.parent / f".{path.name}.harp-v15-{token}.tmp"
    if os.path.lexists(temporary):
        if not temporary.is_file() or temporary.is_symlink() or temporary.read_bytes() != raw:
            raise ProtocolError("HARP v15 activation staging path is unsafe.")
    else:
        _write_exclusive(temporary, raw)
    os.replace(temporary, path)
    _fsync_directories((path.parent,))


def _write_exclusive(path: Path, raw: bytes) -> None:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as exc:
        raise ProtocolError("HARP v15 activation exclusive write failed.") from exc
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


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"HARP v15 {label} is malformed.")
    return value


def _encode(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _decode(value: object, *, label: str) -> bytes:
    if type(value) is not str:
        raise ProtocolError(f"HARP v15 journal {label} is malformed.")
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise ProtocolError(f"HARP v15 journal {label} is malformed.") from exc


def _inject(fault_injector: FaultInjector | None, point: str) -> None:
    if fault_injector is not None:
        fault_injector(point)


__all__ = (
    "ActivationJournal",
    "HarpV15ActivationReceipt",
    "LOCK_RELATIVE_PATH",
    "TRANSACTION_RELATIVE_PATH",
    "activation_lock",
    "commit_activation",
    "inspect_activation_recovery",
    "journal_path",
    "load_journal",
    "recover_activation",
)
