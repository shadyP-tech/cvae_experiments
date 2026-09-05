"""Read-only audit and recovery planning for active v17 authority."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import os
from pathlib import Path

from ....protocol import ProtocolError
from ....routing.harp_protocol import canonical_bytes, canonical_hash
from .....workspace.runtime import MidogppWorkspace, WorkspaceError
from .. import authorization
from ..activation_paths import RepositoryBoundary
from ..activation_transaction import ActivationJournal, load_journal
from ..activation_workspace import validate_rendered_workspace, yaml_mapping
from ..config import HarpStage90V17Config, load_config
from ..identity import EXPERIMENT_ID
from ..source_seal import source_snapshot_identity
from .admin_snapshot import (
    inspect_workspace_admin_output,
    load_recovery_admin_snapshot,
    validate_partial_snapshot_tree,
)
from .audit import authenticate_prior_amendment, require_exact_regular
from .contracts import (
    ACTIVE_SUPERSESSION_RECEIPT,
    ARCHIVED_ADMIN_CONTENT,
    ARCHIVED_ADMIN_MANIFEST,
    ARCHIVED_AMENDMENT,
    ARCHIVED_FINAL_CATALOG,
    ARCHIVED_FINAL_CONFIG,
    ARCHIVED_FINAL_REGISTRY,
    ARCHIVED_JOURNAL,
    ARCHIVED_RETIREMENT_FENCE,
    HarpV17ActiveActivationSupersessionPlan,
    RETIRED_ADMIN_OUTPUT,
    SUPERSEDED_ACTIVE_ROOT_RELATIVE,
    active_receipt_payload,
    is_sha256,
    retirement_fence_payload,
)


def plan_harp_v17_active_activation_supersession(
    repository_root: str | Path,
) -> HarpV17ActiveActivationSupersessionPlan:
    """Plan revocation of an exact active, unclaimed v17 activation."""

    return build_active_supersession_plan(RepositoryBoundary.open(repository_root))


def build_active_supersession_plan(
    boundary: RepositoryBoundary,
) -> HarpV17ActiveActivationSupersessionPlan:
    """Authenticate an initial or explicitly fenced recovery state, read-only."""

    # The live immutable journal is retained until the final cleanup operation,
    # so every recoverable injected-crash state remains self-identifying.
    journal = load_journal(boundary)
    if not is_sha256(journal.activation_plan_hash):
        raise ProtocolError("HARP v17 activation plan hash is malformed.")
    archive_root = _active_archive_root(boundary, journal.journal_hash)
    metadata_states = _active_metadata_states(journal)
    retirement_started = _retirement_evidence_exists(
        boundary,
        archive_root=archive_root,
        journal=journal,
    )
    if not retirement_started and metadata_states != {
        "registry": "final",
        "catalog": "final",
        "config": "final",
    }:
        raise ProtocolError(
            "HARP v17 active supersession initial metadata is not fully activated."
        )
    _require_allowed_metadata_state(metadata_states)

    amendment_location = _exact_live_or_archived(
        journal.amendment_path,
        archive_root / ARCHIVED_AMENDMENT,
        journal.amendment_bytes,
        label="execution amendment",
        require_live=not retirement_started,
    )
    amendment, prior_source = authenticate_prior_amendment(journal.amendment_bytes)
    scratch_root = _scratch_root_from_journal(journal)
    _require_scratch_absent(scratch_root)

    metadata_fully_active = metadata_states == {
        "registry": "final",
        "catalog": "final",
        "config": "final",
    }
    archived_admin_manifest = archive_root / ARCHIVED_ADMIN_MANIFEST
    if metadata_fully_active:
        # While the activated workspace still exists, derive the snapshot from
        # that authenticated live surface even when a partial archive exists.
        # The archive validator below then requires any persisted copy to match.
        active_config = _require_active_workspace(boundary, journal)
        output_root, admin_manifest, admin_files = inspect_workspace_admin_output(
            boundary,
            journal=journal,
            active_config=active_config,
        )
        output_location = (
            "absent" if admin_manifest["state"] == "ABSENT" else "live"
        )
    else:
        if not os.path.lexists(archived_admin_manifest):
            raise ProtocolError(
                "HARP v17 recovery lacks its durable admin snapshot manifest."
            )
        output_root, admin_manifest, admin_files, output_location = (
            load_recovery_admin_snapshot(boundary, archive_root=archive_root)
        )

    replacement_source = dict(source_snapshot_identity(boundary.resolved_root))
    if dict(prior_source) == replacement_source:
        raise ProtocolError(
            "HARP v17 source snapshot is unchanged; active authority must not be "
            "superseded."
        )
    provisional = HarpV17ActiveActivationSupersessionPlan(
        repository_root=boundary.resolved_root,
        journal=journal,
        archive_root=archive_root,
        output_root=output_root,
        scratch_root=scratch_root,
        prior_source_snapshot=dict(prior_source),
        replacement_source_snapshot=replacement_source,
        amendment_hash=str(amendment["amendment_hash"]),
        admin_snapshot_manifest=admin_manifest,
        admin_snapshot_files=admin_files,
        supersession_plan_hash="",
        recovery_state={},
    )
    plan_hash = canonical_hash(provisional.hash_payload())
    fence_location = _validate_retirement_fence_state(
        boundary,
        archive_root=archive_root,
        plan_hash=plan_hash,
        journal=journal,
        retirement_started=retirement_started,
    )
    if (
        not metadata_fully_active
        or amendment_location == "archived"
        or output_location == "retired"
    ) and fence_location not in {"live", "archived"}:
        raise ProtocolError(
            "HARP v17 recovery mutation lacks its authenticated retirement fence."
        )
    archive_complete = _validate_active_archive_if_present(
        archive_root,
        plan=provisional,
    )
    receipt_present = _validate_active_receipt_if_present(
        archive_root,
        plan=provisional,
        plan_hash=plan_hash,
    )
    recovery_state = {
        "retirement_started": retirement_started,
        "metadata_states": metadata_states,
        "amendment_location": amendment_location,
        "output_location": output_location,
        "retirement_fence_location": fence_location,
        "archive_complete": archive_complete,
        "terminal_receipt_present": receipt_present,
    }
    return replace(
        provisional,
        supersession_plan_hash=plan_hash,
        recovery_state=recovery_state,
    )


def _active_metadata_states(journal: ActivationJournal) -> dict[str, str]:
    states: dict[str, str] = {}
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
        if original == final or not path.is_file() or path.is_symlink():
            raise ProtocolError(f"HARP v17 active supersession {name} is unsafe.")
        raw = path.read_bytes()
        if raw == final:
            states[name] = "final"
        elif raw == original:
            states[name] = "original"
        else:
            raise ProtocolError(
                f"HARP v17 active supersession {name} bytes are unrecognized."
            )
    return states


def _require_allowed_metadata_state(states: Mapping[str, str]) -> None:
    ordered = (
        {"registry": "final", "catalog": "final", "config": "final"},
        {"registry": "original", "catalog": "final", "config": "final"},
        {"registry": "original", "catalog": "original", "config": "final"},
        {"registry": "original", "catalog": "original", "config": "original"},
    )
    if dict(states) not in ordered:
        raise ProtocolError(
            "HARP v17 active supersession metadata order is impossible."
        )


def _retirement_evidence_exists(
    boundary: RepositoryBoundary,
    *,
    archive_root: Path,
    journal: ActivationJournal,
) -> bool:
    lease = authorization.lease_path(boundary.resolved_root)
    if os.path.lexists(lease):
        if not lease.is_file() or lease.is_symlink():
            raise ProtocolError(
                "HARP v17 scientific authorization lease already exists; "
                "supersession is forbidden."
            )
        return True
    if not os.path.lexists(archive_root):
        return False
    if not archive_root.is_dir() or archive_root.is_symlink():
        raise ProtocolError("HARP v17 active supersession archive is unsafe.")
    archived_journal = archive_root / ARCHIVED_JOURNAL
    if not os.path.lexists(archived_journal):
        return False
    require_exact_regular(
        archived_journal,
        journal.to_bytes(),
        label="archived activation journal",
    )
    return True


def _exact_live_or_archived(
    live: Path,
    archived: Path,
    expected: bytes,
    *,
    label: str,
    require_live: bool,
) -> str:
    live_exact = (
        live.is_file() and not live.is_symlink() and live.read_bytes() == expected
    )
    archived_exact = (
        archived.is_file()
        and not archived.is_symlink()
        and archived.read_bytes() == expected
    )
    if os.path.lexists(live) and not live_exact:
        raise ProtocolError(f"HARP v17 live {label} drifted.")
    if os.path.lexists(archived) and not archived_exact:
        raise ProtocolError(f"HARP v17 archived {label} drifted.")
    if require_live and not live_exact:
        raise ProtocolError(f"HARP v17 live {label} is absent.")
    if not live_exact and not archived_exact:
        raise ProtocolError(f"HARP v17 {label} evidence is absent.")
    return "live" if live_exact else "archived"


def _scratch_root_from_journal(journal: ActivationJournal) -> Path:
    config = yaml_mapping(journal.final_config_bytes, label="activated config")
    runtime = config.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ProtocolError("HARP v17 activated scratch binding is malformed.")
    value = runtime.get("scratch_root")
    if type(value) is not str:
        raise ProtocolError("HARP v17 activated scratch binding is absent.")
    path = Path(value)
    if not path.is_absolute():
        raise ProtocolError("HARP v17 configured scratch root is not absolute.")
    return path


def _require_scratch_absent(scratch: Path) -> None:
    if os.path.lexists(scratch):
        raise ProtocolError(
            "HARP v17 active supersession requires the scratch root to be absent."
        )


def _require_active_workspace(
    boundary: RepositoryBoundary,
    journal: ActivationJournal,
) -> HarpStage90V17Config:
    """Authenticate the committed gate without revalidating the stale source seal."""

    validate_rendered_workspace(
        yaml_mapping(journal.final_registry_bytes, label="activated registry"),
        yaml_mapping(journal.final_catalog_bytes, label="activated catalog"),
    )
    config = load_config(journal.config_path)
    if (
        not config.execution_authorized
        or config.expected_execution_amendment_sha256 != journal.amendment_sha256
    ):
        raise ProtocolError("HARP v17 activated configuration authority drifted.")
    try:
        workspace = MidogppWorkspace.load(boundary.resolved_root)
        workspace.validate()
        experiment = workspace.get_experiment(EXPERIMENT_ID)
    except WorkspaceError as exc:
        raise ProtocolError("HARP v17 activated workspace failed validation.") from exc
    if experiment.status != "diagnostic" or not experiment.runnable:
        raise ProtocolError("HARP v17 workspace is not in the activated state.")
    return config


def _validate_retirement_fence_state(
    boundary: RepositoryBoundary,
    *,
    archive_root: Path,
    plan_hash: str,
    journal: ActivationJournal,
    retirement_started: bool,
) -> str:
    plan = HarpV17ActiveActivationSupersessionPlan(
        repository_root=boundary.resolved_root,
        journal=journal,
        archive_root=archive_root,
        output_root=boundary.resolved_root
        / authorization.WORKSPACE_OUTPUT_CANONICAL_PATH,
        scratch_root=_scratch_root_from_journal(journal),
        prior_source_snapshot={},
        replacement_source_snapshot={},
        amendment_hash="",
        admin_snapshot_manifest={},
        admin_snapshot_files={},
        supersession_plan_hash=plan_hash,
    )
    # retirement_fence_payload only consumes immutable journal and plan hashes.
    expected = canonical_bytes(retirement_fence_payload(plan)) + b"\n"
    live = authorization.lease_path(boundary.resolved_root)
    archived = archive_root / ARCHIVED_RETIREMENT_FENCE
    live_exact = live.is_file() and not live.is_symlink() and live.read_bytes() == expected
    archived_exact = (
        archived.is_file()
        and not archived.is_symlink()
        and archived.read_bytes() == expected
    )
    if os.path.lexists(live) and not live_exact:
        raise ProtocolError(
            "HARP v17 scientific lease or malformed retirement fence blocks "
            "supersession."
        )
    if os.path.lexists(archived) and not archived_exact:
        raise ProtocolError("HARP v17 archived retirement fence drifted.")
    if live_exact and archived_exact:
        raise ProtocolError("HARP v17 retirement fence location is ambiguous.")
    if live_exact:
        return "live"
    if archived_exact:
        return "archived"
    return "absent" if retirement_started else "not_started"


def _validate_active_archive_if_present(
    archive_root: Path,
    *,
    plan: HarpV17ActiveActivationSupersessionPlan,
) -> bool:
    if not os.path.lexists(archive_root):
        return False
    if not archive_root.is_dir() or archive_root.is_symlink():
        raise ProtocolError("HARP v17 active supersession archive is unsafe.")
    journal = plan.journal
    expected = {
        ARCHIVED_JOURNAL: journal.to_bytes(),
        ARCHIVED_AMENDMENT: journal.amendment_bytes,
        ARCHIVED_FINAL_CONFIG: journal.final_config_bytes,
        ARCHIVED_FINAL_REGISTRY: journal.final_registry_bytes,
        ARCHIVED_FINAL_CATALOG: journal.final_catalog_bytes,
        ARCHIVED_ADMIN_MANIFEST: canonical_bytes(plan.admin_snapshot_manifest) + b"\n",
    }
    complete = True
    for name, raw in expected.items():
        path = archive_root / name
        if not os.path.lexists(path):
            complete = False
        else:
            require_exact_regular(path, raw, label=f"archived {name}")
    allowed = {
        *expected,
        ARCHIVED_ADMIN_CONTENT,
        RETIRED_ADMIN_OUTPUT,
        ARCHIVED_RETIREMENT_FENCE,
        ACTIVE_SUPERSESSION_RECEIPT,
    }
    if any(path.name not in allowed for path in archive_root.iterdir()):
        raise ProtocolError(
            "HARP v17 active supersession archive has an unknown member."
        )
    if plan.admin_snapshot_manifest.get("state") == "WORKSPACE_ADMIN_PRISTINE":
        content = archive_root / ARCHIVED_ADMIN_CONTENT
        if not content.exists():
            complete = False
        else:
            content_complete = validate_partial_snapshot_tree(
                content,
                directories=tuple(plan.admin_snapshot_manifest["directories"]),
                files=plan.admin_snapshot_files,
            )
            complete = complete and content_complete
    elif os.path.lexists(archive_root / ARCHIVED_ADMIN_CONTENT):
        raise ProtocolError("HARP v17 absent admin state has archived content.")
    return complete


def _validate_active_receipt_if_present(
    archive_root: Path,
    *,
    plan: HarpV17ActiveActivationSupersessionPlan,
    plan_hash: str,
) -> bool:
    path = archive_root / ACTIVE_SUPERSESSION_RECEIPT
    if not os.path.lexists(path):
        return False
    expected = active_receipt_payload(
        replace(plan, supersession_plan_hash=plan_hash)
    )
    require_exact_regular(
        path,
        canonical_bytes(expected) + b"\n",
        label="active supersession receipt",
    )
    return True


def _active_archive_root(boundary: RepositoryBoundary, journal_hash: str) -> Path:
    if not is_sha256(journal_hash):
        raise ProtocolError("HARP v17 activation journal hash is malformed.")
    relative = f"{SUPERSEDED_ACTIVE_ROOT_RELATIVE}/{journal_hash}"
    raw = boundary.lexical_root / relative
    kind = "directory" if os.path.lexists(raw) else "future"
    return boundary.member(
        relative,
        label="superseded active activation archive",
        kind=kind,
    )


__all__ = (
    "build_active_supersession_plan",
    "plan_harp_v17_active_activation_supersession",
)
