"""Pure audit of a rolled-back HARP v13 activation attempt."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path

from ....protocol import ProtocolError
from ....routing.harp_protocol import canonical_hash
from .....workspace.runtime import MidogppWorkspace, WorkspaceError
from .. import authorization
from ..activation_paths import RepositoryBoundary
from ..activation_transaction import ActivationJournal, load_journal
from ..identity import (
    AUTHORIZATION_SCOPE,
    EXECUTION_REVISION,
    EXPERIMENT_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
)
from ..source_seal import source_snapshot_identity
from .contracts import (
    ARCHIVED_AMENDMENT,
    HarpV13ActivationSupersessionPlan,
    SUPERSEDED_ROOT_RELATIVE,
    is_sha256,
)


def plan_harp_v13_activation_supersession(
    repository_root: str | Path,
) -> HarpV13ActivationSupersessionPlan:
    """Authenticate a rolled-back attempt without changing any byte."""

    return build_supersession_plan(RepositoryBoundary.open(repository_root))


def build_supersession_plan(
    boundary: RepositoryBoundary,
) -> HarpV13ActivationSupersessionPlan:
    journal = load_journal(boundary)
    _require_rolled_back_metadata(journal)
    _require_unconsumed_surface(boundary)
    _require_planned_workspace(boundary.resolved_root)

    archive_root = _archive_root(boundary, journal.journal_hash)
    archived_amendment = archive_root / ARCHIVED_AMENDMENT
    if os.path.lexists(journal.amendment_path):
        require_exact_regular(
            journal.amendment_path,
            journal.amendment_bytes,
            label="active amendment",
        )
    elif os.path.lexists(archived_amendment):
        require_exact_regular(
            archived_amendment,
            journal.amendment_bytes,
            label="archived amendment",
        )
    else:
        raise ProtocolError("HARP v13 supersession amendment evidence is absent.")

    amendment, prior_source = authenticate_prior_amendment(journal.amendment_bytes)
    replacement_source = dict(source_snapshot_identity(boundary.resolved_root))
    if dict(prior_source) == replacement_source:
        raise ProtocolError(
            "HARP v13 source snapshot is unchanged; use exact activation recovery."
        )
    provisional = HarpV13ActivationSupersessionPlan(
        repository_root=boundary.resolved_root,
        journal=journal,
        archive_root=archive_root,
        prior_source_snapshot=dict(prior_source),
        replacement_source_snapshot=replacement_source,
        amendment_hash=str(amendment["amendment_hash"]),
        supersession_plan_hash="",
    )
    return HarpV13ActivationSupersessionPlan(
        repository_root=provisional.repository_root,
        journal=provisional.journal,
        archive_root=provisional.archive_root,
        prior_source_snapshot=provisional.prior_source_snapshot,
        replacement_source_snapshot=provisional.replacement_source_snapshot,
        amendment_hash=provisional.amendment_hash,
        supersession_plan_hash=canonical_hash(provisional.hash_payload()),
    )


def recovery_source_snapshot_changed(repository_root: str | Path) -> bool:
    """Report source drift for an authenticated live journal without mutation."""

    boundary = RepositoryBoundary.open(repository_root)
    journal = load_journal(boundary)
    _amendment, prior_source = authenticate_prior_amendment(journal.amendment_bytes)
    current_source = dict(source_snapshot_identity(boundary.resolved_root))
    return dict(prior_source) != current_source


def require_harp_v13_recovery_source_current(
    repository_root: str | Path,
) -> None:
    """Block same-plan recovery before writes when its sealed source drifted."""

    if recovery_source_snapshot_changed(repository_root):
        raise ProtocolError(
            "HARP v13 recovery source snapshot drifted; archive the rolled-back "
            "activation before replanning."
        )


def authenticate_prior_amendment(
    raw: bytes,
) -> tuple[dict[str, object], Mapping[str, object]]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("HARP v13 prior amendment is unreadable.") from exc
    if not isinstance(value, dict):
        raise ProtocolError("HARP v13 prior amendment is malformed.")
    amendment_hash = value.get("amendment_hash")
    body = {key: item for key, item in value.items() if key != "amendment_hash"}
    if type(amendment_hash) is not str or amendment_hash != canonical_hash(body):
        raise ProtocolError("HARP v13 prior amendment self-hash drifted.")
    required = {
        "schema_version": authorization.EXECUTION_AMENDMENT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "execution_revision": EXECUTION_REVISION,
        "authorization_scope": AUTHORIZATION_SCOPE,
        "execution_authorized": True,
        "single_use": True,
        "authorization_exhausted": False,
        "consumed_test_reuse": True,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "fresh_evidence": False,
        "output_deletion_restores_authority": False,
    }
    if any(value.get(key) != expected for key, expected in required.items()):
        raise ProtocolError("HARP v13 prior amendment authority envelope drifted.")
    source = value.get("source_snapshot_identity")
    if not isinstance(source, Mapping):
        raise ProtocolError("HARP v13 prior source snapshot is absent.")
    for key in (
        "source_snapshot_manifest_sha256",
        "source_snapshot_tree_sha256",
    ):
        if not is_sha256(source.get(key)):
            raise ProtocolError("HARP v13 prior source snapshot is malformed.")
    if (
        type(source.get("source_snapshot_schema")) is not str
        or type(source.get("source_snapshot_member_count")) is not int
        or int(source["source_snapshot_member_count"]) < 1
    ):
        raise ProtocolError("HARP v13 prior source snapshot is malformed.")
    return value, dict(source)


def require_exact_regular(path: Path, raw: bytes, *, label: str) -> None:
    if not path.is_file() or path.is_symlink() or path.read_bytes() != raw:
        raise ProtocolError(f"HARP v13 {label} is absent or drifted.")


def _require_rolled_back_metadata(journal: ActivationJournal) -> None:
    for path, original, final, label in (
        (
            journal.config_path,
            journal.original_config_bytes,
            journal.final_config_bytes,
            "config",
        ),
        (
            journal.catalog_path,
            journal.original_catalog_bytes,
            journal.final_catalog_bytes,
            "catalog",
        ),
        (
            journal.registry_path,
            journal.original_registry_bytes,
            journal.final_registry_bytes,
            "registry",
        ),
    ):
        if original == final:
            raise ProtocolError(f"HARP v13 supersession {label} states are ambiguous.")
        require_exact_regular(path, original, label=f"rolled-back {label}")


def _require_unconsumed_surface(boundary: RepositoryBoundary) -> None:
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


def _require_planned_workspace(repository_root: Path) -> None:
    try:
        workspace = MidogppWorkspace.load(repository_root)
        workspace.validate()
        experiment = workspace.get_experiment(EXPERIMENT_ID)
    except WorkspaceError as exc:
        raise ProtocolError("HARP v13 rolled-back workspace failed validation.") from exc
    if experiment.status != "planned" or experiment.runnable:
        raise ProtocolError("HARP v13 workspace is not in the planned state.")


def _archive_root(boundary: RepositoryBoundary, journal_hash: str) -> Path:
    if not is_sha256(journal_hash):
        raise ProtocolError("HARP v13 activation journal hash is malformed.")
    relative = f"{SUPERSEDED_ROOT_RELATIVE}/{journal_hash}"
    raw = boundary.lexical_root / relative
    kind = "directory" if os.path.lexists(raw) else "future"
    return boundary.member(relative, label="superseded activation archive", kind=kind)


__all__ = (
    "authenticate_prior_amendment",
    "build_supersession_plan",
    "plan_harp_v13_activation_supersession",
    "recovery_source_snapshot_changed",
    "require_exact_regular",
    "require_harp_v13_recovery_source_current",
)
