"""Single-purpose publication of the workspace-sealed v4 amendment.

The publisher uses exclusive creation and durable file/directory sync.  It
deliberately leaves the final envelope unrendered and cannot claim a lease,
open labels, mount FUSE, or launch the router.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

from ...protocol import ProtocolError
from .hashing import bytes_sha256, payload_sha256, require_sha256
from .inputs import inventory_existing_inputs
from .host import capture_workstation_topology
from .snapshot import capture_workspace_snapshot
from .validation import (
    AmendmentPublicationValidationReceipt,
    observe_publication_surfaces,
    validate_amendment_only_postpublication,
)
from .workspace import (
    WorkspacePreparationContext,
    replay_prepublication,
    validate_preflight_document,
)


@dataclass(frozen=True, slots=True)
class AmendmentPublicationReceipt:
    validation: AmendmentPublicationValidationReceipt
    amendment_path: Path
    amendment_sha256: str
    pre_amendment_plan_sha256: str
    prospective_final_envelope_sha256: str
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.validation) is not AmendmentPublicationValidationReceipt
            or not isinstance(self.amendment_path, Path)
            or not self.amendment_path.is_absolute()
            or self.validation.amendment_sha256 != self.amendment_sha256
        ):
            raise ProtocolError("OE-PPUR v4 amendment publication receipt drifted.")
        for role in (
            "amendment_sha256",
            "pre_amendment_plan_sha256",
            "prospective_final_envelope_sha256",
        ):
            object.__setattr__(
                self, role, require_sha256(getattr(self, role), role.replace("_", " "))
            )
        object.__setattr__(self, "receipt_hash", payload_sha256(self.to_payload()))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v4_amendment_publication_receipt_v1",
            "validation": self.validation.to_payload(),
            "amendment_path": self.amendment_path.as_posix(),
            "amendment_sha256": self.amendment_sha256,
            "pre_amendment_plan_sha256": self.pre_amendment_plan_sha256,
            "prospective_final_envelope_sha256": (
                self.prospective_final_envelope_sha256
            ),
            "final_envelope_rendered": False,
            "authorization_consumed": False,
            "launch_authorized": False,
            "target_labels_opened": False,
            "experiment_launched": False,
        }


def publish_amendment_only(
    context: WorkspacePreparationContext,
    *,
    preflight_raw: bytes,
) -> AmendmentPublicationReceipt:
    """Replay an exact preflight, then issue input #7 once with O_EXCL."""

    if type(context) is not WorkspacePreparationContext or type(preflight_raw) is not bytes:
        raise ProtocolError("OE-PPUR v4 amendment publication input is untyped.")
    candidate = context.candidate
    preflight = replay_prepublication(context)
    validate_preflight_document(
        preflight_raw,
        context=context,
        receipt=preflight,
    )
    amendment = candidate.plan.topology.amendment_path
    amendment_root = amendment.parent
    if os.path.lexists(amendment_root):
        raise ProtocolError("OE-PPUR v4 amendment root already exists.")
    try:
        amendment_root.mkdir(mode=0o700, parents=False, exist_ok=False)
        _fsync_directory(amendment_root.parent)
        _write_exclusive(amendment, candidate.amendment_raw)
        _fsync_directory(amendment_root)
    except (OSError, FileExistsError) as exc:
        raise ProtocolError(
            "OE-PPUR v4 amendment publication failed; partial state is terminal."
        ) from exc

    try:
        published = amendment.read_bytes()
    except OSError as exc:
        raise ProtocolError("OE-PPUR v4 issued amendment could not be read back.") from exc
    observed_workspace = capture_workspace_snapshot(context.seal_spec)
    observed_inputs = inventory_existing_inputs(context.input_specs)
    validation = validate_amendment_only_postpublication(
        candidate,
        preflight,
        observed_workspace=observed_workspace,
        observed_existing_inputs=observed_inputs,
        observed_topology=candidate.plan.topology,
        observed_scientific=candidate.plan.scientific,
        observed_workstation=capture_workstation_topology(
            artifact_parent=candidate.plan.topology.canonical_output_parent,
            scratch_root=candidate.plan.topology.scratch_root,
        ),
        observed_surfaces=observe_publication_surfaces(candidate.plan),
        published_amendment_raw=published,
    )
    return AmendmentPublicationReceipt(
        validation=validation,
        amendment_path=amendment,
        amendment_sha256=bytes_sha256(published),
        pre_amendment_plan_sha256=candidate.plan.plan_hash,
        prospective_final_envelope_sha256=bytes_sha256(candidate.envelope_raw),
    )


def _write_exclusive(path: Path, raw: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
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


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = ("AmendmentPublicationReceipt", "publish_amendment_only")
