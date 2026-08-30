"""Read-only launch admission for the workspace-sealed OE-PPUR v4 lifecycle."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ...protocol import ProtocolError
from .config import RouterV4Config, SEALED_STATE
from .hashing import canonical_hash, require_sha256
from .identity import (
    AUTHORIZATION_AMENDMENT_ARTIFACT_ID,
    DIRECT_INPUT_ARTIFACT_IDS,
    FORBIDDEN_OPERATIONAL_PATH_FRAGMENTS,
    OUTPUT_ARTIFACT_ID,
)


@dataclass(frozen=True, slots=True)
class SealedEnvelopeAdmission:
    config: RouterV4Config
    workspace_snapshot_sha256: str
    workspace_plan_sha256: str
    authorization_amendment_sha256: str
    final_envelope_sha256: str
    direct_input_artifact_ids: tuple[str, ...]
    resolved_paths: tuple[Path, ...]
    topology_contract_sha256: str
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.config) is not RouterV4Config
            or self.config.authorization_state != SEALED_STATE
            or self.config.execution_amendment_issued is not True
            or self.config.launch_authorized is not False
            or tuple(self.direct_input_artifact_ids) != DIRECT_INPUT_ARTIFACT_IDS
            or len(self.resolved_paths) != 7
            or not all(
                isinstance(path, Path) and path.is_absolute()
                for path in self.resolved_paths
            )
        ):
            raise ProtocolError("OE-PPUR v4 sealed-envelope admission drifted.")
        for role in (
            "workspace_snapshot_sha256",
            "workspace_plan_sha256",
            "authorization_amendment_sha256",
            "final_envelope_sha256",
            "topology_contract_sha256",
        ):
            object.__setattr__(
                self,
                role,
                require_sha256(getattr(self, role), role.replace("_", " ")),
            )
        if (
            self.workspace_plan_sha256 != self.config.workspace_plan_sha256
            or self.authorization_amendment_sha256
            != self.config.authorization_amendment_sha256
        ):
            raise ProtocolError("OE-PPUR v4 config/envelope authority drifted.")
        _reject_operational_predecessor_paths(self.resolved_paths)
        object.__setattr__(self, "receipt_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v4_sealed_envelope_admission_v1",
            "experiment_id": self.config.experiment_id,
            "output_artifact_id": OUTPUT_ARTIFACT_ID,
            "workspace_snapshot_sha256": self.workspace_snapshot_sha256,
            "workspace_plan_sha256": self.workspace_plan_sha256,
            "authorization_amendment_sha256": self.authorization_amendment_sha256,
            "final_envelope_sha256": self.final_envelope_sha256,
            "direct_input_artifact_ids": list(self.direct_input_artifact_ids),
            "resolved_paths": [path.as_posix() for path in self.resolved_paths],
            "topology_contract_sha256": self.topology_contract_sha256,
            "v3_amendment_used_as_authority": False,
            "v3_operational_state_used": False,
            "target_labels_opened": False,
            "filesystem_mutation_performed": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "receipt_hash": self.receipt_hash}


@dataclass(frozen=True, slots=True)
class LaunchAuthority:
    experiment_id: str
    workspace_plan_sha256: str
    authorization_amendment_sha256: str
    final_envelope_sha256: str
    authorization_phrase_sha256: str

    def __post_init__(self) -> None:
        if self.experiment_id == "" or not self.experiment_id.endswith(".v4"):
            raise ProtocolError("OE-PPUR v4 launch authority identity drifted.")
        for role in (
            "workspace_plan_sha256",
            "authorization_amendment_sha256",
            "final_envelope_sha256",
            "authorization_phrase_sha256",
        ):
            object.__setattr__(
                self,
                role,
                require_sha256(getattr(self, role), role.replace("_", " ")),
            )


def validate_launch_authority(
    admission: SealedEnvelopeAdmission,
    authority: LaunchAuthority,
) -> LaunchAuthority:
    if (
        type(admission) is not SealedEnvelopeAdmission
        or type(authority) is not LaunchAuthority
        or authority.experiment_id != admission.config.experiment_id
        or authority.workspace_plan_sha256 != admission.workspace_plan_sha256
        or authority.authorization_amendment_sha256
        != admission.authorization_amendment_sha256
        or authority.final_envelope_sha256 != admission.final_envelope_sha256
    ):
        raise ProtocolError("OE-PPUR v4 separate launch authority is absent or drifted.")
    return authority


def _reject_operational_predecessor_paths(paths: Sequence[Path]) -> None:
    for path in paths:
        rendered = path.as_posix()
        if any(fragment in rendered for fragment in FORBIDDEN_OPERATIONAL_PATH_FRAGMENTS):
            raise ProtocolError("OE-PPUR v4 predecessor operational path detected.")


def assert_no_v3_authority_payload(payload: Mapping[str, object]) -> None:
    rendered = repr(dict(payload))
    if (
        AUTHORIZATION_AMENDMENT_ARTIFACT_ID not in rendered
        or "amendment_v3" in rendered
        or "output_" in rendered and "router_v3" in rendered
    ):
        raise ProtocolError("OE-PPUR v4 authority payload contains predecessor state.")


__all__ = (
    "LaunchAuthority",
    "SealedEnvelopeAdmission",
    "assert_no_v3_authority_payload",
    "validate_launch_authority",
)
