"""Thin no-launch lifecycle coordinator for OE-PPUR v4."""

from __future__ import annotations

from ...protocol import ProtocolError
from .admission import LaunchAuthority, SealedEnvelopeAdmission, validate_launch_authority
from .config import RouterV4Config, build_planned_config
from .hashing import canonical_hash
from .identity import DIRECT_INPUT_ARTIFACT_IDS, EXPERIMENT_ID


def inspect_planned_router(config: RouterV4Config | None = None) -> dict[str, object]:
    planned = build_planned_config() if config is None else config
    if type(planned) is not RouterV4Config or planned != build_planned_config():
        raise ProtocolError("OE-PPUR v4 inspection requires the planned config.")
    body = {
        "schema_version": "oe_ppur_v4_workspace_sealed_successor_inspection_v1",
        "experiment_id": EXPERIMENT_ID,
        "authorization_state": planned.authorization_state,
        "execution_amendment_issued": False,
        "launch_authorized": False,
        "config_contract_hash": planned.contract_hash,
        "protocol_hash": planned.protocol_hash,
        "seven_input_contract_hash": planned.seven_input_contract_hash,
        "direct_input_count": 7,
        "direct_input_artifact_ids": list(DIRECT_INPUT_ARTIFACT_IDS),
        "workspace_seal_required": True,
        "pre_amendment_plan_required": True,
        "nfs_safe_in_place_commit_required": True,
        "v3_amendment_preservation_witness_only": True,
        "v3_operational_state_reuse": False,
        "filesystem_mutation_performed": False,
        "target_labels_opened": False,
        "experiment_launched": False,
    }
    return {**body, "inspection_hash": canonical_hash(body)}


def run_oe_ppur_v4(
    config: RouterV4Config,
    *,
    admission: SealedEnvelopeAdmission | None = None,
    launch_authority: LaunchAuthority | None = None,
) -> None:
    """Fail closed until the separately sealed launch edge is supplied.

    The current authorization explicitly covers implementation, preflight, and
    amendment sealing—not execution or topology activation.  Keeping this
    boundary inside the dedicated runner prevents the generic workspace path
    from turning a prepared amendment into an implicit launch.
    """

    if type(config) is not RouterV4Config or not config.execution_amendment_issued:
        raise ProtocolError(
            "OE-PPUR v4 execution is not authorized: the workspace-sealed "
            "amendment is absent."
        )
    if admission is None or launch_authority is None:
        raise ProtocolError(
            "OE-PPUR v4 launch is not authorized by the preparation amendment."
        )
    validate_launch_authority(admission, launch_authority)
    raise ProtocolError(
        "OE-PPUR v4 launch edge remains closed pending separate execution "
        "authorization and workstation topology validation."
    )


__all__ = ("inspect_planned_router", "run_oe_ppur_v4")
