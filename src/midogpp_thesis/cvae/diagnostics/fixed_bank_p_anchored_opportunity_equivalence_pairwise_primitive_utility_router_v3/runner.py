"""Thin path-free planned runner for the OE-PPUR v3 successor."""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

from ...protocol import ProtocolError
from .config import RouterV3Config, build_planned_config, validate_planned_config
from .execution.inputs import build_planned_seven_input_contract
from .hashing import canonical_hash
from .source_seal import build_source_seal
from .workstation import workstation_plan_payload


def inspect_planned_router(
    config: RouterV3Config | None = None,
) -> dict[str, object]:
    """Inspect source and contracts without resolving inputs or probing hardware."""

    planned = validate_planned_config(
        build_planned_config() if config is None else config
    )
    inputs = build_planned_seven_input_contract()
    source = build_source_seal()
    body = {
        "schema_version": "oe_ppur_v3_implementation_inspection_v1",
        "experiment_id": planned.experiment_id,
        "authorization_state": planned.authorization_state,
        "execution_authorized": False,
        "config_contract_hash": planned.contract_hash,
        "protocol_hash": planned.protocol_hash,
        "seven_input_contract_hash": inputs.receipt_hash,
        "direct_input_count": inputs.input_count,
        "direct_input_artifact_ids": list(planned.direct_input_artifact_ids),
        "source_supervision_direct_input_ordinal": 3,
        "source_supervision_content_hash_known": False,
        "authorization_amendment_input_ordinal": 7,
        "authorization_amendment_issued": False,
        "current_source_seal_hash": source.combined_source_sha256,
        "current_source_seal_receipt_hash": source.receipt_hash,
        "nominal_scientific_service_implemented": True,
        "end_to_end_scientific_execution_implemented": False,
        "canonical_terminal_aggregate_evaluator_implemented": True,
        "physical_terminal_manifest_factory_open": False,
        "workstation_plan": workstation_plan_payload(),
        "artifact_paths_present": False,
        "hardware_probed": False,
        "filesystem_mutation_performed": False,
        "authorization_consumed": False,
        "labels_opened": False,
        "experiment_launched": False,
    }
    return {**body, "inspection_hash": canonical_hash(body)}


def run_oe_ppur_v3(
    config: RouterV3Config,
    *,
    artifact_root: str | Path,
    scratch_root: str | Path,
) -> NoReturn:
    """Reject planned execution before source, path, hardware, or service access."""

    planned = validate_planned_config(config)
    if (
        planned.execution_authorized is not False
        or planned.authorization_amendment_sha256 is not None
    ):
        raise ProtocolError("OE-PPUR v3 planned runner identity drifted.")
    # Deliberately do not coerce or inspect either requested root in the
    # planned state.  The future authorized branch owns their admission.
    _ = artifact_root, scratch_root
    raise ProtocolError(
        "OE-PPUR v3 execution is not authorized: direct input #7 is absent "
        "and the source-supervision artifact has not been admitted."
    )


run_opportunity_equivalence_pairwise_primitive_utility_router_v3 = run_oe_ppur_v3


__all__ = (
    "inspect_planned_router",
    "run_oe_ppur_v3",
    "run_opportunity_equivalence_pairwise_primitive_utility_router_v3",
)
