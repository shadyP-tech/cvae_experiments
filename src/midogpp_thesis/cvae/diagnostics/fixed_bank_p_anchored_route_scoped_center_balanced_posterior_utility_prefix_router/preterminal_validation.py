"""Content-first validation while target terminal labels remain closed."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from .bundle import (
    assert_preterminal_closed_world,
    validate_preterminal_content_index,
)
from .config import (
    load_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_config,
)
from .constants import (
    EXPECTED_OUTER_PLAN_COUNT,
    EXPECTED_PSEUDO_ROUTE_COUNT,
    EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT,
)
from .preflight import load_validated_workstation_preflight
from .preterminal_gate import (
    preterminal_validation_checks_payload,
    validate_preterminal_gate_artifacts,
)
from .protocol import FROZEN_PROTOCOL_HASH
from .validation_origin import validate_physical_origin
from .validation_storage import load_table, validate_npz_manifest
from .validation_topology import validate_preterminal_topology_and_lineage


PRETERMINAL_VALIDATION_PHASE = (
    "PRETERMINAL_PARENT_AND_TWO_FRESH_PROCESS_VALIDATION"
)


def validate_preterminal_bundle(
    root: str | Path, *, require_attested: bool = False
) -> dict[str, object]:
    """Rebuild the complete decision surface without terminal artifacts."""

    path = Path(root).resolve()
    assert_preterminal_closed_world(
        path, phase="attested" if require_attested else "validation"
    )
    _validate_preterminal_run_state(path)
    config = load_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_config(
        path / "config.resolved.yaml"
    )
    if Path(config.artifact_root).resolve() != path:
        raise ProtocolError("CBPUPR preterminal config/output binding drifted.")
    load_validated_workstation_preflight(path, runtime=getattr(config, "runtime"))
    content = validate_preterminal_content_index(path)
    protocol = read_json(path / "manifests/protocol_manifest.json")
    physical = read_json(path / "manifests/physical_surface_seal.json")
    plans = load_table(path, "outer_plans")
    fingerprints = load_table(path, "physical_fingerprints")
    support = load_table(path, "route_support_capabilities")
    models = load_table(path, "target_local_posterior_models")
    posteriors = load_table(path, "target_local_posterior_predictions")
    pseudo_references = load_table(path, "pseudo_posterior_references")
    target_candidates = load_table(path, "target_candidate_policies")
    pseudo_candidates = load_table(path, "pseudo_candidate_policies")
    decisions = load_table(path, "route_decisions")
    composed = load_table(path, "composed_predictions")
    capability = read_json(
        path / "reports/preterminal_label_capability_report.json"
    )
    for manifest_name, array_name in (
        ("route_endpoint_probability_index", "route_endpoint_probabilities"),
        (
            "pseudo_route_endpoint_probability_index",
            "pseudo_route_endpoint_probabilities",
        ),
        (
            "target_local_posterior_probability_index",
            "target_local_posterior_probabilities",
        ),
        ("candidate_probability_index", "candidate_probabilities"),
        ("composed_probability_index", "composed_probabilities"),
    ):
        validate_npz_manifest(path, manifest_name, array_name)
    if (
        protocol.get("protocol_contract_hash") != FROZEN_PROTOCOL_HASH
        or physical.get("target_probability_cell_count") != 810
        or len(plans) != EXPECTED_OUTER_PLAN_COUNT
        or len(fingerprints) != 18
        or len(support) != EXPECTED_OUTER_PLAN_COUNT
        or len(models) != EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT
        or len(posteriors) != EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT
        or len(pseudo_references) != 2 * EXPECTED_PSEUDO_ROUTE_COUNT
        or len(target_candidates) != 2 * EXPECTED_OUTER_PLAN_COUNT
        or len(pseudo_candidates) != 2 * EXPECTED_PSEUDO_ROUTE_COUNT
        or len(decisions) != 18
        or len(composed) != 72
        or capability.get("pseudo_evaluation_route_count")
        != EXPECTED_PSEUDO_ROUTE_COUNT
        or capability.get("decision_count") != 4 * EXPECTED_OUTER_PLAN_COUNT
        or capability.get("aggregate_seal_complete") is not True
        or capability.get("terminal_opened") is not False
        or capability.get("raw_labels_persisted") is not False
    ):
        raise ProtocolError("CBPUPR preterminal scientific contract drifted.")
    origin = validate_physical_origin(
        path,
        config=config,
        protocol=protocol,
        physical=physical,
        fingerprint_rows=fingerprints,
    )
    rebuilt = validate_preterminal_topology_and_lineage(
        path,
        config=config,
        origin=origin,
        physical=physical,
        fingerprints=fingerprints,
        plans=plans,
        support=support,
        models=models,
        posteriors=posteriors,
        pseudo_references=pseudo_references,
        target_candidates=target_candidates,
        pseudo_candidates=pseudo_candidates,
        decisions=decisions,
        composed=composed,
        capability=capability,
    )
    checks = preterminal_validation_checks_payload(
        config_contract_hash=config.contract_hash,
        protocol_contract_hash=FROZEN_PROTOCOL_HASH,
        content_index_hash=str(content["content_index_hash"]),
        outer_route_count=len(plans),
        target_posterior_model_fit_count=len(models),
        pseudo_posterior_reference_count=len(pseudo_references),
        preterminal_hash=rebuilt.preterminal_hash,
    )
    if require_attested:
        validate_preterminal_gate_artifacts(path, expected_checks=checks)
    return checks


def verify_preterminal_attested_bundle(
    root: str | Path, *, expected_checks: Mapping[str, object]
) -> dict[str, object]:
    """Re-hash the closed bundle and gate just before label capability use.

    The parent and both fresh workers already performed full scientific
    reconstruction.  Repeating optimizer-free endpoint/posterior replay a
    fourth time here would add workstation cost without strengthening the
    independent-process claim.  This last edge therefore revalidates the exact
    inventory, every indexed byte, the agreed checks hash, both child results,
    the report, and the durable seal.
    """

    path = Path(root).resolve()
    expected = dict(expected_checks)
    assert_preterminal_closed_world(path, phase="attested")
    _validate_preterminal_run_state(path)
    content = validate_preterminal_content_index(path)
    if content.get("content_index_hash") != expected.get("content_index_hash"):
        raise ProtocolError("CBPUPR preterminal content/check binding drifted.")
    validate_preterminal_gate_artifacts(path, expected_checks=expected)
    return expected


def _validate_preterminal_run_state(root: Path) -> None:
    state = read_json(root / "reports/run_state.json")
    try:
        timestamp = datetime.fromisoformat(str(state.get("updated_at_utc")))
    except ValueError as exc:
        raise ProtocolError("CBPUPR preterminal run-state timestamp drifted.") from exc
    if (
        set(state)
        != {
            "schema_version",
            "status",
            "phase",
            "error",
            "error_class",
            "updated_at_utc",
            "cross_run_recovery_allowed",
            "terminal_recovery_allowed",
        }
        or state.get("schema_version") != "fixed_bank_cbpupr_run_state_v1"
        or state.get("status") != "RUNNING"
        or state.get("phase") != PRETERMINAL_VALIDATION_PHASE
        or state.get("error") is not None
        or state.get("error_class") is not None
        or timestamp.tzinfo is None
        or state.get("cross_run_recovery_allowed") is not False
        or state.get("terminal_recovery_allowed") is not False
    ):
        raise ProtocolError("CBPUPR preterminal run-state contract drifted.")


__all__ = (
    "PRETERMINAL_VALIDATION_PHASE",
    "validate_preterminal_bundle",
    "verify_preterminal_attested_bundle",
)
