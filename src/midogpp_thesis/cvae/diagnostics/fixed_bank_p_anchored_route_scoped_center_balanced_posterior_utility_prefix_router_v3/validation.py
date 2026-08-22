"""Content-first validation for the exact terminal CBPUPR bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from .bundle import (
    REQUIRED_FILES,
    assert_closed_world,
    validate_content_index,
    validate_preterminal_content_index,
)
from .config import (
    load_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_config,
)
from .constants import (
    EXPECTED_OUTER_PLAN_COUNT,
    EXPECTED_PSEUDO_ROUTE_COUNT,
    EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT,
    EXPECTED_TOTAL_POSTERIOR_MODEL_FIT_COUNT,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
)
from .hashing import canonical_hash
from .preterminal_gate import (
    preterminal_validation_checks_payload,
    validate_preterminal_gate_artifacts,
)
from .protocol import FROZEN_PROTOCOL_HASH
from .terminal_access_journal import validate_terminal_label_access_journal
from .validation_origin import validate_physical_origin
from .validation_reports import (
    validate_final_attestation,
    validate_scientific_reports,
)
from .validation_storage import load_table, validate_npz_manifest
from .validation_topology import validate_exact_topology_and_lineage


def validate_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_bundle(
    root: str | Path,
    *,
    require_final: bool = True,
) -> dict[str, object]:
    path = Path(root).resolve()
    assert_closed_world(path, allow_incomplete=not require_final)
    config = load_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_config(
        path / "config.resolved.yaml"
    )
    if Path(config.artifact_root).resolve() != path:
        raise ProtocolError("CBPUPR validation config/output binding drifted.")
    content = validate_content_index(path)
    preterminal_content = validate_preterminal_content_index(path)
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
    method_metrics = load_table(path, "terminal_method_metrics")
    center_metrics = load_table(path, "terminal_center_contrasts")
    oracle_rows = load_table(path, "terminal_case_oracles")
    capability = read_json(path / "reports/label_capability_report.json")
    summary = read_json(path / "reports/diagnostic_summary.json")
    leakage = read_json(path / "reports/leakage_report.json")
    publication = read_json(path / "reports/publication_decision.json")
    runtime = read_json(path / "reports/runtime_summary.json")
    terminal = read_json(path / "manifests/terminal_evaluation_seal.json")
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
        or len(method_metrics) != 8
        or capability.get("pseudo_evaluation_route_count")
        != EXPECTED_PSEUDO_ROUTE_COUNT
        or capability.get("decision_count") != 4 * EXPECTED_OUTER_PLAN_COUNT
        or capability.get("aggregate_seal_complete") is not True
        or capability.get("terminal_opened") is not True
        or summary.get("target_posterior_model_fit_count")
        != EXPECTED_TOTAL_POSTERIOR_MODEL_FIT_COUNT
        or summary.get("pseudo_posterior_model_fit_count") != 0
        or summary.get("formal_claim_authorized") is not False
        or "selection_aware_center_sign_flip" not in summary
        or leakage.get("posterior_model_fit_count")
        != EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT
        or leakage.get("pseudo_support_reopen_or_refit_used") is not False
        or leakage.get("policy_replay_bias_used") is not False
        or publication.get("status") != PUBLICATION_STATUS
        or publication.get("terminal_decision") != TERMINAL_DECISION
        or runtime.get("target_posterior_model_fit_count")
        != EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT
        or runtime.get("pseudo_posterior_model_fit_count") != 0
        or terminal.get("raw_labels_persisted") is not False
    ):
        raise ProtocolError("CBPUPR scientific bundle contract drifted.")
    origin = validate_physical_origin(
        path,
        config=config,
        protocol=protocol,
        physical=physical,
        fingerprint_rows=fingerprints,
    )
    rebuilt = validate_exact_topology_and_lineage(
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
        method_metrics=method_metrics,
        center_metrics=center_metrics,
        oracle_rows=oracle_rows,
        summary=summary,
        leakage=leakage,
        terminal=terminal,
    )
    preterminal_checks = preterminal_validation_checks_payload(
        config_contract_hash=config.contract_hash,
        protocol_contract_hash=FROZEN_PROTOCOL_HASH,
        content_index_hash=str(preterminal_content["content_index_hash"]),
        outer_route_count=len(plans),
        target_posterior_model_fit_count=len(models),
        pseudo_posterior_reference_count=len(pseudo_references),
        preterminal_hash=rebuilt.preterminal_hash,
    )
    validate_preterminal_gate_artifacts(
        path, expected_checks=preterminal_checks
    )
    validate_terminal_label_access_journal(
        path, expected_checks=preterminal_checks
    )
    validate_scientific_reports(
        path,
        config=config,
        origin=origin,
        physical=physical,
        capability=capability,
        summary=summary,
        leakage=leakage,
        publication=publication,
        runtime=runtime,
        require_final=require_final,
    )
    check_payload = {
        "schema_version": "fixed_bank_cbpupr_validation_checks_v1",
        "status": "PASS",
        "config_contract_hash": config.contract_hash,
        "protocol_contract_hash": FROZEN_PROTOCOL_HASH,
        "content_index_hash": content["content_index_hash"],
        "required_file_count": len(REQUIRED_FILES),
        "outer_route_count": len(plans),
        "target_posterior_model_fit_count": len(models),
        "pseudo_posterior_model_fit_count": 0,
        "pseudo_posterior_reference_count": len(pseudo_references),
        "terminal_result_hash": terminal["terminal_result_hash"],
        "terminal_only_consumed_test": True,
        "formal_claim_authorized": False,
    }
    checks = {
        **check_payload,
        "validation_checks_hash": canonical_hash(check_payload),
    }
    if require_final:
        validate_final_attestation(path, expected_checks=checks)
    return checks


def verify_completed_attested_bundle(
    root: str | Path,
    *,
    expected_checks: Mapping[str, object],
) -> dict[str, object]:
    path = Path(root).resolve()
    observed = validate_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_bundle(
        path, require_final=True
    )
    report = read_json(path / "reports/validation_report.json")
    attestation = read_json(path / "reports/fresh_process_attestation.json")
    if (
        observed != dict(expected_checks)
        or report.get("checks") != observed
        or report.get("fresh_process_attestation_hash")
        != attestation.get("attestation_hash")
        or report.get("status") != "PASS"
    ):
        raise ProtocolError("CBPUPR final attested validation drifted.")
    return observed


__all__ = (
    "validate_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_bundle",
    "verify_completed_attested_bundle",
)
