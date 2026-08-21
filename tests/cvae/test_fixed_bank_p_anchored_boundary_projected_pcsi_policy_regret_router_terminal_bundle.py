from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router import (
    persistence,
    reports,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router.bundle import (
    CONTENT_INDEX_MEMBERS,
    REQUIRED_FILES,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router.constants import (
    CENTERS,
    DIRECTION_IDS,
    ENDPOINT_METHOD_IDS,
    PORTFOLIO_METHOD_ID,
    candidate_sources,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router.controls import (
    CONTROL_SPECS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router.donor_runtime import (
    DoubleExcludedDonorPriorProvenance,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router.hashing import (
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router.information_diagnostics import (
    _classify_signed_contribution,
    _spearman,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router.reports import (
    validate_transport_endpoint_lineage_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router.telemetry import (
    REQUIRED_PHASE_WORKLOAD_COUNTS,
    PhaseTelemetryRecorder,
    validate_phase_telemetry_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router.terminal_metrics import (
    score_methods,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router.validation import (
    RECONSTRUCTIVE_MEMBERS,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def test_bundle_inventory_contains_pcsi_parc_runtime_and_six_terminal_diagnostics() -> None:
    required = set(REQUIRED_FILES)
    assert set(RECONSTRUCTIVE_MEMBERS) <= set(CONTENT_INDEX_MEMBERS)
    assert {
        "tables/double_exclusion_plans.json",
        "tables/projected_utility_models.json",
        "tables/fresh_legacy_utility_models.json",
        "tables/transport_screens.json",
        "tables/policy_regret_replays.json",
        "tables/policy_authorizations.json",
        "tables/final_policy_predictions.json",
        "tables/terminal_projected_action_diagnostics.json",
        "tables/terminal_policy_regret_diagnostics.json",
        "tables/terminal_transport_diagnostics.json",
        "tables/terminal_selected_case_diagnostics.json",
        "reports/phase_telemetry.json",
    } <= required
    assert not {
        "tables/donor_veto_models.json",
        "tables/composed_predictions.json",
        "tables/information_gate.json",
    } & required


def test_control_menu_is_exactly_five_hash_bound_specs() -> None:
    payloads = [row.to_payload() for row in CONTROL_SPECS]
    assert len(payloads) == 5
    assert len({row["policy_id"] for row in payloads}) == 5
    assert all(row["control_hash"] == canonical_hash({
        key: value for key, value in row.items() if key != "control_hash"
    }) for row in payloads)
    assert sum(bool(row["uses_policy_regret"]) for row in payloads) == 2


def test_phase_telemetry_uses_unique_semantic_workload_count_names() -> None:
    recorder = PhaseTelemetryRecorder()
    for index, (name, value) in enumerate(REQUIRED_PHASE_WORKLOAD_COUNTS.items()):
        recorder.begin(f"phase-{index}")
        recorder.finish({name: value})
    payload = recorder.payload()
    assert validate_phase_telemetry_payload(
        payload, required_counts=REQUIRED_PHASE_WORKLOAD_COUNTS
    ) == payload

    poisoned = deepcopy(payload)
    poisoned["phases"][1]["workload_counts"]["input_artifact_count"] = 6
    unhashed = {key: value for key, value in poisoned.items() if key != "telemetry_hash"}
    poisoned["telemetry_hash"] = canonical_hash(unhashed)
    with pytest.raises(ProtocolError, match="reported twice"):
        validate_phase_telemetry_payload(
            poisoned, required_counts=REQUIRED_PHASE_WORKLOAD_COUNTS
        )


def test_changed_case_count_uses_exact_binary32_bytes_and_reports_active_centers() -> None:
    sample_ids = {
        center: {f"case-{center}": (f"{center}-positive", f"{center}-negative")}
        for center in CENTERS
    }
    labels = {
        (center, f"case-{center}", f"{center}-positive"): 1
        for center in CENTERS
    } | {
        (center, f"case-{center}", f"{center}-negative"): 0
        for center in CENTERS
    }
    portfolio = {
        center: {f"case-{center}": (float(np.float32(0.75)), float(np.float32(0.25)))}
        for center in CENTERS
    }
    same_bytes = {
        center: {
            f"case-{center}": (
                float(np.float32(0.75)) + 1.0e-16,
                float(np.float32(0.25)) - 1.0e-16,
            )
        }
        for center in CENTERS
    }
    probabilities = {method: portfolio for method in ENDPOINT_METHOD_IDS}
    probabilities["SAME_BYTES"] = same_bytes
    method_rows, _centers, _oracles, _metrics = score_methods(
        probabilities,
        sample_ids,
        labels,
        method_order=(PORTFOLIO_METHOD_ID, "SAME_BYTES"),
    )
    same = next(row for row in method_rows if row["method_id"] == "SAME_BYTES")
    assert same["route_count"] == 0
    assert same["active_center_count"] == 0

    changed = deepcopy(same_bytes)
    changed["0"]["case-0"] = (
        float(np.nextafter(np.float32(0.75), np.float32(1.0), dtype=np.float32)),
        float(np.float32(0.25)),
    )
    probabilities["CHANGED"] = changed
    method_rows, _centers, _oracles, _metrics = score_methods(
        probabilities,
        sample_ids,
        labels,
        method_order=(PORTFOLIO_METHOD_ID, "CHANGED"),
    )
    row = next(item for item in method_rows if item["method_id"] == "CHANGED")
    assert row["route_count"] == 1
    assert row["active_center_count"] == 1


def test_midrank_spearman_is_coordinatewise_and_tie_stable() -> None:
    assert _spearman((1.0, 2.0, 2.0, 4.0), (10.0, 20.0, 20.0, 40.0)) == pytest.approx(1.0)
    assert _spearman((1.0, 1.0, 1.0), (0.0, 1.0, 2.0)) == 0.0


def test_selected_case_helpfulness_uses_exact_zero_without_tolerance() -> None:
    assert _classify_signed_contribution(1.0e-300) == "helpful"
    assert _classify_signed_contribution(-1.0e-300) == "harmful"
    assert _classify_signed_contribution(0.0) == "neutral"


def test_double_excluded_prior_rejects_scope_and_hash_poison() -> None:
    outer, pseudo, donor = CENTERS[:3]
    sources = candidate_sources(donor)
    clean = DoubleExcludedDonorPriorProvenance.create(
        outer_target_center=outer,
        pseudo_target_center=pseudo,
        donor_center=donor,
        query_centers_by_source=tuple(
            (
                source,
                tuple(
                    center
                    for center in CENTERS
                    if center not in {outer, pseudo, donor, source}
                ),
            )
            for source in sources
        ),
        prior_values={
            (source, direction): 0.0
            for source in sources
            for direction in DIRECTION_IDS
        },
    )
    poisoned_scope = tuple(clean.query_centers_by_source)
    source, legal_queries = poisoned_scope[0]
    poisoned_scope = (
        (source, (*legal_queries, pseudo)),
        *poisoned_scope[1:],
    )
    with pytest.raises(ProtocolError, match="prior provenance drifted"):
        replace(clean, query_centers_by_source=poisoned_scope)
    with pytest.raises(ProtocolError, match="prior hash drifted"):
        replace(clean, prior_hash="0" * 64)


def test_transport_lineage_rejects_false_label_free_and_hash_poison() -> None:
    unhashed = {
        "schema_version": "fixed_bank_pcsi_parc_transport_endpoint_lineage_v2",
        "transport_semantics": (
            "support_conditioned_endpoint_reconstructed_P_B_I_R"
        ),
        "outer_target_center": "0",
        "endpoint_target_center": "1",
        "endpoint_support_scope": "endpoint_target_T_minus_held_case_c",
        "source_prior_scope": (
            "q_not_in_outer_H_or_endpoint_target_T_or_source_e"
        ),
        "endpoint_state_matrix_hash": "a" * 64,
        "source_prior_labels_used_upstream": True,
        "route_local_support_labels_used_upstream": True,
        "held_case_evaluation_capability_used_directly": False,
        "pseudo_evaluation_capability_used_directly": False,
        "terminal_evaluation_capability_used_directly": False,
        "label_free_claim": False,
        "uses_pre_equivalence_endpoint_crossing_rates": True,
        "identity_level_route_noninterference_required": True,
        "identity_level_route_noninterference_proven": False,
        "authorization_valid": False,
        "protocol_status": "BLOCKED_IDENTITY_LEVEL_ROUTE_FEEDBACK",
        "raw_labels_persisted": False,
    }
    clean = {**unhashed, "lineage_hash": canonical_hash(unhashed)}
    assert validate_transport_endpoint_lineage_payload(
        clean,
        outer_target_center="0",
        endpoint_target_center="1",
        endpoint_state_matrix_hash="a" * 64,
    ) == clean

    false_label_free = {**clean, "label_free_claim": True}
    false_label_free["lineage_hash"] = canonical_hash(
        {
            key: value
            for key, value in false_label_free.items()
            if key != "lineage_hash"
        }
    )
    with pytest.raises(ProtocolError, match="transport endpoint lineage drifted"):
        validate_transport_endpoint_lineage_payload(
            false_label_free,
            outer_target_center="0",
            endpoint_target_center="1",
            endpoint_state_matrix_hash="a" * 64,
        )

    tampered_hash = {**clean, "lineage_hash": "0" * 64}
    with pytest.raises(ProtocolError, match="transport endpoint lineage drifted"):
        validate_transport_endpoint_lineage_payload(
            tampered_hash,
            outer_target_center="0",
            endpoint_target_center="1",
            endpoint_state_matrix_hash="a" * 64,
        )


def test_transport_authorization_cannot_persist_without_poison_invariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        reports,
        "validate_transport_lineage_evidence",
        lambda _preterminal: {"transport_authorization_valid": False},
    )
    with pytest.raises(ProtocolError, match="Canonical persistence is prohibited"):
        reports.assert_transport_authorization_lineage_valid(object())


def test_preterminal_persistence_fails_before_writing_blocked_transport(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def blocked(_preterminal: object) -> None:
        raise ProtocolError("BLOCKED_IDENTITY_FEEDBACK")

    monkeypatch.setattr(
        persistence,
        "assert_transport_authorization_lineage_valid",
        blocked,
    )
    with pytest.raises(ProtocolError, match="BLOCKED_IDENTITY_FEEDBACK"):
        persistence.persist_preterminal(tmp_path, object())
    assert not tuple(tmp_path.rglob("*"))
