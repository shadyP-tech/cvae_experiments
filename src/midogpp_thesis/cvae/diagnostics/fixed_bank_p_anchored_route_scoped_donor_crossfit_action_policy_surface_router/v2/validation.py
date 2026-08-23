"""Fresh-process semantic validation for durable P-DCAPS v2 artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ....expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ....protocol import ProtocolError
from ....runtime.artifact_io import read_json
from ..action_surface import probability_sha256
from ..identity import (
    ACTION_STRATA,
    METHOD_MENU,
    PRIMARY_METHOD_ID,
    PUBLICATION_STATUS,
    P_METHOD_ID,
    TERMINAL_DECISION,
    canonical_hash,
    require_sha256,
)
from ..inventory import (
    CANONICAL_CASE_COUNT,
    CANONICAL_ROW_COUNT,
    ExpectedRouteInventory,
)
from ..persistence.arrays import load_dense_arrays
from ..persistence.safety import reject_forbidden_persisted_values
from ..lifecycle import DurablePreterminalAttestation
from ..preterminal import PreterminalOutputHashes
from ..seals import verify_phase_seal
from ..target_local_runtime import POSTERIOR_CONTROL_IDS
from .bundle import verify_closed_world_index, verify_index_payload_members
from .config import (
    load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v2_config,
)
from .input_contracts import build_source_snapshot_payload
from .inputs import load_label_free_test_frame
from .lineage import build_six_input_binding
from .persistence import (
    COMPOSED_ARRAY_MEMBER,
    FINAL_INDEXED_MEMBERS,
    PRETERMINAL_SCIENCE_MEMBER,
    PRETERMINAL_REQUIRED_MEMBERS,
    PRETERMINAL_ATTESTATION_MEMBER,
    TERMINAL_RESULT_MEMBER,
)
from .protocol import frozen_protocol_payload
from .reports import FINAL_REPORT_MEMBERS, validate_final_report_payloads
from .terminal.inference import exact_shared_center_max_sign_flip
from .validation_records import validate_persisted_preterminal_records
from .validation_terminal import validate_terminal_row_inventory
from .workspace_inputs import validate_workspace_provenance


_TERMINAL_RESULT_KEYS = {
    "schema_version",
    "method_rows",
    "center_rows",
    "case_diagnostic_rows",
    "selection_control",
    "router_diagnostics",
    "preterminal_seal_hash",
    "label_identity_hash",
    "publication_status",
    "terminal_decision",
    "raw_labels_persisted",
    "routing_authorized",
    "promotion_allowed",
    "result_hash",
}
_ROUTER_DIAGNOSTIC_KEYS = {
    "schema_version",
    "action_expected_vs_realized_midrank_spearman_by_stratum",
    "policy_expected_vs_realized_midrank_spearman",
    "policy_expected_vs_realized_pair_count",
    "routed_center_count",
    "joint_safe_routed_center_count",
    "joint_safe_routed_policy_rate",
    "normalized_endpoint_oracle_gap_definition",
    "normalized_endpoint_oracle_gap_defined_case_count",
    "mean_normalized_endpoint_oracle_gap",
    "primary_case_harm_count",
    "primary_case_harm_rate",
    "center_action_frequencies",
    "terminal_labels_changed_preterminal_decisions",
    "nonzero_route_count_is_not_success",
    "descriptive_only",
    "formal_claim_authorized",
}


def validate_preterminal_bundle(root: Path) -> dict[str, object]:
    return _validate_preterminal_science(Path(root), verify_preterminal_index=True)


def verify_durable_preterminal_attestation(
    root: Path,
    expected: DurablePreterminalAttestation,
) -> dict[str, object]:
    """Revalidate the durable barrier immediately before terminal access."""

    path = Path(root)
    preterminal = validate_preterminal_bundle(path)
    payload = read_json(path / PRETERMINAL_ATTESTATION_MEMBER)
    base = {key: value for key, value in payload.items() if key != "attestation_hash"}
    if (
        not isinstance(expected, DurablePreterminalAttestation)
        or payload != expected.to_payload()
        or payload.get("attestation_hash") != canonical_hash(base)
        or payload.get("preterminal_seal_hash")
        != preterminal["preterminal_seal_hash"]
        or payload.get("durable_bundle_hash")
        != preterminal["content_index_hash"]
        or payload.get("validator_count") != 2
        or payload.get("fresh_processes_required") is not True
        or payload.get("target_labels_opened") is not False
    ):
        raise ProtocolError("P-DCAPS v2 durable preterminal barrier drifted.")
    return {
        "status": "PASS",
        "preterminal_seal_hash": preterminal["preterminal_seal_hash"],
        "preterminal_content_index_hash": preterminal["content_index_hash"],
        "durable_attestation_hash": expected.attestation_hash,
        "terminal_label_access_authorized": True,
        "target_labels_opened": False,
    }


def _validate_preterminal_science(
    path: Path, *, verify_preterminal_index: bool
) -> dict[str, object]:
    index = (
        verify_closed_world_index(
            path,
            phase="preterminal",
            expected_members=PRETERMINAL_REQUIRED_MEMBERS,
        )
        if verify_preterminal_index
        else read_json(path / "manifests/preterminal_content_index.json")
    )
    if not verify_preterminal_index:
        verify_index_payload_members(
            path,
            index,
            phase="preterminal",
            expected_members=PRETERMINAL_REQUIRED_MEMBERS,
        )
    science = read_json(path / PRETERMINAL_SCIENCE_MEMBER)
    reject_forbidden_persisted_values(science)
    base = {key: value for key, value in science.items() if key != "bundle_hash"}
    if (
        science.get("schema_version")
        != "pdcaps_v2_preterminal_science_bundle_v1"
        or science.get("bundle_hash") != canonical_hash(base)
        or science.get("protocol") != frozen_protocol_payload()
        or science.get("target_labels_opened") is not False
        or science.get("raw_labels_persisted") is not False
        or science.get("routing_authorized") is not False
        or science.get("promotion_allowed") is not False
    ):
        raise ProtocolError("P-DCAPS v2 preterminal science bundle drifted.")
    config = (
        load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v2_config(
            path / "config.resolved.yaml"
        )
    )
    provenance = validate_workspace_provenance(path, config)
    binding = build_six_input_binding(config, provenance)
    source_snapshot = build_source_snapshot_payload()
    if (
        config.artifact_root.resolve() != path.resolve()
        or science.get("config_hash") != config.config_hash
        or science.get("input_binding") != binding.to_payload()
        or science.get("source_snapshot") != source_snapshot
        or source_snapshot.get("manifest_sha256")
        != config.expected_source_snapshot_manifest_sha256
        or source_snapshot.get("tree_sha256")
        != config.expected_source_snapshot_tree_sha256
        or source_snapshot.get("member_count")
        != config.expected_source_snapshot_member_count
    ):
        raise ProtocolError("P-DCAPS v2 durable config or lineage drifted.")
    frame = load_label_free_test_frame(config)
    canonical_inventory = ExpectedRouteInventory.from_label_free_keys(
        tuple(
            (row.center, row.case_id, row.sample_id)
            for row in frame.rows
        ),
        manifest_sha256=config.expected_manifest_sha256,
        row_order_hash=config.expected_test_cache_row_order_hash,
    )
    canonical_case_ids_by_center = {
        center: [
            case.case_id
            for case in canonical_inventory.cases
            if case.center == center
        ]
        for center in CENTERS
    }
    canonical_case_sample_counts_by_center = {
        center: {
            case.case_id: len(case.sample_ids)
            for case in canonical_inventory.cases
            if case.center == center
        }
        for center in CENTERS
    }
    canonical_sample_ids_by_center = {
        center: tuple(row.sample_id for row in frame.rows_by_center[center])
        for center in CENTERS
    }
    arrays, array_manifest = load_dense_arrays(path / COMPOSED_ARRAY_MEMBER)
    if science.get("composed_probability_manifest_hash") != array_manifest.get(
        "manifest_hash"
    ):
        raise ProtocolError("P-DCAPS v2 composed array lineage drifted.")

    output_payload = _mapping(science, "preterminal_output_hashes")
    output = PreterminalOutputHashes(
        str(output_payload["action_surface_set_seal_hash"]),
        tuple(tuple(row) for row in output_payload["action_surface_seals"]),
        str(output_payload["expected_inventory_hash"]),
        tuple(tuple(row) for row in output_payload["control_result_hashes"]),
        tuple(tuple(row) for row in output_payload["legacy_control_seal_hashes"]),
        tuple(tuple(row) for row in output_payload["method_decision_hashes"]),
        tuple(tuple(row) for row in output_payload["method_composition_hashes"]),
    )
    if output.output_bundle_hash != output_payload.get("output_bundle_hash"):
        raise ProtocolError("P-DCAPS v2 preterminal output reconstruction drifted.")
    record_checks = validate_persisted_preterminal_records(science, output)
    if (
        output.centers != CENTERS
        or output.expected_inventory_hash != canonical_inventory.inventory_hash
        or canonical_inventory.case_count != CANONICAL_CASE_COUNT
        or canonical_inventory.row_count != CANONICAL_ROW_COUNT
        or record_checks.get("identity_result_count") != len(CENTERS)
        or record_checks.get("cyclic_result_count") != len(CENTERS)
        or record_checks.get("legacy_control_count")
        != len(POSTERIOR_CONTROL_IDS) * len(CENTERS)
        or record_checks.get("method_decision_count")
        != len(CENTERS) * len(METHOD_MENU)
        or record_checks.get("target_decision_counts_by_control")
        != {
            control: CANONICAL_CASE_COUNT for control in POSTERIOR_CONTROL_IDS
        }
        or record_checks.get("pseudo_decision_counts_by_control")
        != {
            control: canonical_inventory.pseudo_route_count
            for control in POSTERIOR_CONTROL_IDS
        }
        or record_checks.get("case_ids_by_center")
        != canonical_case_ids_by_center
    ):
        raise ProtocolError("P-DCAPS v2 canonical preterminal rectangle drifted.")
    preterminal_seal = _mapping(science, "preterminal_seal")
    seal_hash = verify_phase_seal(preterminal_seal)
    if (
        preterminal_seal.get("phase") != "PRETERMINAL"
        or preterminal_seal.get("target_labels_opened") is not False
        or output.output_bundle_hash not in preterminal_seal.get("row_hashes", ())
    ):
        raise ProtocolError("P-DCAPS v2 preterminal phase seal drifted.")

    surface_set = _mapping(science, "surface_set")
    expected_surface_set_hash = canonical_hash(
        {
            "schema_version": "pdcaps_action_surface_set_v1",
            "expected_inventory_hash": surface_set["expected_inventory_hash"],
            "physical_surface_hash": surface_set["physical_surface_hash"],
            "control_surface_seals": tuple(
                tuple(row) for row in surface_set["control_surface_seals"]
            ),
            "route_inventory_seal_hashes": tuple(
                tuple(row) for row in surface_set["route_inventory_seal_hashes"]
            ),
            "pseudo_labels_used": False,
            "target_labels_used": False,
        }
    )
    if (
        surface_set.get("surface_set_seal_hash") != expected_surface_set_hash
        or expected_surface_set_hash != output.action_surface_set_seal_hash
        or surface_set.get("expected_inventory_hash")
        != canonical_inventory.inventory_hash
        or tuple(row[0] for row in surface_set["control_surface_seals"])
        != POSTERIOR_CONTROL_IDS
    ):
        raise ProtocolError("P-DCAPS v2 joint surface-set seal drifted.")
    _validate_preterminal_lifecycle_audit(
        science,
        config=config,
        canonical_inventory=canonical_inventory,
        surface_set=surface_set,
        preterminal_seal_hash=seal_hash,
    )

    decisions = _mapping_rows(science, "method_decisions")
    compositions = _mapping_rows(science, "method_compositions")
    expected_keys = tuple(
        (center, method) for center in output.centers for method in METHOD_MENU
    )
    decision_by_key = {
        (str(row["outer_center"]), str(row["method_id"])): row for row in decisions
    }
    composition_by_key = {
        (
            str(_mapping(row, "decision")["outer_center"]),
            str(_mapping(row, "decision")["method_id"]),
        ): row
        for row in compositions
    }
    if (
        tuple(decision_by_key) != expected_keys
        or tuple(composition_by_key) != expected_keys
        or len(arrays) != len(expected_keys)
    ):
        raise ProtocolError("P-DCAPS v2 fixed method rectangle drifted.")
    p_array_by_center = {
        center: arrays[_array_key(center, P_METHOD_ID)] for center in output.centers
    }
    for center, method in expected_keys:
        decision = decision_by_key[(center, method)]
        composition = composition_by_key[(center, method)]
        nested_decision = _mapping(composition, "decision")
        prediction = _mapping(composition, "prediction")
        array = arrays.get(_array_key(center, method))
        if (
            array is None
            or array.dtype != np.float32
            or array.shape != (len(canonical_sample_ids_by_center[center]),)
            or tuple(prediction.get("sample_ids", ()))
            != canonical_sample_ids_by_center[center]
            or probability_sha256(array) != prediction.get("probability_hash")
            or nested_decision.get("decision_hash") != decision.get("decision_hash")
            or composition.get("method_composition_hash")
            != _composition_hash(composition)
            or prediction.get("composition_hash") != _prediction_hash(prediction)
        ):
            raise ProtocolError("P-DCAPS v2 composition semantics drifted.")
        if bool(decision.get("exact_p_fallback")) and not np.array_equal(
            array, p_array_by_center[center]
        ):
            raise ProtocolError("P-DCAPS v2 exact-P fallback changed bytes.")

    expected_decision_hashes = {
        (center, method): digest
        for center, method, digest in output.method_decision_hashes
    }
    expected_composition_hashes = {
        (center, method): digest
        for center, method, digest in output.method_composition_hashes
    }
    if any(
        decision_by_key[key].get("decision_hash") != expected_decision_hashes[key]
        or composition_by_key[key].get("method_composition_hash")
        != expected_composition_hashes[key]
        for key in expected_keys
    ):
        raise ProtocolError("P-DCAPS v2 preterminal hash table drifted.")
    primary_routed_centers = [
        center
        for center in CENTERS
        if bool(
            composition_by_key[(center, PRIMARY_METHOD_ID)].get(
                "selection_enabled"
            )
        )
    ]
    primary_selected_action_count = sum(
        len(
            _mapping(
                composition_by_key[(center, PRIMARY_METHOD_ID)], "decision"
            ).get("selected_action_hashes", ())
        )
        for center in primary_routed_centers
    )
    return {
        "validation_phase": "preterminal",
        "status": "PASS",
        "content_index_hash": index["content_index_hash"],
        "preterminal_science_bundle_hash": science["bundle_hash"],
        "preterminal_output_bundle_hash": output.output_bundle_hash,
        "preterminal_seal_hash": seal_hash,
        "surface_set_seal_hash": expected_surface_set_hash,
        "center_count": len(output.centers),
        "method_composition_count": len(compositions),
        **record_checks,
        "case_sample_counts_by_center": canonical_case_sample_counts_by_center,
        "primary_routed_centers": primary_routed_centers,
        "primary_selected_action_count": primary_selected_action_count,
        "semantic_reconstruction_without_refit": True,
        "target_labels_opened": False,
        "raw_labels_persisted": False,
    }


def validate_final_bundle(root: Path) -> dict[str, object]:
    path = Path(root)
    preterminal = _validate_preterminal_science(
        path, verify_preterminal_index=False
    )
    index = verify_closed_world_index(
        path,
        phase="final",
        expected_members=FINAL_INDEXED_MEMBERS,
    )
    attestation = read_json(path / PRETERMINAL_ATTESTATION_MEMBER)
    attestation_base = {
        key: value for key, value in attestation.items() if key != "attestation_hash"
    }
    validator_pids = attestation.get("validator_process_ids")
    validator_hashes = attestation.get("validator_result_hashes")
    if (
        attestation.get("schema_version")
        != "pdcaps_durable_preterminal_attestation_v1"
        or attestation.get("attestation_hash") != canonical_hash(attestation_base)
        or attestation.get("preterminal_seal_hash")
        != preterminal["preterminal_seal_hash"]
        or not isinstance(validator_pids, list)
        or len(validator_pids) != 2
        or len(set(validator_pids)) != 2
        or not isinstance(validator_hashes, list)
        or len(validator_hashes) != 2
        or len(set(validator_hashes)) != 2
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in validator_hashes
        )
        or attestation.get("durable_bundle_hash")
        != preterminal["content_index_hash"]
        or attestation.get("validator_count") != 2
        or attestation.get("fresh_processes_required") is not True
        or attestation.get("target_labels_opened") is not False
    ):
        raise ProtocolError("P-DCAPS v2 durable attestation drifted.")
    result = read_json(path / TERMINAL_RESULT_MEMBER)
    reject_forbidden_persisted_values(result)
    result_hash = _terminal_result_hash(result)
    method_rows = result.get("method_rows")
    center_rows = result.get("center_rows")
    case_rows = result.get("case_diagnostic_rows")
    selection = result.get("selection_control")
    router_diagnostics = result.get("router_diagnostics")
    if (
        set(result) != _TERMINAL_RESULT_KEYS
        or result.get("schema_version")
        != "pdcaps_v2_terminal_evaluation_result_v1"
        or result.get("result_hash") != result_hash
        or result.get("preterminal_seal_hash")
        != preterminal["preterminal_seal_hash"]
        or not isinstance(method_rows, list)
        or len(method_rows) != len(METHOD_MENU)
        or not isinstance(center_rows, list)
        or len(center_rows) != 9 * len(METHOD_MENU)
        or not isinstance(case_rows, list)
        or not isinstance(selection, dict)
        or not isinstance(router_diagnostics, dict)
        or selection.get("null_replicate_count") != 512
        or result.get("raw_labels_persisted") is not False
        or result.get("routing_authorized") is not False
        or result.get("promotion_allowed") is not False
    ):
        raise ProtocolError("P-DCAPS v2 final terminal bundle drifted.")
    row_inventory = validate_terminal_row_inventory(
        method_rows=method_rows,
        center_rows=center_rows,
        case_rows=case_rows,
        case_ids_by_center=preterminal["case_ids_by_center"],
        case_sample_counts_by_center=preterminal[
            "case_sample_counts_by_center"
        ],
    )
    _validate_terminal_claim_boundary(
        result,
        center_rows=center_rows,
        case_rows=case_rows,
        primary_routed_centers=preterminal["primary_routed_centers"],
        primary_selected_action_count=preterminal[
            "primary_selected_action_count"
        ],
    )
    report_hashes = _validate_final_reports(
        path,
        terminal_result_hash=result_hash,
        terminal_result_payload=result,
    )
    return {
        "status": "PASS",
        "content_index_hash": index["content_index_hash"],
        "terminal_result_hash": result_hash,
        "preterminal_seal_hash": preterminal["preterminal_seal_hash"],
        **row_inventory,
        "sign_flip_replicate_count": 512,
        "final_report_set_hash": canonical_hash(report_hashes),
        "semantic_reconstruction_without_refit": True,
        "raw_labels_persisted": False,
        "formal_claim_authorized": False,
    }


def _validate_final_reports(
    path: Path,
    *,
    terminal_result_hash: str,
    terminal_result_payload: dict[str, object],
) -> dict[str, str]:
    reports: dict[str, dict[str, object]] = {}
    for member in FINAL_REPORT_MEMBERS:
        report = read_json(path / member)
        reject_forbidden_persisted_values(report)
        reports[member] = report
    return validate_final_report_payloads(
        reports,
        terminal_result_hash=terminal_result_hash,
        terminal_result_payload=terminal_result_payload,
    )


def _validate_preterminal_lifecycle_audit(
    science: dict[str, object],
    *,
    config: object,
    canonical_inventory: ExpectedRouteInventory,
    surface_set: dict[str, object],
    preterminal_seal_hash: str,
) -> None:
    """Authenticate the persisted label lifecycle at its durable boundary."""

    lifecycle = _mapping(science, "lifecycle_audit")
    lifecycle_base = {
        key: value for key, value in lifecycle.items() if key != "lifecycle_hash"
    }
    control_seals = surface_set.get("control_surface_seals")
    identity_surface_seal = (
        control_seals[0][1]
        if isinstance(control_seals, list)
        and control_seals
        and isinstance(control_seals[0], list)
        and len(control_seals[0]) == 2
        else None
    )
    expected_keys = {
        "schema_version",
        "phase",
        "protocol_hash",
        "expected_outer_centers",
        "expected_inventory",
        "action_surface_set",
        "action_surface_seal_hash",
        "pseudo_response_surface_count",
        "preterminal_seal_hash",
        "terminal_centers_opened",
        "firewall_hash",
        "target_labels_can_change_preterminal_decisions",
        "support_class_count_scope_count",
        "response_denominators_derived_inside_label_lifecycle",
        "durable_terminal_attestation_required",
        "durable_preterminal_attestation_hash",
        "publication_status",
        "terminal_decision",
        "raw_labels_persisted",
        "lifecycle_hash",
    }
    if (
        set(lifecycle) != expected_keys
        or lifecycle.get("schema_version") != "pdcaps_label_lifecycle_v2"
        or lifecycle.get("lifecycle_hash") != canonical_hash(lifecycle_base)
        or lifecycle.get("phase") != "PRETERMINAL_ATTESTED"
        or lifecycle.get("protocol_hash")
        != getattr(config, "protocol")["protocol_hash"]
        or lifecycle.get("expected_outer_centers")
        != list(canonical_inventory.centers)
        or lifecycle.get("expected_inventory")
        != canonical_inventory.to_payload()
        or lifecycle.get("action_surface_set") != surface_set
        or lifecycle.get("action_surface_seal_hash") != identity_surface_seal
        or lifecycle.get("pseudo_response_surface_count")
        != len(POSTERIOR_CONTROL_IDS) * canonical_inventory.pseudo_route_count
        or lifecycle.get("preterminal_seal_hash") != preterminal_seal_hash
        or lifecycle.get("terminal_centers_opened") != []
        or require_sha256(
            lifecycle.get("firewall_hash"), "persisted label firewall"
        )
        != lifecycle.get("firewall_hash")
        or lifecycle.get("target_labels_can_change_preterminal_decisions")
        is not False
        or lifecycle.get("support_class_count_scope_count")
        != canonical_inventory.case_count
        or lifecycle.get(
            "response_denominators_derived_inside_label_lifecycle"
        )
        is not True
        or lifecycle.get("durable_terminal_attestation_required") is not True
        or lifecycle.get("durable_preterminal_attestation_hash") is not None
        or lifecycle.get("publication_status") != PUBLICATION_STATUS
        or lifecycle.get("terminal_decision") != TERMINAL_DECISION
        or lifecycle.get("raw_labels_persisted") is not False
    ):
        raise ProtocolError("P-DCAPS v2 persisted label lifecycle drifted.")


def _validate_terminal_claim_boundary(
    payload: dict[str, object],
    *,
    center_rows: list[dict[str, object]],
    case_rows: list[dict[str, object]],
    primary_routed_centers: object,
    primary_selected_action_count: object,
) -> None:
    """Reject any terminal record that broadens the consumed-test claim."""

    selection = payload.get("selection_control")
    diagnostics = payload.get("router_diagnostics")
    center_metrics: dict[str, dict[str, dict[str, object]]] = {
        method: {} for method in METHOD_MENU
    }
    for row in center_rows:
        center_metrics[str(row.get("method_id"))][
            str(row.get("target_center"))
        ] = row
    reconstructed_selection = exact_shared_center_max_sign_flip(center_metrics)
    primary_cases = [
        row for row in case_rows if row.get("method_id") == PRIMARY_METHOD_ID
    ]
    primary_harm_count = sum(
        int(bool(row.get("case_harmed_vs_P"))) for row in primary_cases
    )
    if (
        not isinstance(primary_routed_centers, (list, tuple))
        or any(
            not isinstance(center, str) or center not in CENTERS
            for center in primary_routed_centers
        )
        or tuple(primary_routed_centers)
        != tuple(
            center for center in CENTERS if center in set(primary_routed_centers)
        )
        or not isinstance(primary_selected_action_count, int)
        or isinstance(primary_selected_action_count, bool)
        or primary_selected_action_count < 0
    ):
        raise ProtocolError("P-DCAPS v2 terminal claim boundary drifted.")
    routed_center_ids = tuple(str(center) for center in primary_routed_centers)
    selected_action_count = int(primary_selected_action_count)
    primary_center_rows = center_metrics[PRIMARY_METHOD_ID]
    expected_jointly_safe = sum(
        float(primary_center_rows[center]["center_bacc_delta_vs_P"]) > 0.0
        and float(primary_center_rows[center]["center_brier_delta_vs_P"]) <= 0.0
        and float(primary_center_rows[center]["center_log_loss_delta_vs_P"]) <= 0.0
        for center in routed_center_ids
    )
    frequencies = (
        diagnostics.get("center_action_frequencies")
        if isinstance(diagnostics, dict)
        else None
    )
    action_correlations = (
        diagnostics.get(
            "action_expected_vs_realized_midrank_spearman_by_stratum"
        )
        if isinstance(diagnostics, dict)
        else None
    )
    policy_correlation = (
        diagnostics.get("policy_expected_vs_realized_midrank_spearman")
        if isinstance(diagnostics, dict)
        else None
    )
    oracle_gap_count = (
        diagnostics.get("normalized_endpoint_oracle_gap_defined_case_count")
        if isinstance(diagnostics, dict)
        else None
    )
    oracle_gap_mean = (
        diagnostics.get("mean_normalized_endpoint_oracle_gap")
        if isinstance(diagnostics, dict)
        else None
    )
    routed = (
        diagnostics.get("routed_center_count")
        if isinstance(diagnostics, dict)
        else None
    )
    jointly_safe = (
        diagnostics.get("joint_safe_routed_center_count")
        if isinstance(diagnostics, dict)
        else None
    )
    expected_safe_rate = (
        None
        if routed == 0
        else float(jointly_safe) / float(routed)
        if isinstance(routed, int)
        and not isinstance(routed, bool)
        and isinstance(jointly_safe, int)
        and not isinstance(jointly_safe, bool)
        and routed > 0
        else object()
    )
    if (
        payload.get("publication_status") != PUBLICATION_STATUS
        or payload.get("terminal_decision") != TERMINAL_DECISION
        or not isinstance(selection, dict)
        or selection != reconstructed_selection
        or selection.get("schema_version")
        != "pdcaps_v2_shared_center_max_sign_flip_v1"
        or selection.get("descriptive_only") is not True
        or selection.get("formal_claim_authorized") is not False
        or selection.get("nominal_significance_claimed") is not False
        or selection.get("route_pipeline_refit_inside_null_replicate") is not False
        or not isinstance(diagnostics, dict)
        or diagnostics.get("schema_version")
        != "pdcaps_v2_terminal_router_diagnostics_v1"
        or set(diagnostics) != _ROUTER_DIAGNOSTIC_KEYS
        or diagnostics.get("terminal_labels_changed_preterminal_decisions")
        is not False
        or diagnostics.get("nonzero_route_count_is_not_success") is not True
        or diagnostics.get("descriptive_only") is not True
        or diagnostics.get("formal_claim_authorized") is not False
        or not isinstance(routed, int)
        or isinstance(routed, bool)
        or not isinstance(jointly_safe, int)
        or isinstance(jointly_safe, bool)
        or routed != len(routed_center_ids)
        or jointly_safe != expected_jointly_safe
        or diagnostics.get("joint_safe_routed_policy_rate") != expected_safe_rate
        or diagnostics.get("policy_expected_vs_realized_pair_count")
        != len(CENTERS)
        or not _optional_bounded_float(policy_correlation, lower=-1.0, upper=1.0)
        or not isinstance(action_correlations, list)
        or tuple(
            (row.get("family"), row.get("direction"))
            for row in action_correlations
            if isinstance(row, dict)
        )
        != ACTION_STRATA
        or any(
            set(row) != {"family", "direction", "pair_count", "midrank_spearman"}
            or not isinstance(row.get("pair_count"), int)
            or isinstance(row.get("pair_count"), bool)
            or not (0 <= int(row["pair_count"]) <= CANONICAL_CASE_COUNT)
            or not _optional_bounded_float(
                row.get("midrank_spearman"), lower=-1.0, upper=1.0
            )
            for row in action_correlations
            if isinstance(row, dict)
        )
        or sum(int(row["pair_count"]) for row in action_correlations)
        > CANONICAL_CASE_COUNT
        or diagnostics.get("normalized_endpoint_oracle_gap_definition")
        != (
            "best_sealed_case_action_minus_primary_over_best_minus_worst_"
            "sealed_case_action"
        )
        or not isinstance(oracle_gap_count, int)
        or isinstance(oracle_gap_count, bool)
        or not (0 <= oracle_gap_count <= CANONICAL_CASE_COUNT)
        or (oracle_gap_count == 0) is not (oracle_gap_mean is None)
        or (
            oracle_gap_mean is not None
            and not _optional_bounded_float(
                oracle_gap_mean, lower=-1.0e-12, upper=1.0 + 1.0e-12
            )
        )
        or not isinstance(frequencies, list)
        or tuple(
            (row.get("family"), row.get("direction"))
            for row in frequencies
            if isinstance(row, dict)
        )
        != ACTION_STRATA
        or any(
            not isinstance(row.get("selected_count"), int)
            or isinstance(row.get("selected_count"), bool)
            or int(row["selected_count"]) < 0
            for row in frequencies
            if isinstance(row, dict)
        )
        or sum(int(row["selected_count"]) for row in frequencies)
        != selected_action_count
        or diagnostics.get("primary_case_harm_count") != primary_harm_count
        or diagnostics.get("primary_case_harm_rate")
        != primary_harm_count / len(primary_cases)
    ):
        raise ProtocolError("P-DCAPS v2 terminal claim boundary drifted.")


def _optional_bounded_float(
    value: object, *, lower: float, upper: float
) -> bool:
    if value is None:
        return True
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    return bool(np.isfinite(number) and lower <= number <= upper)


def _composition_hash(payload: dict[str, object]) -> str:
    decision = _mapping(payload, "decision")
    prediction = _mapping(payload, "prediction")
    return canonical_hash(
        {
            "schema_version": "pdcaps_composed_method_prediction_v1",
            "method_decision_hash": decision["decision_hash"],
            "prediction_composition_hash": prediction["composition_hash"],
            "outer_admission_applied": payload["outer_admission_applied"],
            "outer_admission_passed": payload["outer_admission_passed"],
            "selection_enabled": payload["selection_enabled"],
            "terminal_diagnostic_only": True,
            "target_labels_used": False,
        }
    )


def _prediction_hash(payload: dict[str, object]) -> str:
    return canonical_hash(
        {
            "schema_version": "pdcaps_composed_center_prediction_v2",
            "center": payload["center"],
            "method_id": payload["method_id"],
            "sample_ids": tuple(payload["sample_ids"]),
            "probability_hash": payload["probability_hash"],
            "protected_probability_hash": payload["protected_probability_hash"],
            "selected_action_hashes": tuple(payload["selected_action_hashes"]),
            "selection_enabled": payload["selection_enabled"],
            "target_labels_used": False,
        }
    )


def _terminal_result_hash(payload: dict[str, object]) -> str:
    require_sha256(payload.get("preterminal_seal_hash"), "final preterminal seal")
    require_sha256(payload.get("label_identity_hash"), "final label identity")
    return canonical_hash(
        {
            "schema_version": "pdcaps_v2_terminal_evaluation_result_v1",
            "method_rows": payload["method_rows"],
            "center_rows": payload["center_rows"],
            "case_diagnostic_rows": payload["case_diagnostic_rows"],
            "selection_control": payload["selection_control"],
            "router_diagnostics": payload["router_diagnostics"],
            "preterminal_seal_hash": payload["preterminal_seal_hash"],
            "label_identity_hash": payload["label_identity_hash"],
            "publication_status": payload["publication_status"],
            "terminal_decision": payload["terminal_decision"],
            "raw_labels_persisted": False,
            "routing_authorized": False,
            "promotion_allowed": False,
        }
    )


def _mapping(payload: dict[str, object], key: str) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ProtocolError(f"P-DCAPS v2 payload mapping is absent: {key}.")
    return value


def _mapping_rows(
    payload: dict[str, object], key: str
) -> tuple[dict[str, object], ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ProtocolError(f"P-DCAPS v2 payload rows are absent: {key}.")
    return tuple(value)


def _array_key(center: str, method: str) -> str:
    return f"center_{center}__{method}"


__all__ = (
    "validate_final_bundle",
    "validate_preterminal_bundle",
    "verify_durable_preterminal_attestation",
)
