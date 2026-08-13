"""Content-first, exact scientific reconstruction of the terminal DCSE bundle."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

from ....data.contract.stage70_target_evaluation.contracts import evaluation_row_id
from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from .actions import action_library_by_target, build_action_library
from .bundle import assert_closed_world, validate_content_index
from .constants import CENTERS, PRE_TERMINAL_METHOD_IDS, physical_action_ids
from .decisions import (
    select_arm_decisions,
    select_matched_g_decisions,
    select_nested_frequency_committee_control,
    select_raw_directional_loo_control,
)
from .donor_priors import compute_donor_priors
from .endpoint_library import build_endpoint_arms
from .ensemble import (
    DESCRIPTIVE_METHOD_IDS,
    compose_control_predictions,
    compose_descriptive_control_predictions,
    compose_method_predictions,
    fixed_physical_method_predictions,
)
from .execution_adapter import load_validated_workstation_preflight
from .experiment_contracts import SCRATCH_ROOT
from .hashing import canonical_hash
from .inputs import (
    assert_input_fence,
    load_label_free_test_frame,
    load_validated_locks,
    validate_active_diagnostic_workspace_binding,
    validate_pre_gpu_firewall,
    validate_workspace_provenance,
)
from .label_capabilities import LabelCapabilityFirewall
from .loo_plans import build_whole_case_loo_plans, seal_loo_plans
from .nulls import build_candidate_identity_null_plan
from .persistence import read_rows
from .protocol import canonical_consumed_test_protocol
from .reports import (
    leakage_report_payload,
    protocol_manifest_payload,
    publication_decision_payload,
    run_state_payload,
)
from .scoring import score_case_action_confusions, score_loo_directional_gains
from .validation_plans import validate_plan_and_decision_products
from .validation_prelabel import reconstruct_prelabel
from .validation_terminal import validate_terminal_products


VALIDATION_SCHEMA = "fixed_bank_dcse_validation_v1"


def validate_fixed_bank_loo_directional_shrinkage_ensemble_bundle(
    root: str | Path,
    *,
    config: object,
    allow_pending_validation: bool = False,
) -> Mapping[str, object]:
    """Rebuild every scientific product without mutating persisted evidence."""

    path = Path(root)
    assert_closed_world(
        path,
        allow_incomplete=False,
        allow_pending_validation=allow_pending_validation,
    )
    protocol = canonical_consumed_test_protocol()
    content = validate_content_index(
        path,
        config_contract_hash=str(getattr(config, "contract_hash")),
        protocol_contract_hash=protocol.contract_hash,
    )
    _reject_forbidden_persistence(path)

    assert_input_fence(config)
    workspace = validate_active_diagnostic_workspace_binding(config)
    provenance = validate_workspace_provenance(path, config)
    locks = load_validated_locks(config)
    frame = load_label_free_test_frame(config)
    firewall = dict(validate_pre_gpu_firewall(config, frame, locks))
    firewall["workspace_binding"] = workspace
    expected_protocol = protocol_manifest_payload(
        config,
        protocol=protocol,
        input_artifact_hashes={
            artifact_id: canonical_hash(provenance[artifact_id])
            for artifact_id in getattr(config, "input_artifact_ids")
        },
        cache_binding_hash=frame.cache_binding_hash,
        firewall=firewall,
    )
    if read_json(path / "manifests/protocol_manifest.json") != expected_protocol:
        raise ProtocolError(
            "Directional-shrinkage protocol manifest is not reconstructive."
        )
    _validate_action_library(path)
    preflight = load_validated_workstation_preflight(
        path, runtime=getattr(config, "runtime")
    )
    prelabel = reconstruct_prelabel(
        path,
        config=config,
        frame=frame,
        generation_lock_hash=locks.generation.generation_lock_hash,
    )
    surface = prelabel["probability_surface"]
    plans = build_whole_case_loo_plans(
        frame.rows,
        probability_surface_hash=str(prelabel["probability_surface_hash"]),
    )
    global_plan_seal = seal_loo_plans(
        plans,
        probability_surface_hash=str(prelabel["probability_surface_hash"]),
    )
    label_firewall = LabelCapabilityFirewall(
        global_plan_seal,
        lambda allowed: _read_scoped_manifest_labels(
            config, frame, allowed_keys=allowed
        ),
    )

    donor_counts: list[object] = []
    donor_priors: list[object] = []
    priors_by_target: dict[str, tuple[object, ...]] = {}
    for target in CENTERS:
        counts_by_source: dict[str, tuple[object, ...]] = {}
        for action in action_library_by_target()[target]:
            source = action.selected_source
            if source is None:
                continue
            counts = score_case_action_confusions(
                surface, label_firewall.open_donor_labels(target, source)
            )
            counts_by_source[source] = counts
            donor_counts.extend(counts)
        priors = compute_donor_priors(counts_by_source, heldout_center=target)
        priors_by_target[target] = priors
        donor_priors.extend(priors)

    repeated_counts: list[object] = []
    gains: list[object] = []
    endpoint_decisions: list[object] = []
    control_decisions: list[object] = []
    for plan in plans:
        labels = label_firewall.open_route_support_labels(
            plan.target_center, plan.case_id, plan_hash=plan.plan_hash
        )
        route_counts = score_case_action_confusions(surface, labels)
        route_gains = score_loo_directional_gains(route_counts, plan)
        priors = priors_by_target[plan.target_center]
        endpoint_decisions.extend(
            select_arm_decisions(
                method_id="DCSE_LOO",
                target_center=plan.target_center,
                case_id=plan.case_id,
                support_gains=route_gains,
                donor_priors=priors,
            )
        )
        endpoint_decisions.extend(
            select_matched_g_decisions(
                target_center=plan.target_center,
                case_id=plan.case_id,
                donor_priors=priors,
            )
        )
        control_decisions.extend(
            (
                select_raw_directional_loo_control(
                    target_center=plan.target_center,
                    case_id=plan.case_id,
                    support_gains=route_gains,
                ),
                select_nested_frequency_committee_control(
                    plan=plan,
                    support_counts=route_counts,
                ),
            )
        )
        repeated_counts.extend(route_counts)
        gains.extend(route_gains)
    case_action_confusions = _deduplicate_case_action_confusions(repeated_counts)
    method_predictions, descriptive_predictions = _compose_predictions(
        surface,
        plans=plans,
        endpoint_decisions=endpoint_decisions,
        control_decisions=control_decisions,
    )
    for plan in plans:
        key = (plan.target_center, plan.case_id)
        label_firewall.record_route_decision_seal(
            *key,
            canonical_hash(
                {
                    "endpoint_decisions": [
                        row.to_payload()
                        for row in endpoint_decisions
                        if (row.target_center, row.case_id) == key
                    ],
                    "control_decisions": [
                        row.to_payload()
                        for row in control_decisions
                        if (row.target_center, row.case_id) == key
                    ],
                    "preterminal_method_predictions": [
                        row.to_payload()
                        for row in method_predictions
                        if (row.target_center, row.case_id) == key
                    ],
                    "descriptive_control_predictions": [
                        row.to_payload()
                        for row in descriptive_predictions
                        if (row.target_center, row.case_id) == key
                    ],
                }
            ),
        )
    route_barrier = label_firewall.decision_barrier_payload()
    null_plan = build_candidate_identity_null_plan(
        tuple((plan.target_center, plan.case_id) for plan in plans),
        seed=int(getattr(config, "nulls")["seed"]),
        replicates=int(getattr(config, "nulls")["replicates"]),
    )
    plan_checks = validate_plan_and_decision_products(
        path,
        plans=plans,
        case_action_confusions=case_action_confusions,
        directional_gains=gains,
        donor_priors=donor_priors,
        endpoint_arms=build_endpoint_arms(),
        arm_decisions=endpoint_decisions,
        control_decisions=control_decisions,
        method_predictions=method_predictions,
        descriptive_control_predictions=descriptive_predictions,
        physical_prelabel_seal_hash=str(prelabel["physical_prelabel_seal_hash"]),
        global_plan_seal_hash=global_plan_seal.plan_seal_hash,
        route_decision_barrier=route_barrier,
        null_plan=null_plan,
        null_contract=getattr(config, "nulls"),
    )
    aggregate = read_json(path / "manifests/aggregate_plan_decision_seal.json")
    bindings = aggregate.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ProtocolError("Directional-shrinkage aggregate bindings are absent.")
    label_firewall.record_aggregate_plan_decision_seal(
        str(aggregate["seal_hash"]),
        plan_seal_hash=str(bindings["global_plan_seal_hash"]),
        decision_barrier_hash=str(bindings["route_decision_barrier_hash"]),
    )
    terminal_labels = label_firewall.open_terminal_labels()
    terminal = _evaluate_terminal(
        probability_surface=surface,
        plans=plans,
        donor_counts=tuple(donor_counts),
        case_action_confusions=case_action_confusions,
        donor_priors=tuple(donor_priors),
        arm_decisions=tuple(endpoint_decisions),
        method_predictions=method_predictions,
        descriptive_predictions=descriptive_predictions,
        aggregate_plan_decision_seal_hash=str(aggregate["seal_hash"]),
        terminal_labels=terminal_labels,
        config=config,
        null_plan=null_plan,
    )
    terminal_checks = validate_terminal_products(
        path,
        reconstructed=terminal,
        expected_lineage_bindings={
            "probability_surface_hash": str(surface.surface_hash),
            "ordered_loo_plan_hashes_hash": canonical_hash(
                [plan.plan_hash for plan in plans]
            ),
            "ordered_donor_prior_hashes_hash": canonical_hash(
                [row.prior_hash for row in donor_priors]
            ),
            "ordered_arm_decision_hashes_hash": canonical_hash(
                [row.decision_hash for row in endpoint_decisions]
            ),
            "ordered_preterminal_prediction_hashes_hash": canonical_hash(
                [row.probability_hash for row in method_predictions]
            ),
            "ordered_descriptive_prediction_hashes_hash": canonical_hash(
                [row.probability_hash for row in descriptive_predictions]
            ),
            "candidate_identity_null_plan_hash": null_plan.plan_hash,
            "aggregate_plan_decision_seal_hash": str(aggregate["seal_hash"]),
        },
    )
    capability = label_firewall.report_payload()
    _validate_reports(
        path,
        config=config,
        preflight=preflight,
        prelabel=prelabel,
        aggregate=aggregate,
        capability=capability,
        terminal=terminal,
        allow_pending_validation=allow_pending_validation,
    )
    checks = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "PASS",
        "content_hash": content["content_hash"],
        "config_contract_hash": str(getattr(config, "contract_hash")),
        "protocol_contract_hash": protocol.contract_hash,
        "workspace_binding": workspace,
        "input_artifact_count": len(provenance),
        "pre_gpu_firewall_status": firewall["status"],
        "workstation_preflight_status": preflight["status"],
        "physical_cell_count": len(prelabel["prediction"].store.cells),
        "probability_index_count": prelabel["probability_index_count"],
        "probability_surface_hash": prelabel["probability_surface_hash"],
        **dict(plan_checks),
        **dict(terminal_checks),
        "all_six_preterminal_methods_reconstructed": True,
        "all_five_descriptive_controls_reconstructed": True,
        "candidate_identity_null_plan_reconstructed": True,
        "exact_topology_and_confusions_compared": True,
        "fitted_numeric_tolerance_used": False,
        "content_index_validated_before_scientific_members": True,
        "two_fresh_cuda_free_process_replays_required": True,
        "nonrepairing_validation": True,
        "closed_world": True,
        "raw_labels_persisted": False,
        "image_or_sample_paths_persisted": False,
        "terminal_diagnostic_only": True,
        "fresh_evidence": False,
        "promotion_eligible": False,
        "may_feed_another_experiment": False,
    }
    if allow_pending_validation:
        return checks
    from .fresh_process_validation import verify_attested_validation_checks

    report = read_json(path / "reports/validation_report.json")
    attested = verify_attested_validation_checks(
        report, expected_reconstructed_checks=checks
    )
    if report != attested:
        raise ProtocolError(
            "Directional-shrinkage validation report is not reconstructive."
        )
    return attested


def _compose_predictions(
    surface: object,
    *,
    plans: Sequence[object],
    endpoint_decisions: Sequence[object],
    control_decisions: Sequence[object],
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    endpoint = {
        method: compose_method_predictions(
            surface,
            tuple(row for row in endpoint_decisions if row.method_id == method),
            method_id=method,
        )
        for method in ("DCSE_LOO", "G_directional_matched")
    }
    controls = {
        method: compose_control_predictions(
            surface,
            tuple(row for row in control_decisions if row.method_id == method),
            method_id=method,
        )
        for method in ("DLOO_raw", "LOO_frequency_committee")
    }
    by_method = {
        "B": fixed_physical_method_predictions(surface, method_id="B"),
        "U": fixed_physical_method_predictions(surface, method_id="U"),
        **endpoint,
        **controls,
    }
    canonical = tuple(
        row for method in PRE_TERMINAL_METHOD_IDS for row in by_method[method]
    )
    descriptive = tuple(
        compose_descriptive_control_predictions(
            surface,
            tuple(
                row for row in endpoint_decisions if row.method_id == "DCSE_LOO"
            ),
        )
    )
    if (
        len(descriptive) != 9_928 * 5
        or tuple(dict.fromkeys(row.method_id for row in descriptive))
        != DESCRIPTIVE_METHOD_IDS
    ):
        raise ProtocolError(
            "Directional-shrinkage validation descriptive topology drifted."
        )
    return canonical, descriptive


def _deduplicate_case_action_confusions(
    repeated: Sequence[object],
) -> tuple[object, ...]:
    rows: dict[tuple[str, str, str], object] = {}
    payloads: dict[tuple[str, str, str], Mapping[str, object]] = {}
    for row in repeated:
        key = (row.target_center, row.case_id, row.action_id)
        payload = row.to_payload()
        if key in payloads and payloads[key] != payload:
            raise ProtocolError(
                "Directional-shrinkage repeated validation confusion drifted."
            )
        rows.setdefault(key, row)
        payloads.setdefault(key, payload)
    ordered = tuple(
        rows[(center, case_id, action)]
        for center in CENTERS
        for case_id in sorted(
            {case for target, case, _action in rows if target == center}
        )
        for action in physical_action_ids(center)
    )
    if len(ordered) != 2_180 or len(rows) != len(ordered):
        raise ProtocolError(
            "Directional-shrinkage validation confusion topology is not 218 x 10."
        )
    return ordered


def _validate_action_library(root: Path) -> None:
    actions = build_action_library()
    rows = tuple(action.to_payload() for action in actions)
    if read_rows(root / "tables/action_library.csv") != rows:
        raise ProtocolError("Directional-shrinkage action table drifted.")
    by_target: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_target.setdefault(str(row["target_center"]), []).append(row)
    unhashed = {
        "schema_version": "fixed_bank_dcse_action_library_manifest_v1",
        "actions_by_target": by_target,
        "action_count": len(rows),
        "physical_actions_per_target": 10,
        "target_expert_used": False,
        "labels_used": False,
        "previous_probability_surface_used": False,
    }
    expected = {**unhashed, "action_library_hash": canonical_hash(unhashed)}
    if read_json(root / "manifests/action_library.json") != expected:
        raise ProtocolError("Directional-shrinkage action manifest drifted.")


def _read_scoped_manifest_labels(
    config: object,
    frame: object,
    *,
    allowed_keys: frozenset[tuple[str, str, str]],
) -> Sequence[object]:
    from .products import BinaryLabel

    universe = {(row.center, row.case_id, row.sample_id): row for row in frame.rows}
    if not allowed_keys or not set(allowed_keys) <= set(universe):
        raise ProtocolError("Directional-shrinkage validation label grant escaped.")
    ordered_keys = tuple(
        (row.center, row.case_id, row.sample_id)
        for row in frame.rows
        if (row.center, row.case_id, row.sample_id) in allowed_keys
    )
    if len(ordered_keys) != len(allowed_keys):
        raise ProtocolError(
            "Directional-shrinkage validation label grant order drifted."
        )
    requested = {key: universe[key] for key in ordered_keys}
    found: dict[tuple[str, str, str], object] = {}
    manifest = Path(getattr(config, "test_manifest_path"))
    manifest_hash = str(getattr(config, "expected_manifest_sha256"))
    try:
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            for ordinal, raw in enumerate(csv.DictReader(handle)):
                key = (
                    str(raw.get("center", "")),
                    str(raw.get("case_id", "")),
                    evaluation_row_id(manifest_hash, ordinal),
                )
                if key not in requested:
                    continue
                if requested[key].manifest_row_index != ordinal or key in found:
                    raise ProtocolError(
                        "Directional-shrinkage validation manifest order drifted."
                    )
                found[key] = BinaryLabel(*key, int(raw["label"]), "validator_loader")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ProtocolError(
            "Cannot load scoped directional-shrinkage validation labels."
        ) from exc
    if set(found) != set(requested):
        raise ProtocolError(
            "Directional-shrinkage validation label coverage drifted."
        )
    return tuple(found[key] for key in ordered_keys)


def _evaluate_terminal(**kwargs: object) -> Mapping[str, object]:
    try:
        from .terminal import evaluate_terminal
    except ImportError as exc:
        raise ProtocolError(
            "Directional-shrinkage terminal evaluator is unavailable."
        ) from exc
    result = evaluate_terminal(**kwargs)
    if not isinstance(result, Mapping):
        raise ProtocolError("Directional-shrinkage terminal result is not a mapping.")
    return result


def _validate_reports(
    root: Path,
    *,
    config: object,
    preflight: Mapping[str, object],
    prelabel: Mapping[str, object],
    aggregate: Mapping[str, object],
    capability: Mapping[str, object],
    terminal: Mapping[str, object],
    allow_pending_validation: bool,
) -> None:
    prediction = prelabel["prediction"]
    terminal_seal = terminal.get("terminal_seal")
    if not isinstance(terminal_seal, Mapping):
        raise ProtocolError("Directional-shrinkage terminal seal is absent.")
    expected_leakage = leakage_report_payload(
        prediction_seal_hash=str(prediction.seal_hash),
        physical_prelabel_seal_hash=str(prelabel["physical_prelabel_seal_hash"]),
        aggregate_plan_decision_seal_hash=str(aggregate["seal_hash"]),
        capability_report=capability,
    )
    expected_publication = publication_decision_payload(
        str(
            terminal_seal.get(
                "seal_hash", terminal_seal.get("terminal_seal_hash")
            )
        ),
        descriptive_success_rubric=terminal_seal.get(
            "descriptive_success_rubric", {}
        ),
    )
    if (
        read_json(root / "reports/label_capability_report.json") != capability
        or read_json(root / "reports/leakage_report.json") != expected_leakage
        or read_json(root / "reports/publication_decision.json")
        != expected_publication
    ):
        raise ProtocolError("Directional-shrinkage terminal reports drifted.")
    runtime = read_json(root / "reports/runtime_summary.json")
    if (
        runtime.get("status") != "PASS"
        or runtime.get("source_stream_lock_hash")
        != prelabel["source"].lock_hash
        or runtime.get("global_prediction_seal_hash") != prediction.seal_hash
        or runtime.get("classifier_cell_count") != 810
        or runtime.get("unique_classifier_fit_count") != 810
        or runtime.get("workstation_preflight") != dict(preflight)
        or runtime.get("classifier_workers") != 4
        or runtime.get("classifier_threads_per_worker") != 3
        or runtime.get("multiprocessing_start_method") != "spawn"
        or runtime.get("classifier_source_cache_root")
        != str((Path(SCRATCH_ROOT) / "source_generation").resolve())
        or runtime.get("canonical_source_cache_root") != str(root.resolve())
        or runtime.get("local_and_canonical_source_lock_identical") is not True
        or runtime.get("terminal_or_cross_run_recovery_used") is not False
        or runtime.get("prior_run_scratch_used_as_evidence") is not False
        or runtime.get(
            "previous_stage90_artifact_checkpoint_or_scratch_reused"
        )
        is not False
        or runtime.get("recomputed_from_original_six_inputs") is not True
    ):
        raise ProtocolError("Directional-shrinkage runtime summary drifted.")
    expected_state = (
        run_state_payload(
            "RUNNING", "CLOSED_WORLD_TWO_FRESH_PROCESS_VALIDATION"
        )
        if allow_pending_validation
        else run_state_payload("COMPLETE", "COMPLETE")
    )
    if read_json(root / "reports/run_state.json") != expected_state:
        raise ProtocolError("Directional-shrinkage run state drifted.")


def _reject_forbidden_persistence(root: Path) -> None:
    forbidden = {
        "label",
        "labels",
        "ground_truth",
        "true_label",
        "image_path",
        "sample_path",
        "manifest_path",
    }
    excluded = {
        "config.resolved.yaml",
        "provenance/input_artifacts.json",
        "manifests/content_index.json",
    }
    for path in root.rglob("*.json"):
        if path.relative_to(root).as_posix() in excluded:
            continue
        value = _json(path)
        if _contains_key(value, forbidden):
            raise ProtocolError(
                "Directional-shrinkage persisted a forbidden raw label/path field."
            )
    for path in root.rglob("*.csv"):
        with path.open("r", encoding="utf-8", newline="") as handle:
            fields = csv.DictReader(handle).fieldnames
        if fields is None or forbidden & set(fields):
            raise ProtocolError(
                "Directional-shrinkage persisted a forbidden raw CSV field."
            )


def _contains_key(value: object, forbidden: set[str]) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).casefold() in forbidden
            or str(key).casefold().endswith("_path")
            or _contains_key(item, forbidden)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_key(item, forbidden) for item in value)
    return False


def _json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(
            f"Directional-shrinkage JSON is unreadable: {path}."
        ) from exc


__all__ = (
    "VALIDATION_SCHEMA",
    "validate_fixed_bank_loo_directional_shrinkage_ensemble_bundle",
)
