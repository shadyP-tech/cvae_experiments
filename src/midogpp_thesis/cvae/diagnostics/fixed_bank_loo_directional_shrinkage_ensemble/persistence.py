"""Atomic, nonrepairing, JSON-native persistence for DCSE phase products."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...runtime.artifact_io import atomic_json, read_json
from .hashing import canonical_hash, json_native
from .reports import protocol_manifest_payload, run_state_payload, seal_payload


_FORBIDDEN_PERSISTED_KEYS = frozenset(
    {
        "label",
        "labels",
        "ground_truth",
        "true_label",
        "image_path",
        "sample_path",
        "manifest_path",
    }
)


def object_payload(value: object) -> dict[str, object]:
    converter = getattr(value, "to_payload", None)
    payload = json_native(converter() if callable(converter) else value)
    if not isinstance(payload, dict):
        raise ProtocolError("Directional-shrinkage row must be a JSON object.")
    _reject_forbidden_keys(payload)
    return payload


def persist_json(path: Path, payload: Mapping[str, object]) -> None:
    converted = json_native(payload)
    if not isinstance(converted, dict):
        raise ProtocolError("Directional-shrinkage JSON product must be an object.")
    _reject_forbidden_keys(converted)
    if path.is_symlink():
        raise ProtocolError("Directional-shrinkage JSON path is a symlink.")
    if path.exists():
        if not path.is_file() or read_json(path) != converted:
            raise ProtocolError(
                f"Directional-shrinkage refuses repair of {path.name}."
            )
        return
    atomic_json(path, converted)


def persist_rows(
    path: Path,
    rows: Sequence[object],
    *,
    fields: Sequence[str] | None = None,
) -> tuple[dict[str, object], ...]:
    payloads = tuple(object_payload(row) for row in rows)
    if not payloads:
        raise ProtocolError(f"Directional-shrinkage table is empty: {path.name}.")
    columns = tuple(fields or sorted({key for row in payloads for key in row}))
    if not columns or len(set(columns)) != len(columns) or any(
        set(row) != set(columns) for row in payloads
    ):
        raise ProtocolError(f"Directional-shrinkage table schema drifted: {path.name}.")
    expected = _csv_bytes(payloads, columns)
    if path.is_symlink():
        raise ProtocolError("Directional-shrinkage table path is a symlink.")
    if path.exists():
        if not path.is_file() or path.read_bytes() != expected:
            raise ProtocolError(
                f"Directional-shrinkage refuses table repair: {path.name}."
            )
        return payloads
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        temporary.write_bytes(expected)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return payloads


def read_rows(path: Path) -> tuple[dict[str, object], ...]:
    if path.is_symlink() or not path.is_file():
        raise ProtocolError(f"Directional-shrinkage table is absent: {path}.")
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or len(set(reader.fieldnames)) != len(
                reader.fieldnames
            ):
                raise ProtocolError("Directional-shrinkage CSV header drifted.")
            rows = []
            for raw in reader:
                row: dict[str, object] = {}
                for key in reader.fieldnames:
                    try:
                        row[key] = json.loads(raw[key])
                    except (TypeError, json.JSONDecodeError) as exc:
                        raise ProtocolError(
                            "Directional-shrinkage CSV cell is not canonical JSON."
                        ) from exc
                rows.append(row)
    except OSError as exc:
        raise ProtocolError(f"Cannot read directional-shrinkage table: {path}.") from exc
    if not rows or path.read_bytes() != _csv_bytes(tuple(rows), tuple(reader.fieldnames)):
        raise ProtocolError("Directional-shrinkage CSV bytes are not canonical.")
    for row in rows:
        _reject_forbidden_keys(row)
    return tuple(rows)


def persist_initial_surfaces(
    root: Path,
    *,
    config: object,
    protocol: object,
    provenance: Mapping[str, Mapping[str, object]],
    frame: object,
    firewall: Mapping[str, object],
    actions: Sequence[object],
) -> Mapping[str, object]:
    input_ids = tuple(getattr(config, "input_artifact_ids"))
    if set(provenance) != set(input_ids) or len(provenance) != 6:
        raise ProtocolError("Directional-shrinkage provenance must cover six inputs.")
    manifest = protocol_manifest_payload(
        config,
        protocol=protocol,
        input_artifact_hashes={
            artifact_id: canonical_hash(provenance[artifact_id])
            for artifact_id in input_ids
        },
        cache_binding_hash=str(getattr(frame, "cache_binding_hash")),
        firewall=firewall,
    )
    persist_json(root / "manifests/protocol_manifest.json", manifest)
    action_rows = persist_rows(root / "tables/action_library.csv", actions)
    by_target: dict[str, list[dict[str, object]]] = {}
    for row in action_rows:
        by_target.setdefault(str(row["target_center"]), []).append(row)
    unhashed = {
        "schema_version": "fixed_bank_dcse_action_library_manifest_v1",
        "actions_by_target": by_target,
        "action_count": len(action_rows),
        "physical_actions_per_target": 10,
        "target_expert_used": False,
        "labels_used": False,
        "previous_probability_surface_used": False,
    }
    action_manifest = {**unhashed, "action_library_hash": canonical_hash(unhashed)}
    persist_json(root / "manifests/action_library.json", action_manifest)
    return action_manifest


def persist_physical_prelabel(
    root: Path,
    *,
    prediction: object,
    probability_index: Sequence[object],
    probability_surface_hash: str,
) -> Mapping[str, object]:
    rows = persist_rows(
        root / "tables/exact_nine_probability_index.csv", probability_index
    )
    seal = seal_payload(
        "fixed_bank_dcse_physical_prelabel_seal_v1",
        bindings={
            "global_prediction_seal_hash": str(getattr(prediction, "seal_hash")),
            "prediction_store_hash": str(getattr(prediction, "store").store_hash),
            "probability_surface_hash": probability_surface_hash,
            "probability_index_hash": canonical_hash(rows),
        },
        physical_cell_count=len(getattr(prediction, "store").cells),
        target_action_index_count=len(rows),
        exact_nine_reduction_dtype="float64",
        stored_probability_dtype="float32",
        labels_used=False,
        sealed_before_label_capabilities=True,
    )
    persist_json(root / "manifests/physical_prelabel_seal.json", seal)
    return seal


def persist_loo_products(
    root: Path,
    *,
    plans: Sequence[object],
    case_action_confusions: Sequence[object],
    directional_gains: Sequence[object],
    physical_prelabel_seal_hash: str,
) -> Mapping[str, object]:
    plan_rows = persist_rows(root / "tables/loo_plans.csv", plans)
    confusion_rows = persist_rows(
        root / "tables/case_action_confusions.csv", case_action_confusions
    )
    gain_rows = persist_rows(root / "tables/directional_gains.csv", directional_gains)
    seal = seal_payload(
        "fixed_bank_dcse_loo_plan_seal_v1",
        bindings={
            "physical_prelabel_seal_hash": physical_prelabel_seal_hash,
            "plans_hash": canonical_hash(plan_rows),
            "case_action_confusions_hash": canonical_hash(confusion_rows),
            "directional_gains_hash": canonical_hash(gain_rows),
        },
        plan_count=len(plan_rows),
        held_case_count=218,
        each_plan_excludes_held_whole_case=True,
        terminal_labels_used=False,
        raw_labels_persisted=False,
    )
    persist_json(root / "manifests/loo_plan_seal.json", seal)
    return seal


def persist_donor_priors(
    root: Path, *, priors: Sequence[object], loo_plan_seal_hash: str
) -> Mapping[str, object]:
    rows = persist_rows(root / "tables/donor_priors.csv", priors)
    seal = seal_payload(
        "fixed_bank_dcse_donor_prior_seal_v1",
        bindings={
            "loo_plan_seal_hash": loo_plan_seal_hash,
            "donor_priors_hash": canonical_hash(rows),
        },
        donor_prior_count=len(rows),
        strict_target_and_source_exclusion=True,
        equal_query_center_aggregation=True,
        terminal_labels_used=False,
    )
    persist_json(root / "manifests/donor_prior_seal.json", seal)
    return seal


def persist_endpoint_library(
    root: Path, *, endpoints: Sequence[object], donor_prior_seal_hash: str
) -> Mapping[str, object]:
    rows = persist_rows(root / "tables/endpoint_arms.csv", endpoints)
    seal = seal_payload(
        "fixed_bank_dcse_endpoint_library_seal_v1",
        bindings={
            "donor_prior_seal_hash": donor_prior_seal_hash,
            "endpoint_library_hash": canonical_hash(rows),
        },
        arm_count=len(rows),
        all_nine_arm_identities_retained=True,
        terminal_labels_used=False,
    )
    persist_json(root / "manifests/endpoint_library_seal.json", seal)
    return seal


def persist_decisions(
    root: Path,
    *,
    decisions: Sequence[object],
    control_decisions: Sequence[object],
    predictions: Sequence[object],
    descriptive_control_predictions: Sequence[object],
    loo_plan_seal_hash: str,
    global_plan_seal_hash: str,
    donor_prior_seal_hash: str,
    endpoint_library_seal_hash: str,
    route_decision_barrier: Mapping[str, object],
    null_plan: object,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    decision_rows = persist_rows(root / "tables/arm_decisions.csv", decisions)
    control_rows = persist_rows(
        root / "tables/control_decisions.csv", control_decisions
    )
    prediction_rows = persist_rows(root / "tables/method_predictions.csv", predictions)
    descriptive_rows = persist_rows(
        root / "tables/descriptive_control_predictions.csv",
        descriptive_control_predictions,
    )
    canonical_methods = {
        str(row.get("method_id")) for row in prediction_rows
    }
    descriptive_methods = {
        str(row.get("method_id")) for row in descriptive_rows
    }
    if (
        len(decision_rows) != 218 * 18
        or len(control_rows) != 218 * 2
        or len(prediction_rows) != 9_928 * 6
        or len(descriptive_rows) != 9_928 * 5
        or canonical_methods
        != {
            "B",
            "U",
            "DCSE_LOO",
            "G_directional_matched",
            "DLOO_raw",
            "LOO_frequency_committee",
        }
        or descriptive_methods
        != {
            "DCSE_hard_vote_descriptive",
            "DCSE_unique_mean_descriptive",
            "uniform_A1_mean_descriptive",
            "DCSE_zero_to_one_only_descriptive",
            "DCSE_one_to_zero_only_descriptive",
        }
    ):
        raise ProtocolError(
            "Directional-shrinkage preterminal decision/prediction topology drifted."
        )
    decision_seal = seal_payload(
        "fixed_bank_dcse_arm_decisions_seal_v1",
        bindings={
            "loo_plan_seal_hash": loo_plan_seal_hash,
            "global_plan_seal_hash": global_plan_seal_hash,
            "donor_prior_seal_hash": donor_prior_seal_hash,
            "endpoint_library_seal_hash": endpoint_library_seal_hash,
            "arm_decisions_hash": canonical_hash(decision_rows),
            "control_decisions_hash": canonical_hash(control_rows),
            "method_predictions_hash": canonical_hash(prediction_rows),
            "descriptive_control_predictions_hash": canonical_hash(
                descriptive_rows
            ),
        },
        arm_decision_count=len(decision_rows),
        control_decision_count=len(control_rows),
        method_prediction_count=len(prediction_rows),
        descriptive_control_prediction_count=len(descriptive_rows),
        preterminal_method_ids=sorted(canonical_methods),
        descriptive_control_method_ids=sorted(descriptive_methods),
        terminal_labels_used=False,
    )
    persist_json(root / "manifests/arm_decisions_seal.json", decision_seal)
    route_seals = route_decision_barrier.get("decision_seals")
    if (
        route_decision_barrier.get("route_count") != 218
        or route_decision_barrier.get("plan_seal_hash") != global_plan_seal_hash
        or not isinstance(route_seals, list)
        or len(route_seals) != 218
        or route_decision_barrier.get("decision_barrier_hash")
        != canonical_hash(
            {
                key: value
                for key, value in route_decision_barrier.items()
                if key != "decision_barrier_hash"
            }
        )
    ):
        raise ProtocolError("Directional-shrinkage route decision barrier drifted.")
    aggregate = seal_payload(
        "fixed_bank_dcse_aggregate_plan_decision_seal_v1",
        bindings={
            "loo_plan_seal_hash": loo_plan_seal_hash,
            "global_plan_seal_hash": global_plan_seal_hash,
            "donor_prior_seal_hash": donor_prior_seal_hash,
            "endpoint_library_seal_hash": endpoint_library_seal_hash,
            "arm_decisions_seal_hash": str(decision_seal["seal_hash"]),
            "control_decisions_hash": canonical_hash(control_rows),
            "method_predictions_hash": canonical_hash(prediction_rows),
            "descriptive_control_predictions_hash": canonical_hash(
                descriptive_rows
            ),
            "route_decision_barrier_hash": str(
                route_decision_barrier["decision_barrier_hash"]
            ),
            "ordered_route_decision_seals_hash": canonical_hash(route_seals),
            "candidate_identity_null_plan_hash": str(
                getattr(null_plan, "plan_hash")
            ),
            "candidate_identity_null_permutation_sha256": str(
                getattr(null_plan, "to_payload")()["permutation_sha256"]
            ),
        },
        all_218_loo_plans_complete=True,
        all_nine_arm_decisions_complete_per_case=True,
        all_two_control_decisions_complete_per_case=True,
        all_preterminal_method_predictions_complete=True,
        all_descriptive_control_predictions_complete=True,
        control_decision_count=len(control_rows),
        method_prediction_count=len(prediction_rows),
        descriptive_control_prediction_count=len(descriptive_rows),
        global_barrier_complete=True,
        candidate_identity_null_plan=dict(getattr(null_plan, "to_payload")()),
        null_plan_sealed_before_terminal_labels=True,
        null_plan_can_change_canonical_decisions=False,
        terminal_labels_used=False,
    )
    persist_json(root / "manifests/aggregate_plan_decision_seal.json", aggregate)
    return decision_seal, aggregate


def persist_terminal(
    root: Path,
    *,
    result: Mapping[str, object],
    capability_report: Mapping[str, object],
    leakage_report: Mapping[str, object],
    publication_decision: Mapping[str, object],
    runtime_summary: Mapping[str, object],
) -> None:
    normalized = _terminal_rows(result)
    table_members = {
        "case_confusions": "tables/terminal_case_confusions.csv",
        "method_metrics": "tables/terminal_method_metrics.csv",
        "center_metrics": "tables/terminal_center_metrics.csv",
        "equal_center_contrasts": "tables/terminal_contrasts.csv",
        "delete_one_center": "tables/whole_pipeline_delete_one_center.csv",
        "leave_one_arm": "tables/leave_one_arm_ablations.csv",
        "null_statistics": "tables/null_statistics.csv",
    }
    for key, member in table_members.items():
        persist_rows(root / member, normalized[key])
    terminal_seal = result.get("terminal_seal")
    if not isinstance(terminal_seal, Mapping):
        raise ProtocolError("Directional-shrinkage terminal seal is absent.")
    persist_json(root / "manifests/terminal_evaluation_seal.json", terminal_seal)
    persist_json(root / "reports/label_capability_report.json", capability_report)
    persist_json(root / "reports/leakage_report.json", leakage_report)
    persist_json(root / "reports/publication_decision.json", publication_decision)
    persist_json(root / "reports/runtime_summary.json", runtime_summary)


def persist_validation_report(root: Path, checks: Mapping[str, object]) -> None:
    from .fresh_process_validation import (
        ATTESTATION_KEY,
        verify_attested_validation_checks,
    )

    converted = json_native(checks)
    if not isinstance(converted, dict):  # pragma: no cover - json_native guard
        raise ProtocolError("Directional-shrinkage validation report is malformed.")
    reconstructed = {
        key: value for key, value in converted.items() if key != ATTESTATION_KEY
    }
    if (
        reconstructed.get("schema_version")
        != "fixed_bank_dcse_validation_v1"
        or reconstructed.get("status") != "PASS"
        or reconstructed.get("all_six_preterminal_methods_reconstructed")
        is not True
        or reconstructed.get("all_five_descriptive_controls_reconstructed")
        is not True
        or reconstructed.get("candidate_identity_null_plan_reconstructed")
        is not True
        or reconstructed.get("exact_topology_and_confusions_compared") is not True
        or reconstructed.get("fitted_numeric_tolerance_used") is not False
        or reconstructed.get(
            "content_index_validated_before_scientific_members"
        )
        is not True
        or reconstructed.get("two_fresh_cuda_free_process_replays_required")
        is not True
        or reconstructed.get("nonrepairing_validation") is not True
        or reconstructed.get("closed_world") is not True
        or reconstructed.get("raw_labels_persisted") is not False
        or reconstructed.get("image_or_sample_paths_persisted") is not False
        or reconstructed.get("terminal_diagnostic_only") is not True
        or reconstructed.get("fresh_evidence") is not False
        or reconstructed.get("promotion_eligible") is not False
        or reconstructed.get("may_feed_another_experiment") is not False
    ):
        raise ProtocolError(
            "Directional-shrinkage validation report claims are incomplete."
        )
    verified = verify_attested_validation_checks(
        converted,
        expected_reconstructed_checks=reconstructed,
    )
    if verified != converted:
        raise ProtocolError(
            "Directional-shrinkage validation report attestation drifted."
        )
    persist_json(root / "reports/validation_report.json", converted)


def write_run_state(
    root: Path,
    *,
    status: str,
    phase: str,
    error: str | None = None,
    error_class: str | None = None,
) -> None:
    # Run state is the sole mutable status product and is excluded from content.
    path = root / "reports/run_state.json"
    if path.is_symlink():
        raise ProtocolError("Directional-shrinkage run state is a symlink.")
    atomic_json(
        path,
        run_state_payload(
            status, phase, error=error, error_class=error_class
        ),
    )


def _terminal_rows(result: Mapping[str, object]) -> dict[str, Sequence[object]]:
    descriptive = result.get("descriptive_inference", {})
    if descriptive is None:
        descriptive = {}
    if not isinstance(descriptive, Mapping):
        descriptive = {"method_metrics": descriptive}
    values = {
        "case_confusions": result.get("case_confusions"),
        "method_metrics": result.get(
            "method_metrics", descriptive.get("method_metrics")
        ),
        "center_metrics": result.get("center_metrics"),
        "equal_center_contrasts": result.get("equal_center_contrasts"),
        "delete_one_center": result.get("delete_one_center"),
        "leave_one_arm": result.get("leave_one_arm"),
        "null_statistics": result.get(
            "null_statistics", descriptive.get("null_statistics")
        ),
    }
    for key, value in values.items():
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
            raise ProtocolError(f"Directional-shrinkage terminal rows absent: {key}.")
    return values  # type: ignore[return-value]


def _csv_bytes(
    rows: Sequence[Mapping[str, object]], columns: Sequence[str]
) -> bytes:
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=list(columns),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: json.dumps(
                    json_native(row[key]),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
                for key in columns
            }
        )
    return stream.getvalue().encode("utf-8")


def _reject_forbidden_keys(value: object) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            name = str(key).casefold()
            if name in _FORBIDDEN_PERSISTED_KEYS or name.endswith("_path"):
                raise ProtocolError(
                    f"Directional-shrinkage persistence forbids key: {key}."
                )
            _reject_forbidden_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_forbidden_keys(nested)


__all__ = (
    "object_payload",
    "persist_decisions",
    "persist_donor_priors",
    "persist_endpoint_library",
    "persist_initial_surfaces",
    "persist_json",
    "persist_loo_products",
    "persist_physical_prelabel",
    "persist_rows",
    "persist_terminal",
    "persist_validation_report",
    "read_rows",
    "write_run_state",
)
