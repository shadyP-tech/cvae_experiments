"""Atomic, nonrepairing persistence for case-directional products."""

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
    {"label", "labels", "ground_truth", "true_label", "image_path", "sample_path"}
)


def object_payload(value: object) -> dict[str, object]:
    converter = getattr(value, "to_payload", None)
    payload = json_native(converter() if callable(converter) else value)
    if not isinstance(payload, dict):
        raise ProtocolError("Case-directional row must be a JSON object.")
    _reject_forbidden_keys(payload)
    return payload


def persist_json(path: Path, payload: Mapping[str, object]) -> None:
    converted = json_native(payload)
    if not isinstance(converted, dict):
        raise ProtocolError("Case-directional JSON product must be an object.")
    _reject_forbidden_keys(converted)
    if path.is_symlink():
        raise ProtocolError("Case-directional JSON path is a symlink.")
    if path.exists():
        if not path.is_file() or read_json(path) != converted:
            raise ProtocolError(f"Case-directional refuses repair of {path.name}.")
        return
    atomic_json(path, converted)


def persist_rows(
    path: Path, rows: Sequence[object], *, fields: Sequence[str] | None = None
) -> tuple[dict[str, object], ...]:
    payloads = tuple(object_payload(row) for row in rows)
    if not payloads:
        raise ProtocolError(f"Case-directional table is empty: {path.name}.")
    columns = tuple(fields or sorted({key for row in payloads for key in row}))
    if not columns or len(set(columns)) != len(columns) or any(
        set(row) != set(columns) for row in payloads
    ):
        raise ProtocolError(f"Case-directional table schema drifted: {path.name}.")
    expected = _csv_bytes(payloads, columns)
    if path.is_symlink():
        raise ProtocolError("Case-directional table path is a symlink.")
    if path.exists():
        if not path.is_file() or path.read_bytes() != expected:
            raise ProtocolError(f"Case-directional refuses repair: {path.name}.")
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
        raise ProtocolError(f"Case-directional table absent: {path}.")
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ProtocolError("Case-directional CSV header drifted.")
            rows = []
            for raw in reader:
                row = {}
                for key in reader.fieldnames:
                    try:
                        row[key] = json.loads(raw[key])
                    except (TypeError, json.JSONDecodeError) as exc:
                        raise ProtocolError(
                            "Case-directional CSV cell is not canonical JSON."
                        ) from exc
                rows.append(row)
    except OSError as exc:
        raise ProtocolError(f"Cannot read case-directional table: {path}.") from exc
    if not rows or path.read_bytes() != _csv_bytes(tuple(rows), tuple(reader.fieldnames)):
        raise ProtocolError("Case-directional CSV bytes are not canonical.")
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
        raise ProtocolError("Case-directional provenance must cover six inputs.")
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
    seal = seal_payload(
        "fixed_bank_cdca_action_library_manifest_v1",
        bindings={"actions_hash": canonical_hash(action_rows)},
        action_count=len(action_rows),
        physical_actions_per_target=10,
        labels_used=False,
        target_expert_used=False,
    )
    persist_json(root / "manifests/action_library.json", seal)
    return seal


def persist_physical_prelabel(
    root: Path,
    *,
    prediction: object,
    probability_index: Sequence[object],
    probability_surface_hash: str,
) -> Mapping[str, object]:
    rows = persist_rows(root / "tables/exact_nine_probability_index.csv", probability_index)
    seal = seal_payload(
        "fixed_bank_cdca_physical_prelabel_seal_v1",
        bindings={
            "global_prediction_seal_hash": str(getattr(prediction, "seal_hash")),
            "prediction_store_hash": str(getattr(prediction, "store").store_hash),
            "probability_surface_hash": probability_surface_hash,
            "probability_index_hash": canonical_hash(rows),
        },
        physical_cell_count=len(getattr(prediction, "store").cells),
        target_action_index_count=len(rows),
        labels_used=False,
        sealed_before_any_label_capability=True,
    )
    persist_json(root / "manifests/physical_prelabel_seal.json", seal)
    return seal


def persist_plans_and_features(
    root: Path,
    *,
    plans: Sequence[object],
    plan_seal: object,
    features: Sequence[object],
    physical_prelabel_seal_hash: str,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    plan_rows = persist_rows(root / "tables/held_case_plans.csv", plans)
    feature_rows = persist_rows(root / "tables/held_case_features.csv", features)
    plan_payload = object_payload(plan_seal)
    if plan_payload.get("plan_seal_hash") is None:
        raise ProtocolError("Case-directional global plan seal is absent.")
    persisted_plan = seal_payload(
        "fixed_bank_cdca_held_case_plan_seal_v1",
        bindings={
            "physical_prelabel_seal_hash": physical_prelabel_seal_hash,
            "plans_hash": canonical_hash(plan_rows),
            "science_plan_seal_hash": plan_payload["plan_seal_hash"],
        },
        plan_count=len(plan_rows),
        held_case_excluded=True,
        labels_used=False,
    )
    persist_json(root / "manifests/held_case_plan_seal.json", persisted_plan)
    feature_seal = seal_payload(
        "fixed_bank_cdca_held_case_feature_seal_v1",
        bindings={
            "physical_prelabel_seal_hash": physical_prelabel_seal_hash,
            "held_case_plan_seal_hash": persisted_plan["seal_hash"],
            "held_case_features_hash": canonical_hash(feature_rows),
        },
        feature_count=len(feature_rows),
        labels_used=False,
        feature_schema_is_label_free=True,
        signed_delta_is_candidate_probability_minus_B=True,
    )
    persist_json(root / "manifests/held_case_feature_seal.json", feature_seal)
    return persisted_plan, feature_seal


def persist_route_science(
    root: Path,
    *,
    support_responses: Sequence[object],
    donor_priors: Sequence[object],
    model_fits: Sequence[object],
    candidate_scores: Sequence[object],
    decisions: Sequence[object],
    method_predictions: Sequence[object],
    descriptive_predictions: Sequence[object],
    held_case_plan_seal_hash: str,
    held_case_feature_seal_hash: str,
    route_barrier: Mapping[str, object],
) -> tuple[Mapping[str, object], Mapping[str, object], Mapping[str, object], Mapping[str, object]]:
    response_rows = persist_rows(root / "tables/support_response_counts.csv", support_responses)
    prior_rows = persist_rows(root / "tables/donor_priors.csv", donor_priors)
    fit_rows = persist_rows(root / "tables/route_model_fits.csv", model_fits)
    score_rows = persist_rows(root / "tables/route_candidate_scores.csv", candidate_scores)
    decision_rows = persist_rows(root / "tables/route_decisions.csv", decisions)
    prediction_rows = persist_rows(root / "tables/method_predictions.csv", method_predictions)
    descriptive_rows = persist_rows(
        root / "tables/descriptive_method_predictions.csv", descriptive_predictions
    )
    prior_seal = seal_payload(
        "fixed_bank_cdca_donor_prior_seal_v1",
        bindings={
            "held_case_plan_seal_hash": held_case_plan_seal_hash,
            "donor_priors_hash": canonical_hash(prior_rows),
        },
        donor_prior_count=len(prior_rows),
        donor_scope="q_not_in_H_or_e",
    )
    model_seal = seal_payload(
        "fixed_bank_cdca_route_model_seal_v1",
        bindings={
            "held_case_feature_seal_hash": held_case_feature_seal_hash,
            "donor_prior_seal_hash": prior_seal["seal_hash"],
            "support_responses_hash": canonical_hash(response_rows),
            "route_model_fits_hash": canonical_hash(fit_rows),
            "route_candidate_scores_hash": canonical_hash(score_rows),
        },
        model_fit_count=len(fit_rows),
        every_fit_is_H_minus_c=True,
        route_local_state_not_shared=True,
    )
    decision_seal = seal_payload(
        "fixed_bank_cdca_route_decision_seal_v1",
        bindings={
            "route_model_seal_hash": model_seal["seal_hash"],
            "route_decisions_hash": canonical_hash(decision_rows),
            "method_predictions_hash": canonical_hash(prediction_rows),
            "descriptive_predictions_hash": canonical_hash(descriptive_rows),
            "route_decision_barrier_hash": route_barrier["decision_barrier_hash"],
        },
        route_direction_decision_count=len(decision_rows),
        final_predictions_sealed=True,
        terminal_labels_used=False,
    )
    aggregate_seal = seal_payload(
        "fixed_bank_cdca_aggregate_plan_decision_seal_v1",
        bindings={
            "held_case_plan_seal_hash": held_case_plan_seal_hash,
            "held_case_feature_seal_hash": held_case_feature_seal_hash,
            "donor_prior_seal_hash": prior_seal["seal_hash"],
            "route_model_seal_hash": model_seal["seal_hash"],
            "route_decision_seal_hash": decision_seal["seal_hash"],
            "route_decision_barrier_hash": route_barrier["decision_barrier_hash"],
        },
        route_count=218,
        all_route_probabilities_and_decisions_sealed=True,
        terminal_labels_used=False,
    )
    for name, payload in (
        ("donor_prior_seal.json", prior_seal),
        ("route_model_seal.json", model_seal),
        ("route_decision_seal.json", decision_seal),
        ("aggregate_plan_decision_seal.json", aggregate_seal),
    ):
        persist_json(root / "manifests" / name, payload)
    return prior_seal, model_seal, decision_seal, aggregate_seal


def persist_terminal(
    root: Path,
    *,
    result: Mapping[str, object],
    capability_report: Mapping[str, object],
    leakage_report: Mapping[str, object],
    publication_decision: Mapping[str, object],
    runtime_summary: Mapping[str, object],
) -> None:
    tables = {
        "terminal_case_confusions.csv": "case_confusions",
        "terminal_method_metrics.csv": "method_metrics",
        "terminal_center_metrics.csv": "center_metrics",
        "terminal_contrasts.csv": "contrasts",
        "router_identification_metrics.csv": "router_identification",
        "feature_permutation_summary.csv": "feature_permutation_summary",
    }
    for filename, key in tables.items():
        rows = result.get(key)
        if not isinstance(rows, (tuple, list)):
            raise ProtocolError(f"Case-directional terminal surface absent: {key}.")
        persist_rows(root / "tables" / filename, rows)
    terminal_seal = result.get("terminal_seal")
    if not isinstance(terminal_seal, Mapping):
        raise ProtocolError("Case-directional terminal seal absent.")
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
    if not isinstance(converted, dict) or "schema_version" in converted:
        raise ProtocolError("Case-directional validation report malformed.")
    reconstructed = {
        key: value for key, value in converted.items() if key != ATTESTATION_KEY
    }
    verified = verify_attested_validation_checks(
        converted,
        expected_reconstructed_checks=reconstructed,
    )
    if verified != converted:
        raise ProtocolError("Case-directional validation report attestation drifted.")
    payload = {
        "schema_version": "fixed_bank_cdca_validation_report_v1",
        **converted,
    }
    persist_json(root / "reports/validation_report.json", payload)


def write_run_state(
    root: Path,
    *,
    status: str,
    phase: str,
    error: str | None = None,
    error_class: str | None = None,
) -> None:
    path = root / "reports/run_state.json"
    if path.is_symlink():
        raise ProtocolError("Case-directional run state is a symlink.")
    atomic_json(path, run_state_payload(status, phase, error=error, error_class=error_class))


def _csv_bytes(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> bytes:
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                field: json.dumps(row[field], sort_keys=True, separators=(",", ":"))
                for field in fields
            }
        )
    return buffer.getvalue().encode("utf-8")


def _reject_forbidden_keys(value: object) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).casefold() in _FORBIDDEN_PERSISTED_KEYS:
                raise ProtocolError(f"Forbidden case-directional key: {key}.")
            _reject_forbidden_keys(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_forbidden_keys(nested)


__all__ = (
    "object_payload",
    "persist_initial_surfaces",
    "persist_json",
    "persist_physical_prelabel",
    "persist_plans_and_features",
    "persist_route_science",
    "persist_rows",
    "persist_terminal",
    "persist_validation_report",
    "read_rows",
    "write_run_state",
)
