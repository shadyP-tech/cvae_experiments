from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_labeled_support_case_conditional_flip_router.label_capabilities import (
    FlipRouterLabelCapabilityManager,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_labeled_support_case_conditional_flip_router.persistence import (
    TERMINAL_TABLE_FIELDS,
    finalize_terminal_checkpoint,
    persist_decisions,
    persist_donor_models,
    persist_fold_plans,
    persist_static_and_calibration,
    persist_terminal,
    persist_terminal_checkpoint,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_labeled_support_case_conditional_flip_router.science_runtime import (
    build_fold_decision_phase,
    evaluate_terminal_phase,
    fit_h_specific_donor_phase,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_labeled_support_case_conditional_flip_router.validation_science_replay import (
    replay_label_aware_surfaces,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import atomic_json, read_json

from flip_router_science_fixture import build_science_fixture


_TERMINAL_TABLE_MEMBERS = {
    key: f"tables/{key}.csv" for key in TERMINAL_TABLE_FIELDS
}


def _serial_process_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    class SerialExecutor:
        def __init__(self, *args: object, initializer=None, initargs=(), **kwargs: object):
            if initializer is not None:
                initializer(*initargs)

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def map(self, function, values):
            return map(function, values)

    monkeypatch.setattr(
        "midogpp_thesis.cvae.diagnostics.fixed_bank_labeled_support_case_conditional_flip_router.science_donor.ProcessPoolExecutor",
        SerialExecutor,
    )
    monkeypatch.setattr(
        "midogpp_thesis.cvae.diagnostics.fixed_bank_labeled_support_case_conditional_flip_router.science_terminal.ProcessPoolExecutor",
        SerialExecutor,
    )


def _materialize(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    through_terminal_checkpoint: bool = False,
):
    _serial_process_pool(monkeypatch)
    fixture = build_science_fixture(root, monkeypatch)
    manager = FlipRouterLabelCapabilityManager(
        fixture.config.test_manifest_path,
        fixture.frame,
        fixture.partition,
        prediction_seal_hash=fixture.prediction.seal_hash,
        feature_seal_hash=fixture.prelabel.feature_surface_hash,
    )
    plans = manager.seal_all_fold_plans()
    persist_fold_plans(root, plans)
    donor = fit_h_specific_donor_phase(
        probability_surface=fixture.probability,
        prelabel=fixture.prelabel,
        partition=fixture.partition,
        manager=manager,
        config=fixture.config,
    )
    persist_donor_models(
        root,
        contribution_targets=donor.contribution_targets,
        models=donor.models,
        seals=donor.seals,
        permutation_payload=donor.permutation_payload,
    )
    decisions = build_fold_decision_phase(
        probability_surface=fixture.probability,
        prelabel=fixture.prelabel,
        partition=fixture.partition,
        manager=manager,
        donor_phase=donor,
        config=fixture.config,
    )
    persist_static_and_calibration(
        root,
        static_rows=decisions.static_rows,
        calibration_rows=decisions.calibration_rows,
        static_seal_payload=decisions.static_seal_payload,
        calibration_seal_payload=decisions.calibration_seal_payload,
    )
    persist_decisions(root, decisions.bundle)
    terminal = evaluate_terminal_phase(
        probability_surface=fixture.probability,
        partition=fixture.partition,
        terminal_labels=manager.open_terminal_evaluation_labels(),
        decision_phase=decisions,
        config=fixture.config,
    )
    reports = {
        "capability_report": manager.report_payload(),
        "leakage_report": {"status": "fixture"},
        "publication_decision": {"status": "fixture"},
        "runtime_summary": {"status": "fixture"},
    }
    if through_terminal_checkpoint:
        persist_terminal_checkpoint(root, result=terminal, **reports)
        finalize_terminal_checkpoint(root)
    else:
        persist_terminal(root, result=terminal, **reports)
    return fixture, donor, decisions, terminal


def _replay(root: Path, fixture: object):
    return replay_label_aware_surfaces(
        root,
        config=fixture.config,
        frame=fixture.frame,
        partition=fixture.partition,
        prediction=fixture.prediction,
        probability_surface=fixture.probability,
        prelabel=fixture.prelabel,
    )


def test_science_phases_persist_and_replay_every_label_aware_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, donor, decisions, terminal = _materialize(tmp_path, monkeypatch)
    result = _replay(tmp_path, fixture)

    assert len(donor.models) == 9
    assert all(
        row["global_static_query_fixed_effect_fit"]["design_rank"] == 15
        and row["global_static_query_fixed_effect_fit"]["required_rank"] == 15
        and row["global_static_query_fixed_effect_fit"]["observation_count"] == 56
        and len(row["global_static_query_fixed_effect_fit"]["candidate_sources"]) == 8
        for row in donor.models
    )
    assert len(decisions.static_rows) == 45
    assert len(decisions.calibration_rows) == 45
    assert len(decisions.bundle.decisions) == 315
    assert all(
        row["F_G"] == row["F_S"]
        and row["ordinary_calibration_shared_by_F_G_and_F_S"] is True
        and len(row["calibration_action_ids"]) == 8
        for row in decisions.calibration_rows
    )
    assert len(terminal["terminal_case_confusions"]) == 405
    assert len(terminal["terminal_center_metrics"]) == 81
    assert len(terminal["terminal_contrasts"]) == 50
    gate = terminal["sealed_terminal_evaluation"]["diagnostic_recoverability_gate"]
    assert gate["status"] in {"PASS", "FAIL"}
    assert set(gate["contrast_one_sided_95_lcb"]) == {
        "F_S-B", "F_S-U", "F_S-F_G", "F_S-F_P", "F_S-S_static"
    }
    assert gate["routing_success_claimed"] is False
    assert gate["promotion_eligible"] is False
    assert result["label_aware_scientific_replay"] == "PASS"


def test_terminal_checkpoint_round_trip_preserves_canonical_csv_schema_and_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the runner's JSON-checkpoint -> CSV-finalization boundary."""

    fixture, _donor, _decisions, terminal = _materialize(
        tmp_path, monkeypatch, through_terminal_checkpoint=True
    )
    expected_by_table = {
        key: tuple(dict(row) for row in terminal[key])
        for key in TERMINAL_TABLE_FIELDS
    }
    for key, fields in TERMINAL_TABLE_FIELDS.items():
        assert set(expected_by_table[key][0]) == set(fields)

    terminal_seal = read_json(
        tmp_path / "manifests/sealed_terminal_evaluation.json"
    )
    for key, fields in TERMINAL_TABLE_FIELDS.items():
        member = _TERMINAL_TABLE_MEMBERS[key]
        with (tmp_path / member).open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            reader = csv.DictReader(handle)
            observed_fields = tuple(reader.fieldnames or ())
            observed_rows = tuple(dict(row) for row in reader)
        expected_rows = expected_by_table[key]
        persisted_rows = tuple(
            {field: _persisted_csv_cell(row[field]) for field in fields}
            for row in expected_rows
        )
        assert observed_fields == fields
        assert observed_rows == persisted_rows
        assert tuple(row["row_hash"] for row in observed_rows) == tuple(
            row["row_hash"] for row in expected_rows
        )
        assert terminal_seal["table_hashes"][key] == terminal[
            "sealed_terminal_evaluation"
        ]["table_hashes"][key]
    assert _replay(tmp_path, fixture)["label_aware_scientific_replay"] == "PASS"


def _persisted_csv_cell(value: object) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return "" if value is None else str(value)


@pytest.mark.parametrize(
    ("member", "field"),
    (
        ("tables/donor_contribution_targets.csv", "delta_tp"),
        ("tables/model_fits.csv", "ordinary_model"),
        ("tables/directional_calibrations.csv", "F_S"),
        ("tables/method_decisions.csv", "action_id"),
        ("tables/terminal_case_confusions.csv", "tp"),
        ("tables/terminal_center_metrics.csv", "bacc"),
        ("tables/terminal_contrasts.csv", "replicates"),
    ),
)
def test_replay_rejects_scientific_tamper_even_without_trusting_outer_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    member: str,
    field: str,
) -> None:
    fixture, *_ = _materialize(tmp_path, monkeypatch)
    path = tmp_path / member
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    assert rows and field in fields
    original = rows[0][field]
    rows[0][field] = _tampered_cell(field, original)
    assert rows[0][field] != original
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ProtocolError, match="replayed table differs|header drifted"):
        _replay(tmp_path, fixture)


def _tampered_cell(field: str, value: str) -> str:
    if field in {"ordinary_model", "F_S"}:
        return value.replace('"schema_version"', '"tampered":true,"schema_version"', 1)
    if field == "action_id":
        return "U" if value != "U" else "B"
    if field == "bacc":
        return str(float(value) + 0.125)
    return str(int(value) + 1)


def test_replay_rejects_rehashed_terminal_or_decision_seal_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, *_ = _materialize(tmp_path, monkeypatch)
    terminal_path = tmp_path / "manifests/sealed_terminal_evaluation.json"
    terminal = read_json(terminal_path)
    terminal["case_confusion_row_count"] += 1
    atomic_json(terminal_path, terminal)

    with pytest.raises(ProtocolError, match="replayed JSON differs"):
        _replay(tmp_path, fixture)
