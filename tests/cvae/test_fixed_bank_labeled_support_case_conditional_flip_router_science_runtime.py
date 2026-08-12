from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

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
from midogpp_thesis.cvae.diagnostics.fixed_bank_labeled_support_case_conditional_flip_router.hashing import (
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_labeled_support_case_conditional_flip_router.constants import (
    CENTERS,
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


def test_scientific_replay_succeeds_in_two_independent_fresh_validation_processes(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    python_paths = (repository / "src", repository / "tests/cvae")
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(python_paths[0]), str(python_paths[1]), environment.get("PYTHONPATH", ""))
    )
    materialize = """
from pathlib import Path
import sys
from pytest import MonkeyPatch
from test_fixed_bank_labeled_support_case_conditional_flip_router_science_runtime import _materialize
root = Path(sys.argv[1])
patch = MonkeyPatch()
_materialize(root, patch)
patch.undo()
"""
    replay = """
from pathlib import Path
import sys
from pytest import MonkeyPatch
from flip_router_science_fixture import build_science_fixture
from test_fixed_bank_labeled_support_case_conditional_flip_router_science_runtime import _replay, _serial_process_pool
root = Path(sys.argv[1])
patch = MonkeyPatch()
_serial_process_pool(patch)
fixture = build_science_fixture(root, patch)
assert _replay(root, fixture)["label_aware_scientific_replay"] == "PASS"
patch.undo()
"""
    # Materialization is intentionally separate from both validators.  Running
    # the replay script twice proves that two independent interpreter/solver
    # processes accept the same persisted bundle.
    for script in (materialize, replay, replay):
        completed = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path)],
            cwd=repository,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr


def _persisted_csv_cell(value: object) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return "" if value is None else str(value)


def _csv_rows(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return tuple(reader.fieldnames or ()), list(reader)


def _write_csv_rows(
    path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _canonical_cell(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _derived_selection(adjusted: dict[str, float]) -> dict[str, object]:
    ranked = sorted(
        adjusted.items(), key=lambda item: (-item[1], f"A1::source={item[0]}")
    )
    best_source, best_gain = ranked[0]
    if best_gain <= 0.0:
        selection: dict[str, object] = {
            "schema_version": "threshold_flip_case_router_core_v1",
            "action_id": "B",
            "exact_gain": 0.0,
            "runner_up_gain": max(0.0, best_gain),
            "fallback_to_b": True,
        }
    else:
        selection = {
            "schema_version": "threshold_flip_case_router_core_v1",
            "action_id": f"A1::source={best_source}",
            "exact_gain": best_gain,
            "runner_up_gain": max(0.0, ranked[1][1]),
            "fallback_to_b": False,
        }
    return {**selection, "selection_hash": canonical_hash(selection)}


def _typed_model_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    nested = {
        "ordinary_model",
        "permutation_model",
        "global_static_selection",
        "global_static_query_fixed_effect_fit",
    }
    return [
        {
            key: json.loads(value) if key in nested else value
            for key, value in row.items()
        }
        for row in rows
    ]


def _typed_static_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    return [
        {
            **row,
            "fold_ordinal": int(row["fold_ordinal"]),
            "selection_case_ids": json.loads(row["selection_case_ids"]),
            "G_static": json.loads(row["G_static"]),
            "S_static": json.loads(row["S_static"]),
        }
        for row in rows
    ]


def _typed_calibration_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in rows:
        typed: dict[str, object] = dict(row)
        for field in (
            "fold_ordinal",
            "calibration_n_positive",
            "calibration_n_negative",
        ):
            typed[field] = int(row[field])
        for field in ("calibration_case_ids", "calibration_action_ids", "F_G", "F_S", "F_P"):
            typed[field] = json.loads(row[field])
        for field in (
            "ordinary_calibration_shared_by_F_G_and_F_S",
            "permutation_calibration_same_capacity_all_actions",
        ):
            typed[field] = row[field] == "True"
        output.append(typed)
    return output


def _typed_decision_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    return [
        {
            **row,
            "fold_ordinal": int(row["fold_ordinal"]),
            "predicted_gain": float(row["predicted_gain"]),
            "gain_standard_error": float(row["gain_standard_error"]),
            "lower_confidence_bound": float(row["lower_confidence_bound"]),
            "evaluation_labels_used": row["evaluation_labels_used"] == "True",
        }
        for row in rows
    ]


def _decision_hash_graph(
    fixture: object,
    *,
    model_rows: list[dict[str, object]],
    static_rows: list[dict[str, object]],
    calibration_rows: list[dict[str, object]],
    decisions: list[dict[str, object]],
) -> dict[str, object]:
    models = {str(row["heldout_target_H"]): row for row in model_rows}
    static = {
        (str(row["target_center"]), int(row["fold_ordinal"])): row
        for row in static_rows
    }
    calibration = {
        (str(row["target_center"]), int(row["fold_ordinal"])): row
        for row in calibration_rows
    }
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in decisions:
        grouped.setdefault(
            (str(row["target_center"]), int(row["fold_ordinal"])), []
        ).append(row)
    fold_seals: dict[tuple[str, int], str] = {}
    for target in CENTERS:
        for fold_ordinal in range(5):
            key = (target, fold_ordinal)
            model = models[target]
            fold = fixture.partition.fold(target, fold_ordinal)
            payload = {
                "schema_version": "fixed_bank_flip_router_fold_decision_seal_v1",
                "target_center": target,
                "fold_ordinal": fold_ordinal,
                "evaluation_case_ids": list(fold.evaluation_case_ids),
                "ordinary_model_hash": model["ordinary_model"]["model_hash"],
                "permutation_model_hash": model["permutation_model"]["model_hash"],
                "static_row_hash": static[key]["row_hash"],
                "calibration_row_hash": calibration[key]["row_hash"],
                "decisions": grouped[key],
                "held_evaluation_labels_used": False,
            }
            fold_seals[key] = canonical_hash(payload)
    serialized_seals = {
        f"{key[0]}::{key[1]}": value for key, value in sorted(fold_seals.items())
    }
    bundle_payload = {
        "schema_version": "fixed_bank_flip_router_decision_bundle_v1",
        "decisions": decisions,
        "fold_seals": serialized_seals,
        "evaluation_labels_used": False,
    }
    return {
        "schema_version": "fixed_bank_labeled_support_flip_all_decisions_v1",
        "decision_count": len(decisions),
        "fold_seals": serialized_seals,
        "fold_seal_count": len(fold_seals),
        "decision_bundle_hash": canonical_hash(bundle_payload),
        "each_fold_decision_without_its_held_evaluation_labels": True,
        "terminal_evaluation_labels_used": False,
    }


def _coherently_shift_query_fixed_effect_graph(
    root: Path, fixture: object, *, delta: float
) -> None:
    """Rewrite every hash/copy downstream of one tiny solve-output change."""

    model_path = root / "tables/model_fits.csv"
    model_fields, model_rows = _csv_rows(model_path)
    row = model_rows[0]
    target = row["heldout_target_H"]
    fit = json.loads(row["global_static_query_fixed_effect_fit"])
    selected_source = str(fit["selection"]["action_id"]).split("=", 1)[1]
    compensator = next(
        source
        for source in fit["candidate_sources"]
        if source != selected_source
    )
    # Preserve the sum-to-zero constraint while changing only solve-derived
    # coefficients.  Recompute every copied value rather than tolerating it
    # independently.
    fit["source_effects"][selected_source] += delta
    fit["source_effects"][compensator] -= delta
    for source in (selected_source, compensator):
        fit["adjusted_source_gains"][source] = (
            fit["grand_mean"] + fit["source_effects"][source]
        )
    selection = _derived_selection(fit["adjusted_source_gains"])
    assert selection["action_id"] == fit["selection"]["action_id"]
    fit["selection"] = selection
    unhashed_fit = dict(fit)
    unhashed_fit.pop("fit_hash")
    fit["fit_hash"] = canonical_hash(unhashed_fit)
    row["global_static_selection"] = _canonical_cell(selection)
    row["global_static_query_fixed_effect_fit"] = _canonical_cell(fit)

    donor_path = root / "manifests/donor_model_seals.json"
    donor = read_json(donor_path)
    seal = donor["models"][target]
    seal["global_static_selection"] = selection
    seal["global_static_query_fixed_effect_fit"] = fit
    unhashed_seal = dict(seal)
    unhashed_seal.pop("seal_hash")
    seal["seal_hash"] = canonical_hash(unhashed_seal)
    row["model_seal_hash"] = seal["seal_hash"]
    atomic_json(donor_path, donor)
    _write_csv_rows(model_path, model_fields, model_rows)

    static_path = root / "tables/static_source_selections.csv"
    static_fields, static_rows = _csv_rows(static_path)
    for static_row in static_rows:
        if static_row["target_center"] != target:
            continue
        static_row["G_static"] = _canonical_cell(selection)
        payload = {
            "target_center": static_row["target_center"],
            "fold_ordinal": int(static_row["fold_ordinal"]),
            "selection_case_ids": json.loads(static_row["selection_case_ids"]),
            "selection_label_identity_hash": static_row[
                "selection_label_identity_hash"
            ],
            "ordinary_model_hash": static_row["ordinary_model_hash"],
            "G_static": selection,
            "S_static": json.loads(static_row["S_static"]),
        }
        static_row["row_hash"] = canonical_hash(payload)
    _write_csv_rows(static_path, static_fields, static_rows)
    static_manifest_path = root / "manifests/static_selection_seals.json"
    static_manifest = read_json(static_manifest_path)
    static_manifest["rows"] = _typed_static_rows(static_rows)
    unhashed_static = dict(static_manifest)
    unhashed_static.pop("static_selection_surface_hash")
    static_manifest["static_selection_surface_hash"] = canonical_hash(
        unhashed_static
    )
    atomic_json(static_manifest_path, static_manifest)

    decision_path = root / "tables/method_decisions.csv"
    decision_fields, decision_rows = _csv_rows(decision_path)
    for decision in decision_rows:
        if decision["target_center"] == target and decision["method_id"] == "G_static":
            decision["predicted_gain"] = str(selection["exact_gain"])
            decision["lower_confidence_bound"] = str(selection["exact_gain"])
    _write_csv_rows(decision_path, decision_fields, decision_rows)

    _, calibration_rows = _csv_rows(root / "tables/directional_calibrations.csv")
    decision_seal = _decision_hash_graph(
        fixture,
        model_rows=_typed_model_rows(model_rows),
        static_rows=_typed_static_rows(static_rows),
        calibration_rows=_typed_calibration_rows(calibration_rows),
        decisions=_typed_decision_rows(decision_rows),
    )
    atomic_json(root / "manifests/all_method_decisions_seal.json", decision_seal)

    terminal_path = root / "manifests/sealed_terminal_evaluation.json"
    terminal = read_json(terminal_path)
    terminal["decision_bundle_hash"] = decision_seal["decision_bundle_hash"]
    unhashed_terminal = dict(terminal)
    unhashed_terminal.pop("sealed_result_hash")
    terminal["sealed_result_hash"] = canonical_hash(unhashed_terminal)
    atomic_json(terminal_path, terminal)


def test_replay_accepts_only_coherent_sub_tolerance_query_fit_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, *_ = _materialize(tmp_path, monkeypatch)
    _coherently_shift_query_fixed_effect_graph(
        tmp_path, fixture, delta=5.0e-16
    )

    result = _replay(tmp_path, fixture)

    persisted_terminal = read_json(
        tmp_path / "manifests/sealed_terminal_evaluation.json"
    )
    persisted_decisions = read_json(
        tmp_path / "manifests/all_method_decisions_seal.json"
    )
    assert result["label_aware_scientific_replay"] == "PASS"
    assert result["decision_bundle_hash"] == persisted_decisions[
        "decision_bundle_hash"
    ]
    assert result["sealed_result_hash"] == persisted_terminal["sealed_result_hash"]


def test_replay_rejects_coherently_rehashed_query_fit_drift_above_tolerance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, *_ = _materialize(tmp_path, monkeypatch)
    _coherently_shift_query_fixed_effect_graph(
        tmp_path, fixture, delta=2.0e-15
    )

    with pytest.raises(ProtocolError, match="exceeds replay tolerance"):
        _replay(tmp_path, fixture)


def test_replay_rejects_rehashed_query_fit_categorical_selection_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, *_ = _materialize(tmp_path, monkeypatch)
    model_path = tmp_path / "tables/model_fits.csv"
    fields, rows = _csv_rows(model_path)
    row = rows[0]
    target = row["heldout_target_H"]
    fit = json.loads(row["global_static_query_fixed_effect_fit"])
    selection = fit["selection"]
    selection["action_id"] = next(
        f"A1::source={source}"
        for source in fit["candidate_sources"]
        if f"A1::source={source}" != selection["action_id"]
    )
    unhashed_selection = dict(selection)
    unhashed_selection.pop("selection_hash")
    selection["selection_hash"] = canonical_hash(unhashed_selection)
    fit["selection"] = selection
    unhashed_fit = dict(fit)
    unhashed_fit.pop("fit_hash")
    fit["fit_hash"] = canonical_hash(unhashed_fit)
    row["global_static_selection"] = _canonical_cell(selection)
    row["global_static_query_fixed_effect_fit"] = _canonical_cell(fit)
    donor_path = tmp_path / "manifests/donor_model_seals.json"
    donor = read_json(donor_path)
    seal = donor["models"][target]
    seal["global_static_selection"] = selection
    seal["global_static_query_fixed_effect_fit"] = fit
    unhashed_seal = dict(seal)
    unhashed_seal.pop("seal_hash")
    seal["seal_hash"] = canonical_hash(unhashed_seal)
    row["model_seal_hash"] = seal["seal_hash"]
    atomic_json(donor_path, donor)
    _write_csv_rows(model_path, fields, rows)

    with pytest.raises(ProtocolError, match="selection.*differs from replay"):
        _replay(tmp_path, fixture)


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

    with pytest.raises(
        ProtocolError,
        match="replayed table differs|header drifted|differs from replay|encoding drifted",
    ):
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

    with pytest.raises(ProtocolError, match="replayed JSON differs|differs from replay"):
        _replay(tmp_path, fixture)
