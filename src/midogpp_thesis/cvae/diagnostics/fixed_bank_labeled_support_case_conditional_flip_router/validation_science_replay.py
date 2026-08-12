"""Exact replay of every label-aware scientific phase.

The durable bundle is treated only as a claim to verify.  This module opens a
fresh set of typed label capabilities, reruns the public scientific facade,
and compares every persisted row and seal with the independently reconstructed
result.  No generated report is trusted as an input to the replay.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from .artifact_io import json_value, object_payload, read_rows
from .constants import CENTERS
from .hashing import canonical_hash
from .label_capabilities import FlipRouterLabelCapabilityManager
from .science_runtime import (
    build_fold_decision_phase,
    evaluate_terminal_phase,
    fit_h_specific_donor_phase,
)


_TERMINAL_TABLES = {
    "terminal_case_confusions": "tables/terminal_case_confusions.csv",
    "terminal_center_metrics": "tables/terminal_center_metrics.csv",
    "terminal_contrasts": "tables/terminal_contrasts.csv",
    "router_identification_metrics": "tables/router_identification_metrics.csv",
    "permutation_metrics": "tables/permutation_metrics.csv",
}


def replay_label_aware_surfaces(
    root: Path,
    *,
    config: object,
    frame: object,
    partition: object,
    prediction: object,
    probability_surface: object,
    prelabel: object,
) -> Mapping[str, object]:
    """Recompute the label-aware pipeline and exact-compare its durable form."""

    manager = FlipRouterLabelCapabilityManager(
        Path(getattr(config, "test_manifest_path")),
        frame,
        partition,
        prediction_seal_hash=str(getattr(prediction, "seal_hash")),
        feature_seal_hash=str(getattr(prelabel, "feature_surface_hash")),
    )
    plans = manager.seal_all_fold_plans()
    _assert_json(root / "manifests/fold_plan_seals.json", _fold_plan_seal(plans))

    donor = fit_h_specific_donor_phase(
        probability_surface=probability_surface,
        prelabel=prelabel,
        partition=partition,
        manager=manager,
        config=config,
    )
    _assert_table(
        root / "tables/donor_contribution_targets.csv",
        donor.contribution_targets,
    )
    _assert_table(root / "tables/model_fits.csv", donor.models)
    _assert_json(
        root / "manifests/donor_model_seals.json",
        {
            "schema_version": "fixed_bank_labeled_support_flip_donor_model_seals_v1",
            "models": {
                key: dict(value) for key, value in sorted(donor.seals.items())
            },
            "model_count": len(donor.seals),
            "models_are_H_specific": True,
            "heldout_H_labels_used": False,
        },
    )
    _assert_json(
        root / "manifests/permutation_provenance_seal.json",
        donor.permutation_payload,
    )

    decisions = build_fold_decision_phase(
        probability_surface=probability_surface,
        prelabel=prelabel,
        partition=partition,
        manager=manager,
        donor_phase=donor,
        config=config,
    )
    _assert_table(
        root / "tables/static_source_selections.csv", decisions.static_rows
    )
    _assert_table(
        root / "tables/directional_calibrations.csv", decisions.calibration_rows
    )
    _assert_json(
        root / "manifests/static_selection_seals.json",
        decisions.static_seal_payload,
    )
    _assert_json(
        root / "manifests/calibration_seals.json",
        decisions.calibration_seal_payload,
    )
    _assert_table(
        root / "tables/method_decisions.csv", decisions.bundle.decisions
    )
    _assert_json(
        root / "manifests/all_method_decisions_seal.json",
        _decision_seal(decisions.bundle),
    )

    terminal_labels = manager.open_terminal_evaluation_labels()
    terminal = evaluate_terminal_phase(
        probability_surface=probability_surface,
        partition=partition,
        terminal_labels=terminal_labels,
        decision_phase=decisions,
        config=config,
    )
    for key, member in _TERMINAL_TABLES.items():
        rows = terminal.get(key)
        if not _rows(rows):
            raise ProtocolError(f"Replayed flip-router terminal table is absent: {key}.")
        _assert_table(root / member, rows)
    sealed = terminal.get("sealed_terminal_evaluation")
    if not isinstance(sealed, Mapping):
        raise ProtocolError("Replayed flip-router terminal seal is absent.")
    _assert_json(root / "manifests/sealed_terminal_evaluation.json", sealed)
    capability = manager.report_payload()
    _assert_json(root / "reports/label_capability_report.json", capability)

    return {
        "fold_plan_count": len(plans),
        "donor_contribution_target_count": len(donor.contribution_targets),
        "H_specific_model_count": len(donor.models),
        "static_selection_count": len(decisions.static_rows),
        "directional_calibration_count": len(decisions.calibration_rows),
        "method_decision_count": len(decisions.bundle.decisions),
        "terminal_case_confusion_count": len(terminal["terminal_case_confusions"]),
        "terminal_center_metric_count": len(terminal["terminal_center_metrics"]),
        "terminal_contrast_count": len(terminal["terminal_contrasts"]),
        "router_identification_metric_count": len(
            terminal["router_identification_metrics"]
        ),
        "permutation_metric_count": len(terminal["permutation_metrics"]),
        "decision_bundle_hash": decisions.bundle.decision_bundle_hash,
        "sealed_result_hash": sealed["sealed_result_hash"],
        "label_aware_scientific_replay": "PASS",
    }


def _fold_plan_seal(plans: Sequence[object]) -> Mapping[str, object]:
    rows = [object_payload(plan) for plan in plans]
    unhashed = {
        "schema_version": "fixed_bank_labeled_support_flip_fold_plan_seals_v1",
        "plans": rows,
        "plan_count": len(rows),
        "held_evaluation_labels_used": False,
        "each_plan_invariant_to_held_evaluation_label_values": True,
    }
    return {**unhashed, "fold_plan_surface_hash": canonical_hash(unhashed)}


def _decision_seal(bundle: object) -> Mapping[str, object]:
    decisions = tuple(getattr(bundle, "decisions"))
    fold_seals = dict(getattr(bundle, "fold_seal_hashes"))
    expected_keys = {(center, fold) for center in CENTERS for fold in range(5)}
    if set(fold_seals) != expected_keys:
        raise ProtocolError("Replayed flip-router fold-decision topology drifted.")
    return {
        "schema_version": "fixed_bank_labeled_support_flip_all_decisions_v1",
        "decision_count": len(decisions),
        "fold_seals": {
            f"{key[0]}::{key[1]}": value
            for key, value in sorted(fold_seals.items())
        },
        "fold_seal_count": len(fold_seals),
        "decision_bundle_hash": getattr(bundle, "decision_bundle_hash"),
        "each_fold_decision_without_its_held_evaluation_labels": True,
        "terminal_evaluation_labels_used": False,
    }


def _assert_json(path: Path, expected: object) -> None:
    converted = json_value(expected)
    if not isinstance(converted, Mapping) or read_json(path) != dict(converted):
        raise ProtocolError(f"Flip-router replayed JSON differs: {path}.")


def _assert_table(path: Path, expected_rows: Sequence[object]) -> None:
    payloads = tuple(object_payload(row) for row in expected_rows)
    if not payloads:
        raise ProtocolError(f"Flip-router replay produced an empty table: {path}.")
    fields = tuple(payloads[0])
    if any(tuple(row) != fields for row in payloads):
        raise ProtocolError(f"Flip-router replay table schema is ragged: {path}.")
    observed = read_rows(path)
    expected = tuple(
        {field: _persisted_cell(row[field]) for field in fields} for row in payloads
    )
    if any(tuple(row) != fields for row in observed):
        raise ProtocolError(f"Flip-router persisted table header drifted: {path}.")
    if observed != expected:
        raise ProtocolError(f"Flip-router replayed table differs: {path}.")


def _persisted_cell(value: object) -> str:
    converted = json_value(value)
    if isinstance(converted, (dict, list)):
        return json.dumps(converted, sort_keys=True, separators=(",", ":"))
    return "" if converted is None else str(converted)


def _rows(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and bool(value)


__all__ = ("replay_label_aware_surfaces",)
