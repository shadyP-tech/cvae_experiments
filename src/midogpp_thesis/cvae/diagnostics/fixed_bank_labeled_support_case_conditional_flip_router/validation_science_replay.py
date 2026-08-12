"""Exact replay of every label-aware scientific phase.

The durable bundle is treated only as a claim to verify.  This module opens a
fresh set of typed label capabilities, reruns the public scientific facade,
and compares every persisted row and seal with the independently reconstructed
result.  No generated report is trusted as an input to the replay.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...routing.threshold_flip_case_router import StaticSelection
from ...runtime.artifact_io import read_json
from .artifact_io import json_value, object_payload, read_rows
from .constants import CENTERS
from .hashing import canonical_hash
from .label_capabilities import FlipRouterLabelCapabilityManager
from .persistence import TERMINAL_TABLE_FIELDS
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

# The query/source fixed-effect solve is the sole numerically non-bitwise part
# of scientific replay observed across fresh BLAS processes.  The bound is
# deliberately far below any decision margin in this diagnostic and applies
# with zero relative tolerance only to the explicitly enumerated derived
# fields below.  Direct cell gains, models, ranks, topology, and categorical
# selections remain exact.
_QUERY_FIXED_EFFECT_ATOL = 1.0e-15
_QUERY_FIXED_EFFECT_VECTOR_FIELDS = (
    "query_effects",
    "source_effects",
    "adjusted_source_gains",
)
_QUERY_FIXED_EFFECT_SCALAR_FIELDS = (
    "grand_mean",
    "residual_sum_squares",
)
_STATIC_SELECTION_NUMERIC_FIELDS = ("exact_gain", "runner_up_gain")


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
    observed_model_rows, expected_model_rows = _assert_model_fits_table(
        root / "tables/model_fits.csv", donor.models
    )
    _assert_donor_model_seals(
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
        observed_model_rows=observed_model_rows,
        expected_model_rows=expected_model_rows,
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
        donor_phase=_with_persisted_global_selections(
            donor, observed_model_rows
        ),
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
        _assert_table(
            root / member,
            rows,
            expected_fields=TERMINAL_TABLE_FIELDS[key],
        )
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


def _assert_model_fits_table(
    path: Path,
    expected_rows: Sequence[object],
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    """Semantically compare the sole cross-process floating-point surface."""

    observed, expected = _typed_table_rows(path, expected_rows)
    if tuple(str(row["heldout_target_H"]) for row in observed) != CENTERS:
        raise ProtocolError("Flip-router model-fit target topology drifted.")
    for ordinal, (actual, replayed) in enumerate(zip(observed, expected)):
        role = f"model_fits[{ordinal}]"
        for field in ("ordinary_model", "permutation_model"):
            _require_exact(actual[field], replayed[field], f"{role}.{field}")
        actual_fit = _mapping(actual["global_static_query_fixed_effect_fit"], role)
        replayed_fit = _mapping(
            replayed["global_static_query_fixed_effect_fit"], role
        )
        _assert_query_fixed_effect_fit(actual_fit, replayed_fit, role=role)
        actual_selection = _mapping(actual["global_static_selection"], role)
        replayed_selection = _mapping(replayed["global_static_selection"], role)
        _require_exact(
            actual_selection,
            _mapping(actual_fit["selection"], role),
            f"{role}.global selection copy",
        )
        _require_exact(
            replayed_selection,
            _mapping(replayed_fit["selection"], role),
            f"{role}.replayed global selection copy",
        )
        normalized = deepcopy(actual)
        normalized["global_static_selection"] = deepcopy(replayed_selection)
        normalized["global_static_query_fixed_effect_fit"] = deepcopy(replayed_fit)
        normalized["model_seal_hash"] = replayed["model_seal_hash"]
        _require_exact(normalized, replayed, role)
    return observed, expected


def _assert_query_fixed_effect_fit(
    observed: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    role: str,
) -> None:
    _validate_query_fixed_effect_fit(observed, role=f"persisted {role}")
    _validate_query_fixed_effect_fit(expected, role=f"replayed {role}")
    if set(observed) != set(expected):
        raise ProtocolError(f"Flip-router {role} fit schema drifted.")
    normalized = deepcopy(dict(observed))
    for field in _QUERY_FIXED_EFFECT_SCALAR_FIELDS:
        _require_close_float(observed[field], expected[field], f"{role}.{field}")
        normalized[field] = expected[field]
    for field in _QUERY_FIXED_EFFECT_VECTOR_FIELDS:
        actual = _mapping(observed[field], f"{role}.{field}")
        replayed = _mapping(expected[field], f"{role}.{field}")
        if tuple(actual) != tuple(replayed):
            raise ProtocolError(f"Flip-router {role}.{field} topology drifted.")
        for key in replayed:
            _require_close_float(
                actual[key], replayed[key], f"{role}.{field}.{key}"
            )
        normalized[field] = deepcopy(replayed)
    actual_selection = _mapping(observed["selection"], f"{role}.selection")
    replayed_selection = _mapping(expected["selection"], f"{role}.selection")
    normalized_selection = deepcopy(dict(actual_selection))
    for field in _STATIC_SELECTION_NUMERIC_FIELDS:
        _require_close_float(
            actual_selection[field],
            replayed_selection[field],
            f"{role}.selection.{field}",
        )
        normalized_selection[field] = replayed_selection[field]
    normalized_selection["selection_hash"] = replayed_selection["selection_hash"]
    normalized["selection"] = normalized_selection
    normalized["fit_hash"] = expected["fit_hash"]
    _require_exact(normalized, expected, role)


def _validate_query_fixed_effect_fit(
    payload: Mapping[str, object], *, role: str
) -> None:
    _require_payload_hash(payload, "fit_hash", role)
    selection = _mapping(payload.get("selection"), f"{role}.selection")
    _require_payload_hash(selection, "selection_hash", f"{role}.selection")
    grand_mean = _finite_float(payload.get("grand_mean"), f"{role}.grand_mean")
    _finite_float(
        payload.get("residual_sum_squares"), f"{role}.residual_sum_squares"
    )
    query_centers = _string_list(payload.get("query_centers"), f"{role}.query_centers")
    candidate_sources = _string_list(
        payload.get("candidate_sources"), f"{role}.candidate_sources"
    )
    query_effects = _numeric_vector(
        payload.get("query_effects"), query_centers, f"{role}.query_effects"
    )
    source_effects = _numeric_vector(
        payload.get("source_effects"),
        candidate_sources,
        f"{role}.source_effects",
    )
    adjusted = _numeric_vector(
        payload.get("adjusted_source_gains"),
        candidate_sources,
        f"{role}.adjusted_source_gains",
    )
    # Every adjusted gain is a direct, sealed copy of the corresponding solve
    # coefficients.  Enforce that relationship before allowing cross-process
    # tolerance against another independently valid fit.
    for source in candidate_sources:
        if adjusted[source] != grand_mean + source_effects[source]:
            raise ProtocolError(
                f"Flip-router {role} adjusted source gain is internally inconsistent."
            )
    if set(query_effects) != set(query_centers):  # defensive for exotic mappings
        raise ProtocolError(f"Flip-router {role} query-effect topology drifted.")
    ranked = sorted(
        adjusted.items(), key=lambda item: (-item[1], f"A1::source={item[0]}")
    )
    if len(ranked) < 2:
        raise ProtocolError(f"Flip-router {role} adjusted ranking is incomplete.")
    best_source, best_gain = ranked[0]
    second_gain = ranked[1][1]
    derived_selection = (
        {
            "schema_version": "threshold_flip_case_router_core_v1",
            "action_id": "B",
            "exact_gain": 0.0,
            "runner_up_gain": max(0.0, best_gain),
            "fallback_to_b": True,
        }
        if best_gain <= 0.0
        else {
            "schema_version": "threshold_flip_case_router_core_v1",
            "action_id": f"A1::source={best_source}",
            "exact_gain": best_gain,
            "runner_up_gain": max(0.0, second_gain),
            "fallback_to_b": False,
        }
    )
    derived_selection["selection_hash"] = canonical_hash(derived_selection)
    _require_exact(selection, derived_selection, f"{role}.selection")


def _assert_donor_model_seals(
    path: Path,
    expected: Mapping[str, object],
    *,
    observed_model_rows: Sequence[Mapping[str, object]],
    expected_model_rows: Sequence[Mapping[str, object]],
) -> None:
    observed_by_target = _rows_by_target(observed_model_rows, "heldout_target_H")
    expected_by_target = _rows_by_target(expected_model_rows, "heldout_target_H")
    replayed = json_value(expected)
    if not isinstance(replayed, Mapping):
        raise ProtocolError("Flip-router replayed donor seal payload is malformed.")
    replayed_models = _mapping(replayed.get("models"), "replayed donor seals")
    observed_models: dict[str, object] = {}
    for target in CENTERS:
        replayed_seal = _mapping(
            replayed_models.get(target), f"replayed donor seal {target}"
        )
        expected_seal = _donor_model_seal(
            replayed_seal, expected_by_target[target], role=f"replayed donor seal {target}"
        )
        _require_exact(replayed_seal, expected_seal, f"replayed donor seal {target}")
        observed_seal = _donor_model_seal(
            replayed_seal,
            observed_by_target[target],
            role=f"persisted donor seal {target}",
        )
        observed_models[target] = observed_seal
    normalized = deepcopy(dict(replayed))
    normalized["models"] = observed_models
    observed_payload = _read_canonical_json_object(path)
    _require_exact(observed_payload, normalized, f"donor model seals {path}")


def _donor_model_seal(
    replayed_seal: Mapping[str, object],
    model_row: Mapping[str, object],
    *,
    role: str,
) -> dict[str, object]:
    ordinary = _mapping(model_row.get("ordinary_model"), f"{role}.ordinary_model")
    permutation = _mapping(
        model_row.get("permutation_model"), f"{role}.permutation_model"
    )
    unhashed = dict(replayed_seal)
    unhashed.pop("seal_hash", None)
    unhashed["global_static_selection"] = deepcopy(
        _mapping(model_row.get("global_static_selection"), role)
    )
    unhashed["global_static_query_fixed_effect_fit"] = deepcopy(
        _mapping(model_row.get("global_static_query_fixed_effect_fit"), role)
    )
    expected_links = {
        "heldout_target_H": model_row.get("heldout_target_H"),
        "model_hash": ordinary.get("model_hash"),
        "model_provenance_hash": ordinary.get("provenance_hash"),
        "permutation_model_hash": permutation.get("model_hash"),
        "permutation_provenance_hash": permutation.get("provenance_hash"),
    }
    for field, value in expected_links.items():
        if not _exact_equal(unhashed.get(field), value):
            raise ProtocolError(f"Flip-router {role}.{field} is misbound.")
    result = {**unhashed, "seal_hash": canonical_hash(unhashed)}
    if model_row.get("model_seal_hash") != result["seal_hash"]:
        raise ProtocolError(f"Flip-router {role} model seal hash drifted.")
    return result


def _with_persisted_global_selections(
    donor: object,
    observed_model_rows: Sequence[Mapping[str, object]],
) -> object:
    """Replay downstream hashes from the validated persisted solve result.

    Only the static selection is consumed after donor fitting.  Re-injecting
    that independently hash-validated copy lets the unchanged exact table and
    JSON comparators reconstruct every dependent row, fold, bundle, and
    terminal seal without granting those downstream surfaces any tolerance.
    """

    rows = _rows_by_target(observed_model_rows, "heldout_target_H")
    selections = {
        target: StaticSelection.from_payload(
            _mapping(rows[target]["global_static_selection"], target)
        )
        for target in CENTERS
    }
    return replace(
        donor,
        global_selection_by_target=MappingProxyType(selections),
    )


def _typed_table_rows(
    path: Path,
    expected_rows: Sequence[object],
    *,
    expected_fields: tuple[str, ...] | None = None,
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    expected = tuple(object_payload(row) for row in expected_rows)
    if not expected:
        raise ProtocolError(f"Flip-router replay produced an empty table: {path}.")
    payload_fields = tuple(expected[0])
    if any(tuple(row) != payload_fields for row in expected):
        raise ProtocolError(f"Flip-router replay table schema is ragged: {path}.")
    fields = payload_fields if expected_fields is None else expected_fields
    if len(fields) != len(payload_fields) or set(fields) != set(payload_fields):
        raise ProtocolError(f"Flip-router replay table schema drifted: {path}.")
    raw_rows = read_rows(path)
    if len(raw_rows) != len(expected) or any(tuple(row) != fields for row in raw_rows):
        raise ProtocolError(f"Flip-router persisted table header drifted: {path}.")
    observed = tuple(
        {
            field: _parse_persisted_cell(
                raw[field], replayed[field], role=f"{path}:{ordinal}:{field}"
            )
            for field in fields
        }
        for ordinal, (raw, replayed) in enumerate(zip(raw_rows, expected))
    )
    return observed, expected


def _parse_persisted_cell(raw: str, example: object, *, role: str) -> object:
    if isinstance(example, (dict, list)):
        parsed = _parse_canonical_json(raw, role=role)
        if type(parsed) is not type(example):
            raise ProtocolError(f"Flip-router {role} JSON type drifted.")
        return parsed
    if example is None:
        if raw != "":
            raise ProtocolError(f"Flip-router {role} null cell drifted.")
        return None
    if type(example) is bool:
        if raw not in {"True", "False"}:
            raise ProtocolError(f"Flip-router {role} boolean cell drifted.")
        return raw == "True"
    if type(example) is int:
        try:
            parsed_int = int(raw)
        except ValueError as exc:
            raise ProtocolError(f"Flip-router {role} integer cell drifted.") from exc
        if raw != str(parsed_int):
            raise ProtocolError(f"Flip-router {role} integer encoding drifted.")
        return parsed_int
    if type(example) is float:
        try:
            parsed_float = float(raw)
        except ValueError as exc:
            raise ProtocolError(f"Flip-router {role} floating cell drifted.") from exc
        if not math.isfinite(parsed_float) or raw != str(parsed_float):
            raise ProtocolError(f"Flip-router {role} floating encoding drifted.")
        return parsed_float
    return raw


def _parse_canonical_json(raw: str, *, role: str) -> object:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    try:
        parsed = json.loads(
            raw,
            object_pairs_hook=unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"nonfinite {value}")
            ),
        )
        canonical = json.dumps(
            parsed,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Flip-router {role} is not canonical JSON.") from exc
    if raw != canonical:
        raise ProtocolError(f"Flip-router {role} JSON encoding drifted.")
    return parsed


def _read_canonical_json_object(path: Path) -> dict[str, object]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProtocolError(f"Cannot read flip-router JSON: {path}.") from exc
    if not raw.endswith("\n") or raw.count("\n") != 1:
        raise ProtocolError(f"Flip-router JSON encoding drifted: {path}.")
    parsed = _parse_canonical_json(raw[:-1], role=str(path))
    if not isinstance(parsed, dict):
        raise ProtocolError(f"Flip-router JSON must be an object: {path}.")
    return parsed


def _mapping(value: object, role: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ProtocolError(f"Flip-router {role} is not a string-keyed object.")
    return value


def _string_list(value: object, role: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(type(item) is not str for item in value)
        or len(set(value)) != len(value)
    ):
        raise ProtocolError(f"Flip-router {role} is not a unique string list.")
    return tuple(value)


def _numeric_vector(
    value: object, names: Sequence[str], role: str
) -> dict[str, float]:
    payload = _mapping(value, role)
    if tuple(payload) != tuple(names):
        raise ProtocolError(f"Flip-router {role} topology drifted.")
    return {name: _finite_float(payload[name], f"{role}.{name}") for name in names}


def _finite_float(value: object, role: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ProtocolError(f"Flip-router {role} is not a finite float.")
    return value


def _require_close_float(actual: object, expected: object, role: str) -> None:
    left = _finite_float(actual, f"persisted {role}")
    right = _finite_float(expected, f"replayed {role}")
    if not math.isclose(
        left, right, rel_tol=0.0, abs_tol=_QUERY_FIXED_EFFECT_ATOL
    ):
        raise ProtocolError(f"Flip-router {role} exceeds replay tolerance.")


def _require_payload_hash(
    payload: Mapping[str, object], hash_field: str, role: str
) -> None:
    unhashed = dict(payload)
    observed = unhashed.pop(hash_field, None)
    if type(observed) is not str or observed != canonical_hash(unhashed):
        raise ProtocolError(f"Flip-router {role} {hash_field} drifted.")


def _rows_by_target(
    rows: Sequence[Mapping[str, object]], field: str
) -> dict[str, Mapping[str, object]]:
    result = {str(row.get(field)): row for row in rows}
    if tuple(result) != CENTERS or len(result) != len(rows):
        raise ProtocolError("Flip-router heldout-target topology drifted.")
    return result


def _exact_equal(actual: object, expected: object) -> bool:
    try:
        return json.dumps(
            actual,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ) == json.dumps(
            expected,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError):
        return False


def _require_exact(actual: object, expected: object, role: str) -> None:
    if not _exact_equal(actual, expected):
        raise ProtocolError(f"Flip-router {role} differs from replay.")


def _assert_json(path: Path, expected: object) -> None:
    converted = json_value(expected)
    if (
        not isinstance(converted, Mapping)
        or not _exact_equal(read_json(path), dict(converted))
    ):
        raise ProtocolError(f"Flip-router replayed JSON differs: {path}.")


def _assert_table(
    path: Path,
    expected_rows: Sequence[object],
    *,
    expected_fields: tuple[str, ...] | None = None,
) -> None:
    payloads = tuple(object_payload(row) for row in expected_rows)
    if not payloads:
        raise ProtocolError(f"Flip-router replay produced an empty table: {path}.")
    payload_fields = tuple(payloads[0])
    if any(tuple(row) != payload_fields for row in payloads):
        raise ProtocolError(f"Flip-router replay table schema is ragged: {path}.")
    fields = payload_fields if expected_fields is None else expected_fields
    if len(fields) != len(payload_fields) or set(fields) != set(payload_fields):
        raise ProtocolError(f"Flip-router replay table schema drifted: {path}.")
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
