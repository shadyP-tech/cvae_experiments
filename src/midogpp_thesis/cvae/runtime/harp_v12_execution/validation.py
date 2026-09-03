"""Fresh-process reconstruction of HARP v12 prelabel routing bytes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
import time

import numpy as np

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from .compatibility_adapter import compatibility_state_from_artifact
from .contracts import ActionKind, array_bytes_sha256
from .stores import (
    read_artifact_value,
    read_label_free_outer_menu,
    read_prelabel_routes,
)


FRESH_VALIDATION_TIMEOUT_SECONDS = 300


def _sha256(value: object, *, role: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProtocolError(f"HARP v12 {role} is not SHA-256.")
    return value


def _verified_artifact_hash(artifact: object, field: str, *, role: str) -> str:
    """Recompute a scientific-manifest identity from its durable projection."""

    manifest = dict(artifact.manifest)
    observed = _sha256(manifest.pop(field, None), role=role)
    if canonical_hash(manifest) != observed:
        raise ProtocolError(f"HARP v12 {role} scientific manifest hash drifted.")
    return observed


def _validate_numeric_oof(model: object, admission: object | None) -> None:
    """Require pickle-free pairwise scores and selected-policy OOF replays."""

    cases = np.asarray(model.arrays.get("oof_case_values"))
    scores = np.asarray(model.arrays.get("oof_action_scores"))
    offsets = np.asarray(model.arrays.get("oof_action_score_offsets"))
    if (
        set(model.arrays)
        != {"oof_case_values", "oof_action_scores", "oof_action_score_offsets"}
        or cases.dtype != np.dtype("float64")
        or cases.ndim != 2
        or cases.shape[1:] != (4,)
        or scores.dtype != np.dtype("float64")
        or scores.ndim != 2
        or scores.shape[1:] != (9,)
        or offsets.dtype != np.dtype("int64")
        or offsets.shape != (len(cases) + 1,)
        or int(offsets[0]) != 0
        or np.any(np.diff(offsets) < 0)
        or int(offsets[-1]) != len(scores)
        or not np.isfinite(cases).all()
        or not np.isfinite(scores).all()
    ):
        raise ProtocolError("HARP v12 numeric source-OOF score geometry drifted.")
    if admission is None:
        raise ProtocolError("HARP v12 source-OOF policy replay is absent.")
    rows = admission.manifest.get("source_policy_oof_rows")
    outer_policies = admission.manifest.get("outer_policies")
    replay = np.asarray(admission.arrays.get("source_policy_oof_values"))
    nested = np.asarray(admission.arrays.get("nested_source_policy_oof_values"))
    if (
        not isinstance(rows, list)
        or not isinstance(outer_policies, list)
        or admission.manifest.get("typed_case_outcome_inventory_replayed") is not True
        or admission.manifest.get(
            "empty_effective_menus_retained_as_exact_B_controls"
        )
        is not True
        or set(admission.arrays)
        != {"source_policy_oof_values", "nested_source_policy_oof_values"}
        or replay.dtype != np.dtype("float64")
        or nested.dtype != np.dtype("float64")
        or replay.shape != (len(rows), 8)
        or nested.shape != (len(rows), 8)
        or not np.isfinite(replay).all()
        or not np.isfinite(nested).all()
        or len(rows) != len(cases)
    ):
        raise ProtocolError("HARP v12 numeric policy replay geometry drifted.")
    expected_cases: list[tuple[float, ...]] = []
    expected_scores: list[tuple[float, ...]] = []
    expected_offsets = [0]
    expected_replay: list[tuple[float, ...]] = []
    expected_nested: list[tuple[float, ...]] = []
    policy_by_outer = {
        str(row.get("outer_target_id")): row
        for row in outer_policies
        if isinstance(row, Mapping)
    }
    if len(policy_by_outer) != len(outer_policies):
        raise ProtocolError("HARP v12 outer policy inventory is malformed.")
    control_counts: dict[str, int] = {outer: 0 for outer in policy_by_outer}
    active_counts: dict[str, int] = {outer: 0 for outer in policy_by_outer}
    label_free_context_hashes: dict[str, list[str]] = {
        outer: [] for outer in policy_by_outer
    }
    outcome_context_hashes: dict[str, list[str]] = {
        outer: [] for outer in policy_by_outer
    }
    try:
        for row in rows:
            if not isinstance(row, Mapping):
                raise TypeError
            prediction = row.get("prediction")
            if not isinstance(prediction, Mapping):
                raise TypeError
            action_scores = prediction.get("action_scores")
            if not isinstance(action_scores, list):
                raise TypeError
            outer = str(row.get("outer_target_id"))
            if (
                outer not in policy_by_outer
                or prediction.get("outer_target_id") != outer
                or prediction.get("query_center_id") != row.get("query_center_id")
                or prediction.get("case_id") != row.get("case_id")
            ):
                raise TypeError
            top = prediction.get("top_action_id")
            expected_cases.append(
                (
                    float(prediction.get("acceptance_probability")),
                    float(prediction.get("rank_margin")),
                    float(len(action_scores)),
                    float(top not in (None, "B")),
                )
            )
            for score in action_scores:
                if not isinstance(score, Mapping):
                    raise TypeError
                _sha256(score.get("score_hash"), role="source action-score hash")
                expected_scores.append(
                    (
                        float(score.get("pairwise_score")),
                        float(score.get("predicted_budget_gain")),
                        float(score.get("predicted_allocation_gain")),
                        float(score.get("predicted_total_gain")),
                        float(score.get("predicted_harm_probability")),
                        float(score.get("predicted_brier_delta")),
                        float(score.get("predicted_log_delta")),
                        float(score.get("acceptance_probability")),
                        float(bool(score.get("model_available"))),
                    )
                )
            expected_offsets.append(len(expected_scores))
            selected_id = str(row.get("selected_action_id"))
            nested_selected_id = str(row.get("nested_selected_action_id"))
            action_ids = row.get("active_action_ids")
            action_hashes = row.get("active_action_hashes")
            outcome_hashes = row.get("active_outcome_hashes")
            if (
                not isinstance(action_ids, list)
                or not isinstance(action_hashes, list)
                or not isinstance(outcome_hashes, list)
                or len(action_ids) != len(action_hashes)
                or len(action_ids) != len(outcome_hashes)
                or len(set(str(value) for value in action_ids)) != len(action_ids)
                or {
                    str(action_id): str(action_hash)
                    for action_id, action_hash in zip(
                        action_ids, action_hashes, strict=True
                    )
                }
                != {
                    str(score.get("action_id")): str(score.get("action_hash"))
                    for score in action_scores
                    if isinstance(score, Mapping)
                }
                or selected_id not in {"B", *(str(value) for value in action_ids)}
                or nested_selected_id not in {"B", *(str(value) for value in action_ids)}
            ):
                raise TypeError
            for outcome_hash in outcome_hashes:
                _sha256(outcome_hash, role="source action-outcome hash")
            exact_control = row.get("is_exact_b_control")
            kind = row.get("case_inventory_kind")
            if type(exact_control) is not bool or kind != (
                "EXACT_B_CONTROL" if exact_control else "ACTIVE_MENU"
            ):
                raise TypeError
            if exact_control:
                control_counts[outer] += 1
                if (
                    action_ids
                    or action_scores
                    or prediction.get("raw_top_action_id") != "B"
                    or top != "B"
                    or float(prediction.get("acceptance_probability")) != 0.0
                    or float(prediction.get("rank_margin")) != 0.0
                    or selected_id != "B"
                    or nested_selected_id != "B"
                    or any(
                        float(row.get(name)) != 0.0
                        for name in (
                            "observed_bacc_gain",
                            "observed_brier_delta",
                            "observed_log_delta",
                            "best_observed_bacc_gain",
                            "regret",
                            "nested_observed_bacc_gain",
                            "nested_observed_brier_delta",
                            "nested_observed_log_delta",
                            "nested_regret",
                        )
                    )
                ):
                    raise TypeError
            else:
                active_counts[outer] += 1
                if not action_ids:
                    raise TypeError
            expected_replay.append(
                (
                    float(prediction.get("acceptance_probability")),
                    float(prediction.get("rank_margin")),
                    float(selected_id != "B"),
                    float(row.get("observed_bacc_gain")),
                    float(row.get("observed_brier_delta")),
                    float(row.get("observed_log_delta")),
                    float(row.get("best_observed_bacc_gain")),
                    float(row.get("regret")),
                )
            )
            expected_nested.append(
                (
                    float(row.get("nested_acceptance_probability")),
                    float(row.get("nested_rank_margin")),
                    float(nested_selected_id != "B"),
                    float(row.get("nested_observed_bacc_gain")),
                    float(row.get("nested_observed_brier_delta")),
                    float(row.get("nested_observed_log_delta")),
                    float(row.get("best_observed_bacc_gain")),
                    float(row.get("nested_regret")),
                )
            )
            threshold = float(row.get("nested_acceptance_threshold"))
            margin = float(row.get("nested_rank_margin_threshold"))
            if (not 0.0 <= threshold <= 1.0 and threshold != 2.0) or margin < 0.0:
                raise TypeError
            _sha256(row.get("prediction_hash"), role="source prediction hash")
            _sha256(
                row.get("nested_prediction_hash"),
                role="nested source prediction hash",
            )
            _sha256(
                row.get("nested_policy_fold_hash"),
                role="nested policy-fold hash",
            )
            _sha256(
                row.get("label_free_case_context_hash"),
                role="label-free case-context hash",
            )
            outcome_context_hash = _sha256(
                row.get("case_outcome_context_hash"),
                role="case-outcome context hash",
            )
            label_free_context_body = {
                "schema_version": "policy_calibrated_label_free_case_context_v12",
                "case_key": (
                    outer,
                    str(row.get("query_center_id")),
                    str(row.get("case_id")),
                ),
                "kind": str(kind),
                "effective_menu_hash": str(prediction.get("menu_hash")),
                "action_ids": tuple(str(value) for value in action_ids),
                "action_hashes": tuple(str(value) for value in action_hashes),
                "prediction_hash": str(row.get("prediction_hash")),
                "source_outcomes_consumed": False,
                "evaluation_labels_consumed": False,
            }
            label_free_context_hash = canonical_hash(label_free_context_body)
            if row.get("label_free_case_context_hash") != label_free_context_hash:
                raise TypeError
            expected_outcome_context_hash = canonical_hash(
                {
                    "schema_version": "policy_calibrated_case_outcome_context_v12",
                    "label_free_context_hash": label_free_context_hash,
                    "outcome_hashes": tuple(str(value) for value in outcome_hashes),
                    "exact_b_gain": 0.0,
                    "evaluation_labels_consumed": False,
                }
            )
            if outcome_context_hash != expected_outcome_context_hash:
                raise TypeError
            label_free_context_hashes[outer].append(label_free_context_hash)
            outcome_context_hashes[outer].append(outcome_context_hash)
            universe_hash = _sha256(
                row.get("source_outcome_universe_hash"),
                role="source outcome-universe hash",
            )
            inventory_hash = _sha256(
                row.get("source_oof_inventory_hash"),
                role="source OOF inventory hash",
            )
            policy_row = policy_by_outer[outer]
            if (
                policy_row.get("source_outcome_universe_hash") != universe_hash
                or policy_row.get("source_oof_label_free_inventory_hash")
                != row.get("source_oof_label_free_inventory_hash")
                or policy_row.get("source_oof_inventory_hash") != inventory_hash
            ):
                raise TypeError
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("HARP v12 source-OOF score rows are malformed.") from exc
    for outer, policy_row in policy_by_outer.items():
        admission_row = policy_row.get("admission")
        label_free_inventory_hash = canonical_hash(
            {
                "schema_version": "policy_calibrated_label_free_case_inventory_v12",
                "context_hashes": tuple(label_free_context_hashes[outer]),
                "case_count": control_counts[outer] + active_counts[outer],
                "exact_b_control_count": control_counts[outer],
                "complete_for_menu_universe": True,
                "source_outcomes_consumed": False,
                "evaluation_labels_consumed": False,
            }
        )
        outcome_inventory_hash = canonical_hash(
            {
                "schema_version": "policy_calibrated_case_outcome_inventory_v12",
                "label_free_inventory_hash": label_free_inventory_hash,
                "context_hashes": tuple(outcome_context_hashes[outer]),
                "case_count": control_counts[outer] + active_counts[outer],
                "exact_b_control_count": control_counts[outer],
                "active_menu_count": active_counts[outer],
                "evaluation_labels_consumed": False,
            }
        )
        if (
            policy_row.get("exact_b_control_count") != control_counts[outer]
            or policy_row.get("active_menu_count") != active_counts[outer]
            or policy_row.get("source_oof_label_free_inventory_hash")
            != label_free_inventory_hash
            or policy_row.get("source_oof_inventory_hash")
            != outcome_inventory_hash
            or not isinstance(admission_row, Mapping)
            or admission_row.get("case_count")
            != control_counts[outer] + active_counts[outer]
        ):
            raise ProtocolError("HARP v12 source OOF inventory counts drifted.")
    expected_case_array = np.asarray(expected_cases, dtype=np.float64).reshape((-1, 4))
    expected_score_array = np.asarray(expected_scores, dtype=np.float64).reshape((-1, 9))
    if (
        cases.tobytes(order="C") != expected_case_array.tobytes(order="C")
        or scores.tobytes(order="C") != expected_score_array.tobytes(order="C")
        or offsets.tobytes(order="C")
        != np.asarray(expected_offsets, dtype=np.int64).tobytes(order="C")
        or replay.tobytes(order="C")
        != np.asarray(expected_replay, dtype=np.float64).reshape((-1, 8)).tobytes(order="C")
        or nested.tobytes(order="C")
        != np.asarray(expected_nested, dtype=np.float64).reshape((-1, 8)).tobytes(order="C")
    ):
        raise ProtocolError("HARP v12 source-OOF score/replay binding drifted.")


def _validate_effective_menu_store(
    effective: object,
    compatibility: object,
    *,
    expected_outer_menu_hashes: Mapping[str, str],
) -> tuple[str, object]:
    """Bind the independent menu seal to its compatibility source byte-for-byte."""

    effective_hash = _verified_artifact_hash(
        effective, "effective_menu_hash", role="effective menu hash"
    )
    compatibility_hash = _sha256(
        compatibility.manifest.get("compatibility_hash"), role="compatibility hash"
    )
    array_names = {
        "effective_action_features",
        "effective_menu_baselines",
        "effective_menu_baseline_offsets",
        "effective_action_probabilities",
        "effective_action_probability_offsets",
    }
    if (
        effective.manifest.get("compatibility_hash") != compatibility_hash
        or effective.manifest.get("effective_menus")
        != compatibility.manifest.get("effective_menus")
        or effective.manifest.get("effective_actions")
        != compatibility.manifest.get("effective_actions")
        or set(effective.arrays) != array_names
        or not array_names.issubset(compatibility.arrays)
    ):
        raise ProtocolError("HARP v12 effective-menu/compatibility binding drifted.")
    for name in sorted(array_names):
        left = np.asarray(effective.arrays[name])
        right = np.asarray(compatibility.arrays[name])
        if (
            left.dtype != right.dtype
            or left.shape != right.shape
            or left.tobytes(order="C") != right.tobytes(order="C")
        ):
            raise ProtocolError("HARP v12 effective-menu durable arrays drifted.")
    if (
        effective.manifest.get("directions_retained") != ["D01", "D10"]
        or effective.manifest.get("all_margins_excluded") is not True
        or effective.manifest.get("exact_b_noops_removed") is not True
        or effective.manifest.get("shared_source_target_implementation") is not True
    ):
        raise ProtocolError("HARP v12 effective-menu filter contract drifted.")
    restored = compatibility_state_from_artifact(
        compatibility,
        expected_outer_menu_hashes=expected_outer_menu_hashes,
    )
    if (
        effective.manifest.get("effective_menu_count")
        != len(restored.effective_menus)
        or effective.manifest.get("effective_action_count")
        != sum(len(menu.actions) for menu in restored.effective_menus)
        or [menu.menu_hash for menu in restored.effective_menus]
        != [str(row.get("menu_hash")) for row in effective.manifest["effective_menus"]]
    ):
        raise ProtocolError("HARP v12 reconstructed effective-menu inventory drifted.")
    return effective_hash, restored


def _validate_target_effective_binding(target: object, state: object) -> None:
    """Require every persisted target action to be an exact sealed menu member."""

    effective: dict[tuple[str, str, str], tuple[str, str, bytes]] = {}
    for menu in state.effective_menus:
        if menu.query_center_id != menu.outer_target_id:
            continue
        for action in menu.actions:
            key = (menu.outer_target_id, menu.case_id, action.action_id)
            values = b"".join(bytes.fromhex(value) for value in action.action_probability_hex)
            effective[key] = (action.action_hash, menu.menu_hash, values)
    rows = target.manifest.get("rows")
    probabilities = np.asarray(target.arrays.get("probabilities"))
    offsets = np.asarray(target.arrays.get("probability_offsets"))
    if not isinstance(rows, list):
        raise ProtocolError("HARP v12 target/effective action inventory is absent.")
    observed: set[tuple[str, str, str]] = set()
    for ordinal, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ProtocolError("HARP v12 target/effective action row is malformed.")
        key = (
            str(row.get("outer_target_id")),
            str(row.get("case_id")),
            str(row.get("action_id")),
        )
        start, stop = int(offsets[ordinal]), int(offsets[ordinal + 1])
        sealed = effective.get(key)
        if (
            sealed is None
            or row.get("action_hash") != sealed[0]
            or row.get("menu_hash") != sealed[1]
            or np.ascontiguousarray(probabilities[start:stop], dtype="<f4").tobytes(
                order="C"
            )
            != sealed[2]
        ):
            raise ProtocolError("HARP v12 target action escaped the effective-menu seal.")
        observed.add(key)
    if observed != set(effective):
        raise ProtocolError("HARP v12 target action inventory is not complete.")
    _validate_target_score_projection(target, effective=effective)


def _validate_target_score_projection(
    target: object,
    *,
    effective: Mapping[tuple[str, str, str], tuple[str, str, bytes]],
) -> None:
    """Bind every target pairwise/residual score to durable numeric arrays."""

    prediction_rows = target.manifest.get("prediction_rows")
    case_values = np.asarray(target.arrays.get("case_prediction_values"))
    score_values = np.asarray(target.arrays.get("action_score_values"))
    score_offsets = np.asarray(target.arrays.get("action_score_offsets"))
    expected_array_names = {
        "probabilities",
        "probability_offsets",
        "case_prediction_values",
        "action_score_values",
        "action_score_offsets",
    }
    if (
        not isinstance(prediction_rows, list)
        or set(target.arrays) != expected_array_names
        or case_values.dtype != np.dtype("float64")
        or case_values.shape != (len(prediction_rows), 4)
        or score_values.dtype != np.dtype("float64")
        or score_values.ndim != 2
        or score_values.shape[1:] != (9,)
        or score_offsets.dtype != np.dtype("int64")
        or score_offsets.shape != (len(prediction_rows) + 1,)
        or int(score_offsets[0]) != 0
        or np.any(np.diff(score_offsets) < 0)
        or int(score_offsets[-1]) != len(score_values)
        or not np.isfinite(case_values).all()
        or not np.isfinite(score_values).all()
    ):
        raise ProtocolError("HARP v12 target action-score array geometry drifted.")
    expected_cases: list[tuple[float, float, float, float]] = []
    expected_scores: list[tuple[float, ...]] = []
    expected_offsets = [0]
    observed: set[tuple[str, str, str]] = set()
    try:
        for row in prediction_rows:
            if not isinstance(row, Mapping):
                raise TypeError
            outer = str(row.get("outer_target_id"))
            case_id = str(row.get("case_id"))
            raw_scores = row.get("action_scores")
            training = row.get("training_center_ids")
            candidates = row.get("training_candidate_ids")
            excluded = row.get("excluded_center_ids")
            if (
                not isinstance(raw_scores, list)
                or not isinstance(training, list)
                or not isinstance(candidates, list)
                or not isinstance(excluded, list)
                or outer not in {str(value) for value in excluded}
                or outer in {str(value) for value in training}
                or outer in {str(value) for value in candidates}
            ):
                raise TypeError
            score_ids: list[str] = []
            for score in raw_scores:
                if not isinstance(score, Mapping):
                    raise TypeError
                action_id = str(score.get("action_id"))
                key = (outer, case_id, action_id)
                sealed = effective.get(key)
                if (
                    sealed is None
                    or score.get("action_hash") != sealed[0]
                    or key in observed
                ):
                    raise TypeError
                observed.add(key)
                score_ids.append(action_id)
                _sha256(score.get("score_hash"), role="target action-score hash")
                expected_scores.append(
                    (
                        float(score.get("pairwise_score")),
                        float(score.get("predicted_budget_gain")),
                        float(score.get("predicted_allocation_gain")),
                        float(score.get("predicted_total_gain")),
                        float(score.get("predicted_harm_probability")),
                        float(score.get("predicted_brier_delta")),
                        float(score.get("predicted_log_delta")),
                        float(score.get("acceptance_probability")),
                        float(bool(score.get("model_available"))),
                    )
                )
            top = row.get("top_action_id")
            raw_top = row.get("raw_top_action_id")
            all_ids = set(score_ids) | {"B"}
            if (top is not None and str(top) not in all_ids) or (
                raw_top is not None and str(raw_top) not in all_ids
            ):
                raise TypeError
            expected_cases.append(
                (
                    float(row.get("acceptance_probability")),
                    float(row.get("rank_margin")),
                    float(len(raw_scores)),
                    float(top not in (None, "B")),
                )
            )
            expected_offsets.append(len(expected_scores))
    except (TypeError, ValueError) as exc:
        raise ProtocolError("HARP v12 target action-score rows are malformed.") from exc
    expected_case_values = np.asarray(expected_cases, dtype=np.float64).reshape((-1, 4))
    expected_score_values = np.asarray(expected_scores, dtype=np.float64).reshape((-1, 9))
    expected_offset_values = np.asarray(expected_offsets, dtype=np.int64)
    if (
        observed != set(effective)
        or case_values.tobytes(order="C")
        != expected_case_values.tobytes(order="C")
        or score_values.tobytes(order="C")
        != expected_score_values.tobytes(order="C")
        or score_offsets.tobytes(order="C")
        != expected_offset_values.tobytes(order="C")
    ):
        raise ProtocolError("HARP v12 target action-score row/value binding drifted.")


def _target_action_table(
    target: object,
) -> dict[tuple[str, str, str], tuple[tuple[str, ...], np.ndarray]]:
    """Reconstruct the sealed action-ID to exact probability-byte binding."""

    _verified_artifact_hash(
        target, "target_action_hash", role="target action hash"
    )
    rows = target.manifest.get("rows")
    probabilities = np.asarray(target.arrays.get("probabilities"))
    offsets = np.asarray(target.arrays.get("probability_offsets"))
    if (
        not isinstance(rows, list)
        or probabilities.dtype != np.dtype("float32")
        or probabilities.ndim != 1
        or offsets.dtype != np.dtype("int64")
        or offsets.shape != (len(rows) + 1,)
        or int(offsets[0]) != 0
        or np.any(np.diff(offsets) <= 0)
        or int(offsets[-1]) != len(probabilities)
    ):
        raise ProtocolError("HARP v12 persisted target-action table geometry drifted.")
    table: dict[tuple[str, str, str], tuple[tuple[str, ...], np.ndarray]] = {}
    for ordinal, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ProtocolError("HARP v12 persisted target-action row is malformed.")
        key = (
            str(row.get("outer_target_id")),
            str(row.get("case_id")),
            str(row.get("action_id")),
        )
        sample_ids = row.get("sample_ids")
        start, stop = int(offsets[ordinal]), int(offsets[ordinal + 1])
        values = np.asarray(probabilities[start:stop], dtype=np.float32)
        direction = row.get("direction")
        kind = row.get("action_kind")
        source = row.get("candidate_source_id")
        expected_action_id = (
            f"U:{direction}" if kind == "U" else f"HXE:{source}:{direction}"
        )
        if (
            key in table
            or any(not value or value == "None" for value in key)
            or not isinstance(sample_ids, list)
            or len(sample_ids) != stop - start
            or len(set(str(value) for value in sample_ids)) != len(sample_ids)
            or row.get("probability_offset_start") != start
            or row.get("probability_offset_stop") != stop
            or direction not in {"D01", "D10"}
            or kind not in {"U", "HXE"}
            or (kind == "U" and source is not None)
            or (kind == "HXE" and (type(source) is not str or not source))
            or (kind == "HXE" and source == key[0])
            or key[2] != expected_action_id
            or array_bytes_sha256(values) != row.get("probability_hash")
        ):
            raise ProtocolError("HARP v12 persisted target-action binding drifted.")
        table[key] = (
            tuple(str(value) for value in sample_ids),
            np.ascontiguousarray(values, dtype=np.float32),
        )
    return table


def _validate_route_inputs(
    routes: object,
    menus: Mapping[str, object],
    target: object,
) -> None:
    """Bind every reconstructed route byte to its physical/menu provenance."""

    table = _target_action_table(target)
    menu_hashes = target.manifest.get("outer_menu_hashes")
    if not isinstance(menu_hashes, Mapping) or dict(menu_hashes) != {
        center: menu.menu_hash for center, menu in menus.items()
    }:
        raise ProtocolError("HARP v12 target actions escaped the reconstructed menus.")
    for case in routes.cases:
        menu = menus[case.outer_target_id]
        baseline = menu.target_block(ActionKind.B)
        uniform = menu.target_block(ActionKind.U)
        indices = np.flatnonzero(
            np.asarray(baseline.case_ids, dtype=object) == case.case_id
        )
        if not len(indices):
            raise ProtocolError("HARP v12 route case is absent from its physical menu.")
        samples = tuple(baseline.sample_ids[int(index)] for index in indices)
        if (
            case.sample_ids != samples
            or tuple(uniform.sample_ids[int(index)] for index in indices) != samples
            or case.baseline_probabilities.tobytes(order="C")
            != np.asarray(baseline.probabilities[indices], dtype=np.float32).tobytes(
                order="C"
            )
            or case.uniform_probabilities.tobytes(order="C")
            != np.asarray(uniform.probabilities[indices], dtype=np.float32).tobytes(
                order="C"
            )
        ):
            raise ProtocolError("HARP v12 routed B/U bytes escaped the physical menu.")
        if case.selected_kind is ActionKind.B:
            if (
                case.direction is not None
                or case.shrinkage != 0.0
                or case.component_action_ids
                or case.component_weights
                or case.component_probabilities
                or case.selected_probabilities.tobytes(order="C")
                != case.baseline_probabilities.tobytes(order="C")
                or case.routed_probabilities.tobytes(order="C")
                != case.baseline_probabilities.tobytes(order="C")
            ):
                raise ProtocolError("HARP v12 reconstructed abstention is not exact B.")
        elif (
            case.direction not in {"D01", "D10"}
            or case.shrinkage != 1.0
            or len(case.component_action_ids) != 1
            or case.component_weights != (1.0,)
            or len(case.component_probabilities) != 1
            or case.selected_probabilities.tobytes(order="C")
            != case.component_probabilities[0].tobytes(order="C")
            or case.routed_probabilities.tobytes(order="C")
            != case.component_probabilities[0].tobytes(order="C")
        ):
            raise ProtocolError("HARP v12 reconstructed route is not exact top-1.")
        for action_id, component in zip(
            case.component_action_ids,
            case.component_probabilities,
            strict=True,
        ):
            action = table.get((case.outer_target_id, case.case_id, action_id))
            if (
                action is None
                or action[0] != case.sample_ids
                or action[1].tobytes(order="C") != component.tobytes(order="C")
            ):
                raise ProtocolError(
                    "HARP v12 routed component is not the named persisted target action."
                )


def reconstruct_prelabel_routes(
    route_root: Path,
    menu_roots: Mapping[str, Path],
    development_root: Path,
    model_root: Path,
    target_action_root: Path,
    *,
    validator_id: str,
    expected_center_ids: Sequence[str],
    expected_config_hash: str,
    effective_menu_root: Path,
    compatibility_root: Path | None = None,
    admission_root: Path | None = None,
) -> dict[str, object]:
    """Reopen every durable prelabel input and revalidate exact-top-1 bytes."""

    centers = tuple(str(value) for value in expected_center_ids)
    if not centers or centers != tuple(sorted(set(centers))):
        raise ProtocolError("HARP v12 validator center universe drifted.")
    config_hash = _sha256(expected_config_hash, role="config hash")
    if set(menu_roots) != set(centers):
        raise ProtocolError("HARP v12 validator menu target universe drifted.")
    menus = {
        center: read_label_free_outer_menu(Path(menu_roots[center]))
        for center in centers
    }
    if any(menu.outer_target_id != center for center, menu in menus.items()):
        raise ProtocolError("HARP v12 validator menu/root identity drifted.")
    development = read_artifact_value(
        Path(development_root), role="source_development_case_surface"
    )
    model = read_artifact_value(Path(model_root), role="source_only_router")
    target = read_artifact_value(
        Path(target_action_root), role="complete_target_case_actions"
    )
    compatibility = (
        None
        if compatibility_root is None
        else read_artifact_value(
            Path(compatibility_root), role="label_free_support_compatibility"
        )
    )
    effective_menu = read_artifact_value(
        Path(effective_menu_root), role="label_free_effective_menu"
    )
    admission = (
        None
        if admission_root is None
        else read_artifact_value(
            Path(admission_root), role="source_only_policy_oof_replay"
        )
    )
    routes = read_prelabel_routes(Path(route_root))
    _verified_artifact_hash(
        development, "surface_hash", role="development surface hash"
    )
    model_hash = _verified_artifact_hash(model, "model_hash", role="model hash")
    target_hash = _verified_artifact_hash(
        target, "target_action_hash", role="target action hash"
    )
    development_hash = _sha256(
        development.manifest.get("surface_hash"), role="development surface hash"
    )
    if (
        routes.model_hash != model_hash
        or routes.target_action_hash != target_hash
        or model.manifest.get("development_surface_hash") != development_hash
        or target.manifest.get("model_hash") != model_hash
    ):
        raise ProtocolError("HARP v12 route/model/action binding drifted.")
    _validate_route_inputs(routes, menus, target)
    observed_centers = tuple(sorted({case.outer_target_id for case in routes.cases}))
    if observed_centers != centers:
        raise ProtocolError("HARP v12 routed center coverage drifted.")
    if any(
        case.selected_kind.value == "Hxe"
        and (
            case.selected_source_id == case.outer_target_id
            or any(
                action_id == f"HXE:{case.outer_target_id}"
                for action_id in case.component_action_ids
            )
        )
        for case in routes.cases
    ):
        raise ProtocolError("HARP v12 reconstructed route violates outer exclusion.")
    compatibility_hash = None
    effective_menu_hash = None
    if compatibility is not None:
        compatibility_hash = _verified_artifact_hash(
            compatibility, "compatibility_hash", role="compatibility hash"
        )
        if target.manifest.get("compatibility_hash") != compatibility_hash:
            raise ProtocolError("HARP v12 target/compatibility binding drifted.")
        if model.manifest.get("compatibility_hash") != compatibility_hash:
            raise ProtocolError("HARP v12 model/compatibility binding drifted.")
        effective_menu_hash, compatibility_state = _validate_effective_menu_store(
            effective_menu,
            compatibility,
            expected_outer_menu_hashes={
                center: menus[center].menu_hash for center in centers
            },
        )
        _validate_target_effective_binding(target, compatibility_state)
    else:
        raise ProtocolError(
            "HARP v12 fresh validation requires compatibility for the effective-menu seal."
        )
    admission_hash = None
    if admission is not None:
        admission_hash = _verified_artifact_hash(
            admission, "admission_hash", role="admission hash"
        )
        if target.manifest.get("admission_hash") != admission_hash:
            raise ProtocolError("HARP v12 target/admission binding drifted.")
        if (
            admission.manifest.get("model_hash") != model_hash
            or admission.manifest.get("development_surface_hash")
            != development_hash
        ):
            raise ProtocolError("HARP v12 admission/model/development binding drifted.")
    else:
        raise ProtocolError(
            "HARP v12 fresh validation requires the source policy OOF replay."
        )
    _validate_numeric_oof(model, admission)
    payload = {
        "schema_version": "midogpp_harp_v12_fresh_route_reconstruction_v1",
        "validator_id": str(validator_id),
        "process_id": os.getpid(),
        "config_hash": config_hash,
        "expected_center_ids": list(centers),
        "menu_hashes": {center: menus[center].menu_hash for center in centers},
        "development_surface_hash": development_hash,
        "compatibility_hash": compatibility_hash,
        "effective_menu_hash": effective_menu_hash,
        "model_hash": model_hash,
        "admission_hash": admission_hash,
        "target_action_hash": target_hash,
        "route_hash": routes.route_hash,
        "policy_hash": routes.policy_hash,
        "decision_hashes": [case.decision_hash for case in routes.cases],
        "case_count": len(routes.cases),
        "exact_top1_physical_action_reconstructed": True,
        "unevaluated_action_mixture_used": False,
        "exact_b_fallback_reconstructed": True,
        "evaluation_labels_opened": False,
    }
    return {**payload, "validation_hash": canonical_hash(payload)}


def _worker(
    queue: object,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
) -> None:
    validator_id = str(kwargs["validator_id"])
    try:
        value = reconstruct_prelabel_routes(*args, **kwargs)  # type: ignore[arg-type]
        queue.put((validator_id, True, value))
    except BaseException as exc:  # pragma: no cover - subprocess forwarding
        queue.put((validator_id, False, (exc.__class__.__name__, str(exc))))


def run_two_fresh_validations(
    route_root: Path,
    menu_roots: Mapping[str, Path],
    development_root: Path,
    model_root: Path,
    target_action_root: Path,
    *,
    expected_center_ids: Sequence[str],
    expected_config_hash: str,
    effective_menu_root: Path,
    compatibility_root: Path | None = None,
    admission_root: Path | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Require byte-identical reconstruction in two distinct spawned processes."""

    centers = tuple(str(value) for value in expected_center_ids)
    config_hash = _sha256(expected_config_hash, role="config hash")
    context = mp.get_context("spawn")
    queue = context.Queue()
    args = (
        Path(route_root).resolve(),
        {key: Path(value).resolve() for key, value in menu_roots.items()},
        Path(development_root).resolve(),
        Path(model_root).resolve(),
        Path(target_action_root).resolve(),
    )
    processes = []
    for validator_id in ("fresh_reconstruction_A", "fresh_reconstruction_B"):
        kwargs = {
            "validator_id": validator_id,
            "expected_center_ids": centers,
            "expected_config_hash": config_hash,
            "effective_menu_root": Path(effective_menu_root).resolve(),
            "compatibility_root": (
                None
                if compatibility_root is None
                else Path(compatibility_root).resolve()
            ),
            "admission_root": (
                None if admission_root is None else Path(admission_root).resolve()
            ),
        }
        processes.append(
            context.Process(
                target=_worker,
                args=(queue, args, kwargs),
                name=f"harp-v12-{validator_id}",
            )
        )
    for process in processes:
        process.start()
    messages: dict[str, tuple[bool, object]] = {}
    deadline = time.monotonic() + FRESH_VALIDATION_TIMEOUT_SECONDS
    try:
        for _ in processes:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ProtocolError("HARP v12 fresh validation timed out.")
                try:
                    validator_id, success, value = queue.get(
                        timeout=min(1.0, remaining)
                    )
                    messages[str(validator_id)] = (bool(success), value)
                    break
                except Empty:
                    failed = {
                        process.name: process.exitcode
                        for process in processes
                        if process.exitcode not in (None, 0)
                    }
                    if failed:
                        raise ProtocolError(
                            f"HARP v12 fresh validator exited before receipt: {failed}."
                        )
    finally:
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    if len(messages) != 2 or any(not success for success, _ in messages.values()):
        detail = {
            key: value for key, (success, value) in messages.items() if not success
        }
        raise ProtocolError(f"HARP v12 fresh validation failed: {detail}.")
    first = messages["fresh_reconstruction_A"][1]
    second = messages["fresh_reconstruction_B"][1]
    if not isinstance(first, dict) or not isinstance(second, dict):
        raise ProtocolError("HARP v12 fresh validator returned an invalid result.")
    pids = {first.get("process_id"), second.get("process_id")}
    comparable = {
        key: value
        for key, value in first.items()
        if key not in {"validator_id", "process_id", "validation_hash"}
    }
    second_comparable = {
        key: value
        for key, value in second.items()
        if key not in {"validator_id", "process_id", "validation_hash"}
    }
    if (
        len(pids) != 2
        or os.getpid() in pids
        or comparable != second_comparable
        or first.get("validation_hash") == second.get("validation_hash")
    ):
        raise ProtocolError("HARP v12 fresh reconstruction identities drifted.")
    return first, second


__all__ = ("reconstruct_prelabel_routes", "run_two_fresh_validations")
