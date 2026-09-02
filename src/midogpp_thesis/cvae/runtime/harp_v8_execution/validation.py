"""Fresh-process reconstruction of HARP v8 prelabel routing bytes."""

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
from ...routing.baseline_inclusive_action_safe_router_v8.calibration import (
    DISABLED_CERTIFICATE_CONFIDENCE_THRESHOLD,
)
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
        raise ProtocolError(f"HARP v8 {role} is not SHA-256.")
    return value


def _verified_artifact_hash(artifact: object, field: str, *, role: str) -> str:
    """Recompute a scientific-manifest identity from its durable projection."""

    manifest = dict(artifact.manifest)
    observed = _sha256(manifest.pop(field, None), role=role)
    if canonical_hash(manifest) != observed:
        raise ProtocolError(f"HARP v8 {role} scientific manifest hash drifted.")
    return observed


def _validate_numeric_oof(model: object, admission: object | None) -> None:
    """Require pickle-free action estimates, certificates, and policy replays."""

    cases = np.asarray(model.arrays.get("oof_case_values"))
    certificates = np.asarray(model.arrays.get("oof_action_certificates"))
    offsets = np.asarray(model.arrays.get("oof_action_certificate_offsets"))
    if (
        cases.dtype != np.dtype("float64")
        or cases.ndim != 2
        or cases.shape[1:] != (4,)
        or certificates.dtype != np.dtype("float64")
        or certificates.ndim != 2
        or certificates.shape[1:] != (12,)
        or offsets.dtype != np.dtype("int64")
        or offsets.shape != (len(cases) + 1,)
        or int(offsets[0]) != 0
        or np.any(np.diff(offsets) < 0)
        or int(offsets[-1]) != len(certificates)
        or not np.isfinite(cases).all()
        or not np.isfinite(certificates).all()
    ):
        raise ProtocolError("HARP v8 numeric source-OOF certificate geometry drifted.")
    outer_models = model.manifest.get("outer_models")
    expected_cases: list[tuple[float, float, float, float]] = []
    expected_certificates: list[tuple[float, ...]] = []
    expected_offsets = [0]
    model_row_identities: list[tuple[str, str, str]] = []
    model_prediction_rows: dict[tuple[str, str, str], Mapping[str, object]] = {}
    nested_prediction_rows: dict[
        tuple[str, str, str],
        tuple[Mapping[str, object], str, tuple[str, ...]],
    ] = {}
    if not isinstance(outer_models, list) or not outer_models:
        raise ProtocolError("HARP v8 numeric source-OOF manifest rows are absent.")
    try:
        for outer_model in outer_models:
            if not isinstance(outer_model, Mapping):
                raise TypeError
            numeric = outer_model.get("numeric_oof")
            if not isinstance(numeric, Mapping):
                raise TypeError
            rows = numeric.get("rows")
            nested_folds = numeric.get("nested_policy_folds")
            if not isinstance(rows, list) or not isinstance(nested_folds, list):
                raise TypeError
            outer_target_id = str(outer_model.get("outer_target_id"))
            for fold in nested_folds:
                if not isinstance(fold, Mapping):
                    raise TypeError
                heldout_center_id = str(fold.get("heldout_center_id"))
                training_center_ids = fold.get("training_center_ids")
                heldout_rows = fold.get("heldout_rows")
                fold_hash = _sha256(
                    fold.get("fold_hash"), role="nested policy fold hash"
                )
                if (
                    not isinstance(training_center_ids, list)
                    or not isinstance(heldout_rows, list)
                    or heldout_center_id in {str(value) for value in training_center_ids}
                    or outer_target_id in {str(value) for value in training_center_ids}
                ):
                    raise TypeError
                ordered_training = tuple(str(value) for value in training_center_ids)
                if ordered_training != tuple(sorted(set(ordered_training))):
                    raise TypeError
                for heldout_row in heldout_rows:
                    if not isinstance(heldout_row, Mapping):
                        raise TypeError
                    nested_identity = (
                        outer_target_id,
                        heldout_center_id,
                        str(heldout_row.get("case_id")),
                    )
                    if (
                        str(heldout_row.get("query_center_id"))
                        != heldout_center_id
                        or nested_identity in nested_prediction_rows
                    ):
                        raise TypeError
                    nested_prediction_rows[nested_identity] = (
                        heldout_row,
                        fold_hash,
                        ordered_training,
                    )
            for row in rows:
                if not isinstance(row, Mapping):
                    raise TypeError
                raw_certificates = row.get("action_certificates")
                safe_action_ids = row.get("safe_action_ids")
                if not isinstance(raw_certificates, list) or not isinstance(
                    safe_action_ids, list
                ):
                    raise TypeError
                observed_safe_ids: list[str] = []
                predicted_harm: list[float] = []
                for certificate in raw_certificates:
                    if not isinstance(certificate, Mapping):
                        raise TypeError
                    is_safe = bool(certificate.get("safe"))
                    if is_safe:
                        observed_safe_ids.append(str(certificate.get("action_id")))
                    predicted_harm.append(
                        float(certificate.get("predicted_harm_probability"))
                    )
                    expected_certificates.append(
                        (
                            float(certificate.get("predicted_bacc_gain")),
                            float(certificate.get("predicted_harm_probability")),
                            float(certificate.get("predicted_brier_delta")),
                            float(certificate.get("predicted_log_delta")),
                            float(certificate.get("gain_lcb")),
                            float(certificate.get("harm_probability_ucb")),
                            float(certificate.get("brier_delta_ucb")),
                            float(certificate.get("log_delta_ucb")),
                            float(certificate.get("harm_brier_risk")),
                            float(certificate.get("harm_log_loss_risk")),
                            float(bool(certificate.get("model_available", True))),
                            float(is_safe),
                        )
                    )
                if observed_safe_ids != [str(value) for value in safe_action_ids]:
                    raise TypeError
                diagnostic_confidence = (
                    max(1.0 - value for value in predicted_harm)
                    if predicted_harm
                    else 0.0
                )
                expected_cases.append(
                    (
                        diagnostic_confidence,
                        float(row.get("rank_margin")),
                        float(len(observed_safe_ids)),
                        float(len(raw_certificates)),
                    )
                )
                identity = (
                    outer_target_id,
                    str(row.get("query_center_id")),
                    str(row.get("case_id")),
                )
                if identity in model_prediction_rows:
                    raise TypeError
                model_prediction_rows[identity] = row
                model_row_identities.append(identity)
                expected_offsets.append(len(expected_certificates))
    except (TypeError, ValueError) as exc:
        raise ProtocolError(
            "HARP v8 numeric source-OOF certificate rows are malformed."
        ) from exc
    case_values = np.asarray(expected_cases, dtype=np.float64).reshape((-1, 4))
    certificate_values = np.asarray(
        expected_certificates, dtype=np.float64
    ).reshape((-1, 12))
    offset_values = np.asarray(expected_offsets, dtype=np.int64)
    if (
        set(model.arrays)
        != {
            "oof_case_values",
            "oof_action_certificates",
            "oof_action_certificate_offsets",
        }
        or cases.shape != case_values.shape
        or cases.tobytes(order="C") != case_values.tobytes(order="C")
        or certificates.shape != certificate_values.shape
        or certificates.tobytes(order="C")
        != certificate_values.tobytes(order="C")
        or offsets.tobytes(order="C") != offset_values.tobytes(order="C")
    ):
        raise ProtocolError("HARP v8 numeric source-OOF certificate binding drifted.")
    if admission is None:
        return
    outer_policies = admission.manifest.get("outer_policies")
    if not isinstance(outer_policies, list):
        raise ProtocolError("HARP v8 outer policy manifest is absent.")
    policy_bindings: dict[
        str, tuple[str, dict[str, tuple[float, float]]]
    ] = {}
    try:
        for policy in outer_policies:
            if not isinstance(policy, Mapping):
                raise TypeError
            outer = str(policy.get("outer_target_id"))
            calibration = policy.get("calibration")
            if not isinstance(calibration, Mapping):
                raise TypeError
            nested_replay_hash = _sha256(
                calibration.get("nested_replay_hash"),
                role="nested policy replay hash",
            )
            raw_thresholds = calibration.get("heldout_thresholds")
            if not isinstance(raw_thresholds, list):
                raise TypeError
            thresholds: dict[str, tuple[float, float]] = {}
            for threshold in raw_thresholds:
                if not isinstance(threshold, Mapping):
                    raise TypeError
                heldout = str(threshold.get("heldout_center_id"))
                confidence = float(
                    threshold.get("certificate_confidence_threshold")
                )
                margin = float(threshold.get("rank_margin_threshold"))
                if heldout in thresholds:
                    raise TypeError
                thresholds[heldout] = (confidence, margin)
            if outer in policy_bindings:
                raise TypeError
            policy_bindings[outer] = (nested_replay_hash, thresholds)
    except (TypeError, ValueError) as exc:
        raise ProtocolError("HARP v8 nested policy manifest is malformed.") from exc
    rows = admission.manifest.get("source_policy_oof_rows")
    replay = np.asarray(admission.arrays.get("source_policy_oof_values"))
    nested_replay = np.asarray(
        admission.arrays.get("nested_source_policy_oof_values")
    )
    if (
        not isinstance(rows, list)
        or set(admission.arrays)
        != {"source_policy_oof_values", "nested_source_policy_oof_values"}
        or replay.dtype != np.dtype("float64")
        or replay.shape != (len(rows), 8)
        or len(rows) != len(cases)
        or not np.isfinite(replay).all()
        or nested_replay.dtype != np.dtype("float64")
        or nested_replay.shape != (len(rows), 8)
        or not np.isfinite(nested_replay).all()
        or admission.manifest.get("source_policy_oof_case_count") != len(rows)
        or admission.manifest.get("nested_held_source_threshold_policy_replayed")
        is not True
    ):
        raise ProtocolError("HARP v8 whole-policy numeric OOF replay drifted.")
    for ordinal, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ProtocolError("HARP v8 whole-policy OOF row is malformed.")
        try:
            expected_head = np.asarray(
                (
                    float(row.get("selected_certificate_confidence")),
                    float(row.get("rank_margin")),
                ),
                dtype=np.float64,
            )
            expected_nested_head = np.asarray(
                (
                    float(row.get("nested_certificate_confidence")),
                    float(row.get("nested_rank_margin")),
                ),
                dtype=np.float64,
            )
            expected_tail = np.asarray(
                (
                    float(str(row.get("selected_action_id")) != "B"),
                    float(row.get("observed_bacc_gain")),
                    float(row.get("observed_brier_delta")),
                    float(row.get("observed_log_delta")),
                    float(row.get("best_observed_bacc_gain")),
                    float(row.get("regret")),
                ),
                dtype=np.float64,
            )
            expected_nested_tail = np.asarray(
                (
                    float(str(row.get("nested_selected_action_id")) != "B"),
                    float(row.get("nested_observed_bacc_gain")),
                    float(row.get("nested_observed_brier_delta")),
                    float(row.get("nested_observed_log_delta")),
                    float(row.get("best_observed_bacc_gain")),
                    float(row.get("nested_regret")),
                ),
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise ProtocolError("HARP v8 whole-policy OOF row is nonnumeric.") from exc
        identity = (
            str(row.get("outer_target_id")),
            str(row.get("query_center_id")),
            str(row.get("case_id")),
        )
        model_prediction = model_prediction_rows.get(identity)
        nested_binding = nested_prediction_rows.get(identity)
        policy_binding = policy_bindings.get(identity[0])
        if (
            model_prediction is None
            or nested_binding is None
            or policy_binding is None
        ):
            raise ProtocolError("HARP v8 nested policy/model identity is absent.")
        nested_prediction, expected_fold_hash, expected_training = nested_binding
        expected_nested_replay_hash, expected_thresholds = policy_binding
        expected_threshold = expected_thresholds.get(identity[1])
        if expected_threshold is None:
            raise ProtocolError("HARP v8 nested policy threshold is absent.")
        if (
            identity != model_row_identities[ordinal]
            or replay[ordinal, :2].tobytes(order="C")
            != expected_head.tobytes(order="C")
        ):
            raise ProtocolError("HARP v8 policy/model OOF row binding drifted.")
        if replay[ordinal, 2:].tobytes(order="C") != expected_tail.tobytes(order="C"):
            raise ProtocolError("HARP v8 whole-policy OOF row/value binding drifted.")
        if (
            nested_replay[ordinal, :2].tobytes(order="C")
            != expected_nested_head.tobytes(order="C")
            or nested_replay[ordinal, 2:].tobytes(order="C")
            != expected_nested_tail.tobytes(order="C")
        ):
            raise ProtocolError("HARP v8 nested policy OOF row/value binding drifted.")
        training_centers = row.get("nested_threshold_training_center_ids")
        certificate_confidence_threshold = row.get(
            "nested_certificate_confidence_threshold"
        )
        rank_margin_threshold = row.get("nested_rank_margin_threshold")
        if (
            type(certificate_confidence_threshold) not in (int, float)
            or type(rank_margin_threshold) not in (int, float)
            or not np.isfinite(float(certificate_confidence_threshold))
            or not (
                0.0 <= float(certificate_confidence_threshold) <= 1.0
                or float(certificate_confidence_threshold)
                == DISABLED_CERTIFICATE_CONFIDENCE_THRESHOLD
            )
            or not np.isfinite(float(rank_margin_threshold))
            or float(rank_margin_threshold) < 0.0
            or (
                float(certificate_confidence_threshold)
                == DISABLED_CERTIFICATE_CONFIDENCE_THRESHOLD
                and float(rank_margin_threshold) != 0.0
            )
            or not isinstance(training_centers, list)
            or not training_centers
            or tuple(str(value) for value in training_centers)
            != tuple(sorted({str(value) for value in training_centers}))
            or str(row.get("query_center_id")) in {
                str(value) for value in training_centers
            }
            or str(row.get("outer_target_id")) in {
                str(value) for value in training_centers
            }
            or tuple(str(value) for value in training_centers)
            != expected_training
            or float(certificate_confidence_threshold) != expected_threshold[0]
            or float(rank_margin_threshold) != expected_threshold[1]
            or row.get("nested_policy_fold_hash") != expected_fold_hash
            or row.get("nested_policy_replay_hash")
            != expected_nested_replay_hash
            or row.get("heldout_model_hash")
            != model_prediction.get("model_hash")
            or row.get("nested_heldout_model_hash")
            != nested_prediction.get("model_hash")
            or row.get("nested_prediction_hash")
            != nested_prediction.get("prediction_hash")
            or row.get("prediction_hash")
            != model_prediction.get("prediction_hash")
            or row.get("menu_hash") != model_prediction.get("menu_hash")
            or row.get("safe_action_ids")
            != model_prediction.get("safe_action_ids")
            or row.get("action_certificates")
            != model_prediction.get("action_certificates")
            or row.get("nested_safe_action_ids")
            != nested_prediction.get("safe_action_ids")
        ):
            raise ProtocolError("HARP v8 nested policy threshold provenance drifted.")
        _sha256(row.get("nested_policy_fold_hash"), role="nested policy fold hash")
        _sha256(
            row.get("nested_policy_replay_hash"),
            role="nested policy replay hash",
        )
        _sha256(
            row.get("nested_heldout_model_hash"),
            role="nested heldout model hash",
        )
        _sha256(
            row.get("nested_prediction_hash"),
            role="nested prediction hash",
        )


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
        raise ProtocolError("HARP v8 effective-menu/compatibility binding drifted.")
    for name in sorted(array_names):
        left = np.asarray(effective.arrays[name])
        right = np.asarray(compatibility.arrays[name])
        if (
            left.dtype != right.dtype
            or left.shape != right.shape
            or left.tobytes(order="C") != right.tobytes(order="C")
        ):
            raise ProtocolError("HARP v8 effective-menu durable arrays drifted.")
    if (
        effective.manifest.get("directions_retained") != ["D01", "D10"]
        or effective.manifest.get("all_margins_excluded") is not True
        or effective.manifest.get("exact_b_noops_removed") is not True
        or effective.manifest.get("shared_source_target_implementation") is not True
    ):
        raise ProtocolError("HARP v8 effective-menu filter contract drifted.")
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
        raise ProtocolError("HARP v8 reconstructed effective-menu inventory drifted.")
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
        raise ProtocolError("HARP v8 target/effective action inventory is absent.")
    observed: set[tuple[str, str, str]] = set()
    for ordinal, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ProtocolError("HARP v8 target/effective action row is malformed.")
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
            raise ProtocolError("HARP v8 target action escaped the effective-menu seal.")
        observed.add(key)
    if observed != set(effective):
        raise ProtocolError("HARP v8 target action inventory is not complete.")
    _validate_target_certificate_projection(target, effective=effective)


def _validate_target_certificate_projection(
    target: object,
    *,
    effective: Mapping[tuple[str, str, str], tuple[str, str, bytes]],
) -> None:
    """Bind every target estimate and safety certificate to durable numbers."""

    prediction_rows = target.manifest.get("prediction_rows")
    case_values = np.asarray(target.arrays.get("case_prediction_values"))
    certificate_values = np.asarray(target.arrays.get("action_certificate_values"))
    certificate_offsets = np.asarray(target.arrays.get("action_certificate_offsets"))
    expected_array_names = {
        "probabilities",
        "probability_offsets",
        "case_prediction_values",
        "action_certificate_values",
        "action_certificate_offsets",
    }
    if (
        not isinstance(prediction_rows, list)
        or set(target.arrays) != expected_array_names
        or case_values.dtype != np.dtype("float64")
        or case_values.shape != (len(prediction_rows), 4)
        or certificate_values.dtype != np.dtype("float64")
        or certificate_values.ndim != 2
        or certificate_values.shape[1:] != (12,)
        or certificate_offsets.dtype != np.dtype("int64")
        or certificate_offsets.shape != (len(prediction_rows) + 1,)
        or int(certificate_offsets[0]) != 0
        or np.any(np.diff(certificate_offsets) < 0)
        or int(certificate_offsets[-1]) != len(certificate_values)
        or not np.isfinite(case_values).all()
        or not np.isfinite(certificate_values).all()
    ):
        raise ProtocolError("HARP v8 target certificate array geometry drifted.")
    expected_cases: list[tuple[float, float, float, float]] = []
    expected_certificates: list[tuple[float, ...]] = []
    expected_offsets = [0]
    observed: set[tuple[str, str, str]] = set()
    try:
        for row in prediction_rows:
            if not isinstance(row, Mapping):
                raise TypeError
            outer = str(row.get("outer_target_id"))
            case_id = str(row.get("case_id"))
            raw_certificates = row.get("action_certificates")
            safe_action_ids = row.get("safe_action_ids")
            training = row.get("training_center_ids")
            candidates = row.get("training_candidate_ids")
            excluded = row.get("excluded_center_ids")
            if (
                not isinstance(raw_certificates, list)
                or not isinstance(safe_action_ids, list)
                or not isinstance(training, list)
                or not isinstance(candidates, list)
                or not isinstance(excluded, list)
                or outer not in {str(value) for value in excluded}
                or outer in {str(value) for value in training}
                or outer in {str(value) for value in candidates}
            ):
                raise TypeError
            safe_observed: list[str] = []
            predicted_harm: list[float] = []
            for certificate in raw_certificates:
                if not isinstance(certificate, Mapping):
                    raise TypeError
                action_id = str(certificate.get("action_id"))
                key = (outer, case_id, action_id)
                sealed = effective.get(key)
                is_safe = bool(certificate.get("safe"))
                if (
                    sealed is None
                    or certificate.get("action_hash") != sealed[0]
                    or key in observed
                ):
                    raise TypeError
                observed.add(key)
                if is_safe:
                    safe_observed.append(action_id)
                predicted_harm.append(
                    float(certificate.get("predicted_harm_probability"))
                )
                _sha256(
                    certificate.get("calibration_cell_hash"),
                    role="target calibration-cell hash",
                )
                _sha256(
                    certificate.get("certificate_hash"),
                    role="target action-certificate hash",
                )
                expected_certificates.append(
                    (
                        float(certificate.get("predicted_bacc_gain")),
                        float(certificate.get("predicted_harm_probability")),
                        float(certificate.get("predicted_brier_delta")),
                        float(certificate.get("predicted_log_delta")),
                        float(certificate.get("gain_lcb")),
                        float(certificate.get("harm_probability_ucb")),
                        float(certificate.get("brier_delta_ucb")),
                        float(certificate.get("log_delta_ucb")),
                        float(certificate.get("harm_brier_risk")),
                        float(certificate.get("harm_log_loss_risk")),
                        float(bool(certificate.get("model_available"))),
                        float(is_safe),
                    )
                )
            if safe_observed != [str(value) for value in safe_action_ids]:
                raise TypeError
            top = row.get("top_action_id")
            raw_top = row.get("raw_top_action_id")
            all_ids = {str(value.get("action_id")) for value in raw_certificates}
            if (top is not None and str(top) not in set(safe_observed)) or (
                raw_top is not None and str(raw_top) not in all_ids
            ):
                raise TypeError
            expected_cases.append(
                (
                    max((1.0 - value for value in predicted_harm), default=0.0),
                    float(row.get("rank_margin")),
                    float(len(safe_observed)),
                    float(len(raw_certificates)),
                )
            )
            expected_offsets.append(len(expected_certificates))
    except (TypeError, ValueError) as exc:
        raise ProtocolError("HARP v8 target certificate rows are malformed.") from exc
    expected_case_values = np.asarray(expected_cases, dtype=np.float64).reshape((-1, 4))
    expected_certificate_values = np.asarray(
        expected_certificates, dtype=np.float64
    ).reshape((-1, 12))
    expected_offset_values = np.asarray(expected_offsets, dtype=np.int64)
    if (
        observed != set(effective)
        or case_values.tobytes(order="C")
        != expected_case_values.tobytes(order="C")
        or certificate_values.tobytes(order="C")
        != expected_certificate_values.tobytes(order="C")
        or certificate_offsets.tobytes(order="C")
        != expected_offset_values.tobytes(order="C")
    ):
        raise ProtocolError("HARP v8 target certificate row/value binding drifted.")


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
        raise ProtocolError("HARP v8 persisted target-action table geometry drifted.")
    table: dict[tuple[str, str, str], tuple[tuple[str, ...], np.ndarray]] = {}
    for ordinal, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ProtocolError("HARP v8 persisted target-action row is malformed.")
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
            raise ProtocolError("HARP v8 persisted target-action binding drifted.")
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
        raise ProtocolError("HARP v8 target actions escaped the reconstructed menus.")
    for case in routes.cases:
        menu = menus[case.outer_target_id]
        baseline = menu.target_block(ActionKind.B)
        uniform = menu.target_block(ActionKind.U)
        indices = np.flatnonzero(
            np.asarray(baseline.case_ids, dtype=object) == case.case_id
        )
        if not len(indices):
            raise ProtocolError("HARP v8 route case is absent from its physical menu.")
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
            raise ProtocolError("HARP v8 routed B/U bytes escaped the physical menu.")
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
                raise ProtocolError("HARP v8 reconstructed abstention is not exact B.")
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
            raise ProtocolError("HARP v8 reconstructed route is not exact top-1.")
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
                    "HARP v8 routed component is not the named persisted target action."
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
        raise ProtocolError("HARP v8 validator center universe drifted.")
    config_hash = _sha256(expected_config_hash, role="config hash")
    if set(menu_roots) != set(centers):
        raise ProtocolError("HARP v8 validator menu target universe drifted.")
    menus = {
        center: read_label_free_outer_menu(Path(menu_roots[center]))
        for center in centers
    }
    if any(menu.outer_target_id != center for center, menu in menus.items()):
        raise ProtocolError("HARP v8 validator menu/root identity drifted.")
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
        raise ProtocolError("HARP v8 route/model/action binding drifted.")
    _validate_route_inputs(routes, menus, target)
    observed_centers = tuple(sorted({case.outer_target_id for case in routes.cases}))
    if observed_centers != centers:
        raise ProtocolError("HARP v8 routed center coverage drifted.")
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
        raise ProtocolError("HARP v8 reconstructed route violates outer exclusion.")
    compatibility_hash = None
    effective_menu_hash = None
    if compatibility is not None:
        compatibility_hash = _verified_artifact_hash(
            compatibility, "compatibility_hash", role="compatibility hash"
        )
        if target.manifest.get("compatibility_hash") != compatibility_hash:
            raise ProtocolError("HARP v8 target/compatibility binding drifted.")
        if model.manifest.get("compatibility_hash") != compatibility_hash:
            raise ProtocolError("HARP v8 model/compatibility binding drifted.")
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
            "HARP v8 fresh validation requires compatibility for the effective-menu seal."
        )
    admission_hash = None
    if admission is not None:
        admission_hash = _verified_artifact_hash(
            admission, "admission_hash", role="admission hash"
        )
        if target.manifest.get("admission_hash") != admission_hash:
            raise ProtocolError("HARP v8 target/admission binding drifted.")
        if (
            admission.manifest.get("model_hash") != model_hash
            or admission.manifest.get("development_surface_hash")
            != development_hash
        ):
            raise ProtocolError("HARP v8 admission/model/development binding drifted.")
    else:
        raise ProtocolError(
            "HARP v8 fresh validation requires the source policy OOF replay."
        )
    _validate_numeric_oof(model, admission)
    payload = {
        "schema_version": "midogpp_harp_v8_fresh_route_reconstruction_v1",
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
                name=f"harp-v8-{validator_id}",
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
                    raise ProtocolError("HARP v8 fresh validation timed out.")
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
                            f"HARP v8 fresh validator exited before receipt: {failed}."
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
        raise ProtocolError(f"HARP v8 fresh validation failed: {detail}.")
    first = messages["fresh_reconstruction_A"][1]
    second = messages["fresh_reconstruction_B"][1]
    if not isinstance(first, dict) or not isinstance(second, dict):
        raise ProtocolError("HARP v8 fresh validator returned an invalid result.")
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
        raise ProtocolError("HARP v8 fresh reconstruction identities drifted.")
    return first, second


__all__ = ("reconstruct_prelabel_routes", "run_two_fresh_validations")
