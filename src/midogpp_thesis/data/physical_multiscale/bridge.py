"""Task-semantic validation of the newly extracted JPEG A-prime bridge."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import warnings

from midogpp_thesis.common.midogpp import MIDOGPP_ELIGIBLE_CENTERS


MINIMUM_PREDICTION_AGREEMENT = 0.999
MAXIMUM_EQUAL_CENTER_BACC_DELTA = 0.001
MINIMUM_COSINE = 0.99999
MAXIMUM_RELATIVE_L2 = 0.001
ALLOWED_REFERENCE_SPEC_HASHES = {
    "86378e6ceb12136e": None,
    "878e04f48c4f8c04": "balanced",
}
EXPECTED_REFERENCE_SPEC_BY_CENTER = {
    "0": "878e04f48c4f8c04",
    "1": "878e04f48c4f8c04",
    "2": "878e04f48c4f8c04",
    "3": "86378e6ceb12136e",
    "5": "86378e6ceb12136e",
    "6": "86378e6ceb12136e",
    "7": "86378e6ceb12136e",
    "8": "86378e6ceb12136e",
    "9": "878e04f48c4f8c04",
}
EXPECTED_REFERENCE_PROTOCOL_HASH = "786589b799d61b14"
EXPECTED_REFERENCE_BUNDLE_HASH = "995aa193c82ee7ec"


def evaluate_jpeg_task_bridge(
    b_cache_root: str | Path,
    canonical_reference_root: str | Path,
    *,
    centers: Sequence[str] = MIDOGPP_ELIGIBLE_CENTERS,
    minimum_prediction_agreement: float = MINIMUM_PREDICTION_AGREEMENT,
    maximum_equal_center_bacc_delta: float = MAXIMUM_EQUAL_CENTER_BACC_DELTA,
) -> Mapping[str, object]:
    """Replay frozen reference classifiers on B's newly extracted A-prime prefix."""

    try:
        import numpy as np  # type: ignore
        import torch  # type: ignore
        from sklearn.exceptions import ConvergenceWarning  # type: ignore
        from sklearn.linear_model import LogisticRegression  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - declared dependencies
        raise RuntimeError(
            "Task-semantic JPEG bridge validation requires numpy, torch, and scikit-learn."
        ) from exc

    center_ids = tuple(str(center) for center in centers)
    if center_ids != MIDOGPP_ELIGIBLE_CENTERS:
        raise ValueError("Production JPEG task bridge requires exact eligible centers.")
    reference_root = Path(canonical_reference_root)
    protocol_path = reference_root / "manifests" / "protocol_manifest.json"
    result_path = reference_root / "tables" / "classifier_tuned_source_results.csv"
    prediction_path = reference_root / "tables" / "classifier_tuned_predictions.csv"
    comparator_paths = {
        "protocol_manifest_sha256": protocol_path,
        "source_results_sha256": result_path,
        "predictions_sha256": prediction_path,
    }
    missing = [
        label for label, comparator_path in comparator_paths.items()
        if not comparator_path.is_file()
    ]
    if missing:
        raise ValueError(f"Canonical task bridge comparator is incomplete: {missing}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("schema_version")
        != "midogpp_eligible_tuned_real_reference_v2"
        or protocol.get("protocol_hash") != EXPECTED_REFERENCE_PROTOCOL_HASH
        or protocol.get("reference_bundle_hash") != EXPECTED_REFERENCE_BUNDLE_HASH
        or protocol.get("claim_scope") != "real_feature_transfer_only"
        or protocol.get("classifier_grid_hash") != "5abd0897d02bdcaa"
        or protocol.get("coverage_mode") != "complete"
        or protocol.get("heldout_centers") != list(MIDOGPP_ELIGIBLE_CENTERS)
        or protocol.get("excluded_centers") != ["4"]
        or protocol.get("threshold_policy") != "predict"
        or protocol.get("selection_used_target_labels") is not False
        or protocol.get("fit_used_target_center") is not False
    ):
        raise ValueError("Canonical task bridge protocol manifest failed validation.")
    result_rows = _read_csv(result_path)
    prediction_rows = _read_csv(prediction_path)
    results_by_center = _unique_by(result_rows, "heldout_center", "reference results")
    expected_predictions = {
        (str(row["heldout_center"]), str(row["sample_id"])): int(
            float(str(row["y_pred"]))
        )
        for row in prediction_rows
    }
    embeddings: list[Any] = []
    labels: list[int] = []
    observed_centers: list[str] = []
    sample_ids: list[str] = []
    for center in center_ids:
        payload = _torch_payload(
            torch,
            Path(b_cache_root)
            / "embeddings"
            / "by_center"
            / f"center_{center}.pt",
        )
        array = np.asarray(payload["embeddings"], dtype=float)
        metadata = tuple(dict(row) for row in payload["metadata"])
        if array.shape != (len(metadata), 3840):
            raise ValueError(f"B shard dimension drift during task bridge: {center}.")
        embeddings.append(array[:, :2560])
        labels.extend(int(row["label"]) for row in metadata)
        observed_centers.extend(str(row["center"]) for row in metadata)
        sample_ids.extend(str(row["sample_id"]) for row in metadata)
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Task bridge sample IDs are duplicated.")
    x = np.concatenate(embeddings, axis=0)
    y = np.asarray(labels, dtype=int)
    center_array = np.asarray(observed_centers, dtype=str)
    agreements = 0
    total = 0
    bridge_bacc: list[float] = []
    canonical_bacc: list[float] = []
    fold_rows: list[Mapping[str, object]] = []
    for heldout in center_ids:
        reference = results_by_center.get(heldout)
        if reference is None:
            raise ValueError(f"Canonical reference result is missing H={heldout}.")
        spec_hash = str(reference.get("selected_classifier_config_hash", ""))
        if spec_hash != EXPECTED_REFERENCE_SPEC_BY_CENTER[heldout]:
            raise ValueError(
                f"Canonical reference classifier hash drifted for H={heldout}: {spec_hash}"
            )
        _validate_reference_spec(reference, spec_hash)
        fit_mask = center_array != heldout
        eval_mask = center_array == heldout
        scaler = StandardScaler()
        x_fit = scaler.fit_transform(x[fit_mask])
        x_eval = scaler.transform(x[eval_mask])
        classifier = LogisticRegression(
            C=0.01,
            penalty="l2",
            solver="lbfgs",
            max_iter=5000,
            class_weight=ALLOWED_REFERENCE_SPEC_HASHES[spec_hash],
            random_state=23,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            classifier.fit(x_fit, y[fit_mask])
        if any(issubclass(item.category, ConvergenceWarning) for item in caught):
            raise ValueError(f"JPEG task bridge classifier did not converge for H={heldout}.")
        predicted = classifier.predict(x_eval).astype(int)
        eval_ids = [
            sample_id
            for sample_id, center in zip(sample_ids, observed_centers, strict=True)
            if center == heldout
        ]
        expected = []
        for sample_id in eval_ids:
            key = (heldout, sample_id)
            if key not in expected_predictions:
                raise ValueError(f"Canonical bridge prediction is missing: {key}")
            expected.append(expected_predictions[key])
        unexpected_reference = {
            sample_id
            for center, sample_id in expected_predictions
            if center == heldout
        }.difference(eval_ids)
        if unexpected_reference:
            raise ValueError(
                f"Canonical bridge contains unexpected H={heldout} samples: "
                f"{sorted(unexpected_reference)[:5]}"
            )
        fold_agreements = sum(
            int(actual == expected_value)
            for actual, expected_value in zip(predicted.tolist(), expected, strict=True)
        )
        agreements += fold_agreements
        total += len(expected)
        bacc = _balanced_accuracy(y[eval_mask].tolist(), predicted.tolist())
        canonical = float(reference["heldout_bacc"])
        bridge_bacc.append(bacc)
        canonical_bacc.append(canonical)
        fold_rows.append(
            {
                "heldout_center": heldout,
                "selected_classifier_config_hash": spec_hash,
                "n_eval": len(expected),
                "prediction_agreement": fold_agreements / float(len(expected)),
                "bridge_bacc": bacc,
                "canonical_bacc": canonical,
                "bacc_delta": bacc - canonical,
            }
        )
    agreement = agreements / float(total)
    mean_bridge = sum(bridge_bacc) / float(len(bridge_bacc))
    mean_canonical = sum(canonical_bacc) / float(len(canonical_bacc))
    absolute_delta = abs(mean_bridge - mean_canonical)
    status = (
        "PASS"
        if agreement >= minimum_prediction_agreement
        and absolute_delta <= maximum_equal_center_bacc_delta
        else "FAIL"
    )
    report = {
        "schema_version": "midogpp_virchow2_jpeg_task_bridge_v1",
        "status": status,
        "prediction_agreement": agreement,
        "minimum_prediction_agreement": minimum_prediction_agreement,
        "prediction_matches": agreements,
        "prediction_count": total,
        "equal_center_mean_bridge_bacc": mean_bridge,
        "equal_center_mean_canonical_bacc": mean_canonical,
        "absolute_equal_center_bacc_delta": absolute_delta,
        "maximum_absolute_equal_center_bacc_delta": maximum_equal_center_bacc_delta,
        "canonical_reference_hashes": {
            label: _sha256(comparator_path)
            for label, comparator_path in comparator_paths.items()
        },
        "frozen_reference_spec_reused_without_reselection": True,
        "folds": fold_rows,
    }
    if status != "PASS":
        raise ValueError(f"Task-semantic JPEG bridge failed: {report}")
    return report


def _validate_reference_spec(
    row: Mapping[str, str],
    expected_hash: str,
) -> None:
    raw = json.loads(str(row.get("selected_classifier_spec", "{}")))
    expected_weight = ALLOWED_REFERENCE_SPEC_HASHES[expected_hash]
    if (
        not isinstance(raw, Mapping)
        or float(raw.get("C", -1.0)) != 0.01
        or raw.get("penalty") != "l2"
        or raw.get("solver") != "lbfgs"
        or int(raw.get("max_iter", -1)) != 5000
        or raw.get("class_weight") != expected_weight
        or int(raw.get("random_state", -1)) != 23
        or raw.get("threshold_policy") != "predict"
    ):
        raise ValueError("Canonical JPEG bridge classifier specification drifted.")


def _balanced_accuracy(labels: Sequence[int], predictions: Sequence[int]) -> float:
    recalls = []
    for label in (0, 1):
        indexes = [index for index, value in enumerate(labels) if int(value) == label]
        if not indexes:
            raise ValueError("JPEG bridge BACC requires both target classes.")
        recalls.append(
            sum(int(predictions[index]) == label for index in indexes) / float(len(indexes))
        )
    return sum(recalls) / 2.0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _unique_by(
    rows: Sequence[Mapping[str, str]],
    key: str,
    label: str,
) -> dict[str, Mapping[str, str]]:
    out: dict[str, Mapping[str, str]] = {}
    for row in rows:
        value = str(row.get(key, ""))
        if not value or value in out:
            raise ValueError(f"{label} has missing or duplicate {key}: {value!r}")
        out[value] = row
    return out


def _torch_payload(torch: Any, path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older torch
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected cache mapping: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
