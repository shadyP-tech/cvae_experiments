from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from midogpp_thesis.common.midogpp import MIDOGPP_ELIGIBLE_CENTERS
from midogpp_thesis.data.features.cache_io import write_center_shard
from midogpp_thesis.data.physical_multiscale.bridge import (
    evaluate_jpeg_task_bridge,
)
from midogpp_thesis.real_features.classifier_reference.classifiers import (
    ClassifierSpec,
    fit_logistic_classifier,
)
from midogpp_thesis.real_features.classifier_reference.downstream import (
    balanced_accuracy,
)


def test_jpeg_task_bridge_replays_canonical_predictions_and_fails_closed(
    tmp_path: Path,
) -> None:
    b_root, reference_root = _write_bridge_fixture(tmp_path)

    report = evaluate_jpeg_task_bridge(b_root, reference_root)

    assert report["status"] == "PASS"
    assert report["prediction_agreement"] == 1.0
    assert report["absolute_equal_center_bacc_delta"] == 0.0
    assert set(report["canonical_reference_hashes"]) == {
        "protocol_manifest_sha256",
        "source_results_sha256",
        "predictions_sha256",
    }

    result_path = reference_root / "tables" / "classifier_tuned_source_results.csv"
    results = _csv(result_path)
    original_result = dict(results[0])
    unbalanced = ClassifierSpec(
        C=0.01,
        penalty="l2",
        solver="lbfgs",
        max_iter=5000,
        class_weight=None,
        random_state=23,
        threshold_policy="predict",
    )
    results[0]["selected_classifier_config_hash"] = unbalanced.config_hash
    results[0]["selected_classifier_spec"] = json.dumps(
        unbalanced.to_payload(), sort_keys=True
    )
    _write_csv(result_path, results)
    with pytest.raises(ValueError, match="classifier hash drifted"):
        evaluate_jpeg_task_bridge(b_root, reference_root)
    results[0] = original_result
    _write_csv(result_path, results)

    prediction_path = (
        reference_root / "tables" / "classifier_tuned_predictions.csv"
    )
    predictions = _csv(prediction_path)
    predictions[0]["y_pred"] = str(1 - int(predictions[0]["y_pred"]))
    _write_csv(prediction_path, predictions)
    with pytest.raises(ValueError, match="Task-semantic JPEG bridge failed"):
        evaluate_jpeg_task_bridge(b_root, reference_root)


def _write_bridge_fixture(tmp_path: Path) -> tuple[Path, Path]:
    b_root = tmp_path / "b"
    reference_root = tmp_path / "reference"
    rng = np.random.default_rng(42)
    embeddings_by_center: dict[str, np.ndarray] = {}
    labels_by_center: dict[str, np.ndarray] = {}
    metadata_by_center: dict[str, list[dict[str, object]]] = {}
    for center_index, center in enumerate(MIDOGPP_ELIGIBLE_CENTERS):
        labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=int)
        a = np.zeros((len(labels), 2560), dtype=np.float32)
        a[:, 0] = labels * 3.0 + rng.normal(0.0, 0.05, len(labels))
        a[:, 1] = (1 - labels) * 2.0 + center_index * 0.001
        b = np.zeros((len(labels), 3840), dtype=np.float32)
        b[:, :2560] = a
        metadata = [
            {
                "sample_id": f"center_{center}_sample_{index}",
                "case_id": f"case_{center}_{index}",
                "label": int(label),
                "split": "train",
                "center": center,
            }
            for index, label in enumerate(labels)
        ]
        write_center_shard(
            b_root / "embeddings" / "by_center" / f"center_{center}.pt",
            embeddings=torch.from_numpy(b),
            canonical_a_embeddings=torch.from_numpy(a),
            metadata=metadata,
            feature_extractor={"representation_id": "jpeg_center_b"},
        )
        embeddings_by_center[center] = a
        labels_by_center[center] = labels
        metadata_by_center[center] = metadata

    results: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    for heldout in MIDOGPP_ELIGIBLE_CENTERS:
        spec = ClassifierSpec(
            C=0.01,
            penalty="l2",
            solver="lbfgs",
            max_iter=5000,
            class_weight=(
                "balanced" if heldout in {"0", "1", "2", "9"} else None
            ),
            random_state=23,
            threshold_policy="predict",
        )
        sources = tuple(
            center for center in MIDOGPP_ELIGIBLE_CENTERS if center != heldout
        )
        fitted = fit_logistic_classifier(
            np.concatenate(tuple(embeddings_by_center[center] for center in sources)),
            np.concatenate(tuple(labels_by_center[center] for center in sources)),
            embeddings_by_center[heldout],
            spec=spec,
        )
        predicted = [int(value) for value in fitted.predictions.tolist()]
        results.append(
            {
                "heldout_center": heldout,
                "selected_classifier_config_hash": spec.config_hash,
                "selected_classifier_spec": json.dumps(
                    spec.to_payload(), sort_keys=True
                ),
                "heldout_bacc": balanced_accuracy(
                    labels_by_center[heldout].tolist(), predicted
                ),
            }
        )
        predictions.extend(
            {
                "heldout_center": heldout,
                "sample_id": metadata["sample_id"],
                "y_pred": prediction,
            }
            for metadata, prediction in zip(
                metadata_by_center[heldout], predicted, strict=True
            )
        )
    (reference_root / "manifests").mkdir(parents=True)
    (reference_root / "tables").mkdir()
    (reference_root / "manifests" / "protocol_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "midogpp_eligible_tuned_real_reference_v2",
                "protocol_hash": "786589b799d61b14",
                "reference_bundle_hash": "995aa193c82ee7ec",
                "claim_scope": "real_feature_transfer_only",
                "classifier_grid_hash": "5abd0897d02bdcaa",
                "coverage_mode": "complete",
                "heldout_centers": list(MIDOGPP_ELIGIBLE_CENTERS),
                "excluded_centers": ["4"],
                "threshold_policy": "predict",
                "selection_used_target_labels": False,
                "fit_used_target_center": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(
        reference_root / "tables" / "classifier_tuned_source_results.csv",
        results,
    )
    _write_csv(
        reference_root / "tables" / "classifier_tuned_predictions.csv",
        predictions,
    )
    return b_root, reference_root


def _write_csv(
    path: Path,
    rows: list[dict[str, object]] | list[dict[str, str]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]
