from __future__ import annotations

import csv
import json
from itertools import product
from pathlib import Path

import numpy as np

from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router import (
    source_inner_evidence as evidence_io,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.source_inner_authorization import (
    SourceInnerReuseReceipt,
)
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)


def test_label_free_prediction_packet_builds_strict_outer_and_target_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    arrays = tmp_path / "arrays"
    manifests = tmp_path / "manifests"
    tables = tmp_path / "tables"
    arrays.mkdir()
    manifests.mkdir()
    tables.mkdir()

    fit_grid = tuple(product(CENTERS, TRAINING_SEEDS, GENERATION_SEEDS))
    probability = np.empty((len(fit_grid), len(CENTERS)), dtype=np.float32)
    for ordinal, (source, training_seed, generation_seed) in enumerate(fit_grid):
        base = 0.1 + 0.08 * CENTERS.index(source)
        probability[ordinal] = np.clip(
            base + 1e-4 * training_seed + 1e-5 * generation_seed,
            0.01,
            0.99,
        )
    prediction = (probability >= 0.5).astype(np.uint8)
    array_path = arrays / "candidate_predictions.npz"
    np.savez_compressed(array_path, y_pred=prediction, prob_pos=probability)

    fit_path = tables / "classifier_fits.csv"
    with fit_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = (
            "prediction_array_row",
            "source_center",
            "training_seed",
            "generation_seed",
            "source_stream_id",
            "eval_labels_available_to_fit_or_predict",
            "seed_selection_performed",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for ordinal, (source, training_seed, generation_seed) in enumerate(fit_grid):
            writer.writerow(
                {
                    "prediction_array_row": ordinal,
                    "source_center": source,
                    "training_seed": training_seed,
                    "generation_seed": generation_seed,
                    "source_stream_id": f"stream-{ordinal}",
                    "eval_labels_available_to_fit_or_predict": False,
                    "seed_selection_performed": False,
                }
            )

    evaluation_path = tables / "evaluation_rows.csv"
    with evaluation_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ("row_ordinal", "sample_id", "case_id", "center", "label_present")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for ordinal, center in enumerate(CENTERS):
            writer.writerow(
                {
                    "row_ordinal": ordinal,
                    "sample_id": f"sample-{center}",
                    "case_id": f"case-{center}",
                    "center": center,
                    "label_present": False,
                }
            )

    probability_hash = evidence_io._array_sha256(probability)  # noqa: SLF001
    prediction_hash = evidence_io._array_sha256(prediction)  # noqa: SLF001
    index_path = manifests / "prediction_index.json"
    index_path.write_text(
        json.dumps(
            {
                "allowed_array_keys": ["y_pred", "prob_pos"],
                "array_member": "arrays/candidate_predictions.npz",
                "eval_labels_available_to_fit_or_predict": False,
                "eval_row_count": len(CENTERS),
                "fit_count": len(fit_grid),
                "labels_stored": False,
                "prediction_dtype": "uint8",
                "prediction_shape": [len(fit_grid), len(CENTERS)],
                "probability_dtype": "float32",
                "probability_shape": [len(fit_grid), len(CENTERS)],
                "probability_array_sha256": probability_hash,
                "prediction_array_sha256": prediction_hash,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    file_hashes = {
        "prediction": evidence_io.file_sha256(array_path),
        "index": evidence_io.file_sha256(index_path),
        "fits": evidence_io.file_sha256(fit_path),
        "evaluation": evidence_io.file_sha256(evaluation_path),
    }
    monkeypatch.setattr(
        evidence_io,
        "EXPECTED_SOURCE_EVALUATION_ROW_COUNT",
        len(CENTERS),
    )
    monkeypatch.setattr(
        evidence_io,
        "EXPECTED_SOURCE_PREDICTION_ARRAY_FILE_SHA256",
        file_hashes["prediction"],
    )
    monkeypatch.setattr(
        evidence_io, "EXPECTED_SOURCE_PREDICTION_INDEX_SHA256", file_hashes["index"]
    )
    monkeypatch.setattr(
        evidence_io, "EXPECTED_SOURCE_CLASSIFIER_FITS_SHA256", file_hashes["fits"]
    )
    monkeypatch.setattr(
        evidence_io, "EXPECTED_SOURCE_EVALUATION_ROWS_SHA256", file_hashes["evaluation"]
    )
    monkeypatch.setattr(
        evidence_io, "EXPECTED_PROBABILITY_ARRAY_SHA256", probability_hash
    )
    monkeypatch.setattr(
        evidence_io, "EXPECTED_PREDICTION_ARRAY_SHA256", prediction_hash
    )
    # This unit fixture exercises the byte/schema loader. The repository-owned
    # amendment and consumer fence are covered separately by governance tests.
    monkeypatch.setattr(
        evidence_io,
        "validate_source_inner_reuse_receipt",
        lambda receipt: receipt,
    )

    receipt = SourceInnerReuseReceipt(
        amendment_id="test",
        amendment_sha256="a" * 64,
        consumer_experiment_id="test",
        input_alias_id="test",
        utility_lock_sha256="b" * 64,
        utility_table_sha256="c" * 64,
        case_confusions_sha256="d" * 64,
        prediction_array_file_sha256=file_hashes["prediction"],
        prediction_index_sha256=file_hashes["index"],
        classifier_fits_sha256=file_hashes["fits"],
        evaluation_rows_sha256=file_hashes["evaluation"],
        publication_status="test",
        terminal_decision="test",
    )
    surface = evidence_io.load_source_inner_prediction_surface(
        tmp_path, receipt=receipt
    )
    outer = evidence_io.build_outer_raw_evidence(surface, outer_target="2")
    target = evidence_io.build_target_raw_evidence(surface, target_center="2")

    assert len(outer.rows) == 56
    assert all(row.query_center != "2" for row in outer.rows)
    assert all(row.candidate_center != "2" for row in outer.rows)
    assert len(target.rows) == 8
    assert {row.candidate_center for row in target.rows} == set(CENTERS) - {"2"}
    assert all(row.labels_consumed is False for row in (*outer.rows, *target.rows))
    assert all(row.exact_nelbo is False for row in (*outer.rows, *target.rows))
