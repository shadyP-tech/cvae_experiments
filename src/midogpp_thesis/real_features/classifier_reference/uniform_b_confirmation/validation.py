"""Independent validation of prospective uniform-B confirmation bundles."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Mapping

from midogpp_thesis.common.hashing import stable_hash

from ..downstream import balanced_accuracy, macro_f1
from ..protocol import ProtocolError
from .artifacts import (
    REQUIRED_FILES,
    confirmation_input_hashes,
    prospective_paired_case_bootstrap,
    read_csv,
    read_json,
    validate_confirmation_inputs,
    validate_content_index,
)
from .config import CANONICAL_A, EVALUATION_SPLIT, UNIFORM_B, UniformBConfirmationConfig
from .runner import _summary


def validate_uniform_b_confirmation_bundle(
    root: str | Path,
    *,
    config: UniformBConfirmationConfig,
    allow_pending: bool = False,
) -> dict[str, object]:
    path = Path(root)
    required = set(REQUIRED_FILES)
    if allow_pending:
        required.remove("reports/validation_report.json")
    missing = sorted(relative for relative in required if not (path / relative).is_file())
    if missing:
        raise ProtocolError(f"Uniform-B prospective bundle is incomplete: {missing}.")
    validate_confirmation_inputs(config)
    hashes = confirmation_input_hashes(config)
    frozen = read_json(path / "manifests/frozen_protocol_snapshot.json")
    protocol = read_json(path / "manifests/protocol_manifest.json")
    representation = read_json(path / "manifests/uniform_representation_lock.json")
    source_index = read_json(path / "manifests/source_lock_index.json")
    leakage = read_json(path / "reports/leakage_provenance_report.json")
    expected_status = "PENDING_INDEPENDENT_VALIDATION" if allow_pending else "PASS"
    if (
        stable_hash({key: value for key, value in frozen.items() if key != "protocol_hash"})
        != frozen.get("protocol_hash")
        or protocol.get("status") != expected_status
        or protocol.get("input_hashes") != hashes
        or protocol.get("protocol_hash") != frozen.get("protocol_hash")
        or protocol.get("evaluation_split") != EVALUATION_SPLIT
        or protocol.get("independent_confirmation_within_observed_centers") is not True
        or protocol.get("external_dataset_confirmation") is not False
        or protocol.get("covers_new_center_uncertainty") is not False
        or representation.get("representation_id") != UNIFORM_B
        or representation.get("choice_pre_test_b_extraction_and_scoring") is not True
        or representation.get("test_outcomes_used_for_choice") is not False
        or stable_hash(
            {key: value for key, value in representation.items() if key != "uniform_lock_hash"}
        ) != representation.get("uniform_lock_hash")
        or source_index.get("lock_count") != len(config.heldout_centers)
        or stable_hash(
            {key: value for key, value in source_index.items() if key != "source_lock_bundle_hash"}
        ) != source_index.get("source_lock_bundle_hash")
        or leakage.get("fit_leakage_status") != "PASS"
        or leakage.get("train_test_case_overlap") != 0
        or leakage.get("test_labels_used_for_fit_selection_or_feature_extraction") is not False
    ):
        raise ProtocolError("Uniform-B prospective protocol/claim validation failed.")
    if not allow_pending and (
        protocol.get("independent_validation_status") != "PASS"
        or leakage.get("status") != "PASS"
        or leakage.get("independent_validation_status") != "PASS"
    ):
        raise ProtocolError("Uniform-B prospective final validation status failed.")

    lock_rows = read_csv(path / "tables/source_lock_audit.csv")
    splits = read_csv(path / "tables/split_isolation_audit.csv")
    results = read_csv(path / "tables/prospective_test_results.csv")
    predictions = read_csv(path / "tables/prospective_test_predictions.csv")
    comparisons = read_csv(path / "tables/paired_center_comparison.csv")
    audits = read_csv(path / "tables/outer_fit_audit.csv")
    n_centers = len(config.heldout_centers)
    if (
        len(lock_rows) != n_centers
        or len(splits) != n_centers
        or len(results) != 2 * n_centers
        or len(comparisons) != n_centers
        or len(audits) != 2 * n_centers
        or len(predictions) != 2 * 9928
    ):
        raise ProtocolError("Uniform-B prospective artifact cardinality failed.")
    if any(
        row.get("status") != "PASS"
        or row.get("sample_overlap") != "0"
        or row.get("case_overlap") != "0"
        or row.get("validation_split_used") != "False"
        for row in splits
    ):
        raise ProtocolError("Uniform-B prospective split isolation audit failed.")
    if any(
        row.get("target_center_absent_from_fit") != "True"
        or row.get("scaler_fit_on_train_source_only") != "True"
        or row.get("test_labels_used_for_fit_or_selection") != "False"
        for row in audits
    ):
        raise ProtocolError("Uniform-B prospective fit audit failed.")
    _validate_metrics(config, results, predictions, comparisons)
    bootstrap = prospective_paired_case_bootstrap(
        predictions,
        seed=config.bootstrap.seed,
        valid_replicates=config.bootstrap.valid_replicates,
        max_attempts=config.bootstrap.max_attempts,
    )
    if read_json(path / "reports/conditional_bootstrap.json") != bootstrap:
        raise ProtocolError("Uniform-B prospective bootstrap reconstruction failed.")
    expected_summary = _summary(config, [dict(row) for row in comparisons], bootstrap)
    if read_json(path / "reports/confirmation_summary.json") != expected_summary:
        raise ProtocolError("Uniform-B prospective decision reconstruction failed.")
    validate_content_index(path)
    checks = {
        "status": "PASS",
        "source_locks": len(lock_rows),
        "outer_results": len(results),
        "outer_predictions": len(predictions),
        "center_comparisons": len(comparisons),
        "test_rows": len(predictions) // 2,
    }
    if not allow_pending:
        report = read_json(path / "reports/validation_report.json")
        if (
            report.get("status") != "PASS"
            or report.get("validator") != "validate_uniform_b_confirmation_bundle"
            or report.get("checks") != checks
        ):
            raise ProtocolError("Uniform-B prospective validation report failed.")
    return checks


def _validate_metrics(
    config: UniformBConfirmationConfig,
    results: list[dict[str, str]],
    predictions: list[dict[str, str]],
    comparisons: list[dict[str, str]],
) -> None:
    result_by = {(row["heldout_center"], row["role"]): row for row in results}
    comparison_by = {row["heldout_center"]: row for row in comparisons}
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in predictions:
        if row.get("evaluation_split") != EVALUATION_SPLIT:
            raise ProtocolError("Uniform-B prospective prediction split drifted.")
        grouped[(row["heldout_center"], row["role"])].append(row)
    for center in config.heldout_centers:
        identities = []
        computed = {}
        for role in ("canonical_a", "uniform_b"):
            rows = grouped[(center, role)]
            identities.append({(row["sample_id"], row["case_id"], row["label"]) for row in rows})
            labels = [int(row["label"]) for row in rows]
            pred = [int(row["prediction"]) for row in rows]
            computed[role] = (balanced_accuracy(labels, pred), macro_f1(labels, pred))
            result = result_by[(center, role)]
            if (
                float(result["bacc"]) != computed[role][0]
                or float(result["macro_f1"]) != computed[role][1]
                or int(result["n_eval"]) != len(rows)
            ):
                raise ProtocolError(f"Uniform-B prospective metric drift: {center}/{role}.")
        if identities[0] != identities[1]:
            raise ProtocolError(f"Uniform-B prospective A/B pools differ: {center}.")
        comparison = comparison_by[center]
        if (
            float(comparison["delta_bacc"]) != computed["uniform_b"][0] - computed["canonical_a"][0]
            or comparison.get("outcome_was_untouched_at_protocol_lock") != "True"
        ):
            raise ProtocolError(f"Uniform-B prospective comparison drift: {center}.")
