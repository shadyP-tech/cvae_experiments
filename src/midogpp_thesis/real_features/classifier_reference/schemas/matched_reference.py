"""Eligible-only predict-policy matched real-reference schema."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from ..artifacts import stable_hash
from ..classifiers import ClassifierSpec, classifier_grid_hash
from ..downstream import balanced_accuracy, macro_f1
from ..protocol import ProtocolError
from .midogpp import MIDOGPP_ELIGIBLE_CENTERS, MIDOGPP_EXCLUDED_CENTERS


MATCHED_REFERENCE_SCHEMA_VERSION = "midogpp_eligible_tuned_real_reference_v2"
MATCHED_REFERENCE_RESULT_SCHEMA_VERSION = "midogpp_eligible_tuned_real_result_v2"
MATCHED_REFERENCE_PREDICTION_SCHEMA_VERSION = "midogpp_eligible_tuned_real_prediction_v2"
MATCHED_REFERENCE_METHOD = "source_inner_tuned_predict"

MATCHED_REFERENCE_RESULT_COLUMNS = (
    "schema_version",
    "method",
    "protocol_hash",
    "experiment_seed",
    "classifier_seed",
    "heldout_center",
    "train_centers",
    "n_train",
    "n_eval",
    "fit_row_hash",
    "eval_row_hash",
    "classifier_grid_hash",
    "selected_classifier_config_hash",
    "selected_classifier_spec",
    "selection_metric",
    "selection_source",
    "source_inner_center_bacc_vector",
    "source_inner_mean_bacc",
    "heldout_bacc",
    "heldout_macro_f1",
    "converged",
    "n_iter",
    "status",
    "feature_cache_hash",
    "manifest_hash",
    "target_eval_labels_used_for_scoring_only",
    "selection_used_target_labels",
    "fit_used_target_center",
    "generated_embeddings_used",
    "cvae_checkpoint_used",
    "source_summary_manifest_used",
    "is_router",
    "claim_scope",
    "claim_role",
    "row_role",
    "leakage_status",
    "support_labels_used",
    "oracle_eligible",
    "probabilities_calibrated",
)

MATCHED_REFERENCE_PREDICTION_COLUMNS = (
    "schema_version",
    "method",
    "protocol_hash",
    "heldout_center",
    "sample_id",
    "case_id",
    "center",
    "y_true",
    "y_pred",
    "prob_pos",
    "selected_classifier_config_hash",
    "eval_row_hash",
    "claim_role",
    "row_role",
    "leakage_status",
    "support_labels_used",
    "oracle_eligible",
    "target_eval_labels_used_for_scoring_only",
)

MATCHED_REFERENCE_REQUIRED_OUTPUTS = (
    "tables/source_inner_classifier_tuning.csv",
    "tables/classifier_tuned_source_results.csv",
    "tables/classifier_tuned_predictions.csv",
    "manifests/protocol_manifest.json",
    "reports/leakage_provenance_report.json",
)


def assert_matched_reference_artifacts(root: Path) -> None:
    root = Path(root)
    missing = [relative for relative in MATCHED_REFERENCE_REQUIRED_OUTPUTS if not (root / relative).exists()]
    if missing:
        raise ProtocolError(f"Matched-reference artifact missing outputs: {missing}")
    protocol = _read_json(root / "manifests/protocol_manifest.json")
    leakage = _read_json(root / "reports/leakage_provenance_report.json")
    _validate_protocol(protocol, leakage)
    _validate_workspace_provenance(root, protocol)
    result_rows = _read_csv(root / "tables/classifier_tuned_source_results.csv")
    prediction_rows = _read_csv(root / "tables/classifier_tuned_predictions.csv")
    tuning_rows = _read_csv(root / "tables/source_inner_classifier_tuning.csv")
    bundle_hash = matched_reference_bundle_hash(tuning_rows, result_rows, prediction_rows)
    if protocol.get("reference_bundle_hash") != bundle_hash:
        raise ProtocolError("Matched-reference content bundle hash mismatch.")
    _assert_columns(result_rows, MATCHED_REFERENCE_RESULT_COLUMNS, "classifier_tuned_source_results.csv")
    _assert_columns(prediction_rows, MATCHED_REFERENCE_PREDICTION_COLUMNS, "classifier_tuned_predictions.csv")
    heldouts = tuple(str(value) for value in protocol["heldout_centers"])
    eligible = tuple(str(value) for value in protocol["eligible_centers"])
    if len(result_rows) != len(heldouts):
        raise ProtocolError("Matched-reference result coverage does not match heldout_centers.")
    rows_by_center = _unique_by(result_rows, "heldout_center", "matched-reference result")
    if set(rows_by_center) != set(heldouts):
        raise ProtocolError("Matched-reference result centers differ from the protocol.")
    grid_specs = tuple(_spec(payload) for payload in protocol["classifier_grid"])
    grid_hashes = {spec.config_hash for spec in grid_specs}
    for heldout in heldouts:
        row = rows_by_center[heldout]
        expected_train = tuple(center for center in eligible if center != heldout)
        if tuple(json.loads(row["train_centers"])) != expected_train:
            raise ProtocolError(f"Matched-reference train-center mismatch for center {heldout}.")
        _assert_row_identity(row, protocol, result=True)
        if row["status"] != "ok" or row["converged"].lower() != "true":
            raise ProtocolError(f"Matched-reference final classifier did not converge for center {heldout}.")
        spec = _spec(json.loads(row["selected_classifier_spec"]))
        if row["selected_classifier_config_hash"] != spec.config_hash:
            raise ProtocolError("Matched-reference classifier-spec hash does not match its payload.")
        if spec.config_hash not in grid_hashes:
            raise ProtocolError("Matched-reference selected classifier is outside the frozen grid.")
        center_predictions = [item for item in prediction_rows if item["heldout_center"] == heldout]
        if len(center_predictions) != int(row["n_eval"]):
            raise ProtocolError(f"Prediction coverage mismatch for center {heldout}.")
        sample_ids = []
        y_true: list[int] = []
        y_pred: list[int] = []
        for prediction in center_predictions:
            _assert_row_identity(prediction, protocol, result=False)
            if prediction["center"] != heldout:
                raise ProtocolError("Prediction center differs from heldout_center.")
            if prediction["selected_classifier_config_hash"] != spec.config_hash:
                raise ProtocolError("Prediction classifier hash differs from its result row.")
            if prediction["eval_row_hash"] != row["eval_row_hash"]:
                raise ProtocolError("Prediction eval-row hash differs from its result row.")
            sample_ids.append(prediction["sample_id"])
            truth = int(prediction["y_true"])
            predicted = int(prediction["y_pred"])
            probability = float(prediction["prob_pos"])
            if truth not in {0, 1} or predicted not in {0, 1} or not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ProtocolError("Matched-reference prediction contains invalid values.")
            y_true.append(truth)
            y_pred.append(predicted)
        if len(sample_ids) != len(set(sample_ids)):
            raise ProtocolError(f"Duplicate prediction sample_id for center {heldout}.")
        if _row_hash(sample_ids) != row["eval_row_hash"]:
            raise ProtocolError(f"Prediction sample order/hash mismatch for center {heldout}.")
        if not math.isclose(float(row["heldout_bacc"]), balanced_accuracy(y_true, y_pred), abs_tol=1e-12):
            raise ProtocolError("Matched-reference heldout BACC does not recompute from predictions.")
        if not math.isclose(float(row["heldout_macro_f1"]), macro_f1(y_true, y_pred), abs_tol=1e-12):
            raise ProtocolError("Matched-reference heldout macro-F1 does not recompute from predictions.")
        selected_tuning = _validate_and_select_tuning_rows(
            tuning_rows,
            heldout=heldout,
            expected_validation_centers=expected_train,
            grid_hashes=grid_hashes,
            grid_hash=str(protocol["classifier_grid_hash"]),
        )
        if selected_tuning.get("classifier_config_hash") != spec.config_hash:
            raise ProtocolError("Selected tuning row does not match the final classifier spec.")
        selected_vector = json.loads(selected_tuning["center_bacc_vector"])
        if json.loads(row["source_inner_center_bacc_vector"]) != selected_vector:
            raise ProtocolError("Result source-inner score vector differs from the selected tuning row.")
        if not math.isclose(
            float(row["source_inner_mean_bacc"]),
            float(selected_tuning["aggregate_bacc"]),
            abs_tol=1e-12,
        ):
            raise ProtocolError("Result source-inner mean differs from the selected tuning row.")


def matched_reference_bundle_hash(
    tuning_rows: Sequence[Mapping[str, object]],
    result_rows: Sequence[Mapping[str, object]],
    prediction_rows: Sequence[Mapping[str, object]],
) -> str:
    return stable_hash(
        {
            "tuning": _canonical_table(tuning_rows),
            "results": _canonical_table(result_rows, ignored=("protocol_hash",)),
            "predictions": _canonical_table(prediction_rows, ignored=("protocol_hash",)),
        }
    )


def _validate_and_select_tuning_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    heldout: str,
    expected_validation_centers: Sequence[str],
    grid_hashes: set[str],
    grid_hash: str,
) -> Mapping[str, str]:
    candidates = [row for row in rows if row.get("outer_target_center") == heldout]
    if len(candidates) != len(grid_hashes):
        raise ProtocolError(f"Tuning-grid coverage mismatch for center {heldout}.")
    by_hash: dict[str, tuple[Mapping[str, str], ClassifierSpec, float]] = {}
    for row in candidates:
        if row.get("schema_version") != "midogpp_eligible_predict_spec_selection_v2":
            raise ProtocolError("Unexpected matched-reference tuning schema.")
        spec = _spec(json.loads(row["classifier_spec"]))
        config_hash = row["classifier_config_hash"]
        if config_hash != spec.config_hash or config_hash not in grid_hashes or config_hash in by_hash:
            raise ProtocolError("Matched-reference tuning classifier identity mismatch.")
        if row["classifier_grid_hash"] != grid_hash:
            raise ProtocolError("Matched-reference tuning grid hash mismatch.")
        if row.get("inner_pseudo_target_center", "") != "":
            raise ProtocolError("Stage-10 tuning row unexpectedly declares an inner pseudo-target.")
        if tuple(json.loads(row["deeper_validation_centers"])) != tuple(expected_validation_centers):
            raise ProtocolError("Matched-reference tuning validation-center coverage mismatch.")
        if tuple(json.loads(row["excluded_centers"])) != (heldout,):
            raise ProtocolError("Matched-reference tuning excluded-center contract mismatch.")
        vector = json.loads(row["center_bacc_vector"])
        convergence = json.loads(row["convergence_by_center"])
        if set(vector) != set(expected_validation_centers) or set(convergence) != set(expected_validation_centers):
            raise ProtocolError("Matched-reference tuning score-vector coverage mismatch.")
        if not all(bool(value) for value in convergence.values()):
            raise ProtocolError("Matched-reference tuning contains a nonconverged fold.")
        values = [float(vector[center]) for center in expected_validation_centers]
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
            raise ProtocolError("Matched-reference tuning contains an invalid BACC value.")
        aggregate = sum(values) / float(len(values))
        if not math.isclose(float(row["aggregate_bacc"]), aggregate, abs_tol=1e-12):
            raise ProtocolError("Matched-reference tuning aggregate does not match its center vector.")
        for field in (
            "selection_used_target_labels",
            "fit_used_outer_target_center",
            "fit_used_inner_pseudo_target_center",
        ):
            if row.get(field) != "false":
                raise ProtocolError(f"Matched-reference tuning field {field} must be false.")
        if row.get("selection_source") != "nested_source_inner_predict":
            raise ProtocolError("Matched-reference tuning selection source mismatch.")
        by_hash[config_hash] = (row, spec, aggregate)
    best_score = max(item[2] for item in by_hash.values())
    winners = [item for item in by_hash.values() if item[2] == best_score]
    selected_row, selected_spec, _ = min(winners, key=lambda item: item[1].tie_break_key())
    for row, spec, _ in by_hash.values():
        expected_selected = spec.config_hash == selected_spec.config_hash
        if (row.get("selected", "").lower() == "true") is not expected_selected:
            raise ProtocolError("Matched-reference persisted selection differs from deterministic selection.")
    return selected_row


def _canonical_table(
    rows: Sequence[Mapping[str, object]],
    *,
    ignored: Sequence[str] = (),
) -> list[dict[str, str]]:
    ignored_set = set(ignored)
    normalized = [
        {
            str(key): "" if value is None else str(value)
            for key, value in row.items()
            if key not in ignored_set
        }
        for row in rows
    ]
    return sorted(normalized, key=stable_hash)


def _row_hash(sample_ids: Sequence[str]) -> str:
    return hashlib.sha256(
        "\n".join(str(value) for value in sample_ids).encode("utf-8")
    ).hexdigest()


def _validate_protocol(protocol: Mapping[str, object], leakage: Mapping[str, object]) -> None:
    if protocol.get("schema_version") != MATCHED_REFERENCE_SCHEMA_VERSION:
        raise ProtocolError("Unexpected matched-reference protocol schema version.")
    unhashed = dict(protocol)
    recorded_hash = str(unhashed.pop("protocol_hash", ""))
    if not recorded_hash or stable_hash(unhashed) != recorded_hash:
        raise ProtocolError("Matched-reference protocol hash mismatch.")
    if leakage.get("status") != "PASS" or leakage.get("protocol_hash") != recorded_hash:
        raise ProtocolError("Matched-reference leakage report is not bound to a passing protocol.")
    eligible = tuple(str(value) for value in protocol.get("eligible_centers", ()))
    heldouts = tuple(str(value) for value in protocol.get("heldout_centers", ()))
    if set(eligible).intersection(MIDOGPP_EXCLUDED_CENTERS):
        raise ProtocolError("Quarantined center appears in matched-reference eligible centers.")
    if not set(eligible).issubset(MIDOGPP_ELIGIBLE_CENTERS) or not set(heldouts).issubset(eligible):
        raise ProtocolError("Matched-reference protocol contains unknown or unavailable centers.")
    if protocol.get("coverage_mode") == "complete":
        if eligible != MIDOGPP_ELIGIBLE_CENTERS or heldouts != MIDOGPP_ELIGIBLE_CENTERS:
            raise ProtocolError("Complete matched reference requires exact nine-center coverage.")
    elif protocol.get("coverage_mode") != "partial_test":
        raise ProtocolError("Unknown matched-reference coverage mode.")
    grid = tuple(_spec(payload) for payload in protocol.get("classifier_grid", ()))
    if len(grid) != 10 or classifier_grid_hash(grid) != str(protocol.get("classifier_grid_hash")):
        raise ProtocolError("Matched-reference classifier grid payload/hash mismatch.")
    if protocol.get("method") != MATCHED_REFERENCE_METHOD or protocol.get("threshold_policy") != "predict":
        raise ProtocolError("Matched-reference method/threshold policy drifted.")
    bundle_hash = protocol.get("reference_bundle_hash")
    if not isinstance(bundle_hash, str) or not bundle_hash:
        raise ProtocolError("Matched-reference protocol lacks its content bundle identity.")
    for field in (
        "selection_used_target_labels",
        "fit_used_target_center",
        "generated_embeddings_used",
        "cvae_checkpoint_used",
        "source_summary_manifest_used",
        "is_router",
        "probabilities_calibrated",
        "support_labels_used",
        "oracle_eligible",
    ):
        if str(protocol.get(field)).lower() != "false":
            raise ProtocolError(f"Matched-reference protocol field {field} must be false.")
    overlap = leakage.get("overlap_rows")
    if not isinstance(overlap, list) or len(overlap) != len(heldouts):
        raise ProtocolError("Matched-reference leakage overlap coverage is incomplete.")
    for row in overlap:
        if not isinstance(row, Mapping):
            raise ProtocolError("Malformed matched-reference overlap row.")
        if row.get("target_center_excluded_from_fit") is not True or row.get("quarantined_center_excluded") is not True:
            raise ProtocolError("Matched-reference overlap report does not exclude target/quarantine centers.")


def _validate_workspace_provenance(root: Path, protocol: Mapping[str, object]) -> None:
    if protocol.get("coverage_mode") != "complete":
        return
    config_path = root / "config.resolved.yaml"
    provenance_path = root / "provenance/input_artifacts.json"
    missing = [str(path.relative_to(root)) for path in (config_path, provenance_path) if not path.is_file()]
    if missing:
        raise ProtocolError(f"Complete matched reference lacks workspace provenance: {missing}")

    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("Matched-reference provenance validation requires PyYAML.") from exc
    resolved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(resolved, Mapping):
        raise ProtocolError("Resolved matched-reference config must be a mapping.")
    experiment = resolved.get("experiment")
    inputs = resolved.get("inputs")
    run = resolved.get("run")
    grid = resolved.get("classifier_grid")
    if not all(isinstance(value, Mapping) for value in (experiment, inputs, run, grid)):
        raise ProtocolError("Resolved matched-reference config lacks its required sections.")
    assert isinstance(experiment, Mapping)
    assert isinstance(inputs, Mapping)
    assert isinstance(run, Mapping)
    assert isinstance(grid, Mapping)
    if (
        experiment.get("name") != protocol.get("experiment_name")
        or int(run.get("experiment_seed", -1)) != int(protocol.get("experiment_seed", -2))
        or int(run.get("classifier_seed", -1)) != int(protocol.get("classifier_seed", -2))
        or str(run.get("heldout_centers", "")).lower() != "all"
        or int(grid.get("expected_candidate_count", -1)) != 10
        or grid.get("expected_grid_hash") != protocol.get("classifier_grid_hash")
        or grid.get("threshold_policy") != "predict"
    ):
        raise ProtocolError("Resolved matched-reference config differs from the frozen protocol.")
    manifest_path = Path(str(inputs.get("manifest_path", "")))
    cache_path = Path(str(inputs.get("feature_cache_path", "")))
    if (
        not manifest_path.is_file()
        or not cache_path.is_file()
        or _file_sha256(manifest_path) != protocol.get("manifest_hash")
        or _file_sha256(cache_path) != protocol.get("feature_cache_hash")
    ):
        raise ProtocolError("Resolved matched-reference input files differ from the runtime protocol.")

    is_uniform_b_reference = (
        protocol.get("experiment_name")
        == "uniform_b_canonical_real_feature_reference_v1"
    )
    expected_experiment_id = (
        "midogpp.real_feature.uniform_b_canonical_reference.v1"
        if is_uniform_b_reference
        else "midogpp.real_feature.eligible_tuned_predict_reference.v2"
    )
    expected_cache_id = (
        "midogpp_virchow2_uniform_b_canonical_train_cache_seed42"
        if is_uniform_b_reference
        else "midogpp_virchow2_xyxy_feature_cache_seed42"
    )
    expected_ids = {
        "midogpp_dataset_contract_annotation_patch_v1",
        expected_cache_id,
    }
    if is_uniform_b_reference:
        expected_ids.add("midogpp_output_uniform_b_v3_prospective_test_confirmation_v1")

    provenance = _read_json(provenance_path)
    if (
        provenance.get("schema_version") != "midogpp_input_artifacts_v2"
        or provenance.get("dataset_id") != "midogpp"
        or provenance.get("experiment_id") != expected_experiment_id
        or provenance.get("stage") != "10_real_feature_reference"
        or provenance.get("claim_scope") != "real_feature_transfer_only"
        or provenance.get("selection_used_target_eval_artifacts") is not False
    ):
        raise ProtocolError("Matched-reference workspace provenance identity mismatch.")
    rows = provenance.get("input_artifacts")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ProtocolError("Malformed matched-reference input-artifact records.")
    by_id = {str(row["artifact_id"]): row for row in rows}
    if set(by_id) != expected_ids or len(by_id) != len(rows):
        raise ProtocolError("Matched-reference input artifact IDs differ from the registry contract.")
    for artifact_id, row in by_id.items():
        if row.get("exists") is not True or row.get("semantic_identities_are_file_hashes") is not False:
            raise ProtocolError(f"Matched-reference input {artifact_id} is missing or mislabelled.")
        integrity = row.get("file_integrity")
        if not isinstance(integrity, Mapping) or str(integrity.get("status", "")).startswith("MISSING"):
            raise ProtocolError(f"Matched-reference input {artifact_id} lacks valid file integrity.")
        files = integrity.get("files")
        if not isinstance(files, list) or not files:
            raise ProtocolError(f"Matched-reference input {artifact_id} has no hashed provenance files.")
        for file_row in files:
            if not isinstance(file_row, Mapping) or file_row.get("exists") is not True:
                raise ProtocolError(f"Matched-reference input {artifact_id} has a missing provenance file.")
            computed = file_row.get("computed")
            if not isinstance(computed, Mapping) or not _is_sha256(computed.get("sha256")):
                raise ProtocolError(f"Matched-reference input {artifact_id} lacks a computed SHA-256.")
            expected = file_row.get("expected")
            if isinstance(expected, Mapping):
                algorithm = str(expected.get("algorithm", ""))
                if computed.get(algorithm) != expected.get("digest"):
                    raise ProtocolError(f"Matched-reference input {artifact_id} failed expected hash verification.")
    if (
        _recorded_file_hash(by_id["midogpp_dataset_contract_annotation_patch_v1"], "manifest.csv")
        != protocol.get("manifest_hash")
        or _recorded_file_hash(
            by_id[expected_cache_id],
            "embeddings/train.pt",
        )
        != protocol.get("feature_cache_hash")
    ):
        raise ProtocolError("Matched-reference workspace input hashes differ from the runtime protocol.")


def _recorded_file_hash(artifact: Mapping[str, object], relative_path: str) -> str:
    integrity = artifact.get("file_integrity")
    files = integrity.get("files") if isinstance(integrity, Mapping) else None
    if not isinstance(files, list):
        raise ProtocolError("Malformed matched-reference file-integrity record.")
    matches = [row for row in files if isinstance(row, Mapping) and row.get("path") == relative_path]
    if len(matches) != 1:
        raise ProtocolError(f"Matched-reference provenance lacks unique identity for {relative_path}.")
    computed = matches[0].get("computed")
    if not isinstance(computed, Mapping) or not _is_sha256(computed.get("sha256")):
        raise ProtocolError(f"Matched-reference provenance lacks SHA-256 for {relative_path}.")
    return str(computed["sha256"])


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _assert_row_identity(
    row: Mapping[str, str],
    protocol: Mapping[str, object],
    *,
    result: bool,
) -> None:
    expected = {
        "schema_version": (
            MATCHED_REFERENCE_RESULT_SCHEMA_VERSION if result else MATCHED_REFERENCE_PREDICTION_SCHEMA_VERSION
        ),
        "method": MATCHED_REFERENCE_METHOD,
        "protocol_hash": str(protocol["protocol_hash"]),
        "claim_role": "real_feature_reference",
        "row_role": "heldout_result" if result else "heldout_prediction",
        "leakage_status": "PASS",
        "support_labels_used": "false",
        "oracle_eligible": "false",
        "target_eval_labels_used_for_scoring_only": "true",
    }
    if result:
        expected.update(
            {
                "feature_cache_hash": str(protocol["feature_cache_hash"]),
                "manifest_hash": str(protocol["manifest_hash"]),
                "selection_used_target_labels": "false",
                "fit_used_target_center": "false",
                "generated_embeddings_used": "false",
                "cvae_checkpoint_used": "false",
                "source_summary_manifest_used": "false",
                "is_router": "false",
                "claim_scope": "real_feature_transfer_only",
                "probabilities_calibrated": "false",
            }
        )
    for field, value in expected.items():
        if str(row.get(field, "")) != value:
            raise ProtocolError(f"Matched-reference row field {field} mismatch.")
    if result and (not row.get("fit_row_hash") or not row.get("eval_row_hash")):
        raise ProtocolError("Matched-reference result row lacks fit/eval row hashes.")


def _spec(payload: object) -> ClassifierSpec:
    if not isinstance(payload, Mapping):
        raise ProtocolError("Classifier spec payload must be a mapping.")
    return ClassifierSpec(
        C=float(payload["C"]),
        penalty=str(payload.get("penalty", "l2")),
        solver=str(payload.get("solver", "lbfgs")),
        max_iter=int(payload.get("max_iter", 2000)),
        class_weight=None if payload.get("class_weight") in (None, "", "none") else str(payload["class_weight"]),
        random_state=int(payload.get("random_state", 23)),
        l1_ratio=None if payload.get("l1_ratio") in (None, "") else float(payload["l1_ratio"]),
        threshold_policy=str(payload.get("threshold_policy", "predict")),
        scaler_fit=str(payload.get("scaler_fit", "synthetic_train_only")),
        family=str(payload.get("family", "sklearn_logistic_regression")),
    )


def _unique_by(rows: Sequence[Mapping[str, str]], field: str, label: str) -> dict[str, Mapping[str, str]]:
    output: dict[str, Mapping[str, str]] = {}
    for row in rows:
        key = str(row.get(field, ""))
        if not key or key in output:
            raise ProtocolError(f"Duplicate or empty {field} in {label}.")
        output[key] = row
    return output


def _assert_columns(rows: Sequence[Mapping[str, object]], required: Sequence[str], label: str) -> None:
    if not rows:
        raise ProtocolError(f"{label} is empty.")
    missing = sorted(set(required).difference(rows[0]))
    if missing:
        raise ProtocolError(f"{label} missing columns: {missing}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ProtocolError(f"Empty CSV: {path}")
        return [dict(row) for row in reader]


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"Malformed JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Expected JSON object: {path}")
    return payload
