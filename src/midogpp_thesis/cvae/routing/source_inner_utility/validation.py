"""Independent reconstruction and provenance validation for candidate utility."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ...generation import (
    load_generation_lock_config,
    read_generation_lock,
    source_generation_plan,
    validate_generation_bundle,
)
from ...protocol import ProtocolError
from .bundle import (
    CASE_CONFUSION_TABLE_MEMBER,
    CONTENT_INDEX_MEMBERS,
    EVALUATION_ROW_COLUMNS,
    EVALUATION_ROW_MEMBER,
    FIT_TABLE_MEMBER,
    PREDICTION_ARRAY_MEMBER,
    REQUIRED_FILES,
    UTILITY_TABLE_MEMBER,
    evaluation_row_table,
    label_consumption_report_payload,
    leakage_report_payload,
    policy_consumption_manifest_payload,
    prediction_index_payload,
    protocol_manifest_payload,
    read_prediction_arrays,
    run_state_payload,
    sha256_file,
    utility_decision_payload,
    utility_lock_payload,
)
from .cache_inputs import (
    load_unlabeled_validation_frame,
    open_scoring_labels,
    read_manifest_evaluation_index,
)
from .config import SourceInnerUtilityConfig, load_source_inner_utility_config
from .contracts import (
    CENTERS,
    CLAIM_SCOPE,
    EXPECTED_CASE_CONFUSION_ROW_COUNT,
    EXPECTED_EVAL_CASES,
    EXPECTED_EVAL_ROWS,
    EXPECTED_FIT_COUNT,
    EXPECTED_UTILITY_ROW_COUNT,
    EXPERIMENT_ID,
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    POLICY_CONSUMPTION_LOCK_HASH,
    TRAINING_SEEDS,
    VALIDATION_CACHE_ARTIFACT_ID,
    VALIDATION_MANIFEST_ARTIFACT_ID,
    expected_fit_keys,
    source_identities_from_generation_lock,
)
from .scoring import (
    CASE_CONFUSION_COLUMNS,
    FIT_COLUMNS,
    UTILITY_COLUMNS,
    PredictionPass,
    array_sha256,
    score_prediction_pass,
)


def validate_source_inner_utility_bundle(
    root: str | Path,
    *,
    config: SourceInnerUtilityConfig,
    allow_pending: bool = False,
) -> dict[str, object]:
    """Reconstruct metrics from compact predictions plus the frozen manifest."""

    path = Path(root)
    required = set(REQUIRED_FILES)
    if allow_pending:
        required.remove("reports/validation_report.json")
    missing = sorted(relative for relative in required if not (path / relative).is_file())
    if missing:
        raise ProtocolError(f"Source-inner utility artifact is incomplete: {missing}.")
    _validate_closed_world(path)
    if load_source_inner_utility_config(path / "config.resolved.yaml") != config:
        raise ProtocolError("Source-inner utility resolved config drifted.")
    validate_source_inner_utility_provenance(path, config=config)

    generation_config = load_generation_lock_config(
        config.generation_lock_root / "config.resolved.yaml"
    )
    if generation_config.bank_root.resolve() != config.bank_root.resolve():
        raise ProtocolError("Source-inner utility upstream bank path drifted.")
    validate_generation_bundle(config.generation_lock_root, config=generation_config)
    generation_lock = read_generation_lock(
        config.generation_lock_root / "manifests/generation_lock.json"
    )
    if (
        generation_lock.generation_lock_hash != config.expected_generation_lock_hash
        or generation_lock.bank_lock_hash != config.expected_bank_lock_hash
    ):
        raise ProtocolError("Source-inner utility upstream GenerationLock drifted.")

    manifest_rows = read_manifest_evaluation_index(
        config.manifest_path,
        expected_sha256=config.expected_manifest_sha256,
        expected_rows=EXPECTED_EVAL_ROWS,
    )
    frame = load_unlabeled_validation_frame(
        config.validation_cache_root,
        manifest_rows,
        expected_rows=EXPECTED_EVAL_ROWS,
    )
    if (
        frame.cache_binding.get("cache_name") != config.expected_cache_semantic_id
        or frame.cache_binding.get("representation_id")
        != config.expected_cache_representation_id
        or frame.cache_binding.get("validation_split") != "val"
    ):
        raise ProtocolError("Source-inner utility validation-cache identity drifted.")
    observed_eval = _read_csv_exact(path / EVALUATION_ROW_MEMBER, EVALUATION_ROW_COLUMNS)
    expected_eval = _rows_as_strings(evaluation_row_table(_frame_prediction_stub(frame)))
    if observed_eval != expected_eval:
        raise ProtocolError("Source-inner utility evaluation-row index drifted.")

    y_pred, prob_pos = read_prediction_arrays(path / PREDICTION_ARRAY_MEMBER)
    if y_pred.shape != (EXPECTED_FIT_COUNT, EXPECTED_EVAL_ROWS) or prob_pos.shape != (
        EXPECTED_FIT_COUNT,
        EXPECTED_EVAL_ROWS,
    ):
        raise ProtocolError("Source-inner utility prediction geometry drifted.")
    observed_fits = _read_csv_exact(path / FIT_TABLE_MEMBER, FIT_COLUMNS)
    normalized_fits = _validate_and_normalize_fits(
        observed_fits,
        y_pred=y_pred,
        prob_pos=prob_pos,
        generation_lock_payload=generation_lock.to_payload(),
        evaluation_order_hash=frame.row_order_hash,
        classifier_config_hash=config.classifier.config_hash,
    )
    predictions = PredictionPass(
        evaluation_rows=tuple(frame.rows),
        fit_rows=tuple(normalized_fits),
        y_pred=y_pred,
        prob_pos=prob_pos,
    )
    expected_prediction_index = prediction_index_payload(
        predictions,
        prediction_file_sha256=sha256_file(path / PREDICTION_ARRAY_MEMBER),
    )
    _require_exact_payload(
        path / "manifests/prediction_index.json",
        expected_prediction_index,
        "prediction index",
    )

    labels = open_scoring_labels(
        config.manifest_path,
        frame.rows,
        expected_sha256=config.expected_manifest_sha256,
    )
    expected_utilities, expected_cases = score_prediction_pass(predictions, labels)
    if len(expected_cases) != EXPECTED_CASE_CONFUSION_ROW_COUNT:
        raise ProtocolError("Source-inner case-confusion production geometry drifted.")
    observed_utilities = _read_csv_exact(path / UTILITY_TABLE_MEMBER, UTILITY_COLUMNS)
    observed_cases = _read_csv_exact(
        path / CASE_CONFUSION_TABLE_MEMBER,
        CASE_CONFUSION_COLUMNS,
    )
    if observed_utilities != _rows_as_strings(expected_utilities):
        raise ProtocolError(
            "Source-inner utility rows or reconstructed BACC/macro-F1 drifted."
        )
    if observed_cases != _rows_as_strings(expected_cases):
        raise ProtocolError("Source-inner case-confusion rows drifted from predictions.")

    expected_protocol = protocol_manifest_payload(
        config,
        generation_lock_hash=generation_lock.generation_lock_hash,
        bank_lock_hash=generation_lock.bank_lock_hash,
        cache_binding=frame.cache_binding,
    )
    _require_exact_payload(
        path / "manifests/protocol_manifest.json",
        expected_protocol,
        "protocol manifest",
    )
    _require_exact_payload(
        path / "manifests/policy_consumption_lock.json",
        policy_consumption_manifest_payload(config),
        "policy consumption lock",
    )
    member_hashes = {
        relative: sha256_file(path / relative)
        for relative in (
            EVALUATION_ROW_MEMBER,
            FIT_TABLE_MEMBER,
            UTILITY_TABLE_MEMBER,
            CASE_CONFUSION_TABLE_MEMBER,
            PREDICTION_ARRAY_MEMBER,
        )
    }
    expected_utility_lock = utility_lock_payload(
        config,
        protocol=expected_protocol,
        prediction_index=expected_prediction_index,
        member_sha256=member_hashes,
        case_confusion_row_count=len(expected_cases),
    )
    _require_exact_payload(
        path / "manifests/utility_lock.json",
        expected_utility_lock,
        "utility lock",
    )
    _require_exact_payload(
        path / "reports/utility_decision.json",
        utility_decision_payload(expected_utility_lock),
        "utility decision",
    )
    _require_exact_payload(
        path / "reports/label_consumption_report.json",
        label_consumption_report_payload(),
        "label consumption report",
    )
    _require_exact_payload(
        path / "reports/leakage_report.json",
        leakage_report_payload(),
        "leakage report",
    )
    _require_exact_payload(
        path / "reports/run_state.json",
        run_state_payload("COMPLETE"),
        "run state",
    )
    _validate_content_index(path)

    checks: dict[str, object] = {
        "status": "PASS",
        "generation_lock_hash": generation_lock.generation_lock_hash,
        "bank_lock_hash": generation_lock.bank_lock_hash,
        "cache_name": config.expected_cache_semantic_id,
        "cache_content_hash": frame.cache_binding.get("cache_content_hash"),
        "manifest_sha256": config.expected_manifest_sha256,
        "policy_consumption_lock_hash": POLICY_CONSUMPTION_LOCK_HASH,
        "evaluation_row_count": EXPECTED_EVAL_ROWS,
        "evaluation_case_count": EXPECTED_EVAL_CASES,
        "classifier_fit_count": EXPECTED_FIT_COUNT,
        "candidate_utility_row_count": EXPECTED_UTILITY_ROW_COUNT,
        "case_confusion_row_count": EXPECTED_CASE_CONFUSION_ROW_COUNT,
        "all_nine_seed_pairs_present": True,
        "q_equals_e_rows_present": False,
        "prediction_arrays_contain_labels": False,
        "metrics_reconstructed_from_predictions_and_manifest": True,
        "case_confusions_reconstructed": True,
        "outer_target_instantiated": False,
        "selection_performed": False,
        "seed_selection_performed": False,
        "alternative_router_tuning_authorized": False,
    }
    if not allow_pending:
        _require_exact_payload(
            path / "reports/validation_report.json",
            {
                "schema_version": (
                    "midogpp_uniform_b_v2_source_inner_utility_validation_v1"
                ),
                "status": "PASS",
                "validator": "validate_source_inner_utility_bundle",
                "checks": checks,
            },
            "validation report",
        )
    return checks


def validate_source_inner_utility_provenance(
    root: str | Path,
    *,
    config: SourceInnerUtilityConfig,
) -> None:
    """Require exactly bank, GenerationLock, unlabeled cache, and dataset contract."""

    output_root = Path(root)
    manifest = _json(output_root / "provenance/input_artifacts.json")
    allowed_header = {
        "schema_version",
        "dataset_id",
        "experiment_id",
        "stage",
        "claim_scope",
        "selection_used_target_eval_artifacts",
        "input_artifacts",
        "repository_revision",
        "repository_dirty",
        "repository_status_hash",
    }
    if set(manifest) != allowed_header:
        raise ProtocolError("Source-inner utility provenance schema drifted.")
    if (
        manifest.get("schema_version") != "midogpp_input_artifacts_v2"
        or manifest.get("dataset_id") != "midogpp"
        or manifest.get("experiment_id") != EXPERIMENT_ID
        or manifest.get("stage") != "60_routing_and_composition"
        or manifest.get("claim_scope") != CLAIM_SCOPE
        or manifest.get("selection_used_target_eval_artifacts") is not False
        or not _is_hex(manifest.get("repository_revision"), 40)
        or not isinstance(manifest.get("repository_dirty"), bool)
        or not _is_hex(manifest.get("repository_status_hash"), 64)
    ):
        raise ProtocolError("Source-inner utility provenance header drifted.")
    rows = manifest.get("input_artifacts")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ProtocolError("Source-inner utility provenance inputs are malformed.")
    expected_roots = {
        EXPERT_BANK_ARTIFACT_ID: (
            config.bank_root.resolve(),
            "30_expert_bank",
            "expert_bank_construction_only",
        ),
        GENERATION_LOCK_ARTIFACT_ID: (
            config.generation_lock_root.resolve(),
            "40_prior_and_generation",
            "generation_settings_and_frame_lock",
        ),
        VALIDATION_CACHE_ARTIFACT_ID: (
            config.validation_cache_root.resolve(),
            "derived_features",
            "feature_cache_provenance",
        ),
        VALIDATION_MANIFEST_ARTIFACT_ID: (
            config.manifest_path.parent.resolve(),
            "dataset_contract",
            "dataset_contract_and_split_provenance",
        ),
    }
    by_id = {str(row.get("artifact_id", "")): row for row in rows}
    if len(rows) != len(by_id) or set(by_id) != set(expected_roots):
        raise ProtocolError(
            "Source-inner utility provenance includes missing, duplicate, or forbidden inputs."
        )
    allowed_row_fields = {
        "artifact_id",
        "resolved_path",
        "stage",
        "evidence_label",
        "claim_scope",
        "semantic_identities",
        "semantic_identities_are_file_hashes",
        "file_integrity",
        "exists",
    }
    for artifact_id, (expected_root, expected_stage, expected_scope) in expected_roots.items():
        row = by_id[artifact_id]
        if (
            set(row) != allowed_row_fields
            or Path(str(row.get("resolved_path", ""))).resolve() != expected_root
            or row.get("stage") != expected_stage
            or row.get("claim_scope") != expected_scope
            or row.get("exists") is not True
            or row.get("semantic_identities_are_file_hashes") is not False
            or not isinstance(row.get("semantic_identities"), Mapping)
        ):
            raise ProtocolError(f"Source-inner utility provenance identity drifted: {artifact_id}.")
        semantic = row["semantic_identities"]
        if artifact_id == GENERATION_LOCK_ARTIFACT_ID and (
            semantic.get("generation_lock_hash")
            != config.expected_generation_lock_hash
            or semantic.get("expert_bank_lock_hash") != config.expected_bank_lock_hash
        ):
            raise ProtocolError("Source-inner utility GenerationLock catalog identity drifted.")
        if artifact_id == VALIDATION_CACHE_ARTIFACT_ID and (
            semantic.get("cache_name") != config.expected_cache_semantic_id
            or semantic.get("representation_id")
            != config.expected_cache_representation_id
            or semantic.get("split") != "val"
            or semantic.get("feature_dim") != "3840"
            or semantic.get("manifest_sha256") != config.expected_manifest_sha256
            or semantic.get("labels_persisted") != "false"
            or semantic.get("policy_consumption_lock_hash")
            != POLICY_CONSUMPTION_LOCK_HASH
        ):
            raise ProtocolError("Source-inner utility cache semantic identity drifted.")
        if artifact_id == VALIDATION_MANIFEST_ARTIFACT_ID and (
            semantic.get("manifest_sha256") != config.expected_manifest_sha256
            or semantic.get("split_role")
            != "val_source_inner_pseudo_target_scoring_only"
            or semantic.get("policy_consumption_lock_hash")
            != POLICY_CONSUMPTION_LOCK_HASH
        ):
            raise ProtocolError("Source-inner validation-manifest authorization drifted.")
        _validate_integrity_inventory(expected_root, row.get("file_integrity"))
    dataset_files = _integrity_files(by_id[VALIDATION_MANIFEST_ARTIFACT_ID])
    manifest_record = dataset_files.get("manifest.csv")
    if (
        manifest_record is None
        or not isinstance(manifest_record.get("computed"), Mapping)
        or manifest_record["computed"].get("sha256") != config.expected_manifest_sha256
    ):
        raise ProtocolError("Source-inner utility manifest provenance hash drifted.")


def _validate_and_normalize_fits(
    rows: Sequence[Mapping[str, str]],
    *,
    y_pred: np.ndarray,
    prob_pos: np.ndarray,
    generation_lock_payload: Mapping[str, object],
    evaluation_order_hash: str,
    classifier_config_hash: str,
) -> list[dict[str, object]]:
    identities = source_identities_from_generation_lock(generation_lock_payload)
    plan = {
        (key.source_center, key.training_seed, key.generation_seed): key
        for key in source_generation_plan(
            # GenerationLock construction is already independently validated;
            # rebuild only its immutable wrapper for plan derivation.
            _generation_lock_from_payload(generation_lock_payload)
        )
    }
    normalized: list[dict[str, object]] = []
    observed_keys: set[tuple[str, int, int]] = set()
    canonical_keys = expected_fit_keys()
    for ordinal, raw in enumerate(rows):
        row: dict[str, object] = dict(raw)
        for field in (
            "fit_ordinal",
            "training_seed",
            "generation_seed",
            "generated_row_count",
            "generated_rows_per_class",
            "generated_class_0_count",
            "generated_class_1_count",
            "prediction_array_row",
            "all_eval_row_count",
        ):
            row[field] = _int(raw.get(field), field)
        for field in (
            "classifier_converged",
            "eval_labels_available_to_fit_or_predict",
            "seed_selection_performed",
        ):
            row[field] = _bool(raw.get(field), field)
        key = (
            str(row["source_center"]),
            int(row["training_seed"]),
            int(row["generation_seed"]),
        )
        if (
            ordinal >= len(canonical_keys)
            or key in observed_keys
            or key not in plan
            or key != canonical_keys[ordinal]
        ):
            raise ProtocolError("Source-inner classifier-fit key coverage drifted.")
        observed_keys.add(key)
        source = identities[(key[0], key[1])]
        generation_key = plan[key]
        prediction_row = y_pred[ordinal]
        probability_row = prob_pos[ordinal]
        expected_identity = source.to_payload()
        try:
            classes = json.loads(str(row.get("classifier_classes", "null")))
            n_iter = json.loads(str(row.get("classifier_n_iter", "null")))
        except json.JSONDecodeError as exc:
            raise ProtocolError("Classifier-fit JSON state is malformed.") from exc
        expected_fit_id = stable_hash(
            {
                "source_center": key[0],
                "training_seed": key[1],
                "generation_seed": key[2],
                "source_stream_id": generation_key.stream_id,
                "generated_block_sha256": row.get("generated_block_sha256"),
                "classifier_config_hash": classifier_config_hash,
                "evaluation_order_hash": evaluation_order_hash,
            }
        )
        if any(str(row.get(field, "")) != str(value) for field, value in expected_identity.items()):
            raise ProtocolError("Classifier-fit source provenance drifted.")
        if (
            row.get("schema_version")
            != "midogpp_uniform_b_v2_candidate_classifier_fit_v1"
            or int(row["fit_ordinal"]) != ordinal
            or int(row["prediction_array_row"]) != ordinal
            or row.get("fit_id") != expected_fit_id
            or row.get("source_stream_id") != generation_key.stream_id
            or int(row["generated_row_count"]) != 2048
            or int(row["generated_rows_per_class"]) != 1024
            or int(row["generated_class_0_count"]) != 1024
            or int(row["generated_class_1_count"]) != 1024
            or row.get("classifier_family") != "sklearn_logistic_regression"
            or row.get("classifier_config_hash") != classifier_config_hash
            or not _is_hex(row.get("scaler_state_hash"), 16)
            or classes != [0, 1]
            or raw.get("classifier_classes") != "[0,1]"
            or not isinstance(n_iter, list)
            or not n_iter
            or any(
                type(value) is not int or value <= 0 or value > 3000
                for value in n_iter
            )
            or raw.get("classifier_n_iter")
            != json.dumps(n_iter, sort_keys=True, separators=(",", ":"))
            or row["classifier_converged"] is not True
            or int(row["all_eval_row_count"]) != EXPECTED_EVAL_ROWS
            or row.get("all_eval_row_hash") != evaluation_order_hash
            or row.get("all_eval_prediction_sha256") != array_sha256(prediction_row)
            or row.get("all_eval_probability_sha256") != array_sha256(probability_row)
            or row["eval_labels_available_to_fit_or_predict"] is not False
            or row["seed_selection_performed"] is not False
            or not _is_hex(row.get("generated_block_sha256"), 64)
        ):
            raise ProtocolError("Source-inner classifier-fit row drifted.")
        normalized.append(row)
    if len(rows) != EXPECTED_FIT_COUNT or observed_keys != set(expected_fit_keys()):
        raise ProtocolError("Source-inner classifier-fit coverage is incomplete.")
    return normalized


def _generation_lock_from_payload(payload: Mapping[str, object]) -> object:
    from ...generation.contracts import GenerationLock

    return GenerationLock(dict(payload))


def _frame_prediction_stub(frame: object) -> object:
    """Minimal object accepted by ``evaluation_row_table``."""

    return SimpleNamespace(evaluation_rows=tuple(getattr(frame, "rows")))


def _validate_integrity_inventory(root: Path, value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "status",
        "default_recording_algorithm",
        "files",
    }:
        raise ProtocolError("Source-inner utility input integrity schema drifted.")
    if str(value.get("status", "")).startswith("MISSING"):
        raise ProtocolError("Source-inner utility input integrity is missing files.")
    files = value.get("files")
    if not isinstance(files, list) or not files or not all(isinstance(item, Mapping) for item in files):
        raise ProtocolError("Source-inner utility input file inventory is empty or malformed.")
    seen: set[str] = set()
    for item in files:
        relative = str(item.get("path", ""))
        if not relative or relative in seen:
            raise ProtocolError("Source-inner utility input file inventory duplicates a path.")
        seen.add(relative)
        member = _safe_member(root, relative)
        computed = item.get("computed")
        if (
            Path(str(item.get("resolved_path", ""))).resolve() != member
            or item.get("exists") is not True
            or not member.is_file()
            or not isinstance(computed, Mapping)
            or computed.get("sha256") != sha256_file(member)
            or int(item.get("size_bytes", -1)) != member.stat().st_size
        ):
            raise ProtocolError(f"Source-inner utility input member drifted: {relative}.")
        expected = item.get("expected")
        if expected is None:
            if item.get("verification") != "RECORDED_NO_EXPECTATION":
                raise ProtocolError("Source-inner input verification state drifted.")
        elif (
            not isinstance(expected, Mapping)
            or computed.get(str(expected.get("algorithm", ""))) != expected.get("digest")
            or item.get("verification") != "MATCH"
        ):
            raise ProtocolError("Source-inner input expected hash failed.")


def _integrity_files(row: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    integrity = row.get("file_integrity")
    if not isinstance(integrity, Mapping) or not isinstance(integrity.get("files"), list):
        return {}
    return {
        str(item.get("path", "")): item
        for item in integrity["files"]
        if isinstance(item, Mapping)
    }


def _validate_content_index(root: Path) -> None:
    payload = _json(root / "manifests/content_index.json")
    if set(payload) != {"schema_version", "records", "content_hash"}:
        raise ProtocolError("Source-inner utility content-index schema drifted.")
    unhashed = {key: value for key, value in payload.items() if key != "content_hash"}
    if (
        payload.get("schema_version")
        != "midogpp_uniform_b_v2_source_inner_utility_content_v1"
        or payload.get("content_hash") != stable_hash(unhashed)
    ):
        raise ProtocolError("Source-inner utility content-index identity drifted.")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != len(CONTENT_INDEX_MEMBERS):
        raise ProtocolError("Source-inner utility content-index coverage drifted.")
    observed: list[str] = []
    for raw in records:
        if not isinstance(raw, Mapping) or set(raw) != {
            "relative_path",
            "sha256",
            "size_bytes",
        }:
            raise ProtocolError("Source-inner utility content-index row drifted.")
        relative = str(raw.get("relative_path", ""))
        member = _safe_member(root, relative)
        if (
            not member.is_file()
            or raw.get("sha256") != sha256_file(member)
            or int(raw.get("size_bytes", -1)) != member.stat().st_size
        ):
            raise ProtocolError(f"Source-inner utility content member drifted: {relative}.")
        observed.append(relative)
    if tuple(observed) != CONTENT_INDEX_MEMBERS:
        raise ProtocolError("Source-inner utility content-index order drifted.")


def _validate_closed_world(root: Path) -> None:
    actual = {
        member.relative_to(root).as_posix()
        for member in root.rglob("*")
        if member.is_file()
    }
    unexpected = sorted(actual.difference(REQUIRED_FILES))
    if unexpected:
        raise ProtocolError(
            f"Source-inner utility artifact contains unexpected files: {unexpected}."
        )


def _read_csv_exact(path: Path, columns: Sequence[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(columns):
            raise ProtocolError(f"Source-inner utility table schema drifted: {path.name}.")
        return [dict(row) for row in reader]


def _rows_as_strings(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, str]]:
    return [
        {str(key): "" if value is None else str(value) for key, value in row.items()}
        for row in rows
    ]


def _require_exact_payload(path: Path, expected: Mapping[str, object], label: str) -> None:
    if _json(path) != dict(expected):
        raise ProtocolError(f"Source-inner utility {label} drifted.")


def _json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read source-inner utility JSON: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Source-inner utility JSON must be an object: {path}.")
    return payload


def _safe_member(root: Path, relative: str) -> Path:
    resolved_root = root.resolve()
    member = (resolved_root / relative).resolve()
    if member == resolved_root or not member.is_relative_to(resolved_root):
        raise ProtocolError("Source-inner utility path escapes its artifact root.")
    return member


def _int(value: object, label: str) -> int:
    rendered = str(value)
    try:
        parsed = int(rendered)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"Source-inner utility integer field drifted: {label}.") from exc
    if rendered != str(parsed):
        raise ProtocolError(f"Source-inner utility integer field drifted: {label}.")
    return parsed


def _bool(value: object, label: str) -> bool:
    rendered = str(value)
    if rendered not in {"True", "False"}:
        raise ProtocolError(f"Source-inner utility boolean field drifted: {label}.")
    return rendered == "True"


def _is_hex(value: object, length: int) -> bool:
    rendered = str(value or "")
    return len(rendered) == length and all(char in "0123456789abcdef" for char in rendered)


__all__ = (
    "validate_source_inner_utility_bundle",
    "validate_source_inner_utility_provenance",
)
