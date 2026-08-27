"""Single-consumer authorization and immutable source-inner artifact loader.

This module is intentionally diagnostic-owned.  The label-free SCEPTRE core
must never import it, and this loader never weakens or rewrites the immutable
Stage-60 policy-consumption lock.  It validates a separate, consumer-specific
Stage-90 amendment for adaptive, descriptive development only.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Mapping

from ...protocol import ProtocolError
from .development_surface import HistoricalUtilityCell, SourceInnerDevelopmentSurface
from .hashing import file_sha256, require_sha256
from .identity import (
    EXPERIMENT_ID,
    EXPECTED_SOURCE_CASE_CONFUSIONS_SHA256,
    EXPECTED_SOURCE_CASE_CONFUSION_ROWS,
    EXPECTED_SOURCE_CLASSIFIER_FITS_SHA256,
    EXPECTED_SOURCE_CLASSIFIER_FIT_ROWS,
    EXPECTED_SOURCE_EVALUATION_ROWS_SHA256,
    EXPECTED_SOURCE_EVALUATION_ROW_COUNT,
    EXPECTED_SOURCE_PREDICTION_ARRAY_FILE_SHA256,
    EXPECTED_SOURCE_PREDICTION_INDEX_SHA256,
    EXPECTED_SOURCE_POLICY_LOCK_HASH,
    EXPECTED_SOURCE_REUSE_AMENDMENT_SHA256,
    EXPECTED_SOURCE_UTILITY_LOCK_SHA256,
    EXPECTED_SOURCE_UTILITY_ROWS,
    EXPECTED_SOURCE_UTILITY_TABLE_SHA256,
    PUBLICATION_STATUS,
    SOURCE_INNER_ALIAS_ARTIFACT_ID,
    SOURCE_INNER_AMENDMENT_ARTIFACT_ID,
    SOURCE_INNER_ORIGINAL_ARTIFACT_ID,
    TERMINAL_DECISION,
)


UTILITY_LOCK_MEMBER = "manifests/utility_lock.json"
UTILITY_TABLE_MEMBER = "tables/candidate_utility.csv"
CASE_CONFUSIONS_MEMBER = "tables/case_confusions.csv"
PREDICTION_ARRAY_MEMBER = "arrays/candidate_predictions.npz"
PREDICTION_INDEX_MEMBER = "manifests/prediction_index.json"
CLASSIFIER_FITS_MEMBER = "tables/classifier_fits.csv"
EVALUATION_ROWS_MEMBER = "tables/evaluation_rows.csv"


@dataclass(frozen=True, slots=True)
class SourceInnerReuseReceipt:
    """Validated capability for one exact historical development surface."""

    amendment_id: str
    amendment_sha256: str
    consumer_experiment_id: str
    input_alias_id: str
    utility_lock_sha256: str
    utility_table_sha256: str
    case_confusions_sha256: str
    prediction_array_file_sha256: str
    prediction_index_sha256: str
    classifier_fits_sha256: str
    evaluation_rows_sha256: str
    publication_status: str
    terminal_decision: str


def load_reuse_amendment(
    path: str | Path,
    *,
    expected_sha256: str | None = EXPECTED_SOURCE_REUSE_AMENDMENT_SHA256,
) -> SourceInnerReuseReceipt:
    """Validate the separate SCEPTRE development amendment byte-for-byte."""

    source = _safe_file(Path(path), role="adaptive reuse amendment")
    observed_sha256 = file_sha256(source)
    if expected_sha256 is not None and observed_sha256 != require_sha256(
        expected_sha256, "adaptive reuse amendment"
    ):
        raise ProtocolError("SCEPTRE adaptive reuse amendment hash drifted.")
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("Cannot read SCEPTRE adaptive reuse amendment.") from exc
    if not isinstance(raw, Mapping):
        raise ProtocolError("SCEPTRE adaptive reuse amendment is not a mapping.")

    source_binding = _mapping(raw, "source_binding")
    claims = _mapping(raw, "claim_boundary")
    development = _mapping(raw, "development_protocol")
    execution = _mapping(raw, "execution_authority")
    original = _mapping(raw, "original_contract_treatment")
    evaluation = _mapping(raw, "test_evaluation_protocol")

    exact = {
        "schema_version": "midogpp_sceptre_source_inner_adaptive_reuse_amendment_v1",
        "amendment_id": SOURCE_INNER_AMENDMENT_ARTIFACT_ID,
        "authorized_consumer_experiment_id": EXPERIMENT_ID,
        "authorized_input_alias_id": SOURCE_INNER_ALIAS_ARTIFACT_ID,
        "authorized_use": (
            "adaptive_architecture_development_and_descriptive_nested_lodo_replay_only"
        ),
    }
    for key, expected in exact.items():
        if raw.get(key) != expected:
            raise ProtocolError(f"SCEPTRE amendment field {key!r} drifted.")
    expected_binding = {
        "original_artifact_id": SOURCE_INNER_ORIGINAL_ARTIFACT_ID,
        "utility_lock_sha256": EXPECTED_SOURCE_UTILITY_LOCK_SHA256,
        "candidate_utility_csv_sha256": EXPECTED_SOURCE_UTILITY_TABLE_SHA256,
        "candidate_utility_row_count": EXPECTED_SOURCE_UTILITY_ROWS,
        "case_confusions_csv_sha256": EXPECTED_SOURCE_CASE_CONFUSIONS_SHA256,
        "case_confusions_row_count": EXPECTED_SOURCE_CASE_CONFUSION_ROWS,
        "candidate_predictions_npz_sha256": (
            EXPECTED_SOURCE_PREDICTION_ARRAY_FILE_SHA256
        ),
        "prediction_index_json_sha256": EXPECTED_SOURCE_PREDICTION_INDEX_SHA256,
        "classifier_fits_csv_sha256": EXPECTED_SOURCE_CLASSIFIER_FITS_SHA256,
        "classifier_fit_row_count": EXPECTED_SOURCE_CLASSIFIER_FIT_ROWS,
        "evaluation_rows_csv_sha256": EXPECTED_SOURCE_EVALUATION_ROWS_SHA256,
        "evaluation_row_count": EXPECTED_SOURCE_EVALUATION_ROW_COUNT,
    }
    if source_binding != expected_binding:
        raise ProtocolError("SCEPTRE source-inner amendment binding drifted.")

    required_true = (
        (claims, "adaptive_comparisons_descriptive_only"),
        (development, "complete_nested_lodo_required"),
        (development, "feature_normalization_fit_after_exclusion_only"),
        (development, "hyperparameter_tuning_after_exclusion_only"),
        (development, "nested_validation_center_candidate_rows_excluded"),
        (development, "nested_validation_center_query_rows_excluded_from_fit"),
        (development, "outer_target_candidate_rows_excluded"),
        (development, "outer_target_query_rows_excluded"),
        (development, "source_inner_self_pairs_forbidden"),
        (evaluation, "complete_router_and_thresholds_frozen_before_test_label_access"),
        (evaluation, "disjoint_target_support_and_evaluation_cases_required"),
        (evaluation, "future_execution_result_remains_posthoc_consumed_test_sensitivity"),
        (evaluation, "whole_case_disjointness_required"),
        (execution, "separate_future_single_consumer_authorization_required"),
    )
    if any(section.get(key) is not True for section, key in required_true):
        raise ProtocolError("SCEPTRE amendment lost a required exclusion or freeze gate.")
    required_false = (
        (claims, "confidence_or_significance_claim_allowed"),
        (claims, "deployment_claim_allowed"),
        (claims, "downstream_utility_claim_allowed"),
        (claims, "fresh_evidence"),
        (claims, "generalization_to_new_center_claim_allowed"),
        (claims, "may_feed_another_experiment"),
        (claims, "promotion_allowed"),
        (claims, "routing_success_claim_allowed"),
        (claims, "thesis_confirmatory_claim_allowed"),
        (development, "seed_selection_allowed"),
        (execution, "consumed_test_reuse_authorized"),
        (execution, "execution_authorized"),
        (execution, "implementation_authorizes_execution"),
        (execution, "output_or_scratch_creation_allowed"),
        (execution, "target_support_or_evaluation_labels_may_open"),
        (execution, "test_cache_or_manifest_resolution_authorized"),
        (original, "original_policy_consumption_lock_mutated"),
        (original, "original_policy_consumption_lock_reinterpreted"),
        (original, "original_stage60_artifact_mutated"),
        (evaluation, "this_amendment_authorizes_test_evaluation"),
    )
    if any(section.get(key) is not False for section, key in required_false):
        raise ProtocolError("SCEPTRE amendment illegally broadens execution authority.")
    if (
        raw.get("authorization_basis")
        != "explicit_user_scope_limited_direction_2026_08_27"
        or raw.get("authorization_date") != "2026-08-27"
        or development.get("strict_order")
        != "delete_q_or_e_equal_H_before_any_transform_normalization_fit_or_tuning"
        or development.get("replication_interpretation")
        != (
            "training_generation_seed_cells_are_nuisance_replications_not_"
            "independent_observations"
        )
        or original.get("scope")
        != "separate_stage90_consumer_specific_governance_exception"
        or original.get("source_inner_policy_lock_hash")
        != EXPECTED_SOURCE_POLICY_LOCK_HASH
        or claims.get("publication_status") != PUBLICATION_STATUS
        or claims.get("terminal_decision") != TERMINAL_DECISION
        or claims.get("fresh_evidence") is not False
        or claims.get("may_feed_another_experiment") is not False
    ):
        raise ProtocolError("SCEPTRE amendment claim or lock firewall drifted.")

    return SourceInnerReuseReceipt(
        amendment_id=SOURCE_INNER_AMENDMENT_ARTIFACT_ID,
        amendment_sha256=observed_sha256,
        consumer_experiment_id=EXPERIMENT_ID,
        input_alias_id=SOURCE_INNER_ALIAS_ARTIFACT_ID,
        utility_lock_sha256=EXPECTED_SOURCE_UTILITY_LOCK_SHA256,
        utility_table_sha256=EXPECTED_SOURCE_UTILITY_TABLE_SHA256,
        case_confusions_sha256=EXPECTED_SOURCE_CASE_CONFUSIONS_SHA256,
        prediction_array_file_sha256=(
            EXPECTED_SOURCE_PREDICTION_ARRAY_FILE_SHA256
        ),
        prediction_index_sha256=EXPECTED_SOURCE_PREDICTION_INDEX_SHA256,
        classifier_fits_sha256=EXPECTED_SOURCE_CLASSIFIER_FITS_SHA256,
        evaluation_rows_sha256=EXPECTED_SOURCE_EVALUATION_ROWS_SHA256,
        publication_status=PUBLICATION_STATUS,
        terminal_decision=TERMINAL_DECISION,
    )


def load_authorized_development_surface(
    artifact_root: str | Path,
    *,
    receipt: SourceInnerReuseReceipt,
) -> SourceInnerDevelopmentSurface:
    """Load the exact immutable 648-cell surface after amendment validation."""

    receipt = validate_source_inner_reuse_receipt(receipt)

    root = Path(artifact_root)
    if root.is_symlink() or not root.is_dir():
        raise ProtocolError("SCEPTRE source-inner artifact root is absent or unsafe.")
    utility_lock = _member(root, UTILITY_LOCK_MEMBER)
    utility_table = _member(root, UTILITY_TABLE_MEMBER)
    case_confusions = _member(root, CASE_CONFUSIONS_MEMBER)
    observed = {
        UTILITY_LOCK_MEMBER: file_sha256(utility_lock),
        UTILITY_TABLE_MEMBER: file_sha256(utility_table),
        CASE_CONFUSIONS_MEMBER: file_sha256(case_confusions),
    }
    expected = {
        UTILITY_LOCK_MEMBER: receipt.utility_lock_sha256,
        UTILITY_TABLE_MEMBER: receipt.utility_table_sha256,
        CASE_CONFUSIONS_MEMBER: receipt.case_confusions_sha256,
    }
    if observed != expected:
        raise ProtocolError("SCEPTRE source-inner artifact bytes drifted.")

    try:
        lock_payload = json.loads(utility_lock.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("Cannot read SCEPTRE source-inner utility lock.") from exc
    if not isinstance(lock_payload, Mapping):
        raise ProtocolError("SCEPTRE source-inner utility lock is not a mapping.")
    if lock_payload.get("policy_consumption_lock_hash") != EXPECTED_SOURCE_POLICY_LOCK_HASH:
        raise ProtocolError("SCEPTRE source-inner policy-consumption lock drifted.")

    cells = _read_utility_cells(utility_table)
    _require_csv_row_count(case_confusions, EXPECTED_SOURCE_CASE_CONFUSION_ROWS)
    return SourceInnerDevelopmentSurface(
        cells=cells,
        utility_lock_sha256=observed[UTILITY_LOCK_MEMBER],
        utility_table_sha256=observed[UTILITY_TABLE_MEMBER],
        case_confusions_sha256=observed[CASE_CONFUSIONS_MEMBER],
        amendment_sha256=receipt.amendment_sha256,
    )


def validate_source_inner_reuse_receipt(
    receipt: object,
) -> SourceInnerReuseReceipt:
    """Replay the complete consumer fence before any source member may open."""

    if not isinstance(receipt, SourceInnerReuseReceipt):
        raise ProtocolError("SCEPTRE source-inner reuse receipt type drifted.")
    if (
        receipt.amendment_id != SOURCE_INNER_AMENDMENT_ARTIFACT_ID
        or receipt.consumer_experiment_id != EXPERIMENT_ID
        or receipt.input_alias_id != SOURCE_INNER_ALIAS_ARTIFACT_ID
        or receipt.amendment_sha256 != EXPECTED_SOURCE_REUSE_AMENDMENT_SHA256
        or receipt.utility_lock_sha256 != EXPECTED_SOURCE_UTILITY_LOCK_SHA256
        or receipt.utility_table_sha256 != EXPECTED_SOURCE_UTILITY_TABLE_SHA256
        or receipt.case_confusions_sha256
        != EXPECTED_SOURCE_CASE_CONFUSIONS_SHA256
        or receipt.prediction_array_file_sha256
        != EXPECTED_SOURCE_PREDICTION_ARRAY_FILE_SHA256
        or receipt.prediction_index_sha256
        != EXPECTED_SOURCE_PREDICTION_INDEX_SHA256
        or receipt.classifier_fits_sha256 != EXPECTED_SOURCE_CLASSIFIER_FITS_SHA256
        or receipt.evaluation_rows_sha256 != EXPECTED_SOURCE_EVALUATION_ROWS_SHA256
        or receipt.publication_status != PUBLICATION_STATUS
        or receipt.terminal_decision != TERMINAL_DECISION
    ):
        raise ProtocolError("SCEPTRE source-inner reuse receipt is not consumer-scoped.")
    return receipt


def _read_utility_cells(path: Path) -> tuple[HistoricalUtilityCell, ...]:
    required = {
        "pseudo_target_center",
        "candidate_source_center",
        "training_seed",
        "generation_seed",
        "bacc",
        "macro_f1",
    }
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ProtocolError("SCEPTRE candidate-utility columns drifted.")
            cells = tuple(
                HistoricalUtilityCell(
                    query_center=str(row["pseudo_target_center"]),
                    candidate_center=str(row["candidate_source_center"]),
                    training_seed=int(row["training_seed"]),
                    generation_seed=int(row["generation_seed"]),
                    bacc=float(row["bacc"]),
                    macro_f1=float(row["macro_f1"]),
                )
                for row in reader
            )
    except (OSError, UnicodeDecodeError, TypeError, ValueError) as exc:
        raise ProtocolError("Cannot parse SCEPTRE candidate-utility table.") from exc
    if len(cells) != EXPECTED_SOURCE_UTILITY_ROWS:
        raise ProtocolError("SCEPTRE candidate-utility row count drifted.")
    return cells


def _require_csv_row_count(path: Path, expected: int) -> None:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            next(reader)
            observed = sum(1 for _ in reader)
    except (OSError, UnicodeDecodeError, StopIteration) as exc:
        raise ProtocolError("Cannot count SCEPTRE source-inner CSV rows.") from exc
    if observed != expected:
        raise ProtocolError("SCEPTRE case-confusion row count drifted.")


def _mapping(raw: Mapping[str, object], key: str) -> dict[str, object]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"SCEPTRE amendment section {key!r} drifted.")
    return dict(value)


def _member(root: Path, relative: str) -> Path:
    candidate = root.joinpath(*relative.split("/"))
    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ProtocolError(f"SCEPTRE source-inner member is absent: {relative}.") from exc
    if candidate.is_symlink() or resolved_root not in resolved.parents or not resolved.is_file():
        raise ProtocolError(f"SCEPTRE source-inner member is unsafe: {relative}.")
    return resolved


def _safe_file(path: Path, *, role: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ProtocolError(f"SCEPTRE {role} is absent or unsafe.")
    return path


__all__ = (
    "CASE_CONFUSIONS_MEMBER",
    "CLASSIFIER_FITS_MEMBER",
    "EVALUATION_ROWS_MEMBER",
    "PREDICTION_ARRAY_MEMBER",
    "PREDICTION_INDEX_MEMBER",
    "SourceInnerReuseReceipt",
    "UTILITY_LOCK_MEMBER",
    "UTILITY_TABLE_MEMBER",
    "load_authorized_development_surface",
    "load_reuse_amendment",
    "validate_source_inner_reuse_receipt",
)
