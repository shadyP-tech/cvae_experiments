"""Model-bank reconstruction and frozen test-decision replay."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...routing.disagreement_regret_core import (
    CandidateContrastRow,
    InferenceSelectionDiagnostic,
    LabelFreeInferenceContext,
    build_label_free_inference_selection_diagnostics,
)
from ...runtime.artifact_io import sha256_file
from .constants import CENTERS, GEOMETRY_IDS, candidate_sources, geometry_action_id
from .experiment_contracts import (
    EXPECTED_TEST_CACHE_CONTENT_HASH,
    EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
    MODEL_FAMILY_IDS,
)
from .hashing import canonical_hash
from .persistence import load_model_bank_records
from .validation_common import (
    CONTRAST_FIELDS,
    EXPECTED_MODEL_BANK_COUNT,
    EXPECTED_MODEL_COUNT,
    EXPECTED_TEST_CONTRAST_ROWS,
    EXPECTED_TEST_SELECTION_ROWS,
    MODEL_TABLE_FIELDS,
    SELECTION_FIELDS,
    SUMMARY_FIELDS,
    finite_float,
    is_sha256,
    read_csv,
    read_object,
)


def validate_model_bank(
    root: Path, *, source_prediction_seal_hash: str
) -> tuple[tuple[object, ...], str, Mapping[str, object]]:
    records = load_model_bank_records(root)
    expected_keys = {
        (target, geometry, family)
        for target in CENTERS
        for geometry in GEOMETRY_IDS
        for family in MODEL_FAMILY_IDS
    }
    if len(records) != EXPECTED_MODEL_BANK_COUNT or {
        record.key for record in records
    } != expected_keys:
        raise ProtocolError("Prediction-only reconstructed model-bank topology drifted.")
    model_count = 0
    for record in records:
        target, geometry, family = record.key
        models = tuple(record.bank.models)
        expected_sources = candidate_sources(target)
        expected_actions = tuple(
            sorted(geometry_action_id(geometry, source) for source in expected_sources)
        )
        if (
            record.bank.family != family
            or record.bank.outer_target_id != target
            or len(models) != len(expected_sources)
            or tuple(model.candidate_action_id for model in models) != expected_actions
        ):
            raise ProtocolError("Prediction-only reconstructed bank identity drifted.")
        for model in models:
            source = model.candidate_source_id
            if (
                source not in expected_sources
                or model.candidate_action_id != geometry_action_id(geometry, source)
                or model.heldout_query_id is not None
                or model.training_scope != "AUTHORIZED_POSTHOC_SOURCE_OOF"
                or model.training_surface_role != "source_oof_training_only"
                or model.prediction_seal_hash != source_prediction_seal_hash
                or model.training_query_ids
                != tuple(sorted(set(CENTERS).difference({target, source})))
                or model.excluded_query_ids != tuple(sorted((target, source)))
                or model.baseline_action_id != "B"
                or model.control_action_id != "U"
                or tuple(
                    source_id
                    for _action, source_id in model.candidate_source_by_action
                )
                != tuple(sorted(expected_sources))
            ):
                raise ProtocolError("Prediction-only H/e model exclusion drifted.")
        model_count += len(models)
    if model_count != EXPECTED_MODEL_COUNT:
        raise ProtocolError("Prediction-only reconstructed model count drifted.")
    index = read_object(root / "manifests/model_bank_index.json")
    seal = read_object(root / "manifests/model_bank_seal.json")
    seal_unhashed = {
        key: value
        for key, value in seal.items()
        if key != "regret_model_bank_seal_hash"
    }
    collection_hash = str(index.get("collection_hash", ""))
    if (
        not is_sha256(collection_hash)
        or seal.get("regret_model_bank_seal_hash") != canonical_hash(seal_unhashed)
        or seal.get("schema_version")
        != "midogpp_disagreement_regret_model_bank_seal_v1"
        or seal.get("status") != "SEALED_SOURCE_ONLY_BEFORE_TEST_ADMISSION"
        or seal.get("collection_hash") != collection_hash
        or seal.get("index_sha256")
        != sha256_file(root / "manifests/model_bank_index.json")
        or seal.get("array_sha256") != sha256_file(root / "arrays/model_bank.npz")
        or seal.get("bank_count") != EXPECTED_MODEL_BANK_COUNT
        or seal.get("model_count") != EXPECTED_MODEL_COUNT
        or seal.get("source_labels_only") is not True
        or seal.get("test_cache_admitted") is not False
        or seal.get("target_labels_used") is not False
    ):
        raise ProtocolError("Prediction-only source-only model-bank seal drifted.")
    return records, collection_hash, seal


def validate_model_training_lineage(
    records: Sequence[object],
    *,
    source_surfaces: Mapping[tuple[str, str, str], object],
    response_rows: Sequence[Mapping[str, str]],
    prelabel_feature_seal_hash: str,
    source_prediction_seal_hash: str,
    model_collection_hash: str,
) -> None:
    response_hashes: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in response_rows:
        response_hashes[(row["outer_target_id"], row["geometry_id"])].add(
            row["response_surface_hash"]
        )
    if any(len(values) != 1 for values in response_hashes.values()):
        raise ProtocolError("Prediction-only response surface topology drifted.")
    for record in records:
        feature_hash = source_surfaces[record.key].surface_hash
        response_hash = next(
            iter(response_hashes[(record.outer_target_id, record.geometry_id)])
        )
        if any(
            model.feature_surface_hash != feature_hash
            or model.response_surface_hash != response_hash
            for model in record.bank.models
        ):
            raise ProtocolError("Prediction-only model training surface lineage drifted.")
    expected_collection_hash = canonical_hash(
        {
            "schema_version": "midogpp_disagreement_regret_model_bank_collection_v1",
            "prelabel_feature_seal_hash": prelabel_feature_seal_hash,
            "source_prediction_seal_hash": source_prediction_seal_hash,
            "banks": [
                {
                    "outer_target_id": record.outer_target_id,
                    "geometry_id": record.geometry_id,
                    "family": record.family,
                    "model_bank_hash": record.bank.model_bank_hash,
                }
                for record in records
            ],
            "source_labels_used_for_training_only": True,
            "raw_source_labels_persisted": False,
            "test_labels_used": False,
        }
    )
    if model_collection_hash != expected_collection_hash:
        raise ProtocolError("Prediction-only model-bank collection hash drifted.")


def validate_model_table(path: Path, *, records: Sequence[object]) -> None:
    rows = read_csv(path, fields=MODEL_TABLE_FIELDS)
    expected: list[dict[str, str]] = []
    ordinal = 0
    for record in sorted(records, key=lambda value: value.key):
        for model in record.bank.models:
            expected.append(
                {
                    "ordinal": str(ordinal),
                    "outer_target_id": record.outer_target_id,
                    "geometry_id": record.geometry_id,
                    "family": record.family,
                    "candidate_action_id": model.candidate_action_id,
                    "candidate_source_id": model.candidate_source_id,
                    "observation_count": str(model.observation_count),
                    "iteration_count": str(model.iteration_count),
                    "model_hash": model.model_hash,
                    "model_bank_hash": record.bank.model_bank_hash,
                }
            )
            ordinal += 1
    if tuple(expected) != rows or ordinal != EXPECTED_MODEL_COUNT:
        raise ProtocolError("Prediction-only model index table differs from reconstruction.")


def validate_contrast_table(
    path: Path,
    *,
    model_records: Sequence[object],
    cases_by_query: Mapping[str, Sequence[str]],
    replayed_contrasts: Sequence[tuple[str, CandidateContrastRow]],
) -> tuple[tuple[str, CandidateContrastRow], ...]:
    rows = read_csv(path, fields=CONTRAST_FIELDS)
    model_by_key = {
        (
            record.outer_target_id,
            record.geometry_id,
            record.family,
            model.candidate_action_id,
        ): model
        for record in model_records
        for model in record.bank.models
    }
    expected = {
        (geometry, family, target, case, action)
        for (target, geometry, family, action), _model in model_by_key.items()
        for case in cases_by_query[target]
    }
    typed_rows: list[tuple[str, CandidateContrastRow]] = []
    keys: list[tuple[str, ...]] = []
    for raw in rows:
        typed = CandidateContrastRow(
            family=raw["family"],
            target_query_id=raw["target_query_id"],
            case_id=raw["case_id"],
            candidate_action_id=raw["candidate_action_id"],
            candidate_source_id=raw["candidate_source_id"],
            predicted_preference_margin_vs_control=finite_float(
                raw["predicted_preference_margin_vs_control"]
            ),
            standard_error_vs_control=finite_float(
                raw["standard_error_vs_control"]
            ),
            predicted_preference_margin_vs_baseline=finite_float(
                raw["predicted_preference_margin_vs_baseline"]
            ),
            standard_error_vs_baseline=finite_float(
                raw["standard_error_vs_baseline"]
            ),
            model_hash=raw["model_hash"],
            score_semantics=raw["score_semantics"],
        )
        key = (
            raw["geometry_id"],
            typed.family,
            typed.target_query_id,
            typed.case_id,
            typed.candidate_action_id,
        )
        model = model_by_key.get(
            (
                typed.target_query_id,
                raw["geometry_id"],
                typed.family,
                typed.candidate_action_id,
            )
        )
        if (
            model is None
            or typed.model_hash != model.model_hash
            or typed.candidate_source_id != model.candidate_source_id
        ):
            raise ProtocolError("Prediction-only candidate contrast lineage drifted.")
        typed_rows.append((raw["geometry_id"], typed))
        keys.append(key)
    if (
        len(rows) != EXPECTED_TEST_CONTRAST_ROWS
        or set(keys) != expected
        or keys != sorted(keys)
    ):
        raise ProtocolError("Prediction-only test contrast topology drifted.")
    if tuple(typed_rows) != tuple(replayed_contrasts):
        raise ProtocolError(
            "Prediction-only candidate contrasts differ from frozen-model replay."
        )
    return tuple(typed_rows)


def validate_selection_table(
    path: Path,
    *,
    contrasts: Sequence[tuple[str, CandidateContrastRow]],
    model_records: Sequence[object],
    test_prediction_seal_hash: str,
) -> tuple[tuple[str, InferenceSelectionDiagnostic], ...]:
    rows = read_csv(path, fields=SELECTION_FIELDS)
    bank_by_key = {record.key: record.bank for record in model_records}
    grouped: dict[tuple[str, str, str], list[CandidateContrastRow]] = defaultdict(list)
    for geometry, row in contrasts:
        grouped[(row.target_query_id, geometry, row.family)].append(row)
    replayed: list[tuple[str, InferenceSelectionDiagnostic]] = []
    for (target, geometry, family), block in sorted(
        grouped.items(), key=lambda item: (item[0][1], item[0][2], item[0][0])
    ):
        bank = bank_by_key[(target, geometry, family)]
        context = LabelFreeInferenceContext(
            dataset_family="MIDOGPP_CONSUMED_TEST_LABEL_FREE",
            outer_target_id=target,
            target_cache_content_hash=EXPECTED_TEST_CACHE_CONTENT_HASH,
            target_cache_order_hash=EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
            prediction_seal_hash=test_prediction_seal_hash,
            action_schema=bank.action_schema,
            model_bank_hash=bank.model_bank_hash,
        )
        replayed.extend(
            (geometry, value)
            for value in build_label_free_inference_selection_diagnostics(
                tuple(sorted(block, key=lambda value: value.row_key)), context=context
            )
        )
    replayed.sort(
        key=lambda value: (
            value[0],
            value[1].family,
            value[1].target_query_id,
            value[1].case_id,
        )
    )
    expected = tuple(selection_csv_row(geometry, row) for geometry, row in replayed)
    if len(rows) != EXPECTED_TEST_SELECTION_ROWS or rows != expected:
        raise ProtocolError("Prediction-only R_raw/R_safe diagnostics differ from replay.")
    return tuple(replayed)


def selection_csv_row(
    geometry: str, row: InferenceSelectionDiagnostic
) -> dict[str, str]:
    return {
        "geometry_id": geometry,
        "family": row.family,
        "target_query_id": row.target_query_id,
        "case_id": row.case_id,
        "raw_action_id": row.raw_action_id,
        "safe_action_id": row.safe_action_id,
        "baseline_action_id": row.baseline_action_id,
        "control_action_id": row.control_action_id,
        "simultaneous_z_value": str(row.simultaneous_z_value),
        "safe_margin": str(row.safe_margin),
        "fallback_reason": row.fallback_reason,
        "claim_role": row.claim_role,
        "may_authorize_routing": "False",
        "may_authorize_promotion": "False",
    }


def validate_summary_table(
    path: Path,
    *,
    selections: Sequence[tuple[str, InferenceSelectionDiagnostic]],
) -> None:
    observed = read_csv(path, fields=SUMMARY_FIELDS)
    counts = Counter(
        (
            geometry,
            row.target_query_id,
            row.family,
            row.raw_action_id,
            row.safe_action_id,
            row.fallback_reason,
        )
        for geometry, row in selections
    )
    expected = tuple(
        {
            "geometry_id": key[0],
            "target_query_id": key[1],
            "family": key[2],
            "raw_action_id": key[3],
            "safe_action_id": key[4],
            "fallback_reason": key[5],
            "case_count": str(value),
            "test_labels_used": "False",
            "test_metric_computed": "False",
        }
        for key, value in sorted(counts.items())
    )
    if observed != expected:
        raise ProtocolError("Prediction-only test prediction summary drifted.")


def validate_frozen_test_seal(
    root: Path,
    *,
    model_collection_hash: str,
    test_prediction_seal_hash: str,
    replayed_surfaces: Mapping[tuple[str, str, str], object],
    contrasts: Sequence[tuple[str, CandidateContrastRow]],
    selections: Sequence[tuple[str, InferenceSelectionDiagnostic]],
) -> Mapping[str, object]:
    payload = read_object(root / "manifests/frozen_test_prediction_seal.json")
    unhashed = {key: value for key, value in payload.items() if key != "seal_hash"}
    required = {
        "schema_version": "midogpp_disagreement_regret_frozen_test_prediction_seal_v1",
        "status": "SEALED_UNSCORED_PREDICTIONS_FOR_ALL_TEST_CASES",
        "model_bank_hash": model_collection_hash,
        "test_prediction_seal_hash": test_prediction_seal_hash,
        "feature_surface_count": EXPECTED_MODEL_BANK_COUNT,
        "contrast_row_count": EXPECTED_TEST_CONTRAST_ROWS,
        "selection_row_count": EXPECTED_TEST_SELECTION_ROWS,
        "test_feature_table_sha256": sha256_file(root / "tables/test_case_features.csv"),
        "contrast_table_sha256": sha256_file(
            root / "tables/test_candidate_contrasts.csv"
        ),
        "selection_table_sha256": sha256_file(
            root / "tables/test_selection_diagnostics.csv"
        ),
        "summary_table_sha256": sha256_file(
            root / "tables/test_prediction_summary.csv"
        ),
        "test_labels_opened": False,
        "test_metrics_computed": False,
        "routing_authorized": False,
        "may_feed_another_experiment": False,
    }
    expected_prediction_hash = canonical_hash(
        {
            "schema_version": "midogpp_disagreement_regret_frozen_test_predictions_v1",
            "model_bank_hash": model_collection_hash,
            "test_prediction_seal_hash": test_prediction_seal_hash,
            "feature_surfaces": [
                {
                    "outer_target_id": target,
                    "geometry_id": geometry,
                    "family": family,
                    "surface_hash": replayed_surfaces[
                        (target, geometry, family)
                    ].surface_hash,
                }
                for target, geometry, family in sorted(replayed_surfaces)
            ],
            "contrast_rows": len(contrasts),
            "contrast_row_hashes": [
                row.row_hash
                for geometry, row in sorted(
                    contrasts,
                    key=lambda value: (value[0], *value[1].row_key),
                )
            ],
            "selection_rows": len(selections),
            "selection_row_hashes": [
                row.row_hash
                for geometry, row in sorted(
                    selections,
                    key=lambda value: (
                        value[0],
                        value[1].family,
                        value[1].target_query_id,
                        value[1].case_id,
                    ),
                )
            ],
            "test_labels_used": False,
            "test_metrics_computed": False,
            "may_authorize_routing": False,
        }
    )
    if (
        payload.get("frozen_test_prediction_hash") != expected_prediction_hash
        or any(payload.get(key) != value for key, value in required.items())
        or payload.get("seal_hash") != canonical_hash(unhashed)
    ):
        raise ProtocolError("Prediction-only frozen test prediction seal drifted.")
    return payload


__all__ = tuple(name for name in globals() if name.startswith("validate_"))
