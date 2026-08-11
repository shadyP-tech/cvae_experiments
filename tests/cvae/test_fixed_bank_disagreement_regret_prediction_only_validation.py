from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.constants import (  # noqa: E501
    CENTERS,
    EXPECTED_CLASSIFIER_FIT_COUNT,
    EXPECTED_SOURCE_ROWS_BY_CENTER,
    EXPECTED_TEST_ROWS_BY_CENTER,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.hashing import (
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.input_contracts import (
    TestInferenceAdmission as _TestInferenceAdmission,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.inputs import (
    expected_test_cache_binding_hash_from_provenance,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.experiment_contracts import (
    EXPERIMENT_ID,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_TEST_CACHE_CONTENT_HASH,
    EXPECTED_TEST_CACHE_REPRESENTATION_ID,
    EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
    EXPECTED_TEST_CACHE_SEMANTIC_ID,
    EXPECTED_TRAIN_CACHE_SHA256,
    INPUT_ARTIFACT_IDS,
    TEST_CACHE_ARTIFACT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.validation_contracts import (
    validate_provenance,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only import (
    validation,
    validation_surfaces,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.disagreement_regret_core import (
    CaseActionResponseRow,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_forbidden_scan_allows_only_fail_closed_test_attestations_and_source_aggregates(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "manifests/safe.json",
        {
            "test_labels_opened": False,
            "test_bacc_or_regret_computed": False,
            "target_oracle_computation_permitted": False,
        },
    )
    _write_csv(
        tmp_path / "tables/source_regret_responses.csv",
        (
            "source_exact_bacc_gain_vs_control",
            "source_exact_regret_from_case_best",
            "response_hash",
        ),
        [
            {
                "source_exact_bacc_gain_vs_control": 0.1,
                "source_exact_regret_from_case_best": 0.0,
                "response_hash": "a" * 64,
            }
        ],
    )
    _write_csv(
        tmp_path / "tables/test_prediction_summary.csv",
        ("test_labels_used", "test_metric_computed"),
        [{"test_labels_used": False, "test_metric_computed": False}],
    )

    validation._reject_forbidden_persisted_fields(tmp_path)

    _write_json(
        tmp_path / "reports/tampered.json",
        {"nested": {"y_true": [0, 1]}},
    )
    with pytest.raises(ProtocolError, match="forbidden outcome key"):
        validation._reject_forbidden_persisted_fields(tmp_path)


def test_forbidden_scan_rejects_target_metric_columns_and_unapproved_source_metrics(
    tmp_path: Path,
) -> None:
    _write_csv(
        tmp_path / "tables/test_candidate_contrasts.csv",
        ("case_id", "target_bacc"),
        [{"case_id": "case", "target_bacc": 0.9}],
    )
    with pytest.raises(ProtocolError, match="forbidden target/outcome column"):
        validation._reject_forbidden_persisted_fields(tmp_path)

    (tmp_path / "tables/test_candidate_contrasts.csv").unlink()
    _write_csv(
        tmp_path / "tables/source_regret_responses.csv",
        (
            "source_exact_bacc_gain_vs_control",
            "source_exact_regret_from_case_best",
            "source_accuracy",
        ),
        [
            {
                "source_exact_bacc_gain_vs_control": 0.1,
                "source_exact_regret_from_case_best": 0.0,
                "source_accuracy": 0.9,
            }
        ],
    )
    with pytest.raises(ProtocolError, match="forbidden target/outcome column"):
        validation._reject_forbidden_persisted_fields(tmp_path)


def test_source_response_class_counts_are_query_wide_across_multiple_cases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "tables/source_regret_responses.csv"
    typed_rows = tuple(
        CaseActionResponseRow(
            query_id="1",
            case_id=case_id,
            action_id="A0::source=2",
            source_id="2",
            exact_bacc_gain_vs_control=0.0,
            exact_regret_from_case_best=0.0,
            disagreement_count=0,
            positive_class_count=2,
            negative_class_count=3,
        )
        for case_id in ("case-a", "case-b")
    )
    surface_hash = "f" * 64

    def response_rows(rows: tuple[CaseActionResponseRow, ...]) -> list[dict[str, object]]:
        return [
            {
                "outer_target_id": "0",
                "geometry_id": "A0",
                "query_id": row.query_id,
                "case_id": row.case_id,
                "action_id": row.action_id,
                "source_id": row.source_id,
                "source_exact_bacc_gain_vs_control": row.exact_bacc_gain_vs_control,
                "source_exact_regret_from_case_best": row.exact_regret_from_case_best,
                "disagreement_count": row.disagreement_count,
                "positive_class_count": row.positive_class_count,
                "negative_class_count": row.negative_class_count,
                "response_hash": row.response_hash,
                "response_surface_hash": surface_hash,
            }
            for row in rows
        ]

    source_features = [
        {
            "outer_target_id": "0",
            "geometry_id": "A0",
            "family": "R",
            "query_id": "1",
            "case_id": row.case_id,
            "action_id": row.action_id,
            "disagreement_count": "0",
        }
        for row in typed_rows
    ]
    query_sample_counts = {"1": 5}
    monkeypatch.setattr(validation_surfaces, "EXPECTED_SOURCE_RESPONSE_ROWS", 2)
    _write_csv(
        path,
        validation_surfaces.SOURCE_RESPONSE_FIELDS,
        response_rows(typed_rows),
    )

    validated = validation_surfaces.validate_response_table(
        path,
        source_features=source_features,
        source_query_sample_counts=query_sample_counts,
    )
    assert len(validated) == 2

    drifted = CaseActionResponseRow(
        query_id="1",
        case_id="case-a",
        action_id="A0::source=2",
        source_id="2",
        exact_bacc_gain_vs_control=0.0,
        exact_regret_from_case_best=0.0,
        disagreement_count=0,
        positive_class_count=1,
        negative_class_count=3,
    )
    _write_csv(
        path,
        validation_surfaces.SOURCE_RESPONSE_FIELDS,
        response_rows((drifted, typed_rows[1])),
    )
    with pytest.raises(ProtocolError, match="query class-count total drifted"):
        validation_surfaces.validate_response_table(
            path,
            source_features=source_features,
            source_query_sample_counts=query_sample_counts,
        )

def test_source_capability_is_exact_hashed_and_fail_closed(tmp_path: Path) -> None:
    source_seal = "1" * 64
    source_classifier_seal = "2" * 64
    target_classifier_seal = "3" * 64
    unhashed = {
        "schema_version": "midogpp_prediction_only_source_label_capability_v1",
        "status": "OPEN_SOURCE_ONLY",
        "source_prediction_seal_hash": source_seal,
        "source_oof_classifier_bank_seal_hash": source_classifier_seal,
        "target_classifier_bank_seal_hash": target_classifier_seal,
        "source_row_count": 9_648,
        "outer_targets_accessed": list(CENTERS),
        "outer_target_label_excluded": True,
        "query_excluded_from_every_source_action_composition": True,
        "source_labels_opened": True,
        "source_labels_opened_after_complete_prediction_seal": True,
        "source_oof_physical_classifier_fit_count": 5_184,
        "source_oof_oriented_prediction_cell_count": 10_368,
        "target_compatible_classifier_fit_count": 1_458,
        "test_manifest_opened": False,
        "test_labels_opened": False,
        "test_labels_available": False,
        "raw_source_labels_persisted": False,
        "raw_sample_ids_persisted": False,
    }
    path = tmp_path / "manifests/source_label_capability_report.json"
    _write_json(path, {**unhashed, "access_report_hash": canonical_hash(unhashed)})

    observed = validation._validate_source_capability(
        tmp_path,
        source_prediction_seal_hash=source_seal,
        source_oof_classifier_bank_seal_hash=source_classifier_seal,
        target_classifier_bank_seal_hash=target_classifier_seal,
    )
    assert observed["source_row_count"] == 9_648
    assert observed["test_labels_opened"] is False

    tampered = dict(observed)
    tampered["test_labels_available"] = True
    unhashed_tampered = {
        key: value for key, value in tampered.items() if key != "access_report_hash"
    }
    tampered["access_report_hash"] = canonical_hash(unhashed_tampered)
    _write_json(path, tampered)
    with pytest.raises(ProtocolError, match="source label capability drifted"):
        validation._validate_source_capability(
            tmp_path,
            source_prediction_seal_hash=source_seal,
            source_oof_classifier_bank_seal_hash=source_classifier_seal,
            target_classifier_bank_seal_hash=target_classifier_seal,
        )


def test_provenance_requires_exact_repository_state_schema(tmp_path: Path) -> None:
    path = tmp_path / "provenance/input_artifacts.json"
    payload = {
        "schema_version": "midogpp_input_artifacts_v2",
        "dataset_id": "midogpp",
        "experiment_id": (
            "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_"
            "disagreement_regret_prediction_only.v1"
        ),
        "stage": "90_oracles_and_diagnostics",
        "claim_scope": "diagnostic_only",
        "selection_used_target_eval_artifacts": False,
        "input_artifacts": [
            {"artifact_id": artifact_id}
            for artifact_id in sorted(INPUT_ARTIFACT_IDS)
        ],
        "repository_revision": "1" * 40,
        "repository_dirty": False,
        "repository_status_hash": "2" * 64,
    }
    _write_json(path, payload)
    config = SimpleNamespace(input_artifact_ids=INPUT_ARTIFACT_IDS)

    assert set(validate_provenance(tmp_path, config=config)) == set(
        INPUT_ARTIFACT_IDS
    )

    _write_json(path, {**payload, "unexpected": True})
    with pytest.raises(ProtocolError, match="provenance header drifted"):
        validate_provenance(tmp_path, config=config)


def _repeat_cases(prefix: str, count: int, row_count: int) -> tuple[str, ...]:
    cases = tuple(f"{prefix}_case_{index:03d}" for index in range(count))
    return tuple(cases[index % count] for index in range(row_count))


def _identity_stores() -> tuple[SimpleNamespace, SimpleNamespace]:
    source_rows: list[str] = []
    source_cases: list[str] = []
    source_queries: list[str] = []
    for query in CENTERS:
        count = EXPECTED_SOURCE_ROWS_BY_CENTER[query]
        cursor = len(source_rows)
        source_rows.extend(
            f"src_{cursor + index:064x}" for index in range(count)
        )
        source_cases.extend(_repeat_cases(f"source_{query}", 24, count))
        source_queries.extend((query,) * count)
    source = SimpleNamespace(
        rows_by_query={
            query: tuple(
                row
                for row, observed_query in zip(
                    source_rows, source_queries, strict=True
                )
                if observed_query == query
            )
            for query in CENTERS
        },
        case_ids_by_query={
            query: tuple(
                case
                for case, observed_query in zip(
                    source_cases, source_queries, strict=True
                )
                if observed_query == query
            )
            for query in CENTERS
        },
    )

    # Exactly 218 whole cases, partitioned across the nine test centers.
    case_counts = dict(zip(CENTERS, (24, 24, 24, 24, 24, 24, 24, 24, 26), strict=True))
    test_rows: dict[str, tuple[str, ...]] = {}
    test_cases: dict[str, tuple[str, ...]] = {}
    test_queries: dict[str, tuple[str, ...]] = {}
    cursor = 0
    for target in CENTERS:
        count = EXPECTED_TEST_ROWS_BY_CENTER[target]
        test_rows[target] = tuple(
            f"eval_{cursor + index:064x}" for index in range(count)
        )
        cursor += count
        test_cases[target] = _repeat_cases(
            f"test_{target}", case_counts[target], count
        )
        test_queries[target] = (target,) * count
    test = SimpleNamespace(
        rows_by_outer_target=test_rows,
        case_ids_by_outer_target=test_cases,
        query_ids_by_outer_target=test_queries,
    )
    return source, test


def test_identity_replay_requires_all_9648_source_and_9928_test_rows() -> None:
    source, test = _identity_stores()
    result = validation._validate_identity_topology(source, test)

    assert sum(len(value) for value in result["source_cases_by_query"].values()) == 216
    assert result["source_query_sample_counts"] == EXPECTED_SOURCE_ROWS_BY_CENTER
    assert sum(len(value) for value in result["test_cases_by_query"].values()) == 218
    assert EXPECTED_CLASSIFIER_FIT_COUNT == 1_458

    duplicate = list(test.rows_by_outer_target["0"])
    duplicate[-1] = duplicate[0]
    drifted = SimpleNamespace(
        rows_by_outer_target={**test.rows_by_outer_target, "0": tuple(duplicate)},
        case_ids_by_outer_target=test.case_ids_by_outer_target,
        query_ids_by_outer_target=test.query_ids_by_outer_target,
    )
    with pytest.raises(ProtocolError, match="exact 9928-row test identity drifted"):
        validation._validate_identity_topology(source, drifted)


def test_identity_replay_rejects_legacy_h_only_replicated_source_store() -> None:
    source, test = _identity_stores()
    flat_rows = tuple(row for query in CENTERS for row in source.rows_by_query[query])
    flat_cases = tuple(
        case for query in CENTERS for case in source.case_ids_by_query[query]
    )
    legacy = SimpleNamespace(
        rows_by_outer_target={target: flat_rows for target in CENTERS},
        case_ids_by_outer_target={target: flat_cases for target in CENTERS},
        query_ids_by_outer_target={
            target: tuple(
                query
                for query in CENTERS
                for _row in source.rows_by_query[query]
            )
            for target in CENTERS
        },
    )

    with pytest.raises(ProtocolError, match="Strict source-OOF query identity"):
        validation._validate_identity_topology(legacy, test)


def test_test_feature_schema_requires_prediction_seal_and_permutation_origin(
    tmp_path: Path,
) -> None:
    incomplete = tuple(
        value
        for value in validation._TEST_FEATURE_FIELDS
        if value not in {"prediction_seal_hash", "feature_origin_action_id"}
    )
    _write_csv(tmp_path / "test.csv", incomplete, [])

    with pytest.raises(ProtocolError, match="CSV schema drifted"):
        validation._read_csv(
            tmp_path / "test.csv", fields=validation._TEST_FEATURE_FIELDS
        )


def test_test_prediction_cells_replay_frozen_classifier_parameters() -> None:
    row_id = "eval_" + "a" * 64
    key = ("0", "B", 17, 17)
    classifier = SimpleNamespace(
        action_hash="1" * 64,
        parameter_sha256="2" * 64,
    )
    bank = SimpleNamespace(by_key={key: classifier}, seal_hash="3" * 64)
    reloaded_bank = SimpleNamespace(
        by_key={key: classifier}, seal_hash=bank.seal_hash
    )
    cell = SimpleNamespace(
        key=key,
        target_center="0",
        action_hash=classifier.action_hash,
        classifier_parameter_sha256=classifier.parameter_sha256,
        row_identity_hash=canonical_hash([row_id]),
    )
    prediction = SimpleNamespace(
        test_store=SimpleNamespace(
            cells=(cell,), rows_by_outer_target={"0": (row_id,)}
        ),
        classifier_bank=reloaded_bank,
        admission=SimpleNamespace(
            source_prediction_seal_hash="4" * 64,
            action_classifier_bank_seal_hash=bank.seal_hash,
        ),
    )

    validation._validate_test_prediction_chain(
        prediction,
        target_classifier_bank=bank,
        composite_prediction_seal_hash="4" * 64,
    )
    assert prediction.classifier_bank is not bank

    reloaded_bank.seal_hash = "6" * 64
    with pytest.raises(ProtocolError, match="test admission lineage drifted"):
        validation._validate_test_prediction_chain(
            prediction,
            target_classifier_bank=bank,
            composite_prediction_seal_hash="4" * 64,
        )
    reloaded_bank.seal_hash = bank.seal_hash

    with pytest.raises(ProtocolError, match="test admission lineage drifted"):
        validation._validate_test_prediction_chain(
            SimpleNamespace(
                test_store=prediction.test_store,
                classifier_bank=SimpleNamespace(),
                admission=SimpleNamespace(
                    source_prediction_seal_hash="4" * 64,
                    action_classifier_bank_seal_hash=None,
                ),
            ),
            target_classifier_bank=SimpleNamespace(by_key={}),
            composite_prediction_seal_hash="4" * 64,
        )

    cell.classifier_parameter_sha256 = "5" * 64
    with pytest.raises(ProtocolError, match="escaped its frozen target classifier"):
        validation._validate_test_prediction_chain(
            prediction,
            target_classifier_bank=bank,
            composite_prediction_seal_hash="4" * 64,
        )


def test_test_cache_binding_is_rebuilt_from_preserved_input_provenance() -> None:
    admission_unhashed = {
        "schema_version": "midogpp_prediction_only_test_inference_admission_v1",
        "source_prediction_seal_hash": "1" * 64,
        "action_classifier_bank_seal_hash": "2" * 64,
        "regret_model_bank_seal_hash": "3" * 64,
        "regret_model_bank_status": "SEALED_SOURCE_ONLY_BEFORE_TEST_ADMISSION",
        "target_labels_available": False,
        "test_scoring_permitted": False,
        "classifier_refit_permitted": False,
    }
    admission = _TestInferenceAdmission(
        source_prediction_seal_hash="1" * 64,
        action_classifier_bank_seal_hash="2" * 64,
        regret_model_bank_seal_hash="3" * 64,
        regret_model_bank_status="SEALED_SOURCE_ONLY_BEFORE_TEST_ADMISSION",
        target_labels_available=False,
        test_scoring_permitted=False,
        admission_hash=canonical_hash(admission_unhashed),
    )
    files = [
        {
            "path": f"embeddings/by_center/center_{center}.pt",
            "exists": True,
            "computed": {"sha256": f"{index + 1:064x}"},
        }
        for index, center in enumerate(CENTERS)
    ]
    provenance = {
        TEST_CACHE_ARTIFACT_ID: {
            "semantic_identities": {
                "cache_name": EXPECTED_TEST_CACHE_SEMANTIC_ID,
                "content_hash": EXPECTED_TEST_CACHE_CONTENT_HASH,
                "row_order_hash": EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
                "row_count": "9928",
                "feature_dim": "3840",
                "manifest_sha256": EXPECTED_MANIFEST_SHA256,
                "representation_id": EXPECTED_TEST_CACHE_REPRESENTATION_ID,
                "split": "test",
                "experiment_fenced": "true",
                "fresh_evidence": "false",
                "labels_persisted": "false",
                "labels_absent": "true",
                "sample_ids_persisted": "false",
                "image_paths_persisted": "false",
                "metadata_artifact_used": "false",
                "previous_stage90_output_used": "false",
                "authorized_consumer_experiment_ids": EXPERIMENT_ID,
            },
            "file_integrity": {"files": files},
        }
    }
    config = SimpleNamespace(
        expected_train_cache_sha256=EXPECTED_TRAIN_CACHE_SHA256,
        expected_manifest_sha256=EXPECTED_MANIFEST_SHA256,
        expected_test_cache_semantic_id=EXPECTED_TEST_CACHE_SEMANTIC_ID,
        expected_test_cache_representation_id=(
            EXPECTED_TEST_CACHE_REPRESENTATION_ID
        ),
        expected_test_cache_content_hash=EXPECTED_TEST_CACHE_CONTENT_HASH,
        expected_test_cache_row_order_hash=EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
    )

    expected = expected_test_cache_binding_hash_from_provenance(
        config, admission=admission, provenance=provenance
    )
    changed_files = [dict(row) for row in files]
    changed_files[0] = {
        **changed_files[0],
        "computed": {"sha256": "f" * 64},
    }
    changed = expected_test_cache_binding_hash_from_provenance(
        config,
        admission=admission,
        provenance={
            TEST_CACHE_ARTIFACT_ID: {
                **provenance[TEST_CACHE_ARTIFACT_ID],
                "file_integrity": {"files": changed_files},
            }
        },
    )

    assert len(expected) == 64
    assert changed != expected

    drifted_identities = dict(
        provenance[TEST_CACHE_ARTIFACT_ID]["semantic_identities"]
    )
    drifted_identities["manifest_sha256"] = "0" * 64
    with pytest.raises(ProtocolError, match="test-cache provenance drifted"):
        expected_test_cache_binding_hash_from_provenance(
            config,
            admission=admission,
            provenance={
                TEST_CACHE_ARTIFACT_ID: {
                    **provenance[TEST_CACHE_ARTIFACT_ID],
                    "semantic_identities": drifted_identities,
                }
            },
        )


def test_run_state_can_only_complete_after_validation_report_exists(
    tmp_path: Path,
) -> None:
    path = tmp_path / "reports/run_state.json"
    _write_json(
        path,
        {
            "schema_version": "midogpp_disagreement_regret_prediction_only_run_state_v1",
            "status": "COMPLETE",
            "phase": "COMPLETE",
            "prediction_only": True,
            "test_labels_opened": False,
        },
    )
    with pytest.raises(ProtocolError, match="run state is not validatable"):
        validation._validate_run_state(tmp_path, validation_exists=False)

    validation._validate_run_state(tmp_path, validation_exists=True)
