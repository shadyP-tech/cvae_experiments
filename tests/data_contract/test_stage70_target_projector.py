from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from midogpp_thesis.data.contract.stage70_target_evaluation import (
    CANONICAL_MANIFEST_SHA256,
    ELIGIBLE_CENTERS,
    EXPECTED_TEST_ROWS,
    EXPECTED_TEST_ROWS_BY_CENTER,
    FRESH_EVIDENCE,
    PURPOSE,
    TargetEvaluationContractError,
    evaluation_row_id,
    iter_bound_image_bytes,
    load_target_evaluation_reservation,
    project_target_evaluation_manifest,
    project_target_evaluation_rows,
    validate_target_evaluation_reservation,
    validate_target_evaluation_reservation_against_manifest,
    write_target_evaluation_reservation,
)


CANONICAL_MANIFEST = Path(
    "datasets/midogpp/contract/annotation_patch_v1/manifest.csv"
)


def test_canonical_projector_reserves_exact_previously_consumed_test_rows() -> None:
    reservation = project_target_evaluation_manifest(CANONICAL_MANIFEST)

    assert reservation.manifest_sha256 == CANONICAL_MANIFEST_SHA256
    assert reservation.row_count == EXPECTED_TEST_ROWS == 9928
    assert reservation.rows_by_center == dict(EXPECTED_TEST_ROWS_BY_CENTER)
    assert tuple(reservation.rows_by_center) == ELIGIBLE_CENTERS
    assert not any(row.center == "4" for row in reservation.rows)
    assert all(row.split == "test" for row in reservation.rows)
    assert reservation.purpose == PURPOSE
    assert reservation.fresh_evidence is FRESH_EVIDENCE is False
    assert reservation.coverage_scope == "canonical"
    assert reservation.rows[0].evaluation_row_id == evaluation_row_id(
        CANONICAL_MANIFEST_SHA256,
        reservation.rows[0].contract_row_index,
    )
    assert set(reservation.rows[0].to_dict()) == {
        "evaluation_row_id",
        "contract_row_index",
        "case_id",
        "center",
        "split",
    }
    assert validate_target_evaluation_reservation(reservation)["status"] == "PASS"


def test_projection_accesses_only_whitelisted_fields_and_ids_ignore_forbidden_values() -> None:
    class SentinelRow(dict[str, object]):
        def __init__(self, payload: dict[str, object]) -> None:
            super().__init__(payload)
            self.accessed: list[str] = []

        def __getitem__(self, key: str) -> object:
            self.accessed.append(key)
            if key in {"label", "label_name", "sample_id", "image_path"}:
                raise AssertionError(f"forbidden field accessed: {key}")
            return super().__getitem__(key)

    manifest_sha256 = "a" * 64
    first = SentinelRow(
        {
            "case_id": "case-1",
            "center": "0",
            "split": "test",
            "label": 0,
            "label_name": "negative",
            "sample_id": "source__y0",
            "image_path": "secret__y0.jpg",
        }
    )
    second = SentinelRow(
        {
            "case_id": "case-1",
            "center": "0",
            "split": "test",
            "label": 1,
            "label_name": "positive",
            "sample_id": "different__y1",
            "image_path": "other__y1.jpg",
        }
    )
    access_events = []

    left = project_target_evaluation_rows(
        [first],
        manifest_sha256=manifest_sha256,
        expected_rows_by_center={"0": 1},
        coverage_scope="test_fixture_only",
        access_log=access_events,
    )
    right = project_target_evaluation_rows(
        [second],
        manifest_sha256=manifest_sha256,
        expected_rows_by_center={"0": 1},
        coverage_scope="test_fixture_only",
    )

    assert left.rows == right.rows
    assert left.reservation_id == right.reservation_id
    assert first.accessed == ["split", "center", "case_id"]
    assert second.accessed == ["split", "center", "case_id"]
    assert {event.field for event in access_events} == {"split", "center", "case_id"}


def test_reservation_json_round_trip_contains_no_source_or_outcome_identity(
    tmp_path: Path,
) -> None:
    manifest = _fixture_manifest(tmp_path)
    manifest_sha256 = _sha256(manifest)
    reservation = project_target_evaluation_manifest(
        manifest,
        expected_manifest_sha256=manifest_sha256,
        expected_rows_by_center={"0": 1},
        allow_test_fixture=True,
    )
    path = tmp_path / "reservation.json"

    write_target_evaluation_reservation(
        reservation,
        path,
        allow_test_fixture=True,
        expected_rows_by_center={"0": 1},
    )
    loaded = load_target_evaluation_reservation(
        path,
        expected_manifest_sha256=manifest_sha256,
        expected_rows_by_center={"0": 1},
        allow_test_fixture=True,
    )

    assert loaded == reservation
    serialized = path.read_text(encoding="utf-8").casefold()
    for forbidden in ('"label"', '"label_name"', '"sample_id"', '"image_path"'):
        assert forbidden not in serialized
    assert "__y0" not in serialized
    assert "__y1" not in serialized


def test_opaque_image_resolver_reads_only_bound_source_location_and_discards_it(
    tmp_path: Path,
) -> None:
    manifest = _fixture_manifest(tmp_path)
    manifest_sha256 = _sha256(manifest)
    reservation = project_target_evaluation_manifest(
        manifest,
        expected_manifest_sha256=manifest_sha256,
        expected_rows_by_center={"0": 1},
        allow_test_fixture=True,
    )
    access_events = []
    seen_paths: list[Path] = []

    records = list(
        iter_bound_image_bytes(
            manifest,
            reservation,
            repo_root=tmp_path,
            image_reader=lambda path: seen_paths.append(path) or path.read_bytes(),
            access_log=access_events,
            allow_test_fixture=True,
        )
    )

    assert len(records) == 1
    assert records[0].row == reservation.rows[0]
    assert records[0].jpeg_bytes == b"test-jpeg"
    assert not hasattr(records[0], "image_path")
    assert seen_paths[0].name == "test__y1.jpg"
    assert [(event.field, event.contract_row_index) for event in access_events] == [
        ("image_path", 1)
    ]


def test_independent_validation_reprojects_manifest_and_rejects_extra_identity(
    tmp_path: Path,
) -> None:
    manifest = _fixture_manifest(tmp_path)
    manifest_sha256 = _sha256(manifest)
    reservation = project_target_evaluation_manifest(
        manifest,
        expected_manifest_sha256=manifest_sha256,
        expected_rows_by_center={"0": 1},
        allow_test_fixture=True,
    )
    assert validate_target_evaluation_reservation_against_manifest(
        manifest,
        reservation,
        expected_rows_by_center={"0": 1},
        allow_test_fixture=True,
    )["status"] == "PASS"

    path = tmp_path / "reservation.json"
    write_target_evaluation_reservation(
        reservation,
        path,
        expected_rows_by_center={"0": 1},
        allow_test_fixture=True,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rows"][0]["label"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TargetEvaluationContractError, match="forbidden|schema"):
        load_target_evaluation_reservation(
            path,
            expected_manifest_sha256=manifest_sha256,
            expected_rows_by_center={"0": 1},
            allow_test_fixture=True,
        )


def _fixture_manifest(root: Path) -> Path:
    (root / "train__y0.jpg").write_bytes(b"train-jpeg")
    (root / "test__y1.jpg").write_bytes(b"test-jpeg")
    path = root / "manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sample_id",
                "case_id",
                "image_path",
                "label",
                "label_name",
                "center",
                "split",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "train__y0",
                "case_id": "case-train",
                "image_path": "train__y0.jpg",
                "label": 0,
                "label_name": "negative",
                "center": "0",
                "split": "train",
            }
        )
        writer.writerow(
            {
                "sample_id": "test__y1",
                "case_id": "case-test",
                "image_path": "test__y1.jpg",
                "label": 1,
                "label_name": "positive",
                "center": "0",
                "split": "test",
            }
        )
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
