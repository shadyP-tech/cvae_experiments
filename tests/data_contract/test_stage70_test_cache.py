from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest
import torch

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.data.contract.stage70_target_evaluation import (
    project_target_evaluation_manifest,
)
from midogpp_thesis.data.features.stage70_test_cache import (
    CACHE_REQUIRED_FILES,
    CONFIG_AUTHORIZATION_BINDING_FIELDS,
    CONFIG_CACHE_FIELDS,
    CONFIG_INPUT_FIELDS,
    CONFIG_PROTOCOL_FIELDS,
    CONFIG_RUN_FIELDS,
    CONFIG_TOP_LEVEL_FIELDS,
    FEATURE_DIM,
    FIXED_WINDOW_START,
    FRESH_EVIDENCE,
    PURPOSE,
    RESERVATION_ARTIFACT_REQUIRED_FILES,
    SHARD_METADATA_FIELDS,
    Stage70TestCacheError,
    build_cli_parser,
    build_stage70_test_cache,
    expected_model_identity,
    load_stage70_center_shard,
    load_validated_stage70_test_cache,
    make_stage70_test_cache_config,
    stage70_extractor_protocol,
    stage70_extractor_protocol_hash,
    validate_stage70_test_cache,
    validate_reservation_artifact_binding,
)
from midogpp_thesis.data.features.stage70_test_cache.io import write_content_index


class FakeExtractor:
    identity = {
        **expected_model_identity(),
        "preprocessing_config": {
            "input_size": [3, 224, 224],
            "interpolation": "bicubic",
        },
    }

    def __init__(self, **kwargs: object) -> None:
        assert kwargs["model_ref"] == self.identity["model_ref"]
        assert kwargs["model_revision"] == self.identity["requested_revision"]

    def extract_spatial_windows(
        self,
        images: list[Image.Image],
        *,
        window_starts: list[tuple[int, int]],
    ) -> torch.Tensor:
        assert window_starts == [FIXED_WINDOW_START] * len(images)
        values = [float(image.getpixel((0, 0))[0]) for image in images]
        return torch.stack(
            [torch.full((FEATURE_DIM,), value, dtype=torch.float32) for value in values]
        )


class BadShapeExtractor(FakeExtractor):
    def extract_spatial_windows(
        self,
        images: list[Image.Image],
        *,
        window_starts: list[tuple[int, int]],
    ) -> torch.Tensor:
        return torch.zeros((len(images), FEATURE_DIM - 1), dtype=torch.float32)


def test_extractor_protocol_is_pinned_and_marks_test_as_previously_consumed() -> None:
    protocol = stage70_extractor_protocol()

    assert protocol["feature_dim"] == 3840
    assert protocol["pooling"] == "fixed_center_rows6to9_cols6to9"
    assert protocol["fixed_window_start"] == [6, 6]
    assert protocol["expected_row_count"] == 9928
    assert protocol["purpose"] == PURPOSE
    assert protocol["fresh_evidence"] is FRESH_EVIDENCE is False
    assert protocol["evidence_status"] == "previously_consumed_test"
    assert len(stage70_extractor_protocol_hash()) == 64


def test_public_cli_config_schema_and_required_file_contract_are_exact() -> None:
    args = build_cli_parser().parse_args(
        ["--config", "stage70.yaml", "--validate-only"]
    )

    assert args.config == Path("stage70.yaml")
    assert args.validate_only is True
    assert CONFIG_TOP_LEVEL_FIELDS == {
        "cache",
        "inputs",
        "authorization_binding",
        "run",
        "protocol",
    }
    assert CONFIG_CACHE_FIELDS == {"name", "artifact_id", "root"}
    assert CONFIG_INPUT_FIELDS == {
        "repo_root",
        "manifest_path",
        "target_evaluation_reservation_path",
    }
    assert CONFIG_AUTHORIZATION_BINDING_FIELDS == {
        "scoring_manifest_sha256",
        "target_evaluation_reservation_id",
        "target_evaluation_reservation_protocol_hash",
        "cache_extractor_protocol_hash",
    }
    assert CONFIG_RUN_FIELDS == {
        "eligible_centers",
        "expected_row_count",
        "expected_rows_by_center",
        "experiment_seed",
        "device",
        "batch_size",
    }
    assert CONFIG_PROTOCOL_FIELDS == {
        "authorized_consumer_experiment_id",
        "purpose",
        "fresh_evidence",
    }
    assert CACHE_REQUIRED_FILES == (
        "embeddings/by_center/center_0.pt",
        "embeddings/by_center/center_1.pt",
        "embeddings/by_center/center_2.pt",
        "embeddings/by_center/center_3.pt",
        "embeddings/by_center/center_5.pt",
        "embeddings/by_center/center_6.pt",
        "embeddings/by_center/center_7.pt",
        "embeddings/by_center/center_8.pt",
        "embeddings/by_center/center_9.pt",
        "manifests/content_index.json",
        "manifests/frozen_build_protocol.json",
        "manifests/row_alignment.json",
        "reports/cache_builder_report.json",
        "reports/validation_report.json",
    )


def test_fake_extractor_builds_atomic_label_sealed_center_shards(tmp_path: Path) -> None:
    fixture = _cache_fixture(tmp_path)
    access_events = []

    built = build_stage70_test_cache(
        fixture["config"],
        reservation=fixture["reservation"],
        extractor_factory=FakeExtractor,
        access_log=access_events,
    )
    summary = validate_stage70_test_cache(
        built,
        expected_config=fixture["config"],
        expected_reservation=fixture["reservation"],
    )

    assert built == fixture["config"].cache_root
    assert summary["status"] == "PASS"
    assert summary["row_count"] == 3
    assert summary["rows_by_center"] == {"0": 2, "1": 1}
    assert summary["target_evaluation_reservation_id"] == fixture[
        "reservation"
    ].reservation_id
    assert summary["cache_extractor_protocol_hash"] == stage70_extractor_protocol_hash()
    assert summary["purpose"] == PURPOSE
    assert summary["fresh_evidence"] is False
    assert len(summary["content_hash"]) == 64
    assert [(event.field, event.contract_row_index) for event in access_events] == [
        ("image_path", 1),
        ("image_path", 2),
        ("image_path", 3),
    ]

    all_metadata = []
    for center in ("0", "1"):
        shard = load_stage70_center_shard(
            built / "embeddings" / "by_center" / f"center_{center}.pt",
            expected_center=center,
        )
        assert shard.embeddings.shape[1] == FEATURE_DIM
        assert all(set(row) == SHARD_METADATA_FIELDS for row in shard.metadata)
        all_metadata.extend(shard.metadata)
    serialized_metadata = json.dumps(all_metadata, sort_keys=True).casefold()
    for forbidden in ("label", "label_name", "sample_id", "image_path", "class"):
        assert forbidden not in serialized_metadata
    assert "__y0" not in serialized_metadata
    assert "__y1" not in serialized_metadata

    loaded = load_validated_stage70_test_cache(
        built,
        expected_config=fixture["config"],
        expected_reservation=fixture["reservation"],
    )
    assert loaded.load_center("0").evaluation_row_ids == tuple(
        row.evaluation_row_id
        for row in fixture["reservation"].rows
        if row.center == "0"
    )


def test_validator_rejects_outcome_field_even_after_content_index_is_rehashed(
    tmp_path: Path,
) -> None:
    fixture = _cache_fixture(tmp_path)
    root = build_stage70_test_cache(
        fixture["config"],
        reservation=fixture["reservation"],
        extractor_factory=FakeExtractor,
    )
    shard_path = root / "embeddings" / "by_center" / "center_0.pt"
    payload = torch.load(shard_path, map_location="cpu", weights_only=True)
    payload["metadata"][0]["label"] = 1
    torch.save(payload, shard_path)
    write_content_index(root)

    with pytest.raises(Stage70TestCacheError, match="metadata|firewall"):
        validate_stage70_test_cache(
            root,
            expected_config=fixture["config"],
            expected_reservation=fixture["reservation"],
        )


def test_validator_rejects_row_identity_not_derived_from_manifest_position(
    tmp_path: Path,
) -> None:
    fixture = _cache_fixture(tmp_path)
    root = build_stage70_test_cache(
        fixture["config"],
        reservation=fixture["reservation"],
        extractor_factory=FakeExtractor,
    )
    shard_path = root / "embeddings" / "by_center" / "center_0.pt"
    payload = torch.load(shard_path, map_location="cpu", weights_only=True)
    payload["metadata"][0]["evaluation_row_id"] = "eval_" + hashlib.sha256(
        b"secret__y1.jpg"
    ).hexdigest()
    torch.save(payload, shard_path)
    write_content_index(root)

    with pytest.raises(Stage70TestCacheError, match="identity|reservation|alignment"):
        validate_stage70_test_cache(
            root,
            expected_config=fixture["config"],
            expected_reservation=fixture["reservation"],
        )


def test_content_index_rejects_unindexed_report_tamper(tmp_path: Path) -> None:
    fixture = _cache_fixture(tmp_path)
    root = build_stage70_test_cache(
        fixture["config"],
        reservation=fixture["reservation"],
        extractor_factory=FakeExtractor,
    )
    report_path = root / "reports" / "cache_builder_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["row_count"] = 2
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(Stage70TestCacheError, match="indexed member drifted"):
        validate_stage70_test_cache(
            root,
            expected_config=fixture["config"],
            expected_reservation=fixture["reservation"],
        )


def test_failed_extraction_never_publishes_partial_cache(tmp_path: Path) -> None:
    fixture = _cache_fixture(tmp_path)

    with pytest.raises(Stage70TestCacheError, match="geometry"):
        build_stage70_test_cache(
            fixture["config"],
            reservation=fixture["reservation"],
            extractor_factory=BadShapeExtractor,
        )

    assert not fixture["config"].cache_root.exists()


def test_missing_reservation_artifact_fails_before_model_or_image_access(
    tmp_path: Path,
) -> None:
    fixture = _cache_fixture(tmp_path)
    config = replace(
        fixture["config"],
        cache_root=tmp_path / "missing-reservation-cache",
        reservation_path=tmp_path / "does-not-exist",
    )
    factory_calls: list[dict[str, object]] = []
    image_reads: list[Path] = []

    def factory(**kwargs: object) -> FakeExtractor:
        factory_calls.append(dict(kwargs))
        return FakeExtractor(**kwargs)

    def image_reader(path: Path) -> bytes:
        image_reads.append(path)
        return path.read_bytes()

    with pytest.raises(Stage70TestCacheError, match="reservation artifact root"):
        build_stage70_test_cache(
            config,
            reservation=fixture["reservation"],
            extractor_factory=factory,
            image_reader=image_reader,
        )

    assert factory_calls == []
    assert image_reads == []
    assert not config.cache_root.exists()


def test_rehashed_reservation_identity_tamper_fails_before_model_or_image_access(
    tmp_path: Path,
) -> None:
    fixture = _cache_fixture(tmp_path)
    reservation_root = fixture["config"].reservation_path
    assert reservation_root is not None
    identity_path = reservation_root / "manifests" / "identity_lock.json"
    protocol_path = reservation_root / "manifests" / "protocol_manifest.json"
    validation_path = reservation_root / "reports" / "validation_report.json"

    identity = _json(identity_path)
    identity["cache_extractor_protocol_hash"] = "f" * 64
    _set_embedded_hash(identity, "identity_lock_hash")
    _write_json(identity_path, identity)
    protocol = _json(protocol_path)
    protocol["identity_lock_hash"] = identity["identity_lock_hash"]
    _set_embedded_hash(protocol, "protocol_hash")
    _write_json(protocol_path, protocol)
    content = _write_reservation_content_index(reservation_root)
    validation = _json(validation_path)
    validation["checks"]["authorization_protocol_hash"] = protocol["protocol_hash"]
    validation["checks"]["content_hash"] = content["content_hash"]
    _write_json(validation_path, validation)

    factory_calls: list[dict[str, object]] = []
    image_reads: list[Path] = []

    def factory(**kwargs: object) -> FakeExtractor:
        factory_calls.append(dict(kwargs))
        return FakeExtractor(**kwargs)

    def image_reader(path: Path) -> bytes:
        image_reads.append(path)
        return path.read_bytes()

    with pytest.raises(Stage70TestCacheError, match="identity-lock binding"):
        build_stage70_test_cache(
            replace(fixture["config"], cache_root=tmp_path / "tampered-cache"),
            reservation=fixture["reservation"],
            extractor_factory=factory,
            image_reader=image_reader,
        )

    assert factory_calls == []
    assert image_reads == []


def _cache_fixture(tmp_path: Path) -> dict[str, object]:
    manifest = _write_manifest_and_images(tmp_path)
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    counts = {"0": 2, "1": 1}
    reservation = project_target_evaluation_manifest(
        manifest,
        expected_manifest_sha256=manifest_sha256,
        expected_rows_by_center=counts,
        allow_test_fixture=True,
    )
    reservation_root = _write_reservation_artifact(
        tmp_path / "reservation",
        reservation,
    )
    config = make_stage70_test_cache_config(
        cache_root=tmp_path / "cache",
        repo_root=tmp_path,
        manifest_path=manifest,
        reservation=reservation,
        reservation_path=reservation_root,
        batch_size=2,
        device="cpu",
        allow_test_fixture=True,
    )
    return {"manifest": manifest, "reservation": reservation, "config": config}


def _write_reservation_artifact(root: Path, reservation: object) -> Path:
    rows = [
        {
            "evaluation_row_id": row.evaluation_row_id,
            "contract_row_index": row.contract_row_index,
            "target_center": row.center,
            "split": row.split,
        }
        for row in reservation.rows
    ]
    target_hash = stable_hash(rows)
    (root / "tables").mkdir(parents=True)
    with (root / "tables" / "target_identity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "evaluation_row_id",
                "contract_row_index",
                "target_center",
                "split",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    identity = {
        "schema_version": "midogpp_stage70_target_identity_lock_v1",
        "experiment_id": (
            "midogpp.frozen_policy_downstream."
            "uniform_b_v2_descriptive_test_reservation.v1"
        ),
        "claim_scope": "target_evaluation_authorization",
        "purpose": PURPOSE,
        "fresh_evidence": FRESH_EVIDENCE,
        "scoring_manifest_sha256": reservation.manifest_sha256,
        "target_evaluation_reservation_id": reservation.reservation_id,
        "target_evaluation_reservation_protocol_hash": reservation.protocol_hash,
        "target_identity_table_hash": target_hash,
        "row_count": reservation.row_count,
        "rows_by_center": reservation.rows_by_center,
        "split": "test",
        "opaque_evaluation_row_ids_only": True,
        "sample_ids_persisted": False,
        "image_paths_persisted": False,
        "target_label_values_persisted": False,
        "cache_artifact_id": (
            "midogpp_virchow2_uniform_b_v2_descriptive_test_cache_seed42"
        ),
        "cache_extractor_protocol_hash": stage70_extractor_protocol_hash(),
    }
    _set_embedded_hash(identity, "identity_lock_hash")
    plan = {
        "schema_version": "midogpp_stage70_cache_extraction_reservation_plan_v1",
        "target_evaluation_reservation_id": reservation.reservation_id,
        "identity_lock_hash": identity["identity_lock_hash"],
    }
    _set_embedded_hash(plan, "evaluation_plan_hash")
    protocol = {
        "schema_version": (
            "midogpp_stage70_target_evaluation_reservation_protocol_v1"
        ),
        "phase": "TARGET_EVALUATION_RESERVATION",
        "experiment_id": (
            "midogpp.frozen_policy_downstream."
            "uniform_b_v2_descriptive_test_reservation.v1"
        ),
        "claim_scope": "target_evaluation_authorization",
        "purpose": PURPOSE,
        "fresh_evidence": FRESH_EVIDENCE,
        "authorized_consumer_experiment_id": (
            "midogpp.frozen_policy_downstream."
            "uniform_b_v2_descriptive_frozen_policy_comparison.v1"
        ),
        "scoring_manifest_sha256": reservation.manifest_sha256,
        "descriptive_locked_model_scoring_allowed": True,
        "identity_lock_hash": identity["identity_lock_hash"],
        "evaluation_plan_hash": plan["evaluation_plan_hash"],
        "cache_extractor_protocol_hash": stage70_extractor_protocol_hash(),
        "target_labels_opened": False,
        "generation_performed": False,
        "classifier_fit_performed": False,
        "prediction_performed": False,
        "metric_scoring_performed": False,
    }
    _set_embedded_hash(protocol, "protocol_hash")
    decision = {
        "schema_version": "midogpp_stage70_reservation_decision_v1",
        "status": "COMPLETE",
        "purpose": PURPOSE,
        "fresh_evidence": FRESH_EVIDENCE,
        "cache_extraction_allowed": True,
        "prediction_allowed": False,
        "label_access_allowed": False,
        "metric_scoring_allowed": False,
        "generation_or_policy_refit_allowed": False,
    }
    _set_embedded_hash(decision, "decision_hash")
    leakage = {
        "status": "PASS",
        "purpose": PURPOSE,
        "fresh_evidence": FRESH_EVIDENCE,
        "previously_consumed_test_rows": True,
        "target_label_values_opened": False,
        "target_label_values_persisted": False,
        "sample_ids_persisted": False,
        "image_paths_persisted": False,
        "policy_or_seed_selection_performed": False,
        "generation_performed": False,
        "classifier_fit_performed": False,
        "prediction_performed": False,
        "metric_scoring_performed": False,
    }
    run_state = {
        "schema_version": "midogpp_stage70_reservation_run_state_v1",
        "status": "COMPLETE",
        "purpose": PURPOSE,
        "fresh_evidence": FRESH_EVIDENCE,
    }
    passthrough = {
        "schema_version": "midogpp_stage70_authorization_input_provenance_v1",
        "status": "fixture",
    }
    (root / "config.resolved.yaml").write_text("fixture: true\n", encoding="utf-8")
    _write_json(root / "manifests" / "input_binding.json", passthrough)
    _write_json(root / "provenance" / "input_artifacts.json", passthrough)
    _write_json(root / "manifests" / "identity_lock.json", identity)
    _write_json(root / "manifests" / "evaluation_plan.json", plan)
    _write_json(root / "manifests" / "protocol_manifest.json", protocol)
    _write_json(root / "reports" / "authorization_decision.json", decision)
    _write_json(root / "reports" / "leakage_report.json", leakage)
    _write_json(root / "reports" / "run_state.json", run_state)
    content = _write_reservation_content_index(root)
    _write_json(
        root / "reports" / "validation_report.json",
        {
            "schema_version": (
                "midogpp_stage70_target_evaluation_reservation_validation_v1"
            ),
            "status": "PASS",
            "validator": "validate_target_evaluation_reservation",
            "checks": {
                "status": "PASS",
                "row_count": reservation.row_count,
                "target_evaluation_reservation_id": reservation.reservation_id,
                "target_evaluation_reservation_protocol_hash": reservation.protocol_hash,
                "target_identity_table_hash": target_hash,
                "authorization_protocol_hash": protocol["protocol_hash"],
                "content_hash": content["content_hash"],
                "prediction_performed": False,
                "metric_scoring_performed": False,
                "target_labels_opened": False,
            },
        },
    )
    binding = validate_reservation_artifact_binding(
        root,
        reservation=reservation,
        expected_cache_extractor_protocol_hash=stage70_extractor_protocol_hash(),
    )
    assert binding.validation_status == "PASS"
    return root


def _write_reservation_content_index(root: Path) -> dict[str, object]:
    excluded = {
        "manifests/content_index.json",
        "reports/run_state.json",
        "reports/validation_report.json",
    }
    records = []
    for relative in RESERVATION_ARTIFACT_REQUIRED_FILES:
        if relative in excluded:
            continue
        member = root / relative
        records.append(
            {
                "relative_path": relative,
                "sha256": hashlib.sha256(member.read_bytes()).hexdigest(),
                "size_bytes": member.stat().st_size,
            }
        )
    payload = {
        "schema_version": "midogpp_stage70_authorization_content_index_v1",
        "records": records,
    }
    payload["content_hash"] = stable_hash(payload)
    _write_json(root / "manifests" / "content_index.json", payload)
    return payload


def _set_embedded_hash(payload: dict[str, object], field: str) -> None:
    payload.pop(field, None)
    payload[field] = stable_hash(payload)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_manifest_and_images(root: Path) -> Path:
    rows = (
        ("train__y0", "case-train", "train__y0.jpg", 0, "negative", "0", "train", 5),
        ("test-a__y0", "case-a", "test-a__y0.jpg", 0, "negative", "0", "test", 10),
        ("test-b__y1", "case-b", "test-b__y1.jpg", 1, "positive", "1", "test", 20),
        ("test-c__y0", "case-c", "test-c__y0.jpg", 0, "negative", "0", "test", 30),
        ("excluded__y1", "case-excluded", "excluded__y1.jpg", 1, "positive", "4", "test", 40),
    )
    for _source_id, _case_id, image_name, _outcome, _name, _center, _split, value in rows:
        Image.new("RGB", (8, 8), color=(value, value, value)).save(root / image_name)
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_id",
                "case_id",
                "image_path",
                "label",
                "label_name",
                "center",
                "split",
            ]
        )
        for row in rows:
            writer.writerow(row[:-1])
    return manifest
