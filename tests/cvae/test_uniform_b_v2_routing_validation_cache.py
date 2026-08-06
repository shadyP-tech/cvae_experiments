from __future__ import annotations

from pathlib import Path

import pytest
import torch
from midogpp_thesis.data.features.uniform_b_routing_validation.cache import (
    LabelBlindManifestRow,
    _consumption_lock,
    manifest_split_overlap_counts,
    read_and_reserve_label_blind_manifest,
    write_content_index,
)
from midogpp_thesis.data.features.uniform_b_routing_validation.config import (
    CANONICAL_VALIDATION_URI,
    ELIGIBLE_CENTERS,
    EXPECTED_VALIDATION_ROWS,
    EXPECTED_VALIDATION_ROWS_BY_CENTER,
    MANIFEST_URI,
    OUTPUT_RELATIVE_ROOT,
    RoutingValidationCacheError,
    SOURCE_TRAIN_B_REPORT_URI,
    load_routing_validation_cache_config,
)
from midogpp_thesis.data.features.uniform_b_routing_validation.validation import (
    load_unlabeled_validation_shard,
    validate_content_index,
    validate_uniform_b_routing_validation_cache,
)


CONFIG = Path(
    "experiments/midogpp/stages/60_routing_and_composition/configs/"
    "uniform_b_v2_routing_validation_cache_v1.yaml"
)
MANIFEST = Path("datasets/midogpp/contract/annotation_patch_v1/manifest.csv")


def test_config_freezes_exact_validation_cache_identity_and_artifact_uris() -> None:
    config = load_routing_validation_cache_config(CONFIG)

    assert config.output_root_location == OUTPUT_RELATIVE_ROOT
    assert config.manifest_location == MANIFEST_URI
    assert config.canonical_validation_location == CANONICAL_VALIDATION_URI
    assert config.source_train_b_report_location == SOURCE_TRAIN_B_REPORT_URI
    assert config.eligible_centers == ELIGIBLE_CENTERS
    assert config.expected_validation_rows == EXPECTED_VALIDATION_ROWS == 2615
    assert (
        dict(config.expected_validation_rows_by_center)
        == EXPECTED_VALIDATION_ROWS_BY_CENTER
        == {
            "0": 375,
            "1": 62,
            "2": 1304,
            "3": 152,
            "5": 122,
            "6": 56,
            "7": 122,
            "8": 154,
            "9": 268,
        }
    )
    assert config.protocol["validation_labels_unobserved_before_lock"] is True
    assert config.protocol["feature_extraction_label_free"] is True
    assert config.protocol["output_metric_computed"] is False


@pytest.mark.parametrize(
    ("old", "new"),
    (
        (
            "name: uniform_b_v2_routing_validation_cache_v1",
            "name: posthoc_validation_cache",
        ),
        ("expected_validation_rows: 2615", "expected_validation_rows: 2614"),
        ("output_metric_computed: false", "output_metric_computed: true"),
        (
            "canonical_validation_cache_path: artifact://midogpp_virchow2_xyxy_validation_cache_seed42/embeddings/val.pt",
            "canonical_validation_cache_path: datasets/midogpp/derived/features/val.pt",
        ),
    ),
)
def test_config_rejects_identity_or_claim_drift(
    tmp_path: Path,
    old: str,
    new: str,
) -> None:
    text = CONFIG.read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / "config.yaml"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")

    with pytest.raises(
        RoutingValidationCacheError, match="identity drifted|protocol drifted"
    ):
        load_routing_validation_cache_config(path)


def test_label_blind_manifest_reserves_exact_disjoint_validation_rows() -> None:
    reservation = read_and_reserve_label_blind_manifest(MANIFEST)

    assert len(reservation.eligible_train_rows) == 9648
    assert len(reservation.validation_rows) == EXPECTED_VALIDATION_ROWS
    assert not any(reservation.overlap_counts.values())
    assert {row.center for row in reservation.validation_rows} == set(ELIGIBLE_CENTERS)
    assert all(row.center != "4" for row in reservation.validation_rows)
    assert all(not hasattr(row, "label") for row in reservation.validation_rows)
    counts = {
        center: sum(row.center == center for row in reservation.validation_rows)
        for center in ELIGIBLE_CENTERS
    }
    assert counts == EXPECTED_VALIDATION_ROWS_BY_CENTER


def test_manifest_overlap_audit_catches_case_reuse_across_splits() -> None:
    rows = (
        _row("train-sample", "shared-case", "train", 0),
        _row("val-sample", "shared-case", "val", 1),
        _row("test-sample", "test-case", "test", 2),
    )

    overlap = manifest_split_overlap_counts(rows)

    assert overlap["train_val_sample_overlap"] == 0
    assert overlap["train_val_case_overlap"] == 1
    assert overlap["train_test_case_overlap"] == 0
    assert overlap["val_test_case_overlap"] == 0


def test_content_index_rejects_member_tamper(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    member = root / "reports" / "cache_builder_report.json"
    member.parent.mkdir(parents=True)
    member.write_text('{"status":"PASS"}\n', encoding="utf-8")
    write_content_index(root)
    validate_content_index(root)

    member.write_text('{"status":"FAIL"}\n', encoding="utf-8")

    with pytest.raises(RoutingValidationCacheError, match="member drifted"):
        validate_content_index(root)


@pytest.mark.parametrize("forbidden_key", ("label", "y_true", "class"))
def test_unlabeled_shard_loader_rejects_outcome_metadata(
    tmp_path: Path,
    forbidden_key: str,
) -> None:
    path = tmp_path / f"{forbidden_key}.pt"
    metadata = [_metadata("sample-0", 0), _metadata("sample-1", 1)]
    metadata[0][forbidden_key] = 1
    _write_shard(path, metadata)

    with pytest.raises(RoutingValidationCacheError, match="strictly unlabeled"):
        load_unlabeled_validation_shard(path, expected_center="0")


def test_unlabeled_shard_loader_exposes_only_alignment_attributes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "center_0.pt"
    metadata = [_metadata("sample-0", 4), _metadata("sample-1", 9)]
    _write_shard(path, metadata)

    shard = load_unlabeled_validation_shard(path, expected_center="0")

    assert shard.embeddings.shape == (2, 3840)
    assert shard.canonical_a_embeddings.shape == (2, 2560)
    assert shard.sample_ids == ("sample-0", "sample-1")
    assert shard.case_ids == ("case-sample-0", "case-sample-1")
    assert shard.manifest_row_indices == (4, 9)
    assert all(set(row) == {"sample_id", "case_id", "split", "center", "manifest_row_index"} for row in shard.metadata)


def test_consumption_lock_keeps_metrics_and_prior_splits_closed() -> None:
    config = load_routing_validation_cache_config(CONFIG)

    lock = _consumption_lock(config)

    assert lock["training_split"]["status"] == "CONSUMED_BY_STAGE20_SOURCE_INNER_EVIDENCE"
    assert lock["test_split"]["status"] == "CONSUMED_FOR_UNIFORM_B_REPRESENTATION_ADOPTION"
    assert lock["validation_split"]["labels_unobserved_before_lock"] is True
    assert lock["validation_split"]["labels_used_for_feature_extraction"] is False
    assert lock["validation_split"]["labels_persisted_in_cache"] is False
    assert lock["validation_split"]["labels_may_be_joined_after_predictions_for_scoring_only"] is True
    assert lock["cache_claim_boundary"]["output_metric_computed"] is False
    assert lock["cache_claim_boundary"]["utility_computed"] is False
    assert lock["cache_claim_boundary"]["routing_performed"] is False


def test_validator_fails_closed_on_incomplete_bundle(tmp_path: Path) -> None:
    root = tmp_path / "incomplete"
    root.mkdir()

    with pytest.raises(RoutingValidationCacheError, match="incomplete or unsafe"):
        validate_uniform_b_routing_validation_cache(root)


def _row(
    sample_id: str,
    case_id: str,
    split: str,
    index: int,
) -> LabelBlindManifestRow:
    return LabelBlindManifestRow(
        sample_id=sample_id,
        case_id=case_id,
        image_path=f"patches/{sample_id}.jpg",
        split=split,
        center="0",
        manifest_row_index=index,
    )


def _metadata(sample_id: str, manifest_row_index: int) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "case_id": f"case-{sample_id}",
        "split": "val",
        "center": "0",
        "manifest_row_index": manifest_row_index,
    }


def _write_shard(path: Path, metadata: list[dict[str, object]]) -> None:
    torch.save(
        {
            "embeddings": torch.zeros((len(metadata), 3840), dtype=torch.float32),
            "canonical_a_embeddings": torch.zeros(
                (len(metadata), 2560), dtype=torch.float32
            ),
            "metadata": metadata,
            "feature_extractor": {},
        },
        path,
    )
