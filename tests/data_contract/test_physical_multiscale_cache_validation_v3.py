from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import pytest

from midogpp_thesis.data.physical_multiscale.cache_validation_v3 import (
    _center_grouped_sample_ids,
    _validate_child_input_decoder,
    _validate_content_index,
    _validate_parent_decoders,
    _validate_pooling_audit,
)
from midogpp_thesis.data.physical_multiscale import cache_builder_v3
from midogpp_thesis.data.physical_multiscale import cache_validation_v3


ANCHOR_POLICY = "continuous_half_open_bbox_image_intersection_centroid_v1"


def test_v3_cache_validator_uses_center_grouped_shard_order() -> None:
    contract_rows = [
        {"sample_id": "center0-first", "center": "0"},
        {"sample_id": "center1-first", "center": "1"},
        {"sample_id": "center0-second", "center": "0"},
        {"sample_id": "center9-first", "center": "9"},
        {"sample_id": "center1-second", "center": "1"},
    ]

    assert _center_grouped_sample_ids(contract_rows) == (
        "center0-first",
        "center0-second",
        "center1-first",
        "center1-second",
        "center9-first",
    )


def test_v3_pooling_audit_recomputes_anchor_relative_token_starts() -> None:
    geometries = {
        key: {
            "p_x": 0.1,
            "p_y": 0.5,
            "token_start_row": 6,
            "token_start_col": 0,
            "shift_x": 3,
            "shift_y": 0,
        }
        for key in ("28um", "56um", "112um")
    }
    contract_rows = [
        {
            "row_index": "0",
            "sample_id": "s1",
            "center": "0",
            "label": "1",
            "policy_id": ANCHOR_POLICY,
            "anchor_x": "11.5",
            "anchor_y": "2416.0",
            "scale_geometry_json": json.dumps(geometries),
        }
    ]
    pooling_rows = [
        {
            "sample_id": "s1",
            "contract_row_index": "0",
            "center": "0",
            "label": "1",
            "fov_um": fov,
            "annotation_anchor_policy_id": ANCHOR_POLICY,
            "anchor_x": "11.5",
            "anchor_y": "2416.0",
            "p_x": "0.1",
            "p_y": "0.5",
            "token_start_row": "6",
            "token_start_col": "0",
            "shift_x": "3",
            "shift_y": "0",
        }
        for fov in ("28.0", "56.0", "112.0")
    ]

    index = _validate_pooling_audit(
        pooling_rows,
        contract_rows,
        anchor_policy_id=ANCHOR_POLICY,
    )
    assert set(index) == {("s1", "28"), ("s1", "56"), ("s1", "112")}

    pooling_rows[0]["token_start_col"] = "1"
    with pytest.raises(ValueError, match="pooling recomputation drift"):
        _validate_pooling_audit(
            pooling_rows,
            contract_rows,
            anchor_policy_id=ANCHOR_POLICY,
        )


def test_v3_content_index_rejects_byte_tampering(tmp_path: Path) -> None:
    payload_path = tmp_path / "reports" / "cache_bundle_report.json"
    payload_path.parent.mkdir(parents=True)
    payload_path.write_text('{"status":"PASS"}\n', encoding="utf-8")
    relative = str(payload_path.relative_to(tmp_path))
    content_index = {
        "schema_version": "midogpp_physical_multiscale_content_index_v3",
        "status": "PASS",
        "annotation_anchor_policy_id": ANCHOR_POLICY,
        "physical_contract_hash": "contract-hash",
        "files": {relative: _sha256(payload_path)},
    }

    _validate_content_index(
        tmp_path,
        content_index,
        anchor_policy_id=ANCHOR_POLICY,
        contract_hash="contract-hash",
    )

    payload_path.write_text('{"status":"FAIL"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="differs from bundle bytes"):
        _validate_content_index(
            tmp_path,
            content_index,
            anchor_policy_id=ANCHOR_POLICY,
            contract_hash="contract-hash",
        )


@pytest.mark.parametrize(
    "documents",
    (
        (
            {
                "annotation_jpeg_decoder": "pillow",
                "raw_tiff_slide_reader_backend": "openslide",
            },
        ),
        (
            {
                "annotation_jpeg_decoder": "pillow",
                "raw_tiff_slide_reader_backend": "pyvips",
            },
            {
                "annotation_jpeg_decoder": "opencv",
                "raw_tiff_slide_reader_backend": "pyvips",
            },
        ),
        (
            {
                "annotation_jpeg_decoder": "pillow",
                "raw_tiff_slide_reader_backend": "pyvips",
            },
            {},
        ),
    ),
)
def test_v3_cache_validator_rejects_parent_decoder_drift(
    documents: tuple[dict[str, object], ...],
) -> None:
    with pytest.raises(ValueError, match="parent decoder lineage drifted"):
        _validate_parent_decoders(
            *documents,
            expected_raw_tiff_backend="pyvips",
        )


@pytest.mark.parametrize(
    ("representation_id", "input_decoder"),
    (
        ("annotation_jpeg_fixed_center_b_v3", "pyvips_raw_tiff"),
        (
            "physical_multiscale_clipped_bbox_annotation_local_c_v3",
            "pillow_jpeg",
        ),
    ),
)
def test_v3_cache_validator_rejects_child_decoder_drift(
    representation_id: str,
    input_decoder: str,
) -> None:
    with pytest.raises(ValueError, match="child input decoder drifted"):
        _validate_child_input_decoder(
            {"input_decoder": input_decoder},
            representation_id=representation_id,
        )


def test_v3_cache_builder_creates_only_the_required_publication_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass(frozen=True)
    class FakeConfig:
        contract_root: Path
        cache_bundle_root: Path
        canonical_cache_path: Path
        canonical_reference_root: Path

    final_root = tmp_path / "pilot_v3" / "seed42"
    config = FakeConfig(
        contract_root=tmp_path / "contract",
        cache_bundle_root=final_root,
        canonical_cache_path=tmp_path / "canonical.pt",
        canonical_reference_root=tmp_path / "reference",
    )

    monkeypatch.setattr(
        cache_builder_v3,
        "validate_contract_bundle_v3",
        lambda *_args, **_kwargs: {"status": "PASS"},
    )

    def build(staging: FakeConfig) -> None:
        assert staging.cache_bundle_root.is_dir()
        assert not tuple(staging.cache_bundle_root.iterdir())
        (staging.cache_bundle_root / "validated.txt").write_text(
            "ok",
            encoding="utf-8",
        )

    def validate(
        stage: Path,
        **_kwargs: object,
    ) -> dict[str, str]:
        assert (stage / "validated.txt").read_text(encoding="utf-8") == "ok"
        return {"status": "PASS"}

    monkeypatch.setattr(
        cache_builder_v3,
        "_build_cache_bundle_v3_in_place",
        build,
    )
    monkeypatch.setattr(cache_validation_v3, "validate_cache_bundle_v3", validate)

    assert cache_builder_v3.build_physical_multiscale_caches_v3(config) == final_root  # type: ignore[arg-type]
    assert (final_root / "validated.txt").read_text(encoding="utf-8") == "ok"
    assert final_root.parent.is_dir()
    assert not final_root.with_name(".seed42.staging").exists()


def test_v3_cache_builder_validates_staged_bytes_against_final_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass(frozen=True)
    class FakeConfig:
        contract_root: Path
        cache_bundle_root: Path
        canonical_cache_path: Path
        canonical_reference_root: Path

        @property
        def b_cache_root(self) -> Path:
            return self.cache_bundle_root / "b_3840"

        @property
        def c_cache_root(self) -> Path:
            return self.cache_bundle_root / "c_11520"

    class ReachedRealValidatorBoundary(RuntimeError):
        pass

    final_root = tmp_path / "pilot_v3" / "seed42"
    config = FakeConfig(
        contract_root=tmp_path / "contract",
        cache_bundle_root=final_root,
        canonical_cache_path=tmp_path / "canonical.pt",
        canonical_reference_root=tmp_path / "reference",
    )
    monkeypatch.setattr(
        cache_builder_v3,
        "validate_contract_bundle_v3",
        lambda *_args, **_kwargs: {"status": "PASS"},
    )

    def build(staging: FakeConfig) -> None:
        for relative in (
            "manifests/bundle_manifest.json",
            "manifests/pooling_audit.csv",
            "manifests/content_index.json",
            "reports/cache_bundle_report.json",
        ):
            path = staging.cache_bundle_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")

    def contract_boundary(
        _root: Path,
        *,
        verify_raw_files: bool,
        expected_config: object,
    ) -> dict[str, object]:
        assert verify_raw_files is False
        assert expected_config is config
        assert expected_config.cache_bundle_root == final_root  # type: ignore[attr-defined]
        raise ReachedRealValidatorBoundary

    monkeypatch.setattr(
        cache_builder_v3,
        "_build_cache_bundle_v3_in_place",
        build,
    )
    monkeypatch.setattr(
        cache_validation_v3,
        "validate_contract_bundle_v3",
        contract_boundary,
    )

    with pytest.raises(ReachedRealValidatorBoundary):
        cache_builder_v3.build_physical_multiscale_caches_v3(config)  # type: ignore[arg-type]

    assert not final_root.exists()
    assert not final_root.with_name(".seed42.staging").exists()
    assert len(tuple(final_root.parent.glob(".seed42.quarantine-*"))) == 1


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
