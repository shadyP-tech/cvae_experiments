from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from midogpp_thesis.data.physical_multiscale.contract_validation import (
    validate_contract_document_v2,
)
from midogpp_thesis.data.physical_multiscale.contract_validation_v3 import (
    _validate_against_canonical_inputs,
    _validate_resolved_config,
    validate_contract_document_v3,
)
from midogpp_thesis.data.physical_multiscale.contract_v3 import (
    _resolved_config_payload,
    build_physical_multiscale_contract_v3,
    claim_firewall_v3,
    geometry_policy_v3,
    physical_contract_hash_v3,
)
from midogpp_thesis.data.physical_multiscale.config_v3 import load_build_config_v3
from midogpp_thesis.data.physical_multiscale import contract_v3
from midogpp_thesis.data.physical_multiscale import contract_validation_v3


def _geometry_policy() -> dict[str, object]:
    return {
        "crop_policy": "deterministic_axiswise_in_bounds_translation",
        "annotation_anchor": "clipped_axis_aligned_bbox_continuous_centroid",
        "annotation_anchor_policy_id": (
            "continuous_half_open_bbox_image_intersection_centroid_v1"
        ),
        "bbox_convention": "continuous_half_open_xyxy",
        "image_intersection": (
            "[max(0,x0),min(W,x1)) x [max(0,y0),min(H,y1))"
        ),
        "minimum_clipped_bbox_area_fraction": 0.25,
        "anchor_delta_definition": (
            "clipped_bbox_centroid_minus_original_bbox_centroid"
        ),
        "coordinate_origin": "level_0_top_left_pixel_edge",
        "pixel_rounding": "round_half_up",
        "out_of_bounds_padding_used": False,
        "realized_padding_pixel_count": 0,
        "crop_size_changed": False,
        "resize_interpolation": "bicubic",
        "resize_antialias": True,
        "output_size_px": 224,
        "token_window_start": "axis_start=clamp(floor(16*p_axis-2),0,12)",
        "token_window_shape": [4, 4],
        "geometry_uses_labels": False,
        "geometry_uses_center_identity": False,
    }


def _document() -> dict[str, object]:
    return {
        "schema_version": (
            "midogpp_physical_multiscale_clipped_bbox_annotation_local_contract_v3"
        ),
        "status": "PASS",
        "profile_id": (
            "physical_multiscale_clipped_bbox_annotation_local_pooling_pilot_v3"
        ),
        "contract_hash": "0123456789abcdef",
        "artifact_dataset": "MIDOG++",
        "claim_dataset": "MIDOG++",
        "split": "train",
        "eligible_centers": ["0", "1", "2", "3", "5", "6", "7", "8", "9"],
        "excluded_centers": ["4"],
        "canonical_cache_sha256": "a" * 64,
        "row_count": 9648,
        "slide_count": 216,
        "fov_um": [28.0, 56.0, 112.0],
        "geometry_policy": _geometry_policy(),
        "identity_hashes": {
            "sample_id_order": "1" * 16,
            "case_id_order": "2" * 16,
            "slide_id_order": "3" * 16,
            "center_order": "4" * 16,
            "label_order": "5" * 16,
            "partition_identity": "6" * 16,
        },
        "target_labels_used_for_extraction": False,
        "claim_firewall": claim_firewall_v3(),
    }


def test_v3_schema_is_distinct_from_the_audit_blocked_v2_lineage() -> None:
    document = _document()
    validate_contract_document_v3(document)

    with pytest.raises(ValueError, match="v2 contract schema failed"):
        validate_contract_document_v2(document)

    drifted = deepcopy(document)
    drifted["geometry_policy"]["minimum_clipped_bbox_area_fraction"] = 0.2  # type: ignore[index]
    with pytest.raises(ValueError, match="v3 contract schema failed"):
        validate_contract_document_v3(drifted)


@pytest.mark.parametrize(
    ("field", "mutated"),
    (
        ("bbox_x", -26.0),
        ("clipped_x0", 1.0),
        ("anchor_x", 12.5),
        ("clipped_area_fraction", 0.48),
        ("anchor_delta_x", 14.5),
        (
            "scale_geometry_json",
            (
                '{"28um":{"p_x":0.2,"token_start_col":1,'
                '"token_start_row":6}}'
            ),
        ),
    ),
)
def test_v3_contract_hash_binds_bbox_anchor_crop_and_token_fields(
    field: str,
    mutated: object,
) -> None:
    identities = _document()["identity_hashes"]
    rows = [
        {
            "sample_id": "305__305__ann15363__y1",
            "bbox_x": -27.0,
            "bbox_y": 2391.0,
            "bbox_w": 50.0,
            "bbox_h": 50.0,
            "clipped_x0": 0.0,
            "clipped_y0": 2391.0,
            "clipped_x1": 23.0,
            "clipped_y1": 2441.0,
            "anchor_x": 11.5,
            "anchor_y": 2416.0,
            "clipped_area_fraction": 0.46,
            "anchor_delta_x": 13.5,
            "anchor_delta_y": 0.0,
            "scale_geometry_json": (
                '{"28um":{"p_x":0.1,"token_start_col":0,'
                '"token_start_row":6}}'
            ),
        }
    ]
    original = physical_contract_hash_v3(
        canonical_cache_sha256="a" * 64,
        manifest_rows=rows,
        fov_um=(28.0, 56.0, 112.0),
        geometry_policy=_geometry_policy(),
        identity_hashes=identities,  # type: ignore[arg-type]
    )
    changed = deepcopy(rows)
    changed[0][field] = mutated
    altered = physical_contract_hash_v3(
        canonical_cache_sha256="a" * 64,
        manifest_rows=changed,
        fov_um=(28.0, 56.0, 112.0),
        geometry_policy=_geometry_policy(),
        identity_hashes=identities,  # type: ignore[arg-type]
    )

    assert len(original) == 16
    assert original != altered


def test_v3_contract_hash_binds_anchor_policy() -> None:
    identities = _document()["identity_hashes"]
    rows = [{"sample_id": "s1", "bbox_x": -27.0, "anchor_x": 11.5}]
    original = physical_contract_hash_v3(
        canonical_cache_sha256="a" * 64,
        manifest_rows=rows,
        fov_um=(28.0, 56.0, 112.0),
        geometry_policy=_geometry_policy(),
        identity_hashes=identities,  # type: ignore[arg-type]
    )
    changed = _geometry_policy()
    changed["annotation_anchor_policy_id"] = "mutated"

    assert original != physical_contract_hash_v3(
        canonical_cache_sha256="a" * 64,
        manifest_rows=rows,
        fov_um=(28.0, 56.0, 112.0),
        geometry_policy=changed,
        identity_hashes=identities,  # type: ignore[arg-type]
    )


def test_v3_contract_builder_uses_the_existing_staging_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_root = tmp_path / "contract_v3"
    config = SimpleNamespace(contract_root=final_root)

    def build_in_place(_config: object, stage: Path) -> None:
        assert stage.is_dir()
        assert not tuple(stage.iterdir())
        (stage / "validated.txt").write_text("ok", encoding="utf-8")

    def validate(
        stage: Path,
        *,
        verify_raw_files: bool,
        expected_config: object,
    ) -> dict[str, str]:
        assert verify_raw_files is True
        assert expected_config is config
        assert (stage / "validated.txt").read_text(encoding="utf-8") == "ok"
        return {"status": "PASS"}

    monkeypatch.setattr(contract_v3, "_build_contract_v3_in_place", build_in_place)
    monkeypatch.setattr(contract_validation_v3, "validate_contract_bundle_v3", validate)

    assert build_physical_multiscale_contract_v3(config) == final_root  # type: ignore[arg-type]
    assert (final_root / "validated.txt").read_text(encoding="utf-8") == "ok"
    assert not final_root.with_name(".contract_v3.staging").exists()


def test_v3_validator_parses_resolved_config_and_rejects_model_drift() -> None:
    config = load_build_config_v3(
        "datasets/midogpp/configs/"
        "physical_multiscale_clipped_bbox_annotation_local_pooling_pilot_v3.yaml",
        require_inputs=False,
    )
    payload = dict(_resolved_config_payload(config))
    policy = geometry_policy_v3(config)

    _validate_resolved_config(
        payload,
        geometry_policy=policy,
        expected_config=config,
    )
    drifted = deepcopy(payload)
    drifted["model"]["model_revision"] = "0" * 40  # type: ignore[index]
    with pytest.raises(ValueError, match="section drifted: model"):
        _validate_resolved_config(
            drifted,
            geometry_policy=policy,
            expected_config=config,
        )


@pytest.mark.parametrize(
    ("section", "key", "value"),
    (
        ("runtime_identity", "pyvips", "3.0.0"),
        ("runtime_identity", "libvips", "8.17.0"),
        ("run", "required_slide_reader_backend", "openslide"),
    ),
)
def test_v3_validator_rejects_tiff_backend_identity_drift(
    section: str,
    key: str,
    value: str,
) -> None:
    config = load_build_config_v3(
        "datasets/midogpp/configs/"
        "physical_multiscale_clipped_bbox_annotation_local_pooling_pilot_v3.yaml",
        require_inputs=False,
    )
    payload = deepcopy(_resolved_config_payload(config))
    payload[section][key] = value  # type: ignore[index]

    with pytest.raises(ValueError, match=f"section drifted: {section}"):
        _validate_resolved_config(
            payload,
            geometry_policy=geometry_policy_v3(config),
            expected_config=config,
        )


@pytest.mark.parametrize("backend", ("", "openslide"))
def test_v3_config_requires_exact_pyvips_backend(
    tmp_path: Path,
    backend: str,
) -> None:
    source = Path(
        "datasets/midogpp/configs/"
        "physical_multiscale_clipped_bbox_annotation_local_pooling_pilot_v3.yaml"
    )
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if backend:
        payload["run"]["required_slide_reader_backend"] = backend
    else:
        del payload["run"]["required_slide_reader_backend"]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="exact pyvips slide backend"):
        load_build_config_v3(path, require_inputs=False)


def test_v3_validator_rejects_coherently_rehashed_wrong_canonical_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            "sample_id": "s1",
            "bbox_x": "1",
            "bbox_y": "2",
            "bbox_w": "3",
            "bbox_h": "4",
        }
    ]
    inputs = SimpleNamespace(
        canonical=SimpleNamespace(cache_sha256="actual"),
        selected_metadata=({"sample_id": "s1"},),
        manifest_by_id={
            "s1": {"bbox_x": "1", "bbox_y": "2", "bbox_w": "3", "bbox_h": "4"}
        },
    )
    monkeypatch.setattr(
        contract_validation_v3,
        "load_contract_inputs",
        lambda _config: inputs,
    )

    with pytest.raises(ValueError, match="configured cache bytes"):
        _validate_against_canonical_inputs(
            rows,
            SimpleNamespace(),
            declared_cache_sha256="coherently-edited-wrong-hash",
        )
