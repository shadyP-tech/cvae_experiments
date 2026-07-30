from __future__ import annotations

from copy import deepcopy

import pytest

from midogpp_thesis.data.physical_multiscale.contract_validation import (
    validate_contract_document_v2,
)
from midogpp_thesis.data.physical_multiscale.contract_validation_v1 import (
    validate_contract_document,
)
from midogpp_thesis.data.physical_multiscale.contract_v2 import (
    claim_firewall_v2,
    physical_contract_hash_v2,
)


def _document() -> dict[str, object]:
    return {
        "schema_version": "midogpp_physical_multiscale_annotation_local_contract_v2",
        "status": "PASS",
        "profile_id": "physical_multiscale_annotation_local_pooling_pilot_v2",
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
        "geometry_policy": {
            "crop_policy": "deterministic_axiswise_in_bounds_translation",
            "annotation_anchor": "canonical_xyxy_continuous_center",
            "coordinate_origin": "level_0_top_left_pixel_edge",
            "pixel_rounding": "round_half_up",
            "out_of_bounds_padding_used": False,
            "realized_padding_pixel_count": 0,
            "crop_size_changed": False,
            "resize_interpolation": "bicubic",
            "resize_antialias": True,
            "output_size_px": 224,
            "geometry_uses_labels": False,
            "geometry_uses_center_identity": False,
        },
        "identity_hashes": {
            "sample_id_order": "1" * 16,
            "case_id_order": "2" * 16,
            "slide_id_order": "3" * 16,
            "center_order": "4" * 16,
            "label_order": "5" * 16,
            "partition_identity": "6" * 16,
        },
        "target_labels_used_for_extraction": False,
        "claim_firewall": claim_firewall_v2(),
    }


def test_v2_contract_schema_accepts_only_annotation_local_identity() -> None:
    document = _document()
    validate_contract_document_v2(document)

    with pytest.raises(ValueError, match="contract schema failed"):
        validate_contract_document(document)

    drifted = deepcopy(document)
    drifted["geometry_policy"]["out_of_bounds_padding_used"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="v2 contract schema failed"):
        validate_contract_document_v2(drifted)


def test_v2_contract_hash_binds_requested_and_realized_geometry() -> None:
    policy = _document()["geometry_policy"]
    identities = _document()["identity_hashes"]
    rows = [
        {
            "sample_id": "s1",
            "raw_tiff_sha256": "b" * 64,
            "scale_geometry_json": (
                '{"112um":{"requested_x0":-5,"realized_x0":0,'
                '"shift_x":5,"p_x":0.1}}'
            ),
        }
    ]
    original = physical_contract_hash_v2(
        canonical_cache_sha256="a" * 64,
        manifest_rows=rows,
        fov_um=(28.0, 56.0, 112.0),
        geometry_policy=policy,  # type: ignore[arg-type]
        identity_hashes=identities,  # type: ignore[arg-type]
    )
    mutated_rows = deepcopy(rows)
    mutated_rows[0]["scale_geometry_json"] = (
        '{"112um":{"requested_x0":-5,"realized_x0":1,'
        '"shift_x":6,"p_x":0.09}}'
    )
    mutated = physical_contract_hash_v2(
        canonical_cache_sha256="a" * 64,
        manifest_rows=mutated_rows,
        fov_um=(28.0, 56.0, 112.0),
        geometry_policy=policy,  # type: ignore[arg-type]
        identity_hashes=identities,  # type: ignore[arg-type]
    )

    assert len(original) == 16
    assert original != mutated
