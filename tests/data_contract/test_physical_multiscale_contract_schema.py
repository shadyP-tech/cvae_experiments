from __future__ import annotations

from copy import deepcopy

import pytest

from midogpp_thesis.data.physical_multiscale.contract import (
    physical_contract_hash,
)
from midogpp_thesis.data.physical_multiscale.validation import (
    validate_contract_document,
)


def test_physical_contract_schema_accepts_exact_builder_document() -> None:
    validate_contract_document(_contract())


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing", "required property"),
        ("altered", "was expected"),
        ("extra", "Additional properties"),
    ),
)
def test_physical_contract_schema_rejects_missing_altered_and_extra_fields(
    mutation: str,
    message: str,
) -> None:
    contract = deepcopy(_contract())
    if mutation == "missing":
        contract.pop("canonical_cache_sha256")
    elif mutation == "altered":
        contract["status"] = "TODO"
    else:
        contract["invented_claim"] = True

    with pytest.raises(ValueError, match=message):
        validate_contract_document(contract)


def test_physical_contract_hash_binds_geometry_bearing_manifest_scalars() -> None:
    policy = _contract()["geometry_policy"]
    baseline = physical_contract_hash(
        canonical_cache_sha256="a" * 64,
        manifest_rows=(
            {
                "sample_id": "sample-1",
                "raw_tiff_sha256": "b" * 64,
                "scale_geometry_json": '{"28um":{"side_px":112}}',
            },
        ),
        fov_um=(28.0, 56.0, 112.0),
        geometry_policy=policy,  # type: ignore[arg-type]
    )
    changed = physical_contract_hash(
        canonical_cache_sha256="a" * 64,
        manifest_rows=(
            {
                "sample_id": "sample-1",
                "raw_tiff_sha256": "b" * 64,
                "scale_geometry_json": '{"28um":{"side_px":113}}',
            },
        ),
        fov_um=(28.0, 56.0, 112.0),
        geometry_policy=policy,  # type: ignore[arg-type]
    )

    assert baseline != changed


def _contract() -> dict[str, object]:
    return {
        "schema_version": "midogpp_physical_multiscale_contract_v1",
        "status": "PASS",
        "contract_hash": "1" * 16,
        "artifact_dataset": "MIDOG++",
        "claim_dataset": "MIDOG++",
        "split": "train",
        "eligible_centers": ["0", "1", "2", "3", "5", "6", "7", "8", "9"],
        "excluded_centers": ["4"],
        "canonical_cache_sha256": "a" * 64,
        "row_count": 9,
        "slide_count": 9,
        "fov_um": [28.0, 56.0, 112.0],
        "geometry_policy": {
            "padding_fraction_max": 0.1,
            "padding_rgb": [255, 255, 255],
            "resize_interpolation": "bicubic",
            "resize_antialias": True,
            "coordinate_origin": "level_0_top_left",
            "pixel_rounding": "round_half_up",
        },
        "target_labels_used_for_extraction": False,
    }
