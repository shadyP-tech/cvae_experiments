from __future__ import annotations

import pytest

from midogpp_thesis.data.physical_multiscale.geometry import (
    physical_crop_geometry,
    round_half_up,
)


def test_physical_crop_geometry_uses_frozen_half_up_square_policy() -> None:
    geometry = physical_crop_geometry(
        center_x=56.0,
        center_y=56.0,
        fov_um=28.0,
        mpp_x=0.25,
        mpp_y=0.25,
        image_width=500,
        image_height=400,
    )

    assert round_half_up(112.5) == 113
    assert geometry.side_px == 112
    assert (geometry.x0, geometry.y0, geometry.x1, geometry.y1) == (0, 0, 112, 112)
    assert geometry.padding_fraction == 0.0


def test_physical_crop_geometry_records_boundary_padding_fraction() -> None:
    geometry = physical_crop_geometry(
        center_x=50.0,
        center_y=56.0,
        fov_um=28.0,
        mpp_x=0.25,
        mpp_y=0.25,
        image_width=500,
        image_height=400,
    )

    assert geometry.pad_left == 6
    assert geometry.pad_top == 0
    assert geometry.padding_fraction == pytest.approx(6 / 112)

