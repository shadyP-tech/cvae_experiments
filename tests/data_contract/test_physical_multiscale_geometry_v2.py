from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict
import math

import pytest

from midogpp_thesis.data.physical_multiscale.geometry import (
    CropGeometryV2,
    physical_crop_geometry_v2,
)


def _geometry(
    *,
    anchor_x: float = 10.25,
    anchor_y: float = 7.75,
    side: float = 5.0,
    width: int = 20,
    height: int = 15,
) -> CropGeometryV2:
    return physical_crop_geometry_v2(
        anchor_x=anchor_x,
        anchor_y=anchor_y,
        fov_um=side,
        mpp_x=1.0,
        mpp_y=1.0,
        image_width=width,
        image_height=height,
    )


def test_v2_center_crop_preserves_requested_square_and_anchor_position() -> None:
    geometry = _geometry()

    assert geometry.side_px == 5
    assert (
        geometry.requested_x0,
        geometry.requested_y0,
        geometry.requested_x1,
        geometry.requested_y1,
    ) == (7, 5, 12, 10)
    assert (
        geometry.realized_x0,
        geometry.realized_y0,
        geometry.realized_x1,
        geometry.realized_y1,
    ) == (7, 5, 12, 10)
    assert (geometry.shift_x, geometry.shift_y) == (0, 0)
    assert geometry.p_x == pytest.approx((10.25 - 7) / 5)
    assert geometry.p_y == pytest.approx((7.75 - 5) / 5)
    assert 0.0 <= geometry.p_x < 1.0
    assert 0.0 <= geometry.p_y < 1.0
    assert (
        geometry.pad_left,
        geometry.pad_top,
        geometry.pad_right,
        geometry.pad_bottom,
        geometry.padding_fraction,
    ) == (0, 0, 0, 0, 0.0)


def test_v2_side_rounding_is_half_up() -> None:
    geometry = _geometry(side=4.5)

    assert geometry.side_px == 5


@pytest.mark.parametrize(
    ("anchor_x", "anchor_y", "expected_start", "expected_shift"),
    [
        (0.0, 0.0, (0, 0), (2, 2)),
        (0.0, 14.999, (0, 11), (2, -1)),
        (19.999, 0.0, (16, 0), (-1, 2)),
        (19.999, 14.999, (16, 11), (-1, -1)),
    ],
)
def test_v2_boundaries_and_corners_clamp_to_exact_in_bounds_square(
    anchor_x: float,
    anchor_y: float,
    expected_start: tuple[int, int],
    expected_shift: tuple[int, int],
) -> None:
    geometry = _geometry(
        anchor_x=anchor_x,
        anchor_y=anchor_y,
        side=4.0,
    )

    assert (geometry.realized_x0, geometry.realized_y0) == expected_start
    assert (geometry.realized_x1, geometry.realized_y1) == (
        expected_start[0] + 4,
        expected_start[1] + 4,
    )
    assert (geometry.shift_x, geometry.shift_y) == expected_shift
    assert 0.0 <= geometry.p_x < 1.0
    assert 0.0 <= geometry.p_y < 1.0


def test_v2_side_equal_to_dimensions_is_valid_without_padding() -> None:
    geometry = _geometry(
        anchor_x=5.999,
        anchor_y=0.0,
        side=6.0,
        width=6,
        height=6,
    )

    assert geometry.side_px == 6
    assert (
        geometry.realized_x0,
        geometry.realized_y0,
        geometry.realized_x1,
        geometry.realized_y1,
    ) == (0, 0, 6, 6)
    assert geometry.padding_fraction == 0.0
    assert geometry.p_x == pytest.approx(5.999 / 6)
    assert geometry.p_y == 0.0


@pytest.mark.parametrize(
    ("side", "width", "height"),
    [
        (7.0, 6, 8),
        (9.0, 10, 8),
    ],
)
def test_v2_rejects_side_larger_than_either_dimension(
    side: float,
    width: int,
    height: int,
) -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        _geometry(
            anchor_x=0.0,
            anchor_y=0.0,
            side=side,
            width=width,
            height=height,
        )


@pytest.mark.parametrize("bad_anchor", [math.nan, math.inf, -math.inf])
def test_v2_rejects_nonfinite_anchors(bad_anchor: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        _geometry(anchor_x=bad_anchor)
    with pytest.raises(ValueError, match="must be finite"):
        _geometry(anchor_y=bad_anchor)


@pytest.mark.parametrize(
    ("anchor_x", "anchor_y"),
    [
        (-0.001, 1.0),
        (20.0, 1.0),
        (1.0, -0.001),
        (1.0, 15.0),
    ],
)
def test_v2_rejects_anchors_outside_half_open_image(
    anchor_x: float,
    anchor_y: float,
) -> None:
    with pytest.raises(ValueError, match="must satisfy"):
        _geometry(anchor_x=anchor_x, anchor_y=anchor_y)


@pytest.mark.parametrize(
    ("anchor", "dimension", "side", "expected"),
    [
        (0.0, 20, 6, 0),
        (3.25, 20, 6, 0),
        (10.0, 20, 6, 7),
        (18.75, 20, 6, 14),
        (19.999, 20, 6, 14),
    ],
)
def test_v2_axis_clamp_is_the_minimal_feasible_displacement(
    anchor: float,
    dimension: int,
    side: int,
    expected: int,
) -> None:
    geometry = _geometry(
        anchor_x=anchor,
        anchor_y=5.0,
        side=float(side),
        width=dimension,
        height=20,
    )
    feasible_starts = range(0, dimension - side + 1)

    assert geometry.realized_x0 == expected
    assert abs(geometry.shift_x) == min(
        abs(start - geometry.requested_x0) for start in feasible_starts
    )


def test_v2_geometry_is_frozen_and_plain_dataclass_serializable() -> None:
    geometry = _geometry()

    assert asdict(geometry)["realized_x0"] == geometry.realized_x0
    with pytest.raises(FrozenInstanceError):
        geometry.shift_x = 1  # type: ignore[misc]
