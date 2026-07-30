from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest

from midogpp_thesis.data.physical_multiscale.geometry import (
    CLIPPED_BBOX_ANCHOR_POLICY_ID,
    clipped_annotation_bbox_anchor,
)


def _anchor(**overrides: object):
    values: dict[str, object] = {
        "bbox_x": 10.0,
        "bbox_y": 20.0,
        "bbox_w": 50.0,
        "bbox_h": 50.0,
        "image_width": 100,
        "image_height": 100,
        "minimum_clipped_area_fraction": 0.25,
    }
    values.update(overrides)
    return clipped_annotation_bbox_anchor(**values)  # type: ignore[arg-type]


def test_fully_visible_bbox_is_byte_stable_at_the_canonical_centroid() -> None:
    result = _anchor()

    assert result.policy_id == CLIPPED_BBOX_ANCHOR_POLICY_ID
    assert result.anchor_x == 35.0
    assert result.anchor_y == 45.0
    assert result.original_centroid_x == result.anchor_x
    assert result.original_centroid_y == result.anchor_y
    assert result.clipped_area_fraction == 1.0
    assert result.anchor_delta_x == 0.0
    assert result.anchor_delta_y == 0.0
    assert result.was_clipped is False
    with pytest.raises(FrozenInstanceError):
        result.anchor_x = 0.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("overrides", "expected_anchor", "expected_fraction"),
    (
        ({"bbox_x": -20.0}, (15.0, 45.0), 0.6),
        ({"bbox_y": -20.0}, (35.0, 15.0), 0.6),
        ({"bbox_x": 80.0}, (90.0, 45.0), 0.4),
        ({"bbox_y": 80.0}, (35.0, 90.0), 0.4),
        ({"bbox_x": -20.0, "bbox_y": -20.0}, (15.0, 15.0), 0.36),
    ),
)
def test_edge_and_corner_clipping_use_continuous_half_open_intersection(
    overrides: dict[str, object],
    expected_anchor: tuple[float, float],
    expected_fraction: float,
) -> None:
    result = _anchor(**overrides)

    assert (result.anchor_x, result.anchor_y) == expected_anchor
    assert result.clipped_area_fraction == pytest.approx(expected_fraction)
    assert result.was_clipped is True


def test_original_centroid_may_be_outside_when_positive_support_remains() -> None:
    result = _anchor(bbox_x=-27.0, bbox_y=2391.0, image_width=6447, image_height=4835)

    assert result.original_centroid_x == -2.0
    assert result.anchor_x == 11.5
    assert result.anchor_y == 2416.0
    assert result.clipped_area_fraction == 0.46
    assert result.anchor_delta_x == 13.5
    assert result.anchor_delta_y == 0.0


def test_exact_xai_regression_anchor_for_sample_309() -> None:
    result = _anchor(
        bbox_x=4186.0,
        bbox_y=-30.0,
        image_width=6447,
        image_height=4835,
    )

    assert result.original_centroid_y == -5.0
    assert (result.anchor_x, result.anchor_y) == (4211.0, 10.0)
    assert result.clipped_area_fraction == 0.4
    assert result.anchor_delta_y == 15.0


@pytest.mark.parametrize(
    "overrides",
    (
        {"bbox_w": 0.0},
        {"bbox_h": -1.0},
        {"bbox_x": math.nan},
        {"bbox_y": math.inf},
        {"bbox_w": math.inf},
        {"bbox_x": -60.0},
        {"bbox_y": 100.0},
        {"minimum_clipped_area_fraction": 0.0},
        {"minimum_clipped_area_fraction": 1.01},
    ),
)
def test_invalid_or_empty_bbox_fails_closed(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _anchor(**overrides)


def test_tiny_clipped_sliver_fails_the_frozen_qc_threshold() -> None:
    with pytest.raises(ValueError, match="frozen minimum area fraction"):
        _anchor(bbox_x=-40.0)

