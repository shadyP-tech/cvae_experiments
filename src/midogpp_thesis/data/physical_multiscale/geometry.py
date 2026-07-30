"""Deterministic physical-FOV crop geometry."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CropGeometry:
    fov_um: float
    mpp_x: float
    mpp_y: float
    side_px: int
    x0: int
    y0: int
    x1: int
    y1: int
    source_x0: int
    source_y0: int
    source_x1: int
    source_y1: int
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int
    padding_fraction: float


@dataclass(frozen=True)
class CropGeometryV2:
    """Exact in-bounds square geometry for the v2 physical crop policy."""

    fov_um: float
    mpp_x: float
    mpp_y: float
    anchor_x: float
    anchor_y: float
    side_px: int
    requested_x0: int
    requested_y0: int
    requested_x1: int
    requested_y1: int
    realized_x0: int
    realized_y0: int
    realized_x1: int
    realized_y1: int
    shift_x: int
    shift_y: int
    p_x: float
    p_y: float
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int
    padding_fraction: float


CLIPPED_BBOX_ANCHOR_POLICY_ID = (
    "continuous_half_open_bbox_image_intersection_centroid_v1"
)


@dataclass(frozen=True)
class ClippedAnnotationBBoxAnchor:
    """Continuous clipped-bbox anchor used by the immutable v3 contract."""

    policy_id: str
    bbox_x0: float
    bbox_y0: float
    bbox_x1: float
    bbox_y1: float
    original_area: float
    original_centroid_x: float
    original_centroid_y: float
    clipped_x0: float
    clipped_y0: float
    clipped_x1: float
    clipped_y1: float
    clipped_area: float
    clipped_area_fraction: float
    anchor_x: float
    anchor_y: float
    anchor_delta_x: float
    anchor_delta_y: float
    was_clipped: bool


def round_half_up(value: float) -> int:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("Physical crop size must be finite and positive.")
    return int(math.floor(value + 0.5))


def physical_crop_geometry(
    *,
    center_x: float,
    center_y: float,
    fov_um: float,
    mpp_x: float,
    mpp_y: float,
    image_width: int,
    image_height: int,
) -> CropGeometry:
    """Convert a physical square FOV to level-0 pixel bounds."""

    if min(mpp_x, mpp_y, fov_um) <= 0.0:
        raise ValueError("MPP and field of view must be positive.")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Raw image dimensions must be positive.")
    mean_mpp = (float(mpp_x) + float(mpp_y)) / 2.0
    side = round_half_up(float(fov_um) / mean_mpp)
    x0 = int(math.floor(float(center_x) - side / 2.0))
    y0 = int(math.floor(float(center_y) - side / 2.0))
    x1 = x0 + side
    y1 = y0 + side
    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(int(image_width), x1)
    src_y1 = min(int(image_height), y1)
    pad_left = src_x0 - x0
    pad_top = src_y0 - y0
    pad_right = x1 - src_x1
    pad_bottom = y1 - src_y1
    inside_width = max(0, src_x1 - src_x0)
    inside_height = max(0, src_y1 - src_y0)
    inside_area = inside_width * inside_height
    padding_fraction = 1.0 - inside_area / float(side * side)
    return CropGeometry(
        fov_um=float(fov_um),
        mpp_x=float(mpp_x),
        mpp_y=float(mpp_y),
        side_px=side,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        source_x0=src_x0,
        source_y0=src_y0,
        source_x1=src_x1,
        source_y1=src_y1,
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
        padding_fraction=float(padding_fraction),
    )


def physical_crop_geometry_in_bounds(
    *,
    anchor_x: float,
    anchor_y: float,
    fov_um: float,
    mpp_x: float,
    mpp_y: float,
    image_width: int,
    image_height: int,
) -> CropGeometryV2:
    """Realize the nearest exact in-bounds square around a level-0 anchor.

    The requested top-left coordinate is frozen as
    ``floor(anchor - side_px / 2)`` on each axis. When that request crosses an
    image boundary, its start is clamped to the closest feasible start. The
    crop size is never reduced and no padding pixels are synthesized.
    """

    width = _positive_dimension(image_width, name="image_width")
    height = _positive_dimension(image_height, name="image_height")
    anchor_x_value = _finite_anchor(anchor_x, name="anchor_x")
    anchor_y_value = _finite_anchor(anchor_y, name="anchor_y")
    if not 0.0 <= anchor_x_value < width:
        raise ValueError("anchor_x must satisfy 0 <= anchor_x < image_width.")
    if not 0.0 <= anchor_y_value < height:
        raise ValueError("anchor_y must satisfy 0 <= anchor_y < image_height.")

    physical_values = {
        "fov_um": float(fov_um),
        "mpp_x": float(mpp_x),
        "mpp_y": float(mpp_y),
    }
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in physical_values.values()
    ):
        raise ValueError("MPP and field of view must be finite and positive.")
    mean_mpp = (physical_values["mpp_x"] + physical_values["mpp_y"]) / 2.0
    side = round_half_up(physical_values["fov_um"] / mean_mpp)
    if side > width or side > height:
        raise ValueError(
            "Physical crop side must not exceed either raw image dimension."
        )

    requested_x0 = _requested_axis_start(anchor_x_value, side)
    requested_y0 = _requested_axis_start(anchor_y_value, side)
    realized_x0 = _clamp_axis_start(requested_x0, side=side, dimension=width)
    realized_y0 = _clamp_axis_start(requested_y0, side=side, dimension=height)
    p_x = (anchor_x_value - realized_x0) / side
    p_y = (anchor_y_value - realized_y0) / side
    if not (0.0 <= p_x < 1.0 and 0.0 <= p_y < 1.0):
        raise ArithmeticError("Realized anchor position escaped the frozen [0, 1) range.")
    return CropGeometryV2(
        fov_um=physical_values["fov_um"],
        mpp_x=physical_values["mpp_x"],
        mpp_y=physical_values["mpp_y"],
        anchor_x=anchor_x_value,
        anchor_y=anchor_y_value,
        side_px=side,
        requested_x0=requested_x0,
        requested_y0=requested_y0,
        requested_x1=requested_x0 + side,
        requested_y1=requested_y0 + side,
        realized_x0=realized_x0,
        realized_y0=realized_y0,
        realized_x1=realized_x0 + side,
        realized_y1=realized_y0 + side,
        shift_x=realized_x0 - requested_x0,
        shift_y=realized_y0 - requested_y0,
        p_x=p_x,
        p_y=p_y,
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        padding_fraction=0.0,
    )


def physical_crop_geometry_v2(
    *,
    anchor_x: float,
    anchor_y: float,
    fov_um: float,
    mpp_x: float,
    mpp_y: float,
    image_width: int,
    image_height: int,
) -> CropGeometryV2:
    """Compatibility wrapper for the immutable v2 geometry surface."""

    return physical_crop_geometry_in_bounds(
        anchor_x=anchor_x,
        anchor_y=anchor_y,
        fov_um=fov_um,
        mpp_x=mpp_x,
        mpp_y=mpp_y,
        image_width=image_width,
        image_height=image_height,
    )


def clipped_annotation_bbox_anchor(
    *,
    bbox_x: float,
    bbox_y: float,
    bbox_w: float,
    bbox_h: float,
    image_width: int,
    image_height: int,
    minimum_clipped_area_fraction: float,
) -> ClippedAnnotationBBoxAnchor:
    """Return the centroid of a bbox clipped to continuous image bounds.

    The canonical bbox is interpreted as the continuous half-open rectangle
    ``[x0, x1) × [y0, y1)`` and intersected with
    ``[0, image_width) × [0, image_height)``. The returned centroid is a
    geometric bbox centroid; it is not an object-mask or foreground centroid.
    """

    width = _positive_dimension(image_width, name="image_width")
    height = _positive_dimension(image_height, name="image_height")
    values = {
        "bbox_x": _finite_anchor(bbox_x, name="bbox_x"),
        "bbox_y": _finite_anchor(bbox_y, name="bbox_y"),
        "bbox_w": _finite_anchor(bbox_w, name="bbox_w"),
        "bbox_h": _finite_anchor(bbox_h, name="bbox_h"),
        "minimum_clipped_area_fraction": _finite_anchor(
            minimum_clipped_area_fraction,
            name="minimum_clipped_area_fraction",
        ),
    }
    if values["bbox_w"] <= 0.0 or values["bbox_h"] <= 0.0:
        raise ValueError("Canonical bbox width and height must be positive.")
    threshold = values["minimum_clipped_area_fraction"]
    if not 0.0 < threshold <= 1.0:
        raise ValueError(
            "minimum_clipped_area_fraction must satisfy 0 < value <= 1."
        )

    x0 = values["bbox_x"]
    y0 = values["bbox_y"]
    x1 = x0 + values["bbox_w"]
    y1 = y0 + values["bbox_h"]
    if not all(math.isfinite(value) for value in (x1, y1)):
        raise ValueError("Canonical bbox bounds must be finite.")
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Canonical bbox must have positive continuous area.")

    clipped_x0 = max(0.0, x0)
    clipped_y0 = max(0.0, y0)
    clipped_x1 = min(float(width), x1)
    clipped_y1 = min(float(height), y1)
    if clipped_x1 <= clipped_x0 or clipped_y1 <= clipped_y0:
        raise ValueError("Canonical bbox has no positive-area image intersection.")

    original_area = (x1 - x0) * (y1 - y0)
    clipped_area = (clipped_x1 - clipped_x0) * (clipped_y1 - clipped_y0)
    fraction = clipped_area / original_area
    if not 0.0 < fraction <= 1.0:
        raise ArithmeticError("Clipped bbox area fraction escaped (0, 1].")
    if fraction < threshold:
        raise ValueError(
            "Canonical bbox image intersection is below the frozen minimum "
            f"area fraction: required={threshold}, actual={fraction}."
        )

    original_centroid_x = (x0 + x1) / 2.0
    original_centroid_y = (y0 + y1) / 2.0
    anchor_x = (clipped_x0 + clipped_x1) / 2.0
    anchor_y = (clipped_y0 + clipped_y1) / 2.0
    if not 0.0 <= anchor_x < width or not 0.0 <= anchor_y < height:
        raise ArithmeticError("Clipped bbox centroid escaped image bounds.")
    return ClippedAnnotationBBoxAnchor(
        policy_id=CLIPPED_BBOX_ANCHOR_POLICY_ID,
        bbox_x0=x0,
        bbox_y0=y0,
        bbox_x1=x1,
        bbox_y1=y1,
        original_area=original_area,
        original_centroid_x=original_centroid_x,
        original_centroid_y=original_centroid_y,
        clipped_x0=clipped_x0,
        clipped_y0=clipped_y0,
        clipped_x1=clipped_x1,
        clipped_y1=clipped_y1,
        clipped_area=clipped_area,
        clipped_area_fraction=fraction,
        anchor_x=anchor_x,
        anchor_y=anchor_y,
        anchor_delta_x=anchor_x - original_centroid_x,
        anchor_delta_y=anchor_y - original_centroid_y,
        was_clipped=(
            clipped_x0 != x0
            or clipped_y0 != y0
            or clipped_x1 != x1
            or clipped_y1 != y1
        ),
    )


def _positive_dimension(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _finite_anchor(value: float, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _requested_axis_start(anchor: float, side: int) -> int:
    unrounded_start = anchor - side / 2.0
    return int(math.floor(unrounded_start))


def _clamp_axis_start(requested_start: int, *, side: int, dimension: int) -> int:
    return min(max(requested_start, 0), dimension - side)
