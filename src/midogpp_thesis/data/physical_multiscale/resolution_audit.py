"""Fail-closed TIFF/OME micrometers-per-pixel audit."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class MppAudit:
    mpp_x: float
    mpp_y: float
    source: str
    ome_mpp_x: float | None
    ome_mpp_y: float | None
    tiff_mpp_x: float | None
    tiff_mpp_y: float | None
    relative_anisotropy: float
    dual_source_relative_delta: float | None
    width: int
    height: int
    orientation: int


def audit_tiff_mpp(
    path: str | Path,
    *,
    mpp_min: float,
    mpp_max: float,
    anisotropy_relative_max: float,
    dual_source_relative_max: float,
) -> MppAudit:
    try:
        import tifffile  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - workstation dependency
        raise RuntimeError("MPP audit requires tifffile.") from exc

    slide = Path(path)
    with tifffile.TiffFile(slide) as tif:
        page = tif.pages[0]
        orientation = int(_tag_value(page, "Orientation", 1))
        if orientation != 1:
            raise ValueError(f"Only top-left TIFF orientation is accepted: {slide}")
        width = int(page.imagewidth)
        height = int(page.imagelength)
        ome = parse_ome_mpp(getattr(tif, "ome_metadata", None))
        tiff = parse_tiff_resolution(
            _tag_value(page, "XResolution", None),
            _tag_value(page, "YResolution", None),
            _tag_value(page, "ResolutionUnit", None),
        )
    return resolve_mpp(
        ome=ome,
        tiff=tiff,
        mpp_min=mpp_min,
        mpp_max=mpp_max,
        anisotropy_relative_max=anisotropy_relative_max,
        dual_source_relative_max=dual_source_relative_max,
        width=width,
        height=height,
        orientation=orientation,
    )


def parse_ome_mpp(metadata: str | None) -> tuple[float, float] | None:
    if not metadata:
        return None
    root = ET.fromstring(metadata)
    pixels = next((element for element in root.iter() if element.tag.endswith("Pixels")), None)
    if pixels is None:
        return None
    raw_x = pixels.attrib.get("PhysicalSizeX")
    raw_y = pixels.attrib.get("PhysicalSizeY")
    if raw_x in (None, "") or raw_y in (None, ""):
        return None
    unit_x = pixels.attrib.get("PhysicalSizeXUnit", "µm")
    unit_y = pixels.attrib.get("PhysicalSizeYUnit", "µm")
    return (
        _to_micrometers(float(raw_x), unit_x),
        _to_micrometers(float(raw_y), unit_y),
    )


def parse_tiff_resolution(
    raw_x: object,
    raw_y: object,
    raw_unit: object,
) -> tuple[float, float] | None:
    if raw_x is None or raw_y is None or raw_unit is None:
        return None
    unit = _resolution_unit(raw_unit)
    if unit == "inch":
        micrometers = 25_400.0
    elif unit == "centimeter":
        micrometers = 10_000.0
    else:
        return None
    pixels_x = _rational_float(raw_x)
    pixels_y = _rational_float(raw_y)
    if pixels_x <= 0.0 or pixels_y <= 0.0:
        return None
    return micrometers / pixels_x, micrometers / pixels_y


def resolve_mpp(
    *,
    ome: tuple[float, float] | None,
    tiff: tuple[float, float] | None,
    mpp_min: float,
    mpp_max: float,
    anisotropy_relative_max: float,
    dual_source_relative_max: float,
    width: int,
    height: int,
    orientation: int,
) -> MppAudit:
    if ome is None and tiff is None:
        raise ValueError("Slide lacks explicit-unit OME or TIFF resolution.")
    chosen = ome if ome is not None else tiff
    assert chosen is not None
    source = "ome" if ome is not None else "tiff"
    dual_delta = None
    if ome is not None and tiff is not None:
        dual_delta = max(
            _relative_delta(ome[0], tiff[0]),
            _relative_delta(ome[1], tiff[1]),
        )
        if dual_delta > dual_source_relative_max:
            raise ValueError(
                f"OME/TIFF MPP disagreement exceeds tolerance: {dual_delta:.6f}"
            )
        source = "ome_and_tiff"
    mpp_x, mpp_y = (float(chosen[0]), float(chosen[1]))
    if not all(math.isfinite(value) and mpp_min <= value <= mpp_max for value in chosen):
        raise ValueError(f"MPP is missing, non-finite, or outside [{mpp_min}, {mpp_max}].")
    anisotropy = _relative_delta(mpp_x, mpp_y)
    if anisotropy > anisotropy_relative_max:
        raise ValueError(f"MPP anisotropy exceeds tolerance: {anisotropy:.6f}")
    return MppAudit(
        mpp_x=mpp_x,
        mpp_y=mpp_y,
        source=source,
        ome_mpp_x=None if ome is None else float(ome[0]),
        ome_mpp_y=None if ome is None else float(ome[1]),
        tiff_mpp_x=None if tiff is None else float(tiff[0]),
        tiff_mpp_y=None if tiff is None else float(tiff[1]),
        relative_anisotropy=anisotropy,
        dual_source_relative_delta=dual_delta,
        width=int(width),
        height=int(height),
        orientation=int(orientation),
    )


def _tag_value(page: object, name: str, default: object) -> object:
    tags = getattr(page, "tags", {})
    tag = tags.get(name) if hasattr(tags, "get") else None
    return default if tag is None else getattr(tag, "value", default)


def _resolution_unit(value: object) -> str:
    rendered = str(getattr(value, "name", value)).strip().lower()
    if rendered in {"2", "inch", "inches", "resolutionunit.inch"}:
        return "inch"
    if rendered in {"3", "centimeter", "centimetre", "cm", "resolutionunit.centimeter"}:
        return "centimeter"
    return "none"


def _rational_float(value: object) -> float:
    if isinstance(value, tuple) and len(value) == 2:
        return float(value[0]) / float(value[1])
    return float(value)


def _relative_delta(left: float, right: float) -> float:
    denominator = (abs(float(left)) + abs(float(right))) / 2.0
    return math.inf if denominator == 0.0 else abs(float(left) - float(right)) / denominator


def _to_micrometers(value: float, unit: str) -> float:
    normalized = unit.strip().lower().replace("μ", "µ")
    factors = {
        "µm": 1.0,
        "um": 1.0,
        "micrometer": 1.0,
        "micrometre": 1.0,
        "nm": 0.001,
        "mm": 1000.0,
        "cm": 10_000.0,
        "m": 1_000_000.0,
    }
    try:
        return float(value) * factors[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported OME physical-size unit: {unit!r}") from exc
