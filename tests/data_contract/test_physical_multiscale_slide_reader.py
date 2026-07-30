from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

from PIL import Image
import pytest

from midogpp_thesis.data.physical_multiscale.geometry import (
    CropGeometryV2,
    physical_crop_geometry_v2,
)
from midogpp_thesis.data.physical_multiscale.slide_reader import (
    SlideReader,
    open_slide,
)


def _force_pillow_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pyvips", None)
    monkeypatch.setitem(sys.modules, "openslide", None)


def _write_test_tiff(path: Path, *, width: int = 8, height: int = 6) -> None:
    image = Image.new("RGB", (width, height))
    image.putdata(
        [
            (x * 20, y * 30, (x + y) * 10)
            for y in range(height)
            for x in range(width)
        ]
    )
    image.save(path, format="TIFF")
    image.close()


def _geometry() -> CropGeometryV2:
    return physical_crop_geometry_v2(
        anchor_x=1.0,
        anchor_y=2.0,
        fov_um=4.0,
        mpp_x=1.0,
        mpp_y=1.0,
        image_width=8,
        image_height=6,
    )


def test_v2_reads_exact_realized_square_without_a_canvas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_pillow_backend(monkeypatch)
    path = tmp_path / "slide.tiff"
    _write_test_tiff(path)
    geometry = _geometry()

    with open_slide(path, require_tiled=False) as slide:
        assert isinstance(slide, SlideReader)
        assert slide.backend == "pillow_test_only"
        assert slide.dimensions == (8, 6)
        monkeypatch.setattr(
            Image,
            "new",
            lambda *_args, **_kwargs: pytest.fail("v2 must not create a canvas"),
        )
        crop = slide.read_exact_square_v2(geometry, output_size=None)

    assert crop.size == (4, 4)
    assert crop.mode == "RGB"
    assert crop.getpixel((0, 0)) == (0, 0, 0)
    assert crop.getpixel((3, 3)) == (60, 90, 60)
    crop.close()


def test_v2_can_resize_the_source_square_directly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_pillow_backend(monkeypatch)
    path = tmp_path / "slide.tiff"
    _write_test_tiff(path)

    with open_slide(path, require_tiled=False) as slide:
        resized = slide.read_exact_square_v2(_geometry(), output_size=7)

    assert resized.size == (7, 7)
    assert resized.mode == "RGB"
    resized.close()


@pytest.mark.parametrize(
    "geometry",
    [
        replace(_geometry(), realized_x0=-1, realized_x1=3),
        replace(_geometry(), realized_x1=5),
        replace(_geometry(), realized_y0=3, realized_y1=7),
        replace(_geometry(), pad_left=1),
        replace(_geometry(), padding_fraction=0.1),
    ],
)
def test_v2_rejects_nonexact_out_of_bounds_or_padded_geometry(
    geometry: CropGeometryV2,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_pillow_backend(monkeypatch)
    path = tmp_path / "slide.tiff"
    _write_test_tiff(path)

    with open_slide(path, require_tiled=False) as slide:
        with pytest.raises(ValueError):
            slide.read_exact_square_v2(geometry, output_size=None)


def test_v1_reader_path_retains_padding_and_fixed_resize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_pillow_backend(monkeypatch)
    path = tmp_path / "slide.tiff"
    _write_test_tiff(path)
    geometry = {
        "side_px": 4,
        "source_x0": 0,
        "source_y0": 0,
        "source_x1": 3,
        "source_y1": 4,
        "pad_left": 1,
        "pad_top": 0,
    }

    with open_slide(path, require_tiled=False) as slide:
        image = slide.read_geometry(geometry, padding_rgb=(255, 255, 255))

    assert image.size == (224, 224)
    assert image.mode == "RGB"
    assert image.getpixel((0, 0)) == (255, 255, 255)
    image.close()


def test_reader_refuses_test_only_backend_when_tiled_reader_is_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_pillow_backend(monkeypatch)
    path = tmp_path / "slide.tiff"
    _write_test_tiff(path)

    with pytest.raises(RuntimeError, match="required for production"):
        open_slide(path, require_tiled=True)


def test_required_pyvips_backend_does_not_fall_back_to_openslide(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenVipsImage:
        @staticmethod
        def new_from_file(*_args: object, **_kwargs: object) -> object:
            raise OSError("injected pyvips failure")

    class OpenSlideMustNotRun:
        def __init__(self, _path: str) -> None:
            pytest.fail("required pyvips must not fall back to OpenSlide")

    monkeypatch.setitem(
        sys.modules,
        "pyvips",
        type("FakePyVips", (), {"Image": BrokenVipsImage}),
    )
    monkeypatch.setitem(
        sys.modules,
        "openslide",
        type("FakeOpenSlide", (), {"OpenSlide": OpenSlideMustNotRun}),
    )
    path = tmp_path / "slide.tiff"
    _write_test_tiff(path)

    with pytest.raises(RuntimeError, match="Required pyvips backend"):
        open_slide(
            path,
            require_tiled=True,
            required_backend="pyvips",
        )
