"""Slide-region readers for frozen physical-multiscale crop policies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from PIL import Image


class SlideReader:
    """Read level-0 RGB regions through pyvips, OpenSlide, or test-only PIL."""

    def __init__(
        self,
        path: Path,
        *,
        require_tiled: bool,
        required_backend: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.backend = ""
        self.reader: Any = None
        if required_backend not in (None, "pyvips"):
            raise ValueError(f"Unsupported required slide backend: {required_backend}")
        if required_backend == "pyvips":
            try:
                import pyvips  # type: ignore

                self.reader = pyvips.Image.new_from_file(
                    str(self.path),
                    access="random",
                )
                self.backend = "pyvips"
                return
            except Exception as exc:
                raise RuntimeError(
                    f"Required pyvips backend could not open production slide: "
                    f"{self.path}"
                ) from exc
        try:
            import pyvips  # type: ignore

            self.reader = pyvips.Image.new_from_file(str(self.path), access="random")
            self.backend = "pyvips"
        except (ModuleNotFoundError, OSError):
            try:
                import openslide  # type: ignore

                self.reader = openslide.OpenSlide(str(self.path))
                self.backend = "openslide"
            except (ModuleNotFoundError, OSError):
                if require_tiled:
                    raise RuntimeError(
                        f"Tiled pyvips/openslide reader is required for production: "
                        f"{self.path}"
                    )
                self.reader = Image.open(self.path)
                self.backend = "pillow_test_only"

    def __enter__(self) -> "SlideReader":
        return self

    def __exit__(self, *_args: object) -> None:
        close = getattr(self.reader, "close", None)
        if callable(close):
            close()

    @property
    def dimensions(self) -> tuple[int, int]:
        if self.backend == "pyvips":
            return int(self.reader.width), int(self.reader.height)
        if self.backend == "openslide":
            width, height = self.reader.dimensions
            return int(width), int(height)
        width, height = self.reader.size
        return int(width), int(height)

    def read_geometry(
        self,
        geometry: Mapping[str, object],
        *,
        padding_rgb: tuple[int, int, int],
    ) -> Image.Image:
        """Compatibility alias for the frozen v1 padding-capable path."""

        return self.read_geometry_v1(geometry, padding_rgb=padding_rgb)

    def read_geometry_v1(
        self,
        geometry: Mapping[str, object],
        *,
        padding_rgb: tuple[int, int, int],
    ) -> Image.Image:
        """Read, pad, and resize a v1 geometry exactly as the original builder."""

        side = int(geometry["side_px"])
        x0 = int(geometry["source_x0"])
        y0 = int(geometry["source_y0"])
        x1 = int(geometry["source_x1"])
        y1 = int(geometry["source_y1"])
        image = self._read_region_rgb(x0=x0, y0=y0, x1=x1, y1=y1)
        canvas = Image.new("RGB", (side, side), color=padding_rgb)
        canvas.paste(
            image,
            (int(geometry["pad_left"]), int(geometry["pad_top"])),
        )
        image.close()
        resized = canvas.resize((224, 224), resample=Image.Resampling.BICUBIC)
        canvas.close()
        return resized

    def read_exact_square(
        self,
        geometry: Mapping[str, object] | object,
        *,
        output_size: int | None = 224,
    ) -> Image.Image:
        """Read one validated in-bounds square without synthesizing pixels."""

        side = _geometry_int(geometry, "side_px")
        x0 = _geometry_int(geometry, "realized_x0")
        y0 = _geometry_int(geometry, "realized_y0")
        x1 = _geometry_int(geometry, "realized_x1")
        y1 = _geometry_int(geometry, "realized_y1")
        if side <= 0:
            raise ValueError("side_px must be positive.")
        if x1 - x0 != side or y1 - y0 != side:
            raise ValueError(
                "Realized bounds must describe the exact side_px square."
            )

        image_width, image_height = self.dimensions
        if not (0 <= x0 < x1 <= image_width and 0 <= y0 < y1 <= image_height):
            raise ValueError(
                "Realized bounds must be fully inside the source image."
            )
        _require_zero_realized_padding(geometry)
        if output_size is not None and (
            isinstance(output_size, bool)
            or not isinstance(output_size, int)
            or output_size <= 0
        ):
            raise ValueError("output_size must be a positive integer or None.")

        image = self._read_region_rgb(x0=x0, y0=y0, x1=x1, y1=y1)
        if image.size != (side, side):
            actual_size = image.size
            image.close()
            raise RuntimeError(
                f"Slide backend returned {actual_size}, expected {(side, side)}."
            )
        if output_size is None or output_size == side:
            return image
        resized = image.resize(
            (output_size, output_size),
            resample=Image.Resampling.BICUBIC,
        )
        image.close()
        return resized

    def read_exact_square_v2(
        self,
        geometry: Mapping[str, object] | object,
        *,
        output_size: int | None = 224,
    ) -> Image.Image:
        """Compatibility wrapper for the immutable v2 reader surface."""

        return self.read_exact_square(geometry, output_size=output_size)

    def _read_region_rgb(
        self,
        *,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
    ) -> Image.Image:
        width = x1 - x0
        height = y1 - y0
        if self.backend == "pyvips":
            region = self.reader.crop(x0, y0, width, height)
            if int(region.bands) >= 3:
                region = region.extract_band(0, n=3)
            elif int(region.bands) != 1:
                region = region.extract_band(0)
            data = region.write_to_memory()
            mode = "RGB" if int(region.bands) == 3 else "L"
            image = Image.frombytes(mode, (width, height), data)
            if mode != "RGB":
                image = image.convert("RGB")
            return image
        if self.backend == "openslide":
            return self.reader.read_region(
                (x0, y0),
                0,
                (width, height),
            ).convert("RGB")
        return self.reader.crop((x0, y0, x1, y1)).convert("RGB")


def open_slide(
    path: Path,
    *,
    require_tiled: bool,
    required_backend: str | None = None,
) -> SlideReader:
    """Open a slide reader with an explicit production/test backend policy."""

    return SlideReader(
        path,
        require_tiled=require_tiled,
        required_backend=required_backend,
    )


def _geometry_value(geometry: Mapping[str, object] | object, key: str) -> object:
    if isinstance(geometry, Mapping):
        try:
            return geometry[key]
        except KeyError as exc:
            raise ValueError(f"Exact-square geometry is missing {key!r}.") from exc
    try:
        return getattr(geometry, key)
    except AttributeError as exc:
        raise ValueError(f"Exact-square geometry is missing {key!r}.") from exc


def _geometry_int(geometry: Mapping[str, object] | object, key: str) -> int:
    value = _geometry_value(geometry, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Exact-square geometry field {key!r} must be an integer.")
    return value


def _require_zero_realized_padding(
    geometry: Mapping[str, object] | object,
) -> None:
    for key in ("pad_left", "pad_top", "pad_right", "pad_bottom"):
        try:
            value = _geometry_value(geometry, key)
        except ValueError:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value != 0:
            raise ValueError("Exact-square geometry cannot request padding.")
    try:
        padding_fraction = _geometry_value(geometry, "padding_fraction")
    except ValueError:
        return
    if (
        isinstance(padding_fraction, bool)
        or not isinstance(padding_fraction, (int, float))
        or float(padding_fraction) != 0.0
    ):
        raise ValueError("Exact-square geometry cannot request padding.")
