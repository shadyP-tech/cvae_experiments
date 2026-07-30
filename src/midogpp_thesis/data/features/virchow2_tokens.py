"""Strict Virchow2 token-layout and spatial-window pooling primitives.

The v2 policy is intentionally separate from the historical pooling helper in
``virchow2.py``.  The latter remains the compatibility surface for the
fixed-centre v1 cache, while this module makes every spatial assumption
explicit and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class Virchow2TokenLayout:
    """The exact token layout emitted by Virchow2 ``forward_features``.

    Production tokens have width 1280.  Tests may provide a smaller explicit
    width while retaining the production token ordering and grid geometry.
    """

    width: int = 1280
    cls_token_count: int = 1
    register_token_count: int = 4
    patch_grid_side: int = 16
    window_side: int = 4
    patch_order: str = "row-major"

    def __post_init__(self) -> None:
        integer_fields = {
            "width": self.width,
            "cls_token_count": self.cls_token_count,
            "register_token_count": self.register_token_count,
            "patch_grid_side": self.patch_grid_side,
            "window_side": self.window_side,
        }
        invalid = {
            key: value
            for key, value in integer_fields.items()
            if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0
        }
        if invalid:
            raise ValueError(
                f"Virchow2 token-layout dimensions must be positive integers: {invalid}"
            )
        if int(self.cls_token_count) != 1:
            raise ValueError("Virchow2 token layout requires exactly one CLS token.")
        if int(self.register_token_count) != 4:
            raise ValueError("Virchow2 token layout requires exactly four register tokens.")
        if int(self.patch_grid_side) != 16:
            raise ValueError("Virchow2 token layout requires a 16x16 patch-token grid.")
        if int(self.window_side) != 4:
            raise ValueError("Virchow2 spatial pooling requires a 4x4 token window.")
        if self.patch_order != "row-major":
            raise ValueError("Virchow2 patch tokens must use row-major ordering.")

    @property
    def patch_token_start(self) -> int:
        return int(self.cls_token_count + self.register_token_count)

    @property
    def patch_token_count(self) -> int:
        return int(self.patch_grid_side**2)

    @property
    def token_count(self) -> int:
        return self.patch_token_start + self.patch_token_count

    @property
    def maximum_window_start(self) -> int:
        return int(self.patch_grid_side - self.window_side)

    @property
    def pooled_width(self) -> int:
        return int(3 * self.width)


VIRCHOW2_TOKEN_LAYOUT = Virchow2TokenLayout()


def validate_virchow2_token_layout(
    outputs: Any,
    *,
    layout: Virchow2TokenLayout = VIRCHOW2_TOKEN_LAYOUT,
) -> Any:
    """Resolve and validate an exact ``[batch, 261, width]`` token tensor."""

    tokens = _resolve_tensor(outputs)
    shape = getattr(tokens, "shape", None)
    if getattr(tokens, "ndim", None) != 3:
        raise ValueError(f"Virchow2 tokens must be rank 3, got shape={shape}.")
    actual_token_count = int(tokens.shape[1])
    if actual_token_count != layout.token_count:
        raise ValueError(
            "Virchow2 token count drift: "
            f"expected={layout.token_count}, actual={actual_token_count}; "
            "layout is 1 CLS + 4 registers + 256 row-major patch tokens."
        )
    actual_width = int(tokens.shape[2])
    if actual_width != int(layout.width):
        raise ValueError(
            f"Virchow2 token width drift: expected={layout.width}, actual={actual_width}."
        )
    return tokens


def normalized_coordinate_to_window_start(
    position: float,
    *,
    layout: Virchow2TokenLayout = VIRCHOW2_TOKEN_LAYOUT,
) -> int:
    """Map one normalized coordinate to its clamped 4-token window start.

    The policy is ``clamp(floor(16 * p - 2), 0, 12)`` for finite
    ``p`` in ``[0, 1)``.
    """

    if isinstance(position, bool):
        raise ValueError("Normalized token-window position must be a finite number in [0, 1).")
    try:
        numeric = float(position)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Normalized token-window position must be a finite number in [0, 1)."
        ) from exc
    if not math.isfinite(numeric) or not 0.0 <= numeric < 1.0:
        raise ValueError(
            f"Normalized token-window position must be finite and in [0, 1), got {position!r}."
        )
    raw_start = math.floor(layout.patch_grid_side * numeric - layout.window_side / 2)
    return min(max(int(raw_start), 0), layout.maximum_window_start)


def normalized_position_to_window_start(
    *,
    x: float,
    y: float,
    layout: Virchow2TokenLayout = VIRCHOW2_TOKEN_LAYOUT,
) -> tuple[int, int]:
    """Return ``(row, column)`` start for normalized image position ``(x, y)``."""

    column = normalized_coordinate_to_window_start(x, layout=layout)
    row = normalized_coordinate_to_window_start(y, layout=layout)
    return row, column


def validate_window_starts(
    window_starts: Sequence[Sequence[int]],
    *,
    batch_size: int,
    layout: Virchow2TokenLayout = VIRCHOW2_TOKEN_LAYOUT,
) -> tuple[tuple[int, int], ...]:
    """Validate exactly one integer ``(row, column)`` start per sample."""

    try:
        starts = tuple(window_starts)
    except TypeError as exc:
        raise ValueError("Virchow2 window starts must be a sequence.") from exc
    if len(starts) != int(batch_size):
        raise ValueError(
            "Virchow2 window-start count must match batch size: "
            f"expected={batch_size}, actual={len(starts)}."
        )

    validated: list[tuple[int, int]] = []
    for sample_index, start in enumerate(starts):
        try:
            pair = tuple(start)
        except TypeError as exc:
            raise ValueError(
                f"Virchow2 window start at sample {sample_index} must be a (row, column) pair."
            ) from exc
        if len(pair) != 2:
            raise ValueError(
                f"Virchow2 window start at sample {sample_index} must contain row and column."
            )
        row, column = pair
        for axis, value in (("row", row), ("column", column)):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(
                    f"Virchow2 {axis} start at sample {sample_index} must be an integer, "
                    f"got {value!r}."
                )
            if not 0 <= int(value) <= layout.maximum_window_start:
                raise ValueError(
                    f"Virchow2 {axis} start at sample {sample_index} must be in "
                    f"[0, {layout.maximum_window_start}], got {value!r}."
                )
        validated.append((int(row), int(column)))
    return tuple(validated)


def pool_virchow2_tokens_v2(
    outputs: Any,
    *,
    window_starts: Sequence[Sequence[int]],
    layout: Virchow2TokenLayout = VIRCHOW2_TOKEN_LAYOUT,
) -> Any:
    """Pool CLS, global patches, and one sample-specific 4x4 patch window.

    Register tokens are deliberately excluded.  Patch tokens are reshaped in
    row-major order, and each 4x4 window receives a uniform arithmetic mean.
    """

    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - production dependency
        raise RuntimeError("Virchow2 token pooling requires torch.") from exc

    tokens = validate_virchow2_token_layout(outputs, layout=layout)
    batch_size = int(tokens.shape[0])
    if batch_size == 0:
        raise ValueError("Virchow2 v2 pooling batch may not be empty.")
    starts = validate_window_starts(
        window_starts,
        batch_size=batch_size,
        layout=layout,
    )
    patch = tokens[:, layout.patch_token_start :]
    grid = patch.reshape(
        batch_size,
        layout.patch_grid_side,
        layout.patch_grid_side,
        layout.width,
    )
    local_means = torch.stack(
        tuple(
            grid[
                sample_index,
                row : row + layout.window_side,
                column : column + layout.window_side,
                :,
            ].reshape(-1, layout.width).mean(dim=0)
            for sample_index, (row, column) in enumerate(starts)
        ),
        dim=0,
    )
    pooled = torch.cat(
        (
            tokens[:, 0],
            patch.mean(dim=1),
            local_means,
        ),
        dim=-1,
    )
    expected_shape = (batch_size, layout.pooled_width)
    if tuple(pooled.shape) != expected_shape:
        raise RuntimeError(
            f"Virchow2 v2 pooled shape drift: expected={expected_shape}, "
            f"actual={tuple(pooled.shape)}."
        )
    return pooled


def describe_preprocessing_spatial_identity(
    preprocessing_config: Mapping[str, object],
    *,
    source_size: tuple[int, int] = (224, 224),
) -> dict[str, object]:
    """Describe whether timm eval preprocessing preserves a 224-square grid.

    Normalization and tensor conversion are not spatial operations.  A timm
    evaluation transform is spatially identity for an already-target-sized
    square when its configured input size matches the source and ``crop_pct``
    is exactly 1.
    """

    raw_input_size = preprocessing_config.get("input_size")
    if not isinstance(raw_input_size, (tuple, list)) or len(raw_input_size) not in (2, 3):
        target_size: tuple[int, int] | None = None
    else:
        try:
            target_size = (int(raw_input_size[-2]), int(raw_input_size[-1]))
        except (TypeError, ValueError):
            target_size = None
    raw_crop_pct = preprocessing_config.get("crop_pct")
    try:
        crop_pct = float(raw_crop_pct)
    except (TypeError, ValueError):
        crop_pct = float("nan")
    source = (int(source_size[0]), int(source_size[1]))
    spatial_identity = (
        source == (224, 224)
        and target_size == source
        and math.isfinite(crop_pct)
        and crop_pct == 1.0
    )
    return {
        "schema_version": "midogpp_virchow2_preprocessing_spatial_identity_v1",
        "source_size": list(source),
        "target_size": list(target_size) if target_size is not None else None,
        "crop_pct": raw_crop_pct,
        "spatial_identity": spatial_identity,
        "reason": (
            "already-224 square input; target size unchanged; crop_pct=1"
            if spatial_identity
            else "preprocessing may resize or crop the input token grid"
        ),
    }


def assert_preprocessing_spatial_identity(
    preprocessing_config: Mapping[str, object],
    *,
    source_size: tuple[int, int] = (224, 224),
) -> dict[str, object]:
    """Return the spatial-identity record or reject a spatially changing config."""

    record = describe_preprocessing_spatial_identity(
        preprocessing_config,
        source_size=source_size,
    )
    if not bool(record["spatial_identity"]):
        raise ValueError(f"Virchow2 preprocessing is not spatially identity: {record}")
    return record


def _resolve_tensor(value: Any, *, depth: int = 0) -> Any:
    if depth > 8:
        raise ValueError(f"Could not resolve Virchow2 token tensor from {type(value)}.")
    if getattr(value, "ndim", None) is not None:
        return value
    if isinstance(value, Mapping):
        for candidate in value.values():
            try:
                return _resolve_tensor(candidate, depth=depth + 1)
            except ValueError:
                continue
    if hasattr(value, "to_tuple"):
        return _resolve_tensor(value.to_tuple(), depth=depth + 1)
    if isinstance(value, (tuple, list)):
        for candidate in value:
            try:
                return _resolve_tensor(candidate, depth=depth + 1)
            except ValueError:
                continue
    raise ValueError(f"Could not resolve Virchow2 token tensor from {type(value)}.")


__all__ = [
    "VIRCHOW2_TOKEN_LAYOUT",
    "Virchow2TokenLayout",
    "assert_preprocessing_spatial_identity",
    "describe_preprocessing_spatial_identity",
    "normalized_coordinate_to_window_start",
    "normalized_position_to_window_start",
    "pool_virchow2_tokens_v2",
    "validate_virchow2_token_layout",
    "validate_window_starts",
]
