"""Source-local Uniform-B block frame and frozen Torch adapter."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ....common.hashing import stable_hash
from ...block_frame import PilotFeatureFrame, fit_pilot_frame
from ...protocol import ProtocolError
from ..independent_source import IndependentSourceData


@dataclass(frozen=True)
class SourceBlockFrame:
    source_center: str
    source_row_hash: str
    frame: PilotFeatureFrame

    def __post_init__(self) -> None:
        if (
            self.frame.arm != "b_block_pca96_32"
            or self.frame.input_dim != 3840
            or self.frame.output_dim != 128
            or self.frame.fit_sample_hash != self.source_row_hash
        ):
            raise ProtocolError("Source block-frame identity is invalid.")

    @property
    def state_hash(self) -> str:
        return stable_hash(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_uniform_b_source_block_frame_v1",
            "source_center": self.source_center,
            "source_row_hash": self.source_row_hash,
            "fit_scope": "source_center_rows_only",
            "outer_or_inner_rows_used": False,
            "frame": self.frame.to_payload(),
        }


def fit_source_block_frame(source: IndependentSourceData) -> SourceBlockFrame:
    if source.embeddings.shape[1] != 3840:
        raise ProtocolError("Uniform-B source frame requires 3840-D embeddings.")
    frame = fit_pilot_frame(
        "b_block_pca96_32",
        source.embeddings,
        fit_sample_hash=source.row_hash,
    )
    return SourceBlockFrame(
        source_center=source.center,
        source_row_hash=source.row_hash,
        frame=frame,
    )


class TorchBlockFrame:
    """Differentiable frozen transform/inverse for one fitted block frame."""

    def __init__(
        self,
        source_frame: SourceBlockFrame,
        *,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.source_frame = source_frame
        self.device = str(device)
        self.dtype = dtype
        self._blocks = tuple(
            {
                "start": block.start,
                "stop": block.stop,
                "output_dim": block.output_dim,
                "scaler_mean": torch.as_tensor(
                    block.scaler_mean, dtype=dtype, device=device
                ).detach(),
                "scaler_scale": torch.as_tensor(
                    block.scaler_scale, dtype=dtype, device=device
                ).detach(),
                "pca_mean": torch.as_tensor(
                    block.pca_mean, dtype=dtype, device=device
                ).detach(),
                "components": torch.as_tensor(
                    block.pca_components, dtype=dtype, device=device
                ).detach(),
            }
            for block in source_frame.frame.blocks
        )

    def transform(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2 or embeddings.shape[1] != 3840:
            raise ProtocolError("Torch block-frame transform expects [n,3840].")
        outputs = []
        for block in self._blocks:
            raw = embeddings[:, block["start"] : block["stop"]]
            scaled = (raw - block["scaler_mean"]) / block["scaler_scale"]
            outputs.append(
                (scaled - block["pca_mean"]) @ block["components"].T
            )
        return torch.cat(outputs, dim=1)

    def inverse_transform(self, projected: torch.Tensor) -> torch.Tensor:
        if projected.ndim != 2 or projected.shape[1] != 128:
            raise ProtocolError("Torch block-frame inverse expects [n,128].")
        raw_blocks = []
        cursor = 0
        for block in self._blocks:
            width = int(block["output_dim"])
            block_z = projected[:, cursor : cursor + width]
            scaled = block_z @ block["components"] + block["pca_mean"]
            raw_blocks.append(
                scaled * block["scaler_scale"] + block["scaler_mean"]
            )
            cursor += width
        return torch.cat(raw_blocks, dim=1)


__all__ = (
    "SourceBlockFrame",
    "TorchBlockFrame",
    "fit_source_block_frame",
)
