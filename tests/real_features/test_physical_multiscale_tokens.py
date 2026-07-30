from __future__ import annotations

import pytest
import torch

from midogpp_thesis.data.features.virchow2 import (
    PATCH_GRID_SIDE,
    PATCH_TOKEN_START,
    pool_virchow2_tokens,
)


def test_virchow2_pooling_excludes_registers_and_uses_frozen_center_block() -> None:
    tokens = torch.zeros((1, PATCH_TOKEN_START + PATCH_GRID_SIDE**2, 1280))
    tokens[:, 0] = 2.0
    tokens[:, 1:PATCH_TOKEN_START] = 999.0
    patch_grid = torch.arange(
        PATCH_GRID_SIDE**2, dtype=torch.float32
    ).reshape(1, PATCH_GRID_SIDE, PATCH_GRID_SIDE, 1)
    tokens[:, PATCH_TOKEN_START:] = patch_grid.reshape(1, -1, 1)

    pooled = pool_virchow2_tokens(tokens, include_center=True)

    assert pooled.shape == (1, 3840)
    assert torch.all(pooled[:, :1280] == 2.0)
    assert pooled[0, 1280].item() == pytest.approx(127.5)
    assert pooled[0, 2560].item() == pytest.approx(127.5)
    assert not torch.any(pooled == 999.0)


def test_virchow2_pooling_rejects_token_layout_drift() -> None:
    with pytest.raises(RuntimeError, match="token count drift"):
        pool_virchow2_tokens(torch.zeros((1, 260, 1280)), include_center=False)

