from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from midogpp_thesis.data.features.virchow2_tokens import (
    Virchow2TokenLayout,
    central_virchow2_token_grid,
    pool_central_quadrants,
)
from midogpp_thesis.real_features.classifier_reference.protocol import ProtocolError
from midogpp_thesis.real_features.classifier_reference.uniform_b_spatial_probe.cache import (
    assemble_b_spatial_features,
)
from midogpp_thesis.real_features.classifier_reference.uniform_b_spatial_probe.config import (
    GLOBAL_DIM,
    SPATIAL_DIM,
    load_spatial_cache_config,
    load_spatial_probe_config,
)
from midogpp_thesis.real_features.classifier_reference.uniform_b_spatial_probe.runner import (
    _hard_core_exchange,
)


def test_central_grid_and_quadrants_preserve_order() -> None:
    layout = Virchow2TokenLayout(width=2)
    tokens = torch.zeros((1, layout.token_count, layout.width), dtype=torch.float32)
    patch_values = torch.arange(256, dtype=torch.float32).reshape(16, 16)
    tokens[:, layout.patch_token_start :, :] = patch_values.reshape(1, 256, 1)

    central = central_virchow2_token_grid(tokens, layout=layout)
    expected = patch_values[6:10, 6:10]
    assert central.shape == (1, 4, 4, 2)
    assert torch.equal(central[0, :, :, 0], expected)
    assert torch.equal(central[0, :, :, 1], expected)

    pooled = pool_central_quadrants(central, layout=layout)
    means = torch.tensor(
        [
            expected[:2, :2].mean(),
            expected[:2, :2].mean(),
            expected[:2, 2:].mean(),
            expected[:2, 2:].mean(),
            expected[2:, :2].mean(),
            expected[2:, :2].mean(),
            expected[2:, 2:].mean(),
            expected[2:, 2:].mean(),
        ]
    )
    assert torch.equal(pooled[0], means)


def test_b_spatial_uses_global_prefix_and_ordered_quadrants() -> None:
    canonical = torch.arange(2 * 3840, dtype=torch.float32).reshape(2, 3840)
    tokens = torch.empty((2, 4, 4, 1280), dtype=torch.float16)
    tokens[:, :2, :2] = 1
    tokens[:, :2, 2:] = 2
    tokens[:, 2:, :2] = 3
    tokens[:, 2:, 2:] = 4

    spatial = assemble_b_spatial_features(canonical, tokens)

    assert spatial.shape == (2, SPATIAL_DIM)
    assert spatial.dtype == torch.float32
    assert torch.equal(spatial[:, :GLOBAL_DIM], canonical[:, :GLOBAL_DIM])
    for index, expected in enumerate((1.0, 2.0, 3.0, 4.0)):
        start = GLOBAL_DIM + index * 1280
        assert torch.all(spatial[:, start : start + 1280] == expected)


def test_frozen_configs_load_and_reject_runtime_drift(tmp_path: Path) -> None:
    cache = load_spatial_cache_config(
        "datasets/midogpp/configs/uniform_b_spatial_token_cache_v1.yaml"
    )
    config_path = Path(
        "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/uniform_b_spatial_probe_v1.yaml"
    )
    probe = load_spatial_probe_config(config_path)
    assert cache.devices == ("cuda:0", "cuda:1")
    assert cache.batch_size_per_device == 32
    assert cache.hf_hub_cache_path == Path("/home/stud/spark/.cache/huggingface/hub")
    assert cache.hf_hub_local_files_only is True
    assert probe.expected_feature_dim == 7680
    assert probe.runtime.outer_jobs == 4
    assert probe.runtime.threads_per_job == 3

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["runtime"]["outer_jobs"] = 5
    drifted = tmp_path / "drifted.yaml"
    drifted.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ProtocolError, match="protocol drifted"):
        load_spatial_probe_config(drifted)


def test_hard_core_exchange_emits_zero_categories() -> None:
    baseline = {
        "sample": {"sample_id": "sample", "y_true": 0, "y_pred": 0}
    }
    spatial = [
        {"sample_id": "sample", "center": "0", "y_true": 0, "y_pred": 0}
    ]

    rows = _hard_core_exchange(baseline, baseline, spatial, ("0",))

    required = {
        "current_shared_hard",
        "hard_core_rescued",
        "hard_core_unresolved",
        "current_bplus_correct_spatial_wrong",
        "current_bplus_wrong_spatial_correct",
        "net_rescue_vs_current_bplus",
    }
    assert len(rows) == 3
    assert all(required.issubset(row) for row in rows)
    assert all(int(row[key]) == 0 for row in rows for key in required)
