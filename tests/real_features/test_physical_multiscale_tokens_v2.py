from __future__ import annotations

import math

from PIL import Image
import pytest
import torch

from midogpp_thesis.data.features.virchow2 import (
    PATCH_GRID_SIDE,
    PATCH_TOKEN_START,
    Virchow2TokenExtractor,
    pool_virchow2_tokens,
)
from midogpp_thesis.data.features.virchow2_tokens import (
    VIRCHOW2_TOKEN_LAYOUT,
    Virchow2TokenLayout,
    assert_preprocessing_spatial_identity,
    describe_preprocessing_spatial_identity,
    normalized_coordinate_to_window_start,
    normalized_position_to_window_start,
    pool_virchow2_tokens_v2,
    validate_virchow2_token_layout,
)


FIXTURE_LAYOUT = Virchow2TokenLayout(width=1)


def _fixture_tokens(batch_size: int = 1) -> torch.Tensor:
    tokens = torch.zeros((batch_size, FIXTURE_LAYOUT.token_count, 1))
    tokens[:, 0] = 7.0
    tokens[:, 1 : FIXTURE_LAYOUT.patch_token_start] = 99_999.0
    grid = torch.tensor(
        [
            [row * 100 + column for column in range(FIXTURE_LAYOUT.patch_grid_side)]
            for row in range(FIXTURE_LAYOUT.patch_grid_side)
        ],
        dtype=torch.float32,
    )
    tokens[:, FIXTURE_LAYOUT.patch_token_start :] = grid.reshape(1, -1, 1)
    return tokens


def test_exact_layout_validates_rank_token_count_and_width() -> None:
    tokens = _fixture_tokens()

    assert validate_virchow2_token_layout(tokens, layout=FIXTURE_LAYOUT) is tokens

    with pytest.raises(ValueError, match="rank 3"):
        validate_virchow2_token_layout(tokens[0], layout=FIXTURE_LAYOUT)
    with pytest.raises(ValueError, match="token count drift"):
        validate_virchow2_token_layout(tokens[:, :-1], layout=FIXTURE_LAYOUT)
    with pytest.raises(ValueError, match="token width drift"):
        validate_virchow2_token_layout(
            torch.zeros((1, FIXTURE_LAYOUT.token_count, 2)),
            layout=FIXTURE_LAYOUT,
        )


def test_normalized_coordinates_use_x_for_column_and_y_for_row() -> None:
    assert normalized_coordinate_to_window_start(0.5) == 6
    assert normalized_coordinate_to_window_start(0.0) == 0
    assert normalized_coordinate_to_window_start(math.nextafter(1.0, 0.0)) == 12
    assert normalized_position_to_window_start(x=0.25, y=0.75) == (10, 2)

    pooled = pool_virchow2_tokens_v2(
        _fixture_tokens(),
        window_starts=[
            normalized_position_to_window_start(
                x=0.25,
                y=0.75,
                layout=FIXTURE_LAYOUT,
            )
        ],
        layout=FIXTURE_LAYOUT,
    )
    expected_local = torch.tensor(
        [row * 100 + column for row in range(10, 14) for column in range(2, 6)],
        dtype=torch.float32,
    ).mean()
    assert pooled.shape == (1, 3)
    assert pooled[0, 2].item() == pytest.approx(expected_local.item())
    assert not torch.any(pooled == 99_999.0)


@pytest.mark.parametrize("position", [-0.01, 1.0, float("nan"), float("inf")])
def test_normalized_coordinate_rejects_out_of_domain_values(position: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        normalized_coordinate_to_window_start(position)


@pytest.mark.parametrize(
    ("window_starts", "message"),
    [
        ([], "count must match"),
        ([(0, 0), (1, 1)], "count must match"),
        ([(0,)], "row and column"),
        ([(0.0, 0)], "must be an integer"),
        ([(True, 0)], "must be an integer"),
        ([(-1, 0)], r"must be in \[0, 12\]"),
        ([(0, 13)], r"must be in \[0, 12\]"),
    ],
)
def test_pooling_rejects_malformed_per_sample_windows(
    window_starts: list[tuple[object, ...]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        pool_virchow2_tokens_v2(
            _fixture_tokens(),
            window_starts=window_starts,
            layout=FIXTURE_LAYOUT,
        )


def test_pooling_rejects_an_empty_batch() -> None:
    with pytest.raises(ValueError, match="batch may not be empty"):
        pool_virchow2_tokens_v2(
            torch.zeros((0, FIXTURE_LAYOUT.token_count, FIXTURE_LAYOUT.width)),
            window_starts=[],
            layout=FIXTURE_LAYOUT,
        )


def test_fixed_center_v2_is_byte_equivalent_to_v1_b_representation() -> None:
    tokens = torch.arange(
        (PATCH_TOKEN_START + PATCH_GRID_SIDE**2) * 1280,
        dtype=torch.float32,
    ).reshape(1, PATCH_TOKEN_START + PATCH_GRID_SIDE**2, 1280)

    v1 = pool_virchow2_tokens(tokens, include_center=True)
    v2 = pool_virchow2_tokens_v2(
        tokens,
        window_starts=[(6, 6)],
        layout=VIRCHOW2_TOKEN_LAYOUT,
    )

    assert v1.numpy().tobytes() == v2.numpy().tobytes()


class _ForwardFeaturesOnlyModel:
    def __init__(self, tokens: torch.Tensor) -> None:
        self.tokens = tokens
        self.forward_features_calls = 0

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        raise AssertionError("v2 extraction must not call model.__call__")

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        self.forward_features_calls += 1
        return self.tokens.expand(int(inputs.shape[0]), -1, -1)


def test_v2_extractor_calls_forward_features_and_accepts_per_image_windows() -> None:
    extractor = object.__new__(Virchow2TokenExtractor)
    model = _ForwardFeaturesOnlyModel(_fixture_tokens())
    extractor.model = model
    extractor.transform = lambda _image: torch.zeros((3, 224, 224))
    extractor.device = torch.device("cpu")
    extractor.identity = {}
    images = [Image.new("RGB", (224, 224)), Image.new("RGB", (224, 224))]

    pooled = extractor.extract_images_v2(
        images,
        window_starts=[(0, 0), (12, 12)],
        layout=FIXTURE_LAYOUT,
    )

    assert model.forward_features_calls == 1
    assert pooled.shape == (2, 3)
    assert pooled[0, 2].item() != pooled[1, 2].item()


def test_preprocessing_spatial_identity_is_described_and_asserted_without_timm() -> None:
    config = {
        "input_size": (3, 224, 224),
        "crop_pct": 1.0,
        "interpolation": "bicubic",
    }
    record = describe_preprocessing_spatial_identity(config)

    assert record["spatial_identity"] is True
    assert assert_preprocessing_spatial_identity(config) == record
    with pytest.raises(ValueError, match="not spatially identity"):
        assert_preprocessing_spatial_identity({**config, "crop_pct": 0.875})
    with pytest.raises(ValueError, match="not spatially identity"):
        assert_preprocessing_spatial_identity({"input_size": (3, 224, 224)})
