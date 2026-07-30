from __future__ import annotations

from pathlib import Path

import pytest
import torch

from midogpp_thesis.data.features.cache_io import write_center_shard
from midogpp_thesis.real_features.classifier_reference.physical_multiscale_center_pooling.frames import (
    CenterShardedRepresentationStore,
)
from midogpp_thesis.real_features.classifier_reference.protocol import ProtocolError


def test_center_sharded_store_excludes_outer_center_until_explicit_eval(
    tmp_path: Path,
) -> None:
    b_root = tmp_path / "b"
    c_root = tmp_path / "c"
    for center in ("0", "1"):
        metadata = [
            {
                "sample_id": f"{center}_{index}",
                "case_id": f"case_{center}_{index}",
                "label": index,
                "split": "train",
                "center": center,
                "contract_row_index": int(center) * 2 + index,
            }
            for index in range(2)
        ]
        write_center_shard(
            b_root / "embeddings" / "by_center" / f"center_{center}.pt",
            embeddings=torch.zeros((2, 3840)),
            canonical_a_embeddings=torch.zeros((2, 2560)),
            metadata=metadata,
            feature_extractor={"representation_id": "jpeg_center_b"},
        )
        write_center_shard(
            c_root / "embeddings" / "by_center" / f"center_{center}.pt",
            embeddings=torch.zeros((2, 11520)),
            metadata=metadata,
            feature_extractor={"representation_id": "physical_multiscale_center_c"},
        )
    store = CenterShardedRepresentationStore(
        b_cache_root=b_root,
        c_cache_root=c_root,
    )

    selector = store.selector_frame(
        outer_target_center="0",
        eligible_centers=("0", "1"),
    )
    assert set(selector.centers) == {"1"}
    assert store.access_log == [("selector_outer_0", "1")]

    outer = store.outer_frame("0")
    assert set(outer.centers) == {"0"}
    assert store.access_log[-1] == ("outer_eval_0", "0")


def test_center_sharded_store_restores_global_contract_order(
    tmp_path: Path,
) -> None:
    b_root = tmp_path / "b"
    c_root = tmp_path / "c"
    contract_indices = {
        "0": (1, 3),
        "1": (0, 2),
        "2": (4, 5),
    }
    for center, indices in contract_indices.items():
        metadata = [
            {
                "sample_id": f"{center}_{index}",
                "case_id": f"case_{center}_{index}",
                "label": index,
                "split": "train",
                "center": center,
                "contract_row_index": contract_index,
            }
            for index, contract_index in enumerate(indices)
        ]
        values = torch.tensor(indices, dtype=torch.float32).reshape(2, 1)
        write_center_shard(
            b_root / "embeddings" / "by_center" / f"center_{center}.pt",
            embeddings=values.repeat(1, 3840),
            canonical_a_embeddings=values.repeat(1, 2560),
            metadata=metadata,
            feature_extractor={"representation_id": "jpeg_center_b"},
        )
        write_center_shard(
            c_root / "embeddings" / "by_center" / f"center_{center}.pt",
            embeddings=values.repeat(1, 11520),
            metadata=metadata,
            feature_extractor={"representation_id": "physical_multiscale_center_c"},
        )
    store = CenterShardedRepresentationStore(
        b_cache_root=b_root,
        c_cache_root=c_root,
    )

    selector = store.selector_frame(
        outer_target_center="2",
        eligible_centers=("0", "1", "2"),
    )

    assert selector.sample_ids == ("1_0", "0_0", "1_1", "0_1")
    assert selector.case_ids == (
        "case_1_0",
        "case_0_0",
        "case_1_1",
        "case_0_1",
    )
    assert selector.centers == ("1", "0", "1", "0")
    assert selector.labels.tolist() == [0, 0, 1, 1]
    for representation in store.representation_order:
        assert selector.embeddings[representation][:, 0].tolist() == [
            0.0,
            1.0,
            2.0,
            3.0,
        ]


def test_center_sharded_store_rejects_duplicate_contract_row_indices(
    tmp_path: Path,
) -> None:
    b_root = tmp_path / "b"
    c_root = tmp_path / "c"
    for center in ("0", "1"):
        metadata = [
            {
                "sample_id": f"{center}_0",
                "case_id": f"case_{center}_0",
                "label": int(center),
                "split": "train",
                "center": center,
                "contract_row_index": 0,
            }
        ]
        write_center_shard(
            b_root / "embeddings" / "by_center" / f"center_{center}.pt",
            embeddings=torch.zeros((1, 3840)),
            canonical_a_embeddings=torch.zeros((1, 2560)),
            metadata=metadata,
            feature_extractor={"representation_id": "jpeg_center_b"},
        )
        write_center_shard(
            c_root / "embeddings" / "by_center" / f"center_{center}.pt",
            embeddings=torch.zeros((1, 11520)),
            metadata=metadata,
            feature_extractor={"representation_id": "physical_multiscale_center_c"},
        )
    store = CenterShardedRepresentationStore(
        b_cache_root=b_root,
        c_cache_root=c_root,
    )

    with pytest.raises(ProtocolError, match="duplicate contract_row_index"):
        store.selector_frame(
            outer_target_center="2",
            eligible_centers=("0", "1", "2"),
        )


@pytest.mark.parametrize("invalid_index", [None, -1, True, 1.5, "01"])
def test_center_sharded_store_rejects_invalid_contract_row_index(
    tmp_path: Path,
    invalid_index: object,
) -> None:
    b_root = tmp_path / "b"
    c_root = tmp_path / "c"
    metadata = [
        {
            "sample_id": "0_0",
            "case_id": "case_0_0",
            "label": 0,
            "split": "train",
            "center": "0",
            "contract_row_index": invalid_index,
        }
    ]
    write_center_shard(
        b_root / "embeddings" / "by_center" / "center_0.pt",
        embeddings=torch.zeros((1, 3840)),
        canonical_a_embeddings=torch.zeros((1, 2560)),
        metadata=metadata,
        feature_extractor={"representation_id": "jpeg_center_b"},
    )
    write_center_shard(
        c_root / "embeddings" / "by_center" / "center_0.pt",
        embeddings=torch.zeros((1, 11520)),
        metadata=metadata,
        feature_extractor={"representation_id": "physical_multiscale_center_c"},
    )
    store = CenterShardedRepresentationStore(
        b_cache_root=b_root,
        c_cache_root=c_root,
    )

    with pytest.raises(ProtocolError, match="valid contract_row_index"):
        store.selector_frame(
            outer_target_center="2",
            eligible_centers=("0", "2"),
        )


def test_center_sharded_store_rejects_metadata_tensor_row_mismatch(
    tmp_path: Path,
) -> None:
    b_root = tmp_path / "b"
    c_root = tmp_path / "c"
    metadata = [
        {
            "sample_id": "0_0",
            "case_id": "case_0_0",
            "label": 0,
            "split": "train",
            "center": "0",
            "contract_row_index": 0,
        }
    ]
    write_center_shard(
        b_root / "embeddings" / "by_center" / "center_0.pt",
        embeddings=torch.zeros((2, 3840)),
        canonical_a_embeddings=torch.zeros((2, 2560)),
        metadata=metadata,
        feature_extractor={"representation_id": "jpeg_center_b"},
    )
    write_center_shard(
        c_root / "embeddings" / "by_center" / "center_0.pt",
        embeddings=torch.zeros((2, 11520)),
        metadata=metadata,
        feature_extractor={"representation_id": "physical_multiscale_center_c"},
    )
    store = CenterShardedRepresentationStore(
        b_cache_root=b_root,
        c_cache_root=c_root,
    )

    with pytest.raises(ProtocolError, match="dimension drift"):
        store.selector_frame(
            outer_target_center="2",
            eligible_centers=("0", "2"),
        )
