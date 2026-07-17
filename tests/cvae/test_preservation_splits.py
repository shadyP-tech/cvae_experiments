from __future__ import annotations

from pathlib import Path

import numpy as np

from midogpp_thesis.cvae.preservation.splits import (
    deeper_split,
    inner_split,
    outer_split,
    source_only_frame,
)
from midogpp_thesis.real_features.classifier_reference.real_feature_frame import (
    RealFeatureFrame,
    RealFeatureRow,
)


def test_nested_splits_exclude_h_i_j_and_quarantined_center() -> None:
    outer = outer_split("0")
    inner = inner_split("0", "1")
    deeper = deeper_split("0", "1", "2")
    assert "0" not in outer.fit_centers
    assert {"0", "1"}.isdisjoint(inner.fit_centers)
    assert {"0", "1", "2"}.isdisjoint(deeper.fit_centers)
    assert "4" not in outer.fit_centers
    assert "4" not in inner.fit_centers
    assert "4" not in deeper.fit_centers


def test_source_only_frame_physically_removes_outer_and_quarantined_rows() -> None:
    centers = ("0", "1", "4", "2")
    frame = RealFeatureFrame(
        embeddings=np.arange(8, dtype=np.float32).reshape(4, 2),
        rows=tuple(
            RealFeatureRow(
                row_index=index,
                sample_id=f"sample-{center}",
                case_id=f"case-{center}",
                center=center,
                label=index % 2,
                split="train",
            )
            for index, center in enumerate(centers)
        ),
        feature_extractor={},
        feature_cache_path=Path("cache.npz"),
        feature_cache_hash="cache-hash",
        manifest_path=Path("manifest.csv"),
        manifest_hash="manifest-hash",
        expected_feature_dim=2,
    )

    source = source_only_frame(frame, outer_target_center="0")

    assert tuple(row.center for row in source.rows) == ("1", "2")
    assert tuple(row.row_index for row in source.rows) == (0, 1)
    assert source.embeddings.tolist() == [[2.0, 3.0], [6.0, 7.0]]
