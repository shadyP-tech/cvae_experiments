from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from midogpp_thesis.real_features.classifier_reference.conditional_logit_alignment.folds import (
    make_inner_fold,
    make_outer_fold,
)
from midogpp_thesis.real_features.classifier_reference.protocol import ProtocolError
from midogpp_thesis.real_features.classifier_reference.real_feature_frame import (
    RealFeatureFrame,
    RealFeatureRow,
)


def test_outer_and_inner_folds_exclude_h_i_and_quarantined_center() -> None:
    frame = _frame(("3", "1", "4", "0", "2"))

    outer = make_outer_fold(frame, "0")
    assert outer.outer_target_center == "0"
    assert outer.inner_pseudo_target_center is None
    assert outer.eval_center == "0"
    assert outer.fit_centers == ("1", "2", "3")
    assert set(outer.fit_domains) == {"1", "2", "3"}
    assert "0" not in outer.fit_domains
    assert "4" not in outer.fit_domains
    assert np.asarray(outer.fit_embeddings).dtype == np.float64
    assert outer.fit_row_hash == _row_hash(outer.fit_sample_ids)
    assert outer.eval_row_hash == _row_hash(outer.eval_sample_ids)
    assert outer.training_frame_hash

    inner = make_inner_fold(frame, "0", "2")
    assert inner.outer_target_center == "0"
    assert inner.inner_pseudo_target_center == "2"
    assert inner.eval_center == "2"
    assert inner.fit_centers == ("1", "3")
    assert set(inner.fit_domains) == {"1", "3"}
    assert not {"0", "2", "4"}.intersection(inner.fit_domains)
    assert all(identity.image_path for identity in inner.fit_identities)
    assert len(inner.fit_identities) == inner.n_fit
    assert len(inner.eval_identities) == inner.n_eval


def test_fold_hashes_change_with_outer_inner_identity() -> None:
    frame = _frame(("0", "1", "2", "3"))

    first = make_inner_fold(frame, "0", "1")
    second = make_inner_fold(frame, "0", "2")

    assert first.fit_row_hash != second.fit_row_hash
    assert first.eval_row_hash != second.eval_row_hash
    assert first.training_frame_hash != second.training_frame_hash


def test_fold_fails_closed_on_case_overlap() -> None:
    frame = _frame(("0", "1", "2"), shared_case=("0", "1"))

    with pytest.raises(ProtocolError, match="case identities overlap"):
        make_outer_fold(frame, "0")


def test_fold_fails_closed_when_eval_center_lacks_a_class() -> None:
    frame = _frame(("0", "1", "2"), single_class_center="0")

    with pytest.raises(ProtocolError, match="evaluation rows must contain both"):
        make_outer_fold(frame, "0")


def test_inner_fold_rejects_same_or_absent_center() -> None:
    frame = _frame(("0", "1", "2"))

    with pytest.raises(ProtocolError, match="must differ"):
        make_inner_fold(frame, "0", "0")
    with pytest.raises(ProtocolError, match="absent"):
        make_inner_fold(frame, "0", "3")
    with pytest.raises(ProtocolError, match="Unknown or quarantined"):
        make_outer_fold(frame, "4")


def _frame(
    centers: tuple[str, ...],
    *,
    shared_case: tuple[str, str] | None = None,
    single_class_center: str | None = None,
) -> RealFeatureFrame:
    rows: list[RealFeatureRow] = []
    embeddings: list[list[float]] = []
    for center in centers:
        labels = (0, 0) if center == single_class_center else (0, 1)
        for local_index, label in enumerate(labels):
            sample_id = f"sample-{center}-{local_index}"
            case_id = f"case-{center}-{local_index}"
            if shared_case is not None and center in shared_case and local_index == 0:
                case_id = "shared-case"
            rows.append(
                RealFeatureRow(
                    row_index=len(rows),
                    sample_id=sample_id,
                    case_id=case_id,
                    center=center,
                    label=label,
                    split="train",
                    image_path=f"/images/{sample_id}.png",
                )
            )
            numeric_center = float(center)
            embeddings.append(
                [numeric_center + 0.2 * label, numeric_center * label + local_index]
            )
    return RealFeatureFrame(
        embeddings=np.asarray(embeddings, dtype=np.float32),
        rows=tuple(rows),
        feature_extractor={"name": "virchow2"},
        feature_cache_path=Path("/tmp/midogpp/virchow2/cache.pt"),
        feature_cache_hash="cache-hash",
        manifest_path=Path("/tmp/midogpp/manifest.csv"),
        manifest_hash="manifest-hash",
        expected_feature_dim=2,
    )


def _row_hash(sample_ids: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest()
