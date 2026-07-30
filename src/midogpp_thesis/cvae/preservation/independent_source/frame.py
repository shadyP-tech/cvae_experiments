"""Source-row extraction and H/I/source identity firewalls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ....common.hashing import stable_hash
from ....real_features.classifier_reference.real_feature_frame import RealFeatureFrame
from ...protocol import ProtocolError
from ..splits import frame_arrays, indices_for_centers, row_hash


@dataclass(frozen=True)
class IndependentSourceData:
    center: str
    embeddings: np.ndarray
    labels: tuple[int, ...]
    sample_ids: tuple[str, ...]
    case_ids: tuple[str, ...]
    image_ids: tuple[str, ...]
    row_hash: str
    case_hash: str
    image_hash: str

    @property
    def identity_hash(self) -> str:
        return stable_hash(
            {
                "schema_version": "midogpp_independent_source_data_v1",
                "center": self.center,
                "row_hash": self.row_hash,
                "case_hash": self.case_hash,
                "image_hash": self.image_hash,
                "shape": list(self.embeddings.shape),
            }
        )


def extract_source_data(
    frame: RealFeatureFrame,
    center: str,
) -> IndependentSourceData:
    indices = indices_for_centers(frame, (str(center),))
    if not indices:
        raise ProtocolError(f"Independent source {center!r} has no rows.")
    embeddings, labels, sample_ids = frame_arrays(frame, indices)
    if set(labels) != {0, 1}:
        raise ProtocolError("Independent source must contain both classes.")
    cases = tuple(str(frame.rows[index].case_id) for index in indices)
    images = tuple(str(frame.rows[index].image_path) for index in indices)
    return IndependentSourceData(
        center=str(center),
        embeddings=np.asarray(embeddings, dtype=np.float32),
        labels=tuple(int(value) for value in labels),
        sample_ids=tuple(str(value) for value in sample_ids),
        case_ids=cases,
        image_ids=images,
        row_hash=row_hash(sample_ids),
        case_hash=row_hash(sorted(set(cases))),
        image_hash=row_hash(sorted(value for value in set(images) if value)),
    )


def assert_source_evaluation_isolation(
    source: IndependentSourceData,
    *,
    outer_center: str,
    inner_center: str,
    eval_sample_ids: Sequence[str],
    eval_case_ids: Sequence[str],
    eval_image_ids: Sequence[str],
) -> None:
    if len({source.center, str(outer_center), str(inner_center)}) != 3:
        raise ProtocolError("Independent evaluation requires distinct E, H, and I.")
    overlaps = (
        set(source.sample_ids).intersection(str(value) for value in eval_sample_ids),
        set(source.case_ids).intersection(str(value) for value in eval_case_ids),
        {
            value
            for value in set(source.image_ids).intersection(
                str(item) for item in eval_image_ids
            )
            if value
        },
    )
    if any(overlaps):
        raise ProtocolError("Source and held-out-inner identities overlap.")


__all__ = (
    "IndependentSourceData",
    "assert_source_evaluation_isolation",
    "extract_source_data",
)
