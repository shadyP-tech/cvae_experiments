"""Eligible-center outer, inner, and deeper split contracts."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import Mapping, Sequence

from ...real_features.classifier_reference.real_feature_frame import RealFeatureFrame
from ...real_features.classifier_reference.schemas.midogpp import (
    MIDOGPP_ELIGIBLE_CENTERS,
    MIDOGPP_EXCLUDED_CENTERS,
)
from ...real_features.classifier_reference.protocol import ProtocolError


@dataclass(frozen=True)
class NestedCenterSplit:
    outer_target_center: str
    inner_pseudo_target_center: str | None
    deeper_validation_center: str | None
    fit_centers: tuple[str, ...]

    @property
    def heldout_centers(self) -> tuple[str, ...]:
        return tuple(
            center
            for center in (
                self.outer_target_center,
                self.inner_pseudo_target_center,
                self.deeper_validation_center,
            )
            if center is not None
        )


def outer_split(outer_target_center: str, *, centers: Sequence[str] = MIDOGPP_ELIGIBLE_CENTERS) -> NestedCenterSplit:
    outer = _eligible(outer_target_center)
    return NestedCenterSplit(outer, None, None, tuple(center for center in centers if center != outer))


def inner_split(
    outer_target_center: str,
    inner_pseudo_target_center: str,
    *,
    centers: Sequence[str] = MIDOGPP_ELIGIBLE_CENTERS,
) -> NestedCenterSplit:
    outer = _eligible(outer_target_center)
    inner = _eligible(inner_pseudo_target_center)
    if inner == outer:
        raise ProtocolError("Inner pseudo-target must differ from outer target.")
    return NestedCenterSplit(outer, inner, None, tuple(center for center in centers if center not in {outer, inner}))


def deeper_split(
    outer_target_center: str,
    inner_pseudo_target_center: str,
    deeper_validation_center: str,
    *,
    centers: Sequence[str] = MIDOGPP_ELIGIBLE_CENTERS,
) -> NestedCenterSplit:
    base = inner_split(outer_target_center, inner_pseudo_target_center, centers=centers)
    deeper = _eligible(deeper_validation_center)
    if deeper in {base.outer_target_center, base.inner_pseudo_target_center}:
        raise ProtocolError("Deeper validation center must differ from outer and inner centers.")
    return NestedCenterSplit(
        base.outer_target_center,
        base.inner_pseudo_target_center,
        deeper,
        tuple(center for center in base.fit_centers if center != deeper),
    )


def indices_for_centers(frame: RealFeatureFrame, centers: Sequence[str]) -> tuple[int, ...]:
    center_set = {str(center) for center in centers}
    if center_set.intersection(MIDOGPP_EXCLUDED_CENTERS):
        raise ProtocolError("Quarantined centers cannot enter preservation splits.")
    unknown = center_set.difference(MIDOGPP_ELIGIBLE_CENTERS)
    if unknown:
        raise ProtocolError(f"Unknown MIDOG++ centers: {sorted(unknown)}")
    return tuple(index for index, row in enumerate(frame.rows) if row.center in center_set)


def frame_arrays(frame: RealFeatureFrame, indices: Sequence[int]) -> tuple[object, tuple[int, ...], tuple[str, ...]]:
    import numpy as np

    embeddings = frame.embeddings
    if hasattr(embeddings, "detach"):
        embeddings = embeddings.detach().cpu().numpy()
    array = np.asarray(embeddings, dtype=np.float32)[list(indices)]
    labels = tuple(int(frame.rows[index].label) for index in indices)
    sample_ids = tuple(frame.rows[index].sample_id for index in indices)
    return array, labels, sample_ids


def source_only_frame(frame: RealFeatureFrame, *, outer_target_center: str) -> RealFeatureFrame:
    """Remove the outer target and quarantined rows before source-inner work."""

    import numpy as np

    outer = _eligible(outer_target_center)
    source_centers = tuple(center for center in frame.eligible_centers if center != outer)
    indices = indices_for_centers(frame, source_centers)
    if not indices:
        raise ProtocolError("Source-inner preservation requires nonempty source rows.")
    embeddings = frame.embeddings
    if hasattr(embeddings, "detach"):
        embeddings = embeddings.detach().cpu().numpy()
    rows = tuple(
        replace(frame.rows[old_index], row_index=new_index)
        for new_index, old_index in enumerate(indices)
    )
    source_frame = RealFeatureFrame(
        embeddings=np.asarray(embeddings, dtype=np.float32)[list(indices)],
        rows=rows,
        feature_extractor=frame.feature_extractor,
        feature_cache_path=frame.feature_cache_path,
        feature_cache_hash=frame.feature_cache_hash,
        manifest_path=frame.manifest_path,
        manifest_hash=frame.manifest_hash,
        expected_feature_dim=frame.expected_feature_dim,
    )
    if outer in source_frame.eligible_centers:
        raise ProtocolError("Outer target rows remained in the source-only frame.")
    return source_frame


def row_hash(sample_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(str(value) for value in sample_ids).encode("utf-8")).hexdigest()


def identity_overlap_audit(
    frame: RealFeatureFrame,
    *,
    fit_indices: Sequence[int],
    eval_indices: Sequence[int],
    outer_target_center: str,
    inner_pseudo_target_center: str | None = None,
) -> dict[str, object]:
    fit_samples = {frame.rows[index].sample_id for index in fit_indices}
    eval_samples = {frame.rows[index].sample_id for index in eval_indices}
    fit_cases = {frame.rows[index].case_id for index in fit_indices}
    eval_cases = {frame.rows[index].case_id for index in eval_indices}
    sample_overlap = sorted(fit_samples.intersection(eval_samples))
    case_overlap = sorted(fit_cases.intersection(eval_cases))
    status = "PASS" if not sample_overlap and not case_overlap else "FAIL"
    return {
        "schema_version": "midogpp_prior_recovery_identity_overlap_v1",
        "outer_target_center": str(outer_target_center),
        "inner_pseudo_target_center": "" if inner_pseudo_target_center is None else str(inner_pseudo_target_center),
        "n_fit_samples": len(fit_samples),
        "n_eval_samples": len(eval_samples),
        "n_fit_cases": len(fit_cases),
        "n_eval_cases": len(eval_cases),
        "sample_overlap_count": len(sample_overlap),
        "case_overlap_count": len(case_overlap),
        "sample_overlap_hash": row_hash(sample_overlap),
        "case_overlap_hash": row_hash(case_overlap),
        "status": status,
    }


def assert_identity_overlap_pass(row: Mapping[str, object]) -> None:
    if row.get("status") != "PASS":
        raise ProtocolError(
            "Preservation fit/evaluation identity overlap detected for "
            f"outer={row.get('outer_target_center')} inner={row.get('inner_pseudo_target_center')}"
        )


def assert_split_excludes(split: NestedCenterSplit, *centers: str) -> None:
    forbidden = {str(center) for center in centers}
    leaked = forbidden.intersection(split.fit_centers)
    if leaked:
        raise ProtocolError(f"Held-out centers leaked into fit set: {sorted(leaked)}")
    if set(split.fit_centers).intersection(MIDOGPP_EXCLUDED_CENTERS):
        raise ProtocolError("Quarantined center leaked into fit set.")


def _eligible(center: str) -> str:
    value = str(center)
    if value not in MIDOGPP_ELIGIBLE_CENTERS:
        raise ProtocolError(f"Unknown or quarantined MIDOG++ center: {value!r}")
    return value
