"""Allow-listed v1 fold preparation reused by non-adoptive v2 studies.

The v2 studies intentionally reuse the already-audited nested classifier,
identity-overlap, and source-fit PCA preparation kernel.  They record that
lineage explicitly and own every downstream metric, checkpoint, and decision
schema themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping

from ....real_features.classifier_reference.real_feature_frame import (
    RealFeatureFrame,
    load_midogpp_real_feature_frame,
)
from ....real_features.classifier_reference.protocol import ProtocolError
from ....real_features.classifier_reference.artifacts import stable_hash
from ..prior_recovery_classifier import (
    SOURCE_INNER_CLASSIFIER_GRID_HASH,
    source_inner_classifier_specs,
)
from ..prior_recovery_runtime_cache import (
    FRAME_CACHE_SCHEMA,
    FRAME_INDEX_SCHEMA,
    FeatureFrameCache,
)
from ..prior_recovery_schema import NESTED_REAL_REFERENCE_SCHEMA
from ..prior_recovery_source_preparation import (
    PreparedSourceInnerFold,
    prepare_source_inner_fold,
)
from ..splits import source_only_frame
from .validation_common import StudyTimingRecorder


EMBEDDED_PREPARATION_SCHEMA = "midogpp_source_inner_study_embedded_v1_preparation_v2"


@dataclass(frozen=True)
class PreparedOuterStudyFolds:
    outer_target_center: str
    source_frame: RealFeatureFrame
    folds: tuple[PreparedSourceInnerFold, ...]
    nested_reference_rows: tuple[Mapping[str, object], ...]
    nested_tuning_rows: tuple[Mapping[str, object], ...]
    identity_rows: tuple[Mapping[str, object], ...]

    @property
    def inner_centers(self) -> tuple[str, ...]:
        return tuple(fold.inner for fold in self.folds)


def load_study_frame(config: object) -> RealFeatureFrame:
    """Load only the registered MIDOG++ manifest/cache lineage."""

    frame = load_midogpp_real_feature_frame(
        manifest_path=Path(getattr(config, "manifest_path")),
        feature_cache_path=Path(getattr(config, "feature_cache_path")),
        expected_feature_dim=int(getattr(config, "expected_feature_dim")),
    )
    heldouts = tuple(str(value) for value in getattr(config, "heldout_centers"))
    missing = set(heldouts).difference(frame.eligible_centers)
    if missing:
        raise ProtocolError(f"Study held-out centers absent from feature frame: {sorted(missing)}")
    return frame


def prepare_outer_study_folds(
    config: object,
    *,
    frame: RealFeatureFrame,
    outer_target_center: str,
    preparation_protocol_hash: str,
    frame_cache: FeatureFrameCache,
    timings: StudyTimingRecorder,
) -> PreparedOuterStudyFolds:
    """Prepare all inner pseudo-target folds after removing outer center H."""

    outer = str(outer_target_center)
    source_frame = source_only_frame(frame, outer_target_center=outer)
    specs = source_inner_classifier_specs(classifier_seed=23)
    folds: list[PreparedSourceInnerFold] = []
    nested: list[Mapping[str, object]] = []
    tuning: list[Mapping[str, object]] = []
    identity: list[Mapping[str, object]] = []
    for inner in source_frame.eligible_centers:
        prepared, nested_row, audit_row = prepare_source_inner_fold(
            pca_dim=int(getattr(config, "pca_dim")),
            frame=source_frame,
            outer=outer,
            inner=str(inner),
            candidate_specs=specs,
            preparation_protocol_hash=str(preparation_protocol_hash),
            preparation_code_version=str(getattr(config, "study_version")),
            frame_cache=frame_cache,
            timings=timings,
        )
        folds.append(prepared)
        nested.append(dict(nested_row))
        tuning.extend(dict(row) for row in prepared.selection.candidate_rows)
        identity.append(dict(audit_row))
    expected = tuple(center for center in frame.eligible_centers if center != outer)
    if tuple(fold.inner for fold in folds) != expected:
        raise ProtocolError("Source-inner study preparation has incomplete inner-center coverage.")
    return PreparedOuterStudyFolds(
        outer_target_center=outer,
        source_frame=source_frame,
        folds=tuple(folds),
        nested_reference_rows=tuple(nested),
        nested_tuning_rows=tuple(tuning),
        identity_rows=tuple(identity),
    )


def embedded_v1_preparation_lineage() -> dict[str, object]:
    """Describe and hash the exact v1 preparation components embedded by v2."""

    modules = {
        "prior_recovery_source_preparation": _module_sha256(
            "prior_recovery_source_preparation.py"
        ),
        "prior_recovery_runtime_cache": _module_sha256(
            "prior_recovery_runtime_cache.py"
        ),
        "prior_recovery_classifier": _module_sha256("prior_recovery_classifier.py"),
    }
    payload: dict[str, object] = {
        "schema_version": EMBEDDED_PREPARATION_SCHEMA,
        "used_for_v2_preparation_only": True,
        "symbols": [
            "prepare_source_inner_fold",
            "FeatureFrameCache.fit_or_load",
            "source_inner_classifier_specs",
        ],
        "embedded_schema_ids": {
            "nested_real_reference": NESTED_REAL_REFERENCE_SCHEMA,
            "feature_frame_cache": FRAME_CACHE_SCHEMA,
            "feature_frame_index": FRAME_INDEX_SCHEMA,
        },
        "classifier_grid_hash": SOURCE_INNER_CLASSIFIER_GRID_HASH,
        "component_file_sha256": modules,
        "v1_recipe_or_decision_semantics_reused": False,
    }
    payload["lineage_hash"] = stable_hash(payload)
    return payload


def _module_sha256(filename: str) -> str:
    path = Path(__file__).resolve().parents[1] / filename
    if not path.is_file():
        raise ProtocolError(f"Embedded preparation component is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()
