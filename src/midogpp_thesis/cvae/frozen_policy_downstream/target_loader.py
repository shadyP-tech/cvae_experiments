"""Label-sealed target-cache loading and one-way scoring-label access."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from os import PathLike
from pathlib import Path
from typing import Mapping

import numpy as np

from ...common.hashing import stable_hash
from ...data.contract.stage70_target_evaluation.contracts import (
    CANONICAL_MANIFEST_SHA256,
    EXPECTED_TEST_ROWS,
    EXPECTED_TEST_ROWS_BY_CENTER,
)
from ...data.features.stage70_test_cache.io import ValidatedStage70TestCache
from ..expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ..protocol import ProtocolError
from .contracts import TargetFrame, array_sha256
from .prediction import PersistedPredictionPass
from .prediction_seal import (
    PredictionSealBinding,
    VerifiedPredictionArtifact,
    load_canonical_prediction_seal_binding,
    verify_persisted_prediction_artifact,
)


@dataclass(frozen=True)
class _OpenedScoringLabels:
    """Immutable labels bound to the exact prediction transaction they may score."""

    evaluation_row_ids: tuple[str, ...]
    labels: tuple[int, ...]
    label_manifest_sha256: str
    authorization_binding_hash: str
    phase_01_sha256: str
    prediction_index_sha256: str
    prediction_arrays_sha256: str
    prediction_seal_sha256: str
    phase_02_sha256: str

    def __post_init__(self) -> None:
        if (
            len(self.evaluation_row_ids) != EXPECTED_TEST_ROWS
            or len(self.labels) != EXPECTED_TEST_ROWS
            or len(set(self.evaluation_row_ids)) != EXPECTED_TEST_ROWS
            or set(self.labels) != {0, 1}
            or len(self.label_manifest_sha256) != 64
            or len(self.authorization_binding_hash) != 16
            or any(
                len(value) != 64
                for value in (
                    self.phase_01_sha256,
                    self.prediction_index_sha256,
                    self.prediction_arrays_sha256,
                    self.prediction_seal_sha256,
                    self.phase_02_sha256,
                )
            )
        ):
            raise ProtocolError("Stage-70 opened scoring-label capability is malformed.")


def load_label_sealed_target_frames(
    cache: ValidatedStage70TestCache,
) -> dict[str, TargetFrame]:
    """Load target embeddings/neutral identities without exposing labels."""

    if not isinstance(cache, ValidatedStage70TestCache):
        raise ProtocolError("Stage-70 prediction requires a validated target cache.")
    if not cache.root.is_dir() or cache.root.is_symlink():
        raise ProtocolError("Stage-70 validated target-cache root is unsafe.")
    summary = dict(cache.summary)
    if (
        summary.get("status") != "PASS"
        or summary.get("manifest_sha256") != CANONICAL_MANIFEST_SHA256
        or summary.get("row_count") != EXPECTED_TEST_ROWS
        or summary.get("rows_by_center") != dict(EXPECTED_TEST_ROWS_BY_CENTER)
    ):
        raise ProtocolError("Stage-70 validated target-cache summary drifted.")
    frames: dict[str, TargetFrame] = {}
    global_ids: list[str] = []
    global_indices: list[int] = []
    expected_shards = summary.get("shard_sha256_by_center")
    if not isinstance(expected_shards, Mapping) or set(expected_shards) != set(CENTERS):
        raise ProtocolError("Stage-70 validated target-cache shard identities drifted.")
    for center in CENTERS:
        shard = cache.load_center(center)
        if shard.shard_sha256 != expected_shards[center]:
            raise ProtocolError(
                f"Stage-70 target-cache shard changed after validation: {center}."
            )
        embeddings = np.array(shard.embeddings, dtype=np.float32, copy=True)
        row_ids = shard.evaluation_row_ids
        row_indices = shard.contract_row_indices
        case_ids = shard.case_ids
        if (
            len(row_ids) != EXPECTED_TEST_ROWS_BY_CENTER[center]
            or row_indices != tuple(sorted(row_indices))
        ):
            raise ProtocolError(
                f"Stage-70 target-cache identity coverage drifted for center {center}."
            )
        metadata_payload = [dict(row) for row in shard.metadata]
        row_order_hash = stable_hash(list(row_ids))
        content_hash = stable_hash(
            {
                "target_center": center,
                "metadata": metadata_payload,
                "embedding_sha256": array_sha256(embeddings),
                "shard_sha256": shard.shard_sha256,
            }
        )
        frames[center] = TargetFrame(
            target_center=center,
            evaluation_row_ids=row_ids,
            contract_row_indices=row_indices,
            case_ids=case_ids,
            embeddings=embeddings,
            row_order_hash=row_order_hash,
            content_hash=content_hash,
        )
        global_ids.extend(row_ids)
        global_indices.extend(row_indices)
    if (
        len(global_ids) != EXPECTED_TEST_ROWS
        or len(global_ids) != len(set(global_ids))
        or len(global_indices) != len(set(global_indices))
    ):
        raise ProtocolError("Stage-70 target-cache global row coverage drifted.")
    return frames


def open_scoring_labels_after_prediction_seal(
    sealed: PersistedPredictionPass,
    *,
    manifest_path: str | Path,
    expected_manifest_sha256: str,
    final_authorization_root: str | PathLike[str],
    target_cache_root: str | PathLike[str],
) -> _OpenedScoringLabels:
    """Open labels only after canonical auth/cache and exact disk-seal checks."""

    if not isinstance(sealed, PersistedPredictionPass):
        raise ProtocolError(
            "Stage-70 label access requires a persisted prediction capability."
        )
    binding = load_canonical_prediction_seal_binding(
        final_authorization_root=final_authorization_root,
        target_cache_root=target_cache_root,
        scoring_manifest_path=manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    return _open_scoring_labels_with_expected_binding(
        sealed,
        manifest_path=manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_binding=binding,
    )


def _open_scoring_labels_with_expected_binding(
    sealed: PersistedPredictionPass,
    *,
    manifest_path: str | Path,
    expected_manifest_sha256: str,
    expected_binding: PredictionSealBinding,
) -> _OpenedScoringLabels:
    """Strict test seam; production callers must use the canonical public path."""

    verified = verify_persisted_prediction_artifact(
        sealed,
        expected_binding=expected_binding,
    )
    source = Path(manifest_path)
    if not source.is_file() or source.is_symlink():
        raise ProtocolError("Stage-70 scoring manifest is missing or a symlink.")
    observed_sha = _sha256_file(source)
    if (
        observed_sha != expected_manifest_sha256
        or observed_sha != expected_binding.scoring_manifest_sha256
        or observed_sha != verified.phase_01_binding.get("scoring_manifest_sha256")
    ):
        raise ProtocolError("Stage-70 scoring-manifest SHA-256 drifted.")
    return _labels_from_verified_artifact(verified, source, observed_sha)


def _labels_from_verified_artifact(
    verified: VerifiedPredictionArtifact,
    source: Path,
    observed_sha: str,
) -> _OpenedScoringLabels:
    expected: dict[int, tuple[str, str, str]] = {}
    for cell in verified.cells:
        for row_id, row_index, case_id in zip(
            cell.evaluation_row_ids,
            cell.contract_row_indices,
            cell.case_ids,
            strict=True,
        ):
            identity = (str(row_id), str(case_id), cell.target_center)
            previous = expected.setdefault(int(row_index), identity)
            if previous != identity:
                raise ProtocolError("Stage-70 prediction rows disagree on target identity.")
    if len(expected) != EXPECTED_TEST_ROWS:
        raise ProtocolError("Stage-70 scoring boundary lacks all canonical target rows.")
    labels_by_index: dict[int, int] = {}
    try:
        with source.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"label", "case_id", "center", "split"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ProtocolError("Stage-70 scoring manifest lacks required fields.")
            for row_index, row in enumerate(reader):
                identity = expected.get(row_index)
                if identity is None:
                    continue
                row_id, case_id, center = identity
                if (
                    str(row["case_id"]) != case_id
                    or str(row["center"]) != center
                    or str(row["split"]) != "test"
                ):
                    raise ProtocolError("Stage-70 scoring identity join drifted.")
                label = int(row["label"])
                if label not in (0, 1):
                    raise ProtocolError("Stage-70 scoring label is not binary.")
                labels_by_index[row_index] = label
    except (OSError, ValueError, TypeError) as exc:
        raise ProtocolError(f"Cannot open Stage-70 scoring manifest: {source}.") from exc
    if set(labels_by_index) != set(expected):
        raise ProtocolError("Stage-70 scoring manifest does not cover prediction rows.")
    if _sha256_file(source) != observed_sha:
        raise ProtocolError("Stage-70 scoring manifest changed while labels were opened.")
    ordered_indices = sorted(expected)
    evaluation_row_ids = tuple(expected[index][0] for index in ordered_indices)
    labels = tuple(labels_by_index[index] for index in ordered_indices)
    return _OpenedScoringLabels(
        evaluation_row_ids=evaluation_row_ids,
        labels=labels,
        label_manifest_sha256=observed_sha,
        authorization_binding_hash=verified.authorization_binding_hash,
        phase_01_sha256=verified.phase_01_sha256,
        prediction_index_sha256=verified.prediction_index_sha256,
        prediction_arrays_sha256=verified.prediction_arrays_sha256,
        prediction_seal_sha256=verified.prediction_seal_sha256,
        phase_02_sha256=verified.phase_02_sha256,
    )


def _sha256_file(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ProtocolError("Stage-70 scoring manifest is missing or a symlink.")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ProtocolError(f"Cannot hash Stage-70 scoring manifest: {path}.") from exc
    return digest.hexdigest()


__all__ = (
    "load_label_sealed_target_frames",
    "open_scoring_labels_after_prediction_seal",
)
