"""Streaming, capability-gated access to consumed validation labels.

The functions in this module never materialize a manifest-wide label column.
They first reject every row that is not named by the appropriate durable
prediction seal, then and only then inspect that row's label field.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Sequence

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .contracts import (
    ACTION_IDS,
    EXPECTED_MANIFEST_SHA256,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
    OpenedLabelVector,
    ValidationRowIdentity,
    development_queries,
    row_identity_hash,
)
from .seals import (
    AllActionTargetPredictionSeal,
    DevelopmentPredictionSeal,
    TargetPredictionSeal,
)


_REQUIRED_MANIFEST_FIELDS = frozenset(
    {"sample_id", "case_id", "center", "split", "label"}
)


def open_development_labels(
    manifest_path: str | Path,
    evaluation_rows: Sequence[ValidationRowIdentity],
    *,
    seal: DevelopmentPredictionSeal,
    all_action_target_seal: AllActionTargetPredictionSeal,
    all_action_target_seal_path: str | Path,
    prediction_index_path: str | Path,
    prediction_arrays_path: str | Path,
    target_prediction_index_path: str | Path,
    target_prediction_arrays_path: str | Path,
    expected_manifest_sha256: str = EXPECTED_MANIFEST_SHA256,
) -> OpenedLabelVector:
    """Open ``q != H`` labels only after both global prediction passes seal."""

    if not isinstance(seal, DevelopmentPredictionSeal):
        raise ProtocolError(
            "Development labels require a complete all-action prediction seal."
        )
    if not isinstance(all_action_target_seal, AllActionTargetPredictionSeal):
        raise ProtocolError(
            "Development labels require a complete pre-label all-target-action "
            "prediction seal."
        )
    _verify_development_capability(seal)
    _verify_all_action_target_capability(
        all_action_target_seal,
        development_seal=seal,
    )
    _verify_persisted_all_action_target_seal(
        all_action_target_seal_path,
        seal=all_action_target_seal,
    )
    _verify_persisted_prediction_files(
        prediction_index_path=prediction_index_path,
        prediction_arrays_path=prediction_arrays_path,
        expected_index_sha256=seal.prediction_index_sha256,
        expected_arrays_sha256=seal.prediction_arrays_sha256,
        phase="development",
    )
    _verify_persisted_prediction_files(
        prediction_index_path=target_prediction_index_path,
        prediction_arrays_path=target_prediction_arrays_path,
        expected_index_sha256=all_action_target_seal.prediction_index_sha256,
        expected_arrays_sha256=all_action_target_seal.prediction_arrays_sha256,
        phase="all-target-action",
    )
    if (
        seal.validation_manifest_sha256 != expected_manifest_sha256
        or all_action_target_seal.validation_manifest_sha256
        != expected_manifest_sha256
    ):
        raise ProtocolError("Development label capability binds another manifest.")

    rows = tuple(evaluation_rows)
    _require_unique_requested_rows(rows)
    if any(
        row.partition_role != "evaluation" or row.center == seal.outer_target
        for row in rows
    ):
        raise ProtocolError(
            "Development label access is limited to q != H evaluation rows."
        )
    expected_queries = development_queries(seal.outer_target)
    rows_by_query = {
        query: tuple(row for row in rows if row.center == query)
        for query in expected_queries
    }
    if any(not query_rows for query_rows in rows_by_query.values()):
        raise ProtocolError("Development label request lacks a pseudo-target query.")
    for query in expected_queries:
        observed_ids = tuple(row.sample_id for row in rows_by_query[query])
        if observed_ids != seal.evaluation_row_ids_by_query[query]:
            raise ProtocolError(
                "Development label request differs from sealed prediction coverage."
            )
        if row_identity_hash(rows_by_query[query]) != (
            seal.evaluation_row_identity_hash_by_query[query]
        ):
            raise ProtocolError(
                "Development label request row identities differ from the seal."
            )

    labels = _stream_requested_labels(
        manifest_path,
        rows,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    for query in expected_queries:
        indices = [index for index, row in enumerate(rows) if row.center == query]
        if {labels[index] for index in indices} != {0, 1}:
            raise ProtocolError(
                f"Development pseudo-target {query} lacks binary evaluation support."
            )
    return _opened_label_vector(
        outer_target=seal.outer_target,
        phase="development",
        rows=rows,
        labels=labels,
        manifest_sha256=expected_manifest_sha256,
        prediction_seal_hash=seal.seal_hash,
    )


def open_target_labels(
    manifest_path: str | Path,
    evaluation_rows: Sequence[ValidationRowIdentity],
    *,
    seal: TargetPredictionSeal,
    prediction_index_path: str | Path,
    prediction_arrays_path: str | Path,
    expected_manifest_sha256: str = EXPECTED_MANIFEST_SHA256,
) -> OpenedLabelVector:
    """Open target-H labels only after selected and control arrays are sealed."""

    if not isinstance(seal, TargetPredictionSeal):
        raise ProtocolError(
            "Target labels require a selected-plus-control target prediction seal."
        )
    _verify_target_capability(seal)
    _verify_persisted_prediction_files(
        prediction_index_path=prediction_index_path,
        prediction_arrays_path=prediction_arrays_path,
        expected_index_sha256=seal.prediction_index_sha256,
        expected_arrays_sha256=seal.prediction_arrays_sha256,
        phase="target",
    )
    if seal.validation_manifest_sha256 != expected_manifest_sha256:
        raise ProtocolError("Target label capability binds another manifest.")

    rows = tuple(evaluation_rows)
    _require_unique_requested_rows(rows)
    if any(
        row.partition_role != "evaluation" or row.center != seal.outer_target
        for row in rows
    ):
        raise ProtocolError("Target label access is limited to target-H evaluation rows.")
    if tuple(row.sample_id for row in rows) != seal.evaluation_row_ids:
        raise ProtocolError("Target label request differs from sealed prediction coverage.")
    if row_identity_hash(rows) != seal.evaluation_row_identity_hash:
        raise ProtocolError("Target label request row identities differ from the seal.")

    labels = _stream_requested_labels(
        manifest_path,
        rows,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    if set(labels) != {0, 1}:
        raise ProtocolError("Target-H evaluation rows lack binary label support.")
    return _opened_label_vector(
        outer_target=seal.outer_target,
        phase="target",
        rows=rows,
        labels=labels,
        manifest_sha256=expected_manifest_sha256,
        prediction_seal_hash=seal.seal_hash,
    )


def _stream_requested_labels(
    manifest_path: str | Path,
    rows: tuple[ValidationRowIdentity, ...],
    *,
    expected_manifest_sha256: str,
) -> tuple[int, ...]:
    path = Path(manifest_path)
    _assert_sha256(path, expected_manifest_sha256)
    expected_by_index = {row.manifest_row_index: row for row in rows}
    labels_by_index: dict[int, int] = {}

    try:
        handle = path.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise ProtocolError(f"Cannot open dense residual label manifest: {path}.") from exc
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not _REQUIRED_MANIFEST_FIELDS.issubset(
            reader.fieldnames
        ):
            raise ProtocolError("Dense residual manifest lacks required scoring fields.")
        for manifest_row_index, raw in enumerate(reader):
            expected = expected_by_index.get(manifest_row_index)
            if expected is None:
                # This branch intentionally precedes the only access to
                # ``raw['label']`` below.  Support rows, target-H rows during
                # development, and all train/test/excluded rows stay unopened.
                continue
            observed_identity = (
                str(raw.get("sample_id", "")),
                str(raw.get("case_id", "")),
                str(raw.get("center", "")),
                str(raw.get("split", "")),
            )
            expected_identity = (
                expected.sample_id,
                expected.case_id,
                expected.center,
                expected.split,
            )
            if observed_identity != expected_identity:
                raise ProtocolError("Dense residual scoring-manifest identity drifted.")
            labels_by_index[manifest_row_index] = _binary_label(raw["label"])

    if set(labels_by_index) != set(expected_by_index):
        raise ProtocolError("Dense residual label coverage differs from sealed rows.")
    return tuple(labels_by_index[row.manifest_row_index] for row in rows)


def _opened_label_vector(
    *,
    outer_target: str,
    phase: str,
    rows: tuple[ValidationRowIdentity, ...],
    labels: tuple[int, ...],
    manifest_sha256: str,
    prediction_seal_hash: str,
) -> OpenedLabelVector:
    vector_hash = stable_hash(
        {
            "outer_target": outer_target,
            "phase": phase,
            "row_identity_hash": row_identity_hash(rows),
            "labels": list(labels),
            "manifest_sha256": manifest_sha256,
            "prediction_seal_hash": prediction_seal_hash,
        }
    )
    return OpenedLabelVector(
        outer_target=outer_target,
        phase=phase,
        rows=rows,
        labels=labels,
        manifest_sha256=manifest_sha256,
        prediction_seal_hash=prediction_seal_hash,
        label_vector_hash=vector_hash,
    )


def _verify_development_capability(seal: DevelopmentPredictionSeal) -> None:
    if seal.seal_hash != stable_hash(seal._unhashed_payload()):
        raise ProtocolError("Development prediction capability no longer verifies.")
    # Force evaluation of complete properties after any illicit in-memory
    # mutation through ``object.__setattr__``.
    expected_cells = (
        len(ACTION_IDS)
        * len(development_queries(seal.outer_target))
        * len(TRAINING_SEEDS)
        * len(GENERATION_SEEDS)
    )
    if seal.cell_count != expected_cells:
        raise ProtocolError("Development prediction capability is incomplete.")


def _verify_all_action_target_capability(
    seal: AllActionTargetPredictionSeal,
    *,
    development_seal: DevelopmentPredictionSeal,
) -> None:
    seal.verify_complete()
    if (
        seal.config_contract_hash != development_seal.config_contract_hash
        or seal.action_library_hash != development_seal.action_library_hash
        or seal.support_partition_lock_hash
        != development_seal.support_partition_lock_hash
        or seal.validation_cache_binding_hash
        != development_seal.validation_cache_binding_hash
        or seal.validation_manifest_sha256
        != development_seal.validation_manifest_sha256
    ):
        raise ProtocolError(
            "Pre-label all-target-action seal disagrees with development capability."
        )
    for query in development_queries(development_seal.outer_target):
        if (
            seal.evaluation_row_ids_by_target[query]
            != development_seal.evaluation_row_ids_by_query[query]
            or seal.evaluation_row_identity_hash_by_target[query]
            != development_seal.evaluation_row_identity_hash_by_query[query]
        ):
            raise ProtocolError(
                "Pre-label target identities disagree with development queries."
            )


def _verify_persisted_all_action_target_seal(
    path: str | Path,
    *,
    seal: AllActionTargetPredictionSeal,
) -> None:
    seal_path = Path(path)
    try:
        payload = json.loads(seal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(
            "Pre-label all-target-action seal is not durably persisted."
        ) from exc
    if payload != seal.to_payload():
        raise ProtocolError(
            "Persisted pre-label all-target-action seal differs from its capability."
        )


def _verify_target_capability(seal: TargetPredictionSeal) -> None:
    if seal.seal_hash != stable_hash(seal._unhashed_payload()):
        raise ProtocolError("Target prediction capability no longer verifies.")
    if seal.cell_count != 18:
        raise ProtocolError("Target prediction capability lacks both logical arms.")


def _require_unique_requested_rows(rows: tuple[ValidationRowIdentity, ...]) -> None:
    if not rows:
        raise ProtocolError("Dense residual label request is empty.")
    sample_ids = [row.sample_id for row in rows]
    manifest_indices = [row.manifest_row_index for row in rows]
    if len(sample_ids) != len(set(sample_ids)) or len(manifest_indices) != len(
        set(manifest_indices)
    ):
        raise ProtocolError("Dense residual label request duplicates row identities.")


def _binary_label(value: object) -> int:
    try:
        numeric = float(str(value))
    except ValueError as exc:
        raise ProtocolError("Dense residual requested scoring label is not binary.") from exc
    if numeric not in (0.0, 1.0):
        raise ProtocolError("Dense residual requested scoring label is outside {0,1}.")
    return int(numeric)


def _assert_sha256(path: Path, expected: str) -> None:
    if not path.is_file():
        raise ProtocolError("Dense residual scoring manifest is missing.")
    digest = _sha256_file(path)
    if digest != expected:
        raise ProtocolError("Dense residual scoring-manifest SHA-256 drifted.")


def _verify_persisted_prediction_files(
    *,
    prediction_index_path: str | Path,
    prediction_arrays_path: str | Path,
    expected_index_sha256: str,
    expected_arrays_sha256: str,
    phase: str,
) -> None:
    index_path = Path(prediction_index_path)
    arrays_path = Path(prediction_arrays_path)
    if not index_path.is_file() or not arrays_path.is_file():
        raise ProtocolError(
            f"Dense residual {phase} prediction capability is not persisted."
        )
    observed_index = _sha256_file(index_path)
    observed_arrays = _sha256_file(arrays_path)
    if (
        observed_index != expected_index_sha256
        or observed_arrays != expected_arrays_sha256
    ):
        raise ProtocolError(
            f"Dense residual {phase} persisted prediction bytes drifted."
        )


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


__all__ = ("open_development_labels", "open_target_labels")
