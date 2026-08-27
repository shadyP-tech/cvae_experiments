"""Hash-bound label-free evidence loader for the historical utility surface.

The immutable source-inner artifact persisted predictions before its labels were
opened.  This module reuses only that label-free prediction packet.  Pairwise
utility labels remain confined to :mod:`development_surface` and never enter
the feature constructor.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
    legal_routing_sources,
)
from midogpp_thesis.cvae.protocol import ProtocolError

from .evidence_builder import RawSourceEvidence
from .hashing import canonical_hash, file_sha256
from .identity import (
    EXPECTED_SOURCE_CLASSIFIER_FITS_SHA256,
    EXPECTED_SOURCE_CLASSIFIER_FIT_ROWS,
    EXPECTED_SOURCE_EVALUATION_ROWS_SHA256,
    EXPECTED_SOURCE_EVALUATION_ROW_COUNT,
    EXPECTED_SOURCE_PREDICTION_ARRAY_FILE_SHA256,
    EXPECTED_SOURCE_PREDICTION_INDEX_SHA256,
)
from .source_inner_authorization import (
    CLASSIFIER_FITS_MEMBER,
    EVALUATION_ROWS_MEMBER,
    PREDICTION_ARRAY_MEMBER,
    PREDICTION_INDEX_MEMBER,
    SourceInnerReuseReceipt,
    validate_source_inner_reuse_receipt,
)


EXPECTED_PROBABILITY_ARRAY_SHA256 = (
    "39a74d9148310d1cee361e58fcdfdced49f93af4417068e9df600a5c8b77a911"
)
EXPECTED_PREDICTION_ARRAY_SHA256 = (
    "e786fd957accb4bfe544c128e778f9828e2158e9d2c8055f7df414835270e369"
)


@dataclass(frozen=True, slots=True)
class SourceFitRow:
    prediction_array_row: int
    source_center: str
    training_seed: int
    generation_seed: int
    source_stream_id: str

    def __post_init__(self) -> None:
        if (
            self.prediction_array_row < 0
            or self.source_center not in CENTERS
            or self.training_seed not in TRAINING_SEEDS
            or self.generation_seed not in GENERATION_SEEDS
            or not self.source_stream_id
        ):
            raise ProtocolError("SCEPTRE source-inner fit index drifted.")


@dataclass(frozen=True, slots=True)
class SourceEvaluationRow:
    row_ordinal: int
    sample_id: str
    case_id: str
    center: str

    def __post_init__(self) -> None:
        if (
            self.row_ordinal < 0
            or not self.sample_id
            or not self.case_id
            or self.center not in CENTERS
        ):
            raise ProtocolError("SCEPTRE source-inner evaluation index drifted.")


@dataclass(frozen=True, slots=True)
class PredictionSurfaceReceipt:
    prediction_array_file_sha256: str
    prediction_index_sha256: str
    classifier_fits_sha256: str
    evaluation_rows_sha256: str
    fit_count: int
    evaluation_row_count: int
    probability_array_sha256: str
    prediction_array_sha256: str
    labels_stored: bool
    receipt_hash: str

    def __post_init__(self) -> None:
        if (
            self.prediction_array_file_sha256
            != EXPECTED_SOURCE_PREDICTION_ARRAY_FILE_SHA256
            or self.prediction_index_sha256 != EXPECTED_SOURCE_PREDICTION_INDEX_SHA256
            or self.classifier_fits_sha256 != EXPECTED_SOURCE_CLASSIFIER_FITS_SHA256
            or self.evaluation_rows_sha256 != EXPECTED_SOURCE_EVALUATION_ROWS_SHA256
            or self.fit_count != EXPECTED_SOURCE_CLASSIFIER_FIT_ROWS
            or self.evaluation_row_count != EXPECTED_SOURCE_EVALUATION_ROW_COUNT
            or self.probability_array_sha256 != EXPECTED_PROBABILITY_ARRAY_SHA256
            or self.prediction_array_sha256 != EXPECTED_PREDICTION_ARRAY_SHA256
            or self.labels_stored is not False
            or any(
                len(value) != 64
                for value in (
                    self.prediction_array_file_sha256,
                    self.prediction_index_sha256,
                    self.classifier_fits_sha256,
                    self.evaluation_rows_sha256,
                    self.probability_array_sha256,
                    self.prediction_array_sha256,
                    self.receipt_hash,
                )
            )
        ):
            raise ProtocolError("SCEPTRE prediction-surface receipt drifted.")
        body = _surface_receipt_body(
            prediction_array_file_sha256=self.prediction_array_file_sha256,
            prediction_index_sha256=self.prediction_index_sha256,
            classifier_fits_sha256=self.classifier_fits_sha256,
            evaluation_rows_sha256=self.evaluation_rows_sha256,
            fit_count=self.fit_count,
            evaluation_row_count=self.evaluation_row_count,
            probability_array_sha256=self.probability_array_sha256,
            prediction_array_sha256=self.prediction_array_sha256,
        )
        if self.receipt_hash != canonical_hash(body):
            raise ProtocolError("SCEPTRE prediction-surface receipt hash drifted.")


@dataclass(frozen=True, slots=True)
class SourceInnerPredictionSurface:
    prob_pos: np.ndarray
    y_pred: np.ndarray
    fit_rows: tuple[SourceFitRow, ...]
    evaluation_rows: tuple[SourceEvaluationRow, ...]
    receipt: PredictionSurfaceReceipt

    def __post_init__(self) -> None:
        expected_shape = (
            EXPECTED_SOURCE_CLASSIFIER_FIT_ROWS,
            EXPECTED_SOURCE_EVALUATION_ROW_COUNT,
        )
        if (
            not isinstance(self.prob_pos, np.ndarray)
            or not isinstance(self.y_pred, np.ndarray)
            or self.prob_pos.shape != expected_shape
            or self.y_pred.shape != expected_shape
            or self.prob_pos.dtype != np.dtype("float32")
            or self.y_pred.dtype != np.dtype("uint8")
            or self.prob_pos.flags.writeable
            or self.y_pred.flags.writeable
            or len(self.fit_rows) != expected_shape[0]
            or len(self.evaluation_rows) != expected_shape[1]
        ):
            raise ProtocolError("SCEPTRE prediction-surface geometry drifted.")
        if (
            _array_sha256(self.prob_pos) != self.receipt.probability_array_sha256
            or _array_sha256(self.y_pred) != self.receipt.prediction_array_sha256
            or tuple(row.prediction_array_row for row in self.fit_rows)
            != tuple(range(expected_shape[0]))
            or tuple(row.row_ordinal for row in self.evaluation_rows)
            != tuple(range(expected_shape[1]))
        ):
            raise ProtocolError("SCEPTRE prediction-surface content binding drifted.")


@dataclass(frozen=True, slots=True)
class RawEvidencePacket:
    role: str
    target_center: str
    source_surface_receipt_hash: str
    rows: tuple[RawSourceEvidence, ...]
    strict_filter: str
    labels_consumed: bool
    packet_hash: str

    def __post_init__(self) -> None:
        if (
            self.role not in {"OUTER_DEVELOPMENT_RAW", "TARGET_PREDICTION_RAW"}
            or self.target_center not in CENTERS
            or len(self.source_surface_receipt_hash) != 64
            or not self.rows
            or not self.strict_filter
            or self.labels_consumed is not False
        ):
            raise ProtocolError("SCEPTRE raw-evidence packet drifted.")
        body = _raw_packet_body(
            role=self.role,
            target_center=self.target_center,
            source_surface_receipt_hash=self.source_surface_receipt_hash,
            rows=self.rows,
            strict_filter=self.strict_filter,
        )
        if self.packet_hash != canonical_hash(body):
            raise ProtocolError("SCEPTRE raw-evidence packet hash drifted.")


def load_source_inner_prediction_surface(
    artifact_root: str | Path,
    *,
    receipt: SourceInnerReuseReceipt,
) -> SourceInnerPredictionSurface:
    """Load and validate the exact label-free prediction packet."""

    receipt = validate_source_inner_reuse_receipt(receipt)
    expected_hashes = {
        PREDICTION_ARRAY_MEMBER: EXPECTED_SOURCE_PREDICTION_ARRAY_FILE_SHA256,
        PREDICTION_INDEX_MEMBER: EXPECTED_SOURCE_PREDICTION_INDEX_SHA256,
        CLASSIFIER_FITS_MEMBER: EXPECTED_SOURCE_CLASSIFIER_FITS_SHA256,
        EVALUATION_ROWS_MEMBER: EXPECTED_SOURCE_EVALUATION_ROWS_SHA256,
    }
    receipt_hashes = {
        PREDICTION_ARRAY_MEMBER: receipt.prediction_array_file_sha256,
        PREDICTION_INDEX_MEMBER: receipt.prediction_index_sha256,
        CLASSIFIER_FITS_MEMBER: receipt.classifier_fits_sha256,
        EVALUATION_ROWS_MEMBER: receipt.evaluation_rows_sha256,
    }
    if receipt_hashes != expected_hashes:
        raise ProtocolError("SCEPTRE prediction evidence receipt is not byte-exact.")
    root = Path(artifact_root)
    members = {name: _member(root, name) for name in expected_hashes}
    observed_hashes = {name: file_sha256(path) for name, path in members.items()}
    if observed_hashes != expected_hashes:
        raise ProtocolError("SCEPTRE label-free prediction evidence bytes drifted.")

    index = _load_index(members[PREDICTION_INDEX_MEMBER])
    fit_rows = _load_fit_rows(members[CLASSIFIER_FITS_MEMBER])
    evaluation_rows = _load_evaluation_rows(members[EVALUATION_ROWS_MEMBER])
    prob_pos, y_pred = _load_arrays(members[PREDICTION_ARRAY_MEMBER], index=index)
    receipt_body = _surface_receipt_body(
        prediction_array_file_sha256=observed_hashes[PREDICTION_ARRAY_MEMBER],
        prediction_index_sha256=observed_hashes[PREDICTION_INDEX_MEMBER],
        classifier_fits_sha256=observed_hashes[CLASSIFIER_FITS_MEMBER],
        evaluation_rows_sha256=observed_hashes[EVALUATION_ROWS_MEMBER],
        fit_count=len(fit_rows),
        evaluation_row_count=len(evaluation_rows),
        probability_array_sha256=_array_sha256(prob_pos),
        prediction_array_sha256=_array_sha256(y_pred),
    )
    surface_receipt = PredictionSurfaceReceipt(
        prediction_array_file_sha256=observed_hashes[PREDICTION_ARRAY_MEMBER],
        prediction_index_sha256=observed_hashes[PREDICTION_INDEX_MEMBER],
        classifier_fits_sha256=observed_hashes[CLASSIFIER_FITS_MEMBER],
        evaluation_rows_sha256=observed_hashes[EVALUATION_ROWS_MEMBER],
        fit_count=len(fit_rows),
        evaluation_row_count=len(evaluation_rows),
        probability_array_sha256=_array_sha256(prob_pos),
        prediction_array_sha256=_array_sha256(y_pred),
        labels_stored=False,
        receipt_hash=canonical_hash(receipt_body),
    )
    return SourceInnerPredictionSurface(
        prob_pos=prob_pos,
        y_pred=y_pred,
        fit_rows=fit_rows,
        evaluation_rows=evaluation_rows,
        receipt=surface_receipt,
    )


def build_outer_raw_evidence(
    surface: SourceInnerPredictionSurface,
    *,
    outer_target: str,
) -> RawEvidencePacket:
    """Delete q/e==H before computing any evidence statistic."""

    target = _target(outer_target)
    pairs = tuple(
        (query, candidate)
        for query in CENTERS
        if query != target
        for candidate in CENTERS
        if candidate not in {target, query}
    )
    return _packet(
        surface,
        role="OUTER_DEVELOPMENT_RAW",
        target_center=target,
        pairs=pairs,
        strict_filter="select_q!=H_and_e!=H_before_probability_or_vote_reduction",
    )


def build_target_raw_evidence(
    surface: SourceInnerPredictionSurface,
    *,
    target_center: str,
) -> RawEvidencePacket:
    """Build the label-free H by exact-C-minus-H routing packet."""

    target = _target(target_center)
    pairs = tuple((target, candidate) for candidate in legal_routing_sources(target))
    return _packet(
        surface,
        role="TARGET_PREDICTION_RAW",
        target_center=target,
        pairs=pairs,
        strict_filter="select_q==H_and_e!=H_before_probability_or_vote_reduction",
    )


def _packet(
    surface: SourceInnerPredictionSurface,
    *,
    role: str,
    target_center: str,
    pairs: tuple[tuple[str, str], ...],
    strict_filter: str,
) -> RawEvidencePacket:
    if not isinstance(surface, SourceInnerPredictionSurface):
        raise ProtocolError("SCEPTRE raw evidence requires a typed prediction surface.")
    fit_by_cell = {
        (row.source_center, row.training_seed, row.generation_seed): row
        for row in surface.fit_rows
    }
    eval_by_center = {
        center: np.asarray(
            [row.row_ordinal for row in surface.evaluation_rows if row.center == center],
            dtype=np.int64,
        )
        for center in CENTERS
    }
    if any(indices.size == 0 for indices in eval_by_center.values()):
        raise ProtocolError("SCEPTRE prediction surface lacks a center evaluation slice.")
    rows = tuple(
        _summarize_pair(
            surface,
            query=query,
            candidate=candidate,
            fit_by_cell=fit_by_cell,
            evaluation_indices=eval_by_center[query],
        )
        for query, candidate in pairs
    )
    body = _raw_packet_body(
        role=role,
        target_center=target_center,
        source_surface_receipt_hash=surface.receipt.receipt_hash,
        rows=rows,
        strict_filter=strict_filter,
    )
    return RawEvidencePacket(
        role=role,
        target_center=target_center,
        source_surface_receipt_hash=surface.receipt.receipt_hash,
        rows=rows,
        strict_filter=strict_filter,
        labels_consumed=False,
        packet_hash=canonical_hash(body),
    )


def _summarize_pair(
    surface: SourceInnerPredictionSurface,
    *,
    query: str,
    candidate: str,
    fit_by_cell: Mapping[tuple[str, int, int], SourceFitRow],
    evaluation_indices: np.ndarray,
) -> RawSourceEvidence:
    fit_ordinals = np.asarray(
        [
            fit_by_cell[(candidate, training_seed, generation_seed)].prediction_array_row
            for training_seed in TRAINING_SEEDS
            for generation_seed in GENERATION_SEEDS
        ],
        dtype=np.int64,
    )
    probabilities = np.asarray(
        surface.prob_pos[np.ix_(fit_ordinals, evaluation_indices)],
        dtype=np.float64,
    )
    votes = np.asarray(
        surface.y_pred[np.ix_(fit_ordinals, evaluation_indices)],
        dtype=np.float64,
    )
    if probabilities.shape[0] != 9 or probabilities.shape != votes.shape:
        raise ProtocolError("SCEPTRE source-family evidence geometry drifted.")
    entropy = _binary_entropy(probabilities)
    replica_energy = {
        training_seed: float(
            np.mean(
                entropy[index * len(GENERATION_SEEDS) : (index + 1) * len(GENERATION_SEEDS)],
                dtype=np.float64,
            )
        )
        for index, training_seed in enumerate(TRAINING_SEEDS)
    }
    family_mean_probability = np.mean(probabilities, axis=0, dtype=np.float64)
    predictive_entropy = float(
        np.mean(_binary_entropy(family_mean_probability), dtype=np.float64)
    )
    vote_fraction = np.mean(votes, axis=0, dtype=np.float64)
    vote_disagreement = float(
        np.mean(4.0 * vote_fraction * (1.0 - vote_fraction), dtype=np.float64)
    )
    return RawSourceEvidence(
        query_center=query,
        candidate_center=candidate,
        training_replica_proxy_energy=replica_energy,
        predictive_entropy=predictive_entropy,
        vote_disagreement=vote_disagreement,
        labels_consumed=False,
        exact_nelbo=False,
    )


def _load_index(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("Cannot parse SCEPTRE prediction index.") from exc
    if not isinstance(raw, Mapping):
        raise ProtocolError("SCEPTRE prediction index is not a mapping.")
    index = dict(raw)
    expected = {
        "allowed_array_keys": ["y_pred", "prob_pos"],
        "array_member": PREDICTION_ARRAY_MEMBER,
        "eval_labels_available_to_fit_or_predict": False,
        "eval_row_count": EXPECTED_SOURCE_EVALUATION_ROW_COUNT,
        "fit_count": EXPECTED_SOURCE_CLASSIFIER_FIT_ROWS,
        "labels_stored": False,
        "prediction_dtype": "uint8",
        "prediction_shape": [
            EXPECTED_SOURCE_CLASSIFIER_FIT_ROWS,
            EXPECTED_SOURCE_EVALUATION_ROW_COUNT,
        ],
        "probability_dtype": "float32",
        "probability_shape": [
            EXPECTED_SOURCE_CLASSIFIER_FIT_ROWS,
            EXPECTED_SOURCE_EVALUATION_ROW_COUNT,
        ],
        "probability_array_sha256": EXPECTED_PROBABILITY_ARRAY_SHA256,
        "prediction_array_sha256": EXPECTED_PREDICTION_ARRAY_SHA256,
    }
    if any(index.get(key) != value for key, value in expected.items()):
        raise ProtocolError("SCEPTRE prediction index semantics drifted.")
    return index


def _load_arrays(
    path: Path,
    *,
    index: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"prob_pos", "y_pred"}:
                raise ProtocolError("SCEPTRE prediction archive keys drifted.")
            prob_pos = np.asarray(archive["prob_pos"], dtype=np.float32).copy()
            y_pred = np.asarray(archive["y_pred"], dtype=np.uint8).copy()
    except (OSError, ValueError) as exc:
        raise ProtocolError("Cannot load SCEPTRE prediction arrays.") from exc
    expected_shape = (
        EXPECTED_SOURCE_CLASSIFIER_FIT_ROWS,
        EXPECTED_SOURCE_EVALUATION_ROW_COUNT,
    )
    if (
        prob_pos.shape != expected_shape
        or y_pred.shape != expected_shape
        or not np.all(np.isfinite(prob_pos))
        or np.any((prob_pos < 0.0) | (prob_pos > 1.0))
        or np.any((y_pred != 0) & (y_pred != 1))
        or _array_sha256(prob_pos) != index.get("probability_array_sha256")
        or _array_sha256(y_pred) != index.get("prediction_array_sha256")
    ):
        raise ProtocolError("SCEPTRE prediction array content drifted.")
    prob_pos.flags.writeable = False
    y_pred.flags.writeable = False
    return prob_pos, y_pred


def _load_fit_rows(path: Path) -> tuple[SourceFitRow, ...]:
    required = {
        "prediction_array_row",
        "source_center",
        "training_seed",
        "generation_seed",
        "source_stream_id",
        "eval_labels_available_to_fit_or_predict",
        "seed_selection_performed",
    }
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ProtocolError("SCEPTRE classifier-fit columns drifted.")
            rows = tuple(
                SourceFitRow(
                    prediction_array_row=int(row["prediction_array_row"]),
                    source_center=str(row["source_center"]),
                    training_seed=int(row["training_seed"]),
                    generation_seed=int(row["generation_seed"]),
                    source_stream_id=str(row["source_stream_id"]),
                )
                for row in reader
                if _require_false(
                    row["eval_labels_available_to_fit_or_predict"],
                    "fit-label availability",
                )
                and _require_false(row["seed_selection_performed"], "seed selection")
            )
    except (OSError, UnicodeDecodeError, TypeError, ValueError) as exc:
        raise ProtocolError("Cannot parse SCEPTRE classifier-fit index.") from exc
    expected_grid = {
        (center, training_seed, generation_seed)
        for center in CENTERS
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
    }
    observed_grid = {
        (row.source_center, row.training_seed, row.generation_seed) for row in rows
    }
    if (
        len(rows) != EXPECTED_SOURCE_CLASSIFIER_FIT_ROWS
        or {row.prediction_array_row for row in rows}
        != set(range(EXPECTED_SOURCE_CLASSIFIER_FIT_ROWS))
        or observed_grid != expected_grid
        or len({row.source_stream_id for row in rows}) != len(rows)
    ):
        raise ProtocolError("SCEPTRE classifier-fit inventory drifted.")
    return tuple(sorted(rows, key=lambda row: row.prediction_array_row))


def _load_evaluation_rows(path: Path) -> tuple[SourceEvaluationRow, ...]:
    required = {"row_ordinal", "sample_id", "case_id", "center", "label_present"}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ProtocolError("SCEPTRE evaluation-row columns drifted.")
            rows = tuple(
                SourceEvaluationRow(
                    row_ordinal=int(row["row_ordinal"]),
                    sample_id=str(row["sample_id"]),
                    case_id=str(row["case_id"]),
                    center=str(row["center"]),
                )
                for row in reader
                if _require_false(row["label_present"], "stored evaluation label")
            )
    except (OSError, UnicodeDecodeError, TypeError, ValueError) as exc:
        raise ProtocolError("Cannot parse SCEPTRE evaluation-row index.") from exc
    if (
        len(rows) != EXPECTED_SOURCE_EVALUATION_ROW_COUNT
        or tuple(row.row_ordinal for row in rows)
        != tuple(range(EXPECTED_SOURCE_EVALUATION_ROW_COUNT))
        or len({row.sample_id for row in rows}) != len(rows)
        or set(row.center for row in rows) != set(CENTERS)
    ):
        raise ProtocolError("SCEPTRE evaluation-row inventory drifted.")
    return rows


def _raw_packet_body(
    *,
    role: str,
    target_center: str,
    source_surface_receipt_hash: str,
    rows: tuple[RawSourceEvidence, ...],
    strict_filter: str,
) -> dict[str, object]:
    return {
        "schema_version": "sceptre_raw_evidence_packet_v1",
        "role": role,
        "target_center": target_center,
        "source_surface_receipt_hash": source_surface_receipt_hash,
        "strict_filter": strict_filter,
        "rows": [
            {
                "query_center": row.query_center,
                "candidate_center": row.candidate_center,
                "training_replica_proxy_energy": {
                    str(seed): row.training_replica_proxy_energy[seed]
                    for seed in TRAINING_SEEDS
                },
                "predictive_entropy": row.predictive_entropy,
                "vote_disagreement": row.vote_disagreement,
                "labels_consumed": False,
                "exact_nelbo": False,
            }
            for row in rows
        ],
        "labels_consumed": False,
        "exact_nelbo": False,
    }


def _surface_receipt_body(
    *,
    prediction_array_file_sha256: str,
    prediction_index_sha256: str,
    classifier_fits_sha256: str,
    evaluation_rows_sha256: str,
    fit_count: int,
    evaluation_row_count: int,
    probability_array_sha256: str,
    prediction_array_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": "sceptre_source_inner_prediction_surface_receipt_v1",
        "member_sha256": {
            PREDICTION_ARRAY_MEMBER: prediction_array_file_sha256,
            PREDICTION_INDEX_MEMBER: prediction_index_sha256,
            CLASSIFIER_FITS_MEMBER: classifier_fits_sha256,
            EVALUATION_ROWS_MEMBER: evaluation_rows_sha256,
        },
        "fit_count": fit_count,
        "evaluation_row_count": evaluation_row_count,
        "probability_array_sha256": probability_array_sha256,
        "prediction_array_sha256": prediction_array_sha256,
        "labels_stored": False,
        "eval_labels_available_to_fit_or_predict": False,
    }


def _binary_entropy(values: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float64)
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    result = -(clipped * np.log(clipped) + (1.0 - clipped) * np.log1p(-clipped))
    if not np.all(np.isfinite(result)):
        raise ProtocolError("SCEPTRE entropy evidence is non-finite.")
    return result


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(
        json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _require_false(value: object, role: str) -> bool:
    if str(value).strip().lower() not in {"false", "0"}:
        raise ProtocolError(f"SCEPTRE {role} must remain false.")
    return True


def _target(value: object) -> str:
    target = str(value)
    if target not in CENTERS:
        raise ProtocolError("SCEPTRE evidence target is unknown.")
    return target


def _member(root: Path, relative: str) -> Path:
    candidate = root.joinpath(*relative.split("/"))
    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ProtocolError(f"SCEPTRE evidence member is absent: {relative}.") from exc
    if candidate.is_symlink() or resolved_root not in resolved.parents or not resolved.is_file():
        raise ProtocolError(f"SCEPTRE evidence member is unsafe: {relative}.")
    return resolved


__all__ = (
    "PredictionSurfaceReceipt",
    "RawEvidencePacket",
    "SourceEvaluationRow",
    "SourceFitRow",
    "SourceInnerPredictionSurface",
    "build_outer_raw_evidence",
    "build_target_raw_evidence",
    "load_source_inner_prediction_surface",
)
