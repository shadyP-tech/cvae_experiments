"""H-scoped raw proxy evidence transformation for historical development.

Raw rows are filtered by query and candidate center before any mean, variance,
or rank is computed.  This is the only bridge from the label-free SCEPTRE core
to the adaptive historical-utility ranker.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np

from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    TRAINING_SEEDS,
    legal_routing_sources,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.sceptre.contracts import FamilyProxyScore
from midogpp_thesis.cvae.routing.sceptre.ranking import normalized_true_midranks

from .evidence_contracts import EvidenceFeatureRow
from .hashing import canonical_hash, require_sha256


FEATURE_NAMES = (
    "proxy_energy_mean",
    "proxy_energy_replica_variance",
    "proxy_energy_midrank",
    "predictive_entropy",
    "vote_disagreement",
)


@dataclass(frozen=True, slots=True)
class RawSourceEvidence:
    query_center: str
    candidate_center: str
    training_replica_proxy_energy: Mapping[int, float]
    predictive_entropy: float
    vote_disagreement: float
    labels_consumed: bool = False
    exact_nelbo: bool = False

    def __post_init__(self) -> None:
        if (
            self.query_center not in CENTERS
            or self.candidate_center not in CENTERS
            or self.query_center == self.candidate_center
        ):
            raise ProtocolError("SCEPTRE raw evidence center geometry drifted.")
        if self.labels_consumed is not False or self.exact_nelbo is not False:
            raise ProtocolError("SCEPTRE raw evidence must be label-free and non-NELBO.")
        raw = dict(self.training_replica_proxy_energy)
        if set(raw) != set(TRAINING_SEEDS):
            raise ProtocolError("SCEPTRE raw evidence lacks three training replicas.")
        if any(not _finite(value) for value in raw.values()):
            raise ProtocolError("SCEPTRE raw proxy energy is non-finite.")
        entropy = float(self.predictive_entropy)
        disagreement = float(self.vote_disagreement)
        if (
            not math.isfinite(entropy)
            or entropy < 0.0
            or not math.isfinite(disagreement)
            or not 0.0 <= disagreement <= 1.0
        ):
            raise ProtocolError("SCEPTRE raw predictive uncertainty is invalid.")
        object.__setattr__(
            self,
            "training_replica_proxy_energy",
            MappingProxyType(
                {seed: float(raw[seed]) for seed in TRAINING_SEEDS}
            ),
        )
        object.__setattr__(self, "predictive_entropy", entropy)
        object.__setattr__(self, "vote_disagreement", disagreement)


@dataclass(frozen=True, slots=True)
class EvidenceTransformReceipt:
    role: str
    target_center: str
    input_row_count: int
    retained_row_count: int
    strict_filter: str
    feature_names: tuple[str, ...]
    retained_keys: tuple[tuple[str, str], ...]
    raw_source_receipt_hash: str
    retained_raw_hash: str
    transformed_feature_hash: str
    labels_consumed: bool
    exact_nelbo: bool
    receipt_hash: str

    def __post_init__(self) -> None:
        if (
            not self.role
            or self.target_center not in CENTERS
            or self.input_row_count < self.retained_row_count
            or self.retained_row_count <= 0
            or self.feature_names != FEATURE_NAMES
            or len(self.retained_keys) != self.retained_row_count
            or len(set(self.retained_keys)) != len(self.retained_keys)
            or tuple(sorted(self.retained_keys)) != self.retained_keys
            or self.labels_consumed is not False
            or self.exact_nelbo is not False
        ):
            raise ProtocolError("SCEPTRE evidence-transform receipt drifted.")
        raw_source_hash = require_sha256(
            self.raw_source_receipt_hash, "raw evidence source receipt"
        )
        retained_raw_hash = require_sha256(
            self.retained_raw_hash, "retained raw evidence"
        )
        transformed_feature_hash = require_sha256(
            self.transformed_feature_hash, "transformed evidence"
        )
        receipt_hash = require_sha256(self.receipt_hash, "evidence transform receipt")
        body = _receipt_body(
            role=self.role,
            target_center=self.target_center,
            input_row_count=self.input_row_count,
            retained_row_count=self.retained_row_count,
            strict_filter=self.strict_filter,
            retained_keys=self.retained_keys,
            raw_source_receipt_hash=raw_source_hash,
            retained_raw_hash=retained_raw_hash,
            transformed_feature_hash=transformed_feature_hash,
        )
        if canonical_hash(body) != receipt_hash:
            raise ProtocolError("SCEPTRE evidence-transform receipt hash drifted.")


@dataclass(frozen=True, slots=True)
class EvidenceFeatureBundle:
    """Receipt-bound raw and transformed evidence for one legal context."""

    raw_rows: tuple[RawSourceEvidence, ...]
    rows: tuple[EvidenceFeatureRow, ...]
    receipt: EvidenceTransformReceipt

    def __post_init__(self) -> None:
        raw_rows = tuple(self.raw_rows)
        rows = tuple(self.rows)
        if len(raw_rows) != self.receipt.retained_row_count or len(rows) != len(raw_rows):
            raise ProtocolError("SCEPTRE evidence bundle geometry drifted.")
        if len({(row.query_center, row.candidate_center) for row in raw_rows}) != len(
            raw_rows
        ):
            raise ProtocolError("SCEPTRE evidence bundle contains duplicate raw rows.")
        if tuple((row.query_center, row.candidate_center) for row in rows) != tuple(
            sorted((row.query_center, row.candidate_center) for row in raw_rows)
        ):
            raise ProtocolError("SCEPTRE evidence bundle row ordering drifted.")
        if self.receipt.retained_keys != tuple(
            (row.query_center, row.candidate_center) for row in rows
        ):
            raise ProtocolError("SCEPTRE evidence receipt key binding drifted.")
        replayed = _transform_rows(raw_rows)
        if rows != replayed:
            raise ProtocolError("SCEPTRE transformed evidence does not replay from raw rows.")
        retained_raw_hash, transformed_feature_hash = _content_hashes(raw_rows, rows)
        if (
            retained_raw_hash != self.receipt.retained_raw_hash
            or transformed_feature_hash != self.receipt.transformed_feature_hash
        ):
            raise ProtocolError("SCEPTRE evidence bundle content binding drifted.")
        object.__setattr__(self, "raw_rows", raw_rows)
        object.__setattr__(self, "rows", rows)


def build_outer_development_evidence(
    rows: Iterable[RawSourceEvidence],
    *,
    outer_target: str,
    raw_source_receipt_hash: str,
) -> EvidenceFeatureBundle:
    """Delete q==H/e==H first, then compute all evidence transformations."""

    raw = tuple(rows)
    target = _target(outer_target)
    retained = tuple(
        row
        for row in raw
        if row.query_center != target and row.candidate_center != target
    )
    expected = {
        (query, candidate)
        for query in CENTERS
        if query != target
        for candidate in CENTERS
        if candidate not in {target, query}
    }
    return _transform(
        retained,
        target=target,
        expected_keys=expected,
        role="OUTER_DEVELOPMENT",
        input_row_count=len(raw),
        strict_filter="q!=H_and_e!=H_before_mean_variance_and_rank",
        raw_source_receipt_hash=raw_source_receipt_hash,
    )


def build_target_prediction_evidence(
    rows: Iterable[RawSourceEvidence],
    *,
    target_center: str,
    raw_source_receipt_hash: str,
) -> EvidenceFeatureBundle:
    """Transform exactly the target H by C-minus-H prediction grid."""

    raw = tuple(rows)
    target = _target(target_center)
    retained = tuple(
        row
        for row in raw
        if row.query_center == target and row.candidate_center != target
    )
    expected = {(target, candidate) for candidate in legal_routing_sources(target)}
    return _transform(
        retained,
        target=target,
        expected_keys=expected,
        role="TARGET_PREDICTION",
        input_row_count=len(raw),
        strict_filter="q==H_and_e!=H_before_mean_variance_and_rank",
        raw_source_receipt_hash=raw_source_receipt_hash,
    )


def build_nested_lodo_evidence(
    outer: EvidenceFeatureBundle,
    *,
    held_center: str,
) -> tuple[EvidenceFeatureBundle, EvidenceFeatureBundle]:
    """Re-delete q/e==K and recompute every transform for nested LODO."""

    if not isinstance(outer, EvidenceFeatureBundle):
        raise ProtocolError("SCEPTRE nested evidence requires a bound outer bundle.")
    if outer.receipt.role != "OUTER_DEVELOPMENT":
        raise ProtocolError("SCEPTRE nested evidence requires the outer development role.")
    target = outer.receipt.target_center
    held = _target(held_center)
    if held == target:
        raise ProtocolError("SCEPTRE nested held center equals outer H.")
    train_rows = tuple(
        row
        for row in outer.raw_rows
        if row.query_center != held and row.candidate_center != held
    )
    validation_rows = tuple(
        row
        for row in outer.raw_rows
        if row.query_center == held and row.candidate_center != held
    )
    training_centers = tuple(center for center in CENTERS if center not in {target, held})
    train_expected = {
        (query, candidate)
        for query in training_centers
        for candidate in training_centers
        if candidate != query
    }
    validation_expected = {
        (held, candidate) for candidate in training_centers
    }
    train = _transform(
        train_rows,
        target=target,
        expected_keys=train_expected,
        role=f"NESTED_LODO_TRAIN_K_{held}",
        input_row_count=len(outer.raw_rows),
        strict_filter=(
            f"q!=H_and_e!=H_then_q!={held}_and_e!={held}_before_all_transforms"
        ),
        raw_source_receipt_hash=outer.receipt.raw_source_receipt_hash,
    )
    validation = _transform(
        validation_rows,
        target=target,
        expected_keys=validation_expected,
        role=f"NESTED_LODO_VALIDATION_K_{held}",
        input_row_count=len(outer.raw_rows),
        strict_filter=(
            f"q=={held}_and_e!=H_and_e!={held}_before_all_transforms"
        ),
        raw_source_receipt_hash=outer.receipt.raw_source_receipt_hash,
    )
    return train, validation


def _transform(
    rows: tuple[RawSourceEvidence, ...],
    *,
    target: str,
    expected_keys: set[tuple[str, str]],
    role: str,
    input_row_count: int,
    strict_filter: str,
    raw_source_receipt_hash: str,
) -> EvidenceFeatureBundle:
    by_key = {(row.query_center, row.candidate_center): row for row in rows}
    if len(by_key) != len(rows) or set(by_key) != expected_keys:
        raise ProtocolError("SCEPTRE raw evidence q/e grid is incomplete or duplicated.")
    canonical_raw = tuple(by_key[key] for key in sorted(by_key))
    canonical_transformed = _transform_rows(canonical_raw)
    retained_raw_hash, transformed_feature_hash = _content_hashes(
        canonical_raw, canonical_transformed
    )
    retained_keys = tuple(sorted(by_key))
    body = _receipt_body(
        role=role,
        target_center=target,
        input_row_count=input_row_count,
        retained_row_count=len(rows),
        strict_filter=strict_filter,
        retained_keys=retained_keys,
        raw_source_receipt_hash=raw_source_receipt_hash,
        retained_raw_hash=retained_raw_hash,
        transformed_feature_hash=transformed_feature_hash,
    )
    receipt = EvidenceTransformReceipt(
        role=role,
        target_center=target,
        input_row_count=input_row_count,
        retained_row_count=len(rows),
        strict_filter=strict_filter,
        feature_names=FEATURE_NAMES,
        retained_keys=retained_keys,
        raw_source_receipt_hash=raw_source_receipt_hash,
        retained_raw_hash=retained_raw_hash,
        transformed_feature_hash=transformed_feature_hash,
        labels_consumed=False,
        exact_nelbo=False,
        receipt_hash=canonical_hash(body),
    )
    return EvidenceFeatureBundle(
        raw_rows=canonical_raw,
        rows=canonical_transformed,
        receipt=receipt,
    )


def _transform_rows(
    rows: tuple[RawSourceEvidence, ...],
) -> tuple[EvidenceFeatureRow, ...]:
    by_key = {(row.query_center, row.candidate_center): row for row in rows}
    if len(by_key) != len(rows):
        raise ProtocolError("SCEPTRE raw evidence contains duplicate q/e rows.")
    family_scores = {
        key: FamilyProxyScore(
            target_center=key[0],
            source_center=key[1],
            training_replica_scores=dict(row.training_replica_proxy_energy),
        )
        for key, row in by_key.items()
    }
    ranks: dict[tuple[str, str], float] = {}
    for query in sorted({key[0] for key in by_key}):
        values = {
            candidate: float(family_scores[(query, candidate)].mean_proxy_energy)
            for key_query, candidate in by_key
            if key_query == query
        }
        query_ranks = normalized_true_midranks(
            values,
            candidate_sources=tuple(sorted(values)),
            lower_is_better=True,
        )
        ranks.update(
            {
                (query, candidate): value
                for candidate, value in query_ranks.items()
            }
        )
    transformed: list[EvidenceFeatureRow] = []
    for key in sorted(by_key):
        row = by_key[key]
        score = family_scores[key]
        replica = np.asarray(
            [score.training_replica_scores[seed] for seed in TRAINING_SEEDS],
            dtype=np.float64,
        )
        transformed.append(
            EvidenceFeatureRow(
                query_center=key[0],
                candidate_center=key[1],
                feature_names=FEATURE_NAMES,
                values=(
                    float(score.mean_proxy_energy),
                    float(np.var(replica, dtype=np.float64)),
                    float(ranks[key]),
                    float(row.predictive_entropy),
                    float(row.vote_disagreement),
                ),
                labels_consumed=False,
            )
        )
    return tuple(transformed)


def _receipt_body(
    *,
    role: str,
    target_center: str,
    input_row_count: int,
    retained_row_count: int,
    strict_filter: str,
    retained_keys: tuple[tuple[str, str], ...],
    raw_source_receipt_hash: str,
    retained_raw_hash: str,
    transformed_feature_hash: str,
) -> dict[str, object]:
    return {
        "schema_version": "sceptre_evidence_transform_receipt_v1",
        "role": role,
        "target_center": target_center,
        "input_row_count": input_row_count,
        "retained_row_count": retained_row_count,
        "strict_filter": strict_filter,
        "retained_keys": [list(key) for key in retained_keys],
        "feature_names": list(FEATURE_NAMES),
        "raw_source_receipt_hash": raw_source_receipt_hash,
        "retained_raw_hash": retained_raw_hash,
        "transformed_feature_hash": transformed_feature_hash,
        "labels_consumed": False,
        "exact_nelbo": False,
    }


def _content_hashes(
    raw_rows: tuple[RawSourceEvidence, ...],
    rows: tuple[EvidenceFeatureRow, ...],
) -> tuple[str, str]:
    retained_raw_payload = [
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
        for row in raw_rows
    ]
    transformed_payload = [
        {
            "query_center": row.query_center,
            "candidate_center": row.candidate_center,
            "feature_names": list(row.feature_names),
            "values": list(row.values),
            "feature_scope": row.feature_scope,
            "labels_consumed": False,
        }
        for row in rows
    ]
    return (
        canonical_hash(
            {
                "schema_version": "sceptre_retained_raw_evidence_v1",
                "rows": retained_raw_payload,
            }
        ),
        canonical_hash(
            {
                "schema_version": "sceptre_transformed_evidence_v1",
                "rows": transformed_payload,
            }
        ),
    )


def _target(value: object) -> str:
    target = str(value)
    if target not in CENTERS:
        raise ProtocolError("SCEPTRE evidence transform target is unknown.")
    return target


def _finite(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


__all__ = (
    "EvidenceFeatureBundle",
    "EvidenceTransformReceipt",
    "FEATURE_NAMES",
    "RawSourceEvidence",
    "build_outer_development_evidence",
    "build_nested_lodo_evidence",
    "build_target_prediction_evidence",
)
