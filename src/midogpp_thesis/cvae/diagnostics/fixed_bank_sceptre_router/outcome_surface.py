"""Additive outcome primitives for selection and calibration surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable

from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    legal_routing_sources,
)
from midogpp_thesis.cvae.protocol import ProtocolError

from .hashing import canonical_hash, require_sha256
from .partitions import FOLD_COUNT


OUTCOME_ROLES = frozenset({"SELECTION", "CALIBRATION"})
EXACT_B_CANDIDATE = "B::exact_equal_union"


@dataclass(frozen=True, slots=True)
class ConfusionCounts:
    tn: int
    fp: int
    fn: int
    tp: int

    def __post_init__(self) -> None:
        values = (self.tn, self.fp, self.fn, self.tp)
        if any(isinstance(value, bool) or int(value) != value or value < 0 for value in values):
            raise ProtocolError("SCEPTRE confusion counts are invalid.")

    def __add__(self, other: object) -> "ConfusionCounts":
        if not isinstance(other, ConfusionCounts):
            return NotImplemented
        return ConfusionCounts(
            self.tn + other.tn,
            self.fp + other.fp,
            self.fn + other.fn,
            self.tp + other.tp,
        )

    @property
    def row_count(self) -> int:
        return self.tn + self.fp + self.fn + self.tp

    @property
    def bacc(self) -> float:
        negative = self.tn + self.fp
        positive = self.tp + self.fn
        if negative == 0 or positive == 0:
            raise ProtocolError("SCEPTRE pooled BACC lacks one true class.")
        return 0.5 * (self.tn / negative + self.tp / positive)


@dataclass(frozen=True, slots=True)
class FamilyOutcome:
    target_center: str
    fold_ordinal: int
    role: str
    candidate_center: str
    partition_hash: str
    case_set_hash: str
    candidate_menu_hash: str
    prediction_receipt_hash: str
    confusion: ConfusionCounts
    brier_sum: float
    log_loss_sum: float
    case_count: int
    exact_b_control_receipt_hash: str | None = None
    prediction_family_seed_cells: int = 9
    outcome_hash: str = field(default="", compare=True)

    def __post_init__(self) -> None:
        target = str(self.target_center)
        candidate = str(self.candidate_center)
        role = str(self.role)
        if (
            target not in CENTERS
            or isinstance(self.fold_ordinal, bool)
            or self.fold_ordinal not in range(FOLD_COUNT)
            or role not in OUTCOME_ROLES
        ):
            raise ProtocolError("SCEPTRE family outcome scope is invalid.")
        if candidate != EXACT_B_CANDIDATE and candidate not in legal_routing_sources(target):
            raise ProtocolError("SCEPTRE family outcome candidate is outside C minus H.")
        partition_hash = require_sha256(self.partition_hash, "outcome partition")
        case_set_hash = require_sha256(self.case_set_hash, "outcome case set")
        menu_hash = _identifier(self.candidate_menu_hash, "candidate menu")
        prediction_hash = _identifier(
            self.prediction_receipt_hash, "prediction receipt"
        )
        if candidate == EXACT_B_CANDIDATE:
            control_hash = _identifier(
                self.exact_b_control_receipt_hash, "exact-B control receipt"
            )
        elif self.exact_b_control_receipt_hash is not None:
            raise ProtocolError("SCEPTRE source-family outcome carries an exact-B receipt.")
        else:
            control_hash = None
        if (
            self.case_count <= 0
            or self.confusion.row_count <= 0
            or self.prediction_family_seed_cells != 9
        ):
            raise ProtocolError("SCEPTRE family outcome replication geometry drifted.")
        if (
            not math.isfinite(self.brier_sum)
            or self.brier_sum < 0.0
            or not math.isfinite(self.log_loss_sum)
            or self.log_loss_sum < 0.0
        ):
            raise ProtocolError("SCEPTRE proper-loss sum is invalid.")
        body = {
            "schema_version": "sceptre_family_outcome_v1",
            "target_center": target,
            "fold_ordinal": self.fold_ordinal,
            "role": role,
            "candidate_center": candidate,
            "partition_hash": partition_hash,
            "case_set_hash": case_set_hash,
            "candidate_menu_hash": menu_hash,
            "prediction_receipt_hash": prediction_hash,
            "exact_b_control_receipt_hash": control_hash,
            "confusion": {
                "tn": self.confusion.tn,
                "fp": self.confusion.fp,
                "fn": self.confusion.fn,
                "tp": self.confusion.tp,
            },
            "brier_sum": self.brier_sum,
            "log_loss_sum": self.log_loss_sum,
            "case_count": self.case_count,
            "prediction_family_seed_cells": self.prediction_family_seed_cells,
            "raw_labels_persisted": False,
        }
        expected_hash = canonical_hash(body)
        if self.outcome_hash and self.outcome_hash != expected_hash:
            raise ProtocolError("SCEPTRE family outcome hash drifted.")
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "candidate_center", candidate)
        object.__setattr__(self, "partition_hash", partition_hash)
        object.__setattr__(self, "case_set_hash", case_set_hash)
        object.__setattr__(self, "candidate_menu_hash", menu_hash)
        object.__setattr__(self, "prediction_receipt_hash", prediction_hash)
        object.__setattr__(self, "exact_b_control_receipt_hash", control_hash)
        object.__setattr__(self, "outcome_hash", expected_hash)

    @property
    def scope_key(self) -> tuple[str, int, str, str, str, str]:
        return (
            self.target_center,
            self.fold_ordinal,
            self.role,
            self.partition_hash,
            self.case_set_hash,
            self.candidate_menu_hash,
        )

    @property
    def brier(self) -> float:
        return self.brier_sum / self.confusion.row_count

    @property
    def log_loss(self) -> float:
        return self.log_loss_sum / self.confusion.row_count


def pool_confusions(rows: Iterable[ConfusionCounts]) -> ConfusionCounts:
    pooled = ConfusionCounts(0, 0, 0, 0)
    observed = False
    for row in rows:
        pooled = pooled + row
        observed = True
    if not observed:
        raise ProtocolError("SCEPTRE cannot pool an empty outcome surface.")
    return pooled


def _identifier(value: object, role: str) -> str:
    text = "" if value is None else str(value)
    if not text or text.strip() != text:
        raise ProtocolError(f"SCEPTRE {role} is invalid.")
    return text


__all__ = (
    "ConfusionCounts",
    "EXACT_B_CANDIDATE",
    "FamilyOutcome",
    "OUTCOME_ROLES",
    "pool_confusions",
)
