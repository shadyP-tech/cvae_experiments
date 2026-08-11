"""Source-OOF exact-response contracts for disagreement regret."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from ...protocol import ProtocolError
from ._validation import _canonical_id
from .hashing import canonical_sha256, is_sha256


RESPONSE_SEMANTICS = "source_oof_exact_bacc_gain_vs_control"


@dataclass(frozen=True)
class CaseActionResponseRow:
    query_id: str
    case_id: str
    action_id: str
    source_id: str | None
    exact_bacc_gain_vs_control: float
    exact_regret_from_case_best: float
    disagreement_count: int
    # Query-wide class totals used by exact BACC, repeated on each case/action row.
    positive_class_count: int
    negative_class_count: int
    response_semantics: str = RESPONSE_SEMANTICS
    response_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("query_id", "case_id", "action_id"):
            object.__setattr__(self, name, _canonical_id(getattr(self, name), name=name))
        if self.source_id is not None:
            object.__setattr__(
                self, "source_id", _canonical_id(self.source_id, name="source_id")
            )
        gain = float(self.exact_bacc_gain_vs_control)
        regret = float(self.exact_regret_from_case_best)
        if (
            not math.isfinite(gain)
            or not -1.0 <= gain <= 1.0
            or not math.isfinite(regret)
            or not 0.0 <= regret <= 2.0
        ):
            raise ProtocolError("Exact response rows must contain finite nonnegative regret.")
        object.__setattr__(self, "exact_bacc_gain_vs_control", gain)
        object.__setattr__(self, "exact_regret_from_case_best", max(0.0, regret))
        if self.response_semantics != RESPONSE_SEMANTICS:
            raise ProtocolError("Smooth/proper-loss responses cannot enter exact regret fitting.")
        for name in (
            "disagreement_count",
            "positive_class_count",
            "negative_class_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ProtocolError(f"{name} must be a nonnegative integer.")
            object.__setattr__(self, name, int(value))
        if self.positive_class_count <= 0 or self.negative_class_count <= 0:
            raise ProtocolError("Exact BACC response queries require both classes.")
        if self.disagreement_count > self.positive_class_count + self.negative_class_count:
            raise ProtocolError("Response disagreement_count exceeds its query sample count.")
        payload = {
            "schema_version": "midogpp_disagreement_regret_response_v1",
            "query_id": self.query_id,
            "case_id": self.case_id,
            "action_id": self.action_id,
            "source_id": self.source_id,
            "exact_bacc_gain_vs_control": gain,
            "exact_regret_from_case_best": max(0.0, regret),
            "disagreement_count": int(self.disagreement_count),
            "positive_class_count": int(self.positive_class_count),
            "negative_class_count": int(self.negative_class_count),
            "response_semantics": self.response_semantics,
        }
        object.__setattr__(self, "response_hash", canonical_sha256(payload))

    @property
    def row_key(self) -> tuple[str, str, str]:
        return (self.query_id, self.case_id, self.action_id)


@dataclass(frozen=True)
class ExactRegretSurface:
    rows: tuple[CaseActionResponseRow, ...]
    feature_surface_hash: str
    label_surface_hash: str
    prediction_seal_hash: str
    development_context_hash: str
    response_semantics: str = RESPONSE_SEMANTICS
    surface_hash: str = field(init=False)

    def __post_init__(self) -> None:
        rows = tuple(self.rows)
        if any(not isinstance(row, CaseActionResponseRow) for row in rows):
            raise ProtocolError("Exact response surfaces require typed response rows.")
        if not rows or tuple(sorted(rows, key=lambda row: row.row_key)) != rows:
            raise ProtocolError("Exact response rows must be nonempty and canonically ordered.")
        if len({row.row_key for row in rows}) != len(rows):
            raise ProtocolError("Exact response surfaces contain duplicate rows.")
        if self.response_semantics != RESPONSE_SEMANTICS:
            raise ProtocolError("Exact response surface semantics drifted.")
        if not all(
            is_sha256(value)
            for value in (
                self.feature_surface_hash,
                self.label_surface_hash,
                self.prediction_seal_hash,
                self.development_context_hash,
            )
        ):
            raise ProtocolError("Exact response lineage requires full SHA-256 identities.")
        rows_by_case: dict[tuple[str, str], list[CaseActionResponseRow]] = {}
        for row in rows:
            rows_by_case.setdefault((row.query_id, row.case_id), []).append(row)
        for case_rows in rows_by_case.values():
            case_best = max(
                0.0,
                max(row.exact_bacc_gain_vs_control for row in case_rows),
            )
            if any(
                not math.isclose(
                    row.exact_regret_from_case_best,
                    case_best - row.exact_bacc_gain_vs_control,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                )
                for row in case_rows
            ):
                raise ProtocolError("Exact case-best regret drifted from action gains.")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(
            self,
            "surface_hash",
            canonical_sha256(
                {
                    "schema_version": "midogpp_disagreement_regret_response_surface_v1",
                    "row_hashes": [row.response_hash for row in rows],
                    "feature_surface_hash": self.feature_surface_hash,
                    "label_surface_hash": self.label_surface_hash,
                    "prediction_seal_hash": self.prediction_seal_hash,
                    "development_context_hash": self.development_context_hash,
                    "response_semantics": self.response_semantics,
                }
            ),
        )


__all__ = (
    "RESPONSE_SEMANTICS",
    "CaseActionResponseRow",
    "ExactRegretSurface",
)
