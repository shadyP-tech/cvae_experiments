"""Immutable seal products for pre-support and support-only decisions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from ...protocol import ProtocolError
from .constants import GEOMETRY_IDS, MIDOGPP_CENTERS, U_ACTION_ID, candidate_sources, geometry_action_id
from .contracts import MethodDecision
from .execution_support import decision_payload
from .hashing import canonical_hash, finite, require_sha256


PRE_SUPPORT_GEOMETRY_METHODS = ("U", "G", "R", "P")


@dataclass(frozen=True, order=True)
class FoldDecisionSeal:
    target_center: str
    fold_ordinal: int
    method_id: str
    geometry_id: str | None
    case_count: int
    decision_hash: str

    def __post_init__(self) -> None:
        if self.target_center not in MIDOGPP_CENTERS or self.fold_ordinal not in range(5) or self.case_count <= 0:
            raise ProtocolError("Fold decision seal identity is invalid.")
        if self.method_id == "B":
            if self.geometry_id is not None:
                raise ProtocolError("B fold seal cannot carry a geometry.")
        elif self.method_id in (*PRE_SUPPORT_GEOMETRY_METHODS, "S_y"):
            if self.geometry_id not in GEOMETRY_IDS:
                raise ProtocolError("Geometry method fold seal lacks its geometry.")
        else:
            raise ProtocolError("Fold decision seal has an unknown method.")
        require_sha256(self.decision_hash, "decision_hash")

    @property
    def key(self) -> tuple[str, int, str, str | None]:
        return self.target_center, self.fold_ordinal, self.method_id, self.geometry_id

    def to_payload(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class PreSupportDecisionProducts:
    decisions: tuple[MethodDecision, ...]
    fold_seals: tuple[FoldDecisionSeal, ...]
    pre_support_decision_hashes: Mapping[tuple[str, int, str, str | None], str]
    pre_support_seal_hash: str
    permutation_provenance_hash: str
    partition_hash: str
    protocol_contract_hash: str
    probability_surface_hash: str

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_actionability_pre_support_decisions_v1",
            "decisions": [decision_payload(row) for row in self.decisions],
            "fold_seals": [row.to_payload() for row in self.fold_seals],
            "pre_support_seal_hash": self.pre_support_seal_hash,
            "permutation_provenance_hash": self.permutation_provenance_hash,
            "partition_hash": self.partition_hash,
            "protocol_contract_hash": self.protocol_contract_hash,
            "probability_surface_hash": self.probability_surface_hash,
            "evaluation_labels_used": False,
        }


@dataclass(frozen=True, order=True)
class FoldActionScore:
    target_center: str
    fold_ordinal: int
    geometry_id: str
    action_id: str
    support_exact_bacc: float

    def __post_init__(self) -> None:
        actions = (
            U_ACTION_ID,
            *(geometry_action_id(self.geometry_id, source) for source in candidate_sources(self.target_center)),
        ) if self.target_center in MIDOGPP_CENTERS and self.geometry_id in GEOMETRY_IDS else ()
        if self.fold_ordinal not in range(5) or self.action_id not in actions:
            raise ProtocolError("Fold support action score identity is invalid.")
        value = finite(self.support_exact_bacc, "support_exact_bacc")
        if not 0.0 <= value <= 1.0:
            raise ProtocolError("Support exact BACC must lie in [0, 1].")


@dataclass(frozen=True)
class SupportFoldProduct:
    target_center: str
    fold_ordinal: int
    decisions: tuple[MethodDecision, ...]
    action_scores: tuple[FoldActionScore, ...]
    geometry_seals: tuple[FoldDecisionSeal, ...]
    support_label_surface_hash: str
    fold_hash: str
    support_product_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.target_center not in MIDOGPP_CENTERS or self.fold_ordinal not in range(5):
            raise ProtocolError("Support fold product identity is invalid.")
        if tuple(row.geometry_id for row in self.geometry_seals) != GEOMETRY_IDS:
            raise ProtocolError("Support fold product needs A0 and A1 seals.")
        require_sha256(self.support_label_surface_hash, "support_label_surface_hash")
        require_sha256(self.fold_hash, "fold_hash")
        object.__setattr__(self, "support_product_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_actionability_support_fold_product_v1",
            "target_center": self.target_center,
            "fold_ordinal": self.fold_ordinal,
            "fold_hash": self.fold_hash,
            "support_label_surface_hash": self.support_label_surface_hash,
            "decisions": [decision_payload(row) for row in self.decisions],
            "action_scores": [dict(row.__dict__) for row in self.action_scores],
            "geometry_seals": [row.to_payload() for row in self.geometry_seals],
            "evaluation_labels_used": False,
            "per_case_bacc_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "support_product_hash": self.support_product_hash}


@dataclass(frozen=True)
class DecisionProducts:
    decisions: tuple[MethodDecision, ...]
    support_action_scores: tuple[FoldActionScore, ...]
    support_product_hashes: tuple[tuple[str, int, str], ...]
    pre_support_decision_hashes: Mapping[tuple[str, int, str, str | None], str]
    pre_support_seal_hash: str
    all_decision_hashes: Mapping[tuple[str, int, str, str | None], str]
    all_decisions_seal_hash: str
    permutation_provenance_hash: str
    partition_hash: str
    protocol_contract_hash: str
    probability_surface_hash: str
    decision_products_hash: str = field(init=False)

    def __post_init__(self) -> None:
        rows, pre, all_hashes = tuple(self.decisions), dict(self.pre_support_decision_hashes), dict(self.all_decision_hashes)
        if not rows or any(row.evaluation_labels_used for row in rows):
            raise ProtocolError("Decision products must be complete and pre-evaluation.")
        if len(pre) != 405 or len(all_hashes) != 495 or not set(pre).issubset(all_hashes):
            raise ProtocolError("Decision seal topology must contain 405 pre-support and 495 total cells.")
        support_hashes = tuple(self.support_product_hashes)
        expected_support = tuple(
            (target, fold) for target in MIDOGPP_CENTERS for fold in range(5)
        )
        if tuple(row[:2] for row in support_hashes) != expected_support:
            raise ProtocolError("Support product hashes must cover all 45 folds canonically.")
        for _target, _fold, digest in support_hashes:
            require_sha256(digest, "support_product_hash")
        for value, name in (
            (self.pre_support_seal_hash, "pre_support_seal_hash"),
            (self.all_decisions_seal_hash, "all_decisions_seal_hash"),
            (self.permutation_provenance_hash, "permutation_provenance_hash"),
            (self.partition_hash, "partition_hash"),
            (self.protocol_contract_hash, "protocol_contract_hash"),
            (self.probability_surface_hash, "probability_surface_hash"),
        ):
            require_sha256(value, name)
        object.__setattr__(self, "decisions", tuple(sorted(rows)))
        object.__setattr__(self, "pre_support_decision_hashes", MappingProxyType(pre))
        object.__setattr__(self, "all_decision_hashes", MappingProxyType(all_hashes))
        object.__setattr__(self, "support_product_hashes", support_hashes)
        object.__setattr__(self, "decision_products_hash", canonical_hash(self._unhashed()))

    @staticmethod
    def _seal_rows(values: Mapping[tuple[str, int, str, str | None], str]) -> list[dict[str, object]]:
        return [
            {"target_center": key[0], "fold_ordinal": key[1], "method_id": key[2],
             "geometry_id": key[3], "decision_hash": value}
            for key, value in sorted(values.items())
        ]

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_actionability_decision_products_v1",
            "decisions": [decision_payload(row) for row in self.decisions],
            "support_action_scores": [dict(row.__dict__) for row in self.support_action_scores],
            "support_product_hashes": [list(row) for row in self.support_product_hashes],
            "pre_support_decision_hashes": self._seal_rows(self.pre_support_decision_hashes),
            "pre_support_seal_hash": self.pre_support_seal_hash,
            "all_decision_hashes": self._seal_rows(self.all_decision_hashes),
            "all_decisions_seal_hash": self.all_decisions_seal_hash,
            "permutation_provenance_hash": self.permutation_provenance_hash,
            "partition_hash": self.partition_hash,
            "protocol_contract_hash": self.protocol_contract_hash,
            "probability_surface_hash": self.probability_surface_hash,
            "evaluation_labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "decision_products_hash": self.decision_products_hash}


__all__ = (
    "DecisionProducts", "FoldActionScore", "FoldDecisionSeal", "PRE_SUPPORT_GEOMETRY_METHODS",
    "PreSupportDecisionProducts", "SupportFoldProduct",
)
