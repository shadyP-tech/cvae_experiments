"""Prelabel feature sealing and strict-LOCO utility construction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ...protocol import ProtocolError
from .constants import MIDOGPP_CENTERS, candidate_sources
from .contracts import CaseActionFeatureRow, ExactNineProbabilitySurface, UtilityTargetRow
from .execution_support import coerce_labels, feature_payload, utility_payload
from .features import build_label_free_case_action_features, matched_blocked_feature_permutation
from .hashing import canonical_hash, require_sha256
from .targets import build_class_balanced_proper_loss_targets


@dataclass(frozen=True)
class PrelabelProducts:
    features: tuple[CaseActionFeatureRow, ...]
    feature_surface_hash: str
    permutation_provenance_hash: str
    probability_surface_hash: str
    protocol_contract_hash: str
    prelabel_products_hash: str = field(init=False)

    def __post_init__(self) -> None:
        rows = tuple(self.features)
        if not rows or rows != tuple(sorted(rows, key=lambda row: row.row_key)):
            raise ProtocolError("Prelabel feature rows must be non-empty and canonical.")
        if len({row.row_key for row in rows}) != len(rows):
            raise ProtocolError("Prelabel feature surface contains duplicate rows.")
        if {row.query_center for row in rows} != set(MIDOGPP_CENTERS):
            raise ProtocolError("Prelabel features must cover every MIDOG++ center.")
        for value, name in (
            (self.feature_surface_hash, "feature_surface_hash"),
            (self.permutation_provenance_hash, "permutation_provenance_hash"),
            (self.probability_surface_hash, "probability_surface_hash"),
            (self.protocol_contract_hash, "protocol_contract_hash"),
        ):
            require_sha256(value, name)
        expected = canonical_hash(
            {
                "schema_version": "fixed_bank_actionability_prelabel_products_v1",
                "probability_surface_hash": self.probability_surface_hash,
                "features": [feature_payload(row) for row in rows],
            }
        )
        if self.feature_surface_hash != expected:
            raise ProtocolError("Prelabel feature surface hash drifted.")
        object.__setattr__(self, "features", rows)
        object.__setattr__(self, "prelabel_products_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_actionability_prelabel_envelope_v1",
            "feature_surface_hash": self.feature_surface_hash,
            "permutation_provenance_hash": self.permutation_provenance_hash,
            "probability_surface_hash": self.probability_surface_hash,
            "protocol_contract_hash": self.protocol_contract_hash,
            "feature_count": len(self.features),
            "labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {
            **self._unhashed(),
            "features": [feature_payload(row) for row in self.features],
            "prelabel_products_hash": self.prelabel_products_hash,
        }


def build_prelabel_products(
    probabilities: ExactNineProbabilitySurface, *, protocol_contract_hash: str
) -> PrelabelProducts:
    """Seal aligned features and every final/nested P derangement pre-label."""

    if not isinstance(probabilities, ExactNineProbabilitySurface):
        raise ProtocolError("Prelabel products require a sealed exact-nine surface.")
    require_sha256(protocol_contract_hash, "protocol_contract_hash")
    features = build_label_free_case_action_features(probabilities)
    feature_hash = canonical_hash(
        {
            "schema_version": "fixed_bank_actionability_prelabel_products_v1",
            "probability_surface_hash": probabilities.surface_hash,
            "features": [feature_payload(row) for row in features],
        }
    )
    contexts: list[dict[str, object]] = []
    for outer in MIDOGPP_CENTERS:
        exclusions_by_q = ((None, (outer,)),) + tuple(
            (query, tuple(sorted((outer, query)))) for query in candidate_sources(outer)
        )
        for query, exclusions in exclusions_by_q:
            permuted = matched_blocked_feature_permutation(
                features, excluded_candidate_centers=exclusions
            )
            contexts.append(
                {
                    "outer_target_center": outer,
                    "heldout_query_center": query,
                    "excluded_candidate_centers": list(exclusions),
                    "permuted_feature_hashes": [row.feature_hash for row in permuted],
                }
            )
    permutation_hash = canonical_hash(
        {
            "schema_version": "fixed_bank_actionability_permutation_provenance_v1",
            "permutation": "nonzero_cyclic_candidate_block_derangement",
            "contexts": contexts,
            "labels_used": False,
        }
    )
    return PrelabelProducts(
        features,
        feature_hash,
        permutation_hash,
        probabilities.surface_hash,
        protocol_contract_hash,
    )


@dataclass(frozen=True)
class LocoUtilityProduct:
    outer_target_center: str
    rows: tuple[UtilityTargetRow, ...]
    donor_label_surface_hash: str
    probability_surface_hash: str
    utility_product_hash: str = field(init=False)

    def __post_init__(self) -> None:
        outer, rows = str(self.outer_target_center), tuple(self.rows)
        if outer not in MIDOGPP_CENTERS or not rows:
            raise ProtocolError("LOCO utility product has an invalid outer target.")
        if rows != tuple(sorted(rows, key=lambda row: row.row_key)):
            raise ProtocolError("LOCO utility rows must be canonical.")
        if len({row.row_key for row in rows}) != len(rows):
            raise ProtocolError("LOCO utility rows contain duplicate case-actions.")
        if {row.query_center for row in rows} != set(candidate_sources(outer)):
            raise ProtocolError("LOCO utility rows must exclude H and cover all donors.")
        require_sha256(self.donor_label_surface_hash, "donor_label_surface_hash")
        require_sha256(self.probability_surface_hash, "probability_surface_hash")
        object.__setattr__(self, "outer_target_center", outer)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "utility_product_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_actionability_loco_utility_product_v1",
            "outer_target_center": self.outer_target_center,
            "response": "class_balanced_proper_loss_gain_vs_u",
            "donor_label_surface_hash": self.donor_label_surface_hash,
            "probability_surface_hash": self.probability_surface_hash,
            "rows": [utility_payload(row) for row in self.rows],
            "target_H_labels_used": False,
            "per_case_bacc_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "utility_product_hash": self.utility_product_hash}


def build_loco_utility_product(
    probabilities: ExactNineProbabilitySurface,
    labels: Sequence[object],
    *,
    outer_target_center: str,
) -> LocoUtilityProduct:
    """Create dense case-action responses from one explicit LOCO capability."""

    outer = str(outer_target_center)
    if outer not in MIDOGPP_CENTERS:
        raise ProtocolError("LOCO utility target H is not a MIDOG++ center.")
    scoped = coerce_labels(labels, expected_scope="loco_donor")
    if {row.target_center for row in scoped} != set(candidate_sources(outer)):
        raise ProtocolError("LOCO labels must contain all and only centers other than H.")
    rows = tuple(
        sorted(
            build_class_balanced_proper_loss_targets(probabilities, scoped),
            key=lambda row: row.row_key,
        )
    )
    label_hash = canonical_hash(
        {
            "schema_version": "fixed_bank_actionability_loco_label_surface_v1",
            "outer_target_center": outer,
            "labels": [
                [row.target_center, row.case_id, row.sample_id, row.label] for row in scoped
            ],
        }
    )
    return LocoUtilityProduct(
        outer, rows, label_hash, probabilities.surface_hash
    )


__all__ = (
    "LocoUtilityProduct",
    "PrelabelProducts",
    "build_loco_utility_product",
    "build_prelabel_products",
)
