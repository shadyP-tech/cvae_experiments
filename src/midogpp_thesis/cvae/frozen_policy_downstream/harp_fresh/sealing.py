"""Global prelabel route sealing for fresh HARP evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...runtime.harp_probability_menu import (
    HarpPredictionMenuSeal,
    HarpRoutedVectorSeal,
    route_harp_probability_vector,
)
from ...runtime.harp_probability_menu.hashing import (
    canonical_sha256,
    raw_array_sha256,
    require_sha256,
)
from .contracts import HarpFreshTargetCache
from .policy import FrozenHarpPolicy


def physical_ablation_reference_preserving_vector(
    vector: HarpRoutedVectorSeal,
) -> np.ndarray:
    """Build the label-free Hxe1-or-U vector used for the matched-budget effect."""

    vector.assert_valid()
    if any(row.eligible and row.lambda_value != 1.0 for row in vector.decisions):
        raise ProtocolError(
            "Fresh HARP physical ablation escaped lambda=1 row coverage."
        )
    eligible = np.asarray([row.eligible for row in vector.decisions], dtype=bool)
    result = np.ascontiguousarray(
        np.where(
            eligible,
            vector.selected_action_probabilities,
            vector.reference_probabilities,
        ),
        dtype=np.float64,
    )
    if np.any(
        result[eligible].view(np.uint64)
        != vector.routed_probabilities[eligible].view(np.uint64)
    ):
        raise ProtocolError(
            "Fresh HARP physical lambda=1 endpoint changed probability bytes."
        )
    result.setflags(write=False)
    return result


@dataclass(frozen=True, kw_only=True)
class HarpFreshPrelabelSeal:
    menu: HarpPredictionMenuSeal
    routed_vectors: tuple[HarpRoutedVectorSeal, ...]
    physical_ablation_vectors: tuple[HarpRoutedVectorSeal, ...]
    policy_hash: str
    reservation_hash: str
    target_cache_hash: str
    durable_bundle_hash: str
    independent_validation_hashes: tuple[str, ...]
    labels_opened_before_seal: bool = False
    physical_ablation_reference_preserving_vectors: tuple[np.ndarray, ...] = field(
        init=False,
        repr=False,
    )
    physical_ablation_reference_preserving_sha256: tuple[str, ...] = field(
        init=False
    )
    route_set_hash: str = field(init=False)
    seal_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.menu, HarpPredictionMenuSeal):
            raise ProtocolError("Fresh HARP prelabel sealing requires a complete menu.")
        self.menu.assert_valid()
        vectors = tuple(self.routed_vectors)
        physical_vectors = tuple(self.physical_ablation_vectors)
        if (
            len(vectors) != len(CENTERS)
            or any(not isinstance(row, HarpRoutedVectorSeal) for row in vectors)
            or len(physical_vectors) != len(CENTERS)
            or any(
                not isinstance(row, HarpRoutedVectorSeal)
                for row in physical_vectors
            )
            or self.labels_opened_before_seal is not False
        ):
            raise ProtocolError("Fresh HARP must seal all center vectors before labels open.")
        centers: list[str] = []
        for vector in vectors:
            vector.assert_valid()
            center_set = {decision.outer_target_id for decision in vector.decisions}
            if len(center_set) != 1:
                raise ProtocolError("Fresh HARP routed vector crosses target centers.")
            centers.append(next(iter(center_set)))
        if tuple(centers) != CENTERS:
            raise ProtocolError("Fresh HARP routed-vector center coverage drifted.")
        physical_centers: list[str] = []
        reference_preserving_vectors: list[np.ndarray] = []
        for primary, physical in zip(vectors, physical_vectors, strict=True):
            physical.assert_valid()
            center_set = {
                decision.outer_target_id for decision in physical.decisions
            }
            if len(center_set) != 1:
                raise ProtocolError(
                    "Fresh HARP physical-ablation vector crosses target centers."
                )
            physical_centers.append(next(iter(center_set)))
            if (
                tuple(
                    (row.row_id, row.case_id) for row in primary.decisions
                )
                != tuple(
                    (row.row_id, row.case_id) for row in physical.decisions
                )
                or any(
                    row.eligible and row.lambda_value != 1.0
                    for row in physical.decisions
                )
            ):
                raise ProtocolError(
                    "Fresh HARP physical ablation escaped lambda=1 row coverage."
                )
            reference_preserving_vectors.append(
                physical_ablation_reference_preserving_vector(physical)
            )
        if tuple(physical_centers) != CENTERS:
            raise ProtocolError(
                "Fresh HARP physical-ablation center coverage drifted."
            )

        for name in (
            "policy_hash",
            "reservation_hash",
            "target_cache_hash",
            "durable_bundle_hash",
        ):
            object.__setattr__(self, name, require_sha256(getattr(self, name), name=name))
        validations = tuple(
            require_sha256(value, name="independent validation hash")
            for value in self.independent_validation_hashes
        )
        if len(validations) < 2 or len(set(validations)) != len(validations):
            raise ProtocolError("Fresh HARP requires two independent prelabel validations.")
        if any(
            vector.prediction_menu_seal_hash != self.menu.seal_hash
            or vector.policy_hash != self.policy_hash
            for vector in (*vectors, *physical_vectors)
        ):
            raise ProtocolError("Fresh HARP route seals drifted from menu or policy.")
        route_hash = canonical_sha256(
            {
                "schema_version": "midogpp_harp_fresh_route_set_v2",
                "menu_seal_hash": self.menu.seal_hash,
                "policy_hash": self.policy_hash,
                "reservation_hash": self.reservation_hash,
                "target_cache_hash": self.target_cache_hash,
                "routed_vectors": [
                    vector.routed_vector_seal_hash for vector in vectors
                ],
                "physical_ablation_routed_vectors": [
                    vector.routed_vector_seal_hash
                    for vector in physical_vectors
                ],
                "physical_ablation_reference_preserving_sha256": [
                    raw_array_sha256(vector)
                    for vector in reference_preserving_vectors
                ],
                "physical_ablation_action_universe": "Hxe_lambda_one_only",
                "physical_ablation_reference_preserving_semantics": (
                    "eligible_Hxe_lambda_one_else_exact_U"
                ),
                "physical_ablation_selection_labels_used": False,
                "all_routes_selected": True,
                "exact_b_fallback_byte_identity": True,
                "labels_opened": False,
            }
        )
        seal_hash = canonical_sha256(
            {
                "schema_version": "midogpp_harp_fresh_prelabel_seal_v2",
                "status": "DURABLE_ALL_ROUTES_SEALED_BEFORE_LABELS",
                "route_set_hash": route_hash,
                "menu_seal_hash": self.menu.seal_hash,
                "durable_bundle_hash": self.durable_bundle_hash,
                "independent_validation_hashes": list(validations),
                "labels_opened_before_seal": False,
            }
        )
        object.__setattr__(self, "routed_vectors", vectors)
        object.__setattr__(self, "physical_ablation_vectors", physical_vectors)
        object.__setattr__(
            self,
            "physical_ablation_reference_preserving_vectors",
            tuple(reference_preserving_vectors),
        )
        object.__setattr__(
            self,
            "physical_ablation_reference_preserving_sha256",
            tuple(raw_array_sha256(vector) for vector in reference_preserving_vectors),
        )
        object.__setattr__(self, "independent_validation_hashes", validations)
        object.__setattr__(self, "route_set_hash", route_hash)
        object.__setattr__(self, "seal_hash", seal_hash)

    @property
    def row_keys(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(
            (decision.outer_target_id, decision.case_id, decision.row_id)
            for vector in self.routed_vectors
            for decision in vector.decisions
        )

    @property
    def physical_ablation_row_keys(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(
            (decision.outer_target_id, decision.case_id, decision.row_id)
            for vector in self.physical_ablation_vectors
            for decision in vector.decisions
        )


def select_and_seal_harp_fresh_routes(
    policy: FrozenHarpPolicy,
    cache: HarpFreshTargetCache,
    menu: HarpPredictionMenuSeal,
    *,
    durable_bundle_hash: str,
    independent_validation_hashes: Sequence[str],
) -> HarpFreshPrelabelSeal:
    if not isinstance(policy, FrozenHarpPolicy) or not isinstance(
        cache, HarpFreshTargetCache
    ):
        raise ProtocolError("Fresh HARP route sealing requires policy and cache contracts.")
    decisions = policy.select_all_routes(menu, cache)
    physical_decisions = policy.select_all_physical_routes(menu, cache)
    vectors: list[HarpRoutedVectorSeal] = []
    physical_vectors: list[HarpRoutedVectorSeal] = []
    cursor = 0
    for center in CENTERS:
        row_count = len(cache.frames_by_center[center].row_ids)
        block = decisions[cursor : cursor + row_count]
        if len(block) != row_count:
            raise ProtocolError("Fresh HARP policy decision inventory is incomplete.")
        vectors.append(route_harp_probability_vector(menu, block))
        physical_block = physical_decisions[cursor : cursor + row_count]
        if len(physical_block) != row_count:
            raise ProtocolError(
                "Fresh HARP physical-ablation decision inventory is incomplete."
            )
        physical_vectors.append(
            route_harp_probability_vector(menu, physical_block)
        )
        cursor += row_count
    if cursor != len(decisions):
        raise ProtocolError("Fresh HARP policy emitted surplus decisions.")
    if cursor != len(physical_decisions):
        raise ProtocolError(
            "Fresh HARP policy emitted surplus physical-ablation decisions."
        )
    return HarpFreshPrelabelSeal(
        menu=menu,
        routed_vectors=tuple(vectors),
        physical_ablation_vectors=tuple(physical_vectors),
        policy_hash=policy.metadata.policy_lock_hash,
        reservation_hash=cache.reservation.reservation_hash,
        target_cache_hash=cache.cache_hash,
        durable_bundle_hash=durable_bundle_hash,
        independent_validation_hashes=tuple(independent_validation_hashes),
    )


__all__ = (
    "HarpFreshPrelabelSeal",
    "physical_ablation_reference_preserving_vector",
    "select_and_seal_harp_fresh_routes",
)
