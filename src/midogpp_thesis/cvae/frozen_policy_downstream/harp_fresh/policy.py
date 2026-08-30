"""Outcome-free adapters for an already frozen HARP policy.

The callback binder in this module is deliberately test-only.  Production
Stage-70 execution is admitted only through :func:`load_frozen_harp_policy`,
which reconstructs the complete Stage-60 model state before creating this
object.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...runtime.harp_probability_menu import (
    TARGET_SURFACE,
    HarpPredictionMenuSeal,
    HarpRouteDecision,
    build_all_target_actions,
)
from .contracts import (
    HarpFreshTargetCache,
    HarpFrozenExecutionLineage,
    HarpFrozenPolicyMetadata,
)


RouteSelector = Callable[
    [HarpPredictionMenuSeal, HarpFreshTargetCache, HarpFrozenPolicyMetadata],
    Sequence[HarpRouteDecision],
]
_CTOR_KEY = object()


class FrozenHarpPolicy:
    """A frozen lock plus its label-free inference implementation."""

    __slots__ = (
        "metadata",
        "policy_receipt_hash",
        "model_bank_collection_hash",
        "action_library_hash",
        "execution_lineage",
        "production_ready",
        "_selector",
        "_physical_selector",
    )

    def __init__(
        self,
        key: object,
        *,
        metadata: HarpFrozenPolicyMetadata,
        selector: RouteSelector,
        physical_selector: RouteSelector,
        policy_receipt_hash: str,
        model_bank_collection_hash: str | None,
        action_library_hash: str | None,
        execution_lineage: HarpFrozenExecutionLineage | None,
        production_ready: bool,
    ) -> None:
        if key is not _CTOR_KEY:
            raise ProtocolError("Fresh HARP policies must be bound through their lock.")
        if bool(production_ready) != isinstance(
            execution_lineage, HarpFrozenExecutionLineage
        ):
            raise ProtocolError("Fresh HARP production readiness lacks executable lineage.")
        self.metadata = metadata
        self.policy_receipt_hash = policy_receipt_hash
        self.model_bank_collection_hash = model_bank_collection_hash
        self.action_library_hash = action_library_hash
        self.execution_lineage = execution_lineage
        self.production_ready = production_ready
        self._selector = selector
        self._physical_selector = physical_selector

    def _select_and_validate(
        self,
        menu: HarpPredictionMenuSeal,
        cache: HarpFreshTargetCache,
        *,
        physical_lambda_one_only: bool,
    ) -> tuple[HarpRouteDecision, ...]:
        if not isinstance(menu, HarpPredictionMenuSeal):
            raise ProtocolError("Frozen HARP inference requires a complete menu seal.")
        if not isinstance(cache, HarpFreshTargetCache):
            raise ProtocolError("Frozen HARP inference requires the admitted fresh cache.")
        menu.assert_valid()
        if self.metadata.fresh_reservation_hash != cache.reservation.reservation_hash:
            raise ProtocolError("Frozen HARP policy is bound to another reservation.")
        expected_action_count = len(build_all_target_actions())
        if (
            len(menu.actions) != expected_action_count
            or any(action.surface_kind != TARGET_SURFACE for action in menu.actions)
            or {action.outer_target_id for action in menu.actions} != set(CENTERS)
        ):
            raise ProtocolError("Frozen HARP inference requires the global target menu.")

        selector = self._physical_selector if physical_lambda_one_only else self._selector
        decisions = tuple(selector(menu, cache, self.metadata))
        expected_rows = tuple(
            (center, row_id, case_id)
            for center in CENTERS
            for row_id, case_id in zip(
                cache.frames_by_center[center].row_ids,
                cache.frames_by_center[center].case_ids,
                strict=True,
            )
        )
        observed_rows = tuple(
            (row.outer_target_id, row.row_id, row.case_id) for row in decisions
        )
        if (
            not decisions
            or any(not isinstance(row, HarpRouteDecision) for row in decisions)
            or observed_rows != expected_rows
            or any(
                row.surface_kind != TARGET_SURFACE
                or row.query_center_id != row.outer_target_id
                or row.policy_hash != self.metadata.policy_lock_hash
                or row.prediction_menu_seal_hash != menu.seal_hash
                or row.labels_consumed is not False
                or (
                    physical_lambda_one_only
                    and row.eligible
                    and row.lambda_value != 1.0
                )
                for row in decisions
            )
        ):
            raise ProtocolError("Frozen HARP policy did not seal every fresh target row.")
        return decisions

    def select_all_routes(
        self,
        menu: HarpPredictionMenuSeal,
        cache: HarpFreshTargetCache,
    ) -> tuple[HarpRouteDecision, ...]:
        return self._select_and_validate(
            menu,
            cache,
            physical_lambda_one_only=False,
        )

    def select_all_physical_routes(
        self,
        menu: HarpPredictionMenuSeal,
        cache: HarpFreshTargetCache,
    ) -> tuple[HarpRouteDecision, ...]:
        """Select the label-free Hxe lambda=1 ablation over the same menu."""

        return self._select_and_validate(
            menu,
            cache,
            physical_lambda_one_only=True,
        )

    def __reduce__(self) -> object:
        raise ProtocolError("Bound HARP inference adapters are nonserializable.")


def bind_frozen_harp_policy(
    metadata: HarpFrozenPolicyMetadata,
    selector: RouteSelector,
    physical_selector: RouteSelector | None = None,
) -> FrozenHarpPolicy:
    """Bind a selector for isolated tests; never a production Stage-70 policy."""

    if (
        not isinstance(metadata, HarpFrozenPolicyMetadata)
        or not callable(selector)
        or (physical_selector is not None and not callable(physical_selector))
    ):
        raise ProtocolError("Fresh HARP requires typed frozen policy metadata and selector.")
    return FrozenHarpPolicy(
        _CTOR_KEY,
        metadata=metadata,
        selector=selector,
        physical_selector=selector if physical_selector is None else physical_selector,
        policy_receipt_hash=metadata.policy_lock_hash,
        model_bank_collection_hash=None,
        action_library_hash=None,
        execution_lineage=None,
        production_ready=False,
    )


def _bind_reconstructed_harp_policy(
    *,
    metadata: HarpFrozenPolicyMetadata,
    selector: RouteSelector,
    physical_selector: RouteSelector,
    policy_receipt_hash: str,
    model_bank_collection_hash: str,
    action_library_hash: str,
    execution_lineage: HarpFrozenExecutionLineage,
) -> FrozenHarpPolicy:
    """Internal production constructor used only after complete reconstruction."""

    return FrozenHarpPolicy(
        _CTOR_KEY,
        metadata=metadata,
        selector=selector,
        physical_selector=physical_selector,
        policy_receipt_hash=policy_receipt_hash,
        model_bank_collection_hash=model_bank_collection_hash,
        action_library_hash=action_library_hash,
        execution_lineage=execution_lineage,
        production_ready=True,
    )


__all__ = ("FrozenHarpPolicy", "RouteSelector", "bind_frozen_harp_policy")
