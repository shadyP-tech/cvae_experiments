"""Two-phase fresh HARP runner with an explicit prelabel barrier."""

from __future__ import annotations

from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...runtime.harp_probability_menu import HarpPredictionMenuSeal, HarpWorkstationContract
from .contracts import HarpFreshTargetCache
from .label_access import issue_harp_fresh_evaluation_capability
from .materialization import PredictionProvider, materialize_harp_fresh_probability_menu
from .policy import FrozenHarpPolicy
from .scoring import HarpFreshDescriptiveResult, score_harp_fresh_routes
from .sealing import HarpFreshPrelabelSeal, select_and_seal_harp_fresh_routes


class HarpFreshRunner:
    """Stateful adapter that makes the label-opening order unskippable."""

    __slots__ = ("policy", "cache", "_menu", "_prelabel_seal", "_evaluated")

    def __init__(self, policy: FrozenHarpPolicy, cache: HarpFreshTargetCache) -> None:
        if not isinstance(policy, FrozenHarpPolicy) or not isinstance(
            cache, HarpFreshTargetCache
        ):
            raise ProtocolError("Fresh HARP runner requires policy and cache contracts.")
        if policy.metadata.fresh_reservation_hash != cache.reservation.reservation_hash:
            raise ProtocolError("Fresh HARP runner reservation binding drifted.")
        if policy.production_ready is not True:
            raise ProtocolError(
                "Fresh HARP production runner rejects callback-only policy locks."
            )
        self.policy = policy
        self.cache = cache
        self._menu: HarpPredictionMenuSeal | None = None
        self._prelabel_seal: HarpFreshPrelabelSeal | None = None
        self._evaluated = False

    def materialize_label_free_menu(
        self,
        predictor: PredictionProvider,
        *,
        workstation: HarpWorkstationContract | None = None,
    ) -> HarpPredictionMenuSeal:
        if self._menu is not None or self._prelabel_seal is not None:
            raise ProtocolError("Fresh HARP probability menu was already materialized.")
        if workstation is None:
            self._menu = materialize_harp_fresh_probability_menu(
                self.policy, self.cache, predictor
            )
        else:
            self._menu = materialize_harp_fresh_probability_menu(
                self.policy, self.cache, predictor, workstation=workstation
            )
        return self._menu

    def select_and_seal_all_routes(
        self,
        *,
        durable_bundle_hash: str,
        independent_validation_hashes: Sequence[str],
    ) -> HarpFreshPrelabelSeal:
        if self._menu is None:
            raise ProtocolError("Fresh HARP routes cannot be selected before menu sealing.")
        if self._prelabel_seal is not None:
            raise ProtocolError("Fresh HARP routes were already sealed.")
        self._prelabel_seal = select_and_seal_harp_fresh_routes(
            self.policy,
            self.cache,
            self._menu,
            durable_bundle_hash=durable_bundle_hash,
            independent_validation_hashes=independent_validation_hashes,
        )
        return self._prelabel_seal

    def open_labels_and_score(
        self,
        *,
        labels_by_row_key: Mapping[tuple[str, str, str], int],
        reservation_hash: str,
        target_cache_hash: str,
        authorization_hash: str,
    ) -> HarpFreshDescriptiveResult:
        if self._prelabel_seal is None:
            raise ProtocolError("Fresh HARP labels cannot open before all routes are sealed.")
        if self._evaluated:
            raise ProtocolError("Fresh HARP evaluation is one-shot.")
        capability = issue_harp_fresh_evaluation_capability(
            self._prelabel_seal,
            labels_by_row_key=labels_by_row_key,
            reservation_hash=reservation_hash,
            target_cache_hash=target_cache_hash,
            authorization_hash=authorization_hash,
        )
        result = score_harp_fresh_routes(self._prelabel_seal, capability)
        self._evaluated = True
        return result


__all__ = ("HarpFreshRunner",)
