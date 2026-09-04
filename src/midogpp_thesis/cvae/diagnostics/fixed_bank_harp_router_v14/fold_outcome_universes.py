"""Exact post-seal source outcomes for HARP v14 nested fold calibration.

The final source model is trained on the ordinary ``H/r/r`` prediction
surface.  Nested policy calibration is different: each prelabel prediction
was generated from a physical ``H/q/r`` menu.  Once every q prediction is
sealed and aggregate source labels are legitimately open, this module joins
those labels back to the *same* certified H/q/r physical menus.  It never
projects outcomes from the final-model surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping, Sequence

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...routing.harp_protocol import HarpSourceLabelRow, canonical_hash
from ...routing.policy_calibrated_residual_router_v14 import SourceOutcomeUniverse
from ...runtime.harp_v14_execution.contracts import ActionKind, LabelFreeActionBlock
from ...runtime.harp_v14_execution.directional_surfaces import attach_source_outcomes
from .fold_menu_binding import (
    DurableFoldMenuBindingCertificate,
    FoldLocalMenuBinding,
)

if TYPE_CHECKING:
    from .source_crossfit_orchestration import LabelFreeSourceCrossfitBundle


@dataclass(frozen=True, slots=True)
class ExactFoldOutcomeUniverse:
    """One exact label join for a certified ``(H, q)`` menu universe."""

    outer_target_id: str
    heldout_center_id: str
    fold_menu_binding_hash: str
    fold_menu_binding_certificate_hash: str
    universe: SourceOutcomeUniverse
    binding_hash: str = field(init=False)

    def __post_init__(self) -> None:
        h = str(self.outer_target_id)
        q = str(self.heldout_center_id)
        if (
            h not in CENTERS
            or q not in CENTERS
            or h == q
            or len(str(self.fold_menu_binding_hash)) != 64
            or len(str(self.fold_menu_binding_certificate_hash)) != 64
            or not isinstance(self.universe, SourceOutcomeUniverse)
            or {row.outer_target_id for row in self.universe.effective_menus} != {h}
            or {row.query_center_id for row in self.universe.effective_menus}
            != set(center for center in CENTERS if center != h)
        ):
            raise ProtocolError("HARP v14 exact fold outcome universe escaped H/q.")
        object.__setattr__(self, "outer_target_id", h)
        object.__setattr__(self, "heldout_center_id", q)
        object.__setattr__(
            self,
            "binding_hash",
            canonical_hash(
                {
                    "schema_version": "midogpp_harp_v14_exact_fold_outcome_universe_v1",
                    "outer_target_id": h,
                    "heldout_center_id": q,
                    "fold_menu_binding_hash": self.fold_menu_binding_hash,
                    "fold_menu_binding_certificate_hash": (
                        self.fold_menu_binding_certificate_hash
                    ),
                    "source_outcome_universe_hash": self.universe.universe_hash,
                    "outcomes_joined_to_exact_H_q_r_menus": True,
                    "posthoc_projection_used": False,
                    "evaluation_labels_used": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class ExactFoldOutcomeUniverseSet:
    """Complete, immutable mapping for all 72 nested source folds."""

    fold_menu_binding_certificate_hash: str
    folds: tuple[ExactFoldOutcomeUniverse, ...]
    set_hash: str = field(init=False)
    _by_pair: Mapping[tuple[str, str], ExactFoldOutcomeUniverse] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        certificate_hash = str(self.fold_menu_binding_certificate_hash)
        folds = tuple(
            sorted(
                self.folds,
                key=lambda row: (row.outer_target_id, row.heldout_center_id),
            )
        )
        expected = tuple((h, q) for h in CENTERS for q in CENTERS if h != q)
        observed = tuple((row.outer_target_id, row.heldout_center_id) for row in folds)
        if (
            len(certificate_hash) != 64
            or folds != self.folds
            or observed != expected
            or any(
                row.fold_menu_binding_certificate_hash != certificate_hash
                for row in folds
            )
        ):
            raise ProtocolError("HARP v14 exact fold outcome coverage is incomplete.")
        by_pair = MappingProxyType(
            {
                (row.outer_target_id, row.heldout_center_id): row
                for row in folds
            }
        )
        object.__setattr__(self, "folds", folds)
        object.__setattr__(self, "_by_pair", by_pair)
        object.__setattr__(
            self,
            "set_hash",
            canonical_hash(
                {
                    "schema_version": "midogpp_harp_v14_exact_fold_outcome_universe_set_v1",
                    "fold_menu_binding_certificate_hash": certificate_hash,
                    "fold_binding_hashes": [row.binding_hash for row in folds],
                    "fold_count": len(folds),
                    "outcomes_joined_after_all_q_prediction_seals": True,
                    "posthoc_projection_used": False,
                    "evaluation_labels_used": False,
                }
            ),
        )

    def for_fold(self, outer_target_id: str, heldout_center_id: str) -> ExactFoldOutcomeUniverse:
        try:
            return self._by_pair[(str(outer_target_id), str(heldout_center_id))]
        except KeyError as exc:
            raise ProtocolError("HARP v14 exact fold outcome universe is absent.") from exc

    def for_outer(self, outer_target_id: str) -> Mapping[str, SourceOutcomeUniverse]:
        h = str(outer_target_id)
        rows = {
            q: self.for_fold(h, q).universe
            for q in CENTERS
            if q != h
        }
        if len(rows) != len(CENTERS) - 1:
            raise ProtocolError("HARP v14 outer nested outcome inventory is incomplete.")
        return MappingProxyType(rows)


def build_exact_fold_outcome_universes(
    *,
    bundle: LabelFreeSourceCrossfitBundle,
    binding_certificate: DurableFoldMenuBindingCertificate,
    labels: Sequence[HarpSourceLabelRow],
) -> ExactFoldOutcomeUniverseSet:
    """Join full source labels to each exact certified H/q/r surface."""

    if (
        not isinstance(binding_certificate, DurableFoldMenuBindingCertificate)
        or binding_certificate.certificate.source_surface_receipt_hash
        != bundle.surface_receipt.receipt_hash
        or binding_certificate.certificate.source_surface_hash
        != bundle.physical_surface.surface_hash
        or binding_certificate.certificate.effective_adapter_hash
        != bundle.effective_surface.adapter_hash
    ):
        raise ProtocolError("HARP v14 exact fold outcome inputs are unbound.")
    rows = tuple(labels)
    by_center = {
        center: tuple(row for row in rows if row.center == center)
        for center in CENTERS
    }
    if (
        not rows
        or any(not isinstance(row, HarpSourceLabelRow) for row in rows)
        or {row.center for row in rows} != set(CENTERS)
        or len({row.row_key for row in rows}) != len(rows)
    ):
        raise ProtocolError("HARP v14 aggregate source labels are incomplete.")

    output: list[ExactFoldOutcomeUniverse] = []
    for binding in binding_certificate.certificate.folds:
        outcomes = []
        for r in CENTERS:
            if r == binding.outer_target_id:
                continue
            menus = tuple(
                wrapper.menu
                for wrapper in binding.wrappers
                if wrapper.current_query_center_id == r
            )
            baseline = _baseline_for(
                bundle,
                h=binding.outer_target_id,
                q=binding.heldout_center_id,
                r=r,
            )
            source_labels = _exact_case_labels(by_center[r], baseline, center=r)
            outcomes.extend(
                attach_source_outcomes(menus, baseline, source_labels=source_labels)
            )
        universe = SourceOutcomeUniverse(binding.effective_menus, tuple(outcomes))
        output.append(
            ExactFoldOutcomeUniverse(
                outer_target_id=binding.outer_target_id,
                heldout_center_id=binding.heldout_center_id,
                fold_menu_binding_hash=binding.binding_hash,
                fold_menu_binding_certificate_hash=(
                    binding_certificate.certificate.certificate_hash
                ),
                universe=universe,
            )
        )
    return ExactFoldOutcomeUniverseSet(
        fold_menu_binding_certificate_hash=(
            binding_certificate.certificate.certificate_hash
        ),
        folds=tuple(output),
    )


def _baseline_for(
    bundle: LabelFreeSourceCrossfitBundle,
    *,
    h: str,
    q: str,
    r: str,
) -> LabelFreeActionBlock:
    matches = tuple(
        row
        for row in bundle.physical_surface.blocks_for(h, q, r)
        if row.action.action_id == "B"
    )
    if len(matches) != 1:
        raise ProtocolError("HARP v14 exact fold outcome join lacks physical B.")
    raw = matches[0]
    return LabelFreeActionBlock(
        surface_role="development",
        outer_target_id=h,
        query_center_id=r,
        action_kind=ActionKind.B,
        selected_source_id=None,
        sample_ids=raw.sample_ids,
        case_ids=raw.case_ids,
        probabilities=raw.probabilities,
        seed_dispersion=raw.seed_dispersion,
    )


def _exact_case_labels(
    rows: Sequence[HarpSourceLabelRow],
    baseline: LabelFreeActionBlock,
    *,
    center: str,
) -> dict[tuple[str, str], int]:
    indexed = {row.row_key: row.label for row in rows}
    expected = {
        (center, case, sample)
        for case, sample in zip(
            baseline.case_ids, baseline.sample_ids, strict=True
        )
    }
    if set(indexed) != expected:
        raise ProtocolError("HARP v14 exact fold source labels exceed or omit r.")
    return {
        (case, sample): indexed[(center, case, sample)]
        for _, case, sample in expected
    }


__all__ = (
    "ExactFoldOutcomeUniverse",
    "ExactFoldOutcomeUniverseSet",
    "build_exact_fold_outcome_universes",
)
