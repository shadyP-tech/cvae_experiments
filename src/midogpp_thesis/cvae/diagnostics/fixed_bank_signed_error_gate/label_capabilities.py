"""Signed-gate configuration of the shared scoped label state machine."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Protocol

from ..fixed_bank_hierarchical_residual_stacker.contracts import BinaryLabel
from ..fixed_bank_hierarchical_residual_stacker.core_hashing import canonical_hash
from ..fixed_bank_hierarchical_residual_stacker.label_capabilities import (
    LabelCapabilityManager,
)
from .constants import METHOD_IDS


class SignedErrorLabelCapability(Protocol):
    """Minimal structural label/seal boundary used before evaluation opens."""

    def open_loco_donor_labels(
        self, heldout_target: str
    ) -> tuple[BinaryLabel, ...]: ...

    def open_fold_support_labels(
        self, target_center: str, fold_ordinal: int
    ) -> tuple[BinaryLabel, ...]: ...

    def record_loco_model_seals(
        self,
        heldout_target: str,
        global_model_hash: str,
        residual_model_hash: str,
        permuted_model_hash: str,
    ) -> None: ...

    def record_fold_method_decision(
        self,
        target_center: str,
        fold_ordinal: int,
        method_id: str,
        decision_hash: str,
    ) -> None: ...

    def record_preevaluation_seals(
        self,
        decision_seal_hash: str,
        permutation_provenance_hash: str,
        *,
        decision_count: int,
    ) -> None: ...


class SignedErrorLabelCapabilityManager(LabelCapabilityManager):
    """Require all six signed-gate decisions before terminal labels can open."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        if "method_ids" in kwargs:
            raise TypeError("Signed-error method IDs are frozen and cannot be overridden.")
        super().__init__(*args, method_ids=METHOD_IDS, **kwargs)  # type: ignore[arg-type]

    def access_report(self) -> Mapping[str, object]:
        base = dict(super().access_report())
        base.pop("report_hash", None)
        base.update(
            {
                "schema_version": "midogpp_signed_error_label_capability_report_v1",
                "R_raw_and_R_safe_separately_sealed": True,
                "terminal_consumed_test_diagnostic_only": True,
            }
        )
        return MappingProxyType({**base, "report_hash": canonical_hash(base)})


__all__ = (
    "SignedErrorLabelCapability",
    "SignedErrorLabelCapabilityManager",
)
