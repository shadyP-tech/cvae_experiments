"""Frozen policy-admission contracts for SCALE-BP."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
import math

from .controls import METHOD_IDS, NEGATIVE_CONTROL_IDS
from .hashing import canonical_hash, require_sha256
from .identity import (
    MAXIMUM_HARMFUL_SELECTED_POLICY_COUNT,
    MAXIMUM_NORMALIZED_ORACLE_GAP,
    CENTERS,
    EXPECTED_CASE_COUNT,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    MINIMUM_OPPORTUNITY_CASES,
    MINIMUM_REPRESENTED_CENTERS,
    MINIMUM_WITHIN_CASE_SPEARMAN,
    TIE_TOLERANCE,
)
from .protocol import ProtocolError


@dataclass(frozen=True, slots=True)
class AdmissionThresholds:
    """Non-overridable admission thresholds mirrored from frozen identity."""

    minimum_opportunity_cases: int = MINIMUM_OPPORTUNITY_CASES
    minimum_represented_centers: int = MINIMUM_REPRESENTED_CENTERS
    minimum_spearman: float = MINIMUM_WITHIN_CASE_SPEARMAN
    maximum_normalized_oracle_gap: float = MAXIMUM_NORMALIZED_ORACLE_GAP
    maximum_harmful_selected_policy_count: int = (
        MAXIMUM_HARMFUL_SELECTED_POLICY_COUNT
    )
    tie_tolerance: float = TIE_TOLERANCE

    def __post_init__(self) -> None:
        if (
            self.minimum_opportunity_cases <= 0
            or self.minimum_represented_centers <= 0
            or not math.isfinite(self.minimum_spearman)
            or not math.isfinite(self.maximum_normalized_oracle_gap)
            or self.maximum_normalized_oracle_gap < 0.0
            or self.maximum_harmful_selected_policy_count < 0
            or not math.isfinite(self.tie_tolerance)
            or self.tie_tolerance < 0.0
        ):
            raise ProtocolError("SCALE-BP admission thresholds drifted.")


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    """Immutable result of the complete pseudo-policy admission replay."""

    outer_center: str
    replay_inventory_hash: str
    replay_bundle_hash: str
    replay_input_root: str
    action_evidence_root: str
    policy_evidence_root: str
    oracle_root: str
    method_menu_hash: str
    admitted: bool
    reasons: tuple[str, ...]
    opportunity_case_count: int
    represented_center_count: int
    selected_case_count: int
    equal_center_bacc_gain: float
    equal_center_brier_loss_delta: float
    equal_center_log_loss_delta: float
    opportunity_spearman: float | None
    spearman_case_count: int
    top1_action_agreement: float | None
    normalized_oracle_gap: float | None
    legacy_normalized_oracle_gap: float | None
    harmful_selected_policy_count: int
    control_route_counts: tuple[tuple[str, int], ...]
    context_count: int
    policy_count: int
    result_hash: str = field(init=False)

    def __post_init__(self) -> None:
        outer_center = str(self.outer_center)
        inventory_hash = require_sha256(
            self.replay_inventory_hash,
            "admission replay inventory hash",
        )
        bundle_hash = require_sha256(self.replay_bundle_hash, "admission bundle hash")
        input_root = require_sha256(self.replay_input_root, "admission input root")
        action_root = require_sha256(
            self.action_evidence_root, "admission action-evidence root"
        )
        policy_root = require_sha256(
            self.policy_evidence_root, "admission policy-evidence root"
        )
        oracle_root = require_sha256(self.oracle_root, "admission oracle root")
        menu_hash = require_sha256(self.method_menu_hash, "admission method-menu hash")
        expected_context_count = EXPECTED_CASE_COUNT - dict(
            EXPECTED_CASE_COUNTS_BY_CENTER
        )[outer_center] if outer_center in CENTERS else -1
        scalar_metrics = (
            self.equal_center_bacc_gain,
            self.equal_center_brier_loss_delta,
            self.equal_center_log_loss_delta,
        )
        optional_metrics = (
            self.opportunity_spearman,
            self.top1_action_agreement,
            self.normalized_oracle_gap,
            self.legacy_normalized_oracle_gap,
        )
        if (
            outer_center not in CENTERS
            or self.admitted != (not self.reasons)
            or len(set(self.reasons)) != len(self.reasons)
            or any(value < 0 for value in (
                self.opportunity_case_count,
                self.represented_center_count,
                self.selected_case_count,
                self.spearman_case_count,
                self.harmful_selected_policy_count,
            ))
            or any(not math.isfinite(value) for value in scalar_metrics)
            or any(value is not None and not math.isfinite(value) for value in optional_metrics)
            or self.control_route_counts
            != tuple(sorted(self.control_route_counts, key=lambda row: row[0]))
            or tuple(method for method, _count in self.control_route_counts)
            != tuple(sorted(NEGATIVE_CONTROL_IDS))
            or any(count < 0 for _method, count in self.control_route_counts)
            or self.context_count != expected_context_count
            or self.policy_count != self.context_count * len(METHOD_IDS)
        ):
            raise ProtocolError("SCALE-BP admission control inventory drifted.")
        object.__setattr__(self, "outer_center", outer_center)
        object.__setattr__(self, "replay_inventory_hash", inventory_hash)
        object.__setattr__(self, "replay_bundle_hash", bundle_hash)
        object.__setattr__(self, "replay_input_root", input_root)
        object.__setattr__(self, "action_evidence_root", action_root)
        object.__setattr__(self, "policy_evidence_root", policy_root)
        object.__setattr__(self, "oracle_root", oracle_root)
        object.__setattr__(self, "method_menu_hash", menu_hash)
        payload = {
            "schema_version": "scale_bp_policy_level_pseudo_admission_result_v2",
            "outer_center": outer_center,
            "replay_inventory_hash": inventory_hash,
            "replay_bundle_hash": bundle_hash,
            "replay_input_root": input_root,
            "action_evidence_root": action_root,
            "policy_evidence_root": policy_root,
            "oracle_root": oracle_root,
            "method_menu_hash": menu_hash,
            "admitted": self.admitted,
            "reasons": self.reasons,
            "opportunity_case_count": self.opportunity_case_count,
            "represented_center_count": self.represented_center_count,
            "selected_case_count": self.selected_case_count,
            "equal_center_bacc_gain": self.equal_center_bacc_gain,
            "equal_center_brier_loss_delta": self.equal_center_brier_loss_delta,
            "equal_center_log_loss_delta": self.equal_center_log_loss_delta,
            "opportunity_spearman": self.opportunity_spearman,
            "spearman_case_count": self.spearman_case_count,
            "top1_action_agreement": self.top1_action_agreement,
            "normalized_oracle_gap": self.normalized_oracle_gap,
            "legacy_normalized_oracle_gap": self.legacy_normalized_oracle_gap,
            "harmful_selected_policy_count": self.harmful_selected_policy_count,
            "control_route_counts": self.control_route_counts,
            "context_count": self.context_count,
            "policy_count": self.policy_count,
            "method_ids": METHOD_IDS,
            "within_case_then_equal_center": True,
            "pair_aware_policy_replay": True,
            "undefined_statistics_fail_closed": True,
        }
        object.__setattr__(self, "result_hash", canonical_hash(payload))


_ALL_OUTER_ADMISSION_FACTORY_TOKEN = object()


@dataclass(frozen=True, slots=True)
class AllOuterAdmissionResult:
    """Factory-issued admission outcome requiring every canonical outer H."""

    replay_bundle_hash: str
    replay_input_root: str
    action_evidence_root: str
    policy_evidence_root: str
    oracle_root: str
    method_menu_hash: str
    outer_results: tuple[AdmissionResult, ...]
    _factory_token: InitVar[object] = None
    admitted: bool = field(init=False)
    failed_outer_centers: tuple[str, ...] = field(init=False)
    context_count: int = field(init=False)
    policy_count: int = field(init=False)
    result_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _ALL_OUTER_ADMISSION_FACTORY_TOKEN:
            raise ProtocolError(
                "SCALE-BP all-outer admission result was not factory issued."
            )
        bundle_hash = require_sha256(
            self.replay_bundle_hash, "all-outer admission bundle hash"
        )
        input_root = require_sha256(
            self.replay_input_root, "all-outer admission input root"
        )
        action_root = require_sha256(
            self.action_evidence_root, "all-outer admission action root"
        )
        policy_root = require_sha256(
            self.policy_evidence_root, "all-outer admission policy root"
        )
        oracle_root = require_sha256(
            self.oracle_root, "all-outer admission oracle root"
        )
        menu_hash = require_sha256(
            self.method_menu_hash, "all-outer admission method-menu hash"
        )
        results = tuple(self.outer_results)
        if (
            any(not isinstance(row, AdmissionResult) for row in results)
            or tuple(row.outer_center for row in results) != CENTERS
            or any(row.method_menu_hash != menu_hash for row in results)
            or len({row.result_hash for row in results}) != len(results)
        ):
            raise ProtocolError("SCALE-BP all-outer admission universe drifted.")
        context_count = sum(row.context_count for row in results)
        policy_count = sum(row.policy_count for row in results)
        expected_context_count = len(CENTERS) * EXPECTED_CASE_COUNT - EXPECTED_CASE_COUNT
        if (
            context_count != expected_context_count
            or policy_count != context_count * len(METHOD_IDS)
        ):
            raise ProtocolError("SCALE-BP all-outer admission rectangle drifted.")
        failed = tuple(row.outer_center for row in results if not row.admitted)
        payload = {
            "schema_version": "scale_bp_all_outer_pseudo_admission_result_v1",
            "replay_bundle_hash": bundle_hash,
            "replay_input_root": input_root,
            "action_evidence_root": action_root,
            "policy_evidence_root": policy_root,
            "oracle_root": oracle_root,
            "method_menu_hash": menu_hash,
            "outer_result_hashes": tuple(
                (row.outer_center, row.result_hash) for row in results
            ),
            "admitted": not failed,
            "failed_outer_centers": failed,
            "outer_center_count": len(results),
            "context_count": context_count,
            "policy_count": policy_count,
            "every_outer_must_pass": True,
        }
        object.__setattr__(self, "replay_bundle_hash", bundle_hash)
        object.__setattr__(self, "replay_input_root", input_root)
        object.__setattr__(self, "action_evidence_root", action_root)
        object.__setattr__(self, "policy_evidence_root", policy_root)
        object.__setattr__(self, "oracle_root", oracle_root)
        object.__setattr__(self, "method_menu_hash", menu_hash)
        object.__setattr__(self, "outer_results", results)
        object.__setattr__(self, "admitted", not failed)
        object.__setattr__(self, "failed_outer_centers", failed)
        object.__setattr__(self, "context_count", context_count)
        object.__setattr__(self, "policy_count", policy_count)
        object.__setattr__(self, "result_hash", canonical_hash(payload))


def _issue_all_outer_admission_result(**kwargs: object) -> AllOuterAdmissionResult:
    return AllOuterAdmissionResult(
        **kwargs,
        _factory_token=_ALL_OUTER_ADMISSION_FACTORY_TOKEN,
    )


__all__ = (
    "AdmissionResult",
    "AdmissionThresholds",
    "AllOuterAdmissionResult",
)
