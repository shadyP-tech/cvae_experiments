"""Artifact and action contracts for the utility-aligned residual policy lock."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ...protocol import ProtocolError
from ..utility_aligned.target_features import target_sources
from ..utility_aligned_identities import (
    CENTERS,
    DEVELOPMENT_RESERVATION_ARTIFACT_ID,
    EQUAL_UNION_POLICY_ARTIFACT_ID,
    EXACT_TAIL_OUTPUT_ARTIFACT_ID as EXACT_TAIL_SURFACE_ARTIFACT_ID,
    METADATA_PROFILE_ARTIFACT_ID,
    TARGET_RESERVATION_ARTIFACT_ID,
    TARGET_SUPPORT_PARENT_RESERVATION_ARTIFACT_ID,
    TARGET_SUPPORT_SURFACE_ARTIFACT_ID,
)
from ..residual_topup.hashing import canonical_sha256


EXPERIMENT_ID = (
    "midogpp.routing_and_composition."
    "uniform_b_v2_utility_aligned_residual_policy_lock.v1"
)
OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_utility_aligned_residual_policy_lock_v1"
)
INPUT_ARTIFACT_IDS = (
    EXACT_TAIL_SURFACE_ARTIFACT_ID,
    EQUAL_UNION_POLICY_ARTIFACT_ID,
    TARGET_SUPPORT_SURFACE_ARTIFACT_ID,
    TARGET_SUPPORT_PARENT_RESERVATION_ARTIFACT_ID,
    TARGET_RESERVATION_ARTIFACT_ID,
    METADATA_PROFILE_ARTIFACT_ID,
)

ACTION_LIBRARY_SCHEMA = "midogpp_utility_aligned_residual_action_library_v2"
POLICY_LOCK_SCHEMA = "midogpp_utility_aligned_residual_policy_lock_v2"
TARGET_POLICY_LOCK_SCHEMA = "midogpp_utility_aligned_target_policy_lock_v2"
TARGET_SUPPORT_SCHEMA = "midogpp_utility_aligned_target_support_surface_v1"

BASE_LONG_ID = "base_equal_union"
UNIFORM_LONG_ID = "uniform_residual_topup"
GLOBAL_LONG_ID = "global_delta_single_source_tail"
ROUTED_LONG_ID = "utility_aligned_residual_tail"
PERMUTATION_LONG_ID = "target_feature_permutation_control_tail"
ORACLE_PREFIX = "single_source_tail::"

ACTION_ROLE_BY_ID = {
    BASE_LONG_ID: "base",
    UNIFORM_LONG_ID: "uniform_control",
    GLOBAL_LONG_ID: "global_ablation",
    ROUTED_LONG_ID: "utility_aligned_router",
    PERMUTATION_LONG_ID: "target_feature_permutation_control",
}
TARGET_BASE_PER_SOURCE = 128
TARGET_TOPUP_TOTAL_PER_CLASS = 128
EXPECTED_ACTION_COUNT = len(CENTERS) * 13
MINIMUM_SUPPORT_CASE_COUNT = 8
MINIMUM_BOOTSTRAP_SURFACE_COUNT = 32


@dataclass(frozen=True)
class LockedResidualAction:
    target_id: str
    action_id: str
    action_role: str
    selected_source: str | None
    source_order: tuple[str, ...]
    counts_per_class: Mapping[str, Mapping[str, int]]
    abstained_to_base: bool
    fallback_reason: str | None
    topup_action_hash: str | None
    decision_hash: str

    def __post_init__(self) -> None:
        sources = target_sources(self.target_id)
        if self.source_order != sources:
            raise ProtocolError("Utility-aligned action source order drifted.")
        counts = {
            str(label): MappingProxyType(
                {str(source): int(count) for source, count in values.items()}
            )
            for label, values in self.counts_per_class.items()
        }
        if tuple(counts) != ("0", "1") or any(
            tuple(values) != sources for values in counts.values()
        ):
            raise ProtocolError("Utility-aligned action class/source coverage drifted.")
        if counts["0"] != counts["1"]:
            raise ProtocolError("Utility-aligned action must be class symmetric.")
        expected_role = ACTION_ROLE_BY_ID.get(self.action_id)
        oracle_source = (
            self.action_id.removeprefix(ORACLE_PREFIX)
            if self.action_id.startswith(ORACLE_PREFIX)
            else None
        )
        if oracle_source is not None:
            expected_role = "terminal_oracle_diagnostic"
        if expected_role != self.action_role:
            raise ProtocolError("Utility-aligned action role drifted.")
        values = dict(counts["0"])
        if self.action_id == BASE_LONG_ID:
            if self.selected_source is not None or set(values.values()) != {128}:
                raise ProtocolError("Utility-aligned B is not exact equal union.")
        elif self.action_id == UNIFORM_LONG_ID:
            if self.selected_source is not None or set(values.values()) != {144}:
                raise ProtocolError("Utility-aligned U is not exact uniform top-up.")
        else:
            selected = self.selected_source
            if oracle_source is not None and selected != oracle_source:
                raise ProtocolError("Utility-aligned Hxe identity drifted.")
            if selected is None:
                if oracle_source is not None or set(values.values()) != {128}:
                    raise ProtocolError("Utility-aligned abstention is not bit-exact B.")
            else:
                if selected not in sources or any(
                    count != (256 if source == selected else 128)
                    for source, count in values.items()
                ):
                    raise ProtocolError("Utility-aligned action is not one additive tail.")
        total = sum(values.values())
        if (self.topup_action_hash is not None) != (total == 1152):
            raise ProtocolError("Utility-aligned top-up hash/budget binding drifted.")
        if self.topup_action_hash is not None and self.topup_action_hash != canonical_sha256(
            {
                "schema_version": "midogpp_utility_aligned_topup_geometry_v1",
                "target_center": self.target_id,
                "action_id": self.action_id,
                "selected_source": self.selected_source,
                "counts_per_class": {
                    label: dict(class_counts)
                    for label, class_counts in counts.items()
                },
            }
        ):
            raise ProtocolError("Utility-aligned top-up action hash drifted.")
        if self.abstained_to_base is not (self.selected_source is None and self.action_id in {
            GLOBAL_LONG_ID, ROUTED_LONG_ID, PERMUTATION_LONG_ID
        }):
            raise ProtocolError("Utility-aligned abstention flag drifted.")
        if (self.fallback_reason is not None) is not self.abstained_to_base:
            raise ProtocolError("Utility-aligned fallback reason drifted.")
        if self.decision_hash != canonical_sha256(self._unhashed_payload()):
            raise ProtocolError("Utility-aligned action hash drifted.")
        object.__setattr__(self, "counts_per_class", MappingProxyType(counts))

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "target_center": self.target_id,
            "action_id": self.action_id,
            "action_role": self.action_role,
            "selected_source": self.selected_source,
            "source_order": list(self.source_order),
            "counts_per_class": {
                label: dict(values) for label, values in self.counts_per_class.items()
            },
            "abstained_to_base": self.abstained_to_base,
            "fallback_reason": self.fallback_reason,
            "total_per_class": sum(self.counts_per_class["0"].values()),
            "topup_action_hash": self.topup_action_hash,
            "target_labels_used": False,
            "support_labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "decision_hash": self.decision_hash}


def build_locked_action(
    *,
    target_id: str,
    action_id: str,
    selected_source: str | None,
    fallback_reason: str | None = None,
) -> LockedResidualAction:
    sources = target_sources(target_id)
    if action_id == BASE_LONG_ID:
        counts = {source: 128 for source in sources}
    elif action_id == UNIFORM_LONG_ID:
        counts = {source: 144 for source in sources}
    else:
        counts = {
            source: 256 if source == selected_source else 128 for source in sources
        }
    role = (
        "terminal_oracle_diagnostic"
        if action_id.startswith(ORACLE_PREFIX)
        else ACTION_ROLE_BY_ID.get(action_id, "")
    )
    class_counts = {"0": dict(counts), "1": dict(counts)}
    values: dict[str, object] = {
        "target_id": target_id,
        "action_id": action_id,
        "action_role": role,
        "selected_source": selected_source,
        "source_order": sources,
        "counts_per_class": class_counts,
        "abstained_to_base": (
            selected_source is None
            and action_id in {GLOBAL_LONG_ID, ROUTED_LONG_ID, PERMUTATION_LONG_ID}
        ),
        "fallback_reason": fallback_reason,
        "topup_action_hash": None,
        "decision_hash": "",
    }
    provisional = LockedResidualAction.__new__(LockedResidualAction)
    for key, value in values.items():
        object.__setattr__(provisional, key, value)
    if sum(class_counts["0"].values()) == 1152:
        values["topup_action_hash"] = canonical_sha256(
            {
                "schema_version": "midogpp_utility_aligned_topup_geometry_v1",
                "target_center": target_id,
                "action_id": action_id,
                "selected_source": selected_source,
                "counts_per_class": class_counts,
            }
        )
        object.__setattr__(provisional, "topup_action_hash", values["topup_action_hash"])
    values["decision_hash"] = canonical_sha256(provisional._unhashed_payload())
    return LockedResidualAction(**values)  # type: ignore[arg-type]


__all__ = (
    "ACTION_LIBRARY_SCHEMA",
    "BASE_LONG_ID",
    "EXPECTED_ACTION_COUNT",
    "EXPERIMENT_ID",
    "GLOBAL_LONG_ID",
    "INPUT_ARTIFACT_IDS",
    "LockedResidualAction",
    "MINIMUM_BOOTSTRAP_SURFACE_COUNT",
    "MINIMUM_SUPPORT_CASE_COUNT",
    "ORACLE_PREFIX",
    "OUTPUT_ARTIFACT_ID",
    "PERMUTATION_LONG_ID",
    "POLICY_LOCK_SCHEMA",
    "ROUTED_LONG_ID",
    "TARGET_POLICY_LOCK_SCHEMA",
    "TARGET_RESERVATION_ARTIFACT_ID",
    "TARGET_SUPPORT_PARENT_RESERVATION_ARTIFACT_ID",
    "TARGET_SUPPORT_SCHEMA",
    "UNIFORM_LONG_ID",
    "build_locked_action",
)
