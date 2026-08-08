"""Reconstructively validated frozen actions for the case-OOF action menu."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Iterable, Mapping

from ...protocol import ProtocolError
from ...routing.residual_topup import (
    ResidualTopupAction,
    build_borda_directed_topup_action,
    build_single_source_tail_action,
    build_uniform_topup_action,
    target_topup_geometry,
)
from ...routing.residual_topup.hashing import canonical_sha256
from .contracts import (
    BASE_ACTION_ID,
    CENTERS,
    GLOBAL_ACTION_ID,
    PERMUTATION_ACTION_ID,
    SUPPORT_ACTION_ID,
    UNIFORM_ACTION_ID,
    candidate_sources,
    expected_action_ids,
    tail_source,
)


BASE_PER_SOURCE_PER_CLASS = 128
BASE_TOTAL_PER_CLASS = 1024
TOPUP_TOTAL_PER_CLASS = 128
MATCHED_TOTAL_PER_CLASS = 1152

FROZEN_ACTION_SCHEMA_VERSION = (
    "midogpp_residual_topup_case_oof_frozen_action_v1"
)

BASE_ACTION_KIND = "fixed_equal_union_base_only"
UNIFORM_POLICY_ACTION_KIND = "fixed_uniform_residual_topup_control"
GLOBAL_POLICY_ACTION_KIND = (
    "fixed_support_only_leave_target_and_query_out_global_proxy_"
    "midrank_residual_topup"
)
SUPPORT_POLICY_ACTION_KIND = (
    "fixed_two_case_target_support_proxy_midrank_residual_topup"
)
PERMUTATION_POLICY_ACTION_KIND = (
    "fixed_target_support_proxy_midrank_source_identity_permutation_control"
)
SINGLE_SOURCE_POLICY_ACTION_KIND = (
    "fixed_single_source_tail_residual_topup_diagnostic"
)

BASE_ACTION_SEMANTICS = "equal_union_base_128_per_source_per_class_no_topup"
UNIFORM_ACTION_SEMANTICS = (
    "equal_union_base_plus_exact_uniform_128_per_class_tail"
)
GLOBAL_ACTION_SEMANTICS = (
    "equal_union_base_plus_fixed_support_leave_H_and_q_out_global_"
    "proxy_borda_tail"
)
SUPPORT_ACTION_SEMANTICS = (
    "equal_union_base_plus_fixed_two_case_unlabeled_target_support_"
    "proxy_borda_tail"
)
PERMUTATION_ACTION_SEMANTICS = (
    "fixed_support_proxy_ranks_reassigned_by_frozen_source_identity_"
    "permutation_before_borda_tail"
)
SINGLE_SOURCE_ACTION_SEMANTICS = (
    "equal_union_base_plus_all_128_tail_rows_from_one_predeclared_source"
)


@dataclass(frozen=True)
class FrozenCaseOOFAction:
    """One reconstructively validated action frozen before label access."""

    outer_target: str
    action_id: str
    policy_id: str
    action_kind: str
    action_semantics: str
    source_order: tuple[str, ...]
    base_per_source_per_class: int
    topup_total_per_class: int
    final_total_per_class: int
    mean_normalized_midrank_by_source: Mapping[str, float]
    source_identity_permutation: Mapping[str, str]
    selected_source: str | None
    direction_weights_by_source: Mapping[str, float]
    topup_counts_by_source: Mapping[str, int]
    final_counts_by_class: Mapping[int, Mapping[str, int]]
    core_action_kind: str | None
    core_action_hash: str | None
    diagnostic_control: bool
    action_hash: str
    core_action: ResidualTopupAction | None = None

    def __post_init__(self) -> None:
        target = str(self.outer_target)
        action_id = str(self.action_id)
        sources = tuple(str(source) for source in self.source_order)
        if (
            target not in CENTERS
            or sources != candidate_sources(target)
            or action_id not in expected_action_ids(target)
        ):
            raise ProtocolError("Case-OOF frozen action identity drifted.")
        ranks = _float_mapping(
            self.mean_normalized_midrank_by_source,
            sources=sources,
            allow_empty=True,
        )
        permutation = _permutation_mapping(
            self.source_identity_permutation,
            sources=sources,
        )
        direction = _float_mapping(
            self.direction_weights_by_source,
            sources=sources,
            allow_empty=True,
        )
        topup = _int_mapping(self.topup_counts_by_source, sources=sources)
        final = _nested_counts(self.final_counts_by_class, sources=sources)
        expected_policy_id, expected_diagnostic = _policy_identity(
            action_id,
            target=target,
        )
        if (
            self.policy_id != expected_policy_id
            or type(self.diagnostic_control) is not bool
            or self.diagnostic_control is not expected_diagnostic
            or isinstance(self.base_per_source_per_class, bool)
            or self.base_per_source_per_class != BASE_PER_SOURCE_PER_CLASS
            or isinstance(self.topup_total_per_class, bool)
            or self.topup_total_per_class
            not in {0, TOPUP_TOTAL_PER_CLASS}
            or isinstance(self.final_total_per_class, bool)
            or self.final_total_per_class
            != BASE_TOTAL_PER_CLASS + self.topup_total_per_class
            or sum(topup.values()) != self.topup_total_per_class
            or any(
                sum(counts.values()) != self.final_total_per_class
                for counts in final.values()
            )
            or any(
                counts[source]
                != BASE_PER_SOURCE_PER_CLASS + topup[source]
                for counts in final.values()
                for source in sources
            )
        ):
            raise ProtocolError("Case-OOF frozen action geometry drifted.")

        selected = None if self.selected_source is None else str(self.selected_source)
        expected_core = _expected_core(
            action_id=action_id,
            sources=sources,
            ranks=ranks,
            permutation=permutation,
            selected_source=selected,
        )
        if expected_core is None:
            if (
                self.topup_total_per_class != 0
                or direction
                or any(topup.values())
                or ranks
                or permutation
                or selected is not None
                or self.core_action is not None
                or self.core_action_kind is not None
                or self.core_action_hash is not None
            ):
                raise ProtocolError("Case-OOF base action carries routing state.")
        else:
            core = self.core_action
            if (
                not isinstance(core, ResidualTopupAction)
                or self.topup_total_per_class != TOPUP_TOTAL_PER_CLASS
                or self.core_action_kind != expected_core.action_kind
                or self.core_action_hash != expected_core.action_hash
                or core.action_hash != expected_core.action_hash
                or core.action_kind != expected_core.action_kind
                or core.geometry.source_order != sources
                or dict(core.calibrated_energy_by_source)
                or dict(core.direction_weights) != direction
                or dict(core.topup_counts) != topup
                or {
                    label: dict(core.final_counts_by_class[label])
                    for label in (0, 1)
                }
                != final
            ):
                raise ProtocolError(
                    "Case-OOF action does not reconstruct from pure top-up primitives."
                )

        payload = serialized_action_payload(
            outer_target=target,
            action_id=action_id,
            policy_id=self.policy_id,
            action_kind=self.action_kind,
            action_semantics=self.action_semantics,
            source_order=sources,
            base_per_source_per_class=self.base_per_source_per_class,
            topup_total_per_class=self.topup_total_per_class,
            final_total_per_class=self.final_total_per_class,
            mean_normalized_midrank_by_source=ranks,
            source_identity_permutation=permutation,
            selected_source=selected,
            direction_weights_by_source=direction,
            topup_counts_by_source=topup,
            final_counts_by_class=final,
            core_action_kind=self.core_action_kind,
            core_action_hash=self.core_action_hash,
            diagnostic_control=self.diagnostic_control,
        )
        if self.action_hash != canonical_sha256(payload):
            raise ProtocolError("Case-OOF frozen action hash is invalid.")
        object.__setattr__(self, "outer_target", target)
        object.__setattr__(self, "action_id", action_id)
        object.__setattr__(self, "source_order", sources)
        object.__setattr__(
            self,
            "mean_normalized_midrank_by_source",
            MappingProxyType(ranks),
        )
        object.__setattr__(
            self,
            "source_identity_permutation",
            MappingProxyType(permutation),
        )
        object.__setattr__(
            self,
            "direction_weights_by_source",
            MappingProxyType(direction),
        )
        object.__setattr__(
            self,
            "topup_counts_by_source",
            MappingProxyType(topup),
        )
        object.__setattr__(
            self,
            "final_counts_by_class",
            MappingProxyType(
                {
                    label: MappingProxyType(final[label])
                    for label in (0, 1)
                }
            ),
        )
        object.__setattr__(self, "selected_source", selected)

    def to_payload(self) -> dict[str, object]:
        payload = serialized_action_payload(
            outer_target=self.outer_target,
            action_id=self.action_id,
            policy_id=self.policy_id,
            action_kind=self.action_kind,
            action_semantics=self.action_semantics,
            source_order=self.source_order,
            base_per_source_per_class=self.base_per_source_per_class,
            topup_total_per_class=self.topup_total_per_class,
            final_total_per_class=self.final_total_per_class,
            mean_normalized_midrank_by_source=(
                self.mean_normalized_midrank_by_source
            ),
            source_identity_permutation=self.source_identity_permutation,
            selected_source=self.selected_source,
            direction_weights_by_source=self.direction_weights_by_source,
            topup_counts_by_source=self.topup_counts_by_source,
            final_counts_by_class=self.final_counts_by_class,
            core_action_kind=self.core_action_kind,
            core_action_hash=self.core_action_hash,
            diagnostic_control=self.diagnostic_control,
        )
        payload["action_hash"] = self.action_hash
        return payload


def make_frozen_case_oof_action(
    *,
    target: str,
    action_id: str,
    policy_id: str,
    action_kind: str,
    action_semantics: str,
    sources: tuple[str, ...],
    ranks: Mapping[str, float],
    permutation: Mapping[str, str],
    selected_source: str | None,
    direction: Mapping[str, float],
    topup: Mapping[str, int],
    final: Mapping[int, Mapping[str, int]],
    core: ResidualTopupAction | None,
    diagnostic_control: bool,
) -> FrozenCaseOOFAction:
    """Construct and canonically hash one frozen, validated Case-OOF action."""

    topup_total = 0 if core is None else core.geometry.topup_total_per_class
    payload = serialized_action_payload(
        outer_target=target,
        action_id=action_id,
        policy_id=policy_id,
        action_kind=action_kind,
        action_semantics=action_semantics,
        source_order=sources,
        base_per_source_per_class=BASE_PER_SOURCE_PER_CLASS,
        topup_total_per_class=topup_total,
        final_total_per_class=BASE_TOTAL_PER_CLASS + topup_total,
        mean_normalized_midrank_by_source=ranks,
        source_identity_permutation=permutation,
        selected_source=selected_source,
        direction_weights_by_source=direction,
        topup_counts_by_source=topup,
        final_counts_by_class=final,
        core_action_kind=None if core is None else core.action_kind,
        core_action_hash=None if core is None else core.action_hash,
        diagnostic_control=diagnostic_control,
    )
    return FrozenCaseOOFAction(
        outer_target=target,
        action_id=action_id,
        policy_id=policy_id,
        action_kind=action_kind,
        action_semantics=action_semantics,
        source_order=sources,
        base_per_source_per_class=BASE_PER_SOURCE_PER_CLASS,
        topup_total_per_class=topup_total,
        final_total_per_class=BASE_TOTAL_PER_CLASS + topup_total,
        mean_normalized_midrank_by_source=ranks,
        source_identity_permutation=permutation,
        selected_source=selected_source,
        direction_weights_by_source=direction,
        topup_counts_by_source=topup,
        final_counts_by_class=final,
        core_action_kind=None if core is None else core.action_kind,
        core_action_hash=None if core is None else core.action_hash,
        diagnostic_control=diagnostic_control,
        action_hash=canonical_sha256(payload),
        core_action=core,
    )


def serialized_action_payload(
    *,
    outer_target: str,
    action_id: str,
    policy_id: str,
    action_kind: str,
    action_semantics: str,
    source_order: tuple[str, ...],
    base_per_source_per_class: int,
    topup_total_per_class: int,
    final_total_per_class: int,
    mean_normalized_midrank_by_source: Mapping[str, float],
    source_identity_permutation: Mapping[str, str],
    selected_source: str | None,
    direction_weights_by_source: Mapping[str, float],
    topup_counts_by_source: Mapping[str, int],
    final_counts_by_class: Mapping[int, Mapping[str, int]],
    core_action_kind: str | None,
    core_action_hash: str | None,
    diagnostic_control: bool,
) -> dict[str, object]:
    """Return the canonical unhashed representation of a frozen action."""

    return {
        "schema_version": FROZEN_ACTION_SCHEMA_VERSION,
        "outer_target": outer_target,
        "action_id": action_id,
        "policy_id": policy_id,
        "action_kind": action_kind,
        "action_semantics": action_semantics,
        "source_order": list(source_order),
        "base_per_source_per_class": base_per_source_per_class,
        "topup_total_per_class": topup_total_per_class,
        "final_total_per_class": final_total_per_class,
        "mean_normalized_midrank_by_source": dict(
            mean_normalized_midrank_by_source
        ),
        "source_identity_permutation": dict(source_identity_permutation),
        "selected_source": selected_source,
        "direction_weights_by_source": dict(direction_weights_by_source),
        "topup_counts_by_source": dict(topup_counts_by_source),
        "final_counts_by_class": {
            str(label): dict(final_counts_by_class[label])
            for label in (0, 1)
        },
        "core_action_kind": core_action_kind,
        "core_action_hash": core_action_hash,
        "diagnostic_control": diagnostic_control,
    }


def canonical_source_identity_permutation(
    sources: Iterable[object],
    *,
    permutation_index: int = 1,
) -> Mapping[str, str]:
    """Return the fixed non-identity cyclic source-label permutation."""

    canonical = tuple(sorted(str(source) for source in sources))
    if (
        len(canonical) < 2
        or len(set(canonical)) != len(canonical)
        or isinstance(permutation_index, bool)
        or not isinstance(permutation_index, int)
        or not 1 <= permutation_index < len(canonical)
    ):
        raise ProtocolError("Case-OOF source permutation is invalid.")
    return MappingProxyType(
        {
            source: canonical[(index + permutation_index) % len(canonical)]
            for index, source in enumerate(canonical)
        }
    )


def _expected_core(
    *,
    action_id: str,
    sources: tuple[str, ...],
    ranks: Mapping[str, float],
    permutation: Mapping[str, str],
    selected_source: str | None,
) -> ResidualTopupAction | None:
    geometry = target_topup_geometry(sources)
    if action_id == BASE_ACTION_ID:
        return None
    if action_id == UNIFORM_ACTION_ID:
        if ranks or permutation or selected_source is not None:
            raise ProtocolError("Case-OOF U action carries undeclared routing state.")
        return build_uniform_topup_action(geometry)
    if action_id in {GLOBAL_ACTION_ID, SUPPORT_ACTION_ID}:
        if set(ranks) != set(sources) or permutation or selected_source is not None:
            raise ProtocolError("Case-OOF rank action routing state drifted.")
        return build_borda_directed_topup_action(ranks, geometry=geometry)
    if action_id == PERMUTATION_ACTION_ID:
        if (
            set(ranks) != set(sources)
            or permutation
            != dict(canonical_source_identity_permutation(sources))
            or selected_source is not None
        ):
            raise ProtocolError("Case-OOF P action permutation drifted.")
        return build_borda_directed_topup_action(ranks, geometry=geometry)
    source = tail_source(action_id)
    if source is None or selected_source != source or ranks or permutation:
        raise ProtocolError("Case-OOF Hxe action state drifted.")
    return build_single_source_tail_action(source, geometry=geometry)


def _policy_identity(action_id: str, *, target: str) -> tuple[str, bool]:
    if action_id == BASE_ACTION_ID:
        return "B", False
    if action_id == UNIFORM_ACTION_ID:
        return "U", False
    if action_id == GLOBAL_ACTION_ID:
        return "G", False
    if action_id == SUPPORT_ACTION_ID:
        return "S", False
    if action_id == PERMUTATION_ACTION_ID:
        return "P", True
    source = tail_source(action_id)
    if source not in candidate_sources(target):
        raise ProtocolError("Case-OOF Hxe action source is invalid for H.")
    return f"Hxe::{source}", True


def _float_mapping(
    values: Mapping[str, float],
    *,
    sources: tuple[str, ...],
    allow_empty: bool,
) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise ProtocolError("Case-OOF action floating mapping is invalid.")
    result: dict[str, float] = {}
    try:
        for raw_source, raw_value in values.items():
            source = str(raw_source)
            if source in result or isinstance(raw_value, bool):
                raise ProtocolError(
                    "Case-OOF action floating mapping is invalid."
                )
            value = float(raw_value)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ProtocolError(
                    "Case-OOF action floating mapping is invalid."
                )
            result[source] = value
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError("Case-OOF action floating mapping is invalid.") from exc
    if (not result and not allow_empty) or not set(result).issubset(sources):
        raise ProtocolError("Case-OOF action floating mapping is invalid.")
    return {source: result[source] for source in sources if source in result}


def _permutation_mapping(
    values: Mapping[str, str],
    *,
    sources: tuple[str, ...],
) -> dict[str, str]:
    if not isinstance(values, Mapping):
        raise ProtocolError("Case-OOF action permutation is invalid.")
    result = {str(source): str(value) for source, value in values.items()}
    if result and (
        tuple(result) != sources or set(result.values()) != set(sources)
    ):
        raise ProtocolError("Case-OOF action permutation is invalid.")
    return result


def _int_mapping(
    values: Mapping[str, int],
    *,
    sources: tuple[str, ...],
) -> dict[str, int]:
    if not isinstance(values, Mapping):
        raise ProtocolError("Case-OOF action count mapping is invalid.")
    result: dict[str, int] = {}
    for raw_source, raw_value in values.items():
        source = str(raw_source)
        if (
            source in result
            or isinstance(raw_value, bool)
            or not isinstance(raw_value, int)
            or raw_value < 0
        ):
            raise ProtocolError("Case-OOF action count mapping is invalid.")
        result[source] = raw_value
    if set(result) != set(sources):
        raise ProtocolError("Case-OOF action count grid is incomplete.")
    return {source: result[source] for source in sources}


def _nested_counts(
    values: Mapping[int, Mapping[str, int]],
    *,
    sources: tuple[str, ...],
) -> dict[int, dict[str, int]]:
    if not isinstance(values, Mapping) or set(values) != {0, 1}:
        raise ProtocolError("Case-OOF action class-count grid is invalid.")
    return {
        label: _int_mapping(values[label], sources=sources)
        for label in (0, 1)
    }


__all__ = (
    "BASE_ACTION_KIND",
    "BASE_ACTION_SEMANTICS",
    "BASE_PER_SOURCE_PER_CLASS",
    "BASE_TOTAL_PER_CLASS",
    "FROZEN_ACTION_SCHEMA_VERSION",
    "FrozenCaseOOFAction",
    "GLOBAL_ACTION_SEMANTICS",
    "GLOBAL_POLICY_ACTION_KIND",
    "MATCHED_TOTAL_PER_CLASS",
    "PERMUTATION_ACTION_SEMANTICS",
    "PERMUTATION_POLICY_ACTION_KIND",
    "SINGLE_SOURCE_ACTION_SEMANTICS",
    "SINGLE_SOURCE_POLICY_ACTION_KIND",
    "SUPPORT_ACTION_SEMANTICS",
    "SUPPORT_POLICY_ACTION_KIND",
    "TOPUP_TOTAL_PER_CLASS",
    "UNIFORM_ACTION_SEMANTICS",
    "UNIFORM_POLICY_ACTION_KIND",
    "canonical_source_identity_permutation",
    "make_frozen_case_oof_action",
    "serialized_action_payload",
)
