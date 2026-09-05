"""Role-qualified bridge from physical v20 component surfaces to science menus.

The workstation materializes only B, exact U, and one Hxe surface per retained
expert.  This adapter derives exact ``U:FULL`` plus the D01/D10 Hxe endpoints,
creates one memory-only source-truth capability per train case, and converts a
single pooled policy's target decisions into byte-reconstructible runtime
recipes.  No K/lambda probability lattice is persisted or sent to a worker.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Protocol, TypeAlias

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...routing.risk_aligned_router_v20.contracts import (
    CompositeKind,
    Direction,
    LabelFreeAction,
    LabelFreeCaseMenu,
    SoftTopKComposite,
    SupportActionOutcome,
    SupportCaseClassProfile,
    SurfaceRole,
)
from ...routing.risk_aligned_router_v20.hashing import (
    canonical_hash,
    require_sha256,
)
from ...routing.risk_aligned_router_v20.records import (
    RouteDecision,
)
from ...routing.risk_aligned_router_v20.truth import (
    SupportTruthCapability,
    combine_truth_capabilities,
)
from .contracts import (
    ActionKind,
    LabelFreeActionBlock,
    LabelFreeOuterMenu,
    PrelabelRouteSet,
    RoutedCase,
    reconstruct_selected_probability_blend,
    reconstruct_shrunk_probability_blend,
)


SOURCE_PHYSICAL_ROLE = "source_train"
TARGET_PHYSICAL_ROLE = "target"
FULL_U_ARM_ID = "U:FULL"

# Frozen, label-free inputs to the pooled source policy.  Target/center identity
# and any outcome-bearing quantity are deliberately absent.
from .mechanism_features import LABEL_FREE_FEATURE_NAMES, feature_values as _feature_values


CompatibilityKey: TypeAlias = tuple[str, str, str]
CompatibilityFeatures: TypeAlias = Mapping[
    CompatibilityKey, Sequence[float] | Mapping[str, float]
]


class PooledRouterLike(Protocol):
    policy_hash: str


@dataclass(frozen=True, slots=True)
class SupportTargetMenuBundle:
    """One center-keyed source-q / target-H pair of label-free menus.

    The pairing is an execution optimization only.  Source menus from all nine
    q values are flattened before the sole pooled fit; no q menu is treated as
    same-H target support.
    """

    physical_menu: LabelFreeOuterMenu = field(repr=False, compare=False)
    candidate_source_ids: tuple[str, ...]
    action_identity_hash: str
    feature_schema_hash: str
    support_menus: tuple[LabelFreeCaseMenu, ...]
    target_menus: tuple[LabelFreeCaseMenu, ...]
    support_case_samples: tuple[tuple[str, tuple[str, ...]], ...]
    target_case_samples: tuple[tuple[str, tuple[str, ...]], ...]
    support_menu_hash: str
    target_menu_hash: str
    bundle_hash: str

    def __post_init__(self) -> None:
        center = self.physical_menu.outer_target_id
        expected_candidates = tuple(value for value in CENTERS if value != center)
        source = tuple(sorted(self.support_menus, key=lambda row: row.case_id))
        target = tuple(sorted(self.target_menus, key=lambda row: row.case_id))
        source_samples = tuple(sorted(self.support_case_samples))
        target_samples = tuple(sorted(self.target_case_samples))
        if (
            self.candidate_source_ids != expected_candidates
            or not source
            or not target
            or any(
                row.center_id != center
                or row.surface_role is not SurfaceRole.SOURCE_TRAIN_DEVELOPMENT
                for row in source
            )
            or any(
                row.center_id != center
                or row.surface_role is not SurfaceRole.TARGET_EVALUATION
                for row in target
            )
            or {row.case_id for row in source}.intersection(
                row.case_id for row in target
            )
            or tuple(row.case_id for row in source)
            != tuple(case for case, _samples in source_samples)
            or tuple(row.case_id for row in target)
            != tuple(case for case, _samples in target_samples)
            or any(not samples for _case, samples in (*source_samples, *target_samples))
        ):
            raise ProtocolError("HARP v20 source/target menu bundle drifted.")
        for name in (
            "action_identity_hash",
            "feature_schema_hash",
            "support_menu_hash",
            "target_menu_hash",
            "bundle_hash",
        ):
            require_sha256(getattr(self, name), name=name)
        if self.support_menu_hash == self.target_menu_hash:
            raise ProtocolError("HARP v20 source and target menu identities collided.")
        object.__setattr__(self, "support_menus", source)
        object.__setattr__(self, "target_menus", target)
        object.__setattr__(self, "support_case_samples", source_samples)
        object.__setattr__(self, "target_case_samples", target_samples)

    @property
    def center_id(self) -> str:
        return self.physical_menu.outer_target_id

    @property
    def outer_target_id(self) -> str:
        """Compatibility alias used by the copied runner's center inventory."""

        return self.center_id

    @property
    def source_menus(self) -> tuple[LabelFreeCaseMenu, ...]:
        return self.support_menus

    @property
    def source_case_samples(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return self.support_case_samples

    @property
    def source_menu_hash(self) -> str:
        return self.support_menu_hash

    def case_samples(self, role: SurfaceRole, case_id: str) -> tuple[str, ...]:
        rows = (
            self.support_case_samples
            if role is SurfaceRole.SOURCE_TRAIN_DEVELOPMENT
            else self.target_case_samples
            if role is SurfaceRole.TARGET_EVALUATION
            else ()
        )
        for observed_case, samples in rows:
            if observed_case == case_id:
                return samples
        raise ProtocolError("HARP v20 case/sample membership is absent.")

    def report(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_harp_v20_source_target_menu_report_v1",
            "center_id": self.center_id,
            "source_context_kind": "PSEUDOQUERY_q",
            "target_context_kind": "HELDOUT_H",
            "candidate_source_ids": list(self.candidate_source_ids),
            "candidate_semantics": "C_MINUS_CONTEXT",
            "action_identity_hash": self.action_identity_hash,
            "feature_schema_hash": self.feature_schema_hash,
            "feature_names": list(LABEL_FREE_FEATURE_NAMES),
            "source_train_case_count": len(self.support_menus),
            "target_case_count": len(self.target_menus),
            "source_train_active_action_count": sum(
                len(row.actions) for row in self.support_menus
            ),
            "target_active_action_count": sum(
                len(row.actions) for row in self.target_menus
            ),
            "source_train_menu_hash": self.support_menu_hash,
            "target_menu_hash": self.target_menu_hash,
            "physical_menu_hash": self.physical_menu.menu_hash,
            "bundle_hash": self.bundle_hash,
            "exact_U_FULL_present": True,
            "directional_Hxe_components_only": True,
            "noops_removed_before_source_labels": True,
            "duplicates_removed_before_source_labels": True,
            "target_evaluation_labels_consumed": False,
        }


@dataclass(frozen=True, slots=True)
class AttachedSupportOutcomes:
    """Aggregate endpoints plus one non-serializable capability per source case."""

    center_id: str
    case_profiles: tuple[SupportCaseClassProfile, ...]
    action_outcomes: tuple[SupportActionOutcome, ...]
    truth_capabilities: tuple[SupportTruthCapability, ...] = field(
        repr=False, compare=False
    )
    attachment_hash: str = field(init=False)

    def __post_init__(self) -> None:
        profiles = tuple(sorted(self.case_profiles, key=lambda row: row.case_id))
        outcomes = tuple(
            sorted(
                self.action_outcomes,
                key=lambda row: (row.action.case_id, row.action.arm_id),
            )
        )
        capabilities = tuple(self.truth_capabilities)
        expected_keys = tuple((self.center_id, row.case_id) for row in profiles)
        if (
            not profiles
            or len({row.case_id for row in profiles}) != len(profiles)
            or any(row.center_id != self.center_id for row in profiles)
            or any(
                row.action.center_id != self.center_id
                or row.action.surface_role
                is not SurfaceRole.SOURCE_TRAIN_DEVELOPMENT
                for row in outcomes
            )
            or len(capabilities) != len(profiles)
            or tuple(key for cap in capabilities for key in cap.case_keys)
            != expected_keys
            or any(len(cap.case_keys) != 1 for cap in capabilities)
        ):
            raise ProtocolError("HARP v20 attached source outcomes are malformed.")
        object.__setattr__(self, "case_profiles", profiles)
        object.__setattr__(self, "action_outcomes", outcomes)
        object.__setattr__(self, "truth_capabilities", capabilities)
        object.__setattr__(
            self,
            "attachment_hash",
            canonical_hash(
                {
                    "schema_version": "midogpp_harp_v20_attached_source_outcomes_v1",
                    "center_id": self.center_id,
                    "profile_hashes": tuple(row.profile_hash for row in profiles),
                    "outcome_hashes": tuple(row.outcome_hash for row in outcomes),
                    "truth_capability_hashes": tuple(
                        row.capability_hash for row in capabilities
                    ),
                    "one_memory_only_capability_per_case": True,
                    "raw_source_labels_persisted": False,
                    "target_evaluation_labels_consumed": False,
                }
            ),
        )

    @property
    def outer_target_id(self) -> str:
        return self.center_id

    def public_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_harp_v20_attached_source_outcomes_v1",
            "center_id": self.center_id,
            "case_profiles": [row.public_payload() for row in self.case_profiles],
            "action_outcomes": [row.public_payload() for row in self.action_outcomes],
            "truth_capabilities": [
                row.public_payload() for row in self.truth_capabilities
            ],
            "attachment_hash": self.attachment_hash,
            "one_memory_only_capability_per_case": True,
            "raw_source_labels_persisted": False,
            "target_evaluation_labels_consumed": False,
        }


def _role_blocks(
    menu: LabelFreeOuterMenu, role: str
) -> tuple[LabelFreeActionBlock, ...]:
    rows = tuple(block for block in menu.blocks if block.surface_role == role)
    expected = tuple(center for center in CENTERS if center != menu.outer_target_id)
    donors = tuple(
        block.selected_source_id
        for block in rows
        if block.action_kind is ActionKind.HXE
    )
    if (
        len(rows) != len(expected) + 2
        or sum(block.action_kind is ActionKind.B for block in rows) != 1
        or sum(block.action_kind is ActionKind.U for block in rows) != 1
        or donors != expected
        or any(block.query_center_id != menu.outer_target_id for block in rows)
        or any(
            block.sample_ids != rows[0].sample_ids
            or block.case_ids != rows[0].case_ids
            for block in rows[1:]
        )
    ):
        raise ProtocolError("HARP v20 physical role menu is not exact C-minus-context.")
    return rows


def _single_block(
    blocks: Sequence[LabelFreeActionBlock], kind: ActionKind
) -> LabelFreeActionBlock:
    rows = tuple(row for row in blocks if row.action_kind is kind)
    if len(rows) != 1:
        raise ProtocolError("HARP v20 physical control block is ambiguous.")
    return rows[0]


def _case_order(block: LabelFreeActionBlock) -> tuple[str, ...]:
    return tuple(dict.fromkeys(block.case_ids))


def _case_indices(block: LabelFreeActionBlock, case_id: str) -> np.ndarray:
    values = np.asarray(
        [index for index, value in enumerate(block.case_ids) if value == case_id],
        dtype=np.int64,
    )
    if not len(values):
        raise ProtocolError("HARP v20 case is absent from its physical block.")
    return values


def _probability_hex(values: np.ndarray) -> tuple[str, ...]:
    raw = np.asarray(values)
    if (
        raw.dtype != np.dtype("float32")
        or raw.ndim != 1
        or not len(raw)
        or not np.isfinite(raw).all()
        or np.any((raw < 0.0) | (raw > 1.0))
    ):
        raise ProtocolError("HARP v20 probability bytes are malformed.")
    packed = np.ascontiguousarray(raw, dtype="<f4").tobytes(order="C")
    return tuple(packed[index : index + 4].hex() for index in range(0, len(packed), 4))


def _decode_probability_hex(values: Sequence[str]) -> np.ndarray:
    try:
        raw = b"".join(bytes.fromhex(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ProtocolError("HARP v20 probability cells are malformed.") from exc
    result = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)
    if not len(result) or not np.isfinite(result).all():
        raise ProtocolError("HARP v20 decoded probabilities are malformed.")
    return result


def _compatibility_values(
    compatibility: CompatibilityFeatures | None,
    *,
    role: str,
    case_id: str,
    donor_id: str | None,
) -> tuple[float, float, float, float]:
    if donor_id is None or compatibility is None:
        return (0.0, 0.0, 0.0, 0.0)
    raw = compatibility.get((role, case_id, donor_id))
    if raw is None:
        raise ProtocolError("HARP v20 compatibility features are incomplete.")
    if isinstance(raw, Mapping):
        try:
            values = (
                float(raw["mean_z"]),
                float(raw["std_z"]),
                float(raw["reciprocal_rank"]),
                float(raw["rank_margin"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("HARP v20 compatibility mapping is malformed.") from exc
    else:
        try:
            values = tuple(float(value) for value in raw)
        except (TypeError, ValueError) as exc:
            raise ProtocolError("HARP v20 compatibility row is malformed.") from exc
        if len(values) != 4:
            raise ProtocolError("HARP v20 compatibility row must contain four values.")
    if not all(math.isfinite(value) for value in values):
        raise ProtocolError("HARP v20 compatibility features must be finite.")
    return values


def _compile_role(
    menu: LabelFreeOuterMenu,
    *,
    physical_role: str,
    surface_role: SurfaceRole,
    compatibility_features: CompatibilityFeatures | None,
) -> tuple[tuple[LabelFreeCaseMenu, ...], tuple[tuple[str, tuple[str, ...]], ...]]:
    blocks = _role_blocks(menu, physical_role)
    baseline_block = _single_block(blocks, ActionKind.B)
    uniform_block = _single_block(blocks, ActionKind.U)
    donors = tuple(row for row in blocks if row.action_kind is ActionKind.HXE)
    output: list[LabelFreeCaseMenu] = []
    case_samples: list[tuple[str, tuple[str, ...]]] = []
    for case_id in _case_order(baseline_block):
        indices = _case_indices(baseline_block, case_id)
        sample_ids = tuple(
            baseline_block.sample_ids[int(index)] for index in indices
        )
        baseline = np.ascontiguousarray(
            baseline_block.probabilities[indices], dtype=np.float32
        )
        baseline_dispersion = np.ascontiguousarray(
            baseline_block.seed_dispersion[indices], dtype=np.float32
        )
        baseline_hex = _probability_hex(baseline)
        uniform = np.ascontiguousarray(
            uniform_block.probabilities[indices], dtype=np.float32
        )
        uniform_dispersion = np.ascontiguousarray(
            uniform_block.seed_dispersion[indices], dtype=np.float32
        )
        actions: list[LabelFreeAction] = [
            LabelFreeAction(
                surface_role=surface_role,
                center_id=menu.outer_target_id,
                case_id=case_id,
                arm_id=FULL_U_ARM_ID,
                direction=Direction.FULL,
                donor_id=None,
                feature_names=LABEL_FREE_FEATURE_NAMES,
                feature_values=_feature_values(
                    baseline,
                    uniform,
                    baseline_dispersion,
                    uniform_dispersion,
                    active=np.ones(len(baseline), dtype=bool),
                    direction=Direction.FULL,
                    compatibility=(0.0, 0.0, 0.0, 0.0),
                ),
                sample_ids=sample_ids,
                baseline_probability_hex=baseline_hex,
                action_probability_hex=_probability_hex(uniform),
            )
        ]
        # Duplicate elimination is branch-local.  A D01 endpoint cannot remove
        # an equal D10 endpoint because the two are eligible on disjoint rows.
        seen_outputs = {
            Direction.D01: {baseline_hex},
            Direction.D10: {baseline_hex},
        }
        for block in donors:
            challenger = np.ascontiguousarray(
                block.probabilities[indices], dtype=np.float32
            )
            challenger_dispersion = np.ascontiguousarray(
                block.seed_dispersion[indices], dtype=np.float32
            )
            compatibility = _compatibility_values(
                compatibility_features,
                role=physical_role,
                case_id=case_id,
                donor_id=block.selected_source_id,
            )
            for direction in (Direction.D01, Direction.D10):
                if direction is Direction.D01:
                    active = (baseline < np.float32(0.5)) & (
                        challenger >= np.float32(0.5)
                    )
                else:
                    active = (baseline >= np.float32(0.5)) & (
                        challenger < np.float32(0.5)
                    )
                surface = baseline.copy()
                surface[active] = challenger[active]
                probability_hex = _probability_hex(surface)
                if probability_hex in seen_outputs[direction]:
                    continue
                seen_outputs[direction].add(probability_hex)
                donor = str(block.selected_source_id)
                actions.append(
                    LabelFreeAction(
                        surface_role=surface_role,
                        center_id=menu.outer_target_id,
                        case_id=case_id,
                        arm_id=f"HXE:{donor}:{direction.value}",
                        direction=direction,
                        donor_id=donor,
                        feature_names=LABEL_FREE_FEATURE_NAMES,
                        feature_values=_feature_values(
                            baseline,
                            challenger,
                            baseline_dispersion,
                            challenger_dispersion,
                            active=active,
                            direction=direction,
                            compatibility=compatibility,
                        ),
                        sample_ids=sample_ids,
                        baseline_probability_hex=baseline_hex,
                        action_probability_hex=probability_hex,
                    )
                )
        output.append(
            LabelFreeCaseMenu(
                surface_role=surface_role,
                center_id=menu.outer_target_id,
                case_id=case_id,
                sample_ids=sample_ids,
                baseline_probability_hex=baseline_hex,
                actions=tuple(actions),
                patch_features=tuple(tuple(map(float, row)) for row in menu.patch_features[physical_role][indices]) if menu.patch_features else (),
            )
        )
        case_samples.append((case_id, sample_ids))
    return tuple(output), tuple(case_samples)


def compile_support_target_menus(
    menu: LabelFreeOuterMenu,
    *,
    compatibility_features: CompatibilityFeatures | None = None,
) -> SupportTargetMenuBundle:
    """Compile one center's source-q and target-H label-free menus."""

    if not isinstance(menu, LabelFreeOuterMenu):
        raise ProtocolError("HARP v20 adapter requires a physical outer menu.")
    if set(menu.patch_features) != {SOURCE_PHYSICAL_ROLE, TARGET_PHYSICAL_ROLE}:
        raise ProtocolError("HARP v20 source/target patch evidence must seal before source labels open.")
    candidates = tuple(center for center in CENTERS if center != menu.outer_target_id)
    source, source_samples = _compile_role(
        menu,
        physical_role=SOURCE_PHYSICAL_ROLE,
        surface_role=SurfaceRole.SOURCE_TRAIN_DEVELOPMENT,
        compatibility_features=compatibility_features,
    )
    target, target_samples = _compile_role(
        menu,
        physical_role=TARGET_PHYSICAL_ROLE,
        surface_role=SurfaceRole.TARGET_EVALUATION,
        compatibility_features=compatibility_features,
    )
    if set(sample for _case, rows in source_samples for sample in rows).intersection(
        sample for _case, rows in target_samples for sample in rows
    ):
        raise ProtocolError("HARP v20 source/evaluation sample identities overlap.")
    action_ids = (
        FULL_U_ARM_ID,
        *(
            f"HXE:{donor}:{direction.value}"
            for donor in candidates
            for direction in (Direction.D01, Direction.D10)
        ),
    )
    action_identity_hash = canonical_hash(
        {
            "schema_version": "midogpp_harp_v20_shared_action_identity_v1",
            "center_id": menu.outer_target_id,
            "candidate_source_ids": candidates,
            "physical_action_ids": ("B", "U", *(f"Hxe::{row}" for row in candidates)),
            "effective_action_ids": action_ids,
            "exact_U_FULL_count": 1,
            "directional_Hxe_only": True,
            "source_q_and_target_H_action_identity_shared": True,
            "labels_consumed": False,
        }
    )
    feature_schema_hash = canonical_hash(
        {
            "schema_version": "midogpp_harp_v20_label_free_feature_schema_v1",
            "feature_names": LABEL_FREE_FEATURE_NAMES,
            "source_and_target_schema_shared": True,
            "outcome_features_present": False,
        }
    )
    source_hash = canonical_hash(
        {
            "schema_version": "midogpp_harp_v20_source_train_menu_set_v1",
            "center_id": menu.outer_target_id,
            "surface_role": SurfaceRole.SOURCE_TRAIN_DEVELOPMENT.value,
            "action_identity_hash": action_identity_hash,
            "case_menu_hashes": tuple(row.menu_hash for row in source),
            "physical_menu_hash": menu.menu_hash,
            "labels_consumed": False,
        }
    )
    target_hash = canonical_hash(
        {
            "schema_version": "midogpp_harp_v20_target_menu_set_v1",
            "center_id": menu.outer_target_id,
            "surface_role": SurfaceRole.TARGET_EVALUATION.value,
            "action_identity_hash": action_identity_hash,
            "case_menu_hashes": tuple(row.menu_hash for row in target),
            "physical_menu_hash": menu.menu_hash,
            "labels_consumed": False,
        }
    )
    body = {
        "schema_version": "midogpp_harp_v20_source_target_menu_bundle_v1",
        "center_id": menu.outer_target_id,
        "candidate_source_ids": candidates,
        "action_identity_hash": action_identity_hash,
        "feature_schema_hash": feature_schema_hash,
        "source_train_menu_hash": source_hash,
        "target_menu_hash": target_hash,
        "physical_menu_hash": menu.menu_hash,
        "source_case_samples": source_samples,
        "target_case_samples": target_samples,
        "labels_consumed": False,
    }
    return SupportTargetMenuBundle(
        physical_menu=menu,
        candidate_source_ids=candidates,
        action_identity_hash=action_identity_hash,
        feature_schema_hash=feature_schema_hash,
        support_menus=source,
        target_menus=target,
        support_case_samples=source_samples,
        target_case_samples=target_samples,
        support_menu_hash=source_hash,
        target_menu_hash=target_hash,
        bundle_hash=canonical_hash(body),
    )


def compile_source_target_menus(
    menu: LabelFreeOuterMenu,
    *,
    compatibility_features: CompatibilityFeatures | None = None,
) -> SupportTargetMenuBundle:
    """Canonical v20 spelling for source-q / target-H compilation."""

    return compile_support_target_menus(
        menu, compatibility_features=compatibility_features
    )


def _label_value(row: object, name: str) -> object:
    return row.get(name) if isinstance(row, Mapping) else getattr(row, name, None)


def attach_support_outcome_inventory(
    bundle: SupportTargetMenuBundle,
    label_rows: Sequence[object],
) -> AttachedSupportOutcomes:
    """Open source truth into single-case capabilities and retain aggregates only."""

    expected = {
        (case_id, sample_id)
        for case_id, sample_ids in bundle.support_case_samples
        for sample_id in sample_ids
    }
    labels: dict[tuple[str, str], int] = {}
    for row in tuple(label_rows):
        center = _label_value(row, "center")
        case_id = _label_value(row, "case_id")
        sample_id = _label_value(row, "sample_id")
        label = _label_value(row, "label")
        key = (case_id, sample_id)
        if (
            center != bundle.center_id
            or type(case_id) is not str
            or type(sample_id) is not str
            or type(label) is not int
            or label not in (0, 1)
            or key in labels
        ):
            raise ProtocolError("HARP v20 source labels crossed their q-local scope.")
        labels[key] = label
    if set(labels) != expected:
        raise ProtocolError("HARP v20 source labels do not exactly cover train-q.")
    capabilities: list[SupportTruthCapability] = []
    for case_id, sample_ids in bundle.support_case_samples:
        capability = SupportTruthCapability(
            {
                (bundle.center_id, case_id): {
                    sample_id: labels[(case_id, sample_id)]
                    for sample_id in sample_ids
                }
            },
            capability_id=f"SOURCE_TRAIN_CASE::{bundle.center_id}::{case_id}",
        )
        capabilities.append(capability)
    labels.clear()
    # This persisted inventory is diagnostic only. Single-class cases must use
    # the same center/class estimand as the terminal endpoint. Every learned
    # stack independently derives its training-scope outcomes from capabilities.
    scoped = combine_truth_capabilities(tuple(capabilities))
    profiles, outcomes = scoped.derive_training_surface(bundle.support_menus)
    return AttachedSupportOutcomes(
        center_id=bundle.center_id,
        case_profiles=tuple(profiles),
        action_outcomes=tuple(outcomes),
        truth_capabilities=tuple(capabilities),
    )


def attach_support_outcomes(
    bundle: SupportTargetMenuBundle,
    label_rows: Sequence[object],
) -> tuple[SupportActionOutcome, ...]:
    return attach_support_outcome_inventory(bundle, label_rows).action_outcomes


def _physical_target_case(
    bundle: SupportTargetMenuBundle, case_id: str
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    baseline = bundle.physical_menu.target_block(ActionKind.B)
    uniform = bundle.physical_menu.target_block(ActionKind.U)
    indices = _case_indices(baseline, case_id)
    return (
        tuple(baseline.sample_ids[int(index)] for index in indices),
        np.ascontiguousarray(baseline.probabilities[indices], dtype=np.float32),
        np.ascontiguousarray(uniform.probabilities[indices], dtype=np.float32),
    )


def _component_recipe(
    menu: LabelFreeCaseMenu,
    composite: SoftTopKComposite,
    baseline: np.ndarray,
) -> tuple[
    ActionKind,
    str | None,
    str | None,
    float,
    tuple[str, ...],
    tuple[float, ...],
    tuple[np.ndarray, ...],
    np.ndarray,
]:
    if composite.kind is CompositeKind.B:
        return ActionKind.B, None, None, 0.0, (), (), (), baseline.copy()
    if composite.kind is CompositeKind.U_FULL:
        action = menu.full_action
        values = _decode_probability_hex(action.action_probability_hex)
        return (
            ActionKind.U,
            None,
            Direction.FULL.value,
            1.0,
            (action.arm_id,),
            (1.0,),
            (values,),
            values.copy(),
        )
    if (
        composite.kind not in (
            CompositeKind.D01_ONLY, CompositeKind.D10_ONLY, CompositeKind.BOTH
        )
        or type(composite.k) is not int
        or composite.mixing_lambda is None
    ):
        raise ProtocolError("HARP v20 target decision has an unknown composite.")
    ids = (*composite.d01_action_ids, *composite.d10_action_ids)
    actions = tuple(menu.action_for(value) for value in ids)
    components = tuple(
        _decode_probability_hex(row.action_probability_hex) for row in actions
    )
    weights = (1.0 / float(composite.k),) * len(actions)
    selected = reconstruct_selected_probability_blend(
        components,
        weights,
        baseline_probabilities=baseline,
        component_action_ids=ids,
    )
    return (
        ActionKind.SOFT_TOPK_PROBABILITY_BLEND,
        None,
        {CompositeKind.D01_ONLY: "D01", CompositeKind.D10_ONLY: "D10", CompositeKind.BOTH: "MIXED"}[composite.kind],
        float(composite.mixing_lambda),
        ids,
        weights,
        components,
        selected,
    )


def route_target_bundle(
    bundle: SupportTargetMenuBundle,
    router: PooledRouterLike,
    *,
    decisions: Sequence[RouteDecision] | None = None,
) -> tuple[RoutedCase, ...]:
    """Route Test-H cases with the single pooled policy and seal recipes."""

    if decisions is None:
        try:
            from ...routing.risk_aligned_router_v20 import (
                route_target_cases,
            )
        except ImportError as exc:  # pragma: no cover - incomplete source checkout
            raise ProtocolError("HARP v20 pooled router implementation is absent.") from exc
        decisions = route_target_cases(router, bundle.target_menus)
    rows = tuple(decisions)
    by_case = {row.composite.case_id: row for row in rows}
    if (
        len(by_case) != len(rows)
        or tuple(sorted(by_case)) != tuple(row.case_id for row in bundle.target_menus)
        or any(
            row.composite.center_id != bundle.center_id
            or row.composite.surface_role is not SurfaceRole.TARGET_EVALUATION
            for row in rows
        )
    ):
        raise ProtocolError("HARP v20 pooled target decisions crossed a menu boundary.")
    routed: list[RoutedCase] = []
    for menu in bundle.target_menus:
        decision = by_case[menu.case_id]
        composite = decision.composite
        samples, baseline, uniform = _physical_target_case(bundle, menu.case_id)
        (
            kind,
            source,
            direction,
            shrinkage,
            component_ids,
            weights,
            components,
            selected,
        ) = _component_recipe(menu, composite, baseline)
        final = _decode_probability_hex(composite.probability_hex)
        expected_final = (
            baseline
            if kind is ActionKind.B
            else reconstruct_shrunk_probability_blend(
                baseline, selected, shrinkage
            )
        )
        if final.tobytes(order="C") != expected_final.tobytes(order="C"):
            raise ProtocolError(
                "HARP v20 science composite failed independent byte reconstruction."
            )
        reason = (
            str(decision.fallback_reason)
            if decision.fallback_reason is not None
            else "ROUTED_POOLED_SOURCE_POLICY"
        )
        routed.append(
            RoutedCase(
                outer_target_id=bundle.center_id,
                case_id=menu.case_id,
                sample_ids=samples,
                selected_kind=kind,
                selected_source_id=source,
                reason=reason,
                baseline_probabilities=baseline,
                uniform_probabilities=uniform,
                selected_probabilities=selected,
                routed_probabilities=final,
                direction=direction,
                shrinkage=shrinkage,
                component_action_ids=component_ids,
                component_weights=weights,
                component_probabilities=components,
                decision_payload={
                    **decision.public_payload(),
                    "composite_kind": composite.kind.value,
                    "composite_k": composite.k,
                    "composite_lambda": composite.mixing_lambda,
                    "surface_role": SurfaceRole.TARGET_EVALUATION.value,
                    "router_hash": router.policy_hash,
                    "support_policy_admitted": bool(decision.admitted),
                    "selection_status": (
                        "ROUTE_SELECTED" if decision.route_selected else "EXACT_B_FALLBACK"
                    ),
                    "probability_status": (
                        "CHANGED" if decision.probability_changed else "UNCHANGED"
                    ),
                    "prediction_status": (
                        "CHANGED" if decision.prediction_changed else "UNCHANGED"
                    ),
                    "utility_status": "NOT_OPENED",
                    "all_k_lambda_probability_matrices_persisted": False,
                    "evaluation_labels_used": False,
                },
            )
        )
    return tuple(sorted(routed, key=lambda row: (row.outer_target_id, row.case_id)))


def build_support_prelabel_route_set(
    bundles: Sequence[SupportTargetMenuBundle],
    router: PooledRouterLike | Sequence[PooledRouterLike],
    *,
    target_action_hash: str | None = None,
) -> PrelabelRouteSet:
    """Apply exactly one pooled policy across all nine target menus."""

    bundle_rows = tuple(sorted(bundles, key=lambda row: row.center_id))
    if not bundle_rows or tuple(row.center_id for row in bundle_rows) != tuple(CENTERS):
        raise ProtocolError("HARP v20 route-set center inventory drifted.")
    if isinstance(router, Sequence) and not isinstance(router, (str, bytes)):
        policies = tuple(router)
        if len(policies) != 1:
            raise ProtocolError("HARP v20 requires exactly one pooled policy.")
        policy = policies[0]
    else:
        policy = router  # type: ignore[assignment]
    policy_hash = require_sha256(
        getattr(policy, "policy_hash", None), name="pooled policy hash"
    )
    model_hash = require_sha256(
        getattr(policy, "model_hash", policy_hash), name="pooled model hash"
    )
    target_hash = (
        canonical_hash(
            {
                "schema_version": "midogpp_harp_v20_target_action_set_v1",
                "menus": tuple((row.center_id, row.target_menu_hash) for row in bundle_rows),
                "target_labels_consumed": False,
            }
        )
        if target_action_hash is None
        else require_sha256(target_action_hash, name="target action hash")
    )
    cases = tuple(
        case for bundle in bundle_rows for case in route_target_bundle(bundle, policy)
    )
    return PrelabelRouteSet(
        cases=tuple(sorted(cases, key=lambda row: (row.outer_target_id, row.case_id))),
        policy_hash=policy_hash,
        model_hash=model_hash,
        target_action_hash=target_hash,
    )


__all__ = (
    "AttachedSupportOutcomes",
    "CompatibilityFeatures",
    "CompatibilityKey",
    "FULL_U_ARM_ID",
    "LABEL_FREE_FEATURE_NAMES",
    "SOURCE_PHYSICAL_ROLE",
    "SupportTargetMenuBundle",
    "TARGET_PHYSICAL_ROLE",
    "attach_support_outcome_inventory",
    "attach_support_outcomes",
    "build_support_prelabel_route_set",
    "compile_support_target_menus",
    "compile_source_target_menus",
    "route_target_bundle",
)
