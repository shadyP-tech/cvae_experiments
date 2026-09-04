"""Role-qualified bridge from physical HARP menus to the v15 router.

This module is the only place where the runtime's dense physical probability
blocks are projected into case-level D01/D10 actions.  The projection is
label-free, preserves the selected float32 cells byte-for-byte, and applies the
same feature schema and deterministic no-op/duplicate rule to train support and
test evaluation cases.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import TypeAlias

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...routing.hierarchical_support_action_risk_router_v15 import (
    ActionFamily,
    Direction,
    FittedSupportRouter,
    LabelFreeAction,
    LabelFreeCaseMenu,
    SupportActionOutcome,
    SupportCaseClassProfile,
    SurfaceRole,
    build_effective_menu,
)
from ...routing.hierarchical_support_action_risk_router_v15.hashing import (
    canonical_hash,
    require_sha256,
)
from .contracts import (
    ActionKind,
    LabelFreeActionBlock,
    LabelFreeOuterMenu,
    PrelabelRouteSet,
    RoutedCase,
)


LABEL_FREE_FEATURE_NAMES = (
    "active_mask_fraction",
    "threshold_flip_fraction",
    "direction_aligned_branch_mass",
    "action_delta_mean",
    "action_delta_std",
    "action_delta_abs_mean",
    "baseline_probability_mean",
    "baseline_positive_branch_fraction",
    "baseline_boundary_distance_mean",
    "baseline_boundary_distance_min",
    "surface_boundary_distance_mean",
    "surface_boundary_distance_min",
    "boundary_distance_change_mean",
    "baseline_seed_dispersion_mean",
    "surface_seed_dispersion_mean",
    "surface_seed_dispersion_change_mean",
    "compatibility_mean_z",
    "compatibility_std_z",
    "compatibility_reciprocal_rank",
    "compatibility_rank_margin",
    "compatibility_available",
    "geometry_action_maximum_source_weight",
    "geometry_action_effective_source_count",
    "geometry_density_excess_over_quarter",
    "geometry_effective_sources_shortfall_from_six",
)

CompatibilityKey: TypeAlias = tuple[str, str, str]
CompatibilityFeatures: TypeAlias = Mapping[
    CompatibilityKey, Sequence[float] | Mapping[str, float]
]


@dataclass(frozen=True, slots=True)
class SupportTargetMenuBundle:
    """One H-local pair of effective support and target menu surfaces."""

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
        outer = self.physical_menu.outer_target_id
        expected_candidates = tuple(center for center in CENTERS if center != outer)
        support = tuple(sorted(self.support_menus, key=lambda row: row.case_id))
        target = tuple(sorted(self.target_menus, key=lambda row: row.case_id))
        support_samples = tuple(sorted(self.support_case_samples))
        target_samples = tuple(sorted(self.target_case_samples))
        if (
            self.candidate_source_ids != expected_candidates
            or not support
            or not target
            or any(
                row.outer_target_id != outer
                or row.surface_role is not SurfaceRole.TARGET_TRAIN_SUPPORT
                for row in support
            )
            or any(
                row.outer_target_id != outer
                or row.surface_role is not SurfaceRole.TARGET_EVALUATION
                for row in target
            )
            or {row.case_id for row in support}.intersection(
                row.case_id for row in target
            )
            or tuple(row.case_id for row in support)
            != tuple(case for case, _ in support_samples)
            or tuple(row.case_id for row in target)
            != tuple(case for case, _ in target_samples)
            or any(not samples for _, samples in (*support_samples, *target_samples))
        ):
            raise ProtocolError("HARP v15 support/target menu bundle drifted.")
        for name in (
            "action_identity_hash",
            "feature_schema_hash",
            "support_menu_hash",
            "target_menu_hash",
            "bundle_hash",
        ):
            require_sha256(getattr(self, name), name=name)
        if self.support_menu_hash == self.target_menu_hash:
            raise ProtocolError("HARP v15 support and target menu identities collided.")
        object.__setattr__(self, "support_menus", support)
        object.__setattr__(self, "target_menus", target)
        object.__setattr__(self, "support_case_samples", support_samples)
        object.__setattr__(self, "target_case_samples", target_samples)

    @property
    def outer_target_id(self) -> str:
        return self.physical_menu.outer_target_id

    def case_samples(self, role: SurfaceRole, case_id: str) -> tuple[str, ...]:
        if role is SurfaceRole.TARGET_TRAIN_SUPPORT:
            rows = self.support_case_samples
        elif role is SurfaceRole.TARGET_EVALUATION:
            rows = self.target_case_samples
        else:  # pragma: no cover - SurfaceRole is a closed enum.
            raise ProtocolError("HARP v15 case/sample lookup has an unknown role.")
        for observed_case, samples in rows:
            if observed_case == case_id:
                return samples
        raise ProtocolError("HARP v15 case/sample membership is absent.")

    def report(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_harp_v15_support_target_menu_report_v1",
            "outer_target_id": self.outer_target_id,
            "candidate_source_ids": list(self.candidate_source_ids),
            "action_identity_hash": self.action_identity_hash,
            "feature_schema_hash": self.feature_schema_hash,
            "feature_names": list(LABEL_FREE_FEATURE_NAMES),
            "support_case_count": len(self.support_menus),
            "target_case_count": len(self.target_menus),
            "support_active_action_count": sum(
                len(row.actions) for row in self.support_menus
            ),
            "target_active_action_count": sum(
                len(row.actions) for row in self.target_menus
            ),
            "support_menu_hash": self.support_menu_hash,
            "target_menu_hash": self.target_menu_hash,
            "physical_menu_hash": self.physical_menu.menu_hash,
            "bundle_hash": self.bundle_hash,
            "exact_byte_projection": True,
            "noops_removed_before_support_labels": True,
            "duplicates_removed_before_support_labels": True,
            "evaluation_labels_consumed": False,
        }


@dataclass(frozen=True, slots=True)
class AttachedSupportOutcomes:
    """Primitive label-derived endpoints plus one profile for every Train-H case."""

    outer_target_id: str
    case_profiles: tuple[SupportCaseClassProfile, ...]
    action_outcomes: tuple[SupportActionOutcome, ...]
    attachment_hash: str = field(init=False)

    def __post_init__(self) -> None:
        profiles = tuple(sorted(self.case_profiles, key=lambda row: row.case_id))
        outcomes = tuple(
            sorted(
                self.action_outcomes,
                key=lambda row: (row.action.case_id, row.action.action_id),
            )
        )
        profile_by_case = {row.case_id: row for row in profiles}
        if (
            not profiles
            or len(profile_by_case) != len(profiles)
            or any(row.outer_target_id != self.outer_target_id for row in profiles)
            or any(
                row.action.outer_target_id != self.outer_target_id
                or row.action.case_id not in profile_by_case
                or row.class_support
                != profile_by_case[row.action.case_id].class_support
                or not row.has_class_local_components
                or row.normalization_case_count is not None
                for row in outcomes
            )
        ):
            raise ProtocolError("HARP v15 attached support outcomes are malformed.")
        object.__setattr__(self, "case_profiles", profiles)
        object.__setattr__(self, "action_outcomes", outcomes)
        object.__setattr__(
            self,
            "attachment_hash",
            canonical_hash(
                {
                    "schema_version": "midogpp_harp_v15_attached_support_outcomes_v1",
                    "outer_target_id": self.outer_target_id,
                    "profile_hashes": tuple(row.profile_hash for row in profiles),
                    "outcome_hashes": tuple(row.outcome_hash for row in outcomes),
                    "full_surface_bacc_normalization_used": False,
                    "raw_support_labels_persisted": False,
                    "evaluation_labels_consumed": False,
                }
            ),
        )

    def public_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_harp_v15_attached_support_outcomes_v1",
            "outer_target_id": self.outer_target_id,
            "case_profiles": [row.public_payload() for row in self.case_profiles],
            "action_outcomes": [row.public_payload() for row in self.action_outcomes],
            "attachment_hash": self.attachment_hash,
            "full_surface_bacc_normalization_used": False,
            "raw_support_labels_persisted": False,
            "evaluation_labels_consumed": False,
        }


def _role_blocks(
    menu: LabelFreeOuterMenu, role: str
) -> tuple[LabelFreeActionBlock, ...]:
    rows = tuple(block for block in menu.blocks if block.surface_role == role)
    expected_candidates = tuple(
        center for center in CENTERS if center != menu.outer_target_id
    )
    candidates = tuple(
        block.selected_source_id
        for block in rows
        if block.action_kind is ActionKind.HXE
    )
    if (
        len(rows) != len(expected_candidates) + 2
        or sum(block.action_kind is ActionKind.B for block in rows) != 1
        or sum(block.action_kind is ActionKind.U for block in rows) != 1
        or candidates != expected_candidates
        or any(block.query_center_id != menu.outer_target_id for block in rows)
        or any(
            block.sample_ids != rows[0].sample_ids
            or block.case_ids != rows[0].case_ids
            for block in rows[1:]
        )
    ):
        raise ProtocolError("HARP v15 physical role menu is not exact C-minus-H.")
    return rows


def _single_block(
    blocks: Sequence[LabelFreeActionBlock], kind: ActionKind
) -> LabelFreeActionBlock:
    rows = tuple(row for row in blocks if row.action_kind is kind)
    if len(rows) != 1:
        raise ProtocolError("HARP v15 physical control block is ambiguous.")
    return rows[0]


def _case_order(block: LabelFreeActionBlock) -> tuple[str, ...]:
    return tuple(dict.fromkeys(block.case_ids))


def _case_indices(block: LabelFreeActionBlock, case_id: str) -> np.ndarray:
    values = np.asarray(
        [index for index, value in enumerate(block.case_ids) if value == case_id],
        dtype=np.int64,
    )
    if not len(values):
        raise ProtocolError("HARP v15 case is absent from its physical block.")
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
        raise ProtocolError("HARP v15 probability bytes are malformed.")
    packed = np.ascontiguousarray(raw, dtype="<f4").tobytes(order="C")
    return tuple(packed[index : index + 4].hex() for index in range(0, len(packed), 4))


def _decode_probability_hex(values: Sequence[str]) -> np.ndarray:
    try:
        raw = b"".join(bytes.fromhex(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ProtocolError("HARP v15 selected probability bytes are malformed.") from exc
    result = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)
    if not len(result) or not np.isfinite(result).all():
        raise ProtocolError("HARP v15 decoded probabilities are malformed.")
    return result


def _compatibility_values(
    compatibility: CompatibilityFeatures | None,
    *,
    role: str,
    case_id: str,
    candidate_source_id: str | None,
) -> tuple[float, float, float, float, float]:
    if candidate_source_id is None or compatibility is None:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    key = (role, case_id, candidate_source_id)
    if key not in compatibility:
        raise ProtocolError("HARP v15 compatibility features are incomplete.")
    raw = compatibility[key]
    if isinstance(raw, Mapping):
        try:
            values = (
                float(raw["mean_z"]),
                float(raw["std_z"]),
                float(raw["reciprocal_rank"]),
                float(raw["rank_margin"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("HARP v15 compatibility mapping is malformed.") from exc
    else:
        try:
            values = tuple(float(value) for value in raw)
        except (TypeError, ValueError) as exc:
            raise ProtocolError("HARP v15 compatibility row is malformed.") from exc
        if len(values) != 4:
            raise ProtocolError("HARP v15 compatibility row must contain four values.")
    if not all(math.isfinite(value) for value in values):
        raise ProtocolError("HARP v15 compatibility features must be finite.")
    return (*values, 1.0)


def _geometry_values(kind: ActionKind, candidate_count: int) -> tuple[float, ...]:
    if candidate_count < 1:
        raise ProtocolError("HARP v15 candidate geometry is empty.")
    if kind is ActionKind.U:
        weights = (1.0 / candidate_count,) * candidate_count
    elif kind is ActionKind.HXE:
        # The frozen eight-source geometry is 128 base plus 128 selected top-up.
        final = float(candidate_count * 128 + 128)
        weights = (256.0 / final,) + (128.0 / final,) * (candidate_count - 1)
    else:
        raise ProtocolError("HARP v15 directional actions cannot use exact B.")
    maximum = max(weights)
    effective = 1.0 / sum(value * value for value in weights)
    return (
        maximum,
        effective,
        max(maximum - 0.25, 0.0),
        max(6.0 - effective, 0.0),
    )


def _feature_values(
    baseline: np.ndarray,
    challenger: np.ndarray,
    baseline_dispersion: np.ndarray,
    challenger_dispersion: np.ndarray,
    *,
    active: np.ndarray,
    direction: Direction,
    kind: ActionKind,
    compatibility: tuple[float, float, float, float, float],
    candidate_count: int,
) -> tuple[float, ...]:
    surface = baseline.copy()
    surface[active] = challenger[active]
    surface_dispersion = baseline_dispersion.copy()
    surface_dispersion[active] = challenger_dispersion[active]
    delta = surface.astype(np.float64) - baseline.astype(np.float64)
    raw_delta = challenger.astype(np.float64) - baseline.astype(np.float64)
    baseline64 = baseline.astype(np.float64)
    surface64 = surface.astype(np.float64)
    baseline_boundary = np.abs(baseline64 - 0.5)
    surface_boundary = np.abs(surface64 - 0.5)
    aligned_mass = (
        np.maximum(raw_delta, 0.0)
        if direction is Direction.D01
        else np.maximum(-raw_delta, 0.0)
    )
    values = (
        float(np.mean(active, dtype=np.float64)),
        float(np.mean((baseline >= 0.5) != (challenger >= 0.5), dtype=np.float64)),
        float(np.mean(aligned_mass, dtype=np.float64)),
        float(np.mean(delta, dtype=np.float64)),
        float(np.std(delta, dtype=np.float64)),
        float(np.mean(np.abs(delta), dtype=np.float64)),
        float(np.mean(baseline64, dtype=np.float64)),
        float(np.mean(baseline >= 0.5, dtype=np.float64)),
        float(np.mean(baseline_boundary, dtype=np.float64)),
        float(np.min(baseline_boundary)),
        float(np.mean(surface_boundary, dtype=np.float64)),
        float(np.min(surface_boundary)),
        float(np.mean(surface_boundary - baseline_boundary, dtype=np.float64)),
        float(np.mean(baseline_dispersion, dtype=np.float64)),
        float(np.mean(surface_dispersion, dtype=np.float64)),
        float(
            np.mean(surface_dispersion, dtype=np.float64)
            - np.mean(baseline_dispersion, dtype=np.float64)
        ),
        *compatibility,
        *_geometry_values(kind, candidate_count),
    )
    if len(values) != len(LABEL_FREE_FEATURE_NAMES) or not all(
        math.isfinite(value) for value in values
    ):
        raise ProtocolError("HARP v15 label-free feature vector is malformed.")
    return values


def _action_id(
    kind: ActionKind, source: str | None, direction: Direction
) -> tuple[str, ActionFamily]:
    if kind is ActionKind.U:
        return f"U:{direction.value}", ActionFamily.U
    if kind is ActionKind.HXE and source is not None:
        return f"HXE:{source}:{direction.value}", ActionFamily.HXE
    raise ProtocolError("HARP v15 challenger identity is malformed.")


def _compile_role(
    menu: LabelFreeOuterMenu,
    *,
    role: str,
    surface_role: SurfaceRole,
    compatibility_features: CompatibilityFeatures | None,
) -> tuple[tuple[LabelFreeCaseMenu, ...], tuple[tuple[str, tuple[str, ...]], ...]]:
    blocks = _role_blocks(menu, role)
    baseline_block = _single_block(blocks, ActionKind.B)
    challengers = tuple(
        row for row in blocks if row.action_kind in {ActionKind.U, ActionKind.HXE}
    )
    output: list[LabelFreeCaseMenu] = []
    case_samples: list[tuple[str, tuple[str, ...]]] = []
    for case_id in _case_order(baseline_block):
        indices = _case_indices(baseline_block, case_id)
        baseline = np.asarray(baseline_block.probabilities[indices], dtype=np.float32)
        baseline_dispersion = np.asarray(
            baseline_block.seed_dispersion[indices], dtype=np.float32
        )
        baseline_hex = _probability_hex(baseline)
        raw_actions: list[LabelFreeAction] = []
        for challenger_block in challengers:
            challenger = np.asarray(
                challenger_block.probabilities[indices], dtype=np.float32
            )
            challenger_dispersion = np.asarray(
                challenger_block.seed_dispersion[indices], dtype=np.float32
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
                action_id, family = _action_id(
                    challenger_block.action_kind,
                    challenger_block.selected_source_id,
                    direction,
                )
                compatibility = _compatibility_values(
                    compatibility_features,
                    role=role,
                    case_id=case_id,
                    candidate_source_id=challenger_block.selected_source_id,
                )
                raw_actions.append(
                    LabelFreeAction(
                        outer_target_id=menu.outer_target_id,
                        case_id=case_id,
                        surface_role=surface_role,
                        action_id=action_id,
                        family=family,
                        direction=direction,
                        candidate_source_id=challenger_block.selected_source_id,
                        feature_names=LABEL_FREE_FEATURE_NAMES,
                        feature_values=_feature_values(
                            baseline,
                            challenger,
                            baseline_dispersion,
                            challenger_dispersion,
                            active=active,
                            direction=direction,
                            kind=challenger_block.action_kind,
                            compatibility=compatibility,
                            candidate_count=len(CENTERS) - 1,
                        ),
                        baseline_probability_hex=baseline_hex,
                        action_probability_hex=_probability_hex(surface),
                    )
                )
        output.append(
            build_effective_menu(
                outer_target_id=menu.outer_target_id,
                case_id=case_id,
                surface_role=surface_role,
                baseline_probability_hex=baseline_hex,
                raw_actions=raw_actions,
            )
        )
        case_samples.append(
            (
                case_id,
                tuple(baseline_block.sample_ids[index] for index in indices.tolist()),
            )
        )
    return tuple(output), tuple(case_samples)


def compile_support_target_menus(
    menu: LabelFreeOuterMenu,
    *,
    compatibility_features: CompatibilityFeatures | None = None,
) -> SupportTargetMenuBundle:
    """Compile the shared support/target effective menu for one target H."""

    if not isinstance(menu, LabelFreeOuterMenu):
        raise ProtocolError("HARP v15 support adapter requires a physical outer menu.")
    candidates = tuple(center for center in CENTERS if center != menu.outer_target_id)
    support, support_samples = _compile_role(
        menu,
        role="support",
        surface_role=SurfaceRole.TARGET_TRAIN_SUPPORT,
        compatibility_features=compatibility_features,
    )
    target, target_samples = _compile_role(
        menu,
        role="target",
        surface_role=SurfaceRole.TARGET_EVALUATION,
        compatibility_features=compatibility_features,
    )
    if set(sample for _, rows in support_samples for sample in rows).intersection(
        sample for _, rows in target_samples for sample in rows
    ):
        raise ProtocolError("HARP v15 support/evaluation sample identities overlap.")
    directional_ids = tuple(
        [f"U:{direction.value}" for direction in (Direction.D01, Direction.D10)]
        + [
            f"HXE:{source}:{direction.value}"
            for source in candidates
            for direction in (Direction.D01, Direction.D10)
        ]
    )
    action_identity_hash = canonical_hash(
        {
            "schema_version": "midogpp_harp_v15_shared_action_identity_v1",
            "outer_target_id": menu.outer_target_id,
            "candidate_source_ids": candidates,
            "physical_action_ids": ("B", "U", *(f"Hxe::{row}" for row in candidates)),
            "directional_action_ids": directional_ids,
            "support_and_target_action_identity_shared": True,
            "labels_consumed": False,
        }
    )
    feature_schema_hash = canonical_hash(
        {
            "schema_version": "midogpp_harp_v15_shared_label_free_feature_schema_v1",
            "feature_names": LABEL_FREE_FEATURE_NAMES,
            "support_and_target_schema_shared": True,
            "outcome_features_present": False,
        }
    )
    support_hash = canonical_hash(
        {
            "schema_version": "midogpp_harp_v15_support_effective_menu_set_v1",
            "outer_target_id": menu.outer_target_id,
            "surface_role": SurfaceRole.TARGET_TRAIN_SUPPORT.value,
            "action_identity_hash": action_identity_hash,
            "case_menu_hashes": tuple(row.menu_hash for row in support),
            "physical_menu_hash": menu.menu_hash,
            "labels_consumed": False,
        }
    )
    target_hash = canonical_hash(
        {
            "schema_version": "midogpp_harp_v15_target_effective_menu_set_v1",
            "outer_target_id": menu.outer_target_id,
            "surface_role": SurfaceRole.TARGET_EVALUATION.value,
            "action_identity_hash": action_identity_hash,
            "case_menu_hashes": tuple(row.menu_hash for row in target),
            "physical_menu_hash": menu.menu_hash,
            "labels_consumed": False,
        }
    )
    bundle_body = {
        "schema_version": "midogpp_harp_v15_support_target_menu_bundle_v1",
        "outer_target_id": menu.outer_target_id,
        "candidate_source_ids": candidates,
        "action_identity_hash": action_identity_hash,
        "feature_schema_hash": feature_schema_hash,
        "support_menu_hash": support_hash,
        "target_menu_hash": target_hash,
        "physical_menu_hash": menu.menu_hash,
        "support_case_samples": support_samples,
        "target_case_samples": target_samples,
        "labels_consumed": False,
    }
    return SupportTargetMenuBundle(
        physical_menu=menu,
        candidate_source_ids=candidates,
        action_identity_hash=action_identity_hash,
        feature_schema_hash=feature_schema_hash,
        support_menus=support,
        target_menus=target,
        support_case_samples=support_samples,
        target_case_samples=target_samples,
        support_menu_hash=support_hash,
        target_menu_hash=target_hash,
        bundle_hash=canonical_hash(bundle_body),
    )


def _label_value(row: object, name: str) -> object:
    if isinstance(row, Mapping):
        return row.get(name)
    return getattr(row, name, None)


def attach_support_outcome_inventory(
    bundle: SupportTargetMenuBundle,
    label_rows: Sequence[object],
) -> AttachedSupportOutcomes:
    """Attach exactly one center's authenticated Train-H labels to its menus."""

    rows = tuple(label_rows)
    expected = {
        (case_id, sample_id)
        for case_id, sample_ids in bundle.support_case_samples
        for sample_id in sample_ids
    }
    labels: dict[tuple[str, str], int] = {}
    for row in rows:
        center = _label_value(row, "center")
        case_id = _label_value(row, "case_id")
        sample_id = _label_value(row, "sample_id")
        label = _label_value(row, "label")
        key = (case_id, sample_id)
        if (
            center != bundle.outer_target_id
            or type(case_id) is not str
            or type(sample_id) is not str
            or type(label) is not int
            or label not in (0, 1)
            or key in labels
        ):
            raise ProtocolError("HARP v15 support labels crossed their H-local scope.")
        labels[key] = label
    if set(labels) != expected:
        raise ProtocolError("HARP v15 support labels do not exactly cover Train-H.")
    truth_by_case = {
        case_id: np.asarray(
            [labels[(case_id, sample)] for sample in samples], dtype=np.int64
        )
        for case_id, samples in bundle.support_case_samples
    }
    menu_by_case = {row.case_id: row for row in bundle.support_menus}
    profiles = tuple(
        SupportCaseClassProfile(
            outer_target_id=bundle.outer_target_id,
            case_id=case_id,
            supports_class_0=bool(np.any(truth == 0)),
            supports_class_1=bool(np.any(truth == 1)),
        )
        for case_id, truth in sorted(truth_by_case.items())
    )
    outcomes: list[SupportActionOutcome] = []
    for case_id, _samples in bundle.support_case_samples:
        menu = menu_by_case[case_id]
        truth = truth_by_case[case_id]
        baseline_probability = _decode_probability_hex(menu.baseline_probability_hex)
        baseline_prediction = baseline_probability >= 0.5
        baseline_brier = float(
            np.mean((baseline_probability - truth) ** 2, dtype=np.float64)
        )
        clipped_baseline = np.clip(baseline_probability, 1.0e-6, 1.0 - 1.0e-6)
        baseline_log_loss = float(
            np.mean(
                -(truth * np.log(clipped_baseline) + (1 - truth) * np.log(1.0 - clipped_baseline)),
                dtype=np.float64,
            )
        )
        class_support = tuple(bool(np.any(truth == label)) for label in (0, 1))
        for action in menu.actions:
            action_probability = _decode_probability_hex(action.action_probability_hex)
            action_prediction = action_probability >= 0.5
            recall_deltas = tuple(
                (
                    float(
                        np.mean(action_prediction[truth == label] == truth[truth == label])
                        - np.mean(baseline_prediction[truth == label] == truth[truth == label])
                    )
                    if present
                    else 0.0
                )
                for label, present in zip((0, 1), class_support, strict=True)
            )
            observed_brier = float(
                np.mean((action_probability - truth) ** 2, dtype=np.float64)
            )
            clipped_action = np.clip(action_probability, 1.0e-6, 1.0 - 1.0e-6)
            observed_log_loss = float(
                np.mean(
                    -(truth * np.log(clipped_action) + (1 - truth) * np.log(1.0 - clipped_action)),
                    dtype=np.float64,
                )
            )
            outcomes.append(
                SupportActionOutcome(
                    action=action,
                    menu_hash=menu.menu_hash,
                    bacc_gain=sum(
                        0.5 * delta
                        for delta, present in zip(
                            recall_deltas, class_support, strict=True
                        )
                        if present
                    ),
                    brier_delta=observed_brier - baseline_brier,
                    log_loss_delta=observed_log_loss - baseline_log_loss,
                    class_recall_deltas=recall_deltas,
                    class_support=class_support,
                )
            )
    return AttachedSupportOutcomes(
        outer_target_id=bundle.outer_target_id,
        case_profiles=profiles,
        action_outcomes=tuple(outcomes),
    )


def attach_support_outcomes(
    bundle: SupportTargetMenuBundle,
    label_rows: Sequence[object],
) -> tuple[SupportActionOutcome, ...]:
    """Compatibility view exposing the primitive action rows only."""

    return attach_support_outcome_inventory(bundle, label_rows).action_outcomes


def _physical_case(
    bundle: SupportTargetMenuBundle, case_id: str
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    baseline = bundle.physical_menu.target_block(ActionKind.B)
    uniform = bundle.physical_menu.target_block(ActionKind.U)
    indices = _case_indices(baseline, case_id)
    samples = tuple(baseline.sample_ids[index] for index in indices.tolist())
    return (
        samples,
        np.asarray(baseline.probabilities[indices], dtype=np.float32).copy(),
        np.asarray(uniform.probabilities[indices], dtype=np.float32).copy(),
    )


def route_target_bundle(
    bundle: SupportTargetMenuBundle,
    router: FittedSupportRouter,
) -> tuple[RoutedCase, ...]:
    """Route all Test-H cases and convert exact decisions to runtime objects."""

    if router.outer_target_id != bundle.outer_target_id:
        raise ProtocolError("HARP v15 support router crossed its target H.")
    routed: list[RoutedCase] = []
    for menu in bundle.target_menus:
        decision = router.route(menu)
        samples, baseline, uniform = _physical_case(bundle, menu.case_id)
        if decision.exact_b_fallback:
            selected_kind = ActionKind.B
            selected_source = None
            direction = None
            components: tuple[np.ndarray, ...] = ()
            component_ids: tuple[str, ...] = ()
            weights: tuple[float, ...] = ()
            shrinkage = 0.0
        else:
            action = menu.action_for(decision.selected_action_id)
            if action is None:
                raise ProtocolError("HARP v15 routed action escaped the target menu.")
            selected_kind = (
                ActionKind.U if action.family is ActionFamily.U else ActionKind.HXE
            )
            selected_source = action.candidate_source_id
            direction = action.direction.value
            components = (_decode_probability_hex(action.action_probability_hex),)
            component_ids = (action.action_id,)
            weights = (1.0,)
            shrinkage = 1.0
        selected = _decode_probability_hex(decision.probability_hex)
        routed.append(
            RoutedCase(
                outer_target_id=bundle.outer_target_id,
                case_id=menu.case_id,
                sample_ids=samples,
                selected_kind=selected_kind,
                selected_source_id=selected_source,
                reason=decision.reason,
                baseline_probabilities=baseline,
                uniform_probabilities=uniform,
                selected_probabilities=selected,
                routed_probabilities=selected.copy(),
                direction=direction,
                shrinkage=shrinkage,
                component_action_ids=component_ids,
                component_weights=weights,
                component_probabilities=components,
                decision_payload={
                    **decision.public_payload(),
                    "surface_role": SurfaceRole.TARGET_EVALUATION.value,
                    "support_router_hash": router.router_hash,
                    "support_policy_admitted": router.admission.admitted,
                    "support_model_is_null": router.endpoint_model.is_null,
                    "evaluation_labels_used": False,
                },
            )
        )
    return tuple(sorted(routed, key=lambda row: (row.outer_target_id, row.case_id)))


def build_support_prelabel_route_set(
    bundles: Sequence[SupportTargetMenuBundle],
    routers: Sequence[FittedSupportRouter],
    *,
    target_action_hash: str | None = None,
) -> PrelabelRouteSet:
    """Route every target case with one H-local router and seal exact B fallback."""

    bundle_rows = tuple(sorted(bundles, key=lambda row: row.outer_target_id))
    router_rows = tuple(sorted(routers, key=lambda row: row.outer_target_id))
    if (
        not bundle_rows
        or tuple(row.outer_target_id for row in bundle_rows)
        != tuple(row.outer_target_id for row in router_rows)
        or len({row.outer_target_id for row in bundle_rows}) != len(bundle_rows)
    ):
        raise ProtocolError("HARP v15 route-set outer-target inventory drifted.")
    model_hash = canonical_hash(
        {
            "schema_version": "midogpp_harp_v15_target_local_model_set_v1",
            "models": tuple(
                (row.outer_target_id, row.endpoint_model.model_hash)
                for row in router_rows
            ),
            "evaluation_labels_consumed": False,
        }
    )
    policy_hash = canonical_hash(
        {
            "schema_version": "midogpp_harp_v15_target_local_policy_set_v1",
            "routers": tuple(
                (row.outer_target_id, row.router_hash, row.admission.admission_hash)
                for row in router_rows
            ),
            "evaluation_labels_consumed": False,
        }
    )
    target_hash = (
        canonical_hash(
            {
                "schema_version": "midogpp_harp_v15_target_action_set_v1",
                "menus": tuple(
                    (row.outer_target_id, row.target_menu_hash) for row in bundle_rows
                ),
                "evaluation_labels_consumed": False,
            }
        )
        if target_action_hash is None
        else require_sha256(target_action_hash, name="target action hash")
    )
    cases = tuple(
        case
        for bundle, router in zip(bundle_rows, router_rows, strict=True)
        for case in route_target_bundle(bundle, router)
    )
    return PrelabelRouteSet(
        cases=tuple(sorted(cases, key=lambda row: (row.outer_target_id, row.case_id))),
        policy_hash=policy_hash,
        model_hash=model_hash,
        target_action_hash=target_hash,
    )


__all__ = (
    "CompatibilityFeatures",
    "CompatibilityKey",
    "LABEL_FREE_FEATURE_NAMES",
    "SupportTargetMenuBundle",
    "attach_support_outcomes",
    "attach_support_outcome_inventory",
    "AttachedSupportOutcomes",
    "build_support_prelabel_route_set",
    "compile_support_target_menus",
    "route_target_bundle",
)
