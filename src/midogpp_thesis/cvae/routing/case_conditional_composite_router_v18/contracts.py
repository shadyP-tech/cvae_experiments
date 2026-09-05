"""Typed, phase-separated scientific contracts for the HARP v18 router.

The source surface is keyed by ``(q, case)`` and has role
``SOURCE_TRAIN_DEVELOPMENT``.  Target menus are keyed by ``(H, case)`` and have
role ``TARGET_EVALUATION``.  A target identifier is never a fitted feature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
import math
import struct
from typing import Sequence

from ...protocol import ProtocolError
from .hashing import canonical_hash, require_sha256


PROBABILITY_CLIP = 1.0e-6
BASELINE_THRESHOLD = 0.5

_FORBIDDEN_FEATURE_TOKENS = (
    "label",
    "truth",
    "outcome",
    "oracle",
    "bacc",
    "brier",
    "log_loss",
    "logloss",
    "evaluation_score",
    "center_id",
    "target_id",
)


class SurfaceRole(str, Enum):
    SOURCE_TRAIN_DEVELOPMENT = "SOURCE_TRAIN_DEVELOPMENT"
    # Ergonomic alias; canonical serialization remains the long, phase-explicit value.
    SOURCE_TRAIN = "SOURCE_TRAIN_DEVELOPMENT"
    TARGET_EVALUATION = "TARGET_EVALUATION"


class Direction(str, Enum):
    D01 = "D01"
    D10 = "D10"
    FULL = "FULL"


class CompositeKind(str, Enum):
    B = "B"
    U_FULL = "U_FULL"
    D01_ONLY = "D01_ONLY"
    D10_ONLY = "D10_ONLY"
    BOTH = "BOTH"
    SOFT_TOPK = "BOTH"  # Historical type spelling; the successor serializes BOTH.


class AdmissionStatus(str, Enum):
    ADMITTED = "ADMITTED"
    NO_NONZERO_SAFE_OOF_COVERAGE = "NO_NONZERO_SAFE_OOF_COVERAGE"
    # Source-compatibility spelling only; serialization uses the protocol term above.
    ZERO_FRONTIER = "NO_NONZERO_SAFE_OOF_COVERAGE"
    INSUFFICIENT_ROUTED_OOF = "INSUFFICIENT_ROUTED_OOF"
    APPROXIMATE_BOUNDS_FAILED = "APPROXIMATE_BOUNDS_FAILED"


def canonical_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ProtocolError(f"HARP v18 {name} must be a canonical nonempty string.")
    return value


def finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or type(value) not in (int, float):
        raise ProtocolError(f"HARP v18 {name} must be numeric.")
    output = float(value)
    if not math.isfinite(output):
        raise ProtocolError(f"HARP v18 {name} must be finite.")
    return 0.0 if output == 0.0 else output


def canonical_probability_hex(values: Sequence[str]) -> tuple[str, ...]:
    rows = tuple(values)
    if any(type(value) is not str for value in rows):
        raise ProtocolError("HARP v18 probability cells must be float32 hex strings.")
    return _canonical_probability_tuple(rows)


@lru_cache(maxsize=2048)
def _canonical_probability_tuple(values: tuple[str, ...]) -> tuple[str, ...]:
    """Bounded, pure label-free parsing cache shared by branch configurations."""
    cells: list[str] = []
    for raw in values:
        if type(raw) is not str or len(raw) != 8:
            raise ProtocolError("HARP v18 probability cells must be little-endian float32 hex.")
        try:
            packed = bytes.fromhex(raw)
        except ValueError as exc:
            raise ProtocolError("HARP v18 probability cells must be hexadecimal.") from exc
        value = struct.unpack("<f", packed)[0]
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ProtocolError("HARP v18 probabilities must lie in [0,1].")
        cells.append(raw.lower())
    if not cells:
        raise ProtocolError("HARP v18 probability vectors cannot be empty.")
    return tuple(cells)


def float32_probability_hex(values: Sequence[float]) -> tuple[str, ...]:
    cells: list[str] = []
    for raw in values:
        value = finite(raw, name="probability")
        if not 0.0 <= value <= 1.0:
            raise ProtocolError("HARP v18 probabilities must lie in [0,1].")
        cells.append(struct.pack("<f", value).hex())
    if not cells:
        raise ProtocolError("HARP v18 probability vectors cannot be empty.")
    return tuple(cells)


def decode_probability_hex(values: Sequence[str]) -> tuple[float, ...]:
    return _decode_probability_tuple(canonical_probability_hex(values))


@lru_cache(maxsize=2048)
def _decode_probability_tuple(values: tuple[str, ...]) -> tuple[float, ...]:
    return tuple(
        float(struct.unpack("<f", bytes.fromhex(cell))[0])
        for cell in values
    )


@dataclass(frozen=True, slots=True)
class LabelFreeAction:
    """One immutable, physical action probability vector.

    ``center_id`` means source query center ``q`` on the development surface and
    outer target ``H`` on the target surface.  Directional arms are donor-backed;
    the sole FULL arm is the exact uniform-U comparator and has no donor.
    """

    surface_role: SurfaceRole
    center_id: str
    case_id: str
    arm_id: str
    direction: Direction
    donor_id: str | None
    feature_names: tuple[str, ...]
    feature_values: tuple[float, ...]
    sample_ids: tuple[str, ...]
    baseline_probability_hex: tuple[str, ...]
    action_probability_hex: tuple[str, ...]
    action_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.surface_role, SurfaceRole):
            raise ProtocolError("HARP v18 action role is malformed.")
        center = canonical_text(self.center_id, name="center id")
        case = canonical_text(self.case_id, name="case id")
        arm = canonical_text(self.arm_id, name="arm id")
        if not isinstance(self.direction, Direction):
            raise ProtocolError("HARP v18 action direction is malformed.")
        donor = self.donor_id
        if self.direction is Direction.FULL:
            if donor is not None:
                raise ProtocolError("HARP v18 exact-U FULL arm cannot claim a donor.")
        else:
            donor = canonical_text(donor, name="donor id")
            if donor == center:
                raise ProtocolError("HARP v18 C-minus-q/H donor fence was crossed.")
        names = tuple(canonical_text(value, name="feature name") for value in self.feature_names)
        values = tuple(finite(value, name="feature value") for value in self.feature_values)
        lowered = tuple(name.lower() for name in names)
        if (
            not names
            or len(names) != len(values)
            or len(names) != len(set(names))
            or any(token in name for name in lowered for token in _FORBIDDEN_FEATURE_TOKENS)
        ):
            raise ProtocolError("HARP v18 features are malformed or outcome-bearing.")
        samples = tuple(canonical_text(value, name="sample id") for value in self.sample_ids)
        if not samples or len(samples) != len(set(samples)):
            raise ProtocolError("HARP v18 action sample identities are malformed.")
        baseline = canonical_probability_hex(self.baseline_probability_hex)
        action = canonical_probability_hex(self.action_probability_hex)
        if len(samples) != len(baseline) or len(action) != len(baseline):
            raise ProtocolError("HARP v18 action rows are not sample-aligned.")
        if self.direction is Direction.FULL and arm not in ("U:FULL", "U_FULL"):
            raise ProtocolError("HARP v18 the exact-U primitive must use its registered U:FULL identity.")
        if self.direction in (Direction.D01, Direction.D10):
            base_values = decode_probability_hex(baseline)
            if any(left != right for value, left, right in zip(base_values, baseline, action, strict=True)
                   if (self.direction is Direction.D01 and value >= BASELINE_THRESHOLD)
                   or (self.direction is Direction.D10 and value < BASELINE_THRESHOLD)):
                raise ProtocolError("HARP v18 primitive action changed its unselected branch.")
        object.__setattr__(self, "center_id", center)
        object.__setattr__(self, "case_id", case)
        object.__setattr__(self, "arm_id", arm)
        object.__setattr__(self, "donor_id", donor)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "feature_values", values)
        object.__setattr__(self, "sample_ids", samples)
        object.__setattr__(self, "baseline_probability_hex", baseline)
        object.__setattr__(self, "action_probability_hex", action)
        object.__setattr__(
            self,
            "action_hash",
            canonical_hash(
                {
                    "schema_version": "pooled_pairwise_label_free_action_v18",
                    "surface_role": self.surface_role.value,
                    "center_id": center,
                    "case_id": case,
                    "arm_id": arm,
                    "direction": self.direction.value,
                    "donor_id": donor,
                    "feature_names": names,
                    "feature_values": values,
                    "sample_ids": samples,
                    "baseline_probability_hex": baseline,
                    "action_probability_hex": action,
                    "labels_consumed": False,
                }
            ),
        )

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.surface_role.value, self.center_id, self.case_id, self.arm_id)

    @property
    def is_active(self) -> bool:
        if self.direction is Direction.FULL:
            return self.action_probability_hex != self.baseline_probability_hex
        baseline = decode_probability_hex(self.baseline_probability_hex)
        relevant = (
            tuple(value < BASELINE_THRESHOLD for value in baseline)
            if self.direction is Direction.D01
            else tuple(value >= BASELINE_THRESHOLD for value in baseline)
        )
        return any(
            active and left != right
            for active, left, right in zip(
                relevant,
                self.action_probability_hex,
                self.baseline_probability_hex,
                strict=True,
            )
        )

    def public_payload(self) -> dict[str, object]:
        return {
            "surface_role": self.surface_role.value,
            "center_id": self.center_id,
            "case_id": self.case_id,
            "arm_id": self.arm_id,
            "direction": self.direction.value,
            "donor_id": self.donor_id,
            "feature_names": list(self.feature_names),
            "feature_values": list(self.feature_values),
            "sample_ids": list(self.sample_ids),
            "baseline_probability_hex": list(self.baseline_probability_hex),
            "action_probability_hex": list(self.action_probability_hex),
            "action_hash": self.action_hash,
            "labels_consumed": False,
        }


@dataclass(frozen=True, slots=True)
class LabelFreeCaseMenu:
    surface_role: SurfaceRole
    center_id: str
    case_id: str
    sample_ids: tuple[str, ...]
    baseline_probability_hex: tuple[str, ...]
    actions: tuple[LabelFreeAction, ...]
    menu_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.surface_role, SurfaceRole):
            raise ProtocolError("HARP v18 menu role is malformed.")
        center = canonical_text(self.center_id, name="menu center id")
        case = canonical_text(self.case_id, name="menu case id")
        samples = tuple(canonical_text(value, name="menu sample id") for value in self.sample_ids)
        baseline = canonical_probability_hex(self.baseline_probability_hex)
        actions = tuple(sorted(self.actions, key=lambda row: (row.direction.value, row.arm_id)))
        if (
            not samples
            or len(samples) != len(set(samples))
            or len(samples) != len(baseline)
            or not actions
            or len({row.arm_id for row in actions}) != len(actions)
            or sum(row.direction is Direction.FULL for row in actions) != 1
            or any(
                not isinstance(row, LabelFreeAction)
                or row.surface_role is not self.surface_role
                or row.center_id != center
                or row.case_id != case
                or row.sample_ids != samples
                or row.baseline_probability_hex != baseline
                for row in actions
            )
        ):
            raise ProtocolError("HARP v18 menu is incomplete or crossed a role/case boundary.")
        schemas = {row.feature_names for row in actions}
        if len(schemas) != 1:
            raise ProtocolError("HARP v18 case actions must share one feature schema.")
        object.__setattr__(self, "center_id", center)
        object.__setattr__(self, "case_id", case)
        object.__setattr__(self, "sample_ids", samples)
        object.__setattr__(self, "baseline_probability_hex", baseline)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(
            self,
            "menu_hash",
            canonical_hash(
                {
                    "schema_version": "pooled_pairwise_case_menu_v18",
                    "surface_role": self.surface_role.value,
                    "center_id": center,
                    "case_id": case,
                    "sample_ids": samples,
                    "baseline_probability_hex": baseline,
                    "action_hashes": tuple(row.action_hash for row in actions),
                    "exact_full_u_count": 1,
                    "labels_consumed": False,
                }
            ),
        )

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.surface_role.value, self.center_id, self.case_id)

    @property
    def full_action(self) -> LabelFreeAction:
        return next(row for row in self.actions if row.direction is Direction.FULL)

    def actions_for(self, direction: Direction, *, active_only: bool = True) -> tuple[LabelFreeAction, ...]:
        rows = tuple(row for row in self.actions if row.direction is direction)
        return tuple(row for row in rows if row.is_active) if active_only else rows

    def action_for(self, arm_id: str) -> LabelFreeAction:
        for row in self.actions:
            if row.arm_id == arm_id:
                return row
        raise ProtocolError("HARP v18 sealed menu lacks the requested arm.")

    def public_payload(self) -> dict[str, object]:
        return {
            "surface_role": self.surface_role.value,
            "center_id": self.center_id,
            "case_id": self.case_id,
            "sample_ids": list(self.sample_ids),
            "baseline_probability_hex": list(self.baseline_probability_hex),
            "actions": [row.public_payload() for row in self.actions],
            "menu_hash": self.menu_hash,
            "labels_consumed": False,
        }


@dataclass(frozen=True, slots=True)
class SupportCaseClassProfile:
    center_id: str
    case_id: str
    sample_count: int
    class_0_count: int
    class_1_count: int
    d01_opportunity_count: int
    d10_opportunity_count: int
    split_role: str = SurfaceRole.SOURCE_TRAIN_DEVELOPMENT.value
    profile_hash: str = field(init=False)

    def __post_init__(self) -> None:
        center = canonical_text(self.center_id, name="profile center id")
        case = canonical_text(self.case_id, name="profile case id")
        counts = (
            self.sample_count,
            self.class_0_count,
            self.class_1_count,
            self.d01_opportunity_count,
            self.d10_opportunity_count,
        )
        if (
            any(type(value) is not int or value < 0 for value in counts)
            or self.sample_count < 1
            or self.class_0_count + self.class_1_count != self.sample_count
            or self.d01_opportunity_count > self.class_1_count
            or self.d10_opportunity_count > self.class_0_count
            or self.split_role != SurfaceRole.SOURCE_TRAIN_DEVELOPMENT.value
        ):
            raise ProtocolError("HARP v18 source case class profile is malformed.")
        object.__setattr__(self, "center_id", center)
        object.__setattr__(self, "case_id", case)
        object.__setattr__(
            self,
            "profile_hash",
            canonical_hash(
                {
                    "schema_version": "pooled_pairwise_support_case_profile_v18",
                    "center_id": center,
                    "case_id": case,
                    "sample_count": self.sample_count,
                    "class_0_count": self.class_0_count,
                    "class_1_count": self.class_1_count,
                    "d01_opportunity_count": self.d01_opportunity_count,
                    "d10_opportunity_count": self.d10_opportunity_count,
                    "split_role": self.split_role,
                    "raw_labels_persisted": False,
                    "target_evaluation_labels_consumed": False,
                }
            ),
        )

    def has_opportunity(self, direction: Direction) -> bool:
        if direction is Direction.D01:
            return self.d01_opportunity_count > 0
        if direction is Direction.D10:
            return self.d10_opportunity_count > 0
        raise ProtocolError("HARP v18 FULL has no directional opportunity label.")

    def public_payload(self) -> dict[str, object]:
        return {
            "center_id": self.center_id,
            "case_id": self.case_id,
            "sample_count": self.sample_count,
            "class_0_count": self.class_0_count,
            "class_1_count": self.class_1_count,
            "d01_opportunity_count": self.d01_opportunity_count,
            "d10_opportunity_count": self.d10_opportunity_count,
            "split_role": self.split_role,
            "profile_hash": self.profile_hash,
            "raw_labels_persisted": False,
            "target_evaluation_labels_consumed": False,
        }


@dataclass(frozen=True, slots=True)
class SupportActionOutcome:
    action: LabelFreeAction
    menu_hash: str
    bacc_gain: float
    brier_delta: float
    log_loss_delta: float
    class_0_gain: float | None = None
    class_1_gain: float | None = None
    normalization_hash: str | None = None
    split_role: str = SurfaceRole.SOURCE_TRAIN_DEVELOPMENT.value
    outcome_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.action, LabelFreeAction)
            or self.action.surface_role is not SurfaceRole.SOURCE_TRAIN_DEVELOPMENT
            or self.split_role != SurfaceRole.SOURCE_TRAIN_DEVELOPMENT.value
        ):
            raise ProtocolError("HARP v18 source outcomes require a source-train action.")
        menu_hash = require_sha256(self.menu_hash, name="support menu hash")
        gain = finite(self.bacc_gain, name="BACC gain")
        brier = finite(self.brier_delta, name="Brier delta")
        logloss = finite(self.log_loss_delta, name="log-loss delta")
        if not -1.0 <= brier <= 1.0:
            raise ProtocolError("HARP v18 source action endpoints are outside metric bounds.")
        if any(value is not None and (not math.isfinite(value) or not -1.0 <= value <= 1.0)
               for value in (self.class_0_gain, self.class_1_gain)):
            raise ProtocolError("HARP v18 primitive class recall deltas are malformed.")
        if self.normalization_hash is not None:
            require_sha256(self.normalization_hash, name="primitive normalization hash")
        object.__setattr__(self, "menu_hash", menu_hash)
        object.__setattr__(self, "bacc_gain", gain)
        object.__setattr__(self, "brier_delta", brier)
        object.__setattr__(self, "log_loss_delta", logloss)
        object.__setattr__(
            self,
            "outcome_hash",
            canonical_hash(
                {
                    "schema_version": "pooled_pairwise_source_action_outcome_v18",
                    "action_hash": self.action.action_hash,
                    "menu_hash": menu_hash,
                    "bacc_gain": gain,
                    "brier_delta": brier,
                    "log_loss_delta": logloss,
                    "split_role": self.split_role,
                    "raw_labels_persisted": False,
                    "target_evaluation_labels_consumed": False,
                    "class_0_gain": self.class_0_gain,
                    "class_1_gain": self.class_1_gain,
                    "normalization_hash": self.normalization_hash,
                }
            ),
        )

    @property
    def harmed(self) -> bool:
        return self.bacc_gain < 0.0

    def public_payload(self) -> dict[str, object]:
        return {
            "center_id": self.action.center_id,
            "case_id": self.action.case_id,
            "arm_id": self.action.arm_id,
            "direction": self.action.direction.value,
            "action_hash": self.action.action_hash,
            "menu_hash": self.menu_hash,
            "bacc_gain": self.bacc_gain,
            "harm": self.harmed,
            "brier_delta": self.brier_delta,
            "log_loss_delta": self.log_loss_delta,
            "split_role": self.split_role,
            "outcome_hash": self.outcome_hash,
            "raw_labels_persisted": False,
            "target_evaluation_labels_consumed": False,
            "class_0_gain": self.class_0_gain,
            "class_1_gain": self.class_1_gain,
            "normalization_hash": self.normalization_hash,
        }


@dataclass(frozen=True, slots=True)
class RouterFitConfig:
    outer_folds: int = 5
    inner_folds: int = 4
    opportunity_ridge_alphas: tuple[float, ...] = (1.0,)  # Serialized compatibility only.
    ranker_ridge_alphas: tuple[float, ...] = (1.0,)
    stack_folds: int = 4
    action_ridge_alpha: float = 1.0
    k_values: tuple[int, ...] = (1, 2, 4)
    lambda_values: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    route_thresholds: tuple[float, ...] = (0.0, 0.0005, 0.001, 0.002, 0.005)
    maximum_numeric_features: int = 20
    required_source_case_count: int | None = 216
    required_source_center_count: int | None = 9
    minimum_cases_per_center: int = 2
    minimum_routed_oof_cases: int = 18
    minimum_routed_oof_centers: int = 6
    minimum_routed_oof_cases_per_center: int = 2
    bootstrap_replicates: int = 1024
    bootstrap_alpha: float = 0.05
    bootstrap_seed: int = 18018
    probability_clip: float = PROBABILITY_CLIP

    def __post_init__(self) -> None:
        numeric_grids = (
            self.opportunity_ridge_alphas,
            self.ranker_ridge_alphas,
            self.lambda_values,
            self.route_thresholds,
        )
        if (
            type(self.outer_folds) is not int
            or self.outer_folds < 2
            or type(self.inner_folds) is not int
            or self.inner_folds < 2
            or any(not grid for grid in numeric_grids)
            or any(not math.isfinite(value) for grid in numeric_grids for value in grid)
            or self.opportunity_ridge_alphas != (1.0,)
            or self.ranker_ridge_alphas != (1.0,)
            or self.action_ridge_alpha != 1.0
            or type(self.stack_folds) is not int or self.stack_folds < 2
            or any(value <= 0.0 for value in self.opportunity_ridge_alphas)
            or any(value <= 0.0 for value in self.ranker_ridge_alphas)
            or tuple(sorted(set(self.k_values))) != self.k_values
            or any(type(value) is not int or value < 1 for value in self.k_values)
            or tuple(sorted(set(self.lambda_values))) != self.lambda_values
            or any(not 0.0 < value <= 1.0 for value in self.lambda_values)
            or tuple(sorted(set(self.route_thresholds))) != self.route_thresholds
            or any(value < 0.0 for value in self.route_thresholds)
            or type(self.maximum_numeric_features) is not int
            or self.maximum_numeric_features < 1
            or type(self.minimum_cases_per_center) is not int
            or self.minimum_cases_per_center < 1
            or type(self.minimum_routed_oof_cases) is not int
            or self.minimum_routed_oof_cases < 1
            or type(self.minimum_routed_oof_centers) is not int
            or self.minimum_routed_oof_centers < 1
            or type(self.minimum_routed_oof_cases_per_center) is not int
            or self.minimum_routed_oof_cases_per_center < 1
            or type(self.bootstrap_replicates) is not int
            or self.bootstrap_replicates < 32
            or not 0.0 < self.bootstrap_alpha < 0.5
            or type(self.bootstrap_seed) is not int
            or self.probability_clip != PROBABILITY_CLIP
        ):
            raise ProtocolError("HARP v18 router fit configuration is malformed.")
        for value, name in (
            (self.required_source_case_count, "required source case count"),
            (self.required_source_center_count, "required source center count"),
        ):
            if value is not None and (type(value) is not int or value < 2):
                raise ProtocolError(f"HARP v18 {name} is malformed.")

    def public_payload(self) -> dict[str, object]:
        return {
            name: list(value) if isinstance(value, tuple) else value
            for name, value in (
                (name, getattr(self, name)) for name in self.__dataclass_fields__
            )
        }


@dataclass(frozen=True, slots=True)
class SoftTopKComposite:
    surface_role: SurfaceRole
    center_id: str
    case_id: str
    menu_hash: str
    kind: CompositeKind
    arm_id: str
    sample_ids: tuple[str, ...]
    baseline_probability_hex: tuple[str, ...]
    probability_hex: tuple[str, ...]
    k: int | None = None
    mixing_lambda: float | None = None
    d01_action_ids: tuple[str, ...] = ()
    d10_action_ids: tuple[str, ...] = ()
    donor_ids: tuple[str, ...] = ()
    composite_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.surface_role, SurfaceRole) or not isinstance(self.kind, CompositeKind):
            raise ProtocolError("HARP v18 composite role or kind is malformed.")
        center = canonical_text(self.center_id, name="composite center id")
        case = canonical_text(self.case_id, name="composite case id")
        arm = canonical_text(self.arm_id, name="composite arm id")
        menu_hash = require_sha256(self.menu_hash, name="composite menu hash")
        samples = tuple(canonical_text(value, name="composite sample id") for value in self.sample_ids)
        baseline = canonical_probability_hex(self.baseline_probability_hex)
        probability = canonical_probability_hex(self.probability_hex)
        d01 = tuple(canonical_text(value, name="D01 action id") for value in self.d01_action_ids)
        d10 = tuple(canonical_text(value, name="D10 action id") for value in self.d10_action_ids)
        donors = tuple(canonical_text(value, name="selected donor id") for value in self.donor_ids)
        if len(samples) != len(baseline) or len(probability) != len(baseline):
            raise ProtocolError("HARP v18 composite probability rows are misaligned.")
        if self.kind in (CompositeKind.D01_ONLY, CompositeKind.D10_ONLY, CompositeKind.BOTH):
            decoded_baseline = decode_probability_hex(baseline)
            d01_required = self.kind in (CompositeKind.D01_ONLY, CompositeKind.BOTH) and any(value < BASELINE_THRESHOLD for value in decoded_baseline)
            d10_required = self.kind in (CompositeKind.D10_ONLY, CompositeKind.BOTH) and any(value >= BASELINE_THRESHOLD for value in decoded_baseline)
            if (
                type(self.k) is not int
                or self.k < 1
                or len(d01) != (self.k if d01_required else 0)
                or len(d10) != (self.k if d10_required else 0)
                or self.mixing_lambda is None
                or not 0.0 < finite(self.mixing_lambda, name="mixing lambda") <= 1.0
                or len(donors) != len(d01) + len(d10)
            ):
                raise ProtocolError("HARP v18 soft top-K composite is malformed.")
            if any(
                probability[index] != baseline[index]
                for index, value in enumerate(decoded_baseline)
                if (value < BASELINE_THRESHOLD and not d01_required)
                or (value >= BASELINE_THRESHOLD and not d10_required)
            ):
                raise ProtocolError("HARP v18 unused branches must preserve exact B bytes.")
        elif any((self.k is not None, self.mixing_lambda is not None, bool(d01), bool(d10), bool(donors))):
            raise ProtocolError("HARP v18 B/U controls cannot claim top-K members.")
        if self.kind is CompositeKind.B and probability != baseline:
            raise ProtocolError("HARP v18 B fallback must preserve exact baseline bytes.")
        object.__setattr__(self, "center_id", center)
        object.__setattr__(self, "case_id", case)
        object.__setattr__(self, "arm_id", arm)
        object.__setattr__(self, "menu_hash", menu_hash)
        object.__setattr__(self, "sample_ids", samples)
        object.__setattr__(self, "baseline_probability_hex", baseline)
        object.__setattr__(self, "probability_hex", probability)
        object.__setattr__(self, "d01_action_ids", d01)
        object.__setattr__(self, "d10_action_ids", d10)
        object.__setattr__(self, "donor_ids", donors)
        object.__setattr__(
            self,
            "composite_hash",
            canonical_hash(
                {
                    "schema_version": "pooled_pairwise_soft_topk_composite_v18",
                    "surface_role": self.surface_role.value,
                    "center_id": center,
                    "case_id": case,
                    "menu_hash": menu_hash,
                    "kind": self.kind.value,
                    "arm_id": arm,
                    "sample_ids": samples,
                    "baseline_probability_hex": baseline,
                    "probability_hex": probability,
                    "k": self.k,
                    "mixing_lambda": self.mixing_lambda,
                    "d01_action_ids": d01,
                    "d10_action_ids": d10,
                    "donor_ids": donors,
                    "float64_accumulation": self.kind in (CompositeKind.D01_ONLY, CompositeKind.D10_ONLY, CompositeKind.BOTH),
                    "float32_serialization": True,
                    "labels_consumed": False,
                }
            ),
        )

    @property
    def route_selected(self) -> bool:
        return self.kind is not CompositeKind.B

    @property
    def probability_changed(self) -> bool:
        return self.probability_hex != self.baseline_probability_hex

    @property
    def prediction_changed(self) -> bool:
        baseline = decode_probability_hex(self.baseline_probability_hex)
        selected = decode_probability_hex(self.probability_hex)
        return any((left >= BASELINE_THRESHOLD) != (right >= BASELINE_THRESHOLD) for left, right in zip(baseline, selected, strict=True))

    @property
    def donor_entropy(self) -> float:
        if not self.donor_ids:
            return 0.0
        counts = {donor: self.donor_ids.count(donor) for donor in set(self.donor_ids)}
        total = float(len(self.donor_ids))
        return -sum((count / total) * math.log(count / total) for count in counts.values())

    def public_payload(self) -> dict[str, object]:
        return {
            "surface_role": self.surface_role.value,
            "center_id": self.center_id,
            "case_id": self.case_id,
            "menu_hash": self.menu_hash,
            "kind": self.kind.value,
            "arm_id": self.arm_id,
            "sample_ids": list(self.sample_ids),
            "baseline_probability_hex": list(self.baseline_probability_hex),
            "probability_hex": list(self.probability_hex),
            "k": self.k,
            "mixing_lambda": self.mixing_lambda,
            "d01_action_ids": list(self.d01_action_ids),
            "d10_action_ids": list(self.d10_action_ids),
            "donor_ids": list(self.donor_ids),
            "route_selected": self.route_selected,
            "probability_changed": self.probability_changed,
            "prediction_changed": self.prediction_changed,
            "donor_entropy": self.donor_entropy,
            "composite_hash": self.composite_hash,
            "labels_consumed": False,
        }


__all__ = (
    "AdmissionStatus",
    "BASELINE_THRESHOLD",
    "CompositeKind",
    "Direction",
    "LabelFreeAction",
    "LabelFreeCaseMenu",
    "PROBABILITY_CLIP",
    "RouterFitConfig",
    "SoftTopKComposite",
    "SupportActionOutcome",
    "SupportCaseClassProfile",
    "SurfaceRole",
    "canonical_probability_hex",
    "canonical_text",
    "decode_probability_hex",
    "finite",
    "float32_probability_hex",
)
