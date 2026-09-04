"""Typed, phase-separated contracts for the HARP v16 support router.

The public prediction path has no field through which evaluation outcomes can
enter.  Labels may be attached only to actions from the explicitly declared
``TARGET_TRAIN_SUPPORT`` surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import math
import struct
from typing import Sequence

from ...protocol import ProtocolError
from .hashing import canonical_hash, require_sha256


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
)


class SurfaceRole(str, Enum):
    TARGET_TRAIN_SUPPORT = "TARGET_TRAIN_SUPPORT"
    TARGET_EVALUATION = "TARGET_EVALUATION"


class Direction(str, Enum):
    D01 = "D01"
    D10 = "D10"


class ActionFamily(str, Enum):
    U = "U"
    HXE = "HXE"


def canonical_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ProtocolError(f"HARP v16 {name} must be a canonical nonempty string.")
    return value


def finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or type(value) not in (int, float):
        raise ProtocolError(f"HARP v16 {name} must be numeric.")
    output = float(value)
    if not math.isfinite(output):
        raise ProtocolError(f"HARP v16 {name} must be finite.")
    return 0.0 if output == 0.0 else output


def float32_probability_hex(values: Sequence[float]) -> tuple[str, ...]:
    cells: list[str] = []
    for raw in values:
        value = finite(raw, name="probability")
        if not 0.0 <= value <= 1.0:
            raise ProtocolError("HARP v16 probabilities must lie in [0,1].")
        cells.append(struct.pack("<f", value).hex())
    if not cells:
        raise ProtocolError("HARP v16 probability vectors cannot be empty.")
    return tuple(cells)


def canonical_probability_hex(values: Sequence[str]) -> tuple[str, ...]:
    cells: list[str] = []
    for raw in values:
        if type(raw) is not str or len(raw) != 8:
            raise ProtocolError("HARP v16 probability cells must be float32 hex.")
        try:
            packed = bytes.fromhex(raw)
        except ValueError as exc:
            raise ProtocolError("HARP v16 probability cells must be hexadecimal.") from exc
        value = struct.unpack("<f", packed)[0]
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ProtocolError("HARP v16 probabilities must lie in [0,1].")
        cells.append(raw.lower())
    if not cells:
        raise ProtocolError("HARP v16 probability vectors cannot be empty.")
    return tuple(cells)


@dataclass(frozen=True, slots=True)
class LabelFreeAction:
    outer_target_id: str
    case_id: str
    surface_role: SurfaceRole
    action_id: str
    family: ActionFamily
    direction: Direction
    candidate_source_id: str | None
    feature_names: tuple[str, ...]
    feature_values: tuple[float, ...]
    baseline_probability_hex: tuple[str, ...]
    action_probability_hex: tuple[str, ...]
    action_hash: str = field(init=False)

    def __post_init__(self) -> None:
        h = canonical_text(self.outer_target_id, name="outer target H")
        case = canonical_text(self.case_id, name="case id")
        action_id = canonical_text(self.action_id, name="action id")
        if not isinstance(self.surface_role, SurfaceRole):
            raise ProtocolError("HARP v16 action surface role is malformed.")
        if not isinstance(self.family, ActionFamily) or not isinstance(
            self.direction, Direction
        ):
            raise ProtocolError("HARP v16 action hierarchy is malformed.")
        source = self.candidate_source_id
        if self.family is ActionFamily.HXE:
            source = canonical_text(source, name="candidate source")
            if source == h:
                raise ProtocolError("HARP v16 target expert crossed the candidate fence.")
        elif source is not None:
            raise ProtocolError("HARP v16 uniform actions cannot claim an expert.")
        names = tuple(
            canonical_text(value, name="feature name") for value in self.feature_names
        )
        values = tuple(finite(value, name="feature value") for value in self.feature_values)
        lowered = tuple(value.lower() for value in names)
        if (
            not names
            or len(names) != len(values)
            or len(names) != len(set(names))
            or any(token in name for name in lowered for token in _FORBIDDEN_FEATURE_TOKENS)
        ):
            raise ProtocolError(
                "HARP v16 features are malformed or outcome-bearing."
            )
        baseline = canonical_probability_hex(self.baseline_probability_hex)
        action = canonical_probability_hex(self.action_probability_hex)
        if len(baseline) != len(action):
            raise ProtocolError("HARP v16 baseline/action vectors are misaligned.")
        object.__setattr__(self, "outer_target_id", h)
        object.__setattr__(self, "case_id", case)
        object.__setattr__(self, "action_id", action_id)
        object.__setattr__(self, "candidate_source_id", source)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "feature_values", values)
        object.__setattr__(self, "baseline_probability_hex", baseline)
        object.__setattr__(self, "action_probability_hex", action)
        object.__setattr__(
            self,
            "action_hash",
            canonical_hash(
                {
                    "schema_version": "hierarchical_support_label_free_action_v16",
                    "outer_target_id": h,
                    "case_id": case,
                    "surface_role": self.surface_role.value,
                    "action_id": action_id,
                    "family": self.family.value,
                    "direction": self.direction.value,
                    "candidate_source_id": source,
                    "feature_names": names,
                    "feature_values": values,
                    "baseline_probability_hex": baseline,
                    "action_probability_hex": action,
                    "labels_consumed": False,
                }
            ),
        )

    @property
    def is_active(self) -> bool:
        return self.action_probability_hex != self.baseline_probability_hex

    @property
    def hierarchy_key(self) -> tuple[str, str, str]:
        return (
            self.direction.value,
            self.family.value,
            self.candidate_source_id or "U",
        )

    def public_payload(self) -> dict[str, object]:
        return {
            "outer_target_id": self.outer_target_id,
            "case_id": self.case_id,
            "surface_role": self.surface_role.value,
            "action_id": self.action_id,
            "family": self.family.value,
            "direction": self.direction.value,
            "candidate_source_id": self.candidate_source_id,
            "feature_names": list(self.feature_names),
            "feature_values": list(self.feature_values),
            "baseline_probability_hex": list(self.baseline_probability_hex),
            "action_probability_hex": list(self.action_probability_hex),
            "action_hash": self.action_hash,
        }


@dataclass(frozen=True, slots=True)
class LabelFreeCaseMenu:
    outer_target_id: str
    case_id: str
    surface_role: SurfaceRole
    baseline_probability_hex: tuple[str, ...]
    actions: tuple[LabelFreeAction, ...]
    menu_hash: str = field(init=False)

    def __post_init__(self) -> None:
        h = canonical_text(self.outer_target_id, name="outer target H")
        case = canonical_text(self.case_id, name="case id")
        if not isinstance(self.surface_role, SurfaceRole):
            raise ProtocolError("HARP v16 menu surface role is malformed.")
        baseline = canonical_probability_hex(self.baseline_probability_hex)
        actions = tuple(sorted(self.actions, key=lambda row: row.action_id))
        action_ids = tuple(row.action_id for row in actions)
        physical_outputs = tuple(row.action_probability_hex for row in actions)
        if len(action_ids) != len(set(action_ids)):
            raise ProtocolError("HARP v16 menus cannot contain duplicate action ids.")
        if len(physical_outputs) != len(set(physical_outputs)):
            raise ProtocolError("HARP v16 effective menus must deduplicate physical outputs.")
        if any(
            not isinstance(row, LabelFreeAction)
            or row.outer_target_id != h
            or row.case_id != case
            or row.surface_role is not self.surface_role
            or row.baseline_probability_hex != baseline
            or not row.is_active
            for row in actions
        ):
            raise ProtocolError(
                "HARP v16 menu actions crossed a role/case boundary or retained a no-op."
            )
        schemas = {row.feature_names for row in actions}
        if len(schemas) > 1:
            raise ProtocolError("HARP v16 case actions must share one feature schema.")
        object.__setattr__(self, "outer_target_id", h)
        object.__setattr__(self, "case_id", case)
        object.__setattr__(self, "baseline_probability_hex", baseline)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(
            self,
            "menu_hash",
            canonical_hash(
                {
                    "schema_version": "hierarchical_support_case_menu_v16",
                    "outer_target_id": h,
                    "case_id": case,
                    "surface_role": self.surface_role.value,
                    "baseline_probability_hex": baseline,
                    "action_hashes": tuple(row.action_hash for row in actions),
                    "effective_noops_removed": True,
                    "physical_outputs_deduplicated": True,
                    "labels_consumed": False,
                }
            ),
        )

    def action_for(self, action_id: str) -> LabelFreeAction | None:
        return next((row for row in self.actions if row.action_id == action_id), None)

    def public_payload(self) -> dict[str, object]:
        return {
            "outer_target_id": self.outer_target_id,
            "case_id": self.case_id,
            "surface_role": self.surface_role.value,
            "baseline_probability_hex": list(self.baseline_probability_hex),
            "actions": [row.public_payload() for row in self.actions],
            "menu_hash": self.menu_hash,
        }


@dataclass(frozen=True, slots=True)
class SupportCaseClassProfile:
    """Minimal label-derived class-support state for one Train-H case.

    The profile deliberately retains neither row labels nor class counts.  It is
    sufficient to rebuild the case-equal BACC normalization *inside* every
    leave-one-case-out fold, including for cases whose effective action menu is
    empty and therefore has no action outcome row.
    """

    outer_target_id: str
    case_id: str
    supports_class_0: bool
    supports_class_1: bool
    split_role: str = "TARGET_TRAIN_SUPPORT"
    profile_hash: str = field(init=False)

    def __post_init__(self) -> None:
        outer = canonical_text(self.outer_target_id, name="profile outer target H")
        case = canonical_text(self.case_id, name="profile case id")
        if (
            type(self.supports_class_0) is not bool
            or type(self.supports_class_1) is not bool
            or not (self.supports_class_0 or self.supports_class_1)
            or self.split_role != SurfaceRole.TARGET_TRAIN_SUPPORT.value
        ):
            raise ProtocolError("HARP v16 support case class profile is malformed.")
        object.__setattr__(self, "outer_target_id", outer)
        object.__setattr__(self, "case_id", case)
        object.__setattr__(
            self,
            "profile_hash",
            canonical_hash(
                {
                    "schema_version": "hierarchical_support_case_class_profile_v16",
                    "outer_target_id": outer,
                    "case_id": case,
                    "class_support": self.class_support,
                    "split_role": self.split_role,
                    "raw_labels_persisted": False,
                    "evaluation_labels_consumed": False,
                }
            ),
        )

    @property
    def class_support(self) -> tuple[bool, bool]:
        return (self.supports_class_0, self.supports_class_1)

    def public_payload(self) -> dict[str, object]:
        return {
            "outer_target_id": self.outer_target_id,
            "case_id": self.case_id,
            "class_support": list(self.class_support),
            "split_role": self.split_role,
            "profile_hash": self.profile_hash,
            "raw_labels_persisted": False,
            "evaluation_labels_consumed": False,
        }


@dataclass(frozen=True, slots=True)
class SupportActionOutcome:
    action: LabelFreeAction
    menu_hash: str
    bacc_gain: float
    brier_delta: float
    log_loss_delta: float
    split_role: str = "TARGET_TRAIN_SUPPORT"
    class_recall_deltas: tuple[float, float] | None = None
    class_support: tuple[bool, bool] | None = None
    normalization_case_count: int | None = None
    normalization_class_support_counts: tuple[int, int] | None = None
    normalization_hash: str | None = None
    outcome_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.action, LabelFreeAction)
            or self.action.surface_role is not SurfaceRole.TARGET_TRAIN_SUPPORT
            or self.split_role != SurfaceRole.TARGET_TRAIN_SUPPORT.value
        ):
            raise ProtocolError(
                "HARP v16 outcomes require the explicit target-train support capability."
            )
        menu_hash = require_sha256(self.menu_hash, name="support menu hash")
        gain = finite(self.bacc_gain, name="BACC gain")
        brier = finite(self.brier_delta, name="Brier delta")
        log_delta = finite(self.log_loss_delta, name="log-loss delta")
        if not -1.0 <= brier <= 1.0:
            raise ProtocolError("HARP v16 support endpoints are outside metric bounds.")
        recalls = self.class_recall_deltas
        support = self.class_support
        if (recalls is None) != (support is None):
            raise ProtocolError(
                "HARP v16 class recall deltas and support flags must be paired."
            )
        normalized_recalls: tuple[float, float] | None = None
        normalized_support: tuple[bool, bool] | None = None
        if recalls is not None and support is not None:
            if (
                type(recalls) is not tuple
                or len(recalls) != 2
                or type(support) is not tuple
                or len(support) != 2
                or any(type(value) is not bool for value in support)
                or not any(support)
            ):
                raise ProtocolError("HARP v16 class-local outcome components are malformed.")
            normalized_recalls = tuple(
                finite(value, name="class recall delta") for value in recalls
            )  # type: ignore[assignment]
            normalized_support = support
            if any(not -1.0 <= value <= 1.0 for value in normalized_recalls) or any(
                (not present) and value != 0.0
                for value, present in zip(
                    normalized_recalls, normalized_support, strict=True
                )
            ):
                raise ProtocolError("HARP v16 class-local recall deltas are invalid.")
        count = self.normalization_case_count
        class_counts = self.normalization_class_support_counts
        normalization_hash = self.normalization_hash
        if len({count is None, class_counts is None, normalization_hash is None}) != 1:
            raise ProtocolError("HARP v16 BACC normalization metadata is incomplete.")
        if count is not None and class_counts is not None:
            if (
                normalized_recalls is None
                or type(count) is not int
                or count < 1
                or type(class_counts) is not tuple
                or len(class_counts) != 2
                or any(type(value) is not int or value < 1 or value > count for value in class_counts)
            ):
                raise ProtocolError("HARP v16 fold-local BACC normalization is malformed.")
            expected_gain = sum(
                0.5 * count * delta / class_count
                for delta, present, class_count in zip(
                    normalized_recalls,
                    normalized_support,
                    class_counts,
                    strict=True,
                )
                if present
            )
            if not math.isclose(gain, expected_gain, rel_tol=0.0, abs_tol=1.0e-12):
                raise ProtocolError("HARP v16 normalized BACC gain drifted.")
            normalization_hash = require_sha256(
                normalization_hash, name="fold-local normalization hash"
            )
        elif normalized_recalls is not None and normalized_support is not None:
            expected_gain = sum(
                0.5 * delta
                for delta, present in zip(
                    normalized_recalls, normalized_support, strict=True
                )
                if present
            )
            if not math.isclose(gain, expected_gain, rel_tol=0.0, abs_tol=1.0e-12):
                raise ProtocolError("HARP v16 case-local BACC gain drifted.")
        object.__setattr__(self, "menu_hash", menu_hash)
        object.__setattr__(self, "bacc_gain", gain)
        object.__setattr__(self, "brier_delta", brier)
        object.__setattr__(self, "log_loss_delta", log_delta)
        object.__setattr__(self, "class_recall_deltas", normalized_recalls)
        object.__setattr__(self, "class_support", normalized_support)
        object.__setattr__(self, "normalization_hash", normalization_hash)
        object.__setattr__(
            self,
            "outcome_hash",
            canonical_hash(
                {
                    "schema_version": "hierarchical_support_action_outcome_v16",
                    "action_hash": self.action.action_hash,
                    "menu_hash": menu_hash,
                    "bacc_gain": gain,
                    "brier_delta": brier,
                    "log_loss_delta": log_delta,
                    "class_recall_deltas": normalized_recalls,
                    "class_support": normalized_support,
                    "normalization_case_count": count,
                    "normalization_class_support_counts": class_counts,
                    "normalization_hash": normalization_hash,
                    "split_role": self.split_role,
                    "evaluation_labels_consumed": False,
                }
            ),
        )

    @property
    def harmed(self) -> bool:
        return self.bacc_gain < 0.0

    @property
    def has_class_local_components(self) -> bool:
        return self.class_recall_deltas is not None

    def with_fold_normalization(
        self,
        *,
        case_count: int,
        class_support_counts: tuple[int, int],
        normalization_hash: str,
    ) -> "SupportActionOutcome":
        """Rebuild BACC contribution using labels from the fit fold only."""

        if self.class_recall_deltas is None or self.class_support is None:
            raise ProtocolError(
                "HARP v16 fold normalization requires primitive class-local outcomes."
            )
        if (
            type(case_count) is not int
            or case_count < 1
            or type(class_support_counts) is not tuple
            or len(class_support_counts) != 2
            or any(
                type(value) is not int or value < 1 or value > case_count
                for value in class_support_counts
            )
        ):
            raise ProtocolError("HARP v16 fold normalization inputs are malformed.")
        gain = sum(
            0.5 * int(case_count) * delta / class_count
            for delta, present, class_count in zip(
                self.class_recall_deltas,
                self.class_support,
                class_support_counts,
                strict=True,
            )
            if present
        )
        return replace(
            self,
            bacc_gain=gain,
            normalization_case_count=int(case_count),
            normalization_class_support_counts=tuple(class_support_counts),
            normalization_hash=normalization_hash,
        )

    def public_payload(self) -> dict[str, object]:
        return {
            "action_hash": self.action.action_hash,
            "menu_hash": self.menu_hash,
            "outer_target_id": self.action.outer_target_id,
            "case_id": self.action.case_id,
            "action_id": self.action.action_id,
            "bacc_gain": self.bacc_gain,
            "brier_delta": self.brier_delta,
            "log_loss_delta": self.log_loss_delta,
            "class_recall_deltas": (
                None
                if self.class_recall_deltas is None
                else list(self.class_recall_deltas)
            ),
            "class_support": (
                None if self.class_support is None else list(self.class_support)
            ),
            "normalization_case_count": self.normalization_case_count,
            "normalization_class_support_counts": (
                None
                if self.normalization_class_support_counts is None
                else list(self.normalization_class_support_counts)
            ),
            "normalization_hash": self.normalization_hash,
            "harmed": self.harmed,
            "split_role": self.split_role,
            "outcome_hash": self.outcome_hash,
        }


@dataclass(frozen=True, slots=True)
class RouterFitConfig:
    ridge_alpha: float = 4.0
    maximum_numeric_features: int = 20
    minimum_support_cases: int = 12
    calibration_alpha: float = 0.20
    minimum_gain_lcb: float = 0.0
    maximum_harm_ucb: float = 0.25
    maximum_brier_delta_ucb: float = 0.002
    maximum_log_loss_delta_ucb: float = 0.005
    minimum_policy_gain: float = 0.0
    maximum_policy_harm_rate: float = 0.25
    maximum_policy_brier_delta: float = 0.002
    maximum_policy_log_loss_delta: float = 0.005
    minimum_policy_coverage: float = 0.02
    minimum_policy_routed_cases: int = 3

    def __post_init__(self) -> None:
        numeric = (
            self.ridge_alpha,
            self.calibration_alpha,
            self.minimum_gain_lcb,
            self.maximum_harm_ucb,
            self.maximum_brier_delta_ucb,
            self.maximum_log_loss_delta_ucb,
            self.minimum_policy_gain,
            self.maximum_policy_harm_rate,
            self.maximum_policy_brier_delta,
            self.maximum_policy_log_loss_delta,
            self.minimum_policy_coverage,
        )
        if (
            any(not math.isfinite(value) for value in numeric)
            or self.ridge_alpha <= 0.0
            or not 0.0 < self.calibration_alpha < 1.0
            or not 0.0 <= self.maximum_harm_ucb <= 1.0
            or not 0.0 <= self.maximum_policy_harm_rate <= 1.0
            or not 0.0 <= self.minimum_policy_coverage <= 1.0
            or int(self.maximum_numeric_features) < 1
            or int(self.minimum_support_cases) < 4
            or int(self.minimum_policy_routed_cases) < 1
        ):
            raise ProtocolError("HARP v16 router fit configuration is malformed.")

    def public_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class EndpointPrediction:
    action: LabelFreeAction
    menu_hash: str
    predicted_gain: float
    predicted_harm_probability: float
    predicted_brier_delta: float
    predicted_log_loss_delta: float
    training_case_ids: tuple[str, ...]
    feature_map_hash: str
    model_hash: str
    out_of_fold: bool
    prediction_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.action, LabelFreeAction):
            raise ProtocolError("HARP v16 endpoint predictions require a label-free action.")
        menu_hash = require_sha256(self.menu_hash, name="prediction menu hash")
        values = (
            finite(self.predicted_gain, name="predicted gain"),
            finite(self.predicted_harm_probability, name="predicted harm probability"),
            finite(self.predicted_brier_delta, name="predicted Brier delta"),
            finite(self.predicted_log_loss_delta, name="predicted log-loss delta"),
        )
        if not 0.0 <= values[1] <= 1.0:
            raise ProtocolError("HARP v16 predicted harm must lie in [0,1].")
        cases = tuple(sorted(canonical_text(value, name="training case id") for value in self.training_case_ids))
        if not cases or len(cases) != len(set(cases)):
            raise ProtocolError("HARP v16 prediction training cases are malformed.")
        if self.out_of_fold and self.action.case_id in cases:
            raise ProtocolError("HARP v16 OOF prediction included its held-out case.")
        feature_hash = require_sha256(self.feature_map_hash, name="feature map hash")
        model_hash = require_sha256(self.model_hash, name="model hash")
        object.__setattr__(self, "menu_hash", menu_hash)
        object.__setattr__(self, "predicted_gain", values[0])
        object.__setattr__(self, "predicted_harm_probability", values[1])
        object.__setattr__(self, "predicted_brier_delta", values[2])
        object.__setattr__(self, "predicted_log_loss_delta", values[3])
        object.__setattr__(self, "training_case_ids", cases)
        object.__setattr__(self, "feature_map_hash", feature_hash)
        object.__setattr__(self, "model_hash", model_hash)
        object.__setattr__(
            self,
            "prediction_hash",
            canonical_hash(
                {
                    "schema_version": "hierarchical_support_endpoint_prediction_v16",
                    "action_hash": self.action.action_hash,
                    "menu_hash": menu_hash,
                    "predicted_gain": values[0],
                    "predicted_harm_probability": values[1],
                    "predicted_brier_delta": values[2],
                    "predicted_log_loss_delta": values[3],
                    "training_case_ids": cases,
                    "feature_map_hash": feature_hash,
                    "model_hash": model_hash,
                    "out_of_fold": bool(self.out_of_fold),
                    "evaluation_labels_consumed": False,
                }
            ),
        )

    def public_payload(self) -> dict[str, object]:
        return {
            "action_hash": self.action.action_hash,
            "action_id": self.action.action_id,
            "outer_target_id": self.action.outer_target_id,
            "case_id": self.action.case_id,
            "surface_role": self.action.surface_role.value,
            "menu_hash": self.menu_hash,
            "predicted_gain": self.predicted_gain,
            "predicted_harm_probability": self.predicted_harm_probability,
            "predicted_brier_delta": self.predicted_brier_delta,
            "predicted_log_loss_delta": self.predicted_log_loss_delta,
            "training_case_ids": list(self.training_case_ids),
            "feature_map_hash": self.feature_map_hash,
            "model_hash": self.model_hash,
            "out_of_fold": self.out_of_fold,
            "prediction_hash": self.prediction_hash,
            "evaluation_labels_consumed": False,
        }


@dataclass(frozen=True, slots=True)
class CasePrediction:
    menu_hash: str
    action_predictions: tuple[EndpointPrediction, ...]
    prediction_hash: str = field(init=False)

    def __post_init__(self) -> None:
        menu_hash = require_sha256(self.menu_hash, name="case prediction menu hash")
        rows = tuple(sorted(self.action_predictions, key=lambda row: row.action.action_id))
        if rows:
            first = rows[0].action
            if (
                any(row.menu_hash != menu_hash for row in rows)
                or any(
                    row.action.outer_target_id != first.outer_target_id
                    or row.action.case_id != first.case_id
                    or row.action.surface_role is not first.surface_role
                    for row in rows
                )
                or len({row.action.action_id for row in rows}) != len(rows)
            ):
                raise ProtocolError("HARP v16 case predictions crossed a sealed menu.")
        object.__setattr__(self, "menu_hash", menu_hash)
        object.__setattr__(self, "action_predictions", rows)
        object.__setattr__(
            self,
            "prediction_hash",
            canonical_hash(
                {
                    "schema_version": "hierarchical_support_case_prediction_v16",
                    "menu_hash": menu_hash,
                    "prediction_hashes": tuple(row.prediction_hash for row in rows),
                    "evaluation_labels_consumed": False,
                }
            ),
        )

    def public_payload(self) -> dict[str, object]:
        return {
            "menu_hash": self.menu_hash,
            "action_predictions": [row.public_payload() for row in self.action_predictions],
            "prediction_hash": self.prediction_hash,
            "evaluation_labels_consumed": False,
        }


__all__ = (
    "ActionFamily",
    "CasePrediction",
    "Direction",
    "EndpointPrediction",
    "LabelFreeAction",
    "LabelFreeCaseMenu",
    "RouterFitConfig",
    "SupportCaseClassProfile",
    "SupportActionOutcome",
    "SurfaceRole",
    "canonical_probability_hex",
    "canonical_text",
    "finite",
    "float32_probability_hex",
)
