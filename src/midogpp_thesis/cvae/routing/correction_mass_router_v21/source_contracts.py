"""Authenticated source class profiles and primitive action outcomes."""
from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Sequence
from ...protocol import ProtocolError
from .hashing import canonical_hash, require_sha256
from .contract_values import SurfaceRole, Direction, canonical_text, finite
from .menu_contracts import LabelFreeAction


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
            raise ProtocolError("HARP v21 source case class profile is malformed.")
        object.__setattr__(self, "center_id", center)
        object.__setattr__(self, "case_id", case)
        object.__setattr__(
            self,
            "profile_hash",
            canonical_hash(
                {
                    "schema_version": "pooled_pairwise_support_case_profile_v21",
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
        raise ProtocolError("HARP v21 FULL has no directional opportunity label.")

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
            raise ProtocolError("HARP v21 source outcomes require a source-train action.")
        menu_hash = require_sha256(self.menu_hash, name="support menu hash")
        gain = finite(self.bacc_gain, name="BACC gain")
        brier = finite(self.brier_delta, name="Brier delta")
        logloss = finite(self.log_loss_delta, name="log-loss delta")
        if not -1.0 <= brier <= 1.0:
            raise ProtocolError("HARP v21 source action endpoints are outside metric bounds.")
        if any(value is not None and (not math.isfinite(value) or not -1.0 <= value <= 1.0)
               for value in (self.class_0_gain, self.class_1_gain)):
            raise ProtocolError("HARP v21 primitive class recall deltas are malformed.")
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
                    "schema_version": "pooled_pairwise_source_action_outcome_v21",
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


