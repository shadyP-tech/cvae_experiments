"""Immutable label-free physical actions and full-feature case menus."""
from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Sequence
from ...protocol import ProtocolError
from .hashing import canonical_hash, require_sha256
import hashlib
import numpy as np
from .contract_values import (SurfaceRole, Direction, BASELINE_THRESHOLD, canonical_text, finite,
    canonical_probability_hex, decode_probability_hex, _FORBIDDEN_FEATURE_TOKENS)


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
            raise ProtocolError("HARP v21 action role is malformed.")
        center = canonical_text(self.center_id, name="center id")
        case = canonical_text(self.case_id, name="case id")
        arm = canonical_text(self.arm_id, name="arm id")
        if not isinstance(self.direction, Direction):
            raise ProtocolError("HARP v21 action direction is malformed.")
        donor = self.donor_id
        if self.direction is Direction.FULL:
            if donor is not None:
                raise ProtocolError("HARP v21 exact-U FULL arm cannot claim a donor.")
        else:
            donor = canonical_text(donor, name="donor id")
            if donor == center:
                raise ProtocolError("HARP v21 C-minus-q/H donor fence was crossed.")
        names = tuple(canonical_text(value, name="feature name") for value in self.feature_names)
        values = tuple(finite(value, name="feature value") for value in self.feature_values)
        lowered = tuple(name.lower() for name in names)
        if (
            not names
            or len(names) != len(values)
            or len(names) != len(set(names))
            or any(token in name for name in lowered for token in _FORBIDDEN_FEATURE_TOKENS)
        ):
            raise ProtocolError("HARP v21 features are malformed or outcome-bearing.")
        samples = tuple(canonical_text(value, name="sample id") for value in self.sample_ids)
        if not samples or len(samples) != len(set(samples)):
            raise ProtocolError("HARP v21 action sample identities are malformed.")
        baseline = canonical_probability_hex(self.baseline_probability_hex)
        action = canonical_probability_hex(self.action_probability_hex)
        if len(samples) != len(baseline) or len(action) != len(baseline):
            raise ProtocolError("HARP v21 action rows are not sample-aligned.")
        if self.direction is Direction.FULL and arm not in ("U:FULL", "U_FULL"):
            raise ProtocolError("HARP v21 the exact-U primitive must use its registered U:FULL identity.")
        if self.direction in (Direction.D01, Direction.D10):
            base_values = decode_probability_hex(baseline)
            if any(left != right for value, left, right in zip(base_values, baseline, action, strict=True)
                   if (self.direction is Direction.D01 and value >= BASELINE_THRESHOLD)
                   or (self.direction is Direction.D10 and value < BASELINE_THRESHOLD)):
                raise ProtocolError("HARP v21 primitive action changed its unselected branch.")
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
                    "schema_version": "pooled_pairwise_label_free_action_v21",
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
    patch_features: object = field(default=(), compare=False, repr=False)
    patch_features_hash: str = field(init=False)
    menu_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.surface_role, SurfaceRole):
            raise ProtocolError("HARP v21 menu role is malformed.")
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
            raise ProtocolError("HARP v21 menu is incomplete or crossed a role/case boundary.")
        try:
            patch = np.asarray(self.patch_features, dtype="<f4")
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProtocolError("HARP v21 sealed patch features are malformed.") from exc
        if patch.size == 0:
            patch = np.empty((0, 3840), dtype="<f4")
        if (patch.ndim != 2 or patch.shape[1] != 3840
            or patch.shape[0] not in (0, len(samples)) or not np.isfinite(patch).all()):
            raise ProtocolError("HARP v21 sealed patch features are malformed.")
        raw_patch_bytes = patch.tobytes(order="C")
        # An immutable bytes owner prevents callers from re-enabling writes.
        patch = np.frombuffer(raw_patch_bytes, dtype="<f4").reshape(patch.shape)
        patch_hash = canonical_hash({"dtype": "float32_le", "shape": tuple(patch.shape),
            "sha256": hashlib.sha256(raw_patch_bytes).hexdigest()})
        object.__setattr__(self, "patch_features", patch)
        object.__setattr__(self, "patch_features_hash", patch_hash)
        schemas = {row.feature_names for row in actions}
        if len(schemas) != 1:
            raise ProtocolError("HARP v21 case actions must share one feature schema.")
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
                    "schema_version": "pooled_pairwise_case_menu_v21",
                    "surface_role": self.surface_role.value,
                    "center_id": center,
                    "case_id": case,
                    "sample_ids": samples,
                    "baseline_probability_hex": baseline,
                    "action_hashes": tuple(row.action_hash for row in actions),
                    "patch_features_hash": patch_hash,
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
        raise ProtocolError("HARP v21 sealed menu lacks the requested arm.")

    def public_payload(self) -> dict[str, object]:
        return {
            "surface_role": self.surface_role.value,
            "center_id": self.center_id,
            "case_id": self.case_id,
            "sample_ids": list(self.sample_ids),
            "baseline_probability_hex": list(self.baseline_probability_hex),
            "actions": [row.public_payload() for row in self.actions],
            "patch_features_hash": self.patch_features_hash,
            "menu_hash": self.menu_hash,
            "labels_consumed": False,
        }


