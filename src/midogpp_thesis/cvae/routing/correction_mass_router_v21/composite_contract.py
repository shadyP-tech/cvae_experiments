"""Actual executed directional composites and exact baseline probability invariants."""
from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Sequence
from ...protocol import ProtocolError
from .hashing import canonical_hash, require_sha256
from .contract_values import (SurfaceRole, CompositeKind, BASELINE_THRESHOLD, canonical_text, finite,
    canonical_probability_hex, decode_probability_hex)


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
            raise ProtocolError("HARP v21 composite role or kind is malformed.")
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
            raise ProtocolError("HARP v21 composite probability rows are misaligned.")
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
                raise ProtocolError("HARP v21 soft top-K composite is malformed.")
            if any(
                probability[index] != baseline[index]
                for index, value in enumerate(decoded_baseline)
                if (value < BASELINE_THRESHOLD and not d01_required)
                or (value >= BASELINE_THRESHOLD and not d10_required)
            ):
                raise ProtocolError("HARP v21 unused branches must preserve exact B bytes.")
        elif any((self.k is not None, self.mixing_lambda is not None, bool(d01), bool(d10), bool(donors))):
            raise ProtocolError("HARP v21 B/U controls cannot claim top-K members.")
        if self.kind is CompositeKind.B and probability != baseline:
            raise ProtocolError("HARP v21 B fallback must preserve exact baseline bytes.")
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
                    "schema_version": "pooled_pairwise_soft_topk_composite_v21",
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
