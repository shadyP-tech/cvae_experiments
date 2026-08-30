"""Durable pre-outcome sealing for HARP portfolio predictions."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Sequence

from ...protocol import ProtocolError
from ..harp_portfolio import HarpPortfolioDecision


_SHA256 = re.compile(r"[0-9a-f]{64}")


def _hash(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ProtocolError(f"{name} must be a lowercase SHA-256 identity.")
    return value


def _canonical_hash(payload: object) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class FrozenHarpPredictionSeal:
    decisions: tuple[HarpPortfolioDecision, ...]
    prediction_surface_hash: str
    policy_hash: str
    durable_bundle_hash: str
    independent_validation_hashes: tuple[str, ...]
    globally_durable: bool = True
    outcomes_opened_before_seal: bool = False
    seal_hash: str = field(init=False)

    def __post_init__(self) -> None:
        rows = tuple(self.decisions)
        if not rows or any(not isinstance(row, HarpPortfolioDecision) for row in rows):
            raise ProtocolError("A frozen HARP seal requires typed portfolio decisions.")
        if tuple(sorted(rows, key=lambda row: row.row_key)) != rows or len({row.row_key for row in rows}) != len(rows):
            raise ProtocolError("Frozen HARP predictions must be canonical and unique.")
        for name in ("prediction_surface_hash", "policy_hash", "durable_bundle_hash"):
            object.__setattr__(self, name, _hash(getattr(self, name), name=name))
        validations = tuple(_hash(value, name="independent validation hash") for value in self.independent_validation_hashes)
        if len(validations) < 2 or len(set(validations)) != len(validations):
            raise ProtocolError("HARP replay requires two independent durable validations.")
        if self.globally_durable is not True or self.outcomes_opened_before_seal is not False:
            raise ProtocolError("HARP predictions must be globally durable before outcomes open.")
        object.__setattr__(self, "independent_validation_hashes", validations)
        payload = {
            "schema_version": "midogpp_harp_frozen_prediction_seal_v1",
            "prediction_surface_hash": self.prediction_surface_hash,
            "policy_hash": self.policy_hash,
            "durable_bundle_hash": self.durable_bundle_hash,
            "independent_validation_hashes": list(validations),
            "rows": [
                {
                    "key": list(row.row_key),
                    "baseline": row.baseline_probability_bytes.hex(),
                    "output": row.output_probability_bytes.hex(),
                    "source": row.selected_source_id,
                    "lambda": row.selected_lambda,
                    "routed": row.routed,
                    "reason": row.reason,
                    "gain_lower": row.gain_lower,
                    "brier_upper": row.brier_upper,
                    "log_loss_upper": row.log_loss_upper,
                    "prediction_seal_hash": row.prediction_seal_hash,
                    "ensemble_receipt_hash": row.ensemble_receipt_hash,
                }
                for row in rows
            ],
        }
        object.__setattr__(self, "seal_hash", _canonical_hash(payload))


def freeze_harp_predictions(
    decisions: Sequence[HarpPortfolioDecision],
    *,
    prediction_surface_hash: str,
    policy_hash: str,
    durable_bundle_hash: str,
    independent_validation_hashes: Sequence[str],
) -> FrozenHarpPredictionSeal:
    """Freeze predictions before the replay capability can be issued."""

    return FrozenHarpPredictionSeal(
        tuple(sorted(decisions, key=lambda row: row.row_key)),
        prediction_surface_hash,
        policy_hash,
        durable_bundle_hash,
        tuple(independent_validation_hashes),
    )


__all__ = ("FrozenHarpPredictionSeal", "freeze_harp_predictions")
