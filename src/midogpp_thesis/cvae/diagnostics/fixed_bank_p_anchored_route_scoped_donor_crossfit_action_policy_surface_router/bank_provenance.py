"""Exhaustive immutable-bank provenance used by P-DCAPS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from .identity import canonical_hash, require_sha256


@dataclass(frozen=True)
class BankProvenance:
    expert_checkpoint_hashes: tuple[str, ...]
    generation_lock_hash: str
    generation_settings_hash: str
    conditioning_hash: str
    seed_count_hash: str
    generated_embedding_hashes: tuple[str, ...]
    classifier_settings_hash: str
    prediction_store_hash: str
    canonical_row_order_hash: str
    provenance_hash: str = field(init=False)

    def __post_init__(self) -> None:
        values = (
            *self.expert_checkpoint_hashes,
            self.generation_lock_hash,
            self.generation_settings_hash,
            self.conditioning_hash,
            self.seed_count_hash,
            *self.generated_embedding_hashes,
            self.classifier_settings_hash,
            self.prediction_store_hash,
            self.canonical_row_order_hash,
        )
        if not self.expert_checkpoint_hashes or not self.generated_embedding_hashes:
            raise ProtocolError("P-DCAPS fixed-bank provenance inventory is empty.")
        for digest in values:
            require_sha256(digest, "fixed-bank provenance hash")
        object.__setattr__(
            self,
            "provenance_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_bank_provenance_v1",
                    "expert_checkpoint_hashes": self.expert_checkpoint_hashes,
                    "generation_lock_hash": self.generation_lock_hash,
                    "generation_settings_hash": self.generation_settings_hash,
                    "conditioning_hash": self.conditioning_hash,
                    "seed_count_hash": self.seed_count_hash,
                    "generated_embedding_hashes": self.generated_embedding_hashes,
                    "classifier_settings_hash": self.classifier_settings_hash,
                    "prediction_store_hash": self.prediction_store_hash,
                    "canonical_row_order_hash": self.canonical_row_order_hash,
                    "experts_updated": False,
                    "generation_updated": False,
                    "classifier_updated": False,
                }
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_bank_provenance_v1",
            "expert_checkpoint_hashes": list(self.expert_checkpoint_hashes),
            "generation_lock_hash": self.generation_lock_hash,
            "generation_settings_hash": self.generation_settings_hash,
            "conditioning_hash": self.conditioning_hash,
            "seed_count_hash": self.seed_count_hash,
            "generated_embedding_hashes": list(self.generated_embedding_hashes),
            "classifier_settings_hash": self.classifier_settings_hash,
            "prediction_store_hash": self.prediction_store_hash,
            "canonical_row_order_hash": self.canonical_row_order_hash,
            "experts_updated": False,
            "generation_updated": False,
            "classifier_updated": False,
            "provenance_hash": self.provenance_hash,
        }


def provenance_from_payload(payload: Mapping[str, object]) -> BankProvenance:
    row = BankProvenance(
        tuple(str(value) for value in payload["expert_checkpoint_hashes"]),  # type: ignore[index]
        str(payload["generation_lock_hash"]),
        str(payload["generation_settings_hash"]),
        str(payload["conditioning_hash"]),
        str(payload["seed_count_hash"]),
        tuple(str(value) for value in payload["generated_embedding_hashes"]),  # type: ignore[index]
        str(payload["classifier_settings_hash"]),
        str(payload["prediction_store_hash"]),
        str(payload["canonical_row_order_hash"]),
    )
    if payload.get("provenance_hash") != row.provenance_hash:
        raise ProtocolError("P-DCAPS bank provenance hash drifted.")
    return row


__all__ = ("BankProvenance", "provenance_from_payload")
