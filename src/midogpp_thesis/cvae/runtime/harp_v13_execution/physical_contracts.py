"""Data-only contracts shared by HARP v13 physical execution modules."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ...generation.contracts import GenerationLock
from ....real_features.classifier_reference.classifiers import ClassifierSpec


@dataclass(frozen=True, slots=True)
class PhysicalInputReceipt:
    generation_lock: GenerationLock
    classifier: ClassifierSpec
    bank_hash: str
    generation_hash: str
    bank_index_sha256: str
    generation_file_sha256: str
    cache_hash: str
    receipt_hash: str

    def public_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_harp_v13_physical_input_receipt_v1",
            "bank_hash": self.bank_hash,
            "generation_hash": self.generation_hash,
            "bank_index_sha256": self.bank_index_sha256,
            "generation_file_sha256": self.generation_file_sha256,
            "cache_hash": self.cache_hash,
            "classifier_hash": self.classifier.config_hash,
            "labels_consumed": False,
            "receipt_hash": self.receipt_hash,
        }


@dataclass(frozen=True, slots=True)
class SourceAdapter:
    contract_hash: str
    expert_bank_root: Path
    runtime: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class StagedFrames:
    path: Path
    receipt_path: Path
    contexts: Mapping[tuple[str, str], tuple[int, int]]
    sample_ids: Mapping[tuple[str, str], tuple[str, ...]]
    case_ids: Mapping[tuple[str, str], tuple[str, ...]]
    sha256: str
    provenance_hash: str
    receipt_hash: str
    receipt_sha256: str


__all__ = ("PhysicalInputReceipt", "SourceAdapter", "StagedFrames")
