"""Independent validation of physical HARP executable and byte lineage."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from ...generation import read_generation_lock
from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json, sha256_file
from ...runtime.harp_probability_menu import (
    HarpPredictionMenuSeal,
    harp_source_stream_content_hash,
)
from ....real_features.classifier_reference.classifiers import ClassifierSpec
from ..harp_protocol.hashing import canonical_hash, require_sha256
from .workstation_runtime import LINEAGE_RECEIPT_MEMBER, SOURCE_RUNTIME_ROOT


SOURCE_LOCK_MEMBER = "manifests/frozen_source_stream_lock.json"
SOURCE_INDEX_MEMBER = "manifests/frozen_source_stream_index.json"


@dataclass(frozen=True)
class HarpAuthoritativeLineage:
    bank_semantic_lock_hash: str
    generation_semantic_lock_hash: str
    source_stream_lock_hash: str
    source_stream_index_hash: str
    source_stream_content_hash: str
    classifier_config_hash: str
    expert_bank_index_sha256: str
    generation_lock_file_sha256: str
    source_cache_lock_sha256: str
    source_cache_index_sha256: str
    source_stream_artifact_binding_hash: str
    classifier_contract_sha256: str
    receipt_hash: str

    def semantic_payload(self) -> dict[str, str]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key
            in {
                "bank_semantic_lock_hash",
                "generation_semantic_lock_hash",
                "source_stream_lock_hash",
                "source_stream_index_hash",
                "source_stream_content_hash",
                "classifier_config_hash",
            }
        }

    def authoritative_receipt_payload(self) -> dict[str, str]:
        """Return physical receipts without manufacturing semantic aliases."""

        return {
            "expert_bank_index_sha256": self.expert_bank_index_sha256,
            "generation_lock_file_sha256": self.generation_lock_file_sha256,
            "source_cache_lock_sha256": self.source_cache_lock_sha256,
            "source_cache_index_sha256": self.source_cache_index_sha256,
            "source_stream_artifact_binding_hash": (
                self.source_stream_artifact_binding_hash
            ),
            "classifier_contract_sha256": self.classifier_contract_sha256,
        }


def menu_semantic_lineage(menu: HarpPredictionMenuSeal | None) -> dict[str, str]:
    if not isinstance(menu, HarpPredictionMenuSeal):
        raise ProtocolError("HARP menu lineage requires a typed menu.")
    menu.assert_valid()
    mapping = {
        "bank_semantic_lock_hash": "bank_hash",
        "generation_semantic_lock_hash": "generation_lock_hash",
        "source_stream_lock_hash": "source_cache_hash",
        "classifier_config_hash": "classifier_hash",
    }
    result: dict[str, str] = {}
    for output, cell_field in mapping.items():
        values = {str(getattr(cell, cell_field)) for cell in menu.cells}
        if len(values) != 1 or not next(iter(values)):
            raise ProtocolError(f"HARP menu {cell_field} is not globally uniform.")
        result[output] = next(iter(values))
    return result


def load_authoritative_lineage(
    *,
    artifact_root: Path,
    expert_bank_root: Path,
    generation_lock_root: Path,
    menu: HarpPredictionMenuSeal,
) -> HarpAuthoritativeLineage:
    """Re-read all authoritative bytes and bind them to menu semantics."""

    receipt_path = artifact_root / LINEAGE_RECEIPT_MEMBER
    raw = read_json(receipt_path)
    fields = {
        "schema_version",
        "bank_semantic_lock_hash",
        "generation_semantic_lock_hash",
        "source_stream_lock_hash",
        "source_stream_index_hash",
        "source_stream_content_hash",
        "classifier_config_hash",
        "expert_bank_index_sha256",
        "generation_lock_file_sha256",
        "source_cache_lock_sha256",
        "source_cache_index_sha256",
        "source_stream_artifact_binding_hash",
        "classifier_contract_sha256",
        "receipt_hash",
    }
    if (
        set(raw) != fields
        or raw.get("schema_version")
        != "midogpp_harp_authoritative_lineage_receipt_v1"
        or raw.get("receipt_hash")
        != canonical_hash(
            {key: value for key, value in raw.items() if key != "receipt_hash"}
        )
    ):
        raise ProtocolError("HARP authoritative lineage receipt schema/hash drifted.")
    full_hash_fields = (
        "source_stream_content_hash",
        "expert_bank_index_sha256",
        "generation_lock_file_sha256",
        "source_cache_lock_sha256",
        "source_cache_index_sha256",
        "source_stream_artifact_binding_hash",
        "classifier_contract_sha256",
        "receipt_hash",
    )
    for field in full_hash_fields:
        require_sha256(raw.get(field), name=f"HARP lineage {field}")

    bank_path = expert_bank_root / "manifests/expert_bank_index.json"
    generation_path = generation_lock_root / "manifests/generation_lock.json"
    source_root = artifact_root / SOURCE_RUNTIME_ROOT
    source_lock_path = source_root / SOURCE_LOCK_MEMBER
    source_index_path = source_root / SOURCE_INDEX_MEMBER
    expected_file_hashes = {
        bank_path: raw["expert_bank_index_sha256"],
        generation_path: raw["generation_lock_file_sha256"],
        source_lock_path: raw["source_cache_lock_sha256"],
        source_index_path: raw["source_cache_index_sha256"],
    }
    for path, expected in expected_file_hashes.items():
        if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
            raise ProtocolError(f"HARP authoritative lineage member drifted: {path}.")

    generation = read_generation_lock(generation_path)
    generation_payload = generation.to_payload()
    bank = read_json(bank_path)
    source_lock = read_json(source_lock_path)
    source_index = read_json(source_index_path)
    if (
        bank.get("bank_lock_hash") != generation.bank_lock_hash
        or source_lock.get("source_stream_lock_hash")
        != raw["source_stream_lock_hash"]
        or source_index.get("source_stream_index_hash")
        != raw["source_stream_index_hash"]
        or source_lock.get("source_stream_index_hash")
        != raw["source_stream_index_hash"]
    ):
        raise ProtocolError("HARP authoritative semantic locks drifted.")
    records = source_index.get("records")
    if not isinstance(records, list) or (
        harp_source_stream_content_hash(records) != raw["source_stream_content_hash"]
    ):
        raise ProtocolError("HARP source-stream scientific content drifted.")
    expected_source_binding = canonical_hash(
        {
            "schema_version": "midogpp_harp_source_stream_artifact_binding_v1",
            "source_cache_lock_sha256": raw["source_cache_lock_sha256"],
            "source_cache_index_sha256": raw["source_cache_index_sha256"],
            "source_stream_content_hash": raw["source_stream_content_hash"],
        }
    )
    if expected_source_binding != raw["source_stream_artifact_binding_hash"]:
        raise ProtocolError("HARP source-stream artifact binding drifted.")

    classifier = _classifier_from_generation_payload(generation_payload)
    raw_classifier = generation_payload["classifier"]
    assert isinstance(raw_classifier, Mapping)
    expected_classifier_contract = canonical_hash(
        {
            "schema_version": "midogpp_harp_classifier_semantic_identity_v1",
            "classifier": classifier.to_payload(),
            "scaler_family": raw_classifier["scaler_family"],
            "fit_in_stage_40": False,
        }
    )
    semantic = menu_semantic_lineage(menu)
    if (
        raw["bank_semantic_lock_hash"] != generation.bank_lock_hash
        or raw["generation_semantic_lock_hash"]
        != generation.generation_lock_hash
        or raw["classifier_config_hash"] != classifier.config_hash
        or raw["classifier_contract_sha256"] != expected_classifier_contract
        or any(raw[key] != value for key, value in semantic.items())
    ):
        raise ProtocolError("HARP menu escaped authoritative executable lineage.")
    return HarpAuthoritativeLineage(
        **{
            key: str(value)
            for key, value in raw.items()
            if key != "schema_version"
        }
    )


def _classifier_from_generation_payload(payload: Mapping[str, object]) -> ClassifierSpec:
    raw = payload.get("classifier")
    if not isinstance(raw, Mapping):
        raise ProtocolError("HARP GenerationLock classifier binding is absent.")
    expected = {
        "family",
        "C",
        "penalty",
        "solver",
        "max_iter",
        "class_weight",
        "random_state",
        "l1_ratio",
        "threshold_policy",
        "scaler_fit",
        "config_hash",
        "scaler_family",
        "fit_in_stage_40",
    }
    if set(raw) != expected:
        raise ProtocolError("HARP GenerationLock classifier schema drifted.")
    try:
        classifier = ClassifierSpec(
            family=str(raw["family"]),
            C=float(raw["C"]),
            penalty=str(raw["penalty"]),
            solver=str(raw["solver"]),
            max_iter=int(raw["max_iter"]),
            class_weight=(None if raw["class_weight"] is None else str(raw["class_weight"])),
            random_state=int(raw["random_state"]),
            l1_ratio=(None if raw["l1_ratio"] is None else float(raw["l1_ratio"])),
            threshold_policy=str(raw["threshold_policy"]),
            scaler_fit=str(raw["scaler_fit"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("HARP GenerationLock classifier values are malformed.") from exc
    if (
        raw.get("config_hash") != classifier.config_hash
        or raw.get("scaler_family") != "sklearn.preprocessing.StandardScaler"
        or raw.get("fit_in_stage_40") is not False
    ):
        raise ProtocolError("HARP GenerationLock classifier contract drifted.")
    return classifier


__all__ = (
    "HarpAuthoritativeLineage",
    "load_authoritative_lineage",
    "menu_semantic_lineage",
)
