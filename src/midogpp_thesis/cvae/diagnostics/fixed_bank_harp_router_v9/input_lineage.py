"""Read-only authentication of the frozen expert/generation lineage."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ...expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    N_EXPERTS,
    TRAINING_SEEDS,
)
from ...expert_bank.uniform_b_v2_promotion.serialization import (
    sampler_from_payload,
    source_frame_from_payload,
)
from ...generation import read_generation_lock
from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from ...runtime.artifact_io import read_json, sha256_file
from ....real_features.classifier_reference.classifiers import ClassifierSpec
from .config import HarpStage90V9Config
from .input_surfaces import HarpConsumedCacheIndex


@dataclass(frozen=True, slots=True)
class HarpV9PhysicalInputReceipt:
    bank_semantic_lock_hash: str
    generation_semantic_lock_hash: str
    expert_bank_index_sha256: str
    generation_lock_file_sha256: str
    classifier_config_hash: str
    classifier_contract_sha256: str
    source_local_frame_lineage_hash: str
    expert_replica_count: int
    cache_hash: str
    cache_content_sha256: str
    receipt_hash: str

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_harp_stage90_physical_input_receipt_v9",
            "bank_semantic_lock_hash": self.bank_semantic_lock_hash,
            "generation_semantic_lock_hash": self.generation_semantic_lock_hash,
            "expert_bank_index_sha256": self.expert_bank_index_sha256,
            "generation_lock_file_sha256": self.generation_lock_file_sha256,
            "classifier_config_hash": self.classifier_config_hash,
            "classifier_contract_sha256": self.classifier_contract_sha256,
            "source_local_frame_lineage_hash": self.source_local_frame_lineage_hash,
            "expert_replica_count": self.expert_replica_count,
            "all_expert_frames_source_center_local": True,
            "all_27_seed_cells_fixed_and_unselected": True,
            "cache_hash": self.cache_hash,
            "cache_content_sha256": self.cache_content_sha256,
            "labels_consumed": False,
            "predecessor_policy_used": False,
            "receipt_hash": self.receipt_hash,
        }


def validate_physical_inputs(
    config: HarpStage90V9Config, cache: HarpConsumedCacheIndex
) -> HarpV9PhysicalInputReceipt:
    """Authenticate the bank, GenerationLock, classifier, and query cache."""

    if type(config) is not HarpStage90V9Config or type(cache) is not HarpConsumedCacheIndex:
        raise ProtocolError("HARP v9 physical validation requires typed inputs.")
    bank_root = config.resolved_path("expert_bank_root")
    generation_root = config.resolved_path("generation_lock_root")
    for input_root, name in ((bank_root, "expert bank"), (generation_root, "GenerationLock")):
        if not input_root.is_dir() or input_root.is_symlink():
            raise ProtocolError(f"HARP v9 authoritative {name} root is unsafe.")
        state = read_json(input_root / "reports/run_state.json")
        validation = read_json(input_root / "reports/validation_report.json")
        if state.get("status") != "COMPLETE" or validation.get("status") != "PASS":
            raise ProtocolError(f"HARP v9 authoritative {name} is not complete and valid.")
    bank_path = bank_root / "manifests/expert_bank_index.json"
    generation_path = generation_root / "manifests/generation_lock.json"
    if (
        not bank_path.is_file()
        or bank_path.is_symlink()
        or not generation_path.is_file()
        or generation_path.is_symlink()
    ):
        raise ProtocolError("HARP v9 bank or GenerationLock is absent.")
    bank_sha = sha256_file(bank_path)
    generation_sha = sha256_file(generation_path)
    generation_lock = read_generation_lock(generation_path)
    lock_payload = generation_lock.to_payload()
    bank_payload = read_json(bank_path)
    bank_binding = lock_payload.get("bank")
    raw_classifier = lock_payload.get("classifier")
    if not isinstance(bank_binding, Mapping) or not isinstance(raw_classifier, Mapping):
        raise ProtocolError("HARP v9 GenerationLock lacks bank/classifier bindings.")
    if (
        generation_lock.bank_lock_hash != config.expected_hashes["expert_bank_lock_hash"]
        or generation_lock.generation_lock_hash != config.expected_hashes["generation_lock_hash"]
        or bank_payload.get("bank_lock_hash") != generation_lock.bank_lock_hash
        or bank_binding.get("bank_index_sha256") != bank_sha
    ):
        raise ProtocolError("HARP v9 authoritative generation lineage drifted.")
    expected_classifier_keys = {
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
    if set(raw_classifier) != expected_classifier_keys:
        raise ProtocolError("HARP v9 classifier contract schema drifted.")
    try:
        classifier = ClassifierSpec(
            family=str(raw_classifier["family"]),
            C=float(raw_classifier["C"]),
            penalty=str(raw_classifier["penalty"]),
            solver=str(raw_classifier["solver"]),
            max_iter=int(raw_classifier["max_iter"]),
            class_weight=(
                None
                if raw_classifier["class_weight"] is None
                else str(raw_classifier["class_weight"])
            ),
            random_state=int(raw_classifier["random_state"]),
            l1_ratio=(
                None if raw_classifier["l1_ratio"] is None else float(raw_classifier["l1_ratio"])
            ),
            threshold_policy=str(raw_classifier["threshold_policy"]),
            scaler_fit=str(raw_classifier["scaler_fit"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("HARP v9 classifier contract is malformed.") from exc
    if (
        classifier.config_hash != raw_classifier.get("config_hash")
        or raw_classifier.get("scaler_family") != "sklearn.preprocessing.StandardScaler"
        or raw_classifier.get("fit_in_stage_40") is not False
    ):
        raise ProtocolError("HARP v9 classifier identity drifted.")
    classifier_contract_sha = canonical_hash(
        {
            "schema_version": "midogpp_harp_classifier_semantic_identity_v9",
            "classifier": classifier.to_payload(),
            "scaler_family": raw_classifier["scaler_family"],
            "fit_in_stage_40": False,
        }
    )
    frame_lineage_hash = _audit_source_local_expert_frames(bank_root, bank_payload)
    base = {
        "schema_version": "midogpp_harp_stage90_physical_input_receipt_v9",
        "bank_semantic_lock_hash": generation_lock.bank_lock_hash,
        "generation_semantic_lock_hash": generation_lock.generation_lock_hash,
        "expert_bank_index_sha256": bank_sha,
        "generation_lock_file_sha256": generation_sha,
        "classifier_config_hash": classifier.config_hash,
        "classifier_contract_sha256": classifier_contract_sha,
        "source_local_frame_lineage_hash": frame_lineage_hash,
        "expert_replica_count": N_EXPERTS,
        "all_expert_frames_source_center_local": True,
        "all_27_seed_cells_fixed_and_unselected": True,
        "cache_hash": cache.cache_hash,
        "cache_content_sha256": cache.content_sha256,
        "labels_consumed": False,
        "predecessor_policy_used": False,
    }
    receipt = HarpV9PhysicalInputReceipt(
        bank_semantic_lock_hash=generation_lock.bank_lock_hash,
        generation_semantic_lock_hash=generation_lock.generation_lock_hash,
        expert_bank_index_sha256=bank_sha,
        generation_lock_file_sha256=generation_sha,
        classifier_config_hash=classifier.config_hash,
        classifier_contract_sha256=classifier_contract_sha,
        source_local_frame_lineage_hash=frame_lineage_hash,
        expert_replica_count=N_EXPERTS,
        cache_hash=cache.cache_hash,
        cache_content_sha256=cache.content_sha256,
        receipt_hash=canonical_hash(base),
    )
    if receipt.to_payload() != {**base, "receipt_hash": receipt.receipt_hash}:
        raise ProtocolError("HARP v9 physical input receipt construction drifted.")
    return receipt


def _audit_source_local_expert_frames(
    bank_root: Path,
    bank_payload: Mapping[str, object],
) -> str:
    """Reconstruct the frame/sampler locality fence for all fixed replicas.

    HARP does not refit or choose an expert replica.  This audit proves that
    every admitted expert carries the source-center frame and posterior
    sampler produced for its own center, and that the complete 9 x 3 bank is
    retained.  Fold-level H/q exclusion is then enforced by the action/menu
    layer without requiring a successor expert bank.
    """

    raw_records = bank_payload.get("records")
    if not isinstance(raw_records, list) or len(raw_records) != N_EXPERTS:
        raise ProtocolError("HARP v9 expert-frame lineage coverage drifted.")
    expected_keys = {
        (str(center), int(seed)) for center in CENTERS for seed in TRAINING_SEEDS
    }
    observed_keys: set[tuple[str, int]] = set()
    center_frames: dict[str, tuple[str, str, str]] = {}
    lineage_rows: list[dict[str, object]] = []
    resolved_root = bank_root.resolve()
    for raw in raw_records:
        if not isinstance(raw, Mapping):
            raise ProtocolError("HARP v9 expert-frame lineage record is malformed.")
        center = str(raw.get("source_center"))
        try:
            seed = int(raw.get("training_seed", -1))
        except (TypeError, ValueError) as exc:
            raise ProtocolError("HARP v9 expert seed identity is malformed.") from exc
        key = (center, seed)
        if (
            key in observed_keys
            or key not in expected_keys
            or raw.get("individual_expert_or_seed_selected") is not False
            or raw.get("routing_authorized") is not True
            or raw.get("fresh_source_only_training") is not True
            or raw.get("parent_checkpoint_used") is not False
        ):
            raise ProtocolError("HARP v9 fixed expert-replica policy drifted.")
        observed_keys.add(key)
        frame_path = _safe_bank_member(
            resolved_root, str(raw.get("frame_path", "")), label="source frame"
        )
        sampler_path = _safe_bank_member(
            resolved_root, str(raw.get("sampler_path", "")), label="source sampler"
        )
        if (
            sha256_file(frame_path) != raw.get("frame_file_sha256")
            or sha256_file(sampler_path) != raw.get("sampler_file_sha256")
        ):
            raise ProtocolError("HARP v9 source-local frame member drifted.")
        frame = source_frame_from_payload(read_json(frame_path))
        sampler = sampler_from_payload(read_json(sampler_path))
        frame_identity = (
            frame.source_row_hash,
            frame.state_hash,
            frame.frame.fit_sample_hash,
        )
        if (
            frame.source_center != center
            or frame.state_hash != raw.get("frame_hash")
            or sampler.source_row_hash != frame.source_row_hash
            or sampler.state_hash != raw.get("sampler_state_hash")
        ):
            raise ProtocolError("HARP v9 expert preprocessing is not source-local.")
        prior = center_frames.setdefault(center, frame_identity)
        if prior != frame_identity:
            raise ProtocolError("HARP v9 replicas disagree on their source-local frame.")
        lineage_rows.append(
            {
                "source_center": center,
                "training_seed": seed,
                "source_row_hash": frame.source_row_hash,
                "frame_hash": frame.state_hash,
                "fit_sample_hash": frame.frame.fit_sample_hash,
                "sampler_state_hash": sampler.state_hash,
                "individual_expert_or_seed_selected": False,
            }
        )
    if observed_keys != expected_keys or set(center_frames) != set(CENTERS):
        raise ProtocolError("HARP v9 expert-frame lineage inventory is incomplete.")
    return canonical_hash(
        {
            "schema_version": "midogpp_harp_source_local_expert_frame_audit_v9",
            "records": sorted(
                lineage_rows,
                key=lambda row: (str(row["source_center"]), int(row["training_seed"])),
            ),
            "source_center_count": len(CENTERS),
            "expert_replica_count": N_EXPERTS,
            "source_center_local_frames": True,
            "fixed_seed_cells": list(TRAINING_SEEDS),
            "individual_expert_or_seed_selection": False,
        }
    )


def _safe_bank_member(root: Path, relative: str, *, label: str) -> Path:
    lexical = root / relative
    if lexical.is_symlink():
        raise ProtocolError(f"HARP v9 {label} escaped the expert bank.")
    member = lexical.resolve()
    if (
        member == root
        or not member.is_relative_to(root)
        or not member.is_file()
    ):
        raise ProtocolError(f"HARP v9 {label} escaped the expert bank.")
    return member


__all__ = ("HarpV9PhysicalInputReceipt", "validate_physical_inputs")
