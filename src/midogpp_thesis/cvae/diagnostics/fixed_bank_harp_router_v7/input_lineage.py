"""Read-only authentication of the frozen expert/generation lineage."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ...generation import read_generation_lock
from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from ...runtime.artifact_io import read_json, sha256_file
from ....real_features.classifier_reference.classifiers import ClassifierSpec
from .config import HarpStage90V7Config
from .input_surfaces import HarpConsumedCacheIndex


@dataclass(frozen=True, slots=True)
class HarpV7PhysicalInputReceipt:
    bank_semantic_lock_hash: str
    generation_semantic_lock_hash: str
    expert_bank_index_sha256: str
    generation_lock_file_sha256: str
    classifier_config_hash: str
    classifier_contract_sha256: str
    cache_hash: str
    cache_content_sha256: str
    receipt_hash: str

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_harp_stage90_physical_input_receipt_v7",
            "bank_semantic_lock_hash": self.bank_semantic_lock_hash,
            "generation_semantic_lock_hash": self.generation_semantic_lock_hash,
            "expert_bank_index_sha256": self.expert_bank_index_sha256,
            "generation_lock_file_sha256": self.generation_lock_file_sha256,
            "classifier_config_hash": self.classifier_config_hash,
            "classifier_contract_sha256": self.classifier_contract_sha256,
            "cache_hash": self.cache_hash,
            "cache_content_sha256": self.cache_content_sha256,
            "labels_consumed": False,
            "predecessor_policy_used": False,
            "receipt_hash": self.receipt_hash,
        }


def validate_physical_inputs(
    config: HarpStage90V7Config, cache: HarpConsumedCacheIndex
) -> HarpV7PhysicalInputReceipt:
    """Authenticate the bank, GenerationLock, classifier, and query cache."""

    if type(config) is not HarpStage90V7Config or type(cache) is not HarpConsumedCacheIndex:
        raise ProtocolError("HARP v7 physical validation requires typed inputs.")
    bank_root = config.resolved_path("expert_bank_root")
    generation_root = config.resolved_path("generation_lock_root")
    for input_root, name in ((bank_root, "expert bank"), (generation_root, "GenerationLock")):
        if not input_root.is_dir() or input_root.is_symlink():
            raise ProtocolError(f"HARP v7 authoritative {name} root is unsafe.")
        state = read_json(input_root / "reports/run_state.json")
        validation = read_json(input_root / "reports/validation_report.json")
        if state.get("status") != "COMPLETE" or validation.get("status") != "PASS":
            raise ProtocolError(f"HARP v7 authoritative {name} is not complete and valid.")
    bank_path = bank_root / "manifests/expert_bank_index.json"
    generation_path = generation_root / "manifests/generation_lock.json"
    if (
        not bank_path.is_file()
        or bank_path.is_symlink()
        or not generation_path.is_file()
        or generation_path.is_symlink()
    ):
        raise ProtocolError("HARP v7 bank or GenerationLock is absent.")
    bank_sha = sha256_file(bank_path)
    generation_sha = sha256_file(generation_path)
    generation_lock = read_generation_lock(generation_path)
    lock_payload = generation_lock.to_payload()
    bank_payload = read_json(bank_path)
    bank_binding = lock_payload.get("bank")
    raw_classifier = lock_payload.get("classifier")
    if not isinstance(bank_binding, Mapping) or not isinstance(raw_classifier, Mapping):
        raise ProtocolError("HARP v7 GenerationLock lacks bank/classifier bindings.")
    if (
        generation_lock.bank_lock_hash != config.expected_hashes["expert_bank_lock_hash"]
        or generation_lock.generation_lock_hash != config.expected_hashes["generation_lock_hash"]
        or bank_payload.get("bank_lock_hash") != generation_lock.bank_lock_hash
        or bank_binding.get("bank_index_sha256") != bank_sha
    ):
        raise ProtocolError("HARP v7 authoritative generation lineage drifted.")
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
        raise ProtocolError("HARP v7 classifier contract schema drifted.")
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
        raise ProtocolError("HARP v7 classifier contract is malformed.") from exc
    if (
        classifier.config_hash != raw_classifier.get("config_hash")
        or raw_classifier.get("scaler_family") != "sklearn.preprocessing.StandardScaler"
        or raw_classifier.get("fit_in_stage_40") is not False
    ):
        raise ProtocolError("HARP v7 classifier identity drifted.")
    classifier_contract_sha = canonical_hash(
        {
            "schema_version": "midogpp_harp_classifier_semantic_identity_v7",
            "classifier": classifier.to_payload(),
            "scaler_family": raw_classifier["scaler_family"],
            "fit_in_stage_40": False,
        }
    )
    base = {
        "schema_version": "midogpp_harp_stage90_physical_input_receipt_v7",
        "bank_semantic_lock_hash": generation_lock.bank_lock_hash,
        "generation_semantic_lock_hash": generation_lock.generation_lock_hash,
        "expert_bank_index_sha256": bank_sha,
        "generation_lock_file_sha256": generation_sha,
        "classifier_config_hash": classifier.config_hash,
        "classifier_contract_sha256": classifier_contract_sha,
        "cache_hash": cache.cache_hash,
        "cache_content_sha256": cache.content_sha256,
        "labels_consumed": False,
        "predecessor_policy_used": False,
    }
    receipt = HarpV7PhysicalInputReceipt(
        bank_semantic_lock_hash=generation_lock.bank_lock_hash,
        generation_semantic_lock_hash=generation_lock.generation_lock_hash,
        expert_bank_index_sha256=bank_sha,
        generation_lock_file_sha256=generation_sha,
        classifier_config_hash=classifier.config_hash,
        classifier_contract_sha256=classifier_contract_sha,
        cache_hash=cache.cache_hash,
        cache_content_sha256=cache.content_sha256,
        receipt_hash=canonical_hash(base),
    )
    if receipt.to_payload() != {**base, "receipt_hash": receipt.receipt_hash}:
        raise ProtocolError("HARP v7 physical input receipt construction drifted.")
    return receipt


__all__ = ("HarpV7PhysicalInputReceipt", "validate_physical_inputs")
