"""Fixed-bank independence proof for target-train support adaptation.

Train-H labels may influence only an H-local router.  Before those labels can
open, this module reconstructs the promoted bank and GenerationLock and proves
that every usable expert, frame, sampler, and classifier fit excludes H.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from ...expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    N_EXPERTS,
    TRAINING_SEEDS,
)
from ...expert_bank.uniform_b_v2_promotion.serialization import (
    sampler_from_payload,
    source_frame_from_payload,
)
from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from ...runtime.artifact_io import read_json, sha256_file


@dataclass(frozen=True, slots=True)
class FixedBankSupportIndependenceAttestation:
    """Label-free, per-H proof that support cannot overlap expert fitting."""

    bank_index_sha256: str
    generation_lock_sha256: str
    source_local_lineage_hash: str
    per_target_hashes: Mapping[str, str]
    attestation_hash: str

    def __post_init__(self) -> None:
        hashes = {str(key): str(value) for key, value in self.per_target_hashes.items()}
        if tuple(hashes) != CENTERS or any(len(value) != 64 for value in hashes.values()):
            raise ProtocolError("HARP v15 per-target independence hashes drifted.")
        object.__setattr__(self, "per_target_hashes", MappingProxyType(hashes))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_harp_v15_fixed_bank_support_independence_v1",
            "bank_index_sha256": self.bank_index_sha256,
            "generation_lock_sha256": self.generation_lock_sha256,
            "source_local_lineage_hash": self.source_local_lineage_hash,
            "per_target_hashes": dict(self.per_target_hashes),
            "candidate_pool_semantics": "C_MINUS_H",
            "target_expert_unrepresentable": True,
            "source_frames_and_samplers_source_center_local": True,
            "classifier_scaler_fit": "synthetic_train_only",
            "support_labels_may_update": "H_LOCAL_ROUTER_ONLY",
            "support_labels_may_not_update": [
                "expert_checkpoint",
                "source_frame",
                "aggregate_prior",
                "generation",
                "classifier",
                "menu_geometry",
                "shared_transform",
                "hyperparameter_grid",
            ],
            "labels_consumed": False,
            "attestation_hash": self.attestation_hash,
        }


def audit_fixed_bank_support_independence(
    *,
    bank_root: Path,
    bank_payload: Mapping[str, object],
    generation_payload: Mapping[str, object],
    bank_index_sha256: str,
    generation_lock_sha256: str,
) -> FixedBankSupportIndependenceAttestation:
    """Reconstruct all source-local records and issue exact per-H proofs."""

    bank = generation_payload.get("bank")
    source_frame_contract = generation_payload.get("source_frame")
    sampler_contract = generation_payload.get("aggregate_prior")
    classifier = generation_payload.get("classifier")
    if not all(
        isinstance(value, Mapping)
        for value in (bank, source_frame_contract, sampler_contract, classifier)
    ):
        raise ProtocolError("HARP v15 GenerationLock support proof is incomplete.")
    assert isinstance(bank, Mapping)
    assert isinstance(source_frame_contract, Mapping)
    assert isinstance(sampler_contract, Mapping)
    assert isinstance(classifier, Mapping)
    if (
        source_frame_contract.get("family") != "source_specific_pca"
        or source_frame_contract.get("fit_scope") != "source_center_rows_only"
        or source_frame_contract.get("one_frame_per_source_center") is not True
        or source_frame_contract.get("refit_allowed") is not False
        or source_frame_contract.get("frame_hashes_bound_per_expert") is not True
        or sampler_contract.get("fit_scope") != "source_center_rows_only"
        or sampler_contract.get("sampler_state_hashes_bound_per_expert") is not True
        or classifier.get("scaler_fit") != "synthetic_train_only"
        or classifier.get("fit_in_stage_40") is not False
        or bank.get("all_27_experts_retained") is not True
        or bank.get("individual_expert_or_seed_selection") is not False
    ):
        raise ProtocolError("HARP v15 shared fit could contain target support.")

    candidates_raw = bank.get("candidate_sources_by_target")
    records_raw = bank_payload.get("records")
    if not isinstance(candidates_raw, Mapping) or not isinstance(records_raw, list):
        raise ProtocolError("HARP v15 fixed-bank support inventory is absent.")
    if len(records_raw) != N_EXPERTS:
        raise ProtocolError("HARP v15 fixed-bank replica coverage drifted.")
    expected_keys = {
        (center, int(seed)) for center in CENTERS for seed in TRAINING_SEEDS
    }
    observed: set[tuple[str, int]] = set()
    rows_by_source: dict[str, list[dict[str, object]]] = {center: [] for center in CENTERS}
    resolved_root = bank_root.resolve()
    for raw in records_raw:
        if not isinstance(raw, Mapping):
            raise ProtocolError("HARP v15 fixed-bank record is malformed.")
        center = str(raw.get("source_center"))
        try:
            seed = int(raw.get("training_seed", -1))
        except (TypeError, ValueError) as exc:
            raise ProtocolError("HARP v15 fixed-bank seed is malformed.") from exc
        key = (center, seed)
        if (
            key not in expected_keys
            or key in observed
            or raw.get("fresh_source_only_training") is not True
            or raw.get("parent_checkpoint_used") is not False
            or raw.get("individual_expert_or_seed_selected") is not False
            or raw.get("routing_authorized") is not True
        ):
            raise ProtocolError("HARP v15 fixed-bank training fence drifted.")
        observed.add(key)
        frame_path = _safe_member(resolved_root, raw.get("frame_path"), "frame")
        sampler_path = _safe_member(resolved_root, raw.get("sampler_path"), "sampler")
        frame_sha = sha256_file(frame_path)
        sampler_sha = sha256_file(sampler_path)
        if (
            frame_sha != raw.get("frame_file_sha256")
            or sampler_sha != raw.get("sampler_file_sha256")
        ):
            raise ProtocolError("HARP v15 source-local preprocessing bytes drifted.")
        frame = source_frame_from_payload(read_json(frame_path))
        sampler = sampler_from_payload(read_json(sampler_path))
        if (
            frame.source_center != center
            or frame.state_hash != raw.get("frame_hash")
            or sampler.source_row_hash != frame.source_row_hash
            or sampler.state_hash != raw.get("sampler_state_hash")
        ):
            raise ProtocolError("HARP v15 expert preprocessing is not source-local.")
        rows_by_source[center].append(
            {
                "source_center": center,
                "training_seed": seed,
                "frame_sha256": frame_sha,
                "sampler_sha256": sampler_sha,
                "source_row_hash": frame.source_row_hash,
                "fit_sample_hash": frame.frame.fit_sample_hash,
                "frame_hash": frame.state_hash,
                "sampler_hash": sampler.state_hash,
            }
        )
    if observed != expected_keys or any(
        len(rows_by_source[center]) != len(TRAINING_SEEDS) for center in CENTERS
    ):
        raise ProtocolError("HARP v15 fixed-bank support proof is incomplete.")
    lineage_rows = [
        row
        for center in CENTERS
        for row in sorted(rows_by_source[center], key=lambda value: int(value["training_seed"]))
    ]
    lineage_hash = canonical_hash(
        {
            "schema_version": "midogpp_harp_v15_source_local_lineage_v1",
            "records": lineage_rows,
        }
    )
    per_target: dict[str, str] = {}
    for outer in CENTERS:
        expected_candidates = tuple(center for center in CENTERS if center != outer)
        raw_candidates = candidates_raw.get(outer)
        if (
            not isinstance(raw_candidates, list)
            or tuple(str(value) for value in raw_candidates) != expected_candidates
        ):
            raise ProtocolError("HARP v15 GenerationLock candidate pool leaked H.")
        target_rows = [
            row for source in expected_candidates for row in rows_by_source[source]
        ]
        if any(str(row["source_center"]) == outer for row in target_rows):
            raise ProtocolError("HARP v15 target expert entered support adaptation.")
        per_target[outer] = canonical_hash(
            {
                "schema_version": "midogpp_harp_v15_per_target_support_independence_v1",
                "outer_target_id": outer,
                "candidate_source_ids": list(expected_candidates),
                "candidate_record_hash": canonical_hash({"records": target_rows}),
                "source_frame_fit_scope": "source_center_rows_only",
                "sampler_fit_scope": "source_center_rows_only",
                "classifier_scaler_fit": "synthetic_train_only",
                "target_expert_excluded": True,
                "labels_consumed": False,
            }
        )
    base = {
        "schema_version": "midogpp_harp_v15_fixed_bank_support_independence_v1",
        "bank_index_sha256": bank_index_sha256,
        "generation_lock_sha256": generation_lock_sha256,
        "source_local_lineage_hash": lineage_hash,
        "per_target_hashes": per_target,
        "candidate_pool_semantics": "C_MINUS_H",
        "target_expert_unrepresentable": True,
        "source_frames_and_samplers_source_center_local": True,
        "classifier_scaler_fit": "synthetic_train_only",
        "support_labels_may_update": "H_LOCAL_ROUTER_ONLY",
        "support_labels_may_not_update": [
            "expert_checkpoint",
            "source_frame",
            "aggregate_prior",
            "generation",
            "classifier",
            "menu_geometry",
            "shared_transform",
            "hyperparameter_grid",
        ],
        "labels_consumed": False,
    }
    attestation = FixedBankSupportIndependenceAttestation(
        bank_index_sha256=bank_index_sha256,
        generation_lock_sha256=generation_lock_sha256,
        source_local_lineage_hash=lineage_hash,
        per_target_hashes=per_target,
        attestation_hash=canonical_hash(base),
    )
    if attestation.to_payload() != {**base, "attestation_hash": attestation.attestation_hash}:
        raise ProtocolError("HARP v15 support independence attestation drifted.")
    return attestation


def _safe_member(root: Path, raw: object, label: str) -> Path:
    if type(raw) is not str or not raw or Path(raw).is_absolute():
        raise ProtocolError(f"HARP v15 expert {label} path is unsafe.")
    lexical = root / raw
    if lexical.is_symlink():
        raise ProtocolError(f"HARP v15 expert {label} path is unsafe.")
    path = lexical.resolve()
    if path == root or not path.is_relative_to(root) or not path.is_file():
        raise ProtocolError(f"HARP v15 expert {label} escaped the bank.")
    return path


__all__ = (
    "FixedBankSupportIndependenceAttestation",
    "audit_fixed_bank_support_independence",
)
