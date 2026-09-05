"""Durable pre-label seals for v17 source-q and target-H menu surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...routing.pooled_pairwise_selected_policy_router_v17.hashing import (
    canonical_bytes,
    canonical_hash,
    require_sha256,
)
from ..artifact_io import atomic_json, read_json, sha256_file
from .stores import CompactStoreReceipt
from .support_independence import FixedBankSupportIndependenceAttestation
from .support_target_adapter import SupportTargetMenuBundle


EXPERIMENT_ID = "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_harp_router.v17"
SOURCE_TRAIN_SURFACE_ROLE = "source_train"
TARGET_EVALUATION_SURFACE_ROLE = "target"


@dataclass(frozen=True, slots=True)
class SupportTargetSurfaceSealSet:
    center_id: str
    candidate_source_ids: tuple[str, ...]
    action_identity_hash: str
    physical_store_receipt_hash: str
    source_train_menu_seal_path: Path
    source_train_menu_seal_sha256: str
    source_train_menu_seal_hash: str
    target_evaluation_menu_seal_path: Path
    target_evaluation_menu_seal_sha256: str
    target_evaluation_menu_seal_hash: str
    bank_independence_attestation_path: Path
    bank_independence_attestation_sha256: str
    seal_set_hash: str

    @property
    def outer_target_id(self) -> str:
        return self.center_id

    @property
    def support_menu_seal_path(self) -> Path:
        return self.source_train_menu_seal_path

    @property
    def support_menu_seal_sha256(self) -> str:
        return self.source_train_menu_seal_sha256

    @property
    def support_menu_seal_hash(self) -> str:
        return self.source_train_menu_seal_hash

    @property
    def target_menu_seal_path(self) -> Path:
        return self.target_evaluation_menu_seal_path

    @property
    def target_menu_seal_sha256(self) -> str:
        return self.target_evaluation_menu_seal_sha256

    @property
    def target_menu_seal_hash(self) -> str:
        return self.target_evaluation_menu_seal_hash

    def public_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_harp_v17_source_target_surface_seal_set_v1",
            "experiment_id": EXPERIMENT_ID,
            "center_id": self.center_id,
            "candidate_source_ids": list(self.candidate_source_ids),
            "action_identity_hash": self.action_identity_hash,
            "physical_store_receipt_hash": self.physical_store_receipt_hash,
            "source_train_menu_seal_path": str(self.source_train_menu_seal_path),
            "source_train_menu_seal_sha256": self.source_train_menu_seal_sha256,
            "source_train_menu_seal_hash": self.source_train_menu_seal_hash,
            "target_evaluation_menu_seal_path": str(
                self.target_evaluation_menu_seal_path
            ),
            "target_evaluation_menu_seal_sha256": (
                self.target_evaluation_menu_seal_sha256
            ),
            "target_evaluation_menu_seal_hash": self.target_evaluation_menu_seal_hash,
            "bank_independence_attestation_path": str(
                self.bank_independence_attestation_path
            ),
            "bank_independence_attestation_sha256": (
                self.bank_independence_attestation_sha256
            ),
            "source_q_and_target_H_rows_separate": True,
            "source_train_labels_opened": False,
            "target_evaluation_labels_opened": False,
            "seal_set_hash": self.seal_set_hash,
        }


def _persist_or_validate(path: Path, payload: dict[str, object]) -> str:
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise ProtocolError("HARP v17 durable seal path is unsafe.")
        if canonical_bytes(read_json(path)) != canonical_bytes(payload):
            raise ProtocolError("HARP v17 refuses to overwrite a drifted durable seal.")
    else:
        atomic_json(path, payload)
    if canonical_bytes(read_json(path)) != canonical_bytes(payload):
        raise ProtocolError("HARP v17 durable seal failed its readback.")
    return sha256_file(path)


def _validate_store_receipt(
    bundle: SupportTargetMenuBundle, receipt: CompactStoreReceipt
) -> str:
    if not isinstance(receipt, CompactStoreReceipt):
        raise ProtocolError("HARP v17 menu seal requires a typed store receipt.")
    manifest_path = receipt.manifest_path.resolve()
    npz_path = receipt.npz_path.resolve()
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or not npz_path.is_file()
        or npz_path.is_symlink()
        or sha256_file(manifest_path) != receipt.manifest_sha256
        or sha256_file(npz_path) != receipt.npz_sha256
    ):
        raise ProtocolError("HARP v17 physical menu store receipt drifted.")
    manifest = read_json(manifest_path)
    if (
        manifest.get("manifest_hash") != receipt.manifest_hash
        or manifest.get("menu_hash") != bundle.physical_menu.menu_hash
        or manifest.get("outer_target_id") != bundle.center_id
    ):
        raise ProtocolError("HARP v17 store receipt escaped its physical menu.")
    return canonical_hash(
        {
            "schema_version": "midogpp_harp_v17_physical_menu_store_receipt_v1",
            "center_id": bundle.center_id,
            "physical_menu_hash": bundle.physical_menu.menu_hash,
            "manifest_hash": receipt.manifest_hash,
            "manifest_sha256": receipt.manifest_sha256,
            "npz_sha256": receipt.npz_sha256,
            "chunk_hashes": dict(receipt.chunk_hashes),
        }
    )


def _validate_attestation(
    bundle: SupportTargetMenuBundle,
    attestation: FixedBankSupportIndependenceAttestation,
) -> dict[str, object]:
    if not isinstance(attestation, FixedBankSupportIndependenceAttestation):
        raise ProtocolError("HARP v17 menu seals require the typed bank attestation.")
    payload = attestation.to_payload()
    body = {key: value for key, value in payload.items() if key != "attestation_hash"}
    expected_candidates = tuple(center for center in CENTERS if center != bundle.center_id)
    if (
        bundle.candidate_source_ids != expected_candidates
        or payload.get("candidate_pool_semantics") != "C_MINUS_CONTEXT_CENTER"
        or payload.get("own_center_expert_unrepresentable") is not True
        or payload.get("labels_consumed") is not False
        or payload.get("attestation_hash") != canonical_hash(body)
        or bundle.center_id not in attestation.per_center_hashes
    ):
        raise ProtocolError("HARP v17 bank attestation does not bind C-minus-center.")
    require_sha256(
        attestation.per_center_hashes[bundle.center_id],
        name="per-center independence hash",
    )
    return payload


def _role_seal(
    bundle: SupportTargetMenuBundle,
    *,
    surface_role: str,
    menu_hash: str,
    store_receipt_hash: str,
) -> dict[str, object]:
    schema = (
        "midogpp_harp_v17_source_train_menu_seal_v1"
        if surface_role == SOURCE_TRAIN_SURFACE_ROLE
        else "midogpp_harp_v17_target_evaluation_menu_seal_v1"
        if surface_role == TARGET_EVALUATION_SURFACE_ROLE
        else None
    )
    if schema is None:
        raise ProtocolError("HARP v17 menu seal has an unknown surface role.")
    body = {
        "schema_version": schema,
        "experiment_id": EXPERIMENT_ID,
        "center_id": bundle.center_id,
        "surface_role": surface_role,
        "candidate_source_ids": list(bundle.candidate_source_ids),
        "action_identity_hash": bundle.action_identity_hash,
        "menu_hash": require_sha256(menu_hash, name="effective menu hash"),
        "store_receipt_hash": require_sha256(
            store_receipt_hash, name="physical store receipt hash"
        ),
        "labels_consumed": False,
    }
    return {**body, "seal_hash": canonical_hash(body)}


def write_source_target_surface_seals(
    root: Path,
    *,
    bundle: SupportTargetMenuBundle,
    physical_store_receipt: CompactStoreReceipt,
    fixed_bank_independence: FixedBankSupportIndependenceAttestation,
) -> SupportTargetSurfaceSealSet:
    """Write one center's source and target menu seals before source truth opens."""

    if not isinstance(bundle, SupportTargetMenuBundle):
        raise ProtocolError("HARP v17 surface sealing requires a compiled menu bundle.")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ProtocolError("HARP v17 surface-seal root is unsafe.")
    raw_store_hash = _validate_store_receipt(bundle, physical_store_receipt)
    attestation_payload = _validate_attestation(bundle, fixed_bank_independence)
    store_hash = canonical_hash(
        {
            "schema_version": "midogpp_harp_v17_prelabel_store_and_bank_binding_v1",
            "center_id": bundle.center_id,
            "candidate_source_ids": bundle.candidate_source_ids,
            "physical_store_receipt_hash": raw_store_hash,
            "bank_independence_attestation_hash": fixed_bank_independence.attestation_hash,
            "per_center_independence_hash": (
                fixed_bank_independence.per_center_hashes[bundle.center_id]
            ),
            "labels_consumed": False,
        }
    )
    attestation_path = root / "fixed_bank_independence_attestation.json"
    attestation_sha = _persist_or_validate(attestation_path, attestation_payload)
    center_root = root / f"center_{bundle.center_id}"
    center_root.mkdir(parents=True, exist_ok=True)
    source_payload = _role_seal(
        bundle,
        surface_role=SOURCE_TRAIN_SURFACE_ROLE,
        menu_hash=bundle.source_menu_hash,
        store_receipt_hash=store_hash,
    )
    target_payload = _role_seal(
        bundle,
        surface_role=TARGET_EVALUATION_SURFACE_ROLE,
        menu_hash=bundle.target_menu_hash,
        store_receipt_hash=store_hash,
    )
    source_path = center_root / "source_train_menu_seal.json"
    target_path = center_root / "target_evaluation_menu_seal.json"
    source_sha = _persist_or_validate(source_path, source_payload)
    target_sha = _persist_or_validate(target_path, target_payload)
    set_body = {
        "schema_version": "midogpp_harp_v17_source_target_surface_seal_set_v1",
        "experiment_id": EXPERIMENT_ID,
        "center_id": bundle.center_id,
        "candidate_source_ids": bundle.candidate_source_ids,
        "action_identity_hash": bundle.action_identity_hash,
        "physical_store_receipt_hash": store_hash,
        "source_train_menu_seal_hash": source_payload["seal_hash"],
        "source_train_menu_seal_sha256": source_sha,
        "target_evaluation_menu_seal_hash": target_payload["seal_hash"],
        "target_evaluation_menu_seal_sha256": target_sha,
        "bank_independence_attestation_hash": fixed_bank_independence.attestation_hash,
        "bank_independence_attestation_sha256": attestation_sha,
        "labels_consumed": False,
    }
    return SupportTargetSurfaceSealSet(
        center_id=bundle.center_id,
        candidate_source_ids=bundle.candidate_source_ids,
        action_identity_hash=bundle.action_identity_hash,
        physical_store_receipt_hash=store_hash,
        source_train_menu_seal_path=source_path.resolve(),
        source_train_menu_seal_sha256=source_sha,
        source_train_menu_seal_hash=str(source_payload["seal_hash"]),
        target_evaluation_menu_seal_path=target_path.resolve(),
        target_evaluation_menu_seal_sha256=target_sha,
        target_evaluation_menu_seal_hash=str(target_payload["seal_hash"]),
        bank_independence_attestation_path=attestation_path.resolve(),
        bank_independence_attestation_sha256=attestation_sha,
        seal_set_hash=canonical_hash(set_body),
    )


def write_support_target_surface_seals(*args: object, **kwargs: object) -> SupportTargetSurfaceSealSet:
    """Compatibility alias for the canonical source/target writer."""

    return write_source_target_surface_seals(*args, **kwargs)  # type: ignore[arg-type]


def report_support_target_surface_seals(
    seals: SupportTargetSurfaceSealSet,
) -> dict[str, object]:
    if not isinstance(seals, SupportTargetSurfaceSealSet):
        raise ProtocolError("HARP v17 surface-seal report requires typed evidence.")
    return seals.public_payload()


SourceTargetSurfaceSealSet = SupportTargetSurfaceSealSet


__all__ = (
    "SupportTargetSurfaceSealSet",
    "SourceTargetSurfaceSealSet",
    "report_support_target_surface_seals",
    "write_source_target_surface_seals",
    "write_support_target_surface_seals",
)
