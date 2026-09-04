"""Durable pre-label seals for HARP v15 support and target menu surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...diagnostics.fixed_bank_harp_router_v15.identity import EXPERIMENT_ID
from ...diagnostics.fixed_bank_harp_router_v15.source_label_capability import (
    SUPPORT_SURFACE_ROLE,
    TARGET_SURFACE_ROLE,
)
from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...routing.hierarchical_support_action_risk_router_v15.hashing import (
    canonical_bytes,
    canonical_hash,
    require_sha256,
)
from ..artifact_io import atomic_json, read_json, sha256_file
from .stores import CompactStoreReceipt
from .support_independence import FixedBankSupportIndependenceAttestation
from .support_target_adapter import SupportTargetMenuBundle


@dataclass(frozen=True, slots=True)
class SupportTargetSurfaceSealSet:
    outer_target_id: str
    candidate_source_ids: tuple[str, ...]
    action_identity_hash: str
    physical_store_receipt_hash: str
    support_menu_seal_path: Path
    support_menu_seal_sha256: str
    support_menu_seal_hash: str
    target_menu_seal_path: Path
    target_menu_seal_sha256: str
    target_menu_seal_hash: str
    bank_independence_attestation_path: Path
    bank_independence_attestation_sha256: str
    seal_set_hash: str

    def public_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_harp_v15_support_target_surface_seal_set_v1",
            "experiment_id": EXPERIMENT_ID,
            "outer_target_id": self.outer_target_id,
            "candidate_source_ids": list(self.candidate_source_ids),
            "action_identity_hash": self.action_identity_hash,
            "physical_store_receipt_hash": self.physical_store_receipt_hash,
            "prelabel_store_and_bank_binding_hash": (
                self.physical_store_receipt_hash
            ),
            "support_menu_seal_path": str(self.support_menu_seal_path),
            "support_menu_seal_sha256": self.support_menu_seal_sha256,
            "support_menu_seal_hash": self.support_menu_seal_hash,
            "target_menu_seal_path": str(self.target_menu_seal_path),
            "target_menu_seal_sha256": self.target_menu_seal_sha256,
            "target_menu_seal_hash": self.target_menu_seal_hash,
            "bank_independence_attestation_path": str(
                self.bank_independence_attestation_path
            ),
            "bank_independence_attestation_sha256": (
                self.bank_independence_attestation_sha256
            ),
            "support_and_target_actions_shared": True,
            "support_and_target_physical_rows_separate": True,
            "support_labels_opened": False,
            "evaluation_labels_opened": False,
            "seal_set_hash": self.seal_set_hash,
        }


def _persist_or_validate(path: Path, payload: dict[str, object]) -> str:
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise ProtocolError("HARP v15 durable seal path is unsafe.")
        if canonical_bytes(read_json(path)) != canonical_bytes(payload):
            raise ProtocolError("HARP v15 refuses to overwrite a drifted durable seal.")
    else:
        atomic_json(path, payload)
    if canonical_bytes(read_json(path)) != canonical_bytes(payload):
        raise ProtocolError("HARP v15 durable seal failed its readback.")
    return sha256_file(path)


def _validate_store_receipt(
    bundle: SupportTargetMenuBundle,
    receipt: CompactStoreReceipt,
) -> str:
    if not isinstance(receipt, CompactStoreReceipt):
        raise ProtocolError("HARP v15 menu seal requires a typed store receipt.")
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
        raise ProtocolError("HARP v15 physical menu store receipt drifted.")
    manifest = read_json(manifest_path)
    if (
        manifest.get("manifest_hash") != receipt.manifest_hash
        or manifest.get("menu_hash") != bundle.physical_menu.menu_hash
        or manifest.get("outer_target_id") != bundle.outer_target_id
    ):
        raise ProtocolError("HARP v15 store receipt escaped its physical menu.")
    return canonical_hash(
        {
            "schema_version": "midogpp_harp_v15_physical_menu_store_receipt_v1",
            "outer_target_id": bundle.outer_target_id,
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
        raise ProtocolError("HARP v15 menu seals require the typed bank attestation.")
    payload = attestation.to_payload()
    body = {key: value for key, value in payload.items() if key != "attestation_hash"}
    expected_candidates = tuple(
        center for center in CENTERS if center != bundle.outer_target_id
    )
    if (
        bundle.candidate_source_ids != expected_candidates
        or payload.get("candidate_pool_semantics") != "C_MINUS_H"
        or payload.get("target_expert_unrepresentable") is not True
        or payload.get("labels_consumed") is not False
        or payload.get("attestation_hash") != canonical_hash(body)
        or bundle.outer_target_id not in attestation.per_target_hashes
    ):
        raise ProtocolError("HARP v15 bank attestation does not bind C-minus-H.")
    require_sha256(
        attestation.per_target_hashes[bundle.outer_target_id],
        name="per-target independence hash",
    )
    return payload


def _role_seal(
    bundle: SupportTargetMenuBundle,
    *,
    surface_role: str,
    menu_hash: str,
    store_receipt_hash: str,
) -> dict[str, object]:
    if surface_role == SUPPORT_SURFACE_ROLE:
        schema = "midogpp_harp_v15_target_train_support_menu_seal_v1"
    elif surface_role == TARGET_SURFACE_ROLE:
        schema = "midogpp_harp_v15_target_test_evaluation_menu_seal_v1"
    else:  # pragma: no cover - private caller uses the closed role inventory.
        raise ProtocolError("HARP v15 menu seal has an unknown surface role.")
    body = {
        "schema_version": schema,
        "experiment_id": EXPERIMENT_ID,
        "outer_target_id": bundle.outer_target_id,
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


def write_support_target_surface_seals(
    root: Path,
    *,
    bundle: SupportTargetMenuBundle,
    physical_store_receipt: CompactStoreReceipt,
    fixed_bank_independence: FixedBankSupportIndependenceAttestation,
) -> SupportTargetSurfaceSealSet:
    """Write the exact durable evidence required to open Train-H labels."""

    if not isinstance(bundle, SupportTargetMenuBundle):
        raise ProtocolError("HARP v15 surface sealing requires a compiled menu bundle.")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ProtocolError("HARP v15 surface-seal root is unsafe.")
    raw_store_hash = _validate_store_receipt(bundle, physical_store_receipt)
    attestation_payload = _validate_attestation(bundle, fixed_bank_independence)
    # The exact role-seal schema calls this field ``store_receipt_hash``.  Bind
    # the independently authenticated bank fence into that receipt so the
    # durable label capability cannot pair valid menu bytes with an unrelated
    # C-minus-H attestation.
    store_hash = canonical_hash(
        {
            "schema_version": "midogpp_harp_v15_prelabel_store_and_bank_binding_v1",
            "outer_target_id": bundle.outer_target_id,
            "candidate_source_ids": bundle.candidate_source_ids,
            "physical_store_receipt_hash": raw_store_hash,
            "bank_independence_attestation_hash": (
                fixed_bank_independence.attestation_hash
            ),
            "per_target_independence_hash": (
                fixed_bank_independence.per_target_hashes[bundle.outer_target_id]
            ),
            "labels_consumed": False,
        }
    )
    attestation_path = root / "fixed_bank_support_independence_attestation.json"
    attestation_sha = _persist_or_validate(attestation_path, attestation_payload)

    outer_root = root / f"H{bundle.outer_target_id}"
    outer_root.mkdir(parents=True, exist_ok=True)
    support_payload = _role_seal(
        bundle,
        surface_role=SUPPORT_SURFACE_ROLE,
        menu_hash=bundle.support_menu_hash,
        store_receipt_hash=store_hash,
    )
    target_payload = _role_seal(
        bundle,
        surface_role=TARGET_SURFACE_ROLE,
        menu_hash=bundle.target_menu_hash,
        store_receipt_hash=store_hash,
    )
    support_path = outer_root / "target_train_support_menu_seal.json"
    target_path = outer_root / "target_test_evaluation_menu_seal.json"
    support_sha = _persist_or_validate(support_path, support_payload)
    target_sha = _persist_or_validate(target_path, target_payload)
    set_body = {
        "schema_version": "midogpp_harp_v15_support_target_surface_seal_set_v1",
        "experiment_id": EXPERIMENT_ID,
        "outer_target_id": bundle.outer_target_id,
        "candidate_source_ids": bundle.candidate_source_ids,
        "action_identity_hash": bundle.action_identity_hash,
        "physical_store_receipt_hash": store_hash,
        "support_menu_seal_hash": support_payload["seal_hash"],
        "support_menu_seal_sha256": support_sha,
        "target_menu_seal_hash": target_payload["seal_hash"],
        "target_menu_seal_sha256": target_sha,
        "bank_independence_attestation_hash": fixed_bank_independence.attestation_hash,
        "bank_independence_attestation_sha256": attestation_sha,
        "labels_consumed": False,
    }
    return SupportTargetSurfaceSealSet(
        outer_target_id=bundle.outer_target_id,
        candidate_source_ids=bundle.candidate_source_ids,
        action_identity_hash=bundle.action_identity_hash,
        physical_store_receipt_hash=store_hash,
        support_menu_seal_path=support_path.resolve(),
        support_menu_seal_sha256=support_sha,
        support_menu_seal_hash=str(support_payload["seal_hash"]),
        target_menu_seal_path=target_path.resolve(),
        target_menu_seal_sha256=target_sha,
        target_menu_seal_hash=str(target_payload["seal_hash"]),
        bank_independence_attestation_path=attestation_path.resolve(),
        bank_independence_attestation_sha256=attestation_sha,
        seal_set_hash=canonical_hash(set_body),
    )


def report_support_target_surface_seals(
    seals: SupportTargetSurfaceSealSet,
) -> dict[str, object]:
    """Return a JSON-safe pre-label report without opening either label role."""

    if not isinstance(seals, SupportTargetSurfaceSealSet):
        raise ProtocolError("HARP v15 surface-seal report requires typed evidence.")
    return seals.public_payload()


__all__ = (
    "SupportTargetSurfaceSealSet",
    "report_support_target_surface_seals",
    "write_support_target_surface_seals",
)
