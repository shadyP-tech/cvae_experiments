"""Exact, label-free ``(H, q, r, case)`` menu bindings for HARP v13.

HARP v12 reconstructed fold persistence inputs from an ``H/r/r`` menu
universe.  That reconstruction is scientifically invalid because ``q`` is a
physical part of the action pool and therefore of every action and menu hash.
This module makes the held-out ``q`` an explicit, durable part of the binding
used by source-fold fitting and persistence.

The certificate is created from the already durable label-free physical
surface.  It contains no outcomes and issues no label capability.  Its sole
purpose is to prove, before any source label can be opened, that every one of
the 72 ``(H, q)`` folds has a unique and exact menu inventory and a
deterministic future capability identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from ...routing.policy_calibrated_residual_router_v13 import EffectiveMenu
from ...runtime.artifact_io import atomic_json, read_json, sha256_file
from ...runtime.harp_v13_execution.crossfit_durability import (
    SourceCrossfitSurfaceReceipt,
)
from ...runtime.harp_v13_execution.crossfit_effective_menus import (
    FoldConditionedEffectiveMenu,
    FoldConditionedEffectiveSurface,
)
from ...runtime.harp_v13_execution.durability import durable_barrier
from ...runtime.harp_v13_execution.hash_contracts import require_sha256


CERTIFICATE_SCHEMA = "midogpp_harp_v13_source_fold_menu_binding_certificate_v1"
CERTIFICATE_RELATIVE_PATH = Path("manifests/source_fold_menu_binding_certificate.json")


@dataclass(frozen=True, slots=True)
class FoldLocalMenuBinding:
    """The exact effective-menu universe for one outer ``(H, q)`` fit."""

    outer_target_id: str
    heldout_center_id: str
    source_surface_receipt_hash: str
    source_surface_hash: str
    effective_adapter_hash: str
    label_index_path: Path
    label_index_sha256: str
    wrappers: tuple[FoldConditionedEffectiveMenu, ...]
    prediction_surface_hash: str = field(init=False)
    fitting_surface_hash: str = field(init=False)
    capability_hash: str = field(init=False)
    binding_hash: str = field(init=False)

    def __post_init__(self) -> None:
        h = str(self.outer_target_id)
        q = str(self.heldout_center_id)
        if h not in CENTERS or q not in CENTERS or h == q:
            raise ProtocolError("HARP v13 fold-menu binding has an invalid H/q pair.")
        receipt_hash = require_sha256(
            self.source_surface_receipt_hash, name="fold-menu source receipt"
        )
        source_hash = require_sha256(
            self.source_surface_hash, name="fold-menu source surface"
        )
        adapter_hash = require_sha256(
            self.effective_adapter_hash, name="fold-menu effective adapter"
        )
        label_path = Path(self.label_index_path).resolve()
        label_hash = require_sha256(
            self.label_index_sha256, name="fold-menu label index"
        )
        if (
            not label_path.is_file()
            or label_path.is_symlink()
            or sha256_file(label_path) != label_hash
        ):
            raise ProtocolError("HARP v13 fold-menu label-index binding drifted.")

        wrappers = _deduplicate_exact_wrappers(self.wrappers, h=h, q=q)
        expected_queries = tuple(center for center in CENTERS if center != h)
        observed_queries = tuple(
            sorted({row.current_query_center_id for row in wrappers})
        )
        prediction = tuple(
            row for row in wrappers if row.current_query_center_id == q
        )
        fitting = tuple(
            row for row in wrappers if row.current_query_center_id != q
        )
        if (
            not prediction
            or not fitting
            or observed_queries != tuple(sorted(expected_queries))
            or any(row.prediction_fold is not (row.current_query_center_id == q) for row in wrappers)
            or any(row.heldout_center_id != q for row in wrappers)
        ):
            raise ProtocolError("HARP v13 exact fold-menu coverage is incomplete.")

        prediction_hashes = tuple(row.fold_menu_hash for row in prediction)
        fitting_hashes = tuple(row.fold_menu_hash for row in fitting)
        allowed = tuple(center for center in CENTERS if center not in {h, q})
        prediction_surface_hash = canonical_hash(
            {
                "schema_version": "midogpp_harp_v13_q_prediction_surface_seal_v1",
                "outer_target_id": h,
                "heldout_center_id": q,
                "source_surface_receipt_hash": receipt_hash,
                "source_surface_hash": source_hash,
                "effective_adapter_hash": adapter_hash,
                "prediction_menu_hashes": list(prediction_hashes),
                "heldout_q_labels_consumed": False,
            }
        )
        fitting_surface_hash = canonical_hash(
            {
                "schema_version": "midogpp_harp_v13_q_fitting_surface_seal_v1",
                "outer_target_id": h,
                "heldout_center_id": q,
                "source_surface_receipt_hash": receipt_hash,
                "source_surface_hash": source_hash,
                "effective_adapter_hash": adapter_hash,
                "allowed_center_ids": list(allowed),
                "fitting_menu_hashes": list(fitting_hashes),
                "heldout_q_labels_consumed": False,
            }
        )
        capability_hash = canonical_hash(
            {
                "schema_version": "midogpp_harp_v13_fold_source_label_capability_v1",
                "outer_target_id": h,
                "heldout_center_id": q,
                "allowed_center_ids": list(allowed),
                "excluded_center_ids": [h, q],
                "source_surface_receipt_hash": receipt_hash,
                "source_surface_hash": source_hash,
                "effective_adapter_hash": adapter_hash,
                "prediction_surface_hash": prediction_surface_hash,
                "fitting_surface_hash": fitting_surface_hash,
                "label_index_path": str(label_path),
                "label_index_sha256": label_hash,
                "evaluation_labels_authorized": False,
            }
        )
        body = {
            "schema_version": "midogpp_harp_v13_exact_fold_menu_binding_v1",
            "outer_target_id": h,
            "heldout_center_id": q,
            "source_surface_receipt_hash": receipt_hash,
            "source_surface_hash": source_hash,
            "effective_adapter_hash": adapter_hash,
            "allowed_center_ids": list(allowed),
            "menu_identities": [_wrapper_identity(row) for row in wrappers],
            "prediction_fold_menu_hashes": list(prediction_hashes),
            "fitting_fold_menu_hashes": list(fitting_hashes),
            "prediction_surface_hash": prediction_surface_hash,
            "fitting_surface_hash": fitting_surface_hash,
            "expected_capability_hash": capability_hash,
            "posthoc_projection_used": False,
            "labels_consumed": False,
        }
        object.__setattr__(self, "outer_target_id", h)
        object.__setattr__(self, "heldout_center_id", q)
        object.__setattr__(self, "source_surface_receipt_hash", receipt_hash)
        object.__setattr__(self, "source_surface_hash", source_hash)
        object.__setattr__(self, "effective_adapter_hash", adapter_hash)
        object.__setattr__(self, "label_index_path", label_path)
        object.__setattr__(self, "label_index_sha256", label_hash)
        object.__setattr__(self, "wrappers", wrappers)
        object.__setattr__(self, "prediction_surface_hash", prediction_surface_hash)
        object.__setattr__(self, "fitting_surface_hash", fitting_surface_hash)
        object.__setattr__(self, "capability_hash", capability_hash)
        object.__setattr__(self, "binding_hash", canonical_hash(body))

    @property
    def effective_menus(self) -> tuple[EffectiveMenu, ...]:
        """Return exact H/q/r menus, never reconstructed H/r/r projections."""

        return tuple(row.menu for row in self.wrappers)

    def public_payload(self) -> dict[str, object]:
        return {
            "outer_target_id": self.outer_target_id,
            "heldout_center_id": self.heldout_center_id,
            "binding_hash": self.binding_hash,
            "prediction_surface_hash": self.prediction_surface_hash,
            "fitting_surface_hash": self.fitting_surface_hash,
            "expected_capability_hash": self.capability_hash,
            "menu_count": len(self.wrappers),
            "prediction_menu_count": sum(
                row.current_query_center_id == self.heldout_center_id
                for row in self.wrappers
            ),
            "fitting_menu_count": sum(
                row.current_query_center_id != self.heldout_center_id
                for row in self.wrappers
            ),
            "menu_identities": [_wrapper_identity(row) for row in self.wrappers],
            "posthoc_projection_used": False,
            "labels_consumed": False,
        }


@dataclass(frozen=True, slots=True)
class FoldMenuBindingCertificate:
    """In-memory all-fold certificate produced without opening outcomes."""

    source_surface_receipt_hash: str
    source_surface_hash: str
    effective_adapter_hash: str
    label_index_path: Path
    label_index_sha256: str
    admission_hash: str
    authorization_lease_hash: str
    folds: tuple[FoldLocalMenuBinding, ...]
    certificate_hash: str = field(init=False)

    def __post_init__(self) -> None:
        receipt_hash = require_sha256(
            self.source_surface_receipt_hash, name="binding-certificate receipt"
        )
        source_hash = require_sha256(
            self.source_surface_hash, name="binding-certificate source surface"
        )
        adapter_hash = require_sha256(
            self.effective_adapter_hash, name="binding-certificate adapter"
        )
        label_path = Path(self.label_index_path).resolve()
        label_hash = require_sha256(
            self.label_index_sha256, name="binding-certificate label index"
        )
        admission_hash = require_sha256(
            self.admission_hash, name="binding-certificate admission"
        )
        lease_hash = require_sha256(
            self.authorization_lease_hash, name="binding-certificate lease"
        )
        folds = tuple(
            sorted(
                self.folds,
                key=lambda row: (row.outer_target_id, row.heldout_center_id),
            )
        )
        expected_pairs = tuple(
            (h, q) for h in CENTERS for q in CENTERS if h != q
        )
        observed_pairs = tuple(
            (row.outer_target_id, row.heldout_center_id) for row in folds
        )
        all_identities = tuple(
            (
                row.outer_target_id,
                row.heldout_center_id,
                wrapper.current_query_center_id,
                wrapper.menu.case_id,
            )
            for row in folds
            for wrapper in row.wrappers
        )
        if (
            observed_pairs != expected_pairs
            or folds != self.folds
            or len(set(all_identities)) != len(all_identities)
            or any(
                row.source_surface_receipt_hash != receipt_hash
                or row.source_surface_hash != source_hash
                or row.effective_adapter_hash != adapter_hash
                or row.label_index_path != label_path
                or row.label_index_sha256 != label_hash
                for row in folds
            )
        ):
            raise ProtocolError("HARP v13 all-fold menu binding coverage drifted.")
        body = {
            "schema_version": CERTIFICATE_SCHEMA,
            "source_surface_receipt_hash": receipt_hash,
            "source_surface_hash": source_hash,
            "effective_adapter_hash": adapter_hash,
            "label_index_path": str(label_path),
            "label_index_sha256": label_hash,
            "admission_hash": admission_hash,
            "authorization_lease_hash": lease_hash,
            "expected_center_ids": list(CENTERS),
            "folds": [row.public_payload() for row in folds],
            "fold_count": len(folds),
            "menu_identity_count": len(all_identities),
            "all_H_q_r_case_identities_unique": True,
            "all_future_capability_hashes_precomputed": True,
            "posthoc_projection_used": False,
            "source_labels_opened": False,
            "evaluation_labels_opened": False,
        }
        object.__setattr__(self, "source_surface_receipt_hash", receipt_hash)
        object.__setattr__(self, "source_surface_hash", source_hash)
        object.__setattr__(self, "effective_adapter_hash", adapter_hash)
        object.__setattr__(self, "label_index_path", label_path)
        object.__setattr__(self, "label_index_sha256", label_hash)
        object.__setattr__(self, "admission_hash", admission_hash)
        object.__setattr__(self, "authorization_lease_hash", lease_hash)
        object.__setattr__(self, "folds", folds)
        object.__setattr__(self, "certificate_hash", canonical_hash(body))

    def for_fold(self, outer_target_id: str, heldout_center_id: str) -> FoldLocalMenuBinding:
        key = (str(outer_target_id), str(heldout_center_id))
        rows = tuple(
            row
            for row in self.folds
            if (row.outer_target_id, row.heldout_center_id) == key
        )
        if len(rows) != 1:
            raise ProtocolError(f"HARP v13 exact fold-menu binding is absent: {key}.")
        return rows[0]

    def public_payload(self) -> dict[str, object]:
        body = {
            "schema_version": CERTIFICATE_SCHEMA,
            "source_surface_receipt_hash": self.source_surface_receipt_hash,
            "source_surface_hash": self.source_surface_hash,
            "effective_adapter_hash": self.effective_adapter_hash,
            "label_index_path": str(self.label_index_path),
            "label_index_sha256": self.label_index_sha256,
            "admission_hash": self.admission_hash,
            "authorization_lease_hash": self.authorization_lease_hash,
            "expected_center_ids": list(CENTERS),
            "folds": [row.public_payload() for row in self.folds],
            "fold_count": len(self.folds),
            "menu_identity_count": sum(len(row.wrappers) for row in self.folds),
            "all_H_q_r_case_identities_unique": True,
            "all_future_capability_hashes_precomputed": True,
            "posthoc_projection_used": False,
            "source_labels_opened": False,
            "evaluation_labels_opened": False,
        }
        return {**body, "certificate_hash": self.certificate_hash}


@dataclass(frozen=True, slots=True)
class DurableFoldMenuBindingCertificate:
    """Fresh disk receipt for an exact all-fold binding certificate."""

    path: Path
    manifest_sha256: str
    certificate: FoldMenuBindingCertificate
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        path = Path(self.path).resolve()
        digest = require_sha256(
            self.manifest_sha256, name="fold-menu certificate manifest"
        )
        if (
            not path.is_file()
            or path.is_symlink()
            or path.parent.name != "manifests"
            or sha256_file(path) != digest
            or read_json(path) != self.certificate.public_payload()
        ):
            raise ProtocolError("HARP v13 fold-menu certificate round trip drifted.")
        object.__setattr__(self, "path", path)
        object.__setattr__(
            self,
            "receipt_hash",
            canonical_hash(
                {
                    "schema_version": "midogpp_harp_v13_fold_menu_binding_receipt_v1",
                    "certificate_hash": self.certificate.certificate_hash,
                    "manifest_sha256": digest,
                    "durable_before_source_label_capabilities": True,
                    "source_labels_opened": False,
                }
            ),
        )

    def for_fold(self, outer_target_id: str, heldout_center_id: str) -> FoldLocalMenuBinding:
        return self.certificate.for_fold(outer_target_id, heldout_center_id)


def build_fold_menu_binding_certificate(
    *,
    effective_surface: FoldConditionedEffectiveSurface,
    surface_receipt: SourceCrossfitSurfaceReceipt,
    label_index_path: Path,
    label_index_sha256: str,
    admission_hash: str,
    authorization_lease_hash: str,
) -> FoldMenuBindingCertificate:
    """Build all 72 exact fold bindings without issuing label authority."""

    if (
        not isinstance(effective_surface, FoldConditionedEffectiveSurface)
        or not isinstance(surface_receipt, SourceCrossfitSurfaceReceipt)
        or effective_surface.source_surface_hash != surface_receipt.surface_hash
    ):
        raise ProtocolError("HARP v13 fold-menu certificate surfaces are unbound.")
    folds: list[FoldLocalMenuBinding] = []
    for h in CENTERS:
        for q in CENTERS:
            if h == q:
                continue
            wrappers = _deduplicate_exact_wrappers(
                (
                    *effective_surface.fitting_menus(h, q),
                    *effective_surface.prediction_menus(h, q),
                ),
                h=h,
                q=q,
            )
            folds.append(
                FoldLocalMenuBinding(
                    outer_target_id=h,
                    heldout_center_id=q,
                    source_surface_receipt_hash=surface_receipt.receipt_hash,
                    source_surface_hash=surface_receipt.surface_hash,
                    effective_adapter_hash=effective_surface.adapter_hash,
                    label_index_path=Path(label_index_path),
                    label_index_sha256=label_index_sha256,
                    wrappers=wrappers,
                )
            )
    return FoldMenuBindingCertificate(
        source_surface_receipt_hash=surface_receipt.receipt_hash,
        source_surface_hash=surface_receipt.surface_hash,
        effective_adapter_hash=effective_surface.adapter_hash,
        label_index_path=Path(label_index_path),
        label_index_sha256=label_index_sha256,
        admission_hash=admission_hash,
        authorization_lease_hash=authorization_lease_hash,
        folds=tuple(folds),
    )


def persist_fold_menu_binding_certificate(
    path: Path,
    certificate: FoldMenuBindingCertificate,
) -> DurableFoldMenuBindingCertificate:
    """Durably write and freshly reconstruct the label-free certificate."""

    if not isinstance(certificate, FoldMenuBindingCertificate):
        raise ProtocolError("HARP v13 fold-menu certificate is untyped.")
    path = Path(path).resolve()
    if path.parent.name != "manifests" or path.name != CERTIFICATE_RELATIVE_PATH.name:
        raise ProtocolError("HARP v13 fold-menu certificate path drifted.")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = certificate.public_payload()
    if path.exists():
        if path.is_symlink() or read_json(path) != payload:
            raise ProtocolError(
                "Existing HARP v13 fold-menu certificate differs; refusing repair."
            )
    else:
        atomic_json(path, payload)
    durable_barrier((path,))
    return DurableFoldMenuBindingCertificate(
        path=path,
        manifest_sha256=sha256_file(path),
        certificate=certificate,
    )


def require_capability_matches_binding(
    capability: object,
    binding: FoldLocalMenuBinding,
) -> None:
    """Reject a capability that is not the exact one precomputed label-free."""

    if (
        not isinstance(binding, FoldLocalMenuBinding)
        or getattr(capability, "outer_target_id", None) != binding.outer_target_id
        or getattr(capability, "heldout_center_id", None) != binding.heldout_center_id
        or getattr(capability, "prediction_surface_hash", None)
        != binding.prediction_surface_hash
        or getattr(capability, "fitting_surface_hash", None)
        != binding.fitting_surface_hash
        or getattr(capability, "capability_hash", None) != binding.capability_hash
    ):
        raise ProtocolError("HARP v13 fold capability escaped its certified menu binding.")


def validate_persisted_fold_menu_binding_payload(
    path: Path,
    *,
    expected_admission_hash: str,
    expected_authorization_lease_hash: str,
) -> Mapping[str, object]:
    """Validate a label-free certificate during same-lease recovery admission."""

    path = Path(path).resolve()
    if not path.is_file() or path.is_symlink() or path.parent.name != "manifests":
        raise ProtocolError("HARP v13 recovered fold-menu certificate is unsafe.")
    payload = read_json(path)
    body = {key: value for key, value in payload.items() if key != "certificate_hash"}
    folds = payload.get("folds")
    if (
        payload.get("schema_version") != CERTIFICATE_SCHEMA
        or payload.get("certificate_hash") != canonical_hash(body)
        or payload.get("admission_hash")
        != require_sha256(expected_admission_hash, name="recovery admission")
        or payload.get("authorization_lease_hash")
        != require_sha256(
            expected_authorization_lease_hash, name="recovery authorization lease"
        )
        or payload.get("source_labels_opened") is not False
        or payload.get("evaluation_labels_opened") is not False
        or payload.get("posthoc_projection_used") is not False
        or payload.get("all_H_q_r_case_identities_unique") is not True
        or payload.get("all_future_capability_hashes_precomputed") is not True
        or payload.get("fold_count") != len(CENTERS) * (len(CENTERS) - 1)
        or not isinstance(folds, list)
        or len(folds) != payload.get("fold_count")
    ):
        raise ProtocolError("HARP v13 recovered fold-menu certificate drifted.")
    pairs: list[tuple[str, str]] = []
    identities: list[tuple[str, str, str, str]] = []
    for raw in folds:
        if not isinstance(raw, Mapping) or raw.get("labels_consumed") is not False:
            raise ProtocolError("HARP v13 recovered fold-menu row is malformed.")
        h = str(raw.get("outer_target_id"))
        q = str(raw.get("heldout_center_id"))
        pairs.append((h, q))
        menus = raw.get("menu_identities")
        if not isinstance(menus, list):
            raise ProtocolError("HARP v13 recovered fold-menu inventory is malformed.")
        if raw.get("menu_count") != len(menus):
            raise ProtocolError("HARP v13 recovered fold-menu count drifted.")
        for menu in menus:
            if not isinstance(menu, Mapping):
                raise ProtocolError("HARP v13 recovered menu identity is malformed.")
            identities.append(
                (
                    str(menu.get("outer_target_id")),
                    str(menu.get("heldout_center_id")),
                    str(menu.get("current_query_center_id")),
                    str(menu.get("case_id")),
                )
            )
            if identities[-1][:2] != (h, q):
                raise ProtocolError("HARP v13 recovered menu crossed H/q.")
    expected_pairs = [(h, q) for h in CENTERS for q in CENTERS if h != q]
    if pairs != expected_pairs or len(set(identities)) != len(identities):
        raise ProtocolError("HARP v13 recovered fold-menu coverage drifted.")
    return payload


def _deduplicate_exact_wrappers(
    wrappers: Sequence[FoldConditionedEffectiveMenu],
    *,
    h: str,
    q: str,
) -> tuple[FoldConditionedEffectiveMenu, ...]:
    by_identity: dict[
        tuple[str, str, str, str], FoldConditionedEffectiveMenu
    ] = {}
    for wrapper in wrappers:
        if (
            not isinstance(wrapper, FoldConditionedEffectiveMenu)
            or wrapper.outer_target_id != h
            or wrapper.heldout_center_id != q
        ):
            raise ProtocolError("HARP v13 fold-menu wrapper crossed H/q.")
        key = (
            wrapper.outer_target_id,
            wrapper.heldout_center_id,
            wrapper.current_query_center_id,
            wrapper.menu.case_id,
        )
        previous = by_identity.get(key)
        if previous is not None and previous.fold_menu_hash != wrapper.fold_menu_hash:
            raise ProtocolError("HARP v13 fold-menu identity has conflicting bytes.")
        by_identity[key] = wrapper
    ordered = tuple(by_identity[key] for key in sorted(by_identity))
    if not ordered:
        raise ProtocolError("HARP v13 fold-menu binding is empty.")
    return ordered


def _wrapper_identity(wrapper: FoldConditionedEffectiveMenu) -> dict[str, object]:
    menu = wrapper.menu
    return {
        "outer_target_id": wrapper.outer_target_id,
        "heldout_center_id": wrapper.heldout_center_id,
        "current_query_center_id": wrapper.current_query_center_id,
        "case_id": menu.case_id,
        "candidate_source_ids": list(wrapper.candidate_source_ids),
        "fold_menu_hash": wrapper.fold_menu_hash,
        "effective_menu_hash": menu.menu_hash,
        "action_ids": [row.action_id for row in menu.actions],
        "action_hashes": [row.action_hash for row in menu.actions],
        "prediction_fold": wrapper.prediction_fold,
    }


__all__ = (
    "CERTIFICATE_RELATIVE_PATH",
    "CERTIFICATE_SCHEMA",
    "DurableFoldMenuBindingCertificate",
    "FoldLocalMenuBinding",
    "FoldMenuBindingCertificate",
    "build_fold_menu_binding_certificate",
    "persist_fold_menu_binding_certificate",
    "require_capability_matches_binding",
    "validate_persisted_fold_menu_binding_payload",
)
