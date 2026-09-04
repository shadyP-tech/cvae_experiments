"""Typed, center-local authority for opening HARP v15 support labels.

The train-center labels for target ``H`` remain inaccessible until both the
same-center train-support menu and the full-test target menu are durable and
sealed. The seals must describe one shared action inventory, and the fixed bank
must independently attest that every retained expert excludes ``H``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash, require_sha256
from ...runtime.artifact_io import read_json, sha256_file
from .identity import EXPERIMENT_ID


SUPPORT_CAPABILITY_STATE = (
    "SUPPORT_CENTER_SCOPED_OPEN_AFTER_SUPPORT_AND_TARGET_MENU_SEALS"
)
SUPPORT_SURFACE_ROLE = "target_train_support"
TARGET_SURFACE_ROLE = "target_test_evaluation"
_SEAL_KEYS = {
    "schema_version",
    "experiment_id",
    "outer_target_id",
    "surface_role",
    "candidate_source_ids",
    "action_identity_hash",
    "menu_hash",
    "store_receipt_hash",
    "labels_consumed",
    "seal_hash",
}
_ATTESTATION_KEYS = {
    "schema_version",
    "bank_index_sha256",
    "generation_lock_sha256",
    "source_local_lineage_hash",
    "per_target_hashes",
    "candidate_pool_semantics",
    "target_expert_unrepresentable",
    "source_frames_and_samplers_source_center_local",
    "classifier_scaler_fit",
    "support_labels_may_update",
    "support_labels_may_not_update",
    "labels_consumed",
    "attestation_hash",
}
_SUPPORT_LABEL_FORBIDDEN_UPDATES = (
    "expert_checkpoint",
    "source_frame",
    "aggregate_prior",
    "generation",
    "classifier",
    "menu_geometry",
    "shared_transform",
    "hyperparameter_grid",
)


@dataclass(frozen=True, slots=True)
class TargetSupportLabelCapability:
    """Authority to read exactly one target center's train-support shard."""

    outer_target_id: str
    support_menu_seal_path: Path
    support_menu_seal_sha256: str
    target_menu_seal_path: Path
    target_menu_seal_sha256: str
    bank_independence_attestation_path: Path
    bank_independence_attestation_sha256: str
    label_index_path: Path
    label_index_sha256: str
    candidate_source_ids: tuple[str, ...]
    action_identity_hash: str
    capability_state: str = SUPPORT_CAPABILITY_STATE
    capability_hash: str = field(init=False)

    def __post_init__(self) -> None:
        h = str(self.outer_target_id)
        expected_candidates = tuple(center for center in CENTERS if center != h)
        candidates = tuple(str(value) for value in self.candidate_source_ids)
        if h not in CENTERS or candidates != expected_candidates:
            raise ProtocolError("HARP v15 support capability candidate pool drifted.")
        action_hash = require_sha256(
            self.action_identity_hash, name="support action identity"
        )
        support_path, support_sha, support = _read_role_seal(
            self.support_menu_seal_path,
            self.support_menu_seal_sha256,
            outer_target_id=h,
            surface_role=SUPPORT_SURFACE_ROLE,
            candidate_source_ids=candidates,
        )
        target_path, target_sha, target = _read_role_seal(
            self.target_menu_seal_path,
            self.target_menu_seal_sha256,
            outer_target_id=h,
            surface_role=TARGET_SURFACE_ROLE,
            candidate_source_ids=candidates,
        )
        attestation_path, attestation_sha, attestation = (
            _read_bank_independence_attestation(
                self.bank_independence_attestation_path,
                self.bank_independence_attestation_sha256,
                outer_target_id=h,
            )
        )
        label_path, label_sha = _authenticated_file(
            self.label_index_path,
            self.label_index_sha256,
            name="support-label index",
        )
        if (
            support_path == target_path
            or support.get("action_identity_hash") != action_hash
            or target.get("action_identity_hash") != action_hash
            or support.get("menu_hash") == target.get("menu_hash")
            or self.capability_state != SUPPORT_CAPABILITY_STATE
        ):
            raise ProtocolError("HARP v15 support/target menu seals are not independent.")
        body = {
            "schema_version": "midogpp_harp_v15_target_support_label_capability_v1",
            "experiment_id": EXPERIMENT_ID,
            "outer_target_id": h,
            "capability_state": SUPPORT_CAPABILITY_STATE,
            "candidate_source_ids": list(candidates),
            "action_identity_hash": action_hash,
            "support_menu_seal_path": str(support_path),
            "support_menu_seal_sha256": support_sha,
            "support_menu_seal_hash": support["seal_hash"],
            "target_menu_seal_path": str(target_path),
            "target_menu_seal_sha256": target_sha,
            "target_menu_seal_hash": target["seal_hash"],
            "bank_independence_attestation_path": str(attestation_path),
            "bank_independence_attestation_sha256": attestation_sha,
            "bank_independence_per_target_hash": attestation[
                "per_target_hashes"
            ][h],
            "label_index_path": str(label_path),
            "label_index_sha256": label_sha,
            "evaluation_labels_authorized": False,
        }
        object.__setattr__(self, "outer_target_id", h)
        object.__setattr__(self, "candidate_source_ids", candidates)
        object.__setattr__(self, "action_identity_hash", action_hash)
        object.__setattr__(self, "support_menu_seal_path", support_path)
        object.__setattr__(self, "support_menu_seal_sha256", support_sha)
        object.__setattr__(self, "target_menu_seal_path", target_path)
        object.__setattr__(self, "target_menu_seal_sha256", target_sha)
        object.__setattr__(self, "bank_independence_attestation_path", attestation_path)
        object.__setattr__(self, "bank_independence_attestation_sha256", attestation_sha)
        object.__setattr__(self, "label_index_path", label_path)
        object.__setattr__(self, "label_index_sha256", label_sha)
        object.__setattr__(self, "capability_hash", canonical_hash(body))

    def authorize(self, allowed_center_ids: Sequence[str]) -> None:
        """Reauthenticate every durable member at the label-read boundary."""

        if tuple(str(value) for value in allowed_center_ids) != (
            self.outer_target_id,
        ):
            raise ProtocolError("HARP v15 support capability is cross-scoped.")
        _read_role_seal(
            self.support_menu_seal_path,
            self.support_menu_seal_sha256,
            outer_target_id=self.outer_target_id,
            surface_role=SUPPORT_SURFACE_ROLE,
            candidate_source_ids=self.candidate_source_ids,
        )
        _read_role_seal(
            self.target_menu_seal_path,
            self.target_menu_seal_sha256,
            outer_target_id=self.outer_target_id,
            surface_role=TARGET_SURFACE_ROLE,
            candidate_source_ids=self.candidate_source_ids,
        )
        _read_bank_independence_attestation(
            self.bank_independence_attestation_path,
            self.bank_independence_attestation_sha256,
            outer_target_id=self.outer_target_id,
        )
        _authenticated_file(
            self.label_index_path,
            self.label_index_sha256,
            name="support-label index",
        )


def issue_target_support_label_capability(
    *,
    outer_target_id: str,
    support_menu_seal_path: Path,
    support_menu_seal_sha256: str,
    target_menu_seal_path: Path,
    target_menu_seal_sha256: str,
    bank_independence_attestation_path: Path,
    bank_independence_attestation_sha256: str,
    label_index_path: Path,
    label_index_sha256: str,
) -> TargetSupportLabelCapability:
    """Issue one H-local capability from durable pre-label evidence only."""

    h = str(outer_target_id)
    candidates = tuple(center for center in CENTERS if center != h)
    _support_path, _support_sha, support = _read_role_seal(
        support_menu_seal_path,
        support_menu_seal_sha256,
        outer_target_id=h,
        surface_role=SUPPORT_SURFACE_ROLE,
        candidate_source_ids=candidates,
    )
    return TargetSupportLabelCapability(
        outer_target_id=h,
        support_menu_seal_path=Path(support_menu_seal_path),
        support_menu_seal_sha256=support_menu_seal_sha256,
        target_menu_seal_path=Path(target_menu_seal_path),
        target_menu_seal_sha256=target_menu_seal_sha256,
        bank_independence_attestation_path=Path(bank_independence_attestation_path),
        bank_independence_attestation_sha256=bank_independence_attestation_sha256,
        label_index_path=Path(label_index_path),
        label_index_sha256=label_index_sha256,
        candidate_source_ids=candidates,
        action_identity_hash=str(support["action_identity_hash"]),
    )


def _read_role_seal(
    path: Path,
    digest: str,
    *,
    outer_target_id: str,
    surface_role: str,
    candidate_source_ids: tuple[str, ...],
) -> tuple[Path, str, Mapping[str, object]]:
    resolved, expected_sha = _authenticated_file(path, digest, name="menu seal")
    payload = read_json(resolved)
    body = {key: value for key, value in payload.items() if key != "seal_hash"}
    expected_schema = (
        "midogpp_harp_v15_target_train_support_menu_seal_v1"
        if surface_role == SUPPORT_SURFACE_ROLE
        else "midogpp_harp_v15_target_test_evaluation_menu_seal_v1"
    )
    if (
        set(payload) != _SEAL_KEYS
        or payload.get("schema_version") != expected_schema
        or payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("outer_target_id") != outer_target_id
        or payload.get("surface_role") != surface_role
        or tuple(payload.get("candidate_source_ids", ())) != candidate_source_ids
        or payload.get("labels_consumed") is not False
        or any(
            require_sha256(payload.get(key), name=f"menu seal {key}")
            != payload.get(key)
            for key in ("action_identity_hash", "menu_hash", "store_receipt_hash")
        )
        or payload.get("seal_hash") != canonical_hash(body)
    ):
        raise ProtocolError("HARP v15 target/support role seal drifted.")
    return resolved, expected_sha, payload


def _authenticated_file(path: Path, digest: str, *, name: str) -> tuple[Path, str]:
    resolved = Path(path).resolve()
    expected = require_sha256(digest, name=name)
    if (
        not resolved.is_file()
        or resolved.is_symlink()
        or sha256_file(resolved) != expected
    ):
        raise ProtocolError(f"HARP v15 {name} is absent or drifted.")
    return resolved, expected


def _read_bank_independence_attestation(
    path: Path,
    digest: str,
    *,
    outer_target_id: str,
) -> tuple[Path, str, Mapping[str, object]]:
    resolved, expected_sha = _authenticated_file(
        path,
        digest,
        name="bank-independence attestation",
    )
    payload = read_json(resolved)
    body = {key: value for key, value in payload.items() if key != "attestation_hash"}
    per_target = payload.get("per_target_hashes")
    if (
        set(payload) != _ATTESTATION_KEYS
        or payload.get("schema_version")
        != "midogpp_harp_v15_fixed_bank_support_independence_v1"
        or any(
            require_sha256(payload.get(key), name=f"bank attestation {key}")
            != payload.get(key)
            for key in (
                "bank_index_sha256",
                "generation_lock_sha256",
                "source_local_lineage_hash",
            )
        )
        or not isinstance(per_target, Mapping)
        or tuple(str(key) for key in per_target) != CENTERS
        or any(
            require_sha256(value, name="per-target independence hash") != value
            for value in per_target.values()
        )
        or outer_target_id not in per_target
        or payload.get("candidate_pool_semantics") != "C_MINUS_H"
        or payload.get("target_expert_unrepresentable") is not True
        or payload.get("source_frames_and_samplers_source_center_local") is not True
        or payload.get("classifier_scaler_fit") != "synthetic_train_only"
        or payload.get("support_labels_may_update") != "H_LOCAL_ROUTER_ONLY"
        or tuple(payload.get("support_labels_may_not_update", ()))
        != _SUPPORT_LABEL_FORBIDDEN_UPDATES
        or payload.get("labels_consumed") is not False
        or payload.get("attestation_hash") != canonical_hash(body)
    ):
        raise ProtocolError("HARP v15 bank-independence attestation drifted.")
    return resolved, expected_sha, payload


__all__ = (
    "SUPPORT_CAPABILITY_STATE",
    "SUPPORT_SURFACE_ROLE",
    "TARGET_SURFACE_ROLE",
    "TargetSupportLabelCapability",
    "issue_target_support_label_capability",
)
