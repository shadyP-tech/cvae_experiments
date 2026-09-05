"""Fresh source-train label capabilities for the pooled HARP v17 router.

All nine source-q and all nine target-H menu seals, plus the fixed-bank
attestations, are reauthenticated as one closed inventory before this module
returns any q-scoped capability. A capability can open only its own source
center shard and never authorizes target-test truth.
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
from .source_train_label_access_fence import SourceTrainLabelAccessFence


SOURCE_TRAIN_CAPABILITY_STATE = (
    "SOURCE_TRAIN_CENTER_SCOPED_OPEN_AFTER_ALL_SOURCE_AND_TARGET_MENU_SEALS_AND_BANK_ATTESTATIONS"
)
SOURCE_TRAIN_SURFACE_ROLE = "source_train"
TARGET_EVALUATION_SURFACE_ROLE = "target"
_SEAL_KEYS = {
    "schema_version",
    "experiment_id",
    "center_id",
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
    "per_center_hashes",
    "candidate_pool_semantics",
    "own_center_expert_unrepresentable",
    "source_frames_and_samplers_source_center_local",
    "classifier_scaler_fit",
    "source_train_labels_may_update",
    "source_train_labels_may_not_update",
    "labels_consumed",
    "attestation_hash",
}
_SOURCE_LABEL_FORBIDDEN_UPDATES = (
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
class SourceTrainLabelCapability:
    """Authority to read exactly one known source-center q label shard."""

    center_id: str
    source_train_menu_seal_path: Path
    source_train_menu_seal_sha256: str
    target_evaluation_menu_seal_path: Path
    target_evaluation_menu_seal_sha256: str
    bank_independence_attestation_path: Path
    bank_independence_attestation_sha256: str
    label_index_path: Path
    label_index_sha256: str
    source_train_label_access_fence: SourceTrainLabelAccessFence
    candidate_source_ids: tuple[str, ...]
    capability_state: str = SOURCE_TRAIN_CAPABILITY_STATE
    capability_hash: str = field(init=False)

    def __post_init__(self) -> None:
        center = str(self.center_id)
        candidates = tuple(str(value) for value in self.candidate_source_ids)
        expected = tuple(value for value in CENTERS if value != center)
        if center not in CENTERS or candidates != expected:
            raise ProtocolError("HARP v17 source capability candidate pool drifted.")
        if type(self.source_train_label_access_fence) is not SourceTrainLabelAccessFence:
            raise ProtocolError("HARP v17 source-train label fence is absent.")
        self.source_train_label_access_fence.authorize(center)
        source_path, source_sha, source = _read_role_seal(
            self.source_train_menu_seal_path,
            self.source_train_menu_seal_sha256,
            center_id=center,
            surface_role=SOURCE_TRAIN_SURFACE_ROLE,
            candidate_source_ids=candidates,
        )
        target_path, target_sha, target = _read_role_seal(
            self.target_evaluation_menu_seal_path,
            self.target_evaluation_menu_seal_sha256,
            center_id=center,
            surface_role=TARGET_EVALUATION_SURFACE_ROLE,
            candidate_source_ids=candidates,
        )
        attestation_path, attestation_sha, attestation = _read_bank_attestation(
            self.bank_independence_attestation_path,
            self.bank_independence_attestation_sha256,
            center_id=center,
        )
        label_path, label_sha = _authenticated_file(
            self.label_index_path,
            self.label_index_sha256,
            name="source-train label index",
        )
        if (
            source_path == target_path
            or source.get("menu_hash") == target.get("menu_hash")
            or source.get("action_identity_hash")
            != target.get("action_identity_hash")
            or self.capability_state != SOURCE_TRAIN_CAPABILITY_STATE
        ):
            raise ProtocolError("HARP v17 source/target seals are not independent.")
        body = {
            "schema_version": "midogpp_harp_v17_source_train_label_capability_v1",
            "experiment_id": EXPERIMENT_ID,
            "center_id": center,
            "capability_state": SOURCE_TRAIN_CAPABILITY_STATE,
            "candidate_source_ids": list(candidates),
            "source_train_menu_seal_path": str(source_path),
            "source_train_menu_seal_sha256": source_sha,
            "source_train_menu_seal_hash": source["seal_hash"],
            "target_evaluation_menu_seal_path": str(target_path),
            "target_evaluation_menu_seal_sha256": target_sha,
            "target_evaluation_menu_seal_hash": target["seal_hash"],
            "bank_independence_attestation_path": str(attestation_path),
            "bank_independence_attestation_sha256": attestation_sha,
            "bank_independence_per_center_hash": attestation["per_center_hashes"][center],
            "label_index_path": str(label_path),
            "label_index_sha256": label_sha,
            "source_train_label_access_fence_hash": (
                self.source_train_label_access_fence.fence_hash
            ),
            "exactly_one_source_center_authorized": True,
            "target_evaluation_labels_authorized": False,
        }
        object.__setattr__(self, "center_id", center)
        object.__setattr__(self, "candidate_source_ids", candidates)
        object.__setattr__(self, "source_train_menu_seal_path", source_path)
        object.__setattr__(self, "source_train_menu_seal_sha256", source_sha)
        object.__setattr__(self, "target_evaluation_menu_seal_path", target_path)
        object.__setattr__(self, "target_evaluation_menu_seal_sha256", target_sha)
        object.__setattr__(self, "bank_independence_attestation_path", attestation_path)
        object.__setattr__(self, "bank_independence_attestation_sha256", attestation_sha)
        object.__setattr__(self, "label_index_path", label_path)
        object.__setattr__(self, "label_index_sha256", label_sha)
        object.__setattr__(self, "capability_hash", canonical_hash(body))

    def authorize(self, allowed_center_ids: Sequence[str]) -> None:
        """Reauthenticate all global indexes and this q's durable members."""

        if tuple(str(value) for value in allowed_center_ids) != (self.center_id,):
            raise ProtocolError("HARP v17 source capability is cross-scoped.")
        self.source_train_label_access_fence.authorize(self.center_id)
        _read_role_seal(
            self.source_train_menu_seal_path,
            self.source_train_menu_seal_sha256,
            center_id=self.center_id,
            surface_role=SOURCE_TRAIN_SURFACE_ROLE,
            candidate_source_ids=self.candidate_source_ids,
        )
        _read_role_seal(
            self.target_evaluation_menu_seal_path,
            self.target_evaluation_menu_seal_sha256,
            center_id=self.center_id,
            surface_role=TARGET_EVALUATION_SURFACE_ROLE,
            candidate_source_ids=self.candidate_source_ids,
        )
        _read_bank_attestation(
            self.bank_independence_attestation_path,
            self.bank_independence_attestation_sha256,
            center_id=self.center_id,
        )
        _authenticated_file(
            self.label_index_path,
            self.label_index_sha256,
            name="source-train label index",
        )


@dataclass(frozen=True, slots=True)
class SourceTrainLabelCapabilitySet:
    """Exact, one-per-q capability inventory issued in canonical center order."""

    capabilities: tuple[SourceTrainLabelCapability, ...]
    capability_set_hash: str = field(init=False)

    def __post_init__(self) -> None:
        rows = tuple(self.capabilities)
        if (
            len(rows) != len(CENTERS)
            or any(type(row) is not SourceTrainLabelCapability for row in rows)
            or tuple(row.center_id for row in rows) != CENTERS
            or len({row.center_id for row in rows}) != len(CENTERS)
            or len({row.capability_hash for row in rows}) != len(CENTERS)
        ):
            raise ProtocolError("HARP v17 source capabilities lack exact center coverage.")
        body = {
            "schema_version": "midogpp_harp_v17_source_train_label_capability_set_v1",
            "experiment_id": EXPERIMENT_ID,
            "ordered_center_ids": list(CENTERS),
            "capability_hashes": [row.capability_hash for row in rows],
            "exactly_one_capability_per_source_center": True,
            "exact_source_center_coverage": True,
            "target_evaluation_labels_authorized": False,
        }
        object.__setattr__(self, "capabilities", rows)
        object.__setattr__(self, "capability_set_hash", canonical_hash(body))

    def for_center(self, center_id: str) -> SourceTrainLabelCapability:
        matches = tuple(row for row in self.capabilities if row.center_id == center_id)
        if len(matches) != 1:
            raise ProtocolError("HARP v17 source capability lookup is not singular.")
        return matches[0]


def issue_source_train_label_capabilities(
    *,
    seal_sets: Sequence[object],
    label_index_path: Path,
    label_index_sha256: str,
    source_train_label_access_fence: SourceTrainLabelAccessFence,
) -> SourceTrainLabelCapabilitySet:
    """Authenticate all 18 menus and all bank proofs before issuing any q grant."""

    if type(source_train_label_access_fence) is not SourceTrainLabelAccessFence:
        raise ProtocolError("HARP v17 source-train label fence is absent.")
    rows = tuple(sorted(tuple(seal_sets), key=_seal_center_id))
    if tuple(_seal_center_id(row) for row in rows) != CENTERS:
        raise ProtocolError("HARP v17 seal inventory lacks exact center coverage.")
    label_path, label_sha = _authenticated_file(
        label_index_path, label_index_sha256, name="source-train label index"
    )
    authenticated: list[tuple[object, tuple[str, ...]]] = []
    # Complete the global reauthentication pass before constructing the first
    # q capability; a partial valid prefix can never authorize source truth.
    for row in rows:
        center = _seal_center_id(row)
        candidates = tuple(value for value in CENTERS if value != center)
        source_train_label_access_fence.authorize(center)
        _read_role_seal(
            Path(getattr(row, "source_train_menu_seal_path")),
            str(getattr(row, "source_train_menu_seal_sha256")),
            center_id=center,
            surface_role=SOURCE_TRAIN_SURFACE_ROLE,
            candidate_source_ids=candidates,
        )
        _read_role_seal(
            Path(getattr(row, "target_evaluation_menu_seal_path")),
            str(getattr(row, "target_evaluation_menu_seal_sha256")),
            center_id=center,
            surface_role=TARGET_EVALUATION_SURFACE_ROLE,
            candidate_source_ids=candidates,
        )
        _read_bank_attestation(
            Path(getattr(row, "bank_independence_attestation_path")),
            str(getattr(row, "bank_independence_attestation_sha256")),
            center_id=center,
        )
        authenticated.append((row, candidates))
    capabilities = tuple(
        SourceTrainLabelCapability(
            center_id=_seal_center_id(row),
            source_train_menu_seal_path=Path(getattr(row, "source_train_menu_seal_path")),
            source_train_menu_seal_sha256=str(getattr(row, "source_train_menu_seal_sha256")),
            target_evaluation_menu_seal_path=Path(
                getattr(row, "target_evaluation_menu_seal_path")
            ),
            target_evaluation_menu_seal_sha256=str(
                getattr(row, "target_evaluation_menu_seal_sha256")
            ),
            bank_independence_attestation_path=Path(
                getattr(row, "bank_independence_attestation_path")
            ),
            bank_independence_attestation_sha256=str(
                getattr(row, "bank_independence_attestation_sha256")
            ),
            label_index_path=label_path,
            label_index_sha256=label_sha,
            source_train_label_access_fence=source_train_label_access_fence,
            candidate_source_ids=candidates,
        )
        for row, candidates in authenticated
    )
    return SourceTrainLabelCapabilitySet(capabilities=capabilities)


def _seal_center_id(value: object) -> str:
    center = getattr(value, "center_id", None)
    if type(center) is not str:
        raise ProtocolError("HARP v17 surface seal lacks a center identity.")
    return center


def _read_role_seal(
    path: Path,
    digest: str,
    *,
    center_id: str,
    surface_role: str,
    candidate_source_ids: tuple[str, ...],
) -> tuple[Path, str, Mapping[str, object]]:
    resolved, expected_sha = _authenticated_file(path, digest, name="menu seal")
    payload = read_json(resolved)
    body = {key: value for key, value in payload.items() if key != "seal_hash"}
    schema = (
        "midogpp_harp_v17_source_train_menu_seal_v1"
        if surface_role == SOURCE_TRAIN_SURFACE_ROLE
        else "midogpp_harp_v17_target_evaluation_menu_seal_v1"
    )
    if (
        set(payload) != _SEAL_KEYS
        or payload.get("schema_version") != schema
        or payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("center_id") != center_id
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
        raise ProtocolError("HARP v17 source/target role seal drifted.")
    return resolved, expected_sha, payload


def _read_bank_attestation(
    path: Path, digest: str, *, center_id: str
) -> tuple[Path, str, Mapping[str, object]]:
    resolved, expected_sha = _authenticated_file(
        path, digest, name="bank-independence attestation"
    )
    payload = read_json(resolved)
    body = {key: value for key, value in payload.items() if key != "attestation_hash"}
    per_center = payload.get("per_center_hashes")
    if (
        set(payload) != _ATTESTATION_KEYS
        or payload.get("schema_version") != "midogpp_harp_v17_fixed_bank_independence_v1"
        or any(
            require_sha256(payload.get(key), name=f"bank attestation {key}")
            != payload.get(key)
            for key in (
                "bank_index_sha256",
                "generation_lock_sha256",
                "source_local_lineage_hash",
            )
        )
        or not isinstance(per_center, Mapping)
        or tuple(str(key) for key in per_center) != CENTERS
        or any(
            require_sha256(value, name="per-center independence hash") != value
            for value in per_center.values()
        )
        or center_id not in per_center
        or payload.get("candidate_pool_semantics") != "C_MINUS_CONTEXT_CENTER"
        or payload.get("own_center_expert_unrepresentable") is not True
        or payload.get("source_frames_and_samplers_source_center_local") is not True
        or payload.get("classifier_scaler_fit") != "synthetic_train_only"
        or payload.get("source_train_labels_may_update") != "POOLED_ROUTER_ONLY"
        or tuple(payload.get("source_train_labels_may_not_update", ()))
        != _SOURCE_LABEL_FORBIDDEN_UPDATES
        or payload.get("labels_consumed") is not False
        or payload.get("attestation_hash") != canonical_hash(body)
    ):
        raise ProtocolError("HARP v17 bank-independence attestation drifted.")
    return resolved, expected_sha, payload


def _authenticated_file(path: Path, digest: str, *, name: str) -> tuple[Path, str]:
    raw = Path(path)
    resolved = raw.resolve()
    expected = require_sha256(digest, name=name)
    if raw.is_symlink() or not resolved.is_file() or sha256_file(resolved) != expected:
        raise ProtocolError(f"HARP v17 {name} is absent or drifted.")
    return resolved, expected


__all__ = (
    "SOURCE_TRAIN_CAPABILITY_STATE",
    "SOURCE_TRAIN_SURFACE_ROLE",
    "TARGET_EVALUATION_SURFACE_ROLE",
    "SourceTrainLabelCapability",
    "SourceTrainLabelCapabilitySet",
    "issue_source_train_label_capabilities",
)
