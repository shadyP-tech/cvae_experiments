"""Durable fail-closed fence for the first HARP v19 source-train-label access.

The fence is committed before any center capability can be issued.  Its mere
presence makes label-free recovery ineligible, including the narrow crash
window between the durable write and the first label-shard read.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_bytes, canonical_hash, require_sha256
from ...runtime.artifact_io import atomic_json, read_json, sha256_file
from ...runtime.harp_v19_execution.durability import durable_barrier
from .identity import EXPERIMENT_ID, PUBLICATION_STATUS, TERMINAL_DECISION


SOURCE_TRAIN_LABEL_ACCESS_STATE = "SOURCE_TRAIN_LABEL_ACCESS_BEGUN"
SOURCE_TRAIN_LABEL_ACCESS_FENCE_MEMBER = (
    "manifests/source_train_label_access_begun.json"
)
_SCHEMA = "midogpp_harp_v19_source_train_label_access_begun_v1"
_FENCE_KEYS = {
    "schema_version",
    "state",
    "experiment_id",
    "config_hash",
    "admission_hash",
    "authorization_lease_hash",
    "ordered_center_ids",
    "source_train_surface_seal_index_path",
    "source_train_surface_seal_index_sha256",
    "source_train_surface_seal_index_hash",
    "target_surface_seal_index_path",
    "target_surface_seal_index_sha256",
    "target_surface_seal_index_hash",
    "bank_independence_index_path",
    "bank_independence_index_sha256",
    "bank_independence_index_hash",
    "label_index_sha256",
    "source_train_label_capability_issuance_may_begin",
    "source_train_label_shard_reads_may_begin",
    "source_train_label_access_irreversibly_begun",
    "evaluation_labels_opened",
    "label_free_recovery_allowed",
    "publication_status",
    "terminal_decision",
    "fresh_evidence",
    "fence_hash",
}


@dataclass(frozen=True, slots=True)
class SourceTrainLabelAccessFence:
    """Authenticated evidence that source-train-label access has irreversibly begun."""

    path: Path
    sha256: str
    fence_hash: str
    config_hash: str
    admission_hash: str
    authorization_lease_hash: str
    ordered_center_ids: tuple[str, ...]
    source_train_surface_seal_index_path: Path
    source_train_surface_seal_index_sha256: str
    source_train_surface_seal_index_hash: str
    target_surface_seal_index_path: Path
    target_surface_seal_index_sha256: str
    target_surface_seal_index_hash: str
    bank_independence_index_path: Path
    bank_independence_index_sha256: str
    bank_independence_index_hash: str
    label_index_sha256: str

    def __post_init__(self) -> None:
        raw_path = Path(self.path)
        path = raw_path.resolve()
        sha = require_sha256(self.sha256, name="source-train-label access fence bytes")
        if raw_path.is_symlink() or not path.is_file() or sha256_file(path) != sha:
            raise ProtocolError("HARP v19 source-train-label access fence bytes drifted.")
        for name in (
            "source_train_surface_seal_index_path",
            "target_surface_seal_index_path",
            "bank_independence_index_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)).resolve())
        payload = _validate_payload(read_json(path))
        expected = self._expected_payload()
        if canonical_bytes(payload) != canonical_bytes(expected):
            raise ProtocolError("HARP v19 source-train-label access fence identity drifted.")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "sha256", sha)
        _reauthenticate_bound_indexes(self)

    def _expected_payload(self) -> dict[str, object]:
        body = _fence_body(
            config_hash=self.config_hash,
            admission_hash=self.admission_hash,
            authorization_lease_hash=self.authorization_lease_hash,
            ordered_center_ids=self.ordered_center_ids,
            source_train_surface_seal_index_path=self.source_train_surface_seal_index_path,
            source_train_surface_seal_index_sha256=self.source_train_surface_seal_index_sha256,
            source_train_surface_seal_index_hash=self.source_train_surface_seal_index_hash,
            target_surface_seal_index_path=self.target_surface_seal_index_path,
            target_surface_seal_index_sha256=self.target_surface_seal_index_sha256,
            target_surface_seal_index_hash=self.target_surface_seal_index_hash,
            bank_independence_index_path=self.bank_independence_index_path,
            bank_independence_index_sha256=self.bank_independence_index_sha256,
            bank_independence_index_hash=self.bank_independence_index_hash,
            label_index_sha256=self.label_index_sha256,
        )
        return {**body, "fence_hash": canonical_hash(body)}

    def authorize(self, outer_target_id: str) -> None:
        """Reauthenticate the fence immediately before capability or shard access."""

        if str(outer_target_id) not in self.ordered_center_ids:
            raise ProtocolError("HARP v19 source-train-label fence is cross-scoped.")
        self.reauthenticate()

    def reauthenticate(self) -> None:
        """Revalidate the committed fence and each bound pre-label index byte."""

        if sha256_file(self.path) != self.sha256:
            raise ProtocolError("HARP v19 source-train-label access fence bytes drifted.")
        observed = _validate_payload(read_json(self.path))
        if canonical_bytes(observed) != canonical_bytes(self._expected_payload()):
            raise ProtocolError("HARP v19 source-train-label access fence changed after commit.")
        _reauthenticate_bound_indexes(self)


def begin_source_train_label_access(
    root: Path,
    *,
    config_hash: str,
    admission_hash: str,
    authorization_lease_hash: str,
    ordered_center_ids: Sequence[str],
    source_train_surface_seal_index: Mapping[str, object],
    source_train_surface_seal_index_path: Path,
    target_surface_seal_index: Mapping[str, object],
    target_surface_seal_index_path: Path,
    bank_independence_index: Mapping[str, object],
    bank_independence_index_path: Path,
    label_index_sha256: str,
) -> SourceTrainLabelAccessFence:
    """Commit and fsync the one-way fence before any source-train label can open."""

    centers = tuple(str(value) for value in ordered_center_ids)
    if centers != CENTERS:
        raise ProtocolError("HARP v19 source-train-label fence center inventory drifted.")
    _validate_index(source_train_surface_seal_index, centers=centers, name="source train")
    _validate_index(target_surface_seal_index, centers=centers, name="target")
    _validate_index(bank_independence_index, centers=centers, name="bank independence")
    root = Path(root).resolve()
    source_path, source_sha = _authenticate_index_file(
        root,
        source_train_surface_seal_index_path,
        expected_member="manifests/source_train_menu_seals.json",
        expected_payload=source_train_surface_seal_index,
        centers=centers,
        name="source train",
    )
    target_path, target_sha = _authenticate_index_file(
        root,
        target_surface_seal_index_path,
        expected_member="manifests/target_evaluation_menu_seals.json",
        expected_payload=target_surface_seal_index,
        centers=centers,
        name="target",
    )
    bank_path, bank_sha = _authenticate_index_file(
        root,
        bank_independence_index_path,
        expected_member="manifests/bank_independence_attestations.json",
        expected_payload=bank_independence_index,
        centers=centers,
        name="bank independence",
    )
    body = _fence_body(
        config_hash=config_hash,
        admission_hash=admission_hash,
        authorization_lease_hash=authorization_lease_hash,
        ordered_center_ids=centers,
        source_train_surface_seal_index_path=source_path,
        source_train_surface_seal_index_sha256=source_sha,
        source_train_surface_seal_index_hash=str(source_train_surface_seal_index["index_hash"]),
        target_surface_seal_index_path=target_path,
        target_surface_seal_index_sha256=target_sha,
        target_surface_seal_index_hash=str(target_surface_seal_index["index_hash"]),
        bank_independence_index_path=bank_path,
        bank_independence_index_sha256=bank_sha,
        bank_independence_index_hash=str(bank_independence_index["index_hash"]),
        label_index_sha256=label_index_sha256,
    )
    payload = {**body, "fence_hash": canonical_hash(body)}
    path = root / SOURCE_TRAIN_LABEL_ACCESS_FENCE_MEMBER
    if path.exists() or path.is_symlink():
        raise ProtocolError("HARP v19 source-train-label access fence already exists.")
    atomic_json(path, payload)
    durable_barrier((path,))
    observed = _validate_payload(read_json(path))
    if canonical_bytes(observed) != canonical_bytes(payload):
        raise ProtocolError("HARP v19 source-train-label access fence failed durable readback.")
    return SourceTrainLabelAccessFence(
        path=path,
        sha256=sha256_file(path),
        fence_hash=str(payload["fence_hash"]),
        config_hash=require_sha256(config_hash, name="source-train-label fence config"),
        admission_hash=require_sha256(admission_hash, name="source-train-label fence admission"),
        authorization_lease_hash=require_sha256(
            authorization_lease_hash, name="source-train-label fence lease"
        ),
        ordered_center_ids=centers,
        source_train_surface_seal_index_path=source_path,
        source_train_surface_seal_index_sha256=source_sha,
        source_train_surface_seal_index_hash=str(
            source_train_surface_seal_index["index_hash"]
        ),
        target_surface_seal_index_path=target_path,
        target_surface_seal_index_sha256=target_sha,
        target_surface_seal_index_hash=str(target_surface_seal_index["index_hash"]),
        bank_independence_index_path=bank_path,
        bank_independence_index_sha256=bank_sha,
        bank_independence_index_hash=str(bank_independence_index["index_hash"]),
        label_index_sha256=require_sha256(
            label_index_sha256, name="source-train-label fence label index"
        ),
    )


def source_train_label_access_has_begun(root: Path) -> bool:
    """Return true for any durable or suspicious fence pathname, fail closed."""

    path = Path(root).resolve() / SOURCE_TRAIN_LABEL_ACCESS_FENCE_MEMBER
    return path.exists() or path.is_symlink()


def load_source_train_label_access_fence(root: Path) -> SourceTrainLabelAccessFence:
    """Load the fence and reauthenticate every bound pre-label index byte."""

    path = Path(root).resolve() / SOURCE_TRAIN_LABEL_ACCESS_FENCE_MEMBER
    payload = _validate_payload(read_json(path))
    return SourceTrainLabelAccessFence(
        path=path,
        sha256=sha256_file(path),
        fence_hash=str(payload["fence_hash"]),
        config_hash=str(payload["config_hash"]),
        admission_hash=str(payload["admission_hash"]),
        authorization_lease_hash=str(payload["authorization_lease_hash"]),
        ordered_center_ids=tuple(str(value) for value in payload["ordered_center_ids"]),
        source_train_surface_seal_index_path=Path(
            str(payload["source_train_surface_seal_index_path"])
        ),
        source_train_surface_seal_index_sha256=str(
            payload["source_train_surface_seal_index_sha256"]
        ),
        source_train_surface_seal_index_hash=str(
            payload["source_train_surface_seal_index_hash"]
        ),
        target_surface_seal_index_path=Path(
            str(payload["target_surface_seal_index_path"])
        ),
        target_surface_seal_index_sha256=str(
            payload["target_surface_seal_index_sha256"]
        ),
        target_surface_seal_index_hash=str(payload["target_surface_seal_index_hash"]),
        bank_independence_index_path=Path(
            str(payload["bank_independence_index_path"])
        ),
        bank_independence_index_sha256=str(
            payload["bank_independence_index_sha256"]
        ),
        bank_independence_index_hash=str(payload["bank_independence_index_hash"]),
        label_index_sha256=str(payload["label_index_sha256"]),
    )


def _fence_body(
    *,
    config_hash: str,
    admission_hash: str,
    authorization_lease_hash: str,
    ordered_center_ids: Sequence[str],
    source_train_surface_seal_index_path: Path,
    source_train_surface_seal_index_sha256: str,
    source_train_surface_seal_index_hash: str,
    target_surface_seal_index_path: Path,
    target_surface_seal_index_sha256: str,
    target_surface_seal_index_hash: str,
    bank_independence_index_path: Path,
    bank_independence_index_sha256: str,
    bank_independence_index_hash: str,
    label_index_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA,
        "state": SOURCE_TRAIN_LABEL_ACCESS_STATE,
        "experiment_id": EXPERIMENT_ID,
        "config_hash": require_sha256(config_hash, name="source-train-label fence config"),
        "admission_hash": require_sha256(
            admission_hash, name="source-train-label fence admission"
        ),
        "authorization_lease_hash": require_sha256(
            authorization_lease_hash, name="source-train-label fence lease"
        ),
        "ordered_center_ids": list(ordered_center_ids),
        "source_train_surface_seal_index_path": str(
            Path(source_train_surface_seal_index_path).resolve()
        ),
        "source_train_surface_seal_index_sha256": require_sha256(
            source_train_surface_seal_index_sha256, name="source-train-label source index bytes"
        ),
        "source_train_surface_seal_index_hash": require_sha256(
            source_train_surface_seal_index_hash, name="source-train-label source index"
        ),
        "target_surface_seal_index_path": str(
            Path(target_surface_seal_index_path).resolve()
        ),
        "target_surface_seal_index_sha256": require_sha256(
            target_surface_seal_index_sha256, name="source-train-label target index bytes"
        ),
        "target_surface_seal_index_hash": require_sha256(
            target_surface_seal_index_hash, name="source-train-label target index"
        ),
        "bank_independence_index_path": str(
            Path(bank_independence_index_path).resolve()
        ),
        "bank_independence_index_sha256": require_sha256(
            bank_independence_index_sha256, name="source-train-label bank index bytes"
        ),
        "bank_independence_index_hash": require_sha256(
            bank_independence_index_hash, name="source-train-label bank index"
        ),
        "label_index_sha256": require_sha256(
            label_index_sha256, name="source-train-label label index"
        ),
        "source_train_label_capability_issuance_may_begin": True,
        "source_train_label_shard_reads_may_begin": True,
        "source_train_label_access_irreversibly_begun": True,
        "evaluation_labels_opened": False,
        "label_free_recovery_allowed": False,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "fresh_evidence": False,
    }


def _validate_index(
    value: Mapping[str, object], *, centers: tuple[str, ...], name: str
) -> None:
    if (
        tuple(value.get("ordered_center_ids", ())) != centers
        or require_sha256(value.get("index_hash"), name=f"source-train-label {name} index")
        != value.get("index_hash")
        or value.get("source_train_labels_opened") is not False
        or value.get("evaluation_labels_opened") is not False
    ):
        raise ProtocolError(f"HARP v19 {name} index cannot authorize label access.")


def _authenticate_index_file(
    root: Path,
    path: Path,
    *,
    expected_member: str,
    expected_payload: Mapping[str, object],
    centers: tuple[str, ...],
    name: str,
) -> tuple[Path, str]:
    raw_path = Path(path)
    resolved = raw_path.resolve()
    if (
        resolved != root / expected_member
        or raw_path.is_symlink()
        or not resolved.is_file()
    ):
        raise ProtocolError(f"HARP v19 {name} index path is absent or unsafe.")
    observed = read_json(resolved)
    _validate_index(observed, centers=centers, name=name)
    if canonical_bytes(observed) != canonical_bytes(expected_payload):
        raise ProtocolError(f"HARP v19 {name} index bytes disagree with memory.")
    return resolved, sha256_file(resolved)


def _reauthenticate_bound_indexes(fence: SourceTrainLabelAccessFence) -> None:
    specs = (
        (
            fence.source_train_surface_seal_index_path,
            fence.source_train_surface_seal_index_sha256,
            fence.source_train_surface_seal_index_hash,
            fence.path.parent / "source_train_menu_seals.json",
            "source train",
        ),
        (
            fence.target_surface_seal_index_path,
            fence.target_surface_seal_index_sha256,
            fence.target_surface_seal_index_hash,
            fence.path.parent / "target_evaluation_menu_seals.json",
            "target",
        ),
        (
            fence.bank_independence_index_path,
            fence.bank_independence_index_sha256,
            fence.bank_independence_index_hash,
            fence.path.parent / "bank_independence_attestations.json",
            "bank independence",
        ),
    )
    for path, expected_sha, expected_hash, expected_path, name in specs:
        if (
            path != expected_path
            or path.is_symlink()
            or not path.is_file()
            or sha256_file(path)
            != require_sha256(expected_sha, name=f"source-train-label {name} index bytes")
        ):
            raise ProtocolError(f"HARP v19 {name} index bytes drifted after fence.")
        payload = read_json(path)
        _validate_index(payload, centers=fence.ordered_center_ids, name=name)
        if payload.get("index_hash") != expected_hash:
            raise ProtocolError(f"HARP v19 {name} index identity drifted after fence.")


def _validate_payload(value: Mapping[str, object]) -> dict[str, object]:
    payload = dict(value)
    body = {key: member for key, member in payload.items() if key != "fence_hash"}
    if (
        set(payload) != _FENCE_KEYS
        or payload.get("schema_version") != _SCHEMA
        or payload.get("state") != SOURCE_TRAIN_LABEL_ACCESS_STATE
        or payload.get("experiment_id") != EXPERIMENT_ID
        or tuple(payload.get("ordered_center_ids", ())) != CENTERS
        or payload.get("source_train_label_capability_issuance_may_begin") is not True
        or payload.get("source_train_label_shard_reads_may_begin") is not True
        or payload.get("source_train_label_access_irreversibly_begun") is not True
        or payload.get("evaluation_labels_opened") is not False
        or payload.get("label_free_recovery_allowed") is not False
        or payload.get("publication_status") != PUBLICATION_STATUS
        or payload.get("terminal_decision") != TERMINAL_DECISION
        or payload.get("fresh_evidence") is not False
        or payload.get("fence_hash") != canonical_hash(body)
    ):
        raise ProtocolError("HARP v19 source-train-label access fence drifted.")
    for key in (
        "config_hash",
        "admission_hash",
        "authorization_lease_hash",
        "source_train_surface_seal_index_sha256",
        "source_train_surface_seal_index_hash",
        "target_surface_seal_index_sha256",
        "target_surface_seal_index_hash",
        "bank_independence_index_sha256",
        "bank_independence_index_hash",
        "label_index_sha256",
        "fence_hash",
    ):
        require_sha256(payload.get(key), name=f"source-train-label fence {key}")
    return payload


__all__ = (
    "SOURCE_TRAIN_LABEL_ACCESS_FENCE_MEMBER",
    "SOURCE_TRAIN_LABEL_ACCESS_STATE",
    "SourceTrainLabelAccessFence",
    "begin_source_train_label_access",
    "load_source_train_label_access_fence",
    "source_train_label_access_has_begun",
)
