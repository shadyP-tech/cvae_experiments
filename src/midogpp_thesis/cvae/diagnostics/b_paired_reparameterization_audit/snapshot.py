"""Canonical, hash-bound Stage-90 audit snapshot schema."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.cvae.protocol import ProtocolError

from .config import (
    AUDIT_CENTERS,
    CLAIM_SCOPE,
    EVIDENCE_LABEL,
    STAGE,
    ClaimFirewall,
)
from .protocol import (
    AuditKeyRecord,
    key_inventory_hash,
    key_record_from_mapping,
    validate_key_inventory,
)


SNAPSHOT_SCHEMA = "midogpp_b_paired_reparameterization_audit_snapshot_v1"
MANIFEST_SCHEMA = "midogpp_b_paired_reparameterization_audit_manifest_v1"
CONTENT_INDEX_SCHEMA = "midogpp_b_paired_reparameterization_content_index_v1"
PENDING_HASH_PROMOTION = "PENDING_HASH_PROMOTION"
HASH_PROMOTED = "HASH_PROMOTED"
PUBLICATION_STATES = (PENDING_HASH_PROMOTION, HASH_PROMOTED)

FIT_LABEL_ROLE = "source_fit_only"
EVAL_LABEL_ROLE = "final_diagnostic_scoring_only"


@dataclass(frozen=True)
class ArrayBinding:
    relative_path: str
    sha256: str
    content_hash: str
    dtype: str
    shape: tuple[int, ...]
    role: str

    def to_payload(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "content_hash": self.content_hash,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "role": self.role,
        }


@dataclass(frozen=True)
class PreparedPartition:
    features: ArrayBinding
    sample_ids: ArrayBinding
    case_ids: ArrayBinding
    class_labels: ArrayBinding
    case_id_inventory: tuple[str, ...]
    row_inventory_hash: str
    sample_id_inventory_hash: str
    case_id_inventory_hash: str
    row_count: int
    sample_count: int
    case_count: int

    def to_payload(self) -> dict[str, object]:
        return {
            "features": self.features.to_payload(),
            "sample_ids": self.sample_ids.to_payload(),
            "case_ids": self.case_ids.to_payload(),
            "class_labels": self.class_labels.to_payload(),
            "case_id_inventory": list(self.case_id_inventory),
            "row_inventory_hash": self.row_inventory_hash,
            "sample_id_inventory_hash": self.sample_id_inventory_hash,
            "case_id_inventory_hash": self.case_id_inventory_hash,
            "row_count": self.row_count,
            "sample_count": self.sample_count,
            "case_count": self.case_count,
        }


@dataclass(frozen=True)
class ContentEntry:
    relative_path: str
    sha256: str
    content_hash: str
    size_bytes: int
    role: str

    def to_payload(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "content_hash": self.content_hash,
            "size_bytes": self.size_bytes,
            "role": self.role,
        }


@dataclass(frozen=True)
class CenterPreparedData:
    """Portable prepared bundle and disjoint source-fit/eval partitions for one center."""

    center: str
    prepared_bundle: ContentEntry
    fit: PreparedPartition
    evaluation: PreparedPartition

    def to_payload(self) -> dict[str, object]:
        return {
            "center": self.center,
            "prepared_bundle": self.prepared_bundle.to_payload(),
            "fit": self.fit.to_payload(),
            "evaluation": self.evaluation.to_payload(),
        }


@dataclass(frozen=True)
class AuditSnapshot:
    """Self-contained pre-run inventory; historical paths are never dereferenced."""

    publication_state: str
    config_hash: str
    protocol_hash: str
    dataset_id: str
    feature_frame: str
    domain_axis: str
    prepared_centers: tuple[CenterPreparedData, ...]
    keys: tuple[AuditKeyRecord, ...]
    content_index: tuple[ContentEntry, ...]
    historical_paths: tuple[str, ...]
    manifest_hash: str
    key_inventory_hash: str
    content_index_hash: str
    snapshot_hash: str
    schema_version: str = SNAPSHOT_SCHEMA
    stage: str = STAGE
    evidence_label: str = EVIDENCE_LABEL
    claim_scope: str = CLAIM_SCOPE
    historical_paths_read: bool = False
    fit_class_labels_used_for_source_fit_only: bool = True
    eval_class_labels_used_for_final_diagnostic_scoring_only: bool = True
    eval_class_labels_used_for_training_or_selection: bool = False
    claim_firewall: ClaimFirewall = ClaimFirewall()

    def manifest_payload(self) -> dict[str, object]:
        return {
            "schema_version": MANIFEST_SCHEMA,
            "config_hash": self.config_hash,
            "protocol_hash": self.protocol_hash,
            "dataset_id": self.dataset_id,
            "feature_frame": self.feature_frame,
            "domain_axis": self.domain_axis,
            "prepared_centers": [
                prepared.to_payload()
                for prepared in sorted(
                    self.prepared_centers, key=lambda item: AUDIT_CENTERS.index(item.center)
                )
            ],
            "historical_paths": list(self.historical_paths),
            "historical_paths_read": self.historical_paths_read,
            "fit_class_labels_used_for_source_fit_only": (
                self.fit_class_labels_used_for_source_fit_only
            ),
            "eval_class_labels_used_for_final_diagnostic_scoring_only": (
                self.eval_class_labels_used_for_final_diagnostic_scoring_only
            ),
            "eval_class_labels_used_for_training_or_selection": (
                self.eval_class_labels_used_for_training_or_selection
            ),
            "claim_firewall": self.claim_firewall.to_payload(),
        }

    def content_index_payload(self) -> dict[str, object]:
        return {
            "schema_version": CONTENT_INDEX_SCHEMA,
            "entries": [
                entry.to_payload()
                for entry in sorted(self.content_index, key=lambda item: item.relative_path)
            ],
        }

    def hash_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "stage": self.stage,
            "evidence_label": self.evidence_label,
            "claim_scope": self.claim_scope,
            "publication_state": self.publication_state,
            "manifest_hash": self.manifest_hash,
            "key_inventory_hash": self.key_inventory_hash,
            "content_index_hash": self.content_index_hash,
        }

    def to_payload(self) -> dict[str, object]:
        return {
            **self.hash_payload(),
            "manifest": self.manifest_payload(),
            "key_inventory": {
                "records": [record.to_payload() for record in self.keys]
            },
            "content_index": self.content_index_payload(),
            "snapshot_hash": self.snapshot_hash,
        }


def build_snapshot(
    *,
    publication_state: str,
    config_hash: str,
    protocol_hash: str,
    dataset_id: str,
    feature_frame: str,
    domain_axis: str,
    prepared_centers: Iterable[CenterPreparedData],
    keys: Iterable[AuditKeyRecord],
    content_index: Iterable[ContentEntry],
    historical_paths: Iterable[str] = (),
) -> AuditSnapshot:
    """Construct a snapshot and compute every semantic binding."""

    provisional = AuditSnapshot(
        publication_state=publication_state,
        config_hash=config_hash,
        protocol_hash=protocol_hash,
        dataset_id=dataset_id,
        feature_frame=feature_frame,
        domain_axis=domain_axis,
        prepared_centers=tuple(prepared_centers),
        keys=tuple(keys),
        content_index=tuple(content_index),
        historical_paths=tuple(str(value) for value in historical_paths),
        manifest_hash="",
        key_inventory_hash="",
        content_index_hash="",
        snapshot_hash="",
    )
    manifest_hash = stable_hash(provisional.manifest_payload())
    for record in provisional.keys:
        if record.snapshot_manifest_hash != manifest_hash:
            raise ProtocolError(
                "Audit key snapshot_manifest_hash does not bind the constructed manifest."
            )
    inventory_hash = key_inventory_hash(provisional.keys)
    content_hash = stable_hash(provisional.content_index_payload())
    bound = _replace_hashes(
        provisional,
        manifest_hash=manifest_hash,
        inventory_hash=inventory_hash,
        content_hash=content_hash,
        snapshot_hash="",
    )
    return _replace_hashes(
        bound,
        manifest_hash=manifest_hash,
        inventory_hash=inventory_hash,
        content_hash=content_hash,
        snapshot_hash=stable_hash(bound.hash_payload()),
    )


def snapshot_manifest_hash(
    *,
    config_hash: str,
    protocol_hash: str,
    dataset_id: str,
    feature_frame: str,
    domain_axis: str,
    prepared_centers: Iterable[CenterPreparedData],
    historical_paths: Iterable[str] = (),
) -> str:
    """Compute the manifest binding before constructing the 36 key records."""

    provisional = AuditSnapshot(
        publication_state=PENDING_HASH_PROMOTION,
        config_hash=config_hash,
        protocol_hash=protocol_hash,
        dataset_id=dataset_id,
        feature_frame=feature_frame,
        domain_axis=domain_axis,
        prepared_centers=tuple(prepared_centers),
        keys=(),
        content_index=(),
        historical_paths=tuple(str(value) for value in historical_paths),
        manifest_hash="",
        key_inventory_hash="",
        content_index_hash="",
        snapshot_hash="",
    )
    return stable_hash(provisional.manifest_payload())


def snapshot_from_mapping(payload: Mapping[str, object]) -> AuditSnapshot:
    """Parse a serialized snapshot; call :func:`validate_snapshot` before use."""

    manifest = _mapping(payload, "manifest")
    inventory = _mapping(payload, "key_inventory")
    content = _mapping(payload, "content_index")
    records = inventory.get("records")
    entries = content.get("entries")
    if not isinstance(records, list) or not isinstance(entries, list):
        raise ProtocolError("Snapshot key inventory and content index must be lists.")
    try:
        snapshot = AuditSnapshot(
            schema_version=str(payload.get("schema_version", "")),
            stage=str(payload.get("stage", "")),
            evidence_label=str(payload.get("evidence_label", "")),
            claim_scope=str(payload.get("claim_scope", "")),
            publication_state=str(payload.get("publication_state", "")),
            config_hash=str(manifest["config_hash"]),
            protocol_hash=str(manifest["protocol_hash"]),
            dataset_id=str(manifest["dataset_id"]),
            feature_frame=str(manifest["feature_frame"]),
            domain_axis=str(manifest["domain_axis"]),
            prepared_centers=tuple(
                _center_prepared_from_mapping(_as_mapping(row))
                for row in _as_list(manifest.get("prepared_centers"), "prepared_centers")
            ),
            keys=tuple(key_record_from_mapping(_as_mapping(row)) for row in records),
            content_index=tuple(
                _content_entry_from_mapping(_as_mapping(row)) for row in entries
            ),
            historical_paths=tuple(
                str(value) for value in manifest.get("historical_paths", ())
            ),
            historical_paths_read=_strict_bool(
                manifest.get("historical_paths_read"), "historical_paths_read"
            ),
            fit_class_labels_used_for_source_fit_only=_strict_bool(
                manifest.get("fit_class_labels_used_for_source_fit_only"),
                "fit_class_labels_used_for_source_fit_only",
            ),
            eval_class_labels_used_for_final_diagnostic_scoring_only=_strict_bool(
                manifest.get(
                    "eval_class_labels_used_for_final_diagnostic_scoring_only"
                ),
                "eval_class_labels_used_for_final_diagnostic_scoring_only",
            ),
            eval_class_labels_used_for_training_or_selection=_strict_bool(
                manifest.get("eval_class_labels_used_for_training_or_selection"),
                "eval_class_labels_used_for_training_or_selection",
            ),
            claim_firewall=ClaimFirewall(
                **{
                    key: _strict_bool(value, key)
                    for key, value in _mapping(manifest, "claim_firewall").items()
                }
            ),
            manifest_hash=str(payload.get("manifest_hash", "")),
            key_inventory_hash=str(payload.get("key_inventory_hash", "")),
            content_index_hash=str(payload.get("content_index_hash", "")),
            snapshot_hash=str(payload.get("snapshot_hash", "")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("Malformed Stage-90 audit snapshot.") from exc
    return snapshot


def load_snapshot(
    path: str | Path,
    *,
    artifact_root: str | Path | None = None,
    require_hash_promoted: bool = True,
) -> AuditSnapshot:
    """Load and fully validate one canonical snapshot."""

    snapshot_path = Path(path)
    try:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read Stage-90 audit snapshot: {snapshot_path}") from exc
    if not isinstance(payload, Mapping):
        raise ProtocolError("Audit snapshot root must be a mapping.")
    snapshot = snapshot_from_mapping(payload)
    validate_snapshot(
        snapshot,
        artifact_root=artifact_root,
        require_hash_promoted=require_hash_promoted,
    )
    return snapshot


def validate_snapshot(
    snapshot: AuditSnapshot,
    *,
    artifact_root: str | Path | None = None,
    require_hash_promoted: bool = True,
) -> None:
    """Validate schema, protocol firewall, semantic hashes, and optional file bytes."""

    if (
        snapshot.schema_version != SNAPSHOT_SCHEMA
        or snapshot.stage != STAGE
        or snapshot.evidence_label != EVIDENCE_LABEL
        or snapshot.claim_scope != CLAIM_SCOPE
    ):
        raise ProtocolError("Snapshot is not canonical Stage-90 AUDIT_ONLY.")
    if snapshot.publication_state not in PUBLICATION_STATES:
        raise ProtocolError("Unknown audit snapshot publication state.")
    if require_hash_promoted and snapshot.publication_state != HASH_PROMOTED:
        raise ProtocolError("Audit consumption requires publication_state=HASH_PROMOTED.")
    if snapshot.historical_paths_read:
        raise ProtocolError("Historical paths are opaque provenance and may not be read.")
    if (
        not snapshot.fit_class_labels_used_for_source_fit_only
        or not snapshot.eval_class_labels_used_for_final_diagnostic_scoring_only
        or snapshot.eval_class_labels_used_for_training_or_selection
    ):
        raise ProtocolError("Audit class-label usage crossed its frozen boundary.")
    if snapshot.claim_firewall != ClaimFirewall():
        raise ProtocolError("Audit snapshot claim firewall drifted.")
    prepared_by_center = _validate_prepared_centers(snapshot.prepared_centers)
    if not snapshot.dataset_id or not snapshot.feature_frame or not snapshot.domain_axis:
        raise ProtocolError("Audit dataset/frame/domain identities must be explicit.")
    for digest in (snapshot.config_hash, snapshot.protocol_hash):
        _validate_semantic_hash(digest)
    expected_manifest = stable_hash(snapshot.manifest_payload())
    if snapshot.manifest_hash != expected_manifest:
        raise ProtocolError("Snapshot manifest hash does not recompute.")
    records = validate_key_inventory(
        snapshot.keys, require_publication_hashes=require_hash_promoted
    )
    if any(record.snapshot_manifest_hash != snapshot.manifest_hash for record in records):
        raise ProtocolError("Training keys are not bound to this snapshot manifest.")
    for record in records:
        prepared = prepared_by_center[record.center].prepared_bundle
        if (
            record.prepared_relpath != prepared.relative_path
            or record.prepared_sha256 != prepared.sha256
            or record.prepared_content_hash != prepared.content_hash
        ):
            raise ProtocolError(
                "Training key prepared binding does not match its center partition."
            )
    if snapshot.key_inventory_hash != key_inventory_hash(records):
        raise ProtocolError("Snapshot key-inventory hash does not recompute.")
    _validate_content_index(snapshot.content_index)
    if snapshot.content_index_hash != stable_hash(snapshot.content_index_payload()):
        raise ProtocolError("Snapshot content-index hash does not recompute.")
    if snapshot.snapshot_hash != stable_hash(snapshot.hash_payload()):
        raise ProtocolError("Top-level snapshot hash does not recompute.")
    referenced: set[str] = set()
    for prepared in snapshot.prepared_centers:
        referenced.add(prepared.prepared_bundle.relative_path)
        for partition in (prepared.fit, prepared.evaluation):
            referenced.update(
                binding.relative_path
                for binding in (
                    partition.features,
                    partition.sample_ids,
                    partition.case_ids,
                    partition.class_labels,
                )
            )
    for record in records:
        referenced.update(
            (
                record.prepared_relpath,
                record.schedule_relpath,
                record.epsilon_trace_relpath,
            )
        )
    indexed = {entry.relative_path for entry in snapshot.content_index}
    missing = sorted(referenced.difference(indexed))
    if missing:
        raise ProtocolError(f"Snapshot content index misses referenced files: {missing}")
    if artifact_root is not None:
        _validate_content_files(Path(artifact_root), snapshot.content_index)


def _validate_partition(partition: PreparedPartition, *, label_role: str) -> None:
    if not partition.case_id_inventory or len(partition.case_id_inventory) != len(
        set(partition.case_id_inventory)
    ):
        raise ProtocolError("Prepared partition case inventory must be nonempty and unique.")
    bindings = (
        partition.features,
        partition.sample_ids,
        partition.case_ids,
        partition.class_labels,
    )
    for binding in bindings:
        _validate_relative_path(binding.relative_path)
        _validate_sha256(binding.sha256)
        _validate_sha256(binding.content_hash)
        if not binding.dtype or not binding.shape or min(binding.shape) <= 0:
            raise ProtocolError("Prepared array dtype/shape must be explicit and positive.")
    if partition.features.role not in {"source_fit_features", "diagnostic_eval_features"}:
        raise ProtocolError("Prepared feature role is invalid.")
    if partition.class_labels.role != label_role:
        raise ProtocolError("Prepared class-label role crossed its allowed boundary.")
    row_counts = {binding.shape[0] for binding in bindings}
    if len(row_counts) != 1:
        raise ProtocolError("Prepared partition arrays are not row-aligned.")
    for digest in (
        partition.row_inventory_hash,
        partition.sample_id_inventory_hash,
        partition.case_id_inventory_hash,
    ):
        _validate_sha256(digest)
    observed_rows = next(iter(row_counts))
    if (
        partition.row_count != observed_rows
        or partition.sample_count != observed_rows
        or partition.case_count != len(partition.case_id_inventory)
    ):
        raise ProtocolError("Prepared partition row/sample/case counts do not reconcile.")


def _validate_prepared_centers(
    prepared_centers: tuple[CenterPreparedData, ...],
) -> dict[str, CenterPreparedData]:
    if tuple(item.center for item in prepared_centers) != AUDIT_CENTERS:
        raise ProtocolError(
            "Snapshot prepared centers must be exactly 2,5,6,9 in canonical order."
        )
    by_center = {item.center: item for item in prepared_centers}
    for center, prepared in by_center.items():
        if prepared.prepared_bundle.role != "prepared_center_bundle":
            raise ProtocolError("Center prepared bundle role is invalid.")
        _validate_content_index((prepared.prepared_bundle,))
        _validate_partition(prepared.fit, label_role=FIT_LABEL_ROLE)
        _validate_partition(prepared.evaluation, label_role=EVAL_LABEL_ROLE)
        if set(prepared.fit.case_id_inventory).intersection(
            prepared.evaluation.case_id_inventory
        ):
            raise ProtocolError(
                f"Prepared source-fit and evaluation cases overlap for center {center}."
            )
    return by_center


def _validate_content_index(entries: tuple[ContentEntry, ...]) -> None:
    paths = [entry.relative_path for entry in entries]
    if not paths or len(paths) != len(set(paths)):
        raise ProtocolError("Content index paths must be nonempty and unique.")
    for entry in entries:
        _validate_relative_path(entry.relative_path)
        _validate_sha256(entry.sha256)
        _validate_sha256(entry.content_hash)
        if entry.size_bytes < 0 or not entry.role:
            raise ProtocolError("Content-index size and role must be explicit.")


def _validate_content_files(root: Path, entries: tuple[ContentEntry, ...]) -> None:
    for entry in entries:
        path = root / entry.relative_path
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        except OSError as exc:
            raise ProtocolError(f"Cannot read indexed audit content: {path}") from exc
        if path.stat().st_size != entry.size_bytes or digest.hexdigest() != entry.sha256:
            raise ProtocolError(f"Indexed audit content byte identity mismatch: {path}")


def _replace_hashes(
    value: AuditSnapshot,
    *,
    manifest_hash: str,
    inventory_hash: str,
    content_hash: str,
    snapshot_hash: str,
) -> AuditSnapshot:
    payload = {
        field: getattr(value, field)
        for field in AuditSnapshot.__dataclass_fields__
    }
    payload.update(
        manifest_hash=manifest_hash,
        key_inventory_hash=inventory_hash,
        content_index_hash=content_hash,
        snapshot_hash=snapshot_hash,
    )
    return AuditSnapshot(**payload)


def _partition_from_mapping(payload: Mapping[str, object]) -> PreparedPartition:
    return PreparedPartition(
        features=_array_binding_from_mapping(_mapping(payload, "features")),
        sample_ids=_array_binding_from_mapping(_mapping(payload, "sample_ids")),
        case_ids=_array_binding_from_mapping(_mapping(payload, "case_ids")),
        class_labels=_array_binding_from_mapping(_mapping(payload, "class_labels")),
        case_id_inventory=tuple(
            str(value) for value in payload.get("case_id_inventory", ())
        ),
        row_inventory_hash=str(payload.get("row_inventory_hash", "")),
        sample_id_inventory_hash=str(payload.get("sample_id_inventory_hash", "")),
        case_id_inventory_hash=str(payload.get("case_id_inventory_hash", "")),
        row_count=int(payload.get("row_count", -1)),
        sample_count=int(payload.get("sample_count", -1)),
        case_count=int(payload.get("case_count", -1)),
    )


def _center_prepared_from_mapping(payload: Mapping[str, object]) -> CenterPreparedData:
    return CenterPreparedData(
        center=str(payload.get("center", "")),
        prepared_bundle=_content_entry_from_mapping(
            _mapping(payload, "prepared_bundle")
        ),
        fit=_partition_from_mapping(_mapping(payload, "fit")),
        evaluation=_partition_from_mapping(_mapping(payload, "evaluation")),
    )


def _array_binding_from_mapping(payload: Mapping[str, object]) -> ArrayBinding:
    return ArrayBinding(
        relative_path=str(payload.get("relative_path", "")),
        sha256=str(payload.get("sha256", "")),
        content_hash=str(payload.get("content_hash", "")),
        dtype=str(payload.get("dtype", "")),
        shape=tuple(int(value) for value in payload.get("shape", ())),
        role=str(payload.get("role", "")),
    )


def _content_entry_from_mapping(payload: Mapping[str, object]) -> ContentEntry:
    return ContentEntry(
        relative_path=str(payload.get("relative_path", "")),
        sha256=str(payload.get("sha256", "")),
        content_hash=str(payload.get("content_hash", "")),
        size_bytes=int(payload.get("size_bytes", -1)),
        role=str(payload.get("role", "")),
    )


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Audit snapshot requires mapping section {key!r}.")
    return value


def _as_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProtocolError("Audit snapshot list entry must be a mapping.")
    return value


def _as_list(value: object, key: str) -> list[object]:
    if not isinstance(value, list):
        raise ProtocolError(f"Audit snapshot field {key!r} must be a list.")
    return value


def _strict_bool(value: object, key: str) -> bool:
    if not isinstance(value, bool):
        raise ProtocolError(f"Audit snapshot field {key!r} must be boolean.")
    return value


def _validate_relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ProtocolError("Snapshot references must be safe artifact-relative paths.")


def _validate_sha256(value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ProtocolError("Snapshot bindings require lowercase full SHA-256 digests.")


def _validate_semantic_hash(value: str) -> None:
    if len(value) != 16 or any(character not in "0123456789abcdef" for character in value):
        raise ProtocolError("Snapshot semantic bindings require canonical 16-hex hashes.")
