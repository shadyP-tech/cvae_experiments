"""Strict label-blind loader for the newly reserved MIDOG++ target cache."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...generation.contracts import COMMON_OUTPUT_DIM
from ...protocol import ProtocolError
from ...routing.harp_protocol.hashing import canonical_hash
from ...runtime.harp_probability_menu.hashing import require_sha256
from .config import (
    POLICY_ARTIFACT_ID,
    RESERVATION_ARTIFACT_ID,
    SCORING_MANIFEST_ARTIFACT_ID,
    TARGET_CACHE_ARTIFACT_ID,
)
from .contracts import (
    HarpFreshReservation,
    HarpFreshTargetCache,
    HarpFreshTargetFrame,
)
from .workspace_binding import HarpFreshWorkspaceBinding


RESERVATION_MEMBER = "manifests/reservation.json"
CACHE_PROTOCOL_MEMBER = "manifests/cache_index.json"
CONTENT_INDEX_MEMBER = "manifests/content_index.json"
ROW_INDEX_MEMBER = "tables/row_index.csv"
ROW_COLUMNS = (
    "schema_version",
    "row_id",
    "center",
    "case_id",
    "center_row_index",
    "embedding_file",
)
ROW_SCHEMA = "midogpp_harp_fresh_target_row_v1"


def _json(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Fresh HARP input is unreadable: {path}.") from exc
    if not isinstance(raw, dict):
        raise ProtocolError("Fresh HARP JSON input must be an object.")
    return raw


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ProtocolError(f"Cannot hash fresh HARP input: {path}.") from exc
    return digest.hexdigest()


def _require_payload_hash(
    raw: Mapping[str, object], member: str, *, role: str
) -> str:
    observed = raw.get(member)
    if observed != canonical_hash({key: value for key, value in raw.items() if key != member}):
        raise ProtocolError(f"Fresh HARP {role} hash drifted.")
    return require_sha256(observed, name=f"{role} hash")


def _case_map(raw: object, *, role: str) -> dict[str, tuple[str, ...]]:
    if not isinstance(raw, Mapping) or tuple(str(key) for key in raw) != CENTERS:
        raise ProtocolError(f"Fresh HARP {role} cases must cover centers in order.")
    output: dict[str, tuple[str, ...]] = {}
    for center in CENTERS:
        values = raw[center]
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ProtocolError(f"Fresh HARP {role} case inventory is malformed.")
        cases = tuple(str(value) for value in values)
        if not cases or cases != tuple(dict.fromkeys(cases)) or any(
            not case or case.strip() != case for case in cases
        ):
            raise ProtocolError(f"Fresh HARP {role} case inventory is malformed.")
        output[center] = cases
    return output


def _load_reservation(path: Path) -> tuple[HarpFreshReservation, str]:
    raw = _json(path)
    reservation_hash = _require_payload_hash(raw, "reservation_hash", role="reservation")
    if (
        raw.get("schema_version") != "midogpp_harp_fresh_target_reservation_v1"
        or raw.get("artifact_id") != RESERVATION_ARTIFACT_ID
        or raw.get("dataset_family") != "MIDOG++"
        or raw.get("status") != "ACTIVE"
        or raw.get("fresh_unconsumed_surface") is not True
        or raw.get("support_evaluation_case_disjoint") is not True
        or raw.get("labels_opened") is not False
        or raw.get("previously_evaluated") is not False
        or raw.get("consumed_test_used") is not False
        or raw.get("consumed_validation_used") is not False
        or raw.get("consumed_stage90_used") is not False
        or raw.get("scoring_manifest_artifact_id") != SCORING_MANIFEST_ARTIFACT_ID
    ):
        raise ProtocolError("Fresh HARP reservation is not active, unopened, and fresh.")
    support = _case_map(raw.get("support_case_ids_by_center"), role="support")
    evaluation = _case_map(raw.get("evaluation_case_ids_by_center"), role="evaluation")
    reservation = HarpFreshReservation(
        reservation_id=str(raw.get("reservation_id", "")),
        support_case_ids_by_center=support,
        evaluation_case_ids_by_center=evaluation,
        upstream_reservation_hash=reservation_hash,
    )
    scoring_sha = require_sha256(
        raw.get("scoring_manifest_sha256"), name="scoring-manifest hash"
    )
    return reservation, scoring_sha


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != ROW_COLUMNS:
                raise ProtocolError("Fresh HARP target row-index columns drifted.")
            rows = tuple(dict(row) for row in reader)
    except OSError as exc:
        raise ProtocolError("Cannot read fresh HARP target row index.") from exc
    if not rows:
        raise ProtocolError("Fresh HARP target row index is empty.")
    return rows


@dataclass(frozen=True, kw_only=True)
class HarpFreshLoadedTarget:
    cache: HarpFreshTargetCache
    cache_content_hash: str
    cache_protocol_hash: str
    policy_lock_hash: str
    scoring_manifest_path: Path
    scoring_manifest_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.cache, HarpFreshTargetCache):
            raise ProtocolError("Fresh HARP loaded target lacks its typed cache.")
        for name in (
            "cache_content_hash",
            "cache_protocol_hash",
            "policy_lock_hash",
            "scoring_manifest_sha256",
        ):
            object.__setattr__(
                self, name, require_sha256(getattr(self, name), name=name)
            )
        object.__setattr__(self, "scoring_manifest_path", Path(self.scoring_manifest_path))


def load_harp_fresh_target(
    binding: HarpFreshWorkspaceBinding,
) -> HarpFreshLoadedTarget:
    """Load embeddings/identities only; the scoring CSV remains unopened."""

    if not isinstance(binding, HarpFreshWorkspaceBinding):
        raise ProtocolError("Fresh HARP target loading requires workspace binding.")
    reservation, scoring_sha = _load_reservation(
        binding.reservation_root / RESERVATION_MEMBER
    )
    # Do not read or hash the scoring manifest here: it contains target labels.
    # Its reserved SHA-256 is carried unopened until the post-seal capability.
    content = _json(binding.target_cache_root / CONTENT_INDEX_MEMBER)
    content_hash = _require_payload_hash(content, "content_hash", role="target content")
    if (
        content.get("schema_version") != "midogpp_harp_fresh_target_content_v1"
        or content.get("artifact_id") != TARGET_CACHE_ARTIFACT_ID
        or content.get("status") != "COMPLETE"
        or content.get("labels_persisted") is not False
    ):
        raise ProtocolError("Fresh HARP target content contract drifted.")
    expected_members = {
        ROW_INDEX_MEMBER,
        *(f"embeddings/by_center/center_{center}.npy" for center in CENTERS),
    }
    files = content.get("files")
    if not isinstance(files, list):
        raise ProtocolError("Fresh HARP target content inventory is absent.")
    observed_members: dict[str, str] = {}
    for item in files:
        if not isinstance(item, Mapping):
            raise ProtocolError("Fresh HARP target content row is malformed.")
        member = str(item.get("path", ""))
        digest = require_sha256(item.get("sha256"), name="target member hash")
        if member in observed_members:
            raise ProtocolError("Fresh HARP target content contains duplicates.")
        observed_members[member] = digest
    if set(observed_members) != expected_members:
        raise ProtocolError("Fresh HARP target content is not closed-world.")
    for member, digest in observed_members.items():
        path = (binding.target_cache_root / member).resolve()
        try:
            path.relative_to(binding.target_cache_root)
        except ValueError as exc:
            raise ProtocolError("Fresh HARP target member escaped cache root.") from exc
        if not path.is_file() or path.is_symlink() or _sha256_file(path) != digest:
            raise ProtocolError(f"Fresh HARP target member drifted: {member}.")

    protocol = _json(binding.target_cache_root / CACHE_PROTOCOL_MEMBER)
    protocol_hash = _require_payload_hash(
        protocol, "cache_index_hash", role="target-cache protocol"
    )
    policy_hash = require_sha256(protocol.get("policy_lock_hash"), name="cache policy lock")
    if (
        protocol.get("schema_version") != "midogpp_harp_fresh_target_cache_index_v1"
        or protocol.get("artifact_id") != TARGET_CACHE_ARTIFACT_ID
        or protocol.get("status") != "COMPLETE"
        or protocol.get("dataset_family") != "MIDOG++"
        or protocol.get("feature_dim") != COMMON_OUTPUT_DIM
        or protocol.get("reservation_artifact_id") != RESERVATION_ARTIFACT_ID
        or protocol.get("reservation_hash") != reservation.reservation_hash
        or protocol.get("policy_artifact_id") != POLICY_ARTIFACT_ID
        or protocol.get("policy_frozen_before_cache_extraction") is not True
        or protocol.get("cache_content_hash") != content_hash
        or protocol.get("labels_persisted") is not False
        or protocol.get("fresh_unconsumed_surface") is not True
        or protocol.get("consumed_test_used") is not False
        or protocol.get("consumed_validation_used") is not False
        or protocol.get("consumed_stage90_used") is not False
    ):
        raise ProtocolError("Fresh HARP target-cache protocol drifted.")

    rows = _read_rows(binding.target_cache_root / ROW_INDEX_MEMBER)
    frames: dict[str, HarpFreshTargetFrame] = {}
    seen_rows: set[str] = set()
    for center in CENTERS:
        center_rows = tuple(row for row in rows if row.get("center") == center)
        expected_member = f"embeddings/by_center/center_{center}.npy"
        if not center_rows:
            raise ProtocolError("Fresh HARP target center has no rows.")
        row_ids: list[str] = []
        case_ids: list[str] = []
        for ordinal, row in enumerate(center_rows):
            if (
                row.get("schema_version") != ROW_SCHEMA
                or row.get("embedding_file") != expected_member
                or row.get("center_row_index") != str(ordinal)
            ):
                raise ProtocolError("Fresh HARP target row identity drifted.")
            row_id = str(row.get("row_id", ""))
            case_id = str(row.get("case_id", ""))
            if not row_id or row_id in seen_rows or not case_id:
                raise ProtocolError("Fresh HARP target row/case identity is invalid.")
            seen_rows.add(row_id)
            row_ids.append(row_id)
            case_ids.append(case_id)
        embeddings = np.load(
            binding.target_cache_root / expected_member,
            mmap_mode="r",
            allow_pickle=False,
        )
        frames[center] = HarpFreshTargetFrame(
            center=center,
            embeddings=np.asarray(embeddings),
            row_ids=tuple(row_ids),
            case_ids=tuple(case_ids),
        )
    cache = HarpFreshTargetCache(
        reservation=reservation,
        frames_by_center=MappingProxyType(frames),
    )
    return HarpFreshLoadedTarget(
        cache=cache,
        cache_content_hash=content_hash,
        cache_protocol_hash=protocol_hash,
        policy_lock_hash=policy_hash,
        scoring_manifest_path=binding.scoring_manifest_path,
        scoring_manifest_sha256=scoring_sha,
    )


__all__ = (
    "CACHE_PROTOCOL_MEMBER",
    "CONTENT_INDEX_MEMBER",
    "HarpFreshLoadedTarget",
    "RESERVATION_MEMBER",
    "ROW_COLUMNS",
    "ROW_INDEX_MEMBER",
    "ROW_SCHEMA",
    "load_harp_fresh_target",
)
