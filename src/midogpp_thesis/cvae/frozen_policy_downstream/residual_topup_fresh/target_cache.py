"""Fresh target-surface admission with a hard pre-seal label boundary."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import cached_property
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ...generation.contracts import COMMON_OUTPUT_DIM
from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import canonical_sha256
from .config import (
    POLICY_ARTIFACT_ID,
    RESERVATION_ARTIFACT_ID,
    SCORING_MANIFEST_ARTIFACT_ID,
    TARGET_CACHE_ARTIFACT_ID,
    ResidualTopupFreshConfig,
)
from .contracts import CENTERS


RESERVATION_SCHEMA = "midogpp_residual_topup_fresh_target_reservation_v1"
CACHE_PROTOCOL_SCHEMA = "midogpp_residual_topup_fresh_target_cache_protocol_v1"
CACHE_CONTENT_SCHEMA = "midogpp_residual_topup_fresh_target_cache_content_v1"
ROW_SCHEMA = "midogpp_residual_topup_fresh_target_row_v1"

CACHE_PROTOCOL_MEMBER = "manifests/cache_protocol.json"
CACHE_CONTENT_MEMBER = "manifests/content_index.json"
ROW_INDEX_MEMBER = "tables/row_index.csv"
ROW_COLUMNS = (
    "schema_version",
    "row_id",
    "center",
    "case_id",
    "center_row_index",
    "embedding_file",
)
_FORBIDDEN_CACHE_COLUMN_TOKENS = (
    "label",
    "class",
    "diagnosis",
    "target",
    "y_true",
)


@dataclass(frozen=True)
class FreshReservation:
    reservation_id: str
    reservation_hash: str
    support_case_ids_by_center: Mapping[str, tuple[str, ...]]
    evaluation_case_ids_by_center: Mapping[str, tuple[str, ...]]
    scoring_manifest_sha256: str
    payload: Mapping[str, object]


@dataclass(frozen=True)
class FreshTargetFrame:
    center: str
    embeddings: np.ndarray
    evaluation_row_ids: tuple[str, ...]
    case_ids: tuple[str, ...]
    center_row_indices: tuple[int, ...]
    file_sha256: str

    def __post_init__(self) -> None:
        embeddings = np.asarray(self.embeddings)
        if (
            self.center not in CENTERS
            or embeddings.dtype != np.float32
            or embeddings.ndim != 2
            or embeddings.shape[1] != COMMON_OUTPUT_DIM
            or embeddings.shape[0] != len(self.evaluation_row_ids)
            or embeddings.shape[0] != len(self.case_ids)
            or self.center_row_indices != tuple(range(embeddings.shape[0]))
            or not np.isfinite(embeddings).all()
        ):
            raise ProtocolError("Fresh target frame geometry drifted.")
        if (
            len(self.evaluation_row_ids) != len(set(self.evaluation_row_ids))
            or any(not value or value.strip() != value for value in self.evaluation_row_ids)
            or any(not value or value.strip() != value for value in self.case_ids)
        ):
            raise ProtocolError("Fresh target row identities are invalid.")
        embeddings.setflags(write=False)


@dataclass(frozen=True)
class FreshTargetSurface:
    reservation: FreshReservation
    frames_by_center: Mapping[str, FreshTargetFrame]
    row_index_rows: tuple[Mapping[str, object], ...]
    cache_content_hash: str
    cache_protocol_hash: str
    scoring_manifest_path: Path
    scoring_manifest_sha256: str
    labels_opened: bool = False

    def __post_init__(self) -> None:
        if set(self.frames_by_center) != set(CENTERS) or self.labels_opened:
            raise ProtocolError("Fresh target surface coverage/label seal drifted.")

    @cached_property
    def evaluation_row_ids_by_target(self) -> Mapping[str, tuple[str, ...]]:
        return MappingProxyType(
            {
                center: self.frames_by_center[center].evaluation_row_ids
                for center in CENTERS
            }
        )


def require_fresh_target_artifacts(config: ResidualTopupFreshConfig) -> None:
    """Fail before any fallback can repoint the planned fresh identities."""

    expected = [
        config.fresh_reservation_path,
        config.fresh_scoring_manifest_path,
        config.fresh_target_cache_root / CACHE_PROTOCOL_MEMBER,
        config.fresh_target_cache_root / CACHE_CONTENT_MEMBER,
        config.fresh_target_cache_root / ROW_INDEX_MEMBER,
        *(
            config.fresh_target_cache_root
            / f"embeddings/by_center/center_{center}.npy"
            for center in CENTERS
        ),
    ]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise ProtocolError(
            "Fresh Stage-70 is blocked: planned reservation/cache/scoring "
            f"artifacts are absent ({missing}). Consumed Stage-70/90 data "
            "cannot substitute."
        )


def load_fresh_target_surface(
    config: ResidualTopupFreshConfig,
) -> FreshTargetSurface:
    """Load only label-blind arrays and identities before prediction sealing."""

    require_fresh_target_artifacts(config)
    reservation = _load_reservation(config.fresh_reservation_path)
    policy_lock_hash = _validate_policy_reservation_binding(
        config.policy_root,
        reservation=reservation,
    )
    scoring_sha = _sha256_file(config.fresh_scoring_manifest_path)
    if scoring_sha != reservation.scoring_manifest_sha256:
        raise ProtocolError("Fresh scoring manifest drifted from its reservation.")

    content = _json(config.fresh_target_cache_root / CACHE_CONTENT_MEMBER)
    _require_hash(content, "content_hash", role="target-cache content")
    if (
        content.get("schema_version") != CACHE_CONTENT_SCHEMA
        or content.get("artifact_id") != TARGET_CACHE_ARTIFACT_ID
        or content.get("status") != "COMPLETE"
        or content.get("labels_persisted") is not False
    ):
        raise ProtocolError("Fresh target-cache content identity drifted.")
    files = content.get("files")
    if not isinstance(files, list):
        raise ProtocolError("Fresh target-cache content inventory is absent.")
    expected_members = {
        ROW_INDEX_MEMBER,
        *(f"embeddings/by_center/center_{center}.npy" for center in CENTERS),
    }
    observed_members: dict[str, str] = {}
    for raw in files:
        if not isinstance(raw, Mapping):
            raise ProtocolError("Fresh target-cache content row is malformed.")
        member = str(raw.get("path", ""))
        digest = str(raw.get("sha256", ""))
        if member in observed_members or not _is_sha256(digest):
            raise ProtocolError("Fresh target-cache content inventory is malformed.")
        observed_members[member] = digest
    if set(observed_members) != expected_members:
        raise ProtocolError("Fresh target-cache closed-world members drifted.")
    for member, digest in observed_members.items():
        path = _safe_member(config.fresh_target_cache_root, member)
        if not path.is_file() or _sha256_file(path) != digest:
            raise ProtocolError(f"Fresh target-cache member drifted: {member}.")

    protocol = _json(config.fresh_target_cache_root / CACHE_PROTOCOL_MEMBER)
    _require_hash(protocol, "cache_protocol_hash", role="target-cache protocol")
    required_protocol = {
        "schema_version": CACHE_PROTOCOL_SCHEMA,
        "artifact_id": TARGET_CACHE_ARTIFACT_ID,
        "status": "COMPLETE",
        "dataset_family": "MIDOG++",
        "representation_id": "annotation_jpeg_fixed_center_b_v3",
        "feature_backbone": "Virchow2",
        "feature_dim": COMMON_OUTPUT_DIM,
        "reservation_artifact_id": RESERVATION_ARTIFACT_ID,
        "reservation_hash": reservation.reservation_hash,
        "policy_artifact_id": POLICY_ARTIFACT_ID,
        "policy_lock_hash": policy_lock_hash,
        "policy_lock_frozen_before_target_cache_extraction": True,
        "scoring_manifest_artifact_id": SCORING_MANIFEST_ARTIFACT_ID,
        "scoring_manifest_sha256": scoring_sha,
        "cache_content_hash": content["content_hash"],
        "labels_persisted": False,
        "fresh_unconsumed_surface": True,
        "consumed_test_used": False,
        "consumed_validation_used": False,
        "consumed_stage90_used": False,
        "reservation_frozen_before_cache_extraction": True,
    }
    _require_values(protocol, required_protocol, "target-cache protocol")

    rows = _read_row_index(config.fresh_target_cache_root / ROW_INDEX_MEMBER)
    expected_row_hash = stable_hash([dict(row) for row in rows])
    if protocol.get("row_identity_hash") != expected_row_hash:
        raise ProtocolError("Fresh target-cache row identity hash drifted.")
    frames: dict[str, FreshTargetFrame] = {}
    globally_seen_rows: set[str] = set()
    for center in CENTERS:
        center_rows = tuple(row for row in rows if row["center"] == center)
        if not center_rows:
            raise ProtocolError("Fresh target-cache center is empty.")
        evaluation_cases = {str(row["case_id"]) for row in center_rows}
        if evaluation_cases != set(
            reservation.evaluation_case_ids_by_center[center]
        ):
            raise ProtocolError("Fresh target-cache reservation case coverage drifted.")
        row_ids = tuple(str(row["row_id"]) for row in center_rows)
        if globally_seen_rows.intersection(row_ids):
            raise ProtocolError("Fresh target-cache rows overlap across centers.")
        globally_seen_rows.update(row_ids)
        member = f"embeddings/by_center/center_{center}.npy"
        if any(row["embedding_file"] != member for row in center_rows):
            raise ProtocolError("Fresh target-cache row/file binding drifted.")
        path = config.fresh_target_cache_root / member
        embeddings = np.load(path, mmap_mode="r", allow_pickle=False)
        frames[center] = FreshTargetFrame(
            center=center,
            embeddings=embeddings,
            evaluation_row_ids=row_ids,
            case_ids=tuple(str(row["case_id"]) for row in center_rows),
            center_row_indices=tuple(
                int(row["center_row_index"]) for row in center_rows
            ),
            file_sha256=observed_members[member],
        )
    return FreshTargetSurface(
        reservation=reservation,
        frames_by_center=MappingProxyType(frames),
        row_index_rows=rows,
        cache_content_hash=str(content["content_hash"]),
        cache_protocol_hash=str(protocol["cache_protocol_hash"]),
        scoring_manifest_path=config.fresh_scoring_manifest_path,
        scoring_manifest_sha256=scoring_sha,
    )


def _load_reservation(path: Path) -> FreshReservation:
    payload = _json(path)
    _require_hash(payload, "reservation_hash", role="fresh reservation")
    required = {
        "schema_version": RESERVATION_SCHEMA,
        "artifact_id": RESERVATION_ARTIFACT_ID,
        "status": "COMPLETE",
        "dataset_family": "MIDOG++",
        "centers": list(CENTERS),
        "split_role": "fresh_unconsumed_case_disjoint_target_evaluation",
        "reservation_frozen_before_cache_extraction": True,
        "fresh_unconsumed_surface": True,
        "consumed_test_used": False,
        "consumed_validation_used": False,
        "consumed_stage90_used": False,
        "support_evaluation_case_disjoint": True,
        "labels_opened": False,
        "scoring_manifest_artifact_id": SCORING_MANIFEST_ARTIFACT_ID,
    }
    _require_values(payload, required, "fresh reservation")
    support = _case_mapping(payload.get("support_case_ids_by_center"), role="support")
    evaluation = _case_mapping(
        payload.get("evaluation_case_ids_by_center"), role="evaluation"
    )
    support_cases = tuple(case for center in CENTERS for case in support[center])
    evaluation_cases = tuple(
        case for center in CENTERS for case in evaluation[center]
    )
    if (
        len(support_cases) != len(set(support_cases))
        or len(evaluation_cases) != len(set(evaluation_cases))
        or set(support_cases).intersection(evaluation_cases)
    ):
        raise ProtocolError(
            "Fresh support/evaluation cases must be globally unique and disjoint."
        )
    scoring_sha = str(payload.get("scoring_manifest_sha256", ""))
    reservation_id = str(payload.get("reservation_id", ""))
    if not reservation_id or reservation_id.strip() != reservation_id:
        raise ProtocolError("Fresh reservation identity is invalid.")
    if not _is_sha256(scoring_sha):
        raise ProtocolError("Fresh reservation scoring-manifest hash is invalid.")
    return FreshReservation(
        reservation_id=reservation_id,
        reservation_hash=str(payload["reservation_hash"]),
        support_case_ids_by_center=support,
        evaluation_case_ids_by_center=evaluation,
        scoring_manifest_sha256=scoring_sha,
        payload=MappingProxyType(dict(payload)),
    )


def _validate_policy_reservation_binding(
    policy_root: Path,
    *,
    reservation: FreshReservation,
) -> str:
    """Bind cache admission to the already-frozen Stage-60 reservation."""

    policy = _json(policy_root / "manifests/policy_lock.json")
    observed_hash = policy.get("policy_lock_hash")
    unhashed = {
        key: value for key, value in policy.items() if key != "policy_lock_hash"
    }
    support = policy.get("support_case_ids_by_target")
    evaluation = policy.get("evaluation_case_ids_by_target")
    if (
        observed_hash != canonical_sha256(unhashed)
        or policy.get("fresh_surface_reservation_id") != reservation.reservation_id
        or policy.get("policy_frozen_before_stage70") is not True
        or not isinstance(support, Mapping)
        or not isinstance(evaluation, Mapping)
    ):
        raise ProtocolError("Fresh target-cache Stage-60 policy binding drifted.")
    normalized_support = _case_mapping(support, role="policy support")
    normalized_evaluation = _case_mapping(evaluation, role="policy evaluation")
    if (
        dict(normalized_support)
        != dict(reservation.support_case_ids_by_center)
        or dict(normalized_evaluation)
        != dict(reservation.evaluation_case_ids_by_center)
    ):
        raise ProtocolError(
            "Fresh Stage-70 reservation case grids drifted from the Stage-60 lock."
        )
    state = _json(policy_root / "reports/run_state.json")
    validation = _json(policy_root / "reports/validation_report.json")
    checks = validation.get("checks")
    if (
        state.get("status") != "COMPLETE"
        or validation.get("status") != "PASS"
        or validation.get("validator")
        != "validate_residual_topup_policy_bundle"
        or not isinstance(checks, Mapping)
        or checks.get("status") != "PASS"
        or checks.get("policy_lock_hash") != observed_hash
        or checks.get("labels_consumed") is not False
        or checks.get("target_evaluation_used") is not False
        or checks.get("source_experts_updated") is not False
    ):
        raise ProtocolError(
            "Fresh target-cache policy lock lacks independent PASS authorization."
        )
    return str(observed_hash)


def _case_mapping(value: object, *, role: str) -> Mapping[str, tuple[str, ...]]:
    if not isinstance(value, Mapping) or {str(key) for key in value} != set(CENTERS):
        raise ProtocolError(f"Fresh reservation {role} cases must cover all centers.")
    normalized = {str(key): raw for key, raw in value.items()}
    output: dict[str, tuple[str, ...]] = {}
    for center in CENTERS:
        raw = normalized[center]
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ProtocolError(f"Fresh reservation {role} cases are malformed.")
        cases = tuple(str(item) for item in raw)
        if (
            not cases
            or len(cases) != len(set(cases))
            or any(not case or case.strip() != case for case in cases)
        ):
            raise ProtocolError(f"Fresh reservation {role} cases are malformed.")
        output[center] = cases
    return MappingProxyType(output)


def _read_row_index(path: Path) -> tuple[Mapping[str, object], ...]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            columns = tuple(reader.fieldnames or ())
            if columns != ROW_COLUMNS or any(
                token in column.lower()
                for column in columns
                for token in _FORBIDDEN_CACHE_COLUMN_TOKENS
            ):
                raise ProtocolError("Fresh target-cache row index is not label blind.")
            raw_rows = tuple(dict(row) for row in reader)
    except OSError as exc:
        raise ProtocolError("Cannot read fresh target-cache row index.") from exc
    rows: list[Mapping[str, object]] = []
    seen: set[str] = set()
    previous_key: tuple[int, int] | None = None
    center_order = {center: index for index, center in enumerate(CENTERS)}
    for raw in raw_rows:
        center = str(raw.get("center", ""))
        row_id = str(raw.get("row_id", ""))
        case_id = str(raw.get("case_id", ""))
        try:
            center_index = int(raw.get("center_row_index", ""))
        except (TypeError, ValueError) as exc:
            raise ProtocolError("Fresh target-cache row index is malformed.") from exc
        if (
            raw.get("schema_version") != ROW_SCHEMA
            or center not in CENTERS
            or not row_id
            or row_id in seen
            or not case_id
            or center_index < 0
        ):
            raise ProtocolError("Fresh target-cache row index is malformed.")
        key = (center_order[center], center_index)
        if previous_key is not None and key <= previous_key:
            raise ProtocolError("Fresh target-cache row order is not canonical.")
        previous_key = key
        seen.add(row_id)
        rows.append(
            MappingProxyType(
                {
                    "schema_version": ROW_SCHEMA,
                    "row_id": row_id,
                    "center": center,
                    "case_id": case_id,
                    "center_row_index": center_index,
                    "embedding_file": str(raw.get("embedding_file", "")),
                }
            )
        )
    return tuple(rows)


def _json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read fresh Stage-70 JSON: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Fresh Stage-70 JSON must be a mapping: {path}.")
    return payload


def _require_hash(payload: Mapping[str, object], key: str, *, role: str) -> None:
    observed = payload.get(key)
    unhashed = {name: value for name, value in payload.items() if name != key}
    if observed != stable_hash(unhashed):
        raise ProtocolError(f"Fresh {role} hash drifted.")


def _require_values(
    payload: Mapping[str, object], expected: Mapping[str, object], role: str
) -> None:
    mismatch = [key for key, value in expected.items() if payload.get(key) != value]
    if mismatch:
        raise ProtocolError(f"Fresh {role} drifted: {mismatch}.")


def _safe_member(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ProtocolError("Fresh target-cache member escapes its artifact root.")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


__all__ = (
    "CACHE_CONTENT_MEMBER",
    "CACHE_CONTENT_SCHEMA",
    "CACHE_PROTOCOL_MEMBER",
    "CACHE_PROTOCOL_SCHEMA",
    "FreshReservation",
    "FreshTargetFrame",
    "FreshTargetSurface",
    "ROW_COLUMNS",
    "ROW_INDEX_MEMBER",
    "ROW_SCHEMA",
    "load_fresh_target_surface",
    "require_fresh_target_artifacts",
)
