"""Active fresh target admission with a hard pre-seal label boundary."""

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
from .config import (
    EXPERIMENT_ID,
    RESERVATION_ARTIFACT_ID,
    SCORING_MANIFEST_ARTIFACT_ID,
    TARGET_CACHE_ARTIFACT_ID,
    UtilityAlignedResidualFreshConfig,
)
from .contracts import CENTERS
from .policy_loading import POLICY_EXPERIMENT_ID, FrozenUtilityAlignedPolicySurface


RESERVATION_SCHEMA = "midogpp_utility_aligned_fresh_target_reservation_v1"
CACHE_PROTOCOL_SCHEMA = "midogpp_utility_aligned_fresh_target_cache_protocol_v1"
CACHE_CONTENT_SCHEMA = "midogpp_utility_aligned_fresh_target_cache_content_v1"
ROW_SCHEMA = "midogpp_utility_aligned_fresh_target_row_v1"
ROW_COLUMNS = (
    "schema_version",
    "row_id",
    "center",
    "case_id",
    "center_row_index",
    "embedding_file",
)
CACHE_PROTOCOL_MEMBER = "manifests/cache_protocol.json"
CACHE_CONTENT_MEMBER = "manifests/content_index.json"
ROW_INDEX_MEMBER = "tables/row_index.csv"
AUTHORIZED_CONSUMER_EXPERIMENT_IDS = (POLICY_EXPERIMENT_ID, EXPERIMENT_ID)


@dataclass(frozen=True)
class FreshReservation:
    reservation_id: str
    reservation_hash: str
    target_evaluation_binding_hash: str
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
    file_sha256: str

    def __post_init__(self) -> None:
        values = np.asarray(self.embeddings)
        if (
            self.center not in CENTERS
            or values.dtype != np.float32
            or values.ndim != 2
            or values.shape != (len(self.evaluation_row_ids), COMMON_OUTPUT_DIM)
            or len(self.case_ids) != len(self.evaluation_row_ids)
            or not np.isfinite(values).all()
        ):
            raise ProtocolError("Utility-aligned fresh target frame drifted.")
        values.setflags(write=False)


@dataclass(frozen=True)
class FreshTargetSurface:
    reservation: FreshReservation
    frames_by_center: Mapping[str, FreshTargetFrame]
    cache_content_hash: str
    cache_protocol_hash: str
    scoring_manifest_path: Path
    scoring_manifest_sha256: str
    labels_opened: bool = False

    def __post_init__(self) -> None:
        if set(self.frames_by_center) != set(CENTERS) or self.labels_opened:
            raise ProtocolError("Utility-aligned fresh target coverage drifted.")

    @cached_property
    def evaluation_row_ids_by_target(self) -> Mapping[str, tuple[str, ...]]:
        return MappingProxyType(
            {
                center: self.frames_by_center[center].evaluation_row_ids
                for center in CENTERS
            }
        )


def require_active_fresh_target_artifacts(
    config: UtilityAlignedResidualFreshConfig,
) -> None:
    """Reject absent, inactive, or consumed reservation inputs before GPU work."""

    expected = (
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
    )
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise ProtocolError(
            "Utility-aligned Stage-70 is blocked before runtime admission: "
            f"fresh inputs are absent ({missing})."
        )
    reservation = _json(config.fresh_reservation_path)
    if (
        reservation.get("status") != "ACTIVE"
        or reservation.get("authorized_consumer_experiment_ids")
        != list(AUTHORIZED_CONSUMER_EXPERIMENT_IDS)
        or reservation.get("fresh_unconsumed_surface") is not True
        or reservation.get("labels_opened") is not False
        or any(
            reservation.get(key) is not False
            for key in (
                "consumed_test_used",
                "consumed_validation_used",
                "consumed_stage70_used",
                "consumed_stage90_used",
            )
        )
    ):
        raise ProtocolError(
            "Utility-aligned Stage-70 requires an active unconsumed reservation; "
            "consumed Stage-70/90 inputs cannot substitute."
        )


def load_fresh_target_surface(
    config: UtilityAlignedResidualFreshConfig,
    policy: FrozenUtilityAlignedPolicySurface,
) -> FreshTargetSurface:
    require_active_fresh_target_artifacts(config)
    reservation = _load_reservation(config.fresh_reservation_path)
    policy_binding = str(policy.policy_payload.get("target_evaluation_binding_hash", ""))
    if (
        reservation.target_evaluation_binding_hash != policy_binding
        or reservation.reservation_hash != policy.reservation_hash
        or reservation.reservation_hash
        != str(policy.policy_payload.get("target_reservation_hash", ""))
        or policy.policy_payload.get("target_reservation_artifact_id")
        != RESERVATION_ARTIFACT_ID
        or dict(reservation.support_case_ids_by_center)
        != dict(policy.support_case_ids_by_target)
        or dict(reservation.evaluation_case_ids_by_center)
        != dict(policy.evaluation_case_ids_by_target)
    ):
        raise ProtocolError("Utility-aligned policy/reservation binding drifted.")
    scoring_sha = _sha256_file(config.fresh_scoring_manifest_path)
    if scoring_sha != reservation.scoring_manifest_sha256:
        raise ProtocolError("Utility-aligned scoring manifest drifted from reservation.")

    content = _json(config.fresh_target_cache_root / CACHE_CONTENT_MEMBER)
    _require_hash(content, "content_hash", "target-cache content")
    if (
        content.get("schema_version") != CACHE_CONTENT_SCHEMA
        or content.get("artifact_id") != TARGET_CACHE_ARTIFACT_ID
        or content.get("status") != "COMPLETE"
        or content.get("labels_persisted") is not False
    ):
        raise ProtocolError("Utility-aligned target-cache content drifted.")
    files = content.get("files")
    if not isinstance(files, Sequence) or isinstance(files, (str, bytes)):
        raise ProtocolError("Utility-aligned target-cache inventory is absent.")
    expected_members = {
        ROW_INDEX_MEMBER,
        *(f"embeddings/by_center/center_{center}.npy" for center in CENTERS),
    }
    observed: dict[str, str] = {}
    for raw in files:
        if not isinstance(raw, Mapping):
            raise ProtocolError("Utility-aligned target-cache member is malformed.")
        member = str(raw.get("path", ""))
        digest = str(raw.get("sha256", ""))
        if member in observed or not _is_sha256(digest):
            raise ProtocolError("Utility-aligned target-cache inventory is malformed.")
        path = _safe_member(config.fresh_target_cache_root, member)
        if not path.is_file() or _sha256_file(path) != digest:
            raise ProtocolError("Utility-aligned target-cache member hash drifted.")
        observed[member] = digest
    if set(observed) != expected_members:
        raise ProtocolError("Utility-aligned target-cache is not closed-world complete.")

    protocol = _json(config.fresh_target_cache_root / CACHE_PROTOCOL_MEMBER)
    _require_hash(protocol, "cache_protocol_hash", "target-cache protocol")
    expected_protocol = {
        "schema_version": CACHE_PROTOCOL_SCHEMA,
        "artifact_id": TARGET_CACHE_ARTIFACT_ID,
        "status": "COMPLETE",
        "dataset_family": "MIDOG++",
        "feature_backbone": "Virchow2",
        "feature_dim": COMMON_OUTPUT_DIM,
        "reservation_artifact_id": RESERVATION_ARTIFACT_ID,
        "reservation_hash": reservation.reservation_hash,
        "target_evaluation_binding_hash": reservation.target_evaluation_binding_hash,
        "policy_lock_hash": policy.policy_lock_hash,
        "scoring_manifest_artifact_id": SCORING_MANIFEST_ARTIFACT_ID,
        "scoring_manifest_sha256": scoring_sha,
        "cache_content_hash": content["content_hash"],
        "fresh_unconsumed_surface": True,
        "labels_persisted": False,
        "consumed_test_used": False,
        "consumed_validation_used": False,
        "consumed_stage70_used": False,
        "consumed_stage90_used": False,
    }
    mismatch = [key for key, value in expected_protocol.items() if protocol.get(key) != value]
    if mismatch:
        raise ProtocolError(f"Utility-aligned target-cache protocol drifted: {mismatch}.")

    rows = _read_row_index(config.fresh_target_cache_root / ROW_INDEX_MEMBER)
    if protocol.get("row_identity_hash") != stable_hash([dict(row) for row in rows]):
        raise ProtocolError("Utility-aligned target row identity hash drifted.")
    frames: dict[str, FreshTargetFrame] = {}
    all_rows: set[str] = set()
    for center in CENTERS:
        selected = tuple(row for row in rows if row["center"] == center)
        row_ids = tuple(str(row["row_id"]) for row in selected)
        cases = tuple(str(row["case_id"]) for row in selected)
        if (
            not selected
            or set(cases) != set(reservation.evaluation_case_ids_by_center[center])
            or all_rows.intersection(row_ids)
        ):
            raise ProtocolError("Utility-aligned target reservation coverage drifted.")
        all_rows.update(row_ids)
        member = f"embeddings/by_center/center_{center}.npy"
        if any(row["embedding_file"] != member for row in selected):
            raise ProtocolError("Utility-aligned target row/file binding drifted.")
        array = np.load(
            config.fresh_target_cache_root / member,
            mmap_mode="r",
            allow_pickle=False,
        )
        frames[center] = FreshTargetFrame(
            center=center,
            embeddings=array,
            evaluation_row_ids=row_ids,
            case_ids=cases,
            file_sha256=observed[member],
        )
    return FreshTargetSurface(
        reservation=reservation,
        frames_by_center=MappingProxyType(frames),
        cache_content_hash=str(content["content_hash"]),
        cache_protocol_hash=str(protocol["cache_protocol_hash"]),
        scoring_manifest_path=config.fresh_scoring_manifest_path,
        scoring_manifest_sha256=scoring_sha,
    )


def _load_reservation(path: Path) -> FreshReservation:
    payload = _json(path)
    _require_hash(payload, "reservation_hash", "fresh reservation")
    required = {
        "schema_version": RESERVATION_SCHEMA,
        "artifact_id": RESERVATION_ARTIFACT_ID,
        "status": "ACTIVE",
        "authorized_consumer_experiment_ids": list(
            AUTHORIZED_CONSUMER_EXPERIMENT_IDS
        ),
        "dataset_family": "MIDOG++",
        "fresh_unconsumed_surface": True,
        "support_evaluation_case_disjoint": True,
        "labels_opened": False,
        "consumed_test_used": False,
        "consumed_validation_used": False,
        "consumed_stage70_used": False,
        "consumed_stage90_used": False,
        "scoring_manifest_artifact_id": SCORING_MANIFEST_ARTIFACT_ID,
    }
    mismatch = [key for key, value in required.items() if payload.get(key) != value]
    if mismatch:
        raise ProtocolError(f"Utility-aligned active reservation drifted: {mismatch}.")
    support = _case_mapping(payload.get("support_case_ids_by_center"), "support")
    evaluation = _case_mapping(
        payload.get("evaluation_case_ids_by_center"), "evaluation"
    )
    support_values = tuple(case for center in CENTERS for case in support[center])
    evaluation_values = tuple(
        case for center in CENTERS for case in evaluation[center]
    )
    support_all = set(support_values)
    eval_all = set(evaluation_values)
    if len(support_all) != len(support_values) or len(eval_all) != len(
        evaluation_values
    ):
        raise ProtocolError(
            "Utility-aligned reservation case IDs must be globally unique."
        )
    if support_all.intersection(eval_all):
        raise ProtocolError("Utility-aligned support/evaluation cases overlap.")
    scoring_sha = str(payload.get("scoring_manifest_sha256", ""))
    binding_hash = str(payload.get("target_evaluation_binding_hash", ""))
    reservation_id = str(payload.get("reservation_id", ""))
    if not reservation_id or not _is_sha256(scoring_sha) or not _is_sha256(binding_hash):
        raise ProtocolError("Utility-aligned active reservation hashes are invalid.")
    return FreshReservation(
        reservation_id=reservation_id,
        reservation_hash=str(payload["reservation_hash"]),
        target_evaluation_binding_hash=binding_hash,
        support_case_ids_by_center=support,
        evaluation_case_ids_by_center=evaluation,
        scoring_manifest_sha256=scoring_sha,
        payload=MappingProxyType(payload),
    )


def _case_mapping(value: object, role: str) -> Mapping[str, tuple[str, ...]]:
    if not isinstance(value, Mapping) or {str(key) for key in value} != set(CENTERS):
        raise ProtocolError(f"Utility-aligned reservation {role} cases drifted.")
    raw = {str(key): item for key, item in value.items()}
    output: dict[str, tuple[str, ...]] = {}
    for center in CENTERS:
        values = raw[center]
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ProtocolError(f"Utility-aligned reservation {role} cases drifted.")
        cases = tuple(str(item) for item in values)
        minimum = 8 if role == "support" else 1
        if (
            len(cases) < minimum
            or len(cases) != len(set(cases))
            or any(not case for case in cases)
        ):
            raise ProtocolError(f"Utility-aligned reservation {role} cases drifted.")
        output[center] = cases
    return MappingProxyType(output)


def _read_row_index(path: Path) -> tuple[Mapping[str, object], ...]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != ROW_COLUMNS:
                raise ProtocolError("Utility-aligned target row-index columns drifted.")
            raw_rows = tuple(dict(row) for row in reader)
    except OSError as exc:
        raise ProtocolError("Cannot read utility-aligned target row index.") from exc
    output: list[Mapping[str, object]] = []
    seen: set[str] = set()
    expected_index = {center: 0 for center in CENTERS}
    for raw in raw_rows:
        center = str(raw.get("center", ""))
        row_id = str(raw.get("row_id", ""))
        case_id = str(raw.get("case_id", ""))
        try:
            index = int(raw.get("center_row_index", ""))
        except (TypeError, ValueError) as exc:
            raise ProtocolError("Utility-aligned target row index is malformed.") from exc
        if (
            raw.get("schema_version") != ROW_SCHEMA
            or center not in CENTERS
            or not row_id
            or row_id in seen
            or not case_id
            or index != expected_index[center]
        ):
            raise ProtocolError("Utility-aligned target row index is malformed.")
        expected_index[center] += 1
        seen.add(row_id)
        output.append(MappingProxyType(dict(raw)))
    return tuple(output)


def _safe_member(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ProtocolError("Utility-aligned target-cache member escapes its root.")
    return path


def _json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read utility-aligned JSON: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("Utility-aligned JSON must be a mapping.")
    return payload


def _require_hash(payload: Mapping[str, object], key: str, role: str) -> None:
    observed = payload.get(key)
    unhashed = {name: value for name, value in payload.items() if name != key}
    if observed != stable_hash(unhashed):
        raise ProtocolError(f"Utility-aligned {role} hash drifted.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    rendered = str(value or "")
    return len(rendered) == 64 and all(char in "0123456789abcdef" for char in rendered)


__all__ = (
    "CACHE_CONTENT_MEMBER",
    "CACHE_PROTOCOL_MEMBER",
    "FreshReservation",
    "FreshTargetFrame",
    "FreshTargetSurface",
    "ROW_COLUMNS",
    "ROW_INDEX_MEMBER",
    "load_fresh_target_surface",
    "require_active_fresh_target_artifacts",
)
