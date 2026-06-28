"""Artifact hashing, freeze, and lineage validation utilities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ..protocol import ProtocolError
from ..schemas import ArtifactLineageKey, REQUIRED_LINEAGE_COLUMNS


@dataclass(frozen=True)
class FrozenProtocolSnapshot:
    """Hashable pre-evaluation snapshot for settings that must not drift."""

    candidate_pool_hash: str
    generation_config_hash: str
    classifier_config_hash: str
    metric_config_hash: str
    feature_config_hash: str
    routing_config_hash: str

    @property
    def protocol_hash(self) -> str:
        return stable_hash(self.to_payload())

    def to_payload(self) -> dict[str, str]:
        return {
            "candidate_pool_hash": self.candidate_pool_hash,
            "generation_config_hash": self.generation_config_hash,
            "classifier_config_hash": self.classifier_config_hash,
            "metric_config_hash": self.metric_config_hash,
            "feature_config_hash": self.feature_config_hash,
            "routing_config_hash": self.routing_config_hash,
        }


def stable_hash(payload: object) -> str:
    """Return a deterministic short hash for JSON-serializable protocol payloads."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def write_frozen_snapshot(path: Path, snapshot: FrozenProtocolSnapshot) -> None:
    """Persist a frozen config snapshot before target-evaluation metrics exist."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = snapshot.to_payload() | {"protocol_hash": snapshot.protocol_hash}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def assert_frozen_snapshot_exists(path: Path) -> None:
    if not path.exists():
        raise ProtocolError(f"Missing frozen protocol snapshot before metric write: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"Malformed frozen protocol snapshot: {path}") from exc
    required = {
        "candidate_pool_hash",
        "generation_config_hash",
        "classifier_config_hash",
        "metric_config_hash",
        "feature_config_hash",
        "routing_config_hash",
        "protocol_hash",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ProtocolError(f"Frozen protocol snapshot missing fields: {missing}")


def assert_selection_lineage(
    *,
    selection_rows: Sequence[Mapping[str, object]],
    feature_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
) -> None:
    """Require each selection row to trace to exactly one feature and candidate row."""

    feature_keys = [_lineage_tuple(row) for row in feature_rows]
    candidate_ids = [str(row.get("candidate_id", "")) for row in candidate_rows]
    for row in selection_rows:
        missing = [key for key in REQUIRED_LINEAGE_COLUMNS if key not in row]
        if missing:
            raise ProtocolError(f"Selection row missing lineage columns: {missing}")
        key = _lineage_tuple(row)
        if feature_keys.count(key) != 1:
            raise ProtocolError(
                f"Selection row must trace to exactly one allowed feature row; "
                f"found {feature_keys.count(key)} for {key}"
            )
        candidate_id = str(row["candidate_id"])
        if candidate_ids.count(candidate_id) != 1:
            raise ProtocolError(
                f"Selection row must trace to exactly one candidate manifest row; "
                f"found {candidate_ids.count(candidate_id)} for candidate_id={candidate_id!r}"
            )


def _lineage_tuple(row: Mapping[str, object]) -> tuple[object, ...]:
    key = ArtifactLineageKey.from_mapping(row)
    return (
        key.fold_id,
        key.experiment_seed,
        key.target_domain,
        key.support_split_id,
        key.eval_split_id,
        key.candidate_id,
        key.expert_checkpoint_id,
        key.expert_checkpoint_hash,
        key.generation_mode,
        key.generation_seed,
        key.classifier_seed,
        key.config_hash,
        key.protocol_hash,
        key.eligibility,
    )
