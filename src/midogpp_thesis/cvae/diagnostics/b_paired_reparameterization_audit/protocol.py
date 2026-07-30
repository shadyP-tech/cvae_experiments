"""Key, pairing, and comparison contracts for the Stage-90 audit."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterable, Mapping

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.cvae.protocol import ProtocolError

from .config import (
    AUDIT_CANDIDATES,
    AUDIT_CENTERS,
    CONTROLLED_CANDIDATES,
    INITIALIZATION_SEEDS,
    LEGACY_CANDIDATE,
)


KEY_SCHEMA = "midogpp_b_paired_reparameterization_key_v1"
PAIR_SCHEMA = "midogpp_b_paired_reparameterization_pair_v1"
KEY_INVENTORY_SCHEMA = "midogpp_b_paired_reparameterization_key_inventory_v1"
EXPECTED_KEY_COUNT = 36
EXPECTED_LEGACY_KEY_COUNT = 12
EXPECTED_CONTROLLED_KEY_COUNT = 24
EXPECTED_PAIR_COUNT = 12

REPLAY_VALIDATION_PURPOSE = "replay_validation"
CONTROLLED_COMPARISON_PURPOSE = "controlled_comparison"


@dataclass(frozen=True)
class AuditKeyRecord:
    """One fully addressable training cell and all immutable input bindings."""

    center: str
    initialization_seed: int
    execution_device: str
    candidate: str
    prepared_relpath: str
    prepared_sha256: str
    prepared_content_hash: str
    schedule_relpath: str
    schedule_sha256: str
    schedule_content_hash: str
    epsilon_trace_relpath: str
    epsilon_trace_sha256: str
    epsilon_trace_content_hash: str
    snapshot_manifest_hash: str
    pair_id: str | None
    key_hash: str
    legacy_expected_checkpoint_hash: str | None = None
    legacy_expected_prediction_hash: str | None = None
    legacy_expected_metric_hash: str | None = None
    legacy_expected_initialization_hash: str | None = None
    legacy_historical_training_key_hash: str | None = None
    legacy_historical_schedule_hash: str | None = None
    legacy_historical_posterior_stream_hash: str | None = None
    legacy_historical_frame_hash: str | None = None
    legacy_historical_fit_row_hash: str | None = None
    legacy_historical_eval_row_hash: str | None = None
    legacy_expected_decode_metric: Mapping[str, int | float] | None = None

    @property
    def is_legacy(self) -> bool:
        return self.candidate == LEGACY_CANDIDATE

    def hash_payload(self) -> dict[str, object]:
        return {
            "schema_version": KEY_SCHEMA,
            "center": self.center,
            "initialization_seed": self.initialization_seed,
            "execution_device": self.execution_device,
            "candidate": self.candidate,
            "prepared_relpath": self.prepared_relpath,
            "prepared_sha256": self.prepared_sha256,
            "prepared_content_hash": self.prepared_content_hash,
            "schedule_relpath": self.schedule_relpath,
            "schedule_sha256": self.schedule_sha256,
            "schedule_content_hash": self.schedule_content_hash,
            "epsilon_trace_relpath": self.epsilon_trace_relpath,
            "epsilon_trace_sha256": self.epsilon_trace_sha256,
            "epsilon_trace_content_hash": self.epsilon_trace_content_hash,
            "snapshot_manifest_hash": self.snapshot_manifest_hash,
            "pair_id": self.pair_id,
        }

    def to_payload(self) -> dict[str, object]:
        return {
            **self.hash_payload(),
            "key_hash": self.key_hash,
            "legacy_expected_checkpoint_hash": self.legacy_expected_checkpoint_hash,
            "legacy_expected_prediction_hash": self.legacy_expected_prediction_hash,
            "legacy_expected_metric_hash": self.legacy_expected_metric_hash,
            "legacy_expected_initialization_hash": self.legacy_expected_initialization_hash,
            "legacy_historical_training_key_hash": self.legacy_historical_training_key_hash,
            "legacy_historical_schedule_hash": self.legacy_historical_schedule_hash,
            "legacy_historical_posterior_stream_hash": (
                self.legacy_historical_posterior_stream_hash
            ),
            "legacy_historical_frame_hash": self.legacy_historical_frame_hash,
            "legacy_historical_fit_row_hash": self.legacy_historical_fit_row_hash,
            "legacy_historical_eval_row_hash": self.legacy_historical_eval_row_hash,
            "legacy_expected_decode_metric": (
                None
                if self.legacy_expected_decode_metric is None
                else dict(self.legacy_expected_decode_metric)
            ),
        }


def build_key_record(
    *,
    center: str,
    initialization_seed: int,
    execution_device: str,
    candidate: str,
    prepared_relpath: str,
    prepared_sha256: str,
    prepared_content_hash: str,
    schedule_relpath: str,
    schedule_sha256: str,
    schedule_content_hash: str,
    epsilon_trace_relpath: str,
    epsilon_trace_sha256: str,
    epsilon_trace_content_hash: str,
    snapshot_manifest_hash: str,
    legacy_expected_checkpoint_hash: str | None = None,
    legacy_expected_prediction_hash: str | None = None,
    legacy_expected_metric_hash: str | None = None,
    legacy_expected_initialization_hash: str | None = None,
    legacy_historical_training_key_hash: str | None = None,
    legacy_historical_schedule_hash: str | None = None,
    legacy_historical_posterior_stream_hash: str | None = None,
    legacy_historical_frame_hash: str | None = None,
    legacy_historical_fit_row_hash: str | None = None,
    legacy_historical_eval_row_hash: str | None = None,
    legacy_expected_decode_metric: Mapping[str, int | float] | None = None,
) -> AuditKeyRecord:
    """Construct one record with its canonical pair and key hashes."""

    pair_id = (
        None
        if candidate == LEGACY_CANDIDATE
        else compute_pair_id(
            center=center,
            initialization_seed=initialization_seed,
            execution_device=execution_device,
            snapshot_manifest_hash=snapshot_manifest_hash,
            prepared_content_hash=prepared_content_hash,
            schedule_content_hash=schedule_content_hash,
            epsilon_trace_content_hash=epsilon_trace_content_hash,
        )
    )
    values: dict[str, object] = {
        "center": str(center),
        "initialization_seed": int(initialization_seed),
        "execution_device": str(execution_device),
        "candidate": str(candidate),
        "prepared_relpath": str(prepared_relpath),
        "prepared_sha256": str(prepared_sha256),
        "prepared_content_hash": str(prepared_content_hash),
        "schedule_relpath": str(schedule_relpath),
        "schedule_sha256": str(schedule_sha256),
        "schedule_content_hash": str(schedule_content_hash),
        "epsilon_trace_relpath": str(epsilon_trace_relpath),
        "epsilon_trace_sha256": str(epsilon_trace_sha256),
        "epsilon_trace_content_hash": str(epsilon_trace_content_hash),
        "snapshot_manifest_hash": str(snapshot_manifest_hash),
        "pair_id": pair_id,
    }
    record = AuditKeyRecord(
        **values,
        key_hash=compute_key_hash(values),
        legacy_expected_checkpoint_hash=legacy_expected_checkpoint_hash,
        legacy_expected_prediction_hash=legacy_expected_prediction_hash,
        legacy_expected_metric_hash=legacy_expected_metric_hash,
        legacy_expected_initialization_hash=legacy_expected_initialization_hash,
        legacy_historical_training_key_hash=legacy_historical_training_key_hash,
        legacy_historical_schedule_hash=legacy_historical_schedule_hash,
        legacy_historical_posterior_stream_hash=legacy_historical_posterior_stream_hash,
        legacy_historical_frame_hash=legacy_historical_frame_hash,
        legacy_historical_fit_row_hash=legacy_historical_fit_row_hash,
        legacy_historical_eval_row_hash=legacy_historical_eval_row_hash,
        legacy_expected_decode_metric=legacy_expected_decode_metric,
    )
    validate_key_record(record, require_publication_hashes=False)
    return record


def compute_pair_id(
    *,
    center: str,
    initialization_seed: int,
    execution_device: str,
    snapshot_manifest_hash: str,
    prepared_content_hash: str,
    schedule_content_hash: str,
    epsilon_trace_content_hash: str,
) -> str:
    """Return the identity shared by the two controlled estimators."""

    return stable_hash(
        {
            "schema_version": PAIR_SCHEMA,
            "center": str(center),
            "initialization_seed": int(initialization_seed),
            "execution_device": str(execution_device),
            "snapshot_manifest_hash": str(snapshot_manifest_hash),
            "prepared_content_hash": str(prepared_content_hash),
            "schedule_content_hash": str(schedule_content_hash),
            "epsilon_trace_content_hash": str(epsilon_trace_content_hash),
            "candidates": list(CONTROLLED_CANDIDATES),
        }
    )


def compute_key_hash(record_or_payload: AuditKeyRecord | Mapping[str, object]) -> str:
    """Recompute a training-key hash from its semantic input bindings."""

    if isinstance(record_or_payload, AuditKeyRecord):
        payload = record_or_payload.hash_payload()
    else:
        payload = {key: record_or_payload[key] for key in _KEY_HASH_FIELDS}
        payload["schema_version"] = KEY_SCHEMA
    return stable_hash(payload)


def key_record_from_mapping(payload: Mapping[str, object]) -> AuditKeyRecord:
    """Parse one record and reject any non-recomputable key or pair identity."""

    try:
        record = AuditKeyRecord(
            center=str(payload["center"]),
            initialization_seed=int(payload["initialization_seed"]),
            execution_device=str(payload["execution_device"]),
            candidate=str(payload["candidate"]),
            prepared_relpath=str(payload["prepared_relpath"]),
            prepared_sha256=str(payload["prepared_sha256"]),
            prepared_content_hash=str(payload["prepared_content_hash"]),
            schedule_relpath=str(payload["schedule_relpath"]),
            schedule_sha256=str(payload["schedule_sha256"]),
            schedule_content_hash=str(payload["schedule_content_hash"]),
            epsilon_trace_relpath=str(payload["epsilon_trace_relpath"]),
            epsilon_trace_sha256=str(payload["epsilon_trace_sha256"]),
            epsilon_trace_content_hash=str(payload["epsilon_trace_content_hash"]),
            snapshot_manifest_hash=str(payload["snapshot_manifest_hash"]),
            pair_id=(
                None if payload.get("pair_id") is None else str(payload.get("pair_id"))
            ),
            key_hash=str(payload["key_hash"]),
            legacy_expected_checkpoint_hash=_optional_string(
                payload.get("legacy_expected_checkpoint_hash")
            ),
            legacy_expected_prediction_hash=_optional_string(
                payload.get("legacy_expected_prediction_hash")
            ),
            legacy_expected_metric_hash=_optional_string(
                payload.get("legacy_expected_metric_hash")
            ),
            legacy_expected_initialization_hash=_optional_string(
                payload.get("legacy_expected_initialization_hash")
            ),
            legacy_historical_training_key_hash=_optional_string(
                payload.get("legacy_historical_training_key_hash")
            ),
            legacy_historical_schedule_hash=_optional_string(
                payload.get("legacy_historical_schedule_hash")
            ),
            legacy_historical_posterior_stream_hash=_optional_string(
                payload.get("legacy_historical_posterior_stream_hash")
            ),
            legacy_historical_frame_hash=_optional_string(
                payload.get("legacy_historical_frame_hash")
            ),
            legacy_historical_fit_row_hash=_optional_string(
                payload.get("legacy_historical_fit_row_hash")
            ),
            legacy_historical_eval_row_hash=_optional_string(
                payload.get("legacy_historical_eval_row_hash")
            ),
            legacy_expected_decode_metric=(
                None
                if payload.get("legacy_expected_decode_metric") is None
                else dict(_require_mapping(payload.get("legacy_expected_decode_metric")))
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("Malformed Stage-90 audit key record.") from exc
    validate_key_record(record, require_publication_hashes=False)
    return record


def validate_key_record(
    record: AuditKeyRecord, *, require_publication_hashes: bool
) -> None:
    """Validate one record, including its recomputed key and controlled pair."""

    if record.center not in AUDIT_CENTERS:
        raise ProtocolError(f"Audit key has unsupported center {record.center!r}.")
    if record.initialization_seed not in INITIALIZATION_SEEDS:
        raise ProtocolError("Audit key has an unsupported initialization seed.")
    if record.execution_device not in {"cuda:0", "cuda:1"}:
        raise ProtocolError("Audit execution device must be cuda:0 or cuda:1.")
    if record.candidate not in AUDIT_CANDIDATES:
        raise ProtocolError(f"Audit key has unsupported candidate {record.candidate!r}.")
    for relpath in (
        record.prepared_relpath,
        record.schedule_relpath,
        record.epsilon_trace_relpath,
    ):
        _validate_relative_path(relpath)
    for digest in (
        record.prepared_sha256,
        record.prepared_content_hash,
        record.schedule_sha256,
        record.schedule_content_hash,
        record.epsilon_trace_sha256,
        record.epsilon_trace_content_hash,
    ):
        _validate_sha256(digest)
    _validate_semantic_hash(record.snapshot_manifest_hash)
    if compute_key_hash(record) != record.key_hash:
        raise ProtocolError("Audit training-key hash does not recompute.")
    if record.is_legacy:
        if record.pair_id is not None:
            raise ProtocolError("Legacy replay-validation keys cannot have pair IDs.")
        expected_hashes = (
            record.legacy_expected_checkpoint_hash,
            record.legacy_expected_prediction_hash,
            record.legacy_expected_metric_hash,
            record.legacy_expected_initialization_hash,
            record.legacy_historical_training_key_hash,
            record.legacy_historical_schedule_hash,
            record.legacy_historical_posterior_stream_hash,
            record.legacy_historical_frame_hash,
            record.legacy_historical_fit_row_hash,
            record.legacy_historical_eval_row_hash,
        )
        if require_publication_hashes and (
            any(value is None for value in expected_hashes)
            or record.legacy_expected_decode_metric is None
        ):
            raise ProtocolError(
                "Published legacy replay keys require the complete historical expectation."
            )
        for value in expected_hashes[:4]:
            if value is not None:
                _validate_sha256(value)
        for value in expected_hashes[4:]:
            if value is not None:
                _validate_semantic_hash(value)
        if record.legacy_expected_decode_metric is not None:
            _validate_decode_metric(record.legacy_expected_decode_metric)
            if (
                record.legacy_expected_metric_hash
                != _canonical_sha256(record.legacy_expected_decode_metric)
            ):
                raise ProtocolError("Legacy expected metric hash does not bind its mapping.")
    else:
        if any(
            value is not None
            for value in (
                record.legacy_expected_checkpoint_hash,
                record.legacy_expected_prediction_hash,
                record.legacy_expected_metric_hash,
                record.legacy_expected_initialization_hash,
                record.legacy_historical_training_key_hash,
                record.legacy_historical_schedule_hash,
                record.legacy_historical_posterior_stream_hash,
                record.legacy_historical_frame_hash,
                record.legacy_historical_fit_row_hash,
                record.legacy_historical_eval_row_hash,
                record.legacy_expected_decode_metric,
            )
        ):
            raise ProtocolError("Controlled keys cannot carry legacy expected hashes.")
        expected_pair = compute_pair_id(
            center=record.center,
            initialization_seed=record.initialization_seed,
            execution_device=record.execution_device,
            snapshot_manifest_hash=record.snapshot_manifest_hash,
            prepared_content_hash=record.prepared_content_hash,
            schedule_content_hash=record.schedule_content_hash,
            epsilon_trace_content_hash=record.epsilon_trace_content_hash,
        )
        if record.pair_id != expected_pair:
            raise ProtocolError("Controlled pair ID does not recompute.")


def validate_key_inventory(
    records: Iterable[AuditKeyRecord], *, require_publication_hashes: bool
) -> tuple[AuditKeyRecord, ...]:
    """Enforce the exact 36-key/12-pair predeclared audit panel."""

    frozen = tuple(records)
    if len(frozen) != EXPECTED_KEY_COUNT:
        raise ProtocolError("Audit key inventory must contain exactly 36 records.")
    coordinates = [
        (record.center, record.initialization_seed, record.candidate)
        for record in frozen
    ]
    expected = {
        (center, seed, candidate)
        for center in AUDIT_CENTERS
        for seed in INITIALIZATION_SEEDS
        for candidate in AUDIT_CANDIDATES
    }
    if set(coordinates) != expected or len(coordinates) != len(set(coordinates)):
        raise ProtocolError("Audit inventory is not the exact 4x3x3 key product.")
    for record in frozen:
        validate_key_record(
            record, require_publication_hashes=require_publication_hashes
        )
    legacy = tuple(record for record in frozen if record.is_legacy)
    controlled = tuple(record for record in frozen if not record.is_legacy)
    if len(legacy) != EXPECTED_LEGACY_KEY_COUNT or len(controlled) != EXPECTED_CONTROLLED_KEY_COUNT:
        raise ProtocolError("Audit inventory legacy/controlled counts must be 12/24.")
    _validate_stream_invariance(legacy, controlled)
    for center in AUDIT_CENTERS:
        for seed in INITIALIZATION_SEEDS:
            devices = {
                record.execution_device
                for record in frozen
                if record.center == center and record.initialization_seed == seed
            }
            if len(devices) != 1:
                raise ProtocolError(
                    "Legacy and controlled keys must share one execution device per coordinate."
                )
    pair_groups: dict[str, list[AuditKeyRecord]] = {}
    for record in controlled:
        assert_candidate_use(record.candidate, CONTROLLED_COMPARISON_PURPOSE)
        pair_groups.setdefault(str(record.pair_id), []).append(record)
    if len(pair_groups) != EXPECTED_PAIR_COUNT:
        raise ProtocolError("Audit comparison must contain exactly 12 pair IDs.")
    for pair in pair_groups.values():
        if {record.candidate for record in pair} != set(CONTROLLED_CANDIDATES):
            raise ProtocolError("Each pair ID must contain exactly the fixed estimator pair.")
        if len(pair) != 2:
            raise ProtocolError("Each pair ID must occur exactly twice.")
    if len({record.key_hash for record in frozen}) != EXPECTED_KEY_COUNT:
        raise ProtocolError("Audit key hashes must be unique.")
    return frozen


def key_inventory_hash(records: Iterable[AuditKeyRecord]) -> str:
    """Canonical inventory hash independent of serialization order."""

    ordered = sorted(
        (record.to_payload() for record in records),
        key=lambda row: (
            str(row["center"]),
            int(row["initialization_seed"]),
            str(row["candidate"]),
        ),
    )
    return stable_hash({"schema_version": KEY_INVENTORY_SCHEMA, "records": ordered})


def comparison_pairs(
    records: Iterable[AuditKeyRecord],
) -> tuple[tuple[AuditKeyRecord, AuditKeyRecord], ...]:
    """Return only the 12 fixed one-epsilon/antithetic comparison pairs."""

    frozen = validate_key_inventory(records, require_publication_hashes=False)
    by_pair: dict[str, dict[str, AuditKeyRecord]] = {}
    for record in frozen:
        if record.is_legacy:
            continue
        by_pair.setdefault(str(record.pair_id), {})[record.candidate] = record
    return tuple(
        (
            candidates[CONTROLLED_CANDIDATES[0]],
            candidates[CONTROLLED_CANDIDATES[1]],
        )
        for _, candidates in sorted(by_pair.items())
    )


def assert_candidate_use(candidate: str, purpose: str) -> None:
    """Block legacy rows from comparison and controlled rows from replay claims."""

    if purpose == REPLAY_VALIDATION_PURPOSE:
        if candidate != LEGACY_CANDIDATE:
            raise ProtocolError("Replay validation accepts only the legacy candidate.")
        return
    if purpose == CONTROLLED_COMPARISON_PURPOSE:
        if candidate not in CONTROLLED_CANDIDATES:
            raise ProtocolError("Controlled comparison excludes legacy replay rows.")
        return
    raise ProtocolError(f"Unknown Stage-90 audit purpose {purpose!r}.")


def _validate_stream_invariance(
    legacy: tuple[AuditKeyRecord, ...],
    controlled: tuple[AuditKeyRecord, ...],
) -> None:
    for center in AUDIT_CENTERS:
        fixed = [record for record in controlled if record.center == center]
        prepared = {
            (record.prepared_sha256, record.prepared_content_hash) for record in fixed
        }
        schedules = {
            (record.schedule_sha256, record.schedule_content_hash) for record in fixed
        }
        traces = {
            (record.epsilon_trace_sha256, record.epsilon_trace_content_hash)
            for record in fixed
        }
        if len(prepared) != 1 or len(schedules) != 1 or len(traces) != 1:
            raise ProtocolError(
                "Fixed prepared/schedule/epsilon inputs must be candidate/seed invariant per center."
            )
        legacy_center = [record for record in legacy if record.center == center]
        trace_by_seed = {
            record.initialization_seed: record.epsilon_trace_content_hash
            for record in legacy_center
        }
        if len(trace_by_seed) != len(INITIALIZATION_SEEDS):
            raise ProtocolError("Legacy epsilon traces must be keyed by center and seed.")
        if len(set(trace_by_seed.values())) != len(INITIALIZATION_SEEDS):
            raise ProtocolError("Legacy epsilon traces must be seed-specific within each center.")


def _validate_relative_path(value: str) -> None:
    from pathlib import PurePosixPath

    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ProtocolError("Audit input references must be safe relative paths.")


def _validate_sha256(value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ProtocolError("Audit bindings require lowercase full SHA-256 digests.")


def _validate_semantic_hash(value: str) -> None:
    if len(value) != 16 or any(character not in "0123456789abcdef" for character in value):
        raise ProtocolError("Audit semantic bindings require canonical 16-hex hashes.")


def _optional_string(value: object) -> str | None:
    return None if value is None else str(value)


def _require_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProtocolError("Legacy decode metric must be a mapping.")
    return value


def _validate_decode_metric(metric: Mapping[str, int | float]) -> None:
    required = {"bacc", "positive_recall", "specificity", "fn", "fp", "tn", "tp"}
    if set(metric) != required:
        raise ProtocolError("Legacy decode metric has the wrong canonical subset.")
    if not all(
        math.isfinite(float(metric[key]))
        for key in ("bacc", "positive_recall", "specificity")
    ):
        raise ProtocolError("Legacy decode metric rates must be finite.")
    if any(int(metric[key]) < 0 for key in ("fn", "fp", "tn", "tp")):
        raise ProtocolError("Legacy decode metric counts must be nonnegative.")


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_KEY_HASH_FIELDS = tuple(
    key
    for key in AuditKeyRecord.__dataclass_fields__
    if key
    not in {
        "key_hash",
        "legacy_expected_checkpoint_hash",
        "legacy_expected_prediction_hash",
        "legacy_expected_metric_hash",
        "legacy_expected_initialization_hash",
        "legacy_historical_training_key_hash",
        "legacy_historical_schedule_hash",
        "legacy_historical_posterior_stream_hash",
        "legacy_historical_frame_hash",
        "legacy_historical_fit_row_hash",
        "legacy_historical_eval_row_hash",
        "legacy_expected_decode_metric",
    }
)
