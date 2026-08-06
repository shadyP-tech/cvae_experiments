"""Independent verification of the durable Stage-70 prediction boundary.

The scoring boundary deliberately trusts neither a self-consistent directory nor
the mutable objects used to compute predictions.  A production binding is rebuilt
from the validated final authorization and canonical target cache.  The durable
capability returned by :func:`seal_prediction_pass` then pins the exact bytes that
may be used for label opening and scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...common.hashing import stable_hash
from ...data.contract.stage70_target_evaluation.contracts import (
    CANONICAL_MANIFEST_SHA256,
    EXPECTED_TEST_ROWS,
    EXPECTED_TEST_ROWS_BY_CENTER,
)
from ...data.features.stage70_test_cache import (
    CACHE_ARTIFACT_ID,
    load_validated_stage70_test_cache,
)
from ..expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from ..protocol import ProtocolError
from .authorization import (
    load_final_authorization_config,
    read_final_authorization_token,
    validate_final_prediction_authorization,
)
from .authorization.contracts import FINAL_AUTHORIZATION_OUTPUT_ARTIFACT_ID
from .contracts import (
    CONTROL_ARM,
    EXPECTED_METRIC_ROWS,
    POLICY_ARMS,
    UTILITY_ARM,
    array_sha256,
)
from .prediction import PersistedPredictionPass


_INDEX_SCHEMA = "midogpp_stage70_prediction_index_v2"
_SEAL_SCHEMA = "midogpp_stage70_prediction_seal_v2"
_PHASE_MARKER_SCHEMA = "midogpp_stage70_phase_marker_v2"
_PHASE_01_SCHEMA = "midogpp_stage70_phase_01_authorization_binding_v2"

PREDICTION_RECORD_FIELDS = frozenset(
    {
        "ordinal",
        "policy_id",
        "target_center",
        "training_seed",
        "generation_seed",
        "replicate_id",
        "row_count",
        "evaluation_row_ids",
        "contract_row_indices",
        "case_ids",
        "target_identity_hash",
        "prediction_array_key",
        "probability_array_key",
        "prediction_sha256",
        "probability_sha256",
        "composition_manifest_hash",
        "train_content_sha256",
        "classifier_config_hash",
        "scaler_state_hash",
        "target_row_order_hash",
        "reused_from_policy_id",
        "prediction_cell_hash",
    }
)
PREDICTION_INDEX_FIELDS = frozenset(
    {
        "schema_version",
        "phase",
        "target_labels_opened",
        "cell_count",
        "target_row_count",
        "phase_01_sha256",
        "authorization_binding_hash",
        "prediction_metadata_hash",
        "records",
    }
)
PREDICTION_SEAL_FIELDS = frozenset(
    {
        "schema_version",
        "phase",
        "phase_01_sha256",
        "authorization_binding_hash",
        "prediction_index_sha256",
        "prediction_arrays_sha256",
        "prediction_metadata_hash",
        "cell_count",
        "target_row_count",
        "classifier_fit_count",
        "prediction_reuse_count",
        "target_labels_opened",
    }
)


@dataclass(frozen=True)
class TargetIdentity:
    evaluation_row_id: str
    contract_row_index: int
    case_id: str

    def to_payload(self) -> dict[str, object]:
        return {
            "evaluation_row_id": self.evaluation_row_id,
            "contract_row_index": self.contract_row_index,
            "case_id": self.case_id,
        }


@dataclass(frozen=True)
class PredictionSealBinding:
    """Strict expected identity reconstructed before predictions are sealed.

    Production callers obtain this only through
    :func:`load_canonical_prediction_seal_binding`.  The constructor remains
    useful to focused tests, but the public label-opening function never accepts
    an injected binding and always rebuilds canonical evidence itself.
    """

    final_authorization_artifact_id: str
    final_authorization_hash: str
    final_authorization_content_hash: str
    authorization_protocol_hash: str
    identity_lock_hash: str
    evaluation_plan_hash: str
    reservation_content_hash: str
    reservation_identity_lock_hash: str
    target_evaluation_reservation_id: str
    target_evaluation_reservation_protocol_hash: str
    target_identity_table_hash: str
    target_cache_artifact_id: str
    target_cache_content_hash: str
    target_cache_row_order_hash: str
    target_cache_shard_sha256_by_center: Mapping[str, str]
    target_cache_rows_by_center: Mapping[str, int]
    cache_extractor_protocol_hash: str
    scoring_manifest_sha256: str
    classifier_config_hash: str
    identities_by_center: Mapping[str, tuple[TargetIdentity, ...]]
    replicate_id_by_cell: Mapping[tuple[str, str, int, int], str]

    def __post_init__(self) -> None:
        if self.final_authorization_artifact_id != FINAL_AUTHORIZATION_OUTPUT_ARTIFACT_ID:
            raise ProtocolError("Stage-70 final authorization artifact identity drifted.")
        if self.target_cache_artifact_id != CACHE_ARTIFACT_ID:
            raise ProtocolError("Stage-70 target-cache artifact identity drifted.")
        for role, value in (
            ("final authorization hash", self.final_authorization_hash),
            ("final authorization content hash", self.final_authorization_content_hash),
            ("authorization protocol hash", self.authorization_protocol_hash),
            ("identity-lock hash", self.identity_lock_hash),
            ("evaluation-plan hash", self.evaluation_plan_hash),
            ("reservation content hash", self.reservation_content_hash),
            ("reservation identity-lock hash", self.reservation_identity_lock_hash),
            ("reservation protocol hash", self.target_evaluation_reservation_protocol_hash),
            ("target identity-table hash", self.target_identity_table_hash),
            ("cache content hash", self.target_cache_content_hash),
            ("cache row-order hash", self.target_cache_row_order_hash),
            ("cache extractor protocol hash", self.cache_extractor_protocol_hash),
            ("scoring-manifest SHA-256", self.scoring_manifest_sha256),
            ("classifier configuration hash", self.classifier_config_hash),
        ):
            if not _is_hash(value):
                raise ProtocolError(f"Stage-70 {role} is malformed.")
        if not self.target_evaluation_reservation_id.startswith("reservation_"):
            raise ProtocolError("Stage-70 target reservation identity is malformed.")

        counts = {str(key): int(value) for key, value in self.target_cache_rows_by_center.items()}
        shards = {
            str(key): str(value)
            for key, value in self.target_cache_shard_sha256_by_center.items()
        }
        raw_identities = {
            str(center): tuple(rows)
            for center, rows in self.identities_by_center.items()
        }
        if (
            counts != dict(EXPECTED_TEST_ROWS_BY_CENTER)
            or set(shards) != set(CENTERS)
            or set(raw_identities) != set(CENTERS)
            or any(not _is_sha256(value) for value in shards.values())
        ):
            raise ProtocolError("Stage-70 canonical target-cache coverage drifted.")

        global_ids: list[str] = []
        global_indices: list[int] = []
        normalized_identities: dict[str, tuple[TargetIdentity, ...]] = {}
        for center in CENTERS:
            rows = raw_identities[center]
            if len(rows) != counts[center]:
                raise ProtocolError(
                    f"Stage-70 target identity count drifted for center {center}."
                )
            previous_index = -1
            normalized: list[TargetIdentity] = []
            for row in rows:
                if not isinstance(row, TargetIdentity):
                    raise ProtocolError("Stage-70 target identity binding is malformed.")
                if (
                    not _is_neutral_evaluation_id(row.evaluation_row_id)
                    or isinstance(row.contract_row_index, bool)
                    or row.contract_row_index < 0
                    or row.contract_row_index <= previous_index
                    or not row.case_id
                ):
                    raise ProtocolError(
                        f"Stage-70 target identity order drifted for center {center}."
                    )
                previous_index = row.contract_row_index
                normalized.append(row)
                global_ids.append(row.evaluation_row_id)
                global_indices.append(row.contract_row_index)
            normalized_identities[center] = tuple(normalized)
        if (
            len(global_ids) != EXPECTED_TEST_ROWS
            or len(set(global_ids)) != EXPECTED_TEST_ROWS
            or len(set(global_indices)) != EXPECTED_TEST_ROWS
        ):
            raise ProtocolError("Stage-70 global target identity coverage drifted.")

        replicate_ids = {
            (str(key[0]), str(key[1]), int(key[2]), int(key[3])): str(value)
            for key, value in self.replicate_id_by_cell.items()
        }
        if set(replicate_ids) != set(expected_cell_keys()) or any(
            not value for value in replicate_ids.values()
        ):
            raise ProtocolError("Stage-70 authorized prediction-cell identities drifted.")
        object.__setattr__(self, "target_cache_rows_by_center", MappingProxyType(counts))
        object.__setattr__(
            self,
            "target_cache_shard_sha256_by_center",
            MappingProxyType(shards),
        )
        object.__setattr__(
            self,
            "identities_by_center",
            MappingProxyType(normalized_identities),
        )
        object.__setattr__(
            self,
            "replicate_id_by_cell",
            MappingProxyType(replicate_ids),
        )

    @property
    def target_identity_hash_by_center(self) -> dict[str, str]:
        return {
            center: stable_hash([row.to_payload() for row in self.identities_by_center[center]])
            for center in CENTERS
        }

    @property
    def global_target_identity_hash(self) -> str:
        rows = sorted(
            (row for center in CENTERS for row in self.identities_by_center[center]),
            key=lambda row: row.contract_row_index,
        )
        return stable_hash([row.to_payload() for row in rows])

    @property
    def authorized_cell_hash(self) -> str:
        return stable_hash(
            [
                {
                    "policy_id": key[0],
                    "target_center": key[1],
                    "training_seed": key[2],
                    "generation_seed": key[3],
                    "replicate_id": self.replicate_id_by_cell[key],
                }
                for key in expected_cell_keys()
            ]
        )

    def phase_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": _PHASE_01_SCHEMA,
            "phase": "AUTHORIZATION_COMPLETE",
            "final_authorization_artifact_id": self.final_authorization_artifact_id,
            "final_authorization_hash": self.final_authorization_hash,
            "final_authorization_content_hash": self.final_authorization_content_hash,
            "authorization_protocol_hash": self.authorization_protocol_hash,
            "identity_lock_hash": self.identity_lock_hash,
            "evaluation_plan_hash": self.evaluation_plan_hash,
            "reservation_content_hash": self.reservation_content_hash,
            "reservation_identity_lock_hash": self.reservation_identity_lock_hash,
            "target_evaluation_reservation_id": self.target_evaluation_reservation_id,
            "target_evaluation_reservation_protocol_hash": (
                self.target_evaluation_reservation_protocol_hash
            ),
            "target_identity_table_hash": self.target_identity_table_hash,
            "target_cache_artifact_id": self.target_cache_artifact_id,
            "target_cache_content_hash": self.target_cache_content_hash,
            "target_cache_row_order_hash": self.target_cache_row_order_hash,
            "target_cache_shard_sha256_by_center": dict(
                self.target_cache_shard_sha256_by_center
            ),
            "target_cache_rows_by_center": dict(self.target_cache_rows_by_center),
            "target_cache_row_count": EXPECTED_TEST_ROWS,
            "cache_extractor_protocol_hash": self.cache_extractor_protocol_hash,
            "scoring_manifest_sha256": self.scoring_manifest_sha256,
            "classifier_config_hash": self.classifier_config_hash,
            "authorized_cell_hash": self.authorized_cell_hash,
            "target_identity_hash_by_center": self.target_identity_hash_by_center,
            "global_target_identity_hash": self.global_target_identity_hash,
            "target_labels_opened": False,
        }
        payload["authorization_binding_hash"] = stable_hash(payload)
        return payload


@dataclass(frozen=True)
class VerifiedPredictionCell:
    """Immutable scoring input reconstructed from a verified NPZ/index pair."""

    policy_id: str
    target_center: str
    training_seed: int
    generation_seed: int
    replicate_id: str
    evaluation_row_ids: tuple[str, ...]
    contract_row_indices: tuple[int, ...]
    case_ids: tuple[str, ...]
    predictions: tuple[int, ...]
    target_identity_hash: str
    prediction_sha256: str
    probability_sha256: str
    composition_manifest_hash: str
    train_content_sha256: str
    classifier_config_hash: str
    scaler_state_hash: str
    target_row_order_hash: str
    reused_from_policy_id: str
    prediction_cell_hash: str


@dataclass(frozen=True)
class VerifiedPredictionArtifact:
    """Exact on-disk prediction transaction verified without opening labels."""

    artifact_root: Path
    cells: tuple[VerifiedPredictionCell, ...]
    phase_01_binding: Mapping[str, object]
    authorization_binding_hash: str
    phase_01_sha256: str
    prediction_index_sha256: str
    prediction_arrays_sha256: str
    prediction_seal_sha256: str
    phase_02_sha256: str

    @property
    def records(self) -> tuple[Mapping[str, object], ...]:
        """Compatibility view containing immutable metric provenance only."""

        return tuple(MappingProxyType(_verified_cell_payload(cell)) for cell in self.cells)


def load_canonical_prediction_seal_binding(
    *,
    final_authorization_root: str | PathLike[str],
    target_cache_root: str | PathLike[str],
    scoring_manifest_path: str | PathLike[str],
    expected_manifest_sha256: str,
) -> PredictionSealBinding:
    """Rebuild the production binding from canonical independently validated inputs."""

    auth_root = _safe_directory(Path(final_authorization_root), "final authorization")
    cache_root = _safe_directory(Path(target_cache_root), "target cache")
    manifest = _safe_file(Path(scoring_manifest_path), "scoring manifest")
    if expected_manifest_sha256 != CANONICAL_MANIFEST_SHA256:
        raise ProtocolError("Stage-70 production scoring manifest is not canonical.")
    if _sha256_file(manifest) != expected_manifest_sha256:
        raise ProtocolError("Stage-70 scoring-manifest SHA-256 drifted.")

    final_config = load_final_authorization_config(auth_root / "config.resolved.yaml")
    if (
        _resolved_path(final_config.cache_root) != _resolved_path(cache_root)
        or _resolved_path(final_config.scoring_manifest_path) != _resolved_path(manifest)
        or final_config.expected_scoring_manifest_sha256 != expected_manifest_sha256
    ):
        raise ProtocolError("Stage-70 final authorization input paths drifted.")
    checks = validate_final_prediction_authorization(auth_root, config=final_config)
    token = read_final_authorization_token(auth_root).to_payload()
    cache = load_validated_stage70_test_cache(cache_root)
    summary = dict(cache.summary)
    identity = _json(auth_root / "manifests/identity_lock.json")
    plan = _json(auth_root / "manifests/evaluation_plan.json")

    exact_pairs = {
        "authorization_token_hash": (
            token.get("authorization_token_hash"),
            checks.get("authorization_token_hash"),
        ),
        "authorization_protocol_hash": (
            token.get("authorization_protocol_hash"),
            checks.get("authorization_protocol_hash"),
        ),
        "identity_lock_hash": (
            token.get("identity_lock_hash"),
            identity.get("identity_lock_hash"),
        ),
        "evaluation_plan_hash": (
            token.get("evaluation_plan_hash"),
            plan.get("evaluation_plan_hash"),
        ),
        "reservation_content_hash": (
            token.get("reservation_content_hash"),
            identity.get("reservation_content_hash"),
        ),
        "target_cache_content_hash": (
            token.get("target_cache_content_hash"),
            summary.get("content_hash"),
        ),
        "target_cache_row_order_hash": (
            token.get("target_cache_row_order_hash"),
            summary.get("row_order_hash"),
        ),
        "scoring_manifest_sha256": (
            token.get("scoring_manifest_sha256"),
            expected_manifest_sha256,
        ),
        "identity_cache_content_hash": (
            identity.get("target_cache_content_hash"),
            summary.get("content_hash"),
        ),
        "identity_cache_row_order_hash": (
            identity.get("target_cache_row_order_hash"),
            summary.get("row_order_hash"),
        ),
        "identity_cache_shards": (
            identity.get("target_cache_shard_sha256_by_center"),
            summary.get("shard_sha256_by_center"),
        ),
        "identity_rows_by_center": (
            identity.get("rows_by_center"),
            summary.get("rows_by_center"),
        ),
        "reservation_id": (
            identity.get("target_evaluation_reservation_id"),
            summary.get("target_evaluation_reservation_id"),
        ),
        "reservation_protocol_hash": (
            identity.get("target_evaluation_reservation_protocol_hash"),
            summary.get("target_evaluation_reservation_protocol_hash"),
        ),
        "identity_scoring_manifest": (
            identity.get("scoring_manifest_sha256"),
            expected_manifest_sha256,
        ),
        "identity_row_count": (identity.get("row_count"), EXPECTED_TEST_ROWS),
        "cache_row_count": (summary.get("row_count"), EXPECTED_TEST_ROWS),
        "authorization_row_count": (checks.get("row_count"), EXPECTED_TEST_ROWS),
        "authorization_center_count": (checks.get("center_count"), len(CENTERS)),
        "authorization_evaluation_cells": (
            checks.get("evaluation_plan_rows"),
            EXPECTED_METRIC_ROWS,
        ),
        "cache_manifest": (summary.get("manifest_sha256"), expected_manifest_sha256),
    }
    mismatches = [key for key, pair in exact_pairs.items() if pair[0] != pair[1]]
    if mismatches:
        raise ProtocolError(
            f"Stage-70 final authorization/cache binding drifted: {mismatches}."
        )

    identities: dict[str, tuple[TargetIdentity, ...]] = {}
    expected_shards = _string_mapping(
        summary["shard_sha256_by_center"], "cache shard identities"
    )
    for center in CENTERS:
        shard = cache.load_center(center)
        if shard.shard_sha256 != expected_shards[center]:
            raise ProtocolError(
                f"Stage-70 target-cache shard changed after validation: {center}."
            )
        identities[center] = tuple(
            TargetIdentity(row_id, row_index, case_id)
            for row_id, row_index, case_id in zip(
                shard.evaluation_row_ids,
                shard.contract_row_indices,
                shard.case_ids,
                strict=True,
            )
        )
    replicate_ids = _replicate_ids_from_plan(plan)
    return PredictionSealBinding(
        final_authorization_artifact_id=FINAL_AUTHORIZATION_OUTPUT_ARTIFACT_ID,
        final_authorization_hash=str(token["authorization_token_hash"]),
        final_authorization_content_hash=str(checks["content_hash"]),
        authorization_protocol_hash=str(token["authorization_protocol_hash"]),
        identity_lock_hash=str(identity["identity_lock_hash"]),
        evaluation_plan_hash=str(plan["evaluation_plan_hash"]),
        reservation_content_hash=str(identity["reservation_content_hash"]),
        reservation_identity_lock_hash=str(identity["reservation_identity_lock_hash"]),
        target_evaluation_reservation_id=str(
            identity["target_evaluation_reservation_id"]
        ),
        target_evaluation_reservation_protocol_hash=str(
            identity["target_evaluation_reservation_protocol_hash"]
        ),
        target_identity_table_hash=str(identity["target_identity_table_hash"]),
        target_cache_artifact_id=str(identity["target_cache_artifact_id"]),
        target_cache_content_hash=str(summary["content_hash"]),
        target_cache_row_order_hash=str(summary["row_order_hash"]),
        target_cache_shard_sha256_by_center=expected_shards,
        target_cache_rows_by_center=_integer_mapping(
            summary["rows_by_center"], "cache center counts"
        ),
        cache_extractor_protocol_hash=str(summary["cache_extractor_protocol_hash"]),
        scoring_manifest_sha256=expected_manifest_sha256,
        classifier_config_hash=str(token["classifier_config_hash"]),
        identities_by_center=identities,
        replicate_id_by_cell=replicate_ids,
    )


def verify_persisted_prediction_artifact(
    sealed: PersistedPredictionPass,
    *,
    expected_binding: PredictionSealBinding | None = None,
) -> VerifiedPredictionArtifact:
    """Fail closed unless the immutable capability still names exact disk bytes."""

    if not isinstance(sealed, PersistedPredictionPass):
        raise ProtocolError(
            "Stage-70 prediction verification requires a persisted seal capability."
        )
    root = _safe_directory(sealed.artifact_root, "prediction artifact")
    if any(member.is_symlink() for member in root.rglob("*")):
        raise ProtocolError("Stage-70 prediction artifact contains a symlink.")
    paths = _prediction_paths(root)
    for role, path in paths.items():
        _safe_file(path, f"prediction {role}")
    observed_capability = {
        "phase_01_sha256": _sha256_file(paths["phase_01"]),
        "prediction_index_sha256": _sha256_file(paths["index"]),
        "prediction_arrays_sha256": _sha256_file(paths["arrays"]),
        "prediction_seal_sha256": _sha256_file(paths["seal"]),
        "phase_02_sha256": _sha256_file(paths["phase_02"]),
    }
    mismatches = [
        field
        for field, value in observed_capability.items()
        if getattr(sealed, field) != value
    ]
    if mismatches:
        raise ProtocolError(
            f"Stage-70 persisted seal capability drifted: {mismatches}."
        )

    phase_01 = _json(paths["phase_01"])
    _validate_phase_01(phase_01, expected_binding=expected_binding)
    phase_01_sha = observed_capability["phase_01_sha256"]
    binding_hash = str(phase_01["authorization_binding_hash"])
    if (
        sealed.authorization_binding_hash != binding_hash
        or sealed.phase_01_sha256 != phase_01_sha
    ):
        raise ProtocolError("Stage-70 persisted capability authorization drifted.")

    index = _json(paths["index"])
    if set(index) != PREDICTION_INDEX_FIELDS:
        raise ProtocolError("Stage-70 prediction index schema drifted.")
    raw_records = index.get("records")
    if (
        index.get("schema_version") != _INDEX_SCHEMA
        or index.get("phase") != "PREDICTIONS_PERSISTED"
        or index.get("target_labels_opened") is not False
        or index.get("cell_count") != EXPECTED_METRIC_ROWS
        or index.get("target_row_count") != EXPECTED_TEST_ROWS
        or index.get("phase_01_sha256") != phase_01_sha
        or index.get("authorization_binding_hash") != binding_hash
        or not isinstance(raw_records, list)
        or len(raw_records) != EXPECTED_METRIC_ROWS
        or index.get("prediction_metadata_hash") != stable_hash(raw_records)
    ):
        raise ProtocolError("Stage-70 prediction index coverage or binding drifted.")
    if not all(isinstance(row, Mapping) for row in raw_records):
        raise ProtocolError("Stage-70 prediction index contains a non-object row.")
    records = tuple(dict(row) for row in raw_records)  # type: ignore[arg-type]
    _validate_record_grid(records, phase_01)
    _validate_target_identities(records, phase_01)

    seal = _json(paths["seal"])
    marker = _json(paths["phase_02"])
    _validate_seal_payload(
        seal,
        schema=_SEAL_SCHEMA,
        phase_01_sha256=phase_01_sha,
        binding_hash=binding_hash,
        index_sha256=observed_capability["prediction_index_sha256"],
        arrays_sha256=observed_capability["prediction_arrays_sha256"],
        metadata_hash=str(index["prediction_metadata_hash"]),
    )
    expected_marker = {**seal, "schema_version": _PHASE_MARKER_SCHEMA}
    if marker != expected_marker:
        raise ProtocolError("Stage-70 phase-02 prediction seal drifted.")

    cells = _validate_prediction_arrays(paths["arrays"], records, phase_01)
    # Rehash paths after parsing to close the ordinary replace-while-reading gap.
    for field, path_key in (
        ("phase_01_sha256", "phase_01"),
        ("prediction_index_sha256", "index"),
        ("prediction_arrays_sha256", "arrays"),
        ("prediction_seal_sha256", "seal"),
        ("phase_02_sha256", "phase_02"),
    ):
        if _sha256_file(paths[path_key]) != getattr(sealed, field):
            raise ProtocolError("Stage-70 prediction transaction changed during verification.")
    return VerifiedPredictionArtifact(
        artifact_root=root,
        cells=cells,
        phase_01_binding=MappingProxyType(dict(phase_01)),
        authorization_binding_hash=binding_hash,
        phase_01_sha256=phase_01_sha,
        prediction_index_sha256=observed_capability["prediction_index_sha256"],
        prediction_arrays_sha256=observed_capability["prediction_arrays_sha256"],
        prediction_seal_sha256=observed_capability["prediction_seal_sha256"],
        phase_02_sha256=observed_capability["phase_02_sha256"],
    )


def validate_persisted_prediction_pass(
    sealed: PersistedPredictionPass,
) -> VerifiedPredictionArtifact:
    """Compatibility name for disk-only persisted-capability verification."""

    return verify_persisted_prediction_artifact(sealed)


def expected_cell_keys() -> tuple[tuple[str, str, int, int], ...]:
    """Return the one canonical on-disk cell order."""

    return tuple(
        (arm, center, training_seed, generation_seed)
        for arm in POLICY_ARMS
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
        for center in CENTERS
    )


def prediction_paths(root: str | Path) -> dict[str, Path]:
    """Internal serialization path map shared by the durable writer."""

    return _prediction_paths(Path(root))


def validate_authorization_phase(
    root: str | Path,
    expected_binding: PredictionSealBinding,
) -> str:
    """Verify the exact phase-01 payload and return its byte SHA-256."""

    path = _safe_file(
        Path(root) / "reports/phase_01_authorization_complete.json",
        "phase-01 authorization binding",
    )
    payload = _json(path)
    _validate_phase_01(payload, expected_binding=expected_binding)
    return _sha256_file(path)


def prediction_record_payload(
    *,
    ordinal: int,
    cell: object,
) -> dict[str, object]:
    """Create the exact hash-bound record written for one prediction cell."""

    identities = _identity_payload(
        tuple(str(value) for value in getattr(cell, "evaluation_row_ids")),
        tuple(int(value) for value in getattr(cell, "contract_row_indices")),
        tuple(str(value) for value in getattr(cell, "case_ids")),
    )
    payload: dict[str, object] = {
        "ordinal": ordinal,
        "policy_id": str(getattr(cell, "policy_id")),
        "target_center": str(getattr(cell, "target_center")),
        "training_seed": int(getattr(cell, "training_seed")),
        "generation_seed": int(getattr(cell, "generation_seed")),
        "replicate_id": str(getattr(cell, "replicate_id")),
        "row_count": len(identities),
        "evaluation_row_ids": [row["evaluation_row_id"] for row in identities],
        "contract_row_indices": [row["contract_row_index"] for row in identities],
        "case_ids": [row["case_id"] for row in identities],
        "target_identity_hash": stable_hash(identities),
        "prediction_array_key": f"prediction_{ordinal:03d}",
        "probability_array_key": f"probability_{ordinal:03d}",
        "prediction_sha256": str(getattr(cell, "prediction_sha256")),
        "probability_sha256": str(getattr(cell, "probability_sha256")),
        "composition_manifest_hash": str(getattr(cell, "composition_manifest_hash")),
        "train_content_sha256": str(getattr(cell, "train_content_sha256")),
        "classifier_config_hash": str(getattr(cell, "classifier_config_hash")),
        "scaler_state_hash": str(getattr(cell, "scaler_state_hash")),
        "target_row_order_hash": str(getattr(cell, "target_row_order_hash")),
        "reused_from_policy_id": str(getattr(cell, "reused_from_policy_id")),
    }
    payload["prediction_cell_hash"] = stable_hash(payload)
    return payload


def _validate_phase_01(
    payload: Mapping[str, object],
    *,
    expected_binding: PredictionSealBinding | None,
) -> None:
    if expected_binding is not None and dict(payload) != expected_binding.phase_payload():
        raise ProtocolError("Stage-70 phase-01 canonical binding drifted.")
    required = set(expected_binding.phase_payload()) if expected_binding is not None else {
        "schema_version",
        "phase",
        "final_authorization_artifact_id",
        "final_authorization_hash",
        "final_authorization_content_hash",
        "authorization_protocol_hash",
        "identity_lock_hash",
        "evaluation_plan_hash",
        "reservation_content_hash",
        "reservation_identity_lock_hash",
        "target_evaluation_reservation_id",
        "target_evaluation_reservation_protocol_hash",
        "target_identity_table_hash",
        "target_cache_artifact_id",
        "target_cache_content_hash",
        "target_cache_row_order_hash",
        "target_cache_shard_sha256_by_center",
        "target_cache_rows_by_center",
        "target_cache_row_count",
        "cache_extractor_protocol_hash",
        "scoring_manifest_sha256",
        "classifier_config_hash",
        "authorized_cell_hash",
        "target_identity_hash_by_center",
        "global_target_identity_hash",
        "target_labels_opened",
        "authorization_binding_hash",
    }
    if set(payload) != required:
        raise ProtocolError("Stage-70 phase-01 authorization schema drifted.")
    unhashed = {
        key: value for key, value in payload.items() if key != "authorization_binding_hash"
    }
    if (
        payload.get("schema_version") != _PHASE_01_SCHEMA
        or payload.get("phase") != "AUTHORIZATION_COMPLETE"
        or payload.get("final_authorization_artifact_id")
        != FINAL_AUTHORIZATION_OUTPUT_ARTIFACT_ID
        or payload.get("target_cache_artifact_id") != CACHE_ARTIFACT_ID
        or payload.get("target_cache_rows_by_center")
        != dict(EXPECTED_TEST_ROWS_BY_CENTER)
        or payload.get("target_cache_row_count") != EXPECTED_TEST_ROWS
        or payload.get("target_labels_opened") is not False
        or payload.get("authorization_binding_hash") != stable_hash(unhashed)
    ):
        raise ProtocolError("Stage-70 phase-01 authorization binding drifted.")


def _validate_record_grid(
    records: tuple[Mapping[str, object], ...],
    phase_01: Mapping[str, object],
) -> None:
    observed: list[tuple[str, str, int, int]] = []
    array_keys: set[str] = set()
    authorized_records: list[dict[str, object]] = []
    for ordinal, row in enumerate(records):
        if set(row) != PREDICTION_RECORD_FIELDS:
            raise ProtocolError("Stage-70 prediction-cell metadata schema drifted.")
        try:
            key = (
                str(row["policy_id"]),
                str(row["target_center"]),
                int(row["training_seed"]),
                int(row["generation_seed"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("Stage-70 prediction cell identity is malformed.") from exc
        if row.get("ordinal") != ordinal:
            raise ProtocolError("Stage-70 prediction ordinals are not canonical.")
        if row.get("prediction_cell_hash") != stable_hash(
            {key: value for key, value in row.items() if key != "prediction_cell_hash"}
        ):
            raise ProtocolError("Stage-70 prediction-cell metadata hash drifted.")
        for field, length in (
            ("prediction_sha256", 64),
            ("probability_sha256", 64),
            ("composition_manifest_hash", 16),
            ("train_content_sha256", 64),
            ("classifier_config_hash", 16),
            ("scaler_state_hash", 16),
            ("target_row_order_hash", 16),
            ("target_identity_hash", 16),
            ("prediction_cell_hash", 16),
        ):
            value = str(row.get(field, ""))
            if len(value) != length or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ProtocolError("Stage-70 prediction-cell provenance is malformed.")
        if not isinstance(row.get("replicate_id"), str) or not row["replicate_id"]:
            raise ProtocolError("Stage-70 prediction replicate identity is malformed.")
        observed.append(key)
        authorized_records.append(
            {
                "policy_id": key[0],
                "target_center": key[1],
                "training_seed": key[2],
                "generation_seed": key[3],
                "replicate_id": str(row["replicate_id"]),
            }
        )
        for field in ("prediction_array_key", "probability_array_key"):
            value = row.get(field)
            if not isinstance(value, str) or not value or value in array_keys:
                raise ProtocolError("Stage-70 prediction array keys are invalid or reused.")
            array_keys.add(value)
        if row.get("classifier_config_hash") != phase_01.get("classifier_config_hash"):
            raise ProtocolError("Stage-70 classifier provenance drifted from authorization.")
        reused = row.get("reused_from_policy_id")
        if (key[0] == UTILITY_ARM and reused != CONTROL_ARM) or (
            key[0] != UTILITY_ARM and reused != ""
        ):
            raise ProtocolError("Stage-70 utility/control fallback provenance drifted.")
    if tuple(observed) != expected_cell_keys():
        raise ProtocolError("Stage-70 prediction cell grid/order is not exact.")
    if stable_hash(authorized_records) != phase_01.get("authorized_cell_hash"):
        raise ProtocolError("Stage-70 prediction replicate identities were not authorized.")


def _validate_target_identities(
    records: tuple[Mapping[str, object], ...],
    phase_01: Mapping[str, object],
) -> None:
    expected_counts = phase_01.get("target_cache_rows_by_center")
    expected_hashes = phase_01.get("target_identity_hash_by_center")
    if not isinstance(expected_counts, Mapping) or not isinstance(expected_hashes, Mapping):
        raise ProtocolError("Stage-70 phase-01 target identities are malformed.")
    by_center: dict[str, tuple[tuple[str, ...], tuple[int, ...], tuple[str, ...]]] = {}
    for row in records:
        center = str(row["target_center"])
        raw_ids = row.get("evaluation_row_ids")
        raw_indices = row.get("contract_row_indices")
        raw_cases = row.get("case_ids")
        if not isinstance(raw_ids, list) or not isinstance(raw_indices, list) or not isinstance(raw_cases, list):
            raise ProtocolError("Stage-70 prediction target identities are malformed.")
        try:
            identity = (
                tuple(str(value) for value in raw_ids),
                tuple(int(value) for value in raw_indices),
                tuple(str(value) for value in raw_cases),
            )
            row_count = int(row.get("row_count", -1))
        except (TypeError, ValueError) as exc:
            raise ProtocolError("Stage-70 prediction target identities are malformed.") from exc
        identity_payload = _identity_payload(*identity)
        if (
            row_count != int(expected_counts.get(center, -1))
            or any(len(values) != row_count for values in identity)
            or len(set(identity[0])) != row_count
            or len(set(identity[1])) != row_count
            or identity[1] != tuple(sorted(identity[1]))
            or any(not _is_neutral_evaluation_id(value) for value in identity[0])
            or row.get("target_identity_hash") != stable_hash(identity_payload)
            or row.get("target_identity_hash") != expected_hashes.get(center)
            or row.get("target_row_order_hash") != stable_hash(list(identity[0]))
        ):
            raise ProtocolError("Stage-70 prediction target identity coverage drifted.")
        previous = by_center.setdefault(center, identity)
        if previous != identity:
            raise ProtocolError("Stage-70 target identities disagree across policy cells.")
    if set(by_center) != set(CENTERS):
        raise ProtocolError("Stage-70 prediction target centers are incomplete.")
    global_payload = sorted(
        (
            payload
            for center in CENTERS
            for payload in _identity_payload(*by_center[center])
        ),
        key=lambda row: int(row["contract_row_index"]),
    )
    if (
        len(global_payload) != EXPECTED_TEST_ROWS
        or len({str(row["evaluation_row_id"]) for row in global_payload})
        != EXPECTED_TEST_ROWS
        or len({int(row["contract_row_index"]) for row in global_payload})
        != EXPECTED_TEST_ROWS
        or stable_hash(global_payload) != phase_01.get("global_target_identity_hash")
    ):
        raise ProtocolError("Stage-70 global prediction target identities drifted.")


def _validate_prediction_arrays(
    path: Path,
    records: tuple[Mapping[str, object], ...],
    phase_01: Mapping[str, object],
) -> tuple[VerifiedPredictionCell, ...]:
    verified_cells: list[VerifiedPredictionCell] = []
    with np.load(path, allow_pickle=False) as arrays:
        expected_members: set[str] = set()
        by_cell: dict[tuple[str, str, int, int], tuple[np.ndarray, np.ndarray, Mapping[str, object]]] = {}
        for row in records:
            prediction_key = str(row["prediction_array_key"])
            probability_key = str(row["probability_array_key"])
            expected_members.update((prediction_key, probability_key))
            try:
                predictions = np.asarray(arrays[prediction_key])
                probabilities = np.asarray(arrays[probability_key])
                row_count = int(row["row_count"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ProtocolError("Stage-70 prediction archive member is missing.") from exc
            if (
                predictions.dtype != np.dtype(np.int64)
                or probabilities.dtype != np.dtype(np.float64)
                or predictions.shape != (row_count,)
                or probabilities.shape != (row_count, 2)
                or set(int(value) for value in np.unique(predictions)) - {0, 1}
                or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0.0)
                or np.any(probabilities > 1.0)
                or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-12)
                or not np.array_equal(np.argmax(probabilities, axis=1), predictions)
                or array_sha256(predictions) != row.get("prediction_sha256")
                or array_sha256(probabilities) != row.get("probability_sha256")
            ):
                raise ProtocolError("Stage-70 prediction archive content/geometry drifted.")
            key = (
                str(row["policy_id"]),
                str(row["target_center"]),
                int(row["training_seed"]),
                int(row["generation_seed"]),
            )
            by_cell[key] = (predictions, probabilities, row)
        if set(arrays.files) != expected_members:
            raise ProtocolError("Stage-70 prediction archive is not closed-world.")
        for key in expected_cell_keys():
            predictions, probabilities, row = by_cell[key]
            if key[0] == UTILITY_ARM:
                control_predictions, control_probabilities, control_row = by_cell[
                    (CONTROL_ARM, key[1], key[2], key[3])
                ]
                provenance_fields = (
                    "train_content_sha256",
                    "classifier_config_hash",
                    "scaler_state_hash",
                    "target_row_order_hash",
                    "target_identity_hash",
                    "prediction_sha256",
                    "probability_sha256",
                )
                if (
                    any(row[field] != control_row[field] for field in provenance_fields)
                    or not np.array_equal(predictions, control_predictions)
                    or not np.array_equal(probabilities, control_probabilities)
                ):
                    raise ProtocolError("Stage-70 utility/control predictions are not exact.")
            verified_cells.append(
                VerifiedPredictionCell(
                    policy_id=key[0],
                    target_center=key[1],
                    training_seed=key[2],
                    generation_seed=key[3],
                    replicate_id=str(row["replicate_id"]),
                    evaluation_row_ids=tuple(str(value) for value in row["evaluation_row_ids"]),  # type: ignore[index]
                    contract_row_indices=tuple(int(value) for value in row["contract_row_indices"]),  # type: ignore[index]
                    case_ids=tuple(str(value) for value in row["case_ids"]),  # type: ignore[index]
                    predictions=tuple(int(value) for value in predictions.tolist()),
                    target_identity_hash=str(row["target_identity_hash"]),
                    prediction_sha256=str(row["prediction_sha256"]),
                    probability_sha256=str(row["probability_sha256"]),
                    composition_manifest_hash=str(row["composition_manifest_hash"]),
                    train_content_sha256=str(row["train_content_sha256"]),
                    classifier_config_hash=str(row["classifier_config_hash"]),
                    scaler_state_hash=str(row["scaler_state_hash"]),
                    target_row_order_hash=str(row["target_row_order_hash"]),
                    reused_from_policy_id=str(row["reused_from_policy_id"]),
                    prediction_cell_hash=str(row["prediction_cell_hash"]),
                )
            )
    if str(phase_01.get("classifier_config_hash")) != verified_cells[0].classifier_config_hash:
        raise ProtocolError("Stage-70 verified classifier provenance drifted.")
    return tuple(verified_cells)


def _validate_seal_payload(
    payload: Mapping[str, object],
    *,
    schema: str,
    phase_01_sha256: str,
    binding_hash: str,
    index_sha256: str,
    arrays_sha256: str,
    metadata_hash: str,
) -> None:
    if set(payload) != PREDICTION_SEAL_FIELDS:
        raise ProtocolError("Stage-70 prediction seal schema drifted.")
    expected = {
        "schema_version": schema,
        "phase": "PREDICTIONS_PERSISTED",
        "phase_01_sha256": phase_01_sha256,
        "authorization_binding_hash": binding_hash,
        "prediction_index_sha256": index_sha256,
        "prediction_arrays_sha256": arrays_sha256,
        "prediction_metadata_hash": metadata_hash,
        "cell_count": EXPECTED_METRIC_ROWS,
        "target_row_count": EXPECTED_TEST_ROWS,
        "classifier_fit_count": 162,
        "prediction_reuse_count": 81,
        "target_labels_opened": False,
    }
    if dict(payload) != expected:
        raise ProtocolError("Stage-70 prediction seal binding drifted.")


def _replicate_ids_from_plan(
    plan: Mapping[str, object],
) -> dict[tuple[str, str, int, int], str]:
    raw = plan.get("records")
    if not isinstance(raw, list) or len(raw) != EXPECTED_METRIC_ROWS:
        raise ProtocolError("Stage-70 final evaluation plan coverage drifted.")
    observed: dict[tuple[str, str, int, int], str] = {}
    for row in raw:
        if not isinstance(row, Mapping):
            raise ProtocolError("Stage-70 final evaluation plan row is malformed.")
        key = (
            str(row.get("policy_id")),
            str(row.get("target_center")),
            int(row.get("training_seed", -1)),
            int(row.get("generation_seed", -1)),
        )
        if key in observed:
            raise ProtocolError("Stage-70 final evaluation plan duplicates a cell.")
        observed[key] = str(row.get("replicate_id", ""))
    if set(observed) != set(expected_cell_keys()):
        raise ProtocolError("Stage-70 final evaluation plan cell grid drifted.")
    return observed


def _identity_payload(
    row_ids: Sequence[str],
    row_indices: Sequence[int],
    case_ids: Sequence[str],
) -> list[dict[str, object]]:
    if not (len(row_ids) == len(row_indices) == len(case_ids)):
        raise ProtocolError("Stage-70 target identity columns do not align.")
    return [
        {
            "evaluation_row_id": str(row_id),
            "contract_row_index": int(row_index),
            "case_id": str(case_id),
        }
        for row_id, row_index, case_id in zip(
            row_ids, row_indices, case_ids, strict=True
        )
    ]


def _verified_cell_payload(cell: VerifiedPredictionCell) -> dict[str, object]:
    return {
        "policy_id": cell.policy_id,
        "target_center": cell.target_center,
        "training_seed": cell.training_seed,
        "generation_seed": cell.generation_seed,
        "replicate_id": cell.replicate_id,
        "evaluation_row_ids": cell.evaluation_row_ids,
        "contract_row_indices": cell.contract_row_indices,
        "case_ids": cell.case_ids,
        "target_identity_hash": cell.target_identity_hash,
        "prediction_sha256": cell.prediction_sha256,
        "probability_sha256": cell.probability_sha256,
        "composition_manifest_hash": cell.composition_manifest_hash,
        "train_content_sha256": cell.train_content_sha256,
        "classifier_config_hash": cell.classifier_config_hash,
        "scaler_state_hash": cell.scaler_state_hash,
        "target_row_order_hash": cell.target_row_order_hash,
        "reused_from_policy_id": cell.reused_from_policy_id,
        "prediction_cell_hash": cell.prediction_cell_hash,
    }


def _prediction_paths(root: Path) -> dict[str, Path]:
    return {
        "phase_01": root / "reports/phase_01_authorization_complete.json",
        "index": root / "manifests/prediction_index.json",
        "seal": root / "manifests/prediction_seal.json",
        "arrays": root / "arrays/target_predictions.npz",
        "phase_02": root / "reports/phase_02_predictions_persisted.json",
    }


def _safe_directory(path: Path, role: str) -> Path:
    if not path.is_dir() or path.is_symlink():
        raise ProtocolError(f"Stage-70 {role} root is missing or a symlink: {path}.")
    return path


def _safe_file(path: Path, role: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise ProtocolError(f"Stage-70 {role} is missing or a symlink: {path}.")
    return path


def _resolved_path(path: Path) -> Path:
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise ProtocolError(f"Stage-70 input path is unavailable: {path}.") from exc


def _json(path: Path) -> dict[str, object]:
    _safe_file(path, "JSON member")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read Stage-70 prediction artifact: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Stage-70 prediction JSON must be an object: {path}.")
    return payload


def _sha256_file(path: Path) -> str:
    _safe_file(path, "hashed member")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ProtocolError(f"Cannot hash Stage-70 prediction artifact: {path}.") from exc
    return digest.hexdigest()


def _string_mapping(value: object, role: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Stage-70 {role} are malformed.")
    return {str(key): str(item) for key, item in value.items()}


def _integer_mapping(value: object, role: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Stage-70 {role} are malformed.")
    try:
        return {str(key): int(item) for key, item in value.items()}
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"Stage-70 {role} are malformed.") from exc


def _is_neutral_evaluation_id(value: str) -> bool:
    suffix = value[5:] if value.startswith("eval_") else ""
    return len(suffix) == 64 and all(character in "0123456789abcdef" for character in suffix)


def _is_hash(value: object) -> bool:
    rendered = str(value)
    return len(rendered) in {16, 64} and all(
        character in "0123456789abcdef" for character in rendered
    )


def _is_sha256(value: object) -> bool:
    rendered = str(value)
    return len(rendered) == 64 and all(
        character in "0123456789abcdef" for character in rendered
    )


__all__ = (
    "PREDICTION_INDEX_FIELDS",
    "PREDICTION_RECORD_FIELDS",
    "PREDICTION_SEAL_FIELDS",
    "PredictionSealBinding",
    "TargetIdentity",
    "VerifiedPredictionArtifact",
    "VerifiedPredictionCell",
    "expected_cell_keys",
    "load_canonical_prediction_seal_binding",
    "prediction_paths",
    "prediction_record_payload",
    "validate_authorization_phase",
    "validate_persisted_prediction_pass",
    "verify_persisted_prediction_artifact",
)
