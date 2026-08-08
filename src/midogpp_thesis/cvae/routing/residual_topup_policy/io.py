"""Strict CSV and fresh-surface attestation loading for Stage 60."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from ...generation.runner import read_generation_lock
from ...protocol import ProtocolError
from ..policy import read_policy_lock as read_equal_union_policy_lock
from ..residual_topup.hashing import canonical_sha256
from .config import (
    DATASET_FAMILY,
    EXPERIMENT_ID,
    FEATURE_BACKBONE,
    PROXY_SURFACE_ARTIFACT_ID,
    REPRESENTATION_ID,
    ResidualTopupPolicyLockConfig,
)
from .contracts import (
    GLOBAL_PSEUDOQUERY_ROLE,
    PROXY_ENERGY_SEMANTICS,
    TARGET_SUPPORT_ROLE,
    FreshProxyScoreRow,
)


PROXY_SCORE_COLUMNS = (
    "outer_target",
    "query_role",
    "query_center",
    "case_id",
    "candidate_source",
    "training_seed",
    "proxy_energy",
    "labels_consumed",
    "evaluation_overlap",
    "source_expert_updated",
    "proxy_energy_semantics",
)
ATTESTATION_SCHEMA_VERSION = (
    "midogpp_residual_topup_fresh_proxy_surface_attestation_v1"
)
ATTESTATION_KEYS = {
    "schema_version",
    "artifact_id",
    "authorized_consumer_experiment_id",
    "dataset_family",
    "feature_backbone",
    "representation_id",
    "reservation_id",
    "proxy_surface_hash",
    "query_shard_hashes",
    "fresh_surface",
    "previously_consumed",
    "consumed_stage70_used",
    "consumed_stage90_used",
    "labels_present",
    "labels_consumed",
    "evaluation_labels_opened",
    "target_evaluation_used",
    "source_experts_updated",
    "pseudoquery_support_case_overlap_count",
    "pseudoquery_evaluation_case_overlap_count",
    "support_evaluation_case_overlap_count",
    "pseudoquery_case_ids_by_center",
    "support_case_ids_by_target",
    "evaluation_case_ids_by_target",
    "proxy_score_row_count",
    "input_hashes",
    "attestation_hash",
}
INPUT_HASH_KEYS = {
    "expert_bank_lock_hash",
    "generation_lock_hash",
    "equal_union_policy_lock_hash",
    "expert_bank_index_sha256",
    "generation_lock_sha256",
    "equal_union_policy_lock_sha256",
    "proxy_score_table_sha256",
}


@dataclass(frozen=True)
class FreshSurfaceAttestation:
    reservation_id: str
    proxy_surface_hash: str
    query_shard_hashes: Mapping[str, str]
    pseudoquery_case_ids_by_center: Mapping[str, tuple[str, ...]]
    support_case_ids_by_target: Mapping[str, tuple[str, ...]]
    evaluation_case_ids_by_target: Mapping[str, tuple[str, ...]]
    proxy_score_row_count: int
    input_hashes: Mapping[str, str]
    attestation_hash: str
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "query_shard_hashes",
            MappingProxyType(dict(self.query_shard_hashes)),
        )
        object.__setattr__(
            self,
            "pseudoquery_case_ids_by_center",
            MappingProxyType(dict(self.pseudoquery_case_ids_by_center)),
        )
        object.__setattr__(
            self,
            "support_case_ids_by_target",
            MappingProxyType(dict(self.support_case_ids_by_target)),
        )
        object.__setattr__(
            self,
            "evaluation_case_ids_by_target",
            MappingProxyType(dict(self.evaluation_case_ids_by_target)),
        )
        object.__setattr__(self, "input_hashes", MappingProxyType(dict(self.input_hashes)))
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))

    def to_payload(self) -> dict[str, object]:
        return dict(self.payload)


@dataclass(frozen=True)
class ValidatedFreshProxyInputs:
    rows: tuple[FreshProxyScoreRow, ...]
    attestation: FreshSurfaceAttestation
    proxy_score_table_sha256: str
    attestation_file_sha256: str
    upstream_file_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "upstream_file_sha256", MappingProxyType(dict(self.upstream_file_sha256))
        )


def read_fresh_proxy_score_rows(path: str | Path) -> tuple[FreshProxyScoreRow, ...]:
    score_path = Path(path)
    if not score_path.is_file():
        raise ProtocolError(
            "Fresh proxy score table is absent; the planned Stage-60 policy must remain blocked."
        )
    with score_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PROXY_SCORE_COLUMNS:
            raise ProtocolError("Fresh proxy score CSV columns drifted.")
        rows: list[FreshProxyScoreRow] = []
        for row_number, raw in enumerate(reader, start=2):
            if None in raw or set(raw) != set(PROXY_SCORE_COLUMNS):
                raise ProtocolError(
                    f"Fresh proxy score CSV row {row_number} is malformed."
                )
            try:
                rows.append(
                    FreshProxyScoreRow(
                        outer_target=str(raw["outer_target"]),
                        query_role=str(raw["query_role"]),
                        query_center=str(raw["query_center"]),
                        case_id=str(raw["case_id"]),
                        candidate_source=str(raw["candidate_source"]),
                        training_seed=int(str(raw["training_seed"])),
                        proxy_energy=float(str(raw["proxy_energy"])),
                        labels_consumed=_strict_false(
                            raw["labels_consumed"],
                            field="labels_consumed",
                        ),
                        evaluation_overlap=_strict_false(
                            raw["evaluation_overlap"],
                            field="evaluation_overlap",
                        ),
                        source_expert_updated=_strict_false(
                            raw["source_expert_updated"],
                            field="source_expert_updated",
                        ),
                        proxy_energy_semantics=str(raw["proxy_energy_semantics"]),
                    )
                )
            except ProtocolError:
                raise
            except (TypeError, ValueError, OverflowError) as exc:
                raise ProtocolError(
                    f"Fresh proxy score CSV row {row_number} is invalid."
                ) from exc
    if not rows:
        raise ProtocolError("Fresh proxy score CSV is empty.")
    return tuple(rows)


def read_fresh_surface_attestation(
    path: str | Path,
    *,
    centers: Sequence[str],
) -> FreshSurfaceAttestation:
    attestation_path = Path(path)
    if not attestation_path.is_file():
        raise ProtocolError(
            "Fresh proxy-surface attestation is absent; the planned Stage-60 policy must remain blocked."
        )
    payload = _json(attestation_path)
    if set(payload) != ATTESTATION_KEYS:
        raise ProtocolError("Fresh proxy-surface attestation keys drifted.")
    unhashed = {key: value for key, value in payload.items() if key != "attestation_hash"}
    observed_hash = payload.get("attestation_hash")
    if not isinstance(observed_hash, str) or observed_hash != canonical_sha256(unhashed):
        raise ProtocolError("Fresh proxy-surface attestation hash is invalid.")
    required = {
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "artifact_id": PROXY_SURFACE_ARTIFACT_ID,
        "authorized_consumer_experiment_id": EXPERIMENT_ID,
        "dataset_family": DATASET_FAMILY,
        "feature_backbone": FEATURE_BACKBONE,
        "representation_id": REPRESENTATION_ID,
        "fresh_surface": True,
        "previously_consumed": False,
        "consumed_stage70_used": False,
        "consumed_stage90_used": False,
        "labels_present": False,
        "labels_consumed": False,
        "evaluation_labels_opened": False,
        "target_evaluation_used": False,
        "source_experts_updated": False,
        "pseudoquery_support_case_overlap_count": 0,
        "pseudoquery_evaluation_case_overlap_count": 0,
        "support_evaluation_case_overlap_count": 0,
    }
    mismatch = [
        key for key, value in required.items() if payload.get(key) != value
    ]
    if mismatch:
        raise ProtocolError(
            "Fresh proxy-surface attestation protocol failed: "
            + ", ".join(sorted(mismatch))
            + "."
        )
    reservation_id = payload.get("reservation_id")
    if (
        not isinstance(reservation_id, str)
        or not reservation_id
        or reservation_id.strip() != reservation_id
    ):
        raise ProtocolError("Fresh proxy-surface reservation identity is invalid.")
    canonical_centers = tuple(str(center) for center in centers)
    proxy_surface_hash = _lower_hex(
        payload.get("proxy_surface_hash"),
        length=16,
        label="proxy-surface hash",
    )
    raw_shard_hashes = payload.get("query_shard_hashes")
    expected_shard_keys = {
        f"{target}::{GLOBAL_PSEUDOQUERY_ROLE}::{query}"
        for target in canonical_centers
        for query in canonical_centers
        if query != target
    }.union(
        {
            f"{target}::{TARGET_SUPPORT_ROLE}::{target}"
            for target in canonical_centers
        }
    )
    if (
        not isinstance(raw_shard_hashes, Mapping)
        or set(map(str, raw_shard_hashes)) != expected_shard_keys
    ):
        raise ProtocolError("Fresh query-shard hash grid is incomplete.")
    query_shard_hashes = {
        key: _lower_hex(
            raw_shard_hashes.get(key),
            length=16,
            label="query-shard hash",
        )
        for key in sorted(expected_shard_keys)
    }
    if len(set(query_shard_hashes.values())) != len(query_shard_hashes):
        raise ProtocolError("Fresh query-shard hashes must be distinct by H/role/q.")
    pseudoquery = _case_mapping(
        payload.get("pseudoquery_case_ids_by_center"),
        centers=canonical_centers,
        label="pseudoquery",
    )
    support = _case_mapping(
        payload.get("support_case_ids_by_target"),
        centers=canonical_centers,
        label="support",
    )
    evaluation = _case_mapping(
        payload.get("evaluation_case_ids_by_target"),
        centers=canonical_centers,
        label="evaluation",
    )
    if len({len(values) for values in pseudoquery.values()}) != 1:
        raise ProtocolError("Fresh pseudoquery case coverage must be equal by center.")
    pseudoquery_cases = {case for values in pseudoquery.values() for case in values}
    support_cases = {case for values in support.values() for case in values}
    evaluation_cases = {case for values in evaluation.values() for case in values}
    if (
        pseudoquery_cases & support_cases
        or pseudoquery_cases & evaluation_cases
        or support_cases & evaluation_cases
    ):
        raise ProtocolError(
            "Fresh pseudoquery, support, and evaluation cases must be globally disjoint."
        )
    row_count = payload.get("proxy_score_row_count")
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count <= 0:
        raise ProtocolError("Fresh proxy score row count is invalid.")
    input_hashes_raw = payload.get("input_hashes")
    if not isinstance(input_hashes_raw, Mapping) or set(input_hashes_raw) != INPUT_HASH_KEYS:
        raise ProtocolError("Fresh proxy-surface input-hash grid drifted.")
    input_hashes: dict[str, str] = {}
    for key in sorted(INPUT_HASH_KEYS):
        value = input_hashes_raw.get(key)
        expected_length = 64 if key.endswith("_sha256") else 16
        if (
            not isinstance(value, str)
            or len(value) != expected_length
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ProtocolError("Fresh proxy-surface input hash is invalid.")
        input_hashes[key] = value
    return FreshSurfaceAttestation(
        reservation_id=reservation_id,
        proxy_surface_hash=proxy_surface_hash,
        query_shard_hashes=query_shard_hashes,
        pseudoquery_case_ids_by_center=pseudoquery,
        support_case_ids_by_target=support,
        evaluation_case_ids_by_target=evaluation,
        proxy_score_row_count=row_count,
        input_hashes=input_hashes,
        attestation_hash=observed_hash,
        payload=payload,
    )


def load_validated_fresh_proxy_inputs(
    config: ResidualTopupPolicyLockConfig,
) -> ValidatedFreshProxyInputs:
    """Load and cryptographically bind all fresh Stage-60 proxy inputs."""

    required = (
        config.proxy_score_table_path,
        config.proxy_attestation_path,
        config.expert_bank_root / "manifests/expert_bank_index.json",
        config.generation_lock_root / "manifests/generation_lock.json",
        config.equal_union_policy_root / "manifests/policy_lock.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ProtocolError(
            "Fresh Stage-60 policy inputs are absent; planned artifact remains blocked: "
            + ", ".join(missing)
            + "."
        )
    rows = read_fresh_proxy_score_rows(config.proxy_score_table_path)
    attestation = read_fresh_surface_attestation(
        config.proxy_attestation_path,
        centers=config.centers,
    )
    score_hash = _sha256_file(config.proxy_score_table_path)
    if (
        len(rows) != attestation.proxy_score_row_count
        or score_hash != attestation.input_hashes["proxy_score_table_sha256"]
    ):
        raise ProtocolError("Fresh proxy score table does not match its attestation.")
    upstream = _validate_upstream_locks(config, attestation=attestation)
    _validate_attested_case_grid(rows, config=config, attestation=attestation)
    return ValidatedFreshProxyInputs(
        rows=rows,
        attestation=attestation,
        proxy_score_table_sha256=score_hash,
        attestation_file_sha256=_sha256_file(config.proxy_attestation_path),
        upstream_file_sha256=upstream,
    )


def _validate_upstream_locks(
    config: ResidualTopupPolicyLockConfig,
    *,
    attestation: FreshSurfaceAttestation,
) -> dict[str, str]:
    bank_path = config.expert_bank_root / "manifests/expert_bank_index.json"
    generation_path = config.generation_lock_root / "manifests/generation_lock.json"
    equal_path = config.equal_union_policy_root / "manifests/policy_lock.json"
    bank = _json(bank_path)
    observed_bank_hash = bank.get("bank_lock_hash")
    if observed_bank_hash != config.expected_bank_lock_hash:
        raise ProtocolError("Fresh proxy surface bank-lock binding failed.")
    generation = read_generation_lock(generation_path)
    equal = read_equal_union_policy_lock(equal_path)
    if (
        generation.generation_lock_hash != config.expected_generation_lock_hash
        or equal.policy_lock_hash != config.expected_equal_union_policy_lock_hash
    ):
        raise ProtocolError("Fresh proxy surface upstream lock identity drifted.")
    expected_semantic = {
        "expert_bank_lock_hash": config.expected_bank_lock_hash,
        "generation_lock_hash": config.expected_generation_lock_hash,
        "equal_union_policy_lock_hash": config.expected_equal_union_policy_lock_hash,
    }
    if any(
        attestation.input_hashes[key] != value
        for key, value in expected_semantic.items()
    ):
        raise ProtocolError("Fresh proxy-surface semantic input hashes drifted.")
    file_hashes = {
        "expert_bank_index_sha256": _sha256_file(bank_path),
        "generation_lock_sha256": _sha256_file(generation_path),
        "equal_union_policy_lock_sha256": _sha256_file(equal_path),
    }
    if any(
        attestation.input_hashes[key] != value for key, value in file_hashes.items()
    ):
        raise ProtocolError("Fresh proxy-surface upstream file hashes drifted.")
    return file_hashes


def _validate_attested_case_grid(
    rows: Sequence[FreshProxyScoreRow],
    *,
    config: ResidualTopupPolicyLockConfig,
    attestation: FreshSurfaceAttestation,
) -> None:
    observed_targets = {row.outer_target for row in rows}
    if observed_targets != set(config.centers):
        raise ProtocolError("Fresh proxy score outer-target grid is incomplete.")
    observed_global: dict[tuple[str, str], set[str]] = {}
    observed_support: dict[str, set[str]] = {}
    evaluation_cases = {
        case
        for values in attestation.evaluation_case_ids_by_target.values()
        for case in values
    }
    for target in config.centers:
        for query in config.centers:
            if query != target:
                observed_global[(target, query)] = set()
        observed_support[target] = set()
    for row in rows:
        if row.case_id in evaluation_cases:
            raise ProtocolError("Fresh proxy score row overlaps target evaluation cases.")
        if row.query_role == GLOBAL_PSEUDOQUERY_ROLE:
            try:
                observed_global[(row.outer_target, row.query_center)].add(row.case_id)
            except KeyError as exc:
                raise ProtocolError("Fresh proxy pseudoquery geometry is invalid.") from exc
        elif row.query_role == TARGET_SUPPORT_ROLE:
            observed_support[row.outer_target].add(row.case_id)
        else:  # FreshProxyScoreRow already rejects this; retain fail-closed locality.
            raise ProtocolError("Fresh proxy score role drifted.")
    for (target, query), cases in observed_global.items():
        if cases != set(attestation.pseudoquery_case_ids_by_center[query]):
            raise ProtocolError(
                f"Fresh pseudoquery case grid drifted for H={target}, q={query}."
            )
    for target, cases in observed_support.items():
        if cases != set(attestation.support_case_ids_by_target[target]):
            raise ProtocolError(f"Fresh target-support case grid drifted for H={target}.")


def _case_mapping(
    raw: object,
    *,
    centers: tuple[str, ...],
    label: str,
) -> dict[str, tuple[str, ...]]:
    if not isinstance(raw, Mapping) or set(map(str, raw)) != set(centers):
        raise ProtocolError(f"Fresh {label} case mapping is incomplete.")
    result: dict[str, tuple[str, ...]] = {}
    all_cases: set[str] = set()
    for center in centers:
        values = raw.get(center)
        if not isinstance(values, list) or not values:
            raise ProtocolError(f"Fresh {label} cases must be nonempty lists.")
        cases = tuple(str(value) for value in values)
        if (
            cases != tuple(sorted(cases))
            or any(not value or value.strip() != value for value in cases)
            or len(set(cases)) != len(cases)
            or all_cases.intersection(cases)
        ):
            raise ProtocolError(f"Fresh {label} case identities are invalid.")
        all_cases.update(cases)
        result[center] = cases
    return result


def _strict_false(value: object, *, field: str) -> bool:
    if value != "false":
        raise ProtocolError(f"Fresh proxy score {field} must be the literal false.")
    return False


def _lower_hex(value: object, *, length: int, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProtocolError(f"Fresh {label} is invalid.")
    return value


def _json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Fresh Stage-60 JSON is unreadable: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Fresh Stage-60 JSON must be an object: {path}.")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = (
    "ATTESTATION_SCHEMA_VERSION",
    "INPUT_HASH_KEYS",
    "PROXY_SCORE_COLUMNS",
    "FreshSurfaceAttestation",
    "ValidatedFreshProxyInputs",
    "load_validated_fresh_proxy_inputs",
    "read_fresh_proxy_score_rows",
    "read_fresh_surface_attestation",
)
