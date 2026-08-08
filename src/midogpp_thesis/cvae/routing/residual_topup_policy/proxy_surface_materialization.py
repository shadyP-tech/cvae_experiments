"""Registered CSV and attestation materialization for fresh proxy surfaces."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ..residual_topup.hashing import canonical_sha256
from .contracts import FIXED_TRAINING_SEEDS, FreshProxyScoreRow
from .proxy_surface_checkpoints import atomic_write_json
from .proxy_surface_contracts import (
    EXPECTED_QUERY_SHARD_COUNT,
    FRESH_SURFACE_ATTESTATION_SCHEMA_VERSION,
    PROXY_SCORE_COLUMNS,
    FreshProxyScoreSurface,
    FreshQueryShard,
    MaterializedFreshProxyInputs,
)
from .proxy_surface_validation import (
    derive_case_grids,
    query_shard_attestation_key,
    validate_fresh_proxy_score_surface,
    validate_query_shards,
)

if TYPE_CHECKING:
    from .config import ResidualTopupPolicyLockConfig


def materialize_fresh_proxy_inputs(
    surface: FreshProxyScoreSurface,
    *,
    shards: Iterable[FreshQueryShard],
    config: "ResidualTopupPolicyLockConfig",
    reservation_id: str,
) -> MaterializedFreshProxyInputs:
    """Write the registered Stage-60 CSV and fresh-surface attestation only.

    Upstream semantic hashes are taken from the validated policy configuration,
    checked against the three lock files, and paired with freshly computed
    SHA-256 file hashes.  Query/support/evaluation case grids are derived from
    the already validated shards rather than accepted as caller assertions.
    """

    from .config import ResidualTopupPolicyLockConfig

    if not isinstance(surface, FreshProxyScoreSurface):
        raise ProtocolError("Fresh proxy materialization requires a score surface.")
    if not isinstance(config, ResidualTopupPolicyLockConfig):
        raise ProtocolError(
            "Fresh proxy materialization requires the Stage-60 config."
        )
    reservation = str(reservation_id)
    if not reservation or reservation.strip() != reservation:
        raise ProtocolError("Fresh proxy reservation identity is invalid.")
    if (
        tuple(config.centers) != tuple(CENTERS)
        or tuple(config.training_seeds) != tuple(FIXED_TRAINING_SEEDS)
        or surface.labels_consumed is not False
        or surface.source_experts_updated is not False
        or surface.expert_bank_binding_hash != config.expected_bank_lock_hash
    ):
        raise ProtocolError("Fresh proxy materialization identity drifted.")

    shard_tuple = validate_query_shards(tuple(shards))
    rows = validate_fresh_proxy_score_surface(surface.rows, shards=shard_tuple)
    if rows != surface.rows:
        raise ProtocolError("Fresh proxy surface row order is not canonical.")

    root = config.proxy_surface_root.expanduser().resolve()
    score_path = root / "tables/proxy_scores.csv"
    attestation_path = root / "manifests/fresh_surface_attestation.json"
    if (
        config.proxy_score_table_path.expanduser().resolve() != score_path
        or config.proxy_attestation_path.expanduser().resolve() != attestation_path
    ):
        raise ProtocolError(
            "Fresh proxy config must register the canonical table and attestation paths."
        )

    bank_path = config.expert_bank_root / "manifests/expert_bank_index.json"
    generation_path = config.generation_lock_root / "manifests/generation_lock.json"
    equal_policy_path = config.equal_union_policy_root / "manifests/policy_lock.json"
    upstream_specs = (
        (bank_path, "bank_lock_hash", config.expected_bank_lock_hash),
        (
            generation_path,
            "generation_lock_hash",
            config.expected_generation_lock_hash,
        ),
        (
            equal_policy_path,
            "policy_lock_hash",
            config.expected_equal_union_policy_lock_hash,
        ),
    )
    for path, field, expected in upstream_specs:
        require_lower_hex(expected, length=16, label=field)
        payload = read_json_object(path, label="fresh proxy upstream lock")
        if payload.get(field) != expected:
            raise ProtocolError(
                f"Fresh proxy upstream semantic hash drifted for {field}."
            )

    atomic_write_proxy_score_csv(score_path, rows)
    score_sha256 = sha256_file(score_path)
    pseudoquery_cases, support_cases, evaluation_cases = derive_case_grids(
        shard_tuple
    )
    input_hashes = {
        "expert_bank_lock_hash": config.expected_bank_lock_hash,
        "generation_lock_hash": config.expected_generation_lock_hash,
        "equal_union_policy_lock_hash": (
            config.expected_equal_union_policy_lock_hash
        ),
        "expert_bank_index_sha256": sha256_file(bank_path),
        "generation_lock_sha256": sha256_file(generation_path),
        "equal_union_policy_lock_sha256": sha256_file(equal_policy_path),
        "proxy_score_table_sha256": score_sha256,
    }
    query_shard_hashes = {
        query_shard_attestation_key(shard): shard.shard_hash
        for shard in shard_tuple
    }
    if len(query_shard_hashes) != EXPECTED_QUERY_SHARD_COUNT:
        raise ProtocolError("Fresh proxy shard-provenance coverage drifted.")
    require_lower_hex(
        surface.surface_hash,
        length=16,
        label="proxy_surface_hash",
    )
    unhashed: dict[str, object] = {
        "schema_version": FRESH_SURFACE_ATTESTATION_SCHEMA_VERSION,
        "artifact_id": config.proxy_surface_artifact_id,
        "authorized_consumer_experiment_id": config.experiment_id,
        "dataset_family": config.protocol["dataset_family"],
        "feature_backbone": config.protocol["feature_backbone"],
        "representation_id": config.protocol["feature_frame"],
        "reservation_id": reservation,
        "proxy_surface_hash": surface.surface_hash,
        "query_shard_hashes": query_shard_hashes,
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
        "pseudoquery_case_ids_by_center": pseudoquery_cases,
        "support_case_ids_by_target": support_cases,
        "evaluation_case_ids_by_target": evaluation_cases,
        "proxy_score_row_count": len(rows),
        "input_hashes": input_hashes,
    }
    attestation_hash = canonical_sha256(unhashed)
    atomic_write_json(
        attestation_path,
        {**unhashed, "attestation_hash": attestation_hash},
    )
    return MaterializedFreshProxyInputs(
        proxy_score_table_path=score_path,
        proxy_attestation_path=attestation_path,
        proxy_score_table_sha256=score_sha256,
        proxy_attestation_sha256=sha256_file(attestation_path),
        attestation_hash=attestation_hash,
        row_count=len(rows),
    )


def atomic_write_proxy_score_csv(
    path: Path,
    rows: Sequence[FreshProxyScoreRow],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(PROXY_SCORE_COLUMNS),
                lineterminator="\n",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "outer_target": row.outer_target,
                        "query_role": row.query_role,
                        "query_center": row.query_center,
                        "case_id": row.case_id,
                        "candidate_source": row.candidate_source,
                        "training_seed": str(row.training_seed),
                        "proxy_energy": repr(row.proxy_energy),
                        "labels_consumed": "false",
                        "evaluation_overlap": "false",
                        "source_expert_updated": "false",
                        "proxy_energy_semantics": row.proxy_energy_semantics,
                    }
                )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read {label}: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"{label.capitalize()} must be a JSON object: {path}.")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ProtocolError(f"Cannot hash fresh proxy input: {path}.") from exc
    return digest.hexdigest()


def require_lower_hex(value: object, *, length: int, label: str) -> str:
    text = str(value)
    if (
        len(text) != length
        or any(character not in "0123456789abcdef" for character in text)
    ):
        raise ProtocolError(f"Fresh proxy {label} is not canonical lower hex.")
    return text


__all__ = ("materialize_fresh_proxy_inputs",)
