"""Configuration and authorization binding for the Stage-70 test cache."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import yaml

from midogpp_thesis.data.contract.stage70_target_evaluation.contracts import (
    AUTHORIZED_CONSUMER_EXPERIMENT_ID,
    CANONICAL_MANIFEST_SHA256,
    ELIGIBLE_CENTERS,
    EXPECTED_TEST_ROWS,
    EXPECTED_TEST_ROWS_BY_CENTER,
    FRESH_EVIDENCE,
    PURPOSE,
    TargetEvaluationReservation,
    semantic_sha256,
)

from .contracts import (
    CACHE_ARTIFACT_ID,
    CACHE_NAME,
    CANONICAL_OUTPUT_RELATIVE_ROOT,
    Stage70TestCacheError,
    stage70_extractor_protocol_hash,
)


CONFIG_TOP_LEVEL_FIELDS = frozenset(
    {"cache", "inputs", "authorization_binding", "run", "protocol"}
)
CONFIG_CACHE_FIELDS = frozenset({"name", "artifact_id", "root"})
CONFIG_INPUT_FIELDS = frozenset(
    {"repo_root", "manifest_path", "target_evaluation_reservation_path"}
)
CONFIG_AUTHORIZATION_BINDING_FIELDS = frozenset(
    {
        "scoring_manifest_sha256",
        "target_evaluation_reservation_id",
        "target_evaluation_reservation_protocol_hash",
        "cache_extractor_protocol_hash",
    }
)
CONFIG_RUN_FIELDS = frozenset(
    {
        "eligible_centers",
        "expected_row_count",
        "expected_rows_by_center",
        "experiment_seed",
        "device",
        "batch_size",
    }
)
CONFIG_PROTOCOL_FIELDS = frozenset(
    {"authorized_consumer_experiment_id", "purpose", "fresh_evidence"}
)


@dataclass(frozen=True)
class Stage70TestCacheConfig:
    """Filesystem binding of an already-authorized extraction protocol."""

    repo_root: Path
    manifest_path: Path
    cache_root: Path
    expected_manifest_sha256: str
    expected_reservation_id: str
    expected_reservation_protocol_hash: str
    reservation_path: Path | None = None
    expected_cache_extractor_protocol_hash: str = field(
        default_factory=stage70_extractor_protocol_hash
    )
    cache_name: str = CACHE_NAME
    cache_artifact_id: str = CACHE_ARTIFACT_ID
    authorized_consumer_experiment_id: str = AUTHORIZED_CONSUMER_EXPERIMENT_ID
    purpose: str = PURPOSE
    fresh_evidence: bool = FRESH_EVIDENCE
    eligible_centers: tuple[str, ...] = ELIGIBLE_CENTERS
    expected_row_count: int = EXPECTED_TEST_ROWS
    expected_rows_by_center: Mapping[str, int] = field(
        default_factory=lambda: dict(EXPECTED_TEST_ROWS_BY_CENTER)
    )
    experiment_seed: int = 42
    device: str = "cuda"
    batch_size: int = 32
    canonical_coverage_required: bool = True

    @property
    def config_protocol_hash(self) -> str:
        return semantic_sha256(stage70_cache_config_protocol(self))


def make_stage70_test_cache_config(
    *,
    cache_root: str | Path,
    repo_root: str | Path,
    manifest_path: str | Path,
    reservation: TargetEvaluationReservation,
    reservation_path: str | Path | None = None,
    batch_size: int = 32,
    device: str = "cuda",
    allow_test_fixture: bool = False,
) -> Stage70TestCacheConfig:
    """Create a config bound to one validated reservation identity."""

    config = Stage70TestCacheConfig(
        repo_root=Path(repo_root),
        manifest_path=Path(manifest_path),
        cache_root=Path(cache_root),
        reservation_path=None if reservation_path is None else Path(reservation_path),
        expected_manifest_sha256=reservation.manifest_sha256,
        expected_reservation_id=reservation.reservation_id,
        expected_reservation_protocol_hash=reservation.protocol_hash,
        eligible_centers=tuple(reservation.rows_by_center),
        expected_row_count=reservation.row_count,
        expected_rows_by_center=dict(reservation.rows_by_center),
        batch_size=batch_size,
        device=device,
        canonical_coverage_required=not allow_test_fixture,
    )
    validate_stage70_test_cache_config(config, expected_reservation=reservation)
    return config


def load_stage70_test_cache_config(path: str | Path) -> Stage70TestCacheConfig:
    """Load the canonical production config; fixture overrides are not accepted."""

    config_path = Path(path)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise Stage70TestCacheError(
            f"Cannot read Stage-70 test-cache config: {config_path}."
        ) from exc
    if not isinstance(payload, Mapping):
        raise Stage70TestCacheError("Stage-70 test-cache config must be a mapping.")
    _require_exact_keys(
        payload,
        set(CONFIG_TOP_LEVEL_FIELDS),
        "top-level config",
    )
    cache = _mapping(payload, "cache")
    inputs = _mapping(payload, "inputs")
    binding = _mapping(payload, "authorization_binding")
    run = _mapping(payload, "run")
    protocol = _mapping(payload, "protocol")
    _require_exact_keys(cache, set(CONFIG_CACHE_FIELDS), "cache")
    _require_exact_keys(
        inputs,
        set(CONFIG_INPUT_FIELDS),
        "inputs",
    )
    _require_exact_keys(
        binding,
        set(CONFIG_AUTHORIZATION_BINDING_FIELDS),
        "authorization binding",
    )
    _require_exact_keys(
        run,
        set(CONFIG_RUN_FIELDS),
        "run",
    )
    _require_exact_keys(
        protocol,
        set(CONFIG_PROTOCOL_FIELDS),
        "protocol",
    )
    repo_root = Path(str(inputs["repo_root"]))
    raw_reservation_root = inputs["target_evaluation_reservation_path"]
    if not isinstance(raw_reservation_root, str) or not raw_reservation_root.strip():
        raise Stage70TestCacheError(
            "Stage-70 target-evaluation reservation path must name an artifact root."
        )
    config = Stage70TestCacheConfig(
        repo_root=repo_root,
        manifest_path=_resolve(repo_root, str(inputs["manifest_path"])),
        cache_root=_resolve(repo_root, str(cache["root"])),
        reservation_path=_resolve(
            repo_root,
            raw_reservation_root,
        ),
        expected_manifest_sha256=str(binding["scoring_manifest_sha256"]),
        expected_reservation_id=str(binding["target_evaluation_reservation_id"]),
        expected_reservation_protocol_hash=str(
            binding["target_evaluation_reservation_protocol_hash"]
        ),
        expected_cache_extractor_protocol_hash=str(
            binding["cache_extractor_protocol_hash"]
        ),
        cache_name=str(cache["name"]),
        cache_artifact_id=str(cache["artifact_id"]),
        authorized_consumer_experiment_id=str(
            protocol["authorized_consumer_experiment_id"]
        ),
        purpose=str(protocol["purpose"]),
        fresh_evidence=protocol["fresh_evidence"],
        eligible_centers=tuple(str(value) for value in run["eligible_centers"]),
        expected_row_count=int(run["expected_row_count"]),
        expected_rows_by_center={
            str(key): int(value)
            for key, value in _mapping(run, "expected_rows_by_center").items()
        },
        experiment_seed=int(run["experiment_seed"]),
        device=str(run["device"]),
        batch_size=int(run["batch_size"]),
        canonical_coverage_required=True,
    )
    validate_stage70_test_cache_config(config)
    return config


def validate_stage70_test_cache_config(
    config: Stage70TestCacheConfig,
    *,
    expected_reservation: TargetEvaluationReservation | None = None,
) -> None:
    """Fail closed on cache, authorization, coverage, or protocol drift."""

    if (
        config.cache_name != CACHE_NAME
        or config.cache_artifact_id != CACHE_ARTIFACT_ID
        or config.authorized_consumer_experiment_id
        != AUTHORIZED_CONSUMER_EXPERIMENT_ID
        or config.purpose != PURPOSE
        or config.fresh_evidence is not FRESH_EVIDENCE
        or config.expected_cache_extractor_protocol_hash
        != stage70_extractor_protocol_hash()
    ):
        raise Stage70TestCacheError("Stage-70 test-cache protocol identity drifted.")
    if not config.device or config.batch_size <= 0 or config.experiment_seed != 42:
        raise Stage70TestCacheError("Stage-70 test-cache execution policy drifted.")
    counts = {str(key): int(value) for key, value in config.expected_rows_by_center.items()}
    if (
        tuple(counts) != config.eligible_centers
        or tuple(center for center in ELIGIBLE_CENTERS if center in counts)
        != config.eligible_centers
        or not counts
        or any(count <= 0 for count in counts.values())
        or config.expected_row_count != sum(counts.values())
    ):
        raise Stage70TestCacheError("Stage-70 test-cache row coverage is invalid.")
    if config.canonical_coverage_required:
        expected_root = (config.repo_root / CANONICAL_OUTPUT_RELATIVE_ROOT).resolve()
        staging_root = expected_root.with_name(f".{expected_root.name}.staging")
        if (
            config.reservation_path is None
            or config.expected_manifest_sha256 != CANONICAL_MANIFEST_SHA256
            or config.eligible_centers != ELIGIBLE_CENTERS
            or config.expected_row_count != EXPECTED_TEST_ROWS
            or counts != dict(EXPECTED_TEST_ROWS_BY_CENTER)
            or config.cache_root.resolve() not in {expected_root, staging_root}
        ):
            raise Stage70TestCacheError(
                "Canonical Stage-70 test-cache coverage or output identity drifted."
            )
    if not config.expected_reservation_id.startswith("reservation_"):
        raise Stage70TestCacheError(
            "Stage-70 target-evaluation reservation identity is invalid."
        )
    if len(config.expected_reservation_protocol_hash) != 64:
        raise Stage70TestCacheError(
            "Stage-70 target-evaluation protocol hash is invalid."
        )
    if expected_reservation is not None and (
        expected_reservation.manifest_sha256 != config.expected_manifest_sha256
        or expected_reservation.reservation_id != config.expected_reservation_id
        or expected_reservation.protocol_hash
        != config.expected_reservation_protocol_hash
        or expected_reservation.row_count != config.expected_row_count
        or expected_reservation.rows_by_center != counts
        or expected_reservation.purpose != PURPOSE
        or expected_reservation.fresh_evidence is not FRESH_EVIDENCE
    ):
        raise Stage70TestCacheError(
            "Stage-70 test-cache config/reservation binding drifted."
        )


def stage70_cache_config_protocol(config: Stage70TestCacheConfig) -> dict[str, object]:
    """Return a path-independent cache authorization/config identity."""

    return {
        "cache_name": config.cache_name,
        "cache_artifact_id": config.cache_artifact_id,
        "authorized_consumer_experiment_id": config.authorized_consumer_experiment_id,
        "purpose": config.purpose,
        "fresh_evidence": config.fresh_evidence,
        "scoring_manifest_sha256": config.expected_manifest_sha256,
        "target_evaluation_reservation_id": config.expected_reservation_id,
        "target_evaluation_reservation_protocol_hash": (
            config.expected_reservation_protocol_hash
        ),
        "cache_extractor_protocol_hash": (
            config.expected_cache_extractor_protocol_hash
        ),
        "eligible_centers": list(config.eligible_centers),
        "expected_row_count": config.expected_row_count,
        "expected_rows_by_center": dict(config.expected_rows_by_center),
        "experiment_seed": config.experiment_seed,
        "batch_size": config.batch_size,
        "coverage_scope": (
            "canonical" if config.canonical_coverage_required else "test_fixture_only"
        ),
    }


def _resolve(root: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else root / path


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise Stage70TestCacheError(
            f"Stage-70 test-cache config section {key!r} must be a mapping."
        )
    return value


def _require_exact_keys(
    payload: Mapping[str, object], expected: set[str], role: str
) -> None:
    observed = {str(key) for key in payload}
    if observed != expected:
        raise Stage70TestCacheError(
            f"Stage-70 test-cache {role} keys drifted: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}."
        )


__all__ = (
    "CONFIG_AUTHORIZATION_BINDING_FIELDS",
    "CONFIG_CACHE_FIELDS",
    "CONFIG_INPUT_FIELDS",
    "CONFIG_PROTOCOL_FIELDS",
    "CONFIG_RUN_FIELDS",
    "CONFIG_TOP_LEVEL_FIELDS",
    "Stage70TestCacheConfig",
    "load_stage70_test_cache_config",
    "make_stage70_test_cache_config",
    "stage70_cache_config_protocol",
    "validate_stage70_test_cache_config",
)
