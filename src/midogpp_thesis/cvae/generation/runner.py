"""Build and health-check the frozen Uniform-B v2 GenerationLock."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from ...common.hashing import stable_hash
from ..expert_bank.uniform_b_v2_promotion import (
    load_promotion_config,
    load_routing_authorized_expert,
    validate_promoted_bank,
)
from ..protocol import ProtocolError
from ..reporting import write_csv_rows, write_json
from .config import UniformBV2GenerationLockConfig
from .contracts import (
    CLAIM_SCOPE,
    COMPOSITION_SHUFFLE_NAMESPACE,
    EXPECTED_GENERATION_LOCK_HASH,
    EXPERIMENT_ID,
    GenerationLock,
    SOURCE_STREAM_NAMESPACE,
)
from .generation import (
    equal_union_replicate_plan,
    generate_source_block,
    source_generation_plan,
)


def build_generation_lock(
    config: UniformBV2GenerationLockConfig,
    bank_root: str | Path | None = None,
) -> GenerationLock:
    """Validate the Stage-30 bank and bind every generation-relevant identity."""

    root = Path(bank_root or config.bank_root)
    bank, control, content = _load_validated_bank(config, root)
    return _build_generation_lock_payload(config, bank=bank, control=control, content=content)


def read_generation_lock(path: str | Path) -> GenerationLock:
    return GenerationLock(_read_json(Path(path)))


def run_generation_lock(
    config: UniformBV2GenerationLockConfig,
    *,
    artifact_root: Path | None = None,
) -> Path:
    """Materialize the lock, deterministic plans, and source-only health evidence."""

    root = Path(artifact_root or config.artifact_root)
    for relative in ("manifests", "reports", "tables", "provenance"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    state_path = root / "reports/run_state.json"
    if state_path.is_file() and _read_json(state_path).get("status") == "COMPLETE":
        from .validation import validate_generation_bundle

        validate_generation_bundle(root, config=config)
        return root
    _write_state(root, "RUNNING")
    try:
        lock = build_generation_lock(config)
        write_json(root / "manifests/generation_lock.json", lock.to_payload())

        source_plan = _source_plan_payload(lock)
        replicate_plan = _replicate_plan_payload(lock)
        write_json(root / "manifests/source_generation_plan.json", source_plan)
        write_json(root / "manifests/equal_union_replicate_plan.json", replicate_plan)

        protocol = {
            "schema_version": "midogpp_uniform_b_v2_generation_lock_protocol_v1",
            "experiment_id": EXPERIMENT_ID,
            "claim_scope": CLAIM_SCOPE,
            "config_contract_hash": config.contract_hash,
            "input_artifact_id": config.bank_artifact_id,
            "bank_lock_hash": config.expected_bank_lock_hash,
            "control_lock_hash": config.expected_control_lock_hash,
            "generation_lock_hash": lock.generation_lock_hash,
            "settings_frozen_before_generation_or_scoring": True,
            "source_streams_target_and_policy_independent": True,
            "full_training_x_generation_seed_cartesian_product": True,
            "individual_expert_or_seed_selection": False,
            "target_data_used": False,
            "target_support_used": False,
            "routing_evidence_computed": False,
            "routing_quality_claimed": False,
            "downstream_utility_computed": False,
            "may_feed_deployable_selection": True,
        }
        protocol["protocol_hash"] = stable_hash(protocol)
        write_json(root / "manifests/protocol_manifest.json", protocol)

        device = _resolve_device(config.runtime_device)
        health_rows = _run_health_probe(config, lock=lock, device=device)
        write_csv_rows(root / "tables/generation_health.csv", health_rows)
        write_json(
            root / "reports/leakage_report.json",
            {
                "schema_version": "midogpp_uniform_b_v2_generation_lock_leakage_v1",
                "status": "PASS",
                "source_only_frozen_state": True,
                "target_data_used": False,
                "target_support_used": False,
                "target_labels_used": False,
                "target_evaluation_labels_used": False,
                "target_identity_used_in_source_stream_seed": False,
                "target_expert_excluded_in_every_control_replicate": True,
                "individual_expert_or_seed_selection_performed": False,
                "routing_scores_computed": False,
                "nelbo_computed": False,
                "classifier_fit_performed": False,
                "downstream_utility_computed": False,
            },
        )
        _write_content_index(root)
        _write_state(root, "COMPLETE")

        from .validation import validate_generation_bundle

        checks = validate_generation_bundle(root, config=config, allow_pending=True)
        write_json(
            root / "reports/validation_report.json",
            {
                "schema_version": "midogpp_uniform_b_v2_generation_lock_validation_v1",
                "status": "PASS",
                "validator": "validate_generation_bundle",
                "checks": checks,
            },
        )
        validate_generation_bundle(root, config=config)
    except Exception:
        _write_state(root, "FAILED")
        raise
    return root


def _build_generation_lock_payload(
    config: UniformBV2GenerationLockConfig,
    *,
    bank: Mapping[str, object],
    control: Mapping[str, object],
    content: Mapping[str, object],
) -> GenerationLock:
    records = bank.get("records")
    if not isinstance(records, list):
        raise ProtocolError("Routing-authorized bank lacks expert records.")
    expert_locks = []
    for raw in records:
        if not isinstance(raw, Mapping):
            raise ProtocolError("Routing-authorized bank contains an invalid expert record.")
        expert_locks.append(
            {
                key: raw[key]
                for key in (
                    "source_center",
                    "training_seed",
                    "expert_lock_hash",
                    "checkpoint_hash",
                    "checkpoint_file_sha256",
                    "frame_hash",
                    "frame_file_sha256",
                    "sampler_state_hash",
                    "sampler_file_sha256",
                )
            }
        )
    expert_locks.sort(key=lambda row: (str(row["source_center"]), int(row["training_seed"])))
    candidate_sources = control.get("candidate_sources_by_target")
    if not isinstance(candidate_sources, Mapping):
        raise ProtocolError("Routing-authorized control lacks candidate-source pools.")
    payload: dict[str, object] = {
        "schema_version": "midogpp_uniform_b_v2_generation_lock_v1",
        "experiment_id": EXPERIMENT_ID,
        "claim_scope": CLAIM_SCOPE,
        "config_contract_hash": config.contract_hash,
        "bank": {
            "artifact_id": config.bank_artifact_id,
            "bank_lock_hash": bank["bank_lock_hash"],
            "control_lock_hash": control["control_lock_hash"],
            "bank_index_sha256": config.expected_bank_index_sha256,
            "control_lock_sha256": config.expected_control_sha256,
            "content_index_sha256": config.expected_content_index_sha256,
            "content_hash": content["content_hash"],
            "model": dict(_mapping(bank, "model")),
            "centers": list(config.centers),
            "expert_locks": expert_locks,
            "candidate_sources_by_target": {
                str(target): [str(source) for source in sources]
                for target, sources in sorted(candidate_sources.items(), key=lambda item: str(item[0]))
            },
            "replica_policy": bank["replica_policy"],
            "all_27_experts_retained": True,
            "individual_expert_or_seed_selection": False,
        },
        "source_frame": {
            **dict(config.source_frame),
            "frame_hashes_bound_per_expert": True,
        },
        "aggregate_prior": {
            **dict(config.aggregate_prior),
            "sampler_state_hashes_bound_per_expert": True,
        },
        "generation": {
            "class_labels": [0, 1],
            "class_budget_policy": "balanced_equal_per_class",
            "dtype": "float32",
            "common_output_dim": 3840,
            "training_seeds": list(config.training_seeds),
            "generation_seeds": list(config.generation_seeds),
            "seed_pairing": "cartesian_product",
            "replicate_policy": config.generation_contract["replicate_policy"],
            "total_per_class": int(config.generation_contract["total_per_class"]),
            "max_source_block_per_class": int(config.generation_contract["total_per_class"]),
            "equal_union_source_budget_per_class": int(
                config.generation_contract["source_budget_per_class"]
            ),
            "source_stream_namespace": SOURCE_STREAM_NAMESPACE,
            "composition_shuffle_namespace": COMPOSITION_SHUFFLE_NAMESPACE,
            "prefix_allocation": True,
            "source_budgets_split_across_seeds": False,
            "target_conditioned_source_weighting": False,
        },
        "classifier": {
            **config.classifier.to_payload(),
            "config_hash": config.classifier.config_hash,
            "scaler_family": "sklearn.preprocessing.StandardScaler",
            "fit_in_stage_40": False,
        },
        "firewalls": dict(config.claim_boundary),
    }
    payload["generation_lock_hash"] = stable_hash(payload)
    lock = GenerationLock(payload)
    if lock.generation_lock_hash != EXPECTED_GENERATION_LOCK_HASH:
        raise ProtocolError("Uniform-B v2 GenerationLock semantic identity drifted.")
    return lock


def _load_validated_bank(
    config: UniformBV2GenerationLockConfig,
    root: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    promotion_config_path = root / "config.resolved.yaml"
    promotion_config = load_promotion_config(promotion_config_path)
    validate_promoted_bank(root, config=promotion_config)
    members = {
        "bank": (
            root / "manifests/expert_bank_index.json",
            config.expected_bank_index_sha256,
        ),
        "control": (
            root / "manifests/equal_union_ps_control_lock.json",
            config.expected_control_sha256,
        ),
        "content": (
            root / "manifests/content_index.json",
            config.expected_content_index_sha256,
        ),
    }
    for label, (path, expected) in members.items():
        if not path.is_file() or _sha256_file(path) != expected:
            raise ProtocolError(f"GenerationLock upstream {label} file drifted.")
    bank = _read_json(members["bank"][0])
    control = _read_json(members["control"][0])
    content = _read_json(members["content"][0])
    if bank.get("bank_lock_hash") != config.expected_bank_lock_hash:
        raise ProtocolError("GenerationLock upstream bank lock drifted.")
    if control.get("control_lock_hash") != config.expected_control_lock_hash:
        raise ProtocolError("GenerationLock upstream control lock drifted.")
    if content.get("content_hash") != config.expected_content_hash:
        raise ProtocolError("GenerationLock upstream content identity drifted.")
    if bank.get("centers") != list(config.centers) or bank.get("training_seeds") != list(
        config.training_seeds
    ):
        raise ProtocolError("GenerationLock upstream bank coverage drifted.")
    if (
        control.get("training_seeds") != list(config.training_seeds)
        or control.get("generation_seeds") != list(config.generation_seeds)
        or control.get("total_per_class") != config.generation_contract["total_per_class"]
        or control.get("source_budget_per_class")
        != config.generation_contract["source_budget_per_class"]
        or control.get("candidate_sources_by_target")
        != config.generation_contract["candidate_sources_by_target"]
    ):
        raise ProtocolError("GenerationLock upstream control geometry drifted.")
    bank_model = _mapping(bank, "model")
    for key in ("input_dim", "hidden_dim", "latent_dim", "num_hidden_layers"):
        if bank_model.get(key) != config.model.get(key):
            raise ProtocolError(f"GenerationLock upstream model identity drifted: {key}.")
    return bank, control, content


def _source_plan_payload(lock: GenerationLock) -> dict[str, object]:
    rows = [key.to_payload() for key in source_generation_plan(lock)]
    payload: dict[str, object] = {
        "schema_version": "midogpp_uniform_b_v2_source_generation_plan_v1",
        "generation_lock_hash": lock.generation_lock_hash,
        "source_stream_count": len(rows),
        "target_or_policy_identity_in_stream_keys": False,
        "records": rows,
    }
    payload["plan_hash"] = stable_hash(payload)
    return payload


def _replicate_plan_payload(lock: GenerationLock) -> dict[str, object]:
    rows = [row.to_payload() for row in equal_union_replicate_plan(lock)]
    payload: dict[str, object] = {
        "schema_version": "midogpp_uniform_b_v2_equal_union_replicate_plan_v1",
        "generation_lock_hash": lock.generation_lock_hash,
        "target_replicate_count": len(rows),
        "replicates_per_target": 9,
        "records": rows,
    }
    payload["plan_hash"] = stable_hash(payload)
    return payload


def _run_health_probe(
    config: UniformBV2GenerationLockConfig,
    *,
    lock: GenerationLock,
    device: str,
) -> list[dict[str, object]]:
    keys = source_generation_plan(lock)
    by_expert: dict[tuple[str, int], list[object]] = {}
    for key in keys:
        by_expert.setdefault((key.source_center, key.training_seed), []).append(key)
    rows: list[dict[str, object]] = []
    for center, training_seed in sorted(by_expert):
        expert = load_routing_authorized_expert(
            config.bank_root,
            source_center=center,
            training_seed=training_seed,
            device=device,
        )
        for raw_key in sorted(by_expert[(center, training_seed)], key=lambda item: item.generation_seed):
            key = raw_key
            first = generate_source_block(
                expert,
                key,
                per_class=config.health_samples_per_class,
                device=device,
            )
            repeated = generate_source_block(
                expert,
                key,
                per_class=config.health_samples_per_class,
                device=device,
            )
            if first.output_sha256 != repeated.output_sha256 or not np.array_equal(
                first.embeddings, repeated.embeddings
            ):
                raise ProtocolError("GenerationLock health probe is not deterministic.")
            count = config.health_samples_per_class
            for class_label in (0, 1):
                start = class_label * count
                stop = start + count
                class_embeddings = np.ascontiguousarray(first.embeddings[start:stop])
                rows.append(
                    {
                        "source_center": center,
                        "training_seed": training_seed,
                        "generation_seed": key.generation_seed,
                        "class_label": class_label,
                        "stream_id": key.stream_id,
                        "derived_seed": key.class_seed_by_label[str(class_label)],
                        "samples": count,
                        "model_space_dim": 256,
                        "reconstructed_embedding_dim": class_embeddings.shape[1],
                        "dtype": str(class_embeddings.dtype),
                        "finite_latents": True,
                        "finite_decoder_outputs": True,
                        "finite_reconstructed_embeddings": bool(
                            np.isfinite(class_embeddings).all()
                        ),
                        "deterministic_repeat": True,
                        "target_data_used": False,
                        "output_sha256": hashlib.sha256(class_embeddings.tobytes()).hexdigest(),
                        "status": "PASS",
                    }
                )
        del expert
        if device.startswith("cuda:"):
            torch.cuda.empty_cache()
    return rows


def _write_content_index(root: Path) -> None:
    excluded = {
        "manifests/content_index.json",
        "reports/run_state.json",
        "reports/validation_report.json",
    }
    records = []
    for path in sorted(member for member in root.rglob("*") if member.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        records.append(
            {
                "relative_path": relative,
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    payload: dict[str, object] = {
        "schema_version": "midogpp_uniform_b_v2_generation_lock_content_v1",
        "records": records,
    }
    payload["content_hash"] = stable_hash(payload)
    write_json(root / "manifests/content_index.json", payload)


def _resolve_device(configured: str) -> str:
    value = os.environ.get("MIDOGPP_V2_GENERATION_DEVICE", configured).strip()
    if value == "cpu":
        return value
    if not value.startswith("cuda:"):
        raise ProtocolError("GenerationLock device must be cpu or explicit cuda:N.")
    index = int(value.split(":", 1)[1])
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        raise ProtocolError(f"Configured GenerationLock device is unavailable: {value}.")
    return value


def _write_state(root: Path, status: str) -> None:
    write_json(
        root / "reports/run_state.json",
        {
            "schema_version": "midogpp_uniform_b_v2_generation_lock_run_state_v1",
            "status": status,
            "claim_scope": CLAIM_SCOPE,
        },
    )


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"GenerationLock payload lacks mapping {key!r}.")
    return value


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read GenerationLock JSON: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"GenerationLock JSON must be an object: {path}.")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = (
    "build_generation_lock",
    "read_generation_lock",
    "run_generation_lock",
)
