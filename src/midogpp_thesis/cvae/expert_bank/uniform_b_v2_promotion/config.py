"""Fail-closed configuration for the reviewed Uniform-B v2 bank promotion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Mapping

import yaml

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .contracts import (
    CENTERS,
    CLAIM_SCOPE,
    CONTROL_SAMPLER_FAMILY,
    CONTROL_TOTAL_PER_CLASS,
    EXPERIMENT_NAME,
    GENERATION_SEEDS,
    OUTPUT_ARTIFACT_ID,
    PROMOTION_REVIEW_ID,
    SOURCE_ARTIFACT_ID,
    SOURCE_CHECKPOINT_INDEX_SHA256,
    SOURCE_CONFIG_HASH,
    SOURCE_CONTENT_INDEX_SHA256,
    SOURCE_DECISION_SHA256,
    SOURCE_FRAME_INDEX_SHA256,
    SOURCE_PROTOCOL_HASH,
    TRAINING_SEEDS,
)


@dataclass(frozen=True)
class UniformBV2PromotionConfig:
    name: str
    artifact_root: Path
    source_study_root: Path
    manifest_path: Path
    feature_cache_path: Path
    source_artifact_id: str
    expected_manifest_hash: str
    expected_feature_cache_hash: str
    expected_source_protocol_hash: str
    expected_source_config_hash: str
    expected_source_content_index_sha256: str
    expected_source_checkpoint_index_sha256: str
    expected_source_frame_index_sha256: str
    expected_source_decision_sha256: str
    centers: tuple[str, ...]
    training_seeds: tuple[int, ...]
    generation_seeds: tuple[int, ...]
    min_ps_mean_bacc: float
    min_ps_seed_bacc: float
    min_ps_minus_p0: float
    max_posterior_ceiling_gap: float
    required_checkpoint_records: int
    required_sampler_records: int
    required_task_metric_rows: int
    required_generation_blocks: int
    input_dim: int
    hidden_dim: int
    latent_dim: int
    num_hidden_layers: int
    sampler_family: str
    sampler_min_class_count: int
    sampler_max_condition_number: float
    control_total_per_class: int
    promotion_review: Mapping[str, object]
    claim_boundary: Mapping[str, object]
    runtime_device: str
    checkpoint_materialization: str

    @property
    def contract_hash(self) -> str:
        payload = asdict(self)
        for key in (
            "artifact_root",
            "source_study_root",
            "manifest_path",
            "feature_cache_path",
            "runtime_device",
            "checkpoint_materialization",
        ):
            payload.pop(key)
        return stable_hash(payload)


def load_promotion_config(path: str | Path) -> UniformBV2PromotionConfig:
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ProtocolError("Uniform-B v2 promotion config must be a mapping.")
    experiment = _mapping(payload, "experiment")
    inputs = _mapping(payload, "inputs")
    gates = _mapping(payload, "promotion_gates")
    model = _mapping(payload, "model")
    prior = _mapping(payload, "aggregate_prior")
    control = _mapping(payload, "canonical_control")
    review = _mapping(payload, "promotion_review")
    claim = _mapping(payload, "claim_boundary")
    runtime = _mapping(payload, "runtime")
    base = config_path.parent
    config = UniformBV2PromotionConfig(
        name=str(experiment.get("name", "")),
        artifact_root=_path(base, experiment.get("artifact_root")),
        source_study_root=_path(base, inputs.get("source_study_root")),
        manifest_path=_path(base, inputs.get("manifest_path")),
        feature_cache_path=_path(base, inputs.get("feature_cache_path")),
        source_artifact_id=str(inputs.get("source_artifact_id", "")),
        expected_manifest_hash=str(inputs.get("expected_manifest_hash", "")),
        expected_feature_cache_hash=str(inputs.get("expected_feature_cache_hash", "")),
        expected_source_protocol_hash=str(inputs.get("expected_source_protocol_hash", "")),
        expected_source_config_hash=str(inputs.get("expected_source_config_hash", "")),
        expected_source_content_index_sha256=str(inputs.get("expected_source_content_index_sha256", "")),
        expected_source_checkpoint_index_sha256=str(inputs.get("expected_source_checkpoint_index_sha256", "")),
        expected_source_frame_index_sha256=str(inputs.get("expected_source_frame_index_sha256", "")),
        expected_source_decision_sha256=str(inputs.get("expected_source_decision_sha256", "")),
        centers=tuple(str(value) for value in gates.get("centers", ())),
        training_seeds=_ints(gates.get("training_seeds")),
        generation_seeds=_ints(gates.get("generation_seeds")),
        min_ps_mean_bacc=float(gates.get("min_ps_mean_bacc", float("nan"))),
        min_ps_seed_bacc=float(gates.get("min_ps_seed_bacc", float("nan"))),
        min_ps_minus_p0=float(gates.get("min_ps_minus_p0", float("nan"))),
        max_posterior_ceiling_gap=float(gates.get("max_posterior_ceiling_gap", float("nan"))),
        required_checkpoint_records=int(gates.get("required_checkpoint_records", 0)),
        required_sampler_records=int(gates.get("required_sampler_records", 0)),
        required_task_metric_rows=int(gates.get("required_task_metric_rows", 0)),
        required_generation_blocks=int(gates.get("required_generation_blocks", 0)),
        input_dim=int(model.get("input_dim", 0)),
        hidden_dim=int(model.get("hidden_dim", 0)),
        latent_dim=int(model.get("latent_dim", 0)),
        num_hidden_layers=int(model.get("num_hidden_layers", 0)),
        sampler_family=str(prior.get("family", "")),
        sampler_min_class_count=int(prior.get("min_class_count", 0)),
        sampler_max_condition_number=float(prior.get("max_condition_number", float("nan"))),
        control_total_per_class=int(control.get("total_per_class", 0)),
        promotion_review=dict(review),
        claim_boundary=dict(claim),
        runtime_device=str(runtime.get("device", "cpu")),
        checkpoint_materialization=str(runtime.get("checkpoint_materialization", "")),
    )
    _validate(config)
    return config


def _validate(config: UniformBV2PromotionConfig) -> None:
    exact = {
        "name": (config.name, EXPERIMENT_NAME),
        "source_artifact_id": (config.source_artifact_id, SOURCE_ARTIFACT_ID),
        "expected_manifest_hash": (
            config.expected_manifest_hash,
            "db661ac7e3dbafde8e283528de6706ab35f2c26629b389706c4504e458cc5869",
        ),
        "expected_feature_cache_hash": (
            config.expected_feature_cache_hash,
            "1ed7602f225c592a6f8103b24ebfc93f72dc6d5d0c27565566a8b2260783d1dc",
        ),
        "expected_source_protocol_hash": (config.expected_source_protocol_hash, SOURCE_PROTOCOL_HASH),
        "expected_source_config_hash": (config.expected_source_config_hash, SOURCE_CONFIG_HASH),
        "expected_source_content_index_sha256": (
            config.expected_source_content_index_sha256,
            SOURCE_CONTENT_INDEX_SHA256,
        ),
        "expected_source_checkpoint_index_sha256": (
            config.expected_source_checkpoint_index_sha256,
            SOURCE_CHECKPOINT_INDEX_SHA256,
        ),
        "expected_source_frame_index_sha256": (
            config.expected_source_frame_index_sha256,
            SOURCE_FRAME_INDEX_SHA256,
        ),
        "expected_source_decision_sha256": (
            config.expected_source_decision_sha256,
            SOURCE_DECISION_SHA256,
        ),
        "centers": (config.centers, CENTERS),
        "training_seeds": (config.training_seeds, TRAINING_SEEDS),
        "generation_seeds": (config.generation_seeds, GENERATION_SEEDS),
        "required_checkpoint_records": (config.required_checkpoint_records, 27),
        "required_sampler_records": (config.required_sampler_records, 81),
        "required_task_metric_rows": (config.required_task_metric_rows, 3240),
        "required_generation_blocks": (config.required_generation_blocks, 405),
        "input_dim": (config.input_dim, 256),
        "hidden_dim": (config.hidden_dim, 1024),
        "latent_dim": (config.latent_dim, 64),
        "num_hidden_layers": (config.num_hidden_layers, 3),
        "sampler_family": (config.sampler_family, CONTROL_SAMPLER_FAMILY),
        "control_total_per_class": (config.control_total_per_class, CONTROL_TOTAL_PER_CLASS),
        "checkpoint_materialization": (
            config.checkpoint_materialization,
            "hard_link_same_filesystem_copy_fallback",
        ),
    }
    mismatch = [
        f"{key}: observed={observed!r}, expected={expected!r}"
        for key, (observed, expected) in exact.items()
        if observed != expected
    ]
    if mismatch:
        raise ProtocolError("Uniform-B v2 promotion protocol drifted: " + "; ".join(mismatch))
    numeric = (
        config.min_ps_mean_bacc,
        config.min_ps_seed_bacc,
        config.min_ps_minus_p0,
        config.max_posterior_ceiling_gap,
        config.sampler_max_condition_number,
    )
    if not all(math.isfinite(value) for value in numeric) or not (
        config.min_ps_mean_bacc == 0.70
        and config.min_ps_seed_bacc == 0.75
        and config.min_ps_minus_p0 == 0.005
        and config.max_posterior_ceiling_gap == 0.01
        and config.sampler_min_class_count == 64
        and config.sampler_max_condition_number == 1_000_000.0
    ):
        raise ProtocolError("Uniform-B v2 promotion gates drifted.")
    required_review = {
        "review_id": PROMOTION_REVIEW_ID,
        "status": "approved",
        "review_effect": "authorizes_new_stage30_expert_bank_only",
        "whole_bank_adoption": True,
        "individual_expert_or_seed_selection": False,
        "source_inner_evidence_consumed_for_adoption": True,
    }
    if any(config.promotion_review.get(key) != value for key, value in required_review.items()):
        raise ProtocolError("Uniform-B v2 promotion review drifted.")
    required_claim = {
        "claim_scope": CLAIM_SCOPE,
        "independently_trained_source_experts": True,
        "all_27_experts_retained": True,
        "target_expert_excluded_in_every_routing_fold": True,
        "source_inner_evaluation_labels_consumed_for_whole_bank_adoption": True,
        "target_labels_used_for_individual_expert_selection": False,
        "target_labels_may_be_used_for_routing_selection": False,
        "routing_quality_claimed": False,
        "may_feed_deployable_selection": True,
    }
    if any(config.claim_boundary.get(key) != value for key, value in required_claim.items()):
        raise ProtocolError("Uniform-B v2 promotion claim boundary drifted.")
    if config.runtime_device != "cpu" and not config.runtime_device.startswith("cuda:"):
        raise ProtocolError("Promotion runtime device must be cpu or explicit cuda:N.")
    if str(config.artifact_root).startswith("output:") and config.artifact_root.name != OUTPUT_ARTIFACT_ID:
        raise ProtocolError("Unexpected Uniform-B v2 promotion output identity.")


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Promotion config section {key!r} must be a mapping.")
    return value


def _ints(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ProtocolError("Promotion config expected an integer list.")
    return tuple(int(item) for item in value)


def _path(base: Path, value: object) -> Path:
    rendered = str(value or "")
    if not rendered:
        raise ProtocolError("Promotion config path is empty.")
    if rendered.startswith(("artifact://", "output://")):
        return Path(rendered)
    path = Path(rendered)
    return path if path.is_absolute() else (base / path).resolve()


__all__ = ("UniformBV2PromotionConfig", "load_promotion_config")
