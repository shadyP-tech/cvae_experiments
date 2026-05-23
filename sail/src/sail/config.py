"""Configuration loading for the extracted Virchow2 pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .protocol import ProtocolError


EXPERIMENT_NAME = "virchow2_source_selected_dense_aggregation"


@dataclass(frozen=True)
class PipelineConfig:
    experiment_name: str = EXPERIMENT_NAME
    candidate_centers: tuple[str, ...] = ("0", "1", "2", "3", "4")
    experiment_seeds: tuple[int, ...] = (42, 43, 44)
    support_sizes: tuple[int, ...] = (4, 8, 16, 32)
    support_seeds: tuple[int, ...] = (17, 23, 31)
    primary_backbone: str = "virchow2"
    representations: tuple[str, ...] = ("raw", "PCA64", "PCA128", "PCA256")
    c_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    class_weight_grid: tuple[str, ...] = ("none", "balanced")
    primary_k_values: tuple[int, ...] = (3, 5, 10)
    aggregation_rules: tuple[str, ...] = ("geometric", "arithmetic")
    weak_center_threshold: float = 0.85
    robust_std_weight: float = 0.25
    robust_weak_penalty_weight: float = 0.50
    mean_090_threshold: float = 0.90
    rebuild_mean_bacc: float = 0.92
    rebuild_worst_center_bacc: float = 0.85
    seed_std_mean_bacc_max: float = 0.03
    seed_worst_center_min: float = 0.75
    min_delta_vs_top1: float = 0.005
    pca_low_sample_warning_multiplier: int = 3
    cache_root: str = "sail/artifacts/pathology_embeddings"
    cache_path_template: str = "{cache_root}/{backbone}/seed{seed}/embeddings/{split}.pt"
    artifacts_root: str = "sail/artifacts/virchow2_dense_source_selected"


@dataclass(frozen=True)
class RunLimits:
    experiment_seeds: tuple[int, ...] | None = None
    heldout_centers: tuple[str, ...] | None = None
    k_values: tuple[int, ...] | None = None
    aggregation_rules: tuple[str, ...] | None = None
    representations: tuple[str, ...] | None = None


def load_config(path: Path) -> PipelineConfig:
    text = Path(path).read_text(encoding="utf-8")
    assert_config_text(text)
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise ProtocolError("Loading YAML configs requires PyYAML.") from exc
    data = yaml.safe_load(text) or {}
    assert_config_mapping(data)
    return _from_mapping(data)


def assert_config_text(text: str) -> None:
    required = (
        f"name: {EXPERIMENT_NAME}",
        "primary_backbone: virchow2",
        "target_eval_labels_used_for_scoring_only: true",
        "target_eval_tuned_selection: forbidden",
        "metadata_routing: baseline_only",
        "cvae_preservation: later_diagnostic",
    )
    missing = [snippet for snippet in required if snippet not in text]
    if missing:
        raise ProtocolError(f"Config is missing locked source-only fields: {missing}")
    forbidden = (
        "primary_backbone: dinov2",
        "target_eval_tuned_selection: allowed",
        "metadata_routing: primary",
        "cvae_preservation: proven",
        "target_support_labels_for_selection: true",
    )
    present = [snippet for snippet in forbidden if snippet in text]
    if present:
        raise ProtocolError(f"Config contains forbidden fields: {present}")


def assert_config_mapping(data: Mapping[str, Any]) -> None:
    experiment = _mapping(data.get("experiment"), "experiment")
    if experiment.get("name") != EXPERIMENT_NAME:
        raise ProtocolError(f"Unexpected experiment.name: {experiment.get('name')!r}")
    protocol = _mapping(data.get("protocol"), "protocol")
    if protocol.get("primary_backbone") != "virchow2":
        raise ProtocolError("The extracted primary backbone must be virchow2.")
    if protocol.get("target_eval_labels_used_for_scoring_only") is not True:
        raise ProtocolError("Target-eval labels must be scoring-only.")
    if protocol.get("target_eval_tuned_selection") != "forbidden":
        raise ProtocolError("Target-eval-tuned selection must be forbidden.")
    if protocol.get("metadata_routing") != "baseline_only":
        raise ProtocolError("Metadata routing is not primary in this extraction.")


def _from_mapping(data: Mapping[str, Any]) -> PipelineConfig:
    defaults = PipelineConfig()
    dataset = _mapping(_mapping(data.get("datasets"), "datasets").get("camelyon17"), "datasets.camelyon17")
    feature_cache = _mapping(data.get("feature_cache"), "feature_cache")
    source = _mapping(data.get("source_inner_lodo"), "source_inner_lodo")
    dense = _mapping(data.get("dense_aggregation"), "dense_aggregation")
    robust = _mapping(dense.get("robust_score"), "dense_aggregation.robust_score")
    decision = _mapping(data.get("decision_rule"), "decision_rule")
    artifacts = _mapping(data.get("artifacts"), "artifacts")
    return PipelineConfig(
        candidate_centers=tuple(str(value) for value in dataset.get("candidate_centers", defaults.candidate_centers)),
        experiment_seeds=tuple(int(value) for value in dataset.get("experiment_seeds", defaults.experiment_seeds)),
        support_sizes=tuple(int(value) for value in dataset.get("support_sizes", defaults.support_sizes)),
        support_seeds=tuple(int(value) for value in dataset.get("support_seeds", defaults.support_seeds)),
        primary_backbone=str(_mapping(data.get("protocol"), "protocol").get("primary_backbone", "virchow2")),
        representations=tuple(_representation_label(value) for value in source.get("representations", defaults.representations)),
        c_grid=tuple(float(value) for value in source.get("C_grid", defaults.c_grid)),
        class_weight_grid=tuple(_class_weight_label(value) for value in source.get("class_weight_grid", defaults.class_weight_grid)),
        primary_k_values=tuple(int(value) for value in dense.get("primary_k_values", defaults.primary_k_values)),
        aggregation_rules=tuple(str(value) for value in dense.get("aggregation_rules", defaults.aggregation_rules)),
        weak_center_threshold=float(robust.get("weak_center_threshold", defaults.weak_center_threshold)),
        robust_std_weight=float(robust.get("std_weight", defaults.robust_std_weight)),
        robust_weak_penalty_weight=float(robust.get("weak_penalty_weight", defaults.robust_weak_penalty_weight)),
        mean_090_threshold=float(decision.get("mean_090_threshold", defaults.mean_090_threshold)),
        rebuild_mean_bacc=float(decision.get("rebuild_mean_bacc", defaults.rebuild_mean_bacc)),
        rebuild_worst_center_bacc=float(decision.get("rebuild_worst_center_bacc", defaults.rebuild_worst_center_bacc)),
        seed_std_mean_bacc_max=float(decision.get("seed_std_mean_bacc_max", defaults.seed_std_mean_bacc_max)),
        seed_worst_center_min=float(decision.get("seed_worst_center_min", defaults.seed_worst_center_min)),
        min_delta_vs_top1=float(decision.get("min_delta_vs_top1", defaults.min_delta_vs_top1)),
        pca_low_sample_warning_multiplier=int(source.get("pca_low_sample_warning_multiplier", defaults.pca_low_sample_warning_multiplier)),
        cache_root=str(feature_cache.get("cache_root", defaults.cache_root)),
        cache_path_template=str(feature_cache.get("cache_path_template", defaults.cache_path_template)),
        artifacts_root=str(artifacts.get("root", defaults.artifacts_root)),
    )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{name} must be a mapping")
    return value


def _class_weight_label(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return "none"
    if text == "balanced":
        return "balanced"
    raise ProtocolError(f"Unsupported class_weight value: {value!r}")


def _representation_label(value: object) -> str:
    text = str(value).strip()
    if text.lower() == "raw":
        return "raw"
    if text.upper().startswith("PCA"):
        return "PCA" + text[3:]
    if text.isdigit():
        return f"PCA{text}"
    raise ProtocolError(f"Unsupported representation: {value!r}")
