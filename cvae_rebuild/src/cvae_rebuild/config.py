from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .protocol import PRIMARY_METHOD, ProtocolError


@dataclass(frozen=True)
class RebuildConfig:
    name: str
    artifact_root: Path
    feature_cache_root: Path
    experiment_seeds: tuple[int, ...]
    heldout_centers: tuple[str, ...]
    support_size: int
    support_seeds: tuple[int, ...]
    generation_seeds: tuple[int, ...]
    classifier_seeds: tuple[int, ...]
    candidate_count_per_cell: int
    primary_method: str
    pca_dim: int
    hidden_dim: int
    latent_dim: int
    num_hidden_layers: int
    train_epochs: int
    batch_size: int
    learning_rate: float
    synthetic_per_class_total: int
    eps: float

    @property
    def expected_candidate_count(self) -> int:
        return int(self.candidate_count_per_cell)


def load_config(path: str | Path) -> RebuildConfig:
    source = Path(path).resolve()
    data = _load_mapping(source)
    base_dir = source.parents[2] if len(source.parents) >= 3 else source.parent
    return parse_config(data, base_dir=base_dir)


def parse_config(data: Mapping[str, Any], *, base_dir: str | Path = ".") -> RebuildConfig:
    base = Path(base_dir)
    experiment = _mapping(data, "experiment")
    inputs = _mapping(data, "inputs")
    run = _mapping(data, "run_matrix")
    frame = _mapping(data, "feature_frame")
    model = _mapping(data, "model")
    generation = _mapping(data, "generation")
    downstream = _mapping(data, "downstream")

    cfg = RebuildConfig(
        name=str(experiment["name"]),
        artifact_root=_path(base, str(experiment["artifact_root"])),
        feature_cache_root=_path(base, str(inputs["feature_cache_root"])),
        experiment_seeds=tuple(int(v) for v in run["experiment_seeds"]),
        heldout_centers=tuple(str(v) for v in run["heldout_centers"]),
        support_size=int(run["support_size"]),
        support_seeds=tuple(int(v) for v in run["support_seeds"]),
        generation_seeds=tuple(int(v) for v in run["generation_seeds"]),
        classifier_seeds=tuple(int(v) for v in run["classifier_seeds"]),
        candidate_count_per_cell=int(run["candidate_count_per_cell"]),
        primary_method=str(experiment["primary_method"]),
        pca_dim=int(frame["pca_dim"]),
        hidden_dim=int(model["hidden_dim"]),
        latent_dim=int(model["latent_dim"]),
        num_hidden_layers=int(model["num_hidden_layers"]),
        train_epochs=int(model.get("train_epochs", 25)),
        batch_size=int(model.get("batch_size", 128)),
        learning_rate=float(model.get("learning_rate", 1.0e-3)),
        synthetic_per_class_total=int(generation["synthetic_per_class_total"]),
        eps=float(downstream["eps"]),
    )
    validate_config(cfg, data)
    return cfg


def validate_config(cfg: RebuildConfig, raw: Mapping[str, Any] | None = None) -> None:
    if cfg.primary_method != PRIMARY_METHOD:
        raise ProtocolError(f"primary_method must be {PRIMARY_METHOD!r}.")
    if cfg.support_size != 32:
        raise ProtocolError("support_size must be locked to 32.")
    if cfg.candidate_count_per_cell != 4:
        raise ProtocolError("candidate_count_per_cell must be locked to 4.")
    if cfg.pca_dim != 256:
        raise ProtocolError("pca_dim must be locked to 256 for the full v1 config.")
    if cfg.hidden_dim != 512 or cfg.latent_dim != 64 or cfg.num_hidden_layers != 2:
        raise ProtocolError("CVAE architecture must be locked to hidden=512, latent=64, two layers.")
    if cfg.train_epochs <= 0 or cfg.batch_size <= 0 or cfg.learning_rate <= 0.0:
        raise ProtocolError("CVAE training epochs, batch size, and learning rate must be positive.")
    if cfg.synthetic_per_class_total != 128:
        raise ProtocolError("synthetic_per_class_total must be locked to 128.")
    if raw is not None:
        routing = _mapping(raw, "routing")
        generation = _mapping(raw, "generation")
        model = _mapping(raw, "model")
        if routing.get("primary_score") != "calibrated_marginal_support_nelbo":
            raise ProtocolError("routing.primary_score must be calibrated marginal support-NELBO.")
        if routing.get("support_sampler") != "random_unlabeled_sample_ids":
            raise ProtocolError("routing.support_sampler must be random_unlabeled_sample_ids.")
        if model.get("class_conditioning") != "encoder_decoder_one_hot":
            raise ProtocolError("model.class_conditioning must be encoder_decoder_one_hot.")
        if generation.get("mode") != "class_stratified_reference_posterior":
            raise ProtocolError("generation.mode must be class_stratified_reference_posterior.")


def resolved_config_dict(cfg: RebuildConfig) -> dict[str, object]:
    return {
        "name": cfg.name,
        "artifact_root": str(cfg.artifact_root),
        "feature_cache_root": str(cfg.feature_cache_root),
        "experiment_seeds": list(cfg.experiment_seeds),
        "heldout_centers": list(cfg.heldout_centers),
        "support_size": cfg.support_size,
        "support_seeds": list(cfg.support_seeds),
        "generation_seeds": list(cfg.generation_seeds),
        "classifier_seeds": list(cfg.classifier_seeds),
        "candidate_count_per_cell": cfg.candidate_count_per_cell,
        "primary_method": cfg.primary_method,
        "pca_dim": cfg.pca_dim,
        "hidden_dim": cfg.hidden_dim,
        "latent_dim": cfg.latent_dim,
        "num_hidden_layers": cfg.num_hidden_layers,
        "train_epochs": cfg.train_epochs,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "synthetic_per_class_total": cfg.synthetic_per_class_total,
        "eps": cfg.eps,
    }


def write_resolved_config(path: str | Path, cfg: RebuildConfig) -> None:
    Path(path).write_text(json.dumps(resolved_config_dict(cfg), indent=2, sort_keys=True) + "\n")


def _load_mapping(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:
            raise ProtocolError("YAML config parsing requires PyYAML unless the file is JSON syntax.") from exc
        data = yaml.safe_load(text)
        if not isinstance(data, Mapping):
            raise ProtocolError("Config root must be a mapping.")
        return data


def _mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Config section {key!r} must be a mapping.")
    return value


def _path(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()
