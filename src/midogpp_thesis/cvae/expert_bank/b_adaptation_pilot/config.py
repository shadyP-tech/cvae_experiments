"""Fail-closed configuration for the canonical-B adaptation pilot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ....real_features.classifier_reference.protocol import ProtocolError


PILOT_SCHEMA = "midogpp_uniform_b_source_expert_adaptation_pilot_v1"
PILOT_CENTERS = ("2", "5", "6", "9")
PILOT_ARMS = ("a_global_pca128", "b_joint_pca128", "b_block_pca96_32")
PILOT_TRAINING_SEEDS = (17, 42, 101)
PILOT_GENERATION_SEEDS = (17, 42, 101)


@dataclass(frozen=True)
class PilotConfig:
    name: str
    artifact_root: Path
    manifest_path: Path
    b_feature_cache_path: Path
    a_feature_cache_path: Path
    centers: tuple[str, ...]
    arms: tuple[str, ...]
    training_seeds: tuple[int, ...]
    generation_seeds: tuple[int, ...]
    devices: tuple[str, ...]
    cpu_threads_per_worker: int
    expected_b_dim: int
    global_dim: int
    local_dim: int
    pca_dim: int
    block_global_pca_dim: int
    block_local_pca_dim: int
    validation_fraction: float
    case_split_seed: int
    optimizer_steps: int
    batch_size: int
    kl_warmup_steps: int
    hidden_dim: int
    latent_dim: int
    learning_rate: float
    weight_decay: float
    beta_final: float
    gradient_clip_norm: float
    conditional_prior_rho: float
    conditional_prior_min_rows: int
    conditional_prior_min_cases: int
    conditional_prior_variance_clip: tuple[float, float]
    conditional_prior_max_condition_number: float
    generated_per_class: int
    classifier_c: float
    minimum_real_bacc: float
    code_version: str


def load_pilot_config(path: str | Path) -> PilotConfig:
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Pilot configuration requires PyYAML.") from exc

    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ProtocolError("Pilot config must be a mapping.")
    experiment = _mapping(payload, "experiment")
    inputs = _mapping(payload, "inputs")
    run = _mapping(payload, "run")
    representation = _mapping(payload, "representation")
    split = _mapping(payload, "case_holdout")
    training = _mapping(payload, "training")
    prior = _mapping(payload, "conditional_prior")
    evaluation = _mapping(payload, "evaluation")
    claim = _mapping(payload, "claim_boundary")

    if str(experiment.get("schema_version", "")) != PILOT_SCHEMA:
        raise ProtocolError(f"Pilot schema must be {PILOT_SCHEMA!r}.")
    if tuple(str(v) for v in run.get("centers", ())) != PILOT_CENTERS:
        raise ProtocolError("Pilot centers are frozen to metadata-selected centers 2,5,6,9.")
    if tuple(str(v) for v in run.get("arms", ())) != PILOT_ARMS:
        raise ProtocolError("Pilot arms differ from the frozen three-arm panel.")
    if tuple(int(v) for v in run.get("training_seeds", ())) != PILOT_TRAINING_SEEDS:
        raise ProtocolError("Pilot training seeds differ from 17/42/101.")
    if tuple(int(v) for v in run.get("generation_seeds", ())) != PILOT_GENERATION_SEEDS:
        raise ProtocolError("Pilot generation seeds differ from 17/42/101.")
    frozen = {
        "expected_b_dim": (representation, 3840),
        "global_dim": (representation, 2560),
        "local_dim": (representation, 1280),
        "pca_dim": (representation, 128),
        "block_global_pca_dim": (representation, 96),
        "block_local_pca_dim": (representation, 32),
        "optimizer_steps": (training, 1000),
        "batch_size": (training, 128),
        "kl_warmup_steps": (training, 250),
        "generated_per_class": (evaluation, 512),
        "hidden_dim": (training, 512),
        "latent_dim": (training, 32),
        "num_hidden_layers": (training, 2),
        "kl_warmup_steps": (training, 250),
        "min_rows_per_class": (prior, 64),
        "min_cases_per_class": (prior, 5),
    }
    for key, (section, expected) in frozen.items():
        if int(section.get(key, -1)) != expected:
            raise ProtocolError(f"Pilot {key} must remain frozen at {expected}.")
    if float(prior.get("rho", -1.0)) != 0.25:
        raise ProtocolError("Pilot conditional-prior rho must remain 0.25.")
    exact_scalars = (
        (float(split.get("validation_fraction", -1.0)), 0.20, "validation_fraction"),
        (int(split.get("seed", -1)), 2718, "case split seed"),
        (float(training.get("learning_rate", -1.0)), 0.001, "learning_rate"),
        (float(training.get("weight_decay", -1.0)), 0.0001, "weight_decay"),
        (float(training.get("beta_final", -1.0)), 0.001, "beta_final"),
        (float(training.get("gradient_clip_norm", -1.0)), 5.0, "gradient_clip_norm"),
        (float(evaluation.get("classifier_c", -1.0)), 0.01, "classifier_c"),
        (float(evaluation.get("minimum_real_bacc", -1.0)), 0.60, "minimum_real_bacc"),
    )
    for observed, expected, name in exact_scalars:
        if observed != expected:
            raise ProtocolError(f"Pilot {name} must remain frozen at {expected}.")
    if claim.get("claim_scope") != "diagnostic_only":
        raise ProtocolError("Pilot claim_scope must be diagnostic_only.")
    forbidden_true = (
        "may_export_recipe_lock",
        "may_feed_expert_bank",
        "may_feed_generation",
        "may_feed_routing",
    )
    if any(bool(claim.get(key, True)) for key in forbidden_true):
        raise ProtocolError("Pilot outputs must be non-promotable.")

    devices = tuple(str(v) for v in run.get("devices", ("cpu",)))
    if not devices or len(devices) != len(set(devices)):
        raise ProtocolError("Pilot devices must be a nonempty unique sequence.")
    variance_clip = tuple(float(v) for v in prior.get("variance_clip", ()))
    if variance_clip != (0.25, 4.0):
        raise ProtocolError("Pilot conditional-prior variance clip must be [0.25,4.0].")
    base = config_path.parent
    return PilotConfig(
        name=str(experiment.get("name", "")),
        artifact_root=_path(base, str(experiment.get("artifact_root", ""))),
        manifest_path=_path(base, str(inputs.get("manifest_path", ""))),
        b_feature_cache_path=_path(base, str(inputs.get("b_feature_cache_path", ""))),
        a_feature_cache_path=_path(base, str(inputs.get("a_feature_cache_path", ""))),
        centers=PILOT_CENTERS,
        arms=PILOT_ARMS,
        training_seeds=PILOT_TRAINING_SEEDS,
        generation_seeds=PILOT_GENERATION_SEEDS,
        devices=devices,
        cpu_threads_per_worker=int(run.get("cpu_threads_per_worker", 2)),
        expected_b_dim=3840,
        global_dim=2560,
        local_dim=1280,
        pca_dim=128,
        block_global_pca_dim=96,
        block_local_pca_dim=32,
        validation_fraction=float(split.get("validation_fraction", 0.2)),
        case_split_seed=int(split.get("seed", 2718)),
        optimizer_steps=1000,
        batch_size=128,
        kl_warmup_steps=250,
        hidden_dim=int(training.get("hidden_dim", 512)),
        latent_dim=int(training.get("latent_dim", 32)),
        learning_rate=float(training.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
        beta_final=float(training.get("beta_final", 1e-3)),
        gradient_clip_norm=float(training.get("gradient_clip_norm", 5.0)),
        conditional_prior_rho=0.25,
        conditional_prior_min_rows=int(prior.get("min_rows_per_class", 64)),
        conditional_prior_min_cases=int(prior.get("min_cases_per_class", 5)),
        conditional_prior_variance_clip=(0.25, 4.0),
        conditional_prior_max_condition_number=float(
            prior.get("max_condition_number", 1e4)
        ),
        generated_per_class=512,
        classifier_c=float(evaluation.get("classifier_c", 0.01)),
        minimum_real_bacc=float(evaluation.get("minimum_real_bacc", 0.60)),
        code_version=str(experiment.get("code_version", "")),
    )


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Pilot config requires mapping section {key!r}.")
    return value


def _path(base: Path, value: str) -> Path:
    if not value:
        raise ProtocolError("Pilot input/output paths must be explicit.")
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()
