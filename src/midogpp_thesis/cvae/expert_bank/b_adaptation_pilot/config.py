"""Fail-closed configuration for the canonical-B adaptation pilot."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

from ....real_features.classifier_reference.protocol import ProtocolError


PILOT_SCHEMA_V1 = "midogpp_uniform_b_source_expert_adaptation_pilot_v1"
PILOT_SCHEMA_V2 = "midogpp_uniform_b_source_expert_adaptation_pilot_v2"
PILOT_IDENTITIES = {
    PILOT_SCHEMA_V1: "uniform_b_source_expert_adaptation_pilot_v1",
    PILOT_SCHEMA_V2: "uniform_b_source_expert_adaptation_pilot_v2",
}
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
    predecessor_failure_audit_path: Path | None
    protocol_amendment: Mapping[str, object]
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
    pca_svd_solver: str
    pca_random_state: int
    pca_n_oversamples: int
    pca_iterated_power: int
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
    amendment = (
        _mapping(payload, "protocol_amendment")
        if str(experiment.get("schema_version", "")) == PILOT_SCHEMA_V2
        else {}
    )

    schema = str(experiment.get("schema_version", ""))
    identity = PILOT_IDENTITIES.get(schema)
    if identity is None:
        raise ProtocolError(
            f"Pilot schema must be one of {tuple(PILOT_IDENTITIES)!r}."
        )
    if str(experiment.get("name", "")) != identity:
        raise ProtocolError("Pilot experiment name does not match its schema.")
    if str(experiment.get("code_version", "")) != identity:
        raise ProtocolError("Pilot code version does not match its schema.")
    if tuple(str(v) for v in run.get("centers", ())) != PILOT_CENTERS:
        raise ProtocolError("Pilot centers are frozen to metadata-selected centers 2,5,6,9.")
    if tuple(str(v) for v in run.get("arms", ())) != PILOT_ARMS:
        raise ProtocolError("Pilot arms differ from the frozen three-arm panel.")
    if tuple(int(v) for v in run.get("training_seeds", ())) != PILOT_TRAINING_SEEDS:
        raise ProtocolError("Pilot training seeds differ from 17/42/101.")
    if tuple(int(v) for v in run.get("generation_seeds", ())) != PILOT_GENERATION_SEEDS:
        raise ProtocolError("Pilot generation seeds differ from 17/42/101.")
    if tuple(str(v) for v in run.get("devices", ())) != ("cuda:0", "cuda:1"):
        raise ProtocolError("Pilot runtime is frozen to one worker on each A5000 device.")
    if int(run.get("cpu_threads_per_worker", -1)) != 1:
        raise ProtocolError("Pilot GPU workers must each use one CPU thread.")
    exact_semantics = (
        (representation, "whiten", False),
        (representation, "post_fit_reweighting", False),
        (split, "policy", "deterministic_case_disjoint_single_split_v1"),
        (split, "train_split_only", True),
        (split, "require_both_classes_on_both_sides", True),
        (training, "objective", "stochastic_isotropic_beta_objective_step_normalized_v1"),
        (training, "batch_policy", "class_to_case_to_row_balanced_with_replacement_v1"),
        (prior, "family", "class_conditional_diagonal_standard_shrinkage_rho025_v1"),
        (prior, "fallback_family", "standard_normal"),
        (prior, "selection_on_holdout", False),
        (evaluation, "classifier_family", "standard_scaler_l2_logistic"),
        (evaluation, "classifier_class_weight", "balanced"),
        (evaluation, "classifier_threshold", 0.5),
        (evaluation, "heldout_class_used_for_cvae_reconstruction_conditioning", True),
    )
    for section, key, expected in exact_semantics:
        if section.get(key) != expected:
            raise ProtocolError(f"Pilot semantic {key} must remain frozen at {expected!r}.")
    if schema == PILOT_SCHEMA_V2:
        heldout_semantics = {
            "heldout_labels_used_for_classifier_fit": False,
            "heldout_labels_used_for_cvae_fit": False,
            "heldout_labels_used_for_denominator_preflight_gate": True,
            "heldout_labels_used_for_scoring": True,
            "heldout_labels_used_for_diagnostic_progression_decision": True,
            "heldout_labels_used_for_confirmation": False,
        }
        for key, expected in heldout_semantics.items():
            if evaluation.get(key) is not expected:
                raise ProtocolError(
                    f"Pilot held-out-label semantic {key} must be {expected!r}."
                )
        expected_amendment: dict[str, object] = {
            "predecessor_experiment": "midogpp.oracle.uniform_b_source_expert_adaptation_pilot.v1",
            "predecessor_artifact": "midogpp_output_uniform_b_source_expert_adaptation_pilot_v1",
            "predecessor_failure_audit_artifact": "midogpp_uniform_b_source_expert_adaptation_pilot_v1_failure_audit",
            "predecessor_status": "FAILED_PREDECLARED_REAL_DENOMINATOR",
            "predecessor_protocol_hash": "a8743bf0ae4e9b02",
            "predecessor_config_resolved_sha256": "e364c7f44c858dbce515b041ce73a00c5c963e2fcd59a60a972951333d32ae7b",
            "predecessor_failure_audit_sha256": "ce06ac0156679db15b156c392c475284c6bfa6af05bab34f2bd48e4dc8ebdb35",
            "observed_failure_center": "9",
            "observed_failure_arm": "a_global_pca128",
            "observed_failure_bacc": 0.5942461434523139,
            "observed_failure_pca_n_oversamples": 16,
            "reason": "approval_preflight_and_production_pca_policy_mismatch",
            "preserved_denominator_floor": 0.60,
            "sole_parameter_change": "randomized_pca_n_oversamples_16_to_10",
            "preflight_outcomes_inspected_before_amendment": True,
            "confirmation_eligible": False,
            "rehabilitates_v1": False,
            "may_promote_v1_outputs": False,
        }
        if dict(amendment) != expected_amendment:
            raise ProtocolError("Pilot v2 protocol amendment record is not exact.")
        audit_path = _path(
            config_path.parent,
            str(inputs.get("predecessor_failure_audit_path", "")),
        )
        if _file_sha256(audit_path) != expected_amendment["predecessor_failure_audit_sha256"]:
            raise ProtocolError("Pilot v2 predecessor failure-audit hash mismatch.")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if (
            audit.get("terminal_status") != expected_amendment["predecessor_status"]
            or audit.get("frozen_protocol_hash") != expected_amendment["predecessor_protocol_hash"]
            or audit.get("failed_reference", {}).get("bacc")
            != expected_amendment["observed_failure_bacc"]
        ):
            raise ProtocolError("Pilot v2 predecessor failure-audit semantics mismatch.")
    else:
        if evaluation.get("heldout_labels_used_for_classifier_fit_or_model_selection") is not False:
            raise ProtocolError("Pilot v1 held-out-label declaration is frozen.")
        audit_path = None

    expected_pca = (
        ("svd_solver", "randomized"),
        ("random_state", 0),
        ("n_oversamples", 10 if schema == PILOT_SCHEMA_V2 else 16),
        ("iterated_power", 4),
    )
    for key, expected in expected_pca:
        observed = representation.get(key, expected if schema == PILOT_SCHEMA_V1 else None)
        if observed != expected:
            raise ProtocolError(
                f"Pilot PCA policy {key} must remain frozen at {expected!r}."
            )
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
        (float(prior.get("max_condition_number", -1.0)), 10000.0, "max_condition_number"),
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
        predecessor_failure_audit_path=audit_path,
        protocol_amendment=dict(amendment),
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
        pca_svd_solver="randomized",
        pca_random_state=0,
        pca_n_oversamples=10 if schema == PILOT_SCHEMA_V2 else 16,
        pca_iterated_power=4,
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Pilot config requires mapping section {key!r}.")
    return value


def _path(base: Path, value: str) -> Path:
    if not value:
        raise ProtocolError("Pilot input/output paths must be explicit.")
    if value.startswith(("artifact://", "output://")):
        from ....workspace.runtime import MidogppWorkspace

        scheme, remainder = value.split("://", 1)
        artifact_id, separator, member = remainder.partition("/")
        root = MidogppWorkspace.load().resolve_artifact(
            artifact_id,
            for_output=scheme == "output",
            require_exists=scheme == "artifact",
        )
        return (root / member).resolve() if separator else root.resolve()
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()
