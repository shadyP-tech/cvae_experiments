"""Protocol dataclasses and validation boundaries for downstream v1.

This module owns cheap checks that must pass before generation or classifier
training starts. It deliberately avoids heavyweight experiment dependencies so
config and artifact contracts can be tested on a laptop without Torch/NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .schemas import (
    CLASSIFIER_SEEDS,
    DATASET_NAME,
    DIAGNOSTIC_BUDGETS_PER_CLASS,
    DOMAIN_KEY,
    ESSENTIAL_BASELINES,
    EXPERIMENT_NAME,
    EXPERIMENT_SEEDS,
    FORBIDDEN_ROUTER_INPUTS,
    GENERATION_SEEDS,
    NEGATIVE_CONTROL_GENERATION_MODE,
    PRIMARY_BUDGET_PER_CLASS,
    PRIMARY_GENERATION_MODE,
    SUPPORT_SEEDS,
    SUPPORT_SIZES,
)


class ProtocolError(ValueError):
    """Raised when the locked downstream protocol is violated."""


class ArtifactSyncError(FileNotFoundError):
    """Raised when downstream v1 is missing required frozen artifacts."""


@dataclass(frozen=True)
class ExperimentIdentity:
    """Stable identifiers that should appear in every run artifact."""

    name: str
    dataset_name: str
    heldout_domain: str
    support_size: int
    support_seed: int
    generation_seed: int
    classifier_seed: int


@dataclass(frozen=True)
class LockedV1Config:
    """Normalized v1 config values used by the downstream pipeline."""

    name: str
    dataset_name: str
    domain_key: str
    candidate_domains: tuple[str, ...]
    experiment_seeds: tuple[int, ...]
    support_seeds: tuple[int, ...]
    support_sizes: tuple[int, ...]
    generation_seeds: tuple[int, ...]
    classifier_seeds: tuple[int, ...]
    primary_generation_mode: str
    negative_control_generation_mode: str
    primary_budget_per_class: int
    diagnostic_budgets_per_class: tuple[int, ...]
    support_selection_glob: str
    artifacts_root: str
    required_external_artifacts: tuple[str, ...]


@dataclass(frozen=True)
class ProtocolManifest:
    """Minimum manifest fields required before running downstream evaluation."""

    identity: ExperimentIdentity
    support_manifest: Path
    evaluation_manifest: Path
    expert_provenance: Path
    candidate_expert_domains: Sequence[str]
    forbidden_router_inputs_checked: bool
    support_eval_disjoint_checked: bool
    target_expert_excluded_checked: bool


def default_v1_config() -> LockedV1Config:
    """Return the implementation-locked Camelyon17 v1 protocol config."""

    return LockedV1Config(
        name=EXPERIMENT_NAME,
        dataset_name=DATASET_NAME,
        domain_key=DOMAIN_KEY,
        candidate_domains=("0", "1", "2", "3", "4"),
        experiment_seeds=EXPERIMENT_SEEDS,
        support_seeds=SUPPORT_SEEDS,
        support_sizes=SUPPORT_SIZES,
        generation_seeds=GENERATION_SEEDS,
        classifier_seeds=CLASSIFIER_SEEDS,
        primary_generation_mode=PRIMARY_GENERATION_MODE,
        negative_control_generation_mode=NEGATIVE_CONTROL_GENERATION_MODE,
        primary_budget_per_class=PRIMARY_BUDGET_PER_CLASS,
        diagnostic_budgets_per_class=DIAGNOSTIC_BUDGETS_PER_CLASS,
        support_selection_glob=(
            "cvae_testing/outputs/camelyon17/"
            "camelyon17_support_estimated_utility_routing_v2/"
            "support_utility_v2_seed*/reports/support_response_sample_selections.csv"
        ),
        artifacts_root="cvae_downstream_evaluation/artifacts",
        required_external_artifacts=(
            "cvae_downstream_evaluation/artifacts/manifests/expert_checkpoints.csv",
            "cvae_downstream_evaluation/artifacts/manifests/embedding_cache_manifest.csv",
            "cvae_downstream_evaluation/artifacts/manifests/expert_provenance.csv",
        ),
    )


def load_locked_v1_config(path: Path) -> LockedV1Config:
    """Load and validate the v1 config.

    PyYAML is optional. If it is unavailable, this function still enforces the
    locked protocol by text-scanning the canonical config and returning the
    normalized v1 defaults. On the workstation, PyYAML can provide a deeper
    structured validation path.
    """

    text = Path(path).read_text(encoding="utf-8")
    assert_locked_v1_config_text(text)

    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return default_v1_config()

    loaded = yaml.safe_load(text) or {}
    assert_locked_v1_config_mapping(loaded)
    return _config_from_mapping(loaded)


def assert_locked_v1_config_text(text: str) -> None:
    """Reject stale template text that conflicts with the frozen v1 plan."""

    forbidden_snippets = (
        "TODO_LOCK_BEFORE_RUN",
        "TODO_LINK_TO_SUPPORT_ROUTING_OR_CVAE_TESTING_ARTIFACTS",
        "conditional_cvae_decoder",
        "linear_or_small_mlp",
        "target_support_pseudo_prior",
        "breakhis:\n    enabled: true",
        "midogpp:\n    enabled: true",
    )
    present = [snippet for snippet in forbidden_snippets if snippet in text]
    if present:
        joined = ", ".join(present)
        raise ProtocolError(f"Config contains stale or forbidden v1 fields: {joined}")

    required_snippets = (
        f"name: {EXPERIMENT_NAME}",
        "enabled: true",
        "class_stratified_reference_posterior_resampling",
        "unconditional_prior_sampling_assigned_label_negative_control",
        "primary_budget_per_class: 128",
        "support_size_stratified_downstream_summary.csv",
        "solver: lbfgs",
        "class_weight: null",
    )
    missing = [snippet for snippet in required_snippets if snippet not in text]
    if missing:
        joined = ", ".join(missing)
        raise ProtocolError(f"Config is missing locked v1 fields: {joined}")


def assert_locked_v1_config_mapping(config: Mapping[str, Any]) -> None:
    """Validate the structured v1 config loaded from YAML."""

    experiment = _mapping(config.get("experiment"), "experiment")
    if experiment.get("name") != EXPERIMENT_NAME:
        raise ProtocolError(f"Unexpected experiment.name: {experiment.get('name')!r}")
    if experiment.get("stage") != "second_stage_downstream_evaluation":
        raise ProtocolError("experiment.stage must be second_stage_downstream_evaluation")

    datasets = _mapping(config.get("datasets"), "datasets")
    enabled = [
        str(name)
        for name, value in datasets.items()
        if isinstance(value, Mapping) and bool(value.get("enabled"))
    ]
    if enabled != [DATASET_NAME]:
        raise ProtocolError(f"v1 must enable only {DATASET_NAME}; got {enabled}")

    camelyon = _mapping(datasets.get(DATASET_NAME), f"datasets.{DATASET_NAME}")
    _assert_equal_tuple(camelyon.get("support_sizes"), SUPPORT_SIZES, "support_sizes")
    _assert_equal_tuple(camelyon.get("support_seeds"), SUPPORT_SEEDS, "support_seeds")
    _assert_equal_tuple(camelyon.get("experiment_seeds"), EXPERIMENT_SEEDS, "experiment_seeds")
    _assert_equal_tuple(camelyon.get("generation_seeds"), GENERATION_SEEDS, "generation_seeds")
    _assert_equal_tuple(camelyon.get("classifier_seeds"), CLASSIFIER_SEEDS, "classifier_seeds")

    generation = _mapping(config.get("generation"), "generation")
    if generation.get("primary_mode") != PRIMARY_GENERATION_MODE:
        raise ProtocolError("generation.primary_mode is not locked to reference-posterior resampling")
    if generation.get("negative_control_mode") != NEGATIVE_CONTROL_GENERATION_MODE:
        raise ProtocolError("generation.negative_control_mode is not locked")
    if generation.get("primary_budget_per_class") != PRIMARY_BUDGET_PER_CLASS:
        raise ProtocolError("generation.primary_budget_per_class must be 128")
    _assert_equal_tuple(
        generation.get("diagnostic_budgets_per_class"),
        DIAGNOSTIC_BUDGETS_PER_CLASS,
        "diagnostic_budgets_per_class",
    )
    if generation.get("reference_pool") != "expert_source_train":
        raise ProtocolError("generation.reference_pool must be expert_source_train")

    classifier = _mapping(_mapping(config.get("downstream"), "downstream").get("classifier"), "classifier")
    expected_classifier = {
        "family": "sklearn_logistic_regression",
        "solver": "lbfgs",
        "C": 1.0,
        "max_iter": 2000,
        "class_weight": None,
        "scaler_fit": "synthetic_train_only",
        "hyperparameter_tuning": "forbidden",
    }
    for key, expected in expected_classifier.items():
        if classifier.get(key) != expected:
            raise ProtocolError(f"classifier.{key} must be {expected!r}; got {classifier.get(key)!r}")

    baselines = _mapping(config.get("baselines"), "baselines")
    configured = tuple(str(v) for v in baselines.get("essential", ()))
    missing = sorted(set(ESSENTIAL_BASELINES).difference(configured))
    if missing:
        raise ProtocolError(f"Missing essential baselines: {missing}")


def resolve_required_external_artifacts(config: LockedV1Config, repo_root: Path) -> tuple[Path, ...]:
    """Return required frozen artifact paths or fail with a sync error."""

    missing: list[Path] = []
    resolved: list[Path] = []
    for raw in config.required_external_artifacts:
        path = Path(raw)
        if not path.is_absolute():
            path = repo_root / path
        resolved.append(path)
        if not path.exists():
            missing.append(path)
    if missing:
        preview = "\n".join(f"- {path}" for path in missing)
        raise ArtifactSyncError(
            "Missing required frozen downstream artifacts. Sync the workstation "
            f"manifests/checkpoints before running v1:\n{preview}"
        )
    return tuple(resolved)


def assert_no_forbidden_router_inputs(inputs: Mapping[str, object]) -> None:
    """Reject configs or manifests that expose target-evaluation information."""

    present = sorted(k for k in FORBIDDEN_ROUTER_INPUTS if k in inputs)
    if present:
        joined = ", ".join(present)
        raise ProtocolError(f"Forbidden router inputs present: {joined}")


def assert_manifest_ready(manifest: ProtocolManifest) -> None:
    """Validate manifest-level protocol gates before downstream work starts."""

    if not manifest.forbidden_router_inputs_checked:
        raise ProtocolError("Forbidden router input check has not passed.")
    if not manifest.support_eval_disjoint_checked:
        raise ProtocolError("Support/evaluation disjointness check has not passed.")
    if not manifest.target_expert_excluded_checked:
        raise ProtocolError("Target expert exclusion check has not passed.")
    if not manifest.candidate_expert_domains:
        raise ProtocolError("Candidate expert pool is empty.")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{name} must be a mapping.")
    return value


def _assert_equal_tuple(value: Any, expected: Sequence[int], field: str) -> None:
    observed = tuple(int(v) for v in (value or ()))
    if observed != tuple(expected):
        raise ProtocolError(f"{field} must be {tuple(expected)}, got {observed}")


def _config_from_mapping(config: Mapping[str, Any]) -> LockedV1Config:
    base = default_v1_config()
    artifacts = _mapping(config.get("artifacts"), "artifacts")
    support_inputs = _mapping(config.get("support_inputs"), "support_inputs")
    external = tuple(str(v) for v in artifacts.get("required_external", base.required_external_artifacts))
    return LockedV1Config(
        name=base.name,
        dataset_name=base.dataset_name,
        domain_key=base.domain_key,
        candidate_domains=base.candidate_domains,
        experiment_seeds=base.experiment_seeds,
        support_seeds=base.support_seeds,
        support_sizes=base.support_sizes,
        generation_seeds=base.generation_seeds,
        classifier_seeds=base.classifier_seeds,
        primary_generation_mode=base.primary_generation_mode,
        negative_control_generation_mode=base.negative_control_generation_mode,
        primary_budget_per_class=base.primary_budget_per_class,
        diagnostic_budgets_per_class=base.diagnostic_budgets_per_class,
        support_selection_glob=str(support_inputs.get("selection_glob", base.support_selection_glob)),
        artifacts_root=str(artifacts.get("root", base.artifacts_root)),
        required_external_artifacts=external,
    )
