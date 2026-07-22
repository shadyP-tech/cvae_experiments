"""Frozen configuration contract for conditional-logit alignment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from ..classifiers import ClassifierSpec
from ..protocol import ProtocolError
from ..schemas.midogpp import MIDOGPP_ELIGIBLE_CENTERS


METHOD = "conditional_logit_alignment"
EXPERIMENT_NAME = "conditional_logit_alignment_v1"
CODE_VERSION = "conditional_logit_alignment_v1"
GAMMA_GRID = (0.0, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0)
FIXED_C = 0.01
TRACE_ATOL = 1.0e-12
TRACE_RTOL = 0.0
MIN_PENALTY_SCALE = 1.0e-12
TIE_ATOL = 1.0e-12
TIE_RTOL = 0.0
EXPECTED_VERSIONS: Mapping[str, str] = {
    "numpy": "1.26.4",
    "scipy": "1.17.1",
    "scikit_learn": "1.8.0",
}


@dataclass(frozen=True)
class AlignmentOptimizerConfig:
    """Numerical locks for the positive-gamma SciPy fits."""

    tol: float = 1.0e-4
    gradient_inf_norm_max: float = 1.0e-4
    ftol_float64_eps_multiplier: int = 64
    max_line_search_steps: int = 50
    max_iter: int = 5000
    require_single_thread: bool = True
    objective_atol: float = 1.0e-10
    coefficient_atol: float = 1.0e-6
    coefficient_rtol: float = 1.0e-6
    probability_atol: float = 1.0e-8
    probability_rtol: float = 1.0e-7
    predictions_exact: bool = True


DEFAULT_OPTIMIZER_CONFIG = AlignmentOptimizerConfig()


def canonical_classifier_spec() -> ClassifierSpec:
    """Return the single classifier specification shared by every CLA fit."""

    return ClassifierSpec(
        C=FIXED_C,
        penalty="l2",
        solver="lbfgs",
        max_iter=5000,
        class_weight=None,
        random_state=23,
        threshold_policy="predict",
    )


@dataclass(frozen=True)
class ConditionalLogitAlignmentConfig:
    """Resolved production configuration plus a test-only partial-coverage switch."""

    name: str
    artifact_root: Path
    manifest_path: Path
    feature_cache_path: Path
    heldout_centers: tuple[str, ...] = MIDOGPP_ELIGIBLE_CENTERS
    experiment_seed: int = 42
    expected_feature_dim: int = 2560
    classifier_spec: ClassifierSpec = field(default_factory=canonical_classifier_spec)
    gamma_grid: tuple[float, ...] = GAMMA_GRID
    optimizer: AlignmentOptimizerConfig = DEFAULT_OPTIMIZER_CONFIG
    expected_versions: Mapping[str, str] = field(
        default_factory=lambda: dict(EXPECTED_VERSIONS)
    )
    mode: str = METHOD
    code_version: str = CODE_VERSION
    expected_outer_fold_count: int = 9
    expected_inner_folds_per_outer: int = 8
    expected_gamma_count: int = 7
    expected_inner_score_count: int = 504
    expected_gamma_summary_count: int = 63
    expected_outer_result_count: int = 18
    expected_outer_comparison_count: int = 9
    expected_frame_audit_count: int = 81
    trace_atol: float = TRACE_ATOL
    trace_rtol: float = TRACE_RTOL
    tie_atol: float = TIE_ATOL
    tie_rtol: float = TIE_RTOL
    allow_partial_test_coverage: bool = False
    # Set only by the resolved-config loader. Complete runs must originate from
    # the workspace-owned snapshot at <canonical artifact root>/config.resolved.yaml.
    config_source_path: Path | None = None

    def __post_init__(self) -> None:
        validate_runtime_config(self)


def load_conditional_logit_alignment_config(
    path: str | Path,
) -> ConditionalLogitAlignmentConfig:
    """Load the exact frozen production design from a resolved YAML config."""

    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - package dependency
        raise RuntimeError("Conditional-logit alignment configs require PyYAML.") from exc

    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ProtocolError("Conditional-logit alignment config must be a mapping.")
    _require_exact_keys(
        payload,
        (
            "experiment",
            "inputs",
            "run",
            "classifier",
            "alignment",
            "selection",
            "optimizer",
            "decision",
            "claim_boundary",
        ),
        "config",
    )
    experiment = _mapping(payload["experiment"], "experiment")
    inputs = _mapping(payload["inputs"], "inputs")
    run = _mapping(payload["run"], "run")
    classifier = _mapping(payload["classifier"], "classifier")
    alignment = _mapping(payload["alignment"], "alignment")
    selection = _mapping(payload["selection"], "selection")
    optimizer = _mapping(payload["optimizer"], "optimizer")
    decision = _mapping(payload["decision"], "decision")
    claim = _mapping(payload["claim_boundary"], "claim_boundary")
    _validate_frozen_sections(
        experiment=experiment,
        inputs=inputs,
        run=run,
        classifier=classifier,
        alignment=alignment,
        selection=selection,
        optimizer=optimizer,
        decision=decision,
        claim=claim,
    )

    base = config_path.parent
    config = ConditionalLogitAlignmentConfig(
        name=str(experiment["name"]),
        mode=str(experiment["mode"]),
        code_version=str(experiment["code_version"]),
        artifact_root=_resolve_path(base, str(experiment["artifact_root"])),
        manifest_path=_resolve_path(base, str(inputs["manifest_path"])),
        feature_cache_path=_resolve_path(base, str(inputs["feature_cache_path"])),
        heldout_centers=MIDOGPP_ELIGIBLE_CENTERS,
        experiment_seed=int(run["experiment_seed"]),
        expected_feature_dim=int(run["expected_feature_dim"]),
        classifier_spec=canonical_classifier_spec(),
        gamma_grid=tuple(float(value) for value in alignment["gamma_grid"]),
        optimizer=AlignmentOptimizerConfig(
            tol=float(optimizer["tol"]),
            gradient_inf_norm_max=float(optimizer["gradient_inf_norm_max"]),
            ftol_float64_eps_multiplier=int(optimizer["ftol_float64_eps_multiplier"]),
            max_line_search_steps=int(optimizer["max_line_search_steps"]),
            max_iter=int(optimizer["max_iter"]),
            require_single_thread=bool(optimizer["require_single_thread"]),
            objective_atol=float(_mapping(optimizer["parity"], "optimizer.parity")["objective_atol"]),
            coefficient_atol=float(_mapping(optimizer["parity"], "optimizer.parity")["coefficient_atol"]),
            coefficient_rtol=float(_mapping(optimizer["parity"], "optimizer.parity")["coefficient_rtol"]),
            probability_atol=float(_mapping(optimizer["parity"], "optimizer.parity")["probability_atol"]),
            probability_rtol=float(_mapping(optimizer["parity"], "optimizer.parity")["probability_rtol"]),
            predictions_exact=bool(_mapping(optimizer["parity"], "optimizer.parity")["predictions_exact"]),
        ),
        expected_versions={
            str(key): str(value)
            for key, value in _mapping(
                optimizer["expected_versions"], "optimizer.expected_versions"
            ).items()
        },
        expected_outer_fold_count=int(run["expected_outer_fold_count"]),
        expected_inner_folds_per_outer=int(run["expected_inner_folds_per_outer"]),
        expected_gamma_count=int(run["expected_gamma_count"]),
        expected_inner_score_count=int(run["expected_inner_score_count"]),
        expected_gamma_summary_count=int(run["expected_gamma_summary_count"]),
        expected_outer_result_count=int(run["expected_outer_result_count"]),
        expected_outer_comparison_count=int(run["expected_outer_comparison_count"]),
        expected_frame_audit_count=int(run["expected_frame_audit_count"]),
        trace_atol=float(alignment["trace_atol"]),
        trace_rtol=float(alignment["trace_rtol"]),
        tie_atol=float(selection["tie_atol"]),
        tie_rtol=float(selection["tie_rtol"]),
        config_source_path=config_path,
    )
    return config


def validate_runtime_config(config: ConditionalLogitAlignmentConfig) -> None:
    """Fail closed if code constructs a runtime config outside the frozen design."""

    if (
        config.name != EXPERIMENT_NAME
        or config.mode != METHOD
        or config.code_version != CODE_VERSION
    ):
        raise ProtocolError("Conditional-logit alignment experiment identity drifted.")
    _validate_classifier_spec(config.classifier_spec)
    if tuple(float(value) for value in config.gamma_grid) != GAMMA_GRID:
        raise ProtocolError("Conditional-logit alignment gamma grid drifted.")
    if config.trace_atol != TRACE_ATOL or config.trace_rtol != TRACE_RTOL:
        raise ProtocolError("Conditional-logit alignment trace tolerance drifted.")
    if config.tie_atol != TIE_ATOL or config.tie_rtol != TIE_RTOL:
        raise ProtocolError("Conditional-logit alignment tie tolerance drifted.")
    if config.optimizer != DEFAULT_OPTIMIZER_CONFIG:
        raise ProtocolError("Conditional-logit alignment optimizer locks drifted.")
    if dict(config.expected_versions) != dict(EXPECTED_VERSIONS):
        raise ProtocolError("Conditional-logit alignment expected library versions drifted.")
    if tuple(config.heldout_centers) != tuple(dict.fromkeys(config.heldout_centers)):
        raise ProtocolError("Conditional-logit alignment held-out centers must be unique.")
    if not config.allow_partial_test_coverage:
        expected_counts = (
            config.expected_outer_fold_count,
            config.expected_inner_folds_per_outer,
            config.expected_gamma_count,
            config.expected_inner_score_count,
            config.expected_gamma_summary_count,
            config.expected_outer_result_count,
            config.expected_outer_comparison_count,
            config.expected_frame_audit_count,
        )
        if (
            config.heldout_centers != MIDOGPP_ELIGIBLE_CENTERS
            or config.experiment_seed != 42
            or config.expected_feature_dim != 2560
            or expected_counts != (9, 8, 7, 504, 63, 18, 9, 81)
        ):
            raise ProtocolError("Conditional-logit alignment production run locks drifted.")


def _validate_frozen_sections(
    *,
    experiment: Mapping[str, object],
    inputs: Mapping[str, object],
    run: Mapping[str, object],
    classifier: Mapping[str, object],
    alignment: Mapping[str, object],
    selection: Mapping[str, object],
    optimizer: Mapping[str, object],
    decision: Mapping[str, object],
    claim: Mapping[str, object],
) -> None:
    _require_exact_keys(experiment, ("name", "mode", "artifact_root", "code_version"), "experiment")
    if (
        experiment["name"] != EXPERIMENT_NAME
        or experiment["mode"] != METHOD
        or experiment["code_version"] != CODE_VERSION
        or not str(experiment["artifact_root"]).strip()
    ):
        raise ProtocolError("Conditional-logit alignment experiment section drifted.")

    _require_exact_keys(inputs, ("manifest_path", "feature_cache_path", "split"), "inputs")
    if (
        not str(inputs["manifest_path"]).strip()
        or not str(inputs["feature_cache_path"]).strip()
        or inputs["split"] != "train"
    ):
        raise ProtocolError("Conditional-logit alignment input contract drifted.")

    expected_run = {
        "experiment_seed": 42,
        "heldout_centers": "all",
        "expected_feature_dim": 2560,
        "expected_outer_fold_count": 9,
        "expected_inner_folds_per_outer": 8,
        "expected_gamma_count": 7,
        "expected_inner_score_count": 504,
        "expected_gamma_summary_count": 63,
        "expected_outer_result_count": 18,
        "expected_outer_comparison_count": 9,
        "expected_frame_audit_count": 81,
    }
    _require_exact_mapping(run, expected_run, "run")

    expected_classifier = {
        "C": FIXED_C,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 5000,
        "class_weight": "none",
        "sample_weight": "none",
        "random_state": 23,
        "threshold_policy": "predict",
        "fit_intercept": True,
        "intercept_penalized": False,
        "dtype": "float64",
        "scaler_fit": "fit_rows_only",
    }
    _require_exact_mapping(classifier, expected_classifier, "classifier")

    expected_alignment = {
        "gamma_grid": list(GAMMA_GRID),
        "centroid_weighting": "equal_domain_class_cells",
        "class_centering": "equal_domain_mean_within_class",
        "construction_frame": "standardized_fit_rows",
        "factor_representation": "rectangular_contrast_factor",
        "normalization": "unit_trace",
        "trace_atol": TRACE_ATOL,
        "trace_rtol": TRACE_RTOL,
        "missing_cell_policy": "fail_closed",
        "nonfinite_policy": "fail_closed",
        "dense_matrix_materialized": False,
    }
    _require_exact_mapping(alignment, expected_alignment, "alignment")

    expected_selection = {
        "outer_protocol": "eligible_center_lodo",
        "inner_protocol": "source_inner_center_lodo",
        "metric": "bacc",
        "aggregation": "equal_center_arithmetic_mean",
        "tie_atol": TIE_ATOL,
        "tie_rtol": TIE_RTOL,
        "tie_break": "smallest_gamma",
        "outer_score_roles": ["selected", "gamma0"],
        "outer_all_gamma_scoring": False,
        "outer_oracle_gamma_computed": False,
    }
    _require_exact_mapping(selection, expected_selection, "selection")

    expected_parity = {
        "objective_atol": 1.0e-10,
        "coefficient_atol": 1.0e-6,
        "coefficient_rtol": 1.0e-6,
        "probability_atol": 1.0e-8,
        "probability_rtol": 1.0e-7,
        "predictions_exact": True,
    }
    parity = _mapping(optimizer.get("parity"), "optimizer.parity")
    _require_exact_mapping(parity, expected_parity, "optimizer.parity")
    expected_optimizer = {
        "pooled_backend": "sklearn_lbfgs",
        "aligned_backend": "scipy_lbfgsb",
        "warm_start": "pooled_gamma0_solution",
        "tol": 1.0e-4,
        "gradient_inf_norm_max": 1.0e-4,
        "ftol_float64_eps_multiplier": 64,
        "max_line_search_steps": 50,
        "max_iter": 5000,
        "expected_versions": dict(EXPECTED_VERSIONS),
        "require_single_thread": True,
        "parity": dict(expected_parity),
    }
    _require_exact_mapping(optimizer, expected_optimizer, "optimizer")

    expected_decision = {
        "primary_contrast": "selected_minus_gamma0",
        "primary_metric": "equal_center_mean_bacc",
        "numerical_epsilon": 1.0e-12,
        "pass_requires_positive_mean_delta": True,
        "pass_requires_nonworse_minimum_center": True,
        "pass_min_nonnegative_center_deltas": 5,
        "macro_f1_role": "secondary_descriptive",
        "invalid_bundle_decision": "REJECTED",
    }
    _require_exact_mapping(decision, expected_decision, "decision")

    expected_claim = {
        "claim_scope": "real_feature_transfer_only",
        "diagnostic_only": True,
        "non_adoptive": True,
        "target_evaluation_labels_used_for_fit": False,
        "target_evaluation_labels_used_for_selection": False,
        "target_evaluation_labels_used_for_scoring_only": True,
        "source_inner_labels_used_for_selection": True,
        "support_labels_used": False,
        "uses_generated_embeddings": False,
        "uses_cvae_checkpoint": False,
        "uses_encoder_posterior": False,
        "uses_decoder_likelihood": False,
        "uses_prior": False,
        "uses_nelbo": False,
        "uses_latent_representation": False,
        "models_embedding_distribution": False,
        "uses_expert_bank": False,
        "uses_router": False,
        "performs_expert_selection": False,
        "performs_expert_weighting": False,
        "performs_aggregation": False,
        "compatibility_signal": "none",
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
    }
    _require_exact_keys(claim, (*expected_claim.keys(), "allowed", "forbidden"), "claim_boundary")
    for key, expected in expected_claim.items():
        if claim[key] != expected:
            raise ProtocolError(f"Conditional-logit alignment claim boundary {key!r} drifted.")
    if not str(claim["allowed"]).strip() or not str(claim["forbidden"]).strip():
        raise ProtocolError("Conditional-logit alignment claim text must be nonempty.")


def _validate_classifier_spec(spec: ClassifierSpec) -> None:
    expected = canonical_classifier_spec()
    if spec.to_payload() != expected.to_payload():
        raise ProtocolError(
            "Conditional-logit alignment requires fixed C=0.01 sklearn lbfgs, "
            "no class weights, and the frozen classifier seed/settings."
        )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Conditional-logit alignment {label} must be a mapping.")
    return value


def _require_exact_keys(
    payload: Mapping[str, object], expected: Sequence[str], label: str
) -> None:
    actual = set(payload)
    required = set(expected)
    if actual != required:
        raise ProtocolError(
            f"Conditional-logit alignment {label} keys drifted: "
            f"missing={sorted(required - actual)} unexpected={sorted(actual - required)}"
        )


def _require_exact_mapping(
    payload: Mapping[str, object], expected: Mapping[str, object], label: str
) -> None:
    _require_exact_keys(payload, tuple(expected), label)
    if dict(payload) != dict(expected):
        raise ProtocolError(f"Conditional-logit alignment {label} values drifted.")


def _resolve_path(base: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (base / path).resolve()


__all__ = [
    "AlignmentOptimizerConfig",
    "CODE_VERSION",
    "ConditionalLogitAlignmentConfig",
    "DEFAULT_OPTIMIZER_CONFIG",
    "EXPERIMENT_NAME",
    "EXPECTED_VERSIONS",
    "FIXED_C",
    "GAMMA_GRID",
    "METHOD",
    "MIN_PENALTY_SCALE",
    "TIE_ATOL",
    "TIE_RTOL",
    "TRACE_ATOL",
    "TRACE_RTOL",
    "canonical_classifier_spec",
    "load_conditional_logit_alignment_config",
    "validate_runtime_config",
]
