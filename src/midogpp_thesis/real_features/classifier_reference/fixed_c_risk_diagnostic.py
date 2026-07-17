"""Fixed-C four-arm risk-weighting diagnostic for MIDOG++ real features."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

from .artifacts import (
    prepare_artifact_dirs,
    stable_hash,
    write_csv_rows,
    write_frozen_snapshot,
    write_json,
)
from .classifiers import ClassifierSpec, fit_logistic_classifier
from .downstream import balanced_accuracy, macro_f1
from .fixed_c_risk_artifact_validation import assert_fixed_c_risk_artifacts
from .fixed_c_risk_reporting import (
    build_diagnostic_summary,
    render_diagnostic_report,
)
from .protocol import ProtocolError
from .real_feature_frame import RealFeatureFrame, load_midogpp_real_feature_frame
from .schemas.fixed_c_risk_diagnostic import (
    FIXED_C_RISK_CODE_VERSION,
    FIXED_C_RISK_EXPERIMENT_ID,
    FIXED_C_RISK_EXPERIMENT_NAME,
    FIXED_C_RISK_METHOD,
    FIXED_C_RISK_PAIRED_COLUMNS,
    FIXED_C_RISK_PAIRED_SCHEMA_VERSION,
    FIXED_C_RISK_PREDICTION_COLUMNS,
    FIXED_C_RISK_PREDICTION_SCHEMA_VERSION,
    FIXED_C_RISK_RESULT_COLUMNS,
    FIXED_C_RISK_RESULT_SCHEMA_VERSION,
    FIXED_C_RISK_SCHEMA_VERSION,
    FIXED_C_RISK_WEIGHT_AUDIT_COLUMNS,
    FIXED_C_RISK_WEIGHT_AUDIT_SCHEMA_VERSION,
    FIXED_CLASSIFIER_CONFIG_HASH,
    PRIMARY_CONTRAST,
    PRIOR_METHOD,
    RISK_POLICY_FORMULAS,
    RISK_POLICY_IDS,
    SELECTION_SOURCE,
    WEIGHT_NORMALIZATION,
    ZERO_CELL_POLICY,
    canonical_fixed_classifier_spec,
    expected_frozen_snapshot,
    fixed_c_risk_bundle_hash,
    risk_policy_hash,
)
from .schemas.midogpp import MIDOGPP_ELIGIBLE_CENTERS, MIDOGPP_EXCLUDED_CENTERS


@dataclass(frozen=True)
class FixedCRiskDiagnosticConfig:
    name: str
    artifact_root: Path
    manifest_path: Path
    feature_cache_path: Path
    heldout_centers: tuple[str, ...]
    experiment_seed: int = 42
    expected_feature_dim: int = 2560
    classifier_spec: ClassifierSpec = canonical_fixed_classifier_spec()
    mode: str = FIXED_C_RISK_METHOD
    code_version: str = FIXED_C_RISK_CODE_VERSION
    expected_outer_fold_count: int = 9
    expected_arm_count: int = 4
    expected_fit_count: int = 36
    allow_partial_test_coverage: bool = False


@dataclass(frozen=True)
class RiskWeightPlan:
    risk_policy_id: str
    formula: str
    weights: tuple[float, ...]
    group_counts: Mapping[str, int]
    group_weights: Mapping[str, float]
    group_masses: Mapping[str, float]
    fit_sample_ids: tuple[str, ...]

    @property
    def risk_policy_hash(self) -> str:
        return risk_policy_hash(self.risk_policy_id)

    @property
    def weight_vector_hash(self) -> str:
        return stable_hash(
            {
                "fit_row_hash": _row_hash(self.fit_sample_ids),
                "ordered_weights": [float(value) for value in self.weights],
            }
        )


def compute_risk_weights(
    labels: Sequence[int],
    domains: Sequence[str],
    risk_policy_id: str,
    *,
    sample_ids: Sequence[str] | None = None,
) -> RiskWeightPlan:
    """Compute one exact, sum-to-N fit-weight vector in input row order."""

    policy = str(risk_policy_id)
    if policy not in RISK_POLICY_IDS:
        raise ProtocolError(f"Unknown fixed-C risk policy: {policy!r}")
    y = tuple(int(value) for value in labels)
    d = tuple(str(value) for value in domains)
    if not y or len(y) != len(d):
        raise ProtocolError("Fixed-C risk weights require aligned nonempty labels/domains.")
    ids = (
        tuple(str(value) for value in sample_ids)
        if sample_ids is not None
        else tuple(str(index) for index in range(len(y)))
    )
    if len(ids) != len(y) or len(set(ids)) != len(ids):
        raise ProtocolError(
            "Fixed-C risk weight hashing requires aligned unique fit sample IDs."
        )
    if set(y) != {0, 1}:
        raise ProtocolError("Fixed-C risk weights require both binary classes.")
    domain_ids = tuple(dict.fromkeys(d))
    if not domain_ids or any(not value for value in domain_ids):
        raise ProtocolError("Fixed-C risk weights require nonempty source domains.")
    n_fit = len(y)
    n_domains = len(domain_ids)

    if policy == "pooled":
        keys = tuple("all" for _ in y)
    elif policy == "global_class":
        keys = tuple(f"class={label}" for label in y)
    elif policy == "domain":
        keys = tuple(f"domain={domain}" for domain in d)
    else:
        present = {(domain, label) for domain, label in zip(d, y)}
        required = {(domain, label) for domain in domain_ids for label in (0, 1)}
        missing = sorted(required.difference(present))
        if missing:
            raise ProtocolError(
                "Fixed-C domain_class weighting is undefined because source "
                f"domain×class cells are missing: {missing}"
            )
        keys = tuple(
            f"domain={domain}|class={label}" for domain, label in zip(d, y)
        )
    counts: dict[str, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    ordered_keys = _canonical_group_keys(policy, domain_ids)
    if set(counts) != set(ordered_keys):
        missing = sorted(set(ordered_keys).difference(counts))
        raise ProtocolError(
            f"Fixed-C risk weighting has missing required groups: {missing}"
        )
    counts = {key: counts[key] for key in ordered_keys}
    group_weights = {
        key: _formula_weight(policy, n_fit, n_domains, count)
        for key, count in counts.items()
    }
    weights = tuple(float(group_weights[key]) for key in keys)
    group_masses = {
        key: float(counts[key]) * float(group_weights[key]) for key in ordered_keys
    }
    if (
        any(not _finite_positive(value) for value in weights)
        or abs(sum(weights) - float(n_fit)) > 1e-10
    ):
        raise ProtocolError("Fixed-C risk weights violate finite positive sum-to-N.")
    return RiskWeightPlan(
        risk_policy_id=policy,
        formula=RISK_POLICY_FORMULAS[policy],
        weights=weights,
        group_counts=counts,
        group_weights=group_weights,
        group_masses=group_masses,
        fit_sample_ids=ids,
    )


def run_fixed_c_risk_diagnostic(
    config: FixedCRiskDiagnosticConfig,
    *,
    artifact_root: Path | None = None,
) -> Path:
    """Run all four predeclared arms and write the complete diagnostic bundle."""

    started = time.perf_counter()
    _validate_runtime_config(config)
    root = prepare_artifact_dirs(artifact_root or config.artifact_root)
    frame = load_midogpp_real_feature_frame(
        manifest_path=config.manifest_path,
        feature_cache_path=config.feature_cache_path,
        expected_feature_dim=config.expected_feature_dim,
    )
    heldouts = tuple(_eligible_present(frame, center) for center in config.heldout_centers)
    coverage_mode = "partial_test" if config.allow_partial_test_coverage else "complete"
    if not config.allow_partial_test_coverage:
        if (
            frame.eligible_centers != MIDOGPP_ELIGIBLE_CENTERS
            or heldouts != MIDOGPP_ELIGIBLE_CENTERS
        ):
            raise ProtocolError(
                "Production fixed-C risk diagnostic requires exact nine-center coverage."
            )

    expected_outer = len(heldouts) if config.allow_partial_test_coverage else 9
    expected_fits = expected_outer * len(RISK_POLICY_IDS)
    protocol: dict[str, object] = {
        "schema_version": FIXED_C_RISK_SCHEMA_VERSION,
        "experiment_id": FIXED_C_RISK_EXPERIMENT_ID,
        "experiment_name": config.name,
        "mode": config.mode,
        "code_version": config.code_version,
        "method": FIXED_C_RISK_METHOD,
        "experiment_seed": int(config.experiment_seed),
        "classifier_seed": int(config.classifier_spec.random_state),
        "fixed_classifier_config_hash": config.classifier_spec.config_hash,
        "fixed_classifier_spec": config.classifier_spec.to_payload(),
        "threshold_policy": config.classifier_spec.threshold_policy,
        "heldout_centers": list(heldouts),
        "eligible_centers": list(frame.eligible_centers),
        "excluded_centers": list(MIDOGPP_EXCLUDED_CENTERS),
        "coverage_mode": coverage_mode,
        "expected_outer_fold_count": expected_outer,
        "expected_arm_count": len(RISK_POLICY_IDS),
        "expected_fit_count": expected_fits,
        "expected_feature_dim": int(config.expected_feature_dim),
        "manifest_path": str(Path(config.manifest_path).resolve()),
        "feature_cache_path": str(Path(config.feature_cache_path).resolve()),
        "manifest_hash": frame.manifest_hash,
        "feature_cache_hash": frame.feature_cache_hash,
        "risk_policy_ids": list(RISK_POLICY_IDS),
        "risk_policy_formulas": dict(RISK_POLICY_FORMULAS),
        "risk_policy_hashes": {
            policy: risk_policy_hash(policy) for policy in RISK_POLICY_IDS
        },
        "normalization": WEIGHT_NORMALIZATION,
        "zero_cell_policy": ZERO_CELL_POLICY,
        "require_finite_positive_weights": True,
        "primary_contrast": PRIMARY_CONTRAST,
        "paired_by": "heldout_center",
        "selection_source": SELECTION_SOURCE,
        "selection_performed": False,
        "scaler_fit_scope": "outer_source_train_only",
        "scaler_weighting": "unweighted",
        "sample_weight_scope": "logistic_regression_fit_only",
        "prior_method": PRIOR_METHOD,
        "claim_scope": "real_feature_transfer_only",
        "claim_role": "risk_weighting_diagnostic",
        "diagnostic_only": True,
        "non_adoptive": True,
        "target_eval_labels_used_for_scoring_only": True,
        "target_eval_labels_used_for_fit": False,
        "target_eval_labels_used_for_selection": False,
        "uses_cvae_checkpoint": False,
        "uses_generated_embeddings": False,
        "uses_prior": False,
        "uses_router": False,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
        "support_labels_used": False,
        "oracle_eligible": False,
    }
    frozen = expected_frozen_snapshot(protocol)
    write_frozen_snapshot(root / "manifests/frozen_protocol_snapshot.json", frozen)

    result_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    weight_audit_rows: list[dict[str, object]] = []
    overlap_rows: list[dict[str, object]] = []
    for heldout in heldouts:
        train_centers = tuple(
            center for center in frame.eligible_centers if center != heldout
        )
        train_idx = _indices(frame, train_centers)
        eval_idx = _indices(frame, (heldout,))
        fit_ids = tuple(frame.rows[index].sample_id for index in train_idx)
        eval_ids = tuple(frame.rows[index].sample_id for index in eval_idx)
        fit_cases = {frame.rows[index].case_id for index in train_idx}
        eval_cases = {frame.rows[index].case_id for index in eval_idx}
        sample_overlap = set(fit_ids).intersection(eval_ids)
        case_overlap = fit_cases.intersection(eval_cases)
        if sample_overlap or case_overlap:
            raise ProtocolError(
                "Fixed-C risk fit/eval sample or case identities overlap; "
                "refusing to fit any classifier."
            )
        x_train, y_train = _arrays(frame, train_idx)
        x_eval, y_eval = _arrays(frame, eval_idx)
        train_domains = tuple(frame.rows[index].center for index in train_idx)
        fit_row_hash = _row_hash(fit_ids)
        eval_row_hash = _row_hash(eval_ids)
        training_frame_hash = stable_hash(
            {
                "manifest_hash": frame.manifest_hash,
                "feature_cache_hash": frame.feature_cache_hash,
                "expected_feature_dim": int(config.expected_feature_dim),
                "train_centers": list(train_centers),
                "fit_row_hash": fit_row_hash,
            }
        )
        weight_plans = {
            policy: compute_risk_weights(
                y_train,
                train_domains,
                policy,
                sample_ids=fit_ids,
            )
            for policy in RISK_POLICY_IDS
        }
        fold_scaler_hash: str | None = None
        for policy in RISK_POLICY_IDS:
            weight_plan = weight_plans[policy]
            fitted = fit_logistic_classifier(
                x_train,
                y_train,
                x_eval,
                spec=config.classifier_spec,
                sample_weight=None if policy == "pooled" else weight_plan.weights,
            )
            if not fitted.converged:
                raise ProtocolError(
                    f"Fixed-C risk classifier did not converge for {heldout}/{policy}."
                )
            scaler_hash = str(fitted.scaler_state_hash)
            if fold_scaler_hash is None:
                fold_scaler_hash = scaler_hash
            elif scaler_hash != fold_scaler_hash:
                raise ProtocolError(
                    "Fixed-C risk scaler state differs across arms on one training frame."
                )
            predictions = tuple(int(value) for value in fitted.predictions.tolist())
            probabilities = tuple(float(row[1]) for row in fitted.probabilities.tolist())
            common = {
                "method": FIXED_C_RISK_METHOD,
                "protocol_hash": "",
                "heldout_center": heldout,
                "risk_policy_id": policy,
                "risk_policy_hash": weight_plan.risk_policy_hash,
                "weight_vector_hash": weight_plan.weight_vector_hash,
                "fixed_classifier_config_hash": FIXED_CLASSIFIER_CONFIG_HASH,
                "fit_row_hash": fit_row_hash,
                "eval_row_hash": eval_row_hash,
                "training_frame_hash": training_frame_hash,
                "scaler_state_hash": scaler_hash,
                "prior_method": PRIOR_METHOD,
                "selection_source": SELECTION_SOURCE,
            }
            result_rows.append(
                {
                    "schema_version": FIXED_C_RISK_RESULT_SCHEMA_VERSION,
                    **common,
                    "experiment_seed": config.experiment_seed,
                    "classifier_seed": config.classifier_spec.random_state,
                    "risk_policy_formula": weight_plan.formula,
                    "train_centers": _json(list(train_centers)),
                    "n_train": len(train_idx),
                    "n_eval": len(eval_idx),
                    "fixed_classifier_spec": _json(
                        config.classifier_spec.to_payload()
                    ),
                    "heldout_bacc": balanced_accuracy(y_eval, predictions),
                    "heldout_macro_f1": macro_f1(y_eval, predictions),
                    "converged": "true",
                    "n_iter": _json(list(fitted.n_iter)),
                    "status": "ok",
                    "manifest_hash": frame.manifest_hash,
                    "feature_cache_hash": frame.feature_cache_hash,
                    "threshold_policy": "predict",
                    "sample_weight_passed_to_fit": (
                        "false" if policy == "pooled" else "true"
                    ),
                    "target_eval_labels_used_for_scoring_only": "true",
                    "selection_used_target_labels": "false",
                    "fit_used_target_center": "false",
                    "target_rows_used_for_fit": "false",
                    "generated_embeddings_used": "false",
                    "cvae_checkpoint_used": "false",
                    "is_router": "false",
                    "claim_scope": "real_feature_transfer_only",
                    "claim_role": "risk_weighting_diagnostic",
                    "row_role": "diagnostic_result",
                    "diagnostic_only": "true",
                    "non_adoptive": "true",
                    "adoption_eligible": "false",
                    "support_labels_used": "false",
                    "oracle_eligible": "false",
                    "leakage_status": "PASS",
                }
            )
            for local_index, row_index in enumerate(eval_idx):
                row = frame.rows[row_index]
                prediction_rows.append(
                    {
                        "schema_version": FIXED_C_RISK_PREDICTION_SCHEMA_VERSION,
                        **common,
                        "sample_id": row.sample_id,
                        "case_id": row.case_id,
                        "center": row.center,
                        "y_true": row.label,
                        "y_pred": predictions[local_index],
                        "prob_pos": probabilities[local_index],
                        "claim_role": "risk_weighting_diagnostic",
                        "row_role": "diagnostic_prediction",
                        "diagnostic_only": "true",
                        "non_adoptive": "true",
                        "adoption_eligible": "false",
                        "support_labels_used": "false",
                        "oracle_eligible": "false",
                        "target_eval_labels_used_for_scoring_only": "true",
                        "leakage_status": "PASS",
                    }
                )
            weight_audit_rows.append(
                {
                    "schema_version": FIXED_C_RISK_WEIGHT_AUDIT_SCHEMA_VERSION,
                    "method": FIXED_C_RISK_METHOD,
                    "protocol_hash": "",
                    "heldout_center": heldout,
                    "risk_policy_id": policy,
                    "risk_policy_formula": weight_plan.formula,
                    "risk_policy_hash": weight_plan.risk_policy_hash,
                    "weight_vector_hash": weight_plan.weight_vector_hash,
                    "train_centers": _json(list(train_centers)),
                    "n_fit": len(train_idx),
                    "n_domains": len(train_centers),
                    "fit_row_hash": fit_row_hash,
                    "training_frame_hash": training_frame_hash,
                    "scaler_state_hash": scaler_hash,
                    "fixed_classifier_config_hash": FIXED_CLASSIFIER_CONFIG_HASH,
                    "group_counts": _json(weight_plan.group_counts),
                    "group_weights": _json(weight_plan.group_weights),
                    "group_masses": _json(weight_plan.group_masses),
                    "weight_min": min(weight_plan.weights),
                    "weight_max": max(weight_plan.weights),
                    "weight_sum": sum(weight_plan.weights),
                    "expected_weight_sum": len(train_idx),
                    "normalization": WEIGHT_NORMALIZATION,
                    "zero_cell_policy": ZERO_CELL_POLICY,
                    "all_weights_finite": "true",
                    "all_weights_positive": "true",
                    "target_rows_used": "false",
                    "scaler_fit_used_sample_weight": "false",
                    "sample_weight_passed_to_fit": (
                        "false" if policy == "pooled" else "true"
                    ),
                    "prior_method": PRIOR_METHOD,
                    "selection_source": SELECTION_SOURCE,
                    "claim_scope": "real_feature_transfer_only",
                    "claim_role": "risk_weighting_diagnostic",
                    "row_role": "fit_level_weight_audit",
                    "diagnostic_only": "true",
                    "non_adoptive": "true",
                    "adoption_eligible": "false",
                    "target_eval_labels_used_for_scoring_only": "true",
                    "selection_used_target_labels": "false",
                    "support_labels_used": "false",
                    "oracle_eligible": "false",
                    "status": "PASS",
                }
            )
        for policy in RISK_POLICY_IDS:
            overlap_rows.append(
                {
                    "heldout_center": heldout,
                    "risk_policy_id": policy,
                    "train_centers": list(train_centers),
                    "fit_row_hash": fit_row_hash,
                    "eval_row_hash": eval_row_hash,
                    "target_center_excluded_from_fit": heldout not in train_centers,
                    "fit_eval_sample_overlap_count": len(sample_overlap),
                    "fit_eval_case_overlap_count": len(case_overlap),
                    "quarantined_center_excluded": not set(
                        train_centers
                    ).intersection(MIDOGPP_EXCLUDED_CENTERS),
                    "target_rows_used_for_weights": False,
                    "target_rows_used_for_scaler_fit": False,
                    "target_rows_used_for_classifier_fit": False,
                    "status": "PASS",
                }
            )

    paired_rows = _paired_rows(result_rows)
    bundle_hash = fixed_c_risk_bundle_hash(
        result_rows, prediction_rows, weight_audit_rows, paired_rows
    )
    protocol["frozen_protocol_hash"] = frozen.protocol_hash
    protocol["bundle_hash"] = bundle_hash
    protocol["protocol_hash"] = stable_hash(protocol)
    protocol_hash = str(protocol["protocol_hash"])
    for row in (*result_rows, *prediction_rows, *weight_audit_rows, *paired_rows):
        row["protocol_hash"] = protocol_hash

    summary = build_diagnostic_summary(
        result_rows,
        paired_rows,
        protocol_hash=protocol_hash,
        bundle_hash=bundle_hash,
        heldout_count=len(heldouts),
    )
    elapsed = time.perf_counter() - started
    write_csv_rows(
        root / "tables/fixed_c_risk_results.csv",
        result_rows,
        FIXED_C_RISK_RESULT_COLUMNS,
    )
    write_csv_rows(
        root / "tables/fixed_c_risk_predictions.csv",
        prediction_rows,
        FIXED_C_RISK_PREDICTION_COLUMNS,
    )
    write_csv_rows(
        root / "tables/fixed_c_risk_weight_audit.csv",
        weight_audit_rows,
        FIXED_C_RISK_WEIGHT_AUDIT_COLUMNS,
    )
    write_csv_rows(
        root / "tables/fixed_c_risk_paired_comparison.csv",
        paired_rows,
        FIXED_C_RISK_PAIRED_COLUMNS,
    )
    write_json(root / "manifests/protocol_manifest.json", protocol)
    write_json(
        root / "reports/leakage_provenance_report.json",
        {
            "schema_version": "midogpp_fixed_c_risk_leakage_v1",
            "status": "PASS",
            "protocol_hash": protocol_hash,
            "frozen_protocol_hash": frozen.protocol_hash,
            "bundle_hash": bundle_hash,
            "target_eval_labels_used_for_scoring_only": True,
            "target_eval_labels_used_for_fit": False,
            "target_eval_labels_used_for_selection": False,
            "target_rows_used_for_weights": False,
            "target_rows_used_for_scaler_fit": False,
            "target_rows_used_for_classifier_fit": False,
            "quarantined_center_excluded": True,
            "scaler_fit_used_sample_weight": False,
            "claim_scope": "real_feature_transfer_only",
            "diagnostic_only": True,
            "overlap_rows": overlap_rows,
        },
    )
    write_json(root / "reports/diagnostic_summary.json", summary)
    (root / "reports/diagnostic_report.md").write_text(
        render_diagnostic_report(summary), encoding="utf-8"
    )
    write_json(
        root / "reports/runtime_summary.json",
        {
            "schema_version": "midogpp_fixed_c_risk_runtime_v1",
            "status": "COMPLETE",
            "protocol_hash": protocol_hash,
            "bundle_hash": bundle_hash,
            "elapsed_seconds": elapsed,
            "n_fits": len(result_rows),
            "used_for_selection": False,
            "claim_scope": "real_feature_transfer_only",
            "diagnostic_only": True,
        },
    )
    assert_fixed_c_risk_artifacts(root, already_loaded_frame=frame)
    return root


def load_fixed_c_risk_config(path: str | Path) -> FixedCRiskDiagnosticConfig:
    """Load only the exact production config; partial coverage is dataclass-only."""

    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("Fixed-C risk configs require PyYAML.") from exc
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ProtocolError("Fixed-C risk config must be a mapping.")
    experiment = _mapping(payload.get("experiment"), "experiment")
    inputs = _mapping(payload.get("inputs"), "inputs")
    run = _mapping(payload.get("run"), "run")
    classifier = _mapping(payload.get("classifier"), "classifier")
    weighting = _mapping(payload.get("weighting"), "weighting")
    comparison = _mapping(payload.get("comparison"), "comparison")
    claim = _mapping(payload.get("claim_boundary"), "claim_boundary")
    _assert_exact_config_sections(
        experiment=experiment,
        run=run,
        classifier=classifier,
        weighting=weighting,
        comparison=comparison,
        claim=claim,
    )
    base = config_path.parent
    spec = _classifier_spec(classifier)
    config = FixedCRiskDiagnosticConfig(
        name=str(experiment["name"]),
        mode=str(experiment["mode"]),
        code_version=str(experiment["code_version"]),
        artifact_root=_resolve_path(base, str(experiment["artifact_root"])),
        manifest_path=_resolve_path(base, str(inputs["manifest_path"])),
        feature_cache_path=_resolve_path(base, str(inputs["feature_cache_path"])),
        heldout_centers=MIDOGPP_ELIGIBLE_CENTERS,
        experiment_seed=int(run["experiment_seed"]),
        expected_feature_dim=int(run["expected_feature_dim"]),
        classifier_spec=spec,
        expected_outer_fold_count=int(run["expected_outer_fold_count"]),
        expected_arm_count=int(run["expected_arm_count"]),
        expected_fit_count=int(run["expected_fit_count"]),
        allow_partial_test_coverage=False,
    )
    _validate_runtime_config(config)
    return config


def _paired_rows(
    result_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    heldouts = tuple(
        dict.fromkeys(str(row["heldout_center"]) for row in result_rows)
    )
    by_key = {
        (str(row["heldout_center"]), str(row["risk_policy_id"])): row
        for row in result_rows
    }
    rows: list[dict[str, object]] = []
    for heldout in heldouts:
        primary = by_key[(heldout, "domain_class")]
        baseline = by_key[(heldout, "pooled")]
        if any(
            primary[field] != baseline[field]
            for field in (
                "eval_row_hash",
                "training_frame_hash",
                "scaler_state_hash",
            )
        ):
            raise ProtocolError("Fixed-C risk primary/baseline frames are not paired.")
        rows.append(
            {
                "schema_version": FIXED_C_RISK_PAIRED_SCHEMA_VERSION,
                "method": FIXED_C_RISK_METHOD,
                "protocol_hash": "",
                "heldout_center": heldout,
                "contrast_id": PRIMARY_CONTRAST,
                "primary_risk_policy_id": "domain_class",
                "baseline_risk_policy_id": "pooled",
                "primary_risk_policy_hash": primary["risk_policy_hash"],
                "baseline_risk_policy_hash": baseline["risk_policy_hash"],
                "primary_weight_vector_hash": primary["weight_vector_hash"],
                "baseline_weight_vector_hash": baseline["weight_vector_hash"],
                "eval_row_hash": primary["eval_row_hash"],
                "training_frame_hash": primary["training_frame_hash"],
                "scaler_state_hash": primary["scaler_state_hash"],
                "primary_bacc": primary["heldout_bacc"],
                "baseline_bacc": baseline["heldout_bacc"],
                "delta_bacc": float(primary["heldout_bacc"])
                - float(baseline["heldout_bacc"]),
                "primary_macro_f1": primary["heldout_macro_f1"],
                "baseline_macro_f1": baseline["heldout_macro_f1"],
                "delta_macro_f1": float(primary["heldout_macro_f1"])
                - float(baseline["heldout_macro_f1"]),
                "selection_source": SELECTION_SOURCE,
                "claim_role": "risk_weighting_diagnostic",
                "row_role": "primary_paired_comparison",
                "claim_scope": "real_feature_transfer_only",
                "diagnostic_only": "true",
                "non_adoptive": "true",
                "adoption_eligible": "false",
                "support_labels_used": "false",
                "oracle_eligible": "false",
                "target_eval_labels_used_for_scoring_only": "true",
            }
        )
    return rows


def _assert_exact_config_sections(
    *,
    experiment: Mapping[str, object],
    run: Mapping[str, object],
    classifier: Mapping[str, object],
    weighting: Mapping[str, object],
    comparison: Mapping[str, object],
    claim: Mapping[str, object],
) -> None:
    for field, expected in {
        "name": FIXED_C_RISK_EXPERIMENT_NAME,
        "mode": FIXED_C_RISK_METHOD,
        "code_version": FIXED_C_RISK_CODE_VERSION,
    }.items():
        if experiment.get(field) != expected:
            raise ProtocolError(f"Fixed-C risk experiment {field} drifted.")
    if (
        int(run.get("experiment_seed", -1)) != 42
        or str(run.get("heldout_centers", "")).lower() != "all"
        or int(run.get("expected_feature_dim", -1)) != 2560
        or int(run.get("expected_outer_fold_count", -1)) != 9
        or int(run.get("expected_arm_count", -1)) != 4
        or int(run.get("expected_fit_count", -1)) != 36
    ):
        raise ProtocolError("Fixed-C risk production run locks drifted.")
    spec = _classifier_spec(classifier)
    if (
        spec.config_hash != FIXED_CLASSIFIER_CONFIG_HASH
        or classifier.get("expected_config_hash") != FIXED_CLASSIFIER_CONFIG_HASH
    ):
        raise ProtocolError("Fixed-C risk classifier config/hash drifted.")
    if (
        tuple(weighting.get("arms", ())) != RISK_POLICY_IDS
        or dict(weighting.get("formulas", {})) != RISK_POLICY_FORMULAS
        or weighting.get("normalization") != WEIGHT_NORMALIZATION
        or weighting.get("zero_cell_policy") != ZERO_CELL_POLICY
        or weighting.get("require_finite_positive_weights") is not True
    ):
        raise ProtocolError("Fixed-C risk weighting config drifted.")
    if dict(comparison) != {
        "primary_contrast": PRIMARY_CONTRAST,
        "paired_by": "heldout_center",
        "selection_rule": "none",
        "adoption_enabled": False,
    }:
        raise ProtocolError("Fixed-C risk comparison config drifted.")
    required_claim = {
        "claim_scope": "real_feature_transfer_only",
        "diagnostic_only": True,
        "non_adoptive": True,
        "target_evaluation_labels_used_for_fit": False,
        "target_evaluation_labels_used_for_selection": False,
        "target_evaluation_labels_used_for_scoring_only": True,
        "uses_cvae_checkpoint": False,
        "uses_generated_embeddings": False,
        "uses_prior": False,
        "uses_router": False,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
    }
    for field, expected in required_claim.items():
        if claim.get(field) is not expected and claim.get(field) != expected:
            raise ProtocolError(f"Fixed-C risk claim boundary {field} drifted.")


def _validate_runtime_config(config: FixedCRiskDiagnosticConfig) -> None:
    if (
        config.name != FIXED_C_RISK_EXPERIMENT_NAME
        or config.mode != FIXED_C_RISK_METHOD
        or config.code_version != FIXED_C_RISK_CODE_VERSION
        or config.classifier_spec.config_hash != FIXED_CLASSIFIER_CONFIG_HASH
    ):
        raise ProtocolError("Fixed-C risk runtime config identity drifted.")
    if tuple(config.heldout_centers) != tuple(dict.fromkeys(config.heldout_centers)):
        raise ProtocolError("Fixed-C risk heldout centers must be unique and ordered.")
    if not config.allow_partial_test_coverage and (
        config.heldout_centers != MIDOGPP_ELIGIBLE_CENTERS
        or config.expected_outer_fold_count != 9
        or config.expected_arm_count != 4
        or config.expected_fit_count != 36
        or config.expected_feature_dim != 2560
        or config.experiment_seed != 42
    ):
        raise ProtocolError("Production fixed-C risk runtime locks drifted.")


def _classifier_spec(payload: Mapping[str, object]) -> ClassifierSpec:
    return ClassifierSpec(
        C=float(payload["C"]),
        penalty=str(payload["penalty"]),
        solver=str(payload["solver"]),
        max_iter=int(payload["max_iter"]),
        class_weight=(
            None
            if payload.get("class_weight") in (None, "", "none")
            else str(payload["class_weight"])
        ),
        random_state=int(payload["random_state"]),
        threshold_policy=str(payload["threshold_policy"]),
    )


def _arrays(
    frame: RealFeatureFrame, indices: Sequence[int]
) -> tuple[object, tuple[int, ...]]:
    import numpy as np

    embeddings = (
        frame.embeddings.detach().cpu().numpy()
        if hasattr(frame.embeddings, "detach")
        else frame.embeddings
    )
    labels = tuple(int(frame.rows[index].label) for index in indices)
    if set(labels) != {0, 1}:
        raise ProtocolError("Fixed-C risk fold must contain both classes.")
    return np.asarray(embeddings, dtype=float)[list(indices)], labels


def _indices(frame: RealFeatureFrame, centers: Sequence[str]) -> tuple[int, ...]:
    center_set = {str(center) for center in centers}
    if center_set.intersection(MIDOGPP_EXCLUDED_CENTERS):
        raise ProtocolError("Quarantined center cannot enter fixed-C risk fitting.")
    return tuple(
        index for index, row in enumerate(frame.rows) if row.center in center_set
    )


def _eligible_present(frame: RealFeatureFrame, center: str) -> str:
    value = str(center)
    if value not in MIDOGPP_ELIGIBLE_CENTERS or value not in frame.eligible_centers:
        raise ProtocolError(
            f"Unknown, quarantined, or absent MIDOG++ center: {value!r}"
        )
    return value


def _canonical_group_keys(
    policy: str, domain_ids: Sequence[str]
) -> tuple[str, ...]:
    if policy == "pooled":
        return ("all",)
    if policy == "global_class":
        return ("class=0", "class=1")
    if policy == "domain":
        return tuple(f"domain={domain}" for domain in domain_ids)
    return tuple(
        f"domain={domain}|class={label}"
        for domain in domain_ids
        for label in (0, 1)
    )


def _formula_weight(
    policy: str, n_fit: int, n_domains: int, count: int
) -> float:
    if count <= 0:
        raise ProtocolError("Fixed-C risk weighting encountered a zero cell.")
    if policy == "pooled":
        return 1.0
    if policy == "global_class":
        return float(n_fit) / float(2 * count)
    if policy == "domain":
        return float(n_fit) / float(n_domains * count)
    if policy == "domain_class":
        return float(n_fit) / float(2 * n_domains * count)
    raise ProtocolError(f"Unknown fixed-C risk policy: {policy!r}")


def _finite_positive(value: float) -> bool:
    import math

    return math.isfinite(float(value)) and float(value) > 0.0


def _row_hash(sample_ids: Sequence[str]) -> str:
    return hashlib.sha256(
        "\n".join(str(value) for value in sample_ids).encode("utf-8")
    ).hexdigest()


def _json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{name} must be a mapping.")
    return value


def _resolve_path(base: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (base / path).resolve()


__all__ = [
    "FixedCRiskDiagnosticConfig",
    "RiskWeightPlan",
    "compute_risk_weights",
    "load_fixed_c_risk_config",
    "run_fixed_c_risk_diagnostic",
]
