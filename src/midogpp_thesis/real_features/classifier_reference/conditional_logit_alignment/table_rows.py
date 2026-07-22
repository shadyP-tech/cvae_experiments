"""Pure scientific row construction for CLA artifact tables."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping, Sequence

from ..artifacts import stable_hash
from ..protocol import ProtocolError
from .artifacts import json_cell
from .schema import (
    AlignmentArtifactTables,
    CLA_METHOD,
    CONDITIONAL_FRAME_AUDIT_SCHEMA_VERSION,
    OUTER_COMPARISON_SCHEMA_VERSION,
    OUTER_PREDICTION_SCHEMA_VERSION,
    OUTER_RESULT_SCHEMA_VERSION,
    PRIMARY_CONTRAST,
    SOLVER_AUDIT_SCHEMA_VERSION,
    SOURCE_INNER_FOLD_SCORE_SCHEMA_VERSION,
    SOURCE_INNER_GAMMA_SUMMARY_SCHEMA_VERSION,
    claim_fields,
)

@dataclass(frozen=True)
class OuterEvaluation:
    """One outer frame with only its selected and gamma-zero physical fits."""

    prepared: object
    selected_gamma: float
    selected_fit: object
    gamma0_fit: object


def build_alignment_artifact_tables(
    selections: Sequence[object],
    outer_evaluations: Sequence[OuterEvaluation],
    *,
    frame: object,
    tie_atol: float = 1e-12,
    tie_rtol: float = 0.0,
) -> AlignmentArtifactTables:
    """Construct all seven normalized tables from typed core results."""

    inner_scores: list[dict[str, object]] = []
    gamma_summaries: list[dict[str, object]] = []
    outer_results: list[dict[str, object]] = []
    outer_predictions: list[dict[str, object]] = []
    frame_audits: list[dict[str, object]] = []
    solver_audits: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []

    seen_frames: set[str] = set()
    seen_fits: set[str] = set()
    for selection in selections:
        heldout = str(getattr(selection, "outer_target_center"))
        prepared_by_inner = {
            str(getattr(item.fold_data, "inner_pseudo_target_center")): item
            for item in getattr(selection, "prepared_folds")
        }
        summaries = tuple(getattr(selection, "gamma_summaries"))
        ranks = _gamma_ranks(summaries, tie_atol=float(tie_atol))
        for prepared in prepared_by_inner.values():
            audit = _frame_audit_row(prepared, fold_scope="source_inner")
            identity = str(audit["conditional_frame_identity"])
            if identity in seen_frames:
                raise ProtocolError("Duplicate CLA source-inner conditional frame.")
            seen_frames.add(identity)
            frame_audits.append(audit)
        for score in getattr(selection, "fold_scores"):
            inner = str(getattr(score, "inner_pseudo_target_center"))
            prepared = prepared_by_inner.get(inner)
            fitted = getattr(score, "fit_result", None)
            if prepared is None or fitted is None:
                raise ProtocolError("CLA source-inner artifact lacks prepared fit state.")
            if not bool(getattr(fitted, "converged")):
                raise ProtocolError(
                    f"CLA source-inner fit did not converge for H={heldout}, I={inner}."
                )
            fold = prepared.fold_data
            frame_identity = conditional_frame_identity(prepared)
            common = _fit_common(prepared, fitted, frame_identity)
            inner_scores.append(
                {
                    "schema_version": SOURCE_INNER_FOLD_SCORE_SCHEMA_VERSION,
                    "method": CLA_METHOD,
                    "protocol_hash": "",
                    "heldout_center": heldout,
                    "inner_center": inner,
                    "gamma": float(getattr(score, "gamma")),
                    **common,
                    "fit_centers": json_cell(fold.fit_centers),
                    "n_fit": int(fold.n_fit),
                    "n_eval": int(fold.n_eval),
                    "eval_row_hash": fold.eval_row_hash,
                    "training_frame_hash": fold.training_frame_hash,
                    "inner_bacc": float(getattr(score, "bacc")),
                    "inner_macro_f1": float(getattr(score, "macro_f1")),
                    "converged": "true",
                    "n_iter": json_cell(getattr(fitted, "n_iter")),
                    "status": "ok",
                    **claim_fields(row_role="source_inner_fold_score"),
                }
            )
            _append_solver_audit(
                solver_audits,
                seen_fits,
                prepared,
                fitted,
                frame_identity,
                fold_scope="source_inner",
            )
        for summary in summaries:
            gamma = float(getattr(summary, "gamma"))
            bacc_by_center = dict(getattr(summary, "inner_center_bacc"))
            gamma_summaries.append(
                {
                    "schema_version": SOURCE_INNER_GAMMA_SUMMARY_SCHEMA_VERSION,
                    "method": CLA_METHOD,
                    "protocol_hash": "",
                    "heldout_center": heldout,
                    "gamma": gamma,
                    "n_inner_centers": len(bacc_by_center),
                    "mean_inner_bacc": float(
                        getattr(summary, "equal_center_mean_bacc")
                    ),
                    "mean_inner_macro_f1": float(
                        getattr(summary, "equal_center_mean_macro_f1")
                    ),
                    "minimum_inner_bacc": min(float(v) for v in bacc_by_center.values()),
                    "selected": _bool_cell(getattr(summary, "selected")),
                    "selection_rank": ranks[gamma],
                    "tie_atol": float(tie_atol),
                    "tie_rtol": float(tie_rtol),
                    "status": "PASS" if bool(getattr(summary, "eligible")) else "REJECTED",
                    **claim_fields(row_role="source_inner_gamma_summary"),
                }
            )

    for evaluation in outer_evaluations:
        prepared = evaluation.prepared
        fold = prepared.fold_data
        heldout = str(fold.outer_target_center)
        selected_gamma = float(evaluation.selected_gamma)
        selected_fit = evaluation.selected_fit
        gamma0_fit = evaluation.gamma0_fit
        if float(getattr(selected_fit, "gamma")) != selected_gamma:
            raise ProtocolError("CLA selected outer fit gamma differs from source-inner lock.")
        if float(getattr(gamma0_fit, "gamma")) != 0.0:
            raise ProtocolError("CLA gamma-zero outer baseline is not gamma=0.")
        shared = selected_gamma == 0.0
        if shared and getattr(selected_fit, "fit_identity") != getattr(
            gamma0_fit, "fit_identity"
        ):
            raise ProtocolError("CLA selected gamma zero must reuse one physical fit.")
        if not shared and getattr(selected_fit, "fit_identity") == getattr(
            gamma0_fit, "fit_identity"
        ):
            raise ProtocolError("Distinct CLA outer gammas cannot share a fit identity.")
        if not bool(getattr(selected_fit, "converged")) or not bool(
            getattr(gamma0_fit, "converged")
        ):
            raise ProtocolError(f"CLA outer fit did not converge for H={heldout}.")

        frame_audit = _frame_audit_row(prepared, fold_scope="outer")
        frame_identity = str(frame_audit["conditional_frame_identity"])
        if frame_identity in seen_frames:
            raise ProtocolError("Duplicate CLA outer conditional frame.")
        seen_frames.add(frame_identity)
        frame_audits.append(frame_audit)
        role_fits = (("selected", selected_fit), ("gamma0", gamma0_fit))
        role_metrics: dict[str, tuple[float, float]] = {}
        for role, fitted in role_fits:
            predictions = tuple(int(value) for value in getattr(fitted, "predictions"))
            probabilities = _positive_probabilities(fitted)
            bacc = _balanced_accuracy(fold.eval_labels, predictions)
            f1 = _macro_f1(fold.eval_labels, predictions)
            role_metrics[role] = (bacc, f1)
            common = _fit_common(prepared, fitted, frame_identity)
            outer_results.append(
                {
                    "schema_version": OUTER_RESULT_SCHEMA_VERSION,
                    "method": CLA_METHOD,
                    "protocol_hash": "",
                    "heldout_center": heldout,
                    "evaluation_role": role,
                    "gamma": float(getattr(fitted, "gamma")),
                    "selected_gamma": selected_gamma,
                    **common,
                    "shared_fit": _bool_cell(shared),
                    "fit_centers": json_cell(fold.fit_centers),
                    "n_fit": int(fold.n_fit),
                    "n_eval": int(fold.n_eval),
                    "eval_row_hash": fold.eval_row_hash,
                    "training_frame_hash": fold.training_frame_hash,
                    "heldout_bacc": bacc,
                    "heldout_macro_f1": f1,
                    "converged": "true",
                    "n_iter": json_cell(getattr(fitted, "n_iter")),
                    "status": "ok",
                    "manifest_hash": str(getattr(frame, "manifest_hash")),
                    "feature_cache_hash": str(getattr(frame, "feature_cache_hash")),
                    **claim_fields(row_role="outer_diagnostic_result"),
                }
            )
            for index, identity in enumerate(fold.eval_identities):
                outer_predictions.append(
                    {
                        "schema_version": OUTER_PREDICTION_SCHEMA_VERSION,
                        "method": CLA_METHOD,
                        "protocol_hash": "",
                        "heldout_center": heldout,
                        "evaluation_role": role,
                        "gamma": float(getattr(fitted, "gamma")),
                        "selected_gamma": selected_gamma,
                        "fit_identity": str(getattr(fitted, "fit_identity")),
                        "conditional_frame_identity": frame_identity,
                        "shared_fit": _bool_cell(shared),
                        "sample_id": identity.sample_id,
                        "case_id": identity.case_id,
                        "center": identity.center,
                        "y_true": int(identity.label),
                        "y_pred": predictions[index],
                        "prob_pos": probabilities[index],
                        "fit_row_hash": fold.fit_row_hash,
                        "eval_row_hash": fold.eval_row_hash,
                        "training_frame_hash": fold.training_frame_hash,
                        "scaler_state_hash": prepared.scaler_state_hash,
                        "penalty_operator_hash": prepared.penalty_operator.factor_hash,
                        "classifier_config_hash": str(
                            getattr(fitted, "classifier_config_hash")
                        ),
                        **claim_fields(row_role="outer_diagnostic_prediction"),
                    }
                )
            _append_solver_audit(
                solver_audits,
                seen_fits,
                prepared,
                fitted,
                frame_identity,
                fold_scope="outer",
            )
        selected_metrics = role_metrics["selected"]
        gamma0_metrics = role_metrics["gamma0"]
        comparisons.append(
            {
                "schema_version": OUTER_COMPARISON_SCHEMA_VERSION,
                "method": CLA_METHOD,
                "protocol_hash": "",
                "heldout_center": heldout,
                "contrast_id": PRIMARY_CONTRAST,
                "selected_gamma": selected_gamma,
                "selected_fit_identity": str(getattr(selected_fit, "fit_identity")),
                "gamma0_fit_identity": str(getattr(gamma0_fit, "fit_identity")),
                "shared_fit": _bool_cell(shared),
                "eval_row_hash": fold.eval_row_hash,
                "selected_bacc": selected_metrics[0],
                "gamma0_bacc": gamma0_metrics[0],
                "delta_bacc": selected_metrics[0] - gamma0_metrics[0],
                "selected_macro_f1": selected_metrics[1],
                "gamma0_macro_f1": gamma0_metrics[1],
                "delta_macro_f1": selected_metrics[1] - gamma0_metrics[1],
                "status": "PASS",
                **claim_fields(row_role="outer_paired_comparison"),
            }
        )

    return AlignmentArtifactTables(
        source_inner_fold_scores=tuple(inner_scores),
        source_inner_gamma_summary=tuple(gamma_summaries),
        outer_results=tuple(outer_results),
        outer_predictions=tuple(outer_predictions),
        conditional_frame_audit=tuple(frame_audits),
        solver_audit=tuple(solver_audits),
        outer_comparison=tuple(comparisons),
    )


def conditional_frame_identity(prepared: object) -> str:
    """Return the semantic identity shared by every gamma on one fit frame."""

    fold = prepared.fold_data
    return stable_hash(
        {
            "training_frame_hash": fold.training_frame_hash,
            "fit_row_hash": fold.fit_row_hash,
            "eval_row_hash": fold.eval_row_hash,
            "scaler_state_hash": prepared.scaler_state_hash,
            "penalty_operator_hash": prepared.penalty_operator.factor_hash,
        }
    )


def _fit_common(
    prepared: object,
    fitted: object,
    frame_identity: str,
) -> dict[str, object]:
    return {
        "fit_identity": str(getattr(fitted, "fit_identity")),
        "conditional_frame_identity": frame_identity,
        "fit_row_hash": prepared.fold_data.fit_row_hash,
        "scaler_state_hash": prepared.scaler_state_hash,
        "penalty_operator_hash": prepared.penalty_operator.factor_hash,
        "classifier_config_hash": str(getattr(fitted, "classifier_config_hash")),
    }


def _frame_audit_row(prepared: object, *, fold_scope: str) -> dict[str, object]:
    fold = prepared.fold_data
    operator = prepared.penalty_operator
    fit_rows = tuple(fold.fit_identities)
    eval_rows = tuple(fold.eval_identities)
    overlap_counts = _overlap_counts(fit_rows, eval_rows)
    if any(overlap_counts.values()):
        raise ProtocolError("CLA fit/evaluation identities overlap.")
    if fold.outer_target_center in set(fold.fit_centers):
        raise ProtocolError("CLA held-out center entered frame construction.")
    inner_excluded = (
        fold.inner_pseudo_target_center is None
        or fold.inner_pseudo_target_center not in set(fold.fit_centers)
    )
    if not inner_excluded:
        raise ProtocolError("CLA inner pseudo-target entered frame construction.")
    return {
        "schema_version": CONDITIONAL_FRAME_AUDIT_SCHEMA_VERSION,
        "method": CLA_METHOD,
        "protocol_hash": "",
        "fold_scope": fold_scope,
        "heldout_center": str(fold.outer_target_center),
        "inner_center": (
            "" if fold.inner_pseudo_target_center is None else str(fold.inner_pseudo_target_center)
        ),
        "conditional_frame_identity": conditional_frame_identity(prepared),
        "fit_centers": json_cell(fold.fit_centers),
        "n_fit": int(fold.n_fit),
        "n_domains": len(fold.fit_centers),
        "fit_row_hash": fold.fit_row_hash,
        "eval_row_hash": fold.eval_row_hash,
        "fit_case_hash": _identity_hash(row.case_id for row in fit_rows),
        "eval_case_hash": _identity_hash(row.case_id for row in eval_rows),
        "fit_image_path_hash": _identity_hash(row.image_path for row in fit_rows),
        "eval_image_path_hash": _identity_hash(row.image_path for row in eval_rows),
        "fit_row_index_hash": _identity_hash(row.row_index for row in fit_rows),
        "eval_row_index_hash": _identity_hash(row.row_index for row in eval_rows),
        "training_frame_hash": fold.training_frame_hash,
        "scaler_state_hash": prepared.scaler_state_hash,
        "penalty_operator_hash": operator.factor_hash,
        "operator_rank": int(operator.rank),
        "maximum_operator_rank": int(operator.maximum_rank),
        "operator_trace": float(operator.trace),
        "required_cell_count": 2 * len(operator.centers),
        "observed_cell_count": len(operator.row_keys),
        "missing_cell_count": 2 * len(operator.centers) - len(operator.row_keys),
        "factor_representation": "rectangular_contrast_factor",
        "normalization": "unit_trace",
        "dense_matrix_materialized": "false",
        "heldout_center_excluded": "true",
        "inner_center_excluded": "true" if fold_scope == "source_inner" else "not_applicable",
        **overlap_counts,
        "target_rows_used_for_scaler": "false",
        "target_rows_used_for_operator": "false",
        "target_rows_used_for_fit": "false",
        "status": "PASS",
        **claim_fields(row_role="conditional_frame_audit"),
    }


def _append_solver_audit(
    rows: list[dict[str, object]],
    seen_fit_identities: set[str],
    prepared: object,
    fitted: object,
    frame_identity: str,
    *,
    fold_scope: str,
) -> None:
    fit_identity = str(getattr(fitted, "fit_identity"))
    if fit_identity in seen_fit_identities:
        return
    seen_fit_identities.add(fit_identity)
    fold = prepared.fold_data
    gamma = float(getattr(fitted, "gamma"))
    rows.append(
        {
            "schema_version": SOLVER_AUDIT_SCHEMA_VERSION,
            "method": CLA_METHOD,
            "protocol_hash": "",
            "fold_scope": fold_scope,
            "heldout_center": str(fold.outer_target_center),
            "inner_center": (
                "" if fold.inner_pseudo_target_center is None else str(fold.inner_pseudo_target_center)
            ),
            "gamma": gamma,
            "fit_identity": fit_identity,
            "conditional_frame_identity": frame_identity,
            "fit_row_hash": fold.fit_row_hash,
            "scaler_state_hash": prepared.scaler_state_hash,
            "penalty_operator_hash": prepared.penalty_operator.factor_hash,
            "classifier_config_hash": str(getattr(fitted, "classifier_config_hash")),
            "backend": str(getattr(fitted, "backend")),
            "warm_start": (
                "not_applicable_shared_sklearn"
                if gamma == 0.0
                else "pooled_gamma0_solution"
            ),
            "objective_value": float(getattr(fitted, "objective")),
            "gradient_inf_norm": float(getattr(fitted, "gradient_inf_norm")),
            "n_iter": json_cell(getattr(fitted, "n_iter")),
            "converged": _bool_cell(getattr(fitted, "converged")),
            "optimizer_status": int(getattr(fitted, "optimizer_status")),
            "l2_normalization": "1/(2*C*N_fit)",
            "intercept_penalized": "false",
            "gamma_zero_shared_sklearn_path": _bool_cell(gamma == 0.0),
            "status": "PASS" if bool(getattr(fitted, "converged")) else "REJECTED",
            **claim_fields(row_role="solver_fit_audit"),
        }
    )


def _positive_probabilities(fitted: object) -> tuple[float, ...]:
    import numpy as np

    values = np.asarray(getattr(fitted, "probabilities"), dtype=float)
    classes = tuple(int(value) for value in getattr(fitted, "classes"))
    if values.ndim != 2 or 1 not in classes or values.shape[1] != len(classes):
        raise ProtocolError("CLA fit returned malformed class probabilities.")
    return tuple(float(value) for value in values[:, classes.index(1)])


def _overlap_counts(
    fit_rows: Sequence[object], eval_rows: Sequence[object]
) -> dict[str, int]:
    def overlap(field: str, *, ignore_blank: bool = False) -> int:
        left = {getattr(row, field) for row in fit_rows}
        right = {getattr(row, field) for row in eval_rows}
        if ignore_blank:
            left.discard("")
            right.discard("")
        return len(left.intersection(right))

    return {
        "fit_eval_sample_overlap_count": overlap("sample_id"),
        "fit_eval_case_overlap_count": overlap("case_id", ignore_blank=True),
        "fit_eval_image_path_overlap_count": overlap("image_path", ignore_blank=True),
        "fit_eval_row_index_overlap_count": overlap("row_index"),
    }


def _identity_hash(values: Sequence[object] | object) -> str:
    return hashlib.sha256(
        "\n".join(str(value) for value in values).encode("utf-8")  # type: ignore[arg-type]
    ).hexdigest()


def _gamma_ranks(summaries: Sequence[object], *, tie_atol: float) -> dict[float, int]:
    values = tuple(summaries)
    if not values:
        raise ProtocolError("CLA gamma summaries are empty.")
    best = max(float(getattr(item, "equal_center_mean_bacc")) for item in values)
    ordered = sorted(
        values,
        key=lambda item: (
            0
            if abs(float(getattr(item, "equal_center_mean_bacc")) - best) <= tie_atol
            else 1,
            float(getattr(item, "gamma"))
            if abs(float(getattr(item, "equal_center_mean_bacc")) - best) <= tie_atol
            else -float(getattr(item, "equal_center_mean_bacc")),
            float(getattr(item, "gamma")),
        ),
    )
    return {float(getattr(item, "gamma")): index + 1 for index, item in enumerate(ordered)}


def _balanced_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    from ..downstream import balanced_accuracy

    return float(balanced_accuracy(y_true, y_pred))


def _macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    from ..downstream import macro_f1

    return float(macro_f1(y_true, y_pred))


def _bool_cell(value: object) -> str:
    return "true" if bool(value) else "false"




__all__ = [
    "OuterEvaluation",
    "build_alignment_artifact_tables",
    "conditional_frame_identity",
]
