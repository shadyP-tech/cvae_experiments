"""Nested source-inner gamma selection for conditional-logit alignment."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable, Mapping, Sequence

from ..classifiers import ClassifierSpec
from ..downstream import balanced_accuracy, macro_f1
from ..protocol import ProtocolError
from .config import (
    AlignmentOptimizerConfig,
    DEFAULT_OPTIMIZER_CONFIG,
    GAMMA_GRID,
    TIE_ATOL,
    TIE_RTOL,
)
from .estimator import (
    AlignmentFitResult,
    PreparedConditionalLogit,
    fit_prepared_conditional_logit,
    prepare_conditional_logit,
)
from .folds import make_inner_fold


PreparedFitFn = Callable[
    [PreparedConditionalLogit, float, AlignmentOptimizerConfig], AlignmentFitResult
]


@dataclass(frozen=True)
class GammaFoldScore:
    """One H/I/gamma source-inner evaluation and its solver result."""

    outer_target_center: str
    inner_pseudo_target_center: str
    gamma: float
    bacc: float
    macro_f1: float = math.nan
    converged: bool = True
    status: str = "ok"
    fit_identity: str = ""
    scaler_state_hash: str = ""
    penalty_operator_hash: str = ""
    training_frame_hash: str = ""
    fit_row_hash: str = ""
    eval_row_hash: str = ""
    fit_result: AlignmentFitResult | None = None

    @property
    def heldout_center(self) -> str:
        return self.outer_target_center

    @property
    def pseudo_target_center(self) -> str:
        return self.inner_pseudo_target_center


@dataclass(frozen=True)
class GammaSummary:
    """Equal-inner-center aggregate for one candidate gamma in outer fold H."""

    outer_target_center: str
    gamma: float
    inner_center_bacc: Mapping[str, float]
    inner_center_macro_f1: Mapping[str, float]
    equal_center_mean_bacc: float
    equal_center_mean_macro_f1: float
    eligible: bool
    selected: bool = False

    @property
    def aggregate_score(self) -> float:
        return self.equal_center_mean_bacc

    @property
    def selected_by_source_inner_lodo(self) -> bool:
        return self.selected


@dataclass(frozen=True)
class OuterEvaluationPlan:
    """Two semantic roles mapped onto one or two unique physical fits."""

    selected_gamma: float
    role_gammas: Mapping[str, float]
    unique_fit_gammas: tuple[float, ...]
    shared_fit: bool

    def gamma_for_role(self, role: str) -> float:
        try:
            return float(self.role_gammas[str(role)])
        except KeyError as exc:
            raise ProtocolError(f"Unknown CLA outer evaluation role: {role!r}") from exc


@dataclass(frozen=True)
class GammaSelection:
    """Complete nested selection state, including all 72 frames and 504 fits."""

    outer_target_center: str
    selected_gamma: float
    gamma_grid: tuple[float, ...]
    inner_centers: tuple[str, ...]
    fold_scores: tuple[GammaFoldScore, ...]
    gamma_summaries: tuple[GammaSummary, ...]
    prepared_folds: tuple[PreparedConditionalLogit, ...] = ()
    selection_metric: str = "bacc"
    aggregation: str = "equal_center_arithmetic_mean"
    tie_atol: float = TIE_ATOL
    tie_rtol: float = TIE_RTOL
    tie_break: str = "smallest_gamma"

    @property
    def heldout_center(self) -> str:
        return self.outer_target_center

    @property
    def summaries(self) -> tuple[GammaSummary, ...]:
        return self.gamma_summaries

    @property
    def selected_summary(self) -> GammaSummary:
        selected = tuple(summary for summary in self.gamma_summaries if summary.selected)
        if len(selected) != 1:
            raise ProtocolError("Gamma selection does not contain exactly one selected summary.")
        return selected[0]

    @property
    def fit_results(self) -> tuple[AlignmentFitResult, ...]:
        results = tuple(
            score.fit_result for score in self.fold_scores if score.fit_result is not None
        )
        if results and len(results) != len(self.fold_scores):
            raise ProtocolError("Gamma selection contains a partial set of fit results.")
        return results

    @property
    def outer_evaluation_plan(self) -> OuterEvaluationPlan:
        return plan_outer_evaluation(self.selected_gamma)


def select_gamma_source_inner(
    frame: object,
    outer_target_center: str,
    gamma_grid: Sequence[float],
    spec: ClassifierSpec,
    *,
    optimizer: AlignmentOptimizerConfig = DEFAULT_OPTIMIZER_CONFIG,
    fit_fn: PreparedFitFn | None = None,
) -> GammaSelection:
    """Run the explicit H/I loop and select gamma without outer-target rows."""

    grid = _validate_gamma_grid(gamma_grid)
    outer = str(outer_target_center)
    observed = tuple(str(center) for center in getattr(frame, "eligible_centers"))
    if outer not in observed:
        raise ProtocolError(
            f"Conditional-logit outer target is absent or ineligible: {outer!r}"
        )
    inner_centers = tuple(sorted((center for center in observed if center != outer), key=_center_key))
    if not inner_centers:
        raise ProtocolError("Conditional-logit source-inner selection has no pseudo-targets.")
    scorer = fit_fn or _default_fit_fn
    prepared_folds: list[PreparedConditionalLogit] = []
    scores: list[GammaFoldScore] = []
    for inner in inner_centers:
        fold = make_inner_fold(frame, outer, inner)  # type: ignore[arg-type]
        if outer in set(fold.fit_domains) or inner in set(fold.fit_domains):
            raise ProtocolError("Conditional-logit H/I rows leaked into a prepared inner fold.")
        prepared = prepare_conditional_logit(fold, spec)
        prepared_folds.append(prepared)
        for gamma in grid:
            fitted = scorer(prepared, gamma, optimizer)
            predictions = tuple(int(value) for value in fitted.predictions)
            scores.append(
                GammaFoldScore(
                    outer_target_center=outer,
                    inner_pseudo_target_center=inner,
                    gamma=float(gamma),
                    bacc=balanced_accuracy(fold.eval_labels, predictions),
                    macro_f1=macro_f1(fold.eval_labels, predictions),
                    converged=bool(fitted.converged),
                    status=fitted.status,
                    fit_identity=fitted.fit_identity,
                    scaler_state_hash=fitted.scaler_state_hash,
                    penalty_operator_hash=fitted.penalty_operator_hash,
                    training_frame_hash=fold.training_frame_hash,
                    fit_row_hash=fold.fit_row_hash,
                    eval_row_hash=fold.eval_row_hash,
                    fit_result=fitted,
                )
            )
    selection = summarize_gamma_scores(
        outer_target_center=outer,
        fold_scores=scores,
        gamma_grid=grid,
        expected_inner_centers=inner_centers,
    )
    return replace(selection, prepared_folds=tuple(prepared_folds))


def summarize_gamma_scores(
    *,
    outer_target_center: str,
    fold_scores: Sequence[GammaFoldScore],
    gamma_grid: Sequence[float] = GAMMA_GRID,
    expected_inner_centers: Sequence[str] | None = None,
    tie_atol: float = TIE_ATOL,
    tie_rtol: float = TIE_RTOL,
) -> GammaSelection:
    """Aggregate one score per inner center and apply the exact gamma tie rule."""

    grid = _validate_gamma_grid(gamma_grid)
    if float(tie_atol) != TIE_ATOL or float(tie_rtol) != TIE_RTOL:
        raise ProtocolError("Conditional-logit gamma tie tolerances drifted.")
    outer = str(outer_target_center)
    if not fold_scores:
        raise ProtocolError("Conditional-logit gamma selection requires fold scores.")
    observed_inner = tuple(
        sorted(
            {str(score.inner_pseudo_target_center) for score in fold_scores},
            key=_center_key,
        )
    )
    inner_centers = (
        tuple(sorted((str(value) for value in expected_inner_centers), key=_center_key))
        if expected_inner_centers is not None
        else observed_inner
    )
    if not inner_centers or observed_inner != inner_centers:
        raise ProtocolError("Conditional-logit inner-center score coverage is incomplete.")
    if outer in inner_centers:
        raise ProtocolError("Outer target center appeared in source-inner gamma scores.")

    by_key: dict[tuple[str, float], GammaFoldScore] = {}
    for score in fold_scores:
        if str(score.outer_target_center) != outer:
            raise ProtocolError("Gamma fold score references the wrong outer target center.")
        key = (str(score.inner_pseudo_target_center), float(score.gamma))
        if key in by_key:
            raise ProtocolError(f"Duplicate conditional-logit gamma fold score: {key!r}")
        if float(score.gamma) not in grid:
            raise ProtocolError("Gamma fold score references a gamma outside the frozen grid.")
        by_key[key] = score

    summaries: list[GammaSummary] = []
    for gamma in grid:
        gamma_scores: list[GammaFoldScore] = []
        for inner in inner_centers:
            try:
                gamma_scores.append(by_key[(inner, gamma)])
            except KeyError as exc:
                raise ProtocolError(
                    "Conditional-logit gamma score matrix is incomplete: "
                    f"missing outer={outer}, inner={inner}, gamma={gamma}."
                ) from exc
        eligible = all(
            score.status == "ok"
            and score.converged
            and math.isfinite(float(score.bacc))
            and math.isfinite(float(score.macro_f1))
            for score in gamma_scores
        )
        bacc_vector = {
            score.inner_pseudo_target_center: float(score.bacc)
            for score in gamma_scores
        }
        macro_vector = {
            score.inner_pseudo_target_center: float(score.macro_f1)
            for score in gamma_scores
        }
        summaries.append(
            GammaSummary(
                outer_target_center=outer,
                gamma=gamma,
                inner_center_bacc=bacc_vector,
                inner_center_macro_f1=macro_vector,
                equal_center_mean_bacc=(
                    sum(bacc_vector.values()) / float(len(inner_centers))
                    if eligible
                    else math.nan
                ),
                equal_center_mean_macro_f1=(
                    sum(macro_vector.values()) / float(len(inner_centers))
                    if eligible
                    else math.nan
                ),
                eligible=eligible,
            )
        )
    if len(by_key) != len(inner_centers) * len(grid):
        raise ProtocolError("Conditional-logit gamma score matrix contains unexpected rows.")
    eligible_summaries = [summary for summary in summaries if summary.eligible]
    if not eligible_summaries:
        raise ProtocolError("No gamma produced valid scores on every source-inner center.")
    best_score = max(summary.equal_center_mean_bacc for summary in eligible_summaries)
    tied = [
        summary
        for summary in eligible_summaries
        if abs(float(summary.equal_center_mean_bacc) - float(best_score)) <= TIE_ATOL
    ]
    selected_gamma = min(float(summary.gamma) for summary in tied)
    selected_summaries = tuple(
        replace(summary, selected=float(summary.gamma) == selected_gamma)
        for summary in summaries
    )
    return GammaSelection(
        outer_target_center=outer,
        selected_gamma=selected_gamma,
        gamma_grid=grid,
        inner_centers=inner_centers,
        fold_scores=tuple(fold_scores),
        gamma_summaries=selected_summaries,
        tie_atol=TIE_ATOL,
        tie_rtol=TIE_RTOL,
    )


def plan_outer_evaluation(selected_gamma: float) -> OuterEvaluationPlan:
    """Plan selected/gamma0 roles while deduplicating selected gamma zero."""

    gamma = float(selected_gamma)
    if gamma not in GAMMA_GRID:
        raise ProtocolError("Selected CLA gamma is outside the frozen grid.")
    shared = gamma == 0.0
    return OuterEvaluationPlan(
        selected_gamma=gamma,
        role_gammas={"selected": gamma, "gamma0": 0.0},
        unique_fit_gammas=(0.0,) if shared else (gamma, 0.0),
        shared_fit=shared,
    )


def _default_fit_fn(
    prepared: PreparedConditionalLogit,
    gamma: float,
    optimizer: AlignmentOptimizerConfig,
) -> AlignmentFitResult:
    return fit_prepared_conditional_logit(prepared, gamma, optimizer=optimizer)


def _validate_gamma_grid(gamma_grid: Sequence[float]) -> tuple[float, ...]:
    grid = tuple(float(value) for value in gamma_grid)
    if grid != GAMMA_GRID:
        raise ProtocolError(
            "Conditional-logit gamma grid must remain exactly "
            "[0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]."
        )
    return grid


def _center_key(center: str) -> tuple[int, str]:
    try:
        return int(str(center)), str(center)
    except ValueError as exc:
        raise ProtocolError(
            f"Conditional-logit center IDs must be numeric: {center!r}"
        ) from exc


__all__ = [
    "GammaFoldScore",
    "GammaSelection",
    "GammaSummary",
    "OuterEvaluationPlan",
    "plan_outer_evaluation",
    "select_gamma_source_inner",
    "summarize_gamma_scores",
]
