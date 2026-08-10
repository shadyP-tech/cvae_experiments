"""Terminal exact-BACC evaluation for sealed signed-error predictions.

Evaluation labels enter only here, after every method prediction is fixed.  The
terminal endpoint pools confusion counts within each target center and then
weights centers equally.  Whole cases are retained as the uncertainty unit;
in particular, single-class cases are never assigned a per-case BACC or
discarded before pooling.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import math
import multiprocessing

from ...protocol import ProtocolError
from ..fixed_bank_hierarchical_residual_stacker.contracts import (
    BinaryLabel,
    CaseConfusionCounts,
    PooledExactBacc,
    PredictionRow,
)
from ..fixed_bank_hierarchical_residual_stacker.core_hashing import canonical_hash
from ..fixed_bank_hierarchical_residual_stacker.pooled_metrics import (
    pooled_exact_bacc,
    score_case_confusions,
)
from ..fixed_bank_hierarchical_residual_stacker.scientific_constants import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    MIDOGPP_CENTERS,
)
from ..fixed_bank_hierarchical_residual_stacker.uncertainty import (
    BootstrapContrast,
    EqualCenterContrast,
    equal_center_contrast,
    whole_case_bootstrap,
)
from .constants import METHOD_IDS


PRIMARY_CONTRASTS = (
    ("R_safe", "B_cal"),
    ("R_safe", "G"),
    ("R_safe", "P"),
)
SECONDARY_CONTRASTS = (
    ("R_raw", "R_safe"),
    ("B_cal", "B"),
)


@dataclass(frozen=True)
class CenterPooledMetric:
    """A center identity paired with the reused pooled exact-BACC contract."""

    target_center: str
    metric: PooledExactBacc

    def __post_init__(self) -> None:
        if self.target_center not in MIDOGPP_CENTERS:
            raise ProtocolError("Center metric uses an unknown MIDOG++ target.")

    def to_payload(self) -> dict[str, object]:
        return {
            "scope": "target_center_pooled",
            "target_center": self.target_center,
            **self.metric.to_payload(),
        }


@dataclass(frozen=True)
class MethodEvaluation:
    """All terminal counts and pooled metrics for one sealed method surface."""

    method_id: str
    case_confusions: tuple[CaseConfusionCounts, ...]
    center_metrics: tuple[CenterPooledMetric, ...]
    equal_center_exact_bacc: float
    single_class_case_count: int
    method_result_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.method_id not in METHOD_IDS:
            raise ProtocolError("Terminal method result uses an unknown method.")
        counts = tuple(self.case_confusions)
        metrics = tuple(self.center_metrics)
        if not counts or any(row.method_id != self.method_id for row in counts):
            raise ProtocolError("Terminal method confusion rows drifted from the method.")
        if tuple(row.target_center for row in metrics) != MIDOGPP_CENTERS:
            raise ProtocolError("Terminal method result requires every MIDOG++ center.")
        if any(row.metric.method_id != self.method_id for row in metrics):
            raise ProtocolError("Terminal pooled metrics drifted from the method.")
        expected_equal_center = math.fsum(
            row.metric.exact_bacc for row in metrics
        ) / len(metrics)
        value = float(self.equal_center_exact_bacc)
        if not math.isfinite(value) or abs(value - expected_equal_center) > 1.0e-12:
            raise ProtocolError("Equal-center BACC drifted from center-pooled metrics.")
        expected_single_class = sum(
            row.n_positive == 0 or row.n_negative == 0 for row in counts
        )
        if (
            isinstance(self.single_class_case_count, bool)
            or self.single_class_case_count != expected_single_class
        ):
            raise ProtocolError("Single-class whole-case accounting drifted.")
        object.__setattr__(self, "case_confusions", counts)
        object.__setattr__(self, "center_metrics", metrics)
        object.__setattr__(self, "equal_center_exact_bacc", value)
        object.__setattr__(self, "method_result_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        total_positive = sum(row.n_positive for row in self.case_confusions)
        total_negative = sum(row.n_negative for row in self.case_confusions)
        return {
            "schema_version": "fixed_bank_signed_error_method_evaluation_v1",
            "method_id": self.method_id,
            "case_confusions": [
                {
                    "method_id": row.method_id,
                    "target_center": row.target_center,
                    "case_id": row.case_id,
                    "n_positive": row.n_positive,
                    "true_positive": row.true_positive,
                    "n_negative": row.n_negative,
                    "true_negative": row.true_negative,
                }
                for row in self.case_confusions
            ],
            "center_metrics": [row.to_payload() for row in self.center_metrics],
            "equal_center_metric": {
                "aggregation": "equal_target_center",
                "center_count": len(self.center_metrics),
                "case_count": len(self.case_confusions),
                "n_positive": total_positive,
                "n_negative": total_negative,
                "exact_bacc": self.equal_center_exact_bacc,
                "pooled_sensitivity_and_specificity_defined": False,
            },
            "single_class_case_count": self.single_class_case_count,
            "single_class_cases_retained": True,
            "per_case_bacc_stored_or_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "method_result_hash": self.method_result_hash}


@dataclass(frozen=True)
class TerminalContrast:
    """Equal-center contrast with bootstrap uncertainty for primary rows only."""

    contrast_role: str
    equal_center: EqualCenterContrast
    bootstrap: BootstrapContrast | None
    contrast_hash: str = field(init=False)

    def __post_init__(self) -> None:
        pair = (
            self.equal_center.challenger_method,
            self.equal_center.reference_method,
        )
        expected_role = (
            "primary"
            if pair in PRIMARY_CONTRASTS
            else "secondary"
            if pair in SECONDARY_CONTRASTS
            else None
        )
        if expected_role is None or self.contrast_role != expected_role:
            raise ProtocolError("Terminal contrast role or method pair drifted.")
        if self.equal_center.center_count != len(MIDOGPP_CENTERS):
            raise ProtocolError("Terminal contrast requires all target centers.")
        if tuple(center for center, _value in self.equal_center.center_differences) != (
            MIDOGPP_CENTERS
        ):
            raise ProtocolError("Terminal contrast center order or coverage drifted.")
        if expected_role == "primary":
            if self.bootstrap is None:
                raise ProtocolError("Every primary contrast requires whole-case bootstrap.")
            bootstrap_pair = (
                self.bootstrap.challenger_method,
                self.bootstrap.reference_method,
            )
            if bootstrap_pair != pair:
                raise ProtocolError("Bootstrap and equal-center contrast pairs differ.")
            if (
                abs(
                    self.bootstrap.observed_equal_center_difference
                    - self.equal_center.mean_difference
                )
                > 1.0e-12
            ):
                raise ProtocolError("Bootstrap observed contrast drifted from the endpoint.")
        elif self.bootstrap is not None:
            raise ProtocolError("Secondary diagnostics must not masquerade as primary inference.")
        object.__setattr__(self, "contrast_hash", canonical_hash(self._unhashed()))

    @property
    def contrast_id(self) -> str:
        return (
            f"{self.equal_center.challenger_method}-"
            f"{self.equal_center.reference_method}"
        )

    def _unhashed(self) -> dict[str, object]:
        equal = self.equal_center
        return {
            "schema_version": "fixed_bank_signed_error_terminal_contrast_v1",
            "contrast_id": self.contrast_id,
            "contrast_role": self.contrast_role,
            "challenger_method": equal.challenger_method,
            "reference_method": equal.reference_method,
            "center_count": equal.center_count,
            "center_differences": [list(value) for value in equal.center_differences],
            "equal_center_difference": equal.mean_difference,
            "center_t_ci95_lower": equal.ci95_lower,
            "center_t_ci95_upper": equal.ci95_upper,
            "whole_case_bootstrap": (
                self.bootstrap.to_payload() if self.bootstrap is not None else None
            ),
            "uncertainty_unit": "whole_case_cluster_within_target_center",
            "per_case_bacc_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "contrast_hash": self.contrast_hash}


@dataclass(frozen=True)
class SignedGateEvaluationResult:
    """Immutable terminal result; runtime parallelism is scientifically inert."""

    method_results: tuple[MethodEvaluation, ...]
    contrasts: tuple[TerminalContrast, ...]
    terminal_label_surface_hash: str
    scientific_result_hash: str = field(init=False)

    def __post_init__(self) -> None:
        methods = tuple(self.method_results)
        contrasts = tuple(self.contrasts)
        if tuple(row.method_id for row in methods) != METHOD_IDS:
            raise ProtocolError("Terminal result method order or coverage drifted.")
        expected_contrasts = PRIMARY_CONTRASTS + SECONDARY_CONTRASTS
        if tuple(
            (
                row.equal_center.challenger_method,
                row.equal_center.reference_method,
            )
            for row in contrasts
        ) != expected_contrasts:
            raise ProtocolError("Terminal result contrast order or coverage drifted.")
        case_scopes = tuple(
            tuple(row.case_key for row in method.case_confusions)
            for method in methods
        )
        label_counts = tuple(
            tuple(
                (row.n_positive, row.n_negative)
                for row in method.case_confusions
            )
            for method in methods
        )
        if len(set(case_scopes)) != 1 or len(set(label_counts)) != 1:
            raise ProtocolError("Terminal methods do not share one whole-case label scope.")
        if (
            not isinstance(self.terminal_label_surface_hash, str)
            or len(self.terminal_label_surface_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.terminal_label_surface_hash
            )
        ):
            raise ProtocolError("Terminal label surface hash must be full SHA-256 text.")
        object.__setattr__(self, "method_results", methods)
        object.__setattr__(self, "contrasts", contrasts)
        object.__setattr__(
            self, "scientific_result_hash", canonical_hash(self._unhashed())
        )

    @property
    def primary_contrasts(self) -> tuple[TerminalContrast, ...]:
        return tuple(row for row in self.contrasts if row.contrast_role == "primary")

    @property
    def secondary_contrasts(self) -> tuple[TerminalContrast, ...]:
        return tuple(row for row in self.contrasts if row.contrast_role == "secondary")

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_signed_error_terminal_evaluation_v1",
            "method_ids": list(METHOD_IDS),
            "methods": [row.to_payload() for row in self.method_results],
            "contrasts": [row.to_payload() for row in self.contrasts],
            "primary_contrasts": [f"{left}-{right}" for left, right in PRIMARY_CONTRASTS],
            "secondary_contrasts": [
                f"{left}-{right}" for left, right in SECONDARY_CONTRASTS
            ],
            "primary_endpoint": "center_pooled_exact_bacc_equal_center_aggregate",
            "terminal_label_surface_hash": self.terminal_label_surface_hash,
            "terminal_evaluation_labels_only": True,
            "single_class_cases_retained": True,
            "per_case_bacc_stored_or_used": False,
            "bootstrap_resampling_unit": "whole_case_within_target_center",
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "scientific_result_hash": self.scientific_result_hash}


def _evaluate_terminal_predictions(
    *,
    predictions_by_method: Mapping[str, Sequence[PredictionRow]],
    labels: Sequence[BinaryLabel],
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_workers: int = 1,
    multiprocessing_start_method: str = "spawn",
    bootstrap_threads_per_worker: int = 1,
) -> SignedGateEvaluationResult:
    """Internally score methods after the sealed adapter opens terminal labels.

    Bootstrap workers run independent primary contrasts.  Every worker uses the
    requested process start method and a bounded native-thread pool.  These
    execution choices cannot change the returned scientific payload or hash.
    """

    label_rows = _validated_terminal_labels(labels)
    _validate_runtime(
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        bootstrap_workers=bootstrap_workers,
        multiprocessing_start_method=multiprocessing_start_method,
        bootstrap_threads_per_worker=bootstrap_threads_per_worker,
    )
    if tuple(predictions_by_method) != METHOD_IDS:
        if set(predictions_by_method) != set(METHOD_IDS):
            raise ProtocolError("Terminal prediction mapping must contain exactly six methods.")

    counts_by_method: dict[str, tuple[CaseConfusionCounts, ...]] = {}
    method_results: list[MethodEvaluation] = []
    for method in METHOD_IDS:
        if method not in predictions_by_method:
            raise ProtocolError("Terminal prediction mapping is missing a method.")
        prediction_rows = tuple(predictions_by_method[method])
        if (
            not prediction_rows
            or any(not isinstance(row, PredictionRow) for row in prediction_rows)
            or any(row.method_id != method for row in prediction_rows)
        ):
            raise ProtocolError("Terminal prediction rows drifted from their mapping key.")
        counts = score_case_confusions(prediction_rows, label_rows)
        counts_by_method[method] = counts
        center_metrics = tuple(
            CenterPooledMetric(
                center,
                pooled_exact_bacc(
                    tuple(row for row in counts if row.target_center == center)
                ),
            )
            for center in MIDOGPP_CENTERS
        )
        method_results.append(
            MethodEvaluation(
                method,
                counts,
                center_metrics,
                math.fsum(row.metric.exact_bacc for row in center_metrics)
                / len(center_metrics),
                sum(row.n_positive == 0 or row.n_negative == 0 for row in counts),
            )
        )

    _validate_common_case_scope(counts_by_method)
    bootstrap_tasks = tuple(
        (
            counts_by_method[challenger],
            counts_by_method[reference],
            bootstrap_replicates,
            bootstrap_seed,
            bootstrap_threads_per_worker,
        )
        for challenger, reference in PRIMARY_CONTRASTS
    )
    bootstrap_values = _run_bootstraps(
        bootstrap_tasks,
        workers=bootstrap_workers,
        start_method=multiprocessing_start_method,
    )
    bootstrap_by_pair = {
        pair: value
        for pair, value in zip(PRIMARY_CONTRASTS, bootstrap_values, strict=True)
    }
    contrasts = tuple(
        TerminalContrast(
            "primary" if pair in PRIMARY_CONTRASTS else "secondary",
            equal_center_contrast(
                counts_by_method[pair[0]], counts_by_method[pair[1]]
            ),
            bootstrap_by_pair.get(pair),
        )
        for pair in PRIMARY_CONTRASTS + SECONDARY_CONTRASTS
    )
    label_hash = canonical_hash(
        {
            "schema_version": "fixed_bank_signed_error_terminal_label_surface_v1",
            "labels": [
                {
                    "target_center": row.target_center,
                    "case_id": row.case_id,
                    "sample_id": row.sample_id,
                    "label": row.label,
                    "label_scope": row.label_scope,
                }
                for row in label_rows
            ],
        }
    )
    return SignedGateEvaluationResult(
        tuple(method_results), contrasts, label_hash
    )


def _validated_terminal_labels(
    labels: Sequence[BinaryLabel],
) -> tuple[BinaryLabel, ...]:
    raw_rows = tuple(labels)
    if not raw_rows:
        raise ProtocolError("Terminal evaluation labels must be non-empty.")
    if any(not isinstance(row, BinaryLabel) for row in raw_rows):
        raise ProtocolError("Terminal evaluation labels use the wrong row contract.")
    rows = tuple(sorted(raw_rows))
    if any(row.label_scope != "terminal_evaluation" for row in rows):
        raise ProtocolError("Only terminal-evaluation labels may enter terminal scoring.")
    if len({row.sample_key for row in rows}) != len(rows):
        raise ProtocolError("Terminal evaluation labels contain duplicate samples.")
    if tuple(sorted({row.target_center for row in rows})) != tuple(
        sorted(MIDOGPP_CENTERS)
    ):
        raise ProtocolError("Terminal evaluation labels must cover all MIDOG++ centers.")
    return rows


def _validate_common_case_scope(
    counts_by_method: Mapping[str, Sequence[CaseConfusionCounts]],
) -> None:
    reference = tuple(counts_by_method[METHOD_IDS[0]])
    reference_scope = tuple(
        (row.case_key, row.n_positive, row.n_negative) for row in reference
    )
    for method in METHOD_IDS[1:]:
        candidate_scope = tuple(
            (row.case_key, row.n_positive, row.n_negative)
            for row in counts_by_method[method]
        )
        if candidate_scope != reference_scope:
            raise ProtocolError("Terminal methods have different whole-case scopes.")


def _validate_runtime(
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    bootstrap_workers: int,
    multiprocessing_start_method: str,
    bootstrap_threads_per_worker: int,
) -> None:
    for value, name in (
        (bootstrap_replicates, "bootstrap_replicates"),
        (bootstrap_workers, "bootstrap_workers"),
        (bootstrap_threads_per_worker, "bootstrap_threads_per_worker"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ProtocolError(f"{name} must be a positive integer.")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise ProtocolError("bootstrap_seed must be an integer.")
    if bootstrap_replicates > BOOTSTRAP_REPLICATES:
        raise ProtocolError(
            "bootstrap_replicates exceeds the frozen canonical replicate budget."
        )
    if (
        bootstrap_workers > 4
        or bootstrap_threads_per_worker > 3
        or bootstrap_workers * bootstrap_threads_per_worker > 12
    ):
        raise ProtocolError("Signed-error terminal CPU pool exceeds the frozen budget.")
    if (
        not isinstance(multiprocessing_start_method, str)
        or multiprocessing_start_method != "spawn"
    ):
        raise ProtocolError("Signed-error terminal evaluation requires spawn.")


def _run_bootstraps(
    tasks: Sequence[
        tuple[
            Sequence[CaseConfusionCounts],
            Sequence[CaseConfusionCounts],
            int,
            int,
            int,
        ]
    ],
    *,
    workers: int,
    start_method: str,
) -> tuple[BootstrapContrast, ...]:
    if workers == 1:
        return tuple(_bootstrap_task(task) for task in tasks)
    context = multiprocessing.get_context(start_method)
    with ProcessPoolExecutor(
        max_workers=min(workers, len(tasks)), mp_context=context
    ) as pool:
        return tuple(pool.map(_bootstrap_task, tasks, chunksize=1))


def _bootstrap_task(
    task: tuple[
        Sequence[CaseConfusionCounts],
        Sequence[CaseConfusionCounts],
        int,
        int,
        int,
    ],
) -> BootstrapContrast:
    from threadpoolctl import threadpool_limits

    challenger, reference, replicates, seed, threads = task
    with threadpool_limits(limits=threads):
        return whole_case_bootstrap(
            challenger,
            reference,
            replicates=replicates,
            seed=seed,
        )


__all__ = (
    "CenterPooledMetric",
    "MethodEvaluation",
    "PRIMARY_CONTRASTS",
    "SECONDARY_CONTRASTS",
    "SignedGateEvaluationResult",
    "TerminalContrast",
)
