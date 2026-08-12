"""Terminal-only pooled utility, oracle, contrast, and routing metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError


@dataclass(frozen=True, order=True)
class CaseConfusion:
    case_id: str
    tp: int
    tn: int
    fp: int
    fn: int

    def __post_init__(self) -> None:
        if not self.case_id or min(self.tp, self.tn, self.fp, self.fn) < 0:
            raise ProtocolError("Case confusion row drifted.")

    @property
    def n_positive(self) -> int:
        return self.tp + self.fn

    @property
    def n_negative(self) -> int:
        return self.tn + self.fp

    def to_payload(self) -> dict[str, object]:
        return dict(self.__dict__)

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "CaseConfusion":
        return cls(
            case_id=str(payload["case_id"]),
            tp=int(payload["tp"]),
            tn=int(payload["tn"]),
            fp=int(payload["fp"]),
            fn=int(payload["fn"]),
        )


@dataclass(frozen=True)
class MethodScore:
    method_id: str
    bacc: float
    tp: int
    tn: int
    n_positive: int
    n_negative: int

    def to_payload(self) -> dict[str, object]:
        return dict(self.__dict__)

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "MethodScore":
        return cls(
            method_id=str(payload["method_id"]),
            bacc=float(payload["bacc"]),
            tp=int(payload["tp"]),
            tn=int(payload["tn"]),
            n_positive=int(payload["n_positive"]),
            n_negative=int(payload["n_negative"]),
        )


@dataclass(frozen=True)
class BootstrapContrast:
    method_id: str
    baseline_id: str
    estimate: float
    ci_low: float
    ci_high: float
    replicates: int
    seed: int

    def to_payload(self) -> dict[str, object]:
        return dict(self.__dict__)

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "BootstrapContrast":
        return cls(
            method_id=str(payload["method_id"]),
            baseline_id=str(payload["baseline_id"]),
            estimate=float(payload["estimate"]),
            ci_low=float(payload["ci_low"]),
            ci_high=float(payload["ci_high"]),
            replicates=int(payload["replicates"]),
            seed=int(payload["seed"]),
        )


@dataclass(frozen=True)
class TerminalOracles:
    static_action_id: str
    static_score: MethodScore
    case_actions: tuple[tuple[str, str], ...]
    case_score: MethodScore

    def to_payload(self) -> dict[str, object]:
        return {
            "static_action_id": self.static_action_id,
            "static_score": self.static_score.to_payload(),
            "case_actions": [list(row) for row in self.case_actions],
            "case_score": self.case_score.to_payload(),
            "terminal_only": True,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "TerminalOracles":
        static = payload["static_score"]
        case_score_payload = payload["case_score"]
        if not isinstance(static, Mapping) or not isinstance(case_score_payload, Mapping):
            raise ProtocolError("Terminal oracle score payload drifted.")
        return cls(
            static_action_id=str(payload["static_action_id"]),
            static_score=MethodScore.from_payload(static),
            case_actions=tuple((str(row[0]), str(row[1])) for row in payload["case_actions"]),  # type: ignore[index]
            case_score=MethodScore.from_payload(case_score_payload),
        )


def case_confusion(
    case_id: str,
    labels: Sequence[int] | np.ndarray,
    predictions: Sequence[int] | np.ndarray,
) -> CaseConfusion:
    truth = np.asarray(labels)
    predicted = np.asarray(predictions)
    if truth.ndim != 1 or truth.shape != predicted.shape or len(truth) == 0:
        raise ProtocolError("Terminal label/prediction vectors are not aligned.")
    if not np.all(np.isin(truth, (0, 1))) or not np.all(np.isin(predicted, (0, 1))):
        raise ProtocolError("Terminal scores require binary labels and predictions.")
    positive = truth == 1
    negative = ~positive
    predicted_positive = predicted == 1
    return CaseConfusion(
        case_id=str(case_id),
        tp=int(np.sum(positive & predicted_positive)),
        tn=int(np.sum(negative & (~predicted_positive))),
        fp=int(np.sum(negative & predicted_positive)),
        fn=int(np.sum(positive & (~predicted_positive))),
    )


def pooled_bacc(rows: Sequence[CaseConfusion]) -> float:
    records = tuple(rows)
    n_positive = sum(row.n_positive for row in records)
    n_negative = sum(row.n_negative for row in records)
    if not records or n_positive <= 0 or n_negative <= 0:
        raise ProtocolError("Pooled BACC requires both classes across non-empty cases.")
    return float(
        0.5 * sum(row.tp for row in records) / n_positive
        + 0.5 * sum(row.tn for row in records) / n_negative
    )


def method_score(method_id: str, rows: Sequence[CaseConfusion]) -> MethodScore:
    records = tuple(rows)
    return MethodScore(
        method_id=str(method_id),
        bacc=pooled_bacc(records),
        tp=sum(row.tp for row in records),
        tn=sum(row.tn for row in records),
        n_positive=sum(row.n_positive for row in records),
        n_negative=sum(row.n_negative for row in records),
    )


def terminal_oracles(
    action_rows: Mapping[str, Sequence[CaseConfusion]],
    *,
    baseline_action_id: str = "B",
) -> TerminalOracles:
    """Open labels only here to compute O_static and additive O_case."""

    normalized = {str(action): tuple(rows) for action, rows in action_rows.items()}
    if baseline_action_id not in normalized or not normalized:
        raise ProtocolError("Terminal oracle surface lacks baseline B.")
    baseline_cases = tuple(row.case_id for row in normalized[baseline_action_id])
    if len(set(baseline_cases)) != len(baseline_cases):
        raise ProtocolError("Terminal oracle baseline cases are duplicated.")
    by_action_case: dict[str, dict[str, CaseConfusion]] = {}
    for action, rows in normalized.items():
        mapping = {row.case_id: row for row in rows}
        if tuple(sorted(mapping)) != tuple(sorted(baseline_cases)) or len(mapping) != len(rows):
            raise ProtocolError("Terminal oracle action/case coverage drifted.")
        by_action_case[action] = mapping
    static_ranked = sorted(
        ((method_score(action, rows).bacc, action) for action, rows in normalized.items()),
        key=lambda item: (-item[0], item[1]),
    )
    static_action = static_ranked[0][1]
    static_score = method_score("O_static", normalized[static_action])
    n_positive = sum(row.n_positive for row in normalized[baseline_action_id])
    n_negative = sum(row.n_negative for row in normalized[baseline_action_id])
    if n_positive <= 0 or n_negative <= 0:
        raise ProtocolError("Terminal case oracle requires both pooled classes.")
    choices: list[tuple[str, str]] = []
    chosen_rows: list[CaseConfusion] = []
    for case_id in sorted(baseline_cases):
        ranked: list[tuple[float, str, CaseConfusion]] = []
        for action, mapping in by_action_case.items():
            row = mapping[case_id]
            contribution = 0.5 * row.tp / n_positive + 0.5 * row.tn / n_negative
            ranked.append((float(contribution), action, row))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        choices.append((case_id, ranked[0][1]))
        chosen_rows.append(ranked[0][2])
    return TerminalOracles(
        static_action_id=static_action,
        static_score=static_score,
        case_actions=tuple(choices),
        case_score=method_score("O_case", chosen_rows),
    )


def paired_case_bootstrap_contrast(
    method_rows: Sequence[CaseConfusion],
    baseline_rows: Sequence[CaseConfusion],
    *,
    method_id: str,
    baseline_id: str,
    replicates: int,
    seed: int,
) -> BootstrapContrast:
    """Resample aligned whole cases, never individual rows."""

    method = {row.case_id: row for row in method_rows}
    baseline = {row.case_id: row for row in baseline_rows}
    if set(method) != set(baseline) or len(method) != len(method_rows) or len(baseline) != len(baseline_rows):
        raise ProtocolError("Paired bootstrap case coverage drifted.")
    if int(replicates) <= 0:
        raise ProtocolError("Bootstrap replicate count must be positive.")
    case_ids = tuple(sorted(method))
    estimate = pooled_bacc(tuple(method.values())) - pooled_bacc(tuple(baseline.values()))
    rng = np.random.default_rng(int(seed))
    values: list[float] = []
    attempts = 0
    while len(values) < int(replicates):
        attempts += 1
        if attempts > int(replicates) * 20:
            raise ProtocolError("Bootstrap could not draw enough two-class pooled samples.")
        indices = rng.integers(0, len(case_ids), size=len(case_ids))
        method_sample = tuple(method[case_ids[int(i)]] for i in indices)
        baseline_sample = tuple(baseline[case_ids[int(i)]] for i in indices)
        try:
            values.append(pooled_bacc(method_sample) - pooled_bacc(baseline_sample))
        except ProtocolError:
            continue
    low, high = np.quantile(np.asarray(values, dtype=np.float64), (0.025, 0.975))
    return BootstrapContrast(
        method_id=str(method_id),
        baseline_id=str(baseline_id),
        estimate=float(estimate),
        ci_low=float(low),
        ci_high=float(high),
        replicates=int(replicates),
        seed=int(seed),
    )


def router_metrics(
    *,
    selected_actions: Sequence[str],
    oracle_actions: Sequence[str],
    predicted_gains: Sequence[float],
    oracle_gains: Sequence[float],
    router_bacc: float,
    baseline_bacc: float,
    oracle_bacc: float,
    fold_static_actions: Sequence[str] = (),
) -> dict[str, float]:
    selected = tuple(str(v) for v in selected_actions)
    oracle = tuple(str(v) for v in oracle_actions)
    predicted = np.asarray(predicted_gains, dtype=np.float64)
    actual = np.asarray(oracle_gains, dtype=np.float64)
    if (
        not selected
        or len(selected) != len(oracle)
        or predicted.shape != actual.shape
        or len(predicted) != len(selected)
    ):
        raise ProtocolError("Router metric vectors are not aligned.")
    if not np.isfinite(predicted).all() or not np.isfinite(actual).all():
        raise ProtocolError("Router metric gains are non-finite.")
    denominator = float(oracle_bacc) - float(baseline_bacc)
    normalized_gap = 0.0 if abs(denominator) <= 1e-15 else (
        float(oracle_bacc) - float(router_bacc)
    ) / denominator
    folds = tuple(str(v) for v in fold_static_actions)
    stability = 1.0
    if folds:
        stability = max(folds.count(value) for value in set(folds)) / len(folds)
    return {
        "top1_oracle_agreement": float(np.mean(np.asarray(selected) == np.asarray(oracle))),
        "spearman": _spearman(predicted, actual),
        "normalized_oracle_gap": float(normalized_gap),
        "fold_stability": float(stability),
    }


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return 0.0
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    if np.std(left_rank) <= 0.0 or np.std(right_rank) <= 0.0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


__all__ = (
    "BootstrapContrast",
    "CaseConfusion",
    "MethodScore",
    "TerminalOracles",
    "case_confusion",
    "method_score",
    "paired_case_bootstrap_contrast",
    "pooled_bacc",
    "router_metrics",
    "terminal_oracles",
)
