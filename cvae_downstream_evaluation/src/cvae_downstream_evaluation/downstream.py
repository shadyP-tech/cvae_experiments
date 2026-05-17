"""Synthetic-only downstream utility utilities.

Heavy classifier dependencies are imported lazily. The aggregation and metric
contracts are pure Python so the protocol can be tested without workstation
packages.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .protocol import ProtocolError
from .schemas import (
    ALL_EXPERT_DOWNSTREAM_COLUMNS,
    ALL_EXPERT_DOWNSTREAM_PRIMARY_KEY,
    MATRIX_SCHEMA_VERSION,
    METHOD_BASELINE_ROW_TYPE,
    PRIMARY_GENERATION_MODE,
    SINGLE_EXPERT_HASH,
    SINGLE_EXPERT_ROW_TYPE,
)


@dataclass(frozen=True)
class DownstreamScore:
    expert_domain: str
    balanced_accuracy: float
    macro_f1: float
    secondary_metrics: Mapping[str, float]


@dataclass(frozen=True)
class DownstreamPrediction:
    """Locked classifier output used by single-expert and late-ensemble rows."""

    score: DownstreamScore
    probabilities: object
    classes: tuple[int, ...]


@dataclass(frozen=True)
class CandidateDownstreamRow:
    """One downstream score for a candidate expert or explicitly tagged baseline."""

    experiment_seed: int
    heldout_center: str
    candidate_expert: str
    generation_mode: str
    budget_per_class: int
    generation_seed: int
    classifier_seed: int
    bacc: float
    macro_f1: float
    auroc: float = math.nan
    auprc: float = math.nan
    row_type: str = SINGLE_EXPERT_ROW_TYPE
    n_synthetic_train: int = 0
    n_target_eval: int = 0
    target_eval_pool_id: str = ""
    candidate_experts_hash: str = SINGLE_EXPERT_HASH
    status: str = "ok"
    error_message: str = ""
    schema_version: str = MATRIX_SCHEMA_VERSION

    def oracle_key(self) -> tuple[int, str, str, int, int, int]:
        return (
            int(self.experiment_seed),
            self.heldout_center,
            self.generation_mode,
            int(self.budget_per_class),
            int(self.generation_seed),
            int(self.classifier_seed),
        )

    def primary_key(self) -> tuple[object, ...]:
        return tuple(getattr(self, field) for field in ALL_EXPERT_DOWNSTREAM_PRIMARY_KEY)

    def is_oracle_eligible(self) -> bool:
        return (
            self.row_type == SINGLE_EXPERT_ROW_TYPE
            and self.generation_mode == PRIMARY_GENERATION_MODE
            and self.status == "ok"
        )

    def to_csv_row(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "experiment_seed": self.experiment_seed,
            "heldout_center": self.heldout_center,
            "candidate_expert": self.candidate_expert,
            "generation_mode": self.generation_mode,
            "budget_per_class": self.budget_per_class,
            "generation_seed": self.generation_seed,
            "classifier_seed": self.classifier_seed,
            "bacc": self.bacc,
            "macro_f1": self.macro_f1,
            "auroc": self.auroc,
            "auprc": self.auprc,
            "row_type": self.row_type,
            "n_synthetic_train": self.n_synthetic_train,
            "n_target_eval": self.n_target_eval,
            "target_eval_pool_id": self.target_eval_pool_id,
            "candidate_experts_hash": self.candidate_experts_hash,
            "status": self.status,
            "error_message": self.error_message,
        }


@dataclass(frozen=True)
class OracleScore:
    experiment_seed: int
    heldout_center: str
    generation_mode: str
    budget_per_class: int
    generation_seed: int
    classifier_seed: int
    expert: str
    bacc: float
    macro_f1: float


def train_and_evaluate_synthetic_only(
    synthetic_embeddings: Sequence[Sequence[float]],
    synthetic_labels: Sequence[int],
    target_embeddings: Sequence[Sequence[float]],
    target_labels: Sequence[int],
    *,
    classifier_seed: int,
) -> DownstreamScore:
    """Train the locked synthetic-only classifier and score target data.

    This function requires NumPy and scikit-learn at runtime. They are not
    imported at module import time so protocol tests can run in minimal Python.
    """

    return fit_locked_logistic_classifier(
        synthetic_embeddings,
        synthetic_labels,
        target_embeddings,
        target_labels,
        classifier_seed=classifier_seed,
    ).score


def fit_locked_logistic_classifier(
    synthetic_embeddings: Sequence[Sequence[float]],
    synthetic_labels: Sequence[int],
    target_embeddings: Sequence[Sequence[float]],
    target_labels: Sequence[int],
    *,
    classifier_seed: int,
) -> DownstreamPrediction:
    """Fit the locked synthetic-only classifier and return probabilities.

    StandardScaler is fit on synthetic training embeddings only. Target
    evaluation labels are consumed only after prediction for final metrics.
    """

    try:
        import numpy as np  # type: ignore
        from sklearn.linear_model import LogisticRegression  # type: ignore
        from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Synthetic-only downstream evaluation requires numpy and scikit-learn."
        ) from exc

    x_syn = np.asarray(synthetic_embeddings, dtype=float)
    y_syn = np.asarray(synthetic_labels, dtype=int)
    x_eval = np.asarray(target_embeddings, dtype=float)
    y_eval = np.asarray(target_labels, dtype=int)
    if x_syn.ndim != 2 or x_eval.ndim != 2:
        raise ValueError("Embeddings must be 2D arrays.")
    if x_syn.shape[1] != x_eval.shape[1]:
        raise ValueError("Synthetic and target embeddings must share the same projection frame.")
    if sorted(set(int(v) for v in y_syn.tolist())) != [0, 1]:
        raise ValueError("Locked v1 classifier expects exactly balanced binary synthetic labels.")

    scaler = StandardScaler()
    x_syn_scaled = scaler.fit_transform(x_syn)
    x_eval_scaled = scaler.transform(x_eval)
    clf = LogisticRegression(
        solver="lbfgs",
        C=1.0,
        max_iter=2000,
        class_weight=None,
        random_state=int(classifier_seed),
    )
    clf.fit(x_syn_scaled, y_syn)
    pred = clf.predict(x_eval_scaled)
    proba = clf.predict_proba(x_eval_scaled)
    classes = tuple(int(v) for v in clf.classes_.tolist())
    secondary: dict[str, float] = {}
    if len(classes) == 2 and proba.shape[1] == 2:
        try:
            secondary["auroc"] = float(roc_auc_score(y_eval, proba[:, 1]))
        except ValueError:
            secondary["auroc"] = math.nan
        try:
            secondary["auprc"] = float(average_precision_score(y_eval, proba[:, 1]))
        except ValueError:
            secondary["auprc"] = math.nan
    return DownstreamPrediction(
        score=DownstreamScore(
            expert_domain="",
            balanced_accuracy=balanced_accuracy(y_eval.tolist(), pred.tolist()),
            macro_f1=macro_f1(y_eval.tolist(), pred.tolist()),
            secondary_metrics=secondary,
        ),
        probabilities=proba,
        classes=classes,
    )


def balanced_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    classes = sorted(set(int(v) for v in y_true))
    if not classes:
        raise ValueError("Cannot compute balanced accuracy with no labels.")
    recalls: list[float] = []
    for cls in classes:
        total = sum(1 for y in y_true if int(y) == cls)
        correct = sum(1 for yt, yp in zip(y_true, y_pred) if int(yt) == cls and int(yp) == cls)
        recalls.append(float(correct) / float(total) if total else 0.0)
    return sum(recalls) / float(len(recalls))


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    classes = sorted(set(int(v) for v in y_true).union(int(v) for v in y_pred))
    if not classes:
        raise ValueError("Cannot compute macro-F1 with no labels.")
    scores: list[float] = []
    for cls in classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if int(yt) == cls and int(yp) == cls)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if int(yt) != cls and int(yp) == cls)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if int(yt) == cls and int(yp) != cls)
        denom = (2 * tp) + fp + fn
        scores.append(float(2 * tp) / float(denom) if denom else 0.0)
    return sum(scores) / float(len(scores))


def compute_single_expert_oracles(rows: Sequence[CandidateDownstreamRow]) -> dict[tuple[int, str, str, int, int, int], OracleScore]:
    """Compute diagnostic downstream oracle over single-expert rows only."""

    grouped: dict[tuple[int, str, str, int, int, int], list[CandidateDownstreamRow]] = {}
    for row in rows:
        if not row.is_oracle_eligible():
            continue
        grouped.setdefault(row.oracle_key(), []).append(row)
    oracles: dict[tuple[int, str, str, int, int, int], OracleScore] = {}
    for key, group in grouped.items():
        winner = max(group, key=lambda row: (float(row.bacc), float(row.macro_f1), _reverse_lex(row.candidate_expert)))
        oracles[key] = OracleScore(
            experiment_seed=winner.experiment_seed,
            heldout_center=winner.heldout_center,
            generation_mode=winner.generation_mode,
            budget_per_class=winner.budget_per_class,
            generation_seed=winner.generation_seed,
            classifier_seed=winner.classifier_seed,
            expert=winner.candidate_expert,
            bacc=float(winner.bacc),
            macro_f1=float(winner.macro_f1),
        )
    return oracles


def validate_candidate_downstream_matrix(rows: Sequence[CandidateDownstreamRow]) -> None:
    """Reject duplicate candidate rows and method baselines inside the oracle matrix."""

    seen: set[tuple[object, ...]] = set()
    for row in rows:
        key = row.primary_key()
        if key in seen:
            raise ProtocolError(f"Duplicate downstream candidate row: {key}")
        seen.add(key)
        if row.row_type not in {SINGLE_EXPERT_ROW_TYPE, METHOD_BASELINE_ROW_TYPE}:
            raise ProtocolError(f"Unknown downstream row_type: {row.row_type}")
        if row.schema_version != MATRIX_SCHEMA_VERSION:
            raise ProtocolError(
                f"Unexpected downstream matrix row schema_version={row.schema_version!r}; "
                f"expected {MATRIX_SCHEMA_VERSION!r}."
            )


def read_candidate_downstream_matrix(path: Path) -> list[CandidateDownstreamRow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [_candidate_from_csv_row(row) for row in csv.DictReader(handle)]
    validate_candidate_downstream_matrix(rows)
    return rows


def write_candidate_downstream_matrix(path: Path, rows: Sequence[CandidateDownstreamRow]) -> None:
    validate_candidate_downstream_matrix(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_matrix_schema(path.with_suffix(".schema.json"))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ALL_EXPERT_DOWNSTREAM_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_row())


def matrix_schema_payload() -> dict[str, object]:
    return {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "primary_key": list(ALL_EXPERT_DOWNSTREAM_PRIMARY_KEY),
        "oracle_eligible_filter": {
            "row_type": SINGLE_EXPERT_ROW_TYPE,
            "generation_mode": PRIMARY_GENERATION_MODE,
            "status": "ok",
        },
    }


def write_matrix_schema(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(matrix_schema_payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def assert_matrix_schema(path: Path) -> None:
    schema_path = path.with_suffix(".schema.json")
    if not schema_path.exists():
        raise ProtocolError(f"Missing downstream matrix schema sidecar: {schema_path}")
    try:
        payload = json.loads(schema_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"Malformed downstream matrix schema sidecar: {schema_path}") from exc
    expected = matrix_schema_payload()
    if payload.get("schema_version") != expected["schema_version"]:
        raise ProtocolError(
            f"Downstream matrix schema_version mismatch: got {payload.get('schema_version')!r}, "
            f"expected {expected['schema_version']!r}."
        )
    if list(payload.get("primary_key") or []) != expected["primary_key"]:
        raise ProtocolError("Downstream matrix primary_key schema mismatch.")
    if payload.get("oracle_eligible_filter") != expected["oracle_eligible_filter"]:
        raise ProtocolError("Downstream matrix oracle_eligible_filter schema mismatch.")


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("Spearman inputs must have equal length.")
    if len(xs) < 2:
        return math.nan
    rx = _rank(xs)
    ry = _rank(ys)
    mean_x = sum(rx) / len(rx)
    mean_y = sum(ry) / len(ry)
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    den_x = math.sqrt(sum((a - mean_x) ** 2 for a in rx))
    den_y = math.sqrt(sum((b - mean_y) ** 2 for b in ry))
    if den_x == 0.0 or den_y == 0.0:
        return math.nan
    return float(num / (den_x * den_y))


def _rank(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(float(v) for v in values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (float(i + 1) + float(j)) / 2.0
        for pos in range(i, j):
            ranks[indexed[pos][0]] = avg_rank
        i = j
    return ranks


def _candidate_from_csv_row(row: Mapping[str, str]) -> CandidateDownstreamRow:
    return CandidateDownstreamRow(
        schema_version=str(row.get("schema_version") or MATRIX_SCHEMA_VERSION),
        experiment_seed=int(row.get("experiment_seed") or 0),
        heldout_center=str(row["heldout_center"]),
        candidate_expert=str(row["candidate_expert"]),
        generation_mode=str(row["generation_mode"]),
        budget_per_class=int(row["budget_per_class"]),
        generation_seed=int(row["generation_seed"]),
        classifier_seed=int(row["classifier_seed"]),
        bacc=_float_or_nan(row.get("bacc", "")),
        macro_f1=_float_or_nan(row.get("macro_f1", "")),
        auroc=_float_or_nan(row.get("auroc", "")),
        auprc=_float_or_nan(row.get("auprc", "")),
        row_type=str(row.get("row_type") or SINGLE_EXPERT_ROW_TYPE),
        n_synthetic_train=int(row.get("n_synthetic_train") or 0),
        n_target_eval=int(row.get("n_target_eval") or 0),
        target_eval_pool_id=str(row.get("target_eval_pool_id") or ""),
        candidate_experts_hash=str(row.get("candidate_experts_hash") or SINGLE_EXPERT_HASH),
        status=str(row.get("status") or "ok"),
        error_message=str(row.get("error_message") or ""),
    )


def _float_or_nan(raw: object) -> float:
    text = str(raw or "").strip()
    if not text:
        return math.nan
    return float(text)


def _reverse_lex(value: str) -> str:
    # Used with max() so lower lexical expert ids win ties deterministically.
    return "".join(chr(255 - ord(ch)) for ch in str(value))
