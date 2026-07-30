"""Deterministic, case-disjoint train-case feasibility splits."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import combinations
from typing import Sequence

from ....real_features.classifier_reference.protocol import ProtocolError


@dataclass(frozen=True)
class CaseHoldout:
    fit_indices: tuple[int, ...]
    eval_indices: tuple[int, ...]
    fit_cases: tuple[str, ...]
    eval_cases: tuple[str, ...]


def deterministic_case_holdout(
    case_ids: Sequence[str],
    labels: Sequence[int],
    *,
    validation_fraction: float,
    seed: int,
) -> CaseHoldout:
    cases = tuple(sorted(set(str(value) for value in case_ids)))
    if not 0.0 < validation_fraction < 0.5 or len(cases) < 10:
        raise ProtocolError("Pilot case holdout requires >=10 cases and fraction in (0,.5).")
    n_eval = max(2, int(round(len(cases) * validation_fraction)))
    ordered = tuple(
        sorted(
            cases,
            key=lambda case: hashlib.sha256(
                f"{seed}|{case}".encode("utf-8")
            ).hexdigest(),
        )
    )
    labels_tuple = tuple(int(value) for value in labels)
    cases_tuple = tuple(str(value) for value in case_ids)
    selected: tuple[str, ...] | None = None
    for candidate in combinations(ordered, n_eval):
        candidate_set = set(candidate)
        eval_labels = {
            labels_tuple[index]
            for index, case in enumerate(cases_tuple)
            if case in candidate_set
        }
        fit_labels = {
            labels_tuple[index]
            for index, case in enumerate(cases_tuple)
            if case not in candidate_set
        }
        if eval_labels == {0, 1} and fit_labels == {0, 1}:
            selected = tuple(sorted(candidate))
            break
    if selected is None:
        raise ProtocolError("No deterministic case split contains both classes on both sides.")
    eval_set = set(selected)
    fit_indices = tuple(i for i, case in enumerate(cases_tuple) if case not in eval_set)
    eval_indices = tuple(i for i, case in enumerate(cases_tuple) if case in eval_set)
    fit_cases = tuple(sorted(set(cases_tuple[i] for i in fit_indices)))
    if set(fit_cases).intersection(selected):
        raise ProtocolError("Pilot case holdout leaked a case.")
    return CaseHoldout(
        fit_indices=fit_indices,
        eval_indices=eval_indices,
        fit_cases=fit_cases,
        eval_cases=selected,
    )
