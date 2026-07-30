"""Case-disjoint source-only cross-fitting identities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ....common.hashing import stable_hash
from ...protocol import ProtocolError


@dataclass(frozen=True)
class CaseFold:
    fold: int
    fit_indices: tuple[int, ...]
    reference_indices: tuple[int, ...]
    fit_cases: tuple[str, ...]
    reference_cases: tuple[str, ...]

    @property
    def identity_hash(self) -> str:
        return stable_hash(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_independent_source_case_fold_v1",
            "fold": self.fold,
            "fit_indices": list(self.fit_indices),
            "reference_indices": list(self.reference_indices),
            "fit_cases": list(self.fit_cases),
            "reference_cases": list(self.reference_cases),
        }


def deterministic_case_folds(
    labels: Sequence[int],
    case_ids: Sequence[str],
    *,
    n_splits: int,
    seed: int,
) -> tuple[CaseFold, ...]:
    """Build deterministic stratified group folds and fail on weak isolation."""

    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold

    y = np.asarray(labels, dtype=np.int64)
    cases = np.asarray([str(value) for value in case_ids], dtype=str)
    if (
        len(y) == 0
        or len(y) != len(cases)
        or set(int(value) for value in y.tolist()) != {0, 1}
        or n_splits < 2
        or len(set(cases.tolist())) < n_splits
    ):
        raise ProtocolError("Source-only case-fold inputs are invalid.")
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=int(seed),
    )
    folds: list[CaseFold] = []
    placeholder = np.zeros((len(y), 1), dtype=np.float32)
    for fold_index, (fit, reference) in enumerate(
        splitter.split(placeholder, y, groups=cases)
    ):
        fit_tuple = tuple(int(value) for value in fit.tolist())
        reference_tuple = tuple(int(value) for value in reference.tolist())
        fit_cases = tuple(sorted(set(cases[list(fit_tuple)].tolist())))
        reference_cases = tuple(
            sorted(set(cases[list(reference_tuple)].tolist()))
        )
        if (
            set(fit_cases).intersection(reference_cases)
            or set(int(value) for value in y[list(fit_tuple)].tolist()) != {0, 1}
            or set(int(value) for value in y[list(reference_tuple)].tolist())
            != {0, 1}
        ):
            raise ProtocolError("Source-only cross-fit fold is not case/class valid.")
        folds.append(
            CaseFold(
                fold=fold_index,
                fit_indices=fit_tuple,
                reference_indices=reference_tuple,
                fit_cases=fit_cases,
                reference_cases=reference_cases,
            )
        )
    if sorted(index for fold in folds for index in fold.reference_indices) != list(
        range(len(y))
    ):
        raise ProtocolError("Cross-fit reference folds do not partition source rows.")
    return tuple(folds)


__all__ = ("CaseFold", "deterministic_case_folds")
