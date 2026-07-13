"""Shared classifier-grid CLI parsing helpers."""

from __future__ import annotations

import argparse
from typing import Sequence

from .classifiers import ClassifierSpec
from .protocol import ProtocolError


def add_classifier_grid_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_c_grid: str = "0.1,1.0,10.0",
    default_penalties: str = "l2",
    default_solvers: str = "lbfgs",
    default_class_weights: str = "none,balanced",
    default_max_iters: str = "2000",
) -> None:
    """Add the shared downstream classifier-grid flags to a parser."""

    parser.add_argument("--classifier-c-grid", default=default_c_grid)
    parser.add_argument("--classifier-penalties", default=default_penalties)
    parser.add_argument("--classifier-solvers", default=default_solvers)
    parser.add_argument("--classifier-class-weights", default=default_class_weights)
    parser.add_argument("--classifier-max-iters", default=default_max_iters)
    parser.add_argument("--classifier-l1-ratios", default="")


def classifier_specs_from_args(args: argparse.Namespace) -> tuple[ClassifierSpec, ...]:
    """Build validated classifier specs from the shared CLI flags."""

    return build_classifier_specs(
        c_grid=str(args.classifier_c_grid),
        penalties=str(args.classifier_penalties),
        solvers=str(args.classifier_solvers),
        class_weights=str(args.classifier_class_weights),
        max_iters=str(args.classifier_max_iters),
        l1_ratios=str(args.classifier_l1_ratios),
        classifier_seed=int(args.classifier_seed),
    )


def build_classifier_specs(
    *,
    c_grid: str,
    penalties: str,
    solvers: str,
    class_weights: str,
    max_iters: str,
    classifier_seed: int,
    l1_ratios: str = "",
) -> tuple[ClassifierSpec, ...]:
    """Build a deterministic grid of validated logistic-regression specs."""

    c_values = _parse_float_list(c_grid, "classifier-c-grid")
    penalty_values = _parse_str_list(penalties, "classifier-penalties")
    solver_values = _parse_str_list(solvers, "classifier-solvers")
    class_weight_values = tuple(_parse_class_weight(value) for value in _parse_str_list(class_weights, "classifier-class-weights"))
    max_iter_values = _parse_int_list(max_iters, "classifier-max-iters")
    l1_ratio_values = _parse_float_list(l1_ratios, "classifier-l1-ratios") if str(l1_ratios).strip() else ()
    specs: list[ClassifierSpec] = []
    for c_value in c_values:
        for penalty in penalty_values:
            for solver in solver_values:
                for class_weight in class_weight_values:
                    for max_iter in max_iter_values:
                        if penalty == "elasticnet":
                            if not l1_ratio_values:
                                raise ProtocolError("elasticnet classifier specs require --classifier-l1-ratios.")
                            for l1_ratio in l1_ratio_values:
                                specs.append(
                                    ClassifierSpec(
                                        C=float(c_value),
                                        penalty=penalty,
                                        solver=solver,
                                        max_iter=int(max_iter),
                                        class_weight=class_weight,
                                        l1_ratio=float(l1_ratio),
                                        random_state=int(classifier_seed),
                                    )
                                )
                        else:
                            specs.append(
                                ClassifierSpec(
                                    C=float(c_value),
                                    penalty=penalty,
                                    solver=solver,
                                    max_iter=int(max_iter),
                                    class_weight=class_weight,
                                    random_state=int(classifier_seed),
                                )
                            )
    if not specs:
        raise ProtocolError("Classifier tuning grid is empty.")
    return tuple(specs)


def _parse_str_list(raw: str, label: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in str(raw).split(",") if part.strip())
    if not values:
        raise ProtocolError(f"--{label} must contain at least one value.")
    return values


def _parse_int_list(raw: str, label: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())
    except ValueError as exc:
        raise ProtocolError(f"Invalid integer in --{label}: {raw!r}") from exc
    if not values:
        raise ProtocolError(f"--{label} must contain at least one value.")
    return values


def _parse_float_list(raw: str, label: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part.strip()) for part in str(raw).split(",") if part.strip())
    except ValueError as exc:
        raise ProtocolError(f"Invalid float in --{label}: {raw!r}") from exc
    if not values:
        raise ProtocolError(f"--{label} must contain at least one value.")
    return values


def _parse_class_weight(raw: object) -> str | None:
    value = str(raw).strip().lower()
    if value in {"none", "null", ""}:
        return None
    if value == "balanced":
        return "balanced"
    raise ProtocolError(f"Unsupported classifier class_weight: {raw!r}")


def csv_values(raw: str | Sequence[object]) -> tuple[str, ...]:
    """Parse a comma-separated list, preserving string center IDs."""

    if isinstance(raw, str):
        values = tuple(part.strip() for part in raw.split(",") if part.strip())
    else:
        values = tuple(str(part).strip() for part in raw if str(part).strip())
    if not values:
        raise ProtocolError("Expected at least one comma-separated value.")
    return values
