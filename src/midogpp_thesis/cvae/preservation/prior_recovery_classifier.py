"""Frozen classifier policy used inside Stage-20 source-inner recipe selection."""

from __future__ import annotations

from ...real_features.classifier_reference.classifier_grid import build_classifier_specs
from ...real_features.classifier_reference.classifiers import (
    ClassifierSpec,
    classifier_grid_hash,
)
from ...real_features.classifier_reference.protocol import ProtocolError


SOURCE_INNER_CLASSIFIER_GRID_HASH = "59b9fa2a008dedc5"
SOURCE_INNER_CLASSIFIER_GRID_SIZE = 2


def source_inner_classifier_specs(*, classifier_seed: int = 23) -> tuple[ClassifierSpec, ...]:
    """Return the predeclared Stage-20 grid with C frozen from Stage-10 evidence."""

    specs = build_classifier_specs(
        c_grid="0.01",
        penalties="l2",
        solvers="lbfgs",
        class_weights="none,balanced",
        max_iters="5000",
        classifier_seed=int(classifier_seed),
    )
    observed = classifier_grid_hash(specs)
    if int(classifier_seed) == 23 and observed != SOURCE_INNER_CLASSIFIER_GRID_HASH:
        raise ProtocolError(
            "Stage-20 source-inner classifier grid drift: "
            f"expected={SOURCE_INNER_CLASSIFIER_GRID_HASH} actual={observed}"
        )
    return specs

