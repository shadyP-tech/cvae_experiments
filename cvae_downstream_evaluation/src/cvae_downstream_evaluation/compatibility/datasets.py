"""Source-inner estimator dataset validation."""

from __future__ import annotations

from typing import Mapping, Sequence

from . import assert_source_inner_training_labels


def validate_source_inner_rows(rows: Sequence[Mapping[str, object]]) -> None:
    assert_source_inner_training_labels(rows)
