"""Label-free feature firewall for the source-only row posterior."""

from __future__ import annotations

from typing import Sequence

from ...protocol import ProtocolError
from .contracts import feature_name_tokens


ROW_POSTERIOR_MAX_FEATURES = 12

_FORBIDDEN_FEATURE_TOKENS = frozenset(
    {
        "target", "label", "labels", "truth", "outcome", "class", "center",
        "centre", "site", "domain", "dataset", "query", "case", "sample",
        "patient", "slide", "image", "filename", "filepath", "path", "uuid",
        "identity", "identifier", "id",
    }
)
_EVIDENCE_FEATURE_TOKENS = frozenset(
    {
        "p", "prob", "probability", "logit", "margin", "disagreement",
        "entropy", "variance", "std", "mean", "median", "quantile",
        "crossing", "flip", "distance", "agreement", "ensemble", "candidate",
        "protected", "baseline", "support", "density",
    }
)


def assert_label_free_feature_names(feature_names: Sequence[str]) -> tuple[str, ...]:
    """Reject identity, target, or label-derived row-posterior features."""

    names = tuple(str(name).strip() for name in feature_names)
    if (
        not names
        or len(names) > ROW_POSTERIOR_MAX_FEATURES
        or len(set(names)) != len(names)
        or any(not name for name in names)
    ):
        raise ProtocolError("Row-posterior feature schema is empty, duplicated, or too wide.")
    for name in names:
        tokens = set(feature_name_tokens(name))
        if tokens.intersection(_FORBIDDEN_FEATURE_TOKENS):
            raise ProtocolError(f"Forbidden target/identity row-posterior feature: {name}")
        if not tokens.intersection(_EVIDENCE_FEATURE_TOKENS):
            raise ProtocolError(
                f"Row-posterior feature is not probability/margin/disagreement evidence: {name}"
            )
    return names


__all__ = ("ROW_POSTERIOR_MAX_FEATURES", "assert_label_free_feature_names")
