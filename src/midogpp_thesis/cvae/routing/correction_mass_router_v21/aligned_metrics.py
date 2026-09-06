"""Fold-scoped contributions to equal-center/class/supporting-case BACC."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence
from ...protocol import ProtocolError
from .contracts import SupportCaseClassProfile
from .hashing import canonical_hash


@dataclass(frozen=True, slots=True)
class ClassSupportNormalizer:
    # Each center: total case count, supporting class-zero cases, class-one cases.
    center_counts: tuple[tuple[str, int, int, int], ...]
    case_keys: tuple[tuple[str, str], ...]
    normalization_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.center_counts or any(n < 1 or c0 < 1 or c1 < 1 or max(c0,c1) > n
                                        for _, n, c0, c1 in self.center_counts):
            raise ProtocolError("HARP v21 aligned BACC requires both classes in each scoring-scope center.")
        object.__setattr__(self, "normalization_hash", canonical_hash({
            "schema_version": "harp_v21_class_support_normalizer", "center_counts": self.center_counts,
            "case_keys": self.case_keys, "estimand": "equal_center_equal_class_equal_supporting_case"}))

    @classmethod
    def fit(cls, profiles: Sequence[SupportCaseClassProfile]) -> "ClassSupportNormalizer":
        rows = tuple(profiles)
        keys = tuple(sorted((p.center_id,p.case_id) for p in rows))
        if not keys or len(keys) != len(set(keys)):
            raise ProtocolError("HARP v21 normalization profiles have empty/duplicate cases.")
        counts = Counter(p.center_id for p in rows)
        return cls(tuple((c, n, sum(p.center_id == c and p.class_0_count > 0 for p in rows),
                          sum(p.center_id == c and p.class_1_count > 0 for p in rows))
                         for c,n in sorted(counts.items())), keys)

    def contribution(self, center_id: str, class_0_gain: float | None,
                     class_1_gain: float | None) -> float:
        match = next((row for row in self.center_counts if row[0] == center_id), None)
        if match is None:
            raise ProtocolError("HARP v21 gain normalization crossed its center scope.")
        _, n, c0, c1 = match
        # These contributions can exceed +/-1. Their within-center mean is BACC gain.
        return .5*n*((0. if class_0_gain is None else class_0_gain/c0)
                     +(0. if class_1_gain is None else class_1_gain/c1))

    def public_payload(self) -> dict[str, object]:
        return {"center_counts": [list(x) for x in self.center_counts],
                "case_keys": [list(x) for x in self.case_keys],
                "normalization_hash": self.normalization_hash}
