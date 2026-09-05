"""Source-only target validation, population filtering and case weights."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ...protocol import ProtocolError
from .aligned_metrics import ClassSupportNormalizer
from .contracts import CompositeKind, SupportCaseClassProfile, SurfaceRole
from .modeling import _case_weights


def category_target(outcome) -> int:
    """Exhaustive disjoint outcomes: safe positive, BACC harm, remaining."""
    return 0 if outcome.safe_positive else 1 if outcome.harmed else 2


@dataclass(frozen=True, slots=True)
class CandidateFittingPopulation:
    composites: tuple
    outcomes: tuple
    row_weights: tuple[float, ...]
    normalizer: ClassSupportNormalizer
    gain_magnitude_bound: float
    excluded_baseline_count: int
    excluded_no_hard_change_count: int
    excluded_duplicate_count: int


def prepare_candidate_population(menus, composites, outcomes, *, normalization_profiles=None):
    menus, rows, truth = tuple(menus), tuple(composites), tuple(outcomes)
    keys = {(m.center_id, m.case_id) for m in menus}
    if (not keys or len(keys) != len(menus) or len(rows) != len(truth)
        or any(m.surface_role is not SurfaceRole.SOURCE_TRAIN_DEVELOPMENT for m in menus)
        or len({c.composite_hash for c in rows}) != len(rows)
        or any((c.center_id, c.case_id) not in keys or c.composite_hash != o.composite.composite_hash
               or o.normalization_hash is None for c, o in zip(rows, truth, strict=True))):
        raise ProtocolError("HARP v20 action fitting needs exact scope-aligned honest composite outcomes.")
    menus_by_key = {(m.center_id, m.case_id): m for m in menus}
    presence = {}
    for composite, outcome in zip(rows, truth, strict=True):
        key = (composite.center_id, composite.case_id)
        if composite.menu_hash != menus_by_key[key].menu_hash:
            raise ProtocolError("HARP v20 action fitting crossed its sealed menu.")
        supported = (outcome.class_0_gain is not None, outcome.class_1_gain is not None)
        if key in presence and presence[key] != supported:
            raise ProtocolError("HARP v20 candidates disagree on class support.")
        presence[key] = supported
    if normalization_profiles is None:
        # Backward-compatible only if UNFILTERED inputs cover the entire scope.
        if set(presence) != keys:
            raise ProtocolError("HARP v20 action fitting requires full-scope normalization profiles.")
        profiles = tuple(SupportCaseClassProfile(c, k, int(p0)+int(p1), int(p0), int(p1), 0, 0)
                         for (c, k), (p0, p1) in sorted(presence.items()))
    else:
        profiles = tuple(normalization_profiles)
        if {(p.center_id, p.case_id) for p in profiles} != keys or len(profiles) != len(keys):
            raise ProtocolError("HARP v20 action normalization crossed its exact fitting scope.")
        if any(presence.get((p.center_id, p.case_id), (p.class_0_count > 0, p.class_1_count > 0))
               != (p.class_0_count > 0, p.class_1_count > 0) for p in profiles):
            raise ProtocolError("HARP v20 outcomes disagree with normalization class support.")
    normalizer = ClassSupportNormalizer.fit(profiles)
    if any(o.normalization_hash != normalizer.normalization_hash or abs(o.bacc_gain - normalizer.contribution(
        o.composite.center_id, o.class_0_gain, o.class_1_gain)) > 1e-12 for o in truth):
        raise ProtocolError("HARP v20 action outcome normalization crossed its exact fitting scope.")
    # Outcome-side full-scope bound; never a target feature. It may exceed one.
    bound = max(.5 * n * (1 / c0 + 1 / c1) for _, n, c0, c1 in normalizer.center_counts)
    selected, seen = [], set()
    baseline_count = noop_count = duplicate_count = 0
    for i, c in enumerate(rows):
        if c.kind is CompositeKind.B:
            baseline_count += 1
        elif not c.prediction_changed:
            noop_count += 1
        elif (c.center_id, c.case_id, c.probability_hex) in seen:
            duplicate_count += 1
        else:
            seen.add((c.center_id, c.case_id, c.probability_hex))
            selected.append(i)
    fitted = tuple(rows[i] for i in selected)
    fitted_truth = tuple(truth[i] for i in selected)
    counts = Counter((c.center_id, c.case_id) for c in fitted)
    case_weights = _case_weights(tuple(counts)) if counts else {}
    weights = tuple(case_weights[(c.center_id, c.case_id)] / counts[(c.center_id, c.case_id)] for c in fitted)
    return CandidateFittingPopulation(fitted, fitted_truth, weights, normalizer, bound,
                                      baseline_count, noop_count, duplicate_count)


__all__ = ("category_target", "prepare_candidate_population", "CandidateFittingPopulation")
