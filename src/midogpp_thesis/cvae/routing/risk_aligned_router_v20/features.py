"""Outcome-blind features of the exact executed composite and its flip sets.

Per-seed composite predictions are absent from this surface. We deliberately
omit composite seed uncertainty instead of substituting mean donor dispersion.
Only explicitly named whole-expert compatibility context may be aggregated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
import math
from threading import Lock
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .contracts import CompositeKind, LabelFreeCaseMenu, SoftTopKComposite, decode_probability_hex
from .hashing import canonical_hash
from .patch_evidence import EVIDENCE_FEATURE_NAMES, evidence_descriptor

KINDS = (CompositeKind.B, CompositeKind.U_FULL, CompositeKind.D01_ONLY,
         CompositeKind.D10_ONLY, CompositeKind.BOTH)
_GLOBAL = ("sample_count", "log_k", "lambda", "baseline_mean", "baseline_positive_fraction",
           "mean_probability_delta", "mean_absolute_delta", "delta_std", "maximum_absolute_delta",
           "hard_change_fraction", "action_margin_min", "action_margin_median")
_BRANCH = ("flip_count", "flip_fraction", "baseline_margin_q10", "baseline_margin_median",
           "action_margin_q10", "action_margin_median", "flip_delta_mean", "flip_delta_std",
           "selected_donor_std_on_flips", "selected_donor_disagreement_on_flips")
EXACT_FEATURE_NAMES = (*_GLOBAL, *(f"{direction}_{name}" for direction in ("d01", "d10") for name in _BRANCH))
_DESCRIPTOR_SCHEMA = "harp_v20_exact_executed_composite_features_v1"


class RawFeatureCache:
    """Bounded pure-feature cache owned by one scientific fit execution.

    Context selection is explicit, so no entries survive implicitly into a
    successor run. Values contain raw label-free floats, never fitted moments.
    """
    def __init__(self, max_entries=8192):
        if type(max_entries) is not int or max_entries < 1:
            raise ProtocolError("HARP v20 raw feature cache bound is malformed.")
        self.max_entries = max_entries
        self._entries = OrderedDict()
        self._lock = Lock()
        self.hits = self.misses = 0

    def get(self, key):
        with self._lock:
            values = self._entries.get(key)
            if values is None:
                self.misses += 1
                return None
            self.hits += 1
            self._entries.move_to_end(key)
            return dict(values)

    def put(self, key, values):
        with self._lock:
            self._entries[key] = tuple(values.items())
            self._entries.move_to_end(key)
            if len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def public_payload(self):
        with self._lock:
            return {"schema": _DESCRIPTOR_SCHEMA, "max_entries": self.max_entries,
                    "entry_count": len(self._entries), "hits": self.hits, "misses": self.misses,
                    "raw_label_free_features_only": True, "fitted_statistics_cached": False}


_RAW_FEATURE_CACHE = ContextVar("harp_v20_raw_feature_cache", default=None)


def current_raw_feature_cache():
    return _RAW_FEATURE_CACHE.get()


@contextmanager
def use_raw_feature_cache(cache):
    """Scope raw feature reuse to a fresh execution; restore context on error."""
    if not isinstance(cache, RawFeatureCache):
        raise ProtocolError("HARP v20 raw feature context needs an explicit cache.")
    token = _RAW_FEATURE_CACHE.set(cache)
    try:
        yield cache
    finally:
        _RAW_FEATURE_CACHE.reset(token)


def exact_composite_features(menu: LabelFreeCaseMenu, composite: SoftTopKComposite, *,
                             baseline_array=None, action_cache=None) -> dict[str, float]:
    if (composite.menu_hash != menu.menu_hash or composite.sample_ids != menu.sample_ids
        or composite.baseline_probability_hex != menu.baseline_probability_hex
        or (composite.center_id, composite.case_id) != (menu.center_id, menu.case_id)):
        raise ProtocolError("HARP v20 executed-action features crossed a sealed menu.")
    cache = _RAW_FEATURE_CACHE.get()
    key = (_DESCRIPTOR_SCHEMA, menu.menu_hash, composite.composite_hash)
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return cached
    base = np.asarray(decode_probability_hex(menu.baseline_probability_hex), dtype=float) if baseline_array is None else baseline_array
    selected = np.asarray(decode_probability_hex(composite.probability_hex), dtype=float)
    delta, base_hard, selected_hard = selected - base, base >= .5, selected >= .5
    flipped = base_hard != selected_hard
    values = (len(base), math.log1p(composite.k or 0), composite.mixing_lambda or 0.,
              base.mean(), base_hard.mean(), delta.mean(), np.abs(delta).mean(), delta.std(),
              np.abs(delta).max(), flipped.mean(), np.abs(selected-.5).min(), np.median(np.abs(selected-.5)))
    result = dict(zip(_GLOBAL, map(float, values), strict=True))
    for direction, mask, ids in (
        ("d01", ~base_hard & selected_hard, composite.d01_action_ids),
        ("d10", base_hard & ~selected_hard, composite.d10_action_ids),
    ):
        stats = {key: 0.0 for key in _BRANCH}
        stats.update(flip_count=float(mask.sum()), flip_fraction=float(mask.mean()))
        if np.any(mask):
            stats.update(baseline_margin_q10=float(np.quantile(np.abs(base[mask]-.5), .1)),
                         baseline_margin_median=float(np.median(np.abs(base[mask]-.5))),
                         action_margin_q10=float(np.quantile(np.abs(selected[mask]-.5), .1)),
                         action_margin_median=float(np.median(np.abs(selected[mask]-.5))),
                         flip_delta_mean=float(delta[mask].mean()), flip_delta_std=float(delta[mask].std()))
            if ids:
                donors = []
                for arm in ids:
                    if action_cache is not None and arm in action_cache:
                        probability = action_cache[arm]
                    else:
                        probability = np.asarray(decode_probability_hex(menu.action_for(arm).action_probability_hex))
                        if action_cache is not None:
                            action_cache[arm] = probability
                    # Dispersion of selected donor endpoints under this lambda,
                    # measured at the exact composite's changed predictions.
                    donors.append((base[mask] + composite.mixing_lambda * (probability[mask]-base[mask])).astype(np.float32))
                donors = np.stack(donors)
                stats["selected_donor_std_on_flips"] = float(np.std(donors.astype(float), axis=0).mean())
                stats["selected_donor_disagreement_on_flips"] = float(((donors >= .5) != selected_hard[mask]).mean())
        result.update({f"{direction}_{key}": value for key, value in stats.items()})
    if cache is not None:
        cache.put(key, result)
    return result


@dataclass(frozen=True, slots=True)
class CompositeFeatureScope:
    feature_names: tuple[str, ...]
    training_case_keys: tuple[tuple[str, str], ...]
    transform_hash: str = field(init=False)

    def __post_init__(self):
        if (not self.training_case_keys or len(set(self.training_case_keys)) != len(self.training_case_keys)
            or any(not name.startswith("compatibility") for name in self.feature_names)):
            raise ProtocolError("HARP v20 composite feature scope is malformed.")
        object.__setattr__(self, "transform_hash", canonical_hash(self._payload()))

    def _payload(self):
        return {"schema_version": "harp_v20_executed_composite_feature_scope", "feature_names": self.feature_names,
                "training_case_keys": self.training_case_keys, "exact_feature_names": EXACT_FEATURE_NAMES,
                "primitive_nonlinear_statistics_averaged": False,
                "composite_seed_uncertainty_available": False, "labels_used": False}

    def public_payload(self):
        return {**self._payload(), "transform_hash": self.transform_hash}


def fit_composite_feature_scope(menus: Sequence[LabelFreeCaseMenu], *, maximum_numeric_features=20):
    rows = tuple(menus)
    schemas = {action.feature_names for menu in rows for action in menu.actions}
    if len(schemas) != 1 or maximum_numeric_features < 1:
        raise ProtocolError("HARP v20 composite context needs one label-free schema.")
    names = tuple(name for name in next(iter(schemas)) if name.startswith("compatibility"))[:maximum_numeric_features]
    return CompositeFeatureScope(names, tuple(sorted((m.center_id, m.case_id) for m in rows)))


def descriptor_names(scope):
    return ("intercept", *(f"context_selected::{name}" for name in scope.feature_names),
            *(f"context_D01::{name}" for name in scope.feature_names),
            *(f"context_D10::{name}" for name in scope.feature_names),
            *(f"kind::{kind.value}" for kind in KINDS), *EXACT_FEATURE_NAMES, *EVIDENCE_FEATURE_NAMES)


def composite_descriptor(menu, composite, scope, *, baseline_array=None, numeric_cache=None, patch_probability=None):
    exact = exact_composite_features(menu, composite, baseline_array=baseline_array, action_cache=numeric_cache)
    selected = (*composite.d01_action_ids, *composite.d10_action_ids)
    if composite.kind is CompositeKind.U_FULL:
        selected = (menu.full_action.arm_id,)

    def context(ids):
        if not ids:
            return [0.] * len(scope.feature_names)
        values = [dict(zip(menu.action_for(arm).feature_names, menu.action_for(arm).feature_values, strict=True)) for arm in ids]
        return [float(np.mean([row[name] for row in values])) for name in scope.feature_names]

    return np.asarray([1., *context(selected), *context(composite.d01_action_ids), *context(composite.d10_action_ids),
                       *(float(composite.kind is kind) for kind in KINDS), *(exact[name] for name in EXACT_FEATURE_NAMES),
                       *evidence_descriptor(menu, composite, patch_probability)])


__all__ = ("EXACT_FEATURE_NAMES", "CompositeFeatureScope", "exact_composite_features",
           "fit_composite_feature_scope", "descriptor_names", "composite_descriptor",
           "RawFeatureCache", "use_raw_feature_cache", "current_raw_feature_cache")
