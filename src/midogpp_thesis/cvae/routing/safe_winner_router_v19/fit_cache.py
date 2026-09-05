"""Bounded, execution-local fit memoization; never serializes truth or models."""
from collections import OrderedDict
from .hashing import canonical_hash


class ScopedFitCache:
    def __init__(self, maximum_entries: int = 96) -> None:
        self.maximum_entries = maximum_entries
        self._models = OrderedDict()
        self.hits = 0
        self.misses = 0

    def key(self, kind, menus, capability, config):
        return canonical_hash({"kind": kind,
            "menu_hashes": tuple(menu.menu_hash for menu in menus),
            "case_keys": tuple((menu.center_id, menu.case_id) for menu in menus),
            "capability_hash": capability.capability_hash,
            "config": config.public_payload(), "execution_namespace": "HARP_V19"})

    def get(self, key):
        value = self._models.get(key)
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
            self._models.move_to_end(key)
        return value

    def put(self, key, model):
        self._models[key] = model
        self._models.move_to_end(key)
        while len(self._models) > self.maximum_entries:
            self._models.popitem(last=False)
        return model

    def public_payload(self):
        return {"maximum_entries": self.maximum_entries,
                "serialized_cache": False, "raw_labels_cached": False,
                "key_includes_exact_menus_scope_capability_and_config": True}


def with_execution_feature_cache(function):
    """Share raw features through a public fit call, restoring nested context."""
    from functools import wraps

    @wraps(function)
    def wrapped(*args, **kwargs):
        from .features import RawFeatureCache, current_raw_feature_cache, use_raw_feature_cache
        if current_raw_feature_cache() is not None:
            return function(*args, **kwargs)
        with use_raw_feature_cache(RawFeatureCache(max_entries=8192)):
            return function(*args, **kwargs)
    return wrapped
