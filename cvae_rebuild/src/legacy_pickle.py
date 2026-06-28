from __future__ import annotations

import importlib
import sys
import types


LEGACY_CVAE_REBUILD_MODULES = (
    "downstream",
    "feature_frame",
    "features",
    "metrics",
    "models",
    "preservation",
    "preservation_repair",
    "protocol",
    "reporting",
    "splits",
)


def install_legacy_cvae_rebuild_pickle_aliases() -> None:
    """Allow old repair/runtime pickles to load after the direct src layout move."""

    package = sys.modules.get("cvae_rebuild")
    if package is None:
        package = types.ModuleType("cvae_rebuild")
        package.__path__ = []
        sys.modules["cvae_rebuild"] = package

    for module_name in LEGACY_CVAE_REBUILD_MODULES:
        legacy_name = f"cvae_rebuild.{module_name}"
        if legacy_name in sys.modules:
            continue
        sys.modules[legacy_name] = importlib.import_module(module_name)
