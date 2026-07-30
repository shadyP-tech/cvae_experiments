"""Bounded canonical-B source-expert adaptation pilot.

Public attributes are loaded on first access so importing neutral CVAE
primitives never imports the pilot runner.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import PilotConfig, load_pilot_config
    from .replay_snapshot import ReplaySnapshot
    from .runner import run_pilot


_EXPORT_MODULES = {
    "PilotConfig": ".config",
    "ReplaySnapshot": ".replay_snapshot",
    "load_pilot_config": ".config",
    "run_pilot": ".runner",
}


def __getattr__(name: str) -> object:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(_EXPORT_MODULES))


__all__ = ("PilotConfig", "ReplaySnapshot", "load_pilot_config", "run_pilot")
