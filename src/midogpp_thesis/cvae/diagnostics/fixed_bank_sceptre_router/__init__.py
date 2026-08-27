"""SCEPTRE: a scoped, selective fixed-bank routing diagnostic.

The package intentionally separates three capabilities:

* :mod:`midogpp_thesis.cvae.routing.sceptre` is label-free and reusable;
* the development modules may consume only the explicitly fenced historical
  source-inner utility surface; and
* the runner is planned and mutation-free until a separate consumed-test
  execution identity is authorized.
"""

from .config import SceptreConfig, load_config
from .identity import EXPERIMENT_ID, OUTPUT_ARTIFACT_ID
from .runner import run_planned_sceptre_router

__all__ = (
    "EXPERIMENT_ID",
    "OUTPUT_ARTIFACT_ID",
    "SceptreConfig",
    "load_config",
    "run_planned_sceptre_router",
)
