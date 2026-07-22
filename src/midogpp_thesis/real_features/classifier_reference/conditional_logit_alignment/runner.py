"""Thin production orchestrator for the Stage-10 CLA diagnostic."""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Mapping

from ..artifacts import prepare_artifact_dirs
from ..protocol import ProtocolError
from ..real_feature_frame import load_midogpp_real_feature_frame
from ..schemas.midogpp import MIDOGPP_ELIGIBLE_CENTERS
from .artifacts import write_completed_bundle, write_frozen_protocol_snapshot
from .config import (
    ConditionalLogitAlignmentConfig,
    EXPECTED_VERSIONS,
    validate_runtime_config,
)
from .estimator import fit_prepared_conditional_logit, prepare_conditional_logit
from .folds import make_outer_fold
from .selection import select_gamma_source_inner
from .table_rows import OuterEvaluation, build_alignment_artifact_tables
from .validation import assert_conditional_logit_alignment_artifacts
from .workspace_binding import validate_production_workspace_binding


THREAD_ENVIRONMENT_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def run_conditional_logit_alignment(
    config: ConditionalLogitAlignmentConfig,
    *,
    artifact_root: Path | None = None,
) -> Path:
    """Run nested source-only gamma selection and two-role outer evaluation."""

    started = time.perf_counter()
    validate_runtime_config(config)
    workspace_binding = (
        None
        if config.allow_partial_test_coverage
        else validate_production_workspace_binding(
            config,
            artifact_root_override=artifact_root,
        )
    )
    runtime_environment = validate_runtime_environment(config)
    root = prepare_artifact_dirs(artifact_root or config.artifact_root)
    frame = load_midogpp_real_feature_frame(
        manifest_path=config.manifest_path,
        feature_cache_path=config.feature_cache_path,
        expected_feature_dim=config.expected_feature_dim,
    )
    heldouts = tuple(str(value) for value in config.heldout_centers)
    _validate_coverage(config, frame, heldouts)
    if workspace_binding is not None:
        manifest_binding = workspace_binding.get("manifest")
        feature_binding = workspace_binding.get("feature_cache")
        if (
            not isinstance(manifest_binding, Mapping)
            or not isinstance(feature_binding, Mapping)
            or frame.manifest_hash != manifest_binding.get("sha256")
            or frame.feature_cache_hash != feature_binding.get("sha256")
        ):
            raise ProtocolError("CLA loaded frame differs from the pre-fit workspace binding.")

    # This is intentionally persisted before source-inner fitting or any outer
    # target scoring.  Later files bind back to this immutable design hash.
    frozen = write_frozen_protocol_snapshot(
        root,
        config,
        workspace_binding=workspace_binding,
    )

    selections = tuple(
        select_gamma_source_inner(
            frame,
            heldout,
            config.gamma_grid,
            config.classifier_spec,
            optimizer=config.optimizer,
        )
        for heldout in heldouts
    )
    outer_evaluations: list[OuterEvaluation] = []
    for selection in selections:
        # Only after H's source-inner gamma is frozen do we materialize the
        # outer fold and score H.  No outer all-gamma surface exists.
        fold = make_outer_fold(frame, selection.outer_target_center)
        prepared = prepare_conditional_logit(fold, config.classifier_spec)
        gamma0 = fit_prepared_conditional_logit(
            prepared,
            0.0,
            optimizer=config.optimizer,
        )
        selected = (
            gamma0
            if float(selection.selected_gamma) == 0.0
            else fit_prepared_conditional_logit(
                prepared,
                float(selection.selected_gamma),
                optimizer=config.optimizer,
            )
        )
        outer_evaluations.append(
            OuterEvaluation(
                prepared=prepared,
                selected_gamma=float(selection.selected_gamma),
                selected_fit=selected,
                gamma0_fit=gamma0,
            )
        )

    tables = build_alignment_artifact_tables(
        selections,
        outer_evaluations,
        frame=frame,
        tie_atol=config.tie_atol,
        tie_rtol=config.tie_rtol,
    )
    expected_counts = _expected_counts(config, frame, selections, tables)
    _assert_table_counts(tables.as_mapping(), expected_counts)
    elapsed = time.perf_counter() - started
    write_completed_bundle(
        root,
        frozen=frozen,
        unbound_tables=tables,
        frame=frame,
        heldout_centers=heldouts,
        gamma_grid=config.gamma_grid,
        classifier_config_hash=config.classifier_spec.config_hash,
        expected_counts=expected_counts,
        coverage_mode=(
            "partial_test" if config.allow_partial_test_coverage else "complete"
        ),
        experiment_seed=config.experiment_seed,
        elapsed_seconds=elapsed,
        runtime_environment=runtime_environment,
    )
    assert_conditional_logit_alignment_artifacts(root, already_loaded_frame=frame)
    return root


def validate_runtime_environment(
    config: ConditionalLogitAlignmentConfig,
) -> dict[str, object]:
    """Fail before scoring if frozen library or single-thread locks drift."""

    try:
        import numpy  # type: ignore
        import scipy  # type: ignore
        import sklearn  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - direct dependencies
        raise RuntimeError("CLA requires numpy, scipy, and scikit-learn.") from exc

    expected = {str(key): str(value) for key, value in config.expected_versions.items()}
    if expected != dict(EXPECTED_VERSIONS):
        raise ProtocolError("CLA expected runtime version contract drifted.")
    observed = {
        "numpy": str(numpy.__version__),
        "scipy": str(scipy.__version__),
        "scikit_learn": str(sklearn.__version__),
    }
    if observed != expected:
        raise ProtocolError(
            "CLA runtime library versions differ from the frozen contract: "
            f"expected={expected}, observed={observed}."
        )
    thread_environment = {key: os.environ.get(key) for key in THREAD_ENVIRONMENT_KEYS}
    if config.optimizer.require_single_thread and any(
        value != "1" for value in thread_environment.values()
    ):
        raise ProtocolError(
            "CLA requires OMP/MKL/OPENBLAS/NUMEXPR thread counts to equal 1: "
            f"observed={thread_environment}."
        )
    return {
        "expected_versions": expected,
        "observed_versions": observed,
        "versions_match": True,
        "require_single_thread": bool(config.optimizer.require_single_thread),
        "thread_environment": thread_environment,
        "single_thread_match": True,
    }


def _validate_coverage(
    config: ConditionalLogitAlignmentConfig,
    frame: object,
    heldouts: tuple[str, ...],
) -> None:
    observed = tuple(str(value) for value in getattr(frame, "eligible_centers"))
    if not heldouts or len(set(heldouts)) != len(heldouts):
        raise ProtocolError("CLA held-out center coverage is empty or duplicated.")
    if any(value not in observed for value in heldouts):
        raise ProtocolError("CLA held-out center is absent from the feature frame.")
    if not config.allow_partial_test_coverage and (
        observed != MIDOGPP_ELIGIBLE_CENTERS
        or heldouts != MIDOGPP_ELIGIBLE_CENTERS
    ):
        raise ProtocolError("Production CLA requires exact nine-center coverage.")


def _expected_counts(
    config: ConditionalLogitAlignmentConfig,
    frame: object,
    selections: tuple[object, ...],
    tables: object,
) -> dict[str, int]:
    n_outer = len(config.heldout_centers)
    n_inner_by_outer = [len(getattr(item, "inner_centers")) for item in selections]
    n_gamma = len(config.gamma_grid)
    return {
        "source_inner_fold_scores": sum(n_inner_by_outer) * n_gamma,
        "source_inner_gamma_summary": n_outer * n_gamma,
        "outer_results": n_outer * 2,
        "outer_predictions": 2
        * sum(
            1
            for heldout in config.heldout_centers
            for row in getattr(frame, "rows")
            if str(row.center) == str(heldout)
        ),
        "conditional_frame_audit": sum(n_inner_by_outer) + n_outer,
        "solver_audit": len(getattr(tables, "solver_audit")),
        "outer_comparison": n_outer,
    }


def _assert_table_counts(
    tables: Mapping[str, object],
    expected: Mapping[str, int],
) -> None:
    actual = {name: len(rows) for name, rows in tables.items()}  # type: ignore[arg-type]
    if actual != dict(expected):
        raise ProtocolError(
            f"CLA logical table cardinalities drifted: expected={dict(expected)}, actual={actual}."
        )


__all__ = [
    "THREAD_ENVIRONMENT_KEYS",
    "run_conditional_logit_alignment",
    "validate_runtime_environment",
]
