"""Fold-local source-crossfit orchestration for the executable HARP v12 run.

This is the sole bridge between label-free H/q/r physical surfaces and
outcome-bearing source fitting.  Each heldout-q fit receives only C-{H,q}
outcomes in a one-task spawned process.  Only predictions return to the
parent, are persisted and freshly reconstructed.  Full source labels can be
joined to q prediction menus only after every one of the 72 fold seals is
durable.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...routing.harp_protocol import HarpSourceLabelRow, canonical_hash
from ...routing.policy_calibrated_residual_router_v12 import (
    EffectiveMenu,
    NestedPolicyFold,
    PairwiseFitConfig,
    SourceActionOutcome,
    assemble_source_lodo_result,
    fit_prelabel_pseudo_target_fold,
)
from ...runtime.harp_v12_execution.contracts import ArtifactValue
from ...runtime.harp_v12_execution.contracts import (
    ActionKind,
    LabelFreeActionBlock,
)
from ...runtime.harp_v12_execution.crossfit_contracts import (
    FoldConditionedSourceSurface,
)
from ...runtime.harp_v12_execution.crossfit_durability import (
    SourceCrossfitSurfaceReceipt,
)
from ...runtime.harp_v12_execution.crossfit_effective_menus import (
    FoldConditionedEffectiveSurface,
)
from ...runtime.harp_v12_execution.model_adapter import (
    OuterRouterBundle,
    RouterFitState,
)
from ...runtime.harp_v12_execution.science_pool import initialize_science_worker
from ...runtime.harp_v12_execution.source_development import SourceDevelopmentState
from .source_crossfit_fold_store import (
    SourceCrossfitFoldSeal,
    SourceCrossfitFoldSealSet,
    persist_source_crossfit_fold,
    persist_source_crossfit_fold_set,
)
from .source_crossfit_artifacts import (
    build_development_artifact as _build_development_artifact,
    build_model_artifact as _build_model_artifact,
    build_source_crossfit_effective_artifact as _build_source_crossfit_effective_artifact,
    build_source_prelabel_prediction_artifact as _build_source_prelabel_prediction_artifact,
    numeric_oof_arrays as _build_numeric_oof_arrays,
    persist_and_reconstruct_source_crossfit_surface,
    source_fold_capability_seal_payload as _source_fold_capability_seal_payload,
    source_prelabel_q_prediction_seal_payload as _source_prelabel_q_prediction_seal_payload,
)
from .source_crossfit_label_joins import (
    attach_prediction_outcomes as _attach_prediction_outcomes_impl,
    join_scoped_worker_outcomes as _join_scoped_worker_outcomes_impl,
)
from .source_label_capability import (
    AggregateSourceLabelCapability,
    FoldSourceLabelCapability,
    issue_aggregate_source_label_capability,
    issue_fold_source_label_capability,
)


SourceLabelLoader = Callable[..., tuple[HarpSourceLabelRow, ...]]
FoldExecutor = Callable[[Sequence["FoldFitTask"], int], tuple["FoldFitExecution", ...]]


@dataclass(frozen=True, slots=True)
class LabelFreeSourceCrossfitBundle:
    physical_surface: FoldConditionedSourceSurface
    surface_receipt: SourceCrossfitSurfaceReceipt
    effective_surface: FoldConditionedEffectiveSurface
    bundle_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.physical_surface, FoldConditionedSourceSurface)
            or not isinstance(self.surface_receipt, SourceCrossfitSurfaceReceipt)
            or not isinstance(self.effective_surface, FoldConditionedEffectiveSurface)
            or self.physical_surface.surface_hash != self.surface_receipt.surface_hash
            or self.effective_surface.source_surface_hash
            != self.physical_surface.surface_hash
            or self.surface_receipt.outer_target_ids != CENTERS
        ):
            raise ProtocolError("HARP v12 label-free source crossfit bundle is unbound.")
        object.__setattr__(
            self,
            "bundle_hash",
            canonical_hash(
                {
                    "schema_version": "midogpp_harp_v12_label_free_source_crossfit_bundle_v1",
                    "source_surface_hash": self.physical_surface.surface_hash,
                    "source_surface_receipt_hash": self.surface_receipt.receipt_hash,
                    "effective_adapter_hash": self.effective_surface.adapter_hash,
                    "labels_consumed": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class FoldFitTask:
    outer_target_id: str
    heldout_center_id: str
    config: object
    cache: object
    source_label_loader: SourceLabelLoader
    label_capability: FoldSourceLabelCapability
    baseline_blocks: tuple[tuple[str, LabelFreeActionBlock], ...]
    fitting_menus: tuple[EffectiveMenu, ...]
    prediction_menus: tuple[EffectiveMenu, ...]
    fit_config: PairwiseFitConfig
    label_capability_hash: str
    source_surface_receipt_hash: str
    source_surface_hash: str
    effective_adapter_hash: str
    prediction_surface_hash: str
    fitting_surface_hash: str
    task_scope_hash: str = field(init=False)

    def __post_init__(self) -> None:
        h = str(self.outer_target_id)
        q = str(self.heldout_center_id)
        baselines = tuple(self.baseline_blocks)
        fitting = tuple(self.fitting_menus)
        prediction = tuple(self.prediction_menus)
        allowed = tuple(center for center in CENTERS if center not in {h, q})
        if (
            h not in CENTERS
            or q not in CENTERS
            or h == q
            or not isinstance(self.label_capability, FoldSourceLabelCapability)
            or self.label_capability.outer_target_id != h
            or self.label_capability.heldout_center_id != q
            or not callable(self.source_label_loader)
            or tuple(row[0] for row in baselines) != allowed
            or any(
                not isinstance(block, LabelFreeActionBlock)
                or block.outer_target_id != h
                or block.query_center_id != query
                or block.action_kind is not ActionKind.B
                for query, block in baselines
            )
            or not fitting
            or not prediction
            or not isinstance(self.fit_config, PairwiseFitConfig)
            or {row.query_center_id for row in fitting} != set(allowed)
            or {row.query_center_id for row in prediction} != {q}
        ):
            raise ProtocolError("HARP v12 isolated fold task escaped C-{H,q}.")
        bindings = (
            self.label_capability_hash,
            self.source_surface_receipt_hash,
            self.source_surface_hash,
            self.effective_adapter_hash,
            self.prediction_surface_hash,
            self.fitting_surface_hash,
        )
        if any(not _is_sha256(value) for value in bindings):
            raise ProtocolError("HARP v12 isolated fold task binding is malformed.")
        body = {
            "schema_version": "midogpp_harp_v12_isolated_fold_fit_task_v1",
            "outer_target_id": h,
            "heldout_center_id": q,
            "allowed_center_ids": list(allowed),
            "baseline_block_hashes": [block.block_hash for _, block in baselines],
            "fitting_menu_hashes": [row.menu_hash for row in fitting],
            "prediction_menu_hashes": [row.menu_hash for row in prediction],
            "label_capability_hash": self.label_capability_hash,
            "source_surface_receipt_hash": self.source_surface_receipt_hash,
            "source_surface_hash": self.source_surface_hash,
            "effective_adapter_hash": self.effective_adapter_hash,
            "prediction_surface_hash": self.prediction_surface_hash,
            "fitting_surface_hash": self.fitting_surface_hash,
            "heldout_q_outcomes_present": False,
        }
        object.__setattr__(self, "outer_target_id", h)
        object.__setattr__(self, "heldout_center_id", q)
        object.__setattr__(self, "baseline_blocks", baselines)
        object.__setattr__(self, "fitting_menus", fitting)
        object.__setattr__(self, "prediction_menus", prediction)
        object.__setattr__(self, "task_scope_hash", canonical_hash(body))


@dataclass(frozen=True, slots=True)
class FoldFitExecution:
    outer_target_id: str
    heldout_center_id: str
    task_scope_hash: str
    nested_fold: NestedPolicyFold
    worker_process_id: int
    cuda_visible_to_worker: bool
    isolation_receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.nested_fold, NestedPolicyFold)
            or self.nested_fold.outer_target_id != self.outer_target_id
            or self.nested_fold.heldout_center_id != self.heldout_center_id
            or type(self.worker_process_id) is not int
            or self.worker_process_id <= 0
            or self.cuda_visible_to_worker is not False
            or not _is_sha256(self.task_scope_hash)
        ):
            raise ProtocolError("HARP v12 isolated fold execution receipt is malformed.")
        object.__setattr__(
            self,
            "isolation_receipt_hash",
            canonical_hash(
                {
                    "schema_version": "midogpp_harp_v12_isolated_fold_execution_v1",
                    "outer_target_id": self.outer_target_id,
                    "heldout_center_id": self.heldout_center_id,
                    "task_scope_hash": self.task_scope_hash,
                    "nested_fold_hash": self.nested_fold.fold_hash,
                    "worker_process_identity_is_operational_only": True,
                    "spawn_start_method": True,
                    "one_task_per_child": True,
                    "cuda_visible_to_worker": False,
                    "heldout_q_outcomes_present": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class SourceCrossfitFitBundle:
    fold_seal_set: SourceCrossfitFoldSealSet
    aggregate_capability: AggregateSourceLabelCapability
    development: ArtifactValue
    fitted: ArtifactValue
    orchestration_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.fold_seal_set, SourceCrossfitFoldSealSet)
            or not isinstance(self.aggregate_capability, AggregateSourceLabelCapability)
            or not isinstance(self.development, ArtifactValue)
            or not isinstance(self.fitted, ArtifactValue)
            or self.aggregate_capability.fold_seal_set.seal_set_hash
            != self.fold_seal_set.seal_set_hash
        ):
            raise ProtocolError("HARP v12 source crossfit fit bundle is malformed.")
        object.__setattr__(
            self,
            "orchestration_hash",
            canonical_hash(
                {
                    "schema_version": "midogpp_harp_v12_source_crossfit_fit_bundle_v1",
                    "fold_seal_set_hash": self.fold_seal_set.seal_set_hash,
                    "aggregate_capability_hash": self.aggregate_capability.capability_hash,
                    "development_surface_hash": self.development.manifest["surface_hash"],
                    "model_hash": self.fitted.manifest["model_hash"],
                    "evaluation_labels_used": False,
                }
            ),
        )


def materialize_label_free_source_crossfit(
    *,
    pipeline: object,
    config: object,
    cache: object,
    centers: Sequence[str],
    scratch_root: Path,
    durable_root: Path,
) -> LabelFreeSourceCrossfitBundle:
    """Materialize, persist, and freshly bind the complete H/q/r surface."""

    typed_centers = tuple(str(value) for value in centers)
    if typed_centers != CENTERS:
        raise ProtocolError("HARP v12 source-crossfit center universe drifted.")
    materialize = getattr(pipeline, "materialize_label_free_source_crossfit_surface", None)
    build_effective = getattr(
        pipeline, "build_label_free_source_crossfit_effective_surface", None
    )
    if not callable(materialize) or not callable(build_effective):
        raise ProtocolError("HARP v12 pipeline lacks the source-crossfit execution seam.")
    physical = materialize(
        config,
        cache,
        outer_targets=typed_centers,
        scratch_root=Path(scratch_root),
    )
    if not isinstance(physical, FoldConditionedSourceSurface):
        raise ProtocolError("HARP v12 pipeline returned an untyped crossfit surface.")
    reconstructed, receipt = persist_and_reconstruct_source_crossfit_surface(
        Path(durable_root), physical
    )
    # Derive the label-free effective adapter from independently reconstructed
    # bytes.  The pre-persistence in-memory surface never authorizes a label.
    effective = build_effective(reconstructed)
    return LabelFreeSourceCrossfitBundle(reconstructed, receipt, effective)


def build_source_crossfit_effective_artifact(
    bundle: LabelFreeSourceCrossfitBundle,
) -> ArtifactValue:
    """Compatibility façade for the extracted effective-menu artifact builder."""

    return _build_source_crossfit_effective_artifact(bundle)


def build_source_prelabel_prediction_artifact(
    fold_seal_set: SourceCrossfitFoldSealSet,
) -> ArtifactValue:
    """Compatibility façade for the extracted prediction artifact builder."""

    return _build_source_prelabel_prediction_artifact(fold_seal_set)


def source_fold_capability_seal_payload(
    fold_seal_set: SourceCrossfitFoldSealSet,
) -> dict[str, object]:
    """Compatibility façade for the extracted fold-capability seal payload."""

    return _source_fold_capability_seal_payload(fold_seal_set)


def source_prelabel_q_prediction_seal_payload(
    fold_seal_set: SourceCrossfitFoldSealSet,
    prediction_artifact: ArtifactValue,
    *,
    store_manifest_sha256: str,
    store_npz_sha256: str,
) -> dict[str, object]:
    """Compatibility façade for the extracted prediction-seal payload."""

    return _source_prelabel_q_prediction_seal_payload(
        fold_seal_set,
        prediction_artifact,
        store_manifest_sha256=store_manifest_sha256,
        store_npz_sha256=store_npz_sha256,
    )


def fit_and_seal_prelabel_source_folds(
    *,
    bundle: LabelFreeSourceCrossfitBundle,
    config: object,
    cache: object,
    source_label_loader: SourceLabelLoader,
    fold_store_root: Path,
    fold_set_path: Path,
    workers: int,
    executor: FoldExecutor | None = None,
) -> SourceCrossfitFoldSealSet:
    """Fit all H/q folds in one-task spawned workers and seal predictions."""

    label_path = Path(getattr(config, "resolved_path")("development_manifest_path"))
    label_hash = str(getattr(config, "expected_hashes")["development_manifest_sha256"])
    fit_config = _fit_config(getattr(config, "model"))
    run_executor = execute_isolated_fold_fits if executor is None else executor
    seals: list[SourceCrossfitFoldSeal] = []
    for h in CENTERS:
        # One canonical, label-free case universe for H.  Each q-fold
        # prediction is rebound to the appropriate H/q/r projection from
        # this universe before that fold can become durable.
        outer_case_menus = tuple(
            wrapper.menu
            for query in CENTERS
            if query != h
            for wrapper in bundle.effective_surface.prediction_menus(h, query)
        )
        tasks: list[FoldFitTask] = []
        capabilities: dict[str, FoldSourceLabelCapability] = {}
        for q in CENTERS:
            if q == h:
                continue
            capability = issue_fold_source_label_capability(
                surface_receipt=bundle.surface_receipt,
                effective_surface=bundle.effective_surface,
                outer_target_id=h,
                heldout_center_id=q,
                label_index_path=label_path,
                label_index_sha256=label_hash,
            )
            allowed = tuple(center for center in CENTERS if center not in {h, q})
            tasks.append(
                _fold_task(
                    bundle,
                    capability,
                    fit_config,
                    config=config,
                    cache=cache,
                    source_label_loader=source_label_loader,
                )
            )
            capabilities[q] = capability
        executions = run_executor(tuple(tasks), workers)
        if {(row.outer_target_id, row.heldout_center_id) for row in executions} != {
            (h, q) for q in CENTERS if q != h
        }:
            raise ProtocolError("HARP v12 isolated fold executor returned incomplete coverage.")
        for execution in sorted(executions, key=lambda row: row.heldout_center_id):
            capability = capabilities[execution.heldout_center_id]
            seals.append(
                persist_source_crossfit_fold(
                    fold_store_root,
                    nested_fold=execution.nested_fold,
                    outer_target_id=h,
                    heldout_center_id=execution.heldout_center_id,
                    source_surface_receipt_hash=bundle.surface_receipt.receipt_hash,
                    source_surface_hash=bundle.physical_surface.surface_hash,
                    effective_adapter_hash=bundle.effective_surface.adapter_hash,
                    prediction_surface_hash=capability.prediction_surface_hash,
                    fitting_surface_hash=capability.fitting_surface_hash,
                    label_capability_hash=capability.capability_hash,
                    isolation_receipt_hash=execution.isolation_receipt_hash,
                    effective_menus=outer_case_menus,
                )
            )
    return persist_source_crossfit_fold_set(
        fold_set_path,
        expected_center_ids=CENTERS,
        source_surface_receipt_hash=bundle.surface_receipt.receipt_hash,
        source_surface_hash=bundle.physical_surface.surface_hash,
        effective_adapter_hash=bundle.effective_surface.adapter_hash,
        fold_seals=tuple(seals),
    )


def assemble_source_crossfit_model(
    *,
    bundle: LabelFreeSourceCrossfitBundle,
    fold_seal_set: SourceCrossfitFoldSealSet,
    config: object,
    cache: object,
    source_label_loader: SourceLabelLoader,
    target_compatibility_hash: str,
) -> SourceCrossfitFitBundle:
    """Open full source labels only after seals, then assemble final models."""

    label_path = Path(getattr(config, "resolved_path")("development_manifest_path"))
    label_hash = str(getattr(config, "expected_hashes")["development_manifest_sha256"])
    aggregate = issue_aggregate_source_label_capability(
        surface_receipt=bundle.surface_receipt,
        fold_seal_set=fold_seal_set,
        label_index_path=label_path,
        label_index_sha256=label_hash,
    )
    labels = source_label_loader(
        config,
        cache,
        allowed_centers=CENTERS,
        source_label_capability=aggregate,
    )
    outcomes, menus = _attach_prediction_outcomes(bundle, labels)
    development_state = SourceDevelopmentState(menus, outcomes)
    development = _development_artifact(
        development_state,
        config_hash=str(getattr(config, "config_hash")),
        bundle=bundle,
        fold_seal_set=fold_seal_set,
        aggregate_capability=aggregate,
    )
    grid = (_fit_config(getattr(config, "model")),)
    bundles: list[OuterRouterBundle] = []
    for h in CENTERS:
        scoped_menus = tuple(row for row in menus if row.outer_target_id == h)
        scoped_outcomes = tuple(row for row in outcomes if row.action.outer_target_id == h)
        folds = tuple(row.nested_fold for row in fold_seal_set.for_outer(h))
        result = assemble_source_lodo_result(
            scoped_outcomes,
            presealed_folds=folds,
            effective_menus=scoped_menus,
            config_grid=grid,
        )
        bundles.append(OuterRouterBundle(h, result))
    fitted_state = RouterFitState(tuple(bundles))
    fitted = _model_artifact(
        fitted_state,
        development=development,
        config_hash=str(getattr(config, "config_hash")),
        target_compatibility_hash=target_compatibility_hash,
        bundle=bundle,
        fold_seal_set=fold_seal_set,
        aggregate_capability=aggregate,
    )
    return SourceCrossfitFitBundle(
        fold_seal_set=fold_seal_set,
        aggregate_capability=aggregate,
        development=development,
        fitted=fitted,
    )


def execute_isolated_fold_fits(
    tasks: Sequence[FoldFitTask], workers: int
) -> tuple[FoldFitExecution, ...]:
    """Bounded spawn pool with one fold per fresh child process."""

    typed = tuple(tasks)
    if not typed or type(workers) is not int or workers < 1:
        raise ProtocolError("HARP v12 isolated fold executor contract drifted.")
    maximum = min(workers, len(typed))
    by_key: dict[tuple[str, str], FoldFitExecution] = {}
    with ProcessPoolExecutor(
        max_workers=maximum,
        mp_context=mp.get_context("spawn"),
        initializer=initialize_science_worker,
        initargs=(1,),
        max_tasks_per_child=1,
    ) as pool:
        iterator = iter(typed)
        pending = {}
        for _ in range(maximum):
            task = next(iterator, None)
            if task is not None:
                pending[pool.submit(_fit_fold_worker, task)] = task
        while pending:
            complete, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in complete:
                task = pending.pop(future)
                result = future.result()
                key = (result.outer_target_id, result.heldout_center_id)
                if key in by_key or key != (task.outer_target_id, task.heldout_center_id):
                    raise ProtocolError("HARP v12 isolated fold result crossed tasks.")
                by_key[key] = result
                replacement = next(iterator, None)
                if replacement is not None:
                    pending[pool.submit(_fit_fold_worker, replacement)] = replacement
    expected = {(task.outer_target_id, task.heldout_center_id) for task in typed}
    if set(by_key) != expected:
        raise ProtocolError("HARP v12 isolated fold execution is incomplete.")
    return tuple(by_key[(task.outer_target_id, task.heldout_center_id)] for task in typed)


def _fit_fold_worker(task: FoldFitTask) -> FoldFitExecution:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ProtocolError("HARP v12 source fold worker can see CUDA devices.")
    allowed = tuple(center for center in CENTERS if center not in {task.outer_target_id, task.heldout_center_id})
    labels = task.source_label_loader(
        task.config,
        task.cache,
        allowed_centers=allowed,
        source_label_capability=task.label_capability,
    )
    outcomes = _join_scoped_worker_outcomes(task, labels)
    fold = fit_prelabel_pseudo_target_fold(
        outcomes,
        heldout_center_id=task.heldout_center_id,
        heldout_menus=task.prediction_menus,
        fixed_excluded_center_ids=(task.outer_target_id, task.heldout_center_id),
        effective_menus=task.fitting_menus,
        config=task.fit_config,
    )
    return FoldFitExecution(
        outer_target_id=task.outer_target_id,
        heldout_center_id=task.heldout_center_id,
        task_scope_hash=task.task_scope_hash,
        nested_fold=fold,
        worker_process_id=os.getpid(),
        cuda_visible_to_worker=False,
    )


def _fold_task(
    bundle: LabelFreeSourceCrossfitBundle,
    capability: FoldSourceLabelCapability,
    fit_config: PairwiseFitConfig,
    *,
    config: object,
    cache: object,
    source_label_loader: SourceLabelLoader,
) -> FoldFitTask:
    h = capability.outer_target_id
    q = capability.heldout_center_id
    fitting = tuple(row.menu for row in bundle.effective_surface.fitting_menus(h, q))
    prediction = tuple(row.menu for row in bundle.effective_surface.prediction_menus(h, q))
    baselines: list[tuple[str, LabelFreeActionBlock]] = []
    for r in CENTERS:
        if r in {h, q}:
            continue
        matches = tuple(
            row
            for row in bundle.physical_surface.blocks_for(h, q, r)
            if row.action.action_id == "B"
        )
        if len(matches) != 1:
            raise ProtocolError("HARP v12 isolated fold lacks its physical B block.")
        row = matches[0]
        baselines.append(
            (
                r,
                LabelFreeActionBlock(
                    surface_role="development",
                    outer_target_id=h,
                    query_center_id=r,
                    action_kind=ActionKind.B,
                    selected_source_id=None,
                    sample_ids=row.sample_ids,
                    case_ids=row.case_ids,
                    probabilities=row.probabilities,
                    seed_dispersion=row.seed_dispersion,
                ),
            )
        )
    return FoldFitTask(
        outer_target_id=h,
        heldout_center_id=q,
        config=config,
        cache=cache,
        source_label_loader=source_label_loader,
        label_capability=capability,
        baseline_blocks=tuple(baselines),
        fitting_menus=fitting,
        prediction_menus=prediction,
        fit_config=fit_config,
        label_capability_hash=capability.capability_hash,
        source_surface_receipt_hash=bundle.surface_receipt.receipt_hash,
        source_surface_hash=bundle.physical_surface.surface_hash,
        effective_adapter_hash=bundle.effective_surface.adapter_hash,
        prediction_surface_hash=capability.prediction_surface_hash,
        fitting_surface_hash=capability.fitting_surface_hash,
    )


def _join_scoped_worker_outcomes(
    task: FoldFitTask,
    labels: Sequence[HarpSourceLabelRow],
) -> tuple[SourceActionOutcome, ...]:
    """Compatibility façade for the isolated C-{H,q} label join."""

    return _join_scoped_worker_outcomes_impl(task, labels)


def _attach_prediction_outcomes(
    bundle: LabelFreeSourceCrossfitBundle,
    labels: Sequence[HarpSourceLabelRow],
) -> tuple[tuple[SourceActionOutcome, ...], tuple[EffectiveMenu, ...]]:
    """Compatibility façade for the aggregate sealed-prediction label join."""

    return _attach_prediction_outcomes_impl(bundle, labels)


def _fit_config(model: Mapping[str, object]) -> PairwiseFitConfig:
    try:
        pairwise = tuple(float(value) for value in model["pairwise_alpha_grid"])
        residual = tuple(float(value) for value in model["residual_alpha_grid"])
        acceptor = tuple(float(value) for value in model["acceptor_alpha_grid"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("HARP v12 crossfit fit configuration is malformed.") from exc
    if len(pairwise) != 1 or len(residual) != 1 or len(acceptor) != 1:
        raise ProtocolError("HARP v12 crossfit requires fixed predeclared regularization.")
    return PairwiseFitConfig(
        pairwise_alpha=pairwise[0],
        residual_alpha=residual[0],
        acceptor_alpha=acceptor[0],
    )


def _development_artifact(
    state: SourceDevelopmentState,
    *,
    config_hash: str,
    bundle: LabelFreeSourceCrossfitBundle,
    fold_seal_set: SourceCrossfitFoldSealSet,
    aggregate_capability: AggregateSourceLabelCapability,
) -> ArtifactValue:
    return _build_development_artifact(
        state,
        config_hash=config_hash,
        bundle=bundle,
        fold_seal_set=fold_seal_set,
        aggregate_capability=aggregate_capability,
    )


def _model_artifact(
    state: RouterFitState,
    *,
    development: ArtifactValue,
    config_hash: str,
    target_compatibility_hash: str,
    bundle: LabelFreeSourceCrossfitBundle,
    fold_seal_set: SourceCrossfitFoldSealSet,
    aggregate_capability: AggregateSourceLabelCapability,
) -> ArtifactValue:
    return _build_model_artifact(
        state,
        development=development,
        config_hash=config_hash,
        target_compatibility_hash=target_compatibility_hash,
        bundle=bundle,
        fold_seal_set=fold_seal_set,
        aggregate_capability=aggregate_capability,
    )


def _numeric_oof_arrays(state: RouterFitState) -> Mapping[str, np.ndarray]:
    """Compatibility façade for the extracted numeric OOF artifact arrays."""

    return _build_numeric_oof_arrays(state)


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = (
    "FoldFitExecution",
    "FoldFitTask",
    "LabelFreeSourceCrossfitBundle",
    "SourceCrossfitFitBundle",
    "assemble_source_crossfit_model",
    "build_source_crossfit_effective_artifact",
    "build_source_prelabel_prediction_artifact",
    "execute_isolated_fold_fits",
    "fit_and_seal_prelabel_source_folds",
    "materialize_label_free_source_crossfit",
    "source_fold_capability_seal_payload",
    "source_prelabel_q_prediction_seal_payload",
)
