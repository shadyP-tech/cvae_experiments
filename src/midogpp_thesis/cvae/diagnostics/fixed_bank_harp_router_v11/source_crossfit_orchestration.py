"""Fold-local source-crossfit orchestration for the executable HARP v11 run.

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
from types import MappingProxyType

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...routing.harp_protocol import HarpSourceLabelRow, canonical_hash
from ...routing.policy_calibrated_residual_router_v11 import (
    EffectiveMenu,
    NestedPolicyFold,
    PairwiseFitConfig,
    SourceActionOutcome,
    assemble_source_lodo_result,
    fit_prelabel_pseudo_target_fold,
)
from ...runtime.harp_v11_execution.contracts import ArtifactValue
from ...runtime.harp_v11_execution.contracts import (
    ActionKind,
    LabelFreeActionBlock,
)
from ...runtime.harp_v11_execution.crossfit_contracts import (
    FoldConditionedSourceSurface,
)
from ...runtime.harp_v11_execution.crossfit_durability import (
    SourceCrossfitSurfaceReceipt,
    persist_source_crossfit_surface,
    reconstruct_source_crossfit_surface,
)
from ...runtime.harp_v11_execution.crossfit_effective_menus import (
    FoldConditionedEffectiveSurface,
)
from ...runtime.harp_v11_execution.directional_surfaces import attach_source_outcomes
from ...runtime.harp_v11_execution.model_adapter import (
    OuterRouterBundle,
    RouterFitState,
    model_manifest,
)
from ...runtime.harp_v11_execution.science_pool import initialize_science_worker
from ...runtime.harp_v11_execution.source_development import SourceDevelopmentState
from .source_crossfit_fold_store import (
    SourceCrossfitFoldSeal,
    SourceCrossfitFoldSealSet,
    persist_source_crossfit_fold,
    persist_source_crossfit_fold_set,
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
            raise ProtocolError("HARP v11 label-free source crossfit bundle is unbound.")
        object.__setattr__(
            self,
            "bundle_hash",
            canonical_hash(
                {
                    "schema_version": "midogpp_harp_v11_label_free_source_crossfit_bundle_v1",
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
            raise ProtocolError("HARP v11 isolated fold task escaped C-{H,q}.")
        bindings = (
            self.label_capability_hash,
            self.source_surface_receipt_hash,
            self.source_surface_hash,
            self.effective_adapter_hash,
            self.prediction_surface_hash,
            self.fitting_surface_hash,
        )
        if any(not _is_sha256(value) for value in bindings):
            raise ProtocolError("HARP v11 isolated fold task binding is malformed.")
        body = {
            "schema_version": "midogpp_harp_v11_isolated_fold_fit_task_v1",
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
            raise ProtocolError("HARP v11 isolated fold execution receipt is malformed.")
        object.__setattr__(
            self,
            "isolation_receipt_hash",
            canonical_hash(
                {
                    "schema_version": "midogpp_harp_v11_isolated_fold_execution_v1",
                    "outer_target_id": self.outer_target_id,
                    "heldout_center_id": self.heldout_center_id,
                    "task_scope_hash": self.task_scope_hash,
                    "nested_fold_hash": self.nested_fold.fold_hash,
                    "worker_process_id": self.worker_process_id,
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
            raise ProtocolError("HARP v11 source crossfit fit bundle is malformed.")
        object.__setattr__(
            self,
            "orchestration_hash",
            canonical_hash(
                {
                    "schema_version": "midogpp_harp_v11_source_crossfit_fit_bundle_v1",
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
        raise ProtocolError("HARP v11 source-crossfit center universe drifted.")
    materialize = getattr(pipeline, "materialize_label_free_source_crossfit_surface", None)
    build_effective = getattr(
        pipeline, "build_label_free_source_crossfit_effective_surface", None
    )
    if not callable(materialize) or not callable(build_effective):
        raise ProtocolError("HARP v11 pipeline lacks the source-crossfit execution seam.")
    physical = materialize(
        config,
        cache,
        outer_targets=typed_centers,
        scratch_root=Path(scratch_root),
    )
    if not isinstance(physical, FoldConditionedSourceSurface):
        raise ProtocolError("HARP v11 pipeline returned an untyped crossfit surface.")
    written_receipt = persist_source_crossfit_surface(Path(durable_root), physical)
    reconstructed, receipt = reconstruct_source_crossfit_surface(
        Path(durable_root), expected_surface_hash=physical.surface_hash
    )
    if (
        receipt.receipt_hash != written_receipt.receipt_hash
        or reconstructed.surface_hash != physical.surface_hash
    ):
        raise ProtocolError(
            "HARP v11 durable source-crossfit reconstruction changed identity."
        )
    # Derive the label-free effective adapter from independently reconstructed
    # bytes.  The pre-persistence in-memory surface never authorizes a label.
    effective = build_effective(reconstructed)
    return LabelFreeSourceCrossfitBundle(reconstructed, receipt, effective)


def build_source_crossfit_effective_artifact(
    bundle: LabelFreeSourceCrossfitBundle,
) -> ArtifactValue:
    """Project fold-effective membership without duplicating physical bytes."""

    rows: list[dict[str, object]] = []
    features: list[tuple[float, ...]] = []
    offsets = [0]
    for wrapper in bundle.effective_surface.menus:
        menu = wrapper.menu
        for action in menu.actions:
            features.append(action.feature_values)
        offsets.append(len(features))
        rows.append(
            {
                "outer_target_id": wrapper.outer_target_id,
                "heldout_center_id": wrapper.heldout_center_id,
                "current_query_center_id": wrapper.current_query_center_id,
                "case_id": menu.case_id,
                "candidate_source_ids": list(wrapper.candidate_source_ids),
                "fold_menu_hash": wrapper.fold_menu_hash,
                "effective_menu_hash": menu.menu_hash,
                "physical_block_hashes": list(wrapper.physical_block_hashes),
                "compatibility_receipt_hashes": list(
                    wrapper.compatibility_receipt_hashes
                ),
                "action_ids": [action.action_id for action in menu.actions],
                "action_hashes": [action.action_hash for action in menu.actions],
                "prediction_fold": wrapper.prediction_fold,
            }
        )
    body = {
        "schema_version": "midogpp_harp_v11_source_crossfit_effective_menu_store_v1",
        "source_surface_hash": bundle.physical_surface.surface_hash,
        "source_surface_receipt_hash": bundle.surface_receipt.receipt_hash,
        "effective_adapter_hash": bundle.effective_surface.adapter_hash,
        "rows": rows,
        "fold_menu_count": len(rows),
        "effective_action_count": len(features),
        "physical_probability_bytes_duplicated": False,
        "labels_consumed": False,
    }
    width = len(features[0]) if features else 0
    return ArtifactValue(
        state=bundle.effective_surface,
        manifest={**body, "effective_menu_hash": canonical_hash(body)},
        arrays={
            "effective_action_features": np.asarray(features, dtype=np.float64).reshape(
                (-1, width)
            ),
            "effective_action_offsets": np.asarray(offsets, dtype=np.int64),
        },
    )


def build_source_prelabel_prediction_artifact(
    fold_seal_set: SourceCrossfitFoldSealSet,
) -> ArtifactValue:
    """Compact all durable heldout-q predictions into one catalogued store."""

    predictions = tuple(
        prediction
        for seal in fold_seal_set.fold_seals
        for prediction in seal.nested_fold.heldout_predictions
    )
    score_rows = [
        (
            score.pairwise_score,
            score.predicted_budget_gain,
            score.predicted_allocation_gain,
            score.predicted_total_gain,
            score.predicted_harm_probability,
            score.predicted_brier_delta,
            score.predicted_log_delta,
            score.acceptance_probability,
            float(score.model_available),
        )
        for prediction in predictions
        for score in prediction.action_scores
    ]
    offsets = [0]
    for prediction in predictions:
        offsets.append(offsets[-1] + len(prediction.action_scores))
    body = {
        "schema_version": "midogpp_harp_v11_source_prelabel_q_prediction_store_v1",
        "source_surface_receipt_hash": fold_seal_set.source_surface_receipt_hash,
        "source_surface_hash": fold_seal_set.source_surface_hash,
        "effective_adapter_hash": fold_seal_set.effective_adapter_hash,
        "fold_seal_set_hash": fold_seal_set.seal_set_hash,
        "fold_seal_hashes": [row.seal_hash for row in fold_seal_set.fold_seals],
        "prediction_rows": [row.public_payload() for row in predictions],
        "prediction_count": len(predictions),
        "heldout_q_outcomes_consumed_by_own_fold": False,
        "aggregate_source_labels_opened": False,
        "evaluation_labels_used": False,
    }
    return ArtifactValue(
        state=fold_seal_set,
        manifest={**body, "prediction_store_hash": canonical_hash(body)},
        arrays={
            "prediction_values": np.asarray(
                [
                    (
                        row.acceptance_probability,
                        row.rank_margin,
                        float(len(row.action_scores)),
                        float(row.top_action_id != "B"),
                    )
                    for row in predictions
                ],
                dtype=np.float64,
            ).reshape((-1, 4)),
            "action_score_values": np.asarray(score_rows, dtype=np.float64).reshape(
                (-1, 9)
            ),
            "action_score_offsets": np.asarray(offsets, dtype=np.int64),
        },
    )


def source_fold_capability_seal_payload(
    fold_seal_set: SourceCrossfitFoldSealSet,
) -> dict[str, object]:
    body = {
        "schema_version": "midogpp_harp_v11_source_fold_label_capability_seals_v1",
        "source_surface_receipt_hash": fold_seal_set.source_surface_receipt_hash,
        "source_surface_hash": fold_seal_set.source_surface_hash,
        "effective_adapter_hash": fold_seal_set.effective_adapter_hash,
        "folds": [
            {
                "outer_target_id": seal.outer_target_id,
                "heldout_center_id": seal.heldout_center_id,
                "allowed_center_ids": [
                    center
                    for center in CENTERS
                    if center not in {seal.outer_target_id, seal.heldout_center_id}
                ],
                "excluded_center_ids": [
                    seal.outer_target_id,
                    seal.heldout_center_id,
                ],
                "label_capability_hash": seal.label_capability_hash,
                "prediction_surface_hash": seal.prediction_surface_hash,
                "fitting_surface_hash": seal.fitting_surface_hash,
                "isolation_receipt_hash": seal.isolation_receipt_hash,
            }
            for seal in fold_seal_set.fold_seals
        ],
        "fold_count": len(fold_seal_set.fold_seals),
        "heldout_q_label_shard_unauthorized_and_not_opened_by_typed_loader_in_own_H_q_worker": True,
        "global_source_label_open_order_claimed": False,
        "evaluation_labels_authorized": False,
    }
    return {**body, "seal_hash": canonical_hash(body)}


def source_prelabel_q_prediction_seal_payload(
    fold_seal_set: SourceCrossfitFoldSealSet,
    prediction_artifact: ArtifactValue,
    *,
    store_manifest_sha256: str,
    store_npz_sha256: str,
) -> dict[str, object]:
    body = {
        "schema_version": "midogpp_harp_v11_source_prelabel_q_prediction_seal_v1",
        "source_surface_receipt_hash": fold_seal_set.source_surface_receipt_hash,
        "source_surface_hash": fold_seal_set.source_surface_hash,
        "effective_adapter_hash": fold_seal_set.effective_adapter_hash,
        "fold_seal_set_hash": fold_seal_set.seal_set_hash,
        "prediction_store_hash": prediction_artifact.manifest[
            "prediction_store_hash"
        ],
        "prediction_store_manifest_sha256": store_manifest_sha256,
        "prediction_store_npz_sha256": store_npz_sha256,
        "fold_seal_hashes": [row.seal_hash for row in fold_seal_set.fold_seals],
        "pseudo_target_q_predictions_sealed_before_q_outcomes_joined_to_same_fold": True,
        "aggregate_source_labels_opened": False,
        "evaluation_labels_opened": False,
    }
    return {**body, "seal_hash": canonical_hash(body)}


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
            raise ProtocolError("HARP v11 isolated fold executor returned incomplete coverage.")
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
        raise ProtocolError("HARP v11 isolated fold executor contract drifted.")
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
                    raise ProtocolError("HARP v11 isolated fold result crossed tasks.")
                by_key[key] = result
                replacement = next(iterator, None)
                if replacement is not None:
                    pending[pool.submit(_fit_fold_worker, replacement)] = replacement
    expected = {(task.outer_target_id, task.heldout_center_id) for task in typed}
    if set(by_key) != expected:
        raise ProtocolError("HARP v11 isolated fold execution is incomplete.")
    return tuple(by_key[(task.outer_target_id, task.heldout_center_id)] for task in typed)


def _fit_fold_worker(task: FoldFitTask) -> FoldFitExecution:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ProtocolError("HARP v11 source fold worker can see CUDA devices.")
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
            raise ProtocolError("HARP v11 isolated fold lacks its physical B block.")
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
    """Join C-{H,q} labels inside the one-task child and nowhere else."""

    allowed = tuple(
        center
        for center in CENTERS
        if center not in {task.outer_target_id, task.heldout_center_id}
    )
    rows = tuple(labels)
    if (
        not rows
        or any(not isinstance(row, HarpSourceLabelRow) for row in rows)
        or {row.center for row in rows} != set(allowed)
        or any(row.center in {task.outer_target_id, task.heldout_center_id} for row in rows)
    ):
        raise ProtocolError("HARP v11 isolated worker labels escaped C-{H,q}.")
    label_index = {row.row_key: row.label for row in rows}
    if len(label_index) != len(rows):
        raise ProtocolError("HARP v11 isolated worker labels duplicate an identity.")
    expected_keys: set[tuple[str, str, str]] = set()
    outcomes: list[SourceActionOutcome] = []
    for query, baseline in task.baseline_blocks:
        keys = {
            (query, case, sample)
            for case, sample in zip(baseline.case_ids, baseline.sample_ids, strict=True)
        }
        expected_keys.update(keys)
        try:
            scoped = {
                (case, sample): label_index[(query, case, sample)]
                for _, case, sample in keys
            }
        except KeyError as exc:
            raise ProtocolError("HARP v11 isolated worker labels omit a physical row.") from exc
        menus = tuple(row for row in task.fitting_menus if row.query_center_id == query)
        outcomes.extend(attach_source_outcomes(menus, baseline, source_labels=scoped))
    if set(label_index) != expected_keys:
        raise ProtocolError("HARP v11 isolated worker label scope exceeds its surface.")
    return tuple(outcomes)


def _attach_prediction_outcomes(
    bundle: LabelFreeSourceCrossfitBundle,
    labels: Sequence[HarpSourceLabelRow],
) -> tuple[tuple[SourceActionOutcome, ...], tuple[EffectiveMenu, ...]]:
    by_center = {center: tuple(row for row in labels if row.center == center) for center in CENTERS}
    output: list[SourceActionOutcome] = []
    menus: list[EffectiveMenu] = []
    for h in CENTERS:
        for q in CENTERS:
            if q == h:
                continue
            matches = tuple(
                row
                for row in bundle.physical_surface.blocks_for(h, q, q)
                if row.action.action_id == "B"
            )
            if len(matches) != 1:
                raise ProtocolError("HARP v11 aggregate q join lacks exact physical B.")
            raw = matches[0]
            baseline = LabelFreeActionBlock(
                surface_role="development",
                outer_target_id=h,
                query_center_id=q,
                action_kind=ActionKind.B,
                selected_source_id=None,
                sample_ids=raw.sample_ids,
                case_ids=raw.case_ids,
                probabilities=raw.probabilities,
                seed_dispersion=raw.seed_dispersion,
            )
            q_rows = by_center[q]
            label_index = {row.row_key: row.label for row in q_rows}
            expected = {
                (q, case, sample)
                for case, sample in zip(
                    baseline.case_ids, baseline.sample_ids, strict=True
                )
            }
            if set(label_index) != expected:
                raise ProtocolError(
                    "HARP v11 aggregate q labels exceed or omit the prediction fold."
                )
            scoped = {
                (case, sample): label_index[(q, case, sample)]
                for _, case, sample in expected
            }
            q_menus = tuple(
                row.menu for row in bundle.effective_surface.prediction_menus(h, q)
            )
            output.extend(attach_source_outcomes(q_menus, baseline, source_labels=scoped))
            menus.extend(q_menus)
    return tuple(output), tuple(menus)


def _fit_config(model: Mapping[str, object]) -> PairwiseFitConfig:
    try:
        pairwise = tuple(float(value) for value in model["pairwise_alpha_grid"])
        residual = tuple(float(value) for value in model["residual_alpha_grid"])
        acceptor = tuple(float(value) for value in model["acceptor_alpha_grid"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("HARP v11 crossfit fit configuration is malformed.") from exc
    if len(pairwise) != 1 or len(residual) != 1 or len(acceptor) != 1:
        raise ProtocolError("HARP v11 crossfit requires fixed predeclared regularization.")
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
    outcomes = state.outcomes
    body = {
        "schema_version": "midogpp_harp_v11_source_train_crossfit_development_surface_v1",
        "config_hash": config_hash,
        "outer_targets": list(CENTERS),
        "expected_center_ids": list(CENTERS),
        "source_surface_hash": bundle.physical_surface.surface_hash,
        "source_surface_receipt_hash": bundle.surface_receipt.receipt_hash,
        "effective_adapter_hash": bundle.effective_surface.adapter_hash,
        "fold_seal_set_hash": fold_seal_set.seal_set_hash,
        "aggregate_source_label_capability_hash": aggregate_capability.capability_hash,
        "observation_count": len(outcomes),
        "effective_menu_count": len(state.effective_menus),
        "source_response_hashes": [row.outcome_hash for row in outcomes],
        "effective_menu_hashes": [row.menu_hash for row in state.effective_menus],
        "strict_outer_H_and_fold_local_q_exclusion": True,
        "q_predictions_presealed_before_same_q_outcomes_joined": True,
        "evaluation_labels_used": False,
    }
    return ArtifactValue(
        state=state,
        manifest={**body, "surface_hash": canonical_hash(body)},
        arrays={
            "feature_values": np.asarray(
                [row.action.feature_values for row in outcomes], dtype=np.float64
            ),
            "endpoint_effects": np.asarray(
                [(row.bacc_gain, row.brier_delta, row.log_delta) for row in outcomes],
                dtype=np.float64,
            ),
        },
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
    body = {
        **model_manifest(state),
        "development_surface_hash": development.manifest["surface_hash"],
        "compatibility_hash": target_compatibility_hash,
        "config_hash": config_hash,
        "expected_center_ids": list(CENTERS),
        "source_surface_hash": bundle.physical_surface.surface_hash,
        "source_surface_receipt_hash": bundle.surface_receipt.receipt_hash,
        "effective_adapter_hash": bundle.effective_surface.adapter_hash,
        "fold_seal_set_hash": fold_seal_set.seal_set_hash,
        "aggregate_source_label_capability_hash": aggregate_capability.capability_hash,
        "legacy_fit_source_lodo_used": False,
        "presealed_fold_assembly_only": True,
        "all_preprocessing_fit_inside_source_lodo": True,
        "evaluation_labels_used": False,
    }
    return ArtifactValue(
        state=state,
        manifest={**body, "model_hash": canonical_hash(body)},
        arrays=_numeric_oof_arrays(state),
    )


def _numeric_oof_arrays(state: RouterFitState) -> Mapping[str, np.ndarray]:
    case_rows: list[tuple[float, ...]] = []
    score_rows: list[tuple[float, ...]] = []
    offsets = [0]
    for bundle in state.bundles:
        for prediction in bundle.lodo.oof_predictions:
            case_rows.append(
                (
                    prediction.acceptance_probability,
                    prediction.rank_margin,
                    float(len(prediction.action_scores)),
                    float(prediction.top_action_id != "B"),
                )
            )
            score_rows.extend(
                (
                    row.pairwise_score,
                    row.predicted_budget_gain,
                    row.predicted_allocation_gain,
                    row.predicted_total_gain,
                    row.predicted_harm_probability,
                    row.predicted_brier_delta,
                    row.predicted_log_delta,
                    row.acceptance_probability,
                    float(row.model_available),
                )
                for row in prediction.action_scores
            )
            offsets.append(len(score_rows))
    return MappingProxyType(
        {
            "oof_case_values": np.asarray(case_rows, dtype=np.float64).reshape((-1, 4)),
            "oof_action_scores": np.asarray(score_rows, dtype=np.float64).reshape((-1, 9)),
            "oof_action_score_offsets": np.asarray(offsets, dtype=np.int64),
        }
    )


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
