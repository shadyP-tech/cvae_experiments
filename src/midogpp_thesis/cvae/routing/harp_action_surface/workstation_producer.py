"""Workstation plan and physical producer boundary for HARP Stage-60 menus.

The orchestration contract is deliberately independent of the action-surface
adapter.  It describes the only permitted physical execution topology and
passes primitive, serializable task DTOs to workers.  The checked-in physical
runtime validates the versioned label-blind frame cache, materializes frozen
source streams, and runs a disjoint CPU classifier phase.  Focused tests may
inject a predictor explicitly; the Stage-60 adapter never uses an injected
provider by default.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

from ...protocol import ProtocolError
from ...runtime.artifact_io import atomic_json, read_json, sha256_file
from ...runtime.harp_probability_menu import (
    DEVELOPMENT_SURFACE,
    EXACT_NINE_SEED_PAIRS,
    TARGET_SURFACE,
    HarpPredictionMenuSeal,
    build_all_development_actions,
    build_all_target_actions,
)
from ..harp_protocol.hashing import canonical_hash, require_sha256
from ..harp_stage60.config import HarpInputReadiness, HarpStage60Config
from ..harp_stage60.constants import ACTION_SURFACE
from .workstation_contracts import (
    HarpClassifierTask,
    HarpGenerationTask,
    HarpPrimitivePredictor,
    HarpWorkstationPlan,
)


PLAN_CHECKPOINT_MEMBER = "checkpoints/harp_workstation_plan_v2.json"
CACHE_INDEX_MEMBER = "manifests/cache_index.json"
CACHE_CONTENT_INDEX_MEMBER = "manifests/content_index.json"
CACHE_ROW_INDEX_MEMBER = "tables/row_index.csv"


def build_harp_workstation_plan(
    config: HarpStage60Config, readiness: HarpInputReadiness
) -> HarpWorkstationPlan:
    """Build the complete spawn task graph without importing torch."""

    _assert_runtime_contract(config)
    surface_kind = DEVELOPMENT_SURFACE if config.contract == ACTION_SURFACE else TARGET_SURFACE
    actions = (
        build_all_development_actions()
        if surface_kind == DEVELOPMENT_SURFACE
        else build_all_target_actions()
    )
    action_by_query: dict[tuple[str, str], tuple[object, ...]] = {}
    for action in actions:
        key = (action.outer_target_id, action.query_center_id)
        action_by_query.setdefault(key, tuple())
        action_by_query[key] = (*action_by_query[key], action)

    centers = tuple(str(value) for value in config.protocol["center_universe"])
    generation_tasks = tuple(
        HarpGenerationTask(
            ordinal=ordinal,
            source_center=source,
            training_seed=training_seed,
            generation_seeds=(17, 42, 101),
            device=("cuda:0", "cuda:1")[ordinal % 2],
            checkpoint_array_member=(
                f"checkpoints/source_streams/source_{source}_train_{training_seed}.npy"
            ),
            checkpoint_receipt_member=(
                f"checkpoints/source_streams/source_{source}_train_{training_seed}.json"
            ),
        )
        for ordinal, (source, training_seed) in enumerate(
            (pair for source in centers for pair in ((source, 17), (source, 42), (source, 101)))
        )
    )
    classifier_tasks: list[HarpClassifierTask] = []
    for (outer, query), query_actions in action_by_query.items():
        for training_seed, generation_seed in EXACT_NINE_SEED_PAIRS:
            ordinal = len(classifier_tasks)
            stem = (
                f"{surface_kind}_H_{outer}_q_{query}_train_{training_seed}"
                f"_generation_{generation_seed}"
            )
            classifier_tasks.append(
                HarpClassifierTask(
                    ordinal=ordinal,
                    surface_kind=surface_kind,
                    outer_target_id=outer,
                    query_center_id=query,
                    training_seed=training_seed,
                    generation_seed=generation_seed,
                    action_hashes=tuple(action.action_hash for action in query_actions),
                    checkpoint_array_member=f"checkpoints/classifiers_v2/{stem}.npz",
                    checkpoint_receipt_member=f"checkpoints/classifiers_v2/{stem}.json",
                )
            )
    readiness_binding = canonical_hash(
        {
            "schema_version": "midogpp_harp_workstation_readiness_binding_v1",
            "surface": readiness.surface,
            "experiment_id": readiness.experiment_id,
            "input_binding_sha256": readiness.input_binding_sha256,
            "reservation_sha256": readiness.reservation_sha256,
            "cache_binding_sha256": readiness.cache_binding_sha256,
            "manifest_sha256": readiness.manifest_sha256,
            "attestation_sha256": readiness.attestation_sha256,
        }
    )
    return HarpWorkstationPlan(
        surface_kind=surface_kind,
        config_contract_hash=config.contract_hash,
        readiness_binding_hash=readiness_binding,
        generation_devices=("cuda:0", "cuda:1"),
        generation_tasks=generation_tasks,
        classifier_tasks=tuple(classifier_tasks),
        action_hashes=tuple(action.action_hash for action in actions),
    )


def materialize_harp_probability_menu(
    config: HarpStage60Config,
    readiness: HarpInputReadiness,
    *,
    primitive_predictor: HarpPrimitivePredictor | None = None,
    allow_test_provider: bool = False,
) -> HarpPredictionMenuSeal:
    """Materialize the neutral menu through the checked-in physical runtime.

    The injected provider switch is intentionally explicit and defaults to
    false, preventing an externally precomputed menu from becoming the
    workstation production path.
    """

    plan = build_harp_workstation_plan(config, readiness)
    _assert_parent_cuda_free()
    _validate_cache_catalog_boundary(config)
    if primitive_predictor is None:
        _persist_or_validate_plan_checkpoint(config.artifact_root, plan)
        from .workstation_runtime import materialize_physical_harp_menu

        return materialize_physical_harp_menu(config, readiness, plan)
    if not allow_test_provider:
        raise ProtocolError("HARP in-memory primitive predictors are test-only.")
    _persist_or_validate_plan_checkpoint(config.artifact_root, plan)
    menu = primitive_predictor.materialize(plan)
    if not isinstance(menu, HarpPredictionMenuSeal):
        raise ProtocolError("HARP primitive predictor returned an untyped menu.")
    menu.assert_valid()
    if tuple(action.action_hash for action in menu.actions) != plan.action_hashes:
        raise ProtocolError("HARP primitive predictor escaped the sealed task plan.")
    return menu


def _assert_runtime_contract(config: HarpStage60Config) -> None:
    runtime = config.runtime
    if (
        tuple(runtime.get("generation_devices", ())) != ("cuda:0", "cuda:1")
        or int(runtime.get("persistent_workers_per_gpu", -1)) != 1
        or int(runtime.get("classifier_workers", -1)) != 4
        or int(runtime.get("classifier_threads_per_worker", -1)) != 3
        or runtime.get("multiprocessing_start_method") != "spawn"
        or runtime.get("parent_cuda_context_forbidden") is not True
        or runtime.get("scientific_reductions_dtype") != "float64"
        or runtime.get("probability_transport_dtype") != "float32"
        or runtime.get("generated_cache_storage") != "float32_memmap"
        or runtime.get("nested_process_pools_allowed") is not False
    ):
        raise ProtocolError("HARP workstation runtime contract drifted.")


def _assert_parent_cuda_free() -> None:
    torch_module = sys.modules.get("torch")
    cuda = None if torch_module is None else getattr(torch_module, "cuda", None)
    if cuda is not None and bool(cuda.is_initialized()):
        raise ProtocolError("HARP workstation parent must remain CUDA-free.")


def _validate_cache_catalog_boundary(config: HarpStage60Config) -> None:
    cache_root = config.input_paths[
        "development_cache_root"
        if config.contract == ACTION_SURFACE
        else "target_support_cache_root"
    ]
    required = (CACHE_INDEX_MEMBER, CACHE_CONTENT_INDEX_MEMBER, CACHE_ROW_INDEX_MEMBER)
    missing = tuple(member for member in required if not (cache_root / member).is_file())
    if missing:
        raise ProtocolError(f"HARP fresh cache is missing catalog members: {missing}.")
    content = read_json(cache_root / CACHE_CONTENT_INDEX_MEMBER)
    members = content.get("members", content.get("member_sha256"))
    if not isinstance(members, dict) and isinstance(content.get("files"), list):
        members = {
            str(item.get("path", "")): item.get("sha256")
            for item in content["files"]
            if isinstance(item, dict)
        }
    if not isinstance(members, dict) or not members:
        raise ProtocolError("HARP fresh cache content index has no closed member map.")
    for member, digest in members.items():
        if type(member) is not str or not member or Path(member).is_absolute() or ".." in Path(member).parts:
            raise ProtocolError("HARP fresh cache content path is unsafe.")
        require_sha256(digest, name="HARP fresh cache member hash")
        path = cache_root / member
        if not path.is_file() or sha256_file(path) != digest:
            raise ProtocolError("HARP fresh cache content member drifted.")
    # Admission establishes the cache schema before any physical worker is
    # launched; the runtime independently validates every row and shard.
    cache_index = read_json(cache_root / CACHE_INDEX_MEMBER)
    if cache_index.get("schema_version") != "midogpp_harp_label_blind_frame_cache_v1":
        raise ProtocolError(
            "HARP fresh cache has no supported label-blind frame schema "
            "(expected midogpp_harp_label_blind_frame_cache_v1)."
        )


def _persist_or_validate_plan_checkpoint(root: Path, plan: HarpWorkstationPlan) -> None:
    path = Path(root) / PLAN_CHECKPOINT_MEMBER
    payload = {**plan.to_payload(), "plan_hash": plan.plan_hash}
    if path.is_file():
        if read_json(path) != payload:
            raise ProtocolError("HARP workstation plan checkpoint drifted on restart.")
        return
    atomic_json(path, payload)


__all__ = (
    "HarpClassifierTask",
    "HarpGenerationTask",
    "HarpPrimitivePredictor",
    "HarpWorkstationPlan",
    "PLAN_CHECKPOINT_MEMBER",
    "build_harp_workstation_plan",
    "materialize_harp_probability_menu",
)
