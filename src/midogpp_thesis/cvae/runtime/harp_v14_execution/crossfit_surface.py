"""Workstation production of the HARP v14 fold-conditioned source surface.

The resident expert stream and staged source-train frames are shared with the
terminal menu.  CPU work is grouped by ``(H, q, r, training seed, generation
seed)`` so one fitted classifier worker receives the complete B/U/Hxe action
slate for that fold cell.  The four one-process, three-BLAS-thread executors
remain phase-disjoint from CUDA generation and use the existing bounded task
queues and per-task checkpoints.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math
from pathlib import Path
from types import MappingProxyType

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from ...runtime.artifact_io import atomic_json, atomic_npy, read_json, sha256_file
from .action_capacity import (
    build_action_capacity_certificate,
    validate_action_capacity,
)
from .classifier_tasks import load_classifier_task_checkpoint
from .crossfit_actions import (
    EXACT_NINE_SEED_PAIRS,
    FoldConditionedActionSpec,
    build_fold_conditioned_action_menu,
    fold_conditioned_action_from_payload,
    six_source_geometry_audit,
)
from .crossfit_contracts import (
    FoldConditionedActionBlock,
    FoldConditionedCompatibility,
    FoldConditionedSourceSurface,
)
from .contracts import (
    ActionKind,
    LabelFreeActionBlock,
    LabelFreeOuterMenu,
    LabelFreeTargetMenu,
)
from .execution_profile import WorkstationProfile
from .gpu_surface import ResidentExpertStreamCache, materialize_resident_expert_streams
from .geometry_features import geometry_feature_audit
from .hash_contracts import require_sha256, require_stable_hash
from .physical import (
    _SourceAdapter,
    _build_tasks as _unused_standard_task_builder,
    _execute_tasks,
    _stage_frames,
    _support_binding,
    validate_physical_inputs,
)
from .physical_contracts import PhysicalInputReceipt, StagedFrames
from .projection_hashing import projection_semantic_hash
from .task_bindings import validate_frame_task_binding, validate_source_task_binding


_TASK_SCHEMA = "midogpp_harp_v14_fold_conditioned_classifier_task_v1"
_SURFACE_MANIFEST_SCHEMA = "midogpp_harp_v14_fold_conditioned_surface_manifest_v1"


def fold_conditioned_physical_plan(
    outer_targets: Sequence[str] = CENTERS,
) -> Mapping[str, object]:
    requested = _outer_targets(outer_targets)
    capacity = dict(build_action_capacity_certificate(centers=requested))
    prediction_contexts = len(requested) * 8
    calibration_contexts = len(requested) * 8 * 7
    context_count = prediction_contexts + calibration_contexts
    action_count = prediction_contexts * 9 + calibration_contexts * 8
    body = {
        "schema_version": "midogpp_harp_v14_fold_conditioned_physical_plan_v1",
        "outer_target_ids": list(requested),
        "identity_axes": ["outer_H", "heldout_q", "current_query_r"],
        "prediction_context_count": prediction_contexts,
        "calibration_context_count": calibration_contexts,
        "context_count": context_count,
        "action_count": action_count,
        "classifier_task_count": context_count * len(EXACT_NINE_SEED_PAIRS),
        "seed_cell_count": action_count * len(EXACT_NINE_SEED_PAIRS),
        "seed_pairs": [list(value) for value in EXACT_NINE_SEED_PAIRS],
        "seed_cells_are_technical_replications": True,
        "seed_selection_performed": False,
        "prediction_source_pool": "C_MINUS_H_MINUS_Q",
        "calibration_source_pool": "C_MINUS_H_MINUS_Q_MINUS_R",
        "six_source_geometry_audit": dict(six_source_geometry_audit()),
        "classifier_task_grouping": "one_H_q_r_seed_pair_complete_action_slate",
        "resident_expert_stream_cache_reused": True,
        "staged_source_train_frames_reused": True,
        "action_capacity_certificate_hash": capacity[
            "capacity_certificate_hash"
        ],
        "stream_rows_per_class": capacity["stream_rows_per_class"],
        "maximum_required_rows_per_class": capacity[
            "global_maximum_required_rows_per_class"
        ],
        "action_capacity_validated_before_scheduling": True,
        "labels_consumed": False,
    }
    return MappingProxyType({**body, "plan_hash": canonical_hash(body)})


def materialize_fold_conditioned_source_surface(
    config: object,
    cache: object,
    *,
    outer_targets: Sequence[str],
    scratch_root: Path,
    source_role: str,
    evaluation_role: str,
) -> FoldConditionedSourceSurface:
    """Materialize every H/q/r source fold before source labels can open."""

    requested = _outer_targets(outer_targets)
    # This is deliberately before scratch creation, frame staging, source
    # generation, or classifier scheduling.
    build_action_capacity_certificate(centers=requested)
    inputs = validate_physical_inputs(config, cache)
    workstation = WorkstationProfile.from_runtime(getattr(config, "runtime"))
    root = Path(scratch_root)
    root.mkdir(parents=True, exist_ok=True)
    frames = _stage_frames(
        config,
        cache,
        inputs=inputs,
        scratch_root=root,
        roles=(str(source_role), str(evaluation_role)),
    )
    source_adapter = _SourceAdapter(
        contract_hash=canonical_hash(
            {
                "schema_version": "midogpp_harp_v14_source_runtime_binding_v1",
                "config_hash": config.config_hash,
                "physical_input_receipt_hash": inputs.receipt_hash,
                "frame_sha256": frames.sha256,
                "frame_provenance_hash": frames.provenance_hash,
                "workstation_profile_hash": workstation.profile_hash,
            }
        ),
        expert_bank_root=config.resolved_path("expert_bank_root"),
        runtime=workstation.source_runtime(),
    )
    support = _support_binding(
        config,
        cache,
        frames=frames,
        development_role=str(source_role),
        evaluation_role=str(evaluation_role),
    )
    source_cache = materialize_resident_expert_streams(
        source_adapter,
        inputs.generation_lock,
        root=root / "source_streams",
        support_binding=support,
    )
    tasks = build_fold_conditioned_classifier_tasks(
        scratch_root=root,
        frames=frames,
        source_cache=source_cache,
        inputs=inputs,
        workstation=workstation,
        source_role=str(source_role),
        outer_targets=requested,
    )
    completed = _execute_tasks(tasks, workstation=workstation)
    blocks = _aggregate_action_blocks(tasks, completed)
    compatibility = build_fold_conditioned_compatibility(
        source_cache.compatibility_payload,
        outer_targets=requested,
    )
    lineage = {
        "physical_input_receipt_hash": inputs.receipt_hash,
        "bank_hash": inputs.bank_hash,
        "generation_hash": inputs.generation_hash,
        "cache_hash": str(getattr(cache, "cache_hash")),
        "frame_array_sha256": frames.sha256,
        "frame_provenance_hash": frames.provenance_hash,
        "source_stream_lock_hash": source_cache.lock_hash,
        "source_stream_index_hash": source_cache.index_hash,
        "compatibility_hash": str(
            source_cache.compatibility_payload.get("compatibility_hash")
        ),
        "workstation_profile_hash": workstation.profile_hash,
        "source_role": str(source_role),
        "physical_plan_hash": str(
            fold_conditioned_physical_plan(requested)["plan_hash"]
        ),
        "source_record_projection_hashes": sorted(
            {
                str(task["source_record_projection_hash"])
                for task in tasks
            }
        ),
        "query_frame_projection_hashes": sorted(
            {str(task["frame_projection_hash"]) for task in tasks}
        ),
        "capability_scoped_source_projection": True,
        "excluded_source_offsets_worker_unreachable": True,
        "query_only_frame_shards": True,
        "shared_geometry_feature_hash": str(
            geometry_feature_audit()["geometry_feature_hash"]
        ),
        "heldout_q_physically_excluded": True,
        "all_transforms_fold_conditioned": True,
    }
    surface = FoldConditionedSourceSurface(
        outer_target_ids=requested,
        blocks=blocks,
        compatibility=compatibility,
        lineage=lineage,
    )
    # Import here to keep durable-store reconstruction independent of the
    # physical worker import graph used by multiprocessing spawn.
    from .crossfit_durability import persist_source_crossfit_surface

    persist_source_crossfit_surface(root / "source_crossfit_surface", surface)
    return surface


def build_fold_conditioned_classifier_tasks(
    *,
    scratch_root: Path,
    frames: StagedFrames,
    source_cache: ResidentExpertStreamCache,
    inputs: PhysicalInputReceipt,
    workstation: WorkstationProfile,
    source_role: str,
    outer_targets: Sequence[str],
) -> tuple[dict[str, object], ...]:
    """Build grouped H/q/r tasks over shared immutable source/frame stores."""

    requested = _outer_targets(outer_targets)
    source_binding = validate_source_task_binding(source_cache)
    checkpoint_root = Path(scratch_root) / "source_crossfit_classifier_checkpoints"
    frame_projections = {
        query: _persist_query_frame_projection(
            Path(scratch_root) / "source_crossfit_frame_projections",
            frames=frames,
            role=source_role,
            query_center_id=query,
        )
        for query in CENTERS
    }
    tasks: list[dict[str, object]] = []
    ordinal = 0
    for outer in requested:
        for heldout in CENTERS:
            if heldout == outer:
                continue
            for query in CENTERS:
                if query == outer:
                    continue
                actions = build_fold_conditioned_action_menu(outer, heldout, query)
                validate_action_capacity(actions)
                allowed_sources = tuple(
                    center for center in CENTERS if center not in {outer, heldout, query}
                )
                projection = _persist_source_record_projection(
                    Path(scratch_root) / "source_crossfit_source_projections",
                    source_binding=source_binding,
                    outer_target_id=outer,
                    heldout_center_id=heldout,
                    current_query_center_id=query,
                    allowed_source_ids=allowed_sources,
                )
                frame_projection = frame_projections[query]
                for training_seed, generation_seed in EXACT_NINE_SEED_PAIRS:
                    stem = (
                        f"H{outer}_q{heldout}_r{query}_"
                        f"t{training_seed}_g{generation_seed}"
                    )
                    body = {
                        "schema_version": _TASK_SCHEMA,
                        "ordinal": ordinal,
                        "surface_kind": "source_crossfit",
                        "outer_target_id": outer,
                        "heldout_center_id": heldout,
                        "current_query_center_id": query,
                        # Worker cache still uses this neutral evaluated-frame alias.
                        "query_center_id": query,
                        "training_seed": training_seed,
                        "generation_seed": generation_seed,
                        "actions": [row.to_payload() for row in actions],
                        "source_pool_semantics": "C_MINUS_H_MINUS_Q_MINUS_R",
                        "heldout_q_physically_excluded": True,
                        "six_source_geometry_audit": (
                            dict(six_source_geometry_audit())
                            if query != heldout
                            else None
                        ),
                        "source_array_path": str(source_binding.array_path),
                        "source_array_sha256": source_binding.array_sha256,
                        "source_index_path": str(projection["path"]),
                        "source_index_sha256": projection["sha256"],
                        "source_stream_index_hash": projection["projection_hash"],
                        "source_records": list(projection["records"]),
                        "source_record_projection_schema": (
                            "midogpp_harp_v14_fold_source_record_projection_v1"
                        ),
                        "source_record_projection_hash": projection[
                            "projection_hash"
                        ],
                        "full_source_stream_index_hash": source_binding.index_hash,
                        "allowed_source_ids": list(allowed_sources),
                        "frame_array_path": str(frame_projection["path"]),
                        "frame_array_sha256": frame_projection["sha256"],
                        "frame_receipt_path": str(frame_projection["receipt_path"]),
                        "frame_receipt_hash": frame_projection["receipt_hash"],
                        "frame_receipt_sha256": frame_projection["receipt_sha256"],
                        "frame_projection_schema": (
                            "midogpp_harp_v14_query_frame_projection_v1"
                        ),
                        "frame_projection_hash": frame_projection["receipt_hash"],
                        "frame_start": 0,
                        "frame_stop": len(frame_projection["sample_ids"]),
                        "sample_ids": list(frames.sample_ids[(source_role, query)]),
                        "case_ids": list(frames.case_ids[(source_role, query)]),
                        "generation_lock_hash": inputs.generation_hash,
                        "bank_hash": inputs.bank_hash,
                        "source_stream_lock_hash": source_binding.lock_hash,
                        "source_stream_lock_sha256": source_binding.lock_sha256,
                        "classifier": inputs.classifier.to_payload(),
                        "threads_per_worker": workstation.blas_threads_per_worker,
                        "workstation_profile_hash": workstation.profile_hash,
                        "labels_available": False,
                    }
                    tasks.append(
                        {
                            **body,
                            "task_hash": canonical_hash(body),
                            "npz_path": str(checkpoint_root / f"{stem}.npz"),
                            "receipt_path": str(checkpoint_root / f"{stem}.json"),
                        }
                    )
                    ordinal += 1
    expected = len(requested) * 8 * 8 * len(EXACT_NINE_SEED_PAIRS)
    if len(tasks) != expected:
        raise ProtocolError("HARP v14 fold-conditioned task coverage drifted.")
    return tuple(tasks)


def _persist_source_record_projection(
    root: Path,
    *,
    source_binding: object,
    outer_target_id: str,
    heldout_center_id: str,
    current_query_center_id: str,
    allowed_source_ids: tuple[str, ...],
) -> Mapping[str, object]:
    """Seal the only stream-record offsets visible to one H/q/r worker."""

    records = tuple(
        row
        for row in source_binding.records
        if str(row.get("source_center")) in set(allowed_source_ids)
    )
    expected = len(allowed_source_ids) * len(EXACT_NINE_SEED_PAIRS)
    if len(records) != expected or {
        str(row.get("source_center")) for row in records
    } != set(allowed_source_ids):
        raise ProtocolError("HARP v14 source projection record coverage drifted.")
    body = {
        "schema_version": "midogpp_harp_v14_fold_source_record_projection_v1",
        "outer_target_id": outer_target_id,
        "heldout_center_id": heldout_center_id,
        "current_query_center_id": current_query_center_id,
        "allowed_source_ids": list(allowed_source_ids),
        "source_pool_semantics": "C_MINUS_H_MINUS_Q_MINUS_R",
        "full_source_stream_index_hash": source_binding.index_hash,
        "full_source_stream_index_sha256": source_binding.index_sha256,
        "records": list(records),
        "excluded_record_offsets_visible": False,
        "labels_consumed": False,
    }
    payload = {
        **body,
        "source_record_projection_hash": projection_semantic_hash(
            body, name="source-record projection hash"
        ),
    }
    root.mkdir(parents=True, exist_ok=True)
    path = root / (
        f"H{outer_target_id}_q{heldout_center_id}_r{current_query_center_id}.json"
    )
    if path.exists():
        if read_json(path) != payload:
            raise ProtocolError("HARP v14 source record projection drifted.")
    else:
        atomic_json(path, payload)
    return MappingProxyType(
        {
            "path": path.resolve(),
            "sha256": sha256_file(path),
            "projection_hash": payload["source_record_projection_hash"],
            "records": records,
        }
    )


def _persist_query_frame_projection(
    root: Path,
    *,
    frames: StagedFrames,
    role: str,
    query_center_id: str,
) -> Mapping[str, object]:
    """Persist a query-only frame so no excluded center rows are worker-visible."""

    start, stop = frames.contexts[(role, query_center_id)]
    source = np.load(frames.path, mmap_mode="r", allow_pickle=False)
    values = np.ascontiguousarray(source[start:stop], dtype=np.float32)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{role}_{query_center_id}.npy"
    receipt_path = root / f"{role}_{query_center_id}.json"
    if path.exists():
        observed = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            observed.dtype != np.float32
            or observed.shape != values.shape
            or not np.array_equal(observed, values)
        ):
            raise ProtocolError("HARP v14 query frame projection drifted.")
    else:
        atomic_npy(path, values)
    body = {
        "schema_version": "midogpp_harp_v14_query_frame_projection_v1",
        "parent_frame_sha256": frames.sha256,
        "parent_frame_receipt_hash": frames.receipt_hash,
        "role": role,
        "query_center_id": query_center_id,
        "sample_ids": list(frames.sample_ids[(role, query_center_id)]),
        "case_ids": list(frames.case_ids[(role, query_center_id)]),
        "shape": list(values.shape),
        "dtype": "float32",
        "array_sha256": sha256_file(path),
        "other_center_rows_visible": False,
        "labels_consumed": False,
    }
    receipt = {
        **body,
        "frame_projection_hash": projection_semantic_hash(
            body, name="frame projection hash"
        ),
    }
    if receipt_path.exists():
        if read_json(receipt_path) != receipt:
            raise ProtocolError("HARP v14 query frame receipt drifted.")
    else:
        atomic_json(receipt_path, receipt)
    return MappingProxyType(
        {
            "path": path.resolve(),
            "sha256": body["array_sha256"],
            "receipt_path": receipt_path.resolve(),
            "receipt_hash": receipt["frame_projection_hash"],
            "receipt_sha256": sha256_file(receipt_path),
            "sample_ids": tuple(body["sample_ids"]),
            "case_ids": tuple(body["case_ids"]),
        }
    )


def _aggregate_action_blocks(
    tasks: Sequence[Mapping[str, object]],
    completed: Mapping[int, Mapping[str, object]],
) -> tuple[FoldConditionedActionBlock, ...]:
    cells: dict[str, list[np.ndarray]] = defaultdict(list)
    action_by_hash: dict[str, FoldConditionedActionSpec] = {}
    identities: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for task in tasks:
        checkpoint = load_classifier_task_checkpoint(task)
        prior = completed.get(int(task["ordinal"]))
        if (
            checkpoint is None
            or prior is None
            or checkpoint.get("checkpoint_hash") != prior.get("checkpoint_hash")
        ):
            raise ProtocolError("HARP v14 crossfit checkpoint changed before aggregation.")
        with np.load(Path(str(task["npz_path"])), allow_pickle=False) as archive:
            values = np.asarray(archive["probabilities"], dtype=np.float32)
        for index, raw in enumerate(task["actions"]):
            action = fold_conditioned_action_from_payload(raw)
            action_by_hash[action.action_hash] = action
            cells[action.action_hash].append(
                np.ascontiguousarray(values[index], dtype=np.float32)
            )
            identity = (
                tuple(str(value) for value in task["sample_ids"]),
                tuple(str(value) for value in task["case_ids"]),
            )
            previous = identities.setdefault(action.action_hash, identity)
            if previous != identity:
                raise ProtocolError("HARP v14 crossfit row identities drifted.")
    output: list[FoldConditionedActionBlock] = []
    for action in sorted(action_by_hash.values(), key=lambda row: row.key):
        members = cells[action.action_hash]
        if len(members) != len(EXACT_NINE_SEED_PAIRS):
            raise ProtocolError("HARP v14 crossfit action lacks all nine seed cells.")
        exact_nine = np.stack(members).astype(np.float64)
        sample_ids, case_ids = identities[action.action_hash]
        output.append(
            FoldConditionedActionBlock(
                action=action,
                sample_ids=sample_ids,
                case_ids=case_ids,
                probabilities=np.ascontiguousarray(
                    np.mean(exact_nine, axis=0, dtype=np.float64),
                    dtype=np.float32,
                ),
                seed_dispersion=np.ascontiguousarray(
                    np.std(exact_nine, axis=0, dtype=np.float64),
                    dtype=np.float32,
                ),
            )
        )
    return tuple(output)


def build_fold_conditioned_compatibility(
    raw: Mapping[str, object], *, outer_targets: Sequence[str]
) -> tuple[FoldConditionedCompatibility, ...]:
    """Compute case-local z-scores/ranks in each exact H/q/r pool.

    Query energies are never reduced to a center mean.  Every source-train
    query case is calibrated against the candidate expert's own-source train
    distribution, then candidates are reranked separately for that case.  It
    is therefore feature-isomorphic to target case-local compatibility while
    remaining label-free and physically excluding H/q/r candidates.
    """

    body = {key: value for key, value in raw.items() if key != "compatibility_hash"}
    replicas = raw.get("replicas")
    if (
        raw.get("schema_version")
        != "midogpp_harp_v14_role_qualified_compatibility_surface_v2"
        or raw.get("compatibility_hash") != canonical_hash(body)
        or not isinstance(replicas, list)
        or raw.get("labels_consumed") is not False
    ):
        raise ProtocolError("HARP v14 crossfit compatibility substrate drifted.")
    by_replica: dict[tuple[str, int], Mapping[str, object]] = {}
    binding = raw.get("support_binding")
    if not isinstance(binding, Mapping):
        raise ProtocolError("HARP v14 crossfit support binding is absent.")
    source_role = str(binding.get("source_role"))
    by_context: dict[tuple[str, int, str, str], Mapping[str, object]] = {}
    for replica in replicas:
        if not isinstance(replica, Mapping):
            raise ProtocolError("HARP v14 crossfit compatibility replica is malformed.")
        source = str(replica.get("source_center"))
        seed = int(replica.get("training_seed", -1))
        contexts = replica.get("contexts")
        if source not in CENTERS or seed not in (17, 42, 101) or not isinstance(contexts, list):
            raise ProtocolError("HARP v14 crossfit compatibility replica drifted.")
        by_replica[(source, seed)] = replica
        for context in contexts:
            if not isinstance(context, Mapping):
                raise ProtocolError("HARP v14 compatibility context is malformed.")
            query = str(context.get("query_center"))
            role = str(context.get("role"))
            by_context[(source, seed, role, query)] = context
    if set(by_replica) != {
        (source, seed) for source in CENTERS for seed in (17, 42, 101)
    }:
        raise ProtocolError("HARP v14 crossfit compatibility grid is incomplete.")

    output: list[FoldConditionedCompatibility] = []
    for outer in _outer_targets(outer_targets):
        for heldout in CENTERS:
            if heldout == outer:
                continue
            for query in CENTERS:
                if query == outer:
                    continue
                candidates = tuple(
                    source
                    for source in CENTERS
                    if source not in {outer, heldout, query}
                )
                by_candidate_case: dict[
                    str,
                    dict[
                        str,
                        tuple[
                            tuple[float, float, float],
                            float,
                            float,
                            tuple[str, str, str],
                        ],
                    ],
                ] = {}
                case_order: tuple[str, ...] | None = None
                for source in candidates:
                    scores_by_case: dict[str, list[float]] = {}
                    hashes: list[str] = []
                    for seed in (17, 42, 101):
                        query_context = by_context[(source, seed, source_role, query)]
                        own_context = by_context[(source, seed, source_role, source)]
                        location, scale = _robust_location_scale(
                            own_context.get("per_case_energy_float32")
                        )
                        raw_cases = query_context.get("case_order")
                        raw_energies = query_context.get("per_case_energy_float32")
                        if (
                            not isinstance(raw_cases, list)
                            or not isinstance(raw_energies, list)
                            or not raw_cases
                            or len(raw_cases) != len(raw_energies)
                        ):
                            raise ProtocolError(
                                "HARP v14 crossfit case-local compatibility rows drifted."
                            )
                        cases = tuple(str(value) for value in raw_cases)
                        energies = tuple(float(value) for value in raw_energies)
                        if len(set(cases)) != len(cases) or not all(
                            math.isfinite(value) for value in energies
                        ):
                            raise ProtocolError(
                                "HARP v14 crossfit compatibility cases are malformed."
                            )
                        if case_order is None:
                            case_order = cases
                        elif cases != case_order:
                            raise ProtocolError(
                                "HARP v14 crossfit compatibility case order differs by replica."
                            )
                        for case_id, energy in zip(cases, energies, strict=True):
                            score = (energy - location) / scale
                            if not math.isfinite(score):
                                raise ProtocolError(
                                    "HARP v14 compatibility z-score is nonfinite."
                                )
                            scores_by_case.setdefault(case_id, []).append(score)
                        hashes.append(
                            require_sha256(
                                by_replica[(source, seed)].get("checkpoint_sha256"),
                                name="compatibility expert checkpoint",
                            )
                        )
                    candidate_rows: dict[
                        str,
                        tuple[
                            tuple[float, float, float],
                            float,
                            float,
                            tuple[str, str, str],
                        ],
                    ] = {}
                    for case_id, raw_scores in scores_by_case.items():
                        replica_scores = tuple(raw_scores)
                        if len(replica_scores) != 3:
                            raise ProtocolError(
                                "HARP v14 case-local compatibility lacks replicas."
                            )
                        mean = sum(replica_scores) / 3.0
                        std = math.sqrt(
                            sum((value - mean) ** 2 for value in replica_scores) / 3.0
                        )
                        candidate_rows[case_id] = (
                            replica_scores,  # type: ignore[assignment]
                            mean,
                            std,
                            tuple(hashes),  # type: ignore[arg-type]
                        )
                    by_candidate_case[source] = candidate_rows
                if case_order is None:
                    raise ProtocolError("HARP v14 crossfit compatibility has no cases.")
                for case_id in case_order:
                    if any(case_id not in by_candidate_case[source] for source in candidates):
                        raise ProtocolError(
                            "HARP v14 crossfit compatibility case coverage drifted."
                        )
                    order = tuple(
                        sorted(
                            candidates,
                            key=lambda source: (
                                by_candidate_case[source][case_id][1],
                                source,
                            ),
                        )
                    )
                    best = by_candidate_case[order[0]][case_id][1]
                    runner_up = by_candidate_case[order[1]][case_id][1]
                    rank_by_source = {
                        source: rank for rank, source in enumerate(order, 1)
                    }
                    for source in candidates:
                        scores, mean, std, hashes = by_candidate_case[source][case_id]
                        rank = rank_by_source[source]
                        output.append(
                            FoldConditionedCompatibility(
                                outer_target_id=outer,
                                heldout_center_id=heldout,
                                current_query_center_id=query,
                                case_id=case_id,
                                candidate_source_id=source,
                                replica_z_scores=scores,
                                mean_z=mean,
                                std_z=std,
                                rank=rank,
                                rank_margin=(
                                    runner_up - mean if rank == 1 else best - mean
                                ),
                                source_checkpoint_hashes=hashes,
                            )
                        )
    return tuple(sorted(output, key=lambda row: row.key))


def persist_fold_conditioned_surface_manifest(
    root: Path, surface: FoldConditionedSourceSurface
) -> Path:
    """Persist the full fold and geometry identities after checkpoint recovery."""

    root = Path(root)
    path = root / "manifest.json"
    body = {
        "schema_version": _SURFACE_MANIFEST_SCHEMA,
        "status": "COMPLETE_LABEL_FREE_SOURCE_CROSSFIT",
        "surface_hash": surface.surface_hash,
        "outer_target_ids": list(surface.outer_target_ids),
        "action_block_count": len(surface.blocks),
        "compatibility_receipt_count": len(surface.compatibility),
        "action_block_hashes": [row.block_hash for row in surface.blocks],
        "compatibility_receipt_hashes": [
            row.receipt_hash for row in surface.compatibility
        ],
        "six_source_geometry_audit": dict(six_source_geometry_audit()),
        "physical_plan": dict(fold_conditioned_physical_plan(surface.outer_target_ids)),
        "lineage": dict(surface.lineage),
        "labels_consumed": False,
    }
    payload = {**body, "manifest_hash": canonical_hash(body)}
    if path.exists():
        if not path.is_file() or path.is_symlink() or read_json(path) != payload:
            raise ProtocolError("Existing HARP v14 crossfit manifest differs; refusing repair.")
    else:
        atomic_json(path, payload)
    if read_json(path) != payload:
        raise ProtocolError("HARP v14 crossfit manifest failed durable round trip.")
    return path


def bind_crossfit_prediction_folds_to_target_menus(
    surface: FoldConditionedSourceSurface,
    target_menus: Sequence[LabelFreeTargetMenu | LabelFreeOuterMenu],
) -> tuple[LabelFreeOuterMenu, ...]:
    """Replace legacy source blocks with the sealed ``r == q`` predictions.

    The target blocks remain exactly the ``C-{H}`` physical menu.  Source
    blocks are projected only from folds whose current query is the held-out
    query; calibration ``r != q`` blocks stay in the separate crossfit surface.
    """

    menus = tuple(target_menus)
    by_outer = {menu.outer_target_id: menu for menu in menus}
    if set(by_outer) != set(surface.outer_target_ids) or len(by_outer) != len(menus):
        raise ProtocolError("HARP v14 target/crossfit outer binding drifted.")
    output: list[LabelFreeOuterMenu] = []
    for outer in surface.outer_target_ids:
        target = tuple(
            block for block in by_outer[outer].blocks if block.surface_role == "target"
        )
        if not target:
            raise ProtocolError("HARP v14 target menu lacks C-{H} actions.")
        source: list[LabelFreeActionBlock] = []
        for block in surface.blocks:
            action = block.action
            if (
                action.outer_target_id != outer
                or action.current_query_center_id != action.heldout_center_id
            ):
                continue
            kind = (
                ActionKind.B
                if action.action_id == "B"
                else ActionKind.U
                if action.action_id == "U"
                else ActionKind.HXE
            )
            source.append(
                LabelFreeActionBlock(
                    surface_role="development",
                    outer_target_id=outer,
                    query_center_id=action.current_query_center_id,
                    action_kind=kind,
                    selected_source_id=action.selected_source_id,
                    sample_ids=block.sample_ids,
                    case_ids=block.case_ids,
                    probabilities=block.probabilities,
                    seed_dispersion=block.seed_dispersion,
                )
            )
        combined = tuple(sorted((*source, *target), key=lambda row: row.key))
        lineage = {
            **dict(by_outer[outer].lineage),
            "source_crossfit_surface_hash": surface.surface_hash,
            "source_crossfit_prediction_semantics": "heldout_q_equals_current_query_r",
            "source_crossfit_calibration_surface_separate": True,
            "six_source_geometry_audit_hash": str(
                six_source_geometry_audit()["geometry_audit_hash"]
            ),
        }
        output.append(
            LabelFreeOuterMenu(
                outer_target_id=outer,
                blocks=combined,
                lineage=lineage,
            )
        )
    return tuple(output)


def _robust_location_scale(raw: object) -> tuple[float, float]:
    if not isinstance(raw, list) or not raw:
        raise ProtocolError("HARP v14 own-source compatibility values are absent.")
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ProtocolError("HARP v14 own-source compatibility values are malformed.")
    location = float(np.median(values))
    mad = float(np.median(np.abs(values - location)))
    scale = 1.4826 * mad
    if not math.isfinite(scale) or scale <= 1e-12:
        scale = max(abs(location) * 1e-6, 1e-6)
    return location, scale


def _outer_targets(values: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(str(value) for value in values)
    if (
        not requested
        or tuple(center for center in CENTERS if center in set(requested)) != requested
        or len(set(requested)) != len(requested)
    ):
        raise ProtocolError("HARP v14 crossfit outer targets are noncanonical.")
    return requested


__all__ = (
    "bind_crossfit_prediction_folds_to_target_menus",
    "build_fold_conditioned_classifier_tasks",
    "build_fold_conditioned_compatibility",
    "fold_conditioned_physical_plan",
    "materialize_fold_conditioned_source_surface",
    "persist_fold_conditioned_surface_manifest",
)
