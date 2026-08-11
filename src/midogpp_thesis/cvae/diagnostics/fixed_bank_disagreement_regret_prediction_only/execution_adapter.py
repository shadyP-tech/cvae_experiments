"""Adapters from the prediction-only diagnostic to neutral workstation runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

from ...protocol import ProtocolError
from ...routing.disagreement_regret_core import ProbabilityRow
from ...runtime.artifact_io import atomic_json, read_json
from ...runtime.frozen_source_streams import (
    FrozenSourceStreamCache,
    load_frozen_source_streams,
    materialize_frozen_source_streams,
    stage_frozen_source_streams,
)
from ...runtime.preflight import run_label_free_workstation_preflight as _preflight
from .actions import actions_for_target
from .constants import (
    B_ACTION_ID,
    CENTERS,
    EXPECTED_CLASSIFIER_FIT_COUNT,
    GEOMETRY_IDS,
    SOURCE_ROWS_PER_CLASS,
    U_ACTION_ID,
    source_from_action_id,
)
from .prediction_contracts import (
    ActionPredictionStore,
    GlobalSourcePredictionSeal,
    GlobalTestPredictionSeal,
)
from .development_actions import development_actions_for
from .development_prediction_contracts import CompositePrelabelPredictionSeal
from .prediction_runtime import (
    issue_test_inference_admission,
    materialize_source_action_predictions,
    materialize_test_action_predictions,
)


SCRATCH_ROOT = "/data/local/fixed_bank_disagreement_regret_prediction_only_v1"
LOCAL_SOURCE_DIRECTORY = "source_cache"


@dataclass(frozen=True)
class _FrozenSourceConfigAdapter:
    contract_hash: str
    expert_bank_root: Path
    runtime: Mapping[str, object]


def _source_generation_runtime(runtime: Mapping[str, object]) -> dict[str, object]:
    """Adapt only the neutral frozen-source generation contract.

    Target-classifier topology is deliberately absent here: this diagnostic
    owns an 18-action, 1,458-fit target bank and a separate strict H/q source
    bank, neither of which is described by the legacy neutral preflight shape.
    """

    return {
        "generation_devices": list(runtime["source_generation_devices"]),
        "cuda_visible_devices": "0,1",
        "source_workers_per_device": int(runtime["source_workers_per_device"]),
        "generation_workers_per_device": 1,
        "persistent_source_workers": True,
        "classifier_workers": int(runtime["cpu_workers"]),
        "classifier_threads_per_worker": int(runtime["threads_per_worker"]),
        "launch_blas_threads": 1,
        "tf32_enabled": False,
        "amp_enabled": False,
        "phase_disjoint_gpu_and_cpu_pools": True,
        "multiprocessing_start_method": "spawn",
        "parent_cuda_context_forbidden": True,
        "generated_cache_format": "float32_npy_memmap",
        "scientific_reductions_dtype": "float64",
        "source_job_count": 27,
        "source_stream_count": 81,
        "source_prefix_rows_per_class": SOURCE_ROWS_PER_CLASS,
        "scratch_preference": [SCRATCH_ROOT, "artifact_parent"],
        "minimum_logical_cpu_count": 12,
        "minimum_physical_ram_bytes": 107_374_182_400,
        "minimum_artifact_disk_free_bytes": 12_884_901_888,
        "minimum_gpu_free_mib_per_device": 18_000,
    }


def _shared_preflight_runtime(runtime: Mapping[str, object]) -> dict[str, object]:
    """Build the exact compatibility payload required by the shared probe.

    The shared probe validates hardware and process isolation but still has a
    historical nine-action topology.  These compatibility-only values never
    enter this diagnostic's persisted topology claims; the enriched report
    below overwrites them with the canonical 5,184/10,368/1,458 counts.
    """

    adapted = _source_generation_runtime(runtime)
    adapted.update(
        {
            "target_task_count": 81,
            "target_action_identity_count": 81,
            "target_probability_cell_count": 729,
            "target_unique_classifier_fit_count": 729,
            "maximum_total_classifier_fit_count": 729,
            "resume_policy": "hash_validated_atomic_phase_and_task_checkpoints",
        }
    )
    return adapted


def run_label_free_workstation_preflight(
    root: Path, *, runtime: Mapping[str, object]
) -> Mapping[str, object]:
    if (
        tuple(runtime.get("source_generation_devices", ())) != ("cuda:0", "cuda:1")
        or int(runtime.get("source_generation_workers", -1)) != 2
        or int(runtime.get("source_workers_per_device", -1)) != 1
        or int(runtime.get("cpu_workers", -1)) != 4
        or int(runtime.get("threads_per_worker", -1)) != 3
        or runtime.get("gpu_phase_precedes_cpu_phase") is not True
        or runtime.get("gpu_and_cpu_phases_disjoint") is not True
        or runtime.get("parent_cuda_context_forbidden_during_cpu_phase") is not True
        or runtime.get("scientific_reduction_dtype") != "float64"
        or runtime.get("surface_storage_dtype") != "float32"
        or runtime.get("hash_validated_resume") is not True
    ):
        raise ProtocolError("Prediction-only workstation execution contract drifted.")
    report_path = root / "reports/workstation_preflight.json"
    fields = {
        "prediction_only_phase_order": [
            "generated_source_streams",
            "target_compatible_classifier_bank_seal",
            "strict_H_q_source_oof_fit_and_prediction_seal",
            "composite_prelabel_prediction_seal",
            "source_label_capability",
            "regret_model_bank_seal",
            "test_cache_admission",
            "frozen_test_inference",
        ],
        "source_oof_physical_classifier_fit_count": 5_184,
        "source_oof_oriented_prediction_cell_count": 10_368,
        "target_compatible_classifier_fit_count": EXPECTED_CLASSIFIER_FIT_COUNT,
        "total_physical_classifier_fit_count": 6_642,
        "test_phase_classifier_fit_count": 0,
        "persistent_a5000_gpu_worker_count": 2,
        "cpu_classifier_worker_count": 4,
        "blas_threads_per_classifier_worker": 3,
        "maximum_dense_fit_bytes": 536_870_912,
    }
    if report_path.is_file():
        payload = read_json(report_path)
        if payload.get("status") != "PASS" or any(
            payload.get(key) != value for key, value in fields.items()
        ):
            raise ProtocolError("Persisted prediction-only preflight drifted.")
        return payload
    with tempfile.TemporaryDirectory(
        prefix=".disagreement-regret-preflight-", dir=root.parent
    ) as probe:
        payload = dict(
            _preflight(
                Path(probe),
                runtime=_shared_preflight_runtime(runtime),
                expected_scratch_root=SCRATCH_ROOT,
            )
        )
    payload["disk_probe_path"] = str(root.resolve())
    payload.update(fields)
    atomic_json(report_path, payload)
    return payload


def materialize_sources(
    config: object, generation_lock: object, *, root: Path
) -> FrozenSourceStreamCache:
    runtime = _source_generation_runtime(getattr(config, "runtime"))
    adapter = _FrozenSourceConfigAdapter(
        contract_hash=str(getattr(config, "contract_hash")),
        expert_bank_root=Path(getattr(config, "expert_bank_root")),
        runtime=runtime,
    )
    cache = materialize_frozen_source_streams(
        adapter, generation_lock, root=root
    )
    shutil.rmtree(root / "checkpoints/frozen_source_streams", ignore_errors=True)
    return cache


def stage_sources_for_cpu(
    cache: FrozenSourceStreamCache, *, root: Path
) -> FrozenSourceStreamCache:
    return stage_frozen_source_streams(
        cache,
        scratch_root=Path(SCRATCH_ROOT),
        canonical_root=root,
        local_directory=LOCAL_SOURCE_DIRECTORY,
    )


def aggregate_probability_rows(
    capability: GlobalSourcePredictionSeal | GlobalTestPredictionSeal,
    *,
    frame_role: str,
    geometry_id: str,
    outer_target_id: object,
) -> tuple[ProbabilityRow, ...]:
    """Return exact-nine core rows for one H and one predeclared geometry."""

    target = str(outer_target_id)
    if target not in CENTERS or geometry_id not in GEOMETRY_IDS:
        raise ProtocolError("Prediction-only probability view identity drifted.")
    if frame_role == "source" and isinstance(capability, GlobalSourcePredictionSeal):
        store = capability.source_store
        seal_hash = capability.seal_hash
        # The store retains every query for replay, but the development view
        # must exclude H from features, responses, and standardization.
        expected_queries = set(CENTERS).difference({target})
    elif frame_role == "test" and isinstance(capability, GlobalTestPredictionSeal):
        store = capability.test_store
        seal_hash = capability.seal_hash
        expected_queries = {target}
    else:
        raise ProtocolError("Prediction-only probability capability/role mismatch.")
    if store.frame_role != frame_role:
        raise ProtocolError("Prediction-only probability store role drifted.")
    row_ids = store.rows_by_outer_target[target]
    case_ids = store.case_ids_by_outer_target[target]
    query_ids = store.query_ids_by_outer_target[target]
    rows: list[ProbabilityRow] = []
    permitted_actions = (
        B_ACTION_ID,
        U_ACTION_ID,
        *(
            action.action_id
            for action in actions_for_target(target)
            if action.geometry_id == geometry_id
        ),
    )
    for action_id in permitted_actions:
        source_id = source_from_action_id(action_id)
        mean, sd, hard_vote = store.exact_nine_summary(target, action_id)
        for sample_id, case_id, query_id, probability, deviation, vote in zip(
            row_ids,
            case_ids,
            query_ids,
            mean,
            sd,
            hard_vote,
            strict=True,
        ):
            if frame_role == "source" and query_id == target:
                continue
            # A candidate may be neither H nor the query's own source.
            if source_id is not None and source_id in (target, query_id):
                continue
            rows.append(
                ProbabilityRow(
                    query_id=query_id,
                    case_id=case_id,
                    sample_id=sample_id,
                    action_id=action_id,
                    source_id=source_id,
                    probability=float(probability),
                    probability_sd=float(deviation),
                    hard_vote_fraction=float(vote),
                    prediction_seal_hash=seal_hash,
                )
            )
    canonical = tuple(sorted(rows, key=lambda row: row.row_key))
    if (
        not canonical
        or {row.query_id for row in canonical} != expected_queries
        or len({row.row_key for row in canonical}) != len(canonical)
    ):
        raise ProtocolError("Prediction-only probability view coverage drifted.")
    return canonical


def aggregate_source_oof_probability_rows(
    capability: CompositePrelabelPredictionSeal,
    *,
    frame_role: str,
    geometry_id: str,
    outer_target_id: object,
) -> tuple[ProbabilityRow, ...]:
    """Return already-valid strict H/q source rows without post-hoc filtering."""

    target = str(outer_target_id)
    if (
        not isinstance(capability, CompositePrelabelPredictionSeal)
        or frame_role != "source"
        or target not in CENTERS
        or geometry_id not in GEOMETRY_IDS
    ):
        raise ProtocolError("Strict source probability view identity drifted.")
    store = capability.source_store
    rows: list[ProbabilityRow] = []
    expected_queries = set(CENTERS).difference({target})
    for query in CENTERS:
        if query == target:
            continue
        row_ids = store.rows_by_query[query]
        case_ids = store.case_ids_by_query[query]
        actions = tuple(
            action
            for action in development_actions_for(target, query)
            if action.action_id in (B_ACTION_ID, U_ACTION_ID)
            or action.geometry_id == geometry_id
        )
        if len(actions) != 9:
            raise ProtocolError("Strict source action menu drifted.")
        for action in actions:
            mean, sd, hard_vote = store.exact_nine_summary(
                target, query, action.action_id
            )
            source_id = source_from_action_id(action.action_id)
            if source_id is not None and source_id in (target, query):
                raise ProtocolError("Strict source action escaped H/q exclusion.")
            for sample_id, case_id, probability, deviation, vote in zip(
                row_ids,
                case_ids,
                mean,
                sd,
                hard_vote,
                strict=True,
            ):
                rows.append(
                    ProbabilityRow(
                        query_id=query,
                        case_id=case_id,
                        sample_id=sample_id,
                        action_id=action.action_id,
                        source_id=source_id,
                        probability=float(probability),
                        probability_sd=float(deviation),
                        hard_vote_fraction=float(vote),
                        prediction_seal_hash=capability.seal_hash,
                    )
                )
    canonical = tuple(sorted(rows, key=lambda row: row.row_key))
    if (
        not canonical
        or {row.query_id for row in canonical} != expected_queries
        or len({row.row_key for row in canonical}) != len(canonical)
        or any(
            row.source_id in (target, row.query_id)
            for row in canonical
            if row.source_id is not None
        )
    ):
        raise ProtocolError("Strict source probability coverage drifted.")
    return canonical


def runtime_summary_payload(
    *,
    generated_sources: FrozenSourceStreamCache,
    source_predictions: GlobalSourcePredictionSeal,
    test_predictions: GlobalTestPredictionSeal,
    runtime: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_prediction_only_runtime_summary_v1",
        "status": "PASS",
        "source_stream_lock_hash": generated_sources.lock_hash,
        "source_oof_classifier_bank_seal_hash": (
            source_predictions.classifier_bank.seal_hash
        ),
        "target_compatible_classifier_bank_seal_hash": (
            source_predictions.target_classifier_bank.seal_hash
        ),
        "source_prediction_seal_hash": source_predictions.seal_hash,
        "regret_model_bank_seal_hash": (
            test_predictions.admission.regret_model_bank_seal_hash
        ),
        "test_prediction_seal_hash": test_predictions.seal_hash,
        "source_stream_count": len(generated_sources.records),
        "source_oof_physical_classifier_fit_count": 5_184,
        "source_oof_oriented_prediction_cell_count": 10_368,
        "target_compatible_classifier_fit_count": EXPECTED_CLASSIFIER_FIT_COUNT,
        "total_physical_classifier_fit_count": 6_642,
        "test_classifier_fit_count": 0,
        "target_labels_available": False,
        "target_scores_computed": False,
        "test_cache_admitted_after_regret_model_bank_seal": True,
        "gpu_source_phase_completed_before_cpu_fit_phase": True,
        "gpu_and_cpu_pools_disjoint": True,
        "cpu_classifier_worker_count": int(runtime["cpu_workers"]),
        "blas_threads_per_classifier_worker": int(runtime["threads_per_worker"]),
        "float32_probability_store": True,
        "float64_exact_nine_reductions": True,
        "float64_frozen_classifier_parameters": True,
        "hash_validated_resume": True,
    }


def probability_views(
    aggregate: object,
    capability: object,
    *,
    frame_role: str,
) -> dict[tuple[str, str], tuple[object, ...]]:
    """Build every target/geometry probability view in canonical order."""

    if not callable(aggregate):
        raise ProtocolError("Probability aggregation dependency is not callable.")
    return {
        (target, geometry): tuple(
            aggregate(
                capability,
                frame_role=frame_role,
                geometry_id=geometry,
                outer_target_id=target,
            )
        )
        for target in CENTERS
        for geometry in GEOMETRY_IDS
    }


__all__ = (
    "GlobalSourcePredictionSeal",
    "GlobalTestPredictionSeal",
    "SCRATCH_ROOT",
    "aggregate_probability_rows",
    "aggregate_source_oof_probability_rows",
    "issue_test_inference_admission",
    "load_frozen_source_streams",
    "materialize_source_action_predictions",
    "materialize_sources",
    "materialize_test_action_predictions",
    "probability_views",
    "run_label_free_workstation_preflight",
    "runtime_summary_payload",
    "stage_sources_for_cpu",
)
