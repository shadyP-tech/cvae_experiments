from __future__ import annotations

import json
import csv
import hashlib
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.frozen_policy_downstream.utility_aligned_residual_fresh.contracts import (
    BASE_ACTION_ID,
    CENTERS,
    EXPECTED_LOGICAL_PREDICTION_COUNT,
    GLOBAL_ACTION_ID,
    PERMUTATION_ACTION_ID,
    ROUTED_ACTION_ID,
    UNIFORM_ACTION_ID,
    FrozenActionPayload,
    PredictionCell,
    expected_action_ids,
    legal_sources,
    tail_action_id,
)
from midogpp_thesis.cvae.frozen_policy_downstream.utility_aligned_residual_fresh.inference import (
    evaluate_sealed_predictions,
)
from midogpp_thesis.cvae.frozen_policy_downstream.utility_aligned_residual_fresh.planning import (
    build_evaluation_plan,
)
from midogpp_thesis.cvae.frozen_policy_downstream.utility_aligned_residual_fresh.prediction_seal import (
    seal_predictions,
    validate_prediction_seal,
)
from midogpp_thesis.cvae.frozen_policy_downstream.utility_aligned_residual_fresh.policy_loading import (
    FrozenUtilityAlignedPolicySurface,
)
from midogpp_thesis.cvae.frozen_policy_downstream.utility_aligned_residual_fresh.prediction_cache import (
    PredictionTaskSpec,
    materialize_prediction_cache,
)
from midogpp_thesis.cvae.frozen_policy_downstream.utility_aligned_residual_fresh.prediction_io import (
    array_sha256 as _array_sha256,
    atomic_json as _atomic_json,
    atomic_save_npy as _atomic_save_npy,
    sha256_file as _sha256_file,
)
from midogpp_thesis.cvae.frozen_policy_downstream.utility_aligned_residual_fresh.prediction_store import (
    try_load_task as _try_load_task,
)
from midogpp_thesis.cvae.frozen_policy_downstream.utility_aligned_residual_fresh.bundle import (
    validate_utility_aligned_residual_fresh_bundle,
    write_utility_aligned_residual_fresh_bundle,
)
from midogpp_thesis.cvae.frozen_policy_downstream.utility_aligned_residual_fresh.label_access import (
    SCORING_COLUMNS,
    SCORING_SCHEMA,
)
from midogpp_thesis.cvae.frozen_policy_downstream.utility_aligned_residual_fresh.runner import (
    _UtilityAlignedFreshRunnerDependencies,
    run_utility_aligned_residual_fresh,
)
from midogpp_thesis.cvae.frozen_policy_downstream.utility_aligned_residual_fresh.target_surface import (
    FreshReservation,
    FreshTargetFrame,
    FreshTargetSurface,
    require_active_fresh_target_artifacts,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.cvae.routing.residual_topup.hashing import canonical_sha256
from midogpp_thesis.cvae.routing.utility_aligned.contracts import (
    ABSTENTION_SEMANTICS,
    BASE_ACTION_ID as CORE_BASE_ACTION_ID,
    GLOBAL_ACTION_ID as CORE_GLOBAL_ACTION_ID,
    PERMUTATION_ACTION_ID as CORE_PERMUTATION_ACTION_ID,
    ROUTED_ACTION_ID as CORE_ROUTED_ACTION_ID,
    build_case_bootstrap_plan,
)
from midogpp_thesis.cvae.frozen_policy_downstream.utility_aligned_residual_fresh.policy_loading import (
    ACTION_LIBRARY_SCHEMA,
    POLICY_EXPERIMENT_ID,
    POLICY_LOCK_SCHEMA,
    TARGET_POLICY_LOCK_SCHEMA,
    load_frozen_utility_aligned_policy,
)


def _counts(target: str, selected: str | None, *, uniform: bool = False):
    sources = legal_sources(target)
    if uniform:
        values = {source: 144 for source in sources}
    else:
        values = {
            source: 128 + (128 if source == selected else 0) for source in sources
        }
    return {0: values, 1: values}


def _actions_by_target():
    output = {}
    for target in CENTERS:
        sources = legal_sources(target)
        specifications = [
            (BASE_ACTION_ID, "base", None, False, None, False),
            (UNIFORM_ACTION_ID, "uniform_control", None, False, None, True),
            (GLOBAL_ACTION_ID, "global_ablation", sources[0], False, None, False),
            (ROUTED_ACTION_ID, "utility_aligned_router", sources[1], False, None, False),
            (
                PERMUTATION_ACTION_ID,
                "target_feature_permutation_control",
                None,
                True,
                "negative_lcb",
                False,
            ),
            *(
                (
                    tail_action_id(source),
                    "terminal_oracle_diagnostic",
                    source,
                    False,
                    None,
                    False,
                )
                for source in sources
            ),
        ]
        output[target] = tuple(
            FrozenActionPayload(
                target_center=target,
                action_id=action_id,
                action_role=role,
                source_counts_by_class=_counts(
                    target,
                    selected,
                    uniform=uniform,
                ),
                action_hash=f"hash::{target}::{action_id}",
                selected_source=selected,
                abstained_to_base=abstained,
                fallback_reason=fallback,
            )
            for action_id, role, selected, abstained, fallback, uniform in specifications
        )
    return output


def _rows_by_target():
    return {
        target: tuple(f"row::{target}::{index}" for index in range(4))
        for target in CENTERS
    }


def _predictions(plan):
    by_composition = {}
    output = []
    for cell in plan.logical_cells:
        probability = by_composition.setdefault(
            (cell.target_center, cell.training_seed, cell.generation_seed, cell.composition_hash),
            np.asarray([0.1, 0.9, 0.2, 0.8], dtype=np.float32),
        )
        output.append(
            PredictionCell(
                target_center=cell.target_center,
                training_seed=cell.training_seed,
                generation_seed=cell.generation_seed,
                action_id=cell.action_id,
                action_hash=cell.action_hash,
                composition_hash=cell.composition_hash,
                evaluation_row_ids=plan.evaluation_row_ids_by_target[cell.target_center],
                probabilities=probability,
            )
        )
    return tuple(output)


def _policy(actions):
    return FrozenUtilityAlignedPolicySurface(
        policy_lock_hash="a" * 64,
        action_library_hash="b" * 64,
        exact_tail_utility_surface_lock_hash="c" * 64,
        reservation_id="fresh-v1",
        reservation_hash="d" * 64,
        support_case_ids_by_target=MappingProxyType(
            {
                target: tuple(f"support::{target}::{i}" for i in range(8))
                for target in CENTERS
            }
        ),
        evaluation_case_ids_by_target=MappingProxyType(
            {target: (f"eval::{target}",) for target in CENTERS}
        ),
        actions_by_target=MappingProxyType(actions),
        raw_actions_by_key=MappingProxyType({}),
        policy_payload=MappingProxyType(
            {
                "target_reservation_artifact_id": (
                    "midogpp_utility_aligned_fresh_target_reservation_v1"
                ),
                "target_reservation_hash": "d" * 64,
                "target_evaluation_binding_hash": "e" * 64,
            }
        ),
    )


def _target_surface(tmp_path: Path, rows):
    frames = {}
    for target in CENTERS:
        frames[target] = FreshTargetFrame(
            center=target,
            embeddings=np.zeros((4, 3840), dtype=np.float32),
            evaluation_row_ids=rows[target],
            case_ids=tuple(f"case::{target}::{i}" for i in range(4)),
            file_sha256="f" * 64,
        )
    reservation = FreshReservation(
        reservation_id="fresh-v1",
        reservation_hash="d" * 64,
        target_evaluation_binding_hash="e" * 64,
        support_case_ids_by_center=MappingProxyType(
            {
                target: tuple(f"support::{target}::{i}" for i in range(8))
                for target in CENTERS
            }
        ),
        evaluation_case_ids_by_center=MappingProxyType(
            {
                target: tuple(f"case::{target}::{i}" for i in range(4))
                for target in CENTERS
            }
        ),
        scoring_manifest_sha256="0" * 64,
        payload=MappingProxyType({}),
    )
    return FreshTargetSurface(
        reservation=reservation,
        frames_by_center=MappingProxyType(frames),
        cache_content_hash="1" * 64,
        cache_protocol_hash="2" * 64,
        scoring_manifest_path=tmp_path / "manifest.csv",
        scoring_manifest_sha256="0" * 64,
    )


def test_package_facade_exposes_only_production_integration_apis():
    import midogpp_thesis.cvae.frozen_policy_downstream.utility_aligned_residual_fresh as api
    from midogpp_thesis.cvae.frozen_policy_downstream.utility_aligned_residual_fresh import (
        prediction_cache as prediction_facade,
    )

    assert api.__all__ == (
        "load_utility_aligned_residual_fresh_config",
        "run_utility_aligned_residual_fresh",
        "validate_utility_aligned_residual_fresh_bundle",
        "validate_utility_aligned_residual_fresh_workspace_binding",
    )
    assert not hasattr(api, "_UtilityAlignedFreshRunnerDependencies")
    assert not hasattr(prediction_facade, "_array_sha256")
    assert not hasattr(prediction_facade, "_atomic_json")
    assert not hasattr(prediction_facade, "_try_load_task")


def test_plan_deduplicates_compositions_but_not_logical_actions():
    plan = build_evaluation_plan(
        _actions_by_target(), evaluation_row_ids_by_target=_rows_by_target()
    )
    assert len(plan.logical_cells) == EXPECTED_LOGICAL_PREDICTION_COUNT == 1053
    assert len(plan.composition_cells) == 810
    target = CENTERS[0]
    assert plan.action_for(target, PERMUTATION_ACTION_ID).composition_hash == plan.action_for(
        target, BASE_ACTION_ID
    ).composition_hash
    routed = plan.action_for(target, ROUTED_ACTION_ID)
    assert routed.composition_hash == plan.action_for(
        target, tail_action_id(routed.selected_source)
    ).composition_hash
    assert expected_action_ids(target)[0:5] == (
        BASE_ACTION_ID,
        UNIFORM_ACTION_ID,
        GLOBAL_ACTION_ID,
        ROUTED_ACTION_ID,
        PERMUTATION_ACTION_ID,
    )


def test_global_seal_precedes_center_level_scoring_and_terminal_oracle():
    rows = _rows_by_target()
    plan = build_evaluation_plan(_actions_by_target(), evaluation_row_ids_by_target=rows)
    predictions = _predictions(plan)
    with pytest.raises(ProtocolError, match="Every utility-aligned"):
        seal_predictions(plan, predictions[:-1])
    capability = seal_predictions(plan, predictions)
    summary = validate_prediction_seal(capability, expected_plan=plan)
    assert summary.logical_prediction_count == 1053
    assert summary.unique_composition_count == 810
    labels = {row: index % 2 for target in CENTERS for index, row in enumerate(rows[target])}
    report = evaluate_sealed_predictions(capability, labels)
    assert len(report.contrast_inference) == 6
    assert {row.contrast_id for row in report.contrast_inference}.issuperset(
        {"R-B", "R-G_delta", "R-U", "R-P"}
    )
    assert all(row.center_count == 9 for row in report.contrast_inference)
    assert len(report.oracle_diagnostics) == 9
    assert all(row.diagnostic_only and not row.may_update_frozen_policy for row in report.oracle_diagnostics)
    assert report.policy_update_emitted is False


def test_active_only_reservation_gate_rejects_inactive(tmp_path: Path):
    reservation_path = tmp_path / "reservation.json"
    reservation_path.write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "authorized_consumer_experiment_ids": [
                    (
                        "midogpp.routing_and_composition."
                        "uniform_b_v2_utility_aligned_residual_policy_lock.v1"
                    ),
                    (
                        "midogpp.frozen_policy_downstream."
                        "uniform_b_v2_utility_aligned_residual_fresh.v1"
                    ),
                ],
                "fresh_unconsumed_surface": True,
                "labels_opened": False,
                "consumed_test_used": False,
                "consumed_validation_used": False,
                "consumed_stage70_used": False,
                "consumed_stage90_used": False,
            }
        ),
        encoding="utf-8",
    )
    cache = tmp_path / "cache"
    for member in (
        "manifests/cache_protocol.json",
        "manifests/content_index.json",
        "tables/row_index.csv",
        *(f"embeddings/by_center/center_{target}.npy" for target in CENTERS),
    ):
        path = cache / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    manifest = tmp_path / "manifest.csv"
    manifest.touch()
    config = SimpleNamespace(
        fresh_reservation_path=reservation_path,
        fresh_scoring_manifest_path=manifest,
        fresh_target_cache_root=cache,
    )
    with pytest.raises(ProtocolError, match="active unconsumed"):
        require_active_fresh_target_artifacts(config)


def test_complete_prediction_checkpoint_tamper_is_fail_closed(tmp_path: Path):
    plan = build_evaluation_plan(
        _actions_by_target(), evaluation_row_ids_by_target=_rows_by_target()
    )
    target = CENTERS[0]
    task = PredictionTaskSpec(
        MappingProxyType(
            {
                "metadata_path": str(tmp_path / "task.json"),
                "probability_path": str(tmp_path / "prob.npy"),
                "task_hash": "task-hash",
                "plan_hash": plan.plan_hash,
                "target_center": target,
                "training_seed": 23,
                "generation_seed": 104729,
                "canonical_root": str(tmp_path),
            }
        )
    )
    (tmp_path / "task.json").write_text(
        json.dumps({"status": "COMPLETE", "checkpoint_hash": "tampered"}),
        encoding="utf-8",
    )
    with pytest.raises(ProtocolError, match="lacks its probability array"):
        _try_load_task(task, plan)


def test_runner_orders_active_gates_and_global_seal_before_labels(tmp_path: Path):
    actions = _actions_by_target()
    rows = _rows_by_target()
    policy = _policy(actions)
    target = _target_surface(tmp_path, rows)
    order = []

    def prediction(config, **kwargs):
        order.append("prediction")
        return SimpleNamespace(predictions=_predictions(kwargs["plan"]))

    def labels(surface, capability):
        order.append("labels")
        assert validate_prediction_seal(capability).logical_prediction_count == 1053
        return {
            row: index % 2
            for center in CENTERS
            for index, row in enumerate(rows[center])
        }

    dependencies = _UtilityAlignedFreshRunnerDependencies(
        validate_workspace=lambda config: order.append("workspace"),
        require_inputs=lambda config: order.append("reservation"),
        load_policy=lambda config: (order.append("policy") or policy),
        load_target=lambda config, loaded: (order.append("target") or target),
        run_preflight=lambda *args, **kwargs: (order.append("preflight") or {"status": "PASS"}),
        load_generation=lambda config: (
            order.append("generation_lock")
            or SimpleNamespace(generation_lock_hash="generation")
        ),
        materialize_source=lambda *args, **kwargs: (
            order.append("source") or SimpleNamespace()
        ),
        materialize_prediction=prediction,
        open_labels=labels,
        evaluate=lambda capability, truth: (
            order.append("evaluate") or evaluate_sealed_predictions(capability, truth)
        ),
        write_bundle=lambda *args, **kwargs: (order.append("bundle") or {}),
        validate_bundle=lambda *args, **kwargs: (order.append("validate") or {"status": "PASS"}),
    )
    config = SimpleNamespace(
        artifact_root=tmp_path / "artifact",
        runtime={"optional_local_scratch_root": "/data/local"},
    )
    run_utility_aligned_residual_fresh(config, dependencies=dependencies)
    assert order == [
        "workspace",
        "reservation",
        "policy",
        "target",
        "preflight",
        "generation_lock",
        "source",
        "prediction",
        "labels",
        "evaluate",
        "bundle",
        "validate",
    ]


def test_bundle_reconstructs_seal_and_rejects_dynamic_array_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import midogpp_thesis.cvae.frozen_policy_downstream.utility_aligned_residual_fresh.bundle_validation as bundle_module
    from midogpp_thesis.cvae.frozen_policy_downstream.utility_aligned_residual_fresh.label_access import (
        open_scoring_labels_after_prediction_seal,
    )
    from midogpp_thesis.cvae.frozen_policy_downstream.residual_topup_fresh.config import (
        DOWNSTREAM_CLASSIFIER,
        canonical_runtime_payload,
    )

    artifact = tmp_path / "artifact"
    source_root = artifact / "checkpoints/source"
    source_root.mkdir(parents=True)
    (source_root / "source_cache.json").write_text("{}\n", encoding="utf-8")
    source = SimpleNamespace(
        root=source_root,
        cache_hash="3" * 64,
        bank_lock_hash="4" * 64,
        generation_lock_hash="generation-lock",
    )
    rows = _rows_by_target()
    actions = _actions_by_target()
    policy = _policy(actions)
    target = _target_surface(tmp_path, rows)
    manifest = target.scoring_manifest_path
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORING_COLUMNS)
        writer.writeheader()
        for center in CENTERS:
            frame = target.frames_by_center[center]
            for index, (row_id, case_id) in enumerate(
                zip(frame.evaluation_row_ids, frame.case_ids, strict=True)
            ):
                writer.writerow(
                    {
                        "schema_version": SCORING_SCHEMA,
                        "row_id": row_id,
                        "center": center,
                        "case_id": case_id,
                        "label": index % 2,
                        "reservation_hash": target.reservation.reservation_hash,
                        "target_cache_content_hash": target.cache_content_hash,
                    }
                )
    scoring_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    reservation = FreshReservation(
        reservation_id=target.reservation.reservation_id,
        reservation_hash=target.reservation.reservation_hash,
        target_evaluation_binding_hash=target.reservation.target_evaluation_binding_hash,
        support_case_ids_by_center=target.reservation.support_case_ids_by_center,
        evaluation_case_ids_by_center=target.reservation.evaluation_case_ids_by_center,
        scoring_manifest_sha256=scoring_sha,
        payload=target.reservation.payload,
    )
    target = FreshTargetSurface(
        reservation=reservation,
        frames_by_center=target.frames_by_center,
        cache_content_hash=target.cache_content_hash,
        cache_protocol_hash=target.cache_protocol_hash,
        scoring_manifest_path=manifest,
        scoring_manifest_sha256=scoring_sha,
    )
    config_source = tmp_path / "config.yaml"
    config_source.write_text("experiment: test\n", encoding="utf-8")
    target_cache_root = tmp_path / "target-cache"
    config = SimpleNamespace(
        source_path=config_source,
        artifact_root=artifact,
        contract_hash="5" * 64,
        classifier=DOWNSTREAM_CLASSIFIER,
        fresh_target_cache_root=target_cache_root,
        runtime=canonical_runtime_payload(),
    )
    plan = build_evaluation_plan(
        actions, evaluation_row_ids_by_target=target.evaluation_row_ids_by_target
    )

    def executor(tasks):
        for task in tasks:
            payload = task.payload
            raw_actions = payload["actions"]
            probabilities = np.stack(
                [np.asarray([0.1, 0.9, 0.2, 0.8], dtype=np.float32)]
                * len(raw_actions),
                axis=0,
            )
            probability_path = Path(str(payload["probability_path"]))
            _atomic_save_npy(probability_path, probabilities)
            logical_rows = [
                {
                    "action_id": raw["action_id"],
                    "action_hash": raw["action_hash"],
                    "composition_hash": raw["composition_hash"],
                    "probability_row": index,
                    "probability_sha256": _array_sha256(probabilities[index]),
                    "labels_available_to_fit_or_predict": False,
                }
                for index, raw in enumerate(raw_actions)
            ]
            unique = len({raw["composition_hash"] for raw in raw_actions})
            unhashed = {
                "schema_version": "midogpp_utility_aligned_prediction_task_v1",
                "status": "COMPLETE",
                "task_id": payload["task_id"],
                "task_hash": payload["task_hash"],
                "plan_hash": payload["plan_hash"],
                "target_center": payload["target_center"],
                "training_seed": payload["training_seed"],
                "generation_seed": payload["generation_seed"],
                "row_ids": list(payload["evaluation_row_ids"]),
                "probability_file_sha256": _sha256_file(probability_path),
                "logical_prediction_count": len(logical_rows),
                "unique_composition_fit_count": unique,
                "logical_rows": logical_rows,
                "fit_rows": [],
                "labels_available_to_fit_or_predict": False,
            }
            _atomic_json(
                Path(str(payload["metadata_path"])),
                {**unhashed, "checkpoint_hash": __import__(
                    "midogpp_thesis.common.hashing", fromlist=["stable_hash"]
                ).stable_hash(unhashed)},
            )

    prediction = materialize_prediction_cache(
        config,
        plan=plan,
        policy=policy,
        source_cache=source,
        target_surface=target,
        generation_lock_hash=source.generation_lock_hash,
        root=artifact / "checkpoints/predictions",
        executor=executor,
    )
    capability = seal_predictions(plan, prediction.predictions)
    labels = open_scoring_labels_after_prediction_seal(target, capability)
    report = evaluate_sealed_predictions(capability, labels)
    monkeypatch.setattr(bundle_module, "load_frozen_utility_aligned_policy", lambda cfg: policy)
    monkeypatch.setattr(bundle_module, "load_fresh_target_surface", lambda cfg, loaded: target)
    monkeypatch.setattr(bundle_module, "load_source_cache", lambda root: source)
    write_utility_aligned_residual_fresh_bundle(
        artifact,
        config=config,
        policy=policy,
        target_surface=target,
        source_cache=source,
        prediction_cache=prediction,
        plan=plan,
        prediction_seal=capability,
        report=report,
        workstation_report={"status": "PASS"},
    )
    assert validate_utility_aligned_residual_fresh_bundle(
        artifact, config=config
    )["status"] == "PASS"
    probability_path = artifact / prediction.records[0].probability_member
    # records are relative to checkpoints/predictions, not the artifact root.
    if not probability_path.is_file():
        probability_path = artifact / "checkpoints/predictions" / prediction.records[0].probability_member
    data = bytearray(probability_path.read_bytes())
    data[-1] ^= 1
    probability_path.write_bytes(data)
    with pytest.raises(ProtocolError, match="content member drifted"):
        validate_utility_aligned_residual_fresh_bundle(artifact, config=config)


def test_independent_policy_loader_binds_exact_reservation_and_case_grid(
    tmp_path: Path,
):
    policy_root = tmp_path / "policy"
    (policy_root / "manifests").mkdir(parents=True)
    (policy_root / "reports").mkdir()
    actions = []
    for target, target_actions in _actions_by_target().items():
        for action in target_actions:
            actions.append(
                {
                    "target_center": target,
                    "action_id": action.action_id,
                    "action_role": action.action_role,
                    "selected_source": action.selected_source,
                    "abstained_to_base": action.abstained_to_base,
                    "fallback_reason": action.fallback_reason,
                    "source_order": list(legal_sources(target)),
                    "counts_per_class": {
                        str(label): dict(action.source_counts_by_class[label])
                        for label in (0, 1)
                    },
                    "total_per_class": action.budget_per_class,
                    "topup_action_hash": (
                        None if action.budget_per_class == 1024 else "a" * 64
                    ),
                    "decision_hash": hashlib.sha256(
                        f"{target}:{action.action_id}".encode()
                    ).hexdigest(),
                    "target_labels_used": False,
                    "support_labels_used": False,
                }
            )
    support = {
        target: [f"support::{target}::{index}" for index in range(8)]
        for target in CENTERS
    }
    evaluation = {target: [f"evaluation::{target}"] for target in CENTERS}
    development_support = {
        target: [f"development-support::{target}::{index}" for index in range(2)]
        for target in CENTERS
    }
    development_evaluation = {
        target: [f"development-evaluation::{target}"] for target in CENTERS
    }
    development_partition_hashes = {
        target: hashlib.sha256(f"development-partition::{target}".encode()).hexdigest()
        for target in CENTERS
    }
    development_manifest_payload = {
        "schema_version": "midogpp_exact_tail_development_case_manifest_v1",
        "reservation_hash": "4" * 16,
        "support_case_ids_by_center": development_support,
        "evaluation_case_ids_by_center": development_evaluation,
        "target_evaluation_case_ids_by_center": evaluation,
        "partition_hashes_by_center": development_partition_hashes,
    }
    common = {
        "experiment_id": POLICY_EXPERIMENT_ID,
        "output_artifact_id": (
            "midogpp_output_uniform_b_v2_utility_aligned_residual_policy_lock_v1"
        ),
        "exact_tail_surface_lock_hash": "1" * 16,
        "equal_union_policy_lock_hash": "2" * 16,
        "metadata_profile_sha256": "3" * 64,
        "development_reservation_artifact_id": "development",
        "development_reservation_hash": "4" * 16,
        "development_case_manifest_hash": canonical_sha256(
            development_manifest_payload
        ),
        "development_support_case_ids_by_query": development_support,
        "development_evaluation_case_ids_by_query": development_evaluation,
        "development_target_evaluation_case_ids_by_target": evaluation,
        "development_partition_hashes_by_query": development_partition_hashes,
        "target_support_surface_artifact_id": "support-surface",
        "target_support_surface_hash": "5" * 64,
        "target_support_parent_reservation_artifact_id": (
            "midogpp_utility_aligned_target_support_reservation_v1"
        ),
        "target_support_parent_reservation_hash": "e" * 16,
        "target_reservation_artifact_id": (
            "midogpp_utility_aligned_fresh_target_reservation_v1"
        ),
        "target_reservation_hash": "6" * 16,
        "target_support_case_ids_by_target": support,
        "target_evaluation_case_ids_by_target": evaluation,
        "target_evaluation_binding_hash": "7" * 64,
        "feature_surface_hash": "8" * 64,
        "feature_schema_hash": "9" * 64,
        "model_lock_hash": "a" * 64,
        "global_ablation_lock_hash": "b" * 64,
        "cardinality_transfer_lock_hash": "c" * 64,
        "target_policy_lock_hash": "d" * 64,
    }
    target_feature_locks = []
    core_policies = []
    for target in CENTERS:
        candidates = list(legal_sources(target))
        plan = build_case_bootstrap_plan(
            target_id=target,
            support_case_ids=tuple(support[target]),
            replicate_count=32,
        )
        bootstrap_hashes = [
            canonical_sha256(
                {
                    "schema_version": "test_bootstrap_surface_v1",
                    "target_id": target,
                    "replicate_index": index,
                    "replicate_hash": plan.replicates[index].replicate_hash,
                }
            )
            for index in range(plan.replicate_count)
        ]
        feature_surface_hash = canonical_sha256(
            {"schema_version": "test_target_feature_surface_v1", "target": target}
        )
        feature_unhashed = {
            "target_id": target,
            "case_bootstrap_plan": plan.to_payload(),
            "target_feature_surface_hash": feature_surface_hash,
            "target_feature_row_count": 72,
            "bootstrap_surface_hashes": bootstrap_hashes,
            "bootstrap_surface_hashes_hash": canonical_sha256(bootstrap_hashes),
            "candidate_sources": candidates,
            "training_seeds": [17, 42, 101],
            "generation_seeds": [17, 42, 101],
            "case_level_resampling": True,
            "labels_used": False,
        }
        target_feature_locks.append(
            {
                **feature_unhashed,
                "target_feature_lock_hash": canonical_sha256(feature_unhashed),
            }
        )
        sorted_bootstrap_hashes = sorted(bootstrap_hashes)
        sorted_replicate_hashes = sorted(
            item.replicate_hash for item in plan.replicates
        )
        for proposed_action, router_kind, global_only, permutation_seed in (
            (
                CORE_GLOBAL_ACTION_ID,
                "global_source_quality_only",
                True,
                None,
            ),
            (
                CORE_ROUTED_ACTION_ID,
                "target_source_interaction",
                False,
                None,
            ),
            (
                CORE_PERMUTATION_ACTION_ID,
                "cyclic_feature_permutation_control",
                False,
                23,
            ),
        ):
            bootstrap_count = 0 if global_only else plan.replicate_count
            used_fallback = proposed_action == CORE_PERMUTATION_ACTION_ID
            selected_source = (
                None
                if used_fallback
                else candidates[1]
                if proposed_action == CORE_ROUTED_ACTION_ID
                else candidates[0]
            )
            policy_unhashed = {
                "schema_version": "midogpp_utility_aligned_policy_v1",
                "target_id": target,
                "candidate_sources": candidates,
                "router_kind": router_kind,
                "proposed_action_id": proposed_action,
                "action_id": CORE_BASE_ACTION_ID if used_fallback else proposed_action,
                "proposed_source": candidates[0],
                "selected_source": selected_source,
                "predicted_gain": 0.1,
                "standard_error": 0.01,
                "lower_confidence_bound": -0.01 if used_fallback else 0.08,
                "confidence_multiplier": 1.96,
                "minimum_gain": 0.0,
                "support_case_count": len(support[target]),
                "minimum_support_case_count": 8,
                "seed_pair_count": 9,
                "replicate_standard_deviation": 0.01,
                "support_bootstrap_replicates": bootstrap_count,
                "minimum_support_bootstrap_replicates": 32,
                "support_bootstrap_standard_deviation": (
                    0.0 if global_only else 0.01
                ),
                "support_bootstrap_surface_hashes": (
                    [] if global_only else sorted_bootstrap_hashes
                ),
                "case_bootstrap_replicate_hashes": (
                    [] if global_only else sorted_replicate_hashes
                ),
                "used_exact_base_fallback": used_fallback,
                "fallback_reason": "negative_lcb" if used_fallback else None,
                "global_only": global_only,
                "permutation_seed": permutation_seed,
                "model_hash": "a" * 64,
                "feature_surface_hash": (
                    canonical_sha256(
                        {
                            "schema_version": "test_permuted_surface_v1",
                            "target": target,
                        }
                    )
                    if proposed_action == CORE_PERMUTATION_ACTION_ID
                    else feature_surface_hash
                ),
                "cardinality_eligibility_hash": "c" * 64,
                "case_bootstrap_plan_hash": None if global_only else plan.plan_hash,
                "target_support_labels_used": False,
                "target_evaluation_used": False,
                "seed_selection_performed": False,
                "abstention_semantics": ABSTENTION_SEMANTICS,
            }
            core_policies.append(
                {
                    **policy_unhashed,
                    "policy_hash": canonical_sha256(policy_unhashed),
                }
            )
    target_policy_unhashed = {
        "schema_version": TARGET_POLICY_LOCK_SCHEMA,
        **{
            key: value
            for key, value in common.items()
            if key
            in {
                    "experiment_id",
                    "output_artifact_id",
                    "exact_tail_surface_lock_hash",
                    "development_case_manifest_hash",
                    "development_support_case_ids_by_query",
                    "development_evaluation_case_ids_by_query",
                    "development_target_evaluation_case_ids_by_target",
                    "development_partition_hashes_by_query",
                    "target_support_surface_artifact_id",
                "target_support_surface_hash",
                "target_support_parent_reservation_artifact_id",
                "target_support_parent_reservation_hash",
                "target_reservation_artifact_id",
                "target_reservation_hash",
                "target_support_case_ids_by_target",
                "target_evaluation_case_ids_by_target",
                "target_evaluation_binding_hash",
                "metadata_profile_sha256",
            }
        },
        "target_feature_locks": target_feature_locks,
        "policies": core_policies,
    }
    target_policy_lock_hash = canonical_sha256(target_policy_unhashed)
    target_policy_payload = {
        **target_policy_unhashed,
        "target_policy_lock_hash": target_policy_lock_hash,
    }
    common["target_policy_lock_hash"] = target_policy_lock_hash
    library_unhashed = {
        "schema_version": ACTION_LIBRARY_SCHEMA,
        **common,
        "action_ids": [row["action_id"] for row in actions],
        "actions": actions,
        "action_count": len(actions),
    }
    library = {
        **library_unhashed,
        "action_library_hash": canonical_sha256(library_unhashed),
    }
    policy_unhashed = {
        "schema_version": POLICY_LOCK_SCHEMA,
        **common,
        "action_library_hash": library["action_library_hash"],
        "candidate_centers": list(CENTERS),
        "primary_contrasts": ["R-B", "R-G_delta", "R-U"],
        "permutation_contrast": "R-P",
        "success_requires_positive_one_sided_lcb": [
            "R-B",
            "R-G_delta",
            "R-U",
            "R-P",
        ],
        "policy_family": "utility_aligned_exact_tail_delta",
        "fallback_policy": "exact_B",
        "outer_target_excluded_from_fit": True,
        "target_support_labels_used": False,
        "target_evaluation_labels_used": False,
        "seed_selection_performed": False,
        "minimum_independent_support_cases_per_target": 8,
        "support_bootstrap_count": 32,
    }
    policy_payload = {
        **policy_unhashed,
        "policy_lock_hash": canonical_sha256(policy_unhashed),
    }
    (policy_root / "manifests/action_library.json").write_text(
        json.dumps(library), encoding="utf-8"
    )
    (policy_root / "manifests/policy_lock.json").write_text(
        json.dumps(policy_payload), encoding="utf-8"
    )
    (policy_root / "manifests/target_policy_lock.json").write_text(
        json.dumps(target_policy_payload), encoding="utf-8"
    )
    (policy_root / "reports/run_state.json").write_text(
        json.dumps({"status": "COMPLETE"}), encoding="utf-8"
    )
    (policy_root / "reports/validation_report.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "checks": {
                    "status": "PASS",
                    "policy_lock_hash": policy_payload["policy_lock_hash"],
                },
            }
        ),
        encoding="utf-8",
    )
    loaded = load_frozen_utility_aligned_policy(
        SimpleNamespace(policy_root=policy_root)
    )
    assert loaded.policy_lock_hash == policy_payload["policy_lock_hash"]
    assert loaded.reservation_hash == "6" * 16
    assert all(len(loaded.support_case_ids_by_target[target]) == 8 for target in CENTERS)

    def _rehash_and_write_bound_policy() -> None:
        manifest = {
            "schema_version": "midogpp_exact_tail_development_case_manifest_v1",
            "reservation_hash": policy_payload["development_reservation_hash"],
            "support_case_ids_by_center": development_support,
            "evaluation_case_ids_by_center": development_evaluation,
            "target_evaluation_case_ids_by_center": evaluation,
            "partition_hashes_by_center": development_partition_hashes,
        }
        manifest_hash = canonical_sha256(manifest)
        for payload in (target_policy_payload, library, policy_payload):
            payload["development_case_manifest_hash"] = manifest_hash
        target_policy_payload["target_policy_lock_hash"] = canonical_sha256(
            {
                key: value
                for key, value in target_policy_payload.items()
                if key != "target_policy_lock_hash"
            }
        )
        library["target_policy_lock_hash"] = target_policy_payload[
            "target_policy_lock_hash"
        ]
        library["action_library_hash"] = canonical_sha256(
            {
                key: value
                for key, value in library.items()
                if key != "action_library_hash"
            }
        )
        policy_payload["target_policy_lock_hash"] = target_policy_payload[
            "target_policy_lock_hash"
        ]
        policy_payload["action_library_hash"] = library["action_library_hash"]
        policy_payload["policy_lock_hash"] = canonical_sha256(
            {
                key: value
                for key, value in policy_payload.items()
                if key != "policy_lock_hash"
            }
        )
        (policy_root / "manifests/action_library.json").write_text(
            json.dumps(library), encoding="utf-8"
        )
        (policy_root / "manifests/target_policy_lock.json").write_text(
            json.dumps(target_policy_payload), encoding="utf-8"
        )
        (policy_root / "manifests/policy_lock.json").write_text(
            json.dumps(policy_payload), encoding="utf-8"
        )
        (policy_root / "reports/validation_report.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "checks": {
                        "status": "PASS",
                        "policy_lock_hash": policy_payload["policy_lock_hash"],
                    },
                }
            ),
            encoding="utf-8",
        )

    original_development_case = development_support[CENTERS[0]][0]
    development_support[CENTERS[0]][0] = support[CENTERS[0]][0]
    _rehash_and_write_bound_policy()
    with pytest.raises(ProtocolError, match="overlap fresh target"):
        load_frozen_utility_aligned_policy(SimpleNamespace(policy_root=policy_root))
    development_support[CENTERS[0]][0] = original_development_case
    _rehash_and_write_bound_policy()
    load_frozen_utility_aligned_policy(SimpleNamespace(policy_root=policy_root))

    original_development_evaluation = development_evaluation[CENTERS[0]][0]
    development_evaluation[CENTERS[0]][0] = support[CENTERS[0]][1]
    _rehash_and_write_bound_policy()
    with pytest.raises(ProtocolError, match="overlap fresh target"):
        load_frozen_utility_aligned_policy(SimpleNamespace(policy_root=policy_root))
    development_evaluation[CENTERS[0]][0] = original_development_evaluation
    _rehash_and_write_bound_policy()
    load_frozen_utility_aligned_policy(SimpleNamespace(policy_root=policy_root))

    # Recompute every outer hash after corrupting one sampled whole-case index.
    # Admission must still fail because it reconstructs the typed PCG64 plan.
    target_policy_payload["target_feature_locks"][0]["case_bootstrap_plan"][
        "replicates"
    ][0]["sampled_indices"][0] = 7
    first_feature = target_policy_payload["target_feature_locks"][0]
    first_feature["target_feature_lock_hash"] = canonical_sha256(
        {
            key: value
            for key, value in first_feature.items()
            if key != "target_feature_lock_hash"
        }
    )
    target_policy_payload["target_policy_lock_hash"] = canonical_sha256(
        {
            key: value
            for key, value in target_policy_payload.items()
            if key != "target_policy_lock_hash"
        }
    )
    library["target_policy_lock_hash"] = target_policy_payload[
        "target_policy_lock_hash"
    ]
    library["action_library_hash"] = canonical_sha256(
        {
            key: value
            for key, value in library.items()
            if key != "action_library_hash"
        }
    )
    policy_payload["target_policy_lock_hash"] = target_policy_payload[
        "target_policy_lock_hash"
    ]
    policy_payload["action_library_hash"] = library["action_library_hash"]
    policy_payload["policy_lock_hash"] = canonical_sha256(
        {
            key: value
            for key, value in policy_payload.items()
            if key != "policy_lock_hash"
        }
    )
    (policy_root / "manifests/action_library.json").write_text(
        json.dumps(library), encoding="utf-8"
    )
    (policy_root / "manifests/target_policy_lock.json").write_text(
        json.dumps(target_policy_payload), encoding="utf-8"
    )
    (policy_root / "manifests/policy_lock.json").write_text(
        json.dumps(policy_payload), encoding="utf-8"
    )
    (policy_root / "reports/validation_report.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "checks": {
                    "status": "PASS",
                    "policy_lock_hash": policy_payload["policy_lock_hash"],
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProtocolError, match="failed reconstruction"):
        load_frozen_utility_aligned_policy(SimpleNamespace(policy_root=policy_root))
