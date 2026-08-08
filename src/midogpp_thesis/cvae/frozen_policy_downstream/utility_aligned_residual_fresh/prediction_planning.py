"""Coarse task planning for composition-deduplicated Stage-70 prediction."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

from ....common.hashing import stable_hash
from .config import CLASSIFIER_THREADS_PER_WORKER, UtilityAlignedResidualFreshConfig
from .contracts import CENTERS, GENERATION_SEEDS, EvaluationPlan, TRAINING_SEEDS, expected_action_ids
from .policy_loading import FrozenUtilityAlignedPolicySurface
from .prediction_contracts import PREDICTION_TASK_SCHEMA, PredictionTaskSpec
from .source_cache import FreshSourceCache
from .target_surface import FreshTargetSurface


def build_prediction_tasks(
    config: UtilityAlignedResidualFreshConfig,
    *,
    plan: EvaluationPlan,
    policy: FrozenUtilityAlignedPolicySurface,
    source_cache: FreshSourceCache,
    target_surface: FreshTargetSurface,
    generation_lock_hash: str,
    root: Path,
    scratch_root: Path | None = None,
) -> tuple[PredictionTaskSpec, ...]:
    tasks: list[PredictionTaskSpec] = []
    for target in CENTERS:
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                task_id = f"target_{target}__train_{training_seed}__gen_{generation_seed}"
                actions = [
                    plan.action_for(target, action_id).to_payload()
                    for action_id in expected_action_ids(target)
                ]
                base = {
                    "schema_version": PREDICTION_TASK_SCHEMA,
                    "task_id": task_id,
                    "target_center": target,
                    "training_seed": training_seed,
                    "generation_seed": generation_seed,
                    "plan_hash": plan.plan_hash,
                    "policy_lock_hash": policy.policy_lock_hash,
                    "action_library_hash": policy.action_library_hash,
                    "source_cache_hash": source_cache.cache_hash,
                    "bank_lock_hash": source_cache.bank_lock_hash,
                    "generation_lock_hash": generation_lock_hash,
                    "target_cache_content_hash": target_surface.cache_content_hash,
                    "target_cache_protocol_hash": target_surface.cache_protocol_hash,
                    "reservation_hash": target_surface.reservation.reservation_hash,
                    "target_frame_sha256": target_surface.frames_by_center[target].file_sha256,
                    "source_cache_root": str(source_cache.root.resolve()),
                    "evaluation_array_path": str(
                        (
                            config.fresh_target_cache_root
                            / f"embeddings/by_center/center_{target}.npy"
                        ).resolve()
                    ),
                    "evaluation_row_ids": list(
                        target_surface.frames_by_center[target].evaluation_row_ids
                    ),
                    "classifier": config.classifier.to_payload(),
                    "threads_per_worker": CLASSIFIER_THREADS_PER_WORKER,
                    "actions": actions,
                    "labels_available_to_fit_or_predict": False,
                }
                task_hash = stable_hash(base)
                probability_path = root / f"arrays/{task_id}.probabilities.npy"
                metadata_path = root / f"tasks/{task_id}.json"
                work_probability = (
                    probability_path
                    if scratch_root is None
                    else scratch_root / f"arrays/{task_id}.probabilities.npy"
                )
                work_metadata = (
                    metadata_path
                    if scratch_root is None
                    else scratch_root / f"tasks/{task_id}.json"
                )
                tasks.append(
                    PredictionTaskSpec(
                        MappingProxyType(
                            {
                                **base,
                                "task_hash": task_hash,
                                "canonical_root": str(root.resolve()),
                                "metadata_path": str(metadata_path.resolve()),
                                "probability_path": str(probability_path.resolve()),
                                "work_metadata_path": str(work_metadata.resolve()),
                                "work_probability_path": str(work_probability.resolve()),
                                "scratch_root": (
                                    None
                                    if scratch_root is None
                                    else str(scratch_root.resolve())
                                ),
                            }
                        )
                    )
                )
    return tuple(tasks)


__all__ = ("build_prediction_tasks",)
