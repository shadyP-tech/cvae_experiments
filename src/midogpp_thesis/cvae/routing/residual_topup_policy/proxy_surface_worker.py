"""One-expert scoring worker and exact case-level aggregation."""

from __future__ import annotations

import gc
import math
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ..dense_residual_soft_router.compatibility import (
    CLASS_PRIOR as COMPATIBILITY_CLASS_PRIOR,
    ENERGY_SEMANTICS as COMPATIBILITY_ENERGY_SEMANTICS,
)
from .contracts import PROXY_ENERGY_SEMANTICS, FreshProxyScoreRow
from .proxy_surface_checkpoints import write_fresh_proxy_score_checkpoint
from .proxy_surface_contracts import (
    ArrayLoader,
    FreshProxyScoreTask,
    FreshProxyTaskResult,
    default_array_loader,
)
from .proxy_surface_validation import (
    deduplicated_task_scoring_groups,
    load_shard_embeddings,
)


ExpertLoader = Callable[..., object]
CompatibilityScorer = Callable[[object, np.ndarray, Sequence[str]], object]


def execute_fresh_proxy_score_task(
    task: FreshProxyScoreTask,
    *,
    array_loader: ArrayLoader | None = None,
    expert_loader: ExpertLoader | None = None,
    scorer: CompatibilityScorer | None = None,
) -> FreshProxyTaskResult:
    """Load one routing-authorized expert once and score all legal shards."""

    if not isinstance(task, FreshProxyScoreTask):
        raise ProtocolError("Fresh proxy worker received an invalid task.")
    active_array_loader = array_loader or default_array_loader
    active_expert_loader = expert_loader or default_expert_loader
    active_scorer = scorer or default_compatibility_scorer
    configure_worker_device(task.device)
    expert = active_expert_loader(
        task.expert_bank_root,
        source_center=task.source_center,
        training_seed=task.training_seed,
        device=task.device,
    )
    try:
        validate_expert_identity(expert, task)
        expert_lock_hash = str(getattr(expert, "expert_lock_hash", ""))
        expert_checkpoint_hash = str(getattr(expert, "checkpoint_hash", ""))
        if not expert_lock_hash or not expert_checkpoint_hash:
            raise ProtocolError("Fresh proxy expert lacks immutable routing hashes.")
        rows: list[FreshProxyScoreRow] = []
        scoring_groups = deduplicated_task_scoring_groups(task)
        for shard, replicated_shards in scoring_groups:
            embeddings = load_shard_embeddings(
                shard, array_loader=active_array_loader
            )
            # Distinct paths may attest the same canonical pseudoquery.  Verify
            # each physical alias once, but never repeat the GPU score.
            checked_paths = {shard.embedding_path}
            for alias in replicated_shards[1:]:
                if alias.embedding_path not in checked_paths:
                    load_shard_embeddings(
                        alias, array_loader=active_array_loader
                    )
                    checked_paths.add(alias.embedding_path)
            per_case = score_shard_exactly(
                expert,
                embeddings,
                shard.case_ids,
                scorer=active_scorer,
                chunk_rows=task.chunk_rows,
                expected_source=task.source_center,
                expected_training_seed=task.training_seed,
            )
            for output_shard in replicated_shards:
                for case_id in output_shard.unique_case_ids:
                    rows.append(
                        FreshProxyScoreRow(
                            outer_target=output_shard.outer_target,
                            query_role=output_shard.query_role,
                            query_center=output_shard.query_center,
                            case_id=case_id,
                            candidate_source=task.source_center,
                            training_seed=task.training_seed,
                            proxy_energy=per_case[case_id],
                            labels_consumed=False,
                            evaluation_overlap=False,
                            source_expert_updated=False,
                            proxy_energy_semantics=PROXY_ENERGY_SEMANTICS,
                        )
                    )
        validate_expert_identity(expert, task)
        if (
            str(getattr(expert, "expert_lock_hash", "")) != expert_lock_hash
            or str(getattr(expert, "checkpoint_hash", ""))
            != expert_checkpoint_hash
        ):
            raise ProtocolError("Fresh proxy scorer mutated expert identity state.")
        return write_fresh_proxy_score_checkpoint(
            task,
            rows=rows,
            expert_lock_hash=expert_lock_hash,
            expert_checkpoint_hash=expert_checkpoint_hash,
        )
    finally:
        del expert
        gc.collect()
        empty_device_cache(task.device)


def score_shard_exactly(
    expert: object,
    embeddings: np.ndarray,
    case_ids: Sequence[str],
    *,
    scorer: CompatibilityScorer,
    chunk_rows: int,
    expected_source: str,
    expected_training_seed: int,
) -> Mapping[str, float]:
    row_energies: list[np.ndarray] = []
    mapping_result: Mapping[str, float] | None = None
    chunk_count = math.ceil(len(embeddings) / chunk_rows)
    for start in range(0, len(embeddings), chunk_rows):
        stop = min(start + chunk_rows, len(embeddings))
        chunk_cases = tuple(case_ids[start:stop])
        result = scorer(expert, embeddings[start:stop], chunk_cases)
        validate_score_result_attestations(
            result,
            expected_source=expected_source,
            expected_training_seed=expected_training_seed,
        )
        per_row = getattr(result, "per_row", None)
        if per_row is not None:
            values = np.asarray(per_row, dtype=np.float64)
            if values.shape != (stop - start,) or not np.isfinite(values).all():
                raise ProtocolError(
                    "Fresh proxy scorer returned invalid row energies."
                )
            row_energies.append(values)
            continue
        if chunk_count != 1:
            raise ProtocolError(
                "Chunked fresh proxy scoring requires per-row energies for exact "
                "case aggregation."
            )
        raw_mapping = (
            result
            if isinstance(result, Mapping)
            else getattr(result, "per_case", None)
        )
        if not isinstance(raw_mapping, Mapping):
            raise ProtocolError("Fresh proxy scorer returned no auditable energies.")
        normalized = {str(key): float(value) for key, value in raw_mapping.items()}
        if set(normalized) != set(case_ids) or not all(
            math.isfinite(value) for value in normalized.values()
        ):
            raise ProtocolError("Fresh proxy scorer returned invalid case energies.")
        mapping_result = MappingProxyType(normalized)
    if mapping_result is not None:
        return mapping_result
    combined = np.concatenate(row_energies, axis=0)
    if combined.shape != (len(case_ids),):
        raise ProtocolError("Fresh proxy chunk reconstruction lost query rows.")
    cases = np.asarray(tuple(case_ids), dtype=object)
    return MappingProxyType(
        {
            case_id: float(np.mean(combined[cases == case_id], dtype=np.float64))
            for case_id in sorted(set(case_ids))
        }
    )


def validate_score_result_attestations(
    result: object,
    *,
    expected_source: str,
    expected_training_seed: int,
) -> None:
    source = getattr(result, "source_center", None)
    seed = getattr(result, "training_seed", None)
    if source is None or seed is None:
        raise ProtocolError("Fresh proxy scorer identity attestation is absent.")
    try:
        seed_matches = int(seed) == expected_training_seed
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError("Fresh proxy scorer/expert identity drifted.") from exc
    if str(source) != expected_source or not seed_matches:
        raise ProtocolError("Fresh proxy scorer/expert identity drifted.")
    if getattr(result, "exact_nelbo", None) is not False:
        raise ProtocolError(
            "Fresh proxy scorer must explicitly attest exact_nelbo=False."
        )
    if getattr(result, "labels_consumed", None) is not False:
        raise ProtocolError(
            "Fresh proxy scorer must explicitly attest labels_consumed=False."
        )
    if (
        getattr(result, "energy_semantics", None)
        != COMPATIBILITY_ENERGY_SEMANTICS
    ):
        raise ProtocolError("Fresh proxy compatibility-energy semantics drifted.")
    raw_prior = getattr(result, "class_prior", None)
    try:
        class_prior = tuple(float(value) for value in raw_prior)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError("Fresh proxy class-prior attestation is absent.") from exc
    if class_prior != tuple(COMPATIBILITY_CLASS_PRIOR):
        raise ProtocolError("Fresh proxy class-prior semantics drifted.")


def validate_expert_identity(expert: object, task: FreshProxyScoreTask) -> None:
    if (
        str(getattr(expert, "source_center", "")) != task.source_center
        or int(getattr(expert, "training_seed", -1)) != task.training_seed
    ):
        raise ProtocolError("Fresh proxy worker loaded the wrong expert replica.")


def configure_worker_device(device: str) -> None:
    if not device.startswith("cuda"):
        return
    import torch

    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def empty_device_cache(device: str) -> None:
    if not device.startswith("cuda"):
        return
    try:
        import torch

        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


def default_expert_loader(
    root: Path,
    *,
    source_center: str,
    training_seed: int,
    device: str,
) -> object:
    from ...expert_bank.uniform_b_v2_promotion.serialization import (
        load_routing_authorized_expert,
    )

    return load_routing_authorized_expert(
        root,
        source_center=source_center,
        training_seed=training_seed,
        device=device,
    )


def default_compatibility_scorer(
    expert: object,
    embeddings: np.ndarray,
    case_ids: Sequence[str],
) -> object:
    from ..dense_residual_soft_router.compatibility import (
        score_variational_compatibility,
    )

    return score_variational_compatibility(expert, embeddings, case_ids)


__all__ = (
    "CompatibilityScorer",
    "ExpertLoader",
    "execute_fresh_proxy_score_task",
)
