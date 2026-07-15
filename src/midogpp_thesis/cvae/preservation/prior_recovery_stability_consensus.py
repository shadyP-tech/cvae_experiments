"""Pure training-seed wrapper and consensus contracts for source-inner locks."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

from ...real_features.classifier_reference.artifacts import stable_hash
from ...real_features.classifier_reference.protocol import ProtocolError
from ..generation_samplers import DIAGONAL_SAMPLER, FULL_SAMPLER, STANDARD_SAMPLER
from ..objectives import ISOTROPIC_OBJECTIVE, TASK_FISHER_OBJECTIVE
from .prior_recovery_config import STABILITY_CONSENSUS_RULE
from .source_inner_selection import RecipeLock, recipe_lock_from_payload


TRAINING_SEED_LOCK_SCHEMA = "midogpp_prior_recovery_training_seed_recipe_lock_v1"
CONSENSUS_LOCK_SCHEMA = "midogpp_prior_recovery_training_seed_consensus_lock_v1"


@dataclass(frozen=True)
class TrainingSeedRecipeLock:
    training_seed: int
    outer_target_center: str
    recipe_lock: RecipeLock
    seed_evidence_hash: str
    per_seed_contract_hash: str
    parent_protocol_hash: str
    checkpoint_hashes: tuple[str, ...]
    sampler_state_hashes: tuple[str, ...]

    @property
    def hash(self) -> str:
        return stable_hash(self.to_payload(include_hash=False))

    def to_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": TRAINING_SEED_LOCK_SCHEMA,
            "training_seed": int(self.training_seed),
            "outer_target_center": self.outer_target_center,
            "recipe_lock": self.recipe_lock.to_payload(),
            "recipe_lock_hash": self.recipe_lock.hash,
            "seed_evidence_hash": self.seed_evidence_hash,
            "per_seed_contract_hash": self.per_seed_contract_hash,
            "parent_protocol_hash": self.parent_protocol_hash,
            "checkpoint_hashes": list(self.checkpoint_hashes),
            "sampler_state_hashes": list(self.sampler_state_hashes),
            "claim_scope": "cvae_recipe_lock_only",
            "routing_performed": False,
            "composition_performed": False,
            "query_object": "none",
            "may_feed_deployable_selection": False,
        }
        if include_hash:
            payload["training_seed_recipe_lock_hash"] = self.hash
        return payload


@dataclass(frozen=True)
class TrainingSeedConsensusLock:
    outer_target_center: str
    integrity_status: str
    primary_arm: str
    objective_id: str
    sampler_family: str
    training_seeds: tuple[int, ...]
    seed_lock_hashes: Mapping[str, str]
    parent_protocol_hash: str
    parent_selection_bundle_hash: str
    consensus_rule_id: str
    consensus_origin: str
    stability_status: str
    recipe_export_ready: bool
    reason: str

    @property
    def hash(self) -> str:
        return stable_hash(self.to_payload(include_hash=False))

    def to_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": CONSENSUS_LOCK_SCHEMA,
            "outer_target_center": self.outer_target_center,
            "integrity_status": self.integrity_status,
            "primary_arm": self.primary_arm,
            "objective_id": self.objective_id,
            "sampler_family": self.sampler_family,
            "training_seeds": list(self.training_seeds),
            "seed_lock_hashes": dict(self.seed_lock_hashes),
            "parent_protocol_hash": self.parent_protocol_hash,
            "parent_selection_bundle_hash": self.parent_selection_bundle_hash,
            "consensus_rule_id": self.consensus_rule_id,
            "consensus_origin": self.consensus_origin,
            "stability_status": self.stability_status,
            "recipe_export_ready": self.recipe_export_ready,
            "reason": self.reason,
            "claim_scope": "cvae_recipe_lock_only",
            "claim_role": "training_seed_consensus_recipe_lock",
            "selection_source": "fully_nested_source_inner_training_seed_consensus",
            "source_inner_labels_used_for_selection": True,
            "target_eval_labels_used_for_selection": False,
            "target_eval_labels_used_for_scoring_only": False,
            "support_labels_used": False,
            "oracle_eligible": False,
            "may_feed_model_recipe": self.recipe_export_ready,
            "may_feed_deployable_selection": False,
            "routing_performed": False,
            "composition_performed": False,
            "query_object": "none",
        }
        if include_hash:
            payload["consensus_recipe_lock_hash"] = self.hash
        return payload


def select_training_seed_consensus(
    locks: Sequence[TrainingSeedRecipeLock],
    *,
    outer_target_center: str,
    training_seeds: Sequence[int],
    parent_protocol_hash: str,
    parent_selection_bundle_hash: str,
    consensus_rule_id: str = STABILITY_CONSENSUS_RULE,
) -> TrainingSeedConsensusLock:
    outer = str(outer_target_center)
    expected_seeds = tuple(int(seed) for seed in training_seeds)
    if consensus_rule_id != STABILITY_CONSENSUS_RULE:
        raise ProtocolError("Training-seed consensus rule is not the frozen v1 rule.")
    by_seed = {int(lock.training_seed): lock for lock in locks}
    if (
        len(by_seed) != len(locks)
        or tuple(seed for seed in expected_seeds if seed in by_seed) != expected_seeds
        or set(by_seed) != set(expected_seeds)
    ):
        raise ProtocolError("Training-seed lock coverage/order mismatch.")
    if any(
        lock.outer_target_center != outer
        or lock.recipe_lock.outer_target_center != outer
        for lock in by_seed.values()
    ):
        raise ProtocolError("Consensus cannot combine locks from different outer centers.")
    seed_hashes = {str(seed): by_seed[seed].hash for seed in expected_seeds}
    invalid = [
        seed
        for seed in expected_seeds
        if by_seed[seed].recipe_lock.status != "VALID"
    ]
    if invalid:
        return _consensus_lock(
            outer,
            training_seeds=expected_seeds,
            seed_lock_hashes=seed_hashes,
            parent_protocol_hash=parent_protocol_hash,
            parent_selection_bundle_hash=parent_selection_bundle_hash,
            integrity_status="INVALID",
            primary_arm="A",
            objective_id=ISOTROPIC_OBJECTIVE,
            sampler_family=STANDARD_SAMPLER,
            consensus_origin="INVALID_SEED_LOCK",
            stability_status="INVALID_SEED_LOCK",
            recipe_export_ready=False,
            reason="invalid_seed_lock:" + ",".join(str(seed) for seed in invalid),
        )
    child_locks = [by_seed[seed].recipe_lock for seed in expected_seeds]
    for lock in child_locks:
        _validate_child_recipe(lock)
    arms = [lock.primary_arm for lock in child_locks]
    if all(arm == "A" for arm in arms):
        return _consensus_lock(
            outer,
            training_seeds=expected_seeds,
            seed_lock_hashes=seed_hashes,
            parent_protocol_hash=parent_protocol_hash,
            parent_selection_bundle_hash=parent_selection_bundle_hash,
            integrity_status="VALID",
            primary_arm="A",
            objective_id=ISOTROPIC_OBJECTIVE,
            sampler_family=STANDARD_SAMPLER,
            consensus_origin="STABLE_UNANIMOUS",
            stability_status="STABLE_STANDARD_FALLBACK",
            recipe_export_ready=True,
            reason="unanimous_standard_normal",
        )
    if any(arm == "A" for arm in arms):
        return _divergence_fallback(
            outer,
            training_seeds=expected_seeds,
            seed_lock_hashes=seed_hashes,
            parent_protocol_hash=parent_protocol_hash,
            parent_selection_bundle_hash=parent_selection_bundle_hash,
            reason="mixed_standard_and_conditional",
        )
    families = {lock.sampler_family for lock in child_locks}
    if len(families) != 1:
        return _divergence_fallback(
            outer,
            training_seeds=expected_seeds,
            seed_lock_hashes=seed_hashes,
            parent_protocol_hash=parent_protocol_hash,
            parent_selection_bundle_hash=parent_selection_bundle_hash,
            reason="conditional_sampler_family_disagreement",
        )
    family = next(iter(families))
    if all(arm == "D" for arm in arms):
        return _consensus_lock(
            outer,
            training_seeds=expected_seeds,
            seed_lock_hashes=seed_hashes,
            parent_protocol_hash=parent_protocol_hash,
            parent_selection_bundle_hash=parent_selection_bundle_hash,
            integrity_status="VALID",
            primary_arm="D",
            objective_id=TASK_FISHER_OBJECTIVE,
            sampler_family=family,
            consensus_origin="STABLE_UNANIMOUS",
            stability_status="STABLE_CONDITIONAL",
            recipe_export_ready=True,
            reason="unanimous_task_fisher_conditional",
        )
    for lock in child_locks:
        sampler_summary = lock.gate_summary.get("sampler")
        if not isinstance(sampler_summary, Mapping):
            raise ProtocolError(
                "Conditional consensus lacks recomputable C-gate evidence."
            )
    mixed = len(set(arms)) > 1
    return _consensus_lock(
        outer,
        training_seeds=expected_seeds,
        seed_lock_hashes=seed_hashes,
        parent_protocol_hash=parent_protocol_hash,
        parent_selection_bundle_hash=parent_selection_bundle_hash,
        integrity_status="VALID",
        primary_arm="C",
        objective_id=ISOTROPIC_OBJECTIVE,
        sampler_family=family,
        consensus_origin=(
            "STABLE_SAMPLER_OBJECTIVE_DIVERGENCE"
            if mixed
            else "STABLE_UNANIMOUS"
        ),
        stability_status=(
            "STABLE_SAMPLER_OBJECTIVE_DIVERGENCE"
            if mixed
            else "STABLE_CONDITIONAL"
        ),
        recipe_export_ready=True,
        reason=(
            "unanimous_sampler_mixed_objective_fallback_to_isotropic"
            if mixed
            else "unanimous_isotropic_conditional"
        ),
    )


def write_training_seed_recipe_lock(path: Path, lock: TrainingSeedRecipeLock) -> None:
    _write_json(path, lock.to_payload())


def load_training_seed_recipe_lock(path: Path) -> TrainingSeedRecipeLock:
    payload = _read_json(path)
    if payload.get("schema_version") != TRAINING_SEED_LOCK_SCHEMA:
        raise ProtocolError("Unexpected training-seed RecipeLock schema.")
    recipe_payload = payload.get("recipe_lock")
    if not isinstance(recipe_payload, Mapping):
        raise ProtocolError("Training-seed RecipeLock lacks its child lock.")
    lock = TrainingSeedRecipeLock(
        training_seed=int(payload["training_seed"]),
        outer_target_center=str(payload["outer_target_center"]),
        recipe_lock=recipe_lock_from_payload(recipe_payload),
        seed_evidence_hash=str(payload["seed_evidence_hash"]),
        per_seed_contract_hash=str(payload["per_seed_contract_hash"]),
        parent_protocol_hash=str(payload["parent_protocol_hash"]),
        checkpoint_hashes=tuple(str(value) for value in payload["checkpoint_hashes"]),
        sampler_state_hashes=tuple(
            str(value) for value in payload["sampler_state_hashes"]
        ),
    )
    if payload.get("recipe_lock_hash") != lock.recipe_lock.hash:
        raise ProtocolError("Wrapped RecipeLock hash mismatch.")
    if payload.get("training_seed_recipe_lock_hash") != lock.hash:
        raise ProtocolError("Training-seed RecipeLock wrapper hash mismatch.")
    expected_flags = {
        "claim_scope": "cvae_recipe_lock_only",
        "routing_performed": False,
        "composition_performed": False,
        "query_object": "none",
        "may_feed_deployable_selection": False,
    }
    if any(payload.get(key) != value for key, value in expected_flags.items()):
        raise ProtocolError("Training-seed RecipeLock claim flags are inconsistent.")
    if not lock.checkpoint_hashes or not lock.sampler_state_hashes:
        raise ProtocolError("Training-seed RecipeLock lacks evidence identities.")
    return lock


def write_consensus_recipe_lock(path: Path, lock: TrainingSeedConsensusLock) -> None:
    _write_json(path, lock.to_payload())


def load_consensus_recipe_lock(path: Path) -> TrainingSeedConsensusLock:
    payload = _read_json(path)
    if payload.get("schema_version") != CONSENSUS_LOCK_SCHEMA:
        raise ProtocolError("Unexpected training-seed consensus schema.")
    if not isinstance(payload.get("recipe_export_ready"), bool):
        raise ProtocolError("Consensus recipe_export_ready must be a JSON boolean.")
    lock = TrainingSeedConsensusLock(
        outer_target_center=str(payload["outer_target_center"]),
        integrity_status=str(payload["integrity_status"]),
        primary_arm=str(payload["primary_arm"]),
        objective_id=str(payload["objective_id"]),
        sampler_family=str(payload["sampler_family"]),
        training_seeds=tuple(int(value) for value in payload["training_seeds"]),
        seed_lock_hashes={
            str(key): str(value)
            for key, value in dict(payload["seed_lock_hashes"]).items()
        },
        parent_protocol_hash=str(payload["parent_protocol_hash"]),
        parent_selection_bundle_hash=str(payload["parent_selection_bundle_hash"]),
        consensus_rule_id=str(payload["consensus_rule_id"]),
        consensus_origin=str(payload["consensus_origin"]),
        stability_status=str(payload["stability_status"]),
        recipe_export_ready=payload["recipe_export_ready"],
        reason=str(payload["reason"]),
    )
    if payload.get("consensus_recipe_lock_hash") != lock.hash:
        raise ProtocolError("Training-seed consensus lock hash mismatch.")
    _validate_consensus_flags(payload, lock)
    return lock


def _validate_child_recipe(lock: RecipeLock) -> None:
    allowed = {
        ("A", ISOTROPIC_OBJECTIVE, STANDARD_SAMPLER),
        ("C", ISOTROPIC_OBJECTIVE, DIAGONAL_SAMPLER),
        ("C", ISOTROPIC_OBJECTIVE, FULL_SAMPLER),
        ("D", TASK_FISHER_OBJECTIVE, DIAGONAL_SAMPLER),
        ("D", TASK_FISHER_OBJECTIVE, FULL_SAMPLER),
    }
    if (lock.primary_arm, lock.objective_id, lock.sampler_family) not in allowed:
        raise ProtocolError("Seed RecipeLock has a malformed arm/objective/sampler tuple.")


def _divergence_fallback(
    outer: str,
    *,
    training_seeds: tuple[int, ...],
    seed_lock_hashes: Mapping[str, str],
    parent_protocol_hash: str,
    parent_selection_bundle_hash: str,
    reason: str,
) -> TrainingSeedConsensusLock:
    return _consensus_lock(
        outer,
        training_seeds=training_seeds,
        seed_lock_hashes=seed_lock_hashes,
        parent_protocol_hash=parent_protocol_hash,
        parent_selection_bundle_hash=parent_selection_bundle_hash,
        integrity_status="VALID",
        primary_arm="A",
        objective_id=ISOTROPIC_OBJECTIVE,
        sampler_family=STANDARD_SAMPLER,
        consensus_origin="CONSERVATIVE_DIVERGENCE_FALLBACK",
        stability_status="CROSS_SEED_DISAGREEMENT",
        recipe_export_ready=True,
        reason=reason,
    )


def _consensus_lock(
    outer: str,
    *,
    training_seeds: tuple[int, ...],
    seed_lock_hashes: Mapping[str, str],
    parent_protocol_hash: str,
    parent_selection_bundle_hash: str,
    integrity_status: str,
    primary_arm: str,
    objective_id: str,
    sampler_family: str,
    consensus_origin: str,
    stability_status: str,
    recipe_export_ready: bool,
    reason: str,
) -> TrainingSeedConsensusLock:
    return TrainingSeedConsensusLock(
        outer_target_center=outer,
        integrity_status=integrity_status,
        primary_arm=primary_arm,
        objective_id=objective_id,
        sampler_family=sampler_family,
        training_seeds=training_seeds,
        seed_lock_hashes=dict(seed_lock_hashes),
        parent_protocol_hash=parent_protocol_hash,
        parent_selection_bundle_hash=parent_selection_bundle_hash,
        consensus_rule_id=STABILITY_CONSENSUS_RULE,
        consensus_origin=consensus_origin,
        stability_status=stability_status,
        recipe_export_ready=recipe_export_ready,
        reason=reason,
    )


def _validate_consensus_flags(
    payload: Mapping[str, object],
    lock: TrainingSeedConsensusLock,
) -> None:
    expected = {
        "claim_scope": "cvae_recipe_lock_only",
        "routing_performed": False,
        "composition_performed": False,
        "query_object": "none",
        "may_feed_deployable_selection": False,
        "may_feed_model_recipe": lock.recipe_export_ready,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ProtocolError("Consensus lock claim flags are inconsistent.")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Malformed training-seed lock JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Expected training-seed lock JSON object: {path}")
    return payload
