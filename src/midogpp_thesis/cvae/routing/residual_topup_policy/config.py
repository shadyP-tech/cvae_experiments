"""Exact configuration contract for the fresh B/U/G/S Stage-60 policy lock."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from ....common.hashing import stable_hash
from ...expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    TRAINING_SEEDS,
)
from ...protocol import ProtocolError


EXPERIMENT_ID = (
    "midogpp.routing_and_composition."
    "uniform_b_v2_residual_topup_b_u_g_s_policy_lock.v1"
)
EXPERIMENT_NAME = "uniform_b_v2_residual_topup_b_u_g_s_policy_lock_v1"
OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_residual_topup_b_u_g_s_policy_lock_v1"
)
EXPERT_BANK_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1"
)
GENERATION_LOCK_ARTIFACT_ID = "midogpp_output_uniform_b_v2_generation_lock_v1"
EQUAL_UNION_POLICY_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_equal_union_policy_lock_v1"
)
PROXY_SURFACE_ARTIFACT_ID = "midogpp_residual_topup_fresh_proxy_surface_v1"
INPUT_ARTIFACT_IDS = (
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    EQUAL_UNION_POLICY_ARTIFACT_ID,
    PROXY_SURFACE_ARTIFACT_ID,
)

EXPECTED_BANK_LOCK_HASH = "9972a41dcd4814cd"
EXPECTED_GENERATION_LOCK_HASH = "34e551425710362e"
EXPECTED_EQUAL_UNION_POLICY_LOCK_HASH = "4b9ea514308b084f"
CLAIM_SCOPE = "routing_and_composition"
STAGE_ID = "60_routing_and_composition"
PLANNED_STATUS = "BLOCKED_PENDING_FRESH_SURFACE"
DATASET_FAMILY = "MIDOG++"
FEATURE_BACKBONE = "Virchow2"
REPRESENTATION_ID = "annotation_jpeg_fixed_center_b_v3"

BASE_ACTION_ID = "base_equal_union"
UNIFORM_ACTION_ID = "uniform_residual_topup"
GLOBAL_ACTION_ID = "global_rank_residual_topup"
SUPPORT_ACTION_ID = "support_rank_residual_topup"
PERMUTATION_ACTION_ID = "support_rank_permutation_control"
SINGLE_SOURCE_ACTION_NAMESPACE = "single_source_tail"
MAIN_ACTION_IDS = (
    BASE_ACTION_ID,
    UNIFORM_ACTION_ID,
    GLOBAL_ACTION_ID,
    SUPPORT_ACTION_ID,
)

BASE_PER_SOURCE_PER_CLASS = 128
BASE_TOTAL_PER_CLASS = 1024
TOPUP_TOTAL_PER_CLASS = 128
MATCHED_TOTAL_PER_CLASS = 1152


@dataclass(frozen=True)
class ResidualTopupPolicyLockConfig:
    experiment_id: str
    name: str
    status: str
    artifact_root: Path
    expert_bank_root: Path
    generation_lock_root: Path
    equal_union_policy_root: Path
    proxy_surface_root: Path
    proxy_score_table_path: Path
    proxy_attestation_path: Path
    expert_bank_artifact_id: str
    generation_lock_artifact_id: str
    equal_union_policy_artifact_id: str
    proxy_surface_artifact_id: str
    expected_bank_lock_hash: str
    expected_generation_lock_hash: str
    expected_equal_union_policy_lock_hash: str
    protocol: Mapping[str, object]
    actions: Mapping[str, object]
    runtime: Mapping[str, object]
    claim_boundary: Mapping[str, object]
    config_source_path: Path

    @property
    def centers(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.protocol["centers"])  # type: ignore[index]

    @property
    def training_seeds(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.protocol["training_seeds"])  # type: ignore[index]

    @property
    def permutation_index(self) -> int:
        return int(self.actions["permutation_index"])

    @property
    def contract_hash(self) -> str:
        return stable_hash(
            {
                "experiment_id": self.experiment_id,
                "name": self.name,
                "status": self.status,
                "input_artifact_ids": list(INPUT_ARTIFACT_IDS),
                "expected_bank_lock_hash": self.expected_bank_lock_hash,
                "expected_generation_lock_hash": self.expected_generation_lock_hash,
                "expected_equal_union_policy_lock_hash": (
                    self.expected_equal_union_policy_lock_hash
                ),
                "protocol": dict(self.protocol),
                "actions": dict(self.actions),
                "runtime": dict(self.runtime),
                "claim_boundary": dict(self.claim_boundary),
            }
        )


def load_residual_topup_policy_lock_config(
    path: str | Path,
) -> ResidualTopupPolicyLockConfig:
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ProtocolError("Residual top-up policy-lock config must be a mapping.")
    _require_exact_keys(
        payload,
        {"experiment", "inputs", "protocol", "actions", "runtime", "claim_boundary"},
        "top-level config",
    )
    experiment = _mapping(payload, "experiment")
    inputs = _mapping(payload, "inputs")
    base = config_path.parent
    config = ResidualTopupPolicyLockConfig(
        experiment_id=str(experiment.get("id", "")),
        name=str(experiment.get("name", "")),
        status=str(experiment.get("status", "")),
        artifact_root=_path(base, experiment.get("artifact_root")),
        expert_bank_root=_path(base, inputs.get("expert_bank_root")),
        generation_lock_root=_path(base, inputs.get("generation_lock_root")),
        equal_union_policy_root=_path(base, inputs.get("equal_union_policy_root")),
        proxy_surface_root=_path(base, inputs.get("proxy_surface_root")),
        proxy_score_table_path=_path(base, inputs.get("proxy_score_table_path")),
        proxy_attestation_path=_path(base, inputs.get("proxy_attestation_path")),
        expert_bank_artifact_id=str(inputs.get("expert_bank_artifact_id", "")),
        generation_lock_artifact_id=str(
            inputs.get("generation_lock_artifact_id", "")
        ),
        equal_union_policy_artifact_id=str(
            inputs.get("equal_union_policy_artifact_id", "")
        ),
        proxy_surface_artifact_id=str(inputs.get("proxy_surface_artifact_id", "")),
        expected_bank_lock_hash=str(inputs.get("expected_bank_lock_hash", "")),
        expected_generation_lock_hash=str(
            inputs.get("expected_generation_lock_hash", "")
        ),
        expected_equal_union_policy_lock_hash=str(
            inputs.get("expected_equal_union_policy_lock_hash", "")
        ),
        protocol=dict(_mapping(payload, "protocol")),
        actions=dict(_mapping(payload, "actions")),
        runtime=dict(_mapping(payload, "runtime")),
        claim_boundary=dict(_mapping(payload, "claim_boundary")),
        config_source_path=config_path,
    )
    _validate(config, experiment=experiment, inputs=inputs)
    return config


def _validate(
    config: ResidualTopupPolicyLockConfig,
    *,
    experiment: Mapping[str, object],
    inputs: Mapping[str, object],
) -> None:
    _require_exact_keys(
        experiment, {"id", "name", "artifact_root", "status"}, "experiment"
    )
    _require_exact_keys(
        inputs,
        {
            "expert_bank_root",
            "generation_lock_root",
            "equal_union_policy_root",
            "proxy_surface_root",
            "proxy_score_table_path",
            "proxy_attestation_path",
            "expert_bank_artifact_id",
            "generation_lock_artifact_id",
            "equal_union_policy_artifact_id",
            "proxy_surface_artifact_id",
            "expected_bank_lock_hash",
            "expected_generation_lock_hash",
            "expected_equal_union_policy_lock_hash",
        },
        "inputs",
    )
    exact = {
        "experiment_id": (config.experiment_id, EXPERIMENT_ID),
        "name": (config.name, EXPERIMENT_NAME),
        "status": (config.status, PLANNED_STATUS),
        "expert_bank_artifact_id": (
            config.expert_bank_artifact_id,
            EXPERT_BANK_ARTIFACT_ID,
        ),
        "generation_lock_artifact_id": (
            config.generation_lock_artifact_id,
            GENERATION_LOCK_ARTIFACT_ID,
        ),
        "equal_union_policy_artifact_id": (
            config.equal_union_policy_artifact_id,
            EQUAL_UNION_POLICY_ARTIFACT_ID,
        ),
        "proxy_surface_artifact_id": (
            config.proxy_surface_artifact_id,
            PROXY_SURFACE_ARTIFACT_ID,
        ),
        "expected_bank_lock_hash": (
            config.expected_bank_lock_hash,
            EXPECTED_BANK_LOCK_HASH,
        ),
        "expected_generation_lock_hash": (
            config.expected_generation_lock_hash,
            EXPECTED_GENERATION_LOCK_HASH,
        ),
        "expected_equal_union_policy_lock_hash": (
            config.expected_equal_union_policy_lock_hash,
            EXPECTED_EQUAL_UNION_POLICY_LOCK_HASH,
        ),
        "centers": (config.centers, CENTERS),
        "training_seeds": (config.training_seeds, TRAINING_SEEDS),
    }
    mismatch = [
        f"{key}: observed={observed!r}, expected={expected!r}"
        for key, (observed, expected) in exact.items()
        if observed != expected
    ]
    if mismatch:
        raise ProtocolError(
            "Residual top-up policy-lock identity drifted: " + "; ".join(mismatch)
        )

    _require_exact_values(
        config.protocol,
        {
            "dataset_family": DATASET_FAMILY,
            "feature_backbone": FEATURE_BACKBONE,
            "feature_frame": REPRESENTATION_ID,
            "stage": STAGE_ID,
            "centers": list(CENTERS),
            "training_seeds": list(TRAINING_SEEDS),
            "query_to_decision_chain": (
                "query_to_proxy_compatibility_to_fixed_rank_action"
            ),
            "proxy_semantics": (
                "class_marginalized_common_space_reconstruction_mse_plus_"
                "latent_dim_normalized_analytic_ps_kl"
            ),
            "proxy_is_nelbo": False,
            "proxy_is_utility": False,
            "proxy_is_bacc_prediction": False,
            "class_hypothesis_prior": [0.5, 0.5],
            "replica_aggregation": (
                "average_all_three_training_replicas_before_each_case_ballot"
            ),
            "ballot": "normalized_true_midrank_lower_is_better",
            "global_rank_source": (
                "independently_reserved_unlabeled_pseudoqueries"
            ),
            "global_outer_exclusion": (
                "exclude_target_H_and_pseudo_target_q_before_each_ballot"
            ),
            "support_rank_source": "unlabeled_target_support_only",
            "support_evaluation_case_disjoint": True,
            "target_labels_used": False,
            "target_evaluation_used": False,
            "source_experts_updated": False,
            "hyperparameters_tuned": False,
        },
        "protocol",
    )
    _require_exact_values(
        config.actions,
        {
            "family": (
                "immutable_equal_union_backbone_with_fixed_residual_topup_b_u_g_s_v1"
            ),
            "main_action_ids": list(MAIN_ACTION_IDS),
            "permutation_control_action_id": PERMUTATION_ACTION_ID,
            "single_source_tail_action_namespace": SINGLE_SOURCE_ACTION_NAMESPACE,
            "base_per_source_per_class": BASE_PER_SOURCE_PER_CLASS,
            "base_total_per_class": BASE_TOTAL_PER_CLASS,
            "topup_total_per_class": TOPUP_TOTAL_PER_CLASS,
            "matched_total_per_class": MATCHED_TOTAL_PER_CLASS,
            "topup_fraction_of_base": 0.125,
            "rank_priority_transform": "one_minus_mean_normalized_midrank",
            "allocation": "hamilton_largest_remainder_canonical_source_ties",
            "freeze_all_H_by_e_single_source_tail_actions": True,
            "freeze_permutation_control": True,
            "permutation_scheme": (
                "canonical_source_order_nonzero_cyclic_rotation"
            ),
            "permutation_index": 1,
            "no_utility_selector": True,
            "no_fallback_gate": True,
            "no_empirical_bayes_shrinkage": True,
        },
        "actions",
    )
    _require_exact_values(
        config.runtime,
        {
            "workstation_profile": "xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb",
            "proxy_scoring_devices": ["cuda:0", "cuda:1"],
            "cuda_visible_devices": "0,1",
            "workers_per_device": 1,
            "score_batch_rows": 2048,
            "multiprocessing_start_method": "spawn",
            "tf32_disabled": True,
            "amp_enabled": False,
            "parent_cuda_context_forbidden": True,
            "checkpoint_policy": "hash_validated_case_score_checkpoints",
        },
        "runtime",
    )
    _require_exact_values(
        config.claim_boundary,
        {
            "strict_claim_firewall": True,
            "claim_scope": CLAIM_SCOPE,
            "current_checkout_has_eligible_fresh_surface": False,
            "policy_lock_may_materialize_only_after_fresh_surface_attestation": True,
            "consumed_stage70_used": False,
            "consumed_stage90_used": False,
            "routing_quality_claimed": False,
            "downstream_utility_claimed": False,
            "may_feed_stage70_only_after_validation_pass": True,
        },
        "claim boundary",
    )
    if (
        config.proxy_score_table_path.parent.parent.resolve()
        != config.proxy_surface_root.resolve()
        or config.proxy_attestation_path.parent.parent.resolve()
        != config.proxy_surface_root.resolve()
    ):
        raise ProtocolError(
            "Residual top-up proxy table and attestation must share the proxy-surface root."
        )


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Residual top-up policy-lock {key} must be a mapping.")
    return value


def _path(base: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ProtocolError("Residual top-up policy-lock path is invalid.")
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def _require_exact_keys(
    payload: Mapping[str, object], expected: set[str], label: str
) -> None:
    observed = set(payload)
    if observed != expected:
        raise ProtocolError(
            f"Residual top-up policy-lock {label} keys drifted: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}."
        )


def _require_exact_values(
    payload: Mapping[str, object], expected: Mapping[str, object], label: str
) -> None:
    _require_exact_keys(payload, set(expected), label)
    mismatch = [
        f"{key}: observed={payload.get(key)!r}, expected={value!r}"
        for key, value in expected.items()
        if payload.get(key) != value
    ]
    if mismatch:
        raise ProtocolError(
            f"Residual top-up policy-lock {label} drifted: " + "; ".join(mismatch)
        )


__all__ = (
    "BASE_ACTION_ID",
    "BASE_PER_SOURCE_PER_CLASS",
    "BASE_TOTAL_PER_CLASS",
    "CLAIM_SCOPE",
    "DATASET_FAMILY",
    "EQUAL_UNION_POLICY_ARTIFACT_ID",
    "EXPECTED_BANK_LOCK_HASH",
    "EXPECTED_EQUAL_UNION_POLICY_LOCK_HASH",
    "EXPECTED_GENERATION_LOCK_HASH",
    "EXPERIMENT_ID",
    "EXPERIMENT_NAME",
    "EXPERT_BANK_ARTIFACT_ID",
    "FEATURE_BACKBONE",
    "GENERATION_LOCK_ARTIFACT_ID",
    "GLOBAL_ACTION_ID",
    "INPUT_ARTIFACT_IDS",
    "MAIN_ACTION_IDS",
    "MATCHED_TOTAL_PER_CLASS",
    "OUTPUT_ARTIFACT_ID",
    "PERMUTATION_ACTION_ID",
    "PLANNED_STATUS",
    "PROXY_SURFACE_ARTIFACT_ID",
    "REPRESENTATION_ID",
    "SINGLE_SOURCE_ACTION_NAMESPACE",
    "STAGE_ID",
    "SUPPORT_ACTION_ID",
    "TOPUP_TOTAL_PER_CLASS",
    "UNIFORM_ACTION_ID",
    "ResidualTopupPolicyLockConfig",
    "load_residual_topup_policy_lock_config",
)
