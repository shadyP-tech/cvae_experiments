"""Frozen identities and row contracts for source-inner candidate utility.

This artifact deliberately has no outer-target variable.  It opens the
authorized validation labels once, after prediction, to create the complete
source-inner ``q != e`` utility surface needed by one predeclared future policy
family.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping

from ....common.hashing import stable_hash
from ....data.features.uniform_b_routing_validation.config import (
    CACHE_NAME as VALIDATION_CACHE_SEMANTIC_ID,
    REPRESENTATION_ID as VALIDATION_CACHE_REPRESENTATION_ID,
)
from ...expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    TRAINING_SEEDS,
)
from ...generation.contracts import (
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_GENERATION_LOCK_HASH,
    GENERATION_SEEDS,
)
from ...protocol import ProtocolError


EXPERIMENT_ID = (
    "midogpp.routing_and_composition."
    "uniform_b_v2_source_inner_candidate_utility.v1"
)
EXPERIMENT_NAME = "uniform_b_v2_source_inner_candidate_utility_v1"
OUTPUT_ARTIFACT_ID = "midogpp_output_uniform_b_v2_source_inner_candidate_utility_v1"
CLAIM_SCOPE = "routing_and_composition"

EXPERT_BANK_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1"
)
GENERATION_LOCK_ARTIFACT_ID = "midogpp_output_uniform_b_v2_generation_lock_v1"
VALIDATION_CACHE_ARTIFACT_ID = (
    "midogpp_virchow2_uniform_b_v2_routing_validation_cache_seed42"
)
VALIDATION_MANIFEST_ARTIFACT_ID = "midogpp_source_inner_validation_manifest_v1"
EQUAL_UNION_POLICY_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_equal_union_policy_lock_v1"
)

MANIFEST_MEMBER = "manifest.csv"
EXPECTED_MANIFEST_SHA256 = (
    "db661ac7e3dbafde8e283528de6706ab35f2c26629b389706c4504e458cc5869"
)
FEATURE_DIM = 3840
FULL_SOURCE_BUDGET_PER_CLASS = 1024
EXPECTED_EVAL_ROWS = 2615
EXPECTED_EVAL_CASES = 44
EXPECTED_FIT_COUNT = 81
EXPECTED_UTILITY_ROW_COUNT = 648
EXPECTED_CASE_CONFUSION_ROW_COUNT = 3168
EXCLUDED_CENTER = "4"

CLASSIFIER_FAMILY = "sklearn_logistic_regression"
CLASSIFIER_C = 0.01
CLASSIFIER_PENALTY = "l2"
CLASSIFIER_SOLVER = "lbfgs"
CLASSIFIER_MAX_ITER = 3000
CLASSIFIER_RANDOM_STATE = 23
CLASSIFIER_THRESHOLD_POLICY = "predict"
CLASSIFIER_SCALER_FIT = "synthetic_train_only"

UTILITY_POLICY_FAMILY = (
    "source_inner_mean_paired_bacc_regret_"
    "three_level_case_cluster_bootstrap_v1"
)
PRIMARY_UTILITY_METRIC = "balanced_accuracy"
PRIMARY_POLICY_OBJECTIVE = "mean_paired_bacc_regret"
SECONDARY_METRIC = "macro_f1_descriptive_only"
BOOTSTRAP_LEVELS = (
    "pseudo_target_centers",
    "cases_within_pseudo_target",
    "paired_training_generation_seed_cells",
)
BOOTSTRAP_VALID_REPLICATES = 2000
BOOTSTRAP_MAX_ATTEMPTS = 20000
BOOTSTRAP_SEED = 6042026
BOOTSTRAP_LOWER_QUANTILE = 0.025
UNIQUE_WINNER_PROBABILITY_MIN = 0.80
PAIRED_MARGIN_LOWER_BOUND = 0.0
FALLBACK_POLICY = "exact_equal_union"

# Patched after the canonical config is materialized.  Keeping this identity in
# source prevents silent scientific-contract edits while allowing runtime paths
# and observed cache hashes to remain machine-specific.
EXPECTED_CONFIG_CONTRACT_HASH = "7eaa6438d04116e8"

def policy_consumption_lock_payload() -> dict[str, object]:
    """Return the sole policy rule authorized by this label consumption.

    The payload is path-free and importable by the future policy artifact so
    the consumer cannot silently redefine regret, uncertainty, gates, or the
    fallback after source-inner validation labels have been opened.
    """

    return {
        "schema_version": "midogpp_uniform_b_v2_policy_consumption_lock_v1",
        "policy_family": UTILITY_POLICY_FAMILY,
        "label_consumption_scope": (
            "single_predeclared_source_inner_utility_regret_policy_family"
        ),
        "candidate_unit": "source_center",
        "candidate_replica_handling": (
            "pair_all_nine_training_generation_seed_cells_no_seed_selection"
        ),
        "primary_metric": PRIMARY_UTILITY_METRIC,
        "primary_objective": PRIMARY_POLICY_OBJECTIVE,
        "regret_definition": (
            "paired_best_candidate_bacc_within_q_training_generation_cell_"
            "minus_candidate_bacc"
        ),
        "regret_aggregation": (
            "equal_mean_over_retained_pseudo_targets_and_paired_"
            "training_generation_seed_cells"
        ),
        "secondary_metric": SECONDARY_METRIC,
        "secondary_metric_may_select": False,
        "future_outer_filter": {
            "rule": "remove_q_equal_H_and_e_equal_H_before_policy_computation",
            "outer_H_instantiated_in_utility_artifact": False,
            "apply_before_bootstrap": True,
        },
        "bootstrap": {
            "family": "paired_three_level_case_cluster_bootstrap",
            "levels": list(BOOTSTRAP_LEVELS),
            "case_sampling": "with_replacement_within_resampled_pseudo_target",
            "case_confusion_reaggregation": (
                "sum_tn_fp_fn_tp_then_recompute_bacc"
            ),
            "class_missing_resample_policy": "reject_and_resample",
            "seed_cell_sampling": (
                "with_replacement_over_paired_training_generation_cells"
            ),
            "valid_replicates": BOOTSTRAP_VALID_REPLICATES,
            "max_attempts": BOOTSTRAP_MAX_ATTEMPTS,
            "seed": BOOTSTRAP_SEED,
            "interval": "percentile",
            "paired_margin_lower_quantile": BOOTSTRAP_LOWER_QUANTILE,
        },
        "winner_rule": {
            "candidate_direction": "minimum_mean_paired_bacc_regret",
            "unique_winner_required": True,
            "minimum_unique_winner_probability": UNIQUE_WINNER_PROBABILITY_MIN,
            "probability_comparator": ">=",
            "paired_margin_definition": (
                "runner_up_mean_paired_bacc_regret_minus_winner_"
                "mean_paired_bacc_regret"
            ),
            "paired_margin_lower_quantile": BOOTSTRAP_LOWER_QUANTILE,
            "paired_margin_lower_bound": PAIRED_MARGIN_LOWER_BOUND,
            "margin_comparator": ">",
        },
        "pass_action": {
            "policy": "unique_winner_single_source_full_budget",
            "source_budget_per_class": FULL_SOURCE_BUDGET_PER_CLASS,
            "source_prefix_start_per_class": 0,
            "generation_stream_reused_from_generation_lock": True,
        },
        "fallback_policy": FALLBACK_POLICY,
        "fallback_policy_artifact_id": EQUAL_UNION_POLICY_ARTIFACT_ID,
        "fallback_is_exact_not_reestimated": True,
        "alternative_router_tuning_authorized": False,
        "alternative_policy_family_authorized": False,
        "hyperparameter_search_authorized": False,
        "seed_selection_authorized": False,
        "macro_f1_selection_authorized": False,
        "policy_selection_performed_in_utility_artifact": False,
    }


POLICY_CONSUMPTION_LOCK_HASH = stable_hash(policy_consumption_lock_payload())

OUTPUT_SEMANTIC_IDENTITIES = {
    "utility_lock_contract": "midogpp_uniform_b_v2_source_inner_utility_lock_v1",
    "config_contract_hash": EXPECTED_CONFIG_CONTRACT_HASH,
    "policy_consumption_lock_hash": POLICY_CONSUMPTION_LOCK_HASH,
    "generation_lock_hash": EXPECTED_GENERATION_LOCK_HASH,
    "expert_bank_lock_hash": EXPECTED_BANK_LOCK_HASH,
    "validation_cache_contract": VALIDATION_CACHE_SEMANTIC_ID,
    "validation_cache_representation": VALIDATION_CACHE_REPRESENTATION_ID,
    "validation_manifest_sha256": EXPECTED_MANIFEST_SHA256,
    "candidate_utility_row_count": str(EXPECTED_UTILITY_ROW_COUNT),
    "case_confusion_row_count": str(EXPECTED_CASE_CONFUSION_ROW_COUNT),
}


def candidate_sources(pseudo_target_center: str) -> tuple[str, ...]:
    """Canonical target-excluded candidate order for one pseudo-target."""

    target = str(pseudo_target_center)
    if target not in CENTERS:
        raise ProtocolError(f"Unknown source-inner pseudo-target center: {target!r}.")
    return tuple(center for center in CENTERS if center != target)


def expected_fit_keys() -> tuple[tuple[str, int, int], ...]:
    """All source/training/generation cells; no seed selection is possible."""

    return tuple(product(CENTERS, TRAINING_SEEDS, GENERATION_SEEDS))


def expected_utility_keys() -> tuple[tuple[str, str, int, int], ...]:
    """All ordered ``q != e`` source-inner utility cells."""

    return tuple(
        (pseudo_target, candidate, training_seed, generation_seed)
        for candidate, training_seed, generation_seed in expected_fit_keys()
        for pseudo_target in CENTERS
        if pseudo_target != candidate
    )


@dataclass(frozen=True)
class EvaluationRow:
    """Label-free identity for one ordered validation embedding."""

    row_ordinal: int
    manifest_row_index: int
    sample_id: str
    case_id: str
    center: str
    split: str
    cache_shard_path: str
    cache_row_index: int

    def __post_init__(self) -> None:
        if self.row_ordinal < 0 or self.manifest_row_index < 0 or self.cache_row_index < 0:
            raise ProtocolError("Evaluation row indices must be nonnegative.")
        if not self.sample_id or not self.case_id:
            raise ProtocolError("Evaluation rows require sample and case identities.")
        if self.center not in CENTERS or self.center == EXCLUDED_CENTER:
            raise ProtocolError("Evaluation row center is outside the eligible source pool.")
        if self.split != "val":
            raise ProtocolError("Source-inner utility may consume validation rows only.")
        if not self.cache_shard_path:
            raise ProtocolError("Evaluation row lacks its cache-shard binding.")

    def identity_payload(self) -> dict[str, object]:
        return {
            "row_ordinal": self.row_ordinal,
            "manifest_row_index": self.manifest_row_index,
            "sample_id": self.sample_id,
            "case_id": self.case_id,
            "center": self.center,
            "split": self.split,
            "cache_shard_path": self.cache_shard_path,
            "cache_row_index": self.cache_row_index,
        }


@dataclass(frozen=True)
class SourceIdentity:
    """GenerationLock-bound source expert/frame/sampler identity."""

    source_center: str
    training_seed: int
    expert_lock_hash: str
    checkpoint_hash: str
    checkpoint_file_sha256: str
    frame_hash: str
    frame_file_sha256: str
    sampler_state_hash: str
    sampler_file_sha256: str

    def __post_init__(self) -> None:
        if self.source_center not in CENTERS or self.training_seed not in TRAINING_SEEDS:
            raise ProtocolError("Source identity is outside the frozen expert bank.")
        for field in (
            self.expert_lock_hash,
            self.checkpoint_hash,
            self.checkpoint_file_sha256,
            self.frame_hash,
            self.frame_file_sha256,
            self.sampler_state_hash,
            self.sampler_file_sha256,
        ):
            if not field:
                raise ProtocolError("Source identity contains an empty provenance hash.")

    def to_payload(self) -> dict[str, object]:
        return {
            "source_center": self.source_center,
            "training_seed": self.training_seed,
            "expert_lock_hash": self.expert_lock_hash,
            "checkpoint_hash": self.checkpoint_hash,
            "checkpoint_file_sha256": self.checkpoint_file_sha256,
            "frame_hash": self.frame_hash,
            "frame_file_sha256": self.frame_file_sha256,
            "sampler_state_hash": self.sampler_state_hash,
            "sampler_file_sha256": self.sampler_file_sha256,
        }


def source_identities_from_generation_lock(
    payload: Mapping[str, object],
) -> dict[tuple[str, int], SourceIdentity]:
    bank = payload.get("bank")
    if not isinstance(bank, Mapping):
        raise ProtocolError("GenerationLock lacks its bank identity.")
    records = bank.get("expert_locks")
    if not isinstance(records, list):
        raise ProtocolError("GenerationLock lacks expert locks.")
    identities: dict[tuple[str, int], SourceIdentity] = {}
    for raw in records:
        if not isinstance(raw, Mapping):
            raise ProtocolError("GenerationLock contains a malformed expert lock.")
        identity = SourceIdentity(
            source_center=str(raw.get("source_center", "")),
            training_seed=int(raw.get("training_seed", -1)),
            expert_lock_hash=str(raw.get("expert_lock_hash", "")),
            checkpoint_hash=str(raw.get("checkpoint_hash", "")),
            checkpoint_file_sha256=str(raw.get("checkpoint_file_sha256", "")),
            frame_hash=str(raw.get("frame_hash", "")),
            frame_file_sha256=str(raw.get("frame_file_sha256", "")),
            sampler_state_hash=str(raw.get("sampler_state_hash", "")),
            sampler_file_sha256=str(raw.get("sampler_file_sha256", "")),
        )
        key = (identity.source_center, identity.training_seed)
        if key in identities:
            raise ProtocolError("GenerationLock duplicates an expert identity.")
        identities[key] = identity
    expected = {(center, seed) for center in CENTERS for seed in TRAINING_SEEDS}
    if set(identities) != expected:
        raise ProtocolError("GenerationLock expert identity coverage drifted.")
    return identities


def evaluation_order_hash(rows: tuple[EvaluationRow, ...]) -> str:
    return stable_hash([row.identity_payload() for row in rows])


__all__ = (
    "BOOTSTRAP_LEVELS",
    "BOOTSTRAP_LOWER_QUANTILE",
    "BOOTSTRAP_MAX_ATTEMPTS",
    "BOOTSTRAP_SEED",
    "BOOTSTRAP_VALID_REPLICATES",
    "CENTERS",
    "CLAIM_SCOPE",
    "CLASSIFIER_C",
    "CLASSIFIER_FAMILY",
    "CLASSIFIER_MAX_ITER",
    "CLASSIFIER_PENALTY",
    "CLASSIFIER_RANDOM_STATE",
    "CLASSIFIER_SCALER_FIT",
    "CLASSIFIER_SOLVER",
    "CLASSIFIER_THRESHOLD_POLICY",
    "EQUAL_UNION_POLICY_ARTIFACT_ID",
    "EXPECTED_BANK_LOCK_HASH",
    "EXPECTED_CASE_CONFUSION_ROW_COUNT",
    "EXPECTED_CONFIG_CONTRACT_HASH",
    "EXPECTED_EVAL_CASES",
    "EXPECTED_EVAL_ROWS",
    "EXPECTED_FIT_COUNT",
    "EXPECTED_GENERATION_LOCK_HASH",
    "EXPECTED_MANIFEST_SHA256",
    "EXPECTED_UTILITY_ROW_COUNT",
    "EXCLUDED_CENTER",
    "EXPERIMENT_ID",
    "EXPERIMENT_NAME",
    "EXPERT_BANK_ARTIFACT_ID",
    "EvaluationRow",
    "FALLBACK_POLICY",
    "FEATURE_DIM",
    "FULL_SOURCE_BUDGET_PER_CLASS",
    "GENERATION_LOCK_ARTIFACT_ID",
    "GENERATION_SEEDS",
    "MANIFEST_MEMBER",
    "OUTPUT_ARTIFACT_ID",
    "OUTPUT_SEMANTIC_IDENTITIES",
    "PAIRED_MARGIN_LOWER_BOUND",
    "POLICY_CONSUMPTION_LOCK_HASH",
    "PRIMARY_POLICY_OBJECTIVE",
    "PRIMARY_UTILITY_METRIC",
    "SECONDARY_METRIC",
    "SourceIdentity",
    "TRAINING_SEEDS",
    "UNIQUE_WINNER_PROBABILITY_MIN",
    "UTILITY_POLICY_FAMILY",
    "VALIDATION_CACHE_ARTIFACT_ID",
    "VALIDATION_CACHE_REPRESENTATION_ID",
    "VALIDATION_CACHE_SEMANTIC_ID",
    "VALIDATION_MANIFEST_ARTIFACT_ID",
    "candidate_sources",
    "evaluation_order_hash",
    "expected_fit_keys",
    "expected_utility_keys",
    "policy_consumption_lock_payload",
    "source_identities_from_generation_lock",
)
