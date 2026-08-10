"""Fail-closed protocol for the terminal consumed-test mechanism diagnostic."""

from __future__ import annotations

from dataclasses import dataclass

from ...protocol import ProtocolError
from .hashing import canonical_hash
from .experiment_contracts import (
    A1_OTHER_ROW_WEIGHT,
    A1_SELECTED_ROW_WEIGHT,
    AUTHORIZATION_SCOPE,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CLAIM_ROLE,
    GEOMETRY_IDS,
    PER_GEOMETRY_METHOD_IDS,
    PRE_EVALUATION_METHOD_IDS,
    PUBLICATION_STATUS,
    TERMINAL_ORACLE_IDS,
)


@dataclass(frozen=True)
class ActionabilityRecoverabilityProtocol:
    evidence_status: str
    consumed_test_data: bool
    fresh_evidence: bool
    may_authorize_routing: bool
    may_authorize_promotion: bool
    may_feed_another_experiment: bool
    geometry_ids: tuple[str, ...]
    pre_evaluation_method_ids: tuple[str, ...]
    per_geometry_method_ids: tuple[str, ...]
    terminal_oracle_ids: tuple[str, ...]
    workstation_profile: str
    cpu_workers: int
    threads_per_worker: int
    bootstrap_replicates: int
    bootstrap_seed: int
    multiprocessing_start_method: str
    source_generation_devices: tuple[str, ...]
    scientific_reduction_dtype: str
    surface_storage_dtype: str
    contract_hash: str

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": (
                "midogpp_fixed_bank_actionability_recoverability_protocol_v1"
            ),
            "dataset_family": "MIDOG++",
            "claim_role": CLAIM_ROLE,
            "authorization_scope": AUTHORIZATION_SCOPE,
            "evidence_status": self.evidence_status,
            "consumed_test_data": self.consumed_test_data,
            "fresh_evidence": self.fresh_evidence,
            "method_development_is_posthoc": True,
            "may_authorize_routing": self.may_authorize_routing,
            "may_authorize_promotion": self.may_authorize_promotion,
            "may_authorize_action_selection": False,
            "may_authorize_action_geometry_update": False,
            "may_authorize_policy_update": False,
            "may_authorize_model_update": False,
            "may_authorize_expert_update": False,
            "may_authorize_deployment": False,
            "may_feed_stage50": False,
            "may_feed_stage60": False,
            "may_feed_stage70": False,
            "may_feed_another_stage90": False,
            "may_feed_another_experiment": self.may_feed_another_experiment,
            "target_expert_used": False,
            "source_expert_updated": False,
            "target_labels_update_shared_model": False,
            "candidate_generalization": "known_fixed_bank_reuse",
            "unseen_expert_transfer_claim": False,
            "geometry_ids": list(self.geometry_ids),
            "geometry_selection_used": False,
            "action_strength_sweep_used": False,
            "class_conditional_action_variant_used": False,
            "source_pair_action_used": False,
            "A1_reuses_exact_A0_rows": True,
            "A1_selected_row_weight": A1_SELECTED_ROW_WEIGHT,
            "A1_other_row_weight": A1_OTHER_ROW_WEIGHT,
            "pre_evaluation_method_ids": list(self.pre_evaluation_method_ids),
            "per_geometry_method_ids": list(self.per_geometry_method_ids),
            "terminal_oracle_ids": list(self.terminal_oracle_ids),
            "terminal_oracles_available_before_evaluation_labels": False,
            "method_rows_carry_geometry_id": True,
            "all_action_probabilities_sealed_before_any_label_access": True,
            "prelabel_features_and_shared_models_sealed_before_target_support": (
                True
            ),
            "all_pre_evaluation_method_decisions_sealed_before_evaluation_labels": (
                True
            ),
            "other_center_label_use": (
                "strict_outer_H_and_nested_query_q_shared_model_fit_only"
            ),
            "same_target_support_use": (
                "S_y_action_selection_within_each_frozen_geometry_only"
            ),
            "evaluation_label_use": (
                "terminal_scoring_complementarity_rank_stability_and_oracles_only"
            ),
            "routing_lcb_relaxed": False,
            "terminal_evaluation_runtime": {
                "bootstrap_replicates": self.bootstrap_replicates,
                "bootstrap_seed": self.bootstrap_seed,
                "multiprocessing_start_method": self.multiprocessing_start_method,
            },
            "workstation": {
                "profile": self.workstation_profile,
                "source_generation_devices": list(
                    self.source_generation_devices
                ),
                "source_generation_workers": 2,
                "source_workers_per_device": 1,
                "gpu_phase_precedes_cpu_phase": True,
                "probability_materialization_device": "cpu",
                "cpu_workers": self.cpu_workers,
                "threads_per_worker": self.threads_per_worker,
                "multiprocessing_start_method": self.multiprocessing_start_method,
                "gpu_and_cpu_phases_disjoint": True,
                "parent_cuda_context_forbidden_during_cpu_phase": True,
                "scientific_reduction_dtype": self.scientific_reduction_dtype,
                "surface_storage_dtype": self.surface_storage_dtype,
                "local_scratch_preferred": True,
                "hash_validated_resume": True,
            },
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "contract_hash": self.contract_hash}


def canonical_consumed_test_protocol() -> ActionabilityRecoverabilityProtocol:
    provisional = ActionabilityRecoverabilityProtocol(
        evidence_status=PUBLICATION_STATUS,
        consumed_test_data=True,
        fresh_evidence=False,
        may_authorize_routing=False,
        may_authorize_promotion=False,
        may_feed_another_experiment=False,
        geometry_ids=GEOMETRY_IDS,
        pre_evaluation_method_ids=PRE_EVALUATION_METHOD_IDS,
        per_geometry_method_ids=PER_GEOMETRY_METHOD_IDS,
        terminal_oracle_ids=TERMINAL_ORACLE_IDS,
        workstation_profile="xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb",
        cpu_workers=4,
        threads_per_worker=3,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        bootstrap_seed=BOOTSTRAP_SEED,
        multiprocessing_start_method="spawn",
        source_generation_devices=("cuda:0", "cuda:1"),
        scientific_reduction_dtype="float64",
        surface_storage_dtype="float32_source_and_probability_arrays",
        contract_hash="",
    )
    return ActionabilityRecoverabilityProtocol(
        **{
            **provisional.__dict__,
            "contract_hash": canonical_hash(provisional._unhashed_payload()),
        }
    )


def assert_consumed_test_diagnostic_only(
    protocol: ActionabilityRecoverabilityProtocol,
) -> None:
    if (
        protocol.evidence_status != PUBLICATION_STATUS
        or protocol.consumed_test_data is not True
        or protocol.fresh_evidence is not False
        or protocol.may_authorize_routing is not False
        or protocol.may_authorize_promotion is not False
        or protocol.may_feed_another_experiment is not False
        or protocol.geometry_ids != GEOMETRY_IDS
        or protocol.pre_evaluation_method_ids != PRE_EVALUATION_METHOD_IDS
        or protocol.per_geometry_method_ids != PER_GEOMETRY_METHOD_IDS
        or protocol.terminal_oracle_ids != TERMINAL_ORACLE_IDS
        or protocol.cpu_workers != 4
        or protocol.threads_per_worker != 3
        or protocol.bootstrap_replicates != BOOTSTRAP_REPLICATES
        or protocol.bootstrap_seed != BOOTSTRAP_SEED
        or protocol.multiprocessing_start_method != "spawn"
        or protocol.source_generation_devices != ("cuda:0", "cuda:1")
        or protocol.scientific_reduction_dtype != "float64"
        or protocol.surface_storage_dtype
        != "float32_source_and_probability_arrays"
    ):
        raise ProtocolError(
            "Actionability/recoverability analysis escaped its consumed-test "
            "boundary."
        )
    expected = canonical_consumed_test_protocol()
    if (
        protocol.contract_hash != canonical_hash(protocol._unhashed_payload())
        or protocol.to_payload() != expected.to_payload()
    ):
        raise ProtocolError(
            "Actionability/recoverability protocol contract hash drifted."
        )


__all__ = (
    "ActionabilityRecoverabilityProtocol",
    "assert_consumed_test_diagnostic_only",
    "canonical_consumed_test_protocol",
)
