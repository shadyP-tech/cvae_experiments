"""Fail-closed protocol for source-trained, test-label-free prediction."""

from __future__ import annotations

from dataclasses import dataclass

from ...protocol import ProtocolError
from ...routing.disagreement_regret_core.hashing import canonical_sha256
from .experiment_contracts import (
    AUTHORIZATION_SCOPE,
    CLAIM_ROLE,
    GEOMETRY_IDS,
    MODEL_FAMILY_IDS,
    PUBLICATION_STATUS,
    SURFACE_IDS,
)


@dataclass(frozen=True)
class DisagreementRegretPredictionOnlyProtocol:
    evidence_status: str
    consumed_test_data: bool
    target_labels_available: bool
    target_scoring_permitted: bool
    fresh_evidence: bool
    source_oof_is_posthoc: bool
    source_labels_previously_available: bool
    may_authorize_routing: bool
    may_authorize_promotion: bool
    may_feed_another_experiment: bool
    geometry_ids: tuple[str, ...]
    model_family_ids: tuple[str, ...]
    surface_ids: tuple[str, ...]
    workstation_profile: str
    gpu_devices: tuple[str, ...]
    cpu_workers: int
    threads_per_worker: int
    contract_hash: str

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": (
                "midogpp_fixed_bank_disagreement_regret_prediction_only_"
                "protocol_v1"
            ),
            "dataset_family": "MIDOG++",
            "claim_role": CLAIM_ROLE,
            "authorization_scope": AUTHORIZATION_SCOPE,
            "evidence_status": self.evidence_status,
            "consumed_test_data": self.consumed_test_data,
            "fresh_evidence": self.fresh_evidence,
            "method_development_is_posthoc": True,
            "source_oof_is_posthoc": self.source_oof_is_posthoc,
            "source_labels_previously_available": (
                self.source_labels_previously_available
            ),
            "source_authorization_is_fresh_or_unused": False,
            "target_labels_available": self.target_labels_available,
            "target_label_capability_exists": False,
            "target_scoring_permitted": self.target_scoring_permitted,
            "target_bacc_or_regret_computation_permitted": False,
            "target_oracle_computation_permitted": False,
            "target_nelbo_or_downstream_metric_permitted": False,
            "may_authorize_routing": self.may_authorize_routing,
            "may_authorize_promotion": self.may_authorize_promotion,
            "may_authorize_action_selection": False,
            "may_authorize_policy_update": False,
            "may_authorize_model_update": False,
            "may_authorize_expert_update": False,
            "may_authorize_deployment": False,
            "may_feed_stage50": False,
            "may_feed_stage60": False,
            "may_feed_stage70": False,
            "may_feed_another_stage90": False,
            "may_feed_another_experiment": self.may_feed_another_experiment,
            "geometry_ids": list(self.geometry_ids),
            "geometry_selection_used": False,
            "model_family_ids": list(self.model_family_ids),
            "surface_ids": list(self.surface_ids),
            "candidate_generalization": "known_fixed_bank_reuse",
            "unseen_expert_transfer_claim": False,
            "training_flow": (
                "posthoc_train_source_oof_labels_fixed_outer_H_query_q_"
                "excluded_from_every_action_composition_candidate_source_e_"
                "response_query_exclusion"
            ),
            "source_oof_query_q_excluded_from_all_action_compositions": True,
            "source_oof_excluded_pair_fit_reuse": (
                "one_physical_fit_per_unordered_H_q_pair_two_oriented_"
                "prediction_contexts"
            ),
            "source_oof_physical_fit_task_count": 324,
            "source_oof_physical_classifier_fit_count": 5_184,
            "source_oof_oriented_prediction_context_count": 648,
            "source_oof_oriented_prediction_cell_count": 10_368,
            "target_inference_fit_task_count": 81,
            "target_inference_classifier_fit_count": 1_458,
            "total_physical_classifier_fit_count_before_test_admission": 6_642,
            "test_phase_classifier_fit_count": 0,
            "source_oof_weight_normalization": {
                "B": "8/7",
                "U": "8/7",
                "A0": "9/8",
                "A1": "72/65",
                "sample_weight_scope": "logistic_regression_fit_only",
                "scaler_fit_used_sample_weight": False,
            },
            "freeze_boundary": (
                "strict_source_oof_predictions_target_compatible_classifier_"
                "bank_and_all_model_families_hyperparameters_and_hashes_"
                "before_target_test_cache_admission"
            ),
            "target_flow": "whole_consumed_test_label_free_inference_only",
            "prediction_output_is_policy": False,
            "workstation": {
                "profile": self.workstation_profile,
                "source_generation_devices": list(self.gpu_devices),
                "source_generation_workers": 2,
                "source_workers_per_device": 1,
                "gpu_phase_precedes_cpu_phase": True,
                "cpu_workers": self.cpu_workers,
                "threads_per_worker": self.threads_per_worker,
                "maximum_total_cpu_threads": 12,
                "maximum_dense_fit_bytes": 536870912,
                "multiprocessing_start_method": "spawn",
                "gpu_and_cpu_phases_disjoint": True,
                "scientific_reduction_dtype": "float64",
                "surface_storage_dtype": "float32",
                "local_scratch_preferred": True,
                "hash_validated_resume": True,
            },
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "contract_hash": self.contract_hash}


def canonical_prediction_only_protocol() -> DisagreementRegretPredictionOnlyProtocol:
    provisional = DisagreementRegretPredictionOnlyProtocol(
        evidence_status=PUBLICATION_STATUS,
        consumed_test_data=True,
        target_labels_available=False,
        target_scoring_permitted=False,
        fresh_evidence=False,
        source_oof_is_posthoc=True,
        source_labels_previously_available=True,
        may_authorize_routing=False,
        may_authorize_promotion=False,
        may_feed_another_experiment=False,
        geometry_ids=GEOMETRY_IDS,
        model_family_ids=MODEL_FAMILY_IDS,
        surface_ids=SURFACE_IDS,
        workstation_profile="xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb",
        gpu_devices=("cuda:0", "cuda:1"),
        cpu_workers=4,
        threads_per_worker=3,
        contract_hash="",
    )
    return DisagreementRegretPredictionOnlyProtocol(
        **{
            **provisional.__dict__,
            "contract_hash": canonical_sha256(provisional._unhashed_payload()),
        }
    )


def assert_prediction_only_diagnostic(
    protocol: DisagreementRegretPredictionOnlyProtocol,
) -> None:
    expected = canonical_prediction_only_protocol()
    if (
        protocol.to_payload() != expected.to_payload()
        or protocol.contract_hash
        != canonical_sha256(protocol._unhashed_payload())
    ):
        raise ProtocolError(
            "Disagreement-regret prediction-only protocol escaped its boundary."
        )


__all__ = (
    "DisagreementRegretPredictionOnlyProtocol",
    "assert_prediction_only_diagnostic",
    "canonical_prediction_only_protocol",
)
