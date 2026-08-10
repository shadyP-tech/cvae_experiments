"""Fail-closed claim and workstation contract for the consumed-data analysis."""

from __future__ import annotations

from dataclasses import dataclass

from ...protocol import ProtocolError
from ..fixed_bank_hierarchical_residual_stacker.core_hashing import canonical_hash
from ..fixed_bank_hierarchical_residual_stacker.scientific_constants import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
)
from .constants import (
    FEATURE_NAMES,
    INTERCEPT_GRID,
    LAMBDA_GRID,
    MARGIN_BANDWIDTH_LOGIT,
    MAX_ABSOLUTE_CORRECTION_LOGIT,
    METHOD_IDS,
    RIDGE_ALPHA_GRID,
    STANDARDIZATION_SCALE_FLOOR,
    UNCERTAINTY_Z,
)


@dataclass(frozen=True)
class SignedErrorGateProtocol:
    evidence_status: str
    consumed_test_data: bool
    fresh_evidence: bool
    may_authorize_routing: bool
    may_feed_another_experiment: bool
    diagnostic_method_ids: tuple[str, ...]
    workstation_profile: str
    cpu_workers: int
    threads_per_worker: int
    bootstrap_replicates: int
    bootstrap_seed: int
    multiprocessing_start_method: str
    probability_generation_devices: tuple[str, ...]
    scientific_reduction_dtype: str
    surface_storage_dtype: str
    contract_hash: str

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_fixed_bank_signed_error_gate_protocol_v1",
            "dataset_family": "MIDOG++",
            "claim_role": "posthoc_signed_error_mechanism_diagnostic",
            "evidence_status": self.evidence_status,
            "consumed_test_data": self.consumed_test_data,
            "fresh_evidence": self.fresh_evidence,
            "may_authorize_routing": self.may_authorize_routing,
            "may_authorize_promotion": False,
            "may_feed_another_experiment": self.may_feed_another_experiment,
            "target_expert_used": False,
            "target_labels_update_shared_model": False,
            "same_target_support_use": "intercept_and_common_lambda_only",
            "evaluation_labels_open_after_all_prediction_seals": True,
            "baseline_predicted_class_branch_used": False,
            "gradient_target": "strict_oof_class_balanced_negative_log_loss_logit_gradient",
            "diagnostic_method_ids": list(self.diagnostic_method_ids),
            "feature_names": list(FEATURE_NAMES),
            "ridge_alpha_grid": list(RIDGE_ALPHA_GRID),
            "intercept_grid": list(INTERCEPT_GRID),
            "lambda_grid": list(LAMBDA_GRID),
            "margin_bandwidth_logit": MARGIN_BANDWIDTH_LOGIT,
            "maximum_absolute_correction_logit": MAX_ABSOLUTE_CORRECTION_LOGIT,
            "uncertainty_z": UNCERTAINTY_Z,
            "standardization_scale_floor": STANDARDIZATION_SCALE_FLOOR,
            "R_raw_and_R_safe_separately_sealed": True,
            "full_lambda_path_threshold_crossings_and_fallback_persisted": True,
            "exact_bacc_lcb_relaxed": False,
            "controls": ["G", "P"],
            "terminal_evaluation_runtime": {
                "bootstrap_replicates": self.bootstrap_replicates,
                "bootstrap_seed": self.bootstrap_seed,
                "multiprocessing_start_method": self.multiprocessing_start_method,
            },
            "workstation": {
                "profile": self.workstation_profile,
                "probability_generation_devices": list(
                    self.probability_generation_devices
                ),
                "gpu_workers": 2,
                "cpu_workers": self.cpu_workers,
                "threads_per_worker": self.threads_per_worker,
                "multiprocessing_start_method": self.multiprocessing_start_method,
                "gpu_and_cpu_phases_disjoint": True,
                "parent_cuda_context_forbidden_during_cpu_phase": True,
                "scientific_reduction_dtype": self.scientific_reduction_dtype,
                "surface_storage_dtype": self.surface_storage_dtype,
                "probability_generation_phase_owned_by_parent_runtime": True,
                "bounded_process_local_probability_surface_copy_count": 4,
                "context_features_streamed_and_hash_revalidated_per_target": True,
                "duplicate_full_context_feature_matrix_forbidden": True,
            },
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "contract_hash": self.contract_hash}


def canonical_consumed_test_protocol() -> SignedErrorGateProtocol:
    provisional = SignedErrorGateProtocol(
        evidence_status="EXPLORATORY_CONSUMED_DATA_ONLY",
        consumed_test_data=True,
        fresh_evidence=False,
        may_authorize_routing=False,
        may_feed_another_experiment=False,
        diagnostic_method_ids=METHOD_IDS,
        workstation_profile="xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb",
        cpu_workers=4,
        threads_per_worker=3,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        bootstrap_seed=BOOTSTRAP_SEED,
        multiprocessing_start_method="spawn",
        probability_generation_devices=("cuda:0", "cuda:1"),
        scientific_reduction_dtype="float64",
        surface_storage_dtype=(
            "sealed_probability_float32_memmap_context_features_streamed_float64"
        ),
        contract_hash="",
    )
    return SignedErrorGateProtocol(
        **{
            **provisional.__dict__,
            "contract_hash": canonical_hash(provisional._unhashed_payload()),
        }
    )


def assert_consumed_test_diagnostic_only(protocol: SignedErrorGateProtocol) -> None:
    if (
        protocol.evidence_status != "EXPLORATORY_CONSUMED_DATA_ONLY"
        or protocol.consumed_test_data is not True
        or protocol.fresh_evidence is not False
        or protocol.may_authorize_routing is not False
        or protocol.may_feed_another_experiment is not False
        or protocol.diagnostic_method_ids != METHOD_IDS
        or protocol.cpu_workers != 4
        or protocol.threads_per_worker != 3
        or protocol.bootstrap_replicates != BOOTSTRAP_REPLICATES
        or protocol.bootstrap_seed != BOOTSTRAP_SEED
        or protocol.multiprocessing_start_method != "spawn"
    ):
        raise ProtocolError("Signed-error analysis escaped its consumed-test boundary.")
    expected = canonical_consumed_test_protocol()
    if (
        protocol.contract_hash != canonical_hash(protocol._unhashed_payload())
        or protocol.to_payload() != expected.to_payload()
    ):
        raise ProtocolError("Signed-error analysis protocol contract hash drifted.")


__all__ = (
    "SignedErrorGateProtocol",
    "assert_consumed_test_diagnostic_only",
    "canonical_consumed_test_protocol",
)
