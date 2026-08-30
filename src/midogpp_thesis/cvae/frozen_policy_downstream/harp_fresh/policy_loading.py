"""Reconstruct and execute a complete frozen Stage-60 HARP policy.

This module is the production inference boundary.  A policy-lock hash alone is
not executable evidence: the lock must carry every model bank, its closed
action library, every policy threshold, and the upstream inference lineage.
Target outcomes are intentionally absent from the loader and selector APIs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import statistics
import struct
from typing import Mapping

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...routing.harp_action_model import (
    HarpActionModelBank,
    HarpTargetAction,
    model_bank_collection_from_payload,
    model_bank_collection_payload,
    score_harp_actions,
)
from ...routing.harp_action_surface import ACTION_FEATURE_NAMES
from ...routing.harp_action_surface.inference_binding import (
    HarpActionInferenceBinding,
)
from ...routing.harp_portfolio import (
    HarpPolicyConfig,
    select_harp_physical_portfolio,
    select_harp_portfolio,
)
from ...routing.harp_portfolio.support_envelope import HarpSupportEnvelope
from ...routing.harp_protocol.hashing import canonical_hash
from ...runtime.harp_probability_menu import (
    BASE_ACTION_ID,
    EXACT_NINE_SEED_PAIRS,
    LAMBDA_GRID,
    TARGET_SURFACE,
    UNIFORM_ACTION_ID,
    HarpPredictionMenuSeal,
    HarpRouteDecision,
    build_all_target_actions,
)
from ...runtime.harp_probability_menu.hashing import (
    canonical_sha256,
    raw_array_sha256,
    require_sha256,
)
from .contracts import (
    HarpFreshTargetCache,
    HarpFrozenExecutionLineage,
    HarpFrozenPolicyMetadata,
)
from .policy import FrozenHarpPolicy, _bind_reconstructed_harp_policy


POLICY_LOCK_MEMBER = Path("manifests/policy_lock.json")
_SELECTION_ORDER = (
    "gain_lower_desc",
    "brier_upper_asc",
    "log_loss_upper_asc",
    "lambda_asc",
    "source_id_asc",
)


def _read_json(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("Frozen HARP policy lock is absent or invalid JSON.") from exc
    if not isinstance(raw, dict):
        raise ProtocolError("Frozen HARP policy lock must be a JSON object.")
    return raw


def _lock_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate / POLICY_LOCK_MEMBER if candidate.is_dir() else candidate


def _number(raw: Mapping[str, object], key: str) -> float:
    value = raw.get(key)
    if type(value) not in (int, float):
        raise ProtocolError(f"Frozen HARP policy threshold is absent or malformed: {key}.")
    number = float(value)
    if not np.isfinite(number):
        raise ProtocolError(f"Frozen HARP policy threshold is nonfinite: {key}.")
    return number


def _integer(raw: Mapping[str, object], key: str) -> int:
    value = raw.get(key)
    if type(value) is not int:
        raise ProtocolError(f"Frozen HARP policy support gate is absent: {key}.")
    return value


def _policy_config(raw: Mapping[str, object]) -> HarpPolicyConfig:
    if _integer(raw, "minimum_truth_classes") != 2:
        raise ProtocolError("Frozen HARP policy must require both source truth classes.")
    return HarpPolicyConfig(
        kappa_gain=_number(raw, "gain_kappa"),
        kappa_loss=_number(raw, "loss_kappa"),
        gain_threshold=_number(raw, "minimum_positive_gain"),
        brier_noninferiority_margin=_number(raw, "maximum_brier_delta"),
        log_loss_noninferiority_margin=_number(raw, "maximum_log_loss_delta"),
        min_donor_count=_integer(raw, "minimum_donor_centers"),
        min_paired_case_count=_integer(raw, "minimum_paired_cases"),
        max_leverage=_number(raw, "maximum_leverage"),
        min_compatibility_shrinkage=_number(
            raw, "minimum_compatibility_shrinkage"
        ),
    )


def _validated_action_library(
    raw: object,
    banks: tuple[HarpActionModelBank, ...],
) -> tuple[str, tuple[str, ...]]:
    if not isinstance(raw, Mapping):
        raise ProtocolError("Frozen HARP policy lacks its complete action library.")
    unhashed = {key: value for key, value in raw.items() if key != "action_library_hash"}
    observed_hash = raw.get("action_library_hash")
    if observed_hash != canonical_hash(unhashed):
        raise ProtocolError("Frozen HARP action-library hash drifted.")
    action_hash = require_sha256(observed_hash, name="action-library hash")
    feature_names = tuple(str(value) for value in raw.get("feature_names", ()))
    expected_candidates = {
        center: [source for source in CENTERS if source != center] for center in CENTERS
    }
    expected_actions = build_all_target_actions()
    expected_runtime_hash = canonical_sha256(
        {
            "schema_version": "midogpp_harp_target_action_library_runtime_v2",
            "actions": [action.to_payload() for action in expected_actions],
            "lambda_grid": list(LAMBDA_GRID),
        }
    )
    if (
        raw.get("schema_version") != "midogpp_harp_action_library_v2"
        or raw.get("candidate_sources_by_target") != expected_candidates
        or tuple(raw.get("lambda_grid", ())) != LAMBDA_GRID
        or tuple(raw.get("directions", ())) != ("D01", "D10", "ALL_MARGINS")
        or feature_names != ACTION_FEATURE_NAMES
        or any(bank.feature_names != feature_names for bank in banks)
        or raw.get("probability_endpoint") != "exact_nine_seed_ensemble_float64"
        or raw.get("predictive_reference_action_id") != "U"
        or raw.get("operational_fallback_action_id") != "B"
        or raw.get("lambda_semantics")
        != "post_classifier_predictive_probability_ensemble_not_generated_distribution"
        or raw.get("lambda_one_is_physical_hxe_endpoint") is not True
        or tuple(raw.get("selection_order", ())) != _SELECTION_ORDER
    ):
        raise ProtocolError("Frozen HARP action library cannot be reconstructed exactly.")
    # The Stage-60 semantic library and neutral runtime library have distinct
    # schemas.  Binding both prevents either action geometry from being swapped.
    return action_hash, (expected_runtime_hash, *feature_names)


@dataclass(frozen=True, kw_only=True)
class HarpFrozenInferenceReceipt:
    """Independently reconstructed executable policy state."""

    metadata: HarpFrozenPolicyMetadata
    model_banks: tuple[HarpActionModelBank, ...]
    policy_config: HarpPolicyConfig
    model_bank_collection_hash: str
    action_library_hash: str
    action_runtime_hash: str
    execution_lineage: HarpFrozenExecutionLineage
    support_envelope: HarpSupportEnvelope
    action_surface_global_prediction_seal_hash: str
    action_inference_binding_sha256: str
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        banks = tuple(self.model_banks)
        if (
            not isinstance(self.metadata, HarpFrozenPolicyMetadata)
            or tuple(bank.outer_target_id for bank in banks) != CENTERS
            or any(not isinstance(bank, HarpActionModelBank) for bank in banks)
            or not isinstance(self.policy_config, HarpPolicyConfig)
        ):
            raise ProtocolError("Frozen HARP executable model-bank coverage drifted.")
        collection_hash = require_sha256(
            self.model_bank_collection_hash, name="model-bank collection hash"
        )
        action_hash = require_sha256(
            self.action_library_hash, name="action-library hash"
        )
        runtime_hash = require_sha256(
            self.action_runtime_hash, name="target action-runtime hash"
        )
        if not isinstance(self.execution_lineage, HarpFrozenExecutionLineage):
            raise ProtocolError("Frozen HARP inference receipt lacks typed lineage.")
        if not isinstance(self.support_envelope, HarpSupportEnvelope):
            raise ProtocolError("Frozen HARP inference receipt lacks support envelope.")
        surface_hash = require_sha256(
            self.action_surface_global_prediction_seal_hash,
            name="action-surface global prediction-seal hash",
        )
        inference_binding_hash = require_sha256(
            self.action_inference_binding_sha256,
            name="action inference-binding hash",
        )
        payload = {
            "schema_version": "midogpp_harp_frozen_inference_receipt_v1",
            "policy_lock_hash": self.metadata.policy_lock_hash,
            "fresh_reservation_hash": self.metadata.fresh_reservation_hash,
            "expert_bank_semantic_id": self.metadata.bank_hash,
            "generation_semantic_id": self.metadata.generation_lock_hash,
            "source_stream_content_semantic_id": self.metadata.source_cache_hash,
            "classifier_config_semantic_id": self.metadata.classifier_hash,
            "model_bank_collection_hash": collection_hash,
            "action_library_hash": action_hash,
            "action_runtime_hash": runtime_hash,
            "execution_lineage": self.execution_lineage.to_payload(),
            "support_compatibility_envelope_sha256": (
                self.support_envelope.envelope_sha256
            ),
            "action_surface_global_prediction_seal_hash": surface_hash,
            "action_inference_binding_sha256": inference_binding_hash,
            "policy_config": {
                "kappa_gain": self.policy_config.kappa_gain,
                "kappa_loss": self.policy_config.kappa_loss,
                "gain_threshold": self.policy_config.gain_threshold,
                "brier_noninferiority_margin": (
                    self.policy_config.brier_noninferiority_margin
                ),
                "log_loss_noninferiority_margin": (
                    self.policy_config.log_loss_noninferiority_margin
                ),
                "min_donor_count": self.policy_config.min_donor_count,
                "min_paired_case_count": self.policy_config.min_paired_case_count,
                "max_leverage": self.policy_config.max_leverage,
                "min_compatibility_shrinkage": (
                    self.policy_config.min_compatibility_shrinkage
                ),
            },
            "model_state_complete": True,
            "threshold_state_complete": True,
            "action_library_complete": True,
            "label_free_physical_lambda_one_ablation_available": True,
            "physical_ablation_lambda": 1.0,
            "target_outcomes_used": False,
        }
        object.__setattr__(self, "model_banks", banks)
        object.__setattr__(self, "model_bank_collection_hash", collection_hash)
        object.__setattr__(self, "action_library_hash", action_hash)
        object.__setattr__(self, "action_runtime_hash", runtime_hash)
        object.__setattr__(
            self, "action_surface_global_prediction_seal_hash", surface_hash
        )
        object.__setattr__(
            self, "action_inference_binding_sha256", inference_binding_hash
        )
        object.__setattr__(self, "receipt_hash", canonical_sha256(payload))


def _feature_values(
    baseline_members: np.ndarray,
    expert_members: np.ndarray,
    baseline: float,
    expert: float,
    lam: float,
) -> tuple[float, ...]:
    action = expert if lam == 1.0 else (1.0 - lam) * baseline + lam * expert
    member_actions = (
        expert_members.copy()
        if lam == 1.0
        else (1.0 - lam) * baseline_members + lam * expert_members
    )
    expert_flips = float(
        np.mean((expert_members >= 0.5) != (baseline_members >= 0.5), dtype=np.float64)
    )
    action_flips = float(
        np.mean((member_actions >= 0.5) != (baseline_members >= 0.5), dtype=np.float64)
    )
    dispersion = float(
        statistics.pstdev(
            tuple(float(value) for value in (expert_members - baseline_members))
        )
    )
    return (
        baseline,
        expert,
        action,
        abs(baseline - 0.5),
        abs(expert - 0.5),
        abs(action - 0.5),
        expert - baseline,
        abs(expert - baseline),
        action - baseline,
        abs(action - baseline),
        expert_flips,
        action_flips,
        dispersion,
        lam,
    )


def _direction(baseline: float, action: float) -> str:
    before = baseline >= 0.5
    after = action >= 0.5
    if not before and after:
        return "D01"
    if before and not after:
        return "D10"
    return "ALL_MARGINS"


def _row_ensemble_receipt(
    menu: HarpPredictionMenuSeal,
    *,
    center: str,
    ordinal: int,
    row_id: str,
    case_id: str,
) -> str:
    actions = tuple(
        action
        for action in menu.actions
        if action.outer_target_id == center and action.query_center_id == center
    )
    members = []
    for action in actions:
        values = np.asarray(
            [cell.probabilities[ordinal] for cell in menu.cells_for(action)],
            dtype=np.float32,
        )
        members.append(
            {
                "action_hash": action.action_hash,
                "member_probability_bytes_sha256": raw_array_sha256(values),
            }
        )
    return canonical_sha256(
        {
            "schema_version": "midogpp_harp_fresh_row_exact_nine_menu_receipt_v1",
            "prediction_menu_seal_hash": menu.seal_hash,
            "outer_target_id": center,
            "row_id": row_id,
            "case_id": case_id,
            "seed_pairs": [list(pair) for pair in EXACT_NINE_SEED_PAIRS],
            "actions": members,
            "labels_consumed": False,
        }
    )


def _select_with_reconstructed_banks(
    receipt: HarpFrozenInferenceReceipt,
    menu: HarpPredictionMenuSeal,
    cache: HarpFreshTargetCache,
    metadata: HarpFrozenPolicyMetadata,
    *,
    physical_lambda_one_only: bool = False,
) -> tuple[HarpRouteDecision, ...]:
    if metadata.policy_lock_hash != receipt.metadata.policy_lock_hash:
        raise ProtocolError("Frozen HARP selector escaped its reconstructed receipt.")
    banks = {bank.outer_target_id: bank for bank in receipt.model_banks}
    decisions: list[HarpRouteDecision] = []
    for center in CENTERS:
        frame = cache.frames_by_center[center]
        baseline_action = menu.action_for(
            surface_kind=TARGET_SURFACE,
            outer_target_id=center,
            query_center_id=center,
            selected_source_id=None,
            action_id=BASE_ACTION_ID,
        )
        baseline = menu.exact_nine(baseline_action)
        reference_action = menu.action_for(
            surface_kind=TARGET_SURFACE,
            outer_target_id=center,
            query_center_id=center,
            selected_source_id=None,
            action_id=UNIFORM_ACTION_ID,
        )
        reference_cells = menu.cells_for(reference_action)
        reference = menu.exact_nine(reference_action)
        source_actions = tuple(
            action
            for action in menu.actions
            if action.surface_kind == TARGET_SURFACE
            and action.outer_target_id == center
            and action.query_center_id == center
            and action.selected_source_id is not None
        )
        expected_sources = tuple(source for source in CENTERS if source != center)
        observed_sources = tuple(
            sorted(
                str(action.selected_source_id)
                for action in source_actions
                if action.selected_source_id is not None
            )
        )
        if observed_sources != expected_sources or len(source_actions) != len(
            expected_sources
        ):
            raise ProtocolError(
                "Frozen HARP target menu lacks the complete legal candidate universe."
            )
        expert_means = {action.selected_source_id: menu.exact_nine(action) for action in source_actions}
        expert_cells = {action.selected_source_id: menu.cells_for(action) for action in source_actions}
        target_actions: list[HarpTargetAction] = []
        action_lookup: dict[tuple[str, str, float], HarpTargetAction] = {}
        for ordinal, (row_id, case_id) in enumerate(
            zip(frame.row_ids, frame.case_ids, strict=True)
        ):
            baseline_probability = float(baseline[ordinal])
            baseline_bytes = struct.pack("<d", baseline_probability)
            reference_members = np.asarray(
                [cell.probabilities[ordinal] for cell in reference_cells],
                dtype=np.float64,
            )
            reference_probability = float(reference[ordinal])
            reference_bytes = struct.pack("<d", reference_probability)
            ensemble_receipt = _row_ensemble_receipt(
                menu,
                center=center,
                ordinal=ordinal,
                row_id=row_id,
                case_id=case_id,
            )
            for action in source_actions:
                assert action.selected_source_id is not None
                source = action.selected_source_id
                expert_members = np.asarray(
                    [cell.probabilities[ordinal] for cell in expert_cells[source]],
                    dtype=np.float64,
                )
                expert_probability = float(expert_means[source][ordinal])
                action_lambdas = (1.0,) if physical_lambda_one_only else LAMBDA_GRID
                for lam in action_lambdas:
                    features = _feature_values(
                        reference_members,
                        expert_members,
                        reference_probability,
                        expert_probability,
                        lam,
                    )
                    action_probability = features[2]
                    target_action = HarpTargetAction(
                        outer_target_id=center,
                        target_query_id=center,
                        candidate_source_id=source,
                        case_id=case_id,
                        sample_id=row_id,
                        lambda_value=lam,
                        direction=_direction(reference_probability, action_probability),
                        feature_names=ACTION_FEATURE_NAMES,
                        feature_values=features,
                        baseline_probability_bytes=reference_bytes,
                        operational_fallback_probability_bytes=baseline_bytes,
                        expert_probability=expert_probability,
                        ensemble_size=len(EXACT_NINE_SEED_PAIRS),
                        ensemble_receipt_hash=ensemble_receipt,
                        prediction_seal_hash=menu.seal_hash,
                        compatibility_shrinkage=(
                            receipt.support_envelope.shrinkage(center, source)
                        ),
                    )
                    target_actions.append(target_action)
                    action_lookup[(row_id, source, lam)] = target_action

        scores = score_harp_actions(banks[center], target_actions)
        expected_lambda_count = 1 if physical_lambda_one_only else len(LAMBDA_GRID)
        expected_action_count = (
            len(frame.row_ids) * len(expected_sources) * expected_lambda_count
        )
        if len(scores) != expected_action_count:
            raise ProtocolError(
                "Frozen HARP target score surface lacks the legal candidate grid."
            )
        portfolio = (
            select_harp_physical_portfolio(scores, config=receipt.policy_config)
            if physical_lambda_one_only
            else select_harp_portfolio(scores, config=receipt.policy_config)
        )
        by_row = {row.sample_id: row for row in portfolio}
        if len(by_row) != len(frame.row_ids) or set(by_row) != set(frame.row_ids):
            raise ProtocolError("Frozen HARP portfolio did not cover every target row.")
        for row_id, case_id in zip(frame.row_ids, frame.case_ids, strict=True):
            selected = by_row[row_id]
            if selected.case_id != case_id:
                raise ProtocolError("Frozen HARP portfolio changed a target case identity.")
            if selected.routed:
                assert selected.selected_source_id is not None
                assert selected.selected_lambda is not None
                chosen = action_lookup[
                    (row_id, selected.selected_source_id, selected.selected_lambda)
                ]
                direction = chosen.direction
                eligible = True
                source = selected.selected_source_id
                lam = selected.selected_lambda
            else:
                direction = "NO_DISAGREEMENT"
                eligible = False
                source = None
                lam = 0.0
                if selected.output_probability_bytes != selected.baseline_probability_bytes:
                    raise ProtocolError("Frozen HARP fallback changed exact-B bytes.")
            decisions.append(
                HarpRouteDecision(
                    surface_kind=TARGET_SURFACE,
                    outer_target_id=center,
                    query_center_id=center,
                    row_id=row_id,
                    case_id=case_id,
                    eligible=eligible,
                    selected_source_id=source,
                    lambda_value=lam,
                    direction=direction,
                    decision_reason=(
                        f"PHYSICAL_LAMBDA_ONE_ABLATION::{selected.reason}"
                        if physical_lambda_one_only
                        else selected.reason
                    ),
                    policy_hash=metadata.policy_lock_hash,
                    prediction_menu_seal_hash=menu.seal_hash,
                )
            )
    return tuple(decisions)


def reconstruct_frozen_harp_policy_receipt(
    policy_lock_path: str | Path,
    *,
    expected_fresh_reservation_hash: str,
) -> HarpFrozenInferenceReceipt:
    """Load and independently reconstruct all executable Stage-60 state."""

    path = _lock_path(policy_lock_path)
    raw = _read_json(path)
    observed_policy_hash = raw.get("policy_lock_hash")
    if (
        raw.get("schema_version") != "midogpp_harp_policy_lock_v2"
        or observed_policy_hash
        != canonical_hash({key: value for key, value in raw.items() if key != "policy_lock_hash"})
    ):
        raise ProtocolError("Frozen HARP policy-lock schema or hash drifted.")
    policy_hash = require_sha256(observed_policy_hash, name="policy-lock hash")
    expected_reservation = require_sha256(
        expected_fresh_reservation_hash, name="expected fresh reservation"
    )
    if (
        raw.get("status") != "FROZEN_BEFORE_TARGET_EVALUATION"
        or raw.get("dataset_family") != "MIDOG++"
        or raw.get("fresh_target_reservation_hash") != expected_reservation
        or tuple(raw.get("lambda_grid", ())) != LAMBDA_GRID
        or raw.get("probability_endpoint") != "exact_nine_seed_ensemble"
        or raw.get("matched_budget_reference_action") != "U"
        or raw.get("utility_deltas_reference_action") != "U"
        or raw.get("lambda_semantics")
        != "post_classifier_predictive_probability_ensemble_not_generated_distribution"
        or raw.get("physical_expert_routing_primary_lambda") != 1.0
        or raw.get("operational_fallback_action") != "B"
        or raw.get("case_equal_weighting") is not True
        or raw.get("delete_donor_predictions") is not True
        or raw.get("proper_loss_noninferiority") is not True
        or raw.get("exact_b_byte_identical_fallback") is not True
        or raw.get("policy_accepts_outcomes") is not False
        or raw.get("target_support_outcomes_used") is not False
        or raw.get("target_support_feature_geometry_used_for_shrink_only")
        is not True
        or raw.get("support_predicted_outcomes_used") is not False
        or raw.get("target_evaluation_outcomes_used") is not False
        or raw.get("stage50_artifacts_used") is not False
        or raw.get("stage90_artifacts_used") is not False
    ):
        raise ProtocolError("Frozen HARP policy is not an outcome-free fresh policy.")

    collection_raw = raw.get("model_bank_collection")
    banks = model_bank_collection_from_payload(collection_raw)
    rebuilt = model_bank_collection_payload(banks)
    if rebuilt != collection_raw or tuple(bank.outer_target_id for bank in banks) != CENTERS:
        raise ProtocolError("Frozen HARP model-bank collection is incomplete or unreconstructable.")
    collection_hash = require_sha256(
        rebuilt["collection_hash"], name="model-bank collection hash"
    )
    library_hash, runtime_binding = _validated_action_library(
        raw.get("action_library"), banks
    )
    runtime_hash = runtime_binding[0]
    config = _policy_config(raw)
    inference_binding = HarpActionInferenceBinding.from_payload(
        raw.get("action_inference_binding")
    )
    if raw.get("action_inference_binding_sha256") != (
        inference_binding.binding_sha256
    ):
        raise ProtocolError("Frozen HARP inference-binding reference drifted.")
    lineage = HarpFrozenExecutionLineage(
        bank_semantic_lock_hash=inference_binding.expert_bank_semantic_id,
        generation_semantic_lock_hash=inference_binding.generation_semantic_id,
        source_stream_lock_hash=(
            inference_binding.source_stream_lock_semantic_id
        ),
        source_stream_index_hash=(
            inference_binding.source_stream_index_semantic_id
        ),
        source_stream_content_hash=(
            inference_binding.source_stream_content_semantic_id
        ),
        classifier_config_hash=inference_binding.classifier_config_semantic_id,
        expert_bank_index_sha256=(
            inference_binding.expert_bank_index_file_sha256
        ),
        generation_lock_file_sha256=(
            inference_binding.generation_lock_file_sha256
        ),
        source_cache_lock_sha256=(
            inference_binding.source_cache_lock_file_sha256
        ),
        source_cache_index_sha256=(
            inference_binding.source_cache_index_file_sha256
        ),
        source_stream_artifact_binding_hash=(
            inference_binding.source_stream_artifact_binding_semantic_id
        ),
        classifier_contract_sha256=(
            inference_binding.classifier_contract_semantic_id
        ),
    )
    expected_source_binding = canonical_sha256(
        {
            "schema_version": "midogpp_harp_source_stream_artifact_binding_v1",
            "source_cache_lock_sha256": lineage.source_cache_lock_sha256,
            "source_cache_index_sha256": lineage.source_cache_index_sha256,
            "source_stream_content_hash": lineage.source_stream_content_hash,
        }
    )
    if expected_source_binding != lineage.source_stream_artifact_binding_hash:
        raise ProtocolError("Frozen HARP source-stream binding drifted.")
    support_envelope = HarpSupportEnvelope.from_payload(
        raw.get("support_compatibility_envelope")
    )
    if raw.get("support_compatibility_envelope_sha256") != (
        support_envelope.envelope_sha256
    ):
        raise ProtocolError("Frozen HARP support-envelope reference drifted.")
    expected_envelope_keys = {
        (outer, source)
        for outer in CENTERS
        for source in CENTERS
        if source != outer
    }
    if (
        support_envelope.maximum_allowed_leverage != config.max_leverage
        or {
            (cell.outer_target_id, cell.candidate_source_id)
            for cell in support_envelope.cells
        }
        != expected_envelope_keys
    ):
        raise ProtocolError("Frozen HARP support-envelope coverage drifted.")
    metadata = HarpFrozenPolicyMetadata(
        policy_lock_hash=policy_hash,
        fresh_reservation_hash=expected_reservation,
        bank_hash=lineage.bank_semantic_lock_hash,
        generation_lock_hash=lineage.generation_semantic_lock_hash,
        source_cache_hash=lineage.source_stream_content_hash,
        classifier_hash=lineage.classifier_config_hash,
    )
    return HarpFrozenInferenceReceipt(
        metadata=metadata,
        model_banks=banks,
        policy_config=config,
        model_bank_collection_hash=collection_hash,
        action_library_hash=library_hash,
        action_runtime_hash=runtime_hash,
        execution_lineage=lineage,
        support_envelope=support_envelope,
        action_surface_global_prediction_seal_hash=(
            inference_binding.global_prediction_seal_semantic_id
        ),
        action_inference_binding_sha256=inference_binding.binding_sha256,
    )


def load_frozen_harp_policy(
    policy_lock_path: str | Path,
    *,
    expected_fresh_reservation_hash: str,
) -> FrozenHarpPolicy:
    """Return an executable policy only after full Stage-60 reconstruction."""

    receipt = reconstruct_frozen_harp_policy_receipt(
        policy_lock_path,
        expected_fresh_reservation_hash=expected_fresh_reservation_hash,
    )

    def selector(
        menu: HarpPredictionMenuSeal,
        cache: HarpFreshTargetCache,
        metadata: HarpFrozenPolicyMetadata,
    ) -> tuple[HarpRouteDecision, ...]:
        return _select_with_reconstructed_banks(receipt, menu, cache, metadata)

    def physical_selector(
        menu: HarpPredictionMenuSeal,
        cache: HarpFreshTargetCache,
        metadata: HarpFrozenPolicyMetadata,
    ) -> tuple[HarpRouteDecision, ...]:
        return _select_with_reconstructed_banks(
            receipt,
            menu,
            cache,
            metadata,
            physical_lambda_one_only=True,
        )

    return _bind_reconstructed_harp_policy(
        metadata=receipt.metadata,
        selector=selector,
        physical_selector=physical_selector,
        policy_receipt_hash=receipt.receipt_hash,
        model_bank_collection_hash=receipt.model_bank_collection_hash,
        action_library_hash=receipt.action_library_hash,
        execution_lineage=receipt.execution_lineage,
    )


__all__ = (
    "HarpFrozenInferenceReceipt",
    "POLICY_LOCK_MEMBER",
    "load_frozen_harp_policy",
    "reconstruct_frozen_harp_policy_receipt",
)
