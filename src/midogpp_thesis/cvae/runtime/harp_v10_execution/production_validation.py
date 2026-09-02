"""Shared validation and byte-transport helpers for HARP v10 production.

These helpers are deliberately label agnostic.  They validate typed in-memory
artifacts, frozen configuration values, and the exact float32/sample geometry
used by both target-action materialization and prelabel routing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .compatibility_adapter import CompatibilityAdapterState
from .contracts import (
    ActionKind,
    ArtifactValue,
    LabelFreeActionBlock,
    LabelFreeOuterMenu,
)


def require_state(value: ArtifactValue, expected: type, *, role: str) -> object:
    """Return a typed opaque artifact state or fail closed."""

    if not isinstance(value, ArtifactValue) or not isinstance(value.state, expected):
        raise ProtocolError(f"HARP v10 {role} in-memory state is absent or untyped.")
    return value.state


def require_sha256(value: object, *, role: str) -> str:
    """Validate and return a lowercase SHA-256 identity."""

    text = str(value)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ProtocolError(f"HARP v10 {role} is not SHA-256.")
    return text


def float32_cells(values: np.ndarray) -> tuple[bytes, ...]:
    """Encode an exact finite one-dimensional float32 transport vector."""

    raw = np.asarray(values)
    if raw.dtype != np.float32 or raw.ndim != 1 or not np.isfinite(raw).all():
        raise ProtocolError("HARP v10 probability transport is not finite float32.")
    packed = np.ascontiguousarray(raw, dtype="<f4").tobytes(order="C")
    return tuple(packed[index : index + 4] for index in range(0, len(packed), 4))


def decode_cells(values: Sequence[bytes]) -> np.ndarray:
    """Decode exact little-endian float32 probability cells."""

    cells = tuple(values)
    if not cells or any(type(value) is not bytes or len(value) != 4 for value in cells):
        raise ProtocolError("HARP v10 probability cells are malformed.")
    return np.frombuffer(b"".join(cells), dtype="<f4").astype(np.float32, copy=True)


def case_ids(block: LabelFreeActionBlock) -> tuple[str, ...]:
    """Return stable first-occurrence case identities from a physical block."""

    return tuple(dict.fromkeys(block.case_ids))


def case_indices(block: LabelFreeActionBlock, case_id: str) -> np.ndarray:
    """Resolve a case to its physical sample indices."""

    indices = np.flatnonzero(np.asarray(block.case_ids, dtype=object) == str(case_id))
    if not len(indices):
        raise ProtocolError("HARP v10 case is absent from its physical target block.")
    return indices


def target_case_blocks(
    menu: LabelFreeOuterMenu, case_id: str
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Return aligned target B/U bytes for a case in one sealed outer menu."""

    baseline = menu.target_block(ActionKind.B)
    uniform = menu.target_block(ActionKind.U)
    indices = case_indices(baseline, case_id)
    samples = tuple(baseline.sample_ids[int(index)] for index in indices)
    if tuple(uniform.sample_ids[int(index)] for index in indices) != samples:
        raise ProtocolError("HARP v10 target B/U sample geometry drifted.")
    return (
        samples,
        np.asarray(baseline.probabilities[indices], dtype=np.float32),
        np.asarray(uniform.probabilities[indices], dtype=np.float32),
    )


def receipts_for_pool(
    state: CompatibilityAdapterState, outer: str, query: str
) -> tuple[object, ...]:
    """Resolve every receipt in the already-sealed candidate-pool order."""

    pool = state.pool(outer, query)
    return tuple(
        state.receipt(outer, query, source) for source in pool.candidate_center_ids
    )


def validate_model_config(config: object) -> None:
    """Fail closed if the predeclared HARP v10 model/policy contract drifts."""

    model = getattr(config, "model", None)
    if not isinstance(model, Mapping):
        raise ProtocolError("HARP v10 frozen model/policy contract drifted.")
    policy = model.get("policy")
    admission = model.get("admission")
    if (
        not isinstance(policy, Mapping)
        or not isinstance(admission, Mapping)
        or model.get("schema_version")
        != "midogpp_harp_stage90_policy_calibrated_pairwise_residual_router_v10"
        or tuple(model.get("action_families", ())) != ("B", "U", "Hxe")
        or tuple(model.get("directional_actions", ())) != ("D01", "D10")
        or tuple(model.get("physical_expert_lambda_grid", ())) != (1.0,)
        or model.get("effective_menu_transform")
        != "label_free_threshold_crossing_then_exact_byte_deduplication"
        or tuple(model.get("effective_menu_excluded_families", ()))
        != ("ALL_MARGINS", "STRUCTURAL_NOOP")
        or model.get("deployed_action_kind") != "exact_top1_physical_action"
        or model.get("unevaluated_action_mixtures_allowed") is not False
        or model.get("budget_residual_contrast") != "U_minus_B"
        or model.get("allocation_residual_contrast") != "Hxe_minus_U"
        or model.get("virtual_baseline_candidate") != "B"
        or model.get("ranker")
        != "source_only_center_case_balanced_tie_aware_pairwise_ridge"
        or model.get("ranker_target")
        != "within_case_downstream_utility_preference"
        or model.get("selected_action_acceptor")
        != "cross_fitted_ranker_selected_action_positive_gain_and_risk"
        or model.get("acceptor_training_surface")
        != "inner_source_lodo_oof_selected_actions_only"
        or tuple(model.get("acceptor_features", ()))
        != (
            "selected_pairwise_score",
            "budget_residual_score",
            "allocation_residual_score",
            "top_runner_up_margin",
            "menu_score_mean",
            "menu_score_std",
            "menu_score_max_abs",
            "physical_action_count",
            "action_kind",
            "direction",
        )
        or model.get("per_action_worst_center_certificate_used") is not False
        or model.get("individual_action_safety_claimed") is not False
        or model.get("policy_family")
        != "one_dimensional_acceptance_threshold_exact_top1_or_B"
        or model.get("whole_policy_admission_scope")
        != "all_held_source_cases_nested_route_or_exact_B"
        or model.get("policy_calibration")
        != "nested_source_center_lodo_complete_rank_accept_route_or_B_replay"
        or model.get("selection_rule")
        != (
            "pairwise_top1_if_crossfit_acceptance_score_meets_nested_policy_"
            "threshold_else_exact_B"
        )
        or model.get("numeric_oof_replay")
        != (
            "required_case_ids_pairwise_scores_selected_actions_acceptance_scores_"
            "policy_routes_endpoints_and_fold_hashes"
        )
        or model.get("all_preprocessing_fit_inside_source_lodo") is not True
        or model.get("policy_hyperparameters_selected_inside_source_lodo") is not False
        or model.get("regularization_hyperparameters_predeclared_fixed_before_source_lodo")
        is not True
        or model.get("acceptance_threshold_selected_inside_source_lodo") is not True
        or float(model.get("rank_margin_fixed_guard", -1.0)) != 0.0
        or model.get("target_thresholds_frozen_before_target_evaluation") is not True
        or model.get("per_outer_policy_admission_required") is not True
        or model.get("policy_admission_null") != "always_exact_B_tie_aware"
        or model.get("exact_b_byte_identical_fallback") is not True
    ):
        raise ProtocolError("HARP v10 frozen model/policy contract drifted.")


__all__ = (
    "case_ids",
    "case_indices",
    "decode_cells",
    "float32_cells",
    "receipts_for_pool",
    "require_sha256",
    "require_state",
    "target_case_blocks",
    "validate_model_config",
)
