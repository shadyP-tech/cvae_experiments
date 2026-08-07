from __future__ import annotations

from pathlib import Path

import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.mmd_kmm_mixture import (
    CROSSFIT_COHORT_SUPPORT_ROLE,
    MMDKMMProtocol,
)


def _kwargs() -> dict[str, object]:
    return {
        "target_center": "target",
        "candidate_sources": tuple(f"source-{index}" for index in range(7)),
        "support_case_ids": ("support-a", "support-b"),
        "evaluation_case_ids": ("eval-a", "eval-b"),
        "common_frame_hash": "common-frame",
    }


@pytest.mark.parametrize(
    ("override", "value"),
    (
        ("artifact_dataset_family", "BreakHis"),
        ("claim_dataset_family", "BreakHis"),
        ("common_frame_semantics", "source_local_pca"),
        ("source_expert_training_role", "target_adapted"),
        ("target_support_role", "labeled_target_support"),
        ("claim_role", "downstream_utility"),
        ("source_experts_frozen", False),
        ("target_expert_excluded", False),
        ("support_labels_used", True),
        ("evaluation_labels_available_to_router", True),
        ("evaluation_embeddings_available_to_router", True),
        ("previous_stage90_router_or_utility_inputs_used", True),
    ),
)
def test_protocol_claim_firewall_rejects_forbidden_inputs(
    override: str, value: object
) -> None:
    kwargs = _kwargs()
    kwargs[override] = value
    with pytest.raises(ProtocolError, match="claim firewall"):
        MMDKMMProtocol(**kwargs)  # type: ignore[arg-type]


def test_protocol_excludes_target_and_requires_disjoint_support() -> None:
    target_in_pool = _kwargs()
    target_in_pool["candidate_sources"] = (
        "target",
        *(f"source-{index}" for index in range(6)),
    )
    with pytest.raises(ProtocolError, match="claim firewall"):
        MMDKMMProtocol(**target_in_pool)  # type: ignore[arg-type]

    overlapping = _kwargs()
    overlapping["evaluation_case_ids"] = ("support-a", "eval-b")
    with pytest.raises(ProtocolError, match="claim firewall"):
        MMDKMMProtocol(**overlapping)  # type: ignore[arg-type]


def test_protocol_requires_all_retained_seeds_without_selection() -> None:
    kwargs = _kwargs()
    kwargs["training_seeds"] = (17, 42)
    with pytest.raises(ProtocolError, match="claim firewall"):
        MMDKMMProtocol(**kwargs)  # type: ignore[arg-type]

    kwargs = _kwargs()
    kwargs["generation_seeds"] = (17, 42, 999)
    with pytest.raises(ProtocolError, match="claim firewall"):
        MMDKMMProtocol(**kwargs)  # type: ignore[arg-type]


def test_protocol_allows_only_explicit_own_case_excluded_transductive_mode() -> None:
    protocol = MMDKMMProtocol(
        **_kwargs(),
        target_support_role=CROSSFIT_COHORT_SUPPORT_ROLE,
        evaluation_embeddings_available_to_router=True,
        cross_fitted_transductive_diagnostic=True,
        cohort_evaluation_embeddings_available_for_other_case_routes=True,
        heldout_evaluation_embeddings_available_to_own_route=False,
    )
    assert protocol.cross_fitted_transductive_diagnostic is True
    assert protocol.evaluation_embeddings_available_to_router is True
    assert protocol.heldout_evaluation_embeddings_available_to_own_route is False

    unsafe = _kwargs()
    unsafe.update(
        target_support_role=CROSSFIT_COHORT_SUPPORT_ROLE,
        evaluation_embeddings_available_to_router=True,
        cross_fitted_transductive_diagnostic=True,
        cohort_evaluation_embeddings_available_for_other_case_routes=True,
        heldout_evaluation_embeddings_available_to_own_route=True,
    )
    with pytest.raises(ProtocolError, match="claim firewall"):
        MMDKMMProtocol(**unsafe)  # type: ignore[arg-type]

def test_router_is_math_only_and_has_no_runnable_experiment_surface() -> None:
    package_root = (
        Path(__file__).parents[2]
        / "src"
        / "midogpp_thesis"
        / "cvae"
        / "routing"
        / "mmd_kmm_mixture"
    )
    assert package_root.is_dir()
    assert not (package_root / "runner.py").exists()
    assert not (package_root / "workspace_binding.py").exists()
    assert not (package_root / "validation.py").exists()
