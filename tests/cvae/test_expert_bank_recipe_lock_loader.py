from __future__ import annotations

import json
from pathlib import Path

import pytest

from midogpp_thesis.cvae.expert_bank import recipe_lock_loader
from midogpp_thesis.cvae.preservation.prior_recovery_stability_consensus import (
    TrainingSeedConsensusLock,
)
from midogpp_thesis.cvae.preservation.source_inner_studies.contracts import (
    FISHER_DECISION_SCHEMA,
    PRIOR_DECISION_SCHEMA,
)
from midogpp_thesis.real_features.classifier_reference.protocol import ProtocolError


def test_stage30_loader_returns_only_the_requested_fold(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    locks = {"0": _lock("0", export_ready=True), "1": _lock("1", export_ready=True)}
    monkeypatch.setattr(recipe_lock_loader, "validate_stability_bundle", lambda root: locks)

    observed = recipe_lock_loader.load_consensus_recipe_for_fold(
        tmp_path,
        outer_target_center="1",
    )
    assert observed.outer_target_center == "1"


def test_stage30_loader_fails_closed_for_missing_or_nonexportable_fold(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        recipe_lock_loader,
        "validate_stability_bundle",
        lambda root: {"0": _lock("0", export_ready=False)},
    )
    with pytest.raises(ProtocolError, match="not globally ready"):
        recipe_lock_loader.load_consensus_recipe_for_fold(
            tmp_path,
            outer_target_center="0",
        )
    monkeypatch.setattr(
        recipe_lock_loader,
        "validate_stability_bundle",
        lambda root: {"0": _lock("0", export_ready=True)},
    )
    with pytest.raises(ProtocolError, match="no consensus lock"):
        recipe_lock_loader.load_consensus_recipe_for_fold(
            tmp_path,
            outer_target_center="1",
        )


def test_stage30_loader_requires_bundle_wide_export_readiness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        recipe_lock_loader,
        "validate_stability_bundle",
        lambda root: {
            "0": _lock("0", export_ready=True),
            "1": _lock("1", export_ready=False),
        },
    )
    with pytest.raises(ProtocolError, match="not globally ready"):
        recipe_lock_loader.load_consensus_recipe_for_fold(
            tmp_path,
            outer_target_center="0",
        )


@pytest.mark.parametrize(
    "study_schema",
    (
        PRIOR_DECISION_SCHEMA,
        FISHER_DECISION_SCHEMA,
    ),
)
def test_stage30_loader_rejects_non_adoptive_v2_study_bundles(
    tmp_path: Path,
    study_schema: str,
) -> None:
    root = tmp_path / study_schema
    (root / "reports").mkdir(parents=True)
    (root / "reports/study_decision.json").write_text(
        json.dumps(
            {
                "schema_version": study_schema,
                "claim_scope": "cvae_source_inner_study_only",
                "may_feed_recipe_selection": False,
                "may_feed_deployable_selection": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ProtocolError, match="missing outputs"):
        recipe_lock_loader.load_consensus_recipe_for_fold(
            root,
            outer_target_center="0",
        )


def _lock(outer: str, *, export_ready: bool) -> TrainingSeedConsensusLock:
    return TrainingSeedConsensusLock(
        outer_target_center=outer,
        integrity_status="VALID",
        primary_arm="A",
        objective_id="isotropic_beta_vae_v1",
        sampler_family="standard_normal",
        training_seeds=(17, 42, 101),
        seed_lock_hashes={"17": "a", "42": "b", "101": "c"},
        parent_protocol_hash="parent",
        parent_selection_bundle_hash="bundle",
        consensus_rule_id="unanimous_conditional_family_d_only_if_all_d_else_c_v1",
        consensus_origin="STABLE_UNANIMOUS",
        stability_status="STABLE_STANDARD_FALLBACK",
        recipe_export_ready=export_ready,
        reason="test",
    )
