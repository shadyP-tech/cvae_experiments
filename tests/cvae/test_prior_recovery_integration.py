from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path

import pytest

from midogpp_thesis.cvae.preservation.prior_recovery import (
    run_outer_prior_recovery,
    run_source_inner_prior_recovery,
)
from midogpp_thesis.cvae.preservation.prior_recovery_artifacts import validate_outer_bundle
from midogpp_thesis.cvae.preservation.source_inner_selection import load_recipe_lock, write_recipe_lock
from midogpp_thesis.real_features.classifier_reference.matched_reference import (
    MatchedReferenceConfig,
    run_matched_reference,
)
from midogpp_thesis.real_features.classifier_reference.protocol import ProtocolError
from tests.cvae.prior_recovery_test_support import (
    prior_recovery_config,
    write_prior_recovery_fixture,
)


def test_source_inner_and_outer_commands_write_separate_complete_bundles(tmp_path: Path) -> None:
    manifest, cache = write_prior_recovery_fixture(tmp_path / "midogpp_prior_recovery")
    source_config = prior_recovery_config(
        mode="source_inner",
        artifact_root=tmp_path / "source",
        manifest=manifest,
        cache=cache,
    )
    source_root = run_source_inner_prior_recovery(source_config)
    assert (source_root / "manifests/recipe_locks/0.json").exists()
    assert (source_root / "reports/gate_decision.json").exists()

    reference_root = run_matched_reference(
        MatchedReferenceConfig(
            name="eligible_tuned_real_reference_v2",
            artifact_root=tmp_path / "reference",
            manifest_path=manifest,
            feature_cache_path=cache,
            heldout_centers=("0",),
            expected_feature_dim=6,
            allow_partial_test_coverage=True,
        )
    )
    lock_root = source_root
    outer_config = prior_recovery_config(
        mode="outer",
        artifact_root=tmp_path / "outer",
        manifest=manifest,
        cache=cache,
        reference=reference_root,
        locks=lock_root,
    )
    outer_root = run_outer_prior_recovery(outer_config)
    coverage = json.loads((outer_root / "manifests/coverage_manifest.json").read_text())
    decision = json.loads((outer_root / "reports/decision_report.json").read_text())
    assert coverage["status"] == "PASS"
    assert decision["status"] == "NEGATIVE_PRESERVATION"
    assert decision["claim_scope"] == "cvae_preservation_only"
    assert coverage["observed_rows_by_role"]["prior"] == 4
    assert coverage["valid_rows_by_role"]["prior"] == 4
    assert coverage["valid_all_representation_rows"] == 12
    rows = list(csv.DictReader((outer_root / "tables/preservation_metrics.csv").open()))
    assert {row["arm"] for row in rows if row["representation_role"] == "prior"} == {"A", "B", "C", "D"}
    assert all(row["may_feed_model_recipe"] == "false" for row in rows)
    assert not (outer_root / "manifests/recipe_locks/0.json").exists()
    with pytest.raises(ProtocolError, match="decision contract differs"):
        validate_outer_bundle(
            outer_root,
            expected_config=replace(outer_config, positive_claim_min_ratio=0.81),
        )

    write_recipe_lock(
        lock_root / "manifests/recipe_locks/0.json",
        replace(load_recipe_lock(lock_root / "manifests/recipe_locks/0.json"), protocol_hash="stale-protocol"),
    )
    with pytest.raises(ProtocolError, match="does not recompute"):
        run_outer_prior_recovery(replace(outer_config, artifact_root=tmp_path / "stale-outer"))
