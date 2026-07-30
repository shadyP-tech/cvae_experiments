from __future__ import annotations

import json
from pathlib import Path

import pytest

from midogpp_thesis.real_features.classifier_reference.matched_reference import (
    canonical_matched_reference_specs,
)
from midogpp_thesis.real_features.classifier_reference.physical_multiscale_center_pooling.decision_lock import (
    read_decision_lock,
    write_decision_lock,
)
from midogpp_thesis.real_features.classifier_reference.physical_multiscale_center_pooling.selection import (
    RepresentationDecision,
)
from midogpp_thesis.real_features.classifier_reference.protocol import ProtocolError


def test_decision_lock_binds_all_representation_specs_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    spec = canonical_matched_reference_specs(classifier_seed=23)[0]
    specs = {
        "canonical_a": spec,
        "jpeg_center_b": spec,
        "physical_multiscale_center_c": spec,
    }
    decision = RepresentationDecision(
        outer_target_center="0",
        selected_representation="canonical_a",
        selected_classifier_hash=spec.config_hash,
        canonical_a_classifier_hash=spec.config_hash,
        source_centers=("1", "2"),
        mean_delta=0.0,
        worst_delta=0.0,
        strict_wins=0,
        gate_passed=False,
        selected_spec=spec,
        canonical_a_spec=spec,
        representation_specs=specs,
    )
    lock = write_decision_lock(
        tmp_path,
        decision=decision,
        config_hash="config",
        candidate_grid_hash="candidates",
        selector_rows=({"outer_target_center": "0", "cell": 1},),
        input_hashes={"cache": "sha"},
    )

    loaded = read_decision_lock(lock.path)
    assert loaded.decision_hash == lock.decision_hash
    assert loaded.payload["posthoc_rows_used_for_lock"] is False
    assert set(loaded.payload["representation_classifier_specs"]) == set(specs)

    payload = json.loads(lock.path.read_text(encoding="utf-8"))
    payload["selected_representation"] = "jpeg_center_b"
    lock.path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProtocolError, match="hash mismatch"):
        read_decision_lock(lock.path)

