from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from midogpp_thesis.cvae.preservation import prior_recovery_common
from midogpp_thesis.cvae.preservation.prior_recovery import (
    run_source_inner_prior_recovery,
)
from midogpp_thesis.cvae.preservation.prior_recovery_runtime_cache import (
    FeatureFrameCache,
)
from midogpp_thesis.real_features.classifier_reference.protocol import ProtocolError
from tests.cvae.prior_recovery_test_support import (
    prior_recovery_config,
    write_prior_recovery_fixture,
)


def test_feature_frame_cache_is_exact_replayable_and_corruption_fails_closed(
    tmp_path: Path,
) -> None:
    embeddings = np.random.default_rng(7).normal(size=(24, 6))
    cache = FeatureFrameCache(tmp_path)
    identity = {
        "expert_id": "H0_I1",
        "source_train_embeddings": embeddings,
        "fit_centers": ("2", "3"),
        "fit_row_hash": "fit-rows",
        "requested_dim": 4,
        "manifest_hash": "manifest",
        "feature_cache_hash": "features",
        "protocol_hash": "protocol",
        "code_version": "test",
    }

    fresh, fresh_hit = cache.fit_or_load(**identity)
    replay, replay_hit = cache.fit_or_load(**identity)
    assert fresh_hit is False
    assert replay_hit is True
    assert replay.state_hash == fresh.state_hash
    np.testing.assert_array_equal(replay.transform(embeddings), fresh.transform(embeddings))
    np.testing.assert_array_equal(
        replay.inverse_transform(replay.transform(embeddings)),
        fresh.inverse_transform(fresh.transform(embeddings)),
    )

    for drift in (
        {"fit_row_hash": "different-rows"},
        {"feature_cache_hash": "different-features"},
        {"protocol_hash": "protocol-v2"},
        {"code_version": "test-v2"},
    ):
        drifted, drifted_hit = cache.fit_or_load(**(identity | drift))
        assert drifted_hit is False
        assert drifted.state_hash == fresh.state_hash

    sidecars = list((tmp_path / "runtime_cache/feature_frames/by_key").glob("*.json"))
    sidecar = next(
        path
        for path in sidecars
        if json.loads(path.read_text(encoding="utf-8"))["frame_cache_key"]["protocol_hash"]
        == "protocol"
        and json.loads(path.read_text(encoding="utf-8"))["frame_cache_key"]["fit_row_hash"]
        == "fit-rows"
        and json.loads(path.read_text(encoding="utf-8"))["frame_cache_key"]["feature_cache_hash"]
        == "features"
        and json.loads(path.read_text(encoding="utf-8"))["frame_cache_key"]["code_version"]
        == "test"
    )
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    escaped = dict(record) | {"relative_path": "../../outside.npz"}
    sidecar.write_text(
        json.dumps(escaped, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ProtocolError, match="noncanonical frame path"):
        cache.fit_or_load(**identity)
    sidecar.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (tmp_path / record["relative_path"]).write_bytes(b"corrupt")
    with pytest.raises(ProtocolError, match="missing or corrupt"):
        cache.fit_or_load(**identity)


def test_source_inner_rerun_uses_exact_checkpoints_and_keeps_timing_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, feature_cache = write_prior_recovery_fixture(tmp_path / "fixture")
    config = prior_recovery_config(
        mode="source_inner",
        artifact_root=tmp_path / "unused-config-root",
        manifest=manifest,
        cache=feature_cache,
    )
    override_root = tmp_path / "source-override"
    root = run_source_inner_prior_recovery(config, artifact_root=override_root)
    assert root == override_root
    evidence_before = (root / "manifests/selection_evidence_manifest.json").read_bytes()
    checkpoints_before = (root / "manifests/checkpoint_index.json").read_bytes()

    def unexpected_training(*args: object, **kwargs: object) -> object:
        raise AssertionError("an exact-key rerun must not retrain a CVAE")

    monkeypatch.setattr(prior_recovery_common, "train_cvae", unexpected_training)
    replay_root = run_source_inner_prior_recovery(config, artifact_root=override_root)
    assert replay_root == root
    assert (root / "manifests/selection_evidence_manifest.json").read_bytes() == evidence_before
    assert (root / "manifests/checkpoint_index.json").read_bytes() == checkpoints_before
    with (root / "tables/runtime_timings.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        timing_rows = list(csv.DictReader(handle))
    assert timing_rows
    assert all(row["used_for_selection"] == "false" for row in timing_rows)
    assert all(row["claim_scope"] == "diagnostic_only" for row in timing_rows)
    assert all(
        row["cache_status"] == "hit"
        for row in timing_rows
        if row["phase"] in {"pca_frame", "cvae_training"}
    )

    sidecar = next((root / "checkpoints/by_training_key").glob("*.json"))
    sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
    escaped = dict(sidecar_payload) | {"relative_path": "../../outside.pt"}
    sidecar.write_text(
        json.dumps(escaped, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ProtocolError, match="noncanonical checkpoint path"):
        run_source_inner_prior_recovery(config, artifact_root=override_root)
    sidecar.write_text(
        json.dumps(sidecar_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sidecar_payload["classifier_spec_hash"] = "forged"
    sidecar.write_text(
        json.dumps(sidecar_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ProtocolError, match="differs from the requested runtime identity"):
        run_source_inner_prior_recovery(config, artifact_root=override_root)
    indexed_record = next(
        record
        for record in json.loads(
            (root / "manifests/checkpoint_index.json").read_text(encoding="utf-8")
        )["records"]
        if record["training_key_hash"] == sidecar_payload["training_key_hash"]
    )
    sidecar_payload["classifier_spec_hash"] = indexed_record["classifier_spec_hash"]
    sidecar_payload["initialization_hash"] = ""
    sidecar.write_text(
        json.dumps(sidecar_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ProtocolError, match="reproducibility metadata"):
        run_source_inner_prior_recovery(config, artifact_root=override_root)
    run_state = json.loads((root / "reports/run_state.json").read_text(encoding="utf-8"))
    runtime_summary = json.loads(
        (root / "reports/runtime_summary.json").read_text(encoding="utf-8")
    )
    assert run_state["status"] == runtime_summary["status"] == "FAILED"
