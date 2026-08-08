from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.residual_topup.hashing import canonical_sha256
from midogpp_thesis.cvae.routing.utility_aligned import (
    TargetCandidateComponents,
    build_target_feature_production,
    target_feature_production_from_payload,
)
from midogpp_thesis.cvae.routing.utility_aligned.target_features import target_sources
from midogpp_thesis.cvae.routing.utility_aligned_target_support_surface import runner
from midogpp_thesis.cvae.routing.utility_aligned_target_support_surface.artifact_writer import (
    feature_payload,
)
from midogpp_thesis.cvae.routing.utility_aligned_target_support_surface.contracts import (
    REQUIRED_FILES,
)
from midogpp_thesis.cvae.routing.utility_aligned_target_support_surface.inputs import (
    parse_support_rows,
)


SEEDS = (17, 42, 101)


def _components(target: str = "0") -> dict[str, TargetCandidateComponents]:
    case_ids = tuple(f"case-{index:02d}" for index in range(8))
    result = {}
    for source_index, source in enumerate(target_sources(target)):
        reconstruction = {
            seed: {
                label: np.linspace(0.1, 0.8, 8, dtype=np.float64)
                + 0.01 * source_index
                + 0.001 * label
                for label in (0, 1)
            }
            for seed in SEEDS
        }
        kl = {
            seed: {
                label: np.linspace(0.05, 0.4, 8, dtype=np.float64)
                + 0.005 * source_index
                + 0.001 * label
                for label in (0, 1)
            }
            for seed in SEEDS
        }
        support_means = {
            case_id: np.full(3840, float(index), dtype=np.float64)
            for index, case_id in enumerate(case_ids)
        }
        generated_means = {
            (training_seed, generation_seed): np.full(
                3840,
                float(source_index) + training_seed / 1000 + generation_seed / 10000,
                dtype=np.float64,
            )
            for training_seed in SEEDS
            for generation_seed in SEEDS
        }
        result[source] = TargetCandidateComponents(
            candidate_source=source,
            reconstruction_by_training_seed=reconstruction,
            normalized_ps_kl_by_training_seed=kl,
            support_case_mean_embeddings=support_means,
            generated_mean_by_seed_pair=generated_means,
            metadata_similarity=0.5,
        )
    return result


def _production():
    return build_target_feature_production(
        target_id="0",
        case_ids=tuple(f"case-{index:02d}" for index in range(8)),
        components_by_source=_components(),
        bootstrap_seed=60920000,
        bootstrap_replicate_count=32,
    )


def test_label_free_target_features_use_exact_seed_grid_and_case_mmd_bootstrap() -> None:
    production = _production()

    assert len(production.point_rows) == 8 * 3 * 3
    assert len(production.bootstrap_surfaces) == 32
    point_mmd = production.point_rows[0].distribution_mmd
    bootstrap_mmd = {
        surface.rows[0].distribution_mmd for surface in production.bootstrap_surfaces
    }
    assert any(value != point_mmd for value in bootstrap_mmd)
    assert len(bootstrap_mmd) > 1


def test_target_components_reject_noncanonical_training_seed() -> None:
    components = _components()
    source = next(iter(components))
    value = components[source]
    bad = dict(value.reconstruction_by_training_seed)
    bad[19] = bad.pop(17)
    with pytest.raises(ProtocolError, match="exact seeds"):
        replace(value, reconstruction_by_training_seed=bad)


def test_target_payload_malformed_numeric_fails_as_protocol_error() -> None:
    payload = dict(feature_payload(_production()))
    point_rows = [dict(value) for value in payload["point_rows"]]
    point_rows[0]["training_seed"] = {"not": "numeric"}
    payload["point_rows"] = point_rows
    payload["target_feature_hash"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "target_feature_hash"}
    )
    with pytest.raises(ProtocolError, match="malformed|numeric"):
        target_feature_production_from_payload(payload)


def test_target_support_reservation_rejects_cross_center_case_reuse() -> None:
    centers = ("0", "1", "2", "3", "5", "6", "7", "8", "9")
    cases = {
        center: [f"{center}-case-{index:02d}" for index in range(8)]
        for center in centers
    }
    cases["1"][0] = cases["0"][0]
    rows = {
        center: [
            {
                "row_ordinal": index,
                "sample_id": f"{center}-sample-{index}",
                "case_id": case_id,
                "center": center,
                "cache_shard_path": f"shards/{center}.npy",
                "cache_row_index": index,
            }
            for index, case_id in enumerate(values)
        ]
        for center, values in cases.items()
    }
    with pytest.raises(ProtocolError, match="unique cases"):
        parse_support_rows(
            {
                "support_case_ids_by_center": cases,
                "support_rows_by_center": rows,
            }
        )


def test_target_support_complete_fast_path_and_incomplete_complete_guard(
    monkeypatch, tmp_path: Path
) -> None:
    config = SimpleNamespace(artifact_root=tmp_path)
    for member in REQUIRED_FILES:
        path = tmp_path / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "validate_target_support_surface_bundle",
        lambda root: {"status": "COMPLETE", "root": str(root)},
    )
    monkeypatch.setattr(
        runner,
        "require_target_support_inputs_ready",
        lambda _config: (_ for _ in ()).throw(AssertionError("fresh input reopened")),
    )
    result = runner.run_utility_aligned_target_support_surface(
        config, workspace_validator=lambda _config: None
    )
    assert result["status"] == "COMPLETE"

    missing = tmp_path / REQUIRED_FILES[-1]
    missing.unlink()
    (tmp_path / "reports/run_state.json").write_text(
        '{"status":"COMPLETE"}\n', encoding="utf-8"
    )
    with pytest.raises(ProtocolError, match="COMPLETE artifact is incomplete"):
        runner.run_utility_aligned_target_support_surface(
            config, workspace_validator=lambda _config: None
        )
