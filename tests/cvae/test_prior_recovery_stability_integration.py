from __future__ import annotations

import json
from pathlib import Path

import pytest

from midogpp_thesis.cvae.preservation.prior_recovery_stability import (
    run_source_inner_training_seed_stability,
)
from midogpp_thesis.cvae.expert_bank.recipe_lock_loader import (
    load_consensus_recipe_for_fold,
)
from midogpp_thesis.cvae.preservation.prior_recovery_stability_artifacts import (
    validate_stability_bundle,
)
from midogpp_thesis.real_features.classifier_reference.protocol import ProtocolError

from tests.cvae.prior_recovery_test_support import (
    prior_recovery_config,
    write_prior_recovery_fixture,
)


def test_tiny_stability_run_writes_recomputable_seed_and_consensus_locks(
    tmp_path: Path,
) -> None:
    manifest, cache = write_prior_recovery_fixture(tmp_path / "inputs")
    root = tmp_path / "stability"
    config = prior_recovery_config(
        mode="stability",
        artifact_root=root,
        manifest=manifest,
        cache=cache,
    )
    run_source_inner_training_seed_stability(config)  # type: ignore[arg-type]
    locks = validate_stability_bundle(root, expected_config=config)  # type: ignore[arg-type]

    assert set(locks) == {"0"}
    assert locks["0"].training_seeds == (17, 42)
    publication_path = root / "reports/publication_state.json"
    assert json.loads(publication_path.read_text(encoding="utf-8"))["status"] == (
        "PUBLISHED"
    )
    assert (
        root / "manifests/training_seed_recipe_locks/seed17/0.json"
    ).is_file()
    assert (
        root / "manifests/training_seed_recipe_locks/seed42/0.json"
    ).is_file()
    consensus_path = root / "manifests/consensus_recipe_locks/0.json"
    before_resume = consensus_path.read_text(encoding="utf-8")

    run_source_inner_training_seed_stability(config)  # type: ignore[arg-type]
    assert consensus_path.read_text(encoding="utf-8") == before_resume
    validate_stability_bundle(root, expected_config=config)  # type: ignore[arg-type]
    assert load_consensus_recipe_for_fold(
        root,
        outer_target_center="0",
    ).outer_target_center == "0"
    with pytest.raises(ProtocolError, match="no consensus lock"):
        load_consensus_recipe_for_fold(root, outer_target_center="1")

    rng_path = root / "tables/rng_pairing_audit.csv"
    pristine_rng = rng_path.read_text(encoding="utf-8")
    rng_path.write_text(
        pristine_rng.replace(
            "PAIRED_BY_GENERATION_SEED",
            "TAMPERED",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProtocolError, match="RNG pairing audit"):
        validate_stability_bundle(root, expected_config=config)  # type: ignore[arg-type]
    rng_path.write_text(pristine_rng, encoding="utf-8")

    publication = json.loads(publication_path.read_text(encoding="utf-8"))
    publication["status"] = "PENDING"
    publication_path.write_text(
        json.dumps(publication, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ProtocolError, match="not published"):
        load_consensus_recipe_for_fold(root, outer_target_center="0")
    publication["status"] = "PUBLISHED"
    publication_path.write_text(
        json.dumps(publication, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    payload = json.loads(consensus_path.read_text(encoding="utf-8"))
    payload["reason"] = "tampered"
    consensus_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ProtocolError, match="hash mismatch"):
        validate_stability_bundle(root, expected_config=config)  # type: ignore[arg-type]
    with pytest.raises(ProtocolError, match="hash mismatch"):
        load_consensus_recipe_for_fold(root, outer_target_center="0")
