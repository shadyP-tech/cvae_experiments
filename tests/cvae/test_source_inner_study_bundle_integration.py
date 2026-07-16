from __future__ import annotations

import pytest

from midogpp_thesis.cvae.preservation.source_inner_studies import (
    fisher_validation,
    prior_validation,
)
from midogpp_thesis.cvae.preservation.source_inner_studies.fisher_validation import (
    validate_fisher_study_bundle,
)
from midogpp_thesis.cvae.preservation.source_inner_studies.prior_validation import (
    validate_prior_study_bundle,
)
from midogpp_thesis.cvae.preservation.source_inner_studies.validation_common import (
    read_csv,
)
from midogpp_thesis.cvae.reporting import write_csv_rows
from midogpp_thesis.real_features.classifier_reference.protocol import ProtocolError

from tests.cvae.source_inner_study_bundle_test_support import (
    assert_exact_fixture_surface,
    fisher_fixture_config,
    fixture_fisher_decisions,
    fixture_prior_decisions,
    patch_common_fixture_validators,
    prior_fixture_config,
    write_fisher_fixture_bundle,
    write_prior_fixture_bundle,
)


def test_prior_writer_public_validator_and_selection_hash_tamper_rejection(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "prior-study"
    config = prior_fixture_config(root)
    write_prior_fixture_bundle(root, config)
    patch_common_fixture_validators(monkeypatch, prior_validation, root)
    monkeypatch.setattr(
        prior_validation,
        "_validate_prior_state_record",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(prior_validation, "_decisions", fixture_prior_decisions)

    assert_exact_fixture_surface(
        root,
        config,
        "manifests/learned_prior_state_index.json",
    )
    decisions = validate_prior_study_bundle(root, expected_config=config)
    assert decisions["0"]["status"] == "FIXTURE_E_PASS"

    delta_path = root / "tables/paired_deltas.csv"
    rows = read_csv(delta_path)
    rows[0]["preservation_ratio_delta"] = "0.081"
    write_csv_rows(delta_path, rows)
    with pytest.raises(ProtocolError, match="selection-evidence hash mismatch"):
        validate_prior_study_bundle(root, expected_config=config)


def test_fisher_writer_public_validator_and_selection_hash_tamper_rejection(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "fisher-study"
    config = fisher_fixture_config(root)
    write_fisher_fixture_bundle(root, config)
    patch_common_fixture_validators(monkeypatch, fisher_validation, root)

    def fixture_states(index, *, config):
        del config
        return {
            (
                str(record["outer_target_center"]),
                str(record["inner_pseudo_target_center"]),
            ): record
            for record in index["records"]
        }

    monkeypatch.setattr(fisher_validation, "_validate_fisher_states", fixture_states)
    monkeypatch.setattr(fisher_validation, "_decisions", fixture_fisher_decisions)

    assert_exact_fixture_surface(
        root,
        config,
        "manifests/task_fisher_shrinkage_state_index.json",
    )
    decisions = validate_fisher_study_bundle(root, expected_config=config)
    assert decisions["0"]["status"] == "FIXTURE_ALPHA_0_10_PASS"

    delta_path = root / "tables/paired_deltas.csv"
    rows = read_csv(delta_path)
    rows[0]["preservation_ratio_delta"] = "0.041"
    write_csv_rows(delta_path, rows)
    with pytest.raises(ProtocolError, match="selection-evidence hash mismatch"):
        validate_fisher_study_bundle(root, expected_config=config)
