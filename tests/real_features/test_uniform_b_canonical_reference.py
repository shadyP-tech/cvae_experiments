from __future__ import annotations

from pathlib import Path

import pytest

from midogpp_thesis.real_features.classifier_reference.protocol import ProtocolError
from midogpp_thesis.real_features.classifier_reference.uniform_b_reference.config import (
    EXPECTED_FEATURE_DIM,
    EXPECTED_TRAIN_ROWS,
    PROMOTION_REVIEW_ID,
    load_uniform_b_canonical_cache_config,
    load_uniform_b_canonical_reference_config,
)
from midogpp_thesis.real_features.classifier_reference.uniform_b_reference.runner import (
    _representation_lock,
    _test_consumption_ledger,
)


CACHE_CONFIG = Path(
    "datasets/midogpp/configs/uniform_b_canonical_train_cache_v1.yaml"
)
REFERENCE_CONFIG = Path(
    "experiments/midogpp/stages/10_real_feature_reference/configs/"
    "uniform_b_canonical_real_feature_reference_v1.yaml"
)


def test_uniform_b_promotion_configs_freeze_review_and_fresh_tuning() -> None:
    cache = load_uniform_b_canonical_cache_config(CACHE_CONFIG)
    reference = load_uniform_b_canonical_reference_config(REFERENCE_CONFIG)

    assert cache.expected_train_rows == EXPECTED_TRAIN_ROWS == 9648
    assert cache.expected_feature_dim == EXPECTED_FEATURE_DIM == 3840
    assert reference.review["review_id"] == PROMOTION_REVIEW_ID
    assert reference.review["classifier_locks_imported_from_diagnostics"] is False
    assert reference.review["test_split_consumed_for_representation_adoption"] is True
    assert reference.review["canonical_a_retained"] is True
    assert reference.review["automatic_downstream_migration"] is False
    assert len(reference.classifier_specs) == 10


def test_uniform_b_promotion_config_rejects_unconsumed_test_claim(
    tmp_path: Path,
) -> None:
    text = REFERENCE_CONFIG.read_text(encoding="utf-8").replace(
        "test_split_consumed_for_representation_adoption: true",
        "test_split_consumed_for_representation_adoption: false",
    )
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ProtocolError, match="promotion review"):
        load_uniform_b_canonical_reference_config(path)


def test_uniform_b_promotion_lock_retains_a_and_blocks_automatic_migration() -> None:
    config = load_uniform_b_canonical_reference_config(REFERENCE_CONFIG)
    confirmation = {
        "decision": "CONFIRMED_WITHIN_CENTER",
        "paired_mean_delta": 0.063,
        "strict_wins": 9,
    }
    lock = _representation_lock(config, confirmation)
    ledger = _test_consumption_ledger(confirmation)

    assert lock["canonical_a_retained"] is True
    assert lock["automatic_downstream_migration"] is False
    assert ledger["status"] == "CONSUMED_FOR_REPRESENTATION_ADOPTION"
    assert ledger["may_be_reused_as_fresh_representation_selection_evidence"] is False
