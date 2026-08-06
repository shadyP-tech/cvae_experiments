from __future__ import annotations

from pathlib import Path

import pytest

from midogpp_thesis.cvae.expert_bank.cli import main as expert_bank_main
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.config import (
    load_promotion_config,
)
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    N_EXPERTS,
    TRAINING_SEEDS,
    legal_routing_sources,
)
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.serialization import (
    source_frame_from_payload,
)
from midogpp_thesis.cvae.preservation.uniform_b_optimized_prior.core import (
    fit_optimized_source_frame,
)
from midogpp_thesis.cvae.preservation.independent_source import IndependentSourceData


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "experiments/midogpp/stages/30_expert_bank/configs/uniform_b_v2_routing_promotion_v1.yaml"


def test_promotion_config_locks_whole_bank_and_strict_gates() -> None:
    config = load_promotion_config(CONFIG)

    assert config.centers == CENTERS
    assert config.training_seeds == TRAINING_SEEDS
    assert N_EXPERTS == 27
    assert config.min_ps_mean_bacc == 0.70
    assert config.min_ps_seed_bacc == 0.75
    assert config.min_ps_minus_p0 == 0.005
    assert config.max_posterior_ceiling_gap == 0.01
    assert config.claim_boundary["all_27_experts_retained"] is True
    assert config.claim_boundary["target_labels_used_for_individual_expert_selection"] is False
    assert config.claim_boundary["may_feed_deployable_selection"] is True


def test_equal_union_control_excludes_only_target_and_has_fixed_eight_sources() -> None:
    for target in CENTERS:
        sources = legal_routing_sources(target)
        assert len(sources) == 8
        assert target not in sources
        assert set(sources) == set(CENTERS).difference({target})


def test_source_frame_serialization_round_trip() -> None:
    import numpy as np

    rng = np.random.default_rng(7)
    rows = 256
    source = IndependentSourceData(
        center="0",
        embeddings=rng.normal(size=(rows, 3840)).astype(np.float32),
        labels=tuple([0] * (rows // 2) + [1] * (rows // 2)),
        case_ids=tuple(f"case-{index}" for index in range(rows)),
        sample_ids=tuple(f"sample-{index}" for index in range(rows)),
        image_ids=tuple(f"image-{index}" for index in range(rows)),
        row_hash="source-row-hash",
        case_hash="source-case-hash",
        image_hash="source-image-hash",
    )
    fitted = fit_optimized_source_frame(source)
    restored = source_frame_from_payload(fitted.to_payload())

    assert restored.state_hash == fitted.state_hash
    assert restored.source_center == "0"
    assert restored.frame.output_dim == 256


def test_cli_help_registers_v2_promotion(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        expert_bank_main(["--help"])
    assert exc.value.code == 0
    assert "uniform-b-v2-routing-promotion" in capsys.readouterr().out
