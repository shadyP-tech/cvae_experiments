from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v14.config import (
    load_config,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.harp_v14_execution.production_validation import (
    validate_model_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_v2_consumed_test_fixed_bank_harp_router_v14.yaml"
)


def test_canonical_v14_model_contract_passes_production_validation() -> None:
    config = load_config(CONFIG)

    validate_model_config(config)


@pytest.mark.parametrize(
    ("field", "stale_value"),
    (
        ("case_outcome_inventory", "typed_outer_H_query_q_case_identity"),
        (
            "active_menu_outcome_rule",
            "exact_menu_prediction_outcome_action_id_and_hash_match",
        ),
    ),
)
def test_v12_outcome_binding_literals_fail_closed(
    field: str, stale_value: str
) -> None:
    config = load_config(CONFIG)
    poisoned_model = dict(config.model)
    poisoned_model[field] = stale_value

    with pytest.raises(
        ProtocolError, match="HARP v14 frozen model/policy contract drifted"
    ):
        validate_model_config(SimpleNamespace(model=poisoned_model))
