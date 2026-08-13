from __future__ import annotations

from pathlib import Path

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router.actions import (
    build_action_library,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router.hashing import (
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router.persistence import (
    object_payload,
    persist_rows,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router.reports import (
    seal_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router.validation_prelabel import (
    validate_action_products,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router.validation_science import (
    ALL_RECONSTRUCTED_TABLE_MEMBERS,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import atomic_json


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = (
    ROOT
    / "src/midogpp_thesis/cvae/diagnostics"
    / "fixed_bank_case_directional_correctness_abstention_router"
)


def _write_action_products(root: Path) -> tuple[dict[str, object], ...]:
    rows = tuple(object_payload(action) for action in build_action_library())
    persist_rows(root / "tables/action_library.csv", rows)
    atomic_json(
        root / "manifests/action_library.json",
        seal_payload(
            "fixed_bank_cdca_action_library_manifest_v1",
            bindings={"actions_hash": canonical_hash(rows)},
            action_count=len(rows),
            physical_actions_per_target=10,
            labels_used=False,
            target_expert_used=False,
        ),
    )
    return rows


def test_action_reconstruction_rejects_canonical_table_and_rehashed_seal_tampering(
    tmp_path: Path,
) -> None:
    rows = _write_action_products(tmp_path)
    assert validate_action_products(tmp_path)["action_count"] == 90

    reordered = tmp_path / "reordered.csv"
    persist_rows(reordered, tuple(reversed(rows)))
    action_table = tmp_path / "tables/action_library.csv"
    original_bytes = action_table.read_bytes()
    action_table.write_bytes(reordered.read_bytes())
    with pytest.raises(ProtocolError, match="action table is not reconstructive"):
        validate_action_products(tmp_path)

    action_table.write_bytes(original_bytes)
    atomic_json(
        tmp_path / "manifests/action_library.json",
        seal_payload(
            "fixed_bank_cdca_action_library_manifest_v1",
            bindings={"actions_hash": canonical_hash(rows)},
            action_count=89,
            physical_actions_per_target=10,
            labels_used=False,
            target_expert_used=False,
        ),
    )
    with pytest.raises(ProtocolError, match="manifest is not reconstructive"):
        validate_action_products(tmp_path)


def test_full_validator_order_and_all_17_table_inventory_are_explicit() -> None:
    assert len(ALL_RECONSTRUCTED_TABLE_MEMBERS) == 17
    assert len(set(ALL_RECONSTRUCTED_TABLE_MEMBERS)) == 17
    assert ALL_RECONSTRUCTED_TABLE_MEMBERS == (
        "tables/action_library.csv",
        "tables/exact_nine_probability_index.csv",
        "tables/held_case_plans.csv",
        "tables/held_case_features.csv",
        "tables/support_response_counts.csv",
        "tables/donor_priors.csv",
        "tables/route_model_fits.csv",
        "tables/route_candidate_scores.csv",
        "tables/route_decisions.csv",
        "tables/method_predictions.csv",
        "tables/descriptive_method_predictions.csv",
        "tables/terminal_case_confusions.csv",
        "tables/terminal_method_metrics.csv",
        "tables/terminal_center_metrics.csv",
        "tables/terminal_contrasts.csv",
        "tables/router_identification_metrics.csv",
        "tables/feature_permutation_summary.csv",
    )

    source = (PACKAGE_ROOT / "validation.py").read_text(encoding="utf-8")
    content_first = source.index("content = validate_content_index(")
    input_admission = source.index("assert_input_fence(config)")
    prelabel = source.index("prelabel = reconstruct_prelabel(")
    plans = source.index("plan_products = reconstruct_plan_and_feature_products(")
    routes = source.index("route_products = reconstruct_route_products(")
    terminal = source.index("terminal_checks = reconstruct_terminal_products(")
    attestation = source.index("return _validate_attested_report(path, checks)")
    assert (
        content_first
        < input_admission
        < prelabel
        < plans
        < routes
        < terminal
        < attestation
    )

    for module in ("validation.py", "validation_prelabel.py", "validation_science.py"):
        module_source = (PACKAGE_ROOT / module).read_text(encoding="utf-8")
        assert "persist_rows(" not in module_source
        assert "persist_json(" not in module_source
        assert "atomic_json(" not in module_source
