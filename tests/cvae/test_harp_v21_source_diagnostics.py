from types import SimpleNamespace

import json
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v21.execution.source_diagnostics import (
    enforce_admitted_target_coverage,
    write_source_diagnostics,
)


def test_missing_frontier_fails_before_any_admission_or_report(tmp_path):
    crossfit = SimpleNamespace(public_payload=lambda: {"frontier_rows": [], "actual_menu_oracle_diagnostics": []})
    fitted = SimpleNamespace(state=SimpleNamespace(policy=SimpleNamespace(crossfit=crossfit)))
    with pytest.raises(ProtocolError, match="without its candidate frontier"):
        write_source_diagnostics(tmp_path, fitted=fitted, source_surface=None, config_hash="c" * 64)
    assert not tuple(tmp_path.rglob("*.json"))


@pytest.mark.parametrize("admitted,kind,aborts", [(True, "B", True), (True, "U", False), (False, "B", False)])
def test_label_free_target_coverage_does_not_force_routes(tmp_path, admitted, kind, aborts):
    routes = SimpleNamespace(cases=(SimpleNamespace(selected_kind=SimpleNamespace(value=kind)),))
    admission = {"source_only_admission": {"admitted": admitted}}
    if aborts:
        with pytest.raises(ProtocolError, match="evaluation truth remains closed"):
            enforce_admitted_target_coverage(tmp_path, routes=routes, policy_admission=admission)
    else:
        enforce_admitted_target_coverage(tmp_path, routes=routes, policy_admission=admission)
    report = json.loads((tmp_path / "reports/label_free_target_coverage.json").read_text())
    assert report["evaluation_labels_opened"] is False
    assert report["threshold_or_policy_changed"] is False
    assert report["nonbaseline_case_count"] == int(kind != "B")


def test_source_reports_persist_prediction_outcome_joins_without_raw_labels(tmp_path):
    from test_harp_v21_nested_policy import _source_surface, _config
    from midogpp_thesis.cvae.routing.correction_mass_router_v21.policy import fit_source_router
    from midogpp_thesis.cvae.routing.correction_mass_router_v21.hashing import canonical_hash
    menus, cap = _source_surface(centers=2, cases_per_center=8)
    policy = fit_source_router(menus, cap, config=_config())
    _, outcomes = cap.scoped(menus).derive_training_surface(menus)
    fitted = SimpleNamespace(state=SimpleNamespace(policy=policy))
    source = SimpleNamespace(state=SimpleNamespace(outcomes_by_outer=(("pooled",outcomes),)))
    paths = write_source_diagnostics(tmp_path, fitted=fitted, source_surface=source, config_hash="c"*64)
    assert len(paths)==3
    joins = json.loads((tmp_path / "reports/source_candidate_winner_joins.json").read_text())
    assert joins["candidate_row_count"] > 0 and joins["winner_row_count"] > 0
    assert joins["raw_sample_labels_persisted"] is False
    for row in joins["candidate_prediction_outcome_joins"]:
        body = dict(row)
        digest = body.pop("join_hash")
        assert canonical_hash(body)==digest
        assert (row["center_id"],row["case_id"]) not in [tuple(key) for key in row["training_case_keys"]]
    assert all("winner_gate_prediction" in row for row in joins["winner_gate_diagnostics"])
    outer = [row for row in joins["winner_gate_diagnostics"] if row["stage"]=="ALL_OUTER_OOF_DIAGNOSTIC"]
    assert len(outer)==len(menus)
