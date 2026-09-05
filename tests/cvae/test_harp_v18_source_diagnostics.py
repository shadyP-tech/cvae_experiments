from types import SimpleNamespace

import json
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v18.execution.source_diagnostics import (
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
