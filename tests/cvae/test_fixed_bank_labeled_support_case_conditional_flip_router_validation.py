from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_labeled_support_case_conditional_flip_router.bundle import (
    CONTENT_INDEX_MEMBERS,
    write_content_index,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_labeled_support_case_conditional_flip_router.protocol import (
    canonical_consumed_test_protocol,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_labeled_support_case_conditional_flip_router.reports import (
    run_state_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_labeled_support_case_conditional_flip_router.validation import (
    assert_completed_bundle_binding,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import atomic_json


def _completed_fixture(root: Path) -> tuple[object, dict[str, object]]:
    config = SimpleNamespace(contract_hash="c" * 64)
    protocol = canonical_consumed_test_protocol()
    for ordinal, member in enumerate(CONTENT_INDEX_MEMBERS):
        path = root / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"member-{ordinal}\n".encode("ascii"))
    content = write_content_index(
        root,
        config_contract_hash=config.contract_hash,
        protocol_contract_hash=protocol.contract_hash,
    )
    checks: dict[str, object] = {
        "schema_version": "fixed_bank_labeled_support_flip_validation_v1",
        "status": "PASS",
        "content_hash": content["content_hash"],
        "config_contract_hash": config.contract_hash,
        "protocol_contract_hash": protocol.contract_hash,
        "scientific_factories_replayed": True,
        "nonrepairing_validation": True,
        "closed_world": True,
    }
    atomic_json(root / "reports/validation_report.json", checks)
    atomic_json(
        root / "reports/run_state.json",
        run_state_payload("COMPLETE", "COMPLETE"),
    )
    return config, checks


def test_completed_binding_is_lightweight_but_exact(tmp_path: Path) -> None:
    config, checks = _completed_fixture(tmp_path)

    assert_completed_bundle_binding(
        tmp_path,
        config=config,
        expected_checks=checks,
    )


def test_completed_binding_cannot_replace_the_mandatory_full_replay(
    tmp_path: Path,
) -> None:
    config, checks = _completed_fixture(tmp_path)
    checks.pop("scientific_factories_replayed")
    atomic_json(tmp_path / "reports/validation_report.json", checks)

    with pytest.raises(ProtocolError, match="not bound to its full validation"):
        assert_completed_bundle_binding(
            tmp_path,
            config=config,
            expected_checks=checks,
        )


@pytest.mark.parametrize("drift", ("content", "report", "state", "extra"))
def test_completed_binding_rejects_post_validation_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    config, checks = _completed_fixture(tmp_path)
    if drift == "content":
        (tmp_path / CONTENT_INDEX_MEMBERS[-1]).write_bytes(b"changed\n")
    elif drift == "report":
        atomic_json(
            tmp_path / "reports/validation_report.json",
            {**checks, "status": "CHANGED"},
        )
    elif drift == "state":
        atomic_json(
            tmp_path / "reports/run_state.json",
            run_state_payload("RUNNING", "FINALIZATION"),
        )
    else:
        (tmp_path / "reports/foreign.json").write_bytes(b"{}\n")

    with pytest.raises(ProtocolError):
        assert_completed_bundle_binding(
            tmp_path,
            config=config,
            expected_checks=checks,
        )
