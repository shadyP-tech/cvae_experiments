from __future__ import annotations

from pathlib import Path

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v17.config import (
    load_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v17.runner import (
    HARP_V17_RUN_CONFIRMATION_TOKEN,
    _enforce_source_policy_admission,
    dry_run_harp_stage90_v17,
    inspect_harp_stage90_v17,
    run_harp_stage90_v17,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v17.source_seal import (
    SOURCE_ENTRYPOINT_PATTERNS,
    build_source_snapshot_payload,
    source_members,
    source_snapshot_identity,
)
from midogpp_thesis.cvae.protocol import ProtocolError


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
    / "uniform_b_v2_consumed_test_fixed_bank_harp_router_v17.yaml"
)


def test_v17_execution_confirmation_precedes_config_and_path_access() -> None:
    with pytest.raises(ProtocolError, match="exact confirmation token"):
        run_harp_stage90_v17(
            object(),
            artifact_root="artifact://must-not-be-resolved",
            confirmation_token=None,
        )
    with pytest.raises(ProtocolError, match="typed configuration"):
        run_harp_stage90_v17(
            object(),
            artifact_root="artifact://must-not-be-resolved",
            confirmation_token=HARP_V17_RUN_CONFIRMATION_TOKEN,
        )


def test_v17_planned_inspection_and_dry_run_are_path_free(tmp_path: Path) -> None:
    config = load_config(CONFIG)
    impossible = tmp_path / "must-not-exist"

    inspection = inspect_harp_stage90_v17(config)
    dry_run = dry_run_harp_stage90_v17(config, artifact_root=impossible)

    assert inspection["status"] == "PLANNED_NEEDS_SEPARATE_EXECUTION_AMENDMENT"
    assert inspection["paths_resolved"] is False
    assert inspection["authorization_probed"] is False
    assert inspection["source_train_labels_opened"] is False
    assert dry_run["status"] == "NEEDS_SEPARATE_EXECUTION_AMENDMENT"
    assert dry_run["paths_resolved"] is False
    assert dry_run["artifact_root_argument_recorded"] is False
    assert dry_run["filesystem_mutations"] == 0
    assert not impossible.exists()


def test_v17_no_frontier_aborts_before_target_actions(tmp_path: Path) -> None:
    config = load_config(CONFIG)
    policy_path = tmp_path / "manifests/source_policy_admission_seal.json"
    policy_path.parent.mkdir(parents=True)
    policy_path.write_text("{}\n", encoding="utf-8")
    seal = {
        "source_only_admission": {
            "status": "NO_NONZERO_SAFE_OOF_COVERAGE",
            "admitted": False,
        },
        "seal_hash": "a" * 64,
    }

    with pytest.raises(ProtocolError, match="target action construction is forbidden"):
        _enforce_source_policy_admission(
            config=config,
            root=tmp_path,
            policy_path=policy_path,
            policy_admission=seal,
        )

    report = tmp_path / "reports/source_policy_nonadmission.json"
    assert report.is_file()
    assert not (tmp_path / "stores/target_case_actions").exists()
    assert not (tmp_path / "reports/evaluation_label_access.json").exists()


def test_v17_source_snapshot_closure_explicitly_excludes_v1_through_v16() -> None:
    identity = source_snapshot_identity(ROOT)
    payload = build_source_snapshot_payload(ROOT)
    members = tuple(
        path.relative_to(ROOT / "src").as_posix() for path in source_members(ROOT)
    )

    assert identity["source_snapshot_member_count"] == len(members)
    assert any("harp_v17_execution/production.py" in member for member in members)
    assert any(
        "pooled_pairwise_selected_policy_router_v17/policy.py" in member
        for member in members
    )
    assert not any(
        f"fixed_bank_harp_router_v{version}/" in member
        or f"harp_v{version}_execution/" in member
        for version in range(1, 17)
        for member in members
    )
    assert set(SOURCE_ENTRYPOINT_PATTERNS).issubset(members)
    assert payload["v17_scientific_import_closure_to_dispatch_boundary_sealed"] is True
    assert (
        payload[
            "v17_scientific_import_closure_to_dispatch_boundary_predecessor_free"
        ]
        is True
    )
    assert payload["shared_dispatch_entrypoint_bytes_sealed"] is True
    assert payload["shared_dispatch_entrypoint_imports_traversed"] is False
    assert payload["shared_dispatch_entrypoints_outside_predecessor_free_closure"] is True
    assert payload["full_dispatch_transitive_local_import_closure_claimed"] is False
    assert "transitive_local_import_closure_sealed" not in payload
    assert (
        "dispatch_leaf_policy="
        "shared_dispatch_entrypoints_byte_sealed_without_import_traversal"
        in identity["source_snapshot_member_pattern"]
    )
