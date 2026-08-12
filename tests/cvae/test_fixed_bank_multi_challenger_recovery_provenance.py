from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_multi_challenger_hierarchical_flip_router import (  # noqa: E501
    recovery,
    recovery_provenance,
    runner,
    validation,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_multi_challenger_hierarchical_flip_router.hashing import (  # noqa: E501
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_multi_challenger_hierarchical_flip_router.recovery_provenance import (  # noqa: E501
    assert_repair_repository_state_unchanged,
    fresh_recovery_audit_payload,
    original_repository_state_from_provenance,
    recovery_audit_payload,
    sealed_recovery_input_hashes,
    validate_recovery_audit_payload,
)
from midogpp_thesis.cvae.protocol import ProtocolError


ORIGINAL_STATE = {
    "repository_revision": "1" * 40,
    "repository_dirty": True,
    "repository_status_hash": "2" * 64,
}
REPAIR_STATE = {
    "repository_revision": "3" * 40,
    "repository_dirty": False,
    "repository_status_hash": "4" * 64,
}
REUSED_HASHES = {
    "source_stream_lock_hash": "5" * 64,
    "global_prediction_seal_hash": "6" * 64,
    "prelabel_feature_surface_hash": "7" * 64,
    "fold_plan_surface_hash": "8" * 64,
}


def test_fresh_audit_is_hash_bound_and_carries_no_recovery_claim() -> None:
    audit = dict(fresh_recovery_audit_payload())

    assert audit["recovery_used"] is False
    assert audit["failed_run_state_hash"] is None
    assert audit["donor_models_refit_during_recovery"] is False
    assert audit["source_generation_recomputed_during_recovery"] is False
    assert audit["predictions_recomputed_during_recovery"] is False
    assert audit["decisions_recomputed_during_recovery"] is False
    assert audit["terminal_evaluation_recomputed_during_recovery"] is False
    assert (
        audit[
            "labels_reopened_only_for_deterministic_reconstruction_during_recovery"
        ]
        is False
    )
    assert audit["terminal_consumed_test_diagnostic_only"] is True
    assert audit["policy_promotion_authorized"] is False
    unhashed = {
        key: value
        for key, value in audit.items()
        if key != "recovery_audit_hash"
    }
    assert audit["recovery_audit_hash"] == canonical_hash(unhashed)
    assert (
        validate_recovery_audit_payload(
            audit,
            original_repository_state=ORIGINAL_STATE,
            current_repository_state=ORIGINAL_STATE,
            **REUSED_HASHES,
        )
        == audit
    )


def test_recovery_audit_binds_exact_failure_seals_and_git_states() -> None:
    audit = dict(
        recovery_audit_payload(
            original_repository_state=ORIGINAL_STATE,
            repair_repository_state=REPAIR_STATE,
            **REUSED_HASHES,
        )
    )

    assert audit["recovery_used"] is True
    assert audit["failed_run_state_hash"] == canonical_hash(
        recovery.FAILED_MAPPINGPROXY_STATE
    )
    assert audit["failed_phase"] == recovery.FAILED_MAPPINGPROXY_STATE["phase"]
    assert audit["failed_error"] == recovery.FAILED_MAPPINGPROXY_STATE["error"]
    assert audit["reused_source_stream_lock_hash"] == "5" * 64
    assert audit["reused_global_prediction_seal_hash"] == "6" * 64
    assert audit["reused_prelabel_feature_surface_hash"] == "7" * 64
    assert audit["reused_fold_plan_surface_hash"] == "8" * 64
    assert audit["donor_models_refit_during_recovery"] is True
    assert audit["source_generation_recomputed_during_recovery"] is False
    assert audit["predictions_recomputed_during_recovery"] is False
    assert audit["decisions_recomputed_during_recovery"] is True
    assert audit["terminal_evaluation_recomputed_during_recovery"] is True
    assert (
        audit[
            "labels_reopened_only_for_deterministic_reconstruction_during_recovery"
        ]
        is True
    )
    assert audit["policy_promotion_authorized"] is False
    assert audit["original_repository_dirty"] is True
    assert audit["repair_repository_dirty"] is False
    assert (
        validate_recovery_audit_payload(
            audit,
            original_repository_state=ORIGINAL_STATE,
            current_repository_state=REPAIR_STATE,
            **REUSED_HASHES,
        )
        == audit
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("donor_models_refit_during_recovery", False),
        ("source_generation_recomputed_during_recovery", True),
        ("predictions_recomputed_during_recovery", True),
        ("decisions_recomputed_during_recovery", False),
        ("terminal_evaluation_recomputed_during_recovery", False),
        (
            "labels_reopened_only_for_deterministic_reconstruction_during_recovery",
            False,
        ),
        ("policy_promotion_authorized", True),
        ("reused_fold_plan_surface_hash", "9" * 64),
        ("failed_error", "TypeError: a different failure"),
    ),
)
def test_recovery_validator_rejects_semantic_or_seal_drift(
    field: str, value: object
) -> None:
    audit = dict(
        recovery_audit_payload(
            original_repository_state=ORIGINAL_STATE,
            repair_repository_state=REPAIR_STATE,
            **REUSED_HASHES,
        )
    )

    with pytest.raises(ProtocolError, match="audit drifted"):
        validate_recovery_audit_payload(
            {**audit, field: value},
            original_repository_state=ORIGINAL_STATE,
            current_repository_state=REPAIR_STATE,
            **REUSED_HASHES,
        )


def test_recovery_requires_clean_changed_repair_checkout() -> None:
    with pytest.raises(ProtocolError, match="repair repository state"):
        recovery_audit_payload(
            original_repository_state=ORIGINAL_STATE,
            repair_repository_state={**REPAIR_STATE, "repository_dirty": True},
            **REUSED_HASHES,
        )
    with pytest.raises(ProtocolError, match="revision did not change"):
        recovery_audit_payload(
            original_repository_state=ORIGINAL_STATE,
            repair_repository_state={
                **REPAIR_STATE,
                "repository_revision": ORIGINAL_STATE["repository_revision"],
            },
            **REUSED_HASHES,
        )


def test_recovery_reused_seals_must_all_be_sha256() -> None:
    for role in REUSED_HASHES:
        invalid = deepcopy(REUSED_HASHES)
        invalid[role] = "a" * 16
        with pytest.raises(ProtocolError, match="reused-seal hash"):
            recovery_audit_payload(
                original_repository_state=ORIGINAL_STATE,
                repair_repository_state=REPAIR_STATE,
                **invalid,
            )


def test_validator_requires_mode_to_match_original_and_current_git_states() -> None:
    fresh = fresh_recovery_audit_payload()
    recovery_audit = recovery_audit_payload(
        original_repository_state=ORIGINAL_STATE,
        repair_repository_state=REPAIR_STATE,
        **REUSED_HASHES,
    )
    with pytest.raises(ProtocolError, match="mode disagrees"):
        validate_recovery_audit_payload(
            fresh,
            original_repository_state=ORIGINAL_STATE,
            current_repository_state=REPAIR_STATE,
            **REUSED_HASHES,
        )
    with pytest.raises(ProtocolError, match="mode disagrees"):
        validate_recovery_audit_payload(
            recovery_audit,
            original_repository_state=ORIGINAL_STATE,
            current_repository_state=ORIGINAL_STATE,
            **REUSED_HASHES,
        )


def test_upstream_bindings_use_file_sha256_and_persisted_surface_hashes(
    tmp_path,
) -> None:
    manifests = tmp_path / "manifests"
    provenance = tmp_path / "provenance"
    manifests.mkdir()
    provenance.mkdir()
    (manifests / "frozen_source_stream_lock.json").write_text(
        '{"source":"lock"}\n', encoding="utf-8"
    )
    (manifests / "fixed_bank_a1_prediction_seal.json").write_text(
        '{"prediction":"seal"}\n', encoding="utf-8"
    )
    (manifests / "prelabel_feature_seal.json").write_text(
        '{"feature_surface_hash":"' + "7" * 64 + '"}\n', encoding="utf-8"
    )
    (manifests / "fold_plan_seals.json").write_text(
        '{"fold_plan_surface_hash":"' + "8" * 64 + '"}\n', encoding="utf-8"
    )
    (provenance / "input_artifacts.json").write_text(
        "{"
        f'"repository_revision":"{ORIGINAL_STATE["repository_revision"]}",'
        '"repository_dirty":true,'
        f'"repository_status_hash":"{ORIGINAL_STATE["repository_status_hash"]}"'
        "}\n",
        encoding="utf-8",
    )

    bindings = sealed_recovery_input_hashes(tmp_path)
    assert all(len(value) == 64 for value in bindings.values())
    assert bindings["prelabel_feature_surface_hash"] == "7" * 64
    assert bindings["fold_plan_surface_hash"] == "8" * 64
    assert original_repository_state_from_provenance(tmp_path) == ORIGINAL_STATE


def test_repair_checkout_must_remain_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = recovery_audit_payload(
        original_repository_state=ORIGINAL_STATE,
        repair_repository_state=REPAIR_STATE,
        **REUSED_HASHES,
    )
    monkeypatch.setattr(
        recovery_provenance,
        "current_repair_repository_state",
        lambda: dict(REPAIR_STATE),
    )
    assert_repair_repository_state_unchanged(audit)

    monkeypatch.setattr(
        recovery_provenance,
        "current_repair_repository_state",
        lambda: {**REPAIR_STATE, "repository_dirty": True},
    )
    with pytest.raises(ProtocolError, match="changed during continuation"):
        assert_repair_repository_state_unchanged(audit)


def test_validator_rebinds_runtime_audit_and_rejects_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = recovery_audit_payload(
        original_repository_state=ORIGINAL_STATE,
        repair_repository_state=REPAIR_STATE,
        **REUSED_HASHES,
    )
    monkeypatch.setattr(
        validation,
        "original_repository_state_from_provenance",
        lambda _root: dict(ORIGINAL_STATE),
    )
    monkeypatch.setattr(
        validation,
        "current_repair_repository_state",
        lambda: dict(REPAIR_STATE),
    )
    monkeypatch.setattr(
        validation,
        "sealed_recovery_input_hashes",
        lambda _root: dict(REUSED_HASHES),
    )

    assert validation._validate_recovery_lineage(  # noqa: SLF001
        Path("/unused"),
        runtime={"mappingproxy_recovery": audit},
    ) == audit
    with pytest.raises(ProtocolError, match="audit drifted"):
        validation._validate_recovery_lineage(  # noqa: SLF001
            Path("/unused"),
            runtime={
                "mappingproxy_recovery": {
                    **audit,
                    "predictions_recomputed_during_recovery": True,
                }
            },
        )

    monkeypatch.setattr(
        validation,
        "sealed_recovery_input_hashes",
        lambda _root: {
            **REUSED_HASHES,
            "global_prediction_seal_hash": "9" * 64,
        },
    )
    with pytest.raises(ProtocolError, match="audit drifted"):
        validation._validate_recovery_lineage(  # noqa: SLF001
            Path("/unused"),
            runtime={"mappingproxy_recovery": audit},
        )

    monkeypatch.setattr(
        validation,
        "sealed_recovery_input_hashes",
        lambda _root: dict(REUSED_HASHES),
    )
    monkeypatch.setattr(
        validation,
        "current_repair_repository_state",
        lambda: {
            **REPAIR_STATE,
            "repository_revision": "a" * 40,
        },
    )
    with pytest.raises(ProtocolError, match="audit drifted"):
        validation._validate_recovery_lineage(  # noqa: SLF001
            Path("/unused"),
            runtime={"mappingproxy_recovery": audit},
        )


@pytest.mark.parametrize(
    "failed_state",
    (
        {
            **recovery.FAILED_MAPPINGPROXY_STATE,
            "phase": "PREDICTION_MATERIALIZATION",
        },
        {
            **recovery.FAILED_MAPPINGPROXY_STATE,
            "error": "some other error",
        },
    ),
)
def test_direct_runner_rejects_unregistered_failed_partial_root(
    tmp_path: Path,
    failed_state: dict[str, object],
) -> None:
    root = tmp_path / "bundle"
    state_path = root / "reports/run_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(failed_state) + "\n", encoding="utf-8"
    )

    with pytest.raises(ProtocolError, match="unregistered FAILED"):
        runner._launch_recovery_audit(root)  # noqa: SLF001


def test_public_runner_rejects_unregistered_failed_root_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "bundle"
    state_path = root / "reports/run_state.json"
    state_path.parent.mkdir(parents=True)
    failed_state = {
        **recovery.FAILED_MAPPINGPROXY_STATE,
        "phase": "PREDICTION_MATERIALIZATION",
    }
    before = json.dumps(failed_state) + "\n"
    state_path.write_text(before, encoding="utf-8")
    config = SimpleNamespace(source_path=tmp_path / "config.yaml")
    monkeypatch.setattr(runner, "assert_launch_files", lambda *args: None)
    monkeypatch.setattr(
        runner, "assert_workspace_resolved_paths", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        runner,
        "recover_if_possible",
        lambda *args, **kwargs: pytest.fail("recovery ran before FAILED-state rejection"),
    )
    monkeypatch.setattr(
        runner,
        "write_state",
        lambda *args, **kwargs: pytest.fail("FAILED state was mutated"),
    )

    with pytest.raises(ProtocolError, match="unregistered FAILED"):
        runner.run_fixed_bank_multi_challenger_hierarchical_flip_router(
            config,
            artifact_root=root,
        )
    assert state_path.read_text(encoding="utf-8") == before
