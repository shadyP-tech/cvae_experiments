from __future__ import annotations

import ast
import inspect
import sys

import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.disagreement_regret_core import provenance, runtime
from midogpp_thesis.cvae.routing.disagreement_regret_core.provenance import (
    DevelopmentContext,
    DevelopmentScope,
    assert_development_context,
)
from midogpp_thesis.cvae.routing.disagreement_regret_core.runtime import (
    WorkstationRuntime,
    assert_dense_fit_within_budget,
    canonical_workstation_runtime,
    estimate_dense_fit_bytes,
    validate_runtime,
)


def _authorization_hash() -> str:
    return "0123456789abcdef" * 4


def test_synthetic_context_is_in_memory_and_has_no_authorization() -> None:
    context = DevelopmentContext(
        scope=DevelopmentScope.SYNTHETIC_TEST,
        dataset_family="SYNTHETIC",
        outer_target_id="outer-0",
    )

    assert assert_development_context(context) is context
    assert context.to_payload() == {
        "schema_version": "midogpp_disagreement_regret_development_context_v1",
        "scope": "SYNTHETIC_TEST",
        "dataset_family": "SYNTHETIC",
        "outer_target_id": "outer-0",
        "authorization_hash": None,
        "authorization_unused": None,
        "authorized_query_ids": [],
        "authorized_sample_keys_hash": None,
        "source_evidence_previously_consumed": False,
        "consumed_data": False,
        "target_labels_available": False,
        "in_memory_only": True,
    }


def test_posthoc_source_oof_context_is_truthful_and_nonfresh() -> None:
    context = DevelopmentContext(
        scope=DevelopmentScope.AUTHORIZED_POSTHOC_SOURCE_OOF,
        dataset_family="MIDOGPP_SOURCE_TRAIN_OOF_POSTHOC",
        outer_target_id="H",
        authorization_hash="a" * 64,
        authorization_unused=False,
        authorized_query_ids=("q0", "q1", "q2"),
        authorized_sample_keys_hash="b" * 64,
        source_evidence_previously_consumed=True,
    )
    assert context.to_payload()["source_evidence_previously_consumed"] is True
    assert context.to_payload()["authorization_unused"] is False
    with pytest.raises(ProtocolError, match="Posthoc source OOF"):
        DevelopmentContext(
            scope=DevelopmentScope.AUTHORIZED_POSTHOC_SOURCE_OOF,
            dataset_family="MIDOGPP_SOURCE_TRAIN_OOF_POSTHOC",
            outer_target_id="H",
            authorization_hash="a" * 64,
            authorization_unused=True,
            authorized_query_ids=("q0", "q1", "q2"),
            authorized_sample_keys_hash="b" * 64,
            source_evidence_previously_consumed=True,
        )


def test_authorized_source_oof_requires_predeclared_unused_authority() -> None:
    context = DevelopmentContext(
        scope="AUTHORIZED_SOURCE_OOF",
        dataset_family="MIDOG++",
        outer_target_id="center-0",
        authorization_hash=_authorization_hash(),
        authorization_unused=True,
        authorized_query_ids=("q0", "q1", "q2"),
        authorized_sample_keys_hash="b" * 64,
    )

    assert context.scope is DevelopmentScope.AUTHORIZED_SOURCE_OOF
    assert assert_development_context(context) is context
    assert context.to_payload()["authorized_query_ids"] == ["q0", "q1", "q2"]


def test_authorized_source_oof_requires_query_and_sample_allowlists() -> None:
    with pytest.raises(ProtocolError, match="donor-query allowlist"):
        DevelopmentContext(
            scope=DevelopmentScope.AUTHORIZED_SOURCE_OOF,
            dataset_family="MIDOG++",
            outer_target_id="center-0",
            authorization_hash=_authorization_hash(),
            authorization_unused=True,
            authorized_sample_keys_hash="b" * 64,
        )
    with pytest.raises(ProtocolError, match="sample-key allowlist"):
        DevelopmentContext(
            scope=DevelopmentScope.AUTHORIZED_SOURCE_OOF,
            dataset_family="MIDOG++",
            outer_target_id="center-0",
            authorization_hash=_authorization_hash(),
            authorization_unused=True,
            authorized_query_ids=("q0", "q1", "q2"),
        )


@pytest.mark.parametrize(
    "dataset_family",
    ("CONSUMED_TEST", "TARGET", "STAGE70", "STAGE-90", "VALIDATION"),
)
def test_authorized_source_oof_rejects_consumed_or_stage_families(
    dataset_family: str,
) -> None:
    with pytest.raises(ProtocolError, match="dataset family cannot name"):
        DevelopmentContext(
            scope=DevelopmentScope.AUTHORIZED_SOURCE_OOF,
            dataset_family=dataset_family,
            outer_target_id="center-0",
            authorization_hash=_authorization_hash(),
            authorization_unused=True,
            authorized_query_ids=("q0", "q1", "q2"),
            authorized_sample_keys_hash="b" * 64,
        )


@pytest.mark.parametrize(
    "scope",
    (
        "CONSUMED_TEST",
        "TEST",
        "TARGET",
        "TARGET_EVALUATION",
        "STAGE70",
        "STAGE90",
        "stage_70",
        "stage_90",
    ),
)
def test_consumed_target_and_stage_scopes_are_rejected(scope: str) -> None:
    with pytest.raises(ProtocolError, match="forbidden"):
        DevelopmentContext(
            scope=scope,
            dataset_family="MIDOG++",
            outer_target_id="center-0",
        )


@pytest.mark.parametrize(
    "authorization_hash",
    (
        None,
        "a" * 63,
        "a" * 65,
        "A" * 64,
        "g" * 64,
        "sha256:" + "a" * 64,
    ),
)
def test_authorized_source_oof_rejects_noncanonical_hashes(
    authorization_hash: str | None,
) -> None:
    with pytest.raises(ProtocolError, match="lowercase 64-hex"):
        DevelopmentContext(
            scope=DevelopmentScope.AUTHORIZED_SOURCE_OOF,
            dataset_family="MIDOG++",
            outer_target_id="center-0",
            authorization_hash=authorization_hash,
            authorization_unused=True,
        )


@pytest.mark.parametrize("authorization_unused", (None, False, 1))
def test_authorized_source_oof_requires_explicit_unused_status(
    authorization_unused: object,
) -> None:
    with pytest.raises(ProtocolError, match="explicit unused"):
        DevelopmentContext(
            scope=DevelopmentScope.AUTHORIZED_SOURCE_OOF,
            dataset_family="MIDOG++",
            outer_target_id="center-0",
            authorization_hash=_authorization_hash(),
            authorization_unused=authorization_unused,
        )


def test_synthetic_context_cannot_smuggle_authority_or_restricted_data() -> None:
    with pytest.raises(ProtocolError, match="authorization metadata"):
        DevelopmentContext(
            scope=DevelopmentScope.SYNTHETIC_TEST,
            dataset_family="SYNTHETIC",
            outer_target_id="outer-0",
            authorization_hash=_authorization_hash(),
            authorization_unused=True,
        )
    with pytest.raises(ProtocolError, match="Consumed data"):
        DevelopmentContext(
            scope=DevelopmentScope.SYNTHETIC_TEST,
            dataset_family="SYNTHETIC",
            outer_target_id="outer-0",
            consumed_data=True,
        )
    with pytest.raises(ProtocolError, match="Target labels"):
        DevelopmentContext(
            scope=DevelopmentScope.SYNTHETIC_TEST,
            dataset_family="SYNTHETIC",
            outer_target_id="outer-0",
            target_labels_available=True,
        )


@pytest.mark.parametrize(
    "dataset_family",
    ("MIDOG++", "STAGE70", "STAGE90", "CONSUMED_TEST", "synthetic"),
)
def test_synthetic_scope_rejects_real_or_stage_dataset_families(
    dataset_family: str,
) -> None:
    with pytest.raises(ProtocolError, match="exact SYNTHETIC dataset family"):
        DevelopmentContext(
            scope=DevelopmentScope.SYNTHETIC_TEST,
            dataset_family=dataset_family,
            outer_target_id="outer-0",
        )


def test_development_context_requires_canonical_nonempty_identities() -> None:
    with pytest.raises(ProtocolError, match="dataset_family"):
        DevelopmentContext(
            scope=DevelopmentScope.SYNTHETIC_TEST,
            dataset_family="",
            outer_target_id="outer-0",
        )
    with pytest.raises(ProtocolError, match="outer_target_id"):
        DevelopmentContext(
            scope=DevelopmentScope.SYNTHETIC_TEST,
            dataset_family="SYNTHETIC",
            outer_target_id=" target ",
        )


def test_canonical_runtime_freezes_cpu_workstation_maxima() -> None:
    profile = canonical_workstation_runtime()

    assert profile == WorkstationRuntime()
    assert validate_runtime(profile) is profile
    assert profile.total_threads == 12
    assert profile.to_payload()["gpu_surfaces"] == "upstream_out_of_scope"
    assert profile.to_payload()["starts_workers"] is False


def test_serial_test_override_is_exactly_one_by_one() -> None:
    serial = canonical_workstation_runtime(serial_test_override=True)

    assert serial.workers == 1
    assert serial.threads_per_worker == 1
    assert serial.total_threads == 1
    assert serial.serial_test_override is True
    with pytest.raises(ProtocolError, match="exact 1 x 1"):
        WorkstationRuntime(
            workers=2,
            threads_per_worker=1,
            serial_test_override=True,
        )


@pytest.mark.parametrize(
    "overrides",
    (
        {"workers": 5},
        {"workers": 0},
        {"threads_per_worker": 4},
        {"threads_per_worker": 0},
        {"workers": True},
        {"threads_per_worker": True},
    ),
)
def test_runtime_rejects_oversubscription_and_non_integer_limits(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ProtocolError, match="frozen 4-worker"):
        WorkstationRuntime(**overrides)


def test_runtime_rejects_gpu_and_nondeterministic_profiles() -> None:
    with pytest.raises(ProtocolError, match="CPU-only"):
        WorkstationRuntime(device="cuda:0")
    with pytest.raises(ProtocolError, match="deterministic"):
        WorkstationRuntime(deterministic=False)
    with pytest.raises(ProtocolError, match="explicit boolean"):
        canonical_workstation_runtime(serial_test_override=1)


def test_dense_fit_memory_budget_is_explicit_and_fail_closed() -> None:
    estimate = estimate_dense_fit_bytes(pair_count=10_000, design_dimension=160)
    assert 0 < estimate < 512 * 1024 * 1024
    assert assert_dense_fit_within_budget(
        pair_count=10_000,
        design_dimension=160,
    ) == estimate
    with pytest.raises(ProtocolError, match="512 MiB"):
        assert_dense_fit_within_budget(
            pair_count=1_000_000,
            design_dimension=1_000,
        )


def test_core_modules_are_pure_non_runnable_and_have_locked_imports() -> None:
    allowed_external_module = "midogpp_thesis.cvae.protocol"
    for module in (provenance, runtime):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(
                    alias.name.partition(".")[0] in sys.stdlib_module_names
                    for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom):
                imported = node.module or ""
                assert (
                    imported == "__future__"
                    or imported in sys.stdlib_module_names
                    or imported == allowed_external_module
                )
        assert not hasattr(module, "main")
        assert not any(
            isinstance(node, (ast.With, ast.AsyncWith)) for node in ast.walk(tree)
        )
