from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
from pathlib import Path
import pickle
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.config import (
    ResolvedConfigBundle,
    build_authorization_ready_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.identity import (
    DIRECT_INPUT_ARTIFACT_IDS,
    DIRECT_INPUT_ROLES,
    EXPECTED_BANK_CONTENT_INDEX_SHA256,
    EXPECTED_GENERATION_CONTENT_INDEX_SHA256,
    EXPECTED_INPUT_KINDS,
    EXPECTED_TERMINAL_CASE_INVENTORY_SHA256,
    INPUT_RELATIVE_MEMBERS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.phase_contracts import (
    AggregateOnlyTerminalReceipt,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.execution.decision_receipts import (
    TypedPreterminalDecisionLedgerReceipt,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.fresh_process_validation import (
    ArtifactFreshProcessAttestationReceipt,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.execution_admission import (
    _issue_six_input_admission_receipt,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.service_factory import (
    CanonicalExecutionServiceFactory,
    build_canonical_execution_services,
    prepare_canonical_execution_service_factory,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2 import (
    terminal_capability as terminal_api,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.source_seal import (
    build_source_contract_receipt,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.terminal_capability import (
    AggregateOnlyTerminalScorer,
    AggregateTerminalScoreRequest,
    GuardedPreterminalBoundary,
    TerminalAggregateCapability,
    issue_terminal_aggregate_capability,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.workspace_inputs import (
    WorkspaceInputBinding,
    hash_ordered_input_locations,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _resolved_bundle(tmp_path: Path):
    source = build_source_contract_receipt()
    config = build_authorization_ready_config(
        source_contract_hash=source.combined_source_sha256,
        expected_authorization_amendment_sha256=_sha("amendment"),
    )
    inputs_root = tmp_path / "immutable-inputs"
    bindings = tuple(
        WorkspaceInputBinding(
            role,
            artifact_id,
            inputs_root / f"input-{index}" / member if member else inputs_root / f"input-{index}",
            kind,
        )
        for index, (role, artifact_id, member, kind) in enumerate(
            zip(
                DIRECT_INPUT_ROLES,
                DIRECT_INPUT_ARTIFACT_IDS,
                INPUT_RELATIVE_MEMBERS,
                EXPECTED_INPUT_KINDS,
                strict=True,
            )
        )
    )
    bundle = ResolvedConfigBundle(
        config=config,
        source_path=tmp_path / "artifact" / "config.resolved.yaml",
        artifact_root=tmp_path / "artifact",
        input_bindings=bindings,
    )
    validated = SimpleNamespace(
        input_binding_hash=_sha("input-binding"),
        input_location_binding_sha256=hash_ordered_input_locations(bindings),
        bank_content_index_sha256=EXPECTED_BANK_CONTENT_INDEX_SHA256,
        generation_content_index_sha256=(
            EXPECTED_GENERATION_CONTENT_INDEX_SHA256
        ),
        cache_content_sha256=(
            "df0bdbf64881ee000fe7c56bc486724313accf373ef8e90896344f8d03d187db"
        ),
        cache_row_order_sha256=(
            "bd1a85b95496203500bfe2dc5232f8bfb383e73d222a8ba083e81b2c6b33c389"
        ),
        manifest_sha256=(
            "db661ac7e3dbafde8e283528de6706ab35f2c26629b389706c4504e458cc5869"
        ),
        parent_ledger_sha256=(
            "8b16eae7bfdb5d20945e8ba3e02447ec74ab857adae441e493b2e37114feab16"
        ),
        artifact_root=bundle.artifact_root.as_posix(),
        scratch_root=(tmp_path / "scratch").as_posix(),
    )
    admission = _issue_six_input_admission_receipt(
        config=config,
        validated=validated,
        protocol_hash=str(config.protocol["protocol_hash"]),
        source_hash=source.combined_source_sha256,
        amendment_sha256=_sha("amendment"),
    )
    return source, bundle, admission


def test_canonical_factory_binds_actual_source_and_immutable_exact_six(
    tmp_path: Path,
) -> None:
    source, bundle, admission = _resolved_bundle(tmp_path)
    factory = prepare_canonical_execution_service_factory(
        bundle, admission=admission, source=source
    )

    assert type(factory) is CanonicalExecutionServiceFactory
    assert tuple(row.role for row in factory.label_free_input_bindings) == (
        DIRECT_INPUT_ROLES[:3]
    )
    assert tuple(row.artifact_id for row in factory.label_free_input_bindings) == (
        DIRECT_INPUT_ARTIFACT_IDS[:3]
    )
    assert tuple(row.kind for row in factory.label_free_input_bindings) == (
        EXPECTED_INPUT_KINDS[:3]
    )
    assert not hasattr(factory, "source_path")
    assert not hasattr(factory, "input_bindings")
    assert all(
        row.role not in DIRECT_INPUT_ROLES[3:]
        for row in factory.label_free_input_bindings
    )
    assert factory.identity.factory_module.endswith(".service_factory")
    assert factory.identity.source_relative_path.endswith("/service_factory.py")
    assert factory.identity.source_contract_hash == source.combined_source_sha256
    assert factory.identity.resolved_config_contract_hash == bundle.contract_hash
    assert (
        factory.identity.admitted_input_location_binding_sha256
        == admission.input_location_binding_sha256
        == factory.identity.resolved_input_location_binding_sha256
    )
    assert factory.identity.to_payload()["structural_service_injection_allowed"] is False
    with pytest.raises(FrozenInstanceError):
        factory.label_free_input_bindings[0].role = "forged"  # type: ignore[misc]


def test_canonical_factory_rejects_structural_bundle_and_direct_construction(
    tmp_path: Path,
) -> None:
    source, bundle, admission = _resolved_bundle(tmp_path)
    fake = SimpleNamespace(**bundle.__dict__) if hasattr(bundle, "__dict__") else SimpleNamespace(
        config=bundle.config,
        source_path=bundle.source_path,
        artifact_root=bundle.artifact_root,
        input_bindings=bundle.input_bindings,
    )
    with pytest.raises(ProtocolError, match="ResolvedConfigBundle"):
        prepare_canonical_execution_service_factory(
            fake,  # type: ignore[arg-type]
            admission=admission,
            source=source,
        )
    with pytest.raises(ProtocolError, match="bypassed resolved admission"):
        CanonicalExecutionServiceFactory(
            artifact_root=bundle.artifact_root,
            bindings=(),
            identity=object(),  # type: ignore[arg-type]
        )


def test_canonical_factory_rejects_admission_from_different_six_path_snapshot(
    tmp_path: Path,
) -> None:
    source, bundle_a, admission_a = _resolved_bundle(tmp_path / "snapshot-a")
    _, other_paths, _ = _resolved_bundle(tmp_path / "snapshot-b")
    bundle_b = ResolvedConfigBundle(
        config=bundle_a.config,
        source_path=bundle_a.source_path,
        artifact_root=bundle_a.artifact_root,
        input_bindings=other_paths.input_bindings,
    )

    assert bundle_a.input_bindings != bundle_b.input_bindings
    with pytest.raises(ProtocolError, match="input-location admission drifted"):
        prepare_canonical_execution_service_factory(
            bundle_b,
            admission=admission_a,
            source=source,
        )


def test_canonical_factory_fails_before_lease_without_mutating_paths(
    tmp_path: Path,
) -> None:
    source, bundle, admission = _resolved_bundle(tmp_path)
    all_paths = (
        bundle.source_path,
        bundle.artifact_root,
        *(row.path for row in bundle.input_bindings),
        tmp_path / ".oe_ppur_v2_single_use_authorization_consumed",
    )
    assert all(not path.exists() for path in all_paths)

    with pytest.raises(ProtocolError, match="before authorization lease"):
        build_canonical_execution_services(
            bundle, admission=admission, source=source
        )

    assert all(not path.exists() for path in all_paths)


def _validated_boundary_stubs(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    TypedPreterminalDecisionLedgerReceipt, ArtifactFreshProcessAttestationReceipt
]:
    """Isolate capability state tests from the upstream guarded factories.

    The genuine typed-ledger and spawned-attestation factories have their own
    focused tests.  These nominal shells are accepted only by monkeypatched
    validator seams in this unit; production issuance still invokes both real
    validators and exact concrete types.
    """

    ledger = object.__new__(TypedPreterminalDecisionLedgerReceipt)
    for name, value in {
        "receipt_hash": _sha("typed-ledger"),
        "six_input_admission_hash": _sha("admission"),
        "input_binding_hash": _sha("input-binding"),
        "parsed_probability_matrix_receipt_hash": _sha("matrix"),
        "matrix_content_sha256": _sha("matrix-content"),
        "row_binding_hash": _sha("row-binding"),
        "outer_fold_receipt_hash": _sha("outer"),
        "decision_source_hash": _sha("decision-source"),
        "case_inventory_sha256": EXPECTED_TERMINAL_CASE_INVENTORY_SHA256,
        "opportunity_surface_hash": _sha("opportunity-surface"),
        "outer_lineage_surface_hash": _sha("outer-lineage-surface"),
        "case_count": 218,
        "exact_p_fallback_count": 17,
    }.items():
        object.__setattr__(ledger, name, value)

    attestation = object.__new__(ArtifactFreshProcessAttestationReceipt)
    for name, value in {
        "phase": "preterminal",
        "sealed_receipt_hash": ledger.receipt_hash,
        "sealed_file_sha256": _sha("ledger-file"),
        "sealed_file_identity_sha256": _sha("ledger-file-identity"),
        "receipt_hash": _sha("fresh-attestation"),
    }.items():
        object.__setattr__(attestation, name, value)

    def validate_ledger(value):
        assert value is ledger
        return ledger

    def validate_attestation(
        value,
        *,
        expected_phase=None,
        expected_sealed_receipt_hash=None,
        expected_file_sha256=None,
    ):
        if (
            value is not attestation
            or value.phase != expected_phase
            or value.sealed_receipt_hash != expected_sealed_receipt_hash
        ):
            raise ProtocolError("guarded attestation mismatch")
        return attestation

    monkeypatch.setattr(
        terminal_api,
        "validate_typed_preterminal_decision_ledger",
        validate_ledger,
    )
    monkeypatch.setattr(
        terminal_api,
        "validate_artifact_fresh_process_attestation",
        validate_attestation,
    )
    return ledger, attestation


class _AggregateScorer(AggregateOnlyTerminalScorer):
    def __init__(self) -> None:
        self.requests: list[AggregateTerminalScoreRequest] = []

    def score_aggregates(
        self, request: AggregateTerminalScoreRequest
    ) -> AggregateOnlyTerminalReceipt:
        self.requests.append(request)
        return AggregateOnlyTerminalReceipt(
            preterminal_attestation_hash=(
                request.preterminal_attestation_receipt_hash
            ),
            preterminal_ledger_receipt_hash=(
                request.preterminal_ledger_receipt_hash
            ),
            metric_names=("bacc", "brier", "log"),
            protected_metrics=(0.70, 0.20, 0.40),
            routed_metrics=(0.71, 0.19, 0.39),
            evaluated_case_count=request.case_count,
            routed_case_count=18,
            center_aggregate_hash=_sha("center-aggregates"),
        )


def test_terminal_capability_requires_attestation_and_exact_ledger_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, attestation = _validated_boundary_stubs(monkeypatch)
    scorer = _AggregateScorer()
    with pytest.raises(ProtocolError, match="requires preterminal attestation"):
        issue_terminal_aggregate_capability(ledger, None, scorer=scorer)

    wrong = object.__new__(ArtifactFreshProcessAttestationReceipt)
    object.__setattr__(wrong, "phase", "preterminal")
    object.__setattr__(wrong, "sealed_receipt_hash", _sha("another-ledger"))
    with pytest.raises(ProtocolError, match="matching preterminal"):
        issue_terminal_aggregate_capability(ledger, wrong, scorer=scorer)

    final = object.__new__(ArtifactFreshProcessAttestationReceipt)
    object.__setattr__(final, "phase", "final")
    object.__setattr__(final, "sealed_receipt_hash", ledger.receipt_hash)
    with pytest.raises(ProtocolError, match="matching preterminal"):
        issue_terminal_aggregate_capability(ledger, final, scorer=scorer)


def test_terminal_capability_is_aggregate_only_ephemeral_and_single_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, attestation = _validated_boundary_stubs(monkeypatch)
    scorer = _AggregateScorer()
    capability = issue_terminal_aggregate_capability(
        ledger, attestation, scorer=scorer
    )
    assert isinstance(capability, TerminalAggregateCapability)
    assert capability.consumed is False
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(capability)

    result = capability.score_aggregates()
    assert result.preterminal_attestation_hash == attestation.receipt_hash
    assert result.preterminal_ledger_receipt_hash == ledger.receipt_hash
    assert result.evaluated_case_count == 218
    assert capability.consumed is True
    assert len(scorer.requests) == 1
    request = scorer.requests[0]
    assert request.case_count == 218
    assert not any(
        fragment in name.casefold()
        for name in request.__slots__
        for fragment in ("path", "label", "manifest")
    )
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(request)
    with pytest.raises(ProtocolError, match="replayed"):
        capability.score_aggregates()
    assert len(scorer.requests) == 1


class _FailingScorer(AggregateOnlyTerminalScorer):
    def score_aggregates(
        self, request: AggregateTerminalScoreRequest
    ) -> AggregateOnlyTerminalReceipt:
        raise RuntimeError("terminal scorer failed")


class _WrongLedgerScorer(_AggregateScorer):
    def score_aggregates(
        self, request: AggregateTerminalScoreRequest
    ) -> AggregateOnlyTerminalReceipt:
        return AggregateOnlyTerminalReceipt(
            preterminal_attestation_hash=(
                request.preterminal_attestation_receipt_hash
            ),
            preterminal_ledger_receipt_hash=_sha("wrong-ledger"),
            metric_names=("bacc", "brier", "log"),
            protected_metrics=(0.70, 0.20, 0.40),
            routed_metrics=(0.71, 0.19, 0.39),
            evaluated_case_count=request.case_count,
            routed_case_count=18,
            center_aggregate_hash=_sha("wrong-center-aggregates"),
        )


def test_terminal_capability_is_spent_before_scorer_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, attestation = _validated_boundary_stubs(monkeypatch)
    capability = issue_terminal_aggregate_capability(
        ledger, attestation, scorer=_FailingScorer()
    )
    with pytest.raises(RuntimeError, match="scorer failed"):
        capability.score_aggregates()
    assert capability.consumed is True
    with pytest.raises(ProtocolError, match="replayed"):
        capability.score_aggregates()


def test_terminal_capability_rejects_wrong_ledger_and_partial_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, attestation = _validated_boundary_stubs(monkeypatch)
    capability = issue_terminal_aggregate_capability(
        ledger, attestation, scorer=_WrongLedgerScorer()
    )
    with pytest.raises(ProtocolError, match="coverage or lineage"):
        capability.score_aggregates()

    with pytest.raises(ProtocolError, match="terminal aggregate receipt drifted"):
        AggregateOnlyTerminalReceipt(
            preterminal_attestation_hash=attestation.receipt_hash,
            preterminal_ledger_receipt_hash=ledger.receipt_hash,
            metric_names=("bacc", "brier", "log"),
            protected_metrics=(0.70, 0.20, 0.40),
            routed_metrics=(0.71, 0.19, 0.39),
            evaluated_case_count=217,
            routed_case_count=18,
            center_aggregate_hash=_sha("partial-center-aggregates"),
        )


def test_terminal_contract_constructors_reject_caller_forgery() -> None:
    with pytest.raises(ProtocolError, match="guarded attestation"):
        GuardedPreterminalBoundary(
            preterminal_ledger_receipt_hash=_sha("ledger"),
            preterminal_attestation_receipt_hash=_sha("attestation"),
            preterminal_ledger_file_sha256=_sha("ledger-file"),
            preterminal_ledger_file_identity_sha256=_sha("ledger-identity"),
            six_input_admission_hash=_sha("admission"),
            input_binding_hash=_sha("binding"),
            parsed_probability_matrix_receipt_hash=_sha("matrix"),
            matrix_content_sha256=_sha("matrix-content"),
            row_binding_hash=_sha("row-binding"),
            outer_fold_receipt_hash=_sha("outer"),
            decision_source_hash=_sha("decision-source"),
            case_inventory_sha256=EXPECTED_TERMINAL_CASE_INVENTORY_SHA256,
            opportunity_surface_hash=_sha("opportunity"),
            outer_lineage_surface_hash=_sha("outer-lineage"),
            case_count=218,
            exact_p_fallback_count=0,
        )
    with pytest.raises(ProtocolError, match="live capability"):
        AggregateTerminalScoreRequest(
            preterminal_ledger_receipt_hash=_sha("ledger"),
            preterminal_attestation_receipt_hash=_sha("attestation"),
            preterminal_ledger_file_sha256=_sha("ledger-file"),
            preterminal_ledger_file_identity_sha256=_sha("ledger-identity"),
            six_input_admission_hash=_sha("admission"),
            input_binding_hash=_sha("binding"),
            parsed_probability_matrix_receipt_hash=_sha("matrix"),
            matrix_content_sha256=_sha("matrix-content"),
            row_binding_hash=_sha("row-binding"),
            outer_fold_receipt_hash=_sha("outer"),
            decision_source_hash=_sha("decision-source"),
            case_inventory_sha256=EXPECTED_TERMINAL_CASE_INVENTORY_SHA256,
            opportunity_surface_hash=_sha("opportunity"),
            outer_lineage_surface_hash=_sha("outer-lineage"),
            case_count=218,
            exact_p_fallback_count=0,
            capability_hash=_sha("capability"),
        )


def test_terminal_capability_rejects_structural_scorer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, attestation = _validated_boundary_stubs(monkeypatch)

    class StructuralFake:
        def score_aggregates(self, request):
            raise AssertionError("structural fake was invoked")

    with pytest.raises(ProtocolError, match="nominal interface"):
        issue_terminal_aggregate_capability(
            ledger,
            attestation,
            scorer=StructuralFake(),  # type: ignore[arg-type]
        )
