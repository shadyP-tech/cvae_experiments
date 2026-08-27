from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2 import runner
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.config import (
    ResolvedConfigBundle,
    build_authorization_ready_config,
    build_planned_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.execution.probability_matrix_receipts import (
    ProbabilityMatrixShardSpec,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.execution_admission import (
    _issue_six_input_admission_receipt,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.identity import (
    CENTERS,
    DIRECT_INPUT_ARTIFACT_IDS,
    DIRECT_INPUT_ROLES,
    EXPECTED_BANK_CONTENT_INDEX_SHA256,
    EXPECTED_GENERATION_CONTENT_INDEX_SHA256,
    EXPECTED_INPUT_KINDS,
    EXPECTED_TERMINAL_CASE_INVENTORY_SHA256,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.phase_contracts import (
    AggregateOnlyTerminalReceipt,
    OuterFoldExecutionReceipt,
    ProbabilityMaterializationReceipt,
    ServicePreflightReceipt,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.row_binding import (
    derive_admitted_row_binding,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.source_seal import (
    build_source_contract_receipt,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.workstation import (
    preflight_workstation,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.workspace_inputs import (
    WorkspaceInputBinding,
    hash_ordered_input_locations,
)
from midogpp_thesis.cvae.protocol import ProtocolError


SHA = "a" * 64
SERVICE_MODULE = (
    "midogpp_thesis.cvae.diagnostics."
    "fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_"
    "router_v2.test_service"
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _PreterminalLedgerStub:
    def __init__(self, admission_hash: str, matrix_hash: str, outer_hash: str) -> None:
        self.six_input_admission_hash = admission_hash
        self.parsed_probability_matrix_receipt_hash = matrix_hash
        self.outer_fold_receipt_hash = outer_hash
        self.receipt_hash = _sha("preterminal-ledger")

    def to_payload(self) -> dict[str, object]:
        body = {
            "schema_version": "oe_ppur_v2_lifecycle_test_ledger_stub_v1",
            "six_input_admission_hash": self.six_input_admission_hash,
            "parsed_probability_matrix_receipt_hash": (
                self.parsed_probability_matrix_receipt_hash
            ),
            "outer_fold_receipt_hash": self.outer_fold_receipt_hash,
            "case_inventory_sha256": EXPECTED_TERMINAL_CASE_INVENTORY_SHA256,
            "case_count": 218,
            "exact_p_fallback_count": 200,
            "labels_opened": False,
            "raw_labels_persisted": False,
        }
        self.receipt_hash = runner.canonical_hash(body)
        return {**body, "receipt_hash": self.receipt_hash}


class _FreshAttestationStub:
    def __init__(self, phase: str, sealed_receipt_hash: str) -> None:
        self.phase = phase
        self.sealed_receipt_hash = sealed_receipt_hash
        self.receipt_hash = _sha(f"{phase}-attestation")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v2_lifecycle_test_attestation_stub_v1",
            "phase": self.phase,
            "sealed_receipt_hash": self.sealed_receipt_hash,
            "receipt_hash": self.receipt_hash,
        }


def _authorized(tmp_path: Path):
    source = build_source_contract_receipt()
    config = build_authorization_ready_config(
        source_contract_hash=source.combined_source_sha256,
        expected_authorization_amendment_sha256=_sha("amendment"),
    )
    artifact = tmp_path / "artifact"
    scratch = tmp_path / "scratch"
    input_bindings = tuple(
        WorkspaceInputBinding(
            role,
            artifact_id,
            tmp_path / "resolved-inputs" / f"input-{index}",
            kind,
        )
        for index, (role, artifact_id, kind) in enumerate(
            zip(
                DIRECT_INPUT_ROLES,
                DIRECT_INPUT_ARTIFACT_IDS,
                EXPECTED_INPUT_KINDS,
                strict=True,
            )
        )
    )
    validated = SimpleNamespace(
        input_binding_hash=_sha("bindings"),
        input_location_binding_sha256=hash_ordered_input_locations(
            input_bindings
        ),
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
        artifact_root=str(artifact),
        scratch_root=str(scratch),
    )
    admission = _issue_six_input_admission_receipt(
        config=config,
        validated=validated,
        protocol_hash=str(config.protocol["protocol_hash"]),
        source_hash=source.combined_source_sha256,
        amendment_sha256=_sha("amendment"),
    )
    resolved = ResolvedConfigBundle(
        config=config,
        source_path=artifact / "config.resolved.yaml",
        artifact_root=artifact,
        input_bindings=input_bindings,
    )
    return source, resolved, admission, artifact, scratch


class _Services:
    def __init__(self, *, fail_after_matrix: bool = False) -> None:
        self.fail_after_matrix = fail_after_matrix
        self.events: list[str] = []

    def preflight(self, admission, source):
        self.events.append("service_preflight")
        return ServicePreflightReceipt(SERVICE_MODULE, _sha("service"), _sha("callbacks"), _sha("spawn"))

    def materialize_probability_matrix(self, context):
        self.events.append("materialize_matrix")
        row_binding = derive_admitted_row_binding(context.admission)
        values = np.linspace(
            0.01,
            0.99,
            9928 * 7,
            dtype=np.dtype("<f4"),
        ).reshape(9928, 7)
        path = context.scratch_root / "probabilities.f32"
        path.write_bytes(values.tobytes(order="C"))
        content = hashlib.sha256(path.read_bytes()).hexdigest()
        spec = ProbabilityMatrixShardSpec(
            path=str(path),
            content_sha256=content,
            six_input_admission_hash=row_binding.six_input_admission_hash,
            row_binding_hash=row_binding.receipt_hash,
            row_index_sha256=row_binding.row_index_sha256,
            row_alignment_receipt_hash=(
                row_binding.row_alignment_receipt_hash
            ),
            gpu_prediction_batch_hash=_sha("gpu-batch"),
            gpu_result_surface_sha256=_sha("gpu-surface"),
            gpu_worker_result_sha256=_sha("gpu-worker"),
            row_start=0,
            row_stop=9928,
            declared_shape=(9928, 7),
        )
        return ProbabilityMaterializationReceipt(
            shards=(spec,),
            row_binding_hash=row_binding.receipt_hash,
            row_index_sha256=row_binding.row_index_sha256,
            row_alignment_receipt_hash=(
                row_binding.row_alignment_receipt_hash
            ),
            gpu_prediction_batch_hash=_sha("gpu-batch"),
            gpu_result_surface_sha256=_sha("gpu-surface"),
            ordered_gpu_worker_result_hashes=(_sha("gpu-worker"),),
            ordered_gpu_result_file_hashes=(content,),
        )

    def run_outer_folds(self, context, matrix):
        self.events.append("run_outer_folds")
        if self.fail_after_matrix:
            raise RuntimeError("post-lease failure")
        return OuterFoldExecutionReceipt(
            matrix.receipt_hash,
            CENTERS,
            tuple(_sha(f"outer-{center}") for center in CENTERS),
            _sha("decision-source"),
        )

    def seal_preterminal_decisions(self, context, matrix, outer):
        self.events.append("seal_preterminal")
        return _PreterminalLedgerStub(
            context.admission.receipt_hash,
            matrix.receipt_hash,
            outer.receipt_hash,
        )

    def build_terminal_scorer(self, context):
        self.events.append("build_terminal_scorer")
        return _TerminalScorer()


class _TerminalScorer:
    def score_aggregates(self, request):
        return AggregateOnlyTerminalReceipt(
            request.preterminal_attestation_hash,
            request.preterminal_ledger_receipt_hash,
            ("bacc", "brier", "log"),
            (0.7, 0.2, 0.4),
            (0.71, 0.19, 0.39),
            218,
            18,
            _sha("center-aggregates"),
        )


class _TerminalCapability:
    def __init__(self, scorer, attestation) -> None:
        self._scorer = scorer
        self._attestation = attestation

    def score_aggregates(self):
        request = SimpleNamespace(
            preterminal_attestation_hash=self._attestation.receipt_hash,
            preterminal_ledger_receipt_hash=_sha("preterminal-ledger"),
        )
        return self._scorer.score_aggregates(request)

def _run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, services: _Services):
    source, resolved, admission, artifact, scratch = _authorized(tmp_path)
    monkeypatch.setattr(runner, "build_source_contract_receipt", lambda: source)
    monkeypatch.setattr(
        runner,
        "admit_six_input_execution",
        lambda *args, **kwargs: admission,
    )
    monkeypatch.setattr(
        runner,
        "build_canonical_execution_services",
        lambda *args, **kwargs: services,
    )
    monkeypatch.setattr(
        runner,
        "preflight_workstation",
        lambda artifact_root, scratch_root: preflight_workstation(
            artifact_root,
            scratch_root,
            observed={
                "gpu_count": 2,
                "gpu_names": ("NVIDIA RTX A5000", "NVIDIA RTX A5000"),
                "cpu_count": 16,
                "scratch_free_bytes": 100 * 1024**3,
            },
        ),
    )
    def fake_attestation(path, *, phase, expected_sealed_receipt_hash, **kwargs):
        return _FreshAttestationStub(phase, expected_sealed_receipt_hash)

    monkeypatch.setattr(
        runner,
        "require_two_fresh_artifact_attestations",
        fake_attestation,
    )
    monkeypatch.setattr(
        runner,
        "validate_artifact_fresh_process_attestation",
        lambda receipt, **kwargs: receipt,
    )
    def validate_ledger(receipt, **kwargs):
        services.events.append("validate_typed_preterminal")
        return receipt

    def issue_capability(preterminal, attestation, *, scorer):
        services.events.append("issue_terminal_capability")
        return _TerminalCapability(scorer, attestation)

    monkeypatch.setattr(
        runner, "validate_typed_preterminal_decision_ledger", validate_ledger
    )
    monkeypatch.setattr(
        runner, "issue_terminal_aggregate_capability", issue_capability
    )
    result = runner.run_oe_ppur_v2(
        resolved,
        scratch_root=scratch,
    )
    return result, artifact, scratch


def test_runner_completes_monotone_matrix_parsed_aggregate_only_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    services = _Services()
    result, artifact, scratch = _run(monkeypatch, tmp_path, services)
    assert result == artifact
    state = runner.read_run_state(artifact)
    assert state["status"] == "COMPLETE"
    assert state["phase"] == "COMPLETE"
    assert state["raw_labels_persisted"] is False
    assert (artifact / "reports/parsed_probability_matrix.json").is_file()
    assert (artifact / "reports/terminal_metrics.json").is_file()
    assert scratch.is_dir()
    lease = tmp_path / ".oe_ppur_v2_single_use_authorization_consumed"
    assert (lease / "claim.json").is_file()
    assert '"status":"COMPLETE"' in (lease / "outcome.json").read_text()
    assert services.events == [
        "service_preflight",
        "materialize_matrix",
        "run_outer_folds",
        "seal_preterminal",
        "validate_typed_preterminal",
        "build_terminal_scorer",
        "issue_terminal_capability",
    ]


def test_postlease_failure_is_permanently_failed_exhausted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="post-lease"):
        _run(monkeypatch, tmp_path, _Services(fail_after_matrix=True))
    artifact = tmp_path / "artifact"
    state = runner.read_run_state(artifact)
    assert state["status"] == "FAILED_EXHAUSTED"
    lease = tmp_path / ".oe_ppur_v2_single_use_authorization_consumed"
    assert '"status":"FAILED_EXHAUSTED"' in (
        lease / "outcome.json"
    ).read_text()


class _PoisonPath:
    def __fspath__(self) -> str:
        raise AssertionError("planned runner touched a run path")


def test_planned_config_rejects_before_source_service_or_path_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "build_source_contract_receipt",
        lambda: (_ for _ in ()).throw(AssertionError("source touched")),
    )
    with pytest.raises(ProtocolError, match="not authorized"):
        runner.run_oe_ppur_v2(
            build_planned_config(),
            scratch_root=_PoisonPath(),
        )


def test_production_runner_does_not_accept_injected_workstation_facts() -> None:
    assert "workstation_observed" not in inspect.signature(
        runner.run_oe_ppur_v2
    ).parameters


def test_workstation_preflight_accepts_only_exact_workspace_launch_envelope(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    scratch = tmp_path / "scratch"
    for relative in ("manifests", "provenance", "reports", "tables"):
        (artifact / relative).mkdir(parents=True, exist_ok=True)
    (artifact / "config.resolved.yaml").write_text("experiment: {}\n", encoding="utf-8")
    (artifact / "provenance/input_artifacts.json").write_text(
        "{}\n", encoding="utf-8"
    )

    receipt = preflight_workstation(
        artifact,
        scratch,
        observed={
            "gpu_count": 2,
            "gpu_names": ("NVIDIA RTX A5000", "NVIDIA RTX A5000"),
            "cpu_count": 16,
            "scratch_free_bytes": 100 * 1024**3,
        },
    )
    assert receipt.artifact_parent == str(tmp_path)
    assert receipt.scratch_parent == str(tmp_path)

    (artifact / "reports/unexpected.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="artifact root already exists"):
        preflight_workstation(
            artifact,
            scratch,
            observed={
                "gpu_count": 2,
                "gpu_names": ("NVIDIA RTX A5000", "NVIDIA RTX A5000"),
                "cpu_count": 16,
                "scratch_free_bytes": 100 * 1024**3,
            },
        )
