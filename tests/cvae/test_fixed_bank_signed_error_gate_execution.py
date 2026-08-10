from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_hierarchical_residual_stacker.contracts import (
    BinaryLabel,
    SampleActionProbability,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_hierarchical_residual_stacker.core_hashing import (
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_hierarchical_residual_stacker.scientific_constants import (  # noqa: E501
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    MIDOGPP_CENTERS,
    candidate_sources,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_signed_error_gate import (
    build_signed_fold_products,
    build_signed_prelabel_products,
    canonical_consumed_test_protocol,
    evaluate_sealed_fold_products,
    fit_all_target_families,
    record_durable_fold_seals,
    record_durable_model_seals,
    terminal as terminal_module,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_signed_error_gate.constants import (
    METHOD_IDS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_signed_error_gate.evaluation import (
    _evaluate_terminal_predictions,
)
from midogpp_thesis.cvae.protocol import ProtocolError


@dataclass(frozen=True)
class _Fold:
    target_center: str
    fold_ordinal: int
    support_case_ids: tuple[str, ...]
    evaluation_case_ids: tuple[str, ...]
    fold_hash: str


@dataclass(frozen=True)
class _Partition:
    folds: tuple[_Fold, ...]
    partition_hash: str


def _surface():
    probabilities: list[SampleActionProbability] = []
    labels: list[BinaryLabel] = []
    folds: list[_Fold] = []
    for center in MIDOGPP_CENTERS:
        cases = tuple(f"case-{center}-{index}" for index in range(5))
        for fold, evaluation in enumerate(cases):
            folds.append(
                _Fold(
                    center,
                    fold,
                    tuple(case for case in cases if case != evaluation),
                    (evaluation,),
                    canonical_hash([center, fold, cases]),
                )
            )
        for case_index, case in enumerate(cases):
            for label in (0, 1):
                sample = f"{case}-sample-{label}"
                baseline = 0.58 if label == 0 else 0.42
                probabilities.append(
                    SampleActionProbability(center, case, sample, "B", baseline)
                )
                for source_index, source in enumerate(candidate_sources(center)):
                    shift = (1 if label else -1) * (
                        0.07 + 0.002 * source_index + 0.001 * case_index
                    )
                    probabilities.append(
                        SampleActionProbability(
                            center,
                            case,
                            sample,
                            source,
                            min(max(baseline + shift, 0.001), 0.999),
                        )
                    )
                labels.append(
                    BinaryLabel(center, case, sample, label, "loco_donor")
                )
    partition_folds = tuple(folds)
    partition = _Partition(
        partition_folds,
        canonical_hash(
            {
                "schema_version": "synthetic_signed_error_partition_v1",
                "fold_hashes": [fold.fold_hash for fold in partition_folds],
            }
        ),
    )
    return tuple(probabilities), tuple(labels), partition


class _Capability:
    def __init__(self, labels):
        self.labels = tuple(labels)
        self.model_seals = {}
        self.support_opened = set()
        self.method_seals = {}
        self.decision_seal = None
        self.permutation_seal = None
        self.evaluation_opened = False

    def open_loco_donor_labels(self, target):
        return tuple(row for row in self.labels if row.target_center != target)

    def record_loco_model_seals(self, target, global_hash, residual_hash, permutation_hash):
        if target in self.model_seals:
            raise ProtocolError("duplicate model seal")
        self.model_seals[target] = (global_hash, residual_hash, permutation_hash)

    def open_fold_support_labels(self, target, fold):
        if set(self.model_seals) != set(MIDOGPP_CENTERS):
            raise ProtocolError("support opened before model seals")
        case = f"case-{target}-{fold}"
        self.support_opened.add((target, fold))
        return tuple(
            replace(row, label_scope="target_support")
            for row in self.labels
            if row.target_center == target and row.case_id != case
        )

    def record_fold_method_decision(self, target, fold, method, decision_hash):
        key = (target, fold, method)
        if (target, fold) not in self.support_opened or key in self.method_seals:
            raise ProtocolError("invalid method seal")
        self.method_seals[key] = decision_hash

    def record_preevaluation_seals(
        self, decision_seal_hash, permutation_provenance_hash, *, decision_count
    ):
        if decision_count != 45 * len(METHOD_IDS):
            raise ProtocolError("invalid decision count")
        self.decision_seal = decision_seal_hash
        self.permutation_seal = permutation_provenance_hash

    def open_oof_evaluation_labels(self):
        if self.decision_seal is None or self.permutation_seal is None:
            raise ProtocolError("evaluation opened before durable seals")
        self.evaluation_opened = True
        return tuple(replace(row, label_scope="terminal_evaluation") for row in self.labels)

    def access_report(self):
        unhashed = {
            "schema_version": "midogpp_signed_error_label_capability_report_v1",
            "status": "PASS" if self.evaluation_opened else "INCOMPLETE",
            "diagnostic_method_ids": list(METHOD_IDS),
            "fold_method_decision_count": len(self.method_seals),
            "all_decisions_seal_hash": self.decision_seal,
            "permutation_provenance_hash": self.permutation_seal,
            "evaluation_labels_opened": self.evaluation_opened,
            "R_raw_and_R_safe_separately_sealed": True,
            "terminal_consumed_test_diagnostic_only": True,
        }
        return {**unhashed, "report_hash": canonical_hash(unhashed)}


def test_full_scientific_lifecycle_requires_durable_seals_before_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probabilities, labels, partition = _surface()
    protocol = canonical_consumed_test_protocol()
    capability = _Capability(labels)
    prelabel = build_signed_prelabel_products(probabilities, protocol=protocol)
    assert len(prelabel.context_hashes) == 162
    models = fit_all_target_families(
        probabilities=probabilities,
        prelabel=prelabel,
        label_manager=capability,
        protocol=protocol,
        worker_count=1,
        threads_per_worker=1,
    )
    with pytest.raises(ProtocolError, match="before model seals"):
        capability.open_fold_support_labels("0", 0)
    record_durable_model_seals(capability, models)
    with pytest.raises(ProtocolError, match="partition_hash"):
        build_signed_fold_products(
            probabilities=probabilities,
            model_products=models,
            partition=replace(partition, partition_hash="not-a-sha256"),
            label_manager=capability,
            protocol=protocol,
        )
    folds = build_signed_fold_products(
        probabilities=probabilities,
        model_products=models,
        partition=partition,
        label_manager=capability,
        protocol=protocol,
    )
    assert folds.partition_hash == partition.partition_hash
    assert all(
        decision["partition_hash"] == partition.partition_hash
        for decision in folds.decisions
    )
    with pytest.raises(ProtocolError, match="before durable seals"):
        capability.open_oof_evaluation_labels()
    record_durable_fold_seals(capability, folds)
    terminal_labels = capability.open_oof_evaluation_labels()
    capability_report = capability.access_report()
    small_scientific_result = _evaluate_terminal_predictions(
        predictions_by_method=folds.predictions_by_method,
        labels=terminal_labels,
        bootstrap_replicates=8,
        bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_workers=1,
        bootstrap_threads_per_worker=1,
    )
    canonical_contrasts = tuple(
        replace(
            contrast,
            bootstrap=(
                replace(
                    contrast.bootstrap,
                    replicate_count=BOOTSTRAP_REPLICATES,
                    seed=BOOTSTRAP_SEED,
                )
                if contrast.bootstrap is not None
                else None
            ),
        )
        for contrast in small_scientific_result.contrasts
    )
    canonical_scientific_result = replace(
        small_scientific_result,
        contrasts=canonical_contrasts,
    )
    observed_runtime: dict[str, object] = {}

    def fake_terminal_evaluator(**kwargs):
        observed_runtime.update(kwargs)
        return canonical_scientific_result

    monkeypatch.setattr(
        terminal_module,
        "_evaluate_terminal_predictions",
        fake_terminal_evaluator,
    )

    with pytest.raises(ProtocolError, match="canonical protocol"):
        evaluate_sealed_fold_products(
            fold_products=folds,
            capability_report=capability_report,
            terminal_labels=terminal_labels,
            protocol=protocol,
            bootstrap_replicates=16,
            bootstrap_seed=BOOTSTRAP_SEED,
        )
    with pytest.raises(ProtocolError, match="canonical protocol"):
        evaluate_sealed_fold_products(
            fold_products=folds,
            capability_report=capability_report,
            terminal_labels=terminal_labels,
            protocol=protocol,
            bootstrap_replicates=BOOTSTRAP_REPLICATES,
            bootstrap_seed=BOOTSTRAP_SEED + 1,
        )
    with pytest.raises(ProtocolError, match="requires spawn"):
        evaluate_sealed_fold_products(
            fold_products=folds,
            capability_report=capability_report,
            terminal_labels=terminal_labels,
            protocol=protocol,
            bootstrap_replicates=BOOTSTRAP_REPLICATES,
            bootstrap_seed=BOOTSTRAP_SEED,
            multiprocessing_start_method="fork",
        )
    with pytest.raises(ProtocolError, match="frozen budget"):
        evaluate_sealed_fold_products(
            fold_products=folds,
            capability_report=capability_report,
            terminal_labels=terminal_labels,
            protocol=protocol,
            bootstrap_replicates=BOOTSTRAP_REPLICATES,
            bootstrap_seed=BOOTSTRAP_SEED,
            bootstrap_workers=5,
            bootstrap_threads_per_worker=3,
        )
    sealed_result = evaluate_sealed_fold_products(
        fold_products=folds,
        capability_report=capability_report,
        terminal_labels=terminal_labels,
        protocol=protocol,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_workers=1,
        bootstrap_threads_per_worker=1,
    )
    assert tuple(
        row.method_id for row in sealed_result.scientific_result.method_results
    ) == METHOD_IDS
    sealed_payload = sealed_result.to_payload()
    assert sealed_payload["evidence_status"] == "EXPLORATORY_CONSUMED_DATA_ONLY"
    assert sealed_payload["terminal_consumed_test_diagnostic_only"] is True
    assert sealed_payload["policy_update_authorized"] is False
    assert sealed_payload["protocol_contract_hash"] == protocol.contract_hash
    assert sealed_payload["partition_hash"] == partition.partition_hash
    assert sealed_payload["capability_report_hash"] == capability_report["report_hash"]
    assert sealed_payload["decision_seal_hash"] == folds.decision_seal_hash
    assert (
        sealed_payload["permutation_provenance_hash"]
        == folds.permutation_provenance_hash
    )
    assert sealed_payload["runtime"] == {
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_workers": 1,
        "bootstrap_threads_per_worker": 1,
        "multiprocessing_start_method": "spawn",
    }
    assert observed_runtime["bootstrap_replicates"] == BOOTSTRAP_REPLICATES
    assert observed_runtime["bootstrap_seed"] == BOOTSTRAP_SEED
    assert observed_runtime["multiprocessing_start_method"] == "spawn"
    assert (
        sealed_payload["scientific_result_hash"]
        == sealed_result.scientific_result.scientific_result_hash
    )
    assert "evidence_status" not in sealed_payload["scientific_result"]
    assert (
        "terminal_consumed_test_diagnostic_only"
        not in sealed_payload["scientific_result"]
    )
    assert "policy_update_authorized" not in sealed_payload["scientific_result"]
    with pytest.raises(ProtocolError, match="canonical protocol runtime"):
        replace(sealed_result, bootstrap_replicates=8)

    duplicate_cases = dict(folds.decisions[0])
    evaluation_case_ids = list(duplicate_cases["evaluation_case_ids"])
    duplicate_cases["evaluation_case_ids"] = [
        *evaluation_case_ids,
        evaluation_case_ids[0],
    ]
    with pytest.raises(ProtocolError, match="malformed"):
        evaluate_sealed_fold_products(
            fold_products=replace(
                folds,
                decisions=(duplicate_cases, *folds.decisions[1:]),
            ),
            capability_report=capability_report,
            terminal_labels=terminal_labels,
            protocol=protocol,
            bootstrap_replicates=BOOTSTRAP_REPLICATES,
            bootstrap_seed=BOOTSTRAP_SEED,
        )

    duplicate_fold = dict(folds.decisions[0])
    duplicate_fold["fold_ordinal"] = folds.decisions[1]["fold_ordinal"]
    with pytest.raises(ProtocolError, match="topology"):
        evaluate_sealed_fold_products(
            fold_products=replace(
                folds,
                decisions=(duplicate_fold, *folds.decisions[1:]),
            ),
            capability_report=capability_report,
            terminal_labels=terminal_labels,
            protocol=protocol,
            bootstrap_replicates=BOOTSTRAP_REPLICATES,
            bootstrap_seed=BOOTSTRAP_SEED,
        )

    different_partition_hash = (
        "0" * 64 if folds.partition_hash != "0" * 64 else "1" * 64
    )
    with pytest.raises(ProtocolError, match="partition hash drifted"):
        evaluate_sealed_fold_products(
            fold_products=replace(folds, partition_hash=different_partition_hash),
            capability_report=capability_report,
            terminal_labels=terminal_labels,
            protocol=protocol,
            bootstrap_replicates=BOOTSTRAP_REPLICATES,
            bootstrap_seed=BOOTSTRAP_SEED,
        )


def test_workstation_cpu_budget_and_support_scope_fail_closed() -> None:
    probabilities, labels, partition = _surface()
    protocol = canonical_consumed_test_protocol()
    capability = _Capability(labels)
    prelabel = build_signed_prelabel_products(probabilities, protocol=protocol)
    with pytest.raises(ProtocolError, match="W-2265"):
        fit_all_target_families(
            probabilities=probabilities,
            prelabel=prelabel,
            label_manager=capability,
            protocol=protocol,
            worker_count=5,
            threads_per_worker=3,
        )

    capability = _Capability(labels)
    models = fit_all_target_families(
        probabilities=probabilities,
        prelabel=prelabel,
        label_manager=capability,
        protocol=protocol,
        worker_count=1,
        threads_per_worker=1,
    )
    record_durable_model_seals(capability, models)
    original = capability.open_fold_support_labels

    def contaminated(target, fold):
        rows = list(original(target, fold))
        rows.append(
            replace(
                next(
                    row
                    for row in labels
                    if row.target_center == target
                    and row.case_id == f"case-{target}-{fold}"
                ),
                label_scope="target_support",
            )
        )
        return tuple(rows)

    capability.open_fold_support_labels = contaminated
    with pytest.raises(ProtocolError, match="mis-scoped"):
        build_signed_fold_products(
            probabilities=probabilities,
            model_products=models,
            partition=partition,
            label_manager=capability,
            protocol=protocol,
        )
