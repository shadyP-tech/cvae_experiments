"""Terminal adapter requiring durable signed-gate seals before scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ..fixed_bank_hierarchical_residual_stacker.contracts import BinaryLabel
from ..fixed_bank_hierarchical_residual_stacker.core_hashing import canonical_hash
from ..fixed_bank_hierarchical_residual_stacker.experiment_contracts import (
    OOF_FOLD_COUNT,
)
from ..fixed_bank_hierarchical_residual_stacker.scientific_constants import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    MIDOGPP_CENTERS,
)
from .constants import METHOD_IDS
from .evaluation import (
    SignedGateEvaluationResult,
    _evaluate_terminal_predictions,
    _validate_runtime,
)
from .execution import SignedFoldProducts, _require_sha256
from .protocol import (
    SignedErrorGateProtocol,
    assert_consumed_test_diagnostic_only,
    canonical_consumed_test_protocol,
)


@dataclass(frozen=True)
class SealedSignedGateEvaluationResult:
    """Claim-bearing terminal envelope for one fully validated sealed surface."""

    scientific_result: SignedGateEvaluationResult
    protocol_contract_hash: str
    partition_hash: str
    capability_report_hash: str
    decision_seal_hash: str
    permutation_provenance_hash: str
    bootstrap_replicates: int
    bootstrap_seed: int
    bootstrap_workers: int
    bootstrap_threads_per_worker: int
    multiprocessing_start_method: str
    sealed_result_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.scientific_result, SignedGateEvaluationResult):
            raise ProtocolError("Sealed terminal envelope lacks a scientific result.")
        for value, name in (
            (self.scientific_result.scientific_result_hash, "scientific_result_hash"),
            (self.protocol_contract_hash, "protocol_contract_hash"),
            (self.partition_hash, "partition_hash"),
            (self.capability_report_hash, "capability_report_hash"),
            (self.decision_seal_hash, "decision_seal_hash"),
            (self.permutation_provenance_hash, "permutation_provenance_hash"),
        ):
            _require_sha256(value, f"Sealed terminal {name}")
        _validate_runtime(
            bootstrap_replicates=self.bootstrap_replicates,
            bootstrap_seed=self.bootstrap_seed,
            bootstrap_workers=self.bootstrap_workers,
            multiprocessing_start_method=self.multiprocessing_start_method,
            bootstrap_threads_per_worker=self.bootstrap_threads_per_worker,
        )
        if (
            self.protocol_contract_hash
            != canonical_consumed_test_protocol().contract_hash
            or self.bootstrap_replicates != BOOTSTRAP_REPLICATES
            or self.bootstrap_seed != BOOTSTRAP_SEED
            or self.multiprocessing_start_method != "spawn"
        ):
            raise ProtocolError(
                "Sealed terminal envelope drifted from the canonical protocol runtime."
            )
        if any(
            row.bootstrap is None
            or row.bootstrap.replicate_count != self.bootstrap_replicates
            or row.bootstrap.seed != self.bootstrap_seed
            for row in self.scientific_result.primary_contrasts
        ):
            raise ProtocolError(
                "Sealed terminal bootstrap summaries drifted from their runtime binding."
            )
        object.__setattr__(
            self, "sealed_result_hash", canonical_hash(self._unhashed())
        )

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_signed_error_sealed_terminal_evaluation_v1",
            "scientific_result": self.scientific_result.to_payload(),
            "scientific_result_hash": self.scientific_result.scientific_result_hash,
            "protocol_contract_hash": self.protocol_contract_hash,
            "partition_hash": self.partition_hash,
            "capability_report_hash": self.capability_report_hash,
            "decision_seal_hash": self.decision_seal_hash,
            "permutation_provenance_hash": self.permutation_provenance_hash,
            "runtime": {
                "bootstrap_replicates": self.bootstrap_replicates,
                "bootstrap_seed": self.bootstrap_seed,
                "bootstrap_workers": self.bootstrap_workers,
                "bootstrap_threads_per_worker": self.bootstrap_threads_per_worker,
                "multiprocessing_start_method": self.multiprocessing_start_method,
            },
            "evidence_status": "EXPLORATORY_CONSUMED_DATA_ONLY",
            "terminal_consumed_test_diagnostic_only": True,
            "policy_update_authorized": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "sealed_result_hash": self.sealed_result_hash}


def evaluate_sealed_fold_products(
    *,
    fold_products: SignedFoldProducts,
    capability_report: Mapping[str, object],
    terminal_labels: Sequence[BinaryLabel],
    protocol: SignedErrorGateProtocol,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    bootstrap_workers: int = 4,
    bootstrap_threads_per_worker: int = 3,
    multiprocessing_start_method: str = "spawn",
) -> SealedSignedGateEvaluationResult:
    """Verify durable prediction/capability seals, then call the pure evaluator."""

    assert_consumed_test_diagnostic_only(protocol)
    _validate_runtime(
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        bootstrap_workers=bootstrap_workers,
        multiprocessing_start_method=multiprocessing_start_method,
        bootstrap_threads_per_worker=bootstrap_threads_per_worker,
    )
    if (
        bootstrap_replicates != protocol.bootstrap_replicates
        or bootstrap_seed != protocol.bootstrap_seed
        or multiprocessing_start_method != protocol.multiprocessing_start_method
    ):
        raise ProtocolError(
            "Sealed terminal runtime drifted from the canonical protocol contract."
        )
    if fold_products.protocol_contract_hash != protocol.contract_hash:
        raise ProtocolError("Terminal signed-gate product has a different protocol hash.")
    _validate_capability_report(capability_report, fold_products)
    _validate_prediction_seals(fold_products)
    scientific_result = _evaluate_terminal_predictions(
        predictions_by_method=fold_products.predictions_by_method,
        labels=terminal_labels,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        bootstrap_workers=bootstrap_workers,
        multiprocessing_start_method=multiprocessing_start_method,
        bootstrap_threads_per_worker=bootstrap_threads_per_worker,
    )
    return SealedSignedGateEvaluationResult(
        scientific_result=scientific_result,
        protocol_contract_hash=fold_products.protocol_contract_hash,
        partition_hash=fold_products.partition_hash,
        capability_report_hash=_require_sha256(
            capability_report.get("report_hash"),
            "Terminal signed-error capability report hash",
        ),
        decision_seal_hash=fold_products.decision_seal_hash,
        permutation_provenance_hash=fold_products.permutation_provenance_hash,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        bootstrap_workers=bootstrap_workers,
        bootstrap_threads_per_worker=bootstrap_threads_per_worker,
        multiprocessing_start_method=multiprocessing_start_method,
    )


def _validate_capability_report(
    report: Mapping[str, object], products: SignedFoldProducts
) -> None:
    unhashed = {key: value for key, value in report.items() if key != "report_hash"}
    if (
        report.get("schema_version")
        != "midogpp_signed_error_label_capability_report_v1"
        or report.get("status") != "PASS"
        or report.get("diagnostic_method_ids") != list(METHOD_IDS)
        or report.get("fold_method_decision_count") != 45 * len(METHOD_IDS)
        or report.get("all_decisions_seal_hash") != products.decision_seal_hash
        or report.get("permutation_provenance_hash")
        != products.permutation_provenance_hash
        or report.get("evaluation_labels_opened") is not True
        or report.get("R_raw_and_R_safe_separately_sealed") is not True
        or report.get("terminal_consumed_test_diagnostic_only") is not True
        or report.get("report_hash") != canonical_hash(unhashed)
    ):
        raise ProtocolError("Terminal labels lack a valid signed-gate capability seal.")


def _validate_prediction_seals(products: SignedFoldProducts) -> None:
    methods = set(METHOD_IDS)
    partition_hash = _require_sha256(
        products.partition_hash, "Terminal signed-error partition_hash"
    )
    expected_topology = {
        (target, ordinal)
        for target in MIDOGPP_CENTERS
        for ordinal in range(OOF_FOLD_COUNT)
    }
    observed_topology: list[tuple[str, int]] = []
    for decision in products.decisions:
        target = decision.get("target_center")
        ordinal = decision.get("fold_ordinal")
        if (
            type(target) is not str
            or target not in MIDOGPP_CENTERS
            or type(ordinal) is not int
        ):
            raise ProtocolError("Signed-gate fold decision identity is malformed.")
        observed_topology.append((target, ordinal))
    if (
        set(products.predictions_by_method) != methods
        or len(observed_topology) != len(expected_topology)
        or set(observed_topology) != expected_topology
        or len(observed_topology) != len(set(observed_topology))
    ):
        raise ProtocolError("Sealed signed-gate prediction topology drifted.")
    observed_decision_hashes: list[str] = []
    covered_cases: dict[str, set[tuple[str, str]]] = {
        method: set() for method in METHOD_IDS
    }
    for decision in products.decisions:
        target = str(decision["target_center"])
        cases_raw = decision.get("evaluation_case_ids")
        prediction_hashes = decision.get("method_prediction_hashes")
        method_decision_hashes = decision.get("method_decision_hashes")
        if (
            not isinstance(cases_raw, list)
            or not cases_raw
            or any(type(value) is not str or not value for value in cases_raw)
            or len(cases_raw) != len(set(cases_raw))
            or not isinstance(prediction_hashes, dict)
            or set(prediction_hashes) != methods
            or not isinstance(method_decision_hashes, dict)
            or set(method_decision_hashes) != methods
        ):
            raise ProtocolError("Signed-gate decision seal payload is malformed.")
        if decision.get("partition_hash") != partition_hash:
            raise ProtocolError("Signed-gate fold decision partition hash drifted.")
        cases = set(cases_raw)
        common = {
            key: value
            for key, value in decision.items()
            if key
            not in (
                "method_prediction_hashes",
                "method_decision_hashes",
                "decision_hash",
            )
        }
        for method in METHOD_IDS:
            rows = tuple(
                row
                for row in products.predictions_by_method[method]
                if row.target_center == target and row.case_id in cases
            )
            case_keys = {(row.target_center, row.case_id) for row in rows}
            prediction_hash = canonical_hash([row.to_payload() for row in rows])
            method_hash = canonical_hash(
                {
                    "common": common,
                    "method_id": method,
                    "prediction_hash": prediction_hash,
                }
            )
            if (
                not rows
                or prediction_hashes[method] != prediction_hash
                or method_decision_hashes[method] != method_hash
            ):
                raise ProtocolError("Signed-gate per-method prediction seal drifted.")
            if covered_cases[method].intersection(case_keys):
                raise ProtocolError("Signed-gate fold decisions reuse evaluation cases.")
            covered_cases[method].update(case_keys)
        unhashed = {
            **common,
            "method_prediction_hashes": prediction_hashes,
            "method_decision_hashes": method_decision_hashes,
        }
        decision_hash = canonical_hash(unhashed)
        if decision.get("decision_hash") != decision_hash:
            raise ProtocolError("Signed-gate fold decision hash drifted.")
        observed_decision_hashes.append(decision_hash)
    expected_seal = canonical_hash(
        {
            "schema_version": "fixed_bank_signed_error_all_fold_decisions_v1",
            "partition_hash": partition_hash,
            "decision_hashes": observed_decision_hashes,
            "R_raw_and_R_safe_prediction_hashes_separate": True,
            "evaluation_labels_used": False,
        }
    )
    if products.decision_seal_hash != expected_seal:
        raise ProtocolError("Signed-gate all-decision seal drifted.")
    for method in METHOD_IDS:
        expected_cases = {
            (row.target_center, row.case_id)
            for row in products.predictions_by_method[method]
        }
        if covered_cases[method] != expected_cases:
            raise ProtocolError("Signed-gate prediction seals lack exact case coverage.")


__all__ = (
    "SealedSignedGateEvaluationResult",
    "evaluate_sealed_fold_products",
)
