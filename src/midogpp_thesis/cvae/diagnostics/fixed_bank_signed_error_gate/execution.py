"""Phase-level assembly for workstation and synthetic signed-gate diagnostics."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing
from typing import Mapping, Protocol, Sequence

from ...protocol import ProtocolError
from ..fixed_bank_hierarchical_residual_stacker.composition import (
    baseline_predictions,
    calibrated_baseline_predictions,
)
from ..fixed_bank_hierarchical_residual_stacker.contracts import (
    BinaryLabel,
    PredictionRow,
    SampleActionProbability,
)
from ..fixed_bank_hierarchical_residual_stacker.core_hashing import canonical_hash
from ..fixed_bank_hierarchical_residual_stacker.scientific_constants import (
    BASELINE_ACTION_ID,
    MIDOGPP_CENTERS,
)
from ..fixed_bank_hierarchical_residual_stacker.experiment_contracts import (
    OOF_FOLD_COUNT,
)
from .calibration import fit_signed_gate_decision
from .composition import compose_signed_predictions, threshold_crossing_count
from .contracts import CorrectionRow, SignedFeatureRow
from .features import (
    build_signed_features,
    feature_context_hash,
    permute_feature_alignment,
)
from .gradients import build_gradient_targets
from .label_capabilities import SignedErrorLabelCapability
from .model import (
    SignedGateFit,
    correction_surface_hash,
    fit_signed_gate,
    predict_corrections,
)
from .protocol import (
    SignedErrorGateProtocol,
    assert_consumed_test_diagnostic_only,
)


@dataclass(frozen=True)
class TargetFamilyFits:
    target_center: str
    global_fit: SignedGateFit
    residual_fit: SignedGateFit
    permutation_fit: SignedGateFit
    global_corrections: tuple[CorrectionRow, ...]
    residual_corrections: tuple[CorrectionRow, ...]
    permutation_corrections: tuple[CorrectionRow, ...]

    def __post_init__(self) -> None:
        fits = (self.global_fit, self.residual_fit, self.permutation_fit)
        corrections = (
            self.global_corrections,
            self.residual_corrections,
            self.permutation_corrections,
        )
        if (
            self.target_center not in MIDOGPP_CENTERS
            or tuple(value.final_model.target_center for value in fits)
            != (self.target_center,) * 3
            or tuple(value.final_model.family for value in fits) != ("G", "R", "P")
            or any(
                not rows
                or any(row.target_center != self.target_center for row in rows)
                for rows in corrections
            )
        ):
            raise ProtocolError("Signed-error target-family products drifted.")

    @property
    def model_seal_hash(self) -> str:
        return canonical_hash(
            {
                "schema_version": "fixed_bank_signed_error_target_models_v1",
                "target_center": self.target_center,
                "model_hashes": [
                    self.global_fit.final_model.model_hash,
                    self.residual_fit.final_model.model_hash,
                    self.permutation_fit.final_model.model_hash,
                ],
                "fit_hashes": [
                    self.global_fit.fit_hash,
                    self.residual_fit.fit_hash,
                    self.permutation_fit.fit_hash,
                ],
                "target_labels_used": False,
            }
        )


@dataclass(frozen=True)
class SignedPrelabelProducts:
    context_hashes: Mapping[str, str]
    feature_surface_hash: str
    protocol_contract_hash: str


@dataclass(frozen=True)
class SignedModelProducts:
    target_fits: tuple[TargetFamilyFits, ...]
    raw_correction_surface_hash: str
    safe_correction_surface_hash: str
    control_correction_surface_hash: str
    protocol_contract_hash: str

    def __post_init__(self) -> None:
        fits = tuple(self.target_fits)
        if tuple(row.target_center for row in fits) != MIDOGPP_CENTERS:
            raise ProtocolError(
                "Signed-error model products require one ordered fit per center."
            )
        for value, name in (
            (self.raw_correction_surface_hash, "raw_correction_surface_hash"),
            (self.safe_correction_surface_hash, "safe_correction_surface_hash"),
            (self.control_correction_surface_hash, "control_correction_surface_hash"),
            (self.protocol_contract_hash, "protocol_contract_hash"),
        ):
            _require_sha256(value, f"Signed-error {name}")
        object.__setattr__(self, "target_fits", fits)


@dataclass(frozen=True)
class SignedFoldProducts:
    decisions: tuple[Mapping[str, object], ...]
    predictions_by_method: Mapping[str, tuple[PredictionRow, ...]]
    decision_seal_hash: str
    permutation_provenance_hash: str
    partition_hash: str
    protocol_contract_hash: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.decision_seal_hash, "decision_seal_hash"),
            (self.permutation_provenance_hash, "permutation_provenance_hash"),
            (self.partition_hash, "partition_hash"),
            (self.protocol_contract_hash, "protocol_contract_hash"),
        ):
            _require_sha256(value, f"Signed-error {name}")
        if set(self.predictions_by_method) != {
            "B",
            "B_cal",
            "G",
            "R_raw",
            "R_safe",
            "P",
        }:
            raise ProtocolError("Signed-error fold products require all six methods.")


class SignedPartitionFold(Protocol):
    """Narrow structural contract consumed from the canonical case partition."""

    target_center: str
    fold_ordinal: int
    support_case_ids: Sequence[str]
    evaluation_case_ids: Sequence[str]
    fold_hash: str


class SignedPartition(Protocol):
    """Validated partition identity and folds needed by signed-gate execution."""

    folds: Sequence[SignedPartitionFold]
    partition_hash: str


def build_signed_prelabel_products(
    probabilities: Sequence[SampleActionProbability],
    *,
    protocol: SignedErrorGateProtocol,
) -> SignedPrelabelProducts:
    assert_consumed_test_diagnostic_only(protocol)
    context_hashes: dict[str, str] = {}
    for target in MIDOGPP_CENTERS:
        outer = build_signed_features(
            probabilities, excluded_candidate_centers=(target,)
        )
        outer_permuted = permute_feature_alignment(outer)
        context_hashes[_context_key(target, None, "aligned")] = (
            feature_context_hash(outer, control="aligned")
        )
        context_hashes[_context_key(target, None, "permuted")] = (
            feature_context_hash(outer_permuted, control="permuted")
        )
        for query in MIDOGPP_CENTERS:
            if query == target:
                continue
            nested = build_signed_features(
                probabilities, excluded_candidate_centers=(target, query)
            )
            nested_permuted = permute_feature_alignment(nested)
            context_hashes[_context_key(target, query, "aligned")] = (
                feature_context_hash(nested, control="aligned")
            )
            context_hashes[_context_key(target, query, "permuted")] = (
                feature_context_hash(nested_permuted, control="permuted")
            )
    surface_hash = canonical_hash(
        {
            "schema_version": "fixed_bank_signed_error_prelabel_features_v1",
            "context_hashes": dict(sorted(context_hashes.items())),
            "outer_context_count": len(MIDOGPP_CENTERS),
            "nested_context_count": len(MIDOGPP_CENTERS)
            * (len(MIDOGPP_CENTERS) - 1),
            "labels_used": False,
            "baseline_predicted_class_branch_used": False,
        }
    )
    return SignedPrelabelProducts(
        dict(sorted(context_hashes.items())), surface_hash, protocol.contract_hash
    )


def fit_target_families(
    *,
    target_center: str,
    probabilities: Sequence[SampleActionProbability],
    prelabel_context_hashes: Mapping[str, str],
    donor_labels: Sequence[BinaryLabel],
    protocol: SignedErrorGateProtocol,
) -> TargetFamilyFits:
    assert_consumed_test_diagnostic_only(protocol)
    outer_features = build_signed_features(
        probabilities, excluded_candidate_centers=(target_center,)
    )
    outer_permuted = permute_feature_alignment(outer_features)
    _verify_context_hash(
        prelabel_context_hashes,
        target_center,
        None,
        "aligned",
        outer_features,
    )
    _verify_context_hash(
        prelabel_context_hashes,
        target_center,
        None,
        "permuted",
        outer_permuted,
    )
    nested_features: dict[str, tuple[SignedFeatureRow, ...]] = {}
    nested_permuted: dict[str, tuple[SignedFeatureRow, ...]] = {}
    for query in MIDOGPP_CENTERS:
        if query == target_center:
            continue
        aligned = build_signed_features(
            probabilities, excluded_candidate_centers=(target_center, query)
        )
        permuted = permute_feature_alignment(aligned)
        _verify_context_hash(
            prelabel_context_hashes,
            target_center,
            query,
            "aligned",
            aligned,
        )
        _verify_context_hash(
            prelabel_context_hashes,
            target_center,
            query,
            "permuted",
            permuted,
        )
        nested_features[query] = aligned
        nested_permuted[query] = permuted
    gradients = build_gradient_targets(
        probabilities, donor_labels, heldout_target=target_center
    )
    keys = {row.sample_key for row in gradients}
    legal_features = tuple(row for row in outer_features if row.sample_key in keys)
    legal_permuted = tuple(row for row in outer_permuted if row.sample_key in keys)
    legal_nested = {
        query: tuple(row for row in rows if row.sample_key in keys)
        for query, rows in nested_features.items()
    }
    legal_nested_permuted = {
        query: tuple(row for row in rows if row.sample_key in keys)
        for query, rows in nested_permuted.items()
    }
    if any(key[0] == target_center for key in keys):
        raise ProtocolError("Target labels entered signed-error shared-model fitting.")
    global_fit = fit_signed_gate(
        legal_features,
        gradients,
        target_center=target_center,
        family="G",
        nested_training_features=legal_nested,
    )
    residual_fit = fit_signed_gate(
        legal_features,
        gradients,
        target_center=target_center,
        family="R",
        nested_training_features=legal_nested,
    )
    permutation_fit = fit_signed_gate(
        legal_permuted,
        gradients,
        target_center=target_center,
        family="P",
        nested_training_features=legal_nested_permuted,
    )
    return TargetFamilyFits(
        target_center,
        global_fit,
        residual_fit,
        permutation_fit,
        predict_corrections(
            global_fit,
            outer_features,
            nested_prediction_features=nested_features,
        ),
        predict_corrections(
            residual_fit,
            outer_features,
            nested_prediction_features=nested_features,
        ),
        predict_corrections(
            permutation_fit,
            outer_permuted,
            nested_prediction_features=nested_permuted,
        ),
    )


def fit_all_target_families(
    *,
    probabilities: Sequence[SampleActionProbability],
    prelabel: SignedPrelabelProducts,
    label_manager: SignedErrorLabelCapability,
    protocol: SignedErrorGateProtocol,
    worker_count: int = 4,
    threads_per_worker: int = 3,
) -> SignedModelProducts:
    assert_consumed_test_diagnostic_only(protocol)
    if prelabel.protocol_contract_hash != protocol.contract_hash:
        raise ProtocolError("Signed-error prelabel surface has a different protocol hash.")
    if (
        type(worker_count) is not int
        or type(threads_per_worker) is not int
        or not 1 <= worker_count <= 4
        or not 1 <= threads_per_worker <= 3
        or worker_count * threads_per_worker > 12
    ):
        raise ProtocolError("Signed-error CPU pool exceeds the frozen W-2265 budget.")
    tasks = []
    for target in MIDOGPP_CENTERS:
        labels = label_manager.open_loco_donor_labels(target)
        tasks.append(
            (
                target,
                tuple(probabilities),
                dict(prelabel.context_hashes),
                tuple(labels),
                int(threads_per_worker),
                protocol,
            )
        )
    if worker_count <= 1:
        fits = tuple(_fit_target_task(task) for task in tasks)
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(int(worker_count), len(tasks)), mp_context=context
        ) as pool:
            fits = tuple(pool.map(_fit_target_task, tasks, chunksize=1))
    residual_rows = tuple(
        row for fit in fits for row in fit.residual_corrections
    )
    control_rows = tuple(
        row
        for fit in fits
        for surface in (
            fit.global_corrections,
            fit.permutation_corrections,
        )
        for row in surface
    )
    return SignedModelProducts(
        fits,
        correction_surface_hash(residual_rows, surface="raw"),
        correction_surface_hash(residual_rows, surface="safe"),
        correction_surface_hash(control_rows, surface="combined"),
        protocol.contract_hash,
    )


def _fit_target_task(task: tuple[object, ...]) -> TargetFamilyFits:
    from threadpoolctl import threadpool_limits

    target, probabilities, context_hashes, labels, thread_count, protocol = task
    with threadpool_limits(limits=int(thread_count)):
        return fit_target_families(
            target_center=str(target),
            probabilities=probabilities,  # type: ignore[arg-type]
            prelabel_context_hashes=context_hashes,  # type: ignore[arg-type]
            donor_labels=labels,  # type: ignore[arg-type]
            protocol=protocol,  # type: ignore[arg-type]
        )


def _context_key(target: str, query: str | None, control: str) -> str:
    return f"H={target}|q={query if query is not None else 'FINAL'}|{control}"


def _verify_context_hash(
    expected: Mapping[str, str],
    target: str,
    query: str | None,
    control: str,
    rows: Sequence[SignedFeatureRow],
) -> None:
    key = _context_key(target, query, control)
    observed = feature_context_hash(rows, control=control)
    if expected.get(key) != observed:
        raise ProtocolError("Signed-error feature context drifted after its prelabel seal.")


def build_signed_fold_products(
    *,
    probabilities: Sequence[SampleActionProbability],
    model_products: SignedModelProducts,
    partition: SignedPartition,
    label_manager: SignedErrorLabelCapability,
    protocol: SignedErrorGateProtocol,
) -> SignedFoldProducts:
    assert_consumed_test_diagnostic_only(protocol)
    if model_products.protocol_contract_hash != protocol.contract_hash:
        raise ProtocolError("Signed-error model surface has a different protocol hash.")
    partition_hash = _require_sha256(
        partition.partition_hash, "Signed-error partition_hash"
    )
    folds = tuple(partition.folds)
    if {
        (fold.target_center, fold.fold_ordinal) for fold in folds
    } != {
        (target, ordinal)
        for target in MIDOGPP_CENTERS
        for ordinal in range(OOF_FOLD_COUNT)
    } or len(folds) != len(MIDOGPP_CENTERS) * OOF_FOLD_COUNT:
        raise ProtocolError("Signed-error partition lacks the exact 45-fold topology.")
    fit_by_target = {row.target_center: row for row in model_products.target_fits}
    if (
        len(model_products.target_fits) != len(MIDOGPP_CENTERS)
        or len(fit_by_target) != len(model_products.target_fits)
        or set(fit_by_target) != set(MIDOGPP_CENTERS)
    ):
        raise ProtocolError("Signed-error model products lack all target centers.")
    all_cases_by_target = {
        target: {
            row.case_id for row in probabilities if row.target_center == target
        }
        for target in MIDOGPP_CENTERS
    }
    evaluated_cases: dict[str, list[str]] = {target: [] for target in MIDOGPP_CENTERS}
    predictions: dict[str, list[PredictionRow]] = {
        method: [] for method in ("B", "B_cal", "G", "R_raw", "R_safe", "P")
    }
    decisions: list[Mapping[str, object]] = []
    for fold in folds:
        target = fold.target_center
        if target not in fit_by_target:
            raise ProtocolError("Signed-error fold uses an unknown target center.")
        fitted = fit_by_target[target]
        support_cases = set(fold.support_case_ids)
        evaluation_cases = set(fold.evaluation_case_ids)
        if (
            not support_cases
            or not evaluation_cases
            or support_cases.intersection(evaluation_cases)
            or support_cases.union(evaluation_cases) != all_cases_by_target[target]
        ):
            raise ProtocolError("Signed-error fold violates whole-case separation.")
        support_labels = label_manager.open_fold_support_labels(
            target, fold.fold_ordinal
        )
        support_keys = {row.sample_key for row in support_labels}
        expected_support_keys = {
            row.sample_key
            for row in probabilities
            if row.target_center == target
            and row.case_id in support_cases
            and row.action_id == BASELINE_ACTION_ID
        }
        if (
            any(row.target_center != target for row in support_labels)
            or {row.case_id for row in support_labels} != support_cases
            or any(row.label_scope != "target_support" for row in support_labels)
            or len(support_keys) != len(tuple(support_labels))
            or support_keys != expected_support_keys
        ):
            raise ProtocolError("Signed-error support capability returned a mis-scoped surface.")
        decision = fit_signed_gate_decision(
            probabilities, fitted.residual_corrections, support_labels
        )
        target_probabilities = tuple(
            row for row in probabilities if row.target_center == target
        )
        method_predictions = {
            "B": baseline_predictions(target_probabilities, method_id="B"),
            "B_cal": calibrated_baseline_predictions(
                target_probabilities,
                intercept=decision.intercept,
                method_id="B_cal",
            ),
            "G": compose_signed_predictions(
                target_probabilities,
                fitted.global_corrections,
                intercept=decision.intercept,
                residual_scale=decision.selected_scale,
                method_id="G",
                safe=True,
            ),
            "R_raw": compose_signed_predictions(
                target_probabilities,
                fitted.residual_corrections,
                intercept=decision.intercept,
                residual_scale=decision.selected_scale,
                method_id="R_raw",
                safe=False,
            ),
            "R_safe": compose_signed_predictions(
                target_probabilities,
                fitted.residual_corrections,
                intercept=decision.intercept,
                residual_scale=decision.selected_scale,
                method_id="R_safe",
                safe=True,
            ),
            "P": compose_signed_predictions(
                target_probabilities,
                fitted.permutation_corrections,
                intercept=decision.intercept,
                residual_scale=decision.selected_scale,
                method_id="P",
                safe=True,
            ),
        }
        selected_rows = {
            method: tuple(row for row in rows if row.case_id in evaluation_cases)
            for method, rows in method_predictions.items()
        }
        reference_keys = {row.sample_key for row in selected_rows["B_cal"]}
        if (
            not reference_keys
            or {row.case_id for row in selected_rows["B_cal"]} != evaluation_cases
            or any(
                {row.sample_key for row in rows} != reference_keys
                for rows in selected_rows.values()
            )
        ):
            raise ProtocolError("Signed-error evaluation prediction coverage drifted.")
        evaluated_cases[target].extend(sorted(evaluation_cases))
        for method, rows in selected_rows.items():
            predictions[method].extend(rows)
        bcal = selected_rows["B_cal"]
        method_prediction_hashes = {
            method: canonical_hash([row.to_payload() for row in rows])
            for method, rows in selected_rows.items()
        }
        common = {
            "schema_version": "fixed_bank_signed_error_fold_decision_v1",
            "target_center": target,
            "fold_ordinal": fold.fold_ordinal,
            "fold_hash": fold.fold_hash,
            "partition_hash": partition_hash,
            "evaluation_case_ids": sorted(evaluation_cases),
            "intercept": decision.intercept,
            "proposed_scale": decision.proposed_scale,
            "selected_scale": decision.selected_scale,
            "support_bacc_lcb": decision.support_bacc_lcb,
            "fallback_reason": decision.fallback_reason,
            "lambda_path": [row.to_payload() for row in decision.lambda_path],
            "evaluation_threshold_crossings": {
                method: threshold_crossing_count(rows, bcal)
                for method, rows in selected_rows.items()
                if method not in ("B", "B_cal")
            },
            "model_seal_hash": fitted.model_seal_hash,
            "R_raw_correction_surface_hash": (
                model_products.raw_correction_surface_hash
            ),
            "R_safe_correction_surface_hash": (
                model_products.safe_correction_surface_hash
            ),
            "control_correction_surface_hash": (
                model_products.control_correction_surface_hash
            ),
            "evaluation_labels_used": False,
            "terminal_consumed_test_diagnostic_only": True,
        }
        method_decision_hashes = {
            method: canonical_hash(
                {
                    "common": common,
                    "method_id": method,
                    "prediction_hash": method_prediction_hashes[method],
                }
            )
            for method in method_prediction_hashes
        }
        unhashed = {
            **common,
            "method_prediction_hashes": method_prediction_hashes,
            "method_decision_hashes": method_decision_hashes,
        }
        decisions.append({**unhashed, "decision_hash": canonical_hash(unhashed)})
    for target, cases in evaluated_cases.items():
        if (
            set(cases) != all_cases_by_target[target]
            or len(cases) != len(set(cases))
        ):
            raise ProtocolError("Signed-error OOF cases are not evaluated exactly once.")
    decision_seal_hash = canonical_hash(
        {
            "schema_version": "fixed_bank_signed_error_all_fold_decisions_v1",
            "partition_hash": partition_hash,
            "decision_hashes": [row["decision_hash"] for row in decisions],
            "R_raw_and_R_safe_prediction_hashes_separate": True,
            "evaluation_labels_used": False,
        }
    )
    permutation_provenance_hash = canonical_hash(
        {
            "schema_version": "fixed_bank_signed_error_permutation_provenance_v1",
            "model_hashes": [
                fit.permutation_fit.final_model.model_hash
                for fit in model_products.target_fits
            ],
            "control_surface_hash": model_products.control_correction_surface_hash,
            "complete_sample_feature_blocks_permuted": True,
            "labels_and_gradients_preserved": True,
        }
    )
    return SignedFoldProducts(
        tuple(decisions),
        {method: tuple(sorted(rows)) for method, rows in predictions.items()},
        decision_seal_hash,
        permutation_provenance_hash,
        partition_hash,
        protocol.contract_hash,
    )


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProtocolError(f"{name} must be a canonical lowercase SHA-256 hash.")
    return value


__all__ = (
    "SignedFoldProducts",
    "SignedModelProducts",
    "SignedPartition",
    "SignedPartitionFold",
    "SignedPrelabelProducts",
    "TargetFamilyFits",
    "build_signed_fold_products",
    "build_signed_prelabel_products",
    "fit_all_target_families",
    "fit_target_families",
)
