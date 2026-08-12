"""Independent scientific replay for the multi-challenger bundle."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from ...routing.hierarchical_multi_challenger import (
    DirectionalCalibration,
    DirectionalLogitModel,
)
from ...routing.threshold_flip_case_router import (
    DirectionSharedCalibration,
    TwoHeadRidgeModel,
)
from ...runtime.artifact_io import read_json
from ...runtime.fixed_bank_a1_action_predictions import load_global_prediction_seal
from .actions import build_action_library
from .artifact_io import object_payload, read_rows
from .constants import CENTERS
from .hashing import fitted_numeric_fingerprint
from .label_capabilities import MultiChallengerLabelCapabilityManager
from .partitions import CaseIdentityRow, build_three_role_partition
from .probability_surfaces import (
    aggregate_exact_nine,
    build_prelabel_surface,
    seed_probability_rows,
)
from .science_decisions import build_fold_decision_phase
from .science_donor import fit_h_specific_donor_phase
from .science_terminal import evaluate_terminal_phase
from .semantic_payloads import (
    calibration_semantic_hash,
    router_metric_semantic_payload,
    score_semantic_payload,
    terminal_table_semantic_hash,
)


_FLOAT_ATOL = 5.0e-12
_FLOAT_RTOL = 5.0e-12

# Replay tolerance is deliberately path-scoped.  These are the only numerics
# whose last bits may change when the same frozen inputs are fitted/derived in a
# fresh process.  Everything else -- including hyperparameters, counts, ranks,
# identities, provenance, actions, reasons, and confusion products -- is exact.
_DIRECTIONAL_MODEL_FLOAT_PATHS = frozenset(
    {
        ("feature_mean", "*"),
        ("feature_scale", "*"),
        ("coefficients", "*"),
        ("covariance", "*", "*"),
    }
)
_LEGACY_MODEL_FLOAT_PATHS = frozenset(
    {
        ("feature_mean", "*"),
        ("feature_scale", "*"),
        ("tp_head", "intercept"),
        ("tp_head", "coefficients", "*"),
        ("tp_head", "covariance", "*", "*"),
        ("tp_head", "residual_variance"),
        ("tn_head", "intercept"),
        ("tn_head", "coefficients", "*"),
        ("tn_head", "covariance", "*", "*"),
        ("tn_head", "residual_variance"),
    }
)
_CALIBRATION_FLOAT_PATHS = frozenset(
    {
        ("family_calibrations", "*", "*", "offset"),
        ("family_calibrations", "*", "*", "offset_variance"),
        ("single_challenger_calibration", "gamma_0to1"),
        ("single_challenger_calibration", "gamma_1to0"),
    }
)
_SCORE_FLOAT_PATHS = frozenset(
    {
        ("expected_gain",),
        ("epistemic_standard_error",),
        ("calibration_standard_error",),
    }
)
_DECISION_FLOAT_PATHS = frozenset(
    {
        ("predicted_gain",),
        ("action_margin",),
        ("epistemic_standard_error",),
        ("calibration_standard_error",),
        ("margin_standard_error",),
        ("margin_lcb",),
    }
)

_DIRECTIONAL_MODEL_IGNORED_PATHS = frozenset({("fit_fingerprint",)})
_LEGACY_MODEL_IGNORED_PATHS = frozenset({("model_hash",)})
_CALIBRATION_IGNORED_PATHS = frozenset(
    {
        (
            "family_calibrations",
            "*",
            "*",
            "calibration_fingerprint",
        ),
        ("single_challenger_calibration", "calibration_hash"),
    }
)


def validate_scientific_surfaces(
    root: Path, *, config: object, frame: object
) -> Mapping[str, object]:
    """Rebuild every scientific phase from the persisted probability seal.

    Persisted fitted values must independently reconstruct their own
    fingerprints and remain within the declared replay tolerance. Categorical
    decisions, menu ranks, actions, reasons, topology, provenance, and terminal
    products stay exact except fitted-score correlation.
    """

    protocol = read_json(root / "manifests/protocol_manifest.json")
    action_manifest = read_json(root / "manifests/action_library.json")
    partition_manifest = read_json(root / "manifests/three_role_partition.json")
    source_lock = read_json(root / "manifests/frozen_source_stream_lock.json")
    action_count = _validate_actions(root, action_manifest)
    partition = _validate_partition(root, partition_manifest)
    prediction = load_global_prediction_seal(
        root,
        expected_config_hash=str(getattr(config, "contract_hash")),
        expected_partition_hash=partition.partition_hash,
        expected_source_lock_hash=str(source_lock["source_stream_lock_hash"]),
        expected_action_library_hash=str(action_manifest["action_library_hash"]),
        expected_target_cache_binding_hash=str(protocol["test_cache_binding_hash"]),
    )
    probabilities, prelabel, seed_count = _validate_prelabel(root, prediction)

    manager = MultiChallengerLabelCapabilityManager(
        Path(getattr(config, "test_manifest_path")),
        frame,
        partition,
        prediction_seal_hash=prediction.seal_hash,
        feature_seal_hash=prelabel.feature_surface_hash,
    )
    plans = manager.seal_all_fold_plans()
    _validate_plan_manifest(root, plans)
    donor = fit_h_specific_donor_phase(
        probability_surface=probabilities,
        prelabel=prelabel,
        partition=partition,
        manager=manager,
        config=config,
    )
    _assert_table_exact(
        root / "tables/directional_donor_responses.csv",
        donor.contribution_rows,
        role="directional donor responses",
    )
    observed_fits = _read_rows_like(
        root / "tables/model_fits.csv", donor.fit_rows
    )
    _validate_model_fit_rows(observed_fits, donor.fit_rows)
    _validate_donor_manifests(root, donor, observed_fits)

    decisions = build_fold_decision_phase(
        probability_surface=probabilities,
        prelabel=prelabel,
        partition=partition,
        manager=manager,
        donor_phase=donor,
        config=config,
    )
    _assert_table_exact(
        root / "tables/candidate_menus.csv",
        decisions.menu_rows,
        role="candidate menus and ranks",
    )
    # Validate each raw calibration fingerprint/hash before permitting any
    # cross-process tolerance in its fitted values.
    _validate_persisted_calibration_fingerprints(root, decisions.calibration_rows)
    _assert_table_derived(
        root / "tables/directional_calibrations.csv",
        decisions.calibration_rows,
        role="directional calibrations",
        allowed_float_paths=_CALIBRATION_FLOAT_PATHS,
        ignored_paths=_CALIBRATION_IGNORED_PATHS,
    )
    _validate_persisted_score_hashes(root, decisions.score_rows)
    _assert_table_derived(
        root / "tables/candidate_scores.csv",
        decisions.score_rows,
        role="candidate scores",
        allowed_float_paths=_SCORE_FLOAT_PATHS,
    )
    _assert_decision_table(
        root / "tables/method_decisions.csv",
        tuple(object_payload(row) for row in decisions.decisions),
    )
    _validate_decision_manifests(root, decisions)

    terminal_labels = manager.open_terminal_evaluation_labels()
    terminal = evaluate_terminal_phase(
        probability_surface=probabilities,
        partition=partition,
        terminal_labels=terminal_labels,
        decision_phase=decisions,
        config=config,
    )
    _validate_terminal_tables(root, terminal)
    _validate_capability_report(root, manager.report_payload())
    return {
        "action_count": action_count,
        "partition_fold_count": len(partition.folds),
        "prediction_cell_count": len(prediction.store.cells),
        "seed_probability_row_count": seed_count,
        "aggregated_probability_row_count": len(probabilities.rows),
        "case_action_feature_count": len(prelabel.features),
        "directional_donor_response_count": len(donor.contribution_rows),
        "H_specific_model_count": len(donor.model_seals),
        "candidate_menu_count": len(decisions.menu_rows),
        "directional_calibration_fold_count": len(decisions.calibration_rows),
        "method_decision_count": len(decisions.decisions),
        "terminal_case_confusion_count": len(terminal["terminal_case_confusions"]),
        "probability_surface_hash": probabilities.surface_hash,
        "feature_surface_hash": prelabel.feature_surface_hash,
        "fit_fingerprint_semantics_validated": True,
        "categorical_decisions_and_terminal_confusions_exact": True,
        "scientific_reconstruction": "PASS",
    }


def _validate_actions(root: Path, manifest: Mapping[str, object]) -> int:
    expected = tuple(build_action_library())
    expected_payloads = tuple(object_payload(row) for row in expected)
    _assert_table_exact(
        root / "tables/action_library.csv", expected_payloads, role="action library"
    )
    by_target = {
        target: [
            object_payload(action)
            for action in expected
            if action.target_center == target
        ]
        for target in CENTERS
    }
    if (
        manifest.get("actions") != list(expected_payloads)
        or manifest.get("action_count") != len(expected)
        or manifest.get("physical_actions_per_target") != 10
        or manifest.get("action_library_hash") != stable_hash(by_target)
        or manifest.get("previous_stage90_predictions_used") is not False
    ):
        raise ProtocolError("Multi-challenger action manifest drifted.")
    return len(expected)


def _validate_partition(root: Path, manifest: Mapping[str, object]):
    identities_raw = manifest.get("identities")
    if not isinstance(identities_raw, list):
        raise ProtocolError("Multi-challenger partition identities are absent.")
    identities = tuple(
        CaseIdentityRow(
            str(row["target_center"]), str(row["case_id"]), str(row["sample_id"])
        )
        for row in identities_raw
        if isinstance(row, Mapping)
    )
    rebuilt = build_three_role_partition(identities)
    if rebuilt.to_payload() != dict(manifest):
        raise ProtocolError("Multi-challenger partition is not reconstructive.")
    rows = []
    for fold in rebuilt.folds:
        for role, cases in (
            ("selection", fold.selection_case_ids),
            ("calibration", fold.calibration_case_ids),
            ("evaluation", fold.evaluation_case_ids),
        ):
            for case_id in cases:
                rows.append(
                    {
                        "target_center": fold.target_center,
                        "fold_ordinal": fold.fold_ordinal,
                        "fold_id": fold.fold_id,
                        "case_id": case_id,
                        "role": role,
                        "fold_hash": fold.fold_hash,
                        "partition_hash": rebuilt.partition_hash,
                    }
                )
    _assert_table_exact(
        root / "tables/three_role_partitions.csv", rows, role="three-role partition"
    )
    return rebuilt


def _validate_prelabel(root: Path, prediction: object):
    seeds = seed_probability_rows(prediction)
    _assert_table_exact(
        root / "tables/seed_probability_rows.csv", seeds, role="seed probabilities"
    )
    probabilities = aggregate_exact_nine(seeds)
    _assert_table_exact(
        root / "tables/aggregated_probability_rows.csv",
        probabilities.rows,
        role="exact-nine aggregates",
    )
    probability_seal = read_json(root / "manifests/sealed_probability_surface.json")
    if probability_seal != {
        "schema_version": "fixed_bank_multi_challenger_probability_seal_v1",
        "global_prediction_seal_hash": prediction.seal_hash,
        "probability_store_hash": probabilities.probability_store_hash,
        "surface_hash": probabilities.surface_hash,
        "row_count": len(probabilities.rows),
        "seed_row_count": len(seeds),
        "exact_nine_ensemble_first": True,
        "labels_used": False,
    }:
        raise ProtocolError("Multi-challenger probability seal drifted.")
    prelabel = build_prelabel_surface(
        probabilities, prediction_seal_hash=prediction.seal_hash
    )
    _assert_table_exact(
        root / "tables/case_action_features.csv",
        prelabel.features,
        role="prelabel feature surface",
    )
    feature_seal = read_json(root / "manifests/prelabel_feature_seal.json")
    if feature_seal != {
        "schema_version": "fixed_bank_multi_challenger_feature_seal_v1",
        "prediction_seal_hash": prediction.seal_hash,
        "probability_surface_hash": prelabel.probability_surface_hash,
        "feature_surface_hash": prelabel.feature_surface_hash,
        "feature_count": len(prelabel.features),
        "sealed_before_label_capabilities": True,
        "feature_hyperparameters_selected_after_labels": False,
        "raw_labels_persisted": False,
    }:
        raise ProtocolError("Multi-challenger prelabel feature seal drifted.")
    return probabilities, prelabel, len(seeds)


def _validate_plan_manifest(root: Path, plans: Sequence[object]) -> None:
    payloads = [object_payload(plan) for plan in plans]
    unhashed = {
        "schema_version": "fixed_bank_multi_challenger_fold_plan_seals_v1",
        "plans": payloads,
        "plan_count": len(payloads),
        "held_evaluation_labels_used": False,
        "each_plan_invariant_to_held_evaluation_label_values": True,
    }
    expected = {**unhashed, "fold_plan_surface_hash": _canonical_hash(unhashed)}
    if read_json(root / "manifests/fold_plan_seals.json") != expected:
        raise ProtocolError("Multi-challenger fold plan seals drifted.")


def _validate_model_fit_rows(
    observed: Sequence[Mapping[str, object]],
    expected: Sequence[Mapping[str, object]],
) -> None:
    if len(observed) != len(expected):
        raise ProtocolError("Multi-challenger model fit row count drifted.")
    for actual, replayed in zip(observed, expected, strict=True):
        if (
            actual.get("heldout_target_H") != replayed.get("heldout_target_H")
            or actual.get("permutation_row_surface_hash")
            != replayed.get("permutation_row_surface_hash")
        ):
            raise ProtocolError("Multi-challenger model fit topology drifted.")
        actual_families = actual.get("families")
        expected_families = replayed.get("families")
        if not isinstance(actual_families, Mapping) or not isinstance(
            expected_families, Mapping
        ):
            raise ProtocolError("Multi-challenger model families are malformed.")
        if set(actual_families) != {"G", "R", "P"}:
            raise ProtocolError("Multi-challenger model family coverage drifted.")
        for family in ("G", "R", "P"):
            actual_directions = actual_families[family]
            replayed_directions = expected_families[family]
            if not isinstance(actual_directions, Mapping) or not isinstance(
                replayed_directions, Mapping
            ):
                raise ProtocolError("Multi-challenger direction models malformed.")
            for direction in ("0to1", "1to0"):
                actual_model = _directional_model(actual_directions[direction])
                replayed_model = _directional_model(replayed_directions[direction])
                if actual_model.provenance_hash != replayed_model.provenance_hash:
                    raise ProtocolError(
                        "Multi-challenger fitted-model provenance drifted."
                    )
                _assert_semantic_equal(
                    actual_model.to_payload(),
                    replayed_model.to_payload(),
                    role="directional model",
                    allowed_float_paths=_DIRECTIONAL_MODEL_FLOAT_PATHS,
                    ignored_paths=_DIRECTIONAL_MODEL_IGNORED_PATHS,
                )
        actual_single = actual.get("single_challenger_model")
        replayed_single = replayed.get("single_challenger_model")
        if not isinstance(actual_single, Mapping) or not isinstance(
            replayed_single, Mapping
        ):
            raise ProtocolError("Multi-challenger single model is malformed.")
        # Each payload validates its own raw hash.  Cross-process equality is
        # tolerance-aware because this legacy control predates fit_fingerprint.
        TwoHeadRidgeModel.from_payload(actual_single)
        TwoHeadRidgeModel.from_payload(replayed_single)
        _assert_semantic_equal(
            actual_single,
            replayed_single,
            role="legacy single model",
            allowed_float_paths=_LEGACY_MODEL_FLOAT_PATHS,
            ignored_paths=_LEGACY_MODEL_IGNORED_PATHS,
        )


def _directional_model(value: object) -> DirectionalLogitModel:
    if not isinstance(value, Mapping):
        raise ProtocolError("Directional model payload is not a mapping.")
    model = DirectionalLogitModel(
        model_target=str(value["model_target"]),
        family=str(value["family"]),
        direction=str(value["direction"]),
        feature_names=tuple(str(item) for item in value["feature_names"]),
        feature_mean=tuple(float(item) for item in value["feature_mean"]),
        feature_scale=tuple(float(item) for item in value["feature_scale"]),
        candidate_sources=tuple(str(item) for item in value["candidate_sources"]),
        query_centers=tuple(str(item) for item in value["query_centers"]),
        coefficients=tuple(float(item) for item in value["coefficients"]),
        covariance=tuple(
            tuple(float(item) for item in row) for row in value["covariance"]
        ),
        feature_alpha=float(value["feature_alpha"]),
        source_alpha=float(value["source_alpha"]),
        query_alpha=float(value["query_alpha"]),
        intercept_alpha=float(value["intercept_alpha"]),
        training_row_count=int(value["training_row_count"]),
        training_trial_count=int(value["training_trial_count"]),
        training_case_clusters=tuple(
            str(item) for item in value["training_case_clusters"]
        ),
        provenance_hash=str(value["provenance_hash"]),
        fit_fingerprint=str(value["fit_fingerprint"]),
    )
    # Explicitly recompute the fitted payload fingerprint rather than trusting
    # the dataclass constructor as the only validation boundary.
    fit_payload = {
        key: value[key]
        for key in (
            "schema_version",
            "model_target",
            "family",
            "direction",
            "feature_names",
            "feature_mean",
            "feature_scale",
            "candidate_sources",
            "query_centers",
            "coefficients",
            "covariance",
            "feature_alpha",
            "source_alpha",
            "query_alpha",
            "intercept_alpha",
            "training_row_count",
            "training_trial_count",
        )
    }
    if fitted_numeric_fingerprint(fit_payload) != model.fit_fingerprint:
        raise ProtocolError("Directional fit fingerprint did not reconstruct.")
    return model


def _validate_donor_manifests(
    root: Path,
    replayed: object,
    persisted_fit_rows: Sequence[Mapping[str, object]],
) -> None:
    persisted = read_json(root / "manifests/donor_model_seals.json")
    if (
        persisted.get("schema_version")
        != "fixed_bank_multi_challenger_donor_model_seals_v1"
        or persisted.get("model_count") != 9
        or persisted.get("models_are_H_specific") is not True
        or persisted.get("strict_H_q_e_exclusion") is not True
        or persisted.get("heldout_H_labels_used") is not False
    ):
        raise ProtocolError("Multi-challenger donor seal header drifted.")
    observed = persisted.get("models")
    if not isinstance(observed, Mapping) or set(observed) != set(CENTERS):
        raise ProtocolError("Multi-challenger donor seal coverage drifted.")
    fit_by_target = {
        str(row["heldout_target_H"]): row for row in persisted_fit_rows
    }
    for target in CENTERS:
        actual = observed[target]
        expected = replayed.model_seals[target]
        if not isinstance(actual, Mapping):
            raise ProtocolError("Multi-challenger donor seal is malformed.")
        fit = fit_by_target[target]
        families = fit["families"]
        single = fit["single_challenger_model"]
        if not isinstance(families, Mapping) or not isinstance(single, Mapping):
            raise ProtocolError("Persisted donor fit payload is malformed.")
        actual_fingerprints = {
            family: {
                direction: families[family][direction]["fit_fingerprint"]
                for direction in ("0to1", "1to0")
            }
            for family in ("G", "R", "P")
        }
        fit_contracts = {
            family: {
                direction: _directional_semantic_contract(
                    _directional_model(families[family][direction])
                )
                for direction in ("0to1", "1to0")
            }
            for family in ("G", "R", "P")
        }
        single_fit_contract = _single_semantic_contract(single)
        composite = _canonical_hash(
            {
                "schema_version": "fixed_bank_multi_challenger_H_models_v1",
                "heldout_target_H": target,
                "fit_contracts": fit_contracts,
                "single_challenger_fit_contract": single_fit_contract,
                "strict_H_q_e_exclusion": True,
                "fitted_numeric_validation": (
                    "raw_values_persisted_and_replayed_with_"
                    "isclose_atol_5e-12_rtol_5e-12"
                ),
            }
        )
        unhashed = {
            key: value
            for key, value in actual.items()
            if key
            not in {
                "fit_fingerprints",
                "single_challenger_model_hash",
                "seal_hash",
            }
        }
        if (
            actual.get("fit_fingerprints") != actual_fingerprints
            or actual.get("single_challenger_model_hash") != single["model_hash"]
            or actual.get("fit_contracts") != fit_contracts
            or actual.get("single_challenger_fit_contract")
            != single_fit_contract
            or actual.get("composite_model_hash") != composite
            or actual.get("seal_hash") != _canonical_hash(unhashed)
        ):
            raise ProtocolError("Persisted donor fitted-numeric seal is inconsistent.")
        for key in (
            "schema_version",
            "heldout_target_H",
            "composite_model_hash",
            "composite_provenance_hash",
            "fit_contracts",
            "single_challenger_fit_contract",
            "fitted_numeric_validation",
            "permutation_row_surface_hash",
            "strict_H_q_e_exclusion",
            "heldout_H_labels_used",
            "seal_hash",
        ):
            if actual.get(key) != expected.get(key):
                raise ProtocolError("Multi-challenger donor seal provenance drifted.")
    if (
        read_json(root / "manifests/permutation_provenance_seal.json")
        != dict(replayed.permutation_provenance)
    ):
        raise ProtocolError("Multi-challenger permutation provenance drifted.")


def _directional_semantic_contract(model: DirectionalLogitModel) -> Mapping[str, object]:
    return {
        "schema_version": "hierarchical_directional_logit_fit_v2",
        "model_target": model.model_target,
        "family": model.family,
        "direction": model.direction,
        "feature_names": list(model.feature_names),
        "candidate_sources": list(model.candidate_sources),
        "query_centers": list(model.query_centers),
        "feature_alpha": model.feature_alpha,
        "source_alpha": model.source_alpha,
        "query_alpha": model.query_alpha,
        "intercept_alpha": model.intercept_alpha,
        "training_row_count": model.training_row_count,
        "training_trial_count": model.training_trial_count,
        "training_case_clusters": list(model.training_case_clusters),
        "provenance_hash": model.provenance_hash,
        "fitted_numeric_fields": [
            "feature_mean",
            "feature_scale",
            "coefficients",
            "covariance",
        ],
        "fitted_numeric_validation": "replay_isclose_atol_5e-12_rtol_5e-12",
    }


def _single_semantic_contract(payload: Mapping[str, object]) -> Mapping[str, object]:
    return {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "feature_mean",
            "feature_scale",
            "tp_head",
            "tn_head",
            "model_hash",
        }
    } | {
        "fitted_numeric_fields": [
            "feature_mean",
            "feature_scale",
            "tp_head",
            "tn_head",
        ],
        "fitted_numeric_validation": "replay_isclose_atol_5e-12_rtol_5e-12",
    }


def _validate_decision_manifests(root: Path, replayed: object) -> None:
    menu_rows = [dict(row) for row in replayed.menu_rows]
    menu_unhashed = {
        "schema_version": "fixed_bank_multi_challenger_candidate_menu_seals_v1",
        "menu_count": len(menu_rows),
        "menu_row_hashes": [str(row["row_hash"]) for row in menu_rows],
        "selection_labels_used_only_for_fixed_B_ranked_menus": True,
        "held_evaluation_labels_used": False,
    }
    if read_json(root / "manifests/candidate_menu_seals.json") != {
        **menu_unhashed,
        "menu_surface_hash": _canonical_hash(menu_unhashed),
    }:
        raise ProtocolError("Multi-challenger candidate-menu seals drifted.")
    calibration = read_json(root / "manifests/calibration_seals.json")
    replayed_calibration_unhashed = {
        "schema_version": "fixed_bank_multi_challenger_calibration_seals_v1",
        "calibration_count": len(replayed.calibration_rows),
        "calibration_row_hashes": [
            str(row["row_hash"]) for row in replayed.calibration_rows
        ],
        "menu_bound": True,
        "shared_model_updated": False,
        "held_evaluation_labels_used": False,
    }
    if (
        calibration
        != {
            **replayed_calibration_unhashed,
            "calibration_surface_hash": _canonical_hash(
                replayed_calibration_unhashed
            ),
        }
    ):
        raise ProtocolError("Multi-challenger calibration semantic seals drifted.")
    decision = read_json(root / "manifests/all_method_decisions_seal.json")
    expected_fold_seals = {
        f"{key[0]}::{key[1]}": value
        for key, value in sorted(replayed.fold_seal_hashes.items())
    }
    persisted_decisions = _read_rows_like(
        root / "tables/method_decisions.csv",
        tuple(object_payload(row) for row in replayed.decisions),
    )
    persisted_bundle_payload = {
        "schema_version": "fixed_bank_multi_challenger_decision_bundle_v1",
        "decisions": [
            _semantic_decision_mapping(row) for row in persisted_decisions
        ],
        "fold_seals": decision.get("fold_seals"),
        "evaluation_labels_used": False,
    }
    if (
        decision.get("schema_version")
        != "fixed_bank_multi_challenger_all_decisions_v1"
        or decision.get("decision_count") != len(replayed.decisions)
        or decision.get("fold_seal_count") != 45
        or decision.get("fold_seals") != expected_fold_seals
        or decision.get("decision_bundle_hash") != replayed.decision_bundle_hash
        or decision.get("decision_bundle_hash")
        != _canonical_hash(persisted_bundle_payload)
        or decision.get("each_fold_decision_without_its_held_evaluation_labels")
        is not True
        or decision.get("terminal_evaluation_labels_used") is not False
    ):
        raise ProtocolError("Multi-challenger decision seal header drifted.")


def _validate_persisted_calibration_fingerprints(
    root: Path, replayed_rows: Sequence[Mapping[str, object]]
) -> None:
    observed = _read_rows_like(
        root / "tables/directional_calibrations.csv", replayed_rows
    )
    for row in observed:
        families = row.get("family_calibrations")
        single = row.get("single_challenger_calibration")
        if (
            not isinstance(families, Mapping)
            or set(families) != {"G", "R", "P"}
            or not isinstance(single, Mapping)
        ):
            raise ProtocolError("Persisted calibration payload is malformed.")
        for family in ("G", "R", "P"):
            by_direction = families[family]
            if (
                not isinstance(by_direction, Mapping)
                or set(by_direction) != {"0to1", "1to0"}
            ):
                raise ProtocolError("Persisted family calibration is malformed.")
            for direction in ("0to1", "1to0"):
                payload = by_direction[direction]
                if not isinstance(payload, Mapping):
                    raise ProtocolError("Persisted direction calibration is malformed.")
                DirectionalCalibration(
                    direction=str(payload["direction"]),
                    offset=float(payload["offset"]),
                    offset_variance=float(payload["offset_variance"]),
                    success_count=int(payload["success_count"]),
                    trial_count=int(payload["trial_count"]),
                    row_count=int(payload["row_count"]),
                    case_count=int(payload["case_count"]),
                    alpha=float(payload["alpha"]),
                    menu_hash=str(payload["menu_hash"]),
                    valid=bool(payload["valid"]),
                    calibration_fingerprint=str(
                        payload["calibration_fingerprint"]
                    ),
                )
        DirectionSharedCalibration.from_payload(single)
        unhashed = {key: value for key, value in row.items() if key != "row_hash"}
        if row.get("row_hash") != calibration_semantic_hash(unhashed):
            raise ProtocolError("Persisted calibration semantic hash drifted.")


def _validate_persisted_score_hashes(
    root: Path, replayed_rows: Sequence[Mapping[str, object]]
) -> None:
    observed = _read_rows_like(root / "tables/candidate_scores.csv", replayed_rows)
    for row in observed:
        if row.get("row_hash") != _canonical_hash(score_semantic_payload(row)):
            raise ProtocolError("Persisted candidate-score semantic hash drifted.")


def _semantic_decision_mapping(row: Mapping[str, object]) -> Mapping[str, object]:
    payload = dict(row)
    numeric_fields = (
        "predicted_gain",
        "action_margin",
        "epistemic_standard_error",
        "calibration_standard_error",
        "margin_standard_error",
        "margin_lcb",
    )
    for field in numeric_fields:
        payload.pop(field)
    return {
        **payload,
        "fitted_numeric_fields": list(numeric_fields),
        "fitted_numeric_validation": "replay_isclose_atol_5e-12_rtol_5e-12",
    }


def _validate_terminal_tables(root: Path, replayed: Mapping[str, object]) -> None:
    members = {
        "terminal_case_confusions": "terminal_case_confusions.csv",
        "terminal_center_metrics": "terminal_center_metrics.csv",
        "terminal_contrasts": "terminal_contrasts.csv",
        "router_identification_metrics": "router_identification_metrics.csv",
        "permutation_metrics": "permutation_metrics.csv",
        "menu_oracle_metrics": "menu_oracle_metrics.csv",
    }
    persisted_tables: dict[str, tuple[dict[str, object], ...]] = {}
    for key, member in members.items():
        rows = replayed[key]
        if not isinstance(rows, Sequence):
            raise ProtocolError(f"Multi-challenger replay table malformed: {key}.")
        payloads = tuple(object_payload(row) for row in rows)
        persisted_tables[key] = _read_rows_like(
            root / "tables" / member, payloads
        )
        for persisted_row in persisted_tables[key]:
            unhashed = {
                field: value
                for field, value in persisted_row.items()
                if field != "row_hash"
            }
            hash_payload = (
                router_metric_semantic_payload(unhashed)
                if key == "router_identification_metrics"
                else unhashed
            )
            if persisted_row.get("row_hash") != _canonical_hash(hash_payload):
                raise ProtocolError(
                    f"Persisted multi-challenger {key} row hash drifted."
                )
        if key != "router_identification_metrics":
            if persisted_tables[key] != payloads:
                raise ProtocolError(
                    f"Multi-challenger exact terminal table drifted: {key}."
                )
        else:
            for actual, expected_row in zip(
                persisted_tables[key], payloads, strict=True
            ):
                actual_exact = dict(actual)
                expected_exact = dict(expected_row)
                actual_spearman = actual_exact.pop("spearman")
                expected_spearman = expected_exact.pop("spearman")
                if actual_exact != expected_exact:
                    raise ProtocolError(
                        "Multi-challenger router identification topology or "
                        "exact action metrics drifted."
                    )
                _assert_semantic_equal(
                    actual_spearman,
                    expected_spearman,
                    role="router identification fitted-score Spearman",
                    allowed_float_paths=frozenset({()}),
                )
    persisted = read_json(root / "manifests/sealed_terminal_evaluation.json")
    expected = replayed["sealed_terminal_evaluation"]
    if not isinstance(expected, Mapping):
        raise ProtocolError("Multi-challenger replay terminal seal malformed.")
    table_hashes = {
        key: terminal_table_semantic_hash(key, rows)
        for key, rows in persisted_tables.items()
    }
    terminal_unhashed = {
        key: value for key, value in persisted.items() if key != "sealed_result_hash"
    }
    if (
        persisted.get("table_hashes") != table_hashes
        or persisted.get("sealed_result_hash") != _canonical_hash(terminal_unhashed)
    ):
        raise ProtocolError("Persisted multi-challenger terminal seal is inconsistent.")
    for key in (
        "schema_version",
        "decision_bundle_hash",
        "terminal_label_identity_hash",
        "diagnostic_routing_gate",
        "terminal_scoring_after_all_45_decision_seals",
        "terminal_oracles_used_for_decisions",
        "raw_labels_persisted",
        "per_case_bacc_persisted",
        "consumed_test_diagnostic_only",
    ):
        if persisted.get(key) != expected.get(key):
            raise ProtocolError("Multi-challenger terminal seal drifted.")


def _validate_capability_report(root: Path, replayed: Mapping[str, object]) -> None:
    persisted = read_json(root / "reports/label_capability_report.json")
    exact_keys = (
        "schema_version",
        "status",
        "experiment_role",
        "manifest_sha256",
        "prediction_seal_hash",
        "feature_seal_hash",
        "fold_plan_count",
        "loco_target_count",
        "H_specific_composite_model_seal_count",
        "H_specific_composite_model_seals",
        "selection_capability_count",
        "calibration_capability_count",
        "fold_decision_seal_count",
        "terminal_scoring_opened",
        "every_nonterminal_access_excludes_its_own_evaluation_cases",
        "all_nine_composite_models_sealed_before_target_support",
        "terminal_open_after_all_45_fold_seals",
        "held_evaluation_label_mutation_can_affect_only_terminal_products",
        "events",
        "raw_labels_persisted",
    )
    if any(persisted.get(key) != replayed.get(key) for key in exact_keys):
        raise ProtocolError("Multi-challenger label-capability replay drifted.")


def _assert_table_exact(
    path: Path, expected_rows: Sequence[object], *, role: str
) -> None:
    payloads = tuple(object_payload(row) for row in expected_rows)
    observed = _read_rows_like(path, payloads)
    if observed != payloads:
        raise ProtocolError(f"Multi-challenger {role} table drifted.")


def _assert_table_derived(
    path: Path,
    expected_rows: Sequence[object],
    *,
    role: str,
    allowed_float_paths: frozenset[tuple[str, ...]],
    ignored_paths: frozenset[tuple[str, ...]] = frozenset(),
) -> None:
    payloads = tuple(object_payload(row) for row in expected_rows)
    observed = _read_rows_like(path, payloads)
    if len(observed) != len(payloads):
        raise ProtocolError(f"Multi-challenger {role} row count drifted.")
    for actual, expected in zip(observed, payloads, strict=True):
        _assert_semantic_equal(
            actual,
            expected,
            role=role,
            allowed_float_paths=allowed_float_paths,
            ignored_paths=ignored_paths,
        )


def _assert_decision_table(
    path: Path, expected_rows: Sequence[Mapping[str, object]]
) -> None:
    """Permit fit-derived drift only for learned-router decision rows.

    B, U, and S_static contain no fitted numerics; their zeroes and exact
    support gain therefore remain byte-semantic values even though the learned
    controls and multi-challenger routers use the same table schema.
    """

    payloads = tuple(object_payload(row) for row in expected_rows)
    observed = _read_rows_like(path, payloads)
    if len(observed) != len(payloads):
        raise ProtocolError("Multi-challenger method decision row count drifted.")
    learned = {"F_single", "G_multi", "R_multi", "P_multi"}
    for actual, expected in zip(observed, payloads, strict=True):
        method_id = expected.get("method_id")
        _assert_semantic_equal(
            actual,
            expected,
            role="method decisions",
            allowed_float_paths=(
                _DECISION_FLOAT_PATHS
                if method_id in learned
                else frozenset()
            ),
        )


def _read_rows_like(
    path: Path, expected_rows: Sequence[Mapping[str, object]]
) -> tuple[dict[str, object], ...]:
    observed = read_rows(path)
    if len(observed) != len(expected_rows):
        raise ProtocolError(f"Multi-challenger table row count drifted: {path}.")
    parsed = []
    for raw, expected in zip(observed, expected_rows, strict=True):
        if tuple(raw) != tuple(expected):
            raise ProtocolError(f"Multi-challenger table schema drifted: {path}.")
        parsed.append(
            {key: _parse_like(raw[key], expected[key]) for key in expected}
        )
    return tuple(parsed)


def _parse_like(raw: str, expected: object) -> object:
    if isinstance(expected, bool):
        if raw not in {"True", "False", "true", "false"}:
            raise ProtocolError("Multi-challenger boolean CSV value is malformed.")
        return raw.casefold() == "true"
    if expected is None:
        return None if raw == "" else json.loads(raw)
    if isinstance(expected, int) and not isinstance(expected, bool):
        return int(raw)
    if isinstance(expected, float):
        return float(raw)
    if isinstance(expected, (list, dict)):
        return json.loads(raw)
    return raw


def _assert_semantic_equal(
    actual: object,
    expected: object,
    *,
    role: str,
    allowed_float_paths: frozenset[tuple[str, ...]],
    ignored_paths: frozenset[tuple[str, ...]] = frozenset(),
    path: tuple[str, ...] = (),
) -> None:
    """Compare a replay payload under an explicit fitted-numeric policy.

    A ``*`` component in a declared path matches exactly one mapping key or
    sequence index.  Hash fields may be ignored only at declared paths and only
    after their raw payload has independently reconstructed that hash.
    """

    if _path_is_declared(path, ignored_paths):
        return
    if isinstance(expected, float):
        if _path_is_declared(path, allowed_float_paths):
            if (
                isinstance(actual, bool)
                or not isinstance(actual, (float, int))
                or not math.isclose(
                    float(actual), expected, rel_tol=_FLOAT_RTOL, abs_tol=_FLOAT_ATOL
                )
            ):
                raise ProtocolError(
                    f"Multi-challenger {role} fitted/derived numeric drifted."
                )
        elif type(actual) is not float or actual != expected:
            raise ProtocolError(
                f"Multi-challenger {role} unallowlisted numeric drifted."
            )
        return
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise ProtocolError(f"Multi-challenger {role} structure drifted.")
        for nested_key in expected:
            _assert_semantic_equal(
                actual[nested_key],
                expected[nested_key],
                role=role,
                allowed_float_paths=allowed_float_paths,
                ignored_paths=ignored_paths,
                path=(*path, str(nested_key)),
            )
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise ProtocolError(f"Multi-challenger {role} sequence drifted.")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _assert_semantic_equal(
                left,
                right,
                role=role,
                allowed_float_paths=allowed_float_paths,
                ignored_paths=ignored_paths,
                path=(*path, str(index)),
            )
        return
    if type(actual) is not type(expected) or actual != expected:
        raise ProtocolError(f"Multi-challenger {role} categorical value drifted.")


def _path_is_declared(
    path: tuple[str, ...], declarations: frozenset[tuple[str, ...]]
) -> bool:
    return any(
        len(path) == len(declaration)
        and all(
            declared == "*" or declared == actual
            for actual, declared in zip(path, declaration, strict=True)
        )
        for declaration in declarations
    )


def _canonical_hash(value: object) -> str:
    from .hashing import canonical_hash

    return canonical_hash(value)


__all__ = ("validate_scientific_surfaces",)
