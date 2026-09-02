from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v7.execution.bindings import (
    reconstruct_frozen_routes_for_evaluation,
)
from midogpp_thesis.cvae.routing.harp_protocol import canonical_hash
from midogpp_thesis.cvae.runtime.harp_v7_execution import validation
from midogpp_thesis.cvae.runtime.harp_v7_execution.contracts import (
    ActionKind,
    ArtifactValue,
    FrozenRouteReceipt,
    PrelabelRouteSet,
    RoutedCase,
)
from midogpp_thesis.cvae.runtime.artifact_io import atomic_json, sha256_file
from midogpp_thesis.cvae.runtime.harp_v7_execution.stores import (
    read_prelabel_routes,
    write_prelabel_routes,
)
from midogpp_thesis.cvae.runtime.harp_v7_execution.terminal import (
    _validate_frozen_route_receipt,
    evaluate_terminal_routes,
)


def _numeric_replay_artifacts() -> tuple[ArtifactValue, ArtifactValue]:
    model = ArtifactValue(
        state=None,
        manifest={
            "outer_models": [
                {
                    "outer_target_id": "H",
                    "numeric_oof": {
                        "rows": [
                            {
                                "query_center_id": "C",
                                "case_id": "case-1",
                                "opportunity_probability": 0.6,
                                "rank_margin": 0.01,
                                "action_scores": [{"score": 0.25}],
                            }
                        ]
                    }
                }
            ]
        },
        arrays={
            "oof_case_values": np.asarray([[0.6, 0.01, 1.0]], dtype=np.float64),
            "oof_action_scores": np.asarray([[0.25]], dtype=np.float64),
            "oof_action_score_offsets": np.asarray([0, 1], dtype=np.int64),
        },
    )
    row = {
        "outer_target_id": "H",
        "query_center_id": "C",
        "case_id": "case-1",
        "opportunity_probability": 0.6,
        "rank_margin": 0.01,
        "selected_action_id": "B",
        "observed_bacc_gain": 0.0,
        "observed_brier_delta": 0.0,
        "observed_log_delta": 0.0,
        "best_observed_bacc_gain": 0.1,
        "regret": 0.1,
        "nested_selected_action_id": "U:D01",
        "nested_observed_bacc_gain": 0.05,
        "nested_observed_brier_delta": -0.01,
        "nested_observed_log_delta": -0.02,
        "nested_regret": 0.05,
        "nested_opportunity_threshold": 0.5,
        "nested_rank_margin_threshold": 0.0,
        "nested_threshold_training_center_ids": ["A", "B"],
        "nested_policy_fold_hash": "a" * 64,
        "nested_policy_replay_hash": "b" * 64,
    }
    admission = ArtifactValue(
        state=None,
        manifest={
            "source_policy_oof_rows": [row],
            "source_policy_oof_case_count": 1,
            "nested_held_source_threshold_policy_replayed": True,
        },
        arrays={
            "source_policy_oof_values": np.asarray(
                [[0.6, 0.01, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1]],
                dtype=np.float64,
            ),
            "nested_source_policy_oof_values": np.asarray(
                [[0.6, 0.01, 1.0, 0.05, -0.01, -0.02, 0.1, 0.05]],
                dtype=np.float64,
            ),
        },
    )
    return model, admission


def test_numeric_oof_replay_requires_deployed_and_nested_values_and_provenance() -> None:
    model, admission = _numeric_replay_artifacts()
    validation._validate_numeric_oof(model, admission)

    changed_model_values = np.asarray(model.arrays["oof_case_values"]).copy()
    changed_model_values[0, 0] = 0.7
    tampered_model = ArtifactValue(
        state=None,
        manifest=model.manifest,
        arrays={**dict(model.arrays), "oof_case_values": changed_model_values},
    )
    with pytest.raises(ProtocolError, match="source-OOF row/value"):
        validation._validate_numeric_oof(tampered_model, admission)

    changed_heads = {
        name: np.asarray(value).copy() for name, value in admission.arrays.items()
    }
    changed_heads["source_policy_oof_values"][0, 0] = 0.7
    changed_heads["nested_source_policy_oof_values"][0, 0] = 0.7
    tampered_heads = ArtifactValue(
        state=None,
        manifest=admission.manifest,
        arrays=changed_heads,
    )
    with pytest.raises(ProtocolError, match="policy/model OOF row binding"):
        validation._validate_numeric_oof(model, tampered_heads)

    missing_nested = ArtifactValue(
        state=None,
        manifest=admission.manifest,
        arrays={
            "source_policy_oof_values": admission.arrays[
                "source_policy_oof_values"
            ]
        },
    )
    with pytest.raises(ProtocolError, match="numeric OOF replay"):
        validation._validate_numeric_oof(model, missing_nested)

    changed = np.asarray(
        admission.arrays["nested_source_policy_oof_values"], dtype=np.float64
    ).copy()
    changed[0, 3] += 0.01
    tampered_nested = ArtifactValue(
        state=None,
        manifest=admission.manifest,
        arrays={
            "source_policy_oof_values": admission.arrays[
                "source_policy_oof_values"
            ],
            "nested_source_policy_oof_values": changed,
        },
    )
    with pytest.raises(ProtocolError, match="nested policy OOF row/value"):
        validation._validate_numeric_oof(model, tampered_nested)

    bad_row = dict(admission.manifest["source_policy_oof_rows"][0])
    bad_row["nested_threshold_training_center_ids"] = ["A", "C"]
    bad_provenance = ArtifactValue(
        state=None,
        manifest={
            **dict(admission.manifest),
            "source_policy_oof_rows": [bad_row],
        },
        arrays=admission.arrays,
    )
    with pytest.raises(ProtocolError, match="threshold provenance"):
        validation._validate_numeric_oof(model, bad_provenance)


def test_exact_top1_route_store_roundtrip_and_legacy_mixture_rejection(
    tmp_path: Path,
) -> None:
    baseline = np.asarray([0.2, 0.8], dtype=np.float32)
    selected = np.asarray([0.7, 0.8], dtype=np.float32)
    routed = RoutedCase(
        outer_target_id="H",
        case_id="case-1",
        sample_ids=("sample-1", "sample-2"),
        selected_kind=ActionKind.HXE,
        selected_source_id="E",
        reason="ROUTED_EXACT_TOP1",
        baseline_probabilities=baseline,
        uniform_probabilities=baseline,
        selected_probabilities=selected,
        routed_probabilities=selected,
        direction="D01",
        shrinkage=1.0,
        component_action_ids=("HXE:E:D01",),
        component_weights=(1.0,),
        component_probabilities=(selected,),
    )
    abstained = RoutedCase(
        outer_target_id="H",
        case_id="case-2",
        sample_ids=("sample-3", "sample-4"),
        selected_kind=ActionKind.B,
        selected_source_id=None,
        reason="EXACT_B_POLICY_ABSTENTION",
        baseline_probabilities=baseline,
        uniform_probabilities=baseline,
        selected_probabilities=baseline,
        routed_probabilities=baseline,
    )
    routes = PrelabelRouteSet(
        cases=(routed, abstained),
        policy_hash="a" * 64,
        model_hash="b" * 64,
        target_action_hash="c" * 64,
    )
    route_root = tmp_path / "stores/prelabel_routes"
    write_prelabel_routes(route_root, routes)
    restored = read_prelabel_routes(route_root)
    assert restored.route_hash == routes.route_hash
    assert restored.cases[0].component_weights == (1.0,)
    assert (
        restored.cases[1].routed_probabilities.tobytes(order="C")
        == baseline.tobytes(order="C")
    )

    config_hash = "d" * 64
    validations = []
    for validator_id, process_id in (("validator-a", 101), ("validator-b", 202)):
        body = {"validator_id": validator_id, "process_id": process_id}
        validations.append({**body, "validation_hash": canonical_hash(body)})
    validation_body = {
        "config_hash": config_hash,
        "expected_center_ids": ["H"],
        "validations": validations,
        "distinct_process_ids": True,
        "evaluation_labels_opened": False,
    }
    validation_bundle_hash = canonical_hash(validation_body)
    validation_payload = {
        **validation_body,
        "bundle_hash": validation_bundle_hash,
    }
    prelabel_body = {
        "route_hash": routes.route_hash,
        "policy_hash": routes.policy_hash,
        "model_hash": routes.model_hash,
        "target_action_hash": routes.target_action_hash,
        "route_store_manifest_sha256": sha256_file(route_root / "manifest.json"),
        "route_store_npz_sha256": sha256_file(route_root / "arrays.npz"),
        "evaluation_labels_opened": False,
    }
    prelabel_bundle_hash = canonical_hash(prelabel_body)
    prelabel_payload = {**prelabel_body, "bundle_hash": prelabel_bundle_hash}
    validation_hashes = tuple(
        row["validation_hash"] for row in validations
    )
    frozen_body = {
        "schema_version": "midogpp_harp_v7_frozen_route_seal_v1",
        "status": "FROZEN_AFTER_TWO_FRESH_RECONSTRUCTIONS",
        "prelabel_bundle_hash": prelabel_bundle_hash,
        "config_hash": config_hash,
        "expected_center_ids": ["H"],
        "route_hash": routes.route_hash,
        "policy_hash": routes.policy_hash,
        "model_hash": routes.model_hash,
        "target_action_hash": routes.target_action_hash,
        "validation_bundle_hash": validation_bundle_hash,
        "independent_validation_hashes": list(validation_hashes),
        "case_count": len(routes.cases),
        "evaluation_labels_opened": False,
    }
    frozen = {**frozen_body, "seal_hash": canonical_hash(frozen_body)}
    manifests = tmp_path / "manifests"
    manifests.mkdir(parents=True)
    atomic_json(manifests / "prelabel_route_bundle.json", prelabel_payload)
    atomic_json(manifests / "fresh_validations.json", validation_payload)
    atomic_json(manifests / "frozen_route_seal.json", frozen)
    reconstructed, receipt = reconstruct_frozen_routes_for_evaluation(
        route_root,
        frozen=frozen,
        model_hash=routes.model_hash,
        target_action_hash=routes.target_action_hash,
        centers=("H",),
        config_hash=config_hash,
    )
    assert reconstructed.route_hash == routes.route_hash
    assert receipt.route_hash == routes.route_hash
    _validate_frozen_route_receipt(
        reconstructed,
        receipt,
        menus=(SimpleNamespace(outer_target_id="H"),),
        artifact_root=tmp_path,
        config_hash=config_hash,
    )
    forged = FrozenRouteReceipt(
        seal_hash=receipt.seal_hash,
        config_hash=receipt.config_hash,
        route_hash=receipt.route_hash,
        policy_hash=receipt.policy_hash,
        model_hash=receipt.model_hash,
        target_action_hash=receipt.target_action_hash,
        validation_bundle_hash=receipt.validation_bundle_hash,
        independent_validation_hashes=receipt.independent_validation_hashes,
        expected_center_ids=receipt.expected_center_ids,
        case_count=receipt.case_count,
    )
    with pytest.raises(ProtocolError, match="durable frozen-route store"):
        _validate_frozen_route_receipt(
            reconstructed,
            forged,
            menus=(SimpleNamespace(outer_target_id="H"),),
            artifact_root=tmp_path / "unregistered",
            config_hash=config_hash,
        )
    with pytest.raises(ProtocolError, match="frozen-route receipt"):
        evaluate_terminal_routes(
            routes,
            {},
            menus=(SimpleNamespace(outer_target_id="H"),),
            frozen_receipt=None,  # type: ignore[arg-type]
            artifact_root=tmp_path,
            config_hash=config_hash,
        )

    with pytest.raises(ProtocolError, match="exact-top-1"):
        RoutedCase(
            outer_target_id="H",
            case_id="case-legacy",
            sample_ids=("sample-1", "sample-2"),
            selected_kind=ActionKind.HXE,
            selected_source_id="E",
            reason="LEGACY_MIXTURE",
            baseline_probabilities=baseline,
            uniform_probabilities=baseline,
            selected_probabilities=selected,
            routed_probabilities=selected,
            direction="D01",
            shrinkage=1.0,
            component_action_ids=("HXE:E:D01", "HXE:F:D01"),
            component_weights=(0.5, 0.5),
            component_probabilities=(selected, selected),
        )


def _effective_menu_artifacts() -> tuple[ArtifactValue, ArtifactValue, object]:
    compatibility_hash = "c" * 64
    menu_hash = "d" * 64
    raw_menus = [{"menu_hash": menu_hash}]
    raw_actions = [{}]
    arrays = {
        "effective_action_features": np.asarray([[1.0]], dtype=np.float64),
        "effective_menu_baselines": np.asarray([0.2], dtype=np.float32),
        "effective_menu_baseline_offsets": np.asarray([0, 1], dtype=np.int64),
        "effective_action_probabilities": np.asarray([0.8], dtype=np.float32),
        "effective_action_probability_offsets": np.asarray([0, 1], dtype=np.int64),
    }
    compatibility = ArtifactValue(
        state=None,
        manifest={
            "compatibility_hash": compatibility_hash,
            "effective_menus": raw_menus,
            "effective_actions": raw_actions,
        },
        arrays=arrays,
    )
    body = {
        "schema_version": "midogpp_harp_v7_effective_menu_store_v1",
        "compatibility_hash": compatibility_hash,
        "effective_menus": raw_menus,
        "effective_actions": raw_actions,
        "effective_menu_count": 1,
        "effective_action_count": 1,
        "directions_retained": ["D01", "D10"],
        "all_margins_excluded": True,
        "exact_b_noops_removed": True,
        "shared_source_target_implementation": True,
    }
    effective = ArtifactValue(
        state=None,
        manifest={**body, "effective_menu_hash": canonical_hash(body)},
        arrays=arrays,
    )
    state = SimpleNamespace(
        effective_menus=(
            SimpleNamespace(menu_hash=menu_hash, actions=(object(),)),
        )
    )
    return effective, compatibility, state


def test_effective_menu_store_rejects_array_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    effective, compatibility, state = _effective_menu_artifacts()
    monkeypatch.setattr(
        validation,
        "compatibility_state_from_artifact",
        lambda value, *, expected_outer_menu_hashes: state,
    )
    effective_hash, restored = validation._validate_effective_menu_store(
        effective,
        compatibility,
        expected_outer_menu_hashes={"H": "e" * 64},
    )
    assert effective_hash == effective.manifest["effective_menu_hash"]
    assert restored is state

    arrays = dict(effective.arrays)
    arrays["effective_action_probabilities"] = np.asarray(
        [0.7], dtype=np.float32
    )
    tampered = ArtifactValue(
        state=None,
        manifest=effective.manifest,
        arrays=arrays,
    )
    with pytest.raises(ProtocolError, match="durable arrays"):
        validation._validate_effective_menu_store(
            tampered,
            compatibility,
            expected_outer_menu_hashes={"H": "e" * 64},
        )
