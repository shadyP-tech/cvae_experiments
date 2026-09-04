from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v16.source_label_capability import (
    issue_target_support_label_capability,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v16.support_label_access_fence import (
    begin_support_label_access,
)
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.hierarchical_support_action_risk_router_v16 import (
    RouterFitConfig,
    SurfaceRole,
)
from midogpp_thesis.cvae.routing.hierarchical_support_action_risk_router_v16.hashing import (
    canonical_hash,
)
from midogpp_thesis.cvae.runtime.artifact_io import atomic_json, read_json, sha256_file
from midogpp_thesis.cvae.runtime.harp_v16_execution.contracts import (
    ActionKind,
    LabelFreeActionBlock,
    LabelFreeOuterMenu,
)
from midogpp_thesis.cvae.runtime.harp_v16_execution.prelabel_diagnostics import (
    build_prelabel_diagnostics,
)
from midogpp_thesis.cvae.runtime.harp_v16_execution.stores import (
    write_label_free_outer_menu,
)
from midogpp_thesis.cvae.runtime.harp_v16_execution.support_independence import (
    FixedBankSupportIndependenceAttestation,
)
from midogpp_thesis.cvae.runtime.harp_v16_execution.support_model_artifacts import (
    build_support_outcome_artifact,
    build_support_router_artifact,
    build_support_target_routes,
    report_support_router_artifact,
)
from midogpp_thesis.cvae.runtime.harp_v16_execution.support_surface_seals import (
    report_support_target_surface_seals,
    write_support_target_surface_seals,
)
from midogpp_thesis.cvae.runtime.harp_v16_execution.support_target_adapter import (
    LABEL_FREE_FEATURE_NAMES,
    attach_support_outcomes,
    compile_support_target_menus,
)


H = "0"
CANDIDATES = tuple(center for center in CENTERS if center != H)


def _identities(prefix: str, case_count: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    cases = tuple(f"{prefix}-case-{index:02d}" for index in range(case_count))
    return (
        tuple(f"{case}-sample-{label}" for case in cases for label in (0, 1)),
        tuple(case for case in cases for _label in (0, 1)),
    )


def _probabilities(
    kind: ActionKind, source: str | None, case_count: int
) -> np.ndarray:
    if kind is ActionKind.B:
        pair = (0.20, 0.80)
    elif kind is ActionKind.U:
        pair = (0.60, 0.40)
    else:
        position = CANDIDATES.index(str(source))
        pair = (0.61 + 0.01 * position, 0.39 - 0.01 * position)
    return np.asarray(pair * case_count, dtype=np.float32)


def _block(
    role: str,
    kind: ActionKind,
    source: str | None,
    *,
    case_count: int,
) -> LabelFreeActionBlock:
    samples, cases = _identities("train" if role == "support" else "test", case_count)
    return LabelFreeActionBlock(
        surface_role=role,
        outer_target_id=H,
        query_center_id=H,
        action_kind=kind,
        selected_source_id=source,
        sample_ids=samples,
        case_ids=cases,
        probabilities=_probabilities(kind, source, case_count),
        seed_dispersion=np.full(len(samples), 0.01, dtype=np.float32),
    )


def _physical_menu(*, support_cases: int = 12, target_cases: int = 2) -> LabelFreeOuterMenu:
    blocks = []
    for role, count in (("support", support_cases), ("target", target_cases)):
        blocks.extend(
            (
                _block(role, ActionKind.B, None, case_count=count),
                _block(role, ActionKind.U, None, case_count=count),
                *(
                    _block(role, ActionKind.HXE, source, case_count=count)
                    for source in CANDIDATES
                ),
            )
        )
    return LabelFreeOuterMenu(
        outer_target_id=H,
        blocks=tuple(sorted(blocks, key=lambda row: row.key)),
        lineage={"fixture": "v16-support-target"},
    )


def _support_labels(bundle) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "center": H,
            "case_id": case_id,
            "sample_id": sample_id,
            "label": label,
        }
        for case_id, samples in bundle.support_case_samples
        for sample_id, label in zip(samples, (1, 0), strict=True)
    )


def _attestation() -> FixedBankSupportIndependenceAttestation:
    per_target = {
        center: canonical_hash(
            {
                "outer_target_id": center,
                "candidate_source_ids": tuple(row for row in CENTERS if row != center),
            }
        )
        for center in CENTERS
    }
    body = {
        "schema_version": "midogpp_harp_v16_fixed_bank_support_independence_v1",
        "bank_index_sha256": "a" * 64,
        "generation_lock_sha256": "b" * 64,
        "source_local_lineage_hash": "c" * 64,
        "per_target_hashes": per_target,
        "candidate_pool_semantics": "C_MINUS_H",
        "target_expert_unrepresentable": True,
        "source_frames_and_samplers_source_center_local": True,
        "classifier_scaler_fit": "synthetic_train_only",
        "support_labels_may_update": "H_LOCAL_ROUTER_ONLY",
        "support_labels_may_not_update": [
            "expert_checkpoint",
            "source_frame",
            "aggregate_prior",
            "generation",
            "classifier",
            "menu_geometry",
            "shared_transform",
            "hyperparameter_grid",
        ],
        "labels_consumed": False,
    }
    return FixedBankSupportIndependenceAttestation(
        bank_index_sha256="a" * 64,
        generation_lock_sha256="b" * 64,
        source_local_lineage_hash="c" * 64,
        per_target_hashes=per_target,
        attestation_hash=canonical_hash(body),
    )


def test_bridge_builds_role_qualified_exact_byte_menus() -> None:
    physical = _physical_menu()
    bundle = compile_support_target_menus(physical)

    assert bundle.candidate_source_ids == CANDIDATES
    assert len(bundle.support_menus) == 12
    assert len(bundle.target_menus) == 2
    assert bundle.support_menu_hash != bundle.target_menu_hash
    assert tuple(bundle.support_menus[0].actions[0].feature_names) == LABEL_FREE_FEATURE_NAMES
    assert all(
        row.surface_role is SurfaceRole.TARGET_TRAIN_SUPPORT
        for row in bundle.support_menus
    )
    assert all(
        row.surface_role is SurfaceRole.TARGET_EVALUATION
        for row in bundle.target_menus
    )
    assert all(
        row.candidate_source_id != H
        for menu in (*bundle.support_menus, *bundle.target_menus)
        for row in menu.actions
    )

    target_case = bundle.target_menus[0]
    d01 = target_case.action_for("U:D01")
    assert d01 is not None
    selected = np.frombuffer(
        b"".join(bytes.fromhex(value) for value in d01.action_probability_hex),
        dtype="<f4",
    )
    baseline = physical.target_block(ActionKind.B).probabilities[:2]
    uniform = physical.target_block(ActionKind.U).probabilities[:2]
    assert selected[0].tobytes() == uniform[0].tobytes()
    assert selected[1].tobytes() == baseline[1].tobytes()


def test_noops_and_exact_byte_duplicates_are_removed_before_labels() -> None:
    physical = _physical_menu()
    support_u = physical.support_block(ActionKind.U)
    support_first = physical.support_block(ActionKind.HXE, CANDIDATES[0])
    replacement = replace(
        support_first,
        probabilities=np.asarray(support_u.probabilities, dtype=np.float32).copy(),
    )
    blocks = tuple(
        replacement if row is support_first else row for row in physical.blocks
    )
    compiled = compile_support_target_menus(
        LabelFreeOuterMenu(H, tuple(sorted(blocks, key=lambda row: row.key)), physical.lineage)
    )
    actions = compiled.support_menus[0].actions
    outputs = tuple(row.action_probability_hex for row in actions)
    assert len(outputs) == len(set(outputs))
    assert all(row.action_probability_hex != row.baseline_probability_hex for row in actions)


def test_support_outcomes_fit_h_local_router_and_route_full_target() -> None:
    bundle = compile_support_target_menus(_physical_menu())
    labels = _support_labels(bundle)
    outcomes = attach_support_outcomes(bundle, labels)
    assert outcomes
    assert all(row.bacc_gain > 0.0 for row in outcomes)

    support = build_support_outcome_artifact((bundle,), {H: labels})
    fitted = build_support_router_artifact(
        support,
        config=RouterFitConfig(minimum_support_cases=4),
    )
    report = report_support_router_artifact(fitted)
    json.dumps(report, sort_keys=True)
    assert report["evaluation_labels_consumed"] is False
    assert report["per_outer"][0]["admission"]["admitted"] is True
    oof = fitted.arrays["support_oof_values"]
    columns = tuple(fitted.manifest["oof_array_columns"])
    assert columns == (
        "predicted_gain",
        "predicted_harm_probability",
        "predicted_brier_delta",
        "predicted_log_loss_delta",
        "observed_gain",
        "observed_harm",
        "observed_brier_delta",
        "observed_log_loss_delta",
    )
    assert oof.ndim == 2 and oof.shape[1] == len(columns) == 8
    first = fitted.state.routers[0].support_crossfit.records[0]
    assert tuple(oof[0]) == pytest.approx(
        (
            first.prediction.predicted_gain,
            first.prediction.predicted_harm_probability,
            first.prediction.predicted_brier_delta,
            first.prediction.predicted_log_loss_delta,
            first.outcome.bacc_gain,
            float(first.outcome.harmed),
            first.outcome.brier_delta,
            first.outcome.log_loss_delta,
        )
    )

    routes = build_support_target_routes((bundle,), fitted)
    assert len(routes.cases) == 2
    assert all(case.selected_kind is not ActionKind.B for case in routes.cases)
    assert all(case.selected_source_id != H for case in routes.cases)
    assert all(
        case.selected_probabilities.tobytes(order="C")
        == case.routed_probabilities.tobytes(order="C")
        for case in routes.cases
    )
    assert routes.model_hash == fitted.manifest["model_hash"]
    diagnostics = build_prelabel_diagnostics(routes)
    assert diagnostics["admitted_target_local_router_count"] == 1
    assert diagnostics["cases_with_admitted_router_count"] == len(routes.cases)
    assert diagnostics["selected_certificate_count"] == len(routes.cases)


def test_support_router_science_pool_is_spawn_safe_and_deterministic() -> None:
    bundle = compile_support_target_menus(_physical_menu())
    support = build_support_outcome_artifact(
        (bundle,), {H: _support_labels(bundle)}
    )
    router_config = RouterFitConfig(minimum_support_cases=4)
    runtime_config = SimpleNamespace(
        model=router_config.public_payload(),
        runtime={
            "science_workers": 4,
            "science_blas_threads_per_worker": 1,
            "multiprocessing_start_method": "spawn",
        },
    )
    parallel = build_support_router_artifact(support, config=runtime_config)
    repeated = build_support_router_artifact(support, config=runtime_config)
    assert parallel.state.state_hash == repeated.state.state_hash
    assert parallel.manifest["model_hash"] == repeated.manifest["model_hash"]
    assert parallel.manifest["science_execution"]["worker_count_used"] == 1
    assert parallel.manifest["science_execution"]["cuda_visible_to_workers"] is False


def test_support_label_join_is_exact_and_center_scoped() -> None:
    bundle = compile_support_target_menus(_physical_menu())
    labels = list(_support_labels(bundle))
    with pytest.raises(ProtocolError, match="exactly cover"):
        attach_support_outcomes(bundle, labels[:-1])
    labels[0] = {**labels[0], "center": "1"}
    with pytest.raises(ProtocolError, match="H-local scope"):
        attach_support_outcomes(bundle, labels)


def test_durable_role_seals_issue_the_existing_support_capability(tmp_path) -> None:
    physical = _physical_menu()
    bundle = compile_support_target_menus(physical)
    store = write_label_free_outer_menu(tmp_path / "physical", physical)
    seals = write_support_target_surface_seals(
        tmp_path / "seals",
        bundle=bundle,
        physical_store_receipt=store,
        fixed_bank_independence=_attestation(),
    )
    report = report_support_target_surface_seals(seals)
    assert report["support_labels_opened"] is False
    support_payload = read_json(seals.support_menu_seal_path)
    target_payload = read_json(seals.target_menu_seal_path)
    assert set(support_payload) == {
        "schema_version",
        "experiment_id",
        "outer_target_id",
        "surface_role",
        "candidate_source_ids",
        "action_identity_hash",
        "menu_hash",
        "store_receipt_hash",
        "labels_consumed",
        "seal_hash",
    }
    assert support_payload["action_identity_hash"] == target_payload["action_identity_hash"]
    assert support_payload["menu_hash"] != target_payload["menu_hash"]

    label_index = tmp_path / "support-label-index.json"
    atomic_json(label_index, {"center": H, "labels": "sealed"})
    index_payloads = tuple(
        {
            "ordered_center_ids": list(CENTERS),
            "index_hash": character * 64,
            "support_labels_opened": False,
            "evaluation_labels_opened": False,
        }
        for character in ("d", "e", "f")
    )
    support_index_path = tmp_path / "manifests/target_support_menu_seals.json"
    target_index_path = tmp_path / "manifests/target_evaluation_menu_seals.json"
    bank_index_path = (
        tmp_path / "manifests/target_bank_independence_attestations.json"
    )
    for path, payload in zip(
        (support_index_path, target_index_path, bank_index_path),
        index_payloads,
        strict=True,
    ):
        atomic_json(path, payload)
    access_fence = begin_support_label_access(
        tmp_path,
        config_hash="0" * 64,
        admission_hash="1" * 64,
        authorization_lease_hash="2" * 64,
        ordered_center_ids=CENTERS,
        support_surface_seal_index=index_payloads[0],
        support_surface_seal_index_path=support_index_path,
        target_surface_seal_index=index_payloads[1],
        target_surface_seal_index_path=target_index_path,
        bank_independence_index=index_payloads[2],
        bank_independence_index_path=bank_index_path,
        label_index_sha256=sha256_file(label_index),
    )
    capability = issue_target_support_label_capability(
        outer_target_id=H,
        support_menu_seal_path=seals.support_menu_seal_path,
        support_menu_seal_sha256=seals.support_menu_seal_sha256,
        target_menu_seal_path=seals.target_menu_seal_path,
        target_menu_seal_sha256=seals.target_menu_seal_sha256,
        bank_independence_attestation_path=(
            seals.bank_independence_attestation_path
        ),
        bank_independence_attestation_sha256=(
            seals.bank_independence_attestation_sha256
        ),
        label_index_path=label_index,
        label_index_sha256=sha256_file(label_index),
        support_label_access_fence=access_fence,
    )
    capability.authorize((H,))
    with pytest.raises(ProtocolError, match="cross-scoped"):
        capability.authorize((H, "1"))
