from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import ast
import hashlib
import inspect
import json
import multiprocessing
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.action_surface import (
    ActionDraft,
    ActionResponse,
    RouteActionDraftSurface,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.admission import (
    PseudoPolicyEvidence as V2PseudoPolicyEvidence,
    build_outer_admission as build_v2_outer_admission,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.contracts import (
    BankViability,
    FavorableUtility,
    RouteKey,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.engine import (
    fit_outer_action_policy_surface,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.identity import (
    ACTION_STRATA,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.inventory import (
    ExpectedRouteInventory,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.legacy_control import (
    seal_legacy_control,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.surface_set import (
    seal_action_surface_set,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3 import (
    method_controls as v3_method_controls,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3.admission import (
    ADMISSION_STATISTIC_NAMES,
    CONSTANT_RANK_UNDEFINED_REASON,
    DENOMINATOR_UNDEFINED_REASON,
    NullableStatistic,
    OuterAdmission,
    PseudoPolicyEvidence,
    build_outer_admission,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3.execution_admission import (
    BLOCKED_MESSAGE,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3.identity import (
    CYCLIC_METHOD_ID,
    EXPERIMENT_ID,
    PRIMARY_METHOD_ID,
    canonical_json_bytes,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3.protocol import (
    frozen_protocol_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3.runner import (
    run_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3.source_seal import (
    EXPECTED_V2_SOURCE_MANIFEST_SHA256,
    EXPECTED_V2_SOURCE_MEMBER_COUNT,
    EXPECTED_V2_SOURCE_TREE_SHA256,
    v3_repair_source_snapshot_identity,
    validate_v2_base_source_seal,
    validate_v3_repair_source_seal,
)
from midogpp_thesis.cvae.protocol import ProtocolError


CENTERS = ("0", "1", "2", "3", "5", "6", "7", "8", "9")
DONORS = CENTERS[1:]


def _hash(value: object) -> str:
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


def _utility(values: tuple[float, float, float]) -> FavorableUtility:
    return FavorableUtility(*values)


def _evidence(
    *,
    constant_metric: str | None = None,
    constant_side: str | None = None,
    zero_denominator: bool = False,
) -> tuple[PseudoPolicyEvidence, ...]:
    rows: list[PseudoPolicyEvidence] = []
    metric_index = {"bacc": 0, "brier": 1, "log": 2}
    for index, donor in enumerate(DONORS, start=1):
        predicted = [0.010 * index, 0.020 * index, 0.030 * index]
        realized = [0.011 * index, 0.018 * index, 0.029 * index]
        if constant_metric is not None:
            coordinate = metric_index[constant_metric]
            if constant_side == "predicted":
                predicted[coordinate] = 0.25
            elif constant_side == "realized":
                realized[coordinate] = 0.20
            else:
                raise AssertionError("test fixture side drifted")
        endpoint = 0.0 if zero_denominator and index == 1 else 1.0
        rows.append(
            PseudoPolicyEvidence(
                "0",
                donor,
                _utility(tuple(predicted)),
                _utility(tuple(realized)),
                True,
                True,
                endpoint,
                0.05,
                _utility(tuple(realized)),
                True,
                True,
                0.10,
            )
        )
    return tuple(rows)


def _spawn_build_admission(
    evidence_payloads: tuple[dict[str, object], ...],
) -> tuple[str, dict[str, object]]:
    rows = tuple(PseudoPolicyEvidence.from_payload(row) for row in evidence_payloads)
    admission = build_outer_admission("0", rows)
    return admission.admission_hash, admission.to_payload()


def _spawn_decision_and_exact_p(
    evidence_payloads: tuple[dict[str, object], ...],
) -> tuple[tuple[str, str, str], ...]:
    """Module-level worker proving DTO/decision/composition spawn parity."""

    surface_set, identity, cyclic = _build_method_sources()
    failed = build_outer_admission(
        "0",
        tuple(
            PseudoPolicyEvidence.from_payload(row)
            for row in evidence_payloads
        ),
    )
    if failed.passed:
        raise AssertionError("spawn parity fixture must fail admission")
    original_builder = v3_method_controls.build_admission_from_pseudo_policies
    v3_method_controls.build_admission_from_pseudo_policies = (
        lambda result, control: failed
    )
    try:
        decisions = (
            v3_method_controls.build_primary_method_decision(
                identity, seal_legacy_control(identity)
            ),
            v3_method_controls.build_cyclic_poison_method_decision(
                identity,
                cyclic,
                surface_set,
                seal_legacy_control(cyclic),
            ),
        )
        values: list[tuple[str, str, str]] = []
        for decision, sealed in (
            (decisions[0], surface_set.identity),
            (decisions[1], surface_set.cyclic),
        ):
            if sealed is None:
                raise AssertionError("cyclic seal is absent")
            routes = tuple(
                row
                for row in sealed.routes
                if row.route_key.surface_role == "target"
                and row.route_key.outer_center == "0"
            )
            order = tuple(
                reversed(
                    tuple(sample for row in routes for sample in row.sample_ids)
                )
            )
            composed = v3_method_controls.compose_method_prediction(
                routes,
                center_sample_order=order,
                decision=decision,
            )
            values.append(
                (
                    decision.decision_hash,
                    composed.method_composition_hash,
                    composed.prediction.probabilities.tobytes(order="C").hex(),
                )
            )
        return tuple(values)
    finally:
        v3_method_controls.build_admission_from_pseudo_policies = original_builder


@pytest.mark.parametrize("metric", ("bacc", "brier", "log"))
@pytest.mark.parametrize("side", ("predicted", "realized"))
def test_constant_rank_statistics_are_nullable_and_fail_closed(
    metric: str,
    side: str,
) -> None:
    admission = build_outer_admission(
        "0", _evidence(constant_metric=metric, constant_side=side)
    )
    statistic = admission.statistics_by_name[f"{metric}_spearman"]
    assert statistic.to_payload() == {
        "name": f"{metric}_spearman",
        "value": None,
        "defined": False,
        "undefined_reason": CONSTANT_RANK_UNDEFINED_REASON,
    }
    assert admission.passed is False
    assert (
        f"UNDEFINED_{metric.upper()}_SPEARMAN::CONSTANT_RANK_INPUT"
        in admission.reasons
    )
    assert canonical_json_bytes(admission.to_payload())
    json.dumps(admission.to_payload(), allow_nan=False)


def test_zero_denominator_is_nullable_and_selects_fail_closed_gate() -> None:
    admission = build_outer_admission("0", _evidence(zero_denominator=True))
    for name in (
        "normalized_oracle_gap",
        "legacy_normalized_oracle_gap",
    ):
        statistic = admission.statistics_by_name[name]
        assert statistic.value is None
        assert statistic.defined is False
        assert statistic.undefined_reason == DENOMINATOR_UNDEFINED_REASON
    assert admission.passed is False
    assert "INVALID_NORMALIZED_ORACLE_DENOMINATOR" in admission.reasons
    assert canonical_json_bytes(admission.to_payload())


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf")))
def test_injected_nonfinite_values_are_rejected_not_normalized(value: float) -> None:
    with pytest.raises(ProtocolError, match="finite float"):
        NullableStatistic("bacc_spearman", value, True, None)
    with pytest.raises(ProtocolError, match="finite float"):
        NullableStatistic.from_payload(
            {
                "name": "bacc_spearman",
                "value": value,
                "defined": True,
                "undefined_reason": None,
            }
        )
    with pytest.raises(ProtocolError, match="nonfinite"):
        canonical_json_bytes({"poison": value})
    with pytest.raises(ProtocolError, match="finite float"):
        PseudoPolicyEvidence(
            "0",
            "1",
            FavorableUtility.zeros(),
            FavorableUtility.zeros(),
            False,
            False,
            value,
            0.0,
            FavorableUtility.zeros(),
            False,
            False,
            0.0,
        )


def test_finite_admission_is_gate_equivalent_to_v2() -> None:
    v3_rows = _evidence()
    v2_rows = tuple(
        V2PseudoPolicyEvidence(
            row.outer_center,
            row.donor_center,
            row.predicted,
            row.realized,
            row.routed,
            row.jointly_safe,
            row.endpoint_oracle_bacc_gain,
            row.absolute_oracle_regret,
            row.legacy_realized,
            row.legacy_routed,
            row.legacy_jointly_safe,
            row.legacy_absolute_oracle_regret,
        )
        for row in v3_rows
    )
    v3 = build_outer_admission("0", v3_rows)
    v2 = build_v2_outer_admission("0", v2_rows)
    assert v3.passed == v2.passed
    assert v3.reasons == v2.reasons
    v2_statistics = dict(v2.statistics)
    assert tuple(row.name for row in v3.statistics) == ADMISSION_STATISTIC_NAMES
    assert all(row.defined for row in v3.statistics)
    for row in v3.statistics:
        assert row.value == v2_statistics[row.name]


def test_nullable_schema_round_trip_pickle_and_hash_are_stable() -> None:
    admission = build_outer_admission(
        "0", _evidence(constant_metric="bacc", constant_side="predicted")
    )
    restored = OuterAdmission.from_payload(admission.to_payload())
    pickled = pickle.loads(pickle.dumps(admission))
    assert restored == admission == pickled
    assert restored.admission_hash == admission.admission_hash
    assert pickled.to_payload() == admission.to_payload()
    statistic = admission.statistics_by_name["bacc_spearman"]
    assert NullableStatistic.from_payload(statistic.to_payload()) == statistic

    poisoned = admission.to_payload()
    poisoned["admission_hash"] = "f" * 64
    with pytest.raises(ProtocolError, match="hash drifted"):
        OuterAdmission.from_payload(poisoned)

    malformed = list(admission.statistics)
    malformed[0] = object()  # type: ignore[assignment]
    with pytest.raises(ProtocolError, match="contract drifted"):
        OuterAdmission(
            admission.outer_center,
            admission.donor_centers,
            admission.passed,
            admission.reasons,
            tuple(malformed),  # type: ignore[arg-type]
            admission.evidence_hashes,
        )


def test_spawned_and_serial_nullable_admission_hashes_match() -> None:
    payloads = tuple(
        row.to_payload()
        for row in _evidence(
            constant_metric="log", constant_side="realized"
        )
    )
    serial = _spawn_build_admission(payloads)
    try:
        with ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            spawned = executor.submit(_spawn_build_admission, payloads).result()
    except (NotImplementedError, PermissionError) as exc:
        pytest.skip(f"local sandbox cannot allocate spawned semaphores: {exc}")
    assert spawned == serial
    json.dumps(spawned[1], sort_keys=True, allow_nan=False)


def test_spawned_decisions_and_exact_p_bytes_match_serial() -> None:
    payloads = tuple(
        row.to_payload()
        for row in _evidence(
            constant_metric="bacc", constant_side="predicted"
        )
    )
    serial = _spawn_decision_and_exact_p(payloads)
    try:
        with ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            spawned = executor.submit(
                _spawn_decision_and_exact_p, payloads
            ).result()
    except (NotImplementedError, PermissionError) as exc:
        pytest.skip(f"local sandbox cannot allocate spawned semaphores: {exc}")
    assert spawned == serial
    assert len(spawned) == 2
    assert all(len(decision_hash) == 64 for decision_hash, _, _ in spawned)
    assert all(len(composition_hash) == 64 for _, composition_hash, _ in spawned)
    assert all(probability_hex for _, _, probability_hex in spawned)


def test_admission_surface_is_label_free() -> None:
    assert tuple(inspect.signature(build_outer_admission).parameters) == (
        "outer_center",
        "evidence",
    )
    evidence = _evidence()
    admission = build_outer_admission("0", evidence)
    assert admission.target_labels_opened is False
    assert all(row.to_payload()["target_labels_used"] is False for row in evidence)
    assert admission.to_payload()["target_labels_opened"] is False


def test_inherited_and_repair_source_scopes_are_disjoint_and_strict() -> None:
    base = dict(validate_v2_base_source_seal())
    assert base["v2_base_source_snapshot_manifest_sha256"] == (
        EXPECTED_V2_SOURCE_MANIFEST_SHA256
    )
    assert base["v2_base_source_snapshot_tree_sha256"] == (
        EXPECTED_V2_SOURCE_TREE_SHA256
    )
    assert base["v2_base_source_snapshot_member_count"] == (
        EXPECTED_V2_SOURCE_MEMBER_COUNT
    )
    repair = dict(v3_repair_source_snapshot_identity())
    validated = validate_v3_repair_source_seal(
        expected_manifest_sha256=repair[
            "v3_repair_source_snapshot_manifest_sha256"
        ],
        expected_tree_sha256=repair["v3_repair_source_snapshot_tree_sha256"],
        expected_member_count=repair["v3_repair_source_snapshot_member_count"],
    )
    assert validated["status"] == "PASS"
    assert repair["v3_repair_source_snapshot_member_count"] < 105


def test_repaired_path_has_static_forbidden_import_boundary() -> None:
    package = Path(
        v3_method_controls.__file__
    ).resolve().parent
    forbidden_exact = {
        (
            "fixed_bank_p_anchored_route_scoped_"
            "donor_crossfit_action_policy_surface_router.admission"
        ),
        (
            "fixed_bank_p_anchored_route_scoped_"
            "donor_crossfit_action_policy_surface_router.routing"
        ),
        (
            "fixed_bank_p_anchored_route_scoped_"
            "donor_crossfit_action_policy_surface_router.method_controls"
        ),
    }
    violations: list[str] = []
    for path in sorted(package.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules: tuple[str, ...]
            if isinstance(node, ast.ImportFrom):
                modules = (() if node.module is None else (node.module,))
            elif isinstance(node, ast.Import):
                modules = tuple(alias.name for alias in node.names)
            else:
                continue
            for module in modules:
                if (
                    module in forbidden_exact
                    or any(module.endswith(f".{value}") for value in forbidden_exact)
                    or (
                        "fixed_bank_p_anchored_route_scoped_donor_crossfit_"
                        "action_policy_surface_router.v2" in module
                    )
                ):
                    violations.append(f"{path.name}:{node.lineno}:{module}")
    assert violations == []


def test_planned_runner_rejects_before_mutating_paths(tmp_path: Path) -> None:
    artifact = tmp_path / "must-not-exist"
    scratch = tmp_path / "scratch-must-not-exist"
    config = SimpleNamespace(
        experiment_id=EXPERIMENT_ID,
        execution_authorized=False,
        protocol=frozen_protocol_payload(),
        runtime={"execution_authorized": False},
        claim_boundary={"execution_authorized": False},
    )
    with pytest.raises(ProtocolError, match=BLOCKED_MESSAGE):
        run_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3(
            config,
            artifact_root=artifact,
            scratch_root=scratch,
        )
    assert not artifact.exists()
    assert not scratch.exists()


def _viability(value: object) -> BankViability:
    return BankViability(
        True,
        True,
        (("0", 30.0), ("1", 30.0)),
        5.0,
        _hash((value, "viability")),
    )


def _route(*, outer: str, center: str, target: bool) -> RouteKey:
    return RouteKey(
        "target" if target else "pseudo",
        outer,
        outer if target else center,
        f"case-{center}",
        outer,
        None if target else center,
        _hash((outer, center, target, "fit")),
    )


def _draft_route(
    *, outer: str, center: str, target: bool, control_id: str
) -> RouteActionDraftSurface:
    route = _route(outer=outer, center=center, target=target)
    baseline = np.asarray([0.2, 0.8], dtype=np.float32)
    drafts = []
    center_index = CENTERS.index(center)
    for stratum_index, (family, direction) in enumerate(ACTION_STRATA):
        action = (
            np.asarray([0.7, 0.8], dtype=np.float32)
            if direction == "zero_to_one"
            else np.asarray([0.2, 0.3], dtype=np.float32)
        )
        level = 0.01 * (center_index + 1) + 0.001 * (stratum_index + 1)
        drafts.append(
            ActionDraft(
                route,
                family,
                direction,
                f"{family}::{direction}",
                action,
                FavorableUtility(level, level / 2.0, level / 3.0),
                0.5,
                _viability((center, family, direction)),
                _hash((center, "endpoint")),
                _hash((center, control_id, "posterior")),
            )
        )
    return RouteActionDraftSurface(
        route,
        (f"{center}-a", f"{center}-b"),
        baseline,
        tuple(drafts),
        _hash((center, "endpoint")),
        _hash((center, control_id, "posterior")),
        _hash("physical"),
        control_id,
    )


def _outer_result(sealed, *, outer: str = "0"):
    responses = []
    for prediction in sealed.predictions:
        route = prediction.key.route_key
        if route.outer_center != outer or route.surface_role != "pseudo":
            continue
        center_index = CENTERS.index(route.route_center)
        stratum_index = ACTION_STRATA.index(prediction.key.stratum)
        level = 0.012 * (center_index + 1) + 0.001 * (stratum_index + 1)
        responses.append(
            ActionResponse(
                prediction.key,
                prediction.prediction_hash,
                FavorableUtility(level, level / 2.0, level / 3.0),
                2,
                10,
                10,
                20,
                _hash("P"),
                _hash((route.route_center, "rows")),
            )
        )
    return fit_outer_action_policy_surface(
        sealed,
        responses,
        outer_center=outer,
    )


def _build_method_sources():
    inventory = ExpectedRouteInventory.focused_fixture(
        tuple(
            (center, f"case-{center}", sample_id)
            for center in CENTERS
            for sample_id in (f"{center}-a", f"{center}-b")
        )
    )
    identity_routes = tuple(
        _draft_route(
            outer=outer,
            center=center,
            target=center == outer,
            control_id="IDENTITY",
        )
        for outer in CENTERS
        for center in CENTERS
    )
    cyclic_routes = tuple(
        _draft_route(
            outer=outer,
            center=center,
            target=center == outer,
            control_id="WITHIN_CASE_CYCLIC_SHIFT",
        )
        for outer in CENTERS
        for center in CENTERS
    )
    surface_set = seal_action_surface_set(
        identity_routes,
        expected_inventory=inventory,
        cyclic_routes=cyclic_routes,
    )
    return (
        surface_set,
        _outer_result(surface_set.identity),
        _outer_result(surface_set.cyclic),
    )


@pytest.fixture(scope="module")
def method_sources():
    return _build_method_sources()


def test_primary_and_cyclic_adapters_compose_byte_exact_p_on_undefined_gate(
    method_sources,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    surface_set, identity, cyclic = method_sources
    failed = build_outer_admission(
        "0",
        _evidence(constant_metric="bacc", constant_side="predicted"),
    )
    assert failed.passed is False
    monkeypatch.setattr(
        v3_method_controls,
        "build_admission_from_pseudo_policies",
        lambda result, control: failed,
    )
    identity_legacy = seal_legacy_control(identity)
    cyclic_legacy = seal_legacy_control(cyclic)
    primary = v3_method_controls.build_primary_method_decision(
        identity, identity_legacy
    )
    cyclic_decision = v3_method_controls.build_cyclic_poison_method_decision(
        identity, cyclic, surface_set, cyclic_legacy
    )
    assert primary.method_id == PRIMARY_METHOD_ID
    assert cyclic_decision.method_id == CYCLIC_METHOD_ID
    assert primary.selected_action_hashes == ()
    assert cyclic_decision.selected_action_hashes == ()
    assert primary.exact_p_fallback is True
    assert cyclic_decision.exact_p_fallback is True
    assert primary.outer_admission_hash == failed.admission_hash
    assert cyclic_decision.outer_admission_hash == failed.admission_hash

    for decision, sealed in (
        (primary, surface_set.identity),
        (cyclic_decision, surface_set.cyclic),
    ):
        assert sealed is not None
        routes = tuple(
            row
            for row in sealed.routes
            if row.route_key.surface_role == "target"
            and row.route_key.outer_center == "0"
        )
        order = tuple(
            reversed(tuple(sample for row in routes for sample in row.sample_ids))
        )
        baseline_by_sample = {
            sample: np.float32(value)
            for row in routes
            for sample, value in zip(
                row.sample_ids, row.baseline_probabilities, strict=True
            )
        }
        expected = np.ascontiguousarray(
            [baseline_by_sample[sample] for sample in order], dtype=np.float32
        )
        composed = v3_method_controls.compose_method_prediction(
            routes,
            center_sample_order=order,
            decision=decision,
        )
        assert composed.prediction.sample_ids == order
        assert composed.prediction.probabilities.tobytes(order="C") == (
            expected.tobytes(order="C")
        )
        assert composed.prediction.selected_action_hashes == ()
        assert composed.prediction.selection_enabled is False
        assert composed.to_payload()["target_labels_used"] is False
