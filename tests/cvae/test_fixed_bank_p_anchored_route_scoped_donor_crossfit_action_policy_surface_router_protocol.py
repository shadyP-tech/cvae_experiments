from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.config import (
    load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.label_firewall import (
    LabelFirewall,
    LabelPhase,
    pseudo_response_scope,
    support_scope,
    terminal_scope,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.persistence import (
    build_content_index,
    persist_dense_arrays,
    persist_report,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.physical_adapter import (
    CenterPhysicalSurface,
    action_library_by_target,
    build_physical_surface,
)
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.route_support import (
    BinaryLabel,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.contracts import (
    RouteKey,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.runner import (
    run_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.target_local_runtime import (
    build_fingerprint_surface,
    fit_route_posterior,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.validation.fresh_process import (
    require_two_fresh_process_validations,
    validate_bundle,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.workstation import (
    estimate_workstation_surface,
)
from midogpp_thesis.cvae.protocol import ProtocolError


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
    / "uniform_b_v2_consumed_test_fixed_bank_p_anchored_route_scoped_"
    "donor_crossfit_action_policy_surface_router_v1.yaml"
)


def _hash(value: object) -> str:
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


def _center_surface(center: str = "0") -> CenterPhysicalSurface:
    sample_ids = ("a", "b", "c", "d")
    case_ids = ("case-1", "case-1", "case-2", "case-2")
    arrays = []
    for index, action in enumerate(action_library_by_target()[center]):
        base = np.asarray([0.2, 0.8, 0.3, 0.7], dtype=np.float32)
        values = np.stack(
            [np.clip(base + np.float32(0.001 * (index + seed)), 0.0, 1.0) for seed in range(9)]
        )
        arrays.append((action.action_id, values))
    return CenterPhysicalSurface(center, sample_ids, case_ids, tuple(arrays), _hash("store"))


def test_label_firewall_requires_action_and_preterminal_seals() -> None:
    support_key = ("0", "support-case", "support-sample")
    pseudo_key = ("1", "pseudo-case", "pseudo-sample")
    terminal_key = ("0", "target-case", "target-sample")
    values = {support_key: 0, pseudo_key: 1, terminal_key: 1}

    def loader(keys: object, scope: str) -> tuple[BinaryLabel, ...]:
        return tuple(BinaryLabel(*key, values[key], scope) for key in keys)  # type: ignore[arg-type]

    firewall = LabelFirewall(loader)
    with pytest.raises(ProtocolError, match="terminal-only"):
        firewall.open_terminal_labels(center="0", keys=(terminal_key,))
    with pytest.raises(ProtocolError, match="wrong phase"):
        firewall.open_support(
            center="0", held_case_id="target-case", keys=(support_key,)
        )
    firewall.advance_support()
    with pytest.raises(ProtocolError, match="terminal-only"):
        firewall.open_terminal_labels(center="0", keys=(terminal_key,))
    support = firewall.open_support(
        center="0", held_case_id="target-case", keys=(support_key,)
    )
    assert support.scope == support_scope("0", "target-case")
    assert support.values == (0,)
    firewall.seal_action_surface(_hash("action"))
    with pytest.raises(ProtocolError, match="terminal-only"):
        firewall.open_terminal_labels(center="0", keys=(terminal_key,))
    firewall.advance_pseudo_response()
    pseudo_route = RouteKey(
        "pseudo",
        "0",
        "1",
        "pseudo-case",
        "0",
        "1",
        _hash("pseudo-fit"),
    )
    pseudo = firewall.open_pseudo_response(
        route_key=pseudo_route, sample_ids=("pseudo-sample",)
    )
    assert pseudo.scope == pseudo_response_scope(pseudo_route)
    assert pseudo.values == (1,)
    target_route = RouteKey(
        "target", "0", "0", "target-case", "0", None, _hash("target-fit")
    )
    with pytest.raises(ProtocolError, match="target scope"):
        firewall.open_pseudo_response(
            route_key=target_route, sample_ids=("target-sample",)
        )
    with pytest.raises(ProtocolError, match="terminal-only"):
        firewall.open_terminal_labels(center="0", keys=(terminal_key,))
    with pytest.raises(ProtocolError, match="attested seal"):
        firewall.open_terminal()
    firewall.attest_preterminal(_hash("preterminal"))
    with pytest.raises(ProtocolError, match="terminal-only"):
        firewall.open_terminal_labels(center="0", keys=(terminal_key,))
    firewall.open_terminal()
    terminal = firewall.open_terminal_labels(center="0", keys=(terminal_key,))
    assert terminal.scope == terminal_scope("0")
    assert terminal.values == (1,)
    assert firewall.phase is LabelPhase.TERMINAL
    audit = firewall.audit_payload()
    assert audit["raw_labels_persisted"] is False
    assert audit["generic_label_read_available"] is False
    assert audit["target_preterminal_grants"] == 0


def test_whole_case_posterior_never_fits_the_held_case() -> None:
    surface = _center_surface()
    fingerprint = build_fingerprint_surface(
        surface,
        physical_surface_hash=_hash("physical"),
        control_id="IDENTITY",
    )
    values = {("0", "case-1", "a"): 0, ("0", "case-1", "b"): 1}

    def loader(keys: object, scope: str) -> tuple[BinaryLabel, ...]:
        return tuple(BinaryLabel(*key, values[key], scope) for key in keys)  # type: ignore[arg-type]

    firewall = LabelFirewall(loader)
    firewall.advance_support()
    support = firewall.open_support(
        center="0",
        held_case_id="case-2",
        keys=tuple(values),
    )
    model, prediction = fit_route_posterior(
        fingerprint,
        held_case_id="case-2",
        support_capability=support,
    )
    assert model.training_case_ids == ("case-1",)
    assert model.held_case_id not in model.training_case_ids
    assert prediction.sample_ids == ("c", "d")
    assert prediction.natural_probabilities.flags.writeable is False
    assert model.support_capability_hash == support.capability_hash


def test_physical_adapter_canonicalizes_and_rejects_duplicate_seed_keys() -> None:
    cells = []
    for center, actions in action_library_by_target().items():
        for action in actions:
            for ordinal, (training_seed, generation_seed) in enumerate(
                (
                    (training, generation)
                    for training in TRAINING_SEEDS
                    for generation in GENERATION_SEEDS
                )
            ):
                cells.append(
                    SimpleNamespace(
                        target_center=center,
                        action_id=action.action_id,
                        training_seed=training_seed,
                        generation_seed=generation_seed,
                        probabilities=np.asarray([ordinal / 10.0], dtype=np.float32),
                    )
                )
    store = SimpleNamespace(
        cells=tuple(reversed(cells)),
        store_hash=_hash("store"),
        rows_by_center={center: (f"{center}-sample",) for center in action_library_by_target()},
        case_ids_by_center={center: (f"{center}-case",) for center in action_library_by_target()},
    )
    physical = build_physical_surface(store)
    first = physical.centers[0].seed_probabilities[0][1][:, 0]
    assert first == pytest.approx(np.arange(9, dtype=np.float32) / 10.0)

    malformed = list(cells)
    malformed[0] = SimpleNamespace(
        **{
            **malformed[0].__dict__,
            "training_seed": malformed[1].training_seed,
            "generation_seed": malformed[1].generation_seed,
        }
    )
    store.cells = tuple(malformed)
    with pytest.raises(ProtocolError, match="exact-nine"):
        build_physical_surface(store)


def test_persistence_rejects_labels_and_two_fresh_processes_reconstruct(
    tmp_path: Path,
) -> None:
    with pytest.raises(ProtocolError, match="forbidden field"):
        persist_report(
            tmp_path / "forbidden.json",
            {"labels": [0, 1]},
            report_role="forbidden",
        )
    arrays = tmp_path / "arrays" / "surface.npz"
    manifest = persist_dense_arrays(
        arrays,
        {"probabilities": np.asarray([0.2, 0.8], dtype=np.float32)},
        schema_version="pdcaps_test_surface_v1",
        lineage_hashes={"surface": _hash("surface")},
    )
    report_path = tmp_path / "reports" / "summary.json"
    persist_report(
        report_path,
        {"array_manifest_hash": manifest["manifest_hash"]},
        report_role="test_summary",
    )
    members = (
        "arrays/surface.npz",
        "arrays/surface.npz.manifest.json",
        "reports/summary.json",
    )
    build_content_index(tmp_path, required_members=members, phase="PRETERMINAL")
    checks = validate_bundle(tmp_path)
    attestation = require_two_fresh_process_validations(
        tmp_path, expected_checks=checks
    )
    assert attestation["status"] == "PASS"
    assert len(set(attestation["pids"])) == 2


def test_runner_and_workstation_contract_are_fail_closed_and_honest(
    tmp_path: Path,
) -> None:
    config = load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_config(
        CONFIG
    )
    output = tmp_path / "never-created"
    with pytest.raises(ProtocolError, match="execution is not authorized"):
        run_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router(
            config, artifact_root=output
        )
    assert not output.exists()
    estimate = estimate_workstation_surface()
    assert estimate.action_ridge_fits == 999
    assert estimate.policy_ridge_fits == 999
    assert estimate.estimated_dense_bytes < 64 * 1024 * 1024
