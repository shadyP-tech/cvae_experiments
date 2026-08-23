from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import pickle
import shutil
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics import cli
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3 import execution_admission
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3.execution_admission import (
    BLOCKED_MESSAGE,
    run_read_only_source_preflight,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3.identity import (
    EXPERIMENT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3.protocol import (
    frozen_protocol_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3.runner import (
    run_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3.source_seal import (
    EXPECTED_COMBINED_SOURCE_SEAL_SHA256,
    CombinedSourceSeal,
    v2_base_source_root,
    v3_repair_source_root,
    validate_combined_source_seal,
)
from midogpp_thesis.cvae.protocol import ProtocolError


V2_SURFACE = (
    "fixed-bank-p-anchored-route-scoped-donor-crossfit-"
    "action-policy-surface-router-v2"
)


def _spawn_source_preflight() -> CombinedSourceSeal:
    """Module-level worker used by the spawn/pickle regression."""

    return run_read_only_source_preflight()


def _planned_config() -> SimpleNamespace:
    return SimpleNamespace(
        experiment_id=EXPERIMENT_ID,
        execution_authorized=False,
        protocol=frozen_protocol_payload(),
        runtime={"execution_authorized": False},
        claim_boundary={"execution_authorized": False},
    )


def test_combined_source_preflight_is_pinned_plain_and_pickle_safe() -> None:
    seal = run_read_only_source_preflight()
    restored = pickle.loads(pickle.dumps(seal))

    assert restored == seal
    assert seal.combined_source_seal_hash == EXPECTED_COMBINED_SOURCE_SEAL_SHA256
    assert isinstance(seal.v2_base, dict)
    assert isinstance(seal.v3_repair, dict)
    assert seal.v2_base["status"] == "PASS"
    assert seal.v3_repair["status"] == "PASS"
    assert restored.to_payload() == seal.to_payload()
    pickle.dumps(seal.to_payload())


def test_combined_source_preflight_round_trips_through_spawn_worker() -> None:
    serial = run_read_only_source_preflight()
    try:
        with ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            spawned = executor.submit(_spawn_source_preflight).result(timeout=30)
    except (PermissionError, OSError) as exc:
        pytest.skip(f"local sandbox cannot allocate spawned process state: {exc}")

    assert spawned == serial
    assert spawned.to_payload() == serial.to_payload()


@pytest.mark.parametrize(
    ("source_root", "member", "root_argument", "message"),
    (
        (
            v2_base_source_root,
            "__init__.py",
            "v2_package_root",
            "inherited v2/base source bytes or inventory drifted",
        ),
        (
            v3_repair_source_root,
            "runner.py",
            "v3_package_root",
            "repair source bytes or inventory drifted",
        ),
    ),
)
def test_combined_source_preflight_rejects_tampered_component(
    tmp_path: Path,
    source_root: object,
    member: str,
    root_argument: str,
    message: str,
) -> None:
    copied = tmp_path / root_argument
    shutil.copytree(source_root(), copied)
    tampered = copied / member
    tampered.write_bytes(tampered.read_bytes() + b"\n# tampered\n")

    with pytest.raises(ProtocolError, match=message):
        validate_combined_source_seal(**{root_argument: copied})


def test_combined_source_preflight_rejects_combined_identity_drift() -> None:
    with pytest.raises(ProtocolError, match="combined source seal drifted"):
        validate_combined_source_seal(
            expected_combined_source_seal_hash="0" * 64,
        )


def test_runner_seals_sources_before_config_access_and_never_touches_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    real_preflight = execution_admission.run_read_only_source_preflight

    def tracked_preflight() -> CombinedSourceSeal:
        receipt = real_preflight()
        events.append("source_preflight")
        return receipt

    class TrackingConfig:
        @property
        def experiment_id(self) -> str:
            events.append("config")
            return EXPERIMENT_ID

        execution_authorized = False
        protocol = frozen_protocol_payload()
        runtime = {"execution_authorized": False}
        claim_boundary = {"execution_authorized": False}

        @property
        def target_labels(self) -> object:
            raise AssertionError("target labels must remain sealed")

    class UntouchablePath:
        def __fspath__(self) -> str:
            raise AssertionError("run path must not be resolved")

        def __str__(self) -> str:
            raise AssertionError("run path must not be inspected")

    monkeypatch.setattr(
        execution_admission,
        "run_read_only_source_preflight",
        tracked_preflight,
    )
    untouched_parent = tmp_path / "must-not-exist"
    with pytest.raises(ProtocolError, match=BLOCKED_MESSAGE):
        run_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3(
            TrackingConfig(),
            artifact_root=UntouchablePath(),  # type: ignore[arg-type]
            scratch_root=UntouchablePath(),  # type: ignore[arg-type]
        )

    assert events[0] == "source_preflight"
    assert events[1:] == ["config"]
    assert not untouched_parent.exists()


def test_runner_remains_non_authorizing_after_successful_source_preflight(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifact" / "output"
    scratch_root = tmp_path / "scratch" / "state"

    with pytest.raises(ProtocolError, match=BLOCKED_MESSAGE):
        run_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3(
            _planned_config(),
            artifact_root=artifact_root,
            scratch_root=scratch_root,
        )

    assert not artifact_root.parent.exists()
    assert not scratch_root.parent.exists()


def test_v2_cli_help_reports_failed_and_exhausted() -> None:
    help_text = " ".join(cli.build_parser().format_help().split())

    assert V2_SURFACE in help_text
    assert "P-DCAPS v2" in help_text
    assert "failed preterminally" in help_text
    assert "authorization is exhausted" in help_text
