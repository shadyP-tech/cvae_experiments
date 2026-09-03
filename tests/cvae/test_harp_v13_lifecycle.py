from __future__ import annotations

from pathlib import Path

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v13.execution.admission import (
    validate_pristine_or_label_free_recovery,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.harp_protocol import canonical_hash
from midogpp_thesis.cvae.runtime.artifact_io import atomic_json
from midogpp_thesis.cvae.runtime.harp_v13_execution.action_capacity import (
    build_action_capacity_certificate,
)


def _write_label_free_admission(root: Path, admission_hash: str) -> None:
    manifests = root / "manifests"
    manifests.mkdir(parents=True)
    capacity = dict(build_action_capacity_certificate())
    atomic_json(
        manifests / "admission.json",
        {
            "admission_hash": admission_hash,
            "action_capacity_certificate_hash": capacity[
                "capacity_certificate_hash"
            ],
        },
    )
    atomic_json(manifests / "action_capacity_certificate.json", capacity)
    atomic_json(manifests / "protocol_manifest.json", {})
    journal = {
        "schema_version": "midogpp_harp_v13_label_free_progress_journal_v1",
        "admission_hash": admission_hash,
        "phase": "LABEL_FREE_PHYSICAL_MENU",
        "labels_available": False,
        "entries": [],
    }
    atomic_json(
        manifests / "label_free_progress_journal.json",
        {**journal, "journal_hash": canonical_hash(journal)},
    )


def test_source_crossfit_crash_window_remains_same_lease_recoverable(
    tmp_path: Path,
) -> None:
    admission_hash = "a" * 64
    _write_label_free_admission(tmp_path, admission_hash)
    for relative in (
        "stores/source_crossfit_physical_surface/manifest.json",
        "stores/source_crossfit_physical_surface/probabilities.npy",
        "stores/source_crossfit_physical_surface/seed_dispersion.npy",
        "stores/source_crossfit_physical_surface/compatibility.npy",
        "stores/source_crossfit_effective_menu/manifest.json",
        "stores/source_crossfit_effective_menu/arrays.npz",
        "manifests/source_crossfit_physical_surface_seal.json",
        "manifests/source_crossfit_effective_menu_seal.json",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"label-free-v13-recovery-witness")

    assert (
        validate_pristine_or_label_free_recovery(
            tmp_path,
            admission_hash=admission_hash,
        )
        == "LABEL_FREE_RECOVERY"
    )


def test_source_crossfit_recovery_still_rejects_label_bearing_state(
    tmp_path: Path,
) -> None:
    admission_hash = "b" * 64
    _write_label_free_admission(tmp_path, admission_hash)
    forbidden = tmp_path / "stores/source_crossfit_folds/outer_0_heldout_1.json"
    forbidden.parent.mkdir(parents=True)
    forbidden.write_bytes(b"source-label-capability-opened")

    with pytest.raises(ProtocolError, match="closed after a label capability"):
        validate_pristine_or_label_free_recovery(
            tmp_path,
            admission_hash=admission_hash,
        )
