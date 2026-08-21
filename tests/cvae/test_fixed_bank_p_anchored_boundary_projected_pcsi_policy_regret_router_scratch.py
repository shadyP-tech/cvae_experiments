from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router import (
    scratch,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.frozen_source_streams import (
    SOURCE_ARRAY_MEMBER,
    SOURCE_INDEX_MEMBER,
    SOURCE_LOCK_MEMBER,
)


def _completed_scratch(root: Path) -> scratch.ScratchLease:
    source = root / scratch.SOURCE_DIRECTORY
    prediction = root / scratch.PREDICTION_DIRECTORY
    for member in (SOURCE_ARRAY_MEMBER, SOURCE_INDEX_MEMBER, SOURCE_LOCK_MEMBER):
        path = source / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"sealed fixture")
    (source / "checkpoints").mkdir()
    (prediction / "checkpoints").mkdir(parents=True)
    return scratch.ScratchLease(root, "dedicated_local")


def _matching_locks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        scratch,
        "load_frozen_source_streams",
        lambda *_args, **_kwargs: SimpleNamespace(
            lock_payload={"source_stream_lock_hash": "1" * 16}
        ),
    )


def test_cleanup_accepts_only_empty_owned_checkpoint_parents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease = _completed_scratch(tmp_path / "scratch")
    _matching_locks(monkeypatch)

    scratch.cleanup_scratch(
        lease,
        config=SimpleNamespace(contract_hash="2" * 16),
        artifact_root=tmp_path / "artifact",
    )

    assert not lease.root.exists()


@pytest.mark.parametrize(
    "drift", ("prediction_file", "prediction_nested", "source_file")
)
def test_cleanup_rejects_foreign_state_without_deleting_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    lease = _completed_scratch(tmp_path / "scratch")
    if drift == "prediction_file":
        (lease.root / scratch.PREDICTION_DIRECTORY / "foreign.bin").write_bytes(
            b"foreign"
        )
    elif drift == "prediction_nested":
        (
            lease.root
            / scratch.PREDICTION_DIRECTORY
            / "checkpoints"
            / "foreign"
        ).mkdir()
    else:
        (lease.root / scratch.SOURCE_DIRECTORY / "foreign.bin").write_bytes(
            b"foreign"
        )
    _matching_locks(monkeypatch)

    with pytest.raises(ProtocolError):
        scratch.cleanup_scratch(
            lease,
            config=SimpleNamespace(contract_hash="2" * 16),
            artifact_root=tmp_path / "artifact",
        )

    assert lease.root.exists()
