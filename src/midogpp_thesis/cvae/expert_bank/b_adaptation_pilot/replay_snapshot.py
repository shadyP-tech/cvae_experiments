"""Immutable input contract for non-adoptive B trajectory diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping

from ....real_features.classifier_reference.protocol import ProtocolError


@dataclass(frozen=True)
class ReplaySnapshot:
    """Hash-bound source-only arrays and stochastic trace for one B cell."""

    center: str
    training_seed: int
    fit_case_ids: tuple[str, ...]
    eval_case_ids: tuple[str, ...]
    file_hashes: Mapping[str, str]
    legacy_schedule_hash: str
    legacy_epsilon_hash: str
    expected_checkpoint_hash: str
    expected_prediction_hash: str

    def validate(self, root: Path) -> None:
        if set(self.fit_case_ids).intersection(self.eval_case_ids):
            raise ProtocolError("Replay snapshot has source/evaluation case overlap.")
        for relative, expected in self.file_hashes.items():
            path = Path(root) / relative
            if not path.is_file() or _sha256(path) != expected:
                raise ProtocolError("Replay snapshot input hash mismatch.")
        for value in (
            self.legacy_schedule_hash,
            self.legacy_epsilon_hash,
            self.expected_checkpoint_hash,
            self.expected_prediction_hash,
        ):
            if len(value) < 16:
                raise ProtocolError("Replay snapshot lacks a complete immutable trace identity.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
