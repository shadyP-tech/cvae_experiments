"""Direct validation of the immutable bank and GenerationLock inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from ....generation.contracts import GenerationLock
from ....protocol import ProtocolError
from ..hashing import canonical_hash
from ..identity import (
    CENTERS,
    EXPECTED_BANK_CONTENT_INDEX_SHA256,
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_GENERATION_CONTENT_INDEX_SHA256,
    EXPECTED_GENERATION_LOCK_HASH,
)


@dataclass(frozen=True, slots=True)
class ValidatedUpstreamInputs:
    expert_bank_root: Path
    generation_lock: GenerationLock
    expert_bank_receipt: Mapping[str, object]
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        root = Path(self.expert_bank_root)
        receipt = MappingProxyType(dict(self.expert_bank_receipt))
        if (
            not root.is_absolute()
            or root.is_symlink()
            or not root.is_dir()
            or type(self.generation_lock) is not GenerationLock
            or self.generation_lock.bank_lock_hash != EXPECTED_BANK_LOCK_HASH
            or self.generation_lock.generation_lock_hash
            != EXPECTED_GENERATION_LOCK_HASH
            or receipt.get("status") != "PASS"
        ):
            raise ProtocolError("OE-PPUR v3 validated upstream identity drifted.")
        object.__setattr__(self, "expert_bank_root", root)
        object.__setattr__(self, "expert_bank_receipt", receipt)
        object.__setattr__(
            self,
            "receipt_hash",
            canonical_hash(
                {
                    "schema_version": "oe_ppur_v3_upstream_input_receipt_v1",
                    "bank_lock_hash": EXPECTED_BANK_LOCK_HASH,
                    "generation_lock_hash": EXPECTED_GENERATION_LOCK_HASH,
                    "expert_count": receipt.get("expert_count"),
                    "all_experts_source_only": True,
                    "labels_opened": False,
                    "paths_persisted": False,
                }
            ),
        )


def load_validated_upstream_inputs(
    expert_bank_root: str | Path,
    generation_lock_root: str | Path,
) -> ValidatedUpstreamInputs:
    bank_root = _safe_root(expert_bank_root, role="expert bank")
    generation_root = _safe_root(generation_lock_root, role="GenerationLock")
    if (
        _sha256_file(bank_root / "manifests/content_index.json")
        != EXPECTED_BANK_CONTENT_INDEX_SHA256
        or _sha256_file(generation_root / "manifests/content_index.json")
        != EXPECTED_GENERATION_CONTENT_INDEX_SHA256
    ):
        raise ProtocolError("OE-PPUR v3 upstream content-index bytes drifted.")
    bank = _validate_expert_bank(bank_root)
    generation = GenerationLock(
        _read_json(generation_root / "manifests/generation_lock.json")
    )
    validation = _read_json(generation_root / "reports/validation_report.json")
    state = _read_json(generation_root / "reports/run_state.json")
    if (
        generation.bank_lock_hash != EXPECTED_BANK_LOCK_HASH
        or generation.generation_lock_hash != EXPECTED_GENERATION_LOCK_HASH
        or validation.get("status") != "PASS"
        or state.get("status") != "COMPLETE"
    ):
        raise ProtocolError("OE-PPUR v3 GenerationLock bundle drifted.")
    return ValidatedUpstreamInputs(bank_root, generation, bank)


def _validate_expert_bank(root: Path) -> Mapping[str, object]:
    bank = _read_json(root / "manifests/expert_bank_index.json")
    validation = _read_json(root / "reports/validation_report.json")
    decision = _read_json(root / "reports/promotion_decision.json")
    state = _read_json(root / "reports/run_state.json")
    records = bank.get("records")
    if (
        bank.get("bank_lock_hash")
        != canonical_hash(
            {key: value for key, value in bank.items() if key != "bank_lock_hash"}
        )[:16]
        or bank.get("bank_lock_hash") != EXPECTED_BANK_LOCK_HASH
        or bank.get("centers") != list(CENTERS)
        or bank.get("n_experts") != 27
        or bank.get("routing_authorized") is not True
        or not isinstance(records, list)
        or validation.get("status") != "PASS"
        or decision.get("may_feed_deployable_selection") is not True
        or state.get("status") != "COMPLETE"
    ):
        raise ProtocolError("OE-PPUR v3 expert-bank authorization drifted.")
    observed = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ProtocolError("OE-PPUR v3 expert-bank record is malformed.")
        observed.add((str(record.get("source_center")), int(record.get("training_seed", -1))))
        unhashed = {
            key: value for key, value in record.items() if key != "expert_lock_hash"
        }
        if (
            record.get("expert_lock_hash") != canonical_hash(unhashed)[:16]
            or record.get("fresh_source_only_training") is not True
            or record.get("routing_authorized") is not True
            or record.get("individual_expert_or_seed_selected") is not False
        ):
            raise ProtocolError("OE-PPUR v3 expert-bank source firewall drifted.")
        for path_role, digest_role in (
            ("checkpoint_path", "checkpoint_file_sha256"),
            ("frame_path", "frame_file_sha256"),
            ("sampler_path", "sampler_file_sha256"),
        ):
            member = _safe_member(root, str(record.get(path_role, "")))
            if _sha256_file(member) != record.get(digest_role):
                raise ProtocolError("OE-PPUR v3 expert-bank member drifted.")
    expected = {(center, seed) for center in CENTERS for seed in (17, 42, 101)}
    if observed != expected:
        raise ProtocolError("OE-PPUR v3 expert-bank coverage drifted.")
    return MappingProxyType(
        {
            "status": "PASS",
            "bank_lock_hash": EXPECTED_BANK_LOCK_HASH,
            "expert_count": len(records),
            "all_experts_source_only": True,
        }
    )


def _safe_root(value: str | Path, *, role: str) -> Path:
    candidate = Path(os.path.abspath(Path(value)))
    _reject_symlink_chain(candidate)
    try:
        root = Path(value).resolve(strict=True)
    except OSError as exc:
        raise ProtocolError(f"OE-PPUR v3 {role} root is absent.") from exc
    if (
        root != candidate
        or root.is_symlink()
        or not root.is_dir()
        or root == Path(root.anchor)
    ):
        raise ProtocolError(f"OE-PPUR v3 {role} root is unsafe.")
    return root


def _safe_member(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or not relative or ".." in path.parts:
        raise ProtocolError("OE-PPUR v3 upstream member path is unsafe.")
    candidate = root / path
    _reject_symlink_chain(candidate, stop=root)
    try:
        member = candidate.resolve(strict=True)
        member.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ProtocolError("OE-PPUR v3 upstream member escaped its root.") from exc
    if member.is_symlink() or not member.is_file():
        raise ProtocolError("OE-PPUR v3 upstream member is unsafe.")
    return member


def _reject_symlink_chain(path: Path, *, stop: Path | None = None) -> None:
    current = path
    boundary = None if stop is None else Path(stop)
    while True:
        if current.is_symlink():
            raise ProtocolError("OE-PPUR v3 upstream path contains a symlink.")
        if boundary is not None and current == boundary:
            return
        if current == current.parent:
            return
        current = current.parent


def _read_json(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ProtocolError("OE-PPUR v3 upstream JSON member is unsafe.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("OE-PPUR v3 upstream JSON member is unreadable.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("OE-PPUR v3 upstream JSON member is not an object.")
    return payload


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ProtocolError("OE-PPUR v3 upstream hashed member is unsafe.")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ("ValidatedUpstreamInputs", "load_validated_upstream_inputs")
