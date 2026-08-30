"""Recursive content seal for the complete executable OE-PPUR v4 lifecycle."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
import hashlib
from pathlib import Path

from ...protocol import ProtocolError
from .hashing import canonical_hash, require_sha256
from .identity import PACKAGE_NAME


_ENTRYPOINT_RELATIVE_PATH = "src/midogpp_thesis/oe_ppur_v4.py"
_ADAPTER_RELATIVE_ROOT = "src/midogpp_thesis/cvae/diagnostics/" + PACKAGE_NAME
_PREPARATION_RELATIVE_ROOT = "src/midogpp_thesis/cvae/diagnostics/oe_ppur_v4_preparation"
_FACTORY_TOKEN = object()


@dataclass(frozen=True, slots=True)
class LifecycleSourceSealReceipt:
    repository_root: str
    member_count: int
    member_hashes: tuple[tuple[str, str], ...]
    lifecycle_source_seal_sha256: str
    _factory_token: InitVar[object | None] = None
    receipt_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object | None) -> None:
        members = tuple(
            (str(relative), require_sha256(digest, f"lifecycle member {relative}"))
            for relative, digest in self.member_hashes
        )
        root = Path(self.repository_root)
        required = {
            _ENTRYPOINT_RELATIVE_PATH,
            _ADAPTER_RELATIVE_ROOT + "/runner.py",
            _PREPARATION_RELATIVE_ROOT + "/workspace.py",
        }
        if (
            _factory_token is not _FACTORY_TOKEN
            or not root.is_absolute()
            or root.is_symlink()
            or self.member_count != len(members)
            or self.member_count < 3
            or tuple(relative for relative, _digest in members)
            != tuple(sorted(relative for relative, _digest in members))
            or not required.issubset({relative for relative, _digest in members})
        ):
            raise ProtocolError("OE-PPUR v4 lifecycle seal topology drifted.")
        seal = require_sha256(
            self.lifecycle_source_seal_sha256, "lifecycle source seal"
        )
        if seal != canonical_hash(_seal_payload(members)):
            raise ProtocolError("OE-PPUR v4 lifecycle seal digest drifted.")
        object.__setattr__(self, "member_hashes", members)
        object.__setattr__(self, "lifecycle_source_seal_sha256", seal)
        object.__setattr__(self, "receipt_hash", canonical_hash(self._payload()))

    @property
    def lifecycle_source_sha256(self) -> str:
        return self.lifecycle_source_seal_sha256

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v4_lifecycle_source_seal_receipt_v1",
            "entrypoint_relative_path": _ENTRYPOINT_RELATIVE_PATH,
            "adapter_relative_root": _ADAPTER_RELATIVE_ROOT,
            "preparation_relative_root": _PREPARATION_RELATIVE_ROOT,
            "member_count": self.member_count,
            "member_hashes": [
                {"path": relative, "sha256": digest}
                for relative, digest in self.member_hashes
            ],
            "lifecycle_source_seal_sha256": self.lifecycle_source_seal_sha256,
            "recursive_adapter_sealed": True,
            "recursive_preparation_sealed": True,
            "authorization_state_present": False,
            "filesystem_mutation_performed": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "receipt_hash": self.receipt_hash}


def build_lifecycle_source_seal(
    repository_root: str | Path | None = None,
) -> LifecycleSourceSealReceipt:
    root = _repository_root(repository_root)
    paths = (
        root / _ENTRYPOINT_RELATIVE_PATH,
        *sorted((root / _ADAPTER_RELATIVE_ROOT).rglob("*.py")),
        *sorted((root / _PREPARATION_RELATIVE_ROOT).rglob("*.py")),
    )
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ProtocolError("OE-PPUR v4 lifecycle member is unsafe.")
    members = tuple(
        sorted(
            (path.relative_to(root).as_posix(), _file_hash(path)) for path in paths
        )
    )
    return LifecycleSourceSealReceipt(
        repository_root=root.as_posix(),
        member_count=len(members),
        member_hashes=members,
        lifecycle_source_seal_sha256=canonical_hash(_seal_payload(members)),
        _factory_token=_FACTORY_TOKEN,
    )


def validate_lifecycle_source_seal(
    value: object,
    *,
    expected_sha256: str | None = None,
) -> LifecycleSourceSealReceipt:
    if type(value) is not LifecycleSourceSealReceipt:
        raise ProtocolError("OE-PPUR v4 lifecycle source seal is untyped.")
    rebuilt = build_lifecycle_source_seal(value.repository_root)
    if rebuilt != value:
        raise ProtocolError("OE-PPUR v4 lifecycle source bytes drifted.")
    if expected_sha256 is not None and rebuilt.lifecycle_source_seal_sha256 != require_sha256(
        expected_sha256, "expected lifecycle source seal"
    ):
        raise ProtocolError("OE-PPUR v4 expected lifecycle source seal drifted.")
    return rebuilt


def _repository_root(value: str | Path | None) -> Path:
    root = Path(value) if value is not None else next(
        (
            parent
            for parent in Path(__file__).resolve().parents
            if (parent / "src/midogpp_thesis").is_dir()
        ),
        Path("."),
    )
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise ProtocolError("OE-PPUR v4 lifecycle repository is absent.") from exc
    if resolved.is_symlink() or not resolved.is_dir():
        raise ProtocolError("OE-PPUR v4 lifecycle repository topology drifted.")
    return resolved


def _file_hash(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ProtocolError("OE-PPUR v4 lifecycle member is unreadable.") from exc


def _seal_payload(members: tuple[tuple[str, str], ...]) -> dict[str, object]:
    return {
        "schema_version": "oe_ppur_v4_lifecycle_source_tree_v1",
        "entrypoint_relative_path": _ENTRYPOINT_RELATIVE_PATH,
        "adapter_relative_root": _ADAPTER_RELATIVE_ROOT,
        "preparation_relative_root": _PREPARATION_RELATIVE_ROOT,
        "members": [
            {"path": relative, "sha256": digest} for relative, digest in members
        ],
    }


__all__ = (
    "LifecycleSourceSealReceipt",
    "build_lifecycle_source_seal",
    "validate_lifecycle_source_seal",
)
