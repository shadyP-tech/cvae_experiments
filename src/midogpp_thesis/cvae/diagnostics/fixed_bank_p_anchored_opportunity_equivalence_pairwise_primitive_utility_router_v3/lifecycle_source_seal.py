"""Content seal for the executable OE-PPUR v3 lifecycle boundary.

The scientific producer seal intentionally covers the router and its numerical
dependencies.  This independent receipt covers the code that materializes
direct input #3, issues direct input #7, renders the launch envelope, and
dispatches the single-use run.  It contains no authorization state and performs
no mutation.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
import hashlib
from pathlib import Path

from ...protocol import ProtocolError
from .hashing import canonical_hash, require_sha256


_ENTRYPOINT_RELATIVE_PATH = "src/midogpp_thesis/oe_ppur_v3.py"
_PREPARATION_RELATIVE_ROOT = (
    "src/midogpp_thesis/cvae/diagnostics/oe_ppur_v3_preparation"
)
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
        if _factory_token is not _FACTORY_TOKEN:
            raise ProtocolError("OE-PPUR v3 lifecycle seal bypassed admission.")
        root = Path(self.repository_root)
        members = tuple(
            (str(relative), require_sha256(digest, f"lifecycle member {relative}"))
            for relative, digest in self.member_hashes
        )
        if (
            not root.is_absolute()
            or root.is_symlink()
            or type(self.member_count) is not int
            or self.member_count != len(members)
            or self.member_count < 2
            or tuple(relative for relative, _digest in members)
            != tuple(sorted(relative for relative, _digest in members))
            or _ENTRYPOINT_RELATIVE_PATH
            not in {relative for relative, _digest in members}
        ):
            raise ProtocolError("OE-PPUR v3 lifecycle seal topology drifted.")
        object.__setattr__(self, "member_hashes", members)
        object.__setattr__(
            self,
            "lifecycle_source_seal_sha256",
            require_sha256(
                self.lifecycle_source_seal_sha256,
                "lifecycle source seal",
            ),
        )
        if self.lifecycle_source_seal_sha256 != canonical_hash(
            _seal_payload(members)
        ):
            raise ProtocolError("OE-PPUR v3 lifecycle seal digest drifted.")
        object.__setattr__(self, "receipt_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v3_lifecycle_source_seal_receipt_v1",
            "entrypoint_relative_path": _ENTRYPOINT_RELATIVE_PATH,
            "preparation_relative_root": _PREPARATION_RELATIVE_ROOT,
            "member_count": self.member_count,
            "member_hashes": [
                {"path": relative, "sha256": digest}
                for relative, digest in self.member_hashes
            ],
            "lifecycle_source_seal_sha256": (
                self.lifecycle_source_seal_sha256
            ),
            "authorization_state_present": False,
            "filesystem_mutation_performed": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "receipt_hash": self.receipt_hash}

    @property
    def lifecycle_source_sha256(self) -> str:
        """Concise alias used by preparation and launch orchestration."""

        return self.lifecycle_source_seal_sha256


def build_lifecycle_source_seal(
    repository_root: str | Path | None = None,
) -> LifecycleSourceSealReceipt:
    """Hash the canonical executable and every preparation Python member."""

    root = _repository_root(repository_root)
    entrypoint = root / _ENTRYPOINT_RELATIVE_PATH
    preparation = root / _PREPARATION_RELATIVE_ROOT
    if entrypoint.is_symlink() or not entrypoint.is_file():
        raise ProtocolError("OE-PPUR v3 lifecycle entrypoint is unsafe.")
    if preparation.is_symlink() or not preparation.is_dir():
        raise ProtocolError("OE-PPUR v3 lifecycle preparation root is unsafe.")
    for candidate in preparation.rglob("*"):
        if candidate.is_symlink():
            raise ProtocolError("OE-PPUR v3 lifecycle tree contains a symlink.")
    paths = (entrypoint, *sorted(preparation.rglob("*.py")))
    members = tuple(
        (path.relative_to(root).as_posix(), _file_hash(path)) for path in paths
    )
    members = tuple(sorted(members))
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
    """Rebuild the tree and require the supplied receipt to remain live."""

    if type(value) is not LifecycleSourceSealReceipt:
        raise ProtocolError("OE-PPUR v3 lifecycle source seal is untyped.")
    rebuilt = build_lifecycle_source_seal(value.repository_root)
    if rebuilt != value:
        raise ProtocolError("OE-PPUR v3 lifecycle source bytes drifted.")
    if expected_sha256 is not None:
        expected = require_sha256(expected_sha256, "expected lifecycle source seal")
        if rebuilt.lifecycle_source_seal_sha256 != expected:
            raise ProtocolError("OE-PPUR v3 expected lifecycle source seal drifted.")
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
        raise ProtocolError("OE-PPUR v3 lifecycle repository is absent.") from exc
    if (
        resolved.is_symlink()
        or not resolved.is_dir()
        or not (resolved / _ENTRYPOINT_RELATIVE_PATH).is_file()
        or not (resolved / _PREPARATION_RELATIVE_ROOT).is_dir()
    ):
        raise ProtocolError("OE-PPUR v3 lifecycle repository topology drifted.")
    return resolved


def _file_hash(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ProtocolError("OE-PPUR v3 lifecycle member is unsafe.")
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ProtocolError("OE-PPUR v3 lifecycle member is unreadable.") from exc


def _seal_payload(
    members: tuple[tuple[str, str], ...],
) -> dict[str, object]:
    return {
        "schema_version": "oe_ppur_v3_lifecycle_source_tree_v1",
        "entrypoint_relative_path": _ENTRYPOINT_RELATIVE_PATH,
        "preparation_relative_root": _PREPARATION_RELATIVE_ROOT,
        "members": [
            {"path": relative, "sha256": digest}
            for relative, digest in members
        ],
    }


__all__ = (
    "LifecycleSourceSealReceipt",
    "build_lifecycle_source_seal",
    "validate_lifecycle_source_seal",
)
