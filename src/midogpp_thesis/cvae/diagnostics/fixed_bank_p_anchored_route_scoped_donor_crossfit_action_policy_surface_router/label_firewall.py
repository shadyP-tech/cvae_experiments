"""Typed, phase-scoped label capabilities for P-DCAPS.

The backing label loader is deliberately private to this module. Callers can
only request one of the three protocol capabilities below; there is no generic
``read(scope=...)`` escape hatch that could open target labels preterminally.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Sequence

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from .contracts import RouteKey
from .identity import canonical_hash, require_sha256
from .route_support import BinaryLabel, require_exact_label_scope


LabelKey = tuple[str, str, str]


class LabelPhase(IntEnum):
    LABEL_FREE = 0
    SUPPORT = 1
    ACTION_SURFACE_SEALED = 2
    PSEUDO_RESPONSE = 3
    PRETERMINAL_ATTESTED = 4
    TERMINAL = 5


LabelLoader = Callable[[Sequence[LabelKey], str], Sequence[BinaryLabel]]


# This seal is deliberately process-local and never persisted.  A support-label
# capability is a main-process bearer authority, not a DTO that may be rebuilt
# from arbitrary ``BinaryLabel`` rows or sent to a worker process.
_LABEL_AUTHORITY_SECRET = secrets.token_bytes(32)


def _scope_component(value: object, *, role: str) -> str:
    component = str(value)
    if not component or any(token in component for token in (",", "<", ">")):
        raise ProtocolError(f"P-DCAPS {role} cannot form a canonical label scope.")
    return component


def support_scope(center: object, held_case_id: object) -> str:
    center_id = _scope_component(center, role="support center")
    held = _scope_component(held_case_id, role="support held case")
    if center_id not in CENTERS:
        raise ProtocolError("P-DCAPS support scope center drifted.")
    return f"SUPPORT::<{center_id},{held}>"


def pseudo_response_scope(route_key: RouteKey) -> str:
    if route_key.surface_role != "pseudo":
        raise ProtocolError("P-DCAPS target scope cannot open preterminally.")
    outer = _scope_component(route_key.outer_center, role="pseudo outer center")
    scored = _scope_component(route_key.route_center, role="pseudo scored center")
    held = _scope_component(route_key.held_case_id, role="pseudo held case")
    return f"PSEUDO::<{outer},{scored},{held}>"


def terminal_scope(center: object) -> str:
    center_id = _scope_component(center, role="terminal center")
    if center_id not in CENTERS:
        raise ProtocolError("P-DCAPS terminal scope center drifted.")
    return f"TERMINAL::<{center_id}>"


def _canonical_keys(keys: Sequence[LabelKey]) -> tuple[LabelKey, ...]:
    rows = tuple((str(center), str(case), str(sample)) for center, case, sample in keys)
    if not rows or len(rows) != len(set(rows)):
        raise ProtocolError("P-DCAPS label capability key inventory drifted.")
    return rows


def _support_authority_seal(
    center: str,
    held_case_id: str,
    rows: Sequence[BinaryLabel],
    scope: str,
) -> str:
    payload = canonical_hash(
        {
            "schema_version": "pdcaps_ephemeral_support_authority_v1",
            "center": str(center),
            "held_case_id": str(held_case_id),
            "scope": str(scope),
            "rows": tuple(
                (row.center, row.case_id, row.sample_id, row.value, row.scope)
                for row in rows
            ),
        }
    ).encode("ascii")
    return hmac.new(_LABEL_AUTHORITY_SECRET, payload, hashlib.sha256).hexdigest()


def _pseudo_authority_seal(
    route_key: RouteKey,
    rows: Sequence[BinaryLabel],
    scope: str,
) -> str:
    payload = canonical_hash(
        {
            "schema_version": "pdcaps_ephemeral_pseudo_authority_v1",
            "route_key": route_key.to_payload(),
            "scope": str(scope),
            "rows": tuple(
                (row.center, row.case_id, row.sample_id, row.value, row.scope)
                for row in rows
            ),
        }
    ).encode("ascii")
    return hmac.new(_LABEL_AUTHORITY_SECRET, payload, hashlib.sha256).hexdigest()


def _terminal_authority_seal(
    center: str,
    rows: Sequence[BinaryLabel],
    scope: str,
    preterminal_seal_hash: str,
) -> str:
    payload = canonical_hash(
        {
            "schema_version": "pdcaps_ephemeral_terminal_authority_v1",
            "center": str(center),
            "scope": str(scope),
            "preterminal_seal_hash": str(preterminal_seal_hash),
            "rows": tuple(
                (row.center, row.case_id, row.sample_id, row.value, row.scope)
                for row in rows
            ),
        }
    ).encode("ascii")
    return hmac.new(_LABEL_AUTHORITY_SECRET, payload, hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class SupportLabelCapability:
    center: str
    held_case_id: str
    rows: tuple[BinaryLabel, ...]
    scope: str
    _authority_seal: str | None = field(default=None, repr=False, compare=False)
    key_order_hash: str = field(init=False)
    capability_hash: str = field(init=False)

    def __post_init__(self) -> None:
        center = str(self.center)
        held = str(self.held_case_id)
        rows = tuple(self.rows)
        expected_scope = support_scope(center, held)
        expected_authority = _support_authority_seal(
            center, held, rows, expected_scope
        )
        if (
            self.scope != expected_scope
            or not rows
            or any(row.center != center or row.case_id == held for row in rows)
            or len({row.key for row in rows}) != len(rows)
            or {row.scope for row in rows} != {expected_scope}
            or not hmac.compare_digest(
                str(self._authority_seal), expected_authority
            )
        ):
            raise ProtocolError("P-DCAPS support label capability drifted.")
        key_hash = canonical_hash(tuple(row.key for row in rows))
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "held_case_id", held)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "key_order_hash", key_hash)
        object.__setattr__(
            self,
            "capability_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_support_label_capability_v1",
                    "center": center,
                    "held_case_id": held,
                    "scope": expected_scope,
                    "key_order_hash": key_hash,
                    "row_count": len(rows),
                    "held_case_excluded": True,
                    "firewall_issued": True,
                    "raw_labels_persisted": False,
                }
            ),
        )

    @property
    def values(self) -> tuple[int, ...]:
        return tuple(row.value for row in self.rows)


def require_support_label_capability(
    capability: object,
    *,
    center: object,
    held_case_id: object,
    expected_keys: Sequence[LabelKey],
) -> tuple[BinaryLabel, ...]:
    """Authenticate and bind one support grant to an exact fit inventory."""

    if not isinstance(capability, SupportLabelCapability):
        raise ProtocolError(
            "P-DCAPS support fitting requires a SupportLabelCapability."
        )
    center_id = str(center)
    held = str(held_case_id)
    keys = _canonical_keys(expected_keys)
    expected_scope = support_scope(center_id, held)
    expected_authority = _support_authority_seal(
        capability.center,
        capability.held_case_id,
        capability.rows,
        capability.scope,
    )
    if (
        capability.center != center_id
        or capability.held_case_id != held
        or capability.scope != expected_scope
        or not hmac.compare_digest(
            str(capability._authority_seal), expected_authority
        )
        or tuple(row.key for row in capability.rows) != keys
        or capability.key_order_hash != canonical_hash(keys)
    ):
        raise ProtocolError("P-DCAPS support capability fit binding drifted.")
    return require_exact_label_scope(
        capability.rows,
        expected_keys=keys,
        expected_scope=expected_scope,
    )


@dataclass(frozen=True)
class PseudoResponseLabelCapability:
    route_key: RouteKey
    rows: tuple[BinaryLabel, ...]
    scope: str
    _authority_seal: str | None = field(default=None, repr=False, compare=False)
    evaluation_row_hash: str = field(init=False)
    capability_hash: str = field(init=False)

    def __post_init__(self) -> None:
        rows = tuple(self.rows)
        expected_scope = pseudo_response_scope(self.route_key)
        expected_authority = _pseudo_authority_seal(
            self.route_key, rows, expected_scope
        )
        if (
            self.scope != expected_scope
            or not rows
            or any(
                row.center != self.route_key.route_center
                or row.case_id != self.route_key.held_case_id
                for row in rows
            )
            or len({row.key for row in rows}) != len(rows)
            or {row.scope for row in rows} != {expected_scope}
            or not hmac.compare_digest(
                str(self._authority_seal), expected_authority
            )
        ):
            raise ProtocolError("P-DCAPS pseudo-response label capability drifted.")
        row_hash = canonical_hash(tuple(row.key for row in rows))
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "evaluation_row_hash", row_hash)
        object.__setattr__(
            self,
            "capability_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_pseudo_response_label_capability_v1",
                    "route_key": self.route_key.to_payload(),
                    "scope": expected_scope,
                    "evaluation_row_hash": row_hash,
                    "row_count": len(rows),
                    "target_scope": False,
                    "firewall_issued": True,
                    "raw_labels_persisted": False,
                }
            ),
        )

    @property
    def values(self) -> tuple[int, ...]:
        return tuple(row.value for row in self.rows)


@dataclass(frozen=True)
class TerminalLabelCapability:
    center: str
    rows: tuple[BinaryLabel, ...]
    scope: str
    preterminal_seal_hash: str
    _authority_seal: str | None = field(default=None, repr=False, compare=False)
    key_order_hash: str = field(init=False)
    capability_hash: str = field(init=False)

    def __post_init__(self) -> None:
        center = str(self.center)
        rows = tuple(self.rows)
        expected_scope = terminal_scope(center)
        require_sha256(self.preterminal_seal_hash, "preterminal seal")
        expected_authority = _terminal_authority_seal(
            center, rows, expected_scope, self.preterminal_seal_hash
        )
        if (
            self.scope != expected_scope
            or not rows
            or any(row.center != center for row in rows)
            or len({row.key for row in rows}) != len(rows)
            or {row.scope for row in rows} != {expected_scope}
            or not hmac.compare_digest(
                str(self._authority_seal), expected_authority
            )
        ):
            raise ProtocolError("P-DCAPS terminal label capability drifted.")
        key_hash = canonical_hash(tuple(row.key for row in rows))
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "key_order_hash", key_hash)
        object.__setattr__(
            self,
            "capability_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_terminal_label_capability_v1",
                    "center": center,
                    "scope": expected_scope,
                    "key_order_hash": key_hash,
                    "row_count": len(rows),
                    "preterminal_seal_hash": self.preterminal_seal_hash,
                    "terminal_only": True,
                    "firewall_issued": True,
                    "raw_labels_persisted": False,
                }
            ),
        )

    @property
    def values(self) -> tuple[int, ...]:
        return tuple(row.value for row in self.rows)


@dataclass
class LabelFirewall:
    """Stateful label authority used only through typed capability methods."""

    _loader: LabelLoader = field(repr=False)
    phase: LabelPhase = LabelPhase.LABEL_FREE
    action_surface_seal_hash: str | None = None
    preterminal_seal_hash: str | None = None
    grants: list[dict[str, object]] = field(default_factory=list)

    def advance_support(self) -> None:
        self._advance(LabelPhase.SUPPORT)

    def open_support(
        self,
        *,
        center: str,
        held_case_id: str,
        keys: Sequence[LabelKey],
    ) -> SupportLabelCapability:
        if self.phase != LabelPhase.SUPPORT:
            raise ProtocolError("P-DCAPS support labels opened in the wrong phase.")
        center_id = str(center)
        held = str(held_case_id)
        canonical_keys = _canonical_keys(keys)
        if any(key[0] != center_id or key[1] == held for key in canonical_keys):
            raise ProtocolError("P-DCAPS support capability escaped its held-case scope.")
        scope = support_scope(center_id, held)
        rows = self._read_exact(canonical_keys, scope=scope)
        capability = SupportLabelCapability(
            center_id,
            held,
            rows,
            scope,
            _authority_seal=_support_authority_seal(center_id, held, rows, scope),
        )
        self._record_grant(capability.scope, capability.key_order_hash, len(rows))
        return capability

    def seal_action_surface(self, seal_hash: str) -> None:
        if self.phase != LabelPhase.SUPPORT:
            raise ProtocolError("P-DCAPS action surface cannot be sealed in this phase.")
        require_sha256(str(seal_hash), "action surface seal")
        self.action_surface_seal_hash = str(seal_hash)
        self._advance(LabelPhase.ACTION_SURFACE_SEALED)

    def advance_pseudo_response(self) -> None:
        if self.action_surface_seal_hash is None:
            raise ProtocolError("P-DCAPS pseudo responses require the action seal.")
        self._advance(LabelPhase.PSEUDO_RESPONSE)

    def open_pseudo_response(
        self,
        *,
        route_key: RouteKey,
        sample_ids: Sequence[str],
    ) -> PseudoResponseLabelCapability:
        if self.phase != LabelPhase.PSEUDO_RESPONSE:
            raise ProtocolError("P-DCAPS pseudo responses opened in the wrong phase.")
        scope = pseudo_response_scope(route_key)
        samples = tuple(str(value) for value in sample_ids)
        keys = _canonical_keys(
            tuple(
                (route_key.route_center, route_key.held_case_id, sample_id)
                for sample_id in samples
            )
        )
        rows = self._read_exact(keys, scope=scope)
        capability = PseudoResponseLabelCapability(
            route_key,
            rows,
            scope,
            _authority_seal=_pseudo_authority_seal(route_key, rows, scope),
        )
        self._record_grant(
            capability.scope, capability.evaluation_row_hash, len(rows)
        )
        return capability

    def attest_preterminal(self, seal_hash: str) -> None:
        if self.phase != LabelPhase.PSEUDO_RESPONSE:
            raise ProtocolError("P-DCAPS preterminal attestation phase drifted.")
        require_sha256(str(seal_hash), "preterminal seal")
        self.preterminal_seal_hash = str(seal_hash)
        self._advance(LabelPhase.PRETERMINAL_ATTESTED)

    def open_terminal(self) -> None:
        if self.preterminal_seal_hash is None:
            raise ProtocolError("P-DCAPS terminal labels require an attested seal.")
        self._advance(LabelPhase.TERMINAL)

    def open_terminal_labels(
        self,
        *,
        center: str,
        keys: Sequence[LabelKey],
    ) -> TerminalLabelCapability:
        if self.phase != LabelPhase.TERMINAL or self.preterminal_seal_hash is None:
            raise ProtocolError("P-DCAPS target labels are terminal-only.")
        center_id = str(center)
        canonical_keys = _canonical_keys(keys)
        if any(key[0] != center_id for key in canonical_keys):
            raise ProtocolError("P-DCAPS terminal label capability crossed centers.")
        scope = terminal_scope(center_id)
        rows = self._read_exact(canonical_keys, scope=scope)
        capability = TerminalLabelCapability(
            center_id,
            rows,
            scope,
            self.preterminal_seal_hash,
            _authority_seal=_terminal_authority_seal(
                center_id, rows, scope, self.preterminal_seal_hash
            ),
        )
        self._record_grant(capability.scope, capability.key_order_hash, len(rows))
        return capability

    def audit_payload(self) -> dict[str, object]:
        payload = {
            "schema_version": "pdcaps_label_firewall_v2",
            "phase": self.phase.name,
            "action_surface_seal_hash": self.action_surface_seal_hash,
            "preterminal_seal_hash": self.preterminal_seal_hash,
            "grant_count": len(self.grants),
            "grants": list(self.grants),
            "generic_label_read_available": False,
            "target_preterminal_grants": 0,
            "raw_labels_persisted": False,
        }
        return {**payload, "firewall_hash": canonical_hash(payload)}

    def _read_exact(
        self, keys: tuple[LabelKey, ...], *, scope: str
    ) -> tuple[BinaryLabel, ...]:
        rows = require_exact_label_scope(
            self._loader(keys, scope),
            expected_keys=keys,
            expected_scope=scope,
        )
        by_key = {row.key: row for row in rows}
        return tuple(by_key[key] for key in keys)

    def _record_grant(self, scope: str, key_hash: str, row_count: int) -> None:
        self.grants.append(
            {
                "scope": str(scope),
                "phase": self.phase.name,
                "row_count": int(row_count),
                "key_hash": str(key_hash),
                "raw_labels_persisted": False,
            }
        )

    def _advance(self, target: LabelPhase) -> None:
        if int(target) != int(self.phase) + 1:
            raise ProtocolError("P-DCAPS label firewall phase transition drifted.")
        self.phase = target


__all__ = (
    "LabelFirewall",
    "LabelKey",
    "LabelLoader",
    "LabelPhase",
    "PseudoResponseLabelCapability",
    "SupportLabelCapability",
    "TerminalLabelCapability",
    "pseudo_response_scope",
    "require_support_label_capability",
    "support_scope",
    "terminal_scope",
)
