"""In-memory development provenance for the disagreement-regret core.

The mathematical core is deliberately stage-neutral.  It may be developed
with synthetic fixtures or with an independently authorized source-only OOF
surface.  Consumed evaluation evidence, target data, and stage-bound records are
not representable as accepted scopes here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re

from midogpp_thesis.cvae.protocol import ProtocolError


_LOWERCASE_SHA256 = re.compile(r"[0-9a-f]{64}", flags=re.ASCII)


class DevelopmentScope(str, Enum):
    """The evidence regimes admitted by the pure development core."""

    SYNTHETIC_TEST = "SYNTHETIC_TEST"
    AUTHORIZED_SOURCE_OOF = "AUTHORIZED_SOURCE_OOF"
    AUTHORIZED_POSTHOC_SOURCE_OOF = "AUTHORIZED_POSTHOC_SOURCE_OOF"


@dataclass(frozen=True, kw_only=True)
class DevelopmentContext:
    """Fail-closed authority carried alongside in-memory development values.

    ``authorization_unused`` has no default affirmative value.  Callers using
    source OOF evidence must explicitly attest that the predeclared authority
    has not already been consumed.  Synthetic callers leave both authorization
    fields as ``None``.
    """

    scope: DevelopmentScope | str
    dataset_family: str
    outer_target_id: str
    authorization_hash: str | None = None
    authorization_unused: bool | None = None
    authorized_query_ids: tuple[str, ...] = ()
    authorized_sample_keys_hash: str | None = None
    source_evidence_previously_consumed: bool = False
    consumed_data: bool = False
    target_labels_available: bool = False

    def __post_init__(self) -> None:
        canonical_scope = _canonical_scope(self.scope)
        object.__setattr__(self, "scope", canonical_scope)
        assert_development_context(self)

    def to_payload(self) -> dict[str, object]:
        """Return a stage-neutral, serialization-ready description.

        This method only builds an in-memory dictionary.  Persistence belongs
        to an authorized adapter outside this package.
        """

        assert_development_context(self)
        return {
            "schema_version": "midogpp_disagreement_regret_development_context_v1",
            "scope": self.scope.value,
            "dataset_family": self.dataset_family,
            "outer_target_id": self.outer_target_id,
            "authorization_hash": self.authorization_hash,
            "authorization_unused": self.authorization_unused,
            "authorized_query_ids": list(self.authorized_query_ids),
            "authorized_sample_keys_hash": self.authorized_sample_keys_hash,
            "source_evidence_previously_consumed": (
                self.source_evidence_previously_consumed
            ),
            "consumed_data": False,
            "target_labels_available": False,
            "in_memory_only": True,
        }


def assert_development_context(context: DevelopmentContext) -> DevelopmentContext:
    """Validate that ``context`` cannot carry consumed or target evidence."""

    if not isinstance(context, DevelopmentContext):
        raise ProtocolError("Development context must use the locked context type.")
    if not isinstance(context.scope, DevelopmentScope) or context.scope not in (
        DevelopmentScope.SYNTHETIC_TEST,
        DevelopmentScope.AUTHORIZED_SOURCE_OOF,
        DevelopmentScope.AUTHORIZED_POSTHOC_SOURCE_OOF,
    ):
        raise ProtocolError(
            "Development scope must be synthetic or an explicitly authorized "
            "source-OOF regime; target, Stage-70, and Stage-90 scopes are forbidden."
        )
    _require_nonempty_text(context.dataset_family, "dataset_family")
    _require_nonempty_text(context.outer_target_id, "outer_target_id")
    if context.consumed_data is not False:
        raise ProtocolError("Consumed data cannot enter the disagreement-regret core.")
    if context.target_labels_available is not False:
        raise ProtocolError("Target labels cannot enter the disagreement-regret core.")

    if context.scope in (
        DevelopmentScope.AUTHORIZED_SOURCE_OOF,
        DevelopmentScope.AUTHORIZED_POSTHOC_SOURCE_OOF,
    ):
        forbidden_family_fragments = (
            "CONSUMED",
            "TEST",
            "TARGET",
            "VALIDATION",
            "EVALUATION",
            "STAGE70",
            "STAGE-70",
            "STAGE90",
            "STAGE-90",
            "QUARANTINE",
            "HISTORICAL",
        )
        if any(
            fragment in context.dataset_family.upper()
            for fragment in forbidden_family_fragments
        ):
            raise ProtocolError(
                "Authorized source OOF dataset family cannot name consumed, target, "
                "evaluation, Stage-70, Stage-90, quarantine, or historical evidence."
            )
        if (
            type(context.authorization_hash) is not str
            or _LOWERCASE_SHA256.fullmatch(context.authorization_hash) is None
        ):
            raise ProtocolError(
                "Authorized source OOF requires a lowercase 64-hex predeclared "
                "authorization hash."
            )
        if context.scope is DevelopmentScope.AUTHORIZED_SOURCE_OOF:
            if context.authorization_unused is not True:
                raise ProtocolError(
                    "Fresh authorized source OOF requires explicit unused authorization status."
                )
            if context.source_evidence_previously_consumed is not False:
                raise ProtocolError(
                    "Fresh source OOF cannot claim previously consumed evidence."
                )
        else:
            if context.authorization_unused is not False:
                raise ProtocolError(
                    "Posthoc source OOF requires explicit already-used authorization status."
                )
            if context.source_evidence_previously_consumed is not True:
                raise ProtocolError(
                    "Posthoc source OOF must attest previously consumed source evidence."
                )
        query_ids = tuple(str(value) for value in context.authorized_query_ids)
        if (
            len(query_ids) < 3
            or query_ids != tuple(sorted(query_ids))
            or len(set(query_ids)) != len(query_ids)
            or any(not value or value != value.strip() for value in query_ids)
            or context.outer_target_id in query_ids
        ):
            raise ProtocolError(
                "Authorized source OOF requires a canonical donor-query allowlist."
            )
        if (
            type(context.authorized_sample_keys_hash) is not str
            or _LOWERCASE_SHA256.fullmatch(context.authorized_sample_keys_hash) is None
        ):
            raise ProtocolError(
                "Authorized source OOF requires a sealed sample-key allowlist hash."
            )
        object.__setattr__(context, "authorized_query_ids", query_ids)
    else:
        if context.dataset_family != "SYNTHETIC":
            raise ProtocolError(
                "Synthetic development requires the exact SYNTHETIC dataset family; "
                "real, target, consumed, Stage-70, and Stage-90 families are forbidden."
            )
        if (
            context.authorization_hash is not None
            or context.authorization_unused is not None
            or context.authorized_query_ids
            or context.authorized_sample_keys_hash is not None
            or context.source_evidence_previously_consumed is not False
        ):
            raise ProtocolError(
                "Synthetic development cannot carry source-OOF authorization metadata."
            )
        object.__setattr__(context, "authorized_query_ids", ())
    return context


def _canonical_scope(value: DevelopmentScope | str) -> DevelopmentScope:
    if isinstance(value, DevelopmentScope):
        return value
    if type(value) is not str:
        raise ProtocolError("Development scope must be an exact string identity.")
    try:
        return DevelopmentScope(value)
    except ValueError as exc:
        raise ProtocolError(
            "Development scope must be SYNTHETIC_TEST, AUTHORIZED_SOURCE_OOF, "
            "or AUTHORIZED_POSTHOC_SOURCE_OOF; target and Stage aliases are forbidden."
        ) from exc


def _require_nonempty_text(value: object, name: str) -> None:
    if type(value) is not str or not value or value != value.strip():
        raise ProtocolError(f"{name} must be a non-empty canonical string.")


__all__ = (
    "DevelopmentContext",
    "DevelopmentScope",
    "assert_development_context",
)
