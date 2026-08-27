"""Transient exact-center terminal label receipts for pseudo utility replay."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field

from ..hashing import canonical_hash, require_sha256
from ..protocol import ProtocolError
from ..replay_scope import PseudoReplayScope


_TERMINAL_RECEIPT_FACTORY_TOKEN = object()


@dataclass(frozen=True, slots=True)
class TerminalCaseLabelInput:
    """Ephemeral labels for one exact manifest-bound whole case."""

    case_id: str
    ordered_sample_keys: tuple[tuple[str, str, str], ...]
    labels: tuple[int, ...] = field(repr=False)

    def __post_init__(self) -> None:
        case = str(self.case_id)
        keys = tuple(tuple(str(value) for value in key) for key in self.ordered_sample_keys)
        try:
            labels = tuple(int(value) for value in self.labels)
        except (TypeError, ValueError) as exc:
            raise ProtocolError("SCALE-BP terminal case labels drifted.") from exc
        if (
            not case
            or not keys
            or keys != tuple(sorted(set(keys)))
            or any(len(key) != 3 or key[1] != case for key in keys)
            or len(labels) != len(keys)
            or any(value not in {0, 1} for value in labels)
        ):
            raise ProtocolError("SCALE-BP terminal case-label input drifted.")
        object.__setattr__(self, "case_id", case)
        object.__setattr__(self, "ordered_sample_keys", keys)
        object.__setattr__(self, "labels", labels)

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise ProtocolError("SCALE-BP terminal case labels may not be serialized.")


@dataclass(frozen=True, slots=True)
class TerminalCaseLabelReceipt:
    """Held-d labels plus denominators derived from the complete pseudo center."""

    scope: PseudoReplayScope
    terminal_labels: tuple[int, ...] = field(repr=False)
    terminal_label_hash: str
    center_population_label_hash: str
    positive_denominator: int
    negative_denominator: int
    row_denominator: int
    case_label_hashes: tuple[tuple[str, str], ...]
    _factory_token: InitVar[object] = None
    receipt_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _TERMINAL_RECEIPT_FACTORY_TOKEN:
            raise ProtocolError(
                "SCALE-BP terminal label receipt bypassed the exact-center loader."
            )
        labels = tuple(int(value) for value in self.terminal_labels)
        label_hash = require_sha256(self.terminal_label_hash, "terminal label hash")
        population_hash = require_sha256(
            self.center_population_label_hash,
            "terminal center-population label hash",
        )
        positive = int(self.positive_denominator)
        negative = int(self.negative_denominator)
        total = int(self.row_denominator)
        hashes = tuple((str(case), str(digest)) for case, digest in self.case_label_hashes)
        for _case, digest in hashes:
            require_sha256(digest, "terminal case-label hash")
        if (
            not isinstance(self.scope, PseudoReplayScope)
            or len(labels) != self.scope.route_witness.evaluation_binding.row_count
            or any(value not in {0, 1} for value in labels)
            or positive <= 0
            or negative <= 0
            or total != positive + negative
            or tuple(case for case, _digest in hashes)
            != self.scope.case_inventory.cases(self.scope.pseudo_center)
            or len(set(hashes)) != len(hashes)
        ):
            raise ProtocolError("SCALE-BP terminal label receipt drifted.")
        payload = {
            "schema_version": "scale_bp_terminal_case_label_receipt_v1",
            "scope_hash": self.scope.scope_hash,
            "evaluation_sample_key_hash": (
                self.scope.route_witness.evaluation_binding.sample_key_hash
            ),
            "terminal_label_hash": label_hash,
            "center_population_label_hash": population_hash,
            "positive_denominator": positive,
            "negative_denominator": negative,
            "row_denominator": total,
            "case_label_hashes": hashes,
            "raw_labels_persisted": False,
        }
        object.__setattr__(self, "terminal_labels", labels)
        object.__setattr__(self, "terminal_label_hash", label_hash)
        object.__setattr__(self, "center_population_label_hash", population_hash)
        object.__setattr__(self, "positive_denominator", positive)
        object.__setattr__(self, "negative_denominator", negative)
        object.__setattr__(self, "row_denominator", total)
        object.__setattr__(self, "case_label_hashes", hashes)
        object.__setattr__(self, "receipt_hash", canonical_hash(payload))

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise ProtocolError(
            "SCALE-BP transient terminal label receipts may not be serialized."
        )


def load_terminal_case_label_receipt(
    scope: PseudoReplayScope,
    center_case_labels: object,
) -> TerminalCaseLabelReceipt:
    """Derive held labels and BACC denominators from every exact case in J."""

    if not isinstance(scope, PseudoReplayScope):
        raise ProtocolError("SCALE-BP terminal label scope drifted.")
    rows = tuple(center_case_labels)  # type: ignore[arg-type]
    if any(not isinstance(row, TerminalCaseLabelInput) for row in rows):
        raise ProtocolError("SCALE-BP terminal center-label population drifted.")
    expected_cases = scope.case_inventory.cases(scope.pseudo_center)
    if tuple(row.case_id for row in rows) != expected_cases:
        raise ProtocolError("SCALE-BP terminal center-label universe is incomplete.")
    case_hashes = []
    positive = 0
    total = 0
    held_labels: tuple[int, ...] | None = None
    for row in rows:
        binding = scope.route_witness.identity_inventory.binding(
            scope.pseudo_center, row.case_id
        )
        key_hash = canonical_hash(
            {
                "schema_version": "scale_bp_case_sample_keys_v1",
                "keys": row.ordered_sample_keys,
            }
        )
        if (
            len(row.labels) != binding.row_count
            or key_hash != binding.sample_key_hash
            or any(
                key[:2] != (scope.pseudo_center, row.case_id)
                for key in row.ordered_sample_keys
            )
        ):
            raise ProtocolError("SCALE-BP terminal case-label lineage drifted.")
        digest = canonical_hash(
            {
                "schema_version": "scale_bp_terminal_case_labels_v1",
                "center": scope.pseudo_center,
                "case_id": row.case_id,
                "sample_key_hash": key_hash,
                "row_count": len(row.labels),
                "values": row.labels,
            }
        )
        case_hashes.append((row.case_id, digest))
        positive += sum(row.labels)
        total += len(row.labels)
        if row.case_id == scope.held_case_id:
            held_labels = row.labels
    negative = total - positive
    if held_labels is None or positive <= 0 or negative <= 0:
        raise ProtocolError("SCALE-BP terminal center lacks both outcome classes.")
    held_hash = canonical_hash(
        {
            "schema_version": "scale_bp_binary_terminal_label_vector_v2",
            "scope_hash": scope.scope_hash,
            "sample_key_hash": scope.route_witness.evaluation_binding.sample_key_hash,
            "row_count": len(held_labels),
            "values": held_labels,
        }
    )
    population_hash = canonical_hash(
        {
            "schema_version": "scale_bp_terminal_center_label_population_v1",
            "pseudo_center": scope.pseudo_center,
            "case_label_hashes": tuple(case_hashes),
            "positive_denominator": positive,
            "negative_denominator": negative,
            "row_denominator": total,
        }
    )
    return TerminalCaseLabelReceipt(
        scope=scope,
        terminal_labels=held_labels,
        terminal_label_hash=held_hash,
        center_population_label_hash=population_hash,
        positive_denominator=positive,
        negative_denominator=negative,
        row_denominator=total,
        case_label_hashes=tuple(case_hashes),
        _factory_token=_TERMINAL_RECEIPT_FACTORY_TOKEN,
    )


__all__ = (
    "TerminalCaseLabelInput",
    "TerminalCaseLabelReceipt",
    "load_terminal_case_label_receipt",
)
