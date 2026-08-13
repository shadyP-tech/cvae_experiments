"""Validator-only B-to-C provenance for terminal finalization recovery."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json, sha256_file
from .hashing import canonical_hash
from .recovery_provenance import (
    current_repair_repository_state,
    fresh_recovery_audit_payload,
    is_hex,
    original_repository_state_from_provenance,
    validated_repository_state,
)
from .terminal_schema import TERMINAL_TABLE_FIELDS, TERMINAL_TABLE_MEMBERS


_FINALIZATION_AUDIT_SCHEMA = (
    "midogpp_fixed_bank_multi_challenger_finalization_recovery_audit_v1"
)
_FINALIZATION_TABLE_MEMBERS = tuple(TERMINAL_TABLE_MEMBERS.values())
_FINALIZATION_FIELDS_BY_MEMBER = {
    member: TERMINAL_TABLE_FIELDS[key]
    for key, member in TERMINAL_TABLE_MEMBERS.items()
}


def finalization_recovery_audit_payload_for_root(
    root: Path,
    *,
    failed_state: Mapping[str, object],
    mappingproxy_recovery_audit: Mapping[str, object],
    current_repository_state: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Reconstruct the validator-only B->C finalization-recovery audit.

    This audit deliberately does not live in ``runtime_summary.json`` or the
    content index: those are B-era scientific products.  It is a validation
    claim about a deterministic serialization repair, and is recomputed by
    each fresh validator from the final, indexed files and the immutable
    mappingproxy audit embedded in the runtime summary.
    """

    prior = _prior_scientific_repository_state(
        root, mappingproxy_recovery_audit
    )
    current = validated_repository_state(
        current_repository_state or current_repair_repository_state(),
        role="finalization repair",
        require_clean=False,
    )
    bindings = finalization_recovery_surface_bindings(root)
    if prior == current:
        return _finalization_audit(
            recovery_used=False,
            failed_state=None,
            prior_mappingproxy_recovery_audit=mappingproxy_recovery_audit,
            prior_repository_state=prior,
            repair_repository_state=None,
            bindings=bindings,
        )
    if prior["repository_revision"] == current["repository_revision"]:
        raise ProtocolError(
            "Finalization recovery checkout state drifted without a repair revision."
        )
    current = validated_repository_state(
        current,
        role="finalization repair",
        require_clean=True,
    )
    return _finalization_audit(
        recovery_used=True,
        failed_state=failed_state,
        prior_mappingproxy_recovery_audit=mappingproxy_recovery_audit,
        prior_repository_state=prior,
        repair_repository_state=current,
        bindings=bindings,
    )


def validate_finalization_recovery_audit_payload(
    value: object,
    root: Path,
    *,
    failed_state: Mapping[str, object],
    mappingproxy_recovery_audit: Mapping[str, object],
    current_repository_state: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Require the report-only finalization audit to match the final bundle."""

    if not isinstance(value, Mapping):
        raise ProtocolError("Multi-challenger finalization recovery audit is absent.")
    expected = finalization_recovery_audit_payload_for_root(
        root,
        failed_state=failed_state,
        mappingproxy_recovery_audit=mappingproxy_recovery_audit,
        current_repository_state=current_repository_state,
    )
    if dict(value) != dict(expected):
        raise ProtocolError("Multi-challenger finalization recovery audit drifted.")
    return dict(value)


def assert_finalization_repair_repository_state_unchanged(
    audit: Mapping[str, object],
) -> None:
    """Ensure the clean C checkout did not change after finalization recovery."""

    if audit.get("finalization_recovery_used") is not True:
        return
    expected = validated_repository_state(
        {
            "repository_revision": audit.get("repair_repository_revision"),
            "repository_dirty": audit.get("repair_repository_dirty"),
            "repository_status_hash": audit.get("repair_repository_status_hash"),
        },
        role="finalization repair",
        require_clean=True,
    )
    if dict(current_repair_repository_state()) != expected:
        raise ProtocolError(
            "Finalization recovery repair checkout changed during continuation."
        )


def fresh_finalization_recovery_audit_payload() -> Mapping[str, object]:
    """Canonical no-C-recovery marker for focused callers and fixtures.

    The in-bundle no-C audit additionally binds B and the indexed terminal
    tables, so validators always use :func:`finalization_recovery_audit_payload_for_root`.
    """

    unhashed = {
        "schema_version": _FINALIZATION_AUDIT_SCHEMA,
        "finalization_recovery_used": False,
        "terminal_consumed_test_diagnostic_only": True,
        "policy_promotion_authorized": False,
    }
    return {**unhashed, "finalization_recovery_audit_hash": canonical_hash(unhashed)}


def finalization_recovery_surface_bindings(root: Path) -> Mapping[str, object]:
    """Bind C to the indexed final terminal products, including CSV headers."""

    path = Path(root)
    index_path = path / "manifests/content_index.json"
    if index_path.is_symlink() or not index_path.is_file():
        raise ProtocolError("Finalization recovery content index is absent or unsafe.")
    index = read_json(index_path)
    index_hash = index.get("content_hash")
    if not is_hex(index_hash, length=64):
        raise ProtocolError("Finalization recovery content index hash is invalid.")
    indexed_rows = index.get("members")
    if not isinstance(indexed_rows, list):
        raise ProtocolError("Finalization recovery content index rows are invalid.")
    indexed_sha = {
        str(row.get("member")): row.get("sha256")
        for row in indexed_rows
        if isinstance(row, Mapping)
    }
    tables: dict[str, object] = {}
    for member in _FINALIZATION_TABLE_MEMBERS:
        table = path / member
        if table.is_symlink() or not table.is_file():
            raise ProtocolError(
                "Finalization recovery terminal table is absent or unsafe."
            )
        file_hash = sha256_file(table)
        header = _csv_header(table)
        if (
            not header
            or header != _FINALIZATION_FIELDS_BY_MEMBER[member]
            or indexed_sha.get(member) != file_hash
        ):
            raise ProtocolError("Finalization recovery terminal table index drifted.")
        tables[member] = {
            "sha256": file_hash,
            "header": list(header),
        }
    return {
        "content_index_hash": index_hash,
        "content_index_file_sha256": sha256_file(index_path),
        "terminal_table_surfaces": tables,
    }


def _finalization_audit(
    *,
    recovery_used: bool,
    failed_state: Mapping[str, object] | None,
    prior_mappingproxy_recovery_audit: Mapping[str, object],
    prior_repository_state: Mapping[str, object],
    repair_repository_state: Mapping[str, object] | None,
    bindings: Mapping[str, object],
) -> Mapping[str, object]:
    failed = dict(failed_state or {})
    prior = validated_repository_state(
        prior_repository_state, role="prior scientific checkout", require_clean=False
    )
    repair = repair_repository_state or {}
    unhashed = {
        "schema_version": _FINALIZATION_AUDIT_SCHEMA,
        "finalization_recovery_used": recovery_used,
        "failed_run_state_hash": canonical_hash(failed) if failed else None,
        "failed_status": failed.get("status") if failed else None,
        "failed_phase": failed.get("phase") if failed else None,
        "failed_error": failed.get("error") if failed else None,
        "failed_error_class": failed.get("error_class") if failed else None,
        "prior_mappingproxy_recovery_audit_hash": prior_mappingproxy_recovery_audit.get(
            "recovery_audit_hash"
        ),
        "prior_repository_revision": prior["repository_revision"],
        "prior_repository_dirty": prior["repository_dirty"],
        "prior_repository_status_hash": prior["repository_status_hash"],
        "repair_repository_revision": repair.get("repository_revision"),
        "repair_repository_dirty": repair.get("repository_dirty"),
        "repair_repository_status_hash": repair.get("repository_status_hash"),
        "content_index_hash": bindings.get("content_index_hash"),
        "content_index_file_sha256": bindings.get("content_index_file_sha256"),
        "terminal_table_surfaces": bindings.get("terminal_table_surfaces"),
        "source_generation_recomputed_during_finalization_recovery": False,
        "predictions_recomputed_during_finalization_recovery": False,
        "donor_models_recomputed_during_finalization_recovery": False,
        "decisions_recomputed_during_finalization_recovery": False,
        "terminal_evaluation_recomputed_during_finalization_recovery": False,
        "terminal_table_rows_reordered_during_finalization_recovery": False,
        "terminal_table_bytes_changed_during_finalization_recovery": False,
        "indexed_scientific_bytes_changed_during_finalization_recovery": False,
        "terminal_schema_validation_contract_repaired_by_checkout": recovery_used,
        "labels_reopened_only_for_read_only_validation_during_finalization_recovery": (
            recovery_used
        ),
        "policy_mutated_during_finalization_recovery": False,
        "raw_labels_persisted": False,
        "terminal_consumed_test_diagnostic_only": True,
        "policy_promotion_authorized": False,
    }
    _validate_finalization_audit_shape(unhashed, recovery_used=recovery_used)
    return {**unhashed, "finalization_recovery_audit_hash": canonical_hash(unhashed)}


def _prior_scientific_repository_state(
    root: Path,
    audit: Mapping[str, object],
) -> dict[str, object]:
    if not is_hex(audit.get("recovery_audit_hash"), length=64):
        raise ProtocolError(
            "Finalization recovery requires a persisted mappingproxy audit."
        )
    if audit.get("recovery_used") is False:
        if dict(audit) != dict(fresh_recovery_audit_payload()):
            raise ProtocolError("Fresh mappingproxy audit drifted before finalization.")
        return dict(original_repository_state_from_provenance(root))
    if audit.get("recovery_used") is not True:
        raise ProtocolError("Mappingproxy recovery mode drifted before finalization.")
    return validated_repository_state(
        {
            "repository_revision": audit.get("repair_repository_revision"),
            "repository_dirty": audit.get("repair_repository_dirty"),
            "repository_status_hash": audit.get("repair_repository_status_hash"),
        },
        role="prior mappingproxy repair",
        require_clean=True,
    )


def _csv_header(path: Path) -> tuple[str, ...]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ProtocolError("Finalization recovery CSV header is unreadable.") from exc
    if (
        not isinstance(header, list)
        or not header
        or any(not isinstance(field, str) or not field for field in header)
        or len(set(header)) != len(header)
    ):
        raise ProtocolError("Finalization recovery CSV header is invalid.")
    return tuple(header)


def _validate_finalization_audit_shape(
    payload: Mapping[str, object],
    *,
    recovery_used: bool,
) -> None:
    if (
        payload.get("schema_version") != _FINALIZATION_AUDIT_SCHEMA
        or payload.get("finalization_recovery_used") is not recovery_used
        or not is_hex(payload.get("prior_mappingproxy_recovery_audit_hash"), length=64)
        or not is_hex(payload.get("content_index_hash"), length=64)
        or not is_hex(payload.get("content_index_file_sha256"), length=64)
        or not isinstance(payload.get("terminal_table_surfaces"), Mapping)
        or set(payload["terminal_table_surfaces"]) != set(_FINALIZATION_TABLE_MEMBERS)
        or payload.get("source_generation_recomputed_during_finalization_recovery")
        is not False
        or payload.get("predictions_recomputed_during_finalization_recovery")
        is not False
        or payload.get("donor_models_recomputed_during_finalization_recovery")
        is not False
        or payload.get("decisions_recomputed_during_finalization_recovery")
        is not False
        or payload.get("terminal_evaluation_recomputed_during_finalization_recovery")
        is not False
        or payload.get("terminal_table_rows_reordered_during_finalization_recovery")
        is not False
        or payload.get("terminal_table_bytes_changed_during_finalization_recovery")
        is not False
        or payload.get("indexed_scientific_bytes_changed_during_finalization_recovery")
        is not False
        or payload.get("terminal_schema_validation_contract_repaired_by_checkout")
        is not recovery_used
        or payload.get(
            "labels_reopened_only_for_read_only_validation_during_finalization_recovery"
        )
        is not recovery_used
        or payload.get("policy_mutated_during_finalization_recovery") is not False
        or payload.get("raw_labels_persisted") is not False
        or payload.get("terminal_consumed_test_diagnostic_only") is not True
        or payload.get("policy_promotion_authorized") is not False
    ):
        raise ProtocolError("Finalization recovery audit topology is invalid.")
    for surface in payload["terminal_table_surfaces"].values():
        if (
            not isinstance(surface, Mapping)
            or not is_hex(surface.get("sha256"), length=64)
            or not isinstance(surface.get("header"), list)
            or not surface["header"]
            or len(set(surface["header"])) != len(surface["header"])
        ):
            raise ProtocolError(
                "Finalization recovery terminal table binding is invalid."
            )
    if recovery_used:
        failure = {
            "failed_run_state_hash": payload.get("failed_run_state_hash"),
            "failed_status": payload.get("failed_status"),
            "failed_phase": payload.get("failed_phase"),
            "failed_error": payload.get("failed_error"),
            "failed_error_class": payload.get("failed_error_class"),
        }
        if (
            not is_hex(failure["failed_run_state_hash"], length=64)
            or failure["failed_status"] != "FAILED"
            or failure["failed_phase"] != "FINALIZATION"
            or not isinstance(failure["failed_error"], str)
            or failure["failed_error_class"] != "ProtocolError"
        ):
            raise ProtocolError("Finalization recovery failure state is invalid.")
        validated_repository_state(
            {
                "repository_revision": payload.get("repair_repository_revision"),
                "repository_dirty": payload.get("repair_repository_dirty"),
                "repository_status_hash": payload.get("repair_repository_status_hash"),
            },
            role="finalization repair",
            require_clean=True,
        )
    elif any(
        payload.get(key) is not None
        for key in (
            "failed_run_state_hash",
            "failed_status",
            "failed_phase",
            "failed_error",
            "failed_error_class",
            "repair_repository_revision",
            "repair_repository_dirty",
            "repair_repository_status_hash",
        )
    ):
        raise ProtocolError("Fresh finalization audit carries recovery state.")


__all__ = (
    "assert_finalization_repair_repository_state_unchanged",
    "finalization_recovery_audit_payload_for_root",
    "finalization_recovery_surface_bindings",
    "fresh_finalization_recovery_audit_payload",
    "validate_finalization_recovery_audit_payload",
)
