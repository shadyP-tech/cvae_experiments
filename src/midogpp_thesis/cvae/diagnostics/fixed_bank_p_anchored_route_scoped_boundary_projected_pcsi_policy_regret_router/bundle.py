"""Closed-world bundle inventory and content-first integrity index."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json, sha256_file
from .artifact_io import persist_json
from .constants import PUBLICATION_STATUS, TERMINAL_DECISION
from .hashing import canonical_hash


REQUIRED_FILES = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "arrays/frozen_source_streams.npy",
    "arrays/fixed_bank_a1_action_probabilities.npz",
    "manifests/frozen_source_stream_index.json",
    "manifests/frozen_source_stream_lock.json",
    "manifests/fixed_bank_a1_prediction_index.json",
    "manifests/fixed_bank_a1_prediction_seal.json",
    "manifests/protocol_manifest.json",
    "manifests/action_library.json",
    "manifests/physical_surface_seal.json",
    "manifests/outer_plan_seal.json",
    "manifests/donor_runtime.json",
    "manifests/policy_replay_runtime.json",
    "manifests/policy_menu.json",
    "manifests/decision_barrier.json",
    "manifests/preterminal_aggregate_seal.json",
    "manifests/terminal_evaluation_seal.json",
    "manifests/content_index.json",
    "tables/exact_nine_probability_index.json",
    "tables/outer_plans.json",
    "tables/double_exclusion_plans.json",
    "tables/physical_fingerprints.json",
    "tables/target_local_posterior_models.json",
    "tables/target_local_posterior_predictions.json",
    "tables/action_equivalence_classes.json",
    "tables/projected_utility_descriptors.json",
    "tables/projected_donor_utility_rows.json",
    "tables/double_excluded_prior_provenance.json",
    "tables/double_excluded_endpoint_scopes.json",
    "tables/pseudo_donor_utility_rows.json",
    "tables/projected_utility_models.json",
    "tables/projected_utility_predictions.json",
    "tables/transport_reference_blocks.json",
    "tables/descriptor_matches.json",
    "tables/donor_case_envelopes.json",
    "tables/route_calibrations.json",
    "tables/sample_influence_predictions.json",
    "tables/transport_descriptors.json",
    "tables/transport_screens.json",
    "tables/target_candidate_policies.json",
    "tables/pseudo_candidate_policies.json",
    "tables/policy_regret_replays.json",
    "tables/policy_authorizations.json",
    "tables/final_policy_predictions.json",
    "tables/route_decisions.json",
    "tables/terminal_method_metrics.json",
    "tables/terminal_center_contrasts.json",
    "tables/terminal_case_oracles.json",
    "tables/terminal_projected_action_diagnostics.json",
    "tables/terminal_policy_regret_diagnostics.json",
    "tables/terminal_transport_diagnostics.json",
    "tables/terminal_selected_case_diagnostics.json",
    "tables/terminal_policy_regret_centers.json",
    "tables/terminal_action_frequencies.json",
    "tables/terminal_diagnostic.json",
    "tables/selection_control.json",
    "reports/workstation_preflight.json",
    "reports/diagnostic_summary.json",
    "reports/label_capability_report.json",
    "reports/leakage_report.json",
    "reports/publication_decision.json",
    "reports/runtime_summary.json",
    "reports/phase_telemetry.json",
    "reports/run_state.json",
    "reports/fresh_process_attestation.json",
    "reports/validation_report.json",
)
INDEX_EXCLUDED = frozenset(
    {
        "manifests/content_index.json",
        "reports/run_state.json",
        "reports/fresh_process_attestation.json",
        "reports/validation_report.json",
    }
)
CONTENT_INDEX_MEMBERS = tuple(
    member for member in REQUIRED_FILES if member not in INDEX_EXCLUDED
)
ALLOWED_RELATIVE_DIRECTORIES = frozenset(
    Path(member).parent.as_posix()
    for member in REQUIRED_FILES
    if Path(member).parent.as_posix() != "."
)


def relative_files(root: Path) -> tuple[str, ...]:
    _reject_symlinks(root)
    return tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path != root / ".run.lock"
        )
    )


def assert_closed_world(
    root: Path,
    *,
    allow_incomplete: bool = False,
    allow_pending_validation: bool = False,
) -> None:
    _reject_symlinks(root)
    if root.exists():
        relative_directories = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_dir()
        }
        if not relative_directories <= ALLOWED_RELATIVE_DIRECTORIES:
            raise ProtocolError("PCSI-RACR bundle contains a foreign directory.")
    observed = set(relative_files(root))
    required = set(REQUIRED_FILES)
    extras = sorted(observed - required)
    if allow_incomplete:
        missing: list[str] = []
    else:
        allowed_missing = (
            {
                "reports/fresh_process_attestation.json",
                "reports/validation_report.json",
            }
            if allow_pending_validation
            else set()
        )
        missing = sorted(required - observed - allowed_missing)
    if extras or missing:
        raise ProtocolError(
            f"PCSI-RACR closed-world drifted: extras={extras}, missing={missing}."
        )


def write_content_index(
    root: Path, *, config_contract_hash: str, protocol_contract_hash: str
) -> Mapping[str, object]:
    rows = []
    for member in CONTENT_INDEX_MEMBERS:
        path = root / member
        if path.is_symlink() or not path.is_file():
            raise ProtocolError(f"PCSI-RACR content member absent: {member}.")
        rows.append(
            {
                "member": member,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = {
        "schema_version": "fixed_bank_pcsi_racr_content_index_v1",
        "config_contract_hash": config_contract_hash,
        "protocol_contract_hash": protocol_contract_hash,
        "members": rows,
        "member_count": len(rows),
        "closed_world": True,
        "raw_labels_persisted": False,
        "image_or_sample_paths_persisted": False,
        "terminal_checkpoint_persisted": False,
        "previous_stage90_output_or_checkpoint_used": False,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "consumed_test_diagnostic_only": True,
        "fresh_evidence": False,
        "routing_success_claimed": False,
        "promotion_eligible": False,
        "may_feed_another_experiment": False,
    }
    payload["content_hash"] = canonical_hash(payload)
    persist_json(root / "manifests/content_index.json", payload)
    return payload


def validate_content_index(
    root: Path, *, config_contract_hash: str, protocol_contract_hash: str
) -> Mapping[str, object]:
    payload = read_json(root / "manifests/content_index.json")
    unhashed = {key: value for key, value in payload.items() if key != "content_hash"}
    rows = payload.get("members")
    exact_header = {
        "schema_version": "fixed_bank_pcsi_racr_content_index_v1",
        "config_contract_hash": config_contract_hash,
        "protocol_contract_hash": protocol_contract_hash,
        "member_count": len(CONTENT_INDEX_MEMBERS),
        "closed_world": True,
        "raw_labels_persisted": False,
        "image_or_sample_paths_persisted": False,
        "terminal_checkpoint_persisted": False,
        "previous_stage90_output_or_checkpoint_used": False,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "consumed_test_diagnostic_only": True,
        "fresh_evidence": False,
        "routing_success_claimed": False,
        "promotion_eligible": False,
        "may_feed_another_experiment": False,
    }
    if (
        set(payload) != set(exact_header) | {"members", "content_hash"}
        or any(payload.get(key) != value for key, value in exact_header.items())
        or not isinstance(payload.get("content_hash"), str)
        or re.fullmatch(r"[0-9a-f]{64}", str(payload.get("content_hash"))) is None
        or payload.get("content_hash") != canonical_hash(unhashed)
        or not isinstance(rows, list)
        or len(rows) != len(CONTENT_INDEX_MEMBERS)
        or [row.get("member") for row in rows if isinstance(row, Mapping)]
        != list(CONTENT_INDEX_MEMBERS)
    ):
        raise ProtocolError("PCSI-RACR content-index header drifted.")
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "member",
            "size_bytes",
            "sha256",
        }:
            raise ProtocolError("PCSI-RACR content-index row malformed.")
        path = root / str(row["member"])
        if (
            path.is_symlink()
            or not path.is_file()
            or type(row.get("size_bytes")) is not int
            or int(row["size_bytes"]) < 0
            or not isinstance(row.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256"))) is None
            or path.stat().st_size != row.get("size_bytes")
            or sha256_file(path) != row.get("sha256")
        ):
            raise ProtocolError("PCSI-RACR indexed member drifted.")
    return payload


def _reject_symlinks(root: Path) -> None:
    if root.is_symlink():
        raise ProtocolError("PCSI-RACR bundle root cannot be a symlink.")
    if not root.exists():
        return
    for directory, names, files in os.walk(root, followlinks=False):
        base = Path(directory)
        if any((base / name).is_symlink() for name in (*names, *files)):
            raise ProtocolError("PCSI-RACR bundle contains a symlink.")


__all__ = (
    "CONTENT_INDEX_MEMBERS",
    "REQUIRED_FILES",
    "assert_closed_world",
    "relative_files",
    "validate_content_index",
    "write_content_index",
)
