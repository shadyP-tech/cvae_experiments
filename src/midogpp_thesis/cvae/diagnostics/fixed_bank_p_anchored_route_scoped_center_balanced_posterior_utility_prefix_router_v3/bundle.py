"""Exact closed-world bundle inventory and stable content index."""

from __future__ import annotations

from pathlib import Path

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json, sha256_file
from .artifact_io import persist_json
from .hashing import canonical_hash
from .terminal_access_journal import TERMINAL_ACCESS_JOURNAL_ORDER


ALLOWED_DIRECTORIES = frozenset(
    {"arrays", "manifests", "provenance", "reports", "tables"}
)
PRETERMINAL_SCIENTIFIC_MEMBERS = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "arrays/frozen_source_streams.npy",
    "arrays/fixed_bank_a1_action_probabilities.npz",
    "arrays/route_endpoint_probabilities.npz",
    "arrays/pseudo_route_endpoint_probabilities.npz",
    "arrays/target_local_posterior_probabilities.npz",
    "arrays/candidate_probabilities.npz",
    "arrays/composed_probabilities.npz",
    "manifests/frozen_source_stream_index.json",
    "manifests/frozen_source_stream_lock.json",
    "manifests/fixed_bank_a1_prediction_index.json",
    "manifests/fixed_bank_a1_prediction_seal.json",
    "manifests/route_endpoint_probability_index.json",
    "manifests/pseudo_route_endpoint_probability_index.json",
    "manifests/target_local_posterior_probability_index.json",
    "manifests/candidate_probability_index.json",
    "manifests/composed_probability_index.json",
    "manifests/protocol_manifest.json",
    "manifests/action_library.json",
    "manifests/physical_surface_seal.json",
    "manifests/outer_plan_seal.json",
    "manifests/policy_menu.json",
    "manifests/decision_barrier.json",
    "manifests/preterminal_aggregate_seal.json",
    "tables/exact_nine_probability_index.json",
    "tables/outer_plans.json",
    "tables/physical_fingerprints.json",
    "tables/route_support_capabilities.json",
    "tables/route_endpoint_states.json",
    "tables/pseudo_source_priors.json",
    "tables/target_local_posterior_models.json",
    "tables/target_local_posterior_predictions.json",
    "tables/pseudo_posterior_references.json",
    "tables/expected_utility_predictions.json",
    "tables/candidate_eligibility.json",
    "tables/target_candidate_policies.json",
    "tables/pseudo_candidate_policies.json",
    "tables/pseudo_policy_replays.json",
    "tables/donor_bias_calibrations.json",
    "tables/prefix_decisions.json",
    "tables/transport_diagnostics.json",
    "tables/composed_predictions.json",
    "tables/route_decisions.json",
    "tables/gate_funnel.json",
    "tables/information_diagnostics.json",
    "reports/workstation_preflight.json",
    "reports/preterminal_label_capability_report.json",
)
PRETERMINAL_GATE_MEMBERS = (
    "manifests/preterminal_content_index.json",
    "reports/preterminal_fresh_process_attestation.json",
    "reports/preterminal_validation_report.json",
    "manifests/preterminal_validation_seal.json",
)
TERMINAL_SCIENTIFIC_MEMBERS = (
    "manifests/terminal_evaluation_seal.json",
    "tables/terminal_method_metrics.json",
    "tables/terminal_center_contrasts.json",
    "tables/terminal_case_oracles.json",
    "reports/diagnostic_summary.json",
    "reports/label_capability_report.json",
    "reports/leakage_report.json",
    "reports/publication_decision.json",
    "reports/runtime_summary.json",
)
FINAL_GATE_MEMBERS = (
    "manifests/content_index.json",
    "reports/run_state.json",
    "reports/fresh_process_attestation.json",
    "reports/validation_report.json",
)
REQUIRED_FILES = (
    *PRETERMINAL_SCIENTIFIC_MEMBERS,
    *PRETERMINAL_GATE_MEMBERS,
    *TERMINAL_ACCESS_JOURNAL_ORDER,
    *TERMINAL_SCIENTIFIC_MEMBERS,
    *FINAL_GATE_MEMBERS,
)
PRETERMINAL_CONTENT_INDEX_MEMBERS = PRETERMINAL_SCIENTIFIC_MEMBERS
PRETERMINAL_UNINDEXED_FILES = frozenset(
    (*PRETERMINAL_SCIENTIFIC_MEMBERS, "reports/run_state.json")
)
PRETERMINAL_VALIDATION_FILES = frozenset(
    (*PRETERMINAL_UNINDEXED_FILES, "manifests/preterminal_content_index.json")
)
PRETERMINAL_ATTESTED_FILES = frozenset(
    (*PRETERMINAL_VALIDATION_FILES, *PRETERMINAL_GATE_MEMBERS[1:])
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
LAUNCH_MEMBERS = frozenset(
    {"config.resolved.yaml", "provenance/input_artifacts.json"}
)


def relative_files(root: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.name != ".run.lock"
        )
    )


def assert_closed_world(root: Path, *, allow_incomplete: bool = False) -> None:
    _assert_safe_tree(root)
    observed = set(relative_files(root))
    required = set(REQUIRED_FILES)
    if not observed <= required:
        raise ProtocolError("CBPUPR bundle contains a foreign file.")
    if allow_incomplete:
        if not LAUNCH_MEMBERS <= observed:
            raise ProtocolError("CBPUPR bundle lacks its launch members.")
    elif observed != required:
        raise ProtocolError("CBPUPR final bundle inventory is not exact.")


def assert_preterminal_closed_world(root: Path, *, phase: str) -> None:
    """Require one exact preterminal lifecycle inventory.

    The three inventories are deliberately explicit.  In particular, a fresh
    validator cannot silently accept terminal products or a partly written
    attestation/report/seal sequence.
    """

    _assert_safe_tree(root)
    expected_by_phase = {
        "unindexed": PRETERMINAL_UNINDEXED_FILES,
        "validation": PRETERMINAL_VALIDATION_FILES,
        "attested": PRETERMINAL_ATTESTED_FILES,
    }
    expected = expected_by_phase.get(str(phase))
    if expected is None:
        raise ProtocolError("CBPUPR preterminal inventory phase drifted.")
    if frozenset(relative_files(root)) != expected:
        raise ProtocolError(
            f"CBPUPR preterminal {phase} bundle inventory is not exact."
        )


def write_preterminal_content_index(root: Path) -> dict[str, object]:
    assert_preterminal_closed_world(root, phase="unindexed")
    payload = {
        "schema_version": "fixed_bank_cbpupr_preterminal_content_index_v1",
        "members": [
            {
                "member": member,
                "size_bytes": (root / member).stat().st_size,
                "sha256": sha256_file(root / member),
            }
            for member in PRETERMINAL_CONTENT_INDEX_MEMBERS
        ],
    }
    result = {**payload, "content_index_hash": canonical_hash(payload)}
    persist_json(root / "manifests/preterminal_content_index.json", result)
    assert_preterminal_closed_world(root, phase="validation")
    return result


def validate_preterminal_content_index(root: Path) -> dict[str, object]:
    payload = read_json(root / "manifests/preterminal_content_index.json")
    rows = payload.get("members")
    unhashed = {
        key: value for key, value in payload.items() if key != "content_index_hash"
    }
    if (
        payload.get("schema_version")
        != "fixed_bank_cbpupr_preterminal_content_index_v1"
        or not isinstance(rows, list)
        or tuple(row.get("member") for row in rows if isinstance(row, dict))
        != PRETERMINAL_CONTENT_INDEX_MEMBERS
        or any(
            not isinstance(row, dict)
            or set(row) != {"member", "size_bytes", "sha256"}
            or row.get("size_bytes")
            != (root / str(row.get("member"))).stat().st_size
            or row.get("sha256")
            != sha256_file(root / str(row.get("member")))
            for row in rows
        )
        or payload.get("content_index_hash") != canonical_hash(unhashed)
    ):
        raise ProtocolError("CBPUPR preterminal content index drifted.")
    return payload


def write_content_index(root: Path) -> dict[str, object]:
    observed = set(relative_files(root))
    if not set(CONTENT_INDEX_MEMBERS) <= observed:
        raise ProtocolError("CBPUPR content index was requested before stable products.")
    if not observed <= set(REQUIRED_FILES):
        raise ProtocolError("CBPUPR content index observed a foreign product.")
    payload = {
        "schema_version": "fixed_bank_cbpupr_content_index_v1",
        "members": [
            {
                "member": member,
                "size_bytes": (root / member).stat().st_size,
                "sha256": sha256_file(root / member),
            }
            for member in CONTENT_INDEX_MEMBERS
        ],
    }
    result = {**payload, "content_index_hash": canonical_hash(payload)}
    persist_json(root / "manifests/content_index.json", result)
    return result


def validate_content_index(root: Path) -> dict[str, object]:
    payload = read_json(root / "manifests/content_index.json")
    rows = payload.get("members")
    unhashed = {key: value for key, value in payload.items() if key != "content_index_hash"}
    if (
        payload.get("schema_version") != "fixed_bank_cbpupr_content_index_v1"
        or not isinstance(rows, list)
        or tuple(row.get("member") for row in rows if isinstance(row, dict))
        != CONTENT_INDEX_MEMBERS
        or any(
            not isinstance(row, dict)
            or set(row) != {"member", "size_bytes", "sha256"}
            or row.get("size_bytes") != (root / str(row.get("member"))).stat().st_size
            or row.get("sha256") != sha256_file(root / str(row.get("member")))
            for row in rows
        )
        or payload.get("content_index_hash") != canonical_hash(unhashed)
    ):
        raise ProtocolError("CBPUPR content index drifted.")
    return payload


def _assert_safe_tree(root: Path) -> None:
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise ProtocolError("CBPUPR bundle contains a symlink.")
    if root.exists() and any(
        path.is_dir() and path.name not in ALLOWED_DIRECTORIES
        for path in root.iterdir()
    ):
        raise ProtocolError("CBPUPR bundle contains a foreign directory.")


__all__ = (
    "ALLOWED_DIRECTORIES",
    "CONTENT_INDEX_MEMBERS",
    "FINAL_GATE_MEMBERS",
    "INDEX_EXCLUDED",
    "LAUNCH_MEMBERS",
    "PRETERMINAL_ATTESTED_FILES",
    "PRETERMINAL_CONTENT_INDEX_MEMBERS",
    "PRETERMINAL_GATE_MEMBERS",
    "PRETERMINAL_SCIENTIFIC_MEMBERS",
    "PRETERMINAL_UNINDEXED_FILES",
    "PRETERMINAL_VALIDATION_FILES",
    "REQUIRED_FILES",
    "TERMINAL_SCIENTIFIC_MEMBERS",
    "TERMINAL_ACCESS_JOURNAL_ORDER",
    "assert_closed_world",
    "assert_preterminal_closed_world",
    "relative_files",
    "validate_content_index",
    "validate_preterminal_content_index",
    "write_content_index",
    "write_preterminal_content_index",
)
