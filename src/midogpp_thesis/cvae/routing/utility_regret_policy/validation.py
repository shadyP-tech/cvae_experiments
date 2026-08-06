"""Independent reconstruction of the frozen utility/regret policy artifact."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ...expert_bank.uniform_b_v2_promotion.validation import (
    REQUIRED_FILES as BANK_REQUIRED_FILES,
)
from ...generation.validation import REQUIRED_FILES as GENERATION_REQUIRED_FILES
from ...protocol import ProtocolError
from ..bundle import REQUIRED_FILES as EQUAL_UNION_REQUIRED_FILES
from ..source_inner_utility.bundle import REQUIRED_FILES as UTILITY_REQUIRED_FILES
from .bundle import (
    CONTENT_INDEX_MEMBERS,
    REQUIRED_FILES,
    leakage_report_payload,
    policy_decision_payload,
    protocol_manifest_payload,
    run_state_payload,
    source_inner_training_summary_payload,
)
from .config import UtilityRegretPolicyConfig, load_utility_regret_policy_config
from .contracts import (
    CENTERS,
    CLAIM_SCOPE,
    CONSUMPTION_RULE_HASH,
    EQUAL_UNION_ARTIFACT_ID,
    EXPERIMENT_ID,
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    UTILITY_ARTIFACT_ID,
)
from .inputs import ValidatedUtilityRegretInputs, load_validated_inputs
from .policy import (
    assignment_table_hash,
    bootstrap_table_hash,
    build_policy_assignments,
    build_policy_lock,
    build_policy_plan,
    build_policy_selections,
    case_confusion_table_hash,
    read_policy_lock,
    selection_table_hash,
)
from .regret import (
    build_outer_regret_cells,
    regret_table_hash,
    summarize_candidates,
    summary_table_hash,
    utility_table_hash,
)


def validate_utility_regret_policy_bundle(
    root: str | Path,
    *,
    config: UtilityRegretPolicyConfig,
    allow_pending: bool = False,
    _validated_inputs: ValidatedUtilityRegretInputs | None = None,
) -> dict[str, object]:
    path = Path(root)
    required = set(REQUIRED_FILES)
    if allow_pending:
        required.remove("reports/validation_report.json")
    missing = sorted(relative for relative in required if not (path / relative).is_file())
    if missing:
        raise ProtocolError(f"Utility/regret policy artifact is incomplete: {missing}.")
    _validate_closed_world(path, allow_pending=allow_pending)
    if load_utility_regret_policy_config(path / "config.resolved.yaml") != config:
        raise ProtocolError("Utility/regret resolved config drifted.")
    validate_policy_provenance(path, config=config)

    inputs = _validated_inputs or load_validated_inputs(config)
    utility_hash = utility_table_hash(inputs.utility_rows)
    case_hash = case_confusion_table_hash(inputs.case_confusion_rows)
    cells = build_outer_regret_cells(inputs.utility_rows)
    summaries = summarize_candidates(cells)
    selections = build_policy_selections(summaries, inputs.case_confusion_rows)
    assignments = build_policy_assignments(inputs.generation_lock, selections)
    regret_hash = regret_table_hash(cells)
    summary_hash = summary_table_hash(summaries)
    plan = build_policy_plan(
        config=config,
        generation_lock=inputs.generation_lock,
        selections=selections,
        assignments=assignments,
        utility_surface_hash=inputs.utility_lock_hash,
    )
    expected_lock = build_policy_lock(
        config=config,
        generation_lock=inputs.generation_lock,
        selections=selections,
        assignments=assignments,
        utility_surface_hash=inputs.utility_lock_hash,
        utility_table_hash=utility_hash,
        case_confusion_table_hash=case_hash,
        regret_table_hash=regret_hash,
        summary_table_hash=summary_hash,
        plan_hash=str(plan["plan_hash"]),
    )
    observed_lock = read_policy_lock(path / "manifests/policy_lock.json")
    if observed_lock.to_payload() != expected_lock.to_payload():
        raise ProtocolError("Utility/regret policy lock drifted from its inputs.")
    _require_exact_json(
        path / "manifests/utility_regret_policy_plan.json",
        plan,
        "policy plan",
    )

    _validate_csv(
        path / "tables/outer_regret_cells.csv",
        [row.to_payload() for row in cells],
        "outer regret cells",
    )
    _validate_csv(
        path / "tables/candidate_regret_summary.csv",
        [row.to_payload() for row in summaries],
        "candidate summaries",
    )
    _validate_csv(
        path / "tables/bootstrap_results.csv",
        [row.bootstrap.to_payload() for row in selections],
        "bootstrap results",
    )
    _validate_csv(
        path / "tables/policy_selections.csv",
        [row.to_payload() for row in selections],
        "policy selections",
    )
    _validate_csv(
        path / "tables/policy_assignments.csv",
        [row.to_payload() for row in assignments],
        "policy assignments",
    )

    protocol = protocol_manifest_payload(
        config,
        generation_lock_hash=inputs.generation_lock.generation_lock_hash,
        utility_lock_hash=inputs.utility_lock_hash,
        utility_content_hash=inputs.utility_content_hash,
        utility_table_hash=utility_hash,
        case_confusion_table_hash=case_hash,
        regret_table_hash=regret_hash,
        summary_table_hash=summary_hash,
    )
    _require_exact_json(
        path / "manifests/protocol_manifest.json",
        protocol,
        "protocol manifest",
    )
    _require_exact_json(
        path / "reports/policy_decision.json",
        policy_decision_payload(expected_lock, selections),
        "policy decision",
    )
    _require_exact_json(
        path / "reports/leakage_report.json",
        leakage_report_payload(),
        "leakage report",
    )
    _require_exact_json(
        path / "reports/source_inner_training_summary.json",
        source_inner_training_summary_payload(selections),
        "training summary",
    )
    _require_exact_json(
        path / "reports/run_state.json",
        run_state_payload("COMPLETE"),
        "run state",
    )
    _validate_content_index(path)

    action_by_target = {row.target_center: row.action for row in selections}
    selected_by_target = {
        row.target_center: (row.selected_source or None) for row in selections
    }
    checks: dict[str, object] = {
        "status": "PASS",
        "bank_lock_hash": config.expected_bank_lock_hash,
        "generation_lock_hash": inputs.generation_lock.generation_lock_hash,
        "equal_union_policy_lock_hash": config.expected_equal_union_policy_lock_hash,
        "utility_lock_hash": inputs.utility_lock_hash,
        "policy_consumption_lock_hash": CONSUMPTION_RULE_HASH,
        "utility_table_hash": utility_hash,
        "case_confusion_table_hash": case_hash,
        "regret_table_hash": regret_hash,
        "summary_table_hash": summary_hash,
        "bootstrap_table_hash": bootstrap_table_hash(selections),
        "selection_table_hash": selection_table_hash(selections),
        "assignment_table_hash": assignment_table_hash(assignments),
        "policy_plan_hash": plan["plan_hash"],
        "policy_lock_hash": expected_lock.policy_lock_hash,
        "outer_regret_cell_count": len(cells),
        "candidate_summary_count": len(summaries),
        "selection_count": len(selections),
        "assignment_count": len(assignments),
        "action_by_target": action_by_target,
        "selected_source_by_target": selected_by_target,
        "outer_target_query_excluded_before_transform": True,
        "outer_target_candidate_excluded_before_transform": True,
        "exact_equal_union_fallback": True,
        "raw_labels_opened_by_policy": False,
        "target_support_used": False,
        "seed_selection_performed": False,
        "routing_quality_claimed": False,
        "downstream_utility_computed": False,
        "may_feed_deployable_selection": True,
    }
    if not allow_pending:
        _require_exact_json(
            path / "reports/validation_report.json",
            {
                "schema_version": "midogpp_uniform_b_v2_utility_regret_validation_v1",
                "status": "PASS",
                "validator": "validate_utility_regret_policy_bundle",
                "checks": checks,
            },
            "validation report",
        )
    return checks


def validate_policy_provenance(
    root: str | Path,
    *,
    config: UtilityRegretPolicyConfig,
) -> None:
    """Require exactly the bank, GenerationLock, control, and utility artifact."""

    output_root = Path(root)
    manifest = _json(output_root / "provenance/input_artifacts.json")
    expected_header = {
        "schema_version": "midogpp_input_artifacts_v2",
        "dataset_id": "midogpp",
        "experiment_id": EXPERIMENT_ID,
        "stage": "60_routing_and_composition",
        "claim_scope": CLAIM_SCOPE,
        "selection_used_target_eval_artifacts": False,
    }
    allowed_header = set(expected_header) | {
        "input_artifacts",
        "repository_revision",
        "repository_dirty",
        "repository_status_hash",
    }
    if (
        set(manifest) != allowed_header
        or any(manifest.get(key) != value for key, value in expected_header.items())
        or not _is_hex(manifest.get("repository_revision"), 40)
        or not isinstance(manifest.get("repository_dirty"), bool)
        or not _is_hex(manifest.get("repository_status_hash"), 64)
    ):
        raise ProtocolError("Utility/regret workspace provenance header drifted.")
    raw_rows = manifest.get("input_artifacts")
    if not isinstance(raw_rows, list) or not all(
        isinstance(row, Mapping) for row in raw_rows
    ):
        raise ProtocolError("Utility/regret workspace provenance is malformed.")
    rows = {str(row.get("artifact_id", "")): row for row in raw_rows}
    expected = {
        EXPERT_BANK_ARTIFACT_ID: (
            config.bank_root,
            "30_expert_bank",
            "expert_bank_construction_only",
            set(BANK_REQUIRED_FILES) | {"reports/validation_report.json"},
        ),
        GENERATION_LOCK_ARTIFACT_ID: (
            config.generation_lock_root,
            "40_prior_and_generation",
            "generation_settings_and_frame_lock",
            set(GENERATION_REQUIRED_FILES) | {"reports/validation_report.json"},
        ),
        EQUAL_UNION_ARTIFACT_ID: (
            config.equal_union_root,
            "60_routing_and_composition",
            "routing_and_composition",
            set(EQUAL_UNION_REQUIRED_FILES),
        ),
        UTILITY_ARTIFACT_ID: (
            config.utility_root,
            "60_routing_and_composition",
            "routing_and_composition",
            set(UTILITY_REQUIRED_FILES),
        ),
    }
    if len(raw_rows) != len(rows) or set(rows) != set(expected):
        raise ProtocolError("Utility/regret policy may consume only four frozen inputs.")
    for artifact_id, (expected_root, stage, scope, required_files) in expected.items():
        row = rows[artifact_id]
        if (
            Path(str(row.get("resolved_path", ""))).resolve()
            != Path(expected_root).resolve()
            or row.get("stage") != stage
            or row.get("claim_scope") != scope
            or row.get("exists") is not True
            or row.get("semantic_identities_are_file_hashes") is not False
            or not isinstance(row.get("semantic_identities"), Mapping)
        ):
            raise ProtocolError(f"Utility/regret provenance drifted: {artifact_id}.")
        _validate_integrity_inventory(
            Path(expected_root),
            row.get("file_integrity"),
            required_files=required_files,
        )


def _validate_integrity_inventory(
    root: Path,
    value: object,
    *,
    required_files: set[str],
) -> None:
    if not isinstance(value, Mapping):
        raise ProtocolError("Utility/regret provenance inventory is malformed.")
    files = value.get("files")
    if not isinstance(files, list) or not all(isinstance(row, Mapping) for row in files):
        raise ProtocolError("Utility/regret provenance file rows are malformed.")
    by_path = {str(row.get("path", "")): row for row in files}
    if len(files) != len(by_path) or set(by_path) != required_files:
        raise ProtocolError("Utility/regret provenance file coverage drifted.")
    for relative, row in by_path.items():
        member = root / relative
        computed = row.get("computed")
        if (
            row.get("exists") is not True
            or not member.is_file()
            or Path(str(row.get("resolved_path", ""))).resolve() != member.resolve()
            or not isinstance(computed, Mapping)
            or computed.get("sha256") != _sha256_file(member)
        ):
            raise ProtocolError("Utility/regret provenance member hash drifted.")


def _validate_csv(
    path: Path,
    expected: Sequence[Mapping[str, object]],
    label: str,
) -> None:
    if not expected:
        raise ProtocolError(f"Utility/regret expected {label} are empty.")
    columns = tuple(expected[0])
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != columns:
            raise ProtocolError(f"Utility/regret {label} schema drifted.")
        observed = [dict(row) for row in reader]
    rendered = [
        {key: "" if row.get(key) is None else str(row.get(key)) for key in columns}
        for row in expected
    ]
    if observed != rendered:
        raise ProtocolError(f"Utility/regret {label} values drifted.")


def _validate_content_index(root: Path) -> None:
    payload = _json(root / "manifests/content_index.json")
    _assert_embedded_hash(payload, "content_hash")
    records = payload.get("records")
    if (
        payload.get("schema_version")
        != "midogpp_uniform_b_v2_utility_regret_content_v1"
        or not isinstance(records, list)
    ):
        raise ProtocolError("Utility/regret content index drifted.")
    indexed: dict[str, Mapping[str, object]] = {}
    for row in records:
        if not isinstance(row, Mapping):
            raise ProtocolError("Utility/regret content row is malformed.")
        relative = str(row.get("relative_path", ""))
        member = root / relative
        if (
            set(row) != {"relative_path", "sha256", "size_bytes"}
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative in indexed
            or not member.is_file()
            or row.get("sha256") != _sha256_file(member)
            or row.get("size_bytes") != member.stat().st_size
        ):
            raise ProtocolError("Utility/regret content member drifted.")
        indexed[relative] = row
    if set(indexed) != set(CONTENT_INDEX_MEMBERS):
        raise ProtocolError("Utility/regret content coverage drifted.")


def _validate_closed_world(root: Path, *, allow_pending: bool) -> None:
    actual = {
        member.relative_to(root).as_posix()
        for member in root.rglob("*")
        if member.is_file()
    }
    expected = set(REQUIRED_FILES)
    if allow_pending:
        expected.remove("reports/validation_report.json")
    if actual != expected:
        raise ProtocolError("Utility/regret policy artifact is not closed-world.")


def _require_exact_json(path: Path, expected: Mapping[str, object], label: str) -> None:
    observed = _json(path)
    if observed != dict(expected):
        raise ProtocolError(f"Utility/regret {label} drifted.")


def _assert_embedded_hash(payload: Mapping[str, object], key: str) -> None:
    unhashed = {name: value for name, value in payload.items() if name != key}
    if payload.get(key) != stable_hash(unhashed):
        raise ProtocolError(f"Utility/regret embedded hash drifted: {key}.")


def _json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read utility/regret JSON: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Utility/regret JSON must be an object: {path}.")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_hex(value: object, length: int) -> bool:
    rendered = str(value or "")
    return len(rendered) == length and all(
        char in "0123456789abcdef" for char in rendered
    )


__all__ = (
    "validate_policy_provenance",
    "validate_utility_regret_policy_bundle",
)
