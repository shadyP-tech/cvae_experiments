"""Materialize the frozen source-inner utility/regret Stage-60 policy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from ...reporting import write_csv_rows, write_json
from .bundle import (
    CONTENT_INDEX_MEMBERS,
    REQUIRED_FILES,
    leakage_report_payload,
    policy_decision_payload,
    protocol_manifest_payload,
    run_state_payload,
    source_inner_training_summary_payload,
)
from .config import UtilityRegretPolicyConfig
from .inputs import load_validated_inputs
from .policy import (
    bootstrap_table_hash,
    build_policy_assignments,
    build_policy_lock,
    build_policy_plan,
    build_policy_selections,
    case_confusion_table_hash,
)
from .regret import (
    build_outer_regret_cells,
    regret_table_hash,
    summarize_candidates,
    summary_table_hash,
    utility_table_hash,
)


def run_utility_regret_policy_lock(
    config: UtilityRegretPolicyConfig,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    """Freeze selections/fallbacks without accessing raw samples or labels."""

    root = Path(artifact_root or config.artifact_root)
    for relative in ("manifests", "provenance", "reports", "tables"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    _assert_closed_world(root)
    if not (root / "config.resolved.yaml").is_file() or not (
        root / "provenance/input_artifacts.json"
    ).is_file():
        raise ProtocolError(
            "Utility/regret policy requires workspace-resolved config and provenance."
        )
    state_path = root / "reports/run_state.json"
    if state_path.is_file() and _json(state_path).get("status") == "COMPLETE":
        from .validation import validate_utility_regret_policy_bundle

        validate_utility_regret_policy_bundle(root, config=config)
        return root

    _write_state(root, "RUNNING")
    try:
        inputs = load_validated_inputs(config)
        utility_hash = utility_table_hash(inputs.utility_rows)
        case_hash = case_confusion_table_hash(inputs.case_confusion_rows)
        cells = build_outer_regret_cells(inputs.utility_rows)
        summaries = summarize_candidates(cells)
        selections = build_policy_selections(
            summaries,
            inputs.case_confusion_rows,
        )
        assignments = build_policy_assignments(
            inputs.generation_lock,
            selections,
        )
        regret_hash = regret_table_hash(cells)
        summary_hash = summary_table_hash(summaries)
        plan = build_policy_plan(
            config=config,
            generation_lock=inputs.generation_lock,
            selections=selections,
            assignments=assignments,
            utility_surface_hash=inputs.utility_lock_hash,
        )
        lock = build_policy_lock(
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

        write_json(root / "manifests/protocol_manifest.json", protocol)
        write_json(root / "manifests/policy_lock.json", lock.to_payload())
        write_json(root / "manifests/utility_regret_policy_plan.json", plan)
        write_csv_rows(
            root / "tables/outer_regret_cells.csv",
            [row.to_payload() for row in cells],
        )
        write_csv_rows(
            root / "tables/candidate_regret_summary.csv",
            [row.to_payload() for row in summaries],
        )
        write_csv_rows(
            root / "tables/bootstrap_results.csv",
            [row.bootstrap.to_payload() for row in selections],
        )
        write_csv_rows(
            root / "tables/policy_selections.csv",
            [row.to_payload() for row in selections],
        )
        write_csv_rows(
            root / "tables/policy_assignments.csv",
            [row.to_payload() for row in assignments],
        )
        write_json(
            root / "reports/policy_decision.json",
            policy_decision_payload(lock, selections),
        )
        write_json(root / "reports/leakage_report.json", leakage_report_payload())
        write_json(
            root / "reports/source_inner_training_summary.json",
            source_inner_training_summary_payload(selections),
        )
        _write_content_index(root)
        _write_state(root, "COMPLETE")

        from .validation import validate_utility_regret_policy_bundle

        checks = validate_utility_regret_policy_bundle(
            root,
            config=config,
            allow_pending=True,
            _validated_inputs=inputs,
        )
        write_json(
            root / "reports/validation_report.json",
            {
                "schema_version": (
                    "midogpp_uniform_b_v2_utility_regret_validation_v1"
                ),
                "status": "PASS",
                "validator": "validate_utility_regret_policy_bundle",
                "checks": checks,
            },
        )
        validate_utility_regret_policy_bundle(
            root,
            config=config,
            _validated_inputs=inputs,
        )
    except Exception:
        _write_state(root, "FAILED")
        raise
    return root


def _write_content_index(root: Path) -> None:
    records = []
    for relative in CONTENT_INDEX_MEMBERS:
        member = root / relative
        if not member.is_file():
            raise ProtocolError(f"Utility/regret content member is missing: {relative}.")
        records.append(
            {
                "relative_path": relative,
                "sha256": _sha256_file(member),
                "size_bytes": member.stat().st_size,
            }
        )
    payload: dict[str, object] = {
        "schema_version": "midogpp_uniform_b_v2_utility_regret_content_v1",
        "records": records,
    }
    payload["content_hash"] = stable_hash(payload)
    write_json(root / "manifests/content_index.json", payload)


def _assert_closed_world(root: Path) -> None:
    actual = {
        member.relative_to(root).as_posix()
        for member in root.rglob("*")
        if member.is_file()
    }
    unexpected = sorted(actual.difference(REQUIRED_FILES))
    if unexpected:
        raise ProtocolError(
            f"Utility/regret policy artifact contains unexpected files: {unexpected}."
        )


def _write_state(root: Path, status: str) -> None:
    write_json(root / "reports/run_state.json", run_state_payload(status))


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


__all__ = ("run_utility_regret_policy_lock",)
