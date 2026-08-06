"""Materialize the non-selecting Stage-60 source-inner utility artifact."""

from __future__ import annotations

import json
from pathlib import Path

from ....common.hashing import stable_hash
from ...generation import (
    load_generation_lock_config,
    read_generation_lock,
    validate_generation_bundle,
)
from ...protocol import ProtocolError
from ...reporting import write_csv_rows, write_json
from .bundle import (
    CASE_CONFUSION_TABLE_MEMBER,
    CONTENT_INDEX_MEMBERS,
    REQUIRED_FILES,
    UTILITY_TABLE_MEMBER,
    label_consumption_report_payload,
    leakage_report_payload,
    policy_consumption_manifest_payload,
    protocol_manifest_payload,
    run_state_payload,
    sha256_file,
    utility_decision_payload,
    utility_lock_payload,
)
from .cache_inputs import (
    load_unlabeled_validation_frame,
    open_scoring_labels,
    read_manifest_evaluation_index,
)
from .config import SourceInnerUtilityConfig
from .contracts import (
    EXPECTED_CASE_CONFUSION_ROW_COUNT,
    EXPECTED_EVAL_ROWS,
    EXPECTED_FIT_COUNT,
    EXPECTED_UTILITY_ROW_COUNT,
)
from .metric_scoring import (
    CASE_CONFUSION_COLUMNS,
    UTILITY_COLUMNS,
    score_prediction_pass,
)
from .prediction import FIT_COLUMNS, run_label_free_prediction_pass
from .prediction_io import (
    EVALUATION_ROW_COLUMNS,
    EVALUATION_ROW_MEMBER,
    FIT_TABLE_MEMBER,
    PREDICTION_ARRAY_MEMBER,
    evaluation_row_table,
    prediction_index_payload,
    write_prediction_arrays,
)


def run_source_inner_candidate_utility(
    config: SourceInnerUtilityConfig,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    """Generate, predict, then open labels and score the full ``q != e`` matrix."""

    root = Path(artifact_root or config.artifact_root)
    for relative in ("arrays", "manifests", "provenance", "reports", "tables"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    state_path = root / "reports/run_state.json"
    try:
        _assert_closed_world(root)
        if not (root / "config.resolved.yaml").is_file() or not (
            root / "provenance/input_artifacts.json"
        ).is_file():
            raise ProtocolError(
                "Source-inner utility requires workspace-resolved config and provenance."
            )
        if state_path.is_file() and _json(state_path).get("status") == "COMPLETE":
            from .validation import validate_source_inner_utility_bundle

            try:
                validate_source_inner_utility_bundle(root, config=config)
            except Exception:
                _write_state(root, "FAILED")
                raise
            return root
    except Exception:
        if state_path.exists():
            _write_state(root, "FAILED")
        raise

    _write_state(root, "RUNNING")
    try:
        from .validation import validate_source_inner_utility_provenance

        validate_source_inner_utility_provenance(root, config=config)
        generation_lock = _load_validated_generation_lock(config)

        # Persist the sole consuming rule before validation identities are even
        # loaded.  Labels remain inaccessible until prediction arrays exist.
        write_json(
            root / "manifests/policy_consumption_lock.json",
            policy_consumption_manifest_payload(config),
        )
        manifest_rows = read_manifest_evaluation_index(
            config.manifest_path,
            expected_sha256=config.expected_manifest_sha256,
            expected_rows=EXPECTED_EVAL_ROWS,
        )
        frame = load_unlabeled_validation_frame(
            config.validation_cache_root,
            manifest_rows,
            expected_rows=EXPECTED_EVAL_ROWS,
        )
        if (
            frame.cache_binding.get("cache_name") != config.expected_cache_semantic_id
            or frame.cache_binding.get("representation_id")
            != config.expected_cache_representation_id
            or frame.cache_binding.get("validation_split") != "val"
        ):
            raise ProtocolError("Routing validation-cache semantic identity drifted.")
        protocol = protocol_manifest_payload(
            config,
            generation_lock_hash=generation_lock.generation_lock_hash,
            bank_lock_hash=generation_lock.bank_lock_hash,
            cache_binding=frame.cache_binding,
        )
        write_json(root / "manifests/protocol_manifest.json", protocol)

        predictions = run_label_free_prediction_pass(
            frame,
            bank_root=config.bank_root,
            classifier_spec=config.classifier,
            generation_lock=generation_lock,
            per_class=1024,
            device=config.generation_device,
            threads_per_fit=config.threads_per_fit,
        )
        if len(predictions.fit_rows) != EXPECTED_FIT_COUNT:
            raise ProtocolError("Source-inner utility did not materialize exactly 81 fits.")
        write_prediction_arrays(root / PREDICTION_ARRAY_MEMBER, predictions)
        write_csv_rows(
            root / EVALUATION_ROW_MEMBER,
            evaluation_row_table(predictions),
            columns=EVALUATION_ROW_COLUMNS,
        )
        write_csv_rows(
            root / FIT_TABLE_MEMBER,
            predictions.fit_rows,
            columns=FIT_COLUMNS,
        )
        prediction_index = prediction_index_payload(
            predictions,
            prediction_file_sha256=sha256_file(root / PREDICTION_ARRAY_MEMBER),
        )
        write_json(root / "manifests/prediction_index.json", prediction_index)

        # This is the producer's sole label-opening call.  All predictions and
        # their two label-free indices are already durably materialized.  The
        # independent validator may reopen the same frozen rows audit-only to
        # reconstruct these scores; that audit cannot change this artifact.
        labels = open_scoring_labels(
            config.manifest_path,
            predictions.evaluation_rows,  # type: ignore[arg-type]
            expected_sha256=config.expected_manifest_sha256,
        )
        utilities, case_confusions = score_prediction_pass(predictions, labels)
        if (
            len(utilities) != EXPECTED_UTILITY_ROW_COUNT
            or len(case_confusions) != EXPECTED_CASE_CONFUSION_ROW_COUNT
        ):
            raise ProtocolError("Source-inner utility/case-confusion geometry drifted.")
        write_csv_rows(
            root / UTILITY_TABLE_MEMBER,
            utilities,
            columns=UTILITY_COLUMNS,
        )
        write_csv_rows(
            root / CASE_CONFUSION_TABLE_MEMBER,
            case_confusions,
            columns=CASE_CONFUSION_COLUMNS,
        )
        write_json(
            root / "reports/label_consumption_report.json",
            label_consumption_report_payload(),
        )
        write_json(root / "reports/leakage_report.json", leakage_report_payload())

        hashed_members = {
            relative: sha256_file(root / relative)
            for relative in (
                EVALUATION_ROW_MEMBER,
                FIT_TABLE_MEMBER,
                UTILITY_TABLE_MEMBER,
                CASE_CONFUSION_TABLE_MEMBER,
                PREDICTION_ARRAY_MEMBER,
            )
        }
        utility_lock = utility_lock_payload(
            config,
            protocol=protocol,
            prediction_index=prediction_index,
            member_sha256=hashed_members,
            case_confusion_row_count=len(case_confusions),
        )
        write_json(root / "manifests/utility_lock.json", utility_lock)
        write_json(
            root / "reports/utility_decision.json",
            utility_decision_payload(utility_lock),
        )
        _write_content_index(root)
        _write_state(root, "COMPLETE")

        from .validation import validate_source_inner_utility_bundle

        checks = validate_source_inner_utility_bundle(
            root,
            config=config,
            allow_pending=True,
        )
        write_json(
            root / "reports/validation_report.json",
            {
                "schema_version": (
                    "midogpp_uniform_b_v2_source_inner_utility_validation_v1"
                ),
                "status": "PASS",
                "validator": "validate_source_inner_utility_bundle",
                "checks": checks,
            },
        )
        validate_source_inner_utility_bundle(root, config=config)
    except Exception:
        _write_state(root, "FAILED")
        raise
    return root


def _load_validated_generation_lock(config: SourceInnerUtilityConfig) -> object:
    generation_config = load_generation_lock_config(
        config.generation_lock_root / "config.resolved.yaml"
    )
    if generation_config.bank_root.resolve() != config.bank_root.resolve():
        raise ProtocolError("Source-inner utility bank/GenerationLock roots disagree.")
    validate_generation_bundle(config.generation_lock_root, config=generation_config)
    lock = read_generation_lock(
        config.generation_lock_root / "manifests/generation_lock.json"
    )
    if (
        lock.generation_lock_hash != config.expected_generation_lock_hash
        or lock.bank_lock_hash != config.expected_bank_lock_hash
    ):
        raise ProtocolError("Source-inner utility GenerationLock identity drifted.")
    return lock


def _write_content_index(root: Path) -> None:
    records = []
    for relative in CONTENT_INDEX_MEMBERS:
        member = root / relative
        if not member.is_file():
            raise ProtocolError(f"Source-inner utility content member missing: {relative}.")
        records.append(
            {
                "relative_path": relative,
                "sha256": sha256_file(member),
                "size_bytes": member.stat().st_size,
            }
        )
    payload: dict[str, object] = {
        "schema_version": "midogpp_uniform_b_v2_source_inner_utility_content_v1",
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
            f"Source-inner utility artifact contains unexpected files: {unexpected}."
        )


def _write_state(root: Path, status: str) -> None:
    write_json(root / "reports/run_state.json", run_state_payload(status))


def _json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read source-inner utility JSON: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Source-inner utility JSON must be an object: {path}.")
    return payload


__all__ = ("run_source_inner_candidate_utility",)
