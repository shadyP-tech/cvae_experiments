"""Label-free loading of the four frozen utility/regret policy inputs."""

from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ...generation import (
    GenerationLock,
    load_generation_lock_config,
    read_generation_lock,
    validate_generation_bundle,
)
from ...protocol import ProtocolError
from ..config import load_equal_union_policy_config
from ..policy import read_policy_lock as read_equal_union_policy_lock
from ..validation import validate_equal_union_policy_bundle
from ..source_inner_utility.bundle import (
    CASE_CONFUSION_TABLE_MEMBER,
    REQUIRED_FILES as UTILITY_REQUIRED_FILES,
    UTILITY_TABLE_MEMBER,
)
from ..source_inner_utility.contracts import (
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_CASE_CONFUSION_ROW_COUNT,
    EXPECTED_CONFIG_CONTRACT_HASH as EXPECTED_UTILITY_CONFIG_CONTRACT_HASH,
    EXPECTED_GENERATION_LOCK_HASH,
    EXPECTED_UTILITY_ROW_COUNT,
    EXPERIMENT_ID as UTILITY_EXPERIMENT_ID,
    POLICY_CONSUMPTION_LOCK_HASH,
    policy_consumption_lock_payload,
)
from ..source_inner_utility.scoring import (
    CASE_CONFUSION_COLUMNS,
    UTILITY_COLUMNS,
)
from .bootstrap import validate_case_confusions
from .config import UtilityRegretPolicyConfig
from .regret import validate_utility_rows


@dataclass(frozen=True)
class ValidatedUtilityRegretInputs:
    generation_lock: GenerationLock
    equal_union_policy_lock: object
    utility_lock: Mapping[str, object]
    utility_rows: tuple[Mapping[str, object], ...]
    case_confusion_rows: tuple[Mapping[str, object], ...]
    utility_content_hash: str

    @property
    def utility_lock_hash(self) -> str:
        return str(self.utility_lock["utility_lock_hash"])


def load_validated_inputs(
    config: UtilityRegretPolicyConfig,
) -> ValidatedUtilityRegretInputs:
    """Validate upstream locks without opening a manifest, cache, or raw label."""

    generation_config = load_generation_lock_config(
        config.generation_lock_root / "config.resolved.yaml"
    )
    validate_generation_bundle(config.generation_lock_root, config=generation_config)
    generation_lock = read_generation_lock(
        config.generation_lock_root / "manifests/generation_lock.json"
    )
    if (
        generation_lock.generation_lock_hash != config.expected_generation_lock_hash
        or generation_lock.bank_lock_hash != config.expected_bank_lock_hash
        or generation_config.bank_root.resolve() != config.bank_root.resolve()
        or generation_config.artifact_root.resolve()
        != config.generation_lock_root.resolve()
    ):
        raise ProtocolError("Utility/regret GenerationLock identity drifted.")

    equal_config = load_equal_union_policy_config(
        config.equal_union_root / "config.resolved.yaml"
    )
    validate_equal_union_policy_bundle(config.equal_union_root, config=equal_config)
    equal_lock = read_equal_union_policy_lock(
        config.equal_union_root / "manifests/policy_lock.json"
    )
    equal_payload = equal_lock.to_payload()
    if (
        equal_lock.policy_lock_hash
        != config.expected_equal_union_policy_lock_hash
        or equal_payload.get("policy_plan_hash")
        != config.expected_equal_union_policy_plan_hash
        or equal_payload.get("assignment_table_hash")
        != config.expected_equal_union_assignment_table_hash
        or equal_config.bank_root.resolve() != config.bank_root.resolve()
        or equal_config.generation_lock_root.resolve()
        != config.generation_lock_root.resolve()
        or equal_config.artifact_root.resolve() != config.equal_union_root.resolve()
    ):
        raise ProtocolError("Utility/regret equal-union control identity drifted.")

    utility_lock, content_hash = _validate_utility_artifact(config.utility_root)
    utility_rows = _read_utility_rows(config.utility_root / UTILITY_TABLE_MEMBER)
    case_rows = _read_case_rows(config.utility_root / CASE_CONFUSION_TABLE_MEMBER)
    validate_utility_rows(utility_rows)
    validate_case_confusions(case_rows)
    _validate_utility_case_integrity(utility_rows, case_rows)
    return ValidatedUtilityRegretInputs(
        generation_lock=generation_lock,
        equal_union_policy_lock=equal_lock,
        utility_lock=utility_lock,
        utility_rows=utility_rows,
        case_confusion_rows=case_rows,
        utility_content_hash=content_hash,
    )


def _validate_utility_artifact(
    root: Path,
) -> tuple[Mapping[str, object], str]:
    """Validate the consumer-facing lock surface without reopening labels."""

    symlinks = sorted(
        member.relative_to(root).as_posix()
        for member in root.rglob("*")
        if member.is_symlink()
    )
    if symlinks:
        raise ProtocolError(
            f"Source-inner utility artifact contains symlink members: {symlinks}."
        )
    missing = sorted(
        relative for relative in UTILITY_REQUIRED_FILES if not (root / relative).is_file()
    )
    if missing:
        raise ProtocolError(f"Source-inner utility artifact is incomplete: {missing}.")
    actual = {
        member.relative_to(root).as_posix()
        for member in root.rglob("*")
        if member.is_file()
    }
    if actual != set(UTILITY_REQUIRED_FILES):
        raise ProtocolError("Source-inner utility artifact is not closed-world.")

    content = _json(root / "manifests/content_index.json")
    records = content.get("records")
    unhashed_content = {
        key: value for key, value in content.items() if key != "content_hash"
    }
    if (
        content.get("schema_version")
        != "midogpp_uniform_b_v2_source_inner_utility_content_v1"
        or stable_hash(unhashed_content) != content.get("content_hash")
        or not isinstance(records, list)
    ):
        raise ProtocolError("Source-inner utility content index drifted.")
    indexed: dict[str, Mapping[str, object]] = {}
    for raw in records:
        if not isinstance(raw, Mapping):
            raise ProtocolError("Source-inner utility content row is malformed.")
        relative = str(raw.get("relative_path", ""))
        if (
            set(raw) != {"relative_path", "sha256", "size_bytes"}
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative in indexed
        ):
            raise ProtocolError("Source-inner utility content row drifted.")
        member = root / relative
        if (
            not member.is_file()
            or raw.get("sha256") != _sha256_file(member)
            or raw.get("size_bytes") != member.stat().st_size
        ):
            raise ProtocolError("Source-inner utility content member hash drifted.")
        indexed[relative] = raw
    expected_indexed = set(UTILITY_REQUIRED_FILES).difference(
        {
            "manifests/content_index.json",
            "reports/run_state.json",
            "reports/validation_report.json",
        }
    )
    if set(indexed) != expected_indexed:
        raise ProtocolError("Source-inner utility content coverage drifted.")

    consumption = _json(root / "manifests/policy_consumption_lock.json")
    if (
        consumption.get("policy_consumption_lock_hash")
        != POLICY_CONSUMPTION_LOCK_HASH
        or consumption.get("rule") != policy_consumption_lock_payload()
        or consumption.get("locked_before_validation_labels_opened") is not True
        or consumption.get("label_consumption_authorizes_only_this_rule") is not True
    ):
        raise ProtocolError("Source-inner policy-consumption lock drifted.")

    protocol = _json(root / "manifests/protocol_manifest.json")
    if (
        protocol.get("experiment_id") != UTILITY_EXPERIMENT_ID
        or protocol.get("config_contract_hash")
        != EXPECTED_UTILITY_CONFIG_CONTRACT_HASH
        or protocol.get("generation_lock_hash") != EXPECTED_GENERATION_LOCK_HASH
        or protocol.get("bank_lock_hash") != EXPECTED_BANK_LOCK_HASH
        or protocol.get("policy_consumption_lock_hash")
        != POLICY_CONSUMPTION_LOCK_HASH
        or protocol.get("q_must_differ_from_e") is not True
        or protocol.get("outer_target_instantiated") is not False
        or protocol.get("selection_performed") is not False
        or protocol.get("seed_selection_performed") is not False
    ):
        raise ProtocolError("Source-inner utility protocol drifted.")
    _assert_embedded_hash(protocol, "protocol_hash")

    utility_lock = _json(root / "manifests/utility_lock.json")
    _assert_embedded_hash(utility_lock, "utility_lock_hash")
    member_hashes = utility_lock.get("member_sha256")
    if (
        utility_lock.get("schema_version")
        != "midogpp_uniform_b_v2_source_inner_utility_lock_v1"
        or utility_lock.get("config_contract_hash")
        != EXPECTED_UTILITY_CONFIG_CONTRACT_HASH
        or utility_lock.get("protocol_hash") != protocol.get("protocol_hash")
        or utility_lock.get("policy_consumption_lock_hash")
        != POLICY_CONSUMPTION_LOCK_HASH
        or utility_lock.get("candidate_utility_row_count")
        != EXPECTED_UTILITY_ROW_COUNT
        or utility_lock.get("case_confusion_row_count")
        != EXPECTED_CASE_CONFUSION_ROW_COUNT
        or utility_lock.get("selection_performed") is not False
        or utility_lock.get("alternative_router_tuning_authorized") is not False
        or utility_lock.get("seed_selection_performed") is not False
        or not isinstance(member_hashes, Mapping)
    ):
        raise ProtocolError("Source-inner utility lock drifted.")
    for relative in (UTILITY_TABLE_MEMBER, CASE_CONFUSION_TABLE_MEMBER):
        if member_hashes.get(relative) != _sha256_file(root / relative):
            raise ProtocolError("Source-inner utility table hash drifted.")

    decision = _json(root / "reports/utility_decision.json")
    validation = _json(root / "reports/validation_report.json")
    leakage = _json(root / "reports/leakage_report.json")
    label_consumption = _json(root / "reports/label_consumption_report.json")
    state = _json(root / "reports/run_state.json")
    if (
        decision.get("status")
        != "SOURCE_INNER_UTILITY_READY_FOR_LOCKED_POLICY_CONSUMER"
        or decision.get("utility_lock_hash") != utility_lock.get("utility_lock_hash")
        or decision.get("alternative_router_tuning_authorized") is not False
        or validation.get("status") != "PASS"
        or validation.get("validator") != "validate_source_inner_utility_bundle"
        or leakage.get("status") != "PASS"
        or leakage.get("outer_target_instantiated") is not False
        or leakage.get("policy_selection_performed") is not False
        or label_consumption.get("status")
        != "CONSUMED_FOR_PREDECLARED_POLICY_FAMILY_ONLY"
        or label_consumption.get("policy_consumption_lock_hash")
        != POLICY_CONSUMPTION_LOCK_HASH
        or label_consumption.get("may_authorize_alternative_router_tuning")
        is not False
        or state.get("status") != "COMPLETE"
    ):
        raise ProtocolError("Source-inner utility consumer reports drifted.")
    return utility_lock, str(content["content_hash"])


def _validate_utility_case_integrity(
    utility_rows: Sequence[Mapping[str, object]],
    case_rows: Sequence[Mapping[str, object]],
) -> None:
    """Reconstruct every frozen utility metric from its per-case confusion rows."""

    if len(case_rows) != EXPECTED_CASE_CONFUSION_ROW_COUNT:
        raise ProtocolError("Source-inner case-confusion row coverage drifted.")

    utilities: dict[str, Mapping[str, object]] = {}
    for row in utility_rows:
        utility_row_id = str(row.get("utility_row_id", ""))
        if not utility_row_id or utility_row_id in utilities:
            raise ProtocolError("Source-inner utility-row identity drifted.")
        utilities[utility_row_id] = row

    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for case in case_rows:
        utility_row_id = str(case.get("utility_row_id", ""))
        utility = utilities.get(utility_row_id)
        if utility is None:
            raise ProtocolError("Case-confusion utility-row identity drifted.")
        identity_fields = (
            "pseudo_target_center",
            "candidate_source_center",
            "training_seed",
            "generation_seed",
        )
        if any(case.get(field) != utility.get(field) for field in identity_fields):
            raise ProtocolError("Case-confusion cell identity drifted from utility row.")

        tn, fp, fn, tp, n, true_zero, true_one = (
            _nonnegative_integer(case.get(field), field)
            for field in (
                "tn",
                "fp",
                "fn",
                "tp",
                "n",
                "true_class_0_count",
                "true_class_1_count",
            )
        )
        if (
            tn + fp + fn + tp != n
            or tn + fp != true_zero
            or fn + tp != true_one
        ):
            raise ProtocolError("Case-confusion counts are internally inconsistent.")
        grouped[utility_row_id].append(case)

    if set(grouped) != set(utilities):
        raise ProtocolError("Case-confusion utility-row coverage drifted.")

    for utility_row_id, utility in utilities.items():
        rows = grouped[utility_row_id]
        tn = sum(int(row["tn"]) for row in rows)
        fp = sum(int(row["fp"]) for row in rows)
        fn = sum(int(row["fn"]) for row in rows)
        tp = sum(int(row["tp"]) for row in rows)
        total = sum(int(row["n"]) for row in rows)
        true_zero = sum(int(row["true_class_0_count"]) for row in rows)
        true_one = sum(int(row["true_class_1_count"]) for row in rows)
        if (
            len(rows) != int(utility["eval_case_count"])
            or total != int(utility["eval_row_count"])
            or true_zero != int(utility["eval_class_0_count"])
            or true_one != int(utility["eval_class_1_count"])
        ):
            raise ProtocolError("Case-confusion counts do not cover the utility row.")
        if tn + fp <= 0 or tp + fn <= 0:
            raise ProtocolError("Case-confusion aggregation lacks one true class.")

        bacc = 0.5 * (tn / (tn + fp) + tp / (tp + fn))
        f1_zero_denominator = (2 * tn) + fn + fp
        f1_one_denominator = (2 * tp) + fp + fn
        f1_zero = 0.0 if f1_zero_denominator == 0 else (2 * tn) / f1_zero_denominator
        f1_one = 0.0 if f1_one_denominator == 0 else (2 * tp) / f1_one_denominator
        macro_f1 = 0.5 * (f1_zero + f1_one)
        if not math.isclose(
            bacc, float(utility["bacc"]), rel_tol=0.0, abs_tol=1.0e-15
        ) or not math.isclose(
            macro_f1,
            float(utility["macro_f1"]),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise ProtocolError(
                "Case-confusion metrics do not reconstruct utility metrics."
            )


def _read_utility_rows(path: Path) -> tuple[Mapping[str, object], ...]:
    rows = _read_csv(path, UTILITY_COLUMNS)
    integer_fields = {
        "training_seed",
        "generation_seed",
        "fit_ordinal",
        "prediction_array_row",
        "generated_row_count",
        "generated_rows_per_class",
        "eval_row_count",
        "eval_class_0_count",
        "eval_class_1_count",
        "eval_case_count",
    }
    float_fields = {"bacc", "macro_f1"}
    bool_fields = {
        "classifier_converged",
        "eval_labels_used_for_scoring_only",
        "pseudo_target_expert_excluded",
        "outer_target_instantiated",
        "candidate_ranking_performed",
        "policy_selection_performed",
        "seed_selection_performed",
    }
    return tuple(
        _normalize_row(
            row,
            integer_fields=integer_fields,
            float_fields=float_fields,
            bool_fields=bool_fields,
        )
        for row in rows
    )


def _read_case_rows(path: Path) -> tuple[Mapping[str, object], ...]:
    rows = _read_csv(path, CASE_CONFUSION_COLUMNS)
    integer_fields = {
        "training_seed",
        "generation_seed",
        "tn",
        "fp",
        "fn",
        "tp",
        "n",
        "true_class_0_count",
        "true_class_1_count",
    }
    return tuple(
        _normalize_row(
            row,
            integer_fields=integer_fields,
            float_fields=set(),
            bool_fields={"eval_labels_used_for_scoring_only"},
        )
        for row in rows
    )


def _read_csv(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != columns:
            raise ProtocolError(f"Source-inner utility CSV schema drifted: {path.name}.")
        rows = [dict(row) for row in reader]
    return rows


def _normalize_row(
    row: Mapping[str, str],
    *,
    integer_fields: set[str],
    float_fields: set[str],
    bool_fields: set[str],
) -> Mapping[str, object]:
    normalized: dict[str, object] = dict(row)
    try:
        for field in integer_fields:
            normalized[field] = int(str(row[field]))
        for field in float_fields:
            normalized[field] = float(str(row[field]))
        for field in bool_fields:
            value = str(row[field])
            if value not in {"True", "False"}:
                raise ValueError(field)
            normalized[field] = value == "True"
    except (KeyError, ValueError) as exc:
        raise ProtocolError("Source-inner utility CSV value drifted.") from exc
    return normalized


def _nonnegative_integer(value: object, label: str) -> int:
    try:
        observed = int(str(value))
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"Case-confusion {label} is invalid.") from exc
    if observed < 0:
        raise ProtocolError(f"Case-confusion {label} is negative.")
    return observed


def _assert_embedded_hash(payload: Mapping[str, object], key: str) -> None:
    unhashed = {name: value for name, value in payload.items() if name != key}
    if payload.get(key) != stable_hash(unhashed):
        raise ProtocolError(f"Source-inner utility embedded hash drifted: {key}.")


def _json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read utility/regret upstream JSON: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Utility/regret upstream JSON must be an object: {path}.")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ("ValidatedUtilityRegretInputs", "load_validated_inputs")
