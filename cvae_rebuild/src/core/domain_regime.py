from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.protocol import ProtocolError
from data.splits import candidate_experts


CAMELYON17_DOMAIN_REGIME = "camelyon17_center5"
MIDOGPP_DOMAIN_REGIME = "midogpp_annotation_patch_v1"
CAMELYON17_CENTERS = ("0", "1", "2", "3", "4")


@dataclass(frozen=True)
class MidogppContractInfo:
    artifact_root: Path
    selected_domain_axis: str
    eligible_domain_ids: tuple[str, ...]
    ineligible_domain_ids: tuple[str, ...]
    fingerprints: dict[str, str]
    class_label_names: dict[str, str]
    split_counts: dict[str, int]


def normalize_domain_regime(value: object | None) -> str:
    text = str(value or "").strip()
    return text or CAMELYON17_DOMAIN_REGIME


def validate_unique_domain_ids(domain_ids: Sequence[str], *, label: str = "heldout_centers") -> tuple[str, ...]:
    values = tuple(str(value) for value in domain_ids)
    if not values:
        raise ProtocolError(f"{label} must be non-empty.")
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ProtocolError(f"{label} contains duplicate domain IDs: {duplicates}.")
    return values


def load_midogpp_contract_info(artifact_root: str | Path) -> MidogppContractInfo:
    root = Path(artifact_root)
    required = (
        "dataset_contract.json",
        "domain_mapping.json",
        "domain_feasibility.csv",
        "manifest.csv",
    )
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise ProtocolError(f"MIDOG++ contract artifact is missing required files: {missing}.")

    contract = _read_json(root / "dataset_contract.json")
    mapping = _read_json(root / "domain_mapping.json")
    feasibility_rows = _read_csv(root / "domain_feasibility.csv")
    manifest_rows = _read_csv(root / "manifest.csv")
    selected_axis = str(contract.get("domain_policy", {}).get("selected_domain_axis") or mapping.get("domain_axis") or "")
    if not selected_axis:
        raise ProtocolError("MIDOG++ contract is missing domain_policy.selected_domain_axis.")

    selected_feasibility = [row for row in feasibility_rows if str(row.get("domain_axis", "")) == selected_axis]
    if not selected_feasibility:
        raise ProtocolError(f"MIDOG++ domain_feasibility.csv has no rows for selected axis {selected_axis!r}.")

    mapped_ids = {str(row.get("domain_id", "")) for row in mapping.get("domains", [])}
    eligible = tuple(
        sorted(
            (
                str(row.get("domain_id_for_axis", "")).strip()
                for row in selected_feasibility
                if _truthy(row.get("eligible"))
            ),
            key=_domain_sort_key,
        )
    )
    ineligible = tuple(
        sorted(
            (
                str(row.get("domain_id_for_axis", "")).strip()
                for row in selected_feasibility
                if not _truthy(row.get("eligible"))
            ),
            key=_domain_sort_key,
        )
    )
    if not eligible:
        raise ProtocolError("MIDOG++ contract has no eligible domains.")
    missing_from_mapping = sorted(set(eligible) - mapped_ids, key=_domain_sort_key)
    if missing_from_mapping:
        raise ProtocolError(f"MIDOG++ eligible domains missing from domain_mapping.json: {missing_from_mapping}.")

    split_counts: dict[str, int] = {}
    labels = {str(row.get("label", "")) for row in manifest_rows}
    for row in manifest_rows:
        split = str(row.get("split", ""))
        split_counts[split] = split_counts.get(split, 0) + 1
    if not {"0", "1"}.issubset(labels):
        raise ProtocolError("MIDOG++ manifest must contain class labels 0 and 1.")

    return MidogppContractInfo(
        artifact_root=root,
        selected_domain_axis=selected_axis,
        eligible_domain_ids=eligible,
        ineligible_domain_ids=ineligible,
        fingerprints={name: _sha256(root / name) for name in required},
        class_label_names={"0": "hard_negative", "1": "mitotic"},
        split_counts=split_counts,
    )


def validate_domain_regime_config(
    *,
    domain_regime: str,
    heldout_centers: Sequence[str],
    dataset_contract_artifact_root: str | Path | None,
    artifact_root: Path,
    strict_full_run_matrix: bool = False,
    strict_available_seed_domain_coverage: bool = False,
) -> MidogppContractInfo | None:
    centers = validate_unique_domain_ids(tuple(str(value) for value in heldout_centers))
    regime = normalize_domain_regime(domain_regime)
    if regime == CAMELYON17_DOMAIN_REGIME:
        if centers != CAMELYON17_CENTERS:
            raise ProtocolError("Camelyon17 domain_regime requires heldout_centers=['0', '1', '2', '3', '4'].")
        return None
    if regime != MIDOGPP_DOMAIN_REGIME:
        raise ProtocolError(f"Unsupported domain_regime={regime!r}.")
    if strict_full_run_matrix:
        raise ProtocolError("MIDOG++ v1 must not use strict_full_run_matrix; use strict_available_seed_domain_coverage.")
    if not strict_available_seed_domain_coverage:
        raise ProtocolError("MIDOG++ v1 requires strict_available_seed_domain_coverage=true.")
    if dataset_contract_artifact_root is None:
        raise ProtocolError("MIDOG++ v1 requires inputs.dataset_contract_artifact_root.")
    if "cvae_rebuild/artifacts/midogpp" not in artifact_root.as_posix():
        raise ProtocolError("MIDOG++ C6.3 artifact_root must be under cvae_rebuild/artifacts/midogpp/.")
    info = load_midogpp_contract_info(dataset_contract_artifact_root)
    if centers != info.eligible_domain_ids:
        raise ProtocolError(
            "MIDOG++ heldout_centers must exactly match contract-derived eligible domains: "
            f"expected {list(info.eligible_domain_ids)}, got {list(centers)}."
        )
    return info


def validate_cache_report_split_counts(cache_report_path: str | Path | None, contract_info: MidogppContractInfo | None) -> None:
    if cache_report_path is None or contract_info is None:
        return
    report = _read_json(Path(cache_report_path))
    split_counts = report.get("split_counts", {})
    if not isinstance(split_counts, Mapping):
        raise ProtocolError("cache_report_path JSON must contain split_counts.")
    reported = {str(key): int(value) for key, value in split_counts.items()}
    expected = {str(key): int(value) for key, value in contract_info.split_counts.items() if key}
    if reported != expected:
        raise ProtocolError(f"cache report split_counts contradict MIDOG++ manifest: expected {expected}, got {reported}.")


def validate_runtime_domain_coverage(
    *,
    domain_regime: str,
    eligible_domain_ids: Sequence[str],
    experiment_seed: int,
    train_metadata: Sequence[Mapping[str, object]],
    test_metadata: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if normalize_domain_regime(domain_regime) != MIDOGPP_DOMAIN_REGIME:
        return []
    eligible = tuple(str(value) for value in eligible_domain_ids)
    train_centers = {str(row.get("center", "")) for row in train_metadata}
    test_centers = {str(row.get("center", "")) for row in test_metadata}
    labels = {str(row.get("label", "")) for row in (*train_metadata, *test_metadata)}
    missing_train = sorted(set(eligible) - train_centers, key=_domain_sort_key)
    missing_test = sorted(set(eligible) - test_centers, key=_domain_sort_key)
    if missing_train or missing_test:
        raise ProtocolError(
            "MIDOG++ cache coverage is incomplete for seed "
            f"{experiment_seed}: missing_train={missing_train}, missing_test={missing_test}."
        )
    if not labels.issubset({"0", "1"}):
        raise ProtocolError(f"MIDOG++ cache labels must be 0/1, got {sorted(labels)}.")

    extra_train = sorted(train_centers - set(eligible), key=_domain_sort_key)
    extra_test = sorted(test_centers - set(eligible), key=_domain_sort_key)
    extra_any = sorted(set(extra_train).union(extra_test), key=_domain_sort_key)

    rows = []
    for heldout in eligible:
        sources = candidate_experts(eligible, heldout)
        if len(sources) != len(eligible) - 1 or heldout in sources:
            raise ProtocolError(f"MIDOG++ source pool is invalid for heldout={heldout}.")
        rows.append(
            {
                "experiment_seed": int(experiment_seed),
                "domain_regime": MIDOGPP_DOMAIN_REGIME,
                "heldout_domain_id": heldout,
                "source_domain_ids": json.dumps(list(sources)),
                "expected_source_count": int(len(eligible) - 1),
                "actual_source_count": int(len(sources)),
                "domain_4_excluded": "4" not in set(sources) and heldout != "4",
                "all_eligible_heldouts_complete": True,
                "target_expert_excluded": True,
                "cache_extra_domain_ids": json.dumps(extra_any),
                "cache_extra_train_domain_ids": json.dumps(extra_train),
                "cache_extra_test_domain_ids": json.dumps(extra_test),
            }
        )
    return rows


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ProtocolError(f"Could not read JSON file {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ProtocolError(f"JSON file must contain an object: {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except Exception as exc:
        raise ProtocolError(f"Could not read CSV file {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _domain_sort_key(value: str) -> tuple[int, object]:
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)
