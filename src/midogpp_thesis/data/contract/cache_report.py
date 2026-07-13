from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "midogpp_cache_report_v1"
C63_BLOCK_REASON = "blocked_until_positive_union_generalized"
REQUIRED_ARTIFACT_FILES = (
    "dataset_contract.json",
    "domain_mapping.json",
    "domain_feasibility.csv",
    "manifest.csv",
)


class CacheReportError(RuntimeError):
    pass


def build_cache_domain_report(
    artifact_root: str | Path,
    *,
    cache_report_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(artifact_root)
    _require_files(root, REQUIRED_ARTIFACT_FILES)

    contract = _read_json(root / "dataset_contract.json")
    domain_mapping = _read_json(root / "domain_mapping.json")
    feasibility_rows = _read_csv(root / "domain_feasibility.csv")
    manifest_rows = _read_csv(root / "manifest.csv")
    cache_report = _load_optional_cache_report(cache_report_path)

    domain_axis = _selected_domain_axis(contract, domain_mapping)
    mapped_domains = _mapped_domains(domain_mapping)
    mapped_ids = {str(row["domain_id"]) for row in mapped_domains}
    selected_feasibility = [row for row in feasibility_rows if str(row.get("domain_axis", "")) == domain_axis]
    if not selected_feasibility:
        raise CacheReportError(f"domain_feasibility.csv has no rows for selected domain_axis={domain_axis!r}")

    _assert_no_duplicate_domain_ids(mapped_domains)
    manifest_counts = _manifest_counts(manifest_rows)
    _assert_manifest_domain_ids_mapped(manifest_counts["by_domain"], mapped_ids)

    eligible_rows = [row for row in selected_feasibility if _truthy(row.get("eligible", ""))]
    ineligible_rows = [row for row in selected_feasibility if not _truthy(row.get("eligible", ""))]
    eligible_ids = [str(row.get("domain_id_for_axis", "")).strip() for row in eligible_rows]
    _assert_eligible_domains_valid(eligible_rows, mapped_ids, manifest_counts["by_domain"])
    _assert_feasibility_matches_manifest(selected_feasibility, manifest_rows)

    warnings: list[str] = []
    ineligible_manifest_ids = [str(row.get("domain_id_for_axis", "")).strip() for row in ineligible_rows if int(manifest_counts["by_domain"].get(str(row.get("domain_id_for_axis", "")).strip(), {}).get("total_rows", 0)) > 0]
    if ineligible_manifest_ids:
        warnings.append(
            "manifest_contains_ineligible_domains: "
            + ",".join(_sort_domain_ids(ineligible_manifest_ids))
            + "; downstream configs should use only eligible_domain_ids"
        )

    cache_report_payload = _cache_report_payload(cache_report, cache_report_path, manifest_counts)
    sail_cache_root = _sail_cache_root_hint(contract, cache_report)
    c63_cache_root = f"{sail_cache_root}/virchow2"
    warnings.append("c63_blocked_until_positive_union_generalized")

    mapped_domain_rows = _mapped_domain_rows(mapped_domains, selected_feasibility, manifest_counts["by_domain"])
    ineligible_domains = _ineligible_domain_rows(ineligible_rows, manifest_counts["by_domain"])
    candidate_centers = [_json_domain_id(value) for value in _sort_domain_ids(eligible_ids)]

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_root": str(root),
        "domain_axis": domain_axis,
        "mapped_domains": mapped_domain_rows,
        "eligible_domain_ids": candidate_centers,
        "ineligible_domains": ineligible_domains,
        "manifest_counts": manifest_counts,
        "cache_report": cache_report_payload,
        "hints": {
            "sail": {
                "cache_root": sail_cache_root,
                "cache_path_template": (
                    "{cache_root}/{backbone}/annotation_patch_xyxy/"
                    "seed{seed}/embeddings/{split}.pt"
                ),
                "candidate_centers": candidate_centers,
            },
            "c63": {
                "cache_root": c63_cache_root,
                "blocked": True,
                "reason": C63_BLOCK_REASON,
            },
        },
        "warnings": warnings,
    }


def format_cache_domain_report(report: Mapping[str, Any]) -> str:
    lines = [
        "MIDOG++ Cache/Domain Bridge",
        f"artifact_root: {report['artifact_root']}",
        f"domain_axis: {report['domain_axis']}",
        f"eligible_domain_ids: {_format_id_list(report['eligible_domain_ids'])}",
        "",
        "Mapped pseudo-domains:",
    ]
    for row in report.get("mapped_domains", []):
        status = "eligible" if row.get("eligible") else "ineligible"
        reason = f" ({row.get('ineligible_reasons')})" if row.get("ineligible_reasons") else ""
        lines.append(
            f"  {row['domain_id']}: {row['domain_name']} "
            f"[{status}{reason}; manifest_rows={row.get('manifest_rows', 0)}]"
        )

    ineligible = report.get("ineligible_domains", [])
    lines.append("")
    lines.append("Ineligible pseudo-domains:")
    if ineligible:
        for row in ineligible:
            lines.append(f"  {row['domain_id']}: {row['domain_name']} - {row.get('reason', '')}")
    else:
        lines.append("  none")

    manifest_counts = report.get("manifest_counts", {})
    lines.extend(
        [
            "",
            "Manifest counts:",
            f"  total_rows: {manifest_counts.get('total_rows', 0)}",
            f"  by_split: {json.dumps(manifest_counts.get('by_split', {}), sort_keys=True)}",
            f"  by_label: {json.dumps(manifest_counts.get('by_label', {}), sort_keys=True)}",
        ]
    )

    cache = report.get("cache_report", {})
    lines.append("")
    if cache.get("provided"):
        lines.append(f"Cache report: {cache.get('path')}")
        lines.append(f"  split_counts: {json.dumps(cache.get('counts', {}), sort_keys=True)}")
    else:
        lines.append("Cache report: not provided")

    hints = report.get("hints", {})
    sail = hints.get("sail", {})
    c63 = hints.get("c63", {})
    lines.extend(
        [
            "",
            "SAIL config hint:",
            "```yaml",
            "feature_cache:",
            f"  cache_root: {sail.get('cache_root', '')}",
            '  cache_path_template: "{cache_root}/{backbone}/seed{seed}/embeddings/{split}.pt"',
            "datasets:",
            "  camelyon17:",
            f"    candidate_centers: {_format_id_list(sail.get('candidate_centers', []))}",
            "```",
            "",
            "C6.3 hint:",
            f"  cache_root: {c63.get('cache_root', '')}",
            "  blocked: true",
            f"  reason: {c63.get('reason', C63_BLOCK_REASON)}",
            "  MIDOG++ C6.3 execution is blocked until positive-union logic is generalized beyond the Camelyon 5-domain assumption.",
        ]
    )

    warnings = list(report.get("warnings", [])) + list(cache.get("warnings", []))
    lines.append("")
    lines.append("Warnings:")
    if warnings:
        for warning in warnings:
            lines.append(f"  - {warning}")
    else:
        lines.append("  none")
    return "\n".join(lines)


def _require_files(root: Path, filenames: Sequence[str]) -> None:
    missing = [name for name in filenames if not (root / name).exists()]
    if missing:
        raise CacheReportError(f"missing required artifact files: {missing}")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CacheReportError(f"malformed JSON: {path}: {type(exc).__name__}: {exc}") from exc


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise CacheReportError(f"empty CSV: {path}")
            return [dict(row) for row in reader]
    except CacheReportError:
        raise
    except Exception as exc:
        raise CacheReportError(f"malformed CSV: {path}: {type(exc).__name__}: {exc}") from exc


def _load_optional_cache_report(cache_report_path: str | Path | None) -> Mapping[str, Any] | None:
    if cache_report_path is None:
        return None
    path = Path(cache_report_path)
    if not path.exists():
        raise CacheReportError(f"cache report path does not exist: {path}")
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        raise CacheReportError(f"cache report must be a JSON object: {path}")
    return payload


def _selected_domain_axis(contract: Mapping[str, Any], domain_mapping: Mapping[str, Any]) -> str:
    policy = contract.get("domain_policy", {}) if isinstance(contract.get("domain_policy"), Mapping) else {}
    axis = str(policy.get("selected_domain_axis") or domain_mapping.get("domain_axis") or "").strip()
    if not axis:
        raise CacheReportError("could not determine selected domain axis")
    return axis


def _mapped_domains(domain_mapping: Mapping[str, Any]) -> list[dict[str, Any]]:
    domains = domain_mapping.get("domains", [])
    if not isinstance(domains, list):
        raise CacheReportError("domain_mapping.json field 'domains' must be a list")
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(domains):
        if not isinstance(row, Mapping):
            raise CacheReportError(f"domain_mapping.json domains[{idx}] must be an object")
        domain_id = str(row.get("domain_id", "")).strip()
        domain_name = str(row.get("domain_name", "")).strip()
        if not domain_id or not domain_name:
            raise CacheReportError(f"domain_mapping.json domains[{idx}] lacks domain_id/domain_name")
        rows.append(
            {
                "domain_id": domain_id,
                "domain_name": domain_name,
                "n_cases": _int_or_none(row.get("n_cases")),
                "n_rows": _int_or_none(row.get("n_rows")),
            }
        )
    return rows


def _assert_no_duplicate_domain_ids(mapped_domains: Sequence[Mapping[str, Any]]) -> None:
    ids = [str(row["domain_id"]) for row in mapped_domains]
    duplicates = sorted({domain_id for domain_id in ids if ids.count(domain_id) > 1})
    if duplicates:
        raise CacheReportError(f"duplicate domain_id values in domain_mapping.json: {duplicates}")


def _assert_manifest_domain_ids_mapped(by_domain: Mapping[str, Any], mapped_ids: set[str]) -> None:
    manifest_ids = set(by_domain)
    missing = sorted(manifest_ids.difference(mapped_ids), key=_domain_sort_key)
    if missing:
        raise CacheReportError(f"manifest.csv contains domain_id values missing from domain_mapping.json: {missing}")


def _assert_eligible_domains_valid(
    eligible_rows: Sequence[Mapping[str, str]],
    mapped_ids: set[str],
    by_domain: Mapping[str, Mapping[str, Any]],
) -> None:
    for row in eligible_rows:
        domain_id = str(row.get("domain_id_for_axis", "")).strip()
        if domain_id not in mapped_ids:
            raise CacheReportError(f"eligible domain_id={domain_id!r} is missing from domain_mapping.json")
        if int(by_domain.get(domain_id, {}).get("total_rows", 0)) <= 0:
            raise CacheReportError(f"eligible domain_id={domain_id!r} has zero manifest samples")


def _assert_feasibility_matches_manifest(
    feasibility_rows: Sequence[Mapping[str, str]],
    manifest_rows: Sequence[Mapping[str, str]],
) -> None:
    for row in feasibility_rows:
        domain_id = str(row.get("domain_id_for_axis", "")).strip()
        stats = _manifest_stats_for_domain(manifest_rows, domain_id)
        for key in (
            "total_rows",
            "total_cases",
            "train_cases",
            "eval_cases",
            "train_positives",
            "train_negatives",
            "eval_positives",
            "eval_negatives",
        ):
            expected = _int_or_none(row.get(key))
            if expected is not None and int(stats[key]) != int(expected):
                raise CacheReportError(
                    f"domain_feasibility.csv contradicts manifest.csv for domain_id={domain_id!r} "
                    f"field={key}: feasibility={expected} manifest={stats[key]}"
                )


def _manifest_counts(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    by_split: dict[str, int] = {}
    by_label: dict[str, int] = {}
    by_domain: dict[str, dict[str, Any]] = {}
    by_domain_split: dict[str, dict[str, int]] = {}
    by_domain_label: dict[str, dict[str, int]] = {}
    for row in rows:
        split = str(row.get("split", "")).strip()
        label = str(int(float(str(row.get("label", "0")))))
        domain_id = str(row.get("domain_id", "")).strip()
        case_id = str(row.get("case_id", "")).strip()
        if not domain_id:
            raise CacheReportError("manifest.csv row missing domain_id")
        by_split[split] = by_split.get(split, 0) + 1
        by_label[label] = by_label.get(label, 0) + 1
        by_domain.setdefault(domain_id, {"total_rows": 0, "case_ids": set()})
        by_domain[domain_id]["total_rows"] += 1
        by_domain[domain_id]["case_ids"].add(case_id)
        by_domain_split.setdefault(domain_id, {})
        by_domain_split[domain_id][split] = by_domain_split[domain_id].get(split, 0) + 1
        by_domain_label.setdefault(domain_id, {})
        by_domain_label[domain_id][label] = by_domain_label[domain_id].get(label, 0) + 1

    serial_domain = {}
    for domain_id, counts in by_domain.items():
        serial_domain[domain_id] = {
            "total_rows": int(counts["total_rows"]),
            "case_count": len(counts["case_ids"]),
        }
    return {
        "total_rows": len(rows),
        "by_split": dict(sorted(by_split.items())),
        "by_label": dict(sorted(by_label.items())),
        "by_domain": dict(sorted(serial_domain.items(), key=lambda item: _domain_sort_key(item[0]))),
        "by_domain_split": {key: dict(sorted(value.items())) for key, value in sorted(by_domain_split.items(), key=lambda item: _domain_sort_key(item[0]))},
        "by_domain_label": {key: dict(sorted(value.items())) for key, value in sorted(by_domain_label.items(), key=lambda item: _domain_sort_key(item[0]))},
    }


def _manifest_stats_for_domain(rows: Sequence[Mapping[str, str]], domain_id: str) -> dict[str, int]:
    domain_rows = [row for row in rows if str(row.get("domain_id", "")).strip() == str(domain_id)]
    train_rows = [row for row in domain_rows if str(row.get("split", "")) == "train"]
    eval_rows = [row for row in domain_rows if str(row.get("split", "")) == "test"]
    return {
        "total_rows": len(domain_rows),
        "total_cases": len({str(row.get("case_id", "")) for row in domain_rows}),
        "train_cases": len({str(row.get("case_id", "")) for row in train_rows}),
        "eval_cases": len({str(row.get("case_id", "")) for row in eval_rows}),
        "train_positives": _label_count(train_rows, 1),
        "train_negatives": _label_count(train_rows, 0),
        "eval_positives": _label_count(eval_rows, 1),
        "eval_negatives": _label_count(eval_rows, 0),
    }


def _cache_report_payload(
    cache_report: Mapping[str, Any] | None,
    cache_report_path: str | Path | None,
    manifest_counts: Mapping[str, Any],
) -> dict[str, Any]:
    if cache_report is None:
        return {
            "provided": False,
            "path": "",
            "counts": {},
            "warnings": ["optional_cache_report_absent"],
        }
    split_counts = cache_report.get("split_counts", {})
    if not isinstance(split_counts, Mapping):
        raise CacheReportError("cache report field 'split_counts' must be a mapping")
    cache_counts = {str(key): int(value) for key, value in split_counts.items()}
    manifest_split = {str(key): int(value) for key, value in manifest_counts.get("by_split", {}).items()}
    if cache_counts != manifest_split:
        raise CacheReportError(f"cache report split_counts contradict manifest.csv: cache={cache_counts} manifest={manifest_split}")
    return {
        "provided": True,
        "path": str(cache_report_path),
        "counts": cache_counts,
        "warnings": [],
    }


def _sail_cache_root_hint(contract: Mapping[str, Any], cache_report: Mapping[str, Any] | None) -> str:
    if cache_report is not None:
        output_paths = cache_report.get("output_paths", {})
        backbone = str(cache_report.get("backbone_name", "virchow2"))
        if isinstance(output_paths, Mapping) and output_paths:
            first = Path(str(next(iter(output_paths.values()))))
            parts = first.parts
            if backbone in parts:
                idx = parts.index(backbone)
                return Path(*parts[:idx]).as_posix()
    return "datasets/midogpp/derived/features"


def _mapped_domain_rows(
    mapped_domains: Sequence[Mapping[str, Any]],
    feasibility_rows: Sequence[Mapping[str, str]],
    by_domain: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    feasibility_by_id = {str(row.get("domain_id_for_axis", "")).strip(): row for row in feasibility_rows}
    out = []
    for row in sorted(mapped_domains, key=lambda item: _domain_sort_key(str(item["domain_id"]))):
        domain_id = str(row["domain_id"])
        feasibility = feasibility_by_id.get(domain_id, {})
        counts = by_domain.get(domain_id, {})
        out.append(
            {
                "domain_id": _json_domain_id(domain_id),
                "domain_id_raw": domain_id,
                "domain_name": row["domain_name"],
                "n_cases": row.get("n_cases"),
                "n_rows": row.get("n_rows"),
                "eligible": _truthy(feasibility.get("eligible", "")),
                "ineligible_reasons": str(feasibility.get("ineligible_reasons", "")),
                "manifest_rows": int(counts.get("total_rows", 0)),
                "manifest_cases": int(counts.get("case_count", 0)),
            }
        )
    return out


def _ineligible_domain_rows(
    rows: Sequence[Mapping[str, str]],
    by_domain: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    out = []
    for row in sorted(rows, key=lambda item: _domain_sort_key(str(item.get("domain_id_for_axis", "")))):
        domain_id = str(row.get("domain_id_for_axis", "")).strip()
        out.append(
            {
                "domain_id": _json_domain_id(domain_id),
                "domain_id_raw": domain_id,
                "domain_name": str(row.get("domain_name", "")),
                "reason": str(row.get("ineligible_reasons", "")),
                "manifest_rows": int(by_domain.get(domain_id, {}).get("total_rows", 0)),
            }
        )
    return out


def _label_count(rows: Sequence[Mapping[str, str]], label: int) -> int:
    return sum(1 for row in rows if int(float(str(row.get("label", -1)))) == int(label))


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _int_or_none(value: Any) -> int | None:
    text = str(value if value is not None else "").strip()
    if text == "":
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def _sort_domain_ids(values: Sequence[str]) -> list[str]:
    return sorted((str(value) for value in values), key=_domain_sort_key)


def _domain_sort_key(value: str) -> tuple[int, Any]:
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def _json_domain_id(value: str) -> int | str:
    text = str(value)
    try:
        return int(text)
    except ValueError:
        return text


def _format_id_list(values: Any) -> str:
    return "[" + ", ".join(str(value) for value in values) + "]"
