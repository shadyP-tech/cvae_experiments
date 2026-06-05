from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from .reporting import repo_relative, sha256_file, write_csv_rows, write_json


MANIFEST_COLUMNS = (
    "sample_id",
    "case_id",
    "image_path",
    "annotation_id",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "patch_center_x",
    "patch_center_y",
    "label",
    "label_name",
    "scanner_model",
    "lab_or_origin",
    "tumor_type",
    "species",
    "resolution",
    "domain_axis",
    "domain_name",
    "domain_id",
    "center",
    "magnification",
    "split",
    "negative_match_scope",
)

DEFAULT_CANDIDATE_AXES = (
    "scanner_model",
    "tumor_type",
    "tumor_type|lab_or_origin|scanner_model",
)


@dataclass(frozen=True)
class EligibilityThresholds:
    total_cases_min: int
    train_cases_min: int
    eval_cases_min: int
    train_positives_min: int
    train_negatives_min: int
    eval_positives_min: int
    eval_negatives_min: int

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "EligibilityThresholds":
        return cls(
            total_cases_min=int(data.get("total_cases_min", 20)),
            train_cases_min=int(data.get("train_cases_min", 10)),
            eval_cases_min=int(data.get("eval_cases_min", 10)),
            train_positives_min=int(data.get("train_positives_min", 50)),
            train_negatives_min=int(data.get("train_negatives_min", 50)),
            eval_positives_min=int(data.get("eval_positives_min", 20)),
            eval_negatives_min=int(data.get("eval_negatives_min", 20)),
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "total_cases_min": self.total_cases_min,
            "train_cases_min": self.train_cases_min,
            "eval_cases_min": self.eval_cases_min,
            "train_positives_min": self.train_positives_min,
            "train_negatives_min": self.train_negatives_min,
            "eval_positives_min": self.eval_positives_min,
            "eval_negatives_min": self.eval_negatives_min,
        }


@dataclass(frozen=True)
class BuilderConfig:
    repo_root: Path
    artifact_name: str
    artifact_root: Path
    input_root: Path
    metadata_path: Path
    annotations_path: Path
    patch_dir: Path
    patch_size: int
    image_quality: int
    bbox_format: str
    positive_policy: str
    negative_policy: str
    negative_ratio: float
    negative_seed: int
    split_seed: int
    split_fractions: Mapping[str, float]
    candidate_axes: tuple[str, ...]
    preferred_axis: str
    final_axis_policy: str
    min_eligible_domains: int
    eligibility: EligibilityThresholds

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, repo_root: Path | None = None) -> "BuilderConfig":
        root = Path(repo_root or Path.cwd()).resolve()
        artifact = _as_mapping(data.get("artifact", {}))
        inputs = _as_mapping(data.get("inputs", {}))
        patches = _as_mapping(data.get("patches", {}))
        sampling = _as_mapping(data.get("sampling", {}))
        split = _as_mapping(data.get("split", {}))
        domain = _as_mapping(data.get("domain", {}))
        eligibility = EligibilityThresholds.from_mapping(_as_mapping(data.get("eligibility", {})))

        artifact_name = str(artifact.get("name", "midogpp_annotation_patch_v1"))
        artifact_root = _config_path(root, artifact.get("root", f"datasets/midogpp/artifacts/{artifact_name}"))
        input_root = _config_path(root, inputs.get("root", "cvae_testing/data/MIDOGpp"))
        metadata_path = _config_path(root, inputs.get("metadata", input_root / "metadata.csv"))
        annotations_path = _config_path(root, inputs.get("annotations", input_root / "databases/MIDOG++.json"))
        patch_dir = _config_path(root, patches.get("patch_dir", artifact_root / "patches_224"))

        candidate_axes = tuple(str(axis) for axis in domain.get("candidate_axes", DEFAULT_CANDIDATE_AXES))
        preferred_axis = str(domain.get("preferred_axis", "tumor_type|lab_or_origin|scanner_model"))
        if preferred_axis not in candidate_axes:
            candidate_axes = (*candidate_axes, preferred_axis)

        cfg = cls(
            repo_root=root,
            artifact_name=artifact_name,
            artifact_root=artifact_root,
            input_root=input_root,
            metadata_path=metadata_path,
            annotations_path=annotations_path,
            patch_dir=patch_dir,
            patch_size=int(patches.get("patch_size", 224)),
            image_quality=int(patches.get("image_quality", 92)),
            bbox_format=str(patches.get("bbox_format", "coco_xywh")),
            positive_policy=str(sampling.get("positive_policy", "all_valid_mitotic_annotations")),
            negative_policy=str(sampling.get("negative_policy", "matched_1_to_1")),
            negative_ratio=float(sampling.get("negative_ratio", 1.0)),
            negative_seed=int(sampling.get("negative_seed", 42)),
            split_seed=int(split.get("seed", 42)),
            split_fractions=_as_mapping(split.get("fractions", {"train": 0.45, "val": 0.10, "test": 0.45})),
            candidate_axes=candidate_axes,
            preferred_axis=preferred_axis,
            final_axis_policy=str(domain.get("final_axis_policy", "auto_tumor_lab_scanner")),
            min_eligible_domains=int(domain.get("min_eligible_domains", 6)),
            eligibility=eligibility,
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if self.positive_policy != "all_valid_mitotic_annotations":
            raise ValueError("Only positive_policy=all_valid_mitotic_annotations is supported")
        if self.negative_policy != "matched_1_to_1":
            raise ValueError("Only negative_policy=matched_1_to_1 is supported")
        if self.negative_ratio != 1.0:
            raise ValueError("Only negative_ratio=1.0 is supported by this contract")
        if self.preferred_axis != "tumor_type|lab_or_origin|scanner_model":
            raise ValueError("Final MIDOG++ contract axis must be tumor_type|lab_or_origin|scanner_model")


@dataclass(frozen=True)
class ContractBuildResult:
    status: str
    artifact_root: Path
    manifest_path: Path
    contract_path: Path
    eligible_domain_count: int
    report: Mapping[str, Any]


@dataclass(frozen=True)
class _SourceRecord:
    source_index: int
    image_ref: str
    image_path: Path
    case_id: str
    scanner_model: str
    lab_or_origin: str
    tumor_type: str
    species: str
    resolution: str


@dataclass(frozen=True)
class _AnnotationRecord:
    annotation_id: str
    image_key: str
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    patch_center_x: float
    patch_center_y: float
    label: int
    label_name: str


@dataclass(frozen=True)
class _SampleRecord:
    source: _SourceRecord
    annotation: _AnnotationRecord
    split: str
    sample_id: str
    patch_path: Path
    negative_match_scope: str

    @property
    def label(self) -> int:
        return int(self.annotation.label)


def load_config(path: str | Path, *, repo_root: Path | None = None) -> BuilderConfig:
    config_path = Path(path)
    if repo_root is None:
        repo_root = _find_repo_root(config_path if config_path.is_absolute() else Path.cwd())
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Loading MIDOG++ YAML configs requires PyYAML") from exc
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return BuilderConfig.from_mapping(payload, repo_root=repo_root)


def build_contract(config: BuilderConfig, *, overwrite: bool = False) -> ContractBuildResult:
    if config.artifact_root.exists() and not overwrite and (config.artifact_root / "manifest.csv").exists():
        raise FileExistsError(f"Contract artifact already exists: {config.artifact_root}. Use --overwrite to rebuild.")
    config.artifact_root.mkdir(parents=True, exist_ok=True)
    config.patch_dir.mkdir(parents=True, exist_ok=True)

    sources = _load_sources(config)
    annotations_by_image = _load_annotations(config.annotations_path, bbox_format=config.bbox_format)
    case_splits = _assign_case_splits(sources, axis=config.preferred_axis, config=config)
    candidates, skipped_annotations = _collect_candidates(sources, annotations_by_image, case_splits)
    selected, matching_report = _select_matched_samples(candidates, config)
    selected = _dedupe_sample_ids(selected)

    manifest_rows = _write_patches_and_manifest_rows(selected, config, overwrite=overwrite)
    domain_mapping = _build_domain_mapping(manifest_rows, config)
    manifest_rows = _apply_final_domain_ids(manifest_rows, domain_mapping)

    feasibility_rows = _domain_feasibility_rows(manifest_rows, config)
    class_balance_rows = _class_balance_rows(manifest_rows, config)
    split_manifest_rows = _split_manifest_rows(sources, case_splits, config)
    leakage_report = _leakage_report(manifest_rows, matching_report)

    preferred_feasibility = [row for row in feasibility_rows if row["domain_axis"] == config.preferred_axis]
    eligible_domain_count = sum(1 for row in preferred_feasibility if bool(row["eligible"]))
    final_axis_frozen = eligible_domain_count >= int(config.min_eligible_domains)
    status = "pass" if final_axis_frozen and leakage_report["status"] == "PASS" else "blocked_insufficient_eligible_domains"
    if leakage_report["status"] != "PASS":
        status = "fail_leakage"

    contract = _dataset_contract(
        config=config,
        manifest_rows=manifest_rows,
        domain_mapping=domain_mapping,
        matching_report=matching_report,
        skipped_annotations=skipped_annotations,
        leakage_report=leakage_report,
        eligible_domain_count=eligible_domain_count,
        final_axis_frozen=final_axis_frozen,
        status=status,
    )
    domain_mapping = {
        **domain_mapping,
        "final_axis_frozen": bool(final_axis_frozen),
        "eligible_domain_count": int(eligible_domain_count),
        "status": status,
    }

    manifest_path = config.artifact_root / "manifest.csv"
    write_csv_rows(manifest_path, manifest_rows, fieldnames=MANIFEST_COLUMNS)
    write_csv_rows(config.artifact_root / "split_manifest.csv", split_manifest_rows)
    write_csv_rows(config.artifact_root / "domain_feasibility.csv", feasibility_rows)
    write_csv_rows(config.artifact_root / "class_balance_by_domain.csv", class_balance_rows)
    write_json(config.artifact_root / "domain_mapping.json", domain_mapping)
    write_json(config.artifact_root / "leakage_report.json", leakage_report)
    write_json(config.artifact_root / "dataset_contract.json", contract)

    return ContractBuildResult(
        status=status,
        artifact_root=config.artifact_root,
        manifest_path=manifest_path,
        contract_path=config.artifact_root / "dataset_contract.json",
        eligible_domain_count=eligible_domain_count,
        report={
            "status": status,
            "artifact_root": str(config.artifact_root),
            "manifest_rows": len(manifest_rows),
            "eligible_domain_count": eligible_domain_count,
            "final_axis_frozen": final_axis_frozen,
            "unmatched_positive_count": matching_report["unmatched_positive_count"],
        },
    )


def domain_name(row: Mapping[str, Any], axis: str) -> str:
    parts = []
    for field in str(axis).split("|"):
        value = _clean(row.get(field, ""))
        parts.append(value or "missing")
    return "|".join(parts)


def _load_sources(config: BuilderConfig) -> list[_SourceRecord]:
    rows = _read_csv_dicts(config.metadata_path)
    sources: list[_SourceRecord] = []
    for idx, row in enumerate(rows):
        image_ref = _first(row, ("image_path", "file_name", "filename", "File", "Image", "image"))
        if not image_ref:
            continue
        image_path = _resolve_image_path(config, image_ref)
        case_id = _first(row, ("case_id", "specimen_id", "slide_id", "patient_id", "filename", "file_name", "image_path"))
        case_id = _clean(case_id) or Path(image_ref).stem
        sources.append(
            _SourceRecord(
                source_index=idx,
                image_ref=image_ref,
                image_path=image_path,
                case_id=case_id,
                scanner_model=_first(row, ("scanner_model", "Scanner", "scanner", "scanner_name")) or "missing",
                lab_or_origin=_first(row, ("lab_or_origin", "origin", "Origin", "lab", "Laboratory")) or "missing",
                tumor_type=_first(row, ("tumor_type", "tumor", "Tumor", "diagnosis")) or "missing",
                species=_first(row, ("species", "Species")) or "missing",
                resolution=_first(row, ("resolution", "Resolution", "mpp", "microns_per_pixel")) or "missing",
            )
        )
    if not sources:
        raise ValueError(f"No source rows with image paths found in {config.metadata_path}")
    return sources


def _load_annotations(path: Path, *, bbox_format: str) -> dict[str, list[_AnnotationRecord]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    categories = payload.get("categories", [])
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError(f"Annotation JSON must contain images and annotations lists: {path}")

    id_to_file: dict[int, str] = {}
    for image in images:
        if isinstance(image, Mapping):
            image_id = _as_int(image.get("id"))
            filename = _clean(image.get("file_name", image.get("filename", "")))
            if image_id is not None and filename:
                id_to_file[int(image_id)] = filename

    category_names: dict[int, str] = {}
    for category in categories if isinstance(categories, list) else []:
        if isinstance(category, Mapping):
            category_id = _as_int(category.get("id"))
            if category_id is not None:
                category_names[int(category_id)] = _clean(category.get("name", ""))

    by_image: dict[str, list[_AnnotationRecord]] = {}
    for fallback_idx, ann in enumerate(annotations):
        if not isinstance(ann, Mapping):
            continue
        image_id = _as_int(ann.get("image_id"))
        filename = id_to_file.get(int(image_id)) if image_id is not None else ""
        if not filename:
            continue
        bbox = _bbox_xywh(ann.get("bbox"), bbox_format=bbox_format)
        if bbox is None:
            continue
        category_id = _as_int(ann.get("category_id"))
        label_name = category_names.get(int(category_id), "") if category_id is not None else ""
        label = _label_from_annotation(ann, category_id=category_id, category_name=label_name)
        annotation_id = _clean(ann.get("id", "")) or f"{Path(filename).stem}_{fallback_idx}"
        x, y, w, h = bbox
        record = _AnnotationRecord(
            annotation_id=annotation_id,
            image_key=Path(filename).name,
            bbox_x=x,
            bbox_y=y,
            bbox_w=w,
            bbox_h=h,
            patch_center_x=x + w / 2.0,
            patch_center_y=y + h / 2.0,
            label=label,
            label_name=label_name or ("mitotic" if label == 1 else "hard-negative/non-mitotic"),
        )
        by_image.setdefault(Path(filename).name, []).append(record)
        by_image.setdefault(Path(filename).stem, []).append(record)
    return by_image


def _assign_case_splits(sources: Sequence[_SourceRecord], *, axis: str, config: BuilderConfig) -> dict[str, str]:
    by_case: dict[str, _SourceRecord] = {}
    for source in sources:
        by_case.setdefault(source.case_id, source)

    by_domain: dict[str, list[str]] = {}
    for case_id, source in by_case.items():
        by_domain.setdefault(domain_name(_source_as_row(source), axis), []).append(case_id)

    split_map: dict[str, str] = {}
    for domain in sorted(by_domain):
        cases = sorted(by_domain[domain])
        random.Random(f"{config.split_seed}:{domain}").shuffle(cases)
        for case_id, split in _split_cases(cases, config.split_fractions).items():
            split_map[case_id] = split
    return split_map


def _collect_candidates(
    sources: Sequence[_SourceRecord],
    annotations_by_image: Mapping[str, Sequence[_AnnotationRecord]],
    case_splits: Mapping[str, str],
) -> tuple[list[_SampleRecord], list[dict[str, Any]]]:
    records: list[_SampleRecord] = []
    skipped: list[dict[str, Any]] = []
    for source in sources:
        annotations = annotations_by_image.get(Path(source.image_ref).name) or annotations_by_image.get(Path(source.image_ref).stem) or ()
        if not annotations:
            skipped.append({"case_id": source.case_id, "image_ref": source.image_ref, "reason": "no_valid_annotations"})
            continue
        split = case_splits[source.case_id]
        for ann in annotations:
            sample_id = _sample_id(source, ann)
            patch_path = Path("__pending__")
            records.append(
                _SampleRecord(
                    source=source,
                    annotation=ann,
                    split=split,
                    sample_id=sample_id,
                    patch_path=patch_path,
                    negative_match_scope="candidate",
                )
            )
    return records, skipped


def _select_matched_samples(records: Sequence[_SampleRecord], config: BuilderConfig) -> tuple[list[_SampleRecord], dict[str, Any]]:
    positives = sorted((r for r in records if r.label == 1), key=_sample_sort_key)
    negatives = sorted((r for r in records if r.label == 0), key=_sample_sort_key)

    negatives_by_case: dict[str, list[_SampleRecord]] = {}
    negatives_by_domain_split: dict[tuple[str, str], list[_SampleRecord]] = {}
    for neg in negatives:
        negatives_by_case.setdefault(neg.source.case_id, []).append(neg)
        domain = domain_name(_source_as_row(neg.source), config.preferred_axis)
        negatives_by_domain_split.setdefault((neg.split, domain), []).append(neg)

    for key in sorted(negatives_by_case):
        random.Random(f"{config.negative_seed}:case:{key}").shuffle(negatives_by_case[key])
    for key in sorted(negatives_by_domain_split):
        random.Random(f"{config.negative_seed}:domain:{key[0]}:{key[1]}").shuffle(negatives_by_domain_split[key])

    used_negative_ids: set[tuple[int, str]] = set()
    selected: list[_SampleRecord] = []
    matches_by_scope = {"same_case": 0, "same_domain_same_split": 0, "unmatched": 0}
    cross_split_violations: list[dict[str, str]] = []

    for pos in positives:
        scope = "unmatched_positive_no_negative_available"
        matched: _SampleRecord | None = _pop_unused_negative(
            negatives_by_case.get(pos.source.case_id, ()),
            used_negative_ids,
            required_split=pos.split,
        )
        if matched is not None:
            scope = "same_case"
            matches_by_scope["same_case"] += 1
        else:
            source_domain = domain_name(_source_as_row(pos.source), config.preferred_axis)
            matched = _pop_unused_negative(
                negatives_by_domain_split.get((pos.split, source_domain), ()),
                used_negative_ids,
                required_split=pos.split,
            )
            if matched is not None:
                scope = "same_domain_same_split"
                matches_by_scope["same_domain_same_split"] += 1
            else:
                matches_by_scope["unmatched"] += 1

        selected.append(_replace_scope(pos, f"positive_{scope}"))
        if matched is not None:
            used_negative_ids.add(_negative_identity(matched))
            if matched.split != pos.split:
                cross_split_violations.append(
                    {"positive_sample_id": pos.sample_id, "negative_sample_id": matched.sample_id, "positive_split": pos.split, "negative_split": matched.split}
                )
            selected.append(_replace_scope(matched, scope))

    return selected, {
        "positive_count": len(positives),
        "candidate_negative_count": len(negatives),
        "selected_negative_count": len([r for r in selected if r.label == 0]),
        "unmatched_positive_count": matches_by_scope["unmatched"],
        "matches_by_scope": matches_by_scope,
        "negative_cross_split_violations": cross_split_violations,
        "negative_sampling_seed": int(config.negative_seed),
        "negative_match_order": ["same_case", "same_pseudo_domain_and_same_split", "unmatched_positive"],
    }


def _write_patches_and_manifest_rows(
    records: Sequence[_SampleRecord],
    config: BuilderConfig,
    *,
    overwrite: bool,
) -> list[dict[str, Any]]:
    image_cache: dict[Path, Any] = {}
    rows: list[dict[str, Any]] = []
    for record in records:
        patch_path = config.patch_dir / f"{record.sample_id}.jpg"
        if overwrite or not patch_path.exists():
            if record.source.image_path not in image_cache:
                image_cache[record.source.image_path] = _read_image(record.source.image_path)
            patch = _crop_centered(
                image_cache[record.source.image_path],
                record.annotation.patch_center_x,
                record.annotation.patch_center_y,
                config.patch_size,
            )
            patch_path.parent.mkdir(parents=True, exist_ok=True)
            patch.save(patch_path, quality=int(config.image_quality))
        rows.append(
            {
                "sample_id": record.sample_id,
                "case_id": record.source.case_id,
                "image_path": repo_relative(patch_path, config.repo_root),
                "annotation_id": record.annotation.annotation_id,
                "bbox_x": _format_float(record.annotation.bbox_x),
                "bbox_y": _format_float(record.annotation.bbox_y),
                "bbox_w": _format_float(record.annotation.bbox_w),
                "bbox_h": _format_float(record.annotation.bbox_h),
                "patch_center_x": _format_float(record.annotation.patch_center_x),
                "patch_center_y": _format_float(record.annotation.patch_center_y),
                "label": int(record.label),
                "label_name": record.annotation.label_name,
                "scanner_model": record.source.scanner_model,
                "lab_or_origin": record.source.lab_or_origin,
                "tumor_type": record.source.tumor_type,
                "species": record.source.species,
                "resolution": record.source.resolution,
                "domain_axis": config.preferred_axis,
                "domain_name": domain_name(_source_as_row(record.source), config.preferred_axis),
                "domain_id": "",
                "center": "",
                "magnification": "",
                "split": record.split,
                "negative_match_scope": record.negative_match_scope,
            }
        )
    return rows


def _build_domain_mapping(rows: Sequence[Mapping[str, Any]], config: BuilderConfig) -> dict[str, Any]:
    domain_names = sorted({str(row["domain_name"]) for row in rows})
    domains = []
    for domain_id, name in enumerate(domain_names):
        domain_rows = [row for row in rows if str(row["domain_name"]) == name]
        domains.append(
            {
                "domain_id": str(domain_id),
                "domain_name": name,
                "n_rows": len(domain_rows),
                "n_cases": len({str(row["case_id"]) for row in domain_rows}),
            }
        )
    return {
        "schema_version": "midogpp_domain_mapping_v1",
        "domain_axis": config.preferred_axis,
        "domains": domains,
        "domain_name_to_id": {row["domain_name"]: row["domain_id"] for row in domains},
    }


def _apply_final_domain_ids(rows: Sequence[Mapping[str, Any]], mapping: Mapping[str, Any]) -> list[dict[str, Any]]:
    name_to_id = mapping["domain_name_to_id"]
    out: list[dict[str, Any]] = []
    for row in rows:
        domain_id = str(name_to_id[str(row["domain_name"])])
        updated = dict(row)
        updated["domain_id"] = domain_id
        updated["center"] = domain_id
        updated["magnification"] = domain_id
        out.append(updated)
    return out


def _domain_feasibility_rows(rows: Sequence[Mapping[str, Any]], config: BuilderConfig) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for axis in config.candidate_axes:
        names = sorted({domain_name(row, axis) for row in rows})
        for domain_id, name in enumerate(names):
            domain_rows = [row for row in rows if domain_name(row, axis) == name]
            train_rows = [row for row in domain_rows if str(row["split"]) == "train"]
            eval_rows = [row for row in domain_rows if str(row["split"]) == "test"]
            val_rows = [row for row in domain_rows if str(row["split"]) == "val"]
            stats = {
                "total_cases": len({str(row["case_id"]) for row in domain_rows}),
                "train_cases": len({str(row["case_id"]) for row in train_rows}),
                "val_cases": len({str(row["case_id"]) for row in val_rows}),
                "eval_cases": len({str(row["case_id"]) for row in eval_rows}),
                "train_positives": _label_count(train_rows, 1),
                "train_negatives": _label_count(train_rows, 0),
                "eval_positives": _label_count(eval_rows, 1),
                "eval_negatives": _label_count(eval_rows, 0),
                "total_rows": len(domain_rows),
            }
            reasons = _ineligible_reasons(stats, config.eligibility)
            out.append(
                {
                    "domain_axis": axis,
                    "domain_name": name,
                    "domain_id_for_axis": str(domain_id),
                    **stats,
                    "eligible": not reasons,
                    "ineligible_reasons": ";".join(reasons),
                }
            )
    return out


def _class_balance_rows(rows: Sequence[Mapping[str, Any]], config: BuilderConfig) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for axis in config.candidate_axes:
        names = sorted({domain_name(row, axis) for row in rows})
        for name in names:
            for split in ("train", "val", "test"):
                split_rows = [row for row in rows if domain_name(row, axis) == name and str(row["split"]) == split]
                out.append(
                    {
                        "domain_axis": axis,
                        "domain_name": name,
                        "split": split,
                        "case_count": len({str(row["case_id"]) for row in split_rows}),
                        "positive_count": _label_count(split_rows, 1),
                        "negative_count": _label_count(split_rows, 0),
                        "total_rows": len(split_rows),
                    }
                )
    return out


def _split_manifest_rows(sources: Sequence[_SourceRecord], case_splits: Mapping[str, str], config: BuilderConfig) -> list[dict[str, Any]]:
    by_case: dict[str, _SourceRecord] = {}
    for source in sources:
        by_case.setdefault(source.case_id, source)
    rows = []
    for case_id in sorted(by_case):
        source = by_case[case_id]
        rows.append(
            {
                "case_id": case_id,
                "split": case_splits[case_id],
                "scanner_model": source.scanner_model,
                "lab_or_origin": source.lab_or_origin,
                "tumor_type": source.tumor_type,
                "species": source.species,
                "resolution": source.resolution,
                "preferred_domain_axis": config.preferred_axis,
                "preferred_domain_name": domain_name(_source_as_row(source), config.preferred_axis),
            }
        )
    return rows


def _leakage_report(rows: Sequence[Mapping[str, Any]], matching_report: Mapping[str, Any]) -> dict[str, Any]:
    by_split = {
        "train": {str(row["case_id"]) for row in rows if str(row["split"]) == "train"},
        "val": {str(row["case_id"]) for row in rows if str(row["split"]) == "val"},
        "test": {str(row["case_id"]) for row in rows if str(row["split"]) == "test"},
    }
    overlaps = {
        "train_val": sorted(by_split["train"].intersection(by_split["val"])),
        "train_test": sorted(by_split["train"].intersection(by_split["test"])),
        "val_test": sorted(by_split["val"].intersection(by_split["test"])),
    }
    sample_ids = [str(row["sample_id"]) for row in rows]
    duplicates = sorted({sid for sid in sample_ids if sample_ids.count(sid) > 1})
    cross_split = list(matching_report.get("negative_cross_split_violations", []))
    status = "PASS" if not any(overlaps.values()) and not duplicates and not cross_split else "FAIL"
    return {
        "schema_version": "midogpp_leakage_report_v1",
        "status": status,
        "case_overlap": overlaps,
        "duplicate_sample_ids": duplicates,
        "negative_cross_split_violations": cross_split,
        "unmatched_positive_count": int(matching_report.get("unmatched_positive_count", 0)),
        "split_counts": {split: len([row for row in rows if str(row["split"]) == split]) for split in ("train", "val", "test")},
    }


def _dataset_contract(
    *,
    config: BuilderConfig,
    manifest_rows: Sequence[Mapping[str, Any]],
    domain_mapping: Mapping[str, Any],
    matching_report: Mapping[str, Any],
    skipped_annotations: Sequence[Mapping[str, Any]],
    leakage_report: Mapping[str, Any],
    eligible_domain_count: int,
    final_axis_frozen: bool,
    status: str,
) -> dict[str, Any]:
    class_counts = {
        "mitotic": _label_count(manifest_rows, 1),
        "hard_negative_or_non_mitotic": _label_count(manifest_rows, 0),
    }
    return {
        "schema_version": "midogpp_annotation_patch_dataset_contract_v1",
        "artifact_name": config.artifact_name,
        "status": status,
        "sample_definition": "annotation-centered patch",
        "group_definition": "case_id",
        "class_definition": "mitotic vs hard-negative/non-mitotic",
        "paths": {
            "artifact_root": repo_relative(config.artifact_root, config.repo_root),
            "manifest": repo_relative(config.artifact_root / "manifest.csv", config.repo_root),
            "split_manifest": repo_relative(config.artifact_root / "split_manifest.csv", config.repo_root),
            "patch_dir": repo_relative(config.patch_dir, config.repo_root),
            "raw_root": _path_for_contract(config.input_root, config.repo_root),
            "metadata": _path_for_contract(config.metadata_path, config.repo_root),
            "annotations": _path_for_contract(config.annotations_path, config.repo_root),
        },
        "hashes": {
            "metadata_sha256": sha256_file(config.metadata_path),
            "annotation_sha256": sha256_file(config.annotations_path),
        },
        "extraction_policy": {
            "patch_size": int(config.patch_size),
            "bbox_format": config.bbox_format,
            "positive_policy": config.positive_policy,
            "sample": "annotation-centered patch",
            "group": "case_id",
        },
        "negative_sampling_policy": {
            "negative_policy": config.negative_policy,
            "negative_ratio": float(config.negative_ratio),
            "negative_sampling_seed": int(config.negative_seed),
            "negative_match_order": matching_report["negative_match_order"],
            "selected_negative_count": int(matching_report["selected_negative_count"]),
            "candidate_negative_count": int(matching_report["candidate_negative_count"]),
            "unmatched_positive_count": int(matching_report["unmatched_positive_count"]),
            "matches_by_scope": matching_report["matches_by_scope"],
        },
        "split_policy": {
            "split_seed": int(config.split_seed),
            "fractions": dict(config.split_fractions),
            "group": "case_id",
            "case_disjoint": bool(leakage_report["status"] == "PASS"),
            "split_order": "assign case-disjoint train/val/test before negative sampling",
        },
        "domain_policy": {
            "candidate_axes": list(config.candidate_axes),
            "selected_domain_axis": config.preferred_axis,
            "final_axis_policy": config.final_axis_policy,
            "min_eligible_domains": int(config.min_eligible_domains),
            "eligible_domain_count": int(eligible_domain_count),
            "final_axis_frozen": bool(final_axis_frozen),
            "domain_mapping": domain_mapping["domain_name_to_id"],
        },
        "eligibility_thresholds": config.eligibility.as_dict(),
        "row_counts": {
            "manifest_rows": len(manifest_rows),
            "positive_rows": class_counts["mitotic"],
            "negative_rows": class_counts["hard_negative_or_non_mitotic"],
            "skipped_source_images": len(skipped_annotations),
        },
        "class_counts": class_counts,
    }


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {path}")
        return [dict(row) for row in reader]


def _resolve_image_path(config: BuilderConfig, image_ref: str) -> Path:
    raw = Path(image_ref)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                config.input_root / raw,
                config.input_root / raw.name,
                config.input_root / "images" / raw.name,
                config.metadata_path.parent / raw,
                config.repo_root / raw,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve MIDOG++ source image '{image_ref}' under {config.input_root}")


def _read_image(path: Path) -> Any:
    import numpy as np

    if path.suffix.lower() in {".tif", ".tiff"}:
        try:
            import tifffile  # type: ignore

            return _normalize_image_array(tifffile.imread(path))
        except Exception:
            pass
    with Image.open(path) as image:
        return _normalize_image_array(np.asarray(image.convert("RGB")))


def _normalize_image_array(arr: Any) -> Any:
    import numpy as np

    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3:
        if arr.shape[0] in {3, 4} and arr.shape[-1] not in {3, 4}:
            arr = np.moveaxis(arr, 0, -1)
        arr = arr[..., :3]
    else:
        raise ValueError(f"Unsupported image array shape: {arr.shape}")

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32, copy=False)
        finite = np.isfinite(arr)
        if not finite.any():
            arr = np.zeros(arr.shape, dtype=np.uint8)
        else:
            min_v = float(arr[finite].min())
            max_v = float(arr[finite].max())
            if max_v <= 1.0 and min_v >= 0.0:
                arr = arr * 255.0
            elif max_v > 255.0 or min_v < 0.0:
                arr = 255.0 * (arr - min_v) / max(max_v - min_v, 1.0e-6)
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _crop_centered(arr: Any, center_x: float, center_y: float, patch_size: int) -> Image.Image:
    import numpy as np

    h, w = int(arr.shape[0]), int(arr.shape[1])
    size = int(patch_size)
    half = size // 2
    cx = int(round(float(center_x)))
    cy = int(round(float(center_y)))
    x0 = cx - half
    y0 = cy - half
    x1 = x0 + size
    y1 = y0 + size
    src_x0 = max(x0, 0)
    src_y0 = max(y0, 0)
    src_x1 = min(x1, w)
    src_y1 = min(y1, h)
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    if src_x1 > src_x0 and src_y1 > src_y0:
        dst_x0 = src_x0 - x0
        dst_y0 = src_y0 - y0
        canvas[dst_y0 : dst_y0 + (src_y1 - src_y0), dst_x0 : dst_x0 + (src_x1 - src_x0)] = arr[
            src_y0:src_y1, src_x0:src_x1
        ]
    return Image.fromarray(canvas, mode="RGB")


def _split_cases(cases: Sequence[str], fractions: Mapping[str, float]) -> dict[str, str]:
    n = len(cases)
    train_fraction = float(fractions.get("train", 0.45))
    val_fraction = float(fractions.get("val", 0.10))
    test_fraction = float(fractions.get("test", max(0.0, 1.0 - train_fraction - val_fraction)))
    if n == 0:
        return {}
    if n == 1:
        if train_fraction > 0:
            return {cases[0]: "train"}
        if val_fraction > 0:
            return {cases[0]: "val"}
        return {cases[0]: "test"}
    if n == 2:
        if val_fraction <= 0 and test_fraction <= 0:
            return {cases[0]: "train", cases[1]: "train"}
        if test_fraction <= 0:
            return {cases[0]: "train", cases[1]: "val"}
        return {cases[0]: "train", cases[1]: "test"}
    if train_fraction <= 0:
        train_n = 0
    else:
        train_n = max(1, int(round(n * train_fraction)))
    if val_fraction <= 0:
        val_n = 0
    else:
        val_n = max(1, int(round(n * val_fraction)))
    if test_fraction <= 0:
        test_n = 0
    else:
        test_n = max(1, n - train_n - val_n)

    while train_n + val_n + test_n > n:
        if train_n >= val_n and train_n > 1:
            train_n -= 1
        elif val_n > 0:
            val_n -= 1
        else:
            test_n -= 1
    while train_n + val_n + test_n < n:
        test_n += 1

    split_map: dict[str, str] = {}
    for case in cases[:train_n]:
        split_map[case] = "train"
    for case in cases[train_n : train_n + val_n]:
        split_map[case] = "val"
    for case in cases[train_n + val_n : train_n + val_n + test_n]:
        split_map[case] = "test"
    return split_map


def _bbox_xywh(raw_bbox: Any, *, bbox_format: str) -> tuple[float, float, float, float] | None:
    if not isinstance(raw_bbox, Sequence) or isinstance(raw_bbox, (str, bytes)) or len(raw_bbox) < 4:
        return None
    try:
        a, b, c, d = [float(value) for value in list(raw_bbox)[:4]]
    except Exception:
        return None
    if str(bbox_format).lower() in {"xyxy", "pascal_voc"}:
        x, y, w, h = a, b, c - a, d - b
    else:
        x, y, w, h = a, b, c, d
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def _label_from_annotation(ann: Mapping[str, Any], *, category_id: int | None, category_name: str) -> int:
    if "label" in ann:
        parsed = _as_int(ann.get("label"))
        if parsed in {0, 1}:
            return int(parsed)
    text = _clean(category_name).lower()
    if "hard" in text and "negative" in text:
        return 0
    if "non-mitotic" in text or "non mitotic" in text or "not mitotic" in text or text.startswith("not "):
        return 0
    if "negative" in text:
        return 0
    if "mitotic" in text or "mitosis" in text:
        return 1
    return 1 if int(category_id or 0) == 1 else 0


def _ineligible_reasons(stats: Mapping[str, int], thresholds: EligibilityThresholds) -> list[str]:
    checks = (
        ("total_cases", thresholds.total_cases_min),
        ("train_cases", thresholds.train_cases_min),
        ("eval_cases", thresholds.eval_cases_min),
        ("train_positives", thresholds.train_positives_min),
        ("train_negatives", thresholds.train_negatives_min),
        ("eval_positives", thresholds.eval_positives_min),
        ("eval_negatives", thresholds.eval_negatives_min),
    )
    return [f"{field}<{minimum}" for field, minimum in checks if int(stats[field]) < int(minimum)]


def _dedupe_sample_ids(records: Sequence[_SampleRecord]) -> list[_SampleRecord]:
    counts: dict[str, int] = {}
    out: list[_SampleRecord] = []
    for record in records:
        base = record.sample_id
        count = counts.get(base, 0)
        counts[base] = count + 1
        sample_id = base if count == 0 else f"{base}__dup{count:02d}"
        out.append(
            _SampleRecord(
                source=record.source,
                annotation=record.annotation,
                split=record.split,
                sample_id=sample_id,
                patch_path=record.patch_path,
                negative_match_scope=record.negative_match_scope,
            )
        )
    return out


def _pop_unused_negative(
    candidates: Sequence[_SampleRecord],
    used_negative_ids: set[tuple[int, str]],
    *,
    required_split: str,
) -> _SampleRecord | None:
    for candidate in candidates:
        identity = _negative_identity(candidate)
        if identity not in used_negative_ids and candidate.split == required_split:
            return candidate
    return None


def _negative_identity(record: _SampleRecord) -> tuple[int, str]:
    return (record.source.source_index, record.annotation.annotation_id)


def _replace_scope(record: _SampleRecord, scope: str) -> _SampleRecord:
    return _SampleRecord(
        source=record.source,
        annotation=record.annotation,
        split=record.split,
        sample_id=record.sample_id,
        patch_path=record.patch_path,
        negative_match_scope=scope,
    )


def _sample_id(source: _SourceRecord, ann: _AnnotationRecord) -> str:
    return f"{_safe_stem(source.case_id)}__{_safe_stem(Path(source.image_ref).stem)}__ann{_safe_stem(ann.annotation_id)}__y{ann.label}"


def _sample_sort_key(record: _SampleRecord) -> tuple[str, str, str, str]:
    return (record.split, domain_name(_source_as_row(record.source), "tumor_type|lab_or_origin|scanner_model"), record.source.case_id, record.annotation.annotation_id)


def _label_count(rows: Sequence[Mapping[str, Any]], label: int) -> int:
    return sum(1 for row in rows if int(float(str(row.get("label", -1)))) == int(label))


def _source_as_row(source: _SourceRecord) -> dict[str, str]:
    return {
        "case_id": source.case_id,
        "scanner_model": source.scanner_model,
        "lab_or_origin": source.lab_or_origin,
        "tumor_type": source.tumor_type,
        "species": source.species,
        "resolution": source.resolution,
    }


def _path_for_contract(path: Path, repo_root: Path) -> str:
    try:
        return repo_relative(path, repo_root)
    except ValueError:
        return str(path)


def _format_float(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def _safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(str(value)).stem).strip("_") or "sample"


def _clean(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _first(row: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = _clean(row.get(key, ""))
        if value:
            return value
    return ""


def _as_int(value: Any) -> int | None:
    try:
        return int(float(str(value)))
    except Exception:
        return None


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _config_path(repo_root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else repo_root / path


def _find_repo_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for path in (current, *current.parents):
        if (path / ".git").exists():
            return path
    return Path.cwd().resolve()
