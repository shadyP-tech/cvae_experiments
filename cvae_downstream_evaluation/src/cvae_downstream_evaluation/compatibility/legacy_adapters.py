"""Adapters from legacy downstream artifacts to learned-utility pipeline inputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

from .pipeline import LearnedUtilityPipelineInputs, run_learned_utility_pipeline
from ..downstream import CandidateDownstreamRow, write_candidate_downstream_matrix
from ..features.feature_table_builder import write_allowed_feature_table
from ..protocol import ProtocolError
from ..schemas import SELECTION_ELIGIBLE


def normalize_c52_legacy_artifacts(
    *,
    router_training_examples: Path,
    downstream_matrix: Path,
    target_domain: str,
    out_dir: Path,
    support_size: int | None = None,
    support_seed: int | None = None,
) -> dict[str, Path]:
    """Build protocol-normalized CSVs for one held-out target.

    Source-inner estimator training rows exclude `target_domain`; deployable
    feature/candidate rows include only that target. The downstream matrix is
    diagnostic-only and is used only after selections are materialized.
    """

    examples = _read_csv(router_training_examples)
    if not examples:
        raise ProtocolError(f"No router training examples found: {router_training_examples}")
    matrix_rows = _read_csv(downstream_matrix)
    if not matrix_rows:
        raise ProtocolError(f"No downstream matrix rows found: {downstream_matrix}")

    target = str(target_domain)
    available = _available_downstream_keys(
        matrix_rows,
        target_domain=target,
        support_size=support_size,
        support_seed=support_seed,
    )
    candidate_rows = _candidate_rows_from_examples(
        examples,
        target_domain=target,
        support_size=support_size,
        support_seed=support_seed,
        available_downstream_keys=available,
    )
    support_rows = _support_rows_from_examples(
        examples,
        target_domain=target,
        support_size=support_size,
        support_seed=support_seed,
        available_downstream_keys=available,
    )
    source_inner_feature_rows = _source_inner_feature_rows_from_examples(
        examples,
        target_domain=target,
        support_size=support_size,
        support_seed=support_seed,
        available_downstream_keys=available,
    )
    source_inner_training_rows = _source_inner_training_from_examples(examples, target_domain=target)
    diagnostic_rows = _diagnostic_rows_from_matrix(
        matrix_rows,
        target_domain=target,
        support_size=support_size,
        support_seed=support_seed,
    )

    paths = {
        "candidates": out_dir / "inputs" / "candidate_manifest.csv",
        "support_features": out_dir / "inputs" / "support_features.csv",
        "source_inner_features": out_dir / "inputs" / "source_inner_features.csv",
        "source_inner_training": out_dir / "inputs" / "source_inner_training.csv",
        "diagnostic_matrix": out_dir / "tables" / "diagnostic_downstream_utility.csv",
    }
    _write_csv(paths["candidates"], candidate_rows)
    _write_csv(paths["support_features"], support_rows)
    _write_csv(paths["source_inner_features"], source_inner_feature_rows)
    _write_csv(paths["source_inner_training"], source_inner_training_rows)
    write_candidate_downstream_matrix(paths["diagnostic_matrix"], diagnostic_rows)
    return paths


def discover_c52_contexts(
    *,
    router_training_examples: Path,
    downstream_matrix: Path,
) -> tuple[tuple[str, int, int], ...]:
    examples = _read_csv(router_training_examples)
    matrix_rows = _read_csv(downstream_matrix)
    example_contexts = {
        (str(row.get("heldout_center")), int(row.get("support_size")), int(row.get("support_seed")))
        for row in examples
        if str(row.get("heldout_center", "")).strip()
    }
    matrix_contexts = {
        (str(row.get("heldout_center")), int(row.get("support_size")), int(row.get("support_seed")))
        for row in matrix_rows
        if str(row.get("heldout_center", "")).strip()
        and "single_expert" in str(row.get("row_type", ""))
        and str(row.get("status", "ok")) == "ok"
    }
    return tuple(sorted(example_contexts.intersection(matrix_contexts), key=lambda item: (int(item[0]), item[1], item[2])))


def run_c52_legacy_batch(
    *,
    router_training_examples: Path,
    downstream_matrix: Path,
    out_dir: Path,
    feature_columns: Sequence[str],
    target_domains: Sequence[str] | None = None,
    support_sizes: Sequence[int] | None = None,
    support_seeds: Sequence[int] | None = None,
    ridge_lambda: float = 1e-3,
) -> dict[str, Path]:
    contexts = discover_c52_contexts(
        router_training_examples=router_training_examples,
        downstream_matrix=downstream_matrix,
    )
    contexts = tuple(
        context
        for context in contexts
        if (target_domains is None or context[0] in {str(v) for v in target_domains})
        and (support_sizes is None or context[1] in {int(v) for v in support_sizes})
        and (support_seeds is None or context[2] in {int(v) for v in support_seeds})
    )
    if not contexts:
        raise ProtocolError("No C5.2 legacy contexts matched the requested filters.")

    alignment_paths: list[Path] = []
    baseline_alignment_paths: list[Path] = []
    leakage_paths: list[Path] = []
    manifest_rows: list[dict[str, object]] = []
    for target, support_size, support_seed in contexts:
        context_dir = Path(out_dir) / f"target{target}" / f"support{support_size}" / f"seed{support_seed}"
        normalized = normalize_c52_legacy_artifacts(
            router_training_examples=router_training_examples,
            downstream_matrix=downstream_matrix,
            target_domain=target,
            support_size=support_size,
            support_seed=support_seed,
            out_dir=context_dir,
        )
        outputs = run_learned_utility_pipeline(
            LearnedUtilityPipelineInputs(
                candidates=normalized["candidates"],
                source_inner_training=normalized["source_inner_training"],
                diagnostic_matrix=normalized["diagnostic_matrix"],
                out_dir=context_dir,
                feature_columns=tuple(feature_columns),
                support_features=normalized["support_features"],
                source_inner_features=normalized["source_inner_features"],
                ridge_lambda=float(ridge_lambda),
            )
        )
        alignment_paths.append(outputs.alignment)
        baseline_alignment_paths.append(outputs.baseline_alignment)
        leakage_paths.append(outputs.leakage_report)
        manifest_rows.append(
            {
                "target_domain": target,
                "support_size": support_size,
                "support_seed": support_seed,
                "context_dir": str(context_dir),
                "alignment": str(outputs.alignment),
                "baseline_alignment": str(outputs.baseline_alignment),
                "leakage_report": str(outputs.leakage_report),
            }
        )

    summary_dir = Path(out_dir) / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    alignment_summary = summary_dir / "learned_utility_alignment_all_contexts.csv"
    baseline_summary = summary_dir / "baseline_alignment_all_contexts.csv"
    leakage_summary = summary_dir / "leakage_summary.csv"
    manifest_path = summary_dir / "legacy_batch_manifest.json"
    _concat_csv(alignment_summary, alignment_paths)
    _concat_csv(baseline_summary, baseline_alignment_paths)
    _write_csv(leakage_summary, _leakage_summary_rows(leakage_paths))
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "c52_legacy_batch_manifest_v1",
                "router_training_examples": str(router_training_examples),
                "downstream_matrix": str(downstream_matrix),
                "feature_columns": list(feature_columns),
                "contexts": manifest_rows,
                "alignment_summary": str(alignment_summary),
                "baseline_summary": str(baseline_summary),
                "leakage_summary": str(leakage_summary),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "alignment_summary": alignment_summary,
        "baseline_summary": baseline_summary,
        "leakage_summary": leakage_summary,
        "manifest": manifest_path,
    }


def _candidate_rows_from_examples(
    rows: Sequence[Mapping[str, object]],
    *,
    target_domain: str,
    support_size: int | None,
    support_seed: int | None,
    available_downstream_keys: set[tuple[str, str]],
) -> list[dict[str, object]]:
    out: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        if str(row.get("heldout_center")) != str(target_domain):
            continue
        if not _matches_support_filter(row, support_size=support_size, support_seed=support_seed):
            continue
        if _example_downstream_key(row) not in available_downstream_keys:
            continue
        if str(row.get("primary_candidate_eligible", "1")) not in {"1", "true", "True"}:
            continue
        lineage = _lineage(row)
        source_domain = str(row.get("candidate_expert", ""))
        if source_domain == str(target_domain):
            raise ProtocolError(f"Legacy candidate row leaks target expert {target_domain}: {row}")
        candidate = lineage | {
            "source_domain": source_domain,
            "mode_label": row.get("mode_label", ""),
            "generator_family": row.get("generator_family", ""),
        }
        out[_dedupe_key(candidate)] = candidate
    if not out:
        raise ProtocolError(f"No candidate rows for target_domain={target_domain}")
    return list(out.values())


def _support_rows_from_examples(
    rows: Sequence[Mapping[str, object]],
    *,
    target_domain: str,
    support_size: int | None,
    support_seed: int | None,
    available_downstream_keys: set[tuple[str, str]],
) -> list[dict[str, object]]:
    out: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        if str(row.get("heldout_center")) != str(target_domain):
            continue
        if not _matches_support_filter(row, support_size=support_size, support_seed=support_seed):
            continue
        if _example_downstream_key(row) not in available_downstream_keys:
            continue
        support = _lineage(row) | {
            "support_nelbo": row.get("support_nelbo_mean", ""),
            "support_nelbo_rank": row.get("support_nelbo_rank_within_unit", ""),
            "support_nelbo_z": row.get("support_nelbo_z_within_unit", ""),
            "metadata_match": row.get("metadata_match", ""),
        }
        out[_dedupe_key(support)] = support
    return list(out.values())


def _source_inner_feature_rows_from_examples(
    rows: Sequence[Mapping[str, object]],
    *,
    target_domain: str,
    support_size: int | None,
    support_seed: int | None,
    available_downstream_keys: set[tuple[str, str]],
) -> list[dict[str, object]]:
    out: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        if str(row.get("heldout_center")) != str(target_domain):
            continue
        if not _matches_support_filter(row, support_size=support_size, support_seed=support_seed):
            continue
        if _example_downstream_key(row) not in available_downstream_keys:
            continue
        feature = _lineage(row) | {
            "source_inner_stability": row.get("utility_label_bacc_std", "0"),
            "source_inner_candidate_stability": row.get("utility_label_ge_080_rate", "0"),
        }
        out[_dedupe_key(feature)] = feature
    return list(out.values())


def _source_inner_training_from_examples(rows: Sequence[Mapping[str, object]], *, target_domain: str) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        if str(row.get("heldout_center")) == str(target_domain):
            continue
        if str(row.get("primary_candidate_eligible", "1")) not in {"1", "true", "True"}:
            continue
        out.append(
            {
                "fold_role": "source_inner_pseudo_target",
                "support_nelbo": row.get("support_nelbo_mean", ""),
                "support_nelbo_rank": row.get("support_nelbo_rank_within_unit", ""),
                "support_nelbo_z": row.get("support_nelbo_z_within_unit", ""),
                "metadata_match": row.get("metadata_match", ""),
                "source_inner_stability": row.get("utility_label_bacc_std", "0"),
                "source_inner_candidate_stability": row.get("utility_label_ge_080_rate", "0"),
                "source_inner_heldout_bacc": row.get("utility_label_bacc", ""),
            }
        )
    if not out:
        raise ProtocolError(f"No source-inner training rows after excluding target_domain={target_domain}")
    return out


def _diagnostic_rows_from_matrix(
    rows: Sequence[Mapping[str, object]],
    *,
    target_domain: str,
    support_size: int | None,
    support_seed: int | None,
) -> list[CandidateDownstreamRow]:
    out: list[CandidateDownstreamRow] = []
    for row in rows:
        if str(row.get("heldout_center")) != str(target_domain):
            continue
        if not _matches_support_filter(row, support_size=support_size, support_seed=support_seed):
            continue
        if str(row.get("status", "ok")) != "ok":
            continue
        if "single_expert" not in str(row.get("row_type", "")):
            continue
        out.append(
            CandidateDownstreamRow(
                experiment_seed=int(row.get("experiment_seed", 0)),
                heldout_center=str(row.get("heldout_center", "")),
                candidate_expert=str(row.get("candidate_expert", "")),
                generation_mode=_legacy_generation_id(row),
                budget_per_class=int(row.get("budget_per_class", 0)),
                generation_seed=int(row.get("generation_seed", 0)),
                classifier_seed=int(row.get("classifier_seed", 0)),
                bacc=float(row.get("bacc", "nan")),
                macro_f1=float(row.get("macro_f1", "nan")),
                auroc=_float_or_nan(row.get("auroc", "")),
                auprc=_float_or_nan(row.get("auprc", "")),
                row_type="single_expert",
                n_synthetic_train=int(row.get("n_synthetic_train") or row.get("n_train") or 0),
                n_target_eval=int(row.get("n_target_eval", 0)),
                target_eval_pool_id=str(row.get("target_eval_pool_id", "")),
                status="ok",
            )
        )
    if not out:
        raise ProtocolError(f"No diagnostic downstream rows for target_domain={target_domain}")
    return out


def _available_downstream_keys(
    rows: Sequence[Mapping[str, object]],
    *,
    target_domain: str,
    support_size: int | None,
    support_seed: int | None,
) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for row in rows:
        if str(row.get("heldout_center")) != str(target_domain):
            continue
        if str(row.get("status", "ok")) != "ok":
            continue
        if "single_expert" not in str(row.get("row_type", "")):
            continue
        if not _matches_support_filter(row, support_size=support_size, support_seed=support_seed):
            continue
        keys.add((str(row.get("candidate_expert", "")), _legacy_generation_id(row)))
    if not keys:
        raise ProtocolError(f"No available downstream candidate keys for target_domain={target_domain}")
    return keys


def _example_downstream_key(row: Mapping[str, object]) -> tuple[str, str]:
    return str(row.get("candidate_expert", "")), _legacy_generation_id(row)


def _lineage(row: Mapping[str, object]) -> dict[str, object]:
    candidate = str(row.get("candidate_expert", ""))
    return {
        "fold_id": f"target{row.get('heldout_center')}",
        "experiment_seed": int(row.get("experiment_seed", 0)),
        "target_domain": str(row.get("heldout_center", "")),
        "support_split_id": str(row.get("support_eval_split_id", "")),
        "eval_split_id": str(row.get("support_eval_split_id", "")) + "_eval",
        "candidate_id": _candidate_id(row),
        "expert_checkpoint_id": candidate,
        "expert_checkpoint_hash": f"legacy_expert_{candidate}",
        "generation_mode": _legacy_generation_id(row),
        "generation_seed": 17,
        "classifier_seed": 17,
        "config_hash": "legacy_c52_adapter",
        "protocol_hash": "target_excluded_source_inner_adapter_v1",
        "eligibility": SELECTION_ELIGIBLE,
    }


def _candidate_id(row: Mapping[str, object]) -> str:
    return "|".join(
        [
            f"seed={row.get('experiment_seed')}",
            f"target={row.get('heldout_center')}",
            f"support={row.get('support_eval_split_id')}",
            f"expert={row.get('candidate_expert')}",
            f"family={row.get('generator_family')}",
            f"mode={row.get('generation_mode')}",
        ]
    )


def _legacy_generation_id(row: Mapping[str, object]) -> str:
    family = str(row.get("generator_family", "")).strip()
    mode = str(row.get("generation_mode", "")).strip()
    return f"{family}::{mode}" if family else mode


def _matches_support_filter(
    row: Mapping[str, object],
    *,
    support_size: int | None,
    support_seed: int | None,
) -> bool:
    if support_size is not None and int(row.get("support_size", -1)) != int(support_size):
        return False
    if support_seed is not None and int(row.get("support_seed", -1)) != int(support_seed):
        return False
    return True


def _dedupe_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        row["fold_id"],
        row["experiment_seed"],
        row["target_domain"],
        row["support_split_id"],
        row["candidate_id"],
    )


def _read_csv(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _concat_csv(path: Path, sources: Sequence[Path]) -> None:
    rows: list[dict[str, object]] = []
    columns: list[str] = []
    for source in sources:
        for row in _read_csv(source):
            rows.append(row)
            for key in row:
                if key not in columns:
                    columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _leakage_summary_rows(paths: Sequence[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = {"leakage_report": str(path)}
        row.update(payload)
        rows.append(row)
    return rows


def _float_or_nan(value: object) -> float:
    text = str(value or "").strip()
    return float(text) if text else float("nan")
