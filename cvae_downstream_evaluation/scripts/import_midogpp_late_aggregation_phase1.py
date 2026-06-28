"""Import MIDOG++ phase-1 artifacts from upstream late-aggregation scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvae_downstream_evaluation.adapters.midogpp import write_midogpp_phase1_artifacts  # noqa: E402
from cvae_downstream_evaluation.artifacts import FrozenProtocolSnapshot, stable_hash, write_frozen_snapshot  # noqa: E402
from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402
from cvae_downstream_evaluation.schemas import DIAGNOSTIC_ONLY, SELECTION_ELIGIBLE  # noqa: E402
from cvae_downstream_evaluation.schemas.midogpp import (  # noqa: E402
    MIDOGPP_ELIGIBLE_CENTERS,
    MIDOGPP_METHOD_BASELINE_ROW_TYPE,
    MIDOGPP_SINGLE_SOURCE_ROW_TYPE,
    NO_SUPPORT_SEED,
    NO_SUPPORT_SET_ID,
    MidogppDownstreamRow,
    assert_midogpp_candidate_pool,
    assert_midogpp_frozen_config_file,
)


DEFAULT_IMPORT_SCHEMA_VERSION = "midogpp_late_aggregation_import_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Translate scored MIDOG++ upstream late-aggregation rows into the "
            "canonical diagnostic phase-1 artifact bundle."
        )
    )
    parser.add_argument(
        "--late-aggregation-matrix",
        required=True,
        help="Upstream late_aggregation_matrix.csv containing pooling_rule=single_source rows.",
    )
    parser.add_argument(
        "--dense-matrix",
        default="",
        help="Optional dense_late_all_sources_downstream_matrix.csv for method baseline rows.",
    )
    parser.add_argument("--out-dir", required=True, help="Output artifact root.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "experiments" / "utility_matrix" / "virchow2_midogpp_all_candidates.yaml"),
        help="Frozen MIDOG++ config to validate before writing artifacts.",
    )
    parser.add_argument("--experiment-seed", type=int, default=42)
    parser.add_argument("--heldout-centers", default=",".join(MIDOGPP_ELIGIBLE_CENTERS))
    parser.add_argument(
        "--synthetic-per-class-total",
        type=int,
        default=128,
        help="Total phase-1 synthetic budget context. Upstream single-source component rows keep their per-source budget in the candidate manifest.",
    )
    parser.add_argument(
        "--classifier-seed",
        type=int,
        default=-1,
        help="Integer sentinel for upstream classifier_seed=null.",
    )
    parser.add_argument("--eval-set-id-prefix", default="midogpp_test_center")
    parser.add_argument(
        "--prior-method",
        action="append",
        default=[],
        help="Restrict import to one or more prior methods. By default all single-source prior methods are imported.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = Path(args.config)
    late_matrix = Path(args.late_aggregation_matrix)
    dense_matrix = Path(args.dense_matrix) if args.dense_matrix else None
    out_dir = Path(args.out_dir)
    heldouts = _parse_centers(args.heldout_centers)
    methods = tuple(str(method) for method in args.prior_method)

    assert_midogpp_frozen_config_file(config)
    late_rows = _read_csv(late_matrix)
    dense_rows = _read_csv(dense_matrix) if dense_matrix else []
    import_hashes = _build_import_hashes(
        config=config,
        late_matrix=late_matrix,
        dense_matrix=dense_matrix,
        heldout_centers=heldouts,
        prior_methods=methods,
        synthetic_per_class_total=args.synthetic_per_class_total,
        classifier_seed=args.classifier_seed,
    )

    diagnostic_rows = _translate_single_source_rows(
        late_rows,
        import_hashes=import_hashes,
        heldout_centers=heldouts,
        prior_methods=methods,
        synthetic_per_class_total=args.synthetic_per_class_total,
        classifier_seed=args.classifier_seed,
        eval_set_id_prefix=args.eval_set_id_prefix,
        experiment_seed=args.experiment_seed,
    )
    if dense_rows:
        diagnostic_rows.extend(
            _translate_dense_baseline_rows(
                dense_rows,
                import_hashes=import_hashes,
                single_source_methods={row.candidate_method for row in diagnostic_rows},
                heldout_centers=heldouts,
                prior_methods=methods,
                synthetic_per_class_total=args.synthetic_per_class_total,
                classifier_seed=args.classifier_seed,
                eval_set_id_prefix=args.eval_set_id_prefix,
                experiment_seed=args.experiment_seed,
            )
        )

    candidate_manifest = _candidate_manifest_from_rows(diagnostic_rows, source_rows=late_rows)
    outputs = write_midogpp_phase1_artifacts(
        out_dir,
        rows=diagnostic_rows,
        candidate_manifest_rows=candidate_manifest,
    )
    _write_import_reports(
        out_dir,
        config=config,
        late_matrix=late_matrix,
        dense_matrix=dense_matrix,
        import_hashes=import_hashes,
        diagnostic_rows=diagnostic_rows,
        candidate_manifest=candidate_manifest,
        heldout_centers=heldouts,
        prior_methods=methods,
        synthetic_per_class_total=args.synthetic_per_class_total,
        classifier_seed=args.classifier_seed,
    )
    for label, path in outputs.items():
        print(f"Wrote {label}: {path}")
    print(f"Wrote import_provenance_report: {out_dir / 'reports' / 'import_provenance_report.json'}")


def _translate_single_source_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    import_hashes: Mapping[str, object],
    heldout_centers: Sequence[str],
    prior_methods: Sequence[str],
    synthetic_per_class_total: int,
    classifier_seed: int,
    eval_set_id_prefix: str,
    experiment_seed: int,
) -> list[MidogppDownstreamRow]:
    allowed_methods = set(prior_methods)
    translated: list[MidogppDownstreamRow] = []
    failed: list[tuple[str, str, str, str]] = []
    for row in rows:
        if str(row.get("pooling_rule", "")) != "single_source":
            continue
        if str(row.get("experiment_seed", "")) != str(experiment_seed):
            continue
        heldout = str(row.get("heldout_center", ""))
        method = str(row.get("prior_method", ""))
        if heldout not in heldout_centers:
            continue
        if allowed_methods and method not in allowed_methods:
            continue
        source = str(row.get("expert_id", ""))
        status = str(row.get("status", ""))
        if status != "ok":
            failed.append((heldout, source, method, status))
            continue
        if source == heldout or source not in MIDOGPP_ELIGIBLE_CENTERS:
            continue
        translated.append(
            _midogpp_row_from_upstream(
                row,
                row_type=MIDOGPP_SINGLE_SOURCE_ROW_TYPE,
                candidate_source_center=source,
                candidate_id=f"midogpp_source_{source}_{method}",
                candidate_method=method,
                expert_pool_type="single_source",
                import_hashes=import_hashes,
                synthetic_per_class_total=synthetic_per_class_total,
                classifier_seed=classifier_seed,
                eval_set_id_prefix=eval_set_id_prefix,
            )
        )
    if failed:
        raise ProtocolError(f"Cannot import failed MIDOG++ single-source rows: {failed[:10]}")
    _assert_context_candidate_coverage(translated)
    return translated


def _translate_dense_baseline_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    import_hashes: Mapping[str, object],
    single_source_methods: set[str],
    heldout_centers: Sequence[str],
    prior_methods: Sequence[str],
    synthetic_per_class_total: int,
    classifier_seed: int,
    eval_set_id_prefix: str,
    experiment_seed: int,
) -> list[MidogppDownstreamRow]:
    allowed_methods = set(prior_methods)
    translated: list[MidogppDownstreamRow] = []
    for row in rows:
        if str(row.get("experiment_seed", "")) != str(experiment_seed):
            continue
        heldout = str(row.get("heldout_center", ""))
        method = str(row.get("prior_method", ""))
        if heldout not in heldout_centers:
            continue
        if allowed_methods and method not in allowed_methods:
            continue
        if method not in single_source_methods:
            continue
        if str(row.get("status", "")) != "ok":
            raise ProtocolError(f"Cannot import failed MIDOG++ dense baseline row: {row}")
        translated.append(
            _midogpp_row_from_upstream(
                row,
                row_type=MIDOGPP_METHOD_BASELINE_ROW_TYPE,
                candidate_source_center="__dense_all_sources__",
                candidate_id=f"midogpp_dense_all_sources_{method}",
                candidate_method=method,
                expert_pool_type="decentralized_source_summary",
                import_hashes=import_hashes,
                synthetic_per_class_total=synthetic_per_class_total,
                classifier_seed=classifier_seed,
                eval_set_id_prefix=eval_set_id_prefix,
            )
        )
    return translated


def _midogpp_row_from_upstream(
    row: Mapping[str, str],
    *,
    row_type: str,
    candidate_source_center: str,
    candidate_id: str,
    candidate_method: str,
    expert_pool_type: str,
    import_hashes: Mapping[str, object],
    synthetic_per_class_total: int,
    classifier_seed: int,
    eval_set_id_prefix: str,
) -> MidogppDownstreamRow:
    method_hashes = import_hashes["method_hashes"]
    if not isinstance(method_hashes, Mapping):
        raise ProtocolError("Invalid MIDOG++ import method hash payload.")
    hashes = method_hashes.get(candidate_method)
    if not isinstance(hashes, Mapping):
        raise ProtocolError(f"Missing MIDOG++ import hashes for method={candidate_method!r}.")
    heldout = str(row["heldout_center"])
    replicate_seed = int(row["replicate_seed"])
    return MidogppDownstreamRow(
        heldout_center=heldout,
        candidate_source_center=candidate_source_center,
        candidate_id=candidate_id,
        candidate_method=candidate_method,
        experiment_seed=int(row["experiment_seed"]),
        replicate_seed=replicate_seed,
        support_size=0,
        support_seed=NO_SUPPORT_SEED,
        support_set_id=NO_SUPPORT_SET_ID,
        eval_set_id=f"{eval_set_id_prefix}_{heldout}",
        generation_seed=replicate_seed,
        latent_sample_seed=_optional_int(row.get("latent_sample_seed")),
        classifier_seed=classifier_seed,
        synthetic_per_class_total=synthetic_per_class_total,
        config_hash=str(hashes["config_hash"]),
        protocol_hash=str(hashes["protocol_hash"]),
        checkpoint_hash=_checkpoint_hash(row=row, row_type=row_type),
        feature_frame_hash=str(hashes["feature_frame_hash"]),
        expert_pool_type=expert_pool_type,
        row_type=row_type,
        bacc=float(row["bacc"]),
        macro_f1=float(row["macro_f1"]),
        status="ok",
        error_message="",
        claim_role="oracle_diagnostic",
        target_eval_labels_used_for_scoring_only=True,
        selection_used_target_labels=False,
        support_labels_used=False,
        eligibility=DIAGNOSTIC_ONLY,
    )


def _candidate_manifest_from_rows(
    rows: Sequence[MidogppDownstreamRow],
    *,
    source_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, object]]:
    source_budget_by_key = {
        (
            str(row.get("heldout_center", "")),
            str(row.get("expert_id", "")),
            str(row.get("prior_method", "")),
        ): str(row.get("synthetic_per_class_total", ""))
        for row in source_rows
        if str(row.get("pooling_rule", "")) == "single_source"
    }
    seen: set[tuple[str, str, str]] = set()
    manifest: list[dict[str, object]] = []
    for row in rows:
        if row.row_type != MIDOGPP_SINGLE_SOURCE_ROW_TYPE:
            continue
        key = (row.heldout_center, row.candidate_source_center, row.candidate_id)
        if key in seen:
            continue
        seen.add(key)
        upstream_budget = source_budget_by_key.get(
            (row.heldout_center, row.candidate_source_center, row.candidate_method),
            "",
        )
        manifest.append(
            {
                "dataset": row.dataset,
                "domain_regime": row.domain_regime,
                "heldout_center": row.heldout_center,
                "candidate_source_center": row.candidate_source_center,
                "candidate_id": row.candidate_id,
                "candidate_method": row.candidate_method,
                "expert_pool_type": row.expert_pool_type,
                "row_type": row.row_type,
                "eligibility": SELECTION_ELIGIBLE,
                "upstream_single_source_synthetic_per_class": upstream_budget,
            }
        )
    for heldout in sorted({row["heldout_center"] for row in manifest}):
        assert_midogpp_candidate_pool(
            heldout_center=str(heldout),
            candidate_rows=[row for row in manifest if row["heldout_center"] == heldout],
        )
    return sorted(
        manifest,
        key=lambda row: (
            str(row["heldout_center"]),
            str(row["candidate_method"]),
            str(row["candidate_source_center"]),
        ),
    )


def _assert_context_candidate_coverage(rows: Sequence[MidogppDownstreamRow]) -> None:
    observed: dict[tuple[str, str, int], set[str]] = {}
    for row in rows:
        observed.setdefault(
            (row.heldout_center, row.candidate_method, row.replicate_seed),
            set(),
        ).add(row.candidate_source_center)
    for (heldout, method, replicate_seed), sources in sorted(observed.items()):
        expected = set(MIDOGPP_ELIGIBLE_CENTERS).difference({heldout})
        if sources != expected:
            raise ProtocolError(
                "MIDOG++ late-aggregation import is missing source candidates; "
                f"heldout={heldout}, method={method}, replicate_seed={replicate_seed}, "
                f"missing={sorted(expected.difference(sources))}, extra={sorted(sources.difference(expected))}"
            )


def _build_import_hashes(
    *,
    config: Path,
    late_matrix: Path,
    dense_matrix: Path | None,
    heldout_centers: Sequence[str],
    prior_methods: Sequence[str],
    synthetic_per_class_total: int,
    classifier_seed: int,
) -> dict[str, object]:
    config_hash = _file_hash(config)
    late_matrix_hash = _file_hash(late_matrix)
    dense_matrix_hash = _file_hash(dense_matrix) if dense_matrix else ""
    method_hashes: dict[str, dict[str, str]] = {}
    method_rows = _read_csv(late_matrix)
    methods = sorted(
        {
            str(row.get("prior_method", ""))
            for row in method_rows
            if str(row.get("pooling_rule", "")) == "single_source"
            and (not prior_methods or str(row.get("prior_method", "")) in prior_methods)
        }
    )
    for method in methods:
        feature_frame_hash = stable_hash(
            {
                "source": "upstream_late_aggregation_matrix",
                "late_matrix_hash": late_matrix_hash,
                "dense_matrix_hash": dense_matrix_hash,
                "method": method,
            }
        )
        snapshot = FrozenProtocolSnapshot(
            candidate_pool_hash=stable_hash(
                {
                    "heldout_centers": list(heldout_centers),
                    "eligible_centers": list(MIDOGPP_ELIGIBLE_CENTERS),
                    "method": method,
                    "source": "upstream_late_aggregation_single_source_rows",
                }
            ),
            generation_config_hash=stable_hash(
                {
                    "method": method,
                    "synthetic_per_class_total": synthetic_per_class_total,
                    "generation_seed_source": "upstream_replicate_seed",
                    "upstream_late_matrix_hash": late_matrix_hash,
                }
            ),
            classifier_config_hash=stable_hash(
                {
                    "classifier": "upstream_locked_logistic_regression",
                    "classifier_seed_sentinel": classifier_seed,
                    "scaler": "upstream_artifact",
                }
            ),
            metric_config_hash=stable_hash({"primary": ["bacc", "macro_f1"], "chosen_before_target_eval": True}),
            feature_config_hash=feature_frame_hash,
            routing_config_hash=stable_hash(
                {
                    "selection": "none",
                    "role": "diagnostic_import_only",
                    "method": method,
                }
            ),
        )
        method_hashes[method] = {
            "config_hash": stable_hash({"config_hash": config_hash, "method": method}),
            "protocol_hash": snapshot.protocol_hash,
            "feature_frame_hash": feature_frame_hash,
        }
    return {
        "schema_version": DEFAULT_IMPORT_SCHEMA_VERSION,
        "config_hash": config_hash,
        "late_matrix_hash": late_matrix_hash,
        "dense_matrix_hash": dense_matrix_hash,
        "method_hashes": method_hashes,
    }


def _write_import_reports(
    out_dir: Path,
    *,
    config: Path,
    late_matrix: Path,
    dense_matrix: Path | None,
    import_hashes: Mapping[str, object],
    diagnostic_rows: Sequence[MidogppDownstreamRow],
    candidate_manifest: Sequence[Mapping[str, object]],
    heldout_centers: Sequence[str],
    prior_methods: Sequence[str],
    synthetic_per_class_total: int,
    classifier_seed: int,
) -> None:
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    single_rows = [row for row in diagnostic_rows if row.row_type == MIDOGPP_SINGLE_SOURCE_ROW_TYPE]
    baseline_rows = [row for row in diagnostic_rows if row.row_type == MIDOGPP_METHOD_BASELINE_ROW_TYPE]
    report = {
        "schema_version": DEFAULT_IMPORT_SCHEMA_VERSION,
        "status": "PASS",
        "config": str(config),
        "late_aggregation_matrix": str(late_matrix),
        "dense_matrix": str(dense_matrix) if dense_matrix else "",
        "hashes": import_hashes,
        "heldout_centers": list(heldout_centers),
        "prior_methods": list(prior_methods) if prior_methods else sorted({row.candidate_method for row in single_rows}),
        "synthetic_per_class_total": synthetic_per_class_total,
        "classifier_seed": classifier_seed,
        "diagnostic_rows": len(diagnostic_rows),
        "single_source_rows": len(single_rows),
        "method_baseline_rows": len(baseline_rows),
        "candidate_manifest_rows": len(candidate_manifest),
        "claim_boundary": (
            "Imported from already-scored upstream diagnostic rows; no deployable selection "
            "or 16D-to-target-feature transform was performed."
        ),
    }
    (reports_dir / "import_provenance_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    snapshot = FrozenProtocolSnapshot(
        candidate_pool_hash=stable_hash({"candidate_manifest": list(candidate_manifest)}),
        generation_config_hash=stable_hash(
            {
                "source": "upstream_late_aggregation_import",
                "synthetic_per_class_total": synthetic_per_class_total,
                "method_hashes": import_hashes["method_hashes"],
            }
        ),
        classifier_config_hash=stable_hash(
            {
                "classifier": "upstream_locked_logistic_regression",
                "classifier_seed_sentinel": classifier_seed,
            }
        ),
        metric_config_hash=stable_hash({"primary": ["bacc", "macro_f1"], "chosen_before_target_eval": True}),
        feature_config_hash=stable_hash(
            {
                "late_matrix_hash": import_hashes["late_matrix_hash"],
                "dense_matrix_hash": import_hashes["dense_matrix_hash"],
            }
        ),
        routing_config_hash=stable_hash({"selection": "none", "role": "diagnostic_import_only"}),
    )
    write_frozen_snapshot(out_dir / "configs" / "frozen_protocol_snapshot.json", snapshot)


def _read_csv(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    if not path.exists():
        raise ProtocolError(f"MIDOG++ import input does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _parse_centers(raw: str) -> tuple[str, ...]:
    centers = tuple(part.strip() for part in raw.split(",") if part.strip())
    unknown = sorted(set(centers).difference(MIDOGPP_ELIGIBLE_CENTERS))
    if unknown:
        raise ProtocolError(f"Unknown MIDOG++ heldout centers: {unknown}")
    return centers


def _checkpoint_hash(*, row: Mapping[str, str], row_type: str) -> str:
    fields = {
        "row_type": row_type,
        "generated_features_hash": row.get("generated_features_hash", ""),
        "prediction_hash": row.get("prediction_hash", ""),
        "composed_prior_hash": row.get("composed_prior_hash", ""),
        "summary_set_hash": row.get("summary_set_hash", ""),
    }
    return stable_hash(fields)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _optional_int(raw: object) -> int | None:
    if raw in {None, ""}:
        return None
    return int(raw)


if __name__ == "__main__":
    try:
        main()
    except ProtocolError as exc:
        raise SystemExit(str(exc)) from exc
