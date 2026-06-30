"""Build a real MIDOG++ phase-2 preflight config from frozen routing inputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvae_downstream_evaluation.schemas.midogpp_phase2 import class_prior_hash  # noqa: E402


DEFAULT_EXPERT_MANIFEST = (
    "cvae_rebuild/artifacts/midogpp/support_nelbo_routing_v1/inputs/source_expert_manifest.csv"
)
DEFAULT_SUPPORT_SETS = (
    "cvae_rebuild/artifacts/midogpp/support_nelbo_routing_v1/inputs/frozen_support_sets.csv"
)
DEFAULT_SUPPORT_SCORES = (
    "cvae_rebuild/artifacts/midogpp/support_nelbo_routing_v1/inputs/frozen_support_nelbo_scores.csv"
)
DEFAULT_TARGET_MANIFEST = "datasets/midogpp/artifacts/midogpp_annotation_patch_v1/manifest.csv"
DEFAULT_PHASE2_ROOT = (
    "cvae_downstream_evaluation/artifacts/midogpp/phase2_target_support_adaptation_virchow2_seed42"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a MIDOG++ phase-2 preflight config with locked support/eval rows."
    )
    parser.add_argument("--heldout-center")
    parser.add_argument("--support-seed", type=int)
    parser.add_argument("--all-contexts", action="store_true")
    parser.add_argument("--heldout-centers", default="0,1,2,3,5,6,7,8,9")
    parser.add_argument("--support-seeds", default="17,23,31")
    parser.add_argument("--support-size", default=32, type=int)
    parser.add_argument("--experiment-seed", default=42, type=int)
    parser.add_argument("--replicate", default="0")
    parser.add_argument("--freeze-timestamp", default="2026-06-30T00:00:00Z")
    parser.add_argument("--out", required=True, help="JSON config path to write.")
    parser.add_argument("--out-dir", default=DEFAULT_PHASE2_ROOT)
    parser.add_argument("--expert-manifest", default=DEFAULT_EXPERT_MANIFEST)
    parser.add_argument("--support-sets", default=DEFAULT_SUPPORT_SETS)
    parser.add_argument("--support-scores", default=DEFAULT_SUPPORT_SCORES)
    parser.add_argument("--target-manifest", default=DEFAULT_TARGET_MANIFEST)
    parser.add_argument("--eval-split", default="test")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    experiment_seed = str(args.experiment_seed)
    support_size = int(args.support_size)

    expert_rows = _read_csv(Path(args.expert_manifest))
    support_set_rows = _read_csv(Path(args.support_sets))
    support_score_rows = _read_csv(Path(args.support_scores))
    target_rows = _read_csv(Path(args.target_manifest))

    prior = {"0": 0.5, "1": 0.5}
    prior_hash = class_prior_hash(prior, class_order=("0", "1"))

    source_rows = _source_rows(
        expert_rows,
        experiment_seed=experiment_seed,
        prior=prior,
        prior_hash=prior_hash,
    )
    if args.all_contexts:
        heldouts = _csv_values(args.heldout_centers)
        support_seeds = _csv_values(args.support_seeds)
    else:
        if args.heldout_center is None or args.support_seed is None:
            raise SystemExit("--heldout-center and --support-seed are required unless --all-contexts is set")
        heldouts = [str(args.heldout_center)]
        support_seeds = [str(args.support_seed)]

    contexts = [
        _context_payload(
            support_set_rows=support_set_rows,
            support_score_rows=support_score_rows,
            target_rows=target_rows,
            heldout=heldout,
            support_seed=support_seed,
            support_size=support_size,
            experiment_seed=experiment_seed,
            eval_split=str(args.eval_split),
            scoped_candidate_ids=bool(args.all_contexts),
        )
        for heldout in heldouts
        for support_seed in support_seeds
    ]

    freeze_run_id = "midogpp_phase2_preflight_real_all_contexts" if args.all_contexts else (
        f"midogpp_phase2_preflight_real_center{heldouts[0]}_seed{support_seeds[0]}"
    )
    payload = {
        "out_dir": str(args.out_dir),
        "center_column": "center",
        "freeze_run_id": freeze_run_id,
        "freeze_timestamp": str(args.freeze_timestamp),
        "snapshot_fields": {
            "metric_config_hash": "midogpp_phase2_metric_config_predeclared_v1",
            "protocol_hash": "midogpp_phase2_preflight_v1",
            "class_prior_value_hash": prior_hash,
        },
        "source_rows": source_rows,
        "contexts": contexts,
    }
    if not args.all_contexts:
        context = contexts[0]
        payload.update(
            {
                "heldout_center": context["heldout_center"],
                "support_size": context["support_size"],
                "support_seed": context["support_seed"],
                "replicate": context["replicate"],
                "target_rows": [],
                "support_rows": context["support_rows"],
                "eval_rows": context["eval_rows"],
                "support_scores": context["support_scores"],
            }
        )
        payload.pop("contexts")
    _validate_counts(payload)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(_summary(payload, out), indent=2, sort_keys=True))


def _source_rows(
    rows: list[dict[str, str]],
    *,
    experiment_seed: str,
    prior: dict[str, float],
    prior_hash: str,
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: str(item.get("expert_id", ""))):
        expert_id = str(row.get("expert_id", ""))
        if str(row.get("experiment_seed", "")) != experiment_seed:
            continue
        out.append(
            {
                "source_center": expert_id,
                "candidate_id": f"source_{expert_id}",
                "stable_candidate_id": f"source_{expert_id}",
                "checkpoint_path": row["checkpoint_path"],
                "checkpoint_hash": row["checkpoint_hash"],
                "checkpoint_provenance_hash": row["checkpoint_hash"],
                "feature_frame_hash": row["feature_frame_hash"],
                "feature_provenance": "source_only_label_free",
                "feature_used_target_eval_labels": False,
                "feature_used_downstream_utility": False,
                "feature_used_fidelity": False,
                "feature_used_oracle_gap": False,
                "feature_used_all_target_eval_statistics": False,
                "embedding_representation_hash": "virchow2_pca64",
                "preprocessing_hash": row["feature_frame_hash"],
                "decoder_likelihood_family": "gaussian",
                "embedding_dimensionality": "64",
                "nelbo_reduction": "mean_per_sample",
                "beta_kl_weight": "0.01",
                "checkpoint_objective": "conditional_cvae_elbo",
                "checkpoint_seed": int(experiment_seed),
                "generation_mode": "class_balanced",
                "generation_class_prior_policy": "uniform_generation_policy",
                "synthetic_budget": 128,
                "generation_seed": int(experiment_seed),
                "classifier_seed": int(experiment_seed),
                "class_order": ["0", "1"],
                "class_prior_values": prior,
                "class_prior_rule": "uniform",
                "class_prior_value_hash": prior_hash,
                "scorer_implementation_hash": "prior_weighted_expected_conditional_nelbo_v1",
                "config_hash": "virchow2_cvae_dense_late_all_sources_midogpp_v1",
                "protocol_hash": "midogpp_phase2_preflight_v1",
            }
        )
    return out


def _locked_support_rows(
    rows: list[dict[str, str]],
    *,
    heldout: str,
    support_seed: str,
    support_size: int,
    experiment_seed: str,
) -> list[dict[str, object]]:
    selected = [
        dict(row)
        for row in rows
        if str(row.get("experiment_seed", "")) == experiment_seed
        and str(row.get("heldout_center", "")) == heldout
        and str(row.get("support_seed", "")) == support_seed
    ]
    if len(selected) != support_size:
        raise SystemExit(f"expected {support_size} locked support rows, got {len(selected)}")
    return selected


def _context_payload(
    *,
    support_set_rows: list[dict[str, str]],
    support_score_rows: list[dict[str, str]],
    target_rows: list[dict[str, str]],
    heldout: str,
    support_seed: str,
    support_size: int,
    experiment_seed: str,
    eval_split: str,
    scoped_candidate_ids: bool,
) -> dict[str, object]:
    return {
        "heldout_center": str(heldout),
        "support_seed": int(support_seed),
        "support_size": int(support_size),
        "replicate": "0",
        "support_rows": _locked_support_rows(
            support_set_rows,
            heldout=str(heldout),
            support_seed=str(support_seed),
            support_size=int(support_size),
            experiment_seed=str(experiment_seed),
        ),
        "eval_rows": _eval_rows(target_rows, heldout=str(heldout), eval_split=eval_split),
        "support_scores": _support_scores(
            support_score_rows,
            heldout=str(heldout),
            support_seed=str(support_seed),
            experiment_seed=str(experiment_seed),
            scoped_candidate_ids=scoped_candidate_ids,
        ),
    }


def _eval_rows(rows: list[dict[str, str]], *, heldout: str, eval_split: str) -> list[dict[str, object]]:
    selected = [
        {
            "sample_id": row["sample_id"],
            "case_id": row.get("case_id", ""),
            "patient_id": row.get("case_id", ""),
            "slide_id": row.get("case_id", ""),
            "group_id": row.get("case_id", ""),
            "center": row["center"],
            "split": row.get("split", ""),
            "label": row.get("label", ""),
        }
        for row in rows
        if str(row.get("center", "")) == heldout and str(row.get("split", "")) == eval_split
    ]
    if not selected:
        raise SystemExit(f"no eval rows for center={heldout!r}, split={eval_split!r}")
    return selected


def _support_scores(
    rows: list[dict[str, str]],
    *,
    heldout: str,
    support_seed: str,
    experiment_seed: str,
    scoped_candidate_ids: bool,
) -> list[dict[str, object]]:
    selected = []
    for row in rows:
        if (
            str(row.get("experiment_seed", "")) == experiment_seed
            and str(row.get("heldout_center", "")) == heldout
            and str(row.get("support_seed", "")) == support_seed
        ):
            candidate_id = (
                f"target_{heldout}_source_{row['expert_id']}"
                if scoped_candidate_ids
                else f"source_{row['expert_id']}"
            )
            selected.append(
                {
                    "candidate_id": candidate_id,
                    "support_score": float(row["calibrated_support_nelbo"]),
                    "support_score_variance_or_se": float(row["support_se"]),
                    "support_n": int(float(row["support_n"])),
                    "encoder_mode": "deterministic",
                }
            )
    return selected


def _validate_counts(payload: dict[str, object]) -> None:
    if "contexts" in payload:
        contexts = payload["contexts"]
        if not isinstance(contexts, list) or not contexts:
            raise SystemExit("contexts must be a non-empty list")
        if len(payload["source_rows"]) != 9:  # type: ignore[arg-type]
            raise SystemExit(f"expected 9 source_rows, got {len(payload['source_rows'])}")  # type: ignore[arg-type]
        for idx, context in enumerate(contexts):
            if len(context["support_rows"]) != int(context["support_size"]):  # type: ignore[index,arg-type]
                raise SystemExit(f"context {idx} support row count does not match support_size")
            if len(context["support_scores"]) != 8:  # type: ignore[index,arg-type]
                raise SystemExit(f"context {idx} expected 8 support_scores, got {len(context['support_scores'])}")  # type: ignore[index]
            if not context["eval_rows"]:  # type: ignore[index]
                raise SystemExit(f"context {idx} eval_rows must not be empty")
        return
    checks = {
        "source_rows": 9,
        "support_rows": int(payload["support_size"]),
        "support_scores": 8,
    }
    for key, expected in checks.items():
        observed = len(payload[key])  # type: ignore[arg-type]
        if observed != expected:
            raise SystemExit(f"expected {expected} {key}, got {observed}")
    if not payload["eval_rows"]:
        raise SystemExit("eval_rows must not be empty")


def _summary(payload: dict[str, object], out: Path) -> dict[str, object]:
    if "contexts" in payload:
        contexts = payload["contexts"]
        assert isinstance(contexts, list)
        return {
            "wrote": str(out),
            "contexts": len(contexts),
            "source_rows": len(payload["source_rows"]),  # type: ignore[arg-type]
            "support_rows": sum(len(context["support_rows"]) for context in contexts),
            "eval_rows": sum(len(context["eval_rows"]) for context in contexts),
            "support_scores": sum(len(context["support_scores"]) for context in contexts),
        }
    return {
        "wrote": str(out),
        "heldout_center": payload["heldout_center"],
        "support_seed": payload["support_seed"],
        "source_rows": len(payload["source_rows"]),  # type: ignore[arg-type]
        "support_rows": len(payload["support_rows"]),  # type: ignore[arg-type]
        "eval_rows": len(payload["eval_rows"]),  # type: ignore[arg-type]
        "support_scores": len(payload["support_scores"]),  # type: ignore[arg-type]
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _csv_values(value: str) -> list[str]:
    out = [item.strip() for item in str(value).split(",") if item.strip()]
    if not out:
        raise SystemExit("empty comma-separated value list")
    return out


if __name__ == "__main__":
    main()
