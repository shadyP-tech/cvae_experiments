from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from config import _load_mapping, _mapping
from domain_regime import (
    MIDOGPP_DOMAIN_REGIME,
    MidogppContractInfo,
    validate_cache_report_split_counts,
    validate_domain_regime_config,
)
from features import default_cache_path, load_feature_cache
from metrics import spearman
from preservation_repair import _path
from protocol import ProtocolError, assert_candidate_pool, build_leakage_report
from reporting import prepare_artifact_dirs, write_csv_rows, write_json
from splits import candidate_experts, random_unlabeled_support_eval_split


MIDOGPP_SUPPORT_NELBO_ROUTING_NAME = "midogpp_support_nelbo_routing_v1"
PRIMARY_METHOD = "support_nelbo_top1_marginal_unlabeled"
DEFAULT_ARTIFACT_ROOT = "cvae_rebuild/artifacts/midogpp/support_nelbo_routing_v1"
SUPPORTED_WEIGHTING_POLICIES = ("none", "dense_all_source_softmax", "topk_softmax")
SUPPORTED_CLASS_PRIOR_SOURCES = ("uniform", "source_validation", "contract_metadata")
GROUP_ID_KEYS = ("patient_id", "case_id", "slide_id", "group_id")


@dataclass(frozen=True)
class MidogppSupportNelboRoutingConfig:
    name: str
    artifact_root: Path
    feature_cache_root: Path
    dataset_contract_artifact_root: Path
    cache_report_path: Path
    source_expert_manifest_path: Path
    support_nelbo_scores_path: Path
    eval_nelbo_matrix_path: Path
    backbone: str
    domain_regime: str
    strict_full_run_matrix: bool
    strict_available_seed_domain_coverage: bool
    experiment_seeds: tuple[int, ...]
    heldout_centers: tuple[str, ...]
    support_size: int
    support_seeds: tuple[int, ...]
    primary_method: str
    primary_score: str
    support_sampler: str
    nelbo_target: str
    class_prior_source: str
    selection_rule: str
    tie_breaking: str
    weighting_policy: str
    softmax_tau: float | None
    top_k: int | None
    weight_aggregation_target: str
    calibration_source: str
    scorer_config_hash: str
    feature_frame_policy: str


def load_midogpp_support_nelbo_routing_config(path: str | Path) -> MidogppSupportNelboRoutingConfig:
    source = Path(path).resolve()
    data = _load_mapping(source)
    base_dir = source.parents[2] if len(source.parents) >= 3 else source.parent
    return parse_midogpp_support_nelbo_routing_config(data, base_dir=base_dir)


def parse_midogpp_support_nelbo_routing_config(
    data: Mapping[str, Any],
    *,
    base_dir: str | Path = ".",
) -> MidogppSupportNelboRoutingConfig:
    base = Path(base_dir)
    experiment = _mapping(data, "experiment")
    inputs = _mapping(data, "inputs")
    run = _mapping(data, "run_matrix")
    routing = _mapping(data, "routing")
    scoring = _mapping(data, "scoring")

    cfg = MidogppSupportNelboRoutingConfig(
        name=str(experiment["name"]),
        artifact_root=_path(base, str(experiment["artifact_root"])),
        feature_cache_root=_path(base, str(inputs["feature_cache_root"])),
        dataset_contract_artifact_root=_path(base, str(inputs["dataset_contract_artifact_root"])),
        cache_report_path=_path(base, str(inputs["cache_report_path"])),
        source_expert_manifest_path=_path(base, str(inputs["source_expert_manifest_path"])),
        support_nelbo_scores_path=_path(base, str(inputs["support_nelbo_scores_path"])),
        eval_nelbo_matrix_path=_path(base, str(inputs["eval_nelbo_matrix_path"])),
        backbone=str(inputs.get("backbone", "")),
        domain_regime=str(run["domain_regime"]),
        strict_full_run_matrix=bool(run.get("strict_full_run_matrix", False)),
        strict_available_seed_domain_coverage=bool(run.get("strict_available_seed_domain_coverage", False)),
        experiment_seeds=tuple(int(v) for v in run["experiment_seeds"]),
        heldout_centers=tuple(str(v) for v in run["heldout_centers"]),
        support_size=int(run["support_size"]),
        support_seeds=tuple(int(v) for v in run["support_seeds"]),
        primary_method=str(experiment["primary_method"]),
        primary_score=str(routing["primary_score"]),
        support_sampler=str(routing["support_sampler"]),
        nelbo_target=str(scoring["nelbo_target"]),
        class_prior_source=str(scoring["class_prior_source"]),
        selection_rule=str(routing["selection_rule"]),
        tie_breaking=str(routing["tie_breaking"]),
        weighting_policy=str(routing.get("weighting_policy", "none")),
        softmax_tau=None if routing.get("softmax_tau") is None else float(routing["softmax_tau"]),
        top_k=None if routing.get("top_k") is None else int(routing["top_k"]),
        weight_aggregation_target=str(routing.get("weight_aggregation_target", "none")),
        calibration_source=str(scoring["calibration_source"]),
        scorer_config_hash=str(scoring["scorer_config_hash"]),
        feature_frame_policy=str(scoring["feature_frame_policy"]),
    )
    validate_midogpp_support_nelbo_routing_config(cfg)
    return cfg


def validate_midogpp_support_nelbo_routing_config(cfg: MidogppSupportNelboRoutingConfig) -> MidogppContractInfo:
    if cfg.name != MIDOGPP_SUPPORT_NELBO_ROUTING_NAME:
        raise ProtocolError(f"experiment.name must be {MIDOGPP_SUPPORT_NELBO_ROUTING_NAME!r}.")
    if cfg.primary_method != PRIMARY_METHOD:
        raise ProtocolError(f"experiment.primary_method must be {PRIMARY_METHOD!r}.")
    if cfg.backbone != "virchow2":
        raise ProtocolError("MIDOG++ support-NELBO routing is locked to backbone=virchow2.")
    if cfg.domain_regime != MIDOGPP_DOMAIN_REGIME:
        raise ProtocolError(f"domain_regime must be {MIDOGPP_DOMAIN_REGIME!r}.")
    if cfg.support_size <= 0:
        raise ProtocolError("support_size must be positive.")
    if not cfg.experiment_seeds or not cfg.support_seeds:
        raise ProtocolError("experiment_seeds and support_seeds must be non-empty.")
    if cfg.primary_score != "calibrated_marginal_support_nelbo":
        raise ProtocolError("routing.primary_score must be calibrated_marginal_support_nelbo.")
    if cfg.support_sampler != "random_unlabeled_sample_ids":
        raise ProtocolError("routing.support_sampler must be random_unlabeled_sample_ids.")
    if cfg.nelbo_target != "marginal_unlabeled":
        raise ProtocolError("scoring.nelbo_target must be marginal_unlabeled for unlabeled target support.")
    if cfg.class_prior_source not in SUPPORTED_CLASS_PRIOR_SOURCES:
        raise ProtocolError(f"Unsupported class_prior_source={cfg.class_prior_source!r}.")
    if cfg.calibration_source != "source_validation_only":
        raise ProtocolError("scoring.calibration_source must be source_validation_only.")
    if not cfg.scorer_config_hash:
        raise ProtocolError("scoring.scorer_config_hash must be recorded.")
    if cfg.feature_frame_policy not in {"shared_source_only", "expert_local_source_only"}:
        raise ProtocolError("scoring.feature_frame_policy must be shared_source_only or expert_local_source_only.")
    if cfg.selection_rule != "min_calibrated_support_nelbo":
        raise ProtocolError("routing.selection_rule must be min_calibrated_support_nelbo.")
    if cfg.tie_breaking != "expert_id_ascending":
        raise ProtocolError("routing.tie_breaking must be expert_id_ascending.")
    if cfg.weighting_policy not in SUPPORTED_WEIGHTING_POLICIES:
        raise ProtocolError(f"Unsupported weighting_policy={cfg.weighting_policy!r}.")
    if cfg.weighting_policy == "none":
        if cfg.softmax_tau is not None or cfg.top_k is not None or cfg.weight_aggregation_target != "none":
            raise ProtocolError("weighting_policy=none must not set tau, top_k, or aggregation target.")
    else:
        if cfg.softmax_tau is None or cfg.softmax_tau <= 0.0:
            raise ProtocolError("Weighted routing requires a fixed positive softmax_tau.")
        if cfg.weight_aggregation_target not in {"budget_weighted_expected_expert_utility", "score_aggregation"}:
            raise ProtocolError("Weighted routing must declare a supported aggregation target.")
        if cfg.weighting_policy == "topk_softmax" and (cfg.top_k is None or cfg.top_k <= 0):
            raise ProtocolError("topk_softmax requires a fixed positive top_k.")
        if cfg.weighting_policy == "dense_all_source_softmax" and cfg.top_k is not None:
            raise ProtocolError("dense_all_source_softmax must not set top_k.")
    info = validate_domain_regime_config(
        domain_regime=cfg.domain_regime,
        heldout_centers=cfg.heldout_centers,
        dataset_contract_artifact_root=cfg.dataset_contract_artifact_root,
        artifact_root=cfg.artifact_root,
        strict_full_run_matrix=cfg.strict_full_run_matrix,
        strict_available_seed_domain_coverage=cfg.strict_available_seed_domain_coverage,
    )
    if info is None:
        raise ProtocolError("MIDOG++ support-NELBO routing requires a MIDOG++ contract.")
    validate_cache_report_split_counts(cfg.cache_report_path, info)
    for required in (
        cfg.source_expert_manifest_path,
        cfg.support_nelbo_scores_path,
        cfg.eval_nelbo_matrix_path,
    ):
        if not required.exists():
            raise ProtocolError(f"Required frozen routing input is missing: {required}")
    return info


def run_midogpp_support_nelbo_routing(
    cfg: MidogppSupportNelboRoutingConfig,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    contract_info = validate_midogpp_support_nelbo_routing_config(cfg)
    root = prepare_artifact_dirs(artifact_root or cfg.artifact_root)
    if "cvae_rebuild/artifacts/midogpp" not in root.as_posix():
        raise ProtocolError("MIDOG++ support-NELBO routing artifact root must remain under cvae_rebuild/artifacts/midogpp/.")

    expert_rows = _read_csv(cfg.source_expert_manifest_path)
    support_rows = _read_csv(cfg.support_nelbo_scores_path)
    eval_rows = _read_csv(cfg.eval_nelbo_matrix_path)
    _validate_expert_manifest(cfg, contract_info, expert_rows)

    split_rows = _support_split_manifest_rows(cfg, contract_info)
    ranked_rows, decision_rows = _rank_and_decide(cfg, contract_info, support_rows)

    decision_version = f"frozen_decision_unix_{int(time.time())}"
    for row in decision_rows:
        row["decision_materialized_before_eval"] = True
        row["decision_version"] = decision_version

    write_csv_rows(root / "manifests" / "support_split_manifest.csv", split_rows)
    write_csv_rows(root / "manifests" / "expert_manifest.csv", expert_rows)
    write_csv_rows(root / "tables" / "support_nelbo_scores.csv", ranked_rows)
    write_csv_rows(root / "tables" / "routing_decisions.csv", decision_rows)

    eval_matrix_rows = _eval_matrix_rows(cfg, contract_info, eval_rows)
    alignment_rows = _alignment_rows(cfg, contract_info, ranked_rows, eval_matrix_rows)
    write_csv_rows(root / "tables" / "all_expert_eval_nelbo_matrix.csv", eval_matrix_rows)
    write_csv_rows(root / "tables" / "routing_to_eval_nelbo_alignment.csv", alignment_rows)
    write_json(root / "reports" / "leakage_report.json", _leakage_report_payload(cfg, contract_info))
    write_json(root / "reports" / "checkpoint_cache_contract_provenance.json", _provenance_payload(cfg, contract_info))
    write_json(root / "reports" / "scorer_config_report.json", _scorer_payload(cfg))
    write_json(root / "manifests" / "protocol_manifest.json", _protocol_manifest_payload(cfg, contract_info))
    write_json(root / "run_config_resolved.yaml", _resolved_config_payload(cfg))
    _write_decision_summary(root)
    return root


def _validate_expert_manifest(
    cfg: MidogppSupportNelboRoutingConfig,
    contract_info: MidogppContractInfo,
    rows: Sequence[Mapping[str, str]],
) -> None:
    if not rows:
        raise ProtocolError("source_expert_manifest_path must contain frozen expert rows.")
    expected_domains = set(contract_info.eligible_domain_ids)
    for seed in cfg.experiment_seeds:
        by_domain = {str(row.get("expert_id", "")): row for row in rows if int(float(row.get("experiment_seed", -1))) == seed}
        if set(by_domain) != expected_domains:
            raise ProtocolError(
                f"Expert manifest for seed {seed} must cover exactly eligible domains {sorted(expected_domains)}."
            )
        for expert_id, row in by_domain.items():
            if expert_id in contract_info.ineligible_domain_ids:
                raise ProtocolError(f"Ineligible expert {expert_id} appeared in source expert manifest.")
            if str(row.get("source_only", "")).lower() not in {"1", "true", "yes"}:
                raise ProtocolError(f"Expert {expert_id} must be marked source_only=true.")
            if str(row.get("frozen", "")).lower() not in {"1", "true", "yes"}:
                raise ProtocolError(f"Expert {expert_id} must be marked frozen=true.")
            if not str(row.get("checkpoint_hash", "")):
                raise ProtocolError(f"Expert {expert_id} is missing checkpoint_hash provenance.")


def _support_split_manifest_rows(
    cfg: MidogppSupportNelboRoutingConfig,
    contract_info: MidogppContractInfo,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in cfg.experiment_seeds:
        cache = load_feature_cache(_feature_cache_path(cfg.feature_cache_root, seed=seed, split="test"))
        for heldout in cfg.heldout_centers:
            candidates = candidate_experts(contract_info.eligible_domain_ids, heldout)
            assert_candidate_pool(
                heldout_center=heldout,
                candidate_experts=candidates,
                expected_count=len(contract_info.eligible_domain_ids) - 1,
            )
            for support_seed in cfg.support_seeds:
                split = random_unlabeled_support_eval_split(
                    cache.metadata,
                    heldout_center=heldout,
                    support_size=cfg.support_size,
                    support_seed=support_seed,
                )
                group_report = _group_disjointness_report(cache.metadata, split.support_indices, split.eval_indices)
                rows.append(
                    {
                        "experiment_seed": seed,
                        "heldout_center": heldout,
                        "support_seed": support_seed,
                        "support_size_requested": split.support_size_requested,
                        "support_size_actual": split.support_size_actual,
                        "support_eval_split_id": split.support_eval_split_id,
                        "support_sample_ids": json.dumps(list(split.support_sample_ids)),
                        "eval_sample_ids": json.dumps(list(split.eval_sample_ids)),
                        "support_labels_used": False,
                        "group_id_status": group_report["status"],
                        "group_id_key": group_report["key"],
                        "support_eval_group_disjoint": group_report["disjoint"],
                    }
                )
    return rows


def _feature_cache_path(root: str | Path, *, seed: int, split: str) -> Path:
    path = default_cache_path(root, seed=seed, split=split)
    if path.exists():
        return path
    npz = path.with_suffix(".npz")
    return npz if npz.exists() else path


def _group_disjointness_report(
    metadata: Sequence[Mapping[str, object]],
    support_indices: Sequence[int],
    eval_indices: Sequence[int],
) -> dict[str, object]:
    for key in GROUP_ID_KEYS:
        support = {str(metadata[idx].get(key, "")) for idx in support_indices if str(metadata[idx].get(key, ""))}
        eval_groups = {str(metadata[idx].get(key, "")) for idx in eval_indices if str(metadata[idx].get(key, ""))}
        if support or eval_groups:
            overlap = support.intersection(eval_groups)
            if overlap:
                raise ProtocolError(f"Support/eval group overlap for {key}: {sorted(overlap)[:5]}")
            return {"status": "available", "key": key, "disjoint": True}
    return {"status": "not_available", "key": "", "disjoint": ""}


def _rank_and_decide(
    cfg: MidogppSupportNelboRoutingConfig,
    contract_info: MidogppContractInfo,
    input_rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    out_scores: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    expected_count = len(contract_info.eligible_domain_ids) - 1
    for seed in cfg.experiment_seeds:
        for heldout in cfg.heldout_centers:
            candidates = candidate_experts(contract_info.eligible_domain_ids, heldout)
            assert_candidate_pool(heldout_center=heldout, candidate_experts=candidates, expected_count=expected_count)
            for support_seed in cfg.support_seeds:
                rows = [
                    row
                    for row in input_rows
                    if int(float(row.get("experiment_seed", -1))) == seed
                    and str(row.get("heldout_center", "")) == heldout
                    and int(float(row.get("support_seed", -1))) == support_seed
                ]
                by_expert = {str(row.get("expert_id", "")): row for row in rows}
                if set(by_expert) != set(candidates):
                    raise ProtocolError(
                        "Support scores must cover exactly the heldout-excluded candidate pool: "
                        f"seed={seed}, heldout={heldout}, support_seed={support_seed}."
                    )
                ranked = sorted(
                    by_expert.values(),
                    key=lambda row: (_float_cell(row, "calibrated_support_nelbo"), str(row.get("expert_id", ""))),
                )
                selected = str(ranked[0]["expert_id"])
                weights = _routing_weights(cfg, ranked)
                for rank, row in enumerate(ranked, start=1):
                    support_n = int(float(row.get("support_n", cfg.support_size)))
                    if support_n <= 0:
                        raise ProtocolError("support_n must be positive.")
                    score_row = {
                        "experiment_seed": seed,
                        "heldout_center": heldout,
                        "support_seed": support_seed,
                        "support_size": int(cfg.support_size),
                        "expert_id": str(row["expert_id"]),
                        "eligible_expert_count": expected_count,
                        "candidate_rank": rank,
                        "raw_support_nelbo": _float_cell(row, "raw_support_nelbo"),
                        "calibrated_support_nelbo": _float_cell(row, "calibrated_support_nelbo"),
                        "support_n": support_n,
                        "support_se": row.get("support_se", ""),
                        "selected_top1": int(rank == 1),
                        "selection_source": "target_support_nelbo",
                        "claim_role": "adoption_candidate",
                        "row_role": "routing_score",
                        "support_labels_used": False,
                        "routing_uses_eval_nelbo": 0,
                        "target_eval_used_for_scoring_only": True,
                        "adoption_eligible": True,
                        "oracle_eligible": False,
                        "nelbo_target": cfg.nelbo_target,
                        "class_prior_source": cfg.class_prior_source,
                        "scorer_config_hash": cfg.scorer_config_hash,
                        "feature_frame_policy": cfg.feature_frame_policy,
                        "calibration_source": cfg.calibration_source,
                    }
                    if support_n >= 2 and str(score_row["support_se"]) == "":
                        raise ProtocolError("support_se is required for every support score with n >= 2.")
                    out_scores.append(score_row)
                decisions.append(
                    {
                        "experiment_seed": seed,
                        "heldout_center": heldout,
                        "support_seed": support_seed,
                        "method": cfg.primary_method,
                        "selected_expert_id": selected,
                        "candidate_experts": json.dumps(list(candidates)),
                        "eligible_expert_count": expected_count,
                        "selection_source": "target_support_nelbo",
                        "support_labels_used": False,
                        "routing_uses_eval_nelbo": 0,
                        "target_eval_used_for_scoring_only": True,
                        "claim_role": "adoption_candidate",
                        "row_role": "routing_decision",
                        "adoption_eligible": True,
                        "oracle_eligible": False,
                        "weighting_policy": cfg.weighting_policy,
                        "weights_json": json.dumps(weights, sort_keys=True),
                    }
                )
    return out_scores, decisions


def _routing_weights(
    cfg: MidogppSupportNelboRoutingConfig,
    ranked_rows: Sequence[Mapping[str, str]],
) -> dict[str, float]:
    if cfg.weighting_policy == "none":
        return {str(ranked_rows[0]["expert_id"]): 1.0}
    rows = list(ranked_rows)
    if cfg.weighting_policy == "topk_softmax":
        rows = rows[: int(cfg.top_k)]
    tau = float(cfg.softmax_tau)
    logits = [-_float_cell(row, "calibrated_support_nelbo") / tau for row in rows]
    max_logit = max(logits)
    exps = [math.exp(value - max_logit) for value in logits]
    denom = sum(exps)
    return {str(row["expert_id"]): float(value / denom) for row, value in zip(rows, exps)}


def _eval_matrix_rows(
    cfg: MidogppSupportNelboRoutingConfig,
    contract_info: MidogppContractInfo,
    input_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    expected_count = len(contract_info.eligible_domain_ids) - 1
    for seed in cfg.experiment_seeds:
        for heldout in cfg.heldout_centers:
            candidates = candidate_experts(contract_info.eligible_domain_ids, heldout)
            by_expert = {
                str(row.get("expert_id", "")): row
                for row in input_rows
                if int(float(row.get("experiment_seed", -1))) == seed and str(row.get("heldout_center", "")) == heldout
            }
            if set(by_expert) != set(candidates):
                raise ProtocolError(f"Eval NELBO matrix must cover every candidate for seed={seed}, heldout={heldout}.")
            for expert_id in candidates:
                row = by_expert[expert_id]
                out.append(
                    {
                        "experiment_seed": seed,
                        "heldout_center": heldout,
                        "expert_id": expert_id,
                        "eligible_expert_count": expected_count,
                        "eval_mean_nelbo": _float_cell(row, "eval_mean_nelbo"),
                        "eval_n": int(float(row.get("eval_n", 0))),
                        "utility": -_float_cell(row, "eval_mean_nelbo"),
                        "selection_source": "diagnostic_only",
                        "claim_role": "diagnostic_only",
                        "row_role": "eval_nelbo_oracle_pool",
                        "support_labels_used": False,
                        "routing_uses_eval_nelbo": 1,
                        "target_eval_used_for_scoring_only": True,
                        "adoption_eligible": False,
                        "oracle_eligible": True,
                        "nelbo_target": cfg.nelbo_target,
                        "class_prior_source": cfg.class_prior_source,
                    }
                )
    return out


def _alignment_rows(
    cfg: MidogppSupportNelboRoutingConfig,
    contract_info: MidogppContractInfo,
    support_rows: Sequence[Mapping[str, object]],
    eval_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for seed in cfg.experiment_seeds:
        for heldout in cfg.heldout_centers:
            for support_seed in cfg.support_seeds:
                support_subset = [
                    row
                    for row in support_rows
                    if row["experiment_seed"] == seed and row["heldout_center"] == heldout and row["support_seed"] == support_seed
                ]
                eval_by_expert = {
                    str(row["expert_id"]): row
                    for row in eval_rows
                    if row["experiment_seed"] == seed and row["heldout_center"] == heldout
                }
                selected = min(
                    support_subset,
                    key=lambda row: (float(row["calibrated_support_nelbo"]), str(row["expert_id"])),
                )
                oracle = min(eval_by_expert.values(), key=lambda row: (float(row["eval_mean_nelbo"]), str(row["expert_id"])))
                ordered_support = sorted(support_subset, key=lambda row: str(row["expert_id"]))
                support_utility = [-float(row["calibrated_support_nelbo"]) for row in ordered_support]
                eval_utility = [float(eval_by_expert[str(row["expert_id"])]["utility"]) for row in ordered_support]
                selected_eval_nelbo = float(eval_by_expert[str(selected["expert_id"])]["eval_mean_nelbo"])
                oracle_eval_nelbo = float(oracle["eval_mean_nelbo"])
                out.append(
                    {
                        "experiment_seed": seed,
                        "heldout_center": heldout,
                        "support_seed": support_seed,
                        "method": cfg.primary_method,
                        "top1_oracle_hit": float(str(selected["expert_id"]) == str(oracle["expert_id"])),
                        "spearman_support_vs_eval_utility": spearman(support_utility, eval_utility),
                        "normalized_oracle_gap": _normalized_lower_is_better_gap(selected_eval_nelbo, oracle_eval_nelbo),
                        "oracle_gap_sign_convention": "selected_eval_nelbo_minus_best_eval_nelbo_over_abs_best; lower_is_better",
                        "selected_expert_id": str(selected["expert_id"]),
                        "oracle_expert_id": str(oracle["expert_id"]),
                        "selection_source": "diagnostic_only",
                        "claim_role": "diagnostic_only",
                        "row_role": "routing_eval_alignment",
                        "support_labels_used": False,
                        "routing_uses_eval_nelbo": 1,
                        "target_eval_used_for_scoring_only": True,
                        "adoption_eligible": False,
                        "oracle_eligible": True,
                    }
                )
    return out


def _normalized_lower_is_better_gap(selected: float, oracle: float) -> float:
    denom = abs(float(oracle))
    if denom <= 1.0e-12:
        return float(selected - oracle)
    return float((selected - oracle) / denom)


def _leakage_report_payload(
    cfg: MidogppSupportNelboRoutingConfig,
    contract_info: MidogppContractInfo,
) -> dict[str, object]:
    report = build_leakage_report(
        target_support_labels_for_selection=False,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=True,
        oracle_rows_diagnostic_only=True,
    ).to_json_dict()
    report.update(
        {
            "domain_regime": cfg.domain_regime,
            "contract_eligible_domain_ids": list(contract_info.eligible_domain_ids),
            "contract_ineligible_domain_ids": list(contract_info.ineligible_domain_ids),
            "support_sampler": cfg.support_sampler,
            "routing_uses_eval_nelbo_for_adoption_rows": 0,
            "oracle_rows_adoption_eligible": False,
            "strict_available_seed_domain_coverage": cfg.strict_available_seed_domain_coverage,
            "strict_full_run_matrix": cfg.strict_full_run_matrix,
        }
    )
    return report


def _provenance_payload(
    cfg: MidogppSupportNelboRoutingConfig,
    contract_info: MidogppContractInfo,
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_support_nelbo_routing_provenance_v1",
        "dataset_contract_artifact_root": str(cfg.dataset_contract_artifact_root),
        "dataset_contract_fingerprints": contract_info.fingerprints,
        "feature_cache_root": str(cfg.feature_cache_root),
        "cache_report_path": str(cfg.cache_report_path),
        "source_expert_manifest_path": str(cfg.source_expert_manifest_path),
        "support_nelbo_scores_path": str(cfg.support_nelbo_scores_path),
        "eval_nelbo_matrix_path": str(cfg.eval_nelbo_matrix_path),
        "source_experts_required_source_only": True,
        "source_experts_required_frozen": True,
    }


def _scorer_payload(cfg: MidogppSupportNelboRoutingConfig) -> dict[str, object]:
    return {
        "schema_version": "midogpp_support_nelbo_scorer_config_v1",
        "nelbo_target": cfg.nelbo_target,
        "class_prior_source": cfg.class_prior_source,
        "calibration_source": cfg.calibration_source,
        "primary_score": cfg.primary_score,
        "scorer_config_hash": cfg.scorer_config_hash,
        "feature_frame_policy": cfg.feature_frame_policy,
        "weighted_row_semantics": "sum_e w_e U_e budget-weighted expected expert utility",
    }


def _protocol_manifest_payload(
    cfg: MidogppSupportNelboRoutingConfig,
    contract_info: MidogppContractInfo,
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_support_nelbo_routing_protocol_manifest_v1",
        "experiment_name": cfg.name,
        "primary_method": cfg.primary_method,
        "domain_regime": cfg.domain_regime,
        "heldout_centers": list(cfg.heldout_centers),
        "eligible_domain_ids": list(contract_info.eligible_domain_ids),
        "ineligible_domain_ids": list(contract_info.ineligible_domain_ids),
        "support_labels_used": False,
        "routing_uses_eval_nelbo_for_adoption_rows": 0,
        "target_eval_used_for_scoring_only": True,
        "oracle_role": "diagnostic_only",
        "downstream_bacc_macro_f1_scope": "out_of_scope_v1",
    }


def _resolved_config_payload(cfg: MidogppSupportNelboRoutingConfig) -> dict[str, object]:
    return {
        "name": cfg.name,
        "artifact_root": str(cfg.artifact_root),
        "feature_cache_root": str(cfg.feature_cache_root),
        "dataset_contract_artifact_root": str(cfg.dataset_contract_artifact_root),
        "cache_report_path": str(cfg.cache_report_path),
        "source_expert_manifest_path": str(cfg.source_expert_manifest_path),
        "support_nelbo_scores_path": str(cfg.support_nelbo_scores_path),
        "eval_nelbo_matrix_path": str(cfg.eval_nelbo_matrix_path),
        "domain_regime": cfg.domain_regime,
        "strict_available_seed_domain_coverage": cfg.strict_available_seed_domain_coverage,
        "strict_full_run_matrix": cfg.strict_full_run_matrix,
        "experiment_seeds": list(cfg.experiment_seeds),
        "heldout_centers": list(cfg.heldout_centers),
        "support_size": cfg.support_size,
        "support_seeds": list(cfg.support_seeds),
        "nelbo_target": cfg.nelbo_target,
        "class_prior_source": cfg.class_prior_source,
        "weighting_policy": cfg.weighting_policy,
        "softmax_tau": cfg.softmax_tau,
        "top_k": cfg.top_k,
    }


def _write_decision_summary(root: Path) -> None:
    text = "\n".join(
        [
            "# MIDOG++ Support-NELBO Routing v1",
            "",
            "## Claim Boundary",
            "",
            "This surface evaluates routing-stage marginal NELBO utility only.",
            "Target support is unlabeled and disjoint from held-out target evaluation.",
            "All-expert eval NELBO and oracle rows are diagnostic-only and not adoption-eligible.",
            "Downstream BACC and macro-F1 are out of scope for this v1 surface.",
            "",
        ]
    )
    path = root / "reports" / "decision_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float_cell(row: Mapping[str, object], key: str) -> float:
    try:
        value = float(str(row[key]))
    except Exception as exc:
        raise ProtocolError(f"Missing or invalid numeric field {key!r}: {row}") from exc
    if not math.isfinite(value):
        raise ProtocolError(f"Field {key!r} must be finite.")
    return value
