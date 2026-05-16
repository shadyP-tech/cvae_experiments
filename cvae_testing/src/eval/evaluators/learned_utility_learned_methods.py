from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from src.eval.evaluators.learned_utility_config import PairwiseTournamentConfig
from src.eval.evaluators.learned_utility_models import (
    _LinearRegressor,
    _MLPRegressor,
    _PairwiseRanker,
)
from src.eval.evaluators.learned_utility_pairs import (
    _build_fold_training_pair_features,
    _build_pair_features,
    _build_pairwise_training_pairs,
    _normalize_targets_per_query,
    _zscore_features,
)
from src.eval.evaluators.learned_utility_pairprob import (
    ConformalCalibrationBlock,
    ConformalRegretSetSelection,
    GroupOOFHardpairBoostCalibrationBlock,
    GroupOOFHardpairBoostSelection,
    JackknifeCalibrationBlock,
    JackknifeLCBSelection,
    PairprobModelBundle,
    PairprobPolicySelection,
    Top2DeltaGateCalibrationBlock,
    Top2DeltaGateModelBundle,
    Top2DeltaGateSelection,
    Top2RerankCalibrationBlock,
    Top2RerankModelBundle,
    Top2RerankSelection,
    _gap_pct_for_selected,
    allpair_delta_gate_route_rows,
    build_group_oof_allpair_delta_gate_training_data,
    build_group_oof_top2_delta_gate_training_data,
    build_pairprob_training_data,
    build_group_oof_hardpair_observations,
    clone_direct_pairprob_adoption_rows,
    conformal_pairprob_route_rows,
    fit_pairprob_model,
    fit_allpair_delta_gate_model,
    fit_top2_delta_gate_model,
    hardpair_boost_route_rows,
    hardpair_weight_multipliers_from_observations,
    jackknife_pairprob_route_rows,
    pairprob_evidence_reason,
    pairprob_probability_matrix,
    pairprob_route_rows,
    pairprob_selected_indices,
    pairprob_win_scores,
    select_pairprob_policy,
    select_conformal_regret_set_policy,
    select_group_oof_hardpair_boost_policy,
    select_jackknife_lcb_policy,
    select_allpair_delta_gate_policy,
    select_top2_delta_gate_policy,
    select_top2_margin_reranker_policy,
    top2_delta_gate_route_rows,
    top2_rerank_route_rows,
)
from src.eval.evaluators.learned_utility_protocol import (
    FoldCandidateSet,
    ProtocolError,
    _method_protocol,
    _protocol_row_fields,
)
from src.eval.evaluators.learned_utility_selection import _selection_metrics
from src.eval.evaluators.learned_utility_tournament import (
    DeltaGatePolicySelection,
    TournamentPolicySelection,
    build_delta_gate_calibration_rows,
    delta_gate_route_rows,
    oracle_confidence_set_rows,
    select_delta_gate_policy,
    summarize_tournament_rows,
    tournament_route_rows,
)


@dataclass(frozen=True)
class LearnedFoldOutputs:
    sample_rows: List[Dict[str, Any]]
    pair_rows: List[Dict[str, Any]]
    pair_training_rows: List[Dict[str, Any]]


def _copy_pairprob_selection_with_reason(
    selection: PairprobPolicySelection,
    *,
    diagnostic_only_reason: str,
) -> PairprobPolicySelection:
    return PairprobPolicySelection(
        method=selection.method,
        feature_set=selection.feature_set,
        ridge_l2=selection.ridge_l2,
        selected_by_inner_validation=selection.selected_by_inner_validation,
        diagnostic_only_reason=str(diagnostic_only_reason or selection.diagnostic_only_reason),
        source_inner_validation_domains=selection.source_inner_validation_domains,
        source_inner_rows=selection.source_inner_rows,
        source_inner_mean_oracle_gap_pct=selection.source_inner_mean_oracle_gap_pct,
        source_inner_worst_domain_oracle_gap_pct=selection.source_inner_worst_domain_oracle_gap_pct,
        source_inner_relative_catastrophic_rate=selection.source_inner_relative_catastrophic_rate,
        source_inner_absolute_high_regret_rate=selection.source_inner_absolute_high_regret_rate,
        source_inner_top1=selection.source_inner_top1,
        source_inner_spearman=selection.source_inner_spearman,
        source_inner_std_oracle_gap_pct=selection.source_inner_std_oracle_gap_pct,
        source_inner_std_top1=selection.source_inner_std_top1,
        source_inner_max_minus_min_oracle_gap_pct=selection.source_inner_max_minus_min_oracle_gap_pct,
        pairwise_near_tie_drop_rate=selection.pairwise_near_tie_drop_rate,
        pairwise_train_pairs_after_filter=selection.pairwise_train_pairs_after_filter,
        pairwise_validation_pairs_after_filter=selection.pairwise_validation_pairs_after_filter,
        pairwise_train_domains_after_filter=selection.pairwise_train_domains_after_filter,
    )


def _infer_experts_per_sample(s_train: np.ndarray) -> int:
    counts = [
        int(np.sum(s_train == int(sample_index)))
        for sample_index in sorted(set(int(v) for v in s_train.tolist()))
    ]
    counts = [v for v in counts if v > 0]
    if not counts:
        raise ProtocolError("No pairwise training candidates remain")
    unique = sorted(set(counts))
    if len(unique) != 1:
        raise ProtocolError(f"Pairwise training candidate count is not constant per sample: {unique}")
    if int(unique[0]) < 1:
        raise ProtocolError("Pairwise training requires at least one candidate per sample")
    return int(unique[0])


def _pairwise_variant_features(
    *,
    x_train: np.ndarray,
    x_test: np.ndarray,
    q_train: np.ndarray,
    e_train: np.ndarray,
    q_test: np.ndarray,
    e_test: np.ndarray,
    sample_domains: np.ndarray,
    embedding_feature_dim: int,
    expert_feature_dim: int,
    run_ablations: bool,
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    span = max(float(np.max(sample_domains) - np.min(sample_domains)), 1.0)
    train_abs_diff = np.abs(q_train.astype(np.float64) - e_train.astype(np.float64)) / span
    test_abs_diff = np.abs(q_test.astype(np.float64) - e_test.astype(np.float64)) / span
    train_exact = (q_train == e_train).astype(np.float64)
    test_exact = (q_test == e_test).astype(np.float64)
    train_meta = np.stack([train_abs_diff, train_exact], axis=1)
    test_meta = np.stack([test_abs_diff, test_exact], axis=1)

    expert_oh_train = x_train[:, embedding_feature_dim : embedding_feature_dim + expert_feature_dim]
    expert_oh_test = x_test[:, embedding_feature_dim : embedding_feature_dim + expert_feature_dim]

    latent_train = np.concatenate([x_train[:, :embedding_feature_dim], expert_oh_train], axis=1)
    latent_test = np.concatenate([x_test[:, :embedding_feature_dim], expert_oh_test], axis=1)
    latent_train_z, latent_test_z = _zscore_features(latent_train, latent_test)

    metadata_train = np.concatenate([expert_oh_train, train_meta], axis=1)
    metadata_test = np.concatenate([expert_oh_test, test_meta], axis=1)
    metadata_train_z, metadata_test_z = _zscore_features(metadata_train, metadata_test)

    combined_train = np.concatenate([latent_train, train_meta], axis=1)
    combined_test = np.concatenate([latent_test, test_meta], axis=1)
    combined_train_z, combined_test_z = _zscore_features(combined_train, combined_test)

    if run_ablations:
        return [
            ("pairwise_ranker_metadata_only", metadata_train_z, metadata_test_z),
            ("pairwise_ranker_latent_only", latent_train_z, latent_test_z),
            ("pairwise_ranker_combined", combined_train_z, combined_test_z),
        ]
    return [("pairwise_ranker", combined_train_z, combined_test_z)]


def _fit_pairwise_variant_predictions(
    *,
    x_train: np.ndarray,
    x_test: np.ndarray,
    q_train: np.ndarray,
    e_train: np.ndarray,
    s_train: np.ndarray,
    q_test: np.ndarray,
    e_test: np.ndarray,
    sample_domains: np.ndarray,
    y_train: np.ndarray,
    pairwise_cfg: Dict[str, Any],
    seed: int,
    heldout_domain: int,
    embedding_feature_dim: int,
    expert_feature_dim: int,
) -> Dict[str, np.ndarray]:
    near_tie_delta = float(pairwise_cfg.get("near_tie_delta", 0.0))
    hard_pair_fraction = float(pairwise_cfg.get("hard_pair_fraction", 0.5))
    random_pair_fraction = float(pairwise_cfg.get("random_pair_fraction", 0.5))
    max_pairs_per_sample = int(pairwise_cfg.get("max_pairs_per_sample", 12))
    max_pairs_per_domain = int(pairwise_cfg.get("max_pairs_per_domain", 5000))
    run_ablations = bool(pairwise_cfg.get("run_ablations", True))
    experts_per_sample = _infer_experts_per_sample(s_train)

    train_pairs, _pair_diags = _build_pairwise_training_pairs(
        y_train=y_train,
        q_train=q_train,
        s_train=s_train,
        experts_per_sample=int(experts_per_sample),
        near_tie_delta=near_tie_delta,
        hard_pair_fraction=hard_pair_fraction,
        random_pair_fraction=random_pair_fraction,
        max_pairs_per_sample=max_pairs_per_sample,
        max_pairs_per_domain=max_pairs_per_domain,
        seed=int(seed) + int(heldout_domain),
    )
    if not train_pairs:
        return {}

    predictions: Dict[str, np.ndarray] = {}
    for method_name, x_tr_variant, x_te_variant in _pairwise_variant_features(
        x_train=x_train,
        x_test=x_test,
        q_train=q_train,
        e_train=e_train,
        q_test=q_test,
        e_test=e_test,
        sample_domains=sample_domains,
        embedding_feature_dim=embedding_feature_dim,
        expert_feature_dim=expert_feature_dim,
        run_ablations=run_ablations,
    ):
        ranker = _PairwiseRanker(
            seed=int(seed),
            hidden_dim=int(pairwise_cfg.get("hidden_dim", 128)),
            epochs=int(pairwise_cfg.get("epochs", 40)),
            lr=float(pairwise_cfg.get("lr", 1e-3)),
            batch_size=int(pairwise_cfg.get("batch_size", 2048)),
            margin=float(pairwise_cfg.get("margin", 1.0)),
            device=str(pairwise_cfg.get("device", "auto")),
        )
        ranker.fit(x_tr_variant, train_pairs)
        predictions[method_name] = ranker.predict(x_te_variant)
    return predictions


def _hard_gap_pct_for_score_matrix(
    *,
    fold: FoldCandidateSet,
    query_domains: np.ndarray,
    expert_domains: Sequence[int],
    score_matrix: np.ndarray,
    true_nelbo_matrix: np.ndarray,
    global_true_nelbo_matrix: np.ndarray,
    global_expert_domains: Sequence[int],
    tournament_cfg: PairwiseTournamentConfig,
    base_method: str,
) -> np.ndarray:
    rows = tournament_route_rows(
        method="pairwise_tournament_hard",
        fold=fold,
        query_domains=query_domains,
        expert_domains=expert_domains,
        score_matrix=score_matrix,
        true_nelbo_matrix=true_nelbo_matrix,
        global_true_nelbo_matrix=global_true_nelbo_matrix,
        global_expert_domains=global_expert_domains,
        policy_name=tournament_cfg.policy_name,
        base_method=str(base_method),
        threshold=0.0,
        topk=1,
        temperature=float(tournament_cfg.score_temperature),
        temperature_policy=tournament_cfg.temperature_policy,
        selected_by_inner_validation=False,
        threshold_selection_policy=tournament_cfg.calibration_policy,
    )
    return np.asarray([float(r["oracle_gap_pct"]) for r in rows], dtype=np.float64)


def _metadata_gap_pct_for_similarity(
    *,
    metadata_similarity_eval: np.ndarray | None,
    true_nelbo_matrix: np.ndarray,
    expert_domains: Sequence[int],
) -> np.ndarray:
    if metadata_similarity_eval is None:
        return np.full((true_nelbo_matrix.shape[0],), float("nan"), dtype=np.float64)
    sim = np.asarray(metadata_similarity_eval, dtype=np.float64)
    true = np.asarray(true_nelbo_matrix, dtype=np.float64)
    if sim.shape != true.shape:
        return np.full((true.shape[0],), float("nan"), dtype=np.float64)
    experts = np.asarray([int(v) for v in expert_domains], dtype=np.int64)
    selected = np.zeros((sim.shape[0],), dtype=np.int64)
    oracle = np.zeros((true.shape[0],), dtype=np.int64)
    tie = np.arange(true.shape[1], dtype=np.int64)
    for i in range(sim.shape[0]):
        selected[i] = int(np.lexsort((experts, -sim[i, :]))[0])
        oracle[i] = int(np.lexsort((tie, true[i, :]))[0])
    oracle_nelbo = true[np.arange(true.shape[0]), oracle]
    selected_nelbo = true[np.arange(true.shape[0]), selected]
    return ((selected_nelbo - oracle_nelbo) / np.maximum(np.abs(oracle_nelbo), 1e-12)) * 100.0


def _fit_pairprob_bundle_from_rows(
    *,
    x_rows: np.ndarray,
    q_rows: np.ndarray,
    e_rows: np.ndarray,
    s_rows: np.ndarray,
    y_rows: np.ndarray,
    feature_set: str,
    ridge_l2: float,
    tournament_cfg: PairwiseTournamentConfig,
    embedding_feature_dim: int,
    expert_feature_dim: int,
    device: str,
    pair_weight_multipliers: Dict[Tuple[int, int, int], float] | None = None,
) -> Tuple[PairprobModelBundle | None, Dict[str, float], str]:
    cfg = tournament_cfg.pairprob_tournament
    train_data = build_pairprob_training_data(
        x_rows=x_rows,
        q_rows=q_rows,
        e_rows=e_rows,
        s_rows=s_rows,
        y_rows=y_rows,
        embedding_dim=int(embedding_feature_dim),
        expert_feature_dim=int(expert_feature_dim),
        feature_set=str(feature_set),
        near_tie_delta_pct=float(cfg.near_tie_delta_pct),
        margin_weight_scale_pct=float(cfg.margin_weight_scale_pct),
        margin_weight_clip=cfg.margin_weight_clip,
        pair_weight_multipliers=pair_weight_multipliers,
    )
    reason = pairprob_evidence_reason(
        train_data=train_data,
        validation_data=None,
        validation_domains=max(1, len(train_data.kept_by_domain)),
        cfg=cfg,
    )
    total_pairs = max(int(train_data.total_pairs), 1)
    evidence = {
        "pairwise_near_tie_drop_rate": float(train_data.dropped_near_tie / total_pairs),
        "pairwise_train_pairs_after_filter": float(train_data.x.shape[0]),
        "pairwise_validation_pairs_after_filter": 0.0,
        "pairwise_train_domains_after_filter": float(len(train_data.kept_by_domain)),
        "diagnostic_only_reason": str(reason),
    }
    if train_data.x.shape[0] <= 0:
        return None, evidence, reason or "insufficient_pairwise_evidence"
    bundle = fit_pairprob_model(
        train_data=train_data,
        feature_set=str(feature_set),
        ridge_l2=float(ridge_l2),
        device=str(device),
    )
    return bundle, evidence, reason


def _fit_pairprob_jackknife_win_summary(
    *,
    x_rows: np.ndarray,
    q_rows: np.ndarray,
    e_rows: np.ndarray,
    s_rows: np.ndarray,
    y_rows: np.ndarray,
    x_eval: np.ndarray,
    eval_expert_domains: Sequence[int],
    source_domains: Sequence[int],
    feature_set: str,
    ridge_l2: float,
    tournament_cfg: PairwiseTournamentConfig,
    embedding_feature_dim: int,
    expert_feature_dim: int,
    device: str,
) -> Tuple[np.ndarray | None, np.ndarray | None, int, bool, str]:
    cfg = tournament_cfg.pairprob_tournament
    wins: List[np.ndarray] = []
    reasons: List[str] = []
    expected_shape: Tuple[int, int] | None = None
    for leaveout_domain in [int(v) for v in source_domains]:
        mask = np.asarray(q_rows, dtype=np.int64) != int(leaveout_domain)
        if int(np.sum(mask)) <= 0:
            reasons.append("source_inner_evidence_insufficient")
            continue
        train_data = build_pairprob_training_data(
            x_rows=x_rows[mask],
            q_rows=q_rows[mask],
            e_rows=e_rows[mask],
            s_rows=s_rows[mask],
            y_rows=y_rows[mask],
            embedding_dim=int(embedding_feature_dim),
            expert_feature_dim=int(expert_feature_dim),
            feature_set=str(feature_set),
            near_tie_delta_pct=float(cfg.near_tie_delta_pct),
            margin_weight_scale_pct=float(cfg.margin_weight_scale_pct),
            margin_weight_clip=cfg.margin_weight_clip,
        )
        reason = pairprob_evidence_reason(
            train_data=train_data,
            validation_data=None,
            validation_domains=max(1, len(train_data.kept_by_domain)),
            cfg=cfg,
        )
        if reason or train_data.x.shape[0] <= 0:
            reasons.append(reason or "source_inner_evidence_insufficient")
            continue
        bundle = fit_pairprob_model(
            train_data=train_data,
            feature_set=str(feature_set),
            ridge_l2=float(ridge_l2),
            device=str(device),
        )
        prob = pairprob_probability_matrix(
            bundle=bundle,
            x_rows=x_eval,
            expert_domains=eval_expert_domains,
            embedding_dim=int(embedding_feature_dim),
            expert_feature_dim=int(expert_feature_dim),
        )
        win = pairprob_win_scores(prob)
        if expected_shape is None:
            expected_shape = tuple(int(v) for v in win.shape)
        elif tuple(int(v) for v in win.shape) != expected_shape:
            reasons.append("candidate_pool_inconsistent")
            continue
        wins.append(win)
    if not wins:
        return None, None, 0, False, "source_inner_evidence_insufficient"
    stack = np.stack(wins, axis=0)
    consistent = expected_shape is not None and all(tuple(int(v) for v in win.shape) == expected_shape for win in wins)
    reason_out = "|".join(part for part in dict.fromkeys(reasons) if part)
    return (
        np.mean(stack, axis=0),
        np.std(stack, axis=0),
        int(stack.shape[0]),
        bool(consistent),
        str(reason_out),
    )


def _calibrate_pairprob_tournament(
    *,
    embeddings: np.ndarray,
    sample_domains: np.ndarray,
    true_nelbo: np.ndarray,
    expert_domains: Sequence[int],
    domain_to_idx: Dict[int, int],
    train_idx: np.ndarray,
    outer_heldout_domain: int,
    pairwise_cfg: Dict[str, Any],
    tournament_cfg: PairwiseTournamentConfig,
    include_metadata_features: bool,
    seed: int,
    embedding_feature_dim: int,
    expert_feature_dim: int,
    global_expert_domains: Sequence[int],
) -> Tuple[
    PairprobPolicySelection | None,
    PairprobPolicySelection | None,
    PairprobPolicySelection | None,
    PairprobPolicySelection | None,
    ConformalRegretSetSelection | None,
    JackknifeLCBSelection | None,
    Top2RerankSelection | None,
    Top2DeltaGateSelection | None,
    Top2DeltaGateSelection | None,
    GroupOOFHardpairBoostSelection | None,
]:
    cfg = tournament_cfg.pairprob_tournament
    source_domains = sorted(set(int(sample_domains[int(i)]) for i in np.asarray(train_idx, dtype=np.int64).tolist()))
    if len(source_domains) < int(cfg.min_source_inner_validation_domains):
        return None, None, None, None, None, None, None, None, None, None

    rows_by_key: Dict[Tuple[str, str, float], List[Dict[str, Any]]] = {}
    evidence_by_key: Dict[Tuple[str, str, float], Dict[str, float]] = {}
    validation_domains_by_key: Dict[Tuple[str, str, float], set[int]] = {}
    conformal_blocks_by_key: Dict[Tuple[str, str, float], List[ConformalCalibrationBlock]] = {}
    jackknife_blocks_by_key: Dict[Tuple[str, float], List[JackknifeCalibrationBlock]] = {}
    top2_blocks_by_key: Dict[Tuple[str, float], List[Top2RerankCalibrationBlock]] = {}
    top2_delta_blocks_by_key: Dict[Tuple[str, float], List[Top2DeltaGateCalibrationBlock]] = {}
    allpair_delta_blocks_by_key: Dict[Tuple[str, float], List[Top2DeltaGateCalibrationBlock]] = {}
    hardpair_boost_blocks_by_key: Dict[Tuple[str, float], List[GroupOOFHardpairBoostCalibrationBlock]] = {}
    feature_sets = [str(cfg.adoption_feature_set), *[str(v) for v in cfg.diagnostic_feature_sets]]
    device = str(pairwise_cfg.get("device", "auto"))

    for validation_domain in source_domains:
        train_idx_arr = np.asarray(train_idx, dtype=np.int64)
        inner_train_idx = train_idx_arr[sample_domains[train_idx_arr] != int(validation_domain)]
        validation_idx = train_idx_arr[sample_domains[train_idx_arr] == int(validation_domain)]
        if inner_train_idx.size == 0 or validation_idx.size == 0:
            continue
        try:
            x_inner, q_inner, e_inner, s_inner = _build_fold_training_pair_features(
                sample_embeddings=embeddings,
                sample_domains=sample_domains,
                train_indices=inner_train_idx,
                expert_domains=expert_domains,
                outer_heldout_domain=int(outer_heldout_domain),
                include_metadata_features=include_metadata_features,
                extra_excluded_domains=[int(validation_domain)],
            )
            validation_fold = FoldCandidateSet.for_heldout_domain(
                heldout_domain=int(outer_heldout_domain),
                expert_domains=expert_domains,
                excluded_domains=[int(validation_domain)],
            )
            x_val, q_val, e_val, s_val = _build_pair_features(
                sample_embeddings=embeddings,
                sample_domains=sample_domains,
                sample_indices=validation_idx,
                expert_domains=validation_fold.candidate_expert_domains,
                expert_id_domains=expert_domains,
                include_metadata_features=include_metadata_features,
            )
        except ProtocolError:
            continue

        y_inner = true_nelbo[s_inner, [domain_to_idx[int(ed)] for ed in e_inner]]
        y_val = true_nelbo[s_val, [domain_to_idx[int(ed)] for ed in e_val]]
        val_n = int(validation_idx.size)
        e_n = len(validation_fold.candidate_expert_domains)
        true_matrix = y_val.reshape(val_n, e_n)
        global_eval = true_nelbo[np.asarray(validation_idx, dtype=np.int64)]

        pred_by_method = _fit_pairwise_variant_predictions(
            x_train=x_inner,
            x_test=x_val,
            q_train=q_inner,
            e_train=e_inner,
            s_train=s_inner,
            q_test=q_val,
            e_test=e_val,
            sample_domains=sample_domains,
            y_train=y_inner,
            pairwise_cfg=pairwise_cfg,
            seed=int(seed) + int(validation_domain),
            heldout_domain=int(outer_heldout_domain),
            embedding_feature_dim=embedding_feature_dim,
            expert_feature_dim=expert_feature_dim,
        )
        hard_gap = np.full((val_n,), float("nan"), dtype=np.float64)
        hard_base = "pairwise_ranker_latent_only"
        if hard_base in pred_by_method:
            hard_gap = _hard_gap_pct_for_score_matrix(
                fold=validation_fold,
                query_domains=sample_domains[validation_idx],
                expert_domains=validation_fold.candidate_expert_domains,
                score_matrix=pred_by_method[hard_base].reshape(val_n, e_n),
                true_nelbo_matrix=true_matrix,
                global_true_nelbo_matrix=global_eval,
                global_expert_domains=global_expert_domains,
                tournament_cfg=tournament_cfg,
                base_method=hard_base,
            )

        for feature_set in feature_sets:
            train_data = build_pairprob_training_data(
                x_rows=x_inner,
                q_rows=q_inner,
                e_rows=e_inner,
                s_rows=s_inner,
                y_rows=y_inner,
                embedding_dim=int(embedding_feature_dim),
                expert_feature_dim=int(expert_feature_dim),
                feature_set=str(feature_set),
                near_tie_delta_pct=float(cfg.near_tie_delta_pct),
                margin_weight_scale_pct=float(cfg.margin_weight_scale_pct),
                margin_weight_clip=cfg.margin_weight_clip,
            )
            val_data = build_pairprob_training_data(
                x_rows=x_val,
                q_rows=q_val,
                e_rows=e_val,
                s_rows=s_val,
                y_rows=y_val,
                embedding_dim=int(embedding_feature_dim),
                expert_feature_dim=int(expert_feature_dim),
                feature_set=str(feature_set),
                near_tie_delta_pct=float(cfg.near_tie_delta_pct),
                margin_weight_scale_pct=float(cfg.margin_weight_scale_pct),
                margin_weight_clip=cfg.margin_weight_clip,
            )
            evidence_reason = pairprob_evidence_reason(
                train_data=train_data,
                validation_data=val_data,
                validation_domains=cfg.min_source_inner_validation_domains,
                cfg=cfg,
            )
            if train_data.x.shape[0] <= 0:
                continue
            for l2 in cfg.ridge_l2_values:
                bundle = fit_pairprob_model(
                    train_data=train_data,
                    feature_set=str(feature_set),
                    ridge_l2=float(l2),
                    device=device,
                )
                prob = pairprob_probability_matrix(
                    bundle=bundle,
                    x_rows=x_val,
                    expert_domains=validation_fold.candidate_expert_domains,
                    embedding_dim=int(embedding_feature_dim),
                    expert_feature_dim=int(expert_feature_dim),
                )
                if (
                    bool(cfg.top2_margin_reranker.enabled)
                    and str(feature_set) == str(cfg.top2_margin_reranker.base_feature_set)
                ):
                    top2_blocks_by_key.setdefault((str(feature_set), float(l2)), []).append(
                        Top2RerankCalibrationBlock(
                            validation_domain=int(validation_domain),
                            query_domains=np.asarray(sample_domains[validation_idx], dtype=np.int64),
                            expert_domains=tuple(int(v) for v in validation_fold.candidate_expert_domains),
                            x_rows=x_val,
                            prob_matrix=prob,
                            true_nelbo_matrix=true_matrix,
                            global_true_nelbo_matrix=global_eval,
                            fold=validation_fold,
                            pairprob_direct_gap_pct=_gap_pct_for_selected(
                                true_matrix,
                                pairprob_selected_indices(prob, validation_fold.candidate_expert_domains),
                            ),
                            )
                        )
                if (
                    bool(cfg.top2_delta_gate.enabled)
                    and str(feature_set) == str(cfg.top2_delta_gate.base_feature_set)
                ):
                    top2_delta_blocks_by_key.setdefault((str(feature_set), float(l2)), []).append(
                        Top2DeltaGateCalibrationBlock(
                            validation_domain=int(validation_domain),
                            train_x_rows=x_inner,
                            train_q_rows=q_inner,
                            train_e_rows=e_inner,
                            train_s_rows=s_inner,
                            train_y_rows=y_inner,
                            query_domains=np.asarray(sample_domains[validation_idx], dtype=np.int64),
                            expert_domains=tuple(int(v) for v in validation_fold.candidate_expert_domains),
                            x_rows=x_val,
                            direct_prob_matrix=prob,
                            true_nelbo_matrix=true_matrix,
                            global_true_nelbo_matrix=global_eval,
                            fold=validation_fold,
                            pairprob_direct_gap_pct=_gap_pct_for_selected(
                                true_matrix,
                                pairprob_selected_indices(prob, validation_fold.candidate_expert_domains),
                            ),
                        )
                    )
                if (
                    bool(cfg.allpair_delta_gate.enabled)
                    and str(feature_set) == str(cfg.allpair_delta_gate.base_feature_set)
                ):
                    allpair_delta_blocks_by_key.setdefault((str(feature_set), float(l2)), []).append(
                        Top2DeltaGateCalibrationBlock(
                            validation_domain=int(validation_domain),
                            train_x_rows=x_inner,
                            train_q_rows=q_inner,
                            train_e_rows=e_inner,
                            train_s_rows=s_inner,
                            train_y_rows=y_inner,
                            query_domains=np.asarray(sample_domains[validation_idx], dtype=np.int64),
                            expert_domains=tuple(int(v) for v in validation_fold.candidate_expert_domains),
                            x_rows=x_val,
                            direct_prob_matrix=prob,
                            true_nelbo_matrix=true_matrix,
                            global_true_nelbo_matrix=global_eval,
                            fold=validation_fold,
                            pairprob_direct_gap_pct=_gap_pct_for_selected(
                                true_matrix,
                                pairprob_selected_indices(prob, validation_fold.candidate_expert_domains),
                            ),
                        )
                    )
                if (
                    bool(cfg.group_oof_hardpair_boost.enabled)
                    and str(feature_set) == str(cfg.group_oof_hardpair_boost.feature_set)
                ):
                    hardpair_boost_blocks_by_key.setdefault((str(feature_set), float(l2)), []).append(
                        GroupOOFHardpairBoostCalibrationBlock(
                            validation_domain=int(validation_domain),
                            train_x_rows=x_inner,
                            train_q_rows=q_inner,
                            train_e_rows=e_inner,
                            train_s_rows=s_inner,
                            train_y_rows=y_inner,
                            query_domains=np.asarray(sample_domains[validation_idx], dtype=np.int64),
                            expert_domains=tuple(int(v) for v in validation_fold.candidate_expert_domains),
                            x_rows=x_val,
                            direct_prob_matrix=prob,
                            true_nelbo_matrix=true_matrix,
                            global_true_nelbo_matrix=global_eval,
                            fold=validation_fold,
                            pairprob_direct_gap_pct=_gap_pct_for_selected(
                                true_matrix,
                                pairprob_selected_indices(prob, validation_fold.candidate_expert_domains),
                            ),
                        )
                    )
                if (
                    bool(cfg.jackknife_lcb_tournament.enabled)
                    and str(feature_set) == str(cfg.adoption_feature_set)
                ):
                    inner_source_domains = sorted(
                        set(int(sample_domains[int(i)]) for i in np.asarray(inner_train_idx, dtype=np.int64).tolist())
                    )
                    mean_win, std_win, n_models, candidate_pool_consistent, _jk_reason = (
                        _fit_pairprob_jackknife_win_summary(
                            x_rows=x_inner,
                            q_rows=q_inner,
                            e_rows=e_inner,
                            s_rows=s_inner,
                            y_rows=y_inner,
                            x_eval=x_val,
                            eval_expert_domains=validation_fold.candidate_expert_domains,
                            source_domains=inner_source_domains,
                            feature_set=str(feature_set),
                            ridge_l2=float(l2),
                            tournament_cfg=tournament_cfg,
                            embedding_feature_dim=int(embedding_feature_dim),
                            expert_feature_dim=int(expert_feature_dim),
                            device=device,
                        )
                    )
                    if mean_win is not None and std_win is not None:
                        hard_win = pairprob_win_scores(prob)
                        jackknife_blocks_by_key.setdefault((str(feature_set), float(l2)), []).append(
                            JackknifeCalibrationBlock(
                                validation_domain=int(validation_domain),
                                query_domains=np.asarray(sample_domains[validation_idx], dtype=np.int64),
                                expert_domains=tuple(int(v) for v in validation_fold.candidate_expert_domains),
                                mean_win=mean_win,
                                std_win=std_win,
                                n_models=int(n_models),
                                candidate_pool_consistent=bool(candidate_pool_consistent),
                                true_nelbo_matrix=true_matrix,
                                global_true_nelbo_matrix=global_eval,
                                fold=validation_fold,
                                pairprob_hard_win=hard_win,
                                pairprob_hard_selected_idx=pairprob_selected_indices(
                                    prob,
                                    validation_fold.candidate_expert_domains,
                                ),
                                pairprob_hard_oracle_gap_pct=_gap_pct_for_selected(
                                    true_matrix,
                                    pairprob_selected_indices(prob, validation_fold.candidate_expert_domains),
                                ),
                            )
                        )
                method_names = (
                    [cfg.direct_method, cfg.group_robust_method]
                    if str(feature_set) == str(cfg.adoption_feature_set)
                    else [cfg.combined_diagnostic_method]
                )
                for method_name in method_names:
                    key = (str(method_name), str(feature_set), float(l2))
                    if (
                        bool(cfg.conformal_regret_set.enabled)
                        and str(method_name) == str(cfg.group_robust_method)
                        and str(feature_set) == str(cfg.conformal_regret_set.feature_set)
                    ):
                        conformal_blocks_by_key.setdefault(key, []).append(
                            ConformalCalibrationBlock(
                                validation_domain=int(validation_domain),
                                query_domains=np.asarray(sample_domains[validation_idx], dtype=np.int64),
                                expert_domains=tuple(int(v) for v in validation_fold.candidate_expert_domains),
                                prob_matrix=prob,
                                true_nelbo_matrix=true_matrix,
                                global_true_nelbo_matrix=global_eval,
                                fold=validation_fold,
                                scalar_hard_oracle_gap_pct=hard_gap,
                            )
                        )
                    total_pairs = max(int(train_data.total_pairs), 1)
                    ev = evidence_by_key.setdefault(
                        key,
                        {
                            "pairwise_near_tie_drop_rate_num": 0.0,
                            "pairwise_near_tie_drop_rate_den": 0.0,
                            "pairwise_train_pairs_after_filter": 0.0,
                            "pairwise_validation_pairs_after_filter": 0.0,
                            "pairwise_train_domains_after_filter": 0.0,
                            "diagnostic_only_reason": "",
                        },
                    )
                    ev["pairwise_near_tie_drop_rate_num"] += float(train_data.dropped_near_tie)
                    ev["pairwise_near_tie_drop_rate_den"] += float(total_pairs)
                    ev["pairwise_train_pairs_after_filter"] += float(train_data.x.shape[0])
                    ev["pairwise_validation_pairs_after_filter"] += float(val_data.x.shape[0])
                    ev["pairwise_train_domains_after_filter"] = max(
                        float(ev["pairwise_train_domains_after_filter"]),
                        float(len(train_data.kept_by_domain)),
                    )
                    if evidence_reason:
                        ev["diagnostic_only_reason"] = str(evidence_reason)
                    validation_domains_by_key.setdefault(key, set()).add(int(validation_domain))
                    selection = PairprobPolicySelection(
                        method=str(method_name),
                        feature_set=str(feature_set),
                        ridge_l2=float(l2),
                        selected_by_inner_validation=True,
                        diagnostic_only_reason=str(evidence_reason),
                    )
                    rows = pairprob_route_rows(
                        method=str(method_name),
                        fold=validation_fold,
                        query_domains=sample_domains[validation_idx],
                        expert_domains=validation_fold.candidate_expert_domains,
                        prob_matrix=prob,
                        true_nelbo_matrix=true_matrix,
                        global_true_nelbo_matrix=global_eval,
                        global_expert_domains=global_expert_domains,
                        policy_name=cfg.policy_name,
                        selection=selection,
                        hard_oracle_gap_pct=hard_gap,
                        diagnostic_only_reason=(
                            "diagnostic_only_combined_metadata_features"
                            if str(method_name) == str(cfg.combined_diagnostic_method)
                            else str(evidence_reason)
                        ),
                        absolute_high_regret_gap_pct=float(cfg.absolute_high_regret_gap_pct),
                        catastrophic_regression_vs_hard_gap_pct=float(
                            cfg.catastrophic_regression_vs_hard_gap_pct
                        ),
                    )
                    rows_by_key.setdefault(key, []).extend(rows)

    cleaned_evidence: Dict[Tuple[str, str, float], Dict[str, float]] = {}
    for key, ev in evidence_by_key.items():
        den = max(float(ev.pop("pairwise_near_tie_drop_rate_den", 0.0)), 1.0)
        num = float(ev.pop("pairwise_near_tie_drop_rate_num", 0.0))
        ev["pairwise_near_tie_drop_rate"] = float(num / den)
        if len(validation_domains_by_key.get(key, set())) < int(cfg.min_source_inner_validation_domains):
            ev["diagnostic_only_reason"] = "insufficient_pairwise_evidence"
        cleaned_evidence[key] = ev

    direct = select_pairprob_policy(
        rows_by_key=rows_by_key,
        method=cfg.direct_method,
        cfg=cfg,
        selection_mode="direct",
        evidence_by_key=cleaned_evidence,
    )
    direct_adoption = None
    if direct is not None and str(cfg.direct_adoption_method):
        direct_adoption = replace(
            direct,
            method=str(cfg.direct_adoption_method),
            diagnostic_only_reason="",
        )
    group = select_pairprob_policy(
        rows_by_key=rows_by_key,
        method=cfg.group_robust_method,
        cfg=cfg,
        selection_mode="group_robust",
        evidence_by_key=cleaned_evidence,
    )
    combined = select_pairprob_policy(
        rows_by_key=rows_by_key,
        method=cfg.combined_diagnostic_method,
        cfg=cfg,
        selection_mode="group_robust",
        evidence_by_key=cleaned_evidence,
    )
    if combined is not None:
        combined = _copy_pairprob_selection_with_reason(
            combined,
            diagnostic_only_reason="diagnostic_only_combined_metadata_features",
        )
    outer_candidate_experts = FoldCandidateSet.for_heldout_domain(
        heldout_domain=int(outer_heldout_domain),
        expert_domains=expert_domains,
    ).candidate_expert_domains
    conformal_selection = None
    if bool(cfg.conformal_regret_set.enabled):
        conformal_key = (
            str(cfg.group_robust_method),
            str(cfg.conformal_regret_set.feature_set),
            float(group.ridge_l2) if group is not None else float("nan"),
        )
        conformal_selection = select_conformal_regret_set_policy(
            blocks=conformal_blocks_by_key.get(conformal_key, []),
            base_selection=group,
            outer_candidate_experts=outer_candidate_experts,
            global_expert_domains=global_expert_domains,
            cfg=cfg.conformal_regret_set,
        )
    jackknife_selection = None
    if bool(cfg.jackknife_lcb_tournament.enabled):
        jackknife_key = (
            str(cfg.jackknife_lcb_tournament.adoption_feature_family),
            float(group.ridge_l2) if group is not None else float("nan"),
        )
        jackknife_selection = select_jackknife_lcb_policy(
            blocks=jackknife_blocks_by_key.get(jackknife_key, []),
            base_selection=group,
            global_expert_domains=global_expert_domains,
            cfg=cfg.jackknife_lcb_tournament,
        )
    top2_selection = None
    if bool(cfg.top2_margin_reranker.enabled):
        top2_key = (
            str(cfg.top2_margin_reranker.base_feature_set),
            float(direct.ridge_l2) if direct is not None else float("nan"),
        )
        top2_selection = select_top2_margin_reranker_policy(
            blocks=top2_blocks_by_key.get(top2_key, []),
            base_selection=direct,
            global_expert_domains=global_expert_domains,
            cfg=cfg.top2_margin_reranker,
            embedding_dim=int(embedding_feature_dim),
            expert_feature_dim=int(expert_feature_dim),
            device=device,
        )
    top2_delta_selection = None
    if bool(cfg.top2_delta_gate.enabled):
        top2_delta_key = (
            str(cfg.top2_delta_gate.base_feature_set),
            float(direct.ridge_l2) if direct is not None else float("nan"),
        )
        top2_delta_selection = select_top2_delta_gate_policy(
            blocks=top2_delta_blocks_by_key.get(top2_delta_key, []),
            base_selection=direct,
            global_expert_domains=global_expert_domains,
            pairprob_cfg=cfg,
            cfg=cfg.top2_delta_gate,
            embedding_dim=int(embedding_feature_dim),
            expert_feature_dim=int(expert_feature_dim),
            device=device,
        )
    allpair_delta_selection = None
    if bool(cfg.allpair_delta_gate.enabled):
        allpair_delta_key = (
            str(cfg.allpair_delta_gate.base_feature_set),
            float(direct.ridge_l2) if direct is not None else float("nan"),
        )
        allpair_delta_selection = select_allpair_delta_gate_policy(
            blocks=allpair_delta_blocks_by_key.get(allpair_delta_key, []),
            base_selection=direct,
            global_expert_domains=global_expert_domains,
            pairprob_cfg=cfg,
            cfg=cfg.allpair_delta_gate,
            embedding_dim=int(embedding_feature_dim),
            expert_feature_dim=int(expert_feature_dim),
            device=device,
        )
    hardpair_boost_selection = None
    if bool(cfg.group_oof_hardpair_boost.enabled):
        hardpair_key = (
            str(cfg.group_oof_hardpair_boost.feature_set),
            float(direct.ridge_l2) if direct is not None else float("nan"),
        )
        hardpair_boost_selection = select_group_oof_hardpair_boost_policy(
            blocks=hardpair_boost_blocks_by_key.get(hardpair_key, []),
            base_selection=direct,
            global_expert_domains=global_expert_domains,
            cfg=cfg.group_oof_hardpair_boost,
            embedding_dim=int(embedding_feature_dim),
            expert_feature_dim=int(expert_feature_dim),
            device=device,
            seed=int(seed),
        )
    return (
        direct,
        direct_adoption,
        group,
        combined,
        conformal_selection,
        jackknife_selection,
        top2_selection,
        top2_delta_selection,
        allpair_delta_selection,
        hardpair_boost_selection,
    )


def _calibrate_pairwise_tournament(
    *,
    embeddings: np.ndarray,
    sample_domains: np.ndarray,
    true_nelbo: np.ndarray,
    expert_domains: Sequence[int],
    domain_to_idx: Dict[int, int],
    train_idx: np.ndarray,
    outer_heldout_domain: int,
    pairwise_cfg: Dict[str, Any],
    tournament_cfg: PairwiseTournamentConfig,
    include_metadata_features: bool,
    seed: int,
    embedding_feature_dim: int,
    expert_feature_dim: int,
    global_expert_domains: Sequence[int],
) -> TournamentPolicySelection | None:
    source_domains = sorted(set(int(sample_domains[int(i)]) for i in np.asarray(train_idx, dtype=np.int64).tolist()))
    if len(source_domains) < 2:
        return None

    policy_rows: Dict[Tuple[str, float, int], List[Dict[str, Any]]] = {}
    for validation_domain in source_domains:
        train_idx_arr = np.asarray(train_idx, dtype=np.int64)
        inner_train_idx = train_idx_arr[sample_domains[train_idx_arr] != int(validation_domain)]
        validation_idx = train_idx_arr[sample_domains[train_idx_arr] == int(validation_domain)]
        if inner_train_idx.size == 0 or validation_idx.size == 0:
            continue

        try:
            x_inner, q_inner, e_inner, s_inner = _build_fold_training_pair_features(
                sample_embeddings=embeddings,
                sample_domains=sample_domains,
                train_indices=inner_train_idx,
                expert_domains=expert_domains,
                outer_heldout_domain=int(outer_heldout_domain),
                include_metadata_features=include_metadata_features,
                extra_excluded_domains=[int(validation_domain)],
            )
            validation_fold = FoldCandidateSet.for_heldout_domain(
                heldout_domain=int(outer_heldout_domain),
                expert_domains=expert_domains,
                excluded_domains=[int(validation_domain)],
            )
            x_val, q_val, e_val, s_val = _build_pair_features(
                sample_embeddings=embeddings,
                sample_domains=sample_domains,
                sample_indices=validation_idx,
                expert_domains=validation_fold.candidate_expert_domains,
                expert_id_domains=expert_domains,
                include_metadata_features=include_metadata_features,
            )
        except ProtocolError:
            continue

        y_inner = true_nelbo[s_inner, [domain_to_idx[int(ed)] for ed in e_inner]]
        y_val = true_nelbo[s_val, [domain_to_idx[int(ed)] for ed in e_val]]
        pred_by_method = _fit_pairwise_variant_predictions(
            x_train=x_inner,
            x_test=x_val,
            q_train=q_inner,
            e_train=e_inner,
            s_train=s_inner,
            q_test=q_val,
            e_test=e_val,
            sample_domains=sample_domains,
            y_train=y_inner,
            pairwise_cfg=pairwise_cfg,
            seed=int(seed) + int(validation_domain),
            heldout_domain=int(outer_heldout_domain),
            embedding_feature_dim=embedding_feature_dim,
            expert_feature_dim=expert_feature_dim,
        )
        if not pred_by_method:
            continue

        val_n = int(validation_idx.size)
        e_n = len(validation_fold.candidate_expert_domains)
        true_matrix = y_val.reshape(val_n, e_n)
        global_eval = true_nelbo[np.asarray(validation_idx, dtype=np.int64)]
        for base_method in tournament_cfg.base_methods:
            if str(base_method) not in pred_by_method:
                continue
            score_matrix = pred_by_method[str(base_method)].reshape(val_n, e_n)
            for threshold in tournament_cfg.margin_thresholds:
                for topk in tournament_cfg.sparse_mix_topk_values:
                    rows = tournament_route_rows(
                        method="pairwise_tournament_inner_selected",
                        fold=validation_fold,
                        query_domains=sample_domains[validation_idx],
                        expert_domains=validation_fold.candidate_expert_domains,
                        score_matrix=score_matrix,
                        true_nelbo_matrix=true_matrix,
                        global_true_nelbo_matrix=global_eval,
                        global_expert_domains=global_expert_domains,
                        policy_name=tournament_cfg.policy_name,
                        base_method=str(base_method),
                        threshold=float(threshold),
                        topk=int(topk),
                        temperature=float(tournament_cfg.score_temperature),
                        temperature_policy=tournament_cfg.temperature_policy,
                        selected_by_inner_validation=True,
                        threshold_selection_policy=tournament_cfg.calibration_policy,
                    )
                    policy_rows.setdefault((str(base_method), float(threshold), int(topk)), []).extend(rows)

    if not policy_rows:
        return None

    candidates: List[Tuple[Tuple[float, float, float, float, int], Tuple[str, float, int], Dict[str, float]]] = []
    for key, rows in policy_rows.items():
        summary = summarize_tournament_rows(rows)
        if int(summary["n_rows"]) <= 0:
            continue
        score = (
            -float(summary["mean_oracle_gap_pct"]),
            -float(summary["high_regret_selection_rate"]),
            float(summary["oracle_in_route_set"]),
            float(summary["top1_oracle_hit"]),
            -int(key[2]),
        )
        candidates.append((score, key, summary))
    if not candidates:
        return None

    _score, (base_method, threshold, topk), summary = sorted(candidates, key=lambda item: item[0], reverse=True)[0]
    diagnostic_reason = (
        "diagnostic_only_high_fallback_rate"
        if float(summary["sparse_mix_active"]) > float(tournament_cfg.max_sparse_mix_activation_rate)
        else ""
    )
    return TournamentPolicySelection(
        base_method=str(base_method),
        threshold=float(threshold),
        topk=int(topk),
        selected_by_inner_validation=True,
        diagnostic_only_reason=diagnostic_reason,
        source_inner_rows=int(summary["n_rows"]),
        source_inner_gap_pct=float(summary["mean_oracle_gap_pct"]),
        source_inner_high_regret_rate=float(summary["high_regret_selection_rate"]),
        source_inner_oracle_in_route_set=float(summary["oracle_in_route_set"]),
        source_inner_top1=float(summary["top1_oracle_hit"]),
        source_inner_sparse_mix_rate=float(summary["sparse_mix_active"]),
    )


def _copy_delta_selection_with_reason(
    selection: DeltaGatePolicySelection,
    *,
    diagnostic_only_reason: str,
) -> DeltaGatePolicySelection:
    return DeltaGatePolicySelection(
        base_method=selection.base_method,
        feature_set=selection.feature_set,
        threshold=selection.threshold,
        topk=selection.topk,
        selected_by_inner_validation=selection.selected_by_inner_validation,
        selection_status=selection.selection_status,
        diagnostic_only_reason=str(diagnostic_only_reason or selection.diagnostic_only_reason),
        source_inner_rows=selection.source_inner_rows,
        source_inner_validation_domains=selection.source_inner_validation_domains,
        source_inner_active_rows=selection.source_inner_active_rows,
        source_inner_active_domains=selection.source_inner_active_domains,
        source_inner_gap_pct=selection.source_inner_gap_pct,
        source_inner_paired_gap_reduction_vs_hard=selection.source_inner_paired_gap_reduction_vs_hard,
        source_inner_high_regret_rate=selection.source_inner_high_regret_rate,
        source_inner_paired_high_regret_reduction_vs_hard=selection.source_inner_paired_high_regret_reduction_vs_hard,
        source_inner_activation_rate=selection.source_inner_activation_rate,
        source_inner_help_rate_active_only=selection.source_inner_help_rate_active_only,
        source_inner_harm_rate_active_only=selection.source_inner_harm_rate_active_only,
        source_inner_help_rate_all_rows=selection.source_inner_help_rate_all_rows,
        source_inner_harm_rate_all_rows=selection.source_inner_harm_rate_all_rows,
        source_inner_mean_delta_pct_when_active=selection.source_inner_mean_delta_pct_when_active,
        source_inner_median_delta_pct_when_active=selection.source_inner_median_delta_pct_when_active,
        source_inner_spearman_pred_vs_true_delta=selection.source_inner_spearman_pred_vs_true_delta,
        source_inner_auc_help_vs_harm=selection.source_inner_auc_help_vs_harm,
        model=selection.model,
    )


def _delta_noop_selection(
    *,
    base_method: str,
    feature_set: str,
    topk: int,
    reason: str,
) -> DeltaGatePolicySelection:
    return DeltaGatePolicySelection(
        base_method=str(base_method),
        feature_set=str(feature_set),
        threshold=float("nan"),
        topk=int(topk),
        selected_by_inner_validation=False,
        selection_status="insufficient_evidence_noop",
        diagnostic_only_reason=str(reason),
    )


def _calibrate_delta_gate_tournament(
    *,
    embeddings: np.ndarray,
    sample_domains: np.ndarray,
    true_nelbo: np.ndarray,
    expert_domains: Sequence[int],
    domain_to_idx: Dict[int, int],
    train_idx: np.ndarray,
    outer_heldout_domain: int,
    pairwise_cfg: Dict[str, Any],
    tournament_cfg: PairwiseTournamentConfig,
    include_metadata_features: bool,
    seed: int,
    embedding_feature_dim: int,
    expert_feature_dim: int,
    global_expert_domains: Sequence[int],
) -> Tuple[DeltaGatePolicySelection, DeltaGatePolicySelection | None]:
    gate_cfg = tournament_cfg.fallback_benefit_gate
    source_domains = sorted(set(int(sample_domains[int(i)]) for i in np.asarray(train_idx, dtype=np.int64).tolist()))
    topk = int(tournament_cfg.sparse_mix_topk_values[0])
    adoption_base = str(tournament_cfg.base_methods[0]) if tournament_cfg.base_methods else "pairwise_ranker_latent_only"
    diagnostic_base = (
        str(tournament_cfg.diagnostic_base_methods[0])
        if tournament_cfg.diagnostic_base_methods
        else "pairwise_ranker_combined"
    )
    if len(source_domains) < 2:
        adoption = _delta_noop_selection(
            base_method=adoption_base,
            feature_set=gate_cfg.feature_set,
            topk=topk,
            reason="insufficient_validation_domains",
        )
        diagnostic = _delta_noop_selection(
            base_method=diagnostic_base,
            feature_set=gate_cfg.diagnostic_feature_sets[0] if gate_cfg.diagnostic_feature_sets else gate_cfg.feature_set,
            topk=topk,
            reason="insufficient_validation_domains",
        ) if tournament_cfg.diagnostic_base_methods else None
        return adoption, diagnostic

    rows_by_key: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    for validation_domain in source_domains:
        train_idx_arr = np.asarray(train_idx, dtype=np.int64)
        inner_train_idx = train_idx_arr[sample_domains[train_idx_arr] != int(validation_domain)]
        validation_idx = train_idx_arr[sample_domains[train_idx_arr] == int(validation_domain)]
        if inner_train_idx.size == 0 or validation_idx.size == 0:
            continue

        try:
            x_inner, q_inner, e_inner, s_inner = _build_fold_training_pair_features(
                sample_embeddings=embeddings,
                sample_domains=sample_domains,
                train_indices=inner_train_idx,
                expert_domains=expert_domains,
                outer_heldout_domain=int(outer_heldout_domain),
                include_metadata_features=include_metadata_features,
                extra_excluded_domains=[int(validation_domain)],
            )
            validation_fold = FoldCandidateSet.for_heldout_domain(
                heldout_domain=int(outer_heldout_domain),
                expert_domains=expert_domains,
                excluded_domains=[int(validation_domain)],
            )
            x_val, q_val, e_val, s_val = _build_pair_features(
                sample_embeddings=embeddings,
                sample_domains=sample_domains,
                sample_indices=validation_idx,
                expert_domains=validation_fold.candidate_expert_domains,
                expert_id_domains=expert_domains,
                include_metadata_features=include_metadata_features,
            )
        except ProtocolError:
            continue

        y_inner = true_nelbo[s_inner, [domain_to_idx[int(ed)] for ed in e_inner]]
        y_val = true_nelbo[s_val, [domain_to_idx[int(ed)] for ed in e_val]]
        pred_by_method = _fit_pairwise_variant_predictions(
            x_train=x_inner,
            x_test=x_val,
            q_train=q_inner,
            e_train=e_inner,
            s_train=s_inner,
            q_test=q_val,
            e_test=e_val,
            sample_domains=sample_domains,
            y_train=y_inner,
            pairwise_cfg=pairwise_cfg,
            seed=int(seed) + int(validation_domain),
            heldout_domain=int(outer_heldout_domain),
            embedding_feature_dim=embedding_feature_dim,
            expert_feature_dim=expert_feature_dim,
        )
        if not pred_by_method:
            continue

        val_n = int(validation_idx.size)
        e_n = len(validation_fold.candidate_expert_domains)
        true_matrix = y_val.reshape(val_n, e_n)
        latent_matrix = (
            pred_by_method["pairwise_ranker_latent_only"].reshape(val_n, e_n)
            if "pairwise_ranker_latent_only" in pred_by_method
            else None
        )
        combined_matrix = (
            pred_by_method["pairwise_ranker_combined"].reshape(val_n, e_n)
            if "pairwise_ranker_combined" in pred_by_method
            else None
        )

        for base_method in tournament_cfg.base_methods:
            if str(base_method) not in pred_by_method:
                continue
            score_matrix = pred_by_method[str(base_method)].reshape(val_n, e_n)
            rows = build_delta_gate_calibration_rows(
                validation_domain=int(validation_domain),
                query_domains=sample_domains[validation_idx],
                expert_domains=validation_fold.candidate_expert_domains,
                score_matrix=score_matrix,
                true_nelbo_matrix=true_matrix,
                feature_set=gate_cfg.feature_set,
                base_method=str(base_method),
                topk=topk,
                temperature=float(tournament_cfg.score_temperature),
                gate_cfg=gate_cfg,
            )
            rows_by_key.setdefault((str(base_method), gate_cfg.feature_set, topk), []).extend(rows)

        for base_method in tournament_cfg.diagnostic_base_methods:
            if str(base_method) not in pred_by_method:
                continue
            for feature_set in gate_cfg.diagnostic_feature_sets:
                score_matrix = pred_by_method[str(base_method)].reshape(val_n, e_n)
                try:
                    rows = build_delta_gate_calibration_rows(
                        validation_domain=int(validation_domain),
                        query_domains=sample_domains[validation_idx],
                        expert_domains=validation_fold.candidate_expert_domains,
                        score_matrix=score_matrix,
                        true_nelbo_matrix=true_matrix,
                        feature_set=str(feature_set),
                        base_method=str(base_method),
                        topk=topk,
                        temperature=float(tournament_cfg.score_temperature),
                        gate_cfg=gate_cfg,
                        latent_score_matrix=latent_matrix,
                        combined_score_matrix=combined_matrix,
                    )
                except ProtocolError:
                    continue
                rows_by_key.setdefault((str(base_method), str(feature_set), topk), []).extend(rows)

    adoption_rows = {
        key: rows
        for key, rows in rows_by_key.items()
        if key[0] in set(str(v) for v in tournament_cfg.base_methods) and key[1] == gate_cfg.feature_set
    }
    adoption = select_delta_gate_policy(rows_by_key=adoption_rows, gate_cfg=gate_cfg)
    if adoption is None:
        adoption = _delta_noop_selection(
            base_method=adoption_base,
            feature_set=gate_cfg.feature_set,
            topk=topk,
            reason="insufficient_validation_domains",
        )

    diagnostic: DeltaGatePolicySelection | None = None
    diagnostic_rows = {
        key: rows
        for key, rows in rows_by_key.items()
        if key[0] in set(str(v) for v in tournament_cfg.diagnostic_base_methods)
        and key[1] in set(str(v) for v in gate_cfg.diagnostic_feature_sets)
    }
    if diagnostic_rows:
        diagnostic = select_delta_gate_policy(rows_by_key=diagnostic_rows, gate_cfg=gate_cfg)
        if diagnostic is not None:
            diagnostic = _copy_delta_selection_with_reason(
                diagnostic,
                diagnostic_only_reason=diagnostic.diagnostic_only_reason or "diagnostic_only_combined_metadata_features",
            )
    elif tournament_cfg.diagnostic_base_methods:
        diagnostic = _delta_noop_selection(
            base_method=diagnostic_base,
            feature_set=gate_cfg.diagnostic_feature_sets[0] if gate_cfg.diagnostic_feature_sets else gate_cfg.feature_set,
            topk=topk,
            reason="insufficient_validation_domains",
        )

    return adoption, diagnostic


def _run_pairprob_tournament_for_fold(
    *,
    x_train: np.ndarray,
    q_train: np.ndarray,
    e_train: np.ndarray,
    s_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    query_domains: np.ndarray,
    fold: FoldCandidateSet,
    true_matrix: np.ndarray,
    global_eval: np.ndarray,
    expert_domains: Sequence[int],
    global_expert_domains: Sequence[int],
    selections: Sequence[PairprobPolicySelection | None],
    conformal_selection: ConformalRegretSetSelection | None,
    jackknife_selection: JackknifeLCBSelection | None,
    top2_selection: Top2RerankSelection | None,
    top2_delta_selection: Top2DeltaGateSelection | None,
    allpair_delta_selection: Top2DeltaGateSelection | None,
    hardpair_boost_selection: GroupOOFHardpairBoostSelection | None,
    tournament_cfg: PairwiseTournamentConfig,
    pairwise_cfg: Dict[str, Any],
    embedding_feature_dim: int,
    expert_feature_dim: int,
    hard_oracle_gap_pct: np.ndarray | None,
    metadata_oracle_gap_pct: np.ndarray | None,
) -> List[Dict[str, Any]]:
    cfg = tournament_cfg.pairprob_tournament
    rows: List[Dict[str, Any]] = []
    device = str(pairwise_cfg.get("device", "auto"))
    direct_diagnostic_rows: List[Dict[str, Any]] = []
    direct_prob: np.ndarray | None = None
    direct_selection_for_eval: PairprobPolicySelection | None = None
    for selection in selections:
        if selection is None:
            continue
        if str(selection.method) == str(cfg.direct_adoption_method):
            if direct_diagnostic_rows:
                rows.extend(
                    clone_direct_pairprob_adoption_rows(
                        direct_diagnostic_rows,
                        adoption_method=str(cfg.direct_adoption_method),
                    )
                )
            continue
        bundle, evidence, final_reason = _fit_pairprob_bundle_from_rows(
            x_rows=x_train,
            q_rows=q_train,
            e_rows=e_train,
            s_rows=s_train,
            y_rows=y_train,
            feature_set=str(selection.feature_set),
            ridge_l2=float(selection.ridge_l2),
            tournament_cfg=tournament_cfg,
            embedding_feature_dim=int(embedding_feature_dim),
            expert_feature_dim=int(expert_feature_dim),
            device=device,
        )
        if bundle is None:
            continue
        reason_parts = [
            str(selection.diagnostic_only_reason),
            str(final_reason),
        ]
        if str(selection.method) == str(cfg.combined_diagnostic_method):
            reason_parts.append("diagnostic_only_combined_metadata_features")
        reason = "|".join(part for part in dict.fromkeys(reason_parts) if part)
        selection_for_eval = PairprobPolicySelection(
            method=selection.method,
            feature_set=selection.feature_set,
            ridge_l2=selection.ridge_l2,
            selected_by_inner_validation=selection.selected_by_inner_validation,
            diagnostic_only_reason=str(reason),
            source_inner_validation_domains=selection.source_inner_validation_domains,
            source_inner_rows=selection.source_inner_rows,
            source_inner_mean_oracle_gap_pct=selection.source_inner_mean_oracle_gap_pct,
            source_inner_worst_domain_oracle_gap_pct=selection.source_inner_worst_domain_oracle_gap_pct,
            source_inner_relative_catastrophic_rate=selection.source_inner_relative_catastrophic_rate,
            source_inner_absolute_high_regret_rate=selection.source_inner_absolute_high_regret_rate,
            source_inner_top1=selection.source_inner_top1,
            source_inner_spearman=selection.source_inner_spearman,
            source_inner_std_oracle_gap_pct=selection.source_inner_std_oracle_gap_pct,
            source_inner_std_top1=selection.source_inner_std_top1,
            source_inner_max_minus_min_oracle_gap_pct=selection.source_inner_max_minus_min_oracle_gap_pct,
            pairwise_near_tie_drop_rate=float(evidence.get("pairwise_near_tie_drop_rate", selection.pairwise_near_tie_drop_rate)),
            pairwise_train_pairs_after_filter=int(evidence.get("pairwise_train_pairs_after_filter", selection.pairwise_train_pairs_after_filter)),
            pairwise_validation_pairs_after_filter=int(selection.pairwise_validation_pairs_after_filter),
            pairwise_train_domains_after_filter=int(evidence.get("pairwise_train_domains_after_filter", selection.pairwise_train_domains_after_filter)),
        )
        prob = pairprob_probability_matrix(
            bundle=bundle,
            x_rows=x_test,
            expert_domains=fold.candidate_expert_domains,
            embedding_dim=int(embedding_feature_dim),
            expert_feature_dim=int(expert_feature_dim),
        )
        selection_rows = pairprob_route_rows(
            method=str(selection.method),
            fold=fold,
            query_domains=query_domains,
            expert_domains=fold.candidate_expert_domains,
            prob_matrix=prob,
            true_nelbo_matrix=true_matrix,
            global_true_nelbo_matrix=global_eval,
            global_expert_domains=global_expert_domains,
            policy_name=cfg.policy_name,
            selection=selection_for_eval,
            hard_oracle_gap_pct=hard_oracle_gap_pct,
            diagnostic_only_reason=str(reason),
            absolute_high_regret_gap_pct=float(cfg.absolute_high_regret_gap_pct),
            catastrophic_regression_vs_hard_gap_pct=float(cfg.catastrophic_regression_vs_hard_gap_pct),
        )
        rows.extend(selection_rows)
        if str(selection.method) == str(cfg.direct_method):
            direct_diagnostic_rows = [dict(row) for row in selection_rows]
            direct_prob = prob
            direct_selection_for_eval = selection_for_eval
        if (
            jackknife_selection is not None
            and bool(cfg.jackknife_lcb_tournament.enabled)
            and str(selection.method) == str(cfg.group_robust_method)
            and str(selection.feature_set) == str(cfg.jackknife_lcb_tournament.adoption_feature_family)
            and abs(float(selection.ridge_l2) - float(jackknife_selection.ridge_l2)) < 1e-18
        ):
            source_domains = sorted(set(int(v) for v in np.asarray(q_train, dtype=np.int64).tolist()))
            mean_win, std_win, n_models, candidate_pool_consistent, jackknife_reason = (
                _fit_pairprob_jackknife_win_summary(
                    x_rows=x_train,
                    q_rows=q_train,
                    e_rows=e_train,
                    s_rows=s_train,
                    y_rows=y_train,
                    x_eval=x_test,
                    eval_expert_domains=fold.candidate_expert_domains,
                    source_domains=source_domains,
                    feature_set=str(selection.feature_set),
                    ridge_l2=float(selection.ridge_l2),
                    tournament_cfg=tournament_cfg,
                    embedding_feature_dim=int(embedding_feature_dim),
                    expert_feature_dim=int(expert_feature_dim),
                    device=device,
                )
            )
            if mean_win is not None and std_win is not None:
                pairprob_gap = np.asarray([float(r["oracle_gap_pct"]) for r in selection_rows], dtype=np.float64)
                pairprob_hard_win = pairprob_win_scores(prob)
                pairprob_hard_selected_idx = pairprob_selected_indices(prob, fold.candidate_expert_domains)
                jackknife_reason_all = "|".join(
                    part
                    for part in dict.fromkeys(
                        [
                            str(jackknife_selection.diagnostic_only_reason),
                            str(jackknife_reason),
                            "" if bool(candidate_pool_consistent) else "candidate_pool_inconsistent",
                        ]
                    )
                    if part
                )
                jackknife_selection_for_eval = replace(
                    jackknife_selection,
                    diagnostic_only_reason=str(jackknife_reason_all),
                    candidate_pool_consistent=bool(
                        jackknife_selection.candidate_pool_consistent and candidate_pool_consistent
                    ),
                )
                rows.extend(
                    jackknife_pairprob_route_rows(
                        method=str(cfg.jackknife_lcb_tournament.mean_method_name),
                        fold=fold,
                        query_domains=query_domains,
                        expert_domains=fold.candidate_expert_domains,
                        mean_win=mean_win,
                        std_win=std_win,
                        n_models=int(n_models),
                        candidate_pool_consistent=bool(candidate_pool_consistent),
                        true_nelbo_matrix=true_matrix,
                        global_true_nelbo_matrix=global_eval,
                        global_expert_domains=global_expert_domains,
                        policy_name=cfg.jackknife_lcb_tournament.method_name,
                        selection=jackknife_selection_for_eval,
                        pairprob_hard_win=pairprob_hard_win,
                        pairprob_hard_selected_idx=pairprob_hard_selected_idx,
                        pairprob_hard_oracle_gap_pct=pairprob_gap,
                        metadata_oracle_gap_pct=metadata_oracle_gap_pct,
                        cfg=cfg.jackknife_lcb_tournament,
                        force_lambda=0.0,
                    )
                )
                rows.extend(
                    jackknife_pairprob_route_rows(
                        method=str(cfg.jackknife_lcb_tournament.method_name),
                        fold=fold,
                        query_domains=query_domains,
                        expert_domains=fold.candidate_expert_domains,
                        mean_win=mean_win,
                        std_win=std_win,
                        n_models=int(n_models),
                        candidate_pool_consistent=bool(candidate_pool_consistent),
                        true_nelbo_matrix=true_matrix,
                        global_true_nelbo_matrix=global_eval,
                        global_expert_domains=global_expert_domains,
                        policy_name=cfg.jackknife_lcb_tournament.method_name,
                        selection=jackknife_selection_for_eval,
                        pairprob_hard_win=pairprob_hard_win,
                        pairprob_hard_selected_idx=pairprob_hard_selected_idx,
                        pairprob_hard_oracle_gap_pct=pairprob_gap,
                        metadata_oracle_gap_pct=metadata_oracle_gap_pct,
                        cfg=cfg.jackknife_lcb_tournament,
                    )
                )
        if (
            conformal_selection is not None
            and bool(cfg.conformal_regret_set.enabled)
            and str(selection.method) == str(cfg.group_robust_method)
            and str(selection.feature_set) == str(cfg.conformal_regret_set.feature_set)
            and abs(float(selection.ridge_l2) - float(conformal_selection.ridge_l2)) < 1e-18
        ):
            conformal_reason = "|".join(
                part
                for part in dict.fromkeys([str(conformal_selection.diagnostic_only_reason), str(reason)])
                if part
            )
            conformal_selection_for_eval = replace(
                conformal_selection,
                diagnostic_only_reason=str(conformal_reason),
            )
            pairprob_gap = np.asarray([float(r["oracle_gap_pct"]) for r in selection_rows], dtype=np.float64)
            rows.extend(
                conformal_pairprob_route_rows(
                    method=str(cfg.conformal_regret_set.topwin_diagnostic_method),
                    fold=fold,
                    query_domains=query_domains,
                    expert_domains=fold.candidate_expert_domains,
                    prob_matrix=prob,
                    true_nelbo_matrix=true_matrix,
                    global_true_nelbo_matrix=global_eval,
                    global_expert_domains=global_expert_domains,
                    policy_name=cfg.conformal_regret_set.method_name,
                    selection=conformal_selection_for_eval,
                    cfg=cfg.conformal_regret_set,
                    pairprob_baseline_gap_pct=pairprob_gap,
                    scalar_hard_oracle_gap_pct=hard_oracle_gap_pct,
                    metadata_oracle_gap_pct=metadata_oracle_gap_pct,
                    topwin_diagnostic=True,
                )
            )
            rows.extend(
                conformal_pairprob_route_rows(
                    method=str(cfg.conformal_regret_set.method_name),
                    fold=fold,
                    query_domains=query_domains,
                    expert_domains=fold.candidate_expert_domains,
                    prob_matrix=prob,
                    true_nelbo_matrix=true_matrix,
                    global_true_nelbo_matrix=global_eval,
                    global_expert_domains=global_expert_domains,
                    policy_name=cfg.conformal_regret_set.method_name,
                    selection=conformal_selection_for_eval,
                    cfg=cfg.conformal_regret_set,
                    pairprob_baseline_gap_pct=pairprob_gap,
                    scalar_hard_oracle_gap_pct=hard_oracle_gap_pct,
                    metadata_oracle_gap_pct=metadata_oracle_gap_pct,
                )
            )
            rows.extend(
                conformal_pairprob_route_rows(
                    method=str(cfg.conformal_regret_set.oracle_diagnostic_method),
                    fold=fold,
                    query_domains=query_domains,
                    expert_domains=fold.candidate_expert_domains,
                    prob_matrix=prob,
                    true_nelbo_matrix=true_matrix,
                    global_true_nelbo_matrix=global_eval,
                    global_expert_domains=global_expert_domains,
                    policy_name=cfg.conformal_regret_set.method_name,
                    selection=conformal_selection_for_eval,
                    cfg=cfg.conformal_regret_set,
                    pairprob_baseline_gap_pct=pairprob_gap,
                    scalar_hard_oracle_gap_pct=hard_oracle_gap_pct,
                    metadata_oracle_gap_pct=metadata_oracle_gap_pct,
                    oracle_diagnostic=True,
                )
            )
    if (
        top2_selection is not None
        and bool(cfg.top2_margin_reranker.enabled)
        and direct_prob is not None
        and direct_selection_for_eval is not None
    ):
        pairprob_gap = np.asarray([float(r["oracle_gap_pct"]) for r in direct_diagnostic_rows], dtype=np.float64)
        top2_reason_parts = [str(top2_selection.diagnostic_only_reason)]
        reranker_bundle: Top2RerankModelBundle | None = top2_selection.model
        top2_selection_for_eval = top2_selection
        if not bool(top2_selection.noop) and reranker_bundle is None:
            top2_reason_parts.append("insufficient_source_inner_rerank_rows")
        top2_reason = "|".join(part for part in dict.fromkeys(top2_reason_parts) if part)
        if top2_reason:
            top2_selection_for_eval = replace(
                top2_selection,
                diagnostic_only_reason=str(top2_reason),
                noop=True,
                guard_status="failed_guards_noop",
                selection_stability_status=(
                    top2_selection.selection_stability_status
                    if top2_selection.selection_stability_status == "unstable"
                    else "forced_direct_pairprob"
                ),
            )
            reranker_bundle = None
        rows.extend(
            top2_rerank_route_rows(
                method=str(cfg.top2_margin_reranker.method_name),
                fold=fold,
                query_domains=query_domains,
                expert_domains=fold.candidate_expert_domains,
                x_rows=x_test,
                prob_matrix=direct_prob,
                true_nelbo_matrix=true_matrix,
                global_true_nelbo_matrix=global_eval,
                global_expert_domains=global_expert_domains,
                policy_name=cfg.top2_margin_reranker.method_name,
                selection=top2_selection_for_eval,
                reranker_bundle=reranker_bundle,
                pairprob_direct_gap_pct=pairprob_gap,
                metadata_oracle_gap_pct=metadata_oracle_gap_pct,
                embedding_dim=int(embedding_feature_dim),
                expert_feature_dim=int(expert_feature_dim),
                cfg=cfg.top2_margin_reranker,
            )
        )
        rows.extend(
            top2_rerank_route_rows(
                method=str(cfg.top2_margin_reranker.diagnostic_oracle_method_name),
                fold=fold,
                query_domains=query_domains,
                expert_domains=fold.candidate_expert_domains,
                x_rows=x_test,
                prob_matrix=direct_prob,
                true_nelbo_matrix=true_matrix,
                global_true_nelbo_matrix=global_eval,
                global_expert_domains=global_expert_domains,
                policy_name=cfg.top2_margin_reranker.method_name,
                selection=top2_selection_for_eval,
                reranker_bundle=None,
                pairprob_direct_gap_pct=pairprob_gap,
                metadata_oracle_gap_pct=metadata_oracle_gap_pct,
                embedding_dim=int(embedding_feature_dim),
                expert_feature_dim=int(expert_feature_dim),
                cfg=cfg.top2_margin_reranker,
                oracle_diagnostic=True,
            )
        )
    if (
        top2_delta_selection is not None
        and bool(cfg.top2_delta_gate.enabled)
        and direct_prob is not None
        and direct_selection_for_eval is not None
    ):
        delta_cfg = cfg.top2_delta_gate
        pairprob_gap = np.asarray([float(r["oracle_gap_pct"]) for r in direct_diagnostic_rows], dtype=np.float64)
        final_train = build_group_oof_top2_delta_gate_training_data(
            x_rows=x_train,
            q_rows=q_train,
            e_rows=e_train,
            s_rows=s_train,
            y_rows=y_train,
            feature_set=str(delta_cfg.base_feature_set),
            ridge_l2=float(direct_selection_for_eval.ridge_l2),
            pairprob_cfg=cfg,
            delta_cfg=delta_cfg,
            embedding_dim=int(embedding_feature_dim),
            expert_feature_dim=int(expert_feature_dim),
            margin_threshold=float(top2_delta_selection.margin_threshold),
            device=device,
        )
        delta_reason_parts = [
            str(top2_delta_selection.diagnostic_only_reason),
            str(final_train.diagnostic_reason),
        ]
        delta_bundle: Top2DeltaGateModelBundle | None = None
        if not any(str(part) for part in delta_reason_parts) and not bool(top2_delta_selection.noop):
            try:
                delta_bundle = fit_top2_delta_gate_model(
                    train_data=final_train,
                    ridge_l2=float(top2_delta_selection.ridge_l2),
                )
            except (ProtocolError, ValueError):
                delta_reason_parts.append("insufficient_source_inner_delta_rows")
        delta_reason = "|".join(part for part in dict.fromkeys(delta_reason_parts) if part)
        delta_selection_for_eval = replace(
            top2_delta_selection,
            diagnostic_only_reason=str(delta_reason),
            noop=bool(delta_reason),
            guard_status="selected" if not delta_reason else "failed_guards_noop",
            selection_stability_status=(
                top2_delta_selection.selection_stability_status
                if not delta_reason or top2_delta_selection.selection_stability_status == "unstable"
                else "forced_direct_pairprob"
            ),
            group_oof_grouping_level=str(final_train.group_oof_grouping_level),
            group_oof_unique_groups=int(final_train.group_oof_unique_groups),
            group_oof_min_groups_per_fold=int(final_train.group_oof_min_groups_per_fold),
            group_oof_folds_used=int(final_train.group_oof_folds_used),
            group_oof_train_domains_per_fold_min=int(final_train.group_oof_train_domains_per_fold_min),
            group_oof_candidate_experts_per_fold_min=int(final_train.group_oof_candidate_experts_per_fold_min),
            group_oof_same_group_leakage_rate=float(final_train.group_oof_same_group_leakage_rate),
        )
        rows.extend(
            top2_delta_gate_route_rows(
                method=str(delta_cfg.method_name),
                fold=fold,
                query_domains=query_domains,
                expert_domains=fold.candidate_expert_domains,
                x_rows=x_test,
                prob_matrix=direct_prob,
                true_nelbo_matrix=true_matrix,
                global_true_nelbo_matrix=global_eval,
                global_expert_domains=global_expert_domains,
                policy_name=delta_cfg.method_name,
                selection=delta_selection_for_eval,
                delta_bundle=delta_bundle,
                pairprob_direct_gap_pct=pairprob_gap,
                metadata_oracle_gap_pct=metadata_oracle_gap_pct,
                embedding_dim=int(embedding_feature_dim),
                expert_feature_dim=int(expert_feature_dim),
                cfg=delta_cfg,
            )
        )
        rows.extend(
            top2_delta_gate_route_rows(
                method=str(delta_cfg.oracle_diagnostic_method_name),
                fold=fold,
                query_domains=query_domains,
                expert_domains=fold.candidate_expert_domains,
                x_rows=x_test,
                prob_matrix=direct_prob,
                true_nelbo_matrix=true_matrix,
                global_true_nelbo_matrix=global_eval,
                global_expert_domains=global_expert_domains,
                policy_name=delta_cfg.method_name,
                selection=delta_selection_for_eval,
                delta_bundle=None,
                pairprob_direct_gap_pct=pairprob_gap,
                metadata_oracle_gap_pct=metadata_oracle_gap_pct,
                embedding_dim=int(embedding_feature_dim),
                expert_feature_dim=int(expert_feature_dim),
                cfg=delta_cfg,
                oracle_diagnostic=True,
            )
        )
    if (
        allpair_delta_selection is not None
        and bool(cfg.allpair_delta_gate.enabled)
        and direct_prob is not None
        and direct_selection_for_eval is not None
    ):
        allpair_cfg = cfg.allpair_delta_gate
        pairprob_gap = np.asarray([float(r["oracle_gap_pct"]) for r in direct_diagnostic_rows], dtype=np.float64)
        final_train = build_group_oof_allpair_delta_gate_training_data(
            x_rows=x_train,
            q_rows=q_train,
            e_rows=e_train,
            s_rows=s_train,
            y_rows=y_train,
            feature_set=str(allpair_cfg.base_feature_set),
            ridge_l2=float(direct_selection_for_eval.ridge_l2),
            pairprob_cfg=cfg,
            delta_cfg=allpair_cfg,
            embedding_dim=int(embedding_feature_dim),
            expert_feature_dim=int(expert_feature_dim),
            margin_threshold=float(allpair_delta_selection.margin_threshold),
            device=device,
        )
        allpair_reason_parts = [
            str(allpair_delta_selection.diagnostic_only_reason),
            str(final_train.diagnostic_reason),
        ]
        allpair_bundle: Top2DeltaGateModelBundle | None = None
        if not any(str(part) for part in allpair_reason_parts) and not bool(allpair_delta_selection.noop):
            try:
                allpair_bundle = fit_allpair_delta_gate_model(
                    train_data=final_train,
                    ridge_l2=float(allpair_delta_selection.ridge_l2),
                )
            except (ProtocolError, ValueError):
                allpair_reason_parts.append("insufficient_source_inner_delta_rows")
        allpair_reason = "|".join(part for part in dict.fromkeys(allpair_reason_parts) if part)
        allpair_selection_for_eval = replace(
            allpair_delta_selection,
            diagnostic_only_reason=str(allpair_reason),
            noop=bool(allpair_reason),
            guard_status="selected" if not allpair_reason else "failed_guards_noop",
            selection_stability_status=(
                allpair_delta_selection.selection_stability_status
                if not allpair_reason or allpair_delta_selection.selection_stability_status == "unstable"
                else "forced_direct_pairprob"
            ),
            group_oof_grouping_level=str(final_train.group_oof_grouping_level),
            group_oof_unique_groups=int(final_train.group_oof_unique_groups),
            group_oof_min_groups_per_fold=int(final_train.group_oof_min_groups_per_fold),
            group_oof_folds_used=int(final_train.group_oof_folds_used),
            group_oof_train_domains_per_fold_min=int(final_train.group_oof_train_domains_per_fold_min),
            group_oof_candidate_experts_per_fold_min=int(final_train.group_oof_candidate_experts_per_fold_min),
            group_oof_same_group_leakage_rate=float(final_train.group_oof_same_group_leakage_rate),
            selected_reason=str(allpair_reason or allpair_delta_selection.selected_reason),
        )
        rows.extend(
            allpair_delta_gate_route_rows(
                method=str(allpair_cfg.method_name),
                fold=fold,
                query_domains=query_domains,
                expert_domains=fold.candidate_expert_domains,
                x_rows=x_test,
                prob_matrix=direct_prob,
                true_nelbo_matrix=true_matrix,
                global_true_nelbo_matrix=global_eval,
                global_expert_domains=global_expert_domains,
                policy_name=allpair_cfg.method_name,
                selection=allpair_selection_for_eval,
                delta_bundle=allpair_bundle,
                pairprob_direct_gap_pct=pairprob_gap,
                metadata_oracle_gap_pct=metadata_oracle_gap_pct,
                embedding_dim=int(embedding_feature_dim),
                expert_feature_dim=int(expert_feature_dim),
                cfg=allpair_cfg,
            )
        )
        rows.extend(
            allpair_delta_gate_route_rows(
                method=str(allpair_cfg.oracle_diagnostic_method_name),
                fold=fold,
                query_domains=query_domains,
                expert_domains=fold.candidate_expert_domains,
                x_rows=x_test,
                prob_matrix=direct_prob,
                true_nelbo_matrix=true_matrix,
                global_true_nelbo_matrix=global_eval,
                global_expert_domains=global_expert_domains,
                policy_name=allpair_cfg.method_name,
                selection=allpair_selection_for_eval,
                delta_bundle=None,
                pairprob_direct_gap_pct=pairprob_gap,
                metadata_oracle_gap_pct=metadata_oracle_gap_pct,
                embedding_dim=int(embedding_feature_dim),
                expert_feature_dim=int(expert_feature_dim),
                cfg=allpair_cfg,
                oracle_diagnostic=True,
            )
        )
    if (
        hardpair_boost_selection is not None
        and bool(cfg.group_oof_hardpair_boost.enabled)
        and direct_prob is not None
        and direct_selection_for_eval is not None
    ):
        boost_cfg = cfg.group_oof_hardpair_boost
        source_domains = sorted(set(int(v) for v in np.asarray(q_train, dtype=np.int64).tolist()))
        observations, diag = build_group_oof_hardpair_observations(
            x_rows=x_train,
            q_rows=q_train,
            e_rows=e_train,
            s_rows=s_train,
            y_rows=y_train,
            source_domains=source_domains,
            feature_set=str(boost_cfg.feature_set),
            ridge_l2=float(hardpair_boost_selection.ridge_l2),
            cfg=boost_cfg,
            embedding_dim=int(embedding_feature_dim),
            expert_feature_dim=int(expert_feature_dim),
            device=device,
        )
        overrides, _override_stats = hardpair_weight_multipliers_from_observations(
            observations,
            margin_threshold=float(hardpair_boost_selection.margin_threshold),
            miss_boost_weight=float(hardpair_boost_selection.miss_boost_weight),
            confirm_boost_weight=float(hardpair_boost_selection.confirm_boost_weight),
            max_pair_weight=float(boost_cfg.max_pair_weight),
        )
        final_reason = "|".join(
            part
            for part in dict.fromkeys([str(hardpair_boost_selection.diagnostic_only_reason), str(diag.reason)])
            if part
        )
        boost_bundle: PairprobModelBundle | None = None
        boost_prob = direct_prob
        if not final_reason:
            boost_bundle, _boost_evidence, fit_reason = _fit_pairprob_bundle_from_rows(
                x_rows=x_train,
                q_rows=q_train,
                e_rows=e_train,
                s_rows=s_train,
                y_rows=y_train,
                feature_set=str(boost_cfg.feature_set),
                ridge_l2=float(hardpair_boost_selection.ridge_l2),
                tournament_cfg=tournament_cfg,
                embedding_feature_dim=int(embedding_feature_dim),
                expert_feature_dim=int(expert_feature_dim),
                device=device,
                pair_weight_multipliers=overrides,
            )
            final_reason = "|".join(
                part for part in dict.fromkeys([final_reason, str(fit_reason)]) if part
            )
        if boost_bundle is not None and not final_reason:
            boost_prob = pairprob_probability_matrix(
                bundle=boost_bundle,
                x_rows=x_test,
                expert_domains=fold.candidate_expert_domains,
                embedding_dim=int(embedding_feature_dim),
                expert_feature_dim=int(expert_feature_dim),
            )
        boost_selection_for_eval = replace(
            hardpair_boost_selection,
            diagnostic_only_reason=str(final_reason),
            noop=bool(final_reason),
            guard_status="selected" if not final_reason else "failed_guards_noop",
        )
        rows.extend(
            hardpair_boost_route_rows(
                method=str(boost_cfg.method_name),
                fold=fold,
                query_domains=query_domains,
                expert_domains=fold.candidate_expert_domains,
                prob_matrix=boost_prob,
                direct_prob_matrix=direct_prob,
                true_nelbo_matrix=true_matrix,
                global_true_nelbo_matrix=global_eval,
                global_expert_domains=global_expert_domains,
                policy_name=str(boost_cfg.method_name),
                selection=boost_selection_for_eval,
                cfg=boost_cfg,
                metadata_oracle_gap_pct=metadata_oracle_gap_pct,
            )
        )
        rows.extend(
            hardpair_boost_route_rows(
                method=str(boost_cfg.miss_only_diagnostic_method_name),
                fold=fold,
                query_domains=query_domains,
                expert_domains=fold.candidate_expert_domains,
                prob_matrix=boost_prob,
                direct_prob_matrix=direct_prob,
                true_nelbo_matrix=true_matrix,
                global_true_nelbo_matrix=global_eval,
                global_expert_domains=global_expert_domains,
                policy_name=str(boost_cfg.method_name),
                selection=boost_selection_for_eval,
                cfg=boost_cfg,
                metadata_oracle_gap_pct=metadata_oracle_gap_pct,
                diagnostic_reason="group_oof_hardpair_miss_only_diagnostic",
            )
        )
        random_overrides, _random_stats = hardpair_weight_multipliers_from_observations(
            observations,
            margin_threshold=float(hardpair_boost_selection.margin_threshold),
            miss_boost_weight=float(hardpair_boost_selection.miss_boost_weight),
            confirm_boost_weight=1.0,
            max_pair_weight=float(boost_cfg.max_pair_weight),
            random_control=True,
            seed=int(query_domains[0]) if len(query_domains) else 0,
        )
        random_prob = direct_prob
        if not str(diag.reason):
            random_bundle, _random_evidence, random_reason = _fit_pairprob_bundle_from_rows(
                x_rows=x_train,
                q_rows=q_train,
                e_rows=e_train,
                s_rows=s_train,
                y_rows=y_train,
                feature_set=str(boost_cfg.feature_set),
                ridge_l2=float(hardpair_boost_selection.ridge_l2),
                tournament_cfg=tournament_cfg,
                embedding_feature_dim=int(embedding_feature_dim),
                expert_feature_dim=int(expert_feature_dim),
                device=device,
                pair_weight_multipliers=random_overrides,
            )
            if random_bundle is not None and not random_reason:
                random_prob = pairprob_probability_matrix(
                    bundle=random_bundle,
                    x_rows=x_test,
                    expert_domains=fold.candidate_expert_domains,
                    embedding_dim=int(embedding_feature_dim),
                    expert_feature_dim=int(expert_feature_dim),
                )
        rows.extend(
            hardpair_boost_route_rows(
                method=str(boost_cfg.random_control_method_name),
                fold=fold,
                query_domains=query_domains,
                expert_domains=fold.candidate_expert_domains,
                prob_matrix=random_prob,
                direct_prob_matrix=direct_prob,
                true_nelbo_matrix=true_matrix,
                global_true_nelbo_matrix=global_eval,
                global_expert_domains=global_expert_domains,
                policy_name=str(boost_cfg.method_name),
                selection=boost_selection_for_eval,
                cfg=boost_cfg,
                metadata_oracle_gap_pct=metadata_oracle_gap_pct,
                diagnostic_reason="random_low_margin_boost_diagnostic",
            )
        )
        rows.extend(
            hardpair_boost_route_rows(
                method=str(boost_cfg.oracle_top2_diagnostic_method_name),
                fold=fold,
                query_domains=query_domains,
                expert_domains=fold.candidate_expert_domains,
                prob_matrix=direct_prob,
                direct_prob_matrix=direct_prob,
                true_nelbo_matrix=true_matrix,
                global_true_nelbo_matrix=global_eval,
                global_expert_domains=global_expert_domains,
                policy_name=str(boost_cfg.method_name),
                selection=boost_selection_for_eval,
                cfg=boost_cfg,
                metadata_oracle_gap_pct=metadata_oracle_gap_pct,
                oracle_top2_diagnostic=True,
            )
        )
    return rows


def _run_learned_methods_for_fold(
    *,
    embeddings: np.ndarray,
    sample_domains: np.ndarray,
    true_nelbo: np.ndarray,
    expert_domains: Sequence[int],
    domain_to_idx: Dict[int, int],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold: FoldCandidateSet,
    global_eval: np.ndarray,
    metadata_similarity_eval: np.ndarray | None,
    predictors: Sequence[str],
    mlp_cfg: Dict[str, Any],
    pairwise_cfg: Dict[str, Any],
    tournament_cfg: PairwiseTournamentConfig,
    include_metadata_features: bool,
    seed: int,
    embedding_feature_dim: int,
    expert_feature_dim: int,
    tie_policy: str,
) -> LearnedFoldOutputs:
    x_train, q_train, e_train, s_train = _build_fold_training_pair_features(
        sample_embeddings=embeddings,
        sample_domains=sample_domains,
        train_indices=train_idx,
        expert_domains=expert_domains,
        outer_heldout_domain=int(fold.heldout_domain),
        include_metadata_features=include_metadata_features,
    )
    x_test, q_test, e_test, s_test = _build_pair_features(
        sample_embeddings=embeddings,
        sample_domains=sample_domains,
        sample_indices=test_idx,
        expert_domains=fold.candidate_expert_domains,
        expert_id_domains=expert_domains,
        include_metadata_features=include_metadata_features,
    )

    y_train = true_nelbo[s_train, [domain_to_idx[int(ed)] for ed in e_train]]
    y_test = true_nelbo[s_test, [domain_to_idx[int(ed)] for ed in e_test]]

    y_train_norm = _normalize_targets_per_query(y_train, q_train)
    x_train_z, x_test_z = _zscore_features(x_train, x_test)

    test_n = int(test_idx.size)
    e_n = len(fold.candidate_expert_domains)
    train_candidates_per_sample = _infer_experts_per_sample(s_train)
    if train_candidates_per_sample < 1:
        raise ProtocolError("learned_pair_policy requires at least one non-self source candidate per source query")

    models: Dict[str, Any] = {}
    if "linear_regressor" in predictors:
        linear = _LinearRegressor(l2=1e-4)
        linear.fit(x_train_z, y_train_norm)
        models["linear_regressor"] = linear
    if "mlp_regressor" in predictors:
        mlp = _MLPRegressor(
            seed=int(seed),
            hidden_dim=int(mlp_cfg.get("hidden_dim", 128)),
            epochs=int(mlp_cfg.get("epochs", 40)),
            lr=float(mlp_cfg.get("lr", 1e-3)),
            batch_size=int(mlp_cfg.get("batch_size", 2048)),
            device=str(mlp_cfg.get("device", "auto")),
        )
        mlp.fit(x_train_z, y_train_norm)
        models["mlp_regressor"] = mlp

    if "metadata_only_regressor" in predictors:
        # Minimal metadata-only variant built from exact-domain feature and normalized difference.
        x_train_meta = x_train_z[:, -2:] if include_metadata_features else np.zeros((x_train_z.shape[0], 2), dtype=np.float64)
        x_test_meta = x_test_z[:, -2:] if include_metadata_features else np.zeros((x_test_z.shape[0], 2), dtype=np.float64)
        meta_reg = _LinearRegressor(l2=1e-4)
        meta_reg.fit(x_train_meta, y_train_norm)
        models["metadata_only_regressor"] = (meta_reg, x_test_meta)

    pair_training_rows: List[Dict[str, Any]] = []
    if "pairwise_ranker" in predictors:
        near_tie_delta = float(pairwise_cfg.get("near_tie_delta", 0.0))
        hard_pair_fraction = float(pairwise_cfg.get("hard_pair_fraction", 0.5))
        random_pair_fraction = float(pairwise_cfg.get("random_pair_fraction", 0.5))
        max_pairs_per_sample = int(pairwise_cfg.get("max_pairs_per_sample", 12))
        max_pairs_per_domain = int(pairwise_cfg.get("max_pairs_per_domain", 5000))
        run_ablations = bool(pairwise_cfg.get("run_ablations", True))

        train_pairs, pair_diags = _build_pairwise_training_pairs(
            y_train=y_train,
            q_train=q_train,
            s_train=s_train,
            experts_per_sample=train_candidates_per_sample,
            near_tie_delta=near_tie_delta,
            hard_pair_fraction=hard_pair_fraction,
            random_pair_fraction=random_pair_fraction,
            max_pairs_per_sample=max_pairs_per_sample,
            max_pairs_per_domain=max_pairs_per_domain,
            seed=int(seed) + int(fold.heldout_domain),
        )
        for d in pair_diags:
            training_diag_fold = FoldCandidateSet.for_heldout_domain(
                heldout_domain=int(fold.heldout_domain),
                expert_domains=expert_domains,
                excluded_domains=[int(d["query_domain"])],
            )
            training_diag_protocol = _protocol_row_fields(
                fold=training_diag_fold,
                method_protocol=_method_protocol("pairwise_ranker"),
                method="pairwise_ranker_training_pairs",
            )
            training_diag_protocol["fold_query_domain"] = int(fold.heldout_domain)
            training_diag_protocol["excluded_experts"] = "|".join(
                str(int(v)) for v in sorted({int(fold.heldout_domain), int(d["query_domain"])})
            )
            pair_training_rows.append(
                {
                    **training_diag_protocol,
                    "fold_query_domain": int(fold.heldout_domain),
                    **d,
                }
            )

        if train_pairs:
            for method_name, x_tr_variant, x_te_variant in _pairwise_variant_features(
                x_train=x_train,
                x_test=x_test,
                q_train=q_train,
                e_train=e_train,
                q_test=q_test,
                e_test=e_test,
                sample_domains=sample_domains,
                embedding_feature_dim=embedding_feature_dim,
                expert_feature_dim=expert_feature_dim,
                run_ablations=run_ablations,
            ):
                ranker = _PairwiseRanker(
                    seed=int(seed),
                    hidden_dim=int(pairwise_cfg.get("hidden_dim", 128)),
                    epochs=int(pairwise_cfg.get("epochs", 40)),
                    lr=float(pairwise_cfg.get("lr", 1e-3)),
                    batch_size=int(pairwise_cfg.get("batch_size", 2048)),
                    margin=float(pairwise_cfg.get("margin", 1.0)),
                    device=str(pairwise_cfg.get("device", "auto")),
                )
                ranker.fit(x_tr_variant, train_pairs)
                models[method_name] = (ranker, x_te_variant)

    sample_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    pairwise_pred_matrices: Dict[str, np.ndarray] = {}
    true_matrix_for_tournament: np.ndarray | None = None
    metadata_oracle_gap_pct_for_tournament: np.ndarray | None = None
    for method, model in models.items():
        if isinstance(model, tuple):
            reg, x_m = model
            pred = reg.predict(x_m)
        else:
            pred = model.predict(x_test_z)

        expected_pred_rows = int(test_n) * int(e_n)
        if int(pred.shape[0]) != expected_pred_rows:
            raise ProtocolError(
                f"Evaluation pair predictions for {method} have {pred.shape[0]} rows; "
                f"expected {expected_pred_rows}"
            )
        pred_matrix = pred.reshape(test_n, e_n)
        true_matrix = y_test.reshape(test_n, e_n)
        if str(method).startswith("pairwise_ranker"):
            pairwise_pred_matrices[str(method)] = pred_matrix
            true_matrix_for_tournament = true_matrix
            metadata_oracle_gap_pct_for_tournament = _metadata_gap_pct_for_similarity(
                metadata_similarity_eval=metadata_similarity_eval,
                true_nelbo_matrix=true_matrix,
                expert_domains=fold.candidate_expert_domains,
            )

        _metrics_unused, rows = _selection_metrics(
            method=method,
            query_domains=sample_domains[test_idx],
            expert_domains=fold.candidate_expert_domains,
            score_matrix=pred_matrix,
            true_nelbo_matrix=true_matrix,
            fold=fold,
            global_true_nelbo_matrix=global_eval,
            global_expert_domains=expert_domains,
            tie_policy=tie_policy,
        )

        for row in rows:
            row["sample_index"] = int(test_idx[int(row["sample_index"])])
            sample_rows.append(row)

        row_protocol = _method_protocol(method)
        for k in range(pred.shape[0]):
            pair_rows.append(
                {
                    **_protocol_row_fields(fold=fold, method_protocol=row_protocol, method=method),
                    "method": method,
                    "sample_index": int(s_test[k]),
                    "query_domain": int(q_test[k]),
                    "expert_domain": int(e_test[k]),
                    "predicted_score": float(pred[k]),
                    "true_nelbo": float(y_test[k]),
                }
            )

    if bool(tournament_cfg.enabled) and pairwise_pred_matrices and true_matrix_for_tournament is not None:
        if bool(tournament_cfg.fallback_benefit_gate.enabled):
            delta_selection, diagnostic_delta_selection = _calibrate_delta_gate_tournament(
                embeddings=embeddings,
                sample_domains=sample_domains,
                true_nelbo=true_nelbo,
                expert_domains=expert_domains,
                domain_to_idx=domain_to_idx,
                train_idx=train_idx,
                outer_heldout_domain=int(fold.heldout_domain),
                pairwise_cfg=pairwise_cfg,
                tournament_cfg=tournament_cfg,
                include_metadata_features=include_metadata_features,
                seed=int(seed),
                embedding_feature_dim=embedding_feature_dim,
                expert_feature_dim=expert_feature_dim,
                global_expert_domains=expert_domains,
            )
            available_bases = [str(m) for m in tournament_cfg.base_methods if str(m) in pairwise_pred_matrices]
            if not available_bases:
                available_bases = sorted(pairwise_pred_matrices)
            hard_base = (
                str(delta_selection.base_method)
                if str(delta_selection.base_method) in pairwise_pred_matrices
                else str(available_bases[0])
            )
            hard_score = pairwise_pred_matrices[hard_base]
            sample_rows.extend(
                tournament_route_rows(
                    method="pairwise_tournament_hard",
                    fold=fold,
                    query_domains=sample_domains[test_idx],
                    expert_domains=fold.candidate_expert_domains,
                    score_matrix=hard_score,
                    true_nelbo_matrix=true_matrix_for_tournament,
                    global_true_nelbo_matrix=global_eval,
                    global_expert_domains=expert_domains,
                    policy_name=tournament_cfg.policy_name,
                    base_method=hard_base,
                    threshold=0.0,
                    topk=1,
                    temperature=float(tournament_cfg.score_temperature),
                    temperature_policy=tournament_cfg.temperature_policy,
                    selected_by_inner_validation=bool(delta_selection.selected_by_inner_validation),
                    threshold_selection_policy=tournament_cfg.fallback_benefit_gate.calibration_policy,
                )
            )
            topk_for_reference = int(delta_selection.topk)
            sample_rows.extend(
                tournament_route_rows(
                    method="pairwise_tournament_topk_uniform",
                    fold=fold,
                    query_domains=sample_domains[test_idx],
                    expert_domains=fold.candidate_expert_domains,
                    score_matrix=hard_score,
                    true_nelbo_matrix=true_matrix_for_tournament,
                    global_true_nelbo_matrix=global_eval,
                    global_expert_domains=expert_domains,
                    policy_name=tournament_cfg.policy_name,
                    base_method=hard_base,
                    threshold=float("inf"),
                    topk=topk_for_reference,
                    temperature=float(tournament_cfg.score_temperature),
                    temperature_policy=tournament_cfg.temperature_policy,
                    selected_by_inner_validation=bool(delta_selection.selected_by_inner_validation),
                    threshold_selection_policy="always_sparse_mix_topk_reference",
                    diagnostic_only_reason="diagnostic_only_sparse_mix_always_active",
                )
            )
            sample_rows.extend(
                delta_gate_route_rows(
                    method=tournament_cfg.fallback_benefit_gate.method_name,
                    fold=fold,
                    query_domains=sample_domains[test_idx],
                    expert_domains=fold.candidate_expert_domains,
                    score_matrix=hard_score,
                    true_nelbo_matrix=true_matrix_for_tournament,
                    global_true_nelbo_matrix=global_eval,
                    global_expert_domains=expert_domains,
                    policy_name=tournament_cfg.policy_name,
                    selection=delta_selection,
                    temperature=float(tournament_cfg.score_temperature),
                    temperature_policy=tournament_cfg.temperature_policy,
                    gate_cfg=tournament_cfg.fallback_benefit_gate,
                )
            )
            if (
                diagnostic_delta_selection is not None
                and str(diagnostic_delta_selection.base_method) in pairwise_pred_matrices
                and (
                    str(diagnostic_delta_selection.feature_set) != "tournament_uncertainty_combined_diagnostic_v1"
                    or (
                        "pairwise_ranker_latent_only" in pairwise_pred_matrices
                        and "pairwise_ranker_combined" in pairwise_pred_matrices
                    )
                )
            ):
                sample_rows.extend(
                    delta_gate_route_rows(
                        method="pairwise_tournament_delta_gated_sparse_mix_combined_diagnostic_v1",
                        fold=fold,
                        query_domains=sample_domains[test_idx],
                        expert_domains=fold.candidate_expert_domains,
                        score_matrix=pairwise_pred_matrices[str(diagnostic_delta_selection.base_method)],
                        true_nelbo_matrix=true_matrix_for_tournament,
                        global_true_nelbo_matrix=global_eval,
                        global_expert_domains=expert_domains,
                        policy_name=tournament_cfg.policy_name,
                        selection=diagnostic_delta_selection,
                        temperature=float(tournament_cfg.score_temperature),
                        temperature_policy=tournament_cfg.temperature_policy,
                        gate_cfg=tournament_cfg.fallback_benefit_gate,
                        diagnostic_only_reason="diagnostic_only_combined_metadata_features",
                        latent_score_matrix=pairwise_pred_matrices.get("pairwise_ranker_latent_only"),
                        combined_score_matrix=pairwise_pred_matrices.get("pairwise_ranker_combined"),
                    )
                )
            sample_rows.extend(
                oracle_confidence_set_rows(
                    fold=fold,
                    query_domains=sample_domains[test_idx],
                    expert_domains=fold.candidate_expert_domains,
                    score_matrix=hard_score,
                    true_nelbo_matrix=true_matrix_for_tournament,
                    global_true_nelbo_matrix=global_eval,
                    global_expert_domains=expert_domains,
                    policy_name=tournament_cfg.policy_name,
                    base_method=hard_base,
                    topk=topk_for_reference,
                    temperature=float(tournament_cfg.score_temperature),
                    temperature_policy=tournament_cfg.temperature_policy,
                )
            )
        else:
            selected_policy = _calibrate_pairwise_tournament(
                embeddings=embeddings,
                sample_domains=sample_domains,
                true_nelbo=true_nelbo,
                expert_domains=expert_domains,
                domain_to_idx=domain_to_idx,
                train_idx=train_idx,
                outer_heldout_domain=int(fold.heldout_domain),
                pairwise_cfg=pairwise_cfg,
                tournament_cfg=tournament_cfg,
                include_metadata_features=include_metadata_features,
                seed=int(seed),
                embedding_feature_dim=embedding_feature_dim,
                expert_feature_dim=expert_feature_dim,
                global_expert_domains=expert_domains,
            )
            available_bases = [str(m) for m in tournament_cfg.base_methods if str(m) in pairwise_pred_matrices]
            if not available_bases:
                available_bases = sorted(pairwise_pred_matrices)
            hard_base = (
                str(selected_policy.base_method)
                if selected_policy is not None and str(selected_policy.base_method) in pairwise_pred_matrices
                else str(available_bases[0])
            )
            hard_score = pairwise_pred_matrices[hard_base]
            sample_rows.extend(
                tournament_route_rows(
                    method="pairwise_tournament_hard",
                    fold=fold,
                    query_domains=sample_domains[test_idx],
                    expert_domains=fold.candidate_expert_domains,
                    score_matrix=hard_score,
                    true_nelbo_matrix=true_matrix_for_tournament,
                    global_true_nelbo_matrix=global_eval,
                    global_expert_domains=expert_domains,
                    policy_name=tournament_cfg.policy_name,
                    base_method=hard_base,
                    threshold=0.0,
                    topk=1,
                    temperature=float(tournament_cfg.score_temperature),
                    temperature_policy=tournament_cfg.temperature_policy,
                    selected_by_inner_validation=bool(selected_policy is not None),
                    threshold_selection_policy=tournament_cfg.calibration_policy,
                )
            )

            topk_for_reference = int(
                selected_policy.topk
                if selected_policy is not None
                else min(max(int(tournament_cfg.sparse_mix_topk_values[0]), 1), len(fold.candidate_expert_domains))
            )
            sample_rows.extend(
                tournament_route_rows(
                    method="pairwise_tournament_topk_uniform",
                    fold=fold,
                    query_domains=sample_domains[test_idx],
                    expert_domains=fold.candidate_expert_domains,
                    score_matrix=hard_score,
                    true_nelbo_matrix=true_matrix_for_tournament,
                    global_true_nelbo_matrix=global_eval,
                    global_expert_domains=expert_domains,
                    policy_name=tournament_cfg.policy_name,
                    base_method=hard_base,
                    threshold=float("inf"),
                    topk=topk_for_reference,
                    temperature=float(tournament_cfg.score_temperature),
                    temperature_policy=tournament_cfg.temperature_policy,
                    selected_by_inner_validation=bool(selected_policy is not None),
                    threshold_selection_policy="always_sparse_mix_topk_reference",
                    diagnostic_only_reason="diagnostic_only_sparse_mix_always_active",
                )
            )

            if selected_policy is not None and str(selected_policy.base_method) in pairwise_pred_matrices:
                selected_score = pairwise_pred_matrices[str(selected_policy.base_method)]
                sample_rows.extend(
                    tournament_route_rows(
                        method="pairwise_tournament_inner_selected",
                        fold=fold,
                        query_domains=sample_domains[test_idx],
                        expert_domains=fold.candidate_expert_domains,
                        score_matrix=selected_score,
                        true_nelbo_matrix=true_matrix_for_tournament,
                        global_true_nelbo_matrix=global_eval,
                        global_expert_domains=expert_domains,
                        policy_name=tournament_cfg.policy_name,
                        base_method=str(selected_policy.base_method),
                        threshold=float(selected_policy.threshold),
                        topk=int(selected_policy.topk),
                        temperature=float(tournament_cfg.score_temperature),
                        temperature_policy=tournament_cfg.temperature_policy,
                        selected_by_inner_validation=True,
                        threshold_selection_policy=tournament_cfg.calibration_policy,
                        diagnostic_only_reason=str(selected_policy.diagnostic_only_reason),
                        source_inner_summary=selected_policy,
                    )
                )
                sample_rows.extend(
                    oracle_confidence_set_rows(
                        fold=fold,
                        query_domains=sample_domains[test_idx],
                        expert_domains=fold.candidate_expert_domains,
                        score_matrix=selected_score,
                        true_nelbo_matrix=true_matrix_for_tournament,
                        global_true_nelbo_matrix=global_eval,
                        global_expert_domains=expert_domains,
                        policy_name=tournament_cfg.policy_name,
                        base_method=str(selected_policy.base_method),
                        topk=int(selected_policy.topk),
                        temperature=float(tournament_cfg.score_temperature),
                        temperature_policy=tournament_cfg.temperature_policy,
                    )
                )

        if bool(tournament_cfg.pairprob_tournament.enabled):
            pairprob_hard_base = (
                "pairwise_ranker_latent_only"
                if "pairwise_ranker_latent_only" in pairwise_pred_matrices
                else next(iter(sorted(pairwise_pred_matrices)))
            )
            pairprob_hard_gap = _hard_gap_pct_for_score_matrix(
                fold=fold,
                query_domains=sample_domains[test_idx],
                expert_domains=fold.candidate_expert_domains,
                score_matrix=pairwise_pred_matrices[pairprob_hard_base],
                true_nelbo_matrix=true_matrix_for_tournament,
                global_true_nelbo_matrix=global_eval,
                global_expert_domains=expert_domains,
                tournament_cfg=tournament_cfg,
                base_method=pairprob_hard_base,
            )
            (
                direct_selection,
                direct_adoption_selection,
                group_selection,
                combined_selection,
                conformal_selection,
                jackknife_selection,
                top2_selection,
                top2_delta_selection,
                allpair_delta_selection,
                hardpair_boost_selection,
            ) = _calibrate_pairprob_tournament(
                embeddings=embeddings,
                sample_domains=sample_domains,
                true_nelbo=true_nelbo,
                expert_domains=expert_domains,
                domain_to_idx=domain_to_idx,
                train_idx=train_idx,
                outer_heldout_domain=int(fold.heldout_domain),
                pairwise_cfg=pairwise_cfg,
                tournament_cfg=tournament_cfg,
                include_metadata_features=include_metadata_features,
                seed=int(seed),
                embedding_feature_dim=embedding_feature_dim,
                expert_feature_dim=expert_feature_dim,
                global_expert_domains=expert_domains,
            )
            sample_rows.extend(
                _run_pairprob_tournament_for_fold(
                    x_train=x_train,
                    q_train=q_train,
                    e_train=e_train,
                    s_train=s_train,
                    y_train=y_train,
                    x_test=x_test,
                    query_domains=sample_domains[test_idx],
                    fold=fold,
                    true_matrix=true_matrix_for_tournament,
                    global_eval=global_eval,
                    expert_domains=fold.candidate_expert_domains,
                    global_expert_domains=expert_domains,
                    selections=[
                        direct_selection,
                        direct_adoption_selection,
                        group_selection,
                        combined_selection,
                    ],
                    conformal_selection=conformal_selection,
                    jackknife_selection=jackknife_selection,
                    top2_selection=top2_selection,
                    top2_delta_selection=top2_delta_selection,
                    allpair_delta_selection=allpair_delta_selection,
                    hardpair_boost_selection=hardpair_boost_selection,
                    tournament_cfg=tournament_cfg,
                    pairwise_cfg=pairwise_cfg,
                    embedding_feature_dim=embedding_feature_dim,
                    expert_feature_dim=expert_feature_dim,
                    hard_oracle_gap_pct=pairprob_hard_gap,
                    metadata_oracle_gap_pct=metadata_oracle_gap_pct_for_tournament,
                )
            )

    return LearnedFoldOutputs(
        sample_rows=sample_rows,
        pair_rows=pair_rows,
        pair_training_rows=pair_training_rows,
    )
