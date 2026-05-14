from __future__ import annotations

from dataclasses import dataclass
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
from src.eval.evaluators.learned_utility_protocol import (
    FoldCandidateSet,
    ProtocolError,
    _method_protocol,
    _protocol_row_fields,
)
from src.eval.evaluators.learned_utility_selection import _selection_metrics
from src.eval.evaluators.learned_utility_tournament import (
    TournamentPolicySelection,
    oracle_confidence_set_rows,
    summarize_tournament_rows,
    tournament_route_rows,
)


@dataclass(frozen=True)
class LearnedFoldOutputs:
    sample_rows: List[Dict[str, Any]]
    pair_rows: List[Dict[str, Any]]
    pair_training_rows: List[Dict[str, Any]]


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

    return LearnedFoldOutputs(
        sample_rows=sample_rows,
        pair_rows=pair_rows,
        pair_training_rows=pair_training_rows,
    )
