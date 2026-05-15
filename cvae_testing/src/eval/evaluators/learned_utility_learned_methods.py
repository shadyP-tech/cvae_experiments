from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

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
from src.eval.evaluators.pairwise_ae_combined_v2 import (
    run_pairwise_ae_combined_v2_for_fold,
)


@dataclass(frozen=True)
class LearnedFoldOutputs:
    sample_rows: List[Dict[str, Any]]
    pair_rows: List[Dict[str, Any]]
    pair_training_rows: List[Dict[str, Any]]
    pairwise_v2_training_rows: List[Dict[str, Any]]
    pairwise_v2_feature_rows: List[Dict[str, Any]]
    pairwise_v2_inner_selection_rows: List[Dict[str, Any]]
    pairwise_v2_pair_prediction_rows: List[Dict[str, Any]]
    pairwise_v2_decision_rows: List[Dict[str, Any]]


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
    include_metadata_features: bool,
    seed: int,
    embedding_feature_dim: int,
    expert_feature_dim: int,
    tie_policy: str,
    ae_zscore_matrix: np.ndarray | None = None,
    sample_metadata: Sequence[Mapping[str, Any]] | None = None,
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
    train_candidates_per_sample = max(int(len(expert_domains)) - 2, 0)
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

            ae_train_z = ae_test_z = None
            if ae_zscore_matrix is not None:
                ae_train = np.asarray(
                    [
                        [float(ae_zscore_matrix[int(s), domain_to_idx[int(e)]])]
                        for s, e in zip(s_train.tolist(), e_train.tolist())
                    ],
                    dtype=np.float64,
                )
                ae_test = np.asarray(
                    [
                        [float(ae_zscore_matrix[int(s), domain_to_idx[int(e)]])]
                        for s, e in zip(s_test.tolist(), e_test.tolist())
                    ],
                    dtype=np.float64,
                )
                ae_train_z, ae_test_z = _zscore_features(ae_train, ae_test)
                ae_metadata_train_z, ae_metadata_test_z = _zscore_features(
                    np.concatenate([ae_train, metadata_train], axis=1),
                    np.concatenate([ae_test, metadata_test], axis=1),
                )
                ae_combined_train_z, ae_combined_test_z = _zscore_features(
                    np.concatenate([ae_train, combined_train], axis=1),
                    np.concatenate([ae_test, combined_test], axis=1),
                )

            pair_variants: List[Tuple[str, np.ndarray, np.ndarray]]
            if run_ablations:
                pair_variants = [
                    ("pairwise_ranker_metadata_only", metadata_train_z, metadata_test_z),
                    ("pairwise_ranker_latent_only", latent_train_z, latent_test_z),
                    ("pairwise_ranker_combined", combined_train_z, combined_test_z),
                ]
                if ae_train_z is not None and ae_test_z is not None:
                    pair_variants.extend(
                        [
                            ("pairwise_ranker_ae_only", ae_train_z, ae_test_z),
                            ("pairwise_ranker_ae_metadata", ae_metadata_train_z, ae_metadata_test_z),
                            ("pairwise_ranker_ae_combined", ae_combined_train_z, ae_combined_test_z),
                        ]
                    )
            else:
                pair_variants = [
                    ("pairwise_ranker", combined_train_z, combined_test_z),
                ]

            for method_name, x_tr_variant, x_te_variant in pair_variants:
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

    v2_outputs = run_pairwise_ae_combined_v2_for_fold(
        embeddings=embeddings,
        sample_domains=sample_domains,
        true_nelbo=true_nelbo,
        expert_domains=expert_domains,
        domain_to_idx=domain_to_idx,
        train_idx=train_idx,
        test_idx=test_idx,
        fold=fold,
        global_eval=global_eval,
        pairwise_cfg=pairwise_cfg,
        seed=int(seed),
        embedding_feature_dim=int(embedding_feature_dim),
        expert_feature_dim=int(expert_feature_dim),
        tie_policy=tie_policy,
        ae_zscore_matrix=ae_zscore_matrix,
        sample_metadata=sample_metadata,
    )
    sample_rows.extend(v2_outputs.sample_rows)

    return LearnedFoldOutputs(
        sample_rows=sample_rows,
        pair_rows=pair_rows,
        pair_training_rows=pair_training_rows,
        pairwise_v2_training_rows=v2_outputs.training_pair_rows,
        pairwise_v2_feature_rows=v2_outputs.feature_diagnostic_rows,
        pairwise_v2_inner_selection_rows=v2_outputs.inner_selection_rows,
        pairwise_v2_pair_prediction_rows=v2_outputs.pair_rows,
        pairwise_v2_decision_rows=v2_outputs.decision_rows,
    )
