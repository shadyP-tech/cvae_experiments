from __future__ import annotations

from typing import Any, Dict


REQUIRED_TOP_LEVEL = ["seed", "data", "features", "model", "training", "routing"]


def _ensure_bool(value: object, name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")


def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a dictionary.")

    missing = [k for k in REQUIRED_TOP_LEVEL if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")

    split = cfg.get("data", {}).get("split")
    if not isinstance(split, dict):
        raise ValueError("data.split must be a dictionary containing train/val/test ratios.")

    for key in ["train", "val", "test"]:
        if key not in split:
            raise ValueError(f"data.split must include '{key}'.")

    train = float(split["train"])
    val = float(split["val"])
    test = float(split["test"])
    total = train + val + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"data.split ratios must sum to 1.0, got {total:.6f}")

    if int(cfg["training"]["batch_size"]) <= 0:
        raise ValueError("training.batch_size must be > 0")
    if int(cfg["training"]["epochs"]) <= 0:
        raise ValueError("training.epochs must be > 0")

    features_cfg = cfg.get("features", {})
    if not isinstance(features_cfg, dict):
        raise ValueError("features must be a dictionary")

    image_size = int(features_cfg.get("image_size", 0))
    if image_size <= 0:
        raise ValueError("features.image_size must be > 0")

    embedding_dim = features_cfg.get("embedding_dim")
    if embedding_dim is not None and int(embedding_dim) <= 0:
        raise ValueError("features.embedding_dim must be > 0 when provided")

    extraction_batch_size = features_cfg.get("extraction_batch_size")
    if extraction_batch_size is not None and int(extraction_batch_size) <= 0:
        raise ValueError("features.extraction_batch_size must be > 0 when provided")

    backbone_type = str(features_cfg.get("backbone_type", "resnet18")).strip().lower()
    allowed_backbones = {"resnet18", "resnet50", "dinov2_vitb14"}
    if backbone_type not in allowed_backbones:
        raise ValueError(
            f"features.backbone_type must be one of {sorted(allowed_backbones)}, got: {backbone_type}"
        )

    magnifications = cfg.get("data", {}).get("magnifications", [])
    if not isinstance(magnifications, list) or not magnifications:
        raise ValueError("data.magnifications must be a non-empty list")
    for m in magnifications:
        if int(m) < 0:
            raise ValueError(f"data.magnifications must contain only non-negative integers, got: {m}")

    routing_strategy = str(cfg.get("routing", {}).get("strategy", "")).strip()
    if not routing_strategy:
        raise ValueError("routing.strategy must be provided")
    from src.routing.registry import STRATEGY_REGISTRY

    if routing_strategy not in STRATEGY_REGISTRY:
        raise ValueError(
            f"routing.strategy must be one of {sorted(STRATEGY_REGISTRY)}, got: {routing_strategy}"
        )

    tracking = cfg.get("tracking")
    if tracking is not None:
        if not isinstance(tracking, dict):
            raise ValueError("tracking must be a dictionary when provided")

        backend = str(tracking.get("backend", "wandb")).strip().lower()
        if backend not in {"wandb"}:
            raise ValueError(f"tracking.backend must be one of ['wandb'], got: {backend}")

        tags = tracking.get("tags", [])
        if not isinstance(tags, list):
            raise ValueError("tracking.tags must be a list when provided")

    learned_cfg = cfg.get("learned_utility")
    if learned_cfg is not None:
        if not isinstance(learned_cfg, dict):
            raise ValueError("learned_utility must be a dictionary when provided")

        split_protocol = str(learned_cfg.get("split_protocol", "loqdo_query_domain")).strip().lower()
        if split_protocol != "loqdo_query_domain":
            raise ValueError("learned_utility.split_protocol must be 'loqdo_query_domain'")

        query_domain_field = str(learned_cfg.get("query_domain_field", "magnification")).strip()
        if query_domain_field != "magnification":
            raise ValueError("learned_utility.query_domain_field must be 'magnification'")

        splits = learned_cfg.get("splits", ["test"])
        if not isinstance(splits, list) or not splits:
            raise ValueError("learned_utility.splits must be a non-empty list")
        allowed_splits = {"train", "val", "test"}
        bad_splits = sorted(set(str(s) for s in splits) - allowed_splits)
        if bad_splits:
            raise ValueError(
                f"learned_utility.splits must be subset of {sorted(allowed_splits)}, got unknown {bad_splits}"
            )

        predictors = learned_cfg.get("predictors", ["linear_regressor", "mlp_regressor"])
        if not isinstance(predictors, list) or not predictors:
            raise ValueError("learned_utility.predictors must be a non-empty list")
        allowed_predictors = {"linear_regressor", "mlp_regressor", "metadata_only_regressor"}
        bad_predictors = sorted(set(str(p) for p in predictors) - allowed_predictors)
        if bad_predictors:
            raise ValueError(
                "learned_utility.predictors must be subset of "
                f"{sorted(allowed_predictors)}, got unknown {bad_predictors}"
            )

        target_cfg = learned_cfg.get("target", {})
        if target_cfg is not None and not isinstance(target_cfg, dict):
            raise ValueError("learned_utility.target must be a dictionary when provided")
        target_name = str((target_cfg or {}).get("name", "nelbo")).strip().lower()
        if target_name != "nelbo":
            raise ValueError("learned_utility.target.name must be 'nelbo'")
        target_norm = str((target_cfg or {}).get("normalization", "per_query_domain_zscore")).strip().lower()
        if target_norm != "per_query_domain_zscore":
            raise ValueError("learned_utility.target.normalization must be 'per_query_domain_zscore'")
        norm_source = str((target_cfg or {}).get("normalization_stats_source", "train_fold_only")).strip().lower()
        if norm_source != "train_fold_only":
            raise ValueError("learned_utility.target.normalization_stats_source must be 'train_fold_only'")
        eval_scale = str((target_cfg or {}).get("eval_scale", "raw_nelbo")).strip().lower()
        if eval_scale != "raw_nelbo":
            raise ValueError("learned_utility.target.eval_scale must be 'raw_nelbo'")

        pair_features = learned_cfg.get("pair_features", {})
        if pair_features is not None and not isinstance(pair_features, dict):
            raise ValueError("learned_utility.pair_features must be a dictionary when provided")
        include_sample_embedding = (pair_features or {}).get("include_sample_embedding", True)
        _ensure_bool(include_sample_embedding, "learned_utility.pair_features.include_sample_embedding")
        if not include_sample_embedding:
            raise ValueError("learned_utility.pair_features.include_sample_embedding must be true")
        expert_id_encoding = str((pair_features or {}).get("expert_id_encoding", "one_hot")).strip().lower()
        if expert_id_encoding != "one_hot":
            raise ValueError("learned_utility.pair_features.expert_id_encoding must be 'one_hot'")
        for key in ["include_metadata_features", "include_domain_stats"]:
            _ensure_bool((pair_features or {}).get(key, False), f"learned_utility.pair_features.{key}")

        scoring_cfg = learned_cfg.get("scoring", {})
        if scoring_cfg is not None and not isinstance(scoring_cfg, dict):
            raise ValueError("learned_utility.scoring must be a dictionary when provided")
        granularity = str((scoring_cfg or {}).get("granularity", "sample_expert_pair")).strip().lower()
        if granularity != "sample_expert_pair":
            raise ValueError("learned_utility.scoring.granularity must be 'sample_expert_pair'")
        _ensure_bool(
            (scoring_cfg or {}).get("enforce_full_expert_scoring", True),
            "learned_utility.scoring.enforce_full_expert_scoring",
        )
        pair_batch_size = int((scoring_cfg or {}).get("pair_batch_size", 4096))
        if pair_batch_size <= 0:
            raise ValueError("learned_utility.scoring.pair_batch_size must be > 0")

        latent_cmp = learned_cfg.get("latent_comparator", {})
        if latent_cmp is not None and not isinstance(latent_cmp, dict):
            raise ValueError("learned_utility.latent_comparator must be a dictionary when provided")
        primary_cmp = str((latent_cmp or {}).get("primary", "wasserstein")).strip().lower()
        if primary_cmp != "wasserstein":
            raise ValueError("learned_utility.latent_comparator.primary must be 'wasserstein'")
        diagnostics = (latent_cmp or {}).get("diagnostics", ["centroid", "gaussian_kl"])
        if not isinstance(diagnostics, list):
            raise ValueError("learned_utility.latent_comparator.diagnostics must be a list")
        allowed_diag = {"centroid", "gaussian_kl", "wasserstein"}
        bad_diag = sorted(set(str(v) for v in diagnostics) - allowed_diag)
        if bad_diag:
            raise ValueError(
                "learned_utility.latent_comparator.diagnostics must be subset of "
                f"{sorted(allowed_diag)}, got unknown {bad_diag}"
            )

        leakage_cfg = learned_cfg.get("leakage_guards", {})
        if leakage_cfg is not None and not isinstance(leakage_cfg, dict):
            raise ValueError("learned_utility.leakage_guards must be a dictionary when provided")
        _ensure_bool(
            (leakage_cfg or {}).get("require_zero_query_domain_overlap", True),
            "learned_utility.leakage_guards.require_zero_query_domain_overlap",
        )
        _ensure_bool(
            (leakage_cfg or {}).get("forbid_heldout_oracle_labels_in_training", True),
            "learned_utility.leakage_guards.forbid_heldout_oracle_labels_in_training",
        )

        winner_cfg = learned_cfg.get("winner_rule", {})
        if winner_cfg is not None and not isinstance(winner_cfg, dict):
            raise ValueError("learned_utility.winner_rule must be a dictionary when provided")
        primary_metric = str((winner_cfg or {}).get("primary_metric", "mean_oracle_gap_pct")).strip().lower()
        if primary_metric != "mean_oracle_gap_pct":
            raise ValueError("learned_utility.winner_rule.primary_metric must be 'mean_oracle_gap_pct'")
        tie_breakers = (winner_cfg or {}).get("tie_breakers", ["top1_oracle_hit", "spearman_with_oracle"])
        if not isinstance(tie_breakers, list) or not tie_breakers:
            raise ValueError("learned_utility.winner_rule.tie_breakers must be a non-empty list")
        allowed_tie = {"top1_oracle_hit", "spearman_with_oracle"}
        bad_ties = sorted(set(str(v) for v in tie_breakers) - allowed_tie)
        if bad_ties:
            raise ValueError(
                "learned_utility.winner_rule.tie_breakers must be subset of "
                f"{sorted(allowed_tie)}, got unknown {bad_ties}"
            )
        if float((winner_cfg or {}).get("mlp_min_improvement_abs_pct", 1.0)) < 0:
            raise ValueError("learned_utility.winner_rule.mlp_min_improvement_abs_pct must be >= 0")
        if float((winner_cfg or {}).get("max_allowed_seed_regression_pct", 5.0)) < 0:
            raise ValueError("learned_utility.winner_rule.max_allowed_seed_regression_pct must be >= 0")

        artifacts_cfg = learned_cfg.get("artifacts", {})
        if artifacts_cfg is not None and not isinstance(artifacts_cfg, dict):
            raise ValueError("learned_utility.artifacts must be a dictionary when provided")
        for key in [
            "save_pair_predictions_csv",
            "save_sample_selection_csv",
            "save_domain_breakdown_csv",
            "save_training_fold_manifest_json",
        ]:
            _ensure_bool((artifacts_cfg or {}).get(key, True), f"learned_utility.artifacts.{key}")

        backbone = str(cfg.get("features", {}).get("backbone_type", "")).strip().lower()
        if backbone != "resnet50":
            raise ValueError("features.backbone_type must be 'resnet50' for learned_utility protocol lock")

    latent_cfg = cfg.get("latent_compatibility")
    if latent_cfg is not None:
        if not isinstance(latent_cfg, dict):
            raise ValueError("latent_compatibility must be a dictionary when provided")

        metrics = latent_cfg.get("metrics", ["centroid", "wasserstein", "gaussian_kl"])
        if not isinstance(metrics, list) or not metrics:
            raise ValueError("latent_compatibility.metrics must be a non-empty list")
        allowed_metrics = {"centroid", "wasserstein", "gaussian_kl"}
        unknown_metrics = sorted(set(str(m) for m in metrics) - allowed_metrics)
        if unknown_metrics:
            raise ValueError(
                f"latent_compatibility.metrics must be subset of {sorted(allowed_metrics)}, got unknown {unknown_metrics}"
            )

        similarity_transform = str(latent_cfg.get("similarity_transform", "exp_neg")).strip()
        if similarity_transform != "exp_neg":
            raise ValueError("latent_compatibility.similarity_transform must be 'exp_neg'")

        routing_granularity = str(latent_cfg.get("routing_granularity", "sample")).strip().lower()
        if routing_granularity not in {"domain", "sample"}:
            raise ValueError("latent_compatibility.routing_granularity must be one of ['domain', 'sample']")

        splits = latent_cfg.get("splits", ["test"])
        if not isinstance(splits, list) or not splits:
            raise ValueError("latent_compatibility.splits must be a non-empty list")
        allowed_splits = {"train", "val", "test"}
        bad_splits = sorted(set(str(s) for s in splits) - allowed_splits)
        if bad_splits:
            raise ValueError(
                f"latent_compatibility.splits must be subset of {sorted(allowed_splits)}, got unknown {bad_splits}"
            )

        min_samples = int(latent_cfg.get("min_samples_per_domain", 50))
        if min_samples <= 0:
            raise ValueError("latent_compatibility.min_samples_per_domain must be > 0")

        cov_reg = float(latent_cfg.get("covariance_regularization_lambda", 1e-4))
        if cov_reg <= 0:
            raise ValueError("latent_compatibility.covariance_regularization_lambda must be > 0")

        similarity_cfg = latent_cfg.get("similarity", {})
        if similarity_cfg is not None and not isinstance(similarity_cfg, dict):
            raise ValueError("latent_compatibility.similarity must be a dictionary when provided")
        scale_floor = float((similarity_cfg or {}).get("scale_floor", 1e-8))
        if scale_floor <= 0:
            raise ValueError("latent_compatibility.similarity.scale_floor must be > 0")

        scale_policy = str((similarity_cfg or {}).get("scale_policy", latent_cfg.get("scale_policy", "median_off_diagonal"))).strip()
        if scale_policy != "median_off_diagonal":
            raise ValueError("latent_compatibility similarity scale policy must be 'median_off_diagonal'")

        wasserstein_cfg = latent_cfg.get("wasserstein", {})
        if wasserstein_cfg is not None and not isinstance(wasserstein_cfg, dict):
            raise ValueError("latent_compatibility.wasserstein must be a dictionary when provided")
        eigen_floor = float((wasserstein_cfg or {}).get("eigenvalue_floor", 1e-10))
        if eigen_floor <= 0:
            raise ValueError("latent_compatibility.wasserstein.eigenvalue_floor must be > 0")

        verification_cfg = latent_cfg.get("verification", {})
        if verification_cfg is not None and not isinstance(verification_cfg, dict):
            raise ValueError("latent_compatibility.verification must be a dictionary when provided")
        symmetry_atol = float((verification_cfg or {}).get("symmetry_atol", 1e-6))
        symmetry_rtol = float((verification_cfg or {}).get("symmetry_rtol", 1e-5))
        diag_opt_tol = float((verification_cfg or {}).get("diag_opt_tol", 1e-6))
        if symmetry_atol < 0 or symmetry_rtol < 0 or diag_opt_tol < 0:
            raise ValueError("latent_compatibility verification tolerances must be >= 0")

        umap_cfg = latent_cfg.get("umap", {})
        if umap_cfg is not None and not isinstance(umap_cfg, dict):
            raise ValueError("latent_compatibility.umap must be a dictionary when provided")
        max_points = int((umap_cfg or {}).get("max_points", 5000))
        if max_points <= 0:
            raise ValueError("latent_compatibility.umap.max_points must be > 0")

        for bool_key in [
            "persist_raw_and_transformed_scores",
            "compute_oracle_alignment",
            "include_metadata_oracle_proxy_table",
        ]:
            if bool_key in latent_cfg and not isinstance(latent_cfg.get(bool_key), bool):
                raise ValueError(f"latent_compatibility.{bool_key} must be a boolean when provided")

        learned_cmp = latent_cfg.get("learned_comparison", {})
        if learned_cmp is not None and not isinstance(learned_cmp, dict):
            raise ValueError("latent_compatibility.learned_comparison must be a dictionary when provided")
        strict_context_match = bool((learned_cmp or {}).get("strict_context_match", True))
        _ = strict_context_match

        sample_level_cfg = latent_cfg.get("sample_level_routing", {})
        if sample_level_cfg is not None and not isinstance(sample_level_cfg, dict):
            raise ValueError("latent_compatibility.sample_level_routing must be a dictionary when provided")
        max_samples = int((sample_level_cfg or {}).get("max_samples", 0))
        timing_every = int((sample_level_cfg or {}).get("timing_every", 0))
        if max_samples < 0:
            raise ValueError("latent_compatibility.sample_level_routing.max_samples must be >= 0")
        if timing_every < 0:
            raise ValueError("latent_compatibility.sample_level_routing.timing_every must be >= 0")

        coverage_gates = latent_cfg.get("coverage_gates", {})
        if coverage_gates is not None and not isinstance(coverage_gates, dict):
            raise ValueError("latent_compatibility.coverage_gates must be a dictionary when provided")
        for gate_key in [
            "require_complete_loqdo_folds",
            "require_all_query_domains_present",
            "exclude_partial_metric_rows",
        ]:
            gate_value = (coverage_gates or {}).get(gate_key, True)
            if not isinstance(gate_value, bool):
                raise ValueError(f"latent_compatibility.coverage_gates.{gate_key} must be a boolean")

        thresholds = latent_cfg.get("acceptance_thresholds", {})
        if thresholds is not None and not isinstance(thresholds, dict):
            raise ValueError("latent_compatibility.acceptance_thresholds must be a dictionary when provided")

        thresholds_enabled = (thresholds or {}).get("enabled", True)
        if not isinstance(thresholds_enabled, bool):
            raise ValueError("latent_compatibility.acceptance_thresholds.enabled must be a boolean")

        strong = (thresholds or {}).get("strong", {})
        if strong is not None and not isinstance(strong, dict):
            raise ValueError("latent_compatibility.acceptance_thresholds.strong must be a dictionary when provided")
        spearman_uplift_gt = float((strong or {}).get("spearman_uplift_gt", 0.10))
        top1_uplift_gte = float((strong or {}).get("top1_uplift_gte", 0.25))
        oracle_gap_reduction_gt_pct = float((strong or {}).get("oracle_gap_reduction_gt_pct", 10.0))
        min_backbone_fraction = float((strong or {}).get("min_backbone_fraction", 0.67))
        if min_backbone_fraction <= 0:
            raise ValueError("latent_compatibility.acceptance_thresholds.strong.min_backbone_fraction must be > 0")
        expected_backbones = (strong or {}).get("expected_backbones", [])
        if expected_backbones is not None:
            if not isinstance(expected_backbones, list):
                raise ValueError("latent_compatibility.acceptance_thresholds.strong.expected_backbones must be a list")
            for backbone in expected_backbones:
                if not str(backbone).strip():
                    raise ValueError(
                        "latent_compatibility.acceptance_thresholds.strong.expected_backbones must contain non-empty backbone names"
                    )
        _ = spearman_uplift_gt, top1_uplift_gte, oracle_gap_reduction_gt_pct

        disallow_guardrail_breach = (strong or {}).get("disallow_any_backbone_guardrail_breach", True)
        if not isinstance(disallow_guardrail_breach, bool):
            raise ValueError(
                "latent_compatibility.acceptance_thresholds.strong.disallow_any_backbone_guardrail_breach must be a boolean"
            )

        non_inferiority = (thresholds or {}).get("non_inferiority", {})
        if non_inferiority is not None and not isinstance(non_inferiority, dict):
            raise ValueError(
                "latent_compatibility.acceptance_thresholds.non_inferiority must be a dictionary when provided"
            )
        max_gap_worsening = float((non_inferiority or {}).get("max_oracle_gap_worsening_pct", 5.0))
        if max_gap_worsening < 0:
            raise ValueError(
                "latent_compatibility.acceptance_thresholds.non_inferiority.max_oracle_gap_worsening_pct must be >= 0"
            )

    return cfg
