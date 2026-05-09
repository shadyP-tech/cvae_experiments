from __future__ import annotations

from typing import Any, Dict


REQUIRED_TOP_LEVEL = ["seed", "data", "features", "model", "training", "routing"]
SUPPORTED_EXPERIMENT_MODES = {"hybrid_ablation", "learned_utility_routing"}
QUARANTINED_EXPERIMENT_MODES = {"legacy_routed_cvae", "latent_compatibility"}


def _ensure_bool(value: object, name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")


def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a dictionary.")

    missing = [k for k in REQUIRED_TOP_LEVEL if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")

    experiment_cfg = cfg.get("experiment", {})
    if not isinstance(experiment_cfg, dict):
        raise ValueError("experiment must be a dictionary")
    experiment_mode = str(experiment_cfg.get("mode", "")).strip()
    if not experiment_mode:
        raise ValueError("experiment.mode is required; implicit legacy defaults are quarantined")
    if experiment_mode in QUARANTINED_EXPERIMENT_MODES:
        raise ValueError(
            f"experiment.mode '{experiment_mode}' is quarantined and cannot be used for thesis-facing runs"
        )
    if experiment_mode not in SUPPORTED_EXPERIMENT_MODES:
        raise ValueError(
            f"experiment.mode must be one of {sorted(SUPPORTED_EXPERIMENT_MODES)}, got: {experiment_mode}"
        )
    experiment_name = str((experiment_cfg or {}).get("name", "")).strip().lower()
    is_response_routing_protocol = experiment_name == "learned_utility_response_routing_v1"
    is_support_response_routing_protocol = experiment_name == "learned_utility_support_response_routing_v1"
    is_learned_utility_v2 = experiment_name in {
        "learned_utility_routing_v2",
        "learned_utility_routing_safe_v2",
    }

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

    data_cfg = cfg.get("data", {})
    domain_field = data_cfg.get("domain_field")
    if domain_field is not None and not str(domain_field).strip():
        raise ValueError("data.domain_field must be non-empty when provided")
    legacy_domain_field_alias = data_cfg.get("legacy_domain_field_alias")
    if legacy_domain_field_alias is not None and not str(legacy_domain_field_alias).strip():
        raise ValueError("data.legacy_domain_field_alias must be non-empty when provided")
    dataset_domain_semantics = data_cfg.get("dataset_domain_semantics")
    if dataset_domain_semantics is not None and not str(dataset_domain_semantics).strip():
        raise ValueError("data.dataset_domain_semantics must be non-empty when provided")
    if is_support_response_routing_protocol:
        if str(data_cfg.get("dataset_type", "")).strip().lower() != "camelyon17":
            raise ValueError(
                "data.dataset_type must be 'camelyon17' for learned_utility_support_response_routing_v1"
            )
        if str(data_cfg.get("domain_field", "")).strip().lower() != "center":
            raise ValueError(
                "data.domain_field must be 'center' for learned_utility_support_response_routing_v1"
            )
        if str(data_cfg.get("legacy_domain_field_alias", "")).strip().lower() != "magnification":
            raise ValueError(
                "data.legacy_domain_field_alias must be 'magnification' for learned_utility_support_response_routing_v1"
            )
        if str(data_cfg.get("dataset_domain_semantics", "")).strip().lower() != "camelyon17_center":
            raise ValueError(
                "data.dataset_domain_semantics must be 'camelyon17_center' "
                "for learned_utility_support_response_routing_v1"
            )
    split_cap_profile = str(data_cfg.get("split_cap_profile", "legacy")).strip().lower()
    allowed_split_cap_profiles = {"legacy", "development", "final", "custom"}
    if split_cap_profile not in allowed_split_cap_profiles:
        raise ValueError(
            "data.split_cap_profile must be one of "
            f"{sorted(allowed_split_cap_profiles)}, got: {split_cap_profile}"
        )

    split_caps_cfg = data_cfg.get("split_domain_caps")
    fixed_split_caps_cfg = data_cfg.get("fixed_split_caps")
    if split_caps_cfg is not None and fixed_split_caps_cfg is not None:
        raise ValueError("Use only one of data.split_domain_caps or data.fixed_split_caps, not both")
    resolved_split_caps_cfg = split_caps_cfg if split_caps_cfg is not None else fixed_split_caps_cfg

    if resolved_split_caps_cfg is not None:
        if not isinstance(resolved_split_caps_cfg, dict):
            raise ValueError("data.split_domain_caps/data.fixed_split_caps must be a dictionary when provided")
        expected_keys = {"train", "val", "test"}
        actual_keys = set(str(k) for k in resolved_split_caps_cfg.keys())
        if actual_keys != expected_keys:
            raise ValueError(
                "data.split_domain_caps/data.fixed_split_caps must define exactly train/val/test keys"
            )
        for key in sorted(expected_keys):
            value = int(resolved_split_caps_cfg[key])
            if value < 0:
                raise ValueError(
                    f"data.split_domain_caps.{key} must be >= 0 (got {value})"
                )

    if split_cap_profile in {"development", "final"} and resolved_split_caps_cfg is not None:
        profile_caps = {
            "development": {"train": 250, "val": 100, "test": 200},
            "final": {"train": 1000, "val": 250, "test": 1000},
        }[split_cap_profile]
        for split_name, expected_cap in profile_caps.items():
            configured = int(resolved_split_caps_cfg[split_name])
            if configured != expected_cap:
                raise ValueError(
                    f"data.split_cap_profile='{split_cap_profile}' requires "
                    f"{split_name}={expected_cap}, got {configured}"
                )

    if int(cfg["training"]["batch_size"]) <= 0:
        raise ValueError("training.batch_size must be > 0")
    if int(cfg["training"]["epochs"]) <= 0:
        raise ValueError("training.epochs must be > 0")

    model_cfg = cfg.get("model", {})
    if not isinstance(model_cfg, dict):
        raise ValueError("model must be a dictionary")
    if int(model_cfg.get("hidden_dim", 0)) <= 0:
        raise ValueError("model.hidden_dim must be > 0")
    if int(model_cfg.get("latent_dim", 0)) <= 0:
        raise ValueError("model.latent_dim must be > 0")

    conditioning_cfg = model_cfg.get("conditioning", {})
    if conditioning_cfg is not None and not isinstance(conditioning_cfg, dict):
        raise ValueError("model.conditioning must be a dictionary when provided")

    conditioning_enabled = bool((conditioning_cfg or {}).get("enabled", False))
    _ensure_bool((conditioning_cfg or {}).get("enabled", False), "model.conditioning.enabled")

    metadata_mode = str((conditioning_cfg or {}).get("metadata_mode", "domain_id")).strip().lower()
    if metadata_mode != "domain_id":
        raise ValueError("model.conditioning.metadata_mode must be 'domain_id' in v1")

    encoding = str((conditioning_cfg or {}).get("encoding", "one_hot")).strip().lower()
    if encoding != "one_hot":
        raise ValueError("model.conditioning.encoding must be 'one_hot' in v1")

    _ensure_bool((conditioning_cfg or {}).get("inject_encoder", True), "model.conditioning.inject_encoder")
    _ensure_bool((conditioning_cfg or {}).get("inject_decoder", True), "model.conditioning.inject_decoder")
    inject_encoder = bool((conditioning_cfg or {}).get("inject_encoder", True))
    inject_decoder = bool((conditioning_cfg or {}).get("inject_decoder", True))
    if conditioning_enabled and not inject_encoder:
        raise ValueError("model.conditioning.inject_encoder must be true when conditioning is enabled")
    if conditioning_enabled and not inject_decoder:
        raise ValueError("model.conditioning.inject_decoder must be true when conditioning is enabled")

    metadata_fields = (conditioning_cfg or {}).get("metadata_fields", ["domain_id"])
    if metadata_fields is not None:
        if not isinstance(metadata_fields, list):
            raise ValueError("model.conditioning.metadata_fields must be a list when provided")
        invalid_fields = [str(x) for x in metadata_fields if str(x).strip().lower() != "domain_id"]
        if invalid_fields:
            raise ValueError("model.conditioning.metadata_fields supports only ['domain_id'] in v1")

    metadata_constraint_cfg = model_cfg.get("metadata_constraint", {})
    if metadata_constraint_cfg is not None and not isinstance(metadata_constraint_cfg, dict):
        raise ValueError("model.metadata_constraint must be a dictionary when provided")

    _ensure_bool(
        (metadata_constraint_cfg or {}).get("enabled", False),
        "model.metadata_constraint.enabled",
    )
    metadata_constraint_enabled = bool((metadata_constraint_cfg or {}).get("enabled", False))

    metadata_constraint_variant = str((metadata_constraint_cfg or {}).get("variant", "aux_head")).strip().lower()
    allowed_constraint_variants = {"aux_head", "conditional_prior"}
    if metadata_constraint_variant not in allowed_constraint_variants:
        raise ValueError(
            "model.metadata_constraint.variant must be one of "
            f"{sorted(allowed_constraint_variants)}, got: {metadata_constraint_variant}"
        )

    metadata_constraint_aux_weight = float((metadata_constraint_cfg or {}).get("aux_weight", 0.0))
    if metadata_constraint_aux_weight < 0:
        raise ValueError("model.metadata_constraint.aux_weight must be >= 0")

    _ensure_bool(
        (metadata_constraint_cfg or {}).get("use_mu", True),
        "model.metadata_constraint.use_mu",
    )

    aux_head_hidden_dim = int((metadata_constraint_cfg or {}).get("head_hidden_dim", 0))
    if aux_head_hidden_dim < 0:
        raise ValueError("model.metadata_constraint.head_hidden_dim must be >= 0")

    prior_hidden_dim = int((metadata_constraint_cfg or {}).get("prior_hidden_dim", 0))
    if prior_hidden_dim < 0:
        raise ValueError("model.metadata_constraint.prior_hidden_dim must be >= 0")

    prior_logvar_min = float((metadata_constraint_cfg or {}).get("prior_logvar_min", -6.0))
    prior_logvar_max = float((metadata_constraint_cfg or {}).get("prior_logvar_max", 2.0))
    if prior_logvar_min > prior_logvar_max:
        raise ValueError("model.metadata_constraint.prior_logvar_min must be <= prior_logvar_max")

    if metadata_constraint_enabled and not conditioning_enabled:
        raise ValueError("model.metadata_constraint.enabled requires model.conditioning.enabled=true in v1")

    protocol_cfg = cfg.get("legacy_protocol")
    if protocol_cfg is not None:
        if not isinstance(protocol_cfg, dict):
            raise ValueError("legacy_protocol must be a dictionary when provided")
        seeds = protocol_cfg.get("seeds", [11, 42, 73])
        if not isinstance(seeds, list) or not seeds:
            raise ValueError("legacy_protocol.seeds must be a non-empty list")
        if any(int(s) < 0 for s in seeds):
            raise ValueError("legacy_protocol.seeds must contain non-negative integers")

        gates = protocol_cfg.get("gates", {})
        if gates is not None and not isinstance(gates, dict):
            raise ValueError("legacy_protocol.gates must be a dictionary when provided")
        e1_cfg = (gates or {}).get("e1", {})
        e2_cfg = (gates or {}).get("e2", {})
        e3_cfg = (gates or {}).get("e3", {})
        if e1_cfg is not None and not isinstance(e1_cfg, dict):
            raise ValueError("legacy_protocol.gates.e1 must be a dictionary when provided")
        if e2_cfg is not None and not isinstance(e2_cfg, dict):
            raise ValueError("legacy_protocol.gates.e2 must be a dictionary when provided")
        if e3_cfg is not None and not isinstance(e3_cfg, dict):
            raise ValueError("legacy_protocol.gates.e3 must be a dictionary when provided")

        e1_median_rel_delta_max = float((e1_cfg or {}).get("median_relative_delta_max", -0.03))
        _ = e1_median_rel_delta_max
        e2_top1_min = float((e2_cfg or {}).get("top1_uplift_min", 0.05))
        e2_spearman_min = float((e2_cfg or {}).get("spearman_uplift_min", 0.05))
        _ = e2_top1_min, e2_spearman_min
        e3_rel_gap_reduction_min = float((e3_cfg or {}).get("relative_gap_reduction_min", 0.30))
        e3_abs_norm_gap_max = float((e3_cfg or {}).get("abs_normalized_gap_median_max", 0.05))
        if e3_rel_gap_reduction_min < 0:
            raise ValueError("legacy_protocol.gates.e3.relative_gap_reduction_min must be >= 0")
        if e3_abs_norm_gap_max < 0:
            raise ValueError("legacy_protocol.gates.e3.abs_normalized_gap_median_max must be >= 0")

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

    feature_extractor_name = features_cfg.get("feature_extractor_name")
    if feature_extractor_name is not None:
        feature_extractor_name_norm = str(feature_extractor_name).strip().lower()
        if not feature_extractor_name_norm:
            raise ValueError("features.feature_extractor_name must be non-empty when provided")
        if feature_extractor_name_norm not in allowed_backbones:
            raise ValueError(
                "features.feature_extractor_name must be one of "
                f"{sorted(allowed_backbones)}, got: {feature_extractor_name_norm}"
            )

    feature_extractor_checkpoint = features_cfg.get("feature_extractor_checkpoint")
    if feature_extractor_checkpoint is not None:
        checkpoint_norm = str(feature_extractor_checkpoint).strip()
        if not checkpoint_norm:
            raise ValueError("features.feature_extractor_checkpoint must be non-empty when provided")

    feature_extractor_layer = features_cfg.get("feature_extractor_layer")
    if feature_extractor_layer is not None:
        feature_extractor_layer_norm = str(feature_extractor_layer).strip().lower()
        allowed_layers = {"final_norm_cls", "final_norm_patch_mean", "prenorm_cls"}
        if feature_extractor_layer_norm not in allowed_layers:
            raise ValueError(
                "features.feature_extractor_layer must be one of "
                f"{sorted(allowed_layers)}, got: {feature_extractor_layer_norm}"
            )

    embedding_pooling = features_cfg.get("embedding_pooling")
    if embedding_pooling is not None:
        embedding_pooling_norm = str(embedding_pooling).strip().lower()
        allowed_pooling = {"cls_token", "patch_mean"}
        if embedding_pooling_norm not in allowed_pooling:
            raise ValueError(
                "features.embedding_pooling must be one of "
                f"{sorted(allowed_pooling)}, got: {embedding_pooling_norm}"
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

    hybrid_cfg = cfg.get("hybrid")
    if hybrid_cfg is not None:
        if not isinstance(hybrid_cfg, dict):
            raise ValueError("hybrid must be a dictionary when provided")

        aggregation_mode = str(hybrid_cfg.get("aggregation_mode", "top1_hard")).strip().lower()
        allowed_modes = {"top1_hard", "topk2_uniform", "topk3_uniform", "soft_all_softmax"}
        if aggregation_mode not in allowed_modes:
            raise ValueError(
                f"hybrid.aggregation_mode must be one of {sorted(allowed_modes)}, got: {aggregation_mode}"
            )

        topk = int(hybrid_cfg.get("topk_k", 2))
        if topk <= 0:
            raise ValueError("hybrid.topk_k must be > 0")

        aggregation_temperature = float(hybrid_cfg.get("aggregation_temperature", 1.0))
        if aggregation_temperature <= 0:
            raise ValueError("hybrid.aggregation_temperature must be > 0")

        if aggregation_mode == "topk2_uniform" and topk != 2:
            raise ValueError("hybrid.topk_k must be 2 when hybrid.aggregation_mode is 'topk2_uniform'")
        if aggregation_mode == "topk3_uniform" and topk != 3:
            raise ValueError("hybrid.topk_k must be 3 when hybrid.aggregation_mode is 'topk3_uniform'")

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
        allowed_predictors = {
            "linear_regressor",
            "mlp_regressor",
            "metadata_only_regressor",
            "pairwise_ranker",
        }
        bad_predictors = sorted(set(str(p) for p in predictors) - allowed_predictors)
        if bad_predictors:
            raise ValueError(
                "learned_utility.predictors must be subset of "
                f"{sorted(allowed_predictors)}, got unknown {bad_predictors}"
            )

        pairwise_params = (learned_cfg.get("predictor_params", {}) or {}).get("pairwise_ranker", {})
        if pairwise_params is not None and not isinstance(pairwise_params, dict):
            raise ValueError("learned_utility.predictor_params.pairwise_ranker must be a dictionary when provided")
        if "pairwise_ranker" in predictors:
            hidden_dim = int((pairwise_params or {}).get("hidden_dim", 128))
            if hidden_dim <= 0:
                raise ValueError("learned_utility.predictor_params.pairwise_ranker.hidden_dim must be > 0")
            epochs = int((pairwise_params or {}).get("epochs", 40))
            if epochs <= 0:
                raise ValueError("learned_utility.predictor_params.pairwise_ranker.epochs must be > 0")
            lr = float((pairwise_params or {}).get("lr", 1e-3))
            if lr <= 0:
                raise ValueError("learned_utility.predictor_params.pairwise_ranker.lr must be > 0")
            batch_size = int((pairwise_params or {}).get("batch_size", 2048))
            if batch_size <= 0:
                raise ValueError("learned_utility.predictor_params.pairwise_ranker.batch_size must be > 0")

            near_tie_delta = float((pairwise_params or {}).get("near_tie_delta", 0.0))
            if near_tie_delta < 0:
                raise ValueError("learned_utility.predictor_params.pairwise_ranker.near_tie_delta must be >= 0")
            hard_frac = float((pairwise_params or {}).get("hard_pair_fraction", 0.5))
            rand_frac = float((pairwise_params or {}).get("random_pair_fraction", 0.5))
            if hard_frac < 0 or rand_frac < 0:
                raise ValueError(
                    "learned_utility.predictor_params.pairwise_ranker hard/random pair fractions must be >= 0"
                )
            if (hard_frac + rand_frac) <= 0:
                raise ValueError(
                    "learned_utility.predictor_params.pairwise_ranker hard/random pair fractions must sum to > 0"
                )

            max_pairs_per_sample = int((pairwise_params or {}).get("max_pairs_per_sample", 12))
            if max_pairs_per_sample <= 0:
                raise ValueError(
                    "learned_utility.predictor_params.pairwise_ranker.max_pairs_per_sample must be > 0"
                )
            max_pairs_per_domain = int((pairwise_params or {}).get("max_pairs_per_domain", 5000))
            if max_pairs_per_domain <= 0:
                raise ValueError(
                    "learned_utility.predictor_params.pairwise_ranker.max_pairs_per_domain must be > 0"
                )
            loss_type = str((pairwise_params or {}).get("loss_type", "hinge")).strip().lower()
            if loss_type != "hinge":
                raise ValueError("learned_utility.predictor_params.pairwise_ranker.loss_type must be 'hinge'")
            margin = float((pairwise_params or {}).get("margin", 1.0))
            if margin < 0:
                raise ValueError("learned_utility.predictor_params.pairwise_ranker.margin must be >= 0")
            _ensure_bool(
                (pairwise_params or {}).get("run_ablations", True),
                "learned_utility.predictor_params.pairwise_ranker.run_ablations",
            )

        hybrid_scoring = learned_cfg.get("hybrid_scoring", {})
        if hybrid_scoring is not None and not isinstance(hybrid_scoring, dict):
            raise ValueError("learned_utility.hybrid_scoring must be a dictionary when provided")
        _ensure_bool(
            (hybrid_scoring or {}).get("enabled", False),
            "learned_utility.hybrid_scoring.enabled",
        )
        alphas = (hybrid_scoring or {}).get("alphas", [i / 10.0 for i in range(11)])
        if not isinstance(alphas, list) or not alphas:
            raise ValueError("learned_utility.hybrid_scoring.alphas must be a non-empty list")
        for a in alphas:
            a_float = float(a)
            if a_float < 0.0 or a_float > 1.0:
                raise ValueError(f"learned_utility.hybrid_scoring.alphas must be in [0,1], got: {a}")

        allowed_norm = {"per_query_zscore", "per_query_minmax"}
        norm_primary = str((hybrid_scoring or {}).get("normalization_primary", "per_query_zscore")).strip().lower()
        if norm_primary not in allowed_norm:
            raise ValueError(
                "learned_utility.hybrid_scoring.normalization_primary must be one of "
                f"{sorted(allowed_norm)}, got: {norm_primary}"
            )
        norm_sensitivity = str((hybrid_scoring or {}).get("normalization_sensitivity", "per_query_minmax")).strip().lower()
        if norm_sensitivity not in allowed_norm:
            raise ValueError(
                "learned_utility.hybrid_scoring.normalization_sensitivity must be one of "
                f"{sorted(allowed_norm)}, got: {norm_sensitivity}"
            )
        _ensure_bool(
            (hybrid_scoring or {}).get("run_sensitivity", True),
            "learned_utility.hybrid_scoring.run_sensitivity",
        )
        tie_policy = str((hybrid_scoring or {}).get("tie_policy", "stable_expert_index")).strip().lower()
        if tie_policy != "stable_expert_index":
            raise ValueError(
                "learned_utility.hybrid_scoring.tie_policy must be 'stable_expert_index'"
            )
        latent_metric = str((hybrid_scoring or {}).get("latent_metric", "wasserstein")).strip().lower()
        if latent_metric != "wasserstein":
            raise ValueError("learned_utility.hybrid_scoring.latent_metric must be 'wasserstein'")
        accept_cfg = (hybrid_scoring or {}).get("acceptance", {})
        if accept_cfg is not None and not isinstance(accept_cfg, dict):
            raise ValueError("learned_utility.hybrid_scoring.acceptance must be a dictionary when provided")
        min_rank_impr = float((accept_cfg or {}).get("min_mean_rank_improvement_abs", 0.05))
        if min_rank_impr < 0:
            raise ValueError("learned_utility.hybrid_scoring.acceptance.min_mean_rank_improvement_abs must be >= 0")
        min_gap_pct_impr = float((accept_cfg or {}).get("min_mean_oracle_gap_pct_improvement_abs", 0.50))
        if min_gap_pct_impr < 0:
            raise ValueError(
                "learned_utility.hybrid_scoring.acceptance.min_mean_oracle_gap_pct_improvement_abs must be >= 0"
            )
        max_top1_drop = float((accept_cfg or {}).get("max_top1_drop_abs", 0.0))
        if max_top1_drop < 0:
            raise ValueError("learned_utility.hybrid_scoring.acceptance.max_top1_drop_abs must be >= 0")

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

        residual_cfg = learned_cfg.get("residual_routing", {})
        if residual_cfg is not None and not isinstance(residual_cfg, dict):
            raise ValueError("learned_utility.residual_routing must be a dictionary when provided")
        residual_cfg = residual_cfg or {}
        _ensure_bool(
            residual_cfg.get("enabled", False),
            "learned_utility.residual_routing.enabled",
        )
        residual_policy_version = str(
            residual_cfg.get("residual_policy_version", "metadata_residual_v1")
        ).strip()
        allowed_residual_policies = {"metadata_residual_v1", "metadata_residual_safe_override_v2"}
        if residual_policy_version not in allowed_residual_policies:
            raise ValueError(
                "learned_utility.residual_routing.residual_policy_version must be one of "
                f"{sorted(allowed_residual_policies)}"
            )
        residual_models = residual_cfg.get("models", ["ridge"])
        if not isinstance(residual_models, list) or not residual_models:
            raise ValueError("learned_utility.residual_routing.models must be a non-empty list")
        bad_residual_models = sorted(set(str(v).strip().lower() for v in residual_models) - {"ridge"})
        if bad_residual_models:
            raise ValueError(
                "learned_utility.residual_routing.models must be subset of ['ridge'], "
                f"got unknown {bad_residual_models}"
            )
        residual_thresholds = residual_cfg.get("thresholds", [0, 0.01, 0.05, 0.10, 0.25, 0.50, "inf"])
        if not isinstance(residual_thresholds, list) or not residual_thresholds:
            raise ValueError("learned_utility.residual_routing.thresholds must be a non-empty list")
        has_inf_threshold = False
        for threshold in residual_thresholds:
            if isinstance(threshold, str) and threshold.strip().lower() == "inf":
                has_inf_threshold = True
                continue
            if float(threshold) < 0.0:
                raise ValueError("learned_utility.residual_routing.thresholds must be >= 0 or 'inf'")
        if not has_inf_threshold:
            raise ValueError("learned_utility.residual_routing.thresholds must include 'inf'")
        residual_feature_sets = residual_cfg.get("feature_sets", ["minimal", "latent", "calibrated"])
        if not isinstance(residual_feature_sets, list) or not residual_feature_sets:
            raise ValueError("learned_utility.residual_routing.feature_sets must be a non-empty list")
        allowed_residual_feature_sets = {"minimal", "latent", "calibrated"}
        bad_feature_sets = sorted(set(str(v).strip().lower() for v in residual_feature_sets) - allowed_residual_feature_sets)
        if bad_feature_sets:
            raise ValueError(
                "learned_utility.residual_routing.feature_sets must be subset of "
                f"{sorted(allowed_residual_feature_sets)}, got unknown {bad_feature_sets}"
            )
        for key in ["adoption_feature_sets", "diagnostic_feature_sets"]:
            configured = residual_cfg.get(key)
            if configured is None:
                continue
            if not isinstance(configured, list):
                raise ValueError(f"learned_utility.residual_routing.{key} must be a list when provided")
            bad = sorted(set(str(v).strip().lower() for v in configured) - allowed_residual_feature_sets)
            if bad:
                raise ValueError(
                    f"learned_utility.residual_routing.{key} must be subset of "
                    f"{sorted(allowed_residual_feature_sets)}, got unknown {bad}"
                )
        _ensure_bool(
            residual_cfg.get("allow_calibrated_adoption", False),
            "learned_utility.residual_routing.allow_calibrated_adoption",
        )
        if float(residual_cfg.get("harmful_override_max", 0.05)) < 0.0:
            raise ValueError("learned_utility.residual_routing.harmful_override_max must be >= 0")
        if float(residual_cfg.get("gap_regression_max", 2.0)) < 0.0:
            raise ValueError("learned_utility.residual_routing.gap_regression_max must be >= 0")
        if residual_policy_version == "metadata_residual_safe_override_v2":
            adoption_feature_sets = residual_cfg.get("adoption_feature_sets", ["minimal", "latent"])
            if not isinstance(adoption_feature_sets, list) or not adoption_feature_sets:
                raise ValueError(
                    "learned_utility.residual_routing.adoption_feature_sets must be non-empty for safe v2"
                )
            allow_calibrated = bool(residual_cfg.get("allow_calibrated_adoption", False))
            if not allow_calibrated and "calibrated" in {
                str(v).strip().lower() for v in adoption_feature_sets
            }:
                raise ValueError(
                    "calibrated cannot appear in adoption_feature_sets when allow_calibrated_adoption=false"
                )
        residual_selection_metric = str(
            residual_cfg.get("selection_metric", "validation_safe_gap_then_top1")
        ).strip().lower()
        if residual_selection_metric != "validation_safe_gap_then_top1":
            raise ValueError(
                "learned_utility.residual_routing.selection_metric must be 'validation_safe_gap_then_top1'"
            )
        if float(residual_cfg.get("ridge_l2", 1e-4)) < 0.0:
            raise ValueError("learned_utility.residual_routing.ridge_l2 must be >= 0")

        support_response_cfg = learned_cfg.get("support_response_routing", {})
        if support_response_cfg is not None and not isinstance(support_response_cfg, dict):
            raise ValueError("learned_utility.support_response_routing must be a dictionary when provided")
        support_response_cfg = support_response_cfg or {}
        _ensure_bool(
            support_response_cfg.get("enabled", False),
            "learned_utility.support_response_routing.enabled",
        )
        if support_response_cfg.get("enabled", False):
            support_sizes = support_response_cfg.get("support_sizes", [8, 16, 32])
            if not isinstance(support_sizes, list) or not support_sizes:
                raise ValueError("learned_utility.support_response_routing.support_sizes must be a non-empty list")
            for value in support_sizes:
                if int(value) <= 0:
                    raise ValueError("learned_utility.support_response_routing.support_sizes must be positive")

            support_seeds = support_response_cfg.get("support_seeds", [17, 23])
            if not isinstance(support_seeds, list) or not support_seeds:
                raise ValueError("learned_utility.support_response_routing.support_seeds must be a non-empty list")
            for value in support_seeds:
                if int(value) < 0:
                    raise ValueError("learned_utility.support_response_routing.support_seeds must be non-negative")

            sampling_policies = support_response_cfg.get("sampling_policies", ["random"])
            if not isinstance(sampling_policies, list) or not sampling_policies:
                raise ValueError(
                    "learned_utility.support_response_routing.sampling_policies must be a non-empty list"
                )
            allowed_sampling = {"random", "class_balanced"}
            bad_sampling = sorted(set(str(v).strip().lower() for v in sampling_policies) - allowed_sampling)
            if bad_sampling:
                raise ValueError(
                    "learned_utility.support_response_routing.sampling_policies must be subset of "
                    f"{sorted(allowed_sampling)}, got unknown {bad_sampling}"
                )

            feature_regimes = support_response_cfg.get(
                "feature_regimes",
                ["static_response_indirect", "response_indirect_shuffled"],
            )
            if not isinstance(feature_regimes, list) or not feature_regimes:
                raise ValueError(
                    "learned_utility.support_response_routing.feature_regimes must be a non-empty list"
                )
            allowed_feature_regimes = {
                "static_response_indirect",
                "response_indirect",
                "response_indirect_shuffled",
            }
            normalized_feature_regimes = {str(v).strip().lower() for v in feature_regimes}
            bad_regimes = sorted(normalized_feature_regimes - allowed_feature_regimes)
            if bad_regimes:
                raise ValueError(
                    "learned_utility.support_response_routing.feature_regimes must be subset of "
                    f"{sorted(allowed_feature_regimes)}, got unknown {bad_regimes}"
                )
            primary_feature_regime = str(
                support_response_cfg.get("primary_feature_regime", "static_response_indirect")
            ).strip().lower()
            if primary_feature_regime not in normalized_feature_regimes:
                raise ValueError(
                    "learned_utility.support_response_routing.primary_feature_regime must appear in feature_regimes"
                )
            if primary_feature_regime == "response_indirect_shuffled":
                raise ValueError(
                    "learned_utility.support_response_routing.primary_feature_regime cannot be shuffled control"
                )

            ranker = str(support_response_cfg.get("ranker", "linear_pairwise_ridge")).strip().lower()
            if ranker != "linear_pairwise_ridge":
                raise ValueError(
                    "learned_utility.support_response_routing.ranker must be 'linear_pairwise_ridge'"
                )
            if float(support_response_cfg.get("ridge_l2", 1.0e-3)) < 0.0:
                raise ValueError("learned_utility.support_response_routing.ridge_l2 must be >= 0")
            if int(support_response_cfg.get("num_response_repeats", 8)) <= 0:
                raise ValueError("learned_utility.support_response_routing.num_response_repeats must be > 0")
            tie_policy = str(support_response_cfg.get("tie_policy", "stable_expert_index")).strip().lower()
            if tie_policy != "stable_expert_index":
                raise ValueError(
                    "learned_utility.support_response_routing.tie_policy must be 'stable_expert_index'"
                )
            _ensure_bool(
                support_response_cfg.get("domain_level_aggregation", True),
                "learned_utility.support_response_routing.domain_level_aggregation",
            )
            _ensure_bool(
                support_response_cfg.get("source_leave_pseudo_domain_out_diagnostic", True),
                "learned_utility.support_response_routing.source_leave_pseudo_domain_out_diagnostic",
            )

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

        compatibility_cfg = learned_cfg.get("compatibility_research", {})
        if compatibility_cfg is not None and not isinstance(compatibility_cfg, dict):
            raise ValueError("learned_utility.compatibility_research must be a dictionary when provided")

        floors_cfg = (compatibility_cfg or {}).get("floors", {})
        if floors_cfg is not None and not isinstance(floors_cfg, dict):
            raise ValueError("learned_utility.compatibility_research.floors must be a dictionary when provided")
        _ensure_bool(
            (floors_cfg or {}).get("random_rank_floor", True),
            "learned_utility.compatibility_research.floors.random_rank_floor",
        )
        _ensure_bool(
            (floors_cfg or {}).get("random_score_floor", True),
            "learned_utility.compatibility_research.floors.random_score_floor",
        )

        perm_cfg = (compatibility_cfg or {}).get("permutation_tests", {})
        if perm_cfg is not None and not isinstance(perm_cfg, dict):
            raise ValueError(
                "learned_utility.compatibility_research.permutation_tests must be a dictionary when provided"
            )
        _ensure_bool(
            (perm_cfg or {}).get("expert_label_permutation", True),
            "learned_utility.compatibility_research.permutation_tests.expert_label_permutation",
        )
        _ensure_bool(
            (perm_cfg or {}).get("metadata_permutation", True),
            "learned_utility.compatibility_research.permutation_tests.metadata_permutation",
        )
        repeats = int((perm_cfg or {}).get("repeats", 200))
        if repeats <= 0:
            raise ValueError("learned_utility.compatibility_research.permutation_tests.repeats must be > 0")

        diagnostics_cfg = (compatibility_cfg or {}).get("diagnostics", {})
        if diagnostics_cfg is not None and not isinstance(diagnostics_cfg, dict):
            raise ValueError(
                "learned_utility.compatibility_research.diagnostics must be a dictionary when provided"
            )
        _ensure_bool(
            (diagnostics_cfg or {}).get("save_distribution_plots", True),
            "learned_utility.compatibility_research.diagnostics.save_distribution_plots",
        )

        gate_cfg = (compatibility_cfg or {}).get("gate", {})
        if gate_cfg is not None and not isinstance(gate_cfg, dict):
            raise ValueError("learned_utility.compatibility_research.gate must be a dictionary when provided")

        uplift_ref = str((gate_cfg or {}).get("uplift_reference_method", "metadata_routing")).strip()
        if uplift_ref != "metadata_routing":
            raise ValueError(
                "learned_utility.compatibility_research.gate.uplift_reference_method must be 'metadata_routing'"
            )

        decision_policy_version = str((gate_cfg or {}).get("decision_policy_version", "sign_ci_v2")).strip()
        if decision_policy_version not in {"legacy_std_v1", "sign_ci_v2"}:
            raise ValueError(
                "learned_utility.compatibility_research.gate.decision_policy_version "
                "must be one of ['legacy_std_v1', 'sign_ci_v2']"
            )

        min_improving_seeds = int((gate_cfg or {}).get("min_improving_seeds", 2))
        if min_improving_seeds <= 0:
            raise ValueError("learned_utility.compatibility_research.gate.min_improving_seeds must be > 0")

        strong_gate = (gate_cfg or {}).get("strong", {})
        weak_gate = (gate_cfg or {}).get("weak", {})
        if strong_gate is not None and not isinstance(strong_gate, dict):
            raise ValueError("learned_utility.compatibility_research.gate.strong must be a dictionary when provided")
        if weak_gate is not None and not isinstance(weak_gate, dict):
            raise ValueError("learned_utility.compatibility_research.gate.weak must be a dictionary when provided")

        for block_name, block in [("strong", strong_gate or {}), ("weak", weak_gate or {})]:
            if float(block.get("spearman_uplift_min", 0.0)) < 0.0:
                raise ValueError(
                    f"learned_utility.compatibility_research.gate.{block_name}.spearman_uplift_min must be >= 0"
                )
            if float(block.get("top1_uplift_min", 0.0)) < 0.0:
                raise ValueError(
                    f"learned_utility.compatibility_research.gate.{block_name}.top1_uplift_min must be >= 0"
                )
            if float(block.get("oracle_gap_pct_reduction_min", 0.0)) < 0.0:
                raise ValueError(
                    "learned_utility.compatibility_research.gate."
                    f"{block_name}.oracle_gap_pct_reduction_min must be >= 0"
                )

        instability_cfg = (gate_cfg or {}).get("instability", {})
        if instability_cfg is not None and not isinstance(instability_cfg, dict):
            raise ValueError(
                "learned_utility.compatibility_research.gate.instability must be a dictionary when provided"
            )
        if float((instability_cfg or {}).get("std_threshold", 0.05)) < 0.0:
            raise ValueError(
                "learned_utility.compatibility_research.gate.instability.std_threshold must be >= 0"
            )
        for field_name, default in [
            ("top1_uplift_std_threshold", 0.05),
            ("spearman_uplift_std_threshold", 0.05),
            ("gap_pct_reduction_std_threshold", 3.0),
        ]:
            if float((instability_cfg or {}).get(field_name, default)) < 0.0:
                raise ValueError(
                    "learned_utility.compatibility_research.gate.instability."
                    f"{field_name} must be >= 0"
                )
        if int((instability_cfg or {}).get("sign_inconsistency_min_count", 2)) < 1:
            raise ValueError(
                "learned_utility.compatibility_research.gate.instability.sign_inconsistency_min_count must be >= 1"
            )
        min_positive_fraction = float((instability_cfg or {}).get("min_positive_fraction", 0.67))
        if min_positive_fraction <= 0.0 or min_positive_fraction > 1.0:
            raise ValueError(
                "learned_utility.compatibility_research.gate.instability.min_positive_fraction must be in (0, 1]"
            )
        ci_level = float((instability_cfg or {}).get("ci_level", 0.95))
        if ci_level <= 0.0 or ci_level >= 1.0:
            raise ValueError("learned_utility.compatibility_research.gate.instability.ci_level must be in (0, 1)")
        if int((instability_cfg or {}).get("ci_bootstrap_reps", 10000)) < 0:
            raise ValueError(
                "learned_utility.compatibility_research.gate.instability.ci_bootstrap_reps must be >= 0"
            )

        backbone = str(cfg.get("features", {}).get("backbone_type", "")).strip().lower()
        if is_response_routing_protocol or is_support_response_routing_protocol:
            if backbone != "dinov2_vitb14":
                raise ValueError(
                    "features.backbone_type must be 'dinov2_vitb14' for learned utility response-routing protocols"
                )

            if int(features_cfg.get("image_size", 0)) != 224:
                raise ValueError("features.image_size must be 224 for learned utility response-routing protocols")

            if int(features_cfg.get("embedding_dim", 0)) != 768:
                raise ValueError("features.embedding_dim must be 768 for learned utility response-routing protocols")

            locked_name = str(features_cfg.get("feature_extractor_name", "")).strip().lower()
            if locked_name != "dinov2_vitb14":
                raise ValueError(
                    "features.feature_extractor_name must be 'dinov2_vitb14' for learned utility response-routing protocols"
                )

            locked_checkpoint = str(features_cfg.get("feature_extractor_checkpoint", "")).strip().lower()
            if locked_checkpoint != "facebook/dinov2-base":
                raise ValueError(
                    "features.feature_extractor_checkpoint must be 'facebook/dinov2-base' "
                    "for learned utility response-routing protocols"
                )

            locked_layer = str(features_cfg.get("feature_extractor_layer", "")).strip().lower()
            if locked_layer != "final_norm_cls":
                raise ValueError(
                    "features.feature_extractor_layer must be 'final_norm_cls' "
                    "for learned utility response-routing protocols"
                )

            locked_pooling = str(features_cfg.get("embedding_pooling", "")).strip().lower()
            if locked_pooling != "cls_token":
                raise ValueError(
                    "features.embedding_pooling must be 'cls_token' for learned utility response-routing protocols"
                )

            response_repeat_mode = str(learned_cfg.get("response_repeat_mode", "posterior_sampling")).strip().lower()
            if response_repeat_mode != "posterior_sampling":
                raise ValueError(
                    "learned_utility.response_repeat_mode must be 'posterior_sampling' "
                    "for learned utility response-routing protocols"
                )

            _ensure_bool(
                learned_cfg.get("posterior_sampling_enabled", True),
                "learned_utility.posterior_sampling_enabled",
            )
            _ensure_bool(
                learned_cfg.get("dropout_enabled", False),
                "learned_utility.dropout_enabled",
            )
            _ensure_bool(
                learned_cfg.get("decoder_sampling_enabled", False),
                "learned_utility.decoder_sampling_enabled",
            )

            if not bool(learned_cfg.get("posterior_sampling_enabled", True)):
                raise ValueError(
                    "learned_utility.posterior_sampling_enabled must be true "
                    "for learned utility response-routing protocols"
                )
            if bool(learned_cfg.get("dropout_enabled", False)):
                raise ValueError(
                    "learned_utility.dropout_enabled must be false "
                    "for learned utility response-routing protocols"
                )

            num_response_repeats = int(learned_cfg.get("num_response_repeats", 0))
            if num_response_repeats <= 0:
                raise ValueError(
                    "learned_utility.num_response_repeats must be > 0 for learned utility response-routing protocols"
                )

            response_norm = str(
                learned_cfg.get("response_feature_normalization", "train_fold_standardize")
            ).strip().lower()
            if response_norm != "train_fold_standardize":
                raise ValueError(
                    "learned_utility.response_feature_normalization must be 'train_fold_standardize' "
                    "for learned utility response-routing protocols"
                )

            calibration_definition = str(
                learned_cfg.get("calibration_error_definition", "bin10_mean_abs_gap")
            ).strip().lower()
            if calibration_definition != "bin10_mean_abs_gap":
                raise ValueError(
                    "learned_utility.calibration_error_definition must be 'bin10_mean_abs_gap' "
                    "for learned utility response-routing protocols"
                )

            near_tie_eps = float(learned_cfg.get("near_tie_epsilon_norm", 0.02))
            if near_tie_eps < 0:
                raise ValueError("learned_utility.near_tie_epsilon_norm must be >= 0")
        else:
            if is_learned_utility_v2:
                if backbone != "dinov2_vitb14":
                    raise ValueError(
                        "features.backbone_type must be 'dinov2_vitb14' for learned_utility_routing_v2"
                    )
                if int(features_cfg.get("embedding_dim", 0)) != 768:
                    raise ValueError(
                        "features.embedding_dim must be 768 for learned_utility_routing_v2"
                    )
            elif backbone != "resnet50":
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
