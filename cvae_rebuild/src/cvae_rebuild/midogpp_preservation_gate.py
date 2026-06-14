from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .midogpp_condition_audit import (
    CONDITION_ROW_IDS,
    METRIC_COLUMNS as CONDITION_METRIC_COLUMNS,
    MODEL_MANIFEST_COLUMNS as CONDITION_MODEL_MANIFEST_COLUMNS,
    PREDICTION_COLUMNS as CONDITION_PREDICTION_COLUMNS,
    RECON_COLUMNS as CONDITION_RECON_COLUMNS,
    TRAINING_COLUMNS as CONDITION_TRAINING_COLUMNS,
    TRUE_LABELS,
    TRUE_TRAIN_TRUE_ENCODE_TRUE_DECODE,
    _attach_condition_ratios,
    _condition_rows,
    _decode_with_condition_labels,
    _metric_context,
    _model_manifest_row as _condition_model_manifest_row,
    _normalize_metric_rows,
    _prediction_context,
    _reconstruction_row as _condition_reconstruction_row,
    _training_context,
)
from .midogpp_condition_audit import (
    FULL_DIM_VARIANT,
    REAL_FULL_DIM_REFERENCE,
    REAL_PCA64_REFERENCE,
    REAL_PCA128_REFERENCE,
    REAL_PCA256_REFERENCE,
)
from .midogpp_preservation_sanity import (
    AUDIT_COLUMNS,
    BALANCED_CONTROL,
    CONTROL_ORDER,
    CVAE_DECODED_ROW_SHUFFLE,
    DEFAULT_FEATURE_CACHE_PATH,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_SIGNAL_DECISION_REPORT_PATH,
    DEFAULT_SIGNAL_SPLIT_MANIFEST_PATH,
    DECODE_MU,
    FeatureCache,
    ManifestRow,
    POSTERIOR_SAMPLE,
    PRESERVATION_COLUMNS,
    PRIOR_SAMPLE,
    REAL_FEATURE_ROW_SHUFFLE,
    REAL_LABEL_PERMUTATION,
    SplitSpec,
    VALID_STATUS,
    VariantConfig,
    WITHIN_TUMOR_CONTROL,
    _aggregate_rows,
    _assert_cache_alignment,
    _assert_virchow2_cache,
    _bool_text,
    _chance_corrected_ratio,
    _decode_mu,
    _empty_metric_row,
    _evaluate_representation,
    _finite,
    _float_or_nan,
    _fmt,
    _identity_audit_rows,
    _load_feature_cache,
    _mapping,
    _parse_variant,
    _path,
    _permuted_labels,
    _posterior_sample,
    _prior_sample,
    _read_split_manifest,
    _read_train_manifest,
    _real_frame_negative_inputs,
    _reconstruction_row as _preservation_reconstruction_row,
    _rng,
    _split_counts,
    _split_status,
    _stable_seed,
    _to_numpy,
    _train_runtime,
)
from .protocol import ProtocolError, build_leakage_report
from .reporting import prepare_artifact_dirs, write_csv_rows, write_json


EXPERIMENT_NAME = "virchow2_cvae_midogpp_preservation_gate_pca128_v1"
DEFAULT_ARTIFACT_ROOT = "cvae_rebuild/artifacts/midogpp/virchow2_cvae_midogpp_preservation_gate_pca128_v1"

PRIMARY_VARIANT = "pca128_beta001"
PCA64_CONTEXT_VARIANT = "pca64_beta001"
PCA256_DIAGNOSTIC_VARIANT = "pca256_beta001"

REAL_REFERENCE_BY_VARIANT = {
    PCA64_CONTEXT_VARIANT: REAL_PCA64_REFERENCE,
    PRIMARY_VARIANT: REAL_PCA128_REFERENCE,
    PCA256_DIAGNOSTIC_VARIANT: REAL_PCA256_REFERENCE,
}

PRIMARY_CVAE_ROLE = DECODE_MU
HARD_NEGATIVE_ROLES = (
    REAL_LABEL_PERMUTATION,
    REAL_FEATURE_ROW_SHUFFLE,
    CVAE_DECODED_ROW_SHUFFLE,
)


@dataclass(frozen=True)
class MidogPPPreservationGateConfig:
    name: str
    artifact_root: Path
    manifest_path: Path
    feature_cache_path: Path
    signal_split_manifest_path: Path
    signal_decision_report_path: Path | None
    positive_label: int
    controls: tuple[str, ...]
    variants: tuple[VariantConfig, ...]
    primary_variant: str
    split_seeds: tuple[int, ...] | None
    bootstrap_reps: int
    bootstrap_seed: int
    allow_npz_test_cache: bool
    min_fit: int
    min_eval: int
    min_fit_pos: int
    min_fit_neg: int
    min_eval_pos: int
    min_eval_neg: int
    min_fit_cases: int
    min_eval_cases: int
    gate_min_bacc: float
    ci_low_threshold: float
    preservation_pass_ratio: float
    preservation_strong_ratio: float
    within_tumor_min_above_fraction: float
    pca256_stronger_delta: float

    @property
    def real_gate_min_bacc(self) -> float:
        return self.gate_min_bacc

    @property
    def cvae_gate_min_bacc(self) -> float:
        return self.gate_min_bacc


def load_midogpp_preservation_gate_config(path: str | Path) -> MidogPPPreservationGateConfig:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise ProtocolError("Loading MIDOG++ preservation gate configs requires PyYAML.") from exc
    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ProtocolError(f"Config must be a mapping: {path}")
    base_dir = source.parents[2] if len(source.parents) >= 3 else source.parent
    return parse_midogpp_preservation_gate_config(payload, base_dir=base_dir)


def parse_midogpp_preservation_gate_config(
    data: Mapping[str, object],
    *,
    base_dir: str | Path = ".",
) -> MidogPPPreservationGateConfig:
    base = Path(base_dir)
    experiment = _mapping(data.get("experiment"), "experiment")
    inputs = _mapping(data.get("inputs"), "inputs")
    run = _mapping(data.get("run"), "run", allow_empty=True)
    bootstrap = _mapping(data.get("bootstrap"), "bootstrap", allow_empty=True)
    thresholds = _mapping(data.get("validity_thresholds"), "validity_thresholds", allow_empty=True)
    decisions = _mapping(data.get("decision_thresholds"), "decision_thresholds", allow_empty=True)
    variants_payload = data.get("variants") or []
    variants = tuple(_parse_variant(item) for item in variants_payload) if variants_payload else _default_variants()
    signal_report = inputs.get("signal_decision_report_path", DEFAULT_SIGNAL_DECISION_REPORT_PATH)
    cfg = MidogPPPreservationGateConfig(
        name=str(experiment.get("name", "")),
        artifact_root=_path(base, str(experiment.get("artifact_root", DEFAULT_ARTIFACT_ROOT))),
        manifest_path=_path(base, str(inputs.get("manifest_path", DEFAULT_MANIFEST_PATH))),
        feature_cache_path=_path(base, str(inputs.get("feature_cache_path", DEFAULT_FEATURE_CACHE_PATH))),
        signal_split_manifest_path=_path(
            base,
            str(inputs.get("signal_split_manifest_path", DEFAULT_SIGNAL_SPLIT_MANIFEST_PATH)),
        ),
        signal_decision_report_path=None if signal_report in ("", None) else _path(base, str(signal_report)),
        positive_label=int(data.get("positive_label", 1)),
        controls=tuple(str(v) for v in run.get("controls", CONTROL_ORDER)),
        variants=variants,
        primary_variant=str(run.get("primary_variant", PRIMARY_VARIANT)),
        split_seeds=None
        if run.get("split_seeds") in (None, "", "all")
        else tuple(int(v) for v in run.get("split_seeds", ())),
        bootstrap_reps=int(bootstrap.get("reps", 1000)),
        bootstrap_seed=int(bootstrap.get("seed", 1337)),
        allow_npz_test_cache=bool(inputs.get("allow_npz_test_cache", False)),
        min_fit=int(thresholds.get("min_fit", 20)),
        min_eval=int(thresholds.get("min_eval", 10)),
        min_fit_pos=int(thresholds.get("min_fit_pos", 10)),
        min_fit_neg=int(thresholds.get("min_fit_neg", 10)),
        min_eval_pos=int(thresholds.get("min_eval_pos", 5)),
        min_eval_neg=int(thresholds.get("min_eval_neg", 5)),
        min_fit_cases=int(thresholds.get("min_fit_cases", 3)),
        min_eval_cases=int(thresholds.get("min_eval_cases", 2)),
        gate_min_bacc=float(decisions.get("gate_min_bacc", decisions.get("cvae_gate_min_bacc", 0.60))),
        ci_low_threshold=float(decisions.get("ci_low_threshold", 0.50)),
        preservation_pass_ratio=float(decisions.get("preservation_pass_ratio", 0.80)),
        preservation_strong_ratio=float(decisions.get("preservation_strong_ratio", 0.90)),
        within_tumor_min_above_fraction=float(decisions.get("within_tumor_min_above_fraction", 0.60)),
        pca256_stronger_delta=float(decisions.get("pca256_stronger_delta", 0.03)),
    )
    validate_midogpp_preservation_gate_config(cfg)
    return cfg


def validate_midogpp_preservation_gate_config(cfg: MidogPPPreservationGateConfig) -> None:
    if cfg.name != EXPERIMENT_NAME:
        raise ProtocolError(f"Unexpected experiment.name={cfg.name!r}; expected {EXPERIMENT_NAME!r}.")
    unknown = sorted(set(cfg.controls) - set(CONTROL_ORDER))
    if unknown:
        raise ProtocolError(f"Unknown controls: {unknown}")
    if cfg.primary_variant != PRIMARY_VARIANT:
        raise ProtocolError(f"Primary gate variant must remain {PRIMARY_VARIANT!r}.")
    variants = {variant.variant_id: variant for variant in cfg.variants}
    for required in (PCA64_CONTEXT_VARIANT, PRIMARY_VARIANT, PCA256_DIAGNOSTIC_VARIANT):
        if required not in variants:
            raise ProtocolError(f"variants must include {required!r}.")
    expected_dims = {
        PCA64_CONTEXT_VARIANT: (64, 16),
        PRIMARY_VARIANT: (128, 32),
        PCA256_DIAGNOSTIC_VARIANT: (256, 64),
    }
    for variant in cfg.variants:
        expected = expected_dims.get(variant.variant_id)
        if expected is not None and (variant.pca_dim, variant.latent_dim) != expected:
            raise ProtocolError(
                f"{variant.variant_id} must use pca_dim={expected[0]} and latent_dim={expected[1]}."
            )
        if variant.num_hidden_layers != 2 or variant.hidden_dim != 512:
            raise ProtocolError("CVAE architecture is locked to hidden_dim=512 and two hidden layers.")
        if not math.isclose(variant.beta_final, 0.001):
            raise ProtocolError("CVAE beta_final is locked to 0.001.")


def run_midogpp_preservation_gate(
    cfg: MidogPPPreservationGateConfig,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    import numpy as np  # type: ignore

    root = prepare_artifact_dirs(Path(artifact_root) if artifact_root is not None else cfg.artifact_root)
    manifest_rows = _read_train_manifest(cfg.manifest_path, positive_label=cfg.positive_label)
    cache = _load_feature_cache(cfg.feature_cache_path)
    _assert_virchow2_cache(cache, path=cfg.feature_cache_path, allow_npz_test_cache=cfg.allow_npz_test_cache)
    embeddings = np.asarray(_to_numpy(cache.embeddings), dtype=float)
    _assert_cache_alignment(manifest_rows, cache)
    split_specs = _read_split_manifest(cfg.signal_split_manifest_path, manifest_rows, controls=cfg.controls)
    if cfg.split_seeds is not None:
        allowed = set(int(seed) for seed in cfg.split_seeds)
        split_specs = tuple(spec for spec in split_specs if int(spec.split_seed) in allowed)
    if not split_specs:
        raise ProtocolError("No matching SAIL signal-control splits were found.")

    pca_context_rows: list[dict[str, object]] = []
    gate_metric_rows: list[dict[str, object]] = []
    condition_rows: list[dict[str, object]] = []
    negative_rows: list[dict[str, object]] = []
    reconstruction_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    model_manifest_rows: list[dict[str, object]] = []

    for spec in split_specs:
        audit = _identity_audit_rows(manifest_rows, spec)
        audit_rows.extend(audit)
        status, error = _split_status(cfg, manifest_rows, spec, audit)
        counts = _split_counts(manifest_rows, spec.fit_idx, spec.eval_idx)
        if status != VALID_STATUS:
            empty = _empty_metric_row(
                spec,
                variant_id=PRIMARY_VARIANT,
                representation_role=REAL_PCA128_REFERENCE,
                counts=counts,
                status=status,
                error_message=error,
            )
            pca_context_rows.append(empty)
            gate_metric_rows.append(empty)
            continue

        x_fit_raw = embeddings[list(spec.fit_idx)]
        y_fit = np.asarray([manifest_rows[idx].label for idx in spec.fit_idx], dtype=int)
        x_eval_raw = embeddings[list(spec.eval_idx)]
        y_eval = [manifest_rows[idx].label for idx in spec.eval_idx]

        full_row, full_preds = _evaluate_representation(
            cfg,
            spec,
            rows=manifest_rows,
            x_fit=x_fit_raw,
            y_fit=y_fit,
            x_eval=x_eval_raw,
            y_eval=y_eval,
            variant_id=FULL_DIM_VARIANT,
            representation_role=REAL_FULL_DIM_REFERENCE,
            counts=counts,
            prediction_prefix="",
        )
        pca_context_rows.append(full_row)
        prediction_rows.extend(_prediction_context(full_preds, condition_row_id="", train_condition_labels="", encode_condition_labels="", decode_condition_labels=""))

        for variant in cfg.variants:
            true_runtime = _train_runtime(
                cfg,
                spec,
                variant,
                x_fit_raw=x_fit_raw,
                x_eval_raw=x_eval_raw,
                y_fit=y_fit,
                condition_mode="real_labels",
            )
            training_rows.extend(_training_context(true_runtime.training_rows, TRUE_LABELS))
            model_manifest_rows.append(_condition_model_manifest_row(spec, true_runtime, manifest_rows, TRUE_LABELS))

            pca_role = REAL_REFERENCE_BY_VARIANT.get(variant.variant_id, f"real_pca{variant.pca_dim}_reference")
            pca_row, pca_preds = _evaluate_representation(
                cfg,
                spec,
                rows=manifest_rows,
                x_fit=true_runtime.fit_x,
                y_fit=y_fit,
                x_eval=true_runtime.eval_x,
                y_eval=y_eval,
                variant_id=variant.variant_id,
                representation_role=pca_role,
                counts=counts,
                prediction_prefix="",
            )
            pca_context_rows.append(pca_row)
            prediction_rows.extend(_prediction_context(pca_preds, condition_row_id="", train_condition_labels="", encode_condition_labels="", decode_condition_labels=""))

            for role, neg_x, neg_y in _real_frame_negative_inputs(cfg, spec, true_runtime.fit_x, y_fit):
                neg_row, neg_preds = _evaluate_representation(
                    cfg,
                    spec,
                    rows=manifest_rows,
                    x_fit=neg_x,
                    y_fit=neg_y,
                    x_eval=true_runtime.eval_x,
                    y_eval=y_eval,
                    variant_id=variant.variant_id,
                    representation_role=role,
                    counts=counts,
                    prediction_prefix="",
                )
                negative_rows.append(neg_row)
                prediction_rows.extend(_prediction_context(neg_preds, condition_row_id="", train_condition_labels="", encode_condition_labels="", decode_condition_labels=""))

            decoded, recon = _decode_mu(true_runtime, true_runtime.fit_x, y_fit)
            reconstruction_rows.append(
                _reconstruction_context(
                    _preservation_reconstruction_row(spec, variant, DECODE_MU, recon, n_rows=len(y_fit)),
                    condition_row_id="",
                    train_condition_labels=TRUE_LABELS,
                    encode_condition_labels=TRUE_LABELS,
                    decode_condition_labels=TRUE_LABELS,
                )
            )
            decode_row, decode_preds = _evaluate_representation(
                cfg,
                spec,
                rows=manifest_rows,
                x_fit=decoded,
                y_fit=y_fit,
                x_eval=true_runtime.eval_x,
                y_eval=y_eval,
                variant_id=variant.variant_id,
                representation_role=DECODE_MU,
                counts=counts,
                prediction_prefix="",
            )
            gate_metric_rows.append(decode_row)
            prediction_rows.extend(_prediction_context(decode_preds, condition_row_id="", train_condition_labels=TRUE_LABELS, encode_condition_labels=TRUE_LABELS, decode_condition_labels=TRUE_LABELS))

            posterior = _posterior_sample(true_runtime, true_runtime.fit_x, y_fit, seed=_stable_seed(spec, variant.variant_id, "gate_posterior"))
            posterior_row, posterior_preds = _evaluate_representation(
                cfg,
                spec,
                rows=manifest_rows,
                x_fit=posterior,
                y_fit=y_fit,
                x_eval=true_runtime.eval_x,
                y_eval=y_eval,
                variant_id=variant.variant_id,
                representation_role=POSTERIOR_SAMPLE,
                counts=counts,
                prediction_prefix="",
            )
            gate_metric_rows.append(posterior_row)
            prediction_rows.extend(_prediction_context(posterior_preds, condition_row_id="", train_condition_labels=TRUE_LABELS, encode_condition_labels=TRUE_LABELS, decode_condition_labels=TRUE_LABELS))

            prior_x, prior_y = _prior_sample(true_runtime, y_fit, seed=_stable_seed(spec, variant.variant_id, "gate_prior"))
            prior_row, prior_preds = _evaluate_representation(
                cfg,
                spec,
                rows=manifest_rows,
                x_fit=prior_x,
                y_fit=prior_y,
                x_eval=true_runtime.eval_x,
                y_eval=y_eval,
                variant_id=variant.variant_id,
                representation_role=PRIOR_SAMPLE,
                counts=counts,
                prediction_prefix="",
            )
            gate_metric_rows.append(prior_row)
            prediction_rows.extend(_prediction_context(prior_preds, condition_row_id="", train_condition_labels=TRUE_LABELS, encode_condition_labels=TRUE_LABELS, decode_condition_labels=TRUE_LABELS))

            shuffled_decoded = np.asarray(decoded, dtype=float)[
                _rng(_stable_seed(spec, variant.variant_id, "gate_decoded_shuffle")).permutation(len(decoded))
            ]
            decoded_neg_row, decoded_neg_preds = _evaluate_representation(
                cfg,
                spec,
                rows=manifest_rows,
                x_fit=shuffled_decoded,
                y_fit=y_fit,
                x_eval=true_runtime.eval_x,
                y_eval=y_eval,
                variant_id=variant.variant_id,
                representation_role=CVAE_DECODED_ROW_SHUFFLE,
                counts=counts,
                prediction_prefix="",
            )
            negative_rows.append(decoded_neg_row)
            prediction_rows.extend(_prediction_context(decoded_neg_preds, condition_row_id="", train_condition_labels=TRUE_LABELS, encode_condition_labels=TRUE_LABELS, decode_condition_labels=TRUE_LABELS))

            permuted_labels = _permuted_condition_labels(cfg, spec, y_fit)
            permuted_runtime = _train_runtime(
                cfg,
                spec,
                variant,
                x_fit_raw=x_fit_raw,
                x_eval_raw=x_eval_raw,
                y_fit=y_fit,
                condition_mode="permuted_train_labels",
            )
            training_rows.extend(_training_context(permuted_runtime.training_rows, "permuted"))
            model_manifest_rows.append(_condition_model_manifest_row(spec, permuted_runtime, manifest_rows, "permuted"))
            for condition in _condition_rows(true_runtime, permuted_runtime, y_fit, permuted_labels):
                condition_decoded, condition_recon = _decode_with_condition_labels(
                    condition["runtime"],
                    condition["runtime"].fit_x,
                    condition["encode_labels"],
                    condition["decode_labels"],
                )
                reconstruction_rows.append(
                    _condition_reconstruction_row(
                        spec,
                        condition["runtime"].variant,
                        str(condition["condition_row_id"]),
                        str(condition["train_condition_labels"]),
                        str(condition["encode_condition_labels"]),
                        str(condition["decode_condition_labels"]),
                        condition_recon,
                        n_rows=len(y_fit),
                    )
                )
                condition_row, condition_preds = _evaluate_representation(
                    cfg,
                    spec,
                    rows=manifest_rows,
                    x_fit=condition_decoded,
                    y_fit=y_fit,
                    x_eval=condition["runtime"].eval_x,
                    y_eval=y_eval,
                    variant_id=condition["runtime"].variant.variant_id,
                    representation_role=str(condition["condition_row_id"]),
                    counts=counts,
                    prediction_prefix="",
                )
                condition_row = _metric_context(
                    condition_row,
                    condition_row_id=str(condition["condition_row_id"]),
                    train_condition_labels=str(condition["train_condition_labels"]),
                    encode_condition_labels=str(condition["encode_condition_labels"]),
                    decode_condition_labels=str(condition["decode_condition_labels"]),
                )
                condition_rows.append(condition_row)
                prediction_rows.extend(
                    _prediction_context(
                        condition_preds,
                        condition_row_id=str(condition["condition_row_id"]),
                        train_condition_labels=str(condition["train_condition_labels"]),
                        encode_condition_labels=str(condition["encode_condition_labels"]),
                        decode_condition_labels=str(condition["decode_condition_labels"]),
                    )
                )

    pca_context_all = pca_context_rows + _aggregate_rows(pca_context_rows)
    gate_metric_all = gate_metric_rows + _aggregate_rows(gate_metric_rows)
    preservation_all = pca_context_all + gate_metric_all
    negative_all = negative_rows + _aggregate_rows(negative_rows)
    condition_all = _normalize_metric_rows(condition_rows + _aggregate_rows(condition_rows))
    _attach_condition_ratios(condition_all)
    _attach_gate_preservation_ratios(preservation_all)

    leakage = _leakage_report(cfg, audit_rows, negative_all)
    decision_labels = _decision_labels(cfg, preservation_all, negative_all, condition_all, leakage)

    write_csv_rows(root / "tables" / "preservation_gate_metrics.csv", preservation_all, PRESERVATION_COLUMNS)
    write_csv_rows(root / "tables" / "pca_capacity_context.csv", pca_context_all, PRESERVATION_COLUMNS)
    write_csv_rows(root / "tables" / "condition_warning_matrix.csv", condition_all, CONDITION_METRIC_COLUMNS)
    write_csv_rows(root / "tables" / "reconstruction_diagnostics.csv", reconstruction_rows, CONDITION_RECON_COLUMNS)
    write_csv_rows(root / "tables" / "training_diagnostics.csv", training_rows, CONDITION_TRAINING_COLUMNS)
    write_csv_rows(root / "tables" / "negative_control_metrics.csv", negative_all, PRESERVATION_COLUMNS)
    write_csv_rows(root / "tables" / "identity_overlap_audit.csv", audit_rows, AUDIT_COLUMNS)
    write_csv_rows(root / "tables" / "predictions.csv", prediction_rows, CONDITION_PREDICTION_COLUMNS)
    write_csv_rows(root / "manifests" / "model_manifest.csv", model_manifest_rows, CONDITION_MODEL_MANIFEST_COLUMNS)
    write_json(root / "manifests" / "protocol_manifest.json", _protocol_manifest(cfg, cache, manifest_rows, split_specs))
    write_json(root / "reports" / "leakage_report.json", leakage)
    _write_decision_report(root / "reports" / "decision_report.md", cfg, decision_labels, leakage, preservation_all, negative_all, condition_all)
    return root


def _default_variants() -> tuple[VariantConfig, ...]:
    return (
        VariantConfig(PCA64_CONTEXT_VARIANT, pca_dim=64, latent_dim=16),
        VariantConfig(PRIMARY_VARIANT, pca_dim=128, latent_dim=32),
        VariantConfig(PCA256_DIAGNOSTIC_VARIANT, pca_dim=256, latent_dim=64),
    )


def _permuted_condition_labels(
    cfg: MidogPPPreservationGateConfig,
    spec: SplitSpec,
    y_fit: Sequence[int],
) -> Any:
    return _permuted_labels(cfg, spec, y_fit, "permuted_train_labels")


def _reconstruction_context(
    row: Mapping[str, object],
    *,
    condition_row_id: str,
    train_condition_labels: str,
    encode_condition_labels: str,
    decode_condition_labels: str,
) -> dict[str, object]:
    out = dict(row)
    out.update(
        {
            "condition_row_id": condition_row_id,
            "train_condition_labels": train_condition_labels,
            "encode_condition_labels": encode_condition_labels,
            "decode_condition_labels": decode_condition_labels,
        }
    )
    return out


def _attach_gate_preservation_ratios(rows: Sequence[dict[str, object]]) -> None:
    summary = {
        (row["control_name"], row["domain_name"], row["variant_id"], row["representation_role"]): row
        for row in rows
        if row.get("aggregation_level") == "summary"
    }
    for row in rows:
        if row.get("aggregation_level") != "summary":
            continue
        variant_id = str(row.get("variant_id"))
        role = str(row.get("representation_role"))
        reference_role = REAL_REFERENCE_BY_VARIANT.get(variant_id)
        if reference_role is None or role == reference_role:
            continue
        if role in {REAL_FULL_DIM_REFERENCE, REAL_PCA64_REFERENCE, REAL_PCA128_REFERENCE, REAL_PCA256_REFERENCE}:
            continue
        reference = summary.get((row["control_name"], row["domain_name"], variant_id, reference_role))
        if reference is None:
            continue
        row["preservation_ratio_vs_real_frame"] = _chance_corrected_ratio(row.get("bacc"), reference.get("bacc"))


def _decision_labels(
    cfg: MidogPPPreservationGateConfig,
    metrics: Sequence[Mapping[str, object]],
    negatives: Sequence[Mapping[str, object]],
    condition_rows: Sequence[Mapping[str, object]],
    leakage: Mapping[str, object],
) -> list[str]:
    labels: list[str] = []
    hard_negative_above = _negative_above_chance(cfg, negatives, variant_id=PRIMARY_VARIANT)
    if leakage.get("status") != "PASS" or hard_negative_above:
        return ["LEAKAGE_OR_ALIGNMENT_FAILURE_SUSPECT", "DO_NOT_RUN_GMM_YET"]

    primary_real = _summary(metrics, BALANCED_CONTROL, "", PRIMARY_VARIANT, REAL_PCA128_REFERENCE)
    primary_decode = _summary(metrics, BALANCED_CONTROL, "", PRIMARY_VARIANT, DECODE_MU)
    if primary_real is None or primary_decode is None:
        return ["INSUFFICIENT_VALID_SPLITS", "DO_NOT_RUN_GMM_YET"]

    if _gate_above_chance(cfg, primary_real) and _required_negatives_clean(
        cfg,
        negatives,
        PRIMARY_VARIANT,
        (REAL_LABEL_PERMUTATION, REAL_FEATURE_ROW_SHUFFLE),
    ):
        labels.append("PCA128_REAL_FRAME_GATE_PASS")
    else:
        labels.extend(["REAL_COMPARATOR_UNSUITABLE", "DO_NOT_RUN_GMM_YET"])
        if _condition_permutation_above_chance(cfg, condition_rows):
            labels.append("LATENT_CLASS_SIGNAL_DOMINATES_CONDITION_WARNING")
        return _unique(labels)

    ratio = _chance_corrected_ratio(primary_decode.get("bacc"), primary_real.get("bacc"))
    decode_pass = (
        _gate_above_chance(cfg, primary_decode)
        and _finite(ratio)
        and ratio >= cfg.preservation_pass_ratio
        and _required_negatives_clean(cfg, negatives, PRIMARY_VARIANT, HARD_NEGATIVE_ROLES)
    )
    if decode_pass:
        labels.append("PCA128_CVAE_DECODE_PRESERVATION_PASS")
        if ratio >= cfg.preservation_strong_ratio:
            labels.append("PCA128_CVAE_STRONG_PRESERVATION")
    else:
        labels.append("PCA128_CVAE_RECONSTRUCTION_BOTTLENECK")

    pca256_decode = _summary(metrics, BALANCED_CONTROL, "", PCA256_DIAGNOSTIC_VARIANT, DECODE_MU)
    if (
        pca256_decode is not None
        and _gate_above_chance(cfg, pca256_decode)
        and _finite(pca256_decode.get("bacc"))
        and _finite(primary_decode.get("bacc"))
        and float(pca256_decode.get("bacc")) - float(primary_decode.get("bacc")) >= cfg.pca256_stronger_delta
    ):
        labels.append("PCA256_DIAGNOSTIC_STRONGER_ABSOLUTE_UTILITY")

    within_fraction, within_above, within_total = _within_tumor_decode_fraction(cfg, metrics)
    if decode_pass and within_total and within_fraction >= cfg.within_tumor_min_above_fraction:
        labels.append("GMM_FEASIBILITY_ALLOWED_NEXT")
    else:
        labels.append("DO_NOT_RUN_GMM_YET")
    labels.append(f"WITHIN_TUMOR_DECODE_SUPPORT_{within_above}_OF_{within_total}")

    if _condition_permutation_above_chance(cfg, condition_rows):
        labels.append("LATENT_CLASS_SIGNAL_DOMINATES_CONDITION_WARNING")
    return _unique(labels)


def _gate_above_chance(cfg: MidogPPPreservationGateConfig, row: Mapping[str, object]) -> bool:
    return (
        _finite(row.get("bacc"))
        and _finite(row.get("ci_low"))
        and float(row.get("bacc")) > cfg.gate_min_bacc
        and float(row.get("ci_low")) > cfg.ci_low_threshold
    )


def _negative_above_chance(
    cfg: MidogPPPreservationGateConfig,
    rows: Sequence[Mapping[str, object]],
    *,
    variant_id: str | None = None,
) -> list[Mapping[str, object]]:
    out = []
    for row in rows:
        if row.get("aggregation_level") != "summary":
            continue
        if variant_id is not None and row.get("variant_id") != variant_id:
            continue
        if str(row.get("representation_role")) not in HARD_NEGATIVE_ROLES:
            continue
        if _gate_above_chance(cfg, row):
            out.append(row)
    return out


def _required_negative_above(
    cfg: MidogPPPreservationGateConfig,
    rows: Sequence[Mapping[str, object]],
    variant_id: str,
    roles: Sequence[str],
) -> list[Mapping[str, object]]:
    found = []
    for role in roles:
        row = _summary(rows, BALANCED_CONTROL, "", variant_id, role)
        if row is not None and _gate_above_chance(cfg, row):
            found.append(row)
    return found


def _required_negatives_clean(
    cfg: MidogPPPreservationGateConfig,
    rows: Sequence[Mapping[str, object]],
    variant_id: str,
    roles: Sequence[str],
) -> bool:
    for role in roles:
        row = _summary(rows, BALANCED_CONTROL, "", variant_id, role)
        if row is None or _gate_above_chance(cfg, row):
            return False
    return True


def _condition_permutation_above_chance(
    cfg: MidogPPPreservationGateConfig,
    rows: Sequence[Mapping[str, object]],
) -> bool:
    for row in rows:
        if row.get("aggregation_level") != "summary":
            continue
        if row.get("control_name") != BALANCED_CONTROL:
            continue
        if str(row.get("condition_row_id")) in ("", TRUE_TRAIN_TRUE_ENCODE_TRUE_DECODE):
            continue
        if _gate_above_chance(cfg, row):
            return True
    return False


def _within_tumor_decode_fraction(
    cfg: MidogPPPreservationGateConfig,
    rows: Sequence[Mapping[str, object]],
) -> tuple[float, int, int]:
    within = [
        row
        for row in rows
        if row.get("aggregation_level") == "summary"
        and row.get("control_name") == WITHIN_TUMOR_CONTROL
        and row.get("variant_id") == PRIMARY_VARIANT
        and row.get("representation_role") == DECODE_MU
        and row.get("status") == VALID_STATUS
    ]
    above = [row for row in within if _gate_above_chance(cfg, row)]
    return (float(len(above)) / float(len(within)) if within else 0.0, len(above), len(within))


def _summary(
    rows: Sequence[Mapping[str, object]],
    control: str,
    domain: str,
    variant: str,
    role: str,
) -> Mapping[str, object] | None:
    for row in rows:
        if (
            row.get("aggregation_level") == "summary"
            and row.get("control_name") == control
            and row.get("domain_name") == domain
            and row.get("variant_id") == variant
            and row.get("representation_role") == role
        ):
            return row
    return None


def _leakage_report(
    cfg: MidogPPPreservationGateConfig,
    audit_rows: Sequence[Mapping[str, object]],
    negative_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    overlap_failures = [row for row in audit_rows if int(row.get("overlap_count", 0)) > 0]
    negative_above = _negative_above_chance(cfg, negative_rows, variant_id=PRIMARY_VARIANT)
    extra = []
    if overlap_failures:
        extra.append("identity_overlap_failure")
    if negative_above:
        extra.append("negative_control_above_chance")
    report = build_leakage_report(
        target_support_labels_for_selection=False,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=True,
        oracle_rows_diagnostic_only=True,
        extra_violations=extra,
    ).to_json_dict()
    report.update(
        {
            "identity_overlap_failure_count": len(overlap_failures),
            "negative_controls_above_chance_count": len(negative_above),
            "negative_control_above_chance_rule": f"mean BACC > {cfg.gate_min_bacc} and CI_low > {cfg.ci_low_threshold}",
            "condition_permutation_is_warning_only": True,
            "fit_only_pca": True,
            "fit_only_cvae_training": True,
            "eval_labels_scoring_only": True,
            "fixed_threshold": 0.5,
            "metadata_success_signal": False,
            "routing_or_gmm_composition": False,
        }
    )
    return report


def _protocol_manifest(
    cfg: MidogPPPreservationGateConfig,
    cache: FeatureCache,
    rows: Sequence[ManifestRow],
    splits: Sequence[SplitSpec],
) -> dict[str, object]:
    feature_dim = int(getattr(cache.embeddings, "shape", [0, 0])[1])
    if feature_dim == 0:
        feature_dim = int(_to_numpy(cache.embeddings).shape[1])
    return {
        "schema_version": "midogpp_virchow2_cvae_preservation_gate_pca128_protocol_v1",
        "experiment_name": cfg.name,
        "manifest_path": str(cfg.manifest_path),
        "feature_cache_path": str(cfg.feature_cache_path),
        "signal_split_manifest_path": str(cfg.signal_split_manifest_path),
        "signal_decision_report_path": str(cfg.signal_decision_report_path or ""),
        "train_rows": int(len(rows)),
        "split_count": int(len(splits)),
        "feature_dim": int(feature_dim),
        "feature_extractor": dict(cache.feature_extractor),
        "positive_label": int(cfg.positive_label),
        "cache_building_in_scope": False,
        "split_scope": "train_cache_signal_controls",
        "pca_scaler_fit_scope": "fit_rows_only",
        "cvae_training_scope": "fit_rows_only",
        "eval_labels_role": "scoring_only",
        "threshold_policy": "fixed_0.5_classifier_rule_not_calibrated_probability",
        "primary_gate": f"{PRIMARY_VARIANT}+{DECODE_MU}",
        "primary_comparator": REAL_PCA128_REFERENCE,
        "diagnostic_variants": [variant.variant_id for variant in cfg.variants if variant.variant_id != cfg.primary_variant],
        "above_chance_rule": f"mean BACC > {cfg.gate_min_bacc} and CI_low > {cfg.ci_low_threshold}",
        "negative_control_above_chance_rule": f"mean BACC > {cfg.gate_min_bacc} and CI_low > {cfg.ci_low_threshold}",
        "condition_rows": list(CONDITION_ROW_IDS),
        "condition_permutation_interpretation": (
            "Above-chance condition-permutation rows warn that reconstruction may preserve class signal through z; "
            "they do not prove leakage and do not support controllable class-conditional generation."
        ),
        "gmm_composition_in_scope": False,
        "claim_boundary": {
            "allowed": "pca128 CVAE preservation of corrected MIDOG++ signal-control utility if gate labels pass.",
            "forbidden": "No GMM composition, routing, expert-selection, metadata-compatibility, or controllable class-conditional generation claim.",
        },
    }


def _write_decision_report(
    path: Path,
    cfg: MidogPPPreservationGateConfig,
    decision_labels: Sequence[str],
    leakage: Mapping[str, object],
    metrics: Sequence[Mapping[str, object]],
    negatives: Sequence[Mapping[str, object]],
    condition_rows: Sequence[Mapping[str, object]],
) -> None:
    primary_real = _summary(metrics, BALANCED_CONTROL, "", PRIMARY_VARIANT, REAL_PCA128_REFERENCE)
    primary_decode = _summary(metrics, BALANCED_CONTROL, "", PRIMARY_VARIANT, DECODE_MU)
    ratio = _chance_corrected_ratio(primary_decode.get("bacc"), primary_real.get("bacc")) if primary_real and primary_decode else math.nan
    within_fraction, within_above, within_total = _within_tumor_decode_fraction(cfg, metrics)
    lines = [
        "# MIDOG++ Virchow2 pca128 CVAE Preservation Gate",
        "",
        f"- Decision labels: `{', '.join(decision_labels)}`",
        f"- Leakage status: `{leakage.get('status')}`",
        f"- Above-chance rule: mean BACC `> {cfg.gate_min_bacc}` and CI low `> {cfg.ci_low_threshold}`.",
        "- Primary gate: `pca128_beta001 + decode_mu_fit_to_real_eval`.",
        "- pca256 is diagnostic only; pca64 is context for the known capacity bottleneck.",
        "- This is not GMM composition, routing, expert selection, metadata compatibility, or controllable class-conditional generation evidence.",
        "",
        "## Primary Gate",
        "",
        "| Row | BACC | CI low | Ratio | Status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for label, row in (("real_pca128_reference", primary_real), ("decode_mu_fit_to_real_eval", primary_decode)):
        if row is not None:
            lines.append(
                f"| {label} | {_fmt(row.get('bacc'))} | {_fmt(row.get('ci_low'))} | "
                f"{_fmt(row.get('preservation_ratio_vs_real_frame') if label != 'real_pca128_reference' else ratio)} | {row.get('status')} |"
            )
    lines.extend(
        [
            "",
            "## Within-Tumor Decode Support",
            "",
            f"- Valid within-tumor decode rows above chance: `{within_above}/{within_total}`.",
            f"- Fraction: `{_fmt(within_fraction)}`; required `>= {cfg.within_tumor_min_above_fraction}` for `GMM_FEASIBILITY_ALLOWED_NEXT`.",
            "",
            "## Context Rows",
            "",
            "| Variant | Role | BACC | CI low | Status |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for variant, role in (
        (FULL_DIM_VARIANT, REAL_FULL_DIM_REFERENCE),
        (PCA64_CONTEXT_VARIANT, REAL_PCA64_REFERENCE),
        (PRIMARY_VARIANT, REAL_PCA128_REFERENCE),
        (PCA256_DIAGNOSTIC_VARIANT, REAL_PCA256_REFERENCE),
        (PCA256_DIAGNOSTIC_VARIANT, DECODE_MU),
    ):
        row = _summary(metrics, BALANCED_CONTROL, "", variant, role)
        if row is not None:
            lines.append(f"| {variant} | {role} | {_fmt(row.get('bacc'))} | {_fmt(row.get('ci_low'))} | {row.get('status')} |")
    lines.extend(
        [
            "",
            "## Negative Controls",
            "",
            "| Variant | Role | BACC | CI low | Gate-above chance |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for row in negatives:
        if row.get("aggregation_level") == "summary":
            lines.append(
                f"| {row.get('variant_id')} | {row.get('representation_role')} | {_fmt(row.get('bacc'))} | "
                f"{_fmt(row.get('ci_low'))} | {_bool_text(_gate_above_chance(cfg, row))} |"
            )
    lines.extend(
        [
            "",
            "## Condition Warning Matrix",
            "",
            "Above-chance permuted-condition rows mean the CVAE may preserve class signal through `z`. "
            "That supports reconstruction mechanics only; it does not support controllable class-conditional generation.",
            "",
            "| Variant | Condition row | BACC | CI low | Gate-above chance |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for row in condition_rows:
        if row.get("aggregation_level") == "summary" and row.get("control_name") == BALANCED_CONTROL:
            lines.append(
                f"| {row.get('variant_id')} | {row.get('condition_row_id')} | {_fmt(row.get('bacc'))} | "
                f"{_fmt(row.get('ci_low'))} | {_bool_text(_gate_above_chance(cfg, row))} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _unique(labels: Sequence[str]) -> list[str]:
    out = []
    for label in labels:
        if label not in out:
            out.append(label)
    return out or ["INSUFFICIENT_VALID_SPLITS"]
