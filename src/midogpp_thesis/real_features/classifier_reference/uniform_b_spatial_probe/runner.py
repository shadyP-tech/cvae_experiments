"""Run the frozen-capacity B-spatial representation diagnostic."""

from __future__ import annotations

import csv
from collections import defaultdict
import hashlib
import json
import platform
from pathlib import Path
import time
from typing import Mapping, Sequence

from joblib import Parallel, delayed
import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from midogpp_thesis.common.hashing import stable_hash
from ..artifacts import prepare_artifact_dirs, write_csv_rows, write_json
from ..protocol import ProtocolError
from ..real_feature_frame import load_midogpp_real_feature_frame
from ..uniform_b_nonlinear_probe.config import load_nonlinear_probe_config
from ..uniform_b_nonlinear_probe.estimator import (
    effective_gamma, fit_logistic, kernel_audit_row, median_distance_fit,
)
from ..uniform_b_nonlinear_probe.statistics import binary_metrics, paired_case_bootstrap
from ..uniform_b_nonlinear_probe.validation import validate_nonlinear_probe_bundle
from .cache import validate_uniform_b_spatial_cache
from .config import (
    EXPECTED_ROWS, SPATIAL_DIM, SpatialProbeConfig, load_spatial_cache_config,
)
from .workspace_binding import validate_production_workspace_binding

def run_spatial_probe(config: SpatialProbeConfig) -> Path:
    validate_production_workspace_binding(config)
    started = time.perf_counter()
    root = prepare_artifact_dirs(config.artifact_root); (root / "provenance").mkdir(parents=True, exist_ok=True)
    if (root / "manifests/protocol_manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed B-spatial output: {root}.")
    cache_config = load_spatial_cache_config(config.spatial_cache_config_path)
    validate_uniform_b_spatial_cache(config.spatial_cache_root, config=cache_config)
    nonlinear_config = load_nonlinear_probe_config(config.nonlinear_reference_root / "config.resolved.yaml")
    validate_nonlinear_probe_bundle(config.nonlinear_reference_root, config=nonlinear_config)
    frame = load_midogpp_real_feature_frame(
        manifest_path=config.manifest_path, feature_cache_path=config.spatial_feature_cache_path,
        expected_feature_dim=config.expected_feature_dim, allow_excluded_center_omission=True,
    )
    if len(frame.rows) != config.expected_rows: raise ProtocolError("B-spatial train-row count drifted.")
    x = np.asarray(frame.embeddings, dtype=np.float32)
    y = np.asarray([row.label for row in frame.rows], dtype=np.int8)
    centers = np.asarray([row.center for row in frame.rows], dtype=str)
    sample_ids = np.asarray([row.sample_id for row in frame.rows], dtype=str)
    case_ids = np.asarray([row.case_id for row in frame.rows], dtype=str)
    linear_locks = _linear_locks(config.canonical_reference_root)
    nystroem_locks = _nystroem_locks(config.nonlinear_reference_root)
    canonical_linear, current_bplus = _baseline_predictions(config.nonlinear_reference_root)
    if set(current_bplus) != set(sample_ids.tolist()) or set(canonical_linear) != set(sample_ids.tolist()):
        raise ProtocolError("B-spatial baseline rows do not match spatial features.")
    input_hashes = _input_hashes(config, frame.feature_cache_hash, frame.manifest_hash)
    frozen = _frozen_protocol(config, input_hashes, linear_locks, nystroem_locks)
    write_json(root / "manifests/frozen_protocol_snapshot.json", frozen)
    write_json(root / "manifests/inherited_classifier_locks.json", {
        "schema_version": "midogpp_uniform_b_spatial_inherited_locks_v1",
        "protocol_hash": frozen["protocol_hash"], "linear": linear_locks, "nystroem": nystroem_locks,
        "reselected_on_spatial_features": False, "threshold_policy": "predict",
    })
    write_json(root / "provenance/input_artifacts.json", {
        "schema_version": "midogpp_uniform_b_spatial_inputs_v1", "input_hashes": input_hashes,
        "validation_or_test_inputs_present": False, "multiscale_c_is_not_an_input": True,
        "bounded_change": "central_4x4_mean_replaced_by_ordered_2x2_quadrant_means",
    })
    outputs = Parallel(n_jobs=config.runtime.outer_jobs, backend="loky", max_nbytes="10M", mmap_mode="r")(
        delayed(_fit_outer)(outer, x, y, centers, sample_ids, case_ids, config, linear_locks[outer], nystroem_locks[outer])
        for outer in config.heldout_centers
    )
    materialized = _materialize(outputs, canonical_linear, current_bplus, config)
    bootstrap = paired_case_bootstrap(
        materialized["spatial_primary"], materialized["current_bplus"], centers=config.heldout_centers,
        replicates=config.gate.bootstrap_replicates, seed=config.gate.bootstrap_seed,
    )
    hard_core = _hard_core_exchange(
        canonical_linear, current_bplus, materialized["spatial_primary"], config.heldout_centers
    )
    exchange = _error_exchange(materialized["spatial_primary"], materialized["current_bplus"])
    decision = _progression_decision(materialized["comparisons"], materialized["stability"], hard_core, bootstrap, config)
    summary = _summary(materialized, hard_core, exchange, bootstrap, decision)
    write_csv_rows(root / "tables/kernel_fit_audit.csv", materialized["audits"])
    write_csv_rows(root / "tables/outer_results.csv", materialized["results"])
    write_csv_rows(root / "tables/outer_predictions.csv", materialized["all_predictions"])
    write_csv_rows(root / "tables/stability_predictions.csv", materialized["stability_predictions"])
    write_csv_rows(root / "tables/center_comparison.csv", materialized["comparisons"])
    write_csv_rows(root / "tables/seed_stability.csv", materialized["stability"])
    write_csv_rows(root / "tables/error_exchange.csv", exchange)
    write_csv_rows(root / "tables/hard_core_exchange.csv", hard_core)
    write_json(root / "reports/paired_bootstrap.json", bootstrap)
    write_json(root / "reports/progression_decision.json", decision)
    write_json(root / "reports/diagnostic_summary.json", summary)
    (root / "reports/diagnostic_report.md").write_text(_render_report(summary))
    write_json(root / "reports/runtime_summary.json", {
        "schema_version": "midogpp_uniform_b_spatial_runtime_v1", "status": "COMPLETE",
        "elapsed_seconds": time.perf_counter() - started, "cpu_worker_processes": config.runtime.outer_jobs,
        "threads_per_process": config.runtime.threads_per_job, "gpu_used_for_classifiers": False,
        "outer_linear_fits": 9, "outer_nystroem_fits": 27, "source_inner_selector_cells": 0,
        "python": platform.python_version(),
    })
    write_json(root / "reports/leakage_provenance_report.json", {
        "schema_version": "midogpp_uniform_b_spatial_leakage_v1", "status": "PENDING_INDEPENDENT_VALIDATION",
        "outer_target_used_for_scaler_gamma_landmarks_or_fit": False, "classifier_capacity_reselected": False,
        "target_labels_used_for_scoring_only": True, "validation_features_generated": False,
        "test_features_generated": False, "diagnostic_surface_previously_inspected": True,
    })
    write_json(root / "manifests/protocol_manifest.json", {
        "schema_version": "midogpp_uniform_b_spatial_protocol_manifest_v1", "status": "PENDING_INDEPENDENT_VALIDATION",
        "protocol_hash": frozen["protocol_hash"], "representation_id": "annotation_jpeg_fixed_center_b_spatial_quadrants_v1",
        "claim_scope": "diagnostic_only", "diagnostic_surface_previously_inspected": True,
        "may_replace_canonical_reference": False, "validation_scored": False, "test_scored": False,
    })
    _write_content_index(root)
    from .validation import validate_spatial_probe_bundle
    pending = validate_spatial_probe_bundle(root, config=config, allow_pending=True)
    write_json(root / "reports/validation_report.json", {
        "schema_version": "midogpp_uniform_b_spatial_validation_v1", "status": "PASS",
        "validator": "validate_spatial_probe_bundle", "checks": pending,
    })
    leakage = _read_json(root / "reports/leakage_provenance_report.json"); leakage["status"] = "PASS"; write_json(root / "reports/leakage_provenance_report.json", leakage)
    protocol = _read_json(root / "manifests/protocol_manifest.json"); protocol["status"] = "PASS"; write_json(root / "manifests/protocol_manifest.json", protocol)
    _write_content_index(root); validate_spatial_probe_bundle(root, config=config)
    return root

def _fit_outer(outer: str, x: np.ndarray, y: np.ndarray, centers: np.ndarray, sample_ids: np.ndarray, case_ids: np.ndarray, config: SpatialProbeConfig, linear_lock: Mapping[str, object], nystroem_lock: Mapping[str, object]) -> dict[str, object]:
    with threadpool_limits(limits=config.runtime.threads_per_job):
        train = centers != outer; target = centers == outer
        scaler = StandardScaler(); train_x = scaler.fit_transform(x[train]).astype(np.float32, copy=False); target_x = scaler.transform(x[target]).astype(np.float32, copy=False)
        linear = LogisticRegression(
            C=float(linear_lock["C"]), solver=str(linear_lock["solver"]), penalty=str(linear_lock["penalty"]),
            class_weight=linear_lock["class_weight"], max_iter=int(linear_lock["max_iter"]), random_state=23,
        ); linear.fit(train_x, y[train])
        if int(np.max(linear.n_iter_)) >= int(linear_lock["max_iter"]): raise ProtocolError(f"B-spatial linear fit did not converge: {outer}.")
        linear_rows = _prediction_rows(outer, sample_ids[target], case_ids[target], y[target], linear.predict(target_x), linear.predict_proba(target_x)[:, 1], "b_spatial_linear", "spatial_linear_lock")
        median = median_distance_fit(train_x, sample_ids[train], seed=config.gamma_sample_seed, cap=config.gamma_sample_cap, fit_key="b_spatial_outer:" + outer)
        gamma = effective_gamma(float(nystroem_lock["width_multiplier"]), float(median["median_distance"]))
        fit_hash = _string_hash(sorted(sample_ids[train].tolist())); scaler_hash = _array_hash((scaler.mean_, scaler.scale_)); seed_outputs = []
        for seed in (config.primary_landmark_seed, *config.stability_landmark_seeds):
            transformer = Nystroem(kernel="rbf", gamma=gamma, n_components=int(nystroem_lock["n_components"]), random_state=seed, n_jobs=1)
            train_z = transformer.fit_transform(train_x); target_z = transformer.transform(target_x)
            model = fit_logistic(train_z, y[train], c_value=float(nystroem_lock["logistic_c"]), class_weight=nystroem_lock["class_weight"], max_iter=5000)
            pred = model.predict(target_z).astype(np.int8); prob = model.predict_proba(target_z)[:, 1]
            rows = _prediction_rows(outer, sample_ids[target], case_ids[target], y[target], pred, prob, "b_spatial_nystroem", str(nystroem_lock["candidate_id"]), seed)
            audit = kernel_audit_row(
                role="b_spatial_outer_final", fit_key=outer, train_centers=tuple(c for c in config.heldout_centers if c != outer),
                fit_row_hash=fit_hash, scaler_hash=scaler_hash, median=median, width_multiplier=float(nystroem_lock["width_multiplier"]),
                gamma=gamma, n_components=int(nystroem_lock["n_components"]), landmark_seed=seed, transformer=transformer, train_sample_ids=sample_ids[train],
            )
            seed_outputs.append({"seed": seed, "rows": rows, "metrics": binary_metrics(y[target], pred), "audit": audit, "n_iter": int(np.max(model.n_iter_))})
        return {
            "outer": outer, "n_train": int(train.sum()), "n_eval": int(target.sum()), "fit_row_hash": fit_hash,
            "eval_row_hash": _string_hash(sorted(sample_ids[target].tolist())), "linear_rows": linear_rows,
            "linear_metrics": binary_metrics(y[target], linear.predict(target_x)), "linear_lock": dict(linear_lock),
            "nystroem_lock": dict(nystroem_lock), "seed_outputs": seed_outputs,
        }

def _prediction_rows(outer: str, sample_ids: Sequence[object], case_ids: Sequence[object], truth: np.ndarray, prediction: np.ndarray, probability: np.ndarray, role: str, candidate_id: str, seed: int | None = None) -> list[dict[str, object]]:
    eval_hash = _string_hash(sorted(str(value) for value in sample_ids)); rows = []
    for i, sample_id in enumerate(sample_ids):
        rows.append({
            "schema_version": "midogpp_uniform_b_spatial_prediction_v1", "model_role": role, "outer_center": outer,
            "sample_id": str(sample_id), "case_id": str(case_ids[i]), "center": outer, "y_true": int(truth[i]),
            "y_pred": int(prediction[i]), "prob_pos": float(probability[i]), "candidate_id": candidate_id,
            "landmark_seed": "" if seed is None else seed, "eval_row_hash": eval_hash,
            "target_labels_used_for_scoring_only": True, "fit_used_target_center": False, "capacity_reselected": False,
        })
    return rows

def _materialize(outputs: Sequence[Mapping[str, object]], canonical_linear: Mapping[str, Mapping[str, object]], current_bplus: Mapping[str, Mapping[str, object]], config: SpatialProbeConfig) -> dict[str, object]:
    results=[]; spatial_linear=[]; spatial_primary=[]; stability_predictions=[]; comparisons=[]; stability=[]; audits=[]
    current_rows = [_normalized_baseline(row, "canonical_b_nystroem") for row in current_bplus.values()]
    linear_b_rows = [_normalized_baseline(row, "canonical_b_linear") for row in canonical_linear.values()]
    for output in sorted(outputs, key=lambda row: config.heldout_centers.index(str(row["outer"]))):
        outer=str(output["outer"]); seed_map={int(row["seed"]):row for row in output["seed_outputs"]}; primary=seed_map[config.primary_landmark_seed]
        spatial_linear.extend(output["linear_rows"]); spatial_primary.extend(primary["rows"]); audits.extend(row["audit"] for row in seed_map.values())
        baseline_center=[row for row in current_rows if str(row["center"])==outer]; truth=np.asarray([int(row["y_true"]) for row in baseline_center]); base_metrics=binary_metrics(truth,np.asarray([int(row["y_pred"]) for row in baseline_center]))
        for role, metrics, lock in (("canonical_b_nystroem",base_metrics,{}),("b_spatial_linear",output["linear_metrics"],output["linear_lock"]),("b_spatial_nystroem",primary["metrics"],output["nystroem_lock"])):
            results.append({"schema_version":"midogpp_uniform_b_spatial_outer_result_v1","outer_center":outer,"model_role":role,"n_train":output["n_train"],"n_eval":output["n_eval"],"fit_row_hash":output["fit_row_hash"],"eval_row_hash":output["eval_row_hash"],**dict(metrics),**dict(lock),"diagnostic_only":True})
        for candidate_role, metrics in (("b_spatial_linear",output["linear_metrics"]),("b_spatial_nystroem",primary["metrics"])):
            comparisons.append(_comparison_row(outer,candidate_role,base_metrics,metrics,int(output["n_eval"])))
        for seed in config.stability_landmark_seeds:
            row=seed_map[seed]; stability_predictions.extend(row["rows"]); delta=float(row["metrics"]["bacc"])-float(base_metrics["bacc"]); stability.append({"schema_version":"midogpp_uniform_b_spatial_seed_stability_v1","outer_center":outer,"landmark_seed":seed,"candidate_id":output["nystroem_lock"]["candidate_id"],"current_bplus_bacc":base_metrics["bacc"],"spatial_bacc":row["metrics"]["bacc"],"delta_bacc":delta,"primary_delta_bacc":float(primary["metrics"]["bacc"])-float(base_metrics["bacc"])})
    all_predictions=linear_b_rows+current_rows+spatial_linear+spatial_primary
    return {"results":results,"all_predictions":all_predictions,"current_bplus":current_rows,"spatial_linear":spatial_linear,"spatial_primary":spatial_primary,"stability_predictions":stability_predictions,"comparisons":comparisons,"stability":stability,"audits":audits}

def _comparison_row(center: str, role: str, baseline: Mapping[str, float], candidate: Mapping[str, float], n: int) -> dict[str, object]:
    return {"schema_version":"midogpp_uniform_b_spatial_center_comparison_v1","outer_center":center,"candidate_role":role,"n_eval":n,"current_bplus_bacc":baseline["bacc"],"candidate_bacc":candidate["bacc"],"delta_bacc":candidate["bacc"]-baseline["bacc"],"delta_positive_recall":candidate["positive_recall"]-baseline["positive_recall"],"delta_specificity":candidate["specificity"]-baseline["specificity"],"strict_win":candidate["bacc"]>baseline["bacc"]}

def _hard_core_exchange(canonical_linear: Mapping[str, Mapping[str, object]], current_bplus: Mapping[str, Mapping[str, object]], spatial: Sequence[Mapping[str, object]], centers: Sequence[str]) -> list[dict[str, object]]:
    spatial_by={str(row["sample_id"]):row for row in spatial}; groups=[("overall","all",list(spatial))]
    for center in centers:
        for label in (0,1): groups.append(("center_class",f"{center}:{label}",[row for row in spatial if str(row["center"])==center and int(row["y_true"])==label]))
    output=[]
    for scope,value,rows in groups:
        counts={
            "current_shared_hard": 0,
            "hard_core_rescued": 0,
            "hard_core_unresolved": 0,
            "current_bplus_correct_spatial_wrong": 0,
            "current_bplus_wrong_spatial_correct": 0,
        }
        for row in rows:
            sid=str(row["sample_id"]); truth=int(row["y_true"]); linear_wrong=int(canonical_linear[sid]["y_pred"])!=truth; current_wrong=int(current_bplus[sid]["y_pred"])!=truth; spatial_wrong=int(spatial_by[sid]["y_pred"])!=truth
            if linear_wrong and current_wrong: counts["current_shared_hard"]+=1; counts["hard_core_rescued"]+=int(not spatial_wrong); counts["hard_core_unresolved"]+=int(spatial_wrong)
            counts["current_bplus_correct_spatial_wrong"]+=int(not current_wrong and spatial_wrong); counts["current_bplus_wrong_spatial_correct"]+=int(current_wrong and not spatial_wrong)
        output.append({"schema_version":"midogpp_uniform_b_spatial_hard_core_exchange_v1","scope":scope,"scope_value":value,"n":len(rows),**counts,"net_rescue_vs_current_bplus":counts["current_bplus_wrong_spatial_correct"]-counts["current_bplus_correct_spatial_wrong"]})
    return output

def _error_exchange(candidate: Sequence[Mapping[str, object]], baseline: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    base={str(row["sample_id"]):row for row in baseline}; groups=[("overall","all",list(candidate))]
    groups += [("center",c,[r for r in candidate if str(r["center"])==c]) for c in sorted({str(r["center"]) for r in candidate},key=int)]
    groups += [("class",str(y),[r for r in candidate if int(r["y_true"])==y]) for y in (0,1)]
    out=[]
    for scope,value,rows in groups:
        counts={
            "baseline_wrong_candidate_correct": 0,
            "baseline_correct_candidate_wrong": 0,
            "both_wrong": 0,
            "both_correct": 0,
        }
        for row in rows:
            bc=int(base[str(row["sample_id"])]["y_pred"])==int(row["y_true"]); cc=int(row["y_pred"])==int(row["y_true"]); counts[{(False,True):"baseline_wrong_candidate_correct",(True,False):"baseline_correct_candidate_wrong",(False,False):"both_wrong",(True,True):"both_correct"}[(bc,cc)]]+=1
        out.append({"schema_version":"midogpp_uniform_b_spatial_error_exchange_v1","scope":scope,"scope_value":value,"n":len(rows),**counts,"net_rescue":counts["baseline_wrong_candidate_correct"]-counts["baseline_correct_candidate_wrong"]})
    return out

def _progression_decision(comparisons: Sequence[Mapping[str, object]], stability: Sequence[Mapping[str, object]], hard_core: Sequence[Mapping[str, object]], bootstrap: Mapping[str, object], config: SpatialProbeConfig) -> dict[str, object]:
    primary=[row for row in comparisons if row["candidate_role"]=="b_spatial_nystroem"]; deltas=[float(row["delta_bacc"]) for row in primary]; direction=[float(row[key]) for row in primary for key in ("delta_positive_recall","delta_specificity")]; hard=next(row for row in hard_core if row["scope"]=="overall")
    by_seed=defaultdict(list)
    for row in stability: by_seed[int(row["landmark_seed"])].append(row)
    seed_checks={}
    for seed,rows in by_seed.items():
        mean=float(np.mean([float(r["delta_bacc"]) for r in rows])); primary_mean=float(np.mean([float(r["primary_delta_bacc"]) for r in rows])); worst=min(float(r["delta_bacc"]) for r in rows); seed_checks[str(seed)]={"mean_delta":mean,"absolute_deviation_from_primary":abs(mean-primary_mean),"worst_center_delta":worst,"passed":mean>config.gate.stability_mean_delta_min_exclusive and abs(mean-primary_mean)<=config.gate.stability_primary_deviation_max and worst>=config.gate.stability_worst_center_delta_min}
    observed={"equal_center_mean_delta":float(np.mean(deltas)),"strict_center_wins":sum(float(v)>0 for v in deltas),"worst_center_delta":min(deltas),"worst_center_class_direction_delta":min(direction),"hard_core_rescued":int(hard.get("hard_core_rescued",0)),"net_rescue_vs_current_bplus":int(hard.get("net_rescue_vs_current_bplus",0)),"seed_checks":seed_checks,"bootstrap_lower_supportive":bootstrap.get("percentile_2_5")}
    checks={"mean_bacc_delta":observed["equal_center_mean_delta"]>=config.gate.mean_bacc_delta_min,"strict_center_wins":observed["strict_center_wins"]>=config.gate.strict_center_wins_min,"worst_center_delta":observed["worst_center_delta"]>=config.gate.worst_center_bacc_delta_min,"worst_center_class_direction":observed["worst_center_class_direction_delta"]>=config.gate.worst_center_class_direction_delta_min,"hard_core_net_rescue":observed["net_rescue_vs_current_bplus"]>config.gate.hard_core_net_rescue_min_exclusive,"landmark_seed_stability":all(row["passed"] for row in seed_checks.values())}
    passed=all(checks.values()); return {"schema_version":"midogpp_uniform_b_spatial_progression_v1","decision":"B_SPATIAL_DIAGNOSTIC_GATE_PASS" if passed else "B_SPATIAL_DIAGNOSTIC_GATE_FAIL","passed":passed,"checks":checks,"observed":observed,"thresholds":config.gate.__dict__,"bootstrap_supportive_only":True,"diagnostic_only":True}

def _summary(materialized: Mapping[str, object], hard_core: Sequence[Mapping[str, object]], exchange: Sequence[Mapping[str, object]], bootstrap: Mapping[str, object], decision: Mapping[str, object]) -> dict[str, object]:
    comparisons=materialized["comparisons"]; primary=[r for r in comparisons if r["candidate_role"]=="b_spatial_nystroem"]; linear=[r for r in comparisons if r["candidate_role"]=="b_spatial_linear"]; overall=next(r for r in hard_core if r["scope"]=="overall"); exchange_all=next(r for r in exchange if r["scope"]=="overall")
    current=float(np.mean([float(r["current_bplus_bacc"]) for r in primary])); spatial=float(np.mean([float(r["candidate_bacc"]) for r in primary])); spatial_linear=float(np.mean([float(r["candidate_bacc"]) for r in linear]))
    return {"schema_version":"midogpp_uniform_b_spatial_diagnostic_summary_v1","status":"COMPLETE","decision":decision["decision"],"progression_gate_passed":decision["passed"],"equal_center_current_bplus_bacc":current,"equal_center_b_spatial_linear_bacc":spatial_linear,"equal_center_b_spatial_nystroem_bacc":spatial,"equal_center_delta_vs_current_bplus":spatial-current,"hard_core_exchange":dict(overall),"overall_error_exchange":dict(exchange_all),"bootstrap":dict(bootstrap),"progression":dict(decision),"diagnostic_only":True,"validation_scored":False,"test_scored":False}

def _render_report(summary: Mapping[str, object]) -> str:
    hard=summary["hard_core_exchange"]; return "\n".join(["# Uniform-B Spatial Diagnostic","",f"Decision: `{summary['decision']}`.","","This is a bounded diagnostic on the inspected train surface; it cannot replace canonical B.","",f"- Current B+ BACC: `{summary['equal_center_current_bplus_bacc']:.6f}`",f"- B-spatial linear BACC: `{summary['equal_center_b_spatial_linear_bacc']:.6f}`",f"- B-spatial Nyström BACC: `{summary['equal_center_b_spatial_nystroem_bacc']:.6f}`",f"- Delta versus current B+: `{summary['equal_center_delta_vs_current_bplus']:+.6f}`",f"- Current 1,318 hard-core rescues: `{hard.get('hard_core_rescued',0)}`",f"- Net rescue versus current B+: `{hard.get('net_rescue_vs_current_bplus',0)}`","","Validation and test were not featurized or scored.",""])

def _linear_locks(root: Path) -> dict[str, dict[str, object]]:
    rows=_read_csv(root / "tables/classifier_tuned_source_results.csv"); out={}
    for row in rows:
        spec=json.loads(row["selected_classifier_spec"]); out[str(row["heldout_center"])]={k:spec[k] for k in ("C","class_weight","max_iter","penalty","solver")}
    return out

def _nystroem_locks(root: Path) -> dict[str, dict[str, object]]:
    rows=[r for r in _read_csv(root / "tables/outer_results.csv") if r["model_role"]=="canonical_b_nystroem_primary"]; out={}
    for row in rows: out[str(row["outer_center"])]={"candidate_id":row["candidate_id"],"width_multiplier":float(row["width_multiplier"]),"n_components":int(float(row["n_components"])),"logistic_c":float(row["logistic_c"]),"class_weight":None if row["inherited_class_weight"]=="none" else row["inherited_class_weight"]}
    return out

def _baseline_predictions(root: Path) -> tuple[dict[str,dict[str,object]],dict[str,dict[str,object]]]:
    rows=_read_csv(root / "tables/outer_predictions.csv"); linear={r["sample_id"]:r for r in rows if r["model_role"]=="canonical_b_linear_baseline"}; full={r["sample_id"]:r for r in rows if r["model_role"]=="canonical_b_nystroem_primary"}; return linear,full

def _normalized_baseline(row: Mapping[str, object], role: str) -> dict[str, object]:
    return {"schema_version":"midogpp_uniform_b_spatial_prediction_v1","model_role":role,"outer_center":str(row["outer_center"]),"sample_id":str(row["sample_id"]),"case_id":str(row["case_id"]),"center":str(row["center"]),"y_true":int(row["y_true"]),"y_pred":int(row["y_pred"]),"prob_pos":float(row["prob_pos"]),"candidate_id":str(row["candidate_id"]),"landmark_seed":row.get("landmark_seed",""),"eval_row_hash":str(row["eval_row_hash"]),"target_labels_used_for_scoring_only":True,"fit_used_target_center":False,"capacity_reselected":False}

def _frozen_protocol(config: SpatialProbeConfig, hashes: Mapping[str,str], linear: Mapping[str,object], nystroem: Mapping[str,object]) -> dict[str,object]:
    payload={"schema_version":"midogpp_uniform_b_spatial_frozen_protocol_v1","experiment_name":config.name,"representation_id":"annotation_jpeg_fixed_center_b_spatial_quadrants_v1","heldout_centers":list(config.heldout_centers),"feature_dim":SPATIAL_DIM,"spatial_summary":"ordered_TL_TR_BL_BR_2x2_means_of_central_4x4","linear_locks":linear,"nystroem_locks":nystroem,"classifier_capacity_reselected":False,"primary_landmark_seed":config.primary_landmark_seed,"stability_landmark_seeds":list(config.stability_landmark_seeds),"threshold_policy":"predict","gate":config.gate.__dict__,"input_hashes":dict(sorted(hashes.items())),"claim_scope":"diagnostic_only"}; payload["protocol_hash"]=stable_hash(payload); return payload

def _input_hashes(config: SpatialProbeConfig, spatial_hash: str, manifest_hash: str) -> dict[str,str]:
    return {"dataset_manifest":manifest_hash,"spatial_feature_cache":spatial_hash,"canonical_b_cache":_sha256_file(config.canonical_b_cache_path),"canonical_reference_content_index":_sha256_file(config.canonical_reference_root / "manifests/content_index.json"),"nonlinear_reference_content_index":_sha256_file(config.nonlinear_reference_root / "manifests/content_index.json"),"spatial_cache_content_index":_sha256_file(config.spatial_cache_root / "manifests/content_index.json")}

def _read_csv(path: Path) -> list[dict[str,str]]:
    with path.open(newline="",encoding="utf-8") as handle: return [dict(row) for row in csv.DictReader(handle)]

def _read_json(path: Path) -> dict[str,object]:
    value=json.loads(path.read_text());
    if not isinstance(value,dict): raise ProtocolError(f"Expected JSON object: {path}.")
    return value

def _string_hash(values: Sequence[object]) -> str:
    digest=hashlib.sha256()
    for value in values: digest.update(str(value).encode()); digest.update(b"\0")
    return digest.hexdigest()

def _array_hash(arrays: Sequence[np.ndarray]) -> str:
    digest=hashlib.sha256()
    for array in arrays:
        value=np.ascontiguousarray(array); digest.update(str(value.dtype).encode()); digest.update(json.dumps(list(value.shape)).encode()); digest.update(value.tobytes())
    return digest.hexdigest()

def _sha256_file(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()

def _write_content_index(root: Path) -> None:
    files=[{"path":str(path.relative_to(root)),"sha256":_sha256_file(path)} for path in sorted(root.rglob("*")) if path.is_file() and path.name!="content_index.json"]; payload={"schema_version":"midogpp_uniform_b_spatial_content_index_v1","files":files}; payload["content_hash"]=stable_hash(payload); write_json(root / "manifests/content_index.json",payload)
