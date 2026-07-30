"""Independent validation for the bounded B-spatial diagnostic bundle."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping

from midogpp_thesis.common.hashing import stable_hash
from ..protocol import ProtocolError
from ..uniform_b_nonlinear_probe.statistics import paired_case_bootstrap
from .config import EXPECTED_ROWS, SpatialProbeConfig

def validate_spatial_probe_bundle(root: str | Path, *, config: SpatialProbeConfig, allow_pending: bool = False) -> dict[str, object]:
    from .runner import _hard_core_exchange, _progression_decision
    path=Path(root)
    required={
        "provenance/input_artifacts.json", "manifests/frozen_protocol_snapshot.json",
        "manifests/inherited_classifier_locks.json", "manifests/protocol_manifest.json",
        "manifests/content_index.json", "tables/kernel_fit_audit.csv", "tables/outer_results.csv",
        "tables/outer_predictions.csv", "tables/stability_predictions.csv", "tables/center_comparison.csv",
        "tables/seed_stability.csv", "tables/error_exchange.csv", "tables/hard_core_exchange.csv",
        "reports/paired_bootstrap.json", "reports/progression_decision.json", "reports/diagnostic_summary.json",
        "reports/diagnostic_report.md", "reports/runtime_summary.json", "reports/leakage_provenance_report.json",
    }
    if not allow_pending: required.add("reports/validation_report.json")
    missing=sorted(name for name in required if not (path/name).is_file())
    if missing: raise ProtocolError(f"Uniform-B spatial bundle is incomplete: {missing}.")
    protocol=_read_json(path/"manifests/protocol_manifest.json"); leakage=_read_json(path/"reports/leakage_provenance_report.json")
    expected_status="PENDING_INDEPENDENT_VALIDATION" if allow_pending else "PASS"
    if (protocol.get("status") != expected_status or leakage.get("status") != expected_status
        or protocol.get("claim_scope") != "diagnostic_only" or protocol.get("validation_scored") is not False
        or protocol.get("test_scored") is not False or leakage.get("classifier_capacity_reselected") is not False):
        raise ProtocolError("Uniform-B spatial protocol or leakage status drifted.")
    frozen=_read_json(path/"manifests/frozen_protocol_snapshot.json"); unhashed={k:v for k,v in frozen.items() if k!="protocol_hash"}
    if stable_hash(unhashed) != frozen.get("protocol_hash") or frozen.get("feature_dim") != 7680:
        raise ProtocolError("Uniform-B spatial frozen protocol hash drifted.")
    results=_read_csv(path/"tables/outer_results.csv"); predictions=_read_csv(path/"tables/outer_predictions.csv")
    stability_predictions=_read_csv(path/"tables/stability_predictions.csv"); comparisons=_read_csv(path/"tables/center_comparison.csv")
    stability=_read_csv(path/"tables/seed_stability.csv"); audits=_read_csv(path/"tables/kernel_fit_audit.csv")
    exchange=_read_csv(path/"tables/error_exchange.csv"); hard=_read_csv(path/"tables/hard_core_exchange.csv")
    if (len(results)!=27 or len(predictions)!=4*EXPECTED_ROWS or len(stability_predictions)!=2*EXPECTED_ROWS
        or len(comparisons)!=18 or len(stability)!=18 or len(audits)!=27 or len(exchange)!=12 or len(hard)!=19):
        raise ProtocolError("Uniform-B spatial artifact cardinality drifted.")
    by_role={role:[row for row in predictions if row["model_role"]==role] for role in ("canonical_b_linear","canonical_b_nystroem","b_spatial_linear","b_spatial_nystroem")}
    if any(len(rows)!=EXPECTED_ROWS for rows in by_role.values()): raise ProtocolError("Uniform-B spatial prediction-role coverage drifted.")
    source=[row for row in _read_csv(config.nonlinear_reference_root/"tables/outer_predictions.csv") if row["model_role"] in {"canonical_b_linear_baseline","canonical_b_nystroem_primary"}]
    source_map={("canonical_b_linear" if row["model_role"]=="canonical_b_linear_baseline" else "canonical_b_nystroem",row["sample_id"]):(int(row["y_pred"]),float(row["prob_pos"]),row["eval_row_hash"]) for row in source}
    observed={(row["model_role"],row["sample_id"]):(int(row["y_pred"]),float(row["prob_pos"]),row["eval_row_hash"]) for row in by_role["canonical_b_linear"]+by_role["canonical_b_nystroem"]}
    if source_map != observed: raise ProtocolError("Uniform-B spatial imported baseline identity drifted.")
    val_test=_manifest_split_ids(config.manifest_path,{"val","test"}); predicted={row["sample_id"] for row in predictions}
    if predicted & val_test: raise ProtocolError("Validation/test rows leaked into B-spatial predictions.")
    recomputed_hard=_hard_core_exchange({r["sample_id"]:r for r in by_role["canonical_b_linear"]},{r["sample_id"]:r for r in by_role["canonical_b_nystroem"]},by_role["b_spatial_nystroem"],config.heldout_centers)
    _assert_rows_equal(hard,recomputed_hard)
    bootstrap=paired_case_bootstrap(by_role["b_spatial_nystroem"],by_role["canonical_b_nystroem"],centers=config.heldout_centers,replicates=config.gate.bootstrap_replicates,seed=config.gate.bootstrap_seed)
    _assert_nested_close(_read_json(path/"reports/paired_bootstrap.json"),bootstrap)
    decision=_progression_decision(comparisons,stability,recomputed_hard,bootstrap,config)
    _assert_nested_close(_read_json(path/"reports/progression_decision.json"),decision)
    _validate_content_index(path)
    checks={"status":"PASS","outer_results":len(results),"outer_predictions":len(predictions),"stability_predictions":len(stability_predictions),"baseline_identity":"EXACT","capacity_reselected":False,"validation_scored":False,"test_scored":False,"decision":decision["decision"]}
    if not allow_pending:
        report=_read_json(path/"reports/validation_report.json")
        if report.get("status")!="PASS" or report.get("checks")!=checks: raise ProtocolError("Uniform-B spatial validation report drifted.")
    return checks

def _read_csv(path: Path) -> list[dict[str,str]]:
    with path.open(newline="",encoding="utf-8") as handle: return [dict(row) for row in csv.DictReader(handle)]

def _read_json(path: Path) -> dict[str,object]:
    value=json.loads(path.read_text());
    if not isinstance(value,dict): raise ProtocolError(f"Expected JSON object: {path}.")
    return value

def _manifest_split_ids(path: Path, splits: set[str]) -> set[str]:
    return {row["sample_id"] for row in _read_csv(path) if row["split"].lower() in splits}

def _assert_rows_equal(stored: list[dict[str,str]], expected: list[dict[str,object]]) -> None:
    if len(stored)!=len(expected): raise ProtocolError("B-spatial hard-core row count drifted.")
    for left,right in zip(stored,expected,strict=True):
        if set(left)!=set(right): raise ProtocolError("B-spatial hard-core columns drifted.")
        for key,value in right.items():
            raw=left[key]
            if isinstance(value,bool): ok=raw.lower()==str(value).lower()
            elif isinstance(value,int): ok=int(raw)==value
            elif isinstance(value,float): ok=abs(float(raw)-value)<=1e-10
            else: ok=raw==str(value)
            if not ok: raise ProtocolError(f"B-spatial hard-core value drifted: {key}.")

def _assert_nested_close(left: object,right: object) -> None:
    if isinstance(right,Mapping):
        if not isinstance(left,Mapping) or set(left)!=set(right): raise ProtocolError("B-spatial JSON mapping drifted.")
        for key,value in right.items(): _assert_nested_close(left[key],value)
    elif isinstance(right,list):
        if not isinstance(left,list) or len(left)!=len(right): raise ProtocolError("B-spatial JSON list drifted.")
        for a,b in zip(left,right,strict=True): _assert_nested_close(a,b)
    elif isinstance(right,float):
        if abs(float(left)-right)>1e-10: raise ProtocolError("B-spatial JSON float drifted.")
    elif left!=right: raise ProtocolError("B-spatial JSON value drifted.")

def _sha256_file(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()

def _validate_content_index(root: Path) -> None:
    payload=_read_json(root/"manifests/content_index.json"); unhashed={k:v for k,v in payload.items() if k!="content_hash"}
    if stable_hash(unhashed)!=payload.get("content_hash"): raise ProtocolError("B-spatial content hash drifted.")
    expected={str(path.relative_to(root)) for path in root.rglob("*") if path.is_file() and path.name!="content_index.json"}; observed=set()
    for row in payload.get("files",[]):
        member=root/str(row["path"]); observed.add(str(row["path"]))
        if not member.is_file() or _sha256_file(member)!=row.get("sha256"): raise ProtocolError(f"B-spatial member drifted: {member}.")
    if observed!=expected: raise ProtocolError("B-spatial content-index coverage drifted.")
