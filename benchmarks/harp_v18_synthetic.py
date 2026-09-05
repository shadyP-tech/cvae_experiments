"""CPU-only HARP v18 runtime benchmark using generated, non-scientific fixtures.

Run: PYTHONPATH=src conda run -n thesis python benchmarks/harp_v18_synthetic.py \
    --output-dir /private/tmp/harp_v18_synthetic_benchmark

No original caches, checkpoints, authorities, leases, or evaluation labels are
read. This checks runtime geometry and CPU cost, never real routing utility.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import resource
import time
import tempfile

import numpy as np
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from midogpp_thesis.cvae.routing.case_conditional_composite_router_v18 import RouterFitConfig
from midogpp_thesis.cvae.routing.case_conditional_composite_router_v18 import crossfit, policy, stacked_fitting
from midogpp_thesis.cvae.runtime.harp_v18_execution.contracts import ActionKind, LabelFreeActionBlock, LabelFreeOuterMenu
from midogpp_thesis.cvae.runtime.harp_v18_execution.support_target_adapter import compile_support_target_menus
from midogpp_thesis.cvae.runtime.harp_v18_execution.support_model_artifacts import build_support_outcome_artifact, build_support_router_artifact


def physical_fixture(center: str, *, case_count: int, samples_per_case: int) -> LabelFreeOuterMenu:
    blocks=[]
    donors=tuple(c for c in CENTERS if c != center)
    for role,count in (("source_train",case_count),("target",1)):
        cases=tuple(f"synthetic-{role}-{center}-{i:02d}" for i in range(count))
        case_ids=tuple(c for c in cases for _ in range(samples_per_case))
        samples=tuple(f"{c}:sample{i:03d}" for c in cases for i in range(samples_per_case))
        for kind,donor in ((ActionKind.B,None),(ActionKind.U,None),*((ActionKind.HXE,d) for d in donors)):
            if kind is ActionKind.B:
                four=(.2,.3,.8,.7)
            elif kind is ActionKind.U:
                four=(.58,.57,.42,.43)
            else:
                i=donors.index(donor)
                four=(.60+.02*i,.56+.02*i,.40-.02*i,.44-.02*i)
            probability=np.asarray(four*(samples_per_case//4)*count,dtype=np.float32)
            blocks.append(LabelFreeActionBlock(surface_role=role,outer_target_id=center,query_center_id=center,
                action_kind=kind,selected_source_id=donor,sample_ids=samples,case_ids=case_ids,
                probabilities=probability,seed_dispersion=np.full(len(probability),.01,dtype=np.float32)))
    return LabelFreeOuterMenu(outer_target_id=center,blocks=tuple(sorted(blocks,key=lambda b:b.key)),
                             lineage={"fixture":"harp_v18_synthetic_benchmark_only","center":center})


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--cases-per-center',type=int,default=24)
    parser.add_argument('--samples-per-case',type=int,default=48)
    args=parser.parse_args()
    output=args.output_dir.resolve()
    temporary_roots = {Path(tempfile.gettempdir()).resolve(), Path('/tmp').resolve()}
    if not any(output.is_relative_to(root) and output != root for root in temporary_roots):
        raise ValueError('Synthetic benchmark outputs require a subdirectory of the system temporary directory.')
    if args.cases_per_center!=24 or args.samples_per_case<4 or args.samples_per_case%4:
        raise ValueError('Use exactly216sourcecases and sample counts divisible by4.')
    output.mkdir(parents=True,exist_ok=True)
    result_path=output/'benchmark.json'
    if result_path.exists():
        raise FileExistsError('Refusing to overwrite an existing benchmark result.')
    started=time.perf_counter()
    fit_counts={'stacks':0,'rankers':0}
    original_stack=stacked_fitting.fit_stacked_science_model
    original_proposal=stacked_fitting.fit_proposal_model
    def counted_proposal(*positional,**kwargs):
        fit_counts['rankers']+=1
        return original_proposal(*positional,**kwargs)
    def counted_stack(*positional,**kwargs):
        fit_counts['stacks']+=1
        index=fit_counts['stacks']
        before=time.perf_counter()
        value=original_stack(*positional,**kwargs)
        print(json.dumps({'synthetic_stack':index,'fit_seconds':round(time.perf_counter()-before,3),
                          'elapsed_seconds':round(time.perf_counter()-started,3)}),flush=True)
        return value
    stacked_fitting.fit_proposal_model=counted_proposal
    crossfit.fit_stacked_science_model=counted_stack
    policy.fit_stacked_science_model=counted_stack
    preparation_started=time.perf_counter()
    bundles=tuple(compile_support_target_menus(physical_fixture(c,case_count=args.cases_per_center,
        samples_per_case=args.samples_per_case)) for c in CENTERS)
    labels={bundle.center_id: tuple(
        {"center":bundle.center_id,"case_id":case,"sample_id":sample,"label":label}
        for case,samples in bundle.source_case_samples
        for sample,label in zip(samples,(1,1,0,0)*(args.samples_per_case//4),strict=True)
    ) for bundle in bundles}
    surface=build_support_outcome_artifact(bundles,labels)
    preparation_seconds=time.perf_counter()-preparation_started
    print(json.dumps({'phase':'synthetic_source_ready','seconds':preparation_seconds,
                      'source_cases':216,'samples_per_case':args.samples_per_case}),flush=True)
    fit_started=time.perf_counter()
    result=build_support_router_artifact(surface,config=RouterFitConfig())
    fit_seconds=time.perf_counter()-fit_started
    selected=result.state.policy
    usage=resource.getrusage(resource.RUSAGE_SELF)
    rss_bytes=int(usage.ru_maxrss if platform.system()=='Darwin' else usage.ru_maxrss*1024)
    summary={
        'benchmark_identity':'harp_v18_synthetic_geometry_only','finished_utc':datetime.now(timezone.utc).isoformat(),
        'host_platform':platform.platform(),'machine':platform.machine(),'python':platform.python_version(),
        'source_cases':216,'source_centers':9,'samples_per_case':args.samples_per_case,
        'candidate_configurations_per_case':38,'outer_folds':5,'inner_folds':4,'stack_folds':4,
        'ridge_alpha':1.,'max_numeric_features':20,'shared_outcome_design_columns':len(selected.model.action_model.design_names),
        'runtime_blas_threads':1,'cuda_used':False,'scientific_data_used':False,'evaluation_truth_opened':False,
        'preparation_seconds':preparation_seconds,'fit_seconds':fit_seconds,'total_seconds':time.perf_counter()-started,
        'cpu_user_seconds':usage.ru_utime,'cpu_system_seconds':usage.ru_stime,'peak_rss_bytes':rss_bytes,
        'fit_counts':fit_counts,'frontier_rows':len(selected.crossfit.frontier_rows),
        'outer_oof_cases':len(selected.crossfit.records),'outer_oof_routed':sum(r.route_selected for r in selected.crossfit.records),
        'admission_status':selected.admission.status.value,
        'claim_boundary':'Synthetic CPU performance and construction only; no real routing utility or launch authority.',
    }
    result_path.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
