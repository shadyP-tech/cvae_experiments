"""CPU-only HARP v19 runtime benchmark using generated, non-scientific fixtures.

Run: PYTHONPATH=src conda run -n thesis python benchmarks/harp_v19_synthetic.py \
    --output-dir /private/tmp/harp_v19_synthetic_benchmark

No original caches, checkpoints, authorities, leases, or evaluation labels are
read. This checks runtime geometry and CPU cost, never real routing utility.
"""
from __future__ import annotations
import argparse
import cProfile
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import pstats
import resource
import time
import tempfile

import numpy as np
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from midogpp_thesis.cvae.routing.safe_winner_router_v19 import RouterFitConfig
from midogpp_thesis.cvae.routing.safe_winner_router_v19 import crossfit, learning, policy, proposer
from midogpp_thesis.cvae.routing.safe_winner_router_v19.features import RawFeatureCache, use_raw_feature_cache
from midogpp_thesis.cvae.routing.safe_winner_router_v19.fit_cache import ScopedFitCache
from midogpp_thesis.cvae.routing.safe_winner_router_v19.truth import combine_truth_capabilities
from threadpoolctl import threadpool_limits
from midogpp_thesis.cvae.runtime.harp_v19_execution.contracts import ActionKind, LabelFreeActionBlock, LabelFreeOuterMenu
from midogpp_thesis.cvae.runtime.harp_v19_execution.support_target_adapter import compile_support_target_menus
from midogpp_thesis.cvae.runtime.harp_v19_execution.support_model_artifacts import build_support_outcome_artifact, build_support_router_artifact


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
                             lineage={"fixture":"harp_v19_synthetic_benchmark_only","center":center})


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--cases-per-center',type=int,default=24)
    parser.add_argument('--samples-per-case',type=int,default=48)
    parser.add_argument('--profile', action='store_true', help='Write cProfile data; timings include profiler overhead.')
    parser.add_argument('--complete-only', action='store_true',
                        help='Profile one full 216-case complete learner; skip outer policy evaluation and admission.')
    args=parser.parse_args()
    output=args.output_dir.resolve()
    temporary_roots = {Path(tempfile.gettempdir()).resolve(), Path('/tmp').resolve()}
    if not any(output.is_relative_to(root) and output != root for root in temporary_roots):
        raise ValueError('Synthetic benchmark outputs require a subdirectory of the system temporary directory.')
    if args.cases_per_center!=24 or args.samples_per_case<4 or args.samples_per_case%4:
        raise ValueError('Use exactly 216 source cases and sample counts divisible by 4.')
    output.mkdir(parents=True,exist_ok=True)
    result_path=output/'benchmark.json'
    if result_path.exists():
        raise FileExistsError('Refusing to overwrite an existing benchmark result.')
    started=time.perf_counter()
    # Count actual uncached estimator calls, not a guessed nesting multiplier.
    # These aliases are the call sites used by the complete v19 learner.
    fit_counts = {'complete_learner_requests': 0, 'actual_ranker_fits': 0,
                  'actual_candidate_outcome_fits': 0, 'actual_winner_gate_fits': 0}
    fit_times = {key: 0.0 for key in fit_counts}
    def counted(function, key, *, progress=False):
        def wrapped(*positional, **kwargs):
            fit_counts[key] += 1
            index, before = fit_counts[key], time.perf_counter()
            value = function(*positional, **kwargs)
            elapsed = time.perf_counter() - before
            fit_times[key] += elapsed
            if progress:
                print(json.dumps({'complete_learner_request': index, 'fit_seconds': round(elapsed, 3),
                                  'elapsed_seconds': round(time.perf_counter()-started, 3)}), flush=True)
            return value
        return wrapped
    proposer.fit_proposal_model = counted(proposer.fit_proposal_model, 'actual_ranker_fits')
    proposer.fit_action_outcome_model = counted(proposer.fit_action_outcome_model, 'actual_candidate_outcome_fits')
    learning.fit_winner_gate = counted(learning.fit_winner_gate, 'actual_winner_gate_fits')
    complete = counted(learning.fit_stacked_science_model, 'complete_learner_requests', progress=True)
    crossfit.fit_stacked_science_model = complete
    policy.fit_stacked_science_model = complete
    fit_cache_stats = {'instances': 0, 'hits': 0, 'misses': 0, 'peak_entries': 0}
    original_init, original_get, original_put = ScopedFitCache.__init__, ScopedFitCache.get, ScopedFitCache.put
    def cache_init(instance, *positional, **kwargs):
        fit_cache_stats['instances'] += 1
        original_init(instance, *positional, **kwargs)
    def cache_get(instance, key):
        value = original_get(instance, key)
        fit_cache_stats['misses' if value is None else 'hits'] += 1
        return value
    def cache_put(instance, key, model):
        value = original_put(instance, key, model)
        fit_cache_stats['peak_entries'] = max(fit_cache_stats['peak_entries'], len(instance._models))
        return value
    ScopedFitCache.__init__, ScopedFitCache.get, ScopedFitCache.put = cache_init, cache_get, cache_put
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
    config = RouterFitConfig()
    raw_features = RawFeatureCache(max_entries=8192)
    profiler = cProfile.Profile() if args.profile else None
    if profiler is not None:
        profiler.enable()
    with use_raw_feature_cache(raw_features):
        if args.complete_only:
            with threadpool_limits(limits=1):
                model = complete(surface.state.source_menus,
                    combine_truth_capabilities(surface.state.truth_capabilities), config=config)
            selected = None
        else:
            result = build_support_router_artifact(surface, config=config)
            selected = result.state.policy
            model = selected.model
    if profiler is not None:
        profiler.disable()
        profiler.dump_stats(output / 'fit_profile.pstats')
        with (output / 'fit_profile_top.txt').open('w') as stream:
            pstats.Stats(profiler, stream=stream).sort_stats('cumulative').print_stats(60)
    fit_seconds=time.perf_counter()-fit_started
    usage=resource.getrusage(resource.RUSAGE_SELF)
    rss_bytes=int(usage.ru_maxrss if platform.system()=='Darwin' else usage.ru_maxrss*1024)
    summary={
        'benchmark_identity':'harp_v19_synthetic_geometry_only','finished_utc':datetime.now(timezone.utc).isoformat(),
        'benchmark_mode':'SINGLE_COMPLETE_LEARNER' if args.complete_only else 'FULL_NESTED_SOURCE_POLICY',
        'host_platform':platform.platform(),'machine':platform.machine(),'python':platform.python_version(),
        'source_cases':216,'source_centers':9,'samples_per_case':args.samples_per_case,
        'candidate_configurations_per_case':2+3*len(config.k_values)*len(config.lambda_values),
        'outer_folds':config.outer_folds,'inner_folds':config.inner_folds,'stack_folds':config.stack_folds,
        'winner_folds':config.winner_folds, 'candidate_ridge_alpha':config.candidate_ridge_alpha,
        'winner_gate_ridge_alpha':config.winner_gate_ridge_alpha,'gate_thresholds':config.route_thresholds,
        'max_numeric_features':config.maximum_numeric_features,
        'shared_outcome_design_columns':len(model.action_model.design_names),
        'complete_model_hash':model.model_hash,
        'runtime_blas_threads':1,'cuda_used':False,'scientific_data_used':False,'evaluation_truth_opened':False,
        'preparation_seconds':preparation_seconds,'fit_seconds':fit_seconds,'total_seconds':time.perf_counter()-started,
        'cpu_user_seconds':usage.ru_utime,'cpu_system_seconds':usage.ru_stime,'peak_rss_bytes':rss_bytes,
        'fit_counts':fit_counts,'fit_component_seconds_inclusive':fit_times,
        'fit_cache_stats':fit_cache_stats,'raw_feature_cache_stats':raw_features.public_payload(),
        'profiling_enabled':args.profile,
        'outer_crossfit_performed': selected is not None, 'admission_computed': selected is not None,
        'frontier_rows':None if selected is None else len(selected.crossfit.frontier_rows),
        'outer_oof_cases':None if selected is None else len(selected.crossfit.records),
        'outer_oof_routed':None if selected is None else sum(r.route_selected for r in selected.crossfit.records),
        'admission_status':None if selected is None else selected.admission.status.value,
        'claim_boundary':'Synthetic CPU performance and construction only; no real routing utility or launch authority.',
    }
    result_path.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
