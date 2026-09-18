"""Export complete small historical evidence; CPU only, explicit external roots.

The historical originals are never edited. No CUDA library is loaded.
"""
from pathlib import Path
import argparse,csv,hashlib,json,math,re,statistics

HISTORICAL_LIBRARY_SHA256='e79a7395cbf0b19e04b0a626bcbf5df2a22f908d0983bef6ee431e36031f9dea'
SEAL_SHA256='ec751a08a3a3d7ab1b1db73e10577bf09bd3535bd5d9e0bdd4cf3bf73b07c80d'
COLUMNS=('quartet_id','process','arm','kind','pair_id','K','boundary','method','style','role','order','alias','R',
    'wall_ms','unit_ms','per_job_amortized_ms','jobs_per_second','logical_queries_per_second','completed_solves',
    'equivalent_real_tasks','logical_full_field_queries','status','observed_status','reason','timing_valid',
    'execution_contract_ok','sum_entry_unit_ms','outer_loop_record_overhead_ms','library_sha256','worker_sha256')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def record(root,rel):p=root/rel;return dict(path=rel,bytes=p.stat().st_size,sha256=sha(p))
def read(root,rel):return json.loads((root/rel).read_text(encoding='utf-8-sig'))
def write_json(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False);f.write('\n')

def export(evidence,backend,build,out):
    seal=read(evidence,'delivery/SHA256SUMS.json')
    if sha(evidence/'delivery/SHA256SUMS.json')!=SEAL_SHA256:raise ValueError('Wrong historical seal')
    if seal.get('status')!='SEALED':raise ValueError('Evidence not sealed')
    for rel,r in seal['files'].items():
        p=(evidence/rel).resolve()
        if not p.is_relative_to(evidence.resolve()) or not p.is_file() or p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:
            raise ValueError('Historical seal mismatch: '+rel)
    a=read(evidence,'delivery/ANALYSIS.json');verdict=read(evidence,'delivery/VERDICT.json')
    plan=read(evidence,'PLAN_SCHEDULE.json');inventory=read(evidence,'INPUT_INVENTORY.json')
    original=list(csv.DictReader((evidence/'delivery/TIMINGS.csv').open(encoding='utf-8')))
    if len(original)!=256 or any(r['status']!='OK' for r in original):raise ValueError('Not the full distinct-task study')
    if a['actual_phase_application_solves']!=6306 or len(a['comparisons'])!=8 or len(a['AA'])!=32:raise ValueError('Historical cardinality mismatch')
    if len({(r['quartet_id'],r['arm']) for r in original})!=256:raise ValueError('Duplicate arm')
    grouped={}
    for r in original:grouped.setdefault(r['quartet_id'],[]).append(r)
    for q in a['quartets']:
        rs=grouped[q['quartet_id']]
        av=[float(r['unit_ms']) for r in rs if r['role']=='A'];bv=[float(r['unit_ms']) for r in rs if r['role']=='B']
        ratio=math.sqrt(av[0]*av[1]/(bv[0]*bv[1]))
        if not math.isclose(q['ratio'],ratio,rel_tol=1e-12):raise ValueError('Raw quartet ratio mismatch')
    if sha(backend/'build/frozen/libthree_ideas_production.so')!=HISTORICAL_LIBRARY_SHA256:raise ValueError('Wrong original library')
    bs=read(build,'build/graph_fix_r2/BUILD_STATUS.json')
    if not any(r.get('sha256')==HISTORICAL_LIBRARY_SHA256 for r in bs.get('artifacts',[])):raise ValueError('Build provenance missing')
    artifacts=('SUMMARY.json','PROTOCOL.json','AA.json','TIMINGS.csv','CALIBRATION_R.json','TASK_IDENTITIES.json','REFERENCE_HASHES.json','EVIDENCE_MANIFEST.json')
    if any((out/p).exists() for p in artifacts):raise FileExistsError('Exclusive historical export already exists')
    out.mkdir(parents=True,exist_ok=True)
    comparisons=[]
    for c in a['comparisons']:
        selected={k:c[k] for k in ('pair_id','method','K','boundary','processes','pooled_raw_ratios','pooled_median','AA_complete','AA_stable','verdict')}
        selected['required_AA_quartets']=[x['quartet_id'] for x in c['required_AA']]
        comparisons.append(selected)
    aa=[{k:q[k] for k in ('quartet_id','process','K','boundary','method','control','candidate','order','status','ratio','stable','raw_unit_ms','raw_wall_ms','arm_keys')} for q in a['AA']]
    phases=dict(correctness=42,limited_memcheck=16,benchmark_process_1=3268,benchmark_process_2=2980,total=6306)
    summary=dict(schema=1,scope='HISTORICAL_OFFLINE_DISTINCT_REAL_TASK_BATCHING',date='2026-09-18',
        experiment_id='distinct_real_tasks_07',baseline_commit='328e84fa079216789acde0306dde4b0047b13c43',
        historical_library_sha256=HISTORICAL_LIBRARY_SHA256,hardware='NVIDIA GeForce RTX 5080',
        scenario_count=1,total_tasks=10,different_geometries_per_K=5,K_values=[3,6],no_replication_to_fill_batches=True,
        geometry_shared_between_corresponding_K3_K6_tasks=True,mixed_shape=dict(N=108,M=581,L=60),
        formal_arms=256,formal_status_counts={'OK':256},AA_total=32,AA_pass=23,AA_fail=9,processes=2,
        process_seeds=plan['process_seeds'],actual_application_solves=phases,
        benchmark_count_formula='per process: 484 nonformal + 96*sum(four frozen R); R repeats are not independent samples',
        comparisons=comparisons,absolute_paired_unit_summaries=a['absolute_paired_unit_summaries'],
        correctness=dict(status='PASS_42_ACTUAL_SOLVES',old_golden_jobs=['K3_batch0','K6_batch3'],
            reference_scope='10 fresh independent Published original jobs; 32 forward/reverse mixed runs; two methods, K values and audit flags, repeated twice'),
        memcheck=dict(status='PASS_FOUR_LIMITED_CALLS',methods=['D1_PUBLISHED','D1_GRAPH'],audit_flags=[0,1],
            solves_per_call=4,total_solves=16,tools_not_covered=['racecheck','initcheck','synccheck']),
        memory=dict(max_global_tracked_bytes=121217640,max_global_graph_budget_reservation_bytes=671088640,
            max_global_combined_bytes=792306280,cudaMemGetInfo_free_min=15438184448,cudaMemGetInfo_free_max=15681454080,
            tracked_and_reserved_are_separate=True,graph_reservation_not_actual_graph_bytes=True,device_peak_unknown=True,
            observations_not_summed_across_handles=True),
        historical_new_GPU_worker_seconds=verdict['new_GPU_worker_seconds'],
        historical_cumulative_GPU_worker_seconds=verdict['cumulative_GPU_worker_seconds'],
        integration=dict(published='PRIMARY_PACKAGED_BACKEND',graph='OPTIONAL_BACKEND_NOT_PACKAGED',
            new_build_and_smoke='SEPARATE_REGRESSION_EVIDENCE_NOT_HISTORICAL_PERFORMANCE_RECERTIFICATION',
            historical_256_arms_rerun=False),
        limitations=['Only one scene; ready-host availability assumed, online queue/batch wait unknown',
            'Other six performance comparisons remain INCONCLUSIVE_TIMING',
            'Not full mesher or single-request latency acceleration',
            'No independent stage-median sum or chained paired ratios',
            'Historical A/B sanitizer failures and SPEC2 coverage gaps remain unchanged'])
    write_json(out/'SUMMARY.json',summary)
    # Complete frozen protocol and schedule: byte-identical original, new public filename.
    with (out/'PROTOCOL.json').open('xb') as f:f.write((evidence/'PLAN_SCHEDULE.json').read_bytes())
    write_json(out/'AA.json',dict(schema=1,source='delivery/ANALYSIS.json',AA_interval=[.95,1.05],records=aa))
    with (out/'TIMINGS.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=COLUMNS,extrasaction='ignore');writer.writeheader()
        for r in original:writer.writerow({k:r.get(k,'') for k in COLUMNS})
    write_json(out/'CALIBRATION_R.json',dict(schema=1,processes=[read(evidence,f'results/process_{p}/R_FROZEN.json') for p in (1,2)]))
    tasks={k:inventory[k] for k in ('scenario_count','job_count','field_sha256','numeric_profile','frozen_library_sha256','lattice_count','tasks','geometry_fingerprint_format','cross_task_identical_geometry','groups')}
    write_json(out/'TASK_IDENTITIES.json',tasks)
    with (out/'REFERENCE_HASHES.json').open('xb') as f:f.write((evidence/'results/correctness/reference/MANIFEST.json').read_bytes())
    original_names=['delivery/SHA256SUMS.json','delivery/ANALYSIS.json','delivery/VERDICT.json','delivery/REPORT.md',
        'delivery/TIMINGS.csv','delivery/INDEPENDENT_AUDIT.json','delivery/MEMORY_OBSERVATIONS.csv',
        'delivery/SOLVER_WORKSPACE_AND_GRAPH.csv','delivery/FRESH_HELPER_CHECK.json','INPUT_INVENTORY.json',
        'PLAN_SCHEDULE.json','PROTOCOL.md','EXECUTION_FROZEN.json','mixed_common.py','mixed_worker.py',
        'plan_mixed.py','run_mixed.py','analyze_mixed.py','audit_mixed.py','results/correctness/RESULT.json',
        'results/correctness/reference/MANIFEST.json','results/SAFETY.json','results/process_1/RESULT.json','results/process_2/RESULT.json',
        'results/process_1/R_FROZEN.json','results/process_2/R_FROZEN.json','results/FINAL_SOURCE_RECHECK.json',
        'results/GPU_WORKER_BUDGET.json']
    backend_names=['build/frozen/libthree_ideas_production.so','python/typed_api.py','python/input_loader.py',
        'python/pilot.py','python/experiment_supervisor.py','src/three_ideas_core.cu','src/three_ideas_api.h',
        'src/ti_memory_guard.h','src/field_admission.h','src/field_bridge_isolated.cu','src/d1_graph_control.cuh',
        'src/graph_query.cuh','src/exports.map','results/gates_final/GATES.json',
        'reference/BinocMesher_Three_Ideas_SIMT_20260918/reference_screen/inputs/fields.bin',
        'reference/BinocMesher_Three_Ideas_SIMT_20260918/reference_screen/inputs/natural_jobs.bin']
    build_names=['build/graph_fix_r2/BUILD_PLAN.json','build/graph_fix_r2/BUILD_STATUS.json',
        'build/graph_fix_r2/SOURCE_RECHECK.json','build/graph_fix_r2/INCLUDE_CLOSURE.json']
    for rel in ('src/d4_solver.cu','src/d4_pipeline.cuh','src/d4_field_eval.cuh','src/gpu_solver.cu',
        'src/field_eval.cuh','src/field_types.h','src/field_bridge.cu','include/d4_solver_api.h'):
        build_names.append('checkout/gpu_accelerate/'+rel)
    provenance=dict(schema=1,scope='ORIGINAL_AND_PUBLIC_DERIVED_IDENTITIES_SEPARATELY',
        external_roots={'external_evidence_root':'User-supplied sealed distinct-task experiment containing delivery/SHA256SUMS.json',
            'external_backend_root':'User-supplied frozen historical adapter, production DSO, typed bindings and original inputs',
            'external_build_root':'User-supplied original graph_fix_r2 build evidence and fixed checkout'},
        original_seal_verification=dict(sha256=SEAL_SHA256,files_verified=len(seal['files']),all_sizes_and_hashes_match=True),
        originals={alias:[record(root,p) for p in selected] for alias,root,selected in
            [('external_evidence_root',evidence,original_names),('external_backend_root',backend,backend_names),('external_build_root',build,build_names)]},
        derived=[record(out,p) for p in artifacts if p!='EVIDENCE_MANIFEST.json'],
        transformations={'PROTOCOL.json':'Byte-identical original PLAN_SCHEDULE.json; full fixed schedule preserved',
            'TIMINGS.csv':'All 256 original rows in original order; retained exact selected numeric/text fields; removed output arrays/handles and retained originals externally',
            'SUMMARY.json':'All eight verdicts and direct ratio evidence from sealed ANALYSIS/VERDICT; no integration performance claim',
            'AA.json':'All 32 A/A records, original ratios/raw quartet times and identities',
            'TASK_IDENTITIES.json':'Original identity inventory without machine-local absolute dependency paths',
            'CALIBRATION_R.json':'Original two process freeze objects, all four R and three samples each',
            'REFERENCE_HASHES.json':'Byte-identical original correctness/reference/MANIFEST.json; per-task eleven output byte lengths and SHA256, comparison only, never solver input'},
        retrieve='Obtain original nonpublic assets/evidence from their owner. Set the three external root arguments explicitly and verify their hashes before reuse.',
        not_committed=['Scene geometry/heightmaps/field assets','Shared libraries and compiled outputs','Third-party trees','Large raw analysis/report bundles'],
        schema_note='Historical schema 1 objects; public schema 1. The manifest intentionally does not hash itself.',
        no_absolute_personal_paths=True,no_credentials=True,no_GPU_UUID=True)
    write_json(out/'EVIDENCE_MANIFEST.json',provenance)
    for name in artifacts:
        text=(out/name).read_text(encoding='utf-8')
        if any(marker in text for marker in ('/mnt/','/home/','/Users/','\\\\Users\\\\')) or re.search(r'\bGPU-[0-9a-fA-F]{8}-',text):raise ValueError('Private path/identity in export '+name)
    return dict(status='PASS_COMPLETE_PUBLIC_HISTORY_EXPORT',arms=256,AA=32,comparisons=8,total_solves=6306,
        bytes=sum((out/p).stat().st_size for p in artifacts),artifacts=list(artifacts),GPU_execution=False)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--external-evidence-root',type=Path,required=True);p.add_argument('--external-backend-root',type=Path,required=True)
    p.add_argument('--external-build-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(export(a.external_evidence_root.resolve(),a.external_backend_root.resolve(),a.external_build_root.resolve(),a.output.resolve()),indent=2))
if __name__=='__main__':main()
