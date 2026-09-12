"""Independent stdlib-only audit of completed G1 raw results. Never runs measured code."""
import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

from workspace_paths import ROOT
SUITE=ROOT/'results/performance_01'
MODES=('R0','R1','L0','L1_8','L1_32')
ISSUES=[]
COUNTS=Counter()
CSV_ROWS=[]
RAW_HASHES={}

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def check(ok,msg):
    COUNTS['assertions']+=1
    if not ok:ISSUES.append(msg)
def close(a,b):
    return isinstance(a,(float,int)) and isinstance(b,(float,int)) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
def same_value(a,b):
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(same_value(x,y) for x,y in zip(a,b))
    if isinstance(a,(float,int)) and not isinstance(a,bool):return close(a,b)
    return a==b
def jsonl(path):
    RAW_HASHES[str(path.relative_to(ROOT))]=sha(path)
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
def quantile(values,p):
    values=sorted(values);loc=(len(values)-1)*p;lower=math.floor(loc);upper=math.ceil(loc)
    return values[lower]+(values[upper]-values[lower])*(loc-lower)
def key(row):return (row['table'],row['case'],row['K'],row['comparison'])
def case_key(row):return f"K{row['K']}:{row['id']}"
def field_key(row):return f"{row['field']}:K{row['K']}"
def pair_options(choice):
    pairs=[('R1',mode) for mode in MODES if mode!='R1']
    primary=(choice['resident'],choice['local'])
    if primary not in pairs:pairs.append(primary)
    return [(a,b,(a,b)==primary) for a,b in pairs]
def expected_orders(matrix,worker,scope,comparison):
    seed=int.from_bytes(hashlib.sha256(f"{matrix['seed']}|{worker}|{scope}|{comparison}".encode()).digest()[:8],'little')
    values=['AB']*10+['BA']*10;random.Random(seed).shuffle(values);return values

def audit_event(row,repeats,q,label):
    for field in ('event_ms','host_submission_and_sync_ms','per_solve_ms','host_ms_per_solve'):
        check(isinstance(row[field],(int,float)) and math.isfinite(row[field]) and row[field]>0,label+' finite '+field)
    check(row['repeats']==repeats,label+' frozen repeats')
    check(row['completed_solves_delta']==repeats and row['event_records_delta']==2,label+' solve/event deltas')
    check(row['logical_queries_per_solve']==q and row['logical_queries_in_block']==repeats*q,label+' query deltas')
    check(row['trace']==0 and row['internal_repeats_are_independent_inputs'] is False,label+' trace/independence')
    check(close(row['per_solve_ms'],row['event_ms']/repeats),label+' event normalization')
    check(close(row['host_ms_per_solve'],row['host_submission_and_sync_ms']/repeats),label+' wall normalization')

def output_checks(directory,expected):
    rows=read(directory/'output_checks.json')
    check(len(rows)==len(expected),str(directory)+' output check count')
    check(Counter((r['mode'],r['purpose']) for r in rows)==Counter(expected),str(directory)+' output check purposes')
    check(all(r['status']=='PASS' and r['outside_timing'] is True and r['trace']==0 for r in rows),str(directory)+' output PASS outside timing')
    COUNTS['resident_output_check_records']+=len(rows)

def session_info(directory,case,rows):
    data=read(directory/'preparation.json')
    before=data['info_before_graph_events'];after=data['info_after_graph_events'];final=data['final_info']
    check(all(len(v)==40 for v in (before,after,final)),str(directory)+' info ABI size')
    for v in (before,after,final):
        check(v[0]==1 and v[1]==0 and v[38]==1,str(directory)+' production timing ABI')
        check(v[2]==case['N'] and v[3]==case['M'] and v[5]==case['K'] and v[7]==case['Q'],str(directory)+' N/M/K/Q')
        check(v[10]==0 and v[11]==0,str(directory)+' no debug allocations')
    check(before[31:36]==[0,0,0,0,0],str(directory)+' fresh counters')
    check(after[31:36]==[1,0,0,0,1],str(directory)+' graph/events prepared counters')
    check(final[31]==1 and final[32]==2*len(rows) and final[33]==50 and
          final[34]==50+sum(r['repeats'] for r in rows) and final[35]==1,str(directory)+' final event/solve/graph counters')
    check(all(final[i]==after[i] for i in (2,3,4,5,6,7,8,9,12,16,18,21,22,23,24,25,26,27,28,29,30,39)),str(directory)+' immutable geometry/field/graph identity')
    check(set(data['first_solve_host_ms'])==set(MODES),str(directory)+' first solve all modes')
    check(data['graph_cost_applies_to']=='R1' and data['event_cost_is_measurement_instrumentation'] is True,str(directory)+' setup accounting')
    COUNTS['resident_sessions']+=1

def audit_calibration(matrix,selection):
    expected_real={case_key(r):r for r in matrix['cases'] if r['role']=='calibration'}
    check(len(expected_real)==4,'Four real calibration cases')
    real_dirs={p.name for p in (SUITE/'worker_0/calibration').iterdir() if p.is_dir()}
    check(real_dirs=={k.replace(':','_') for k in expected_real},'Calibration directories exclude held-out inputs')
    check(set(selection['fields'])=={field_key(r) for r in expected_real.values()},'Selection field/K keys')
    expected_plane={f'K{k}:diagnostic_kind0_K{k}':{'id':f'diagnostic_kind0_K{k}','K':k,'N':33,'M':608,'Q':33+(k+1)*608,'field':'analytic_plane'} for k in (3,6)}
    check({p.name for p in (SUITE/'worker_0/calibration_plane').iterdir() if p.is_dir()}=={k.replace(':','_') for k in expected_plane},'Plane calibration directories')
    for analytic,cases,choices,dirname in ((False,expected_real,selection['fields'],'calibration'),(True,expected_plane,selection['analytic_plane'],'calibration_plane')):
        for name,case in cases.items():
            choice=choices[field_key(case)]
            check(choice['calibration_case']==name and choice['repeat_cap']==4096 and choice['target_ms']==10,'Calibration identity/bounds '+name)
            repeat=choice['repeats'];check(repeat>0 and repeat<=4096 and repeat&(repeat-1)==0,'Power-of-two repeats '+name)
            directory=SUITE/'worker_0'/dirname/name.replace(':','_');rows=jsonl(directory/'raw.jsonl')
            phase='plane_repeat_resolution' if analytic else 'repeat_resolution'
            resolution=[r for r in rows if r['phase']==phase]
            levels=[2**i for i in range(repeat.bit_length())]
            check(Counter((r['repeats'],r['mode']) for r in resolution)==Counter((n,m) for n in levels for m in MODES),'Complete bounded resolution grid '+name)
            for n in levels:
                values=[r['event_ms'] for r in resolution if r['repeats']==n]
                if n<repeat:check(min(values)<10,'Calibration continued after resolution target '+name)
                else:
                    reached=min(values)>=10
                    check(choice['resolution_target_reached']==reached and (reached or n==4096),'Calibration termination '+name)
            expected_checks=[(m,p) for m in MODES for p in ('first','repeat','warmup_final')]
            expected_checks += [(r['mode'],('plane_repeat' if analytic else 'calibration_repeat')+str(r['repeats'])) for r in resolution]
            if analytic:
                check(len(rows)==len(resolution),'No plane config selection '+name)
                check(choice['resident']=='R1' and choice['local']=='L1_8' and choice['has_held_out'] is False and choice['not_real_field_coverage'] is True,'Plane scope '+name)
            else:
                selected=[r for r in rows if r['phase']=='configuration_selection']
                check(len(rows)==len(resolution)+15 and Counter((r['trial'],r['mode']) for r in selected)==Counter((t,m) for t in range(3) for m in MODES),'Selection samples '+name)
                check(all(r['repeats']==repeat for r in selected),'Selection uses frozen repeats '+name)
                scores={m:statistics.median([r['per_solve_ms'] for r in selected if r['mode']==m]) for m in MODES}
                check(all(close(scores[m],choice['selection_median_ms'][m]) for m in MODES),'Recomputed selection medians '+name)
                resident=min(('R0','R1'),key=lambda m:(scores[m],0 if m=='R1' else 1))
                local=min(('L0','L1_8','L1_32'),key=lambda m:(scores[m],{'L0':1,'L1_8':8,'L1_32':32}[m]))
                check(choice['resident']==resident and choice['local']==local and choice['selection_samples_per_mode']==3,'Selection winner from calibration only '+name)
                expected_checks += [(r['mode'],'selection_'+str(r['trial'])) for r in selected]
            for r in rows:audit_event(r,r['repeats'],case['Q'],name+' calibration')
            output_checks(directory,expected_checks);session_info(directory,case,rows)
            COUNTS['calibration_measurement_blocks']+=len(rows)


def expected_matrix(matrix,selection):
    rows=[]
    for k in (3,6):
        for field in ('LandTiles','SdfTrees'):
            held=[r for r in matrix['cases'] if r['K']==k and r['field']==field and r['role']=='held_out']
            sizes=sorted({r['N'] for r in held});wanted={sizes[0],sizes[len(sizes)//2],sizes[-1]}
            rows += [r for r in held if r['N'] in wanted]
    expected={}
    for r in rows:
        for a,b,primary in pair_options(selection['fields'][field_key(r)]):
            expected[('resident',r['id'],r['K'],a+'_vs_'+b)]={'case':r,'a':a,'b':b,'primary':primary,'choice':selection['fields'][field_key(r)],'role':'held_out'}
    for k in (3,6):
        plane={'id':f'diagnostic_kind0_K{k}','K':k,'field':'analytic_plane','N':33,'M':608,'Q':33+(k+1)*608}
        choice=selection['analytic_plane'][field_key(plane)]
        for a,b,primary in pair_options(choice):
            expected[('analytic_plane_diagnostic',plane['id'],k,a+'_vs_'+b)]={'case':plane,'a':a,'b':b,'primary':primary,'choice':choice,'role':'diagnostic_no_held_out'}
        common={'id':'whole_checkpoint','K':k,'field':'LandTiles+SdfTrees','N':192,'M':1053,'Q':192+(k+1)*1053}
        for b in MODES:
            expected[('common_checkpoint_to_output','whole_checkpoint',k,'C0_vs_'+b)]={'case':common,'a':'C0','b':b,'primary':False,'choice':{'repeats':1},'role':'complete_original_group'}
    return rows,expected


def recompute_group(rows,label):
    pairs=defaultdict(dict)
    for r in rows:
        check(r['arm'] not in pairs[r['pair']],label+' no duplicate pair arm')
        pairs[r['pair']][r['arm']]=r
    check(set(pairs)==set(range(20)) and all(set(p)=={'A','B'} for p in pairs.values()),label+' complete 20 pairs')
    ordered=[pairs[i] for i in range(20)]
    a=[p['A']['per_solve_ms'] for p in ordered];b=[p['B']['per_solve_ms'] for p in ordered];ratios=[x/y for x,y in zip(a,b)]
    return {'pairs':20,'median_A_ms':statistics.median(a),'p90_A_ms':quantile(a,.9),'median_B_ms':statistics.median(b),
        'p90_B_ms':quantile(b,.9),'paired_ratios_A_over_B':ratios,'median_paired_ratio':statistics.median(ratios),'p90_paired_ratio':quantile(ratios,.9)}


def audit_common_records(worker_dir,common_rows):
    directory=worker_dir/'common'
    paths=list(directory.glob('a*.json'))
    check(len(paths)==520,str(worker_dir)+' 520 common records')
    formal={r['stage_file']:r for r in common_rows};check(len(formal)==400,str(worker_dir)+' 400 unique timed common arms')
    warmups=Counter();seen=set();contract={'xyz':{'shape':[192,3],'dtype':'float32'},'times':{'shape':[192,2],'dtype':'int32'},'tags':{'shape':[192],'dtype':'int8'},'vertex_map':{'shape':[594],'dtype':'int32'}}
    for path in paths:
        record=read(path);name=str(path.relative_to(worker_dir));COUNTS['common_output_contract_records']+=1
        check(record['status']=='EXECUTED' and record['GPU_executed'] is True and record['outputs']==contract,name+' output contract/executed')
        check(record['original_python_loop']['while_ast_unchanged'] is True,name+' original while preserved')
        if record['phase']=='warmup':
            warmups[(record['K'],record['mode'])]+=1
            check(record['performance_test'] is False and record['stages']==[],name+' warmup timing off')
        else:
            seen.add(name);check(name in formal and record['phase']=='formal' and record['performance_test'] is True,name+' formal arm identity')
            row=formal[name]
            check(record['mode']==row['mode'] and record['K']==row['K'] and record['label']==f"{row['comparison']}:{row['pair']}:{row['arm']}",name+' raw/record identity')
            roots=[r for r in record['stages'] if r['parent'] is None]
            check(len(roots)==1 and roots[0]['stage']=='checkpoint_to_output',name+' single timed root')
            check(close(roots[0]['wall_ns']/1e6,row['per_solve_ms']),name+' root time matches raw row')
            children=[r for r in record['stages'] if r['parent']=='checkpoint_to_output']
            check(all(r['inclusive'] is True and r['wall_ns']>=0 for r in record['stages']) and sum(r['wall_ns'] for r in children)<=roots[0]['wall_ns'],name+' child sum within root')
            check(all(r['parent']=='checkpoint_to_output' for r in record['stages'] if r['parent'] is not None),name+' direct-child cost partition')
    check(seen==set(formal),str(worker_dir)+' every timed raw row has record')
    check(warmups==Counter({(k,m):10 for k in (3,6) for m in ('C0',)+MODES}),str(worker_dir)+' 120 fixed warmups')
    check(not any((worker_dir/'private_checkpoints').iterdir()),str(worker_dir)+' private checkpoints released after success')


def main():
    global SUITE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',type=Path,default=SUITE)
    parser.add_argument('--out',type=Path,default=ROOT/'results/G1_AUDIT.json')
    parser.add_argument('--csv-out',type=Path,default=ROOT/'results/G1_TIMINGS.csv')
    args=parser.parse_args()
    SUITE=args.suite.resolve()
    destinations=[args.out.resolve(),args.csv_out.resolve()]
    if not SUITE.is_relative_to(ROOT) or any(not p.is_relative_to(ROOT) for p in destinations):
        parser.error('suite and audit output paths must remain in BM_GPU_WORKSPACE')
    if destinations[0]==destinations[1]:parser.error('audit JSON and CSV must use distinct paths')
    suite_status=read(SUITE/'STATUS.json')
    worker_paths=[SUITE/f'worker_{i}/STATUS.json' for i in range(3)]
    if suite_status.get('status')!='COMPLETED' or not all(p.is_file() and read(p).get('status')=='COMPLETED' for p in worker_paths):
        print(json.dumps({'status':'NOT_READY_NO_AUDIT_RUN','reason':'Require completed suite and all three completed workers; no output files written'}))
        return 2
    if any(p.exists() for p in destinations):raise FileExistsError('Preserve existing audit results')
    for path in destinations:path.parent.mkdir(parents=True,exist_ok=True)
    report={'status':'RUNNING','scope':'Independent standard-library analysis only; no new GPU/application execution','issues':ISSUES}
    try:
        matrix=read(ROOT/'configs/FROZEN_MATRIX.json');selection=read(SUITE/'SELECTION.json');plan=read(SUITE/'PLAN.json');final=read(SUITE/'SUMMARY.json')
        matrix_hash=sha(ROOT/'configs/FROZEN_MATRIX.json');selection_hash=sha(SUITE/'SELECTION.json')
        check(matrix['processes']==3 and matrix['warmups']==10 and matrix['pairs']==20 and matrix['K']==[3,6] and matrix['pair_orders']=={'AB':10,'BA':10},'Frozen protocol')
        check(selection['worker']==0 and selection['held_out_used'] is False and selection['status']=='FROZEN_AFTER_WORKER0_CALIBRATION','Selection owner/no held-out')
        check(selection['matrix_sha256']==matrix_hash==plan['matrix_sha256'],'Matrix fingerprints')
        check(suite_status['completed_workers']==[0,1,2] and suite_status['automatic_retries']==0,'Three sequential workers/no retries')
        check(final['status']=='COMPLETED' and final['selection_sha256']==selection_hash and final['all_slow_and_negative_trials_retained'] is True,'Final summary identity')
        check(sha(ROOT/'configs/PERFORMANCE_AUTHORIZATION.json')==suite_status['authorization_sha256'],'Authorization hash')
        for item in plan['fingerprints']:check(item['sha256']==sha(ROOT/item['path']),'Unchanged prepared artifact '+item['path'])
        for item in plan['checkpoint_files']:check(item['sha256']==sha(ROOT/'inputs/checkpoint_seed'/item['path']),'Unchanged checkpoint '+item['path'])
        audit_calibration(matrix,selection)
        held,expected=expected_matrix(matrix,selection)
        check(set(plan['resident_formal_cases'])=={case_key(r) for r in held},'Prepared held-out matrix')
        check(not ({case_key(r) for r in held}&set(plan['calibration_cases'])),'Calibration/formal case separation')
        computed=defaultdict(list)
        for worker,path in enumerate(worker_paths):
            worker_dir=path.parent;ws=read(path)
            check(ws['worker']==worker and ws['selection_sha256']==selection_hash and ws['automatic_retries']==0,'Worker identity/selection '+str(worker))
            if worker:check(not (worker_dir/'calibration').exists() and not (worker_dir/'calibration_plane').exists(),'No recalibration by worker '+str(worker))
            raw=jsonl(worker_dir/'trials.jsonl');COUNTS['formal_timed_arms']+=len(raw)
            check(ws['raw_timed_arms']==len(raw)==40*len(expected),'Raw timed arm count worker '+str(worker))
            groups=defaultdict(list)
            for row in raw:groups[key(row)].append(row)
            check(set(groups)==set(expected),'Exact case/comparison matrix worker '+str(worker))
            ws_rows={key(r):r for r in ws['summary']};check(len(ws_rows)==len(ws['summary']) and set(ws_rows)==set(expected),'Worker summary keys')
            common_rows=[];session_rows=defaultdict(list);session_expect=defaultdict(list)
            for group,rows in groups.items():
                table,case,k,comparison=group;spec=expected[group];label=f'{worker}/{group}'
                scope='whole_checkpoint_K'+str(k) if table=='common_checkpoint_to_output' else f'K{k}:{case}'
                order=expected_orders(matrix,worker,scope,comparison)
                check(len(rows)==40,label+' arm count')
                check([(r['pair'],r['arm']) for r in rows]==[(p,arm) for p,o in enumerate(order) for arm in o],label+' exact randomized order')
                check(Counter(r['order'] for r in rows if r['arm']=='A')==Counter({'AB':10,'BA':10}),label+' AB BA balance')
                for row in rows:
                    check(row['worker']==worker and row['mode_A']==spec['a'] and row['mode_B']==spec['b'] and row['mode']==(spec['a'] if row['arm']=='A' else spec['b']),label+' mode mapping')
                    check(row['primary']==spec['primary'] and row['role']==spec['role'] and row['field']==spec['case']['field'],label+' role and primary')
                    check(row['order']==order[row['pair']],label+' frozen order')
                    if table=='common_checkpoint_to_output':
                        common_rows.append(row)
                        check(row['full_output_check']=='PASS_OUTSIDE_TIMING' and row['graph_reuse']==1 and row['repeats']==1 and row['checkpoint_provisioning_timed'] is False and row['checkpoint_read_prepare_solve_readback_cleanup_timed'] is True,label+' common output and cost scope')
                        check(math.isfinite(row['per_solve_ms']) and row['per_solve_ms']>0,label+' positive common wall')
                    else:
                        audit_event(row,spec['choice']['repeats'],spec['case']['Q'],label)
                        session_rows[(table,case,k)].append(row)
                        if row['pair'] in (0,1,19):session_expect[(table,case,k)].append((row['mode'],comparison+f"_pair{row['pair']:02d}"))
                recomputed=recompute_group(rows,label)
                for field,value in recomputed.items():check(same_value(value,ws_rows[group][field]),label+' worker summary '+field)
                computed[group].append({'worker':worker,**recomputed})
                CSV_ROWS.append({'level':'process','worker':worker,'table':table,'case':case,'K':k,'field':spec['case']['field'],'N':spec['case']['N'],'M':spec['case']['M'],'comparison':comparison,'mode_A':spec['a'],'mode_B':spec['b'],'primary':spec['primary'],'pairs':20,'repeats_per_arm':spec['choice']['repeats'],
                    **{f:v for f,v in recomputed.items() if f!='paired_ratios_A_over_B'},'ratio_basis':'paired A/B; >1 means B faster','timing_scope':'checkpoint_to_output_host_wall' if table=='common_checkpoint_to_output' else 'resident_reset_solve_stream_event_span'})
            for session,rows in session_rows.items():
                table,case,k=session;spec=next(v for g,v in expected.items() if g[:3]==session)
                directory=worker_dir/table/f'K{k}_{case}'
                expected_checks=[(m,p) for m in MODES for p in ('first','repeat','warmup_final')]+session_expect[session]
                output_checks(directory,expected_checks);session_info(directory,spec['case'],rows)
            audit_common_records(worker_dir,common_rows)
        final_rows={key(r):r for r in final['rows']}
        check(len(final_rows)==len(final['rows']) and set(final_rows)==set(expected),'Aggregate summary exact matrix')
        for group,rows in computed.items():
            aggregate=final_rows[group];spec=expected[group];ratios=[r['median_paired_ratio'] for r in rows]
            check(close(aggregate['median_of_process_paired_medians'],statistics.median(ratios)) and same_value(aggregate['process_ratio_range'],[min(ratios),max(ratios)]),'Aggregate paired median and range '+str(group))
            embedded={r['worker']:r for r in aggregate['process_summaries']}
            check(set(embedded)=={0,1,2},'Aggregate embedded workers '+str(group))
            for r in rows:
                for f,v in r.items():check(same_value(v,embedded[r['worker']][f]),'Aggregate embedded value '+str(group)+' '+f)
            table,case,k,comparison=group
            CSV_ROWS.append({'level':'aggregate_descriptive_three_processes','worker':'all','table':table,'case':case,'K':k,'field':spec['case']['field'],'N':spec['case']['N'],'M':spec['case']['M'],'comparison':comparison,'mode_A':spec['a'],'mode_B':spec['b'],'primary':spec['primary'],'pairs':60,'repeats_per_arm':spec['choice']['repeats'],
                'median_of_process_paired_medians':statistics.median(ratios),'min_process_paired_median':min(ratios),'max_process_paired_median':max(ratios),
                'ratio_basis':'median of three process paired medians; >1 means B faster','timing_scope':'checkpoint_to_output_host_wall' if table=='common_checkpoint_to_output' else 'resident_reset_solve_stream_event_span'})
        report.update(selection_sha256=selection_hash,matrix_sha256=matrix_hash,processes=3,comparisons_per_process=len(expected),held_out_real_cases=len(held),analytic_plane_cases=2,counts=dict(COUNTS),raw_trials_sha256=RAW_HASHES,
            status='PASS_INDEPENDENT_G1_AUDIT' if not ISSUES else 'FAILED_INDEPENDENT_G1_AUDIT',
            limitations=['Every common timed arm has recorded PASS_OUTSIDE_TIMING; this offline audit checks the logs and output shapes, not unsaved array bytes.',
                'Resident correctness first/repeat/final checks are validated from saved PASS records, not by executing an oracle again.',
                'CSV process rows contain all timings and comparisons; aggregate rows contain descriptive ratio summaries, not pooled independent trials.',
                'The analytic plane is diagnostic only; no real-field claim or held-out configuration selection.',
                'CUDA event span includes stream idle gaps and reset; event/wall subtraction is not pure coordination cost.'])
    except BaseException as error:
        ISSUES.append('Audit exception: '+repr(error));report.update(status='FAILED_INDEPENDENT_G1_AUDIT',traceback=traceback.format_exc(),counts=dict(COUNTS))
    report['issues']=ISSUES
    destinations[0].write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    fields=['level','worker','table','case','K','field','N','M','comparison','mode_A','mode_B','primary','pairs','repeats_per_arm','median_A_ms','p90_A_ms','median_B_ms','p90_B_ms','median_paired_ratio','p90_paired_ratio','median_of_process_paired_medians','min_process_paired_median','max_process_paired_median','ratio_basis','timing_scope']
    with destinations[1].open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(CSV_ROWS)
    print(json.dumps({'status':report['status'],'issues':ISSUES,'counts':dict(COUNTS),'csv_rows':len(CSV_ROWS),'outputs':[str(p) for p in destinations]},indent=2))
    return 0 if not ISSUES else 1

if __name__=='__main__':raise SystemExit(main())