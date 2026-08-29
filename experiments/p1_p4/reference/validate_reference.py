#!/usr/bin/env python3
import csv, json, hashlib, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
R=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'results/reference'
s=json.loads((R/'summary.json').read_text())
checks={}
checks['review_pass']=s['review']['verdict']=='PASS_PREVIOUS_ROUND_CODE_REVIEW'
checks['p1_pass']=s['p1']['verdict']=='PASS_P1_READ_ONLY_REGISTRY' and s['p1']['byte_identical'] and s['p1']['order_invariant']
checks['p2_pass']=s['p2']['verdict']=='PASS_P2_EXACT_SADDLE_ONLY' and s['p2']['exact_accuracy']==1.0 and s['p2']['dihedral_mismatches']==0 and s['p2']['eventfree_mismatches']==0
checks['p3_pass']=s['p3']['verdict']=='PASS_P3_SINGLE_ENDPOINT_BATCH' and s['p3']['failures']==0 and s['p3']['proposed_gap_max']==0 and s['p3']['relative_boundary_passes']==s['p3']['cases'] and s['p3']['slice_passes']==s['p3']['cases']
checks['p4_pass']=s['p4']['verdict']=='PASS_P4_LOW_COST_MIXED_SEQUENCE' and s['p4']['registry_invariance_failures']==0 and s['p4']['event_aligned_probe_failures']==0 and s['p4']['eventfree_hash_failures']==0 and s['p4']['simultaneous_batches']>=1
# raw row counts
with (R/'p2_saddles.csv').open() as f: p2rows=sum(1 for _ in f)-1
with (R/'p3_endpoint_batches.csv').open() as f: p3rows=sum(1 for _ in f)-1
with (R/'p4_sequence.csv').open() as f: p4rows=sum(1 for _ in f)-1
checks['raw_counts']=p2rows==s['p2']['event_sides'] and p3rows==s['p3']['cases'] and p4rows==4
checks['scope_qualified']=len(s.get('limitations',[]))>=4 and 'core.so' in ' '.join(s['limitations'])
verdict='PASS_INDEPENDENT_P1_P4_VALIDATION' if all(checks.values()) else 'STOP_INDEPENDENT_P1_P4_VALIDATION'
out={'verdict':verdict,'checks':checks,'raw_counts':{'p2':p2rows,'p3':p3rows,'p4':p4rows}}
(R/'independent_validation.json').write_text(json.dumps(out,indent=2))
print(verdict)
if verdict.startswith('STOP'): sys.exit(2)
