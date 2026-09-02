#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json
from fractions import Fraction
from pathlib import Path


def load(path: Path):
    if not path.is_file(): raise FileNotFoundError(path)
    return json.loads(path.read_text())


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--results',type=Path,required=True); ap.add_argument('--cache-root',type=Path,required=True); ap.add_argument('--repeat',type=Path); ap.add_argument('--event-id'); args=ap.parse_args()
    r=args.results.resolve(); cache=args.cache_root.resolve()
    summary=load(r/'summary.json'); tv0=load(r/'tv0/tv0_summary.json'); tv1=load(r/'tv1/tv1_summary.json'); tv2=load(r/'tv2/tv2_summary.json'); tv3=load(r/'tv3/tv3_summary.json'); tv4=load(r/'tv4/tv4_summary.json')
    selected=(
        load(cache/'event_registry_selected_event.json')
        if args.event_id is None else
        {'event_id':args.event_id}
    )
    with (cache/'event_registry_p1.csv').open(newline='') as f: rows=list(csv.DictReader(f))
    group=[x for x in rows if x['canonical_event_id']==selected['event_id']]
    if not group: raise RuntimeError('requested event has no registry rows')
    times=tuple(int(group[0][f't{i}']) for i in range(4)); A=times[0]+times[2]-times[1]-times[3]; B=times[0]*times[2]-times[1]*times[3]
    root=Fraction(B,A); u=Fraction(times[0]-times[3],A); v=Fraction(times[0]-times[1],A)
    expected_root=Fraction(
        int(group[0]['root_num']),int(group[0]['root_den']))
    expected_u=Fraction(int(group[0]['u_num']),int(group[0]['u_den']))
    expected_v=Fraction(int(group[0]['v_num']),int(group[0]['v_den']))
    checks={
      'overall_pass':summary.get('pass') is True and summary.get('verdict')=='PASS_TV0_TV4_PRODUCTION_DERIVED_THEORY_VALIDATION',
      'all_stage_verdicts_pass':all(v.startswith('PASS_') for v in summary['stages'].values()),
      'tv0_all_checks':all(tv0['checks'].values()),
      'tv0_has_both_routes':tv0['route_counts'].get('regular_monotone',0)>0 and tv0['route_counts'].get('quotient_singular',0)>0,
      'tv0_temporal_events_nonempty':tv0['registry']['temporal_canonical_events']>0,
      'tv1_all_checks':all(tv1['checks'].values()) and tv1['independent_temporal_saddles']==tv1['registry_temporal_saddles'],
      'tv1_probes_nontrivial':tv1['exact_probes']>=1000 and tv1['event_free_intervals_tested']>=100,
      'tv2_all_checks':all(tv2['checks'].values()) and tv2['cases']>=32,
      'tv3_all_checks':all(tv3['checks'].values()) and tv3['tetrahedra']==2,
      'tv3_transition_2_to_1':tv3['lower_slice']['components']==2 and tv3['upper_slice']['components']==1,
      'tv4_all_checks':all(tv4['checks'].values()) and tv4['shared_blocks']==1,
      'selected_event_math_recomputed':(
          root==expected_root and u==expected_u and v==expected_v),
      'requested_event_matches_results':(
          (
              args.event_id is None or
              summary.get('event_id')==selected['event_id']
          ) and
          tv3.get('event_id')==selected['event_id'] and
          tv4.get('event_id')==selected['event_id']),
      'selected_event_has_multiple_incident_records':len(group)>1 and len({x['logical_incidence_id'] for x in group})>1,
      'scope_does_not_claim_runtime':summary['scope']['production_runtime_modified'] is False,
    }
    repeat_result=None
    if args.repeat:
      rr=args.repeat.resolve()
      def manifest(root:Path): return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
      repeat_result={'same_file_set':set(manifest(r))==set(manifest(rr)),'byte_identical':manifest(r)==manifest(rr)}
      checks['repeat_byte_identical']=repeat_result['byte_identical']
    passed=all(checks.values())
    result={'schema':'binoc-tv0-tv4-independent-validation-v1','pass':passed,'verdict':'PASS_INDEPENDENT_TV0_TV4_VALIDATION' if passed else 'STOP_INDEPENDENT_TV0_TV4_VALIDATION','event_id':selected['event_id'],'recomputed_selected_event':{'times':times,'A':A,'B':B,'root':str(root),'u':str(u),'v':str(v),'raw_rows':len(group),'logical_incidences':len({x['logical_incidence_id'] for x in group})},'checks':checks,'repeat':repeat_result}
    (r/'independent_validation.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(result['verdict']); return 0 if passed else 2
if __name__=='__main__': raise SystemExit(main())
