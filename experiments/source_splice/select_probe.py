#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from fractions import Fraction
from pathlib import Path


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    selected=json.loads((args.cache_root/'event_registry_selected_event.json').read_text())
    if not selected.get('selected'):
        raise RuntimeError('no selected production event')
    event_id=selected['event_id']
    rows=[]
    with (args.cache_root/'event_registry_p1.csv').open(newline='',encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            if row['canonical_event_id']==event_id:
                rows.append(row)
    if not rows:
        raise RuntimeError('selected event has no registry rows')
    roots={Fraction(int(row['root_num']),int(row['root_den'])) for row in rows}
    if len(roots)!=1:
        raise RuntimeError('selected event rows disagree on exact root')
    root=next(iter(roots))
    corner_times={int(row[f't{i}']) for row in rows for i in range(4)}
    lower=max(Fraction(t,1) for t in corner_times if Fraction(t,1)<root)
    upper=min(Fraction(t,1) for t in corner_times if Fraction(t,1)>root)
    probe=(lower+root)/2
    payload={
        'event_id':event_id,
        'root':{'numerator':root.numerator,'denominator':root.denominator},
        'lower_corner':{'numerator':lower.numerator,'denominator':lower.denominator},
        'upper_corner':{'numerator':upper.numerator,'denominator':upper.denominator},
        'probe':{'numerator':probe.numerator,'denominator':probe.denominator},
        'registry_rows':len(rows),
    }
    args.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    print(f'{probe.numerator} {probe.denominator}')
    return 0
if __name__=='__main__': raise SystemExit(main())
