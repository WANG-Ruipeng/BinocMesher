"""Independent exact-root source-quotient link witnesses; no native state.

Reads the complete V-halo for the 26 frozen SOURCE_INTERFACE_REJECT events.
Keeps oriented face owner classes so a JSON-only auditor can rebuild every
link graph. This is a source-model contract, not actual C++ identity proof.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import resource
import time
import traceback

from forest_support_reader import read_event_candidates


def require(value, reason):
    if not value:raise ValueError(reason)


def stable(value):
    return json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk:=stream.read(1024*1024):h.update(chunk)
    return h.hexdigest()


def fraction(value):
    return F(value['numerator'],value['denominator']) if isinstance(value,dict) else F(value)


def edge_list(face):
    return [(face[i],face[(i+1)%len(face)]) for i in range(len(face))]


def oriented_key(element,face):
    face=tuple(face)
    return element,min(face[i:]+face[:i] for i in range(len(face)))


def graph_witness(vertex,previous,following,edges):
    """Recompute the necessary directed connected path conditions explicitly."""
    edges=[tuple(e) for e in edges]
    degree=Counter(x for e in edges for x in e)
    incoming=Counter(b for a,b in edges);outgoing=Counter(a for a,b in edges)
    adjacency={v:set() for v in degree}
    for a,b in edges:adjacency[a].add(b);adjacency[b].add(a)
    remaining=set(degree);components=[]
    while remaining:
        seen=set();pending=[min(remaining)]
        while pending:
            v=pending.pop()
            if v in seen:continue
            seen.add(v);pending.extend(adjacency[v]-seen)
        remaining-=seen;components.append(sorted(seen))
    endpoints={previous,following}
    degree_ok=({v for v,d in degree.items() if d==1}==endpoints and
               all(d==(1 if v in endpoints else 2) for v,d in degree.items()))
    oriented=all(incoming[v]==(0 if v==previous else 1) and
                 outgoing[v]==(0 if v==following else 1) for v in degree)
    connected=len(components)==1
    valid=degree_ok and oriented and connected
    return {'source_vid':vertex,'status':'PASS' if valid else 'REJECT',
        'link_edges':len(edges),'link_vertices':len(degree),
        'directed_link_path_compatible':oriented,'expected_path_endpoints':sorted(endpoints),
        'directed_edges':[list(x) for x in edges],'components':components,
        'degree':dict(sorted(degree.items())),
        'incoming':{v:incoming[v] for v in sorted(degree)},
        'outgoing':{v:outgoing[v] for v in sorted(degree)},
        'degree_and_endpoints_compatible':degree_ok,'connected':connected,
        'failure_conditions':[name for name,ok in
            [('DEGREE_OR_ENDPOINTS',degree_ok),('DIRECTED_PATH',oriented),('DISCONNECTED',connected)] if not ok]}


def build_event(snapshot,event,event_binding):
    eid=event['event_id'];root=F(event['root']);element=event['element']
    source=event['compiler']['source']
    require(snapshot.event_id==eid and snapshot.tau==root,'SNAPSHOT_EVENT_TIME_MISMATCH')
    require(snapshot.event_candidates_complete and snapshot.halo_complete,'INCOMPLETE_HALO')
    supports=[p for p in source['breakpoint_points'] if fraction(p['time'])==root]
    require(len(supports)==1,'ROOT_SINGLETON_NOT_UNIQUE')
    support=supports[0];cycle=tuple(support['boundary_cycle']);boundary=set(cycle)
    require(boundary<={v.text() for v in snapshot.halo_vertex_ids},'SOURCE_BOUNDARY_OUTSIDE_COMPLETE_CANDIDATE_HALO')
    source_faces=[tuple(f) for f in support['source_faces']]
    require(len(source_faces)==2 and len(cycle)==4 and len(boundary)==4,'MALFORMED_SOURCE_DISK')
    require(all(len(f)==3 and len(set(f))==3 and set(f)<=boundary for f in source_faces),'INVALID_SOURCE_FACE')
    directions=Counter(e for face in source_faces for e in edge_list(face))
    source_boundary=Counter(edge_list(cycle));inner=directions-source_boundary
    require(all(directions[e]==1 for e in source_boundary) and len(inner)==2 and
            sum(inner.values())==2 and all(inner[b,a]==n==1 for (a,b),n in inner.items()),'SOURCE_DISK_BOUNDARY_OR_DIAGONAL_INVALID')
    diagonal=set(next(iter(inner)))
    groups=defaultdict(list);emitted={}
    for tri in snapshot.actual_raw_triangles:
        owner=tuple(tri.reference.values());face=tuple(v.text() for v in tri.source_vertices)
        require(owner not in emitted,'DUPLICATE_RAW_OWNER')
        key=oriented_key(tri.reference.element,face)
        emitted[owner]=key;groups[key].append(owner)
    requested={tuple(o) for o in support['owners']}
    selected={oriented_key(element,face) for face in source_faces}
    require(requested<=set(emitted),'SOURCE_OWNER_NOT_EMITTED')
    require({emitted[o] for o in requested}==selected,'SOURCE_OWNER_FACE_DISAGREEMENT')
    require({o for key in selected for o in groups[key]}==requested,'PARTIAL_OWNER_CLASS_SUPPRESSION')
    faces=[{'element':e,'source_vertices':list(face),'owners':[list(o) for o in sorted(owners)],
            'suppressed':(e,face) in selected}
           for (e,face),owners in sorted(groups.items())]
    links={v:[] for v in cycle};retained_directions=Counter();excluded=[]
    for index,row in enumerate(faces):
        face=tuple(row['source_vertices'])
        if row['suppressed'] or row['element']!=element or not boundary&set(face):continue
        if len(set(face))!=3:
            excluded.append({'face_index':index,'reason':'RETAINED_INTERFACE_REPEATED_ID'});continue
        if diagonal<=set(face):
            excluded.append({'face_index':index,'reason':'RETAINED_USES_ORIGINAL_INTERNAL_DIAGONAL'});continue
        retained_directions.update(edge_list(face))
        for v in boundary&set(face):
            j=face.index(v);links[v].append((face[(j+1)%3],face[(j+2)%3]))
    root_rows=[q for q in event['requested_source_interface']['queries'] if q['query']['kind']=='exact_root']
    require(len(root_rows)==1 and F(root_rows[0]['query']['evaluation_tau'])==root,'FROZEN_ROOT_QUERY_BINDING')
    original=root_rows[0];require(original['status']=='REJECT','FROZEN_ROOT_NOT_REJECTED')
    old_links={p['source_vid']:p for p in original['interface']['vertex_link_checks']}
    graphs=[graph_witness(v,cycle[i-1],cycle[(i+1)%4],links[v]) for i,v in enumerate(cycle)]
    require(all({k:row[k] for k in old_links[row['source_vid']]}==old_links[row['source_vid']] for row in graphs),
            'INDEPENDENT_GRAPH_SUMMARY_DIFFERS_FROM_FROZEN_ROOT')
    edge_checks=[]
    for a,b in edge_list(cycle):
        same,opposite=retained_directions[a,b],retained_directions[b,a]
        edge_checks.append({'edge':[a,b],'same_direction':same,'opposite_direction':opposite,
                            'status':'PASS' if same==0 and opposite==1 else 'REJECT'})
    require(edge_checks==original['interface']['boundary_edge_checks'],'BOUNDARY_ATTACHMENT_SUMMARY_DIFFERS')
    require(any(row['status']=='REJECT' for row in graphs),'NO_INDEPENDENT_LINK_REJECTION')
    return {'event_id':eid,'element':element,'root':str(root),'frozen_event':event_binding,
        'source_contract_sha256':hashlib.sha256(stable(source)).hexdigest(),
        'source_support':{k:support[k] for k in ('time','boundary_cycle','source_faces','owners')},
        'frozen_root_query':original['query'],'snapshot':snapshot.summary(),
        'all_oriented_quotient_faces_in_complete_halo':faces,
        'suppressed_oriented_face_keys':[[e,list(f)] for e,f in sorted(selected)],
        'internal_diagonal':sorted(diagonal),'excluded_from_link_graph':excluded,
        'boundary_edge_checks':edge_checks,'vertex_links':graphs,
        'frozen_root_interface_sha256':hashlib.sha256(stable(original['interface'])).hexdigest(),
        'independent_summary_equal_frozen':True,'status':'CONFIRMED_FIXED_SOURCE_LINK_REJECTION'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--attempt',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();start=time.monotonic();cpu=time.process_time()
    require(not args.output.exists(),'OUTPUT_EXISTS')
    require(args.cache.resolve() not in args.output.resolve().parents,'OUTPUT_INSIDE_ORIGINAL_CACHE')
    input_paths=[args.attempt/x for x in ('summary.json','events_index.json','input_verification.json')]
    old_verification=json.loads(input_paths[2].read_text())
    require(old_verification['pass'],'FROZEN_INPUT_VERIFICATION_NOT_PASS')
    index=json.loads(input_paths[1].read_text())
    rows=[r for r in index if r['reason']=='SOURCE_INTERFACE_REJECT']
    require(len(rows)==26,'EXPECTED_FIXED_26_SOURCE_INTERFACE_REJECTIONS')
    events={};event_bindings={}
    for row in rows:
        path=args.attempt/row['artifact']['path'];require(file_sha(path)==row['artifact']['sha256'],'FROZEN_EVENT_HASH_MISMATCH')
        value=json.loads(path.read_text());require(value['event_id']==row['event_id'],'FROZEN_EVENT_ID_MISMATCH')
        events[row['event_id']]=value;event_bindings[row['event_id']]=row['artifact'];input_paths.append(path)
    bindings={str(p):file_sha(p) for p in input_paths}
    sources={str(Path(__file__).resolve()):file_sha(__file__),**old_verification['executed_source_sha256']}
    require(all(file_sha(p)==h for p,h in sources.items()),'FROZEN_READER_SOURCE_CHANGED')
    result={'schema':'forest-independent-source-link-witness-v1','status':'STOP_NOT_A_METHOD_DECISION',
        'scope':'26 fixed source-quotient exact-root link rejections; not actual C++ identity or all-time admission.',
        'frozen_artifact_sha256':bindings,'executed_sources_sha256':sources,'events':[]}
    try:
        dataset=read_event_candidates(args.cache,events,seconds=300)
        print(json.dumps({'stage':'26_candidates_read','seconds':time.monotonic()-start}),flush=True)
        snapshots=dataset.load_halos({eid:[F(e['root'])] for eid,e in events.items()})
        for eid,event in events.items():
            result['events'].append(build_event(snapshots[eid,F(event['root'])],event,event_bindings[eid]))
        verification=dataset.verify_inputs()
        require(verification['input_full_sha256']==old_verification['input_full_sha256'],'COMPLETE_INPUT_HASH_SET_DIFFERS_FROM_FROZEN_ATTEMPT')
        require(all(file_sha(p)==h for p,h in bindings.items()),'FROZEN_ARTIFACT_CHANGED_DURING_READ')
        require(all(file_sha(p)==h for p,h in sources.items()),'EXECUTED_SOURCE_CHANGED_DURING_READ')
        failures=[g for e in result['events'] for g in e['vertex_links'] if g['status']=='REJECT']
        result.update(status='PASS_INDEPENDENT_SOURCE_LINK_REPLAY',event_count=len(events),
            failed_link_count=len(failures),equal_e_v_minus_1_failed_links=sum(g['link_edges']==g['link_vertices']-1 for g in failures),
            failure_condition_counts=dict(Counter(k for g in failures for k in g['failure_conditions'])),
            input_verification=verification,full_input_hash_set_equal_frozen=True,reader=dataset.summary())
    except Exception as error:
        result.update(reason=type(error).__name__+': '+str(error),traceback=traceback.format_exc())
    result['cost']={'wall_seconds':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu,
                    'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    data=json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n'
    require(len(data.encode())<=2*1024*1024,'WITNESS_REPORT_EXCEEDS_2_MIB')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:stream.write(data)
    print(json.dumps({k:result[k] for k in ('status','cost')}|{'bytes':len(data.encode()),'output':str(args.output)}),flush=True)
    return 0 if result['status']=='PASS_INDEPENDENT_SOURCE_LINK_REPLAY' else 2


if __name__=='__main__':raise SystemExit(main())
