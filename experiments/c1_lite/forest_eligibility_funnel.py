#!/usr/bin/env python3
"""Bounded read-only Forest131 eligibility; never a runtime admission.

Only referenced HV objects and a boundary-incidence halo are retained. The
variable-length processed streams are scanned once; unrelated polygons are
discarded immediately. Integer source predicates certify combinatorial state
only. All geometry here is ideal rational interpolation of serialized float32
HV, before Forest's missing post-mesh displacement stage.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from fractions import Fraction as F
import hashlib
import itertools
import json
import mmap
import os
from pathlib import Path
import resource
import struct
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'tv0_tv4'))
from theory_audit import (HVID, HV, Cell, SourceRecord, admissible_saddle,
    canonical_partition, build_production_half_handle, complete_production_event_star)
from interface_topology import original_disk, audit_interface
from window_geometry import certify_segment

STAGES = ('registry_saddle', 'source_provenance', 'beb1_kernel', 'candidate_window',
          'source_owner_closure', 'source_interface', 'candidate_local_geometry',
          'same_root_compatibility', 'natural_frame_hits', 'global_exterior',
          'runtime_implementation', 'runtime_admission')
HV_DTYPE = np.dtype({'names':['node','group'], 'formats':['<i4','i1'],
                     'offsets':[0,4], 'itemsize':28})

def fj(x):
    x=F(x); return {'numerator':x.numerator,'denominator':x.denominator}

def fr(x):
    return F(x['numerator'],x['denominator']) if isinstance(x,dict) else F(x)

def key(a,b):
    return tuple(sorted((a,b)))

def token(v):
    return '|'.join(f'{a}:{b}' for a,b in v)

def hp(text):
    return tuple(map(int,text.split(':')))

def canon(face):
    face=tuple(face); return min(face[i:]+face[:i] for i in range(3))

def verdict(status,reason,**evidence):
    return {'status':status,'reason_code':reason,**evidence}

def require(value,reason):
    if not value: raise ValueError(reason)

def sub(a,b): return tuple(x-y for x,y in zip(a,b))
def cross(a,b): return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])

def rank3(points):
    rows=[list(sub(p,points[0])) for p in points[1:]]
    rank=0
    for col in range(4):
        pivot=next((i for i in range(rank,3) if rows[i][col]),None)
        if pivot is None: continue
        rows[rank],rows[pivot]=rows[pivot],rows[rank]
        divisor=rows[rank][col]; rows[rank]=[x/divisor for x in rows[rank]]
        for i in range(rank+1,3):
            a=rows[i][col]; rows[i]=[x-a*y for x,y in zip(rows[i],rows[rank])]
        rank+=1
        if rank==3: break
    return rank==3

def effective(pair,t,hvs):
    """Ideal coordinates with native original-ordered-VID collapse semantics.

    Unlike window_source's ideal canonical model, native compute_slice swaps
    coordinates by time but keeps the original VID endpoints for collapse.
    Do not repair that native rule here or call this actual float32 identity.
    """
    first,second=pair; p1,p2=hvs[first],hvs[second]
    if p1['time']>p2['time']: p1,p2=p2,p1
    t1,t2=p1['time'],p2['time']; raw=max(F(t1),min(F(t2),t))
    e=raw
    if p1['view']!=p2['view'] and t1+p1['span']<=t2-p2['span']:
        if p2['view']: t2-=p2['span']
        else: t1+=p1['span']
        e=max(F(t1),min(F(t2),raw))
    if t1!=t2:
        position=tuple((a*(t2-e)+b*(e-t1))/(t2-t1) for a,b in zip(p1['position'],p2['position']))
        which=1 if e==t1 else 2 if e==t2 else 0
    else:
        which=1 if raw<t1 else 2
        position=(p1 if which==1 else p2)['position']
    out=(first,first) if which==1 else (second,second) if which==2 else pair
    return key(*out),position

def time_ordered_edge(a,b,hvs):
    require(hvs[a]['time']!=hvs[b]['time'],'SADDLE_BOUNDARY_EQUAL_ENDPOINT_TIMES')
    return (a,b) if hvs[a]['time']<hvs[b]['time'] else (b,a)

def candidate_levels(times,root):
    clearance=min(abs(F(t)-root) for t in times)
    require(clearance>0,'NO_REGULAR_SOURCE_TIME_CLEARANCE')
    epsilon=clearance/2
    return root-epsilon,root,root+epsilon

def contains_integer(a,b):
    return (a.numerator+a.denominator-1)//a.denominator <= b.numerator//b.denominator

class Reader:
    def __init__(self,seconds=1200):
        self.start=time.monotonic(); self.seconds=seconds; self.bytes=0
        self.files={}; self.used={}; self.full_hashes={}; self.stages=[]
    def check(self):
        if time.monotonic()-self.start>self.seconds: raise TimeoutError('COOPERATIVE_20_MINUTE_BUDGET')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>2*1024**3:
            raise MemoryError('TWO_GIB_RSS_BUDGET')
    def watch(self,path):
        path=Path(path)
        if path not in self.files:
            require(path.is_file() and not path.is_symlink(),'MISSING_OR_SYMLINK_INPUT')
            stat=path.stat(); self.files[path]=(stat.st_size,stat.st_mtime_ns)
    def take(self,path,offset,size):
        path=Path(path); self.watch(path)
        with path.open('rb') as f: f.seek(offset); data=f.read(size)
        require(len(data)==size,'TRUNCATED_SELECTED_INPUT')
        self.bytes+=size; self.used[(path,offset,size)]=hashlib.sha256(data).hexdigest()
        return data
    def whole(self,path):
        path=Path(path); self.watch(path); data=path.read_bytes(); self.bytes+=len(data)
        self.full_hashes[path]=hashlib.sha256(data).hexdigest(); return data
    @contextmanager
    def stage(self,name):
        wall,cpu,read=time.monotonic(),time.process_time(),self.bytes
        try: yield
        finally:
            self.stages.append({'stage':name,'wall_seconds':time.monotonic()-wall,
                'cpu_seconds':time.process_time()-cpu,'logical_bytes_examined':self.bytes-read,
                'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})
    def verify(self):
        bad=[]
        for p,expected in self.files.items():
            stat=p.stat()
            if (stat.st_size,stat.st_mtime_ns)!=expected: bad.append(str(p))
        for (p,offset,size),expected in self.used.items():
            with p.open('rb') as f: f.seek(offset); data=f.read(size)
            self.bytes+=size
            if hashlib.sha256(data).hexdigest()!=expected: bad.append(str(p)+':'+str(offset))
        for p,expected in self.full_hashes.items():
            h=hashlib.sha256()
            with p.open('rb') as f:
                while chunk:=f.read(1024*1024): h.update(chunk); self.bytes+=len(chunk)
            if h.hexdigest()!=expected: bad.append(str(p))
        return {'pass':not bad,'changed':sorted(set(bad)),'stat_checked_files':len(self.files),
                'selected_byte_ranges_rehashed':len(self.used),'complete_files_rehashed':len(self.full_hashes),
                'scope':'All opened file stats; full registry/camera/HV SHA256; referenced provenance and selected primary byte ranges, not full unrelated processed payload.'}

def load_hvs(cache,wanted,reader,existing=None):
    result={} if existing is None else existing
    wanted=set(wanted)-set(result)
    packed=np.array([n*256+g+128 for n,g in wanted],dtype=np.int64)
    if not wanted: return result
    for path in sorted((cache/'hypervertices').glob('*.bin')):
        reader.watch(path); digest=hashlib.sha256()
        with path.open('rb') as stream:
            header=stream.read(4); reader.bytes+=4; digest.update(header)
            count=struct.unpack('<i',header)[0]
            require(count>=0 and path.stat().st_size==4+28*count,'HV_VECTOR_LENGTH')
            for base in range(0,count,65536):
                reader.check(); data=stream.read(min(65536,count-base)*28)
                reader.bytes+=len(data); digest.update(data)
                records=np.frombuffer(data,dtype=HV_DTYPE)
                keys=records['node'].astype(np.int64)*256+records['group'].astype(np.int64)+128
                for index in np.flatnonzero(np.isin(keys,packed)):
                    off=int(index)*28; n=struct.unpack_from('<i',data,off)[0]; g=struct.unpack_from('<b',data,off+4)[0]
                    row={'position':tuple(F.from_float(x) for x in struct.unpack_from('<3f',data,off+8)),
                         'time':struct.unpack_from('<b',data,off+20)[0],
                         'span':struct.unpack_from('<b',data,off+21)[0],
                         'view':struct.unpack_from('<b',data,off+24)[0]}
                    require(row['span']>=0 and row['view'] in (0,1),'HV_ATTRIBUTES_INVALID')
                    if (n,g) in result: require(result[n,g]==row,'DUPLICATE_HV_DISAGREEMENT')
                    result[n,g]=row
        actual=digest.hexdigest()
        if path in reader.full_hashes: require(reader.full_hashes[path]==actual,'HV_CHANGED_BETWEEN_PASSES')
        reader.full_hashes[path]=actual
    require(wanted<=set(result),'REFERENCED_HV_MISSING')
    return result

def source_values(row):
    return (int(row['source_t_group']),int(row['source_record_index']),
            *(int(row[name]) for name in ('edge_x','edge_y','edge_z','edge_L','edge_tcoord','edge_tL','edge_dir','element')),
            *(hp(row[f'source_h{i}'])[0] for i in range(8)),
            *(hp(row[f'source_h{i}'])[1] for i in range(8)))

def verify_sources(cache,rows,reader):
    seen=set()
    for row in rows:
        values=source_values(row); ident=values[:2]
        if ident in seen: continue
        seen.add(ident); path=cache/'hyperpoly_meta'/f'{ident[0]}.bin'
        magic,version,size,layout,count=struct.unpack('<IIIIQ',reader.take(path,0,24))
        require((magic,version,size,layout)==(0x32504842,2,104,1),'BHP2_HEADER')
        require(path.stat().st_size==24+count*104 and 0<=ident[1]<count,'BHP2_LENGTH_OR_INDEX')
        actual=struct.unpack('<26i',reader.take(path,24+ident[1]*104,104))
        require(actual==values,'REGISTRY_BHP2_SOURCE_DISAGREEMENT')
    return len(seen)

def classify(event,hvs):
    rows=event['_rows']; row=rows[0]; root=event['_root']; stages=event['stages']
    hs=[HVID(*hp(row[f'h{i}'])) for i in range(4)]
    ts=[hvs[(h.node,h.group)]['time'] for h in hs]
    saddle=admissible_saddle(hs,ts,int(row['element']))
    require(saddle is not None and saddle.root==root and saddle.event_id()==event['event_id'],'REGISTRY_SADDLE_RECHECK_FAILED')
    require(all(int(row[f't{i}'])==ts[i] for i in range(4)),'REGISTRY_HV_TIME_DISAGREEMENT')
    stages['registry_saddle']=verdict('PASS','EXACT_CANONICAL_TEMPORAL_FACE_SADDLE')
    p=[hvs[(h.node,h.group)]['position'] for h in saddle.face_hvids]; u,v=saddle.u,saddle.v
    star=tuple((1-u)*(1-v)*p[0][k]+u*(1-v)*p[1][k]+u*v*p[2][k]+(1-u)*v*p[3][k] for k in range(3))
    tu=tuple((1-v)*(p[1][k]-p[0][k])+v*(p[2][k]-p[3][k]) for k in range(3))
    tv=tuple((1-u)*(p[3][k]-p[0][k])+u*(p[2][k]-p[1][k]) for k in range(3))
    event['_star']=star
    branches=[h for h,t in zip(saddle.face_hvids,saddle.face_times) if t<root]+[h for h,t in zip(saddle.face_hvids,saddle.face_times) if t>root]
    vertices=[(*star,root)]+[(*hvs[(h.node,h.group)]['position'],F(hvs[(h.node,h.group)]['time'])) for h in branches]
    tets,completion=complete_production_event_star(np.asarray([(0,3,1,4),(0,3,4,2)],np.int64))
    exact_regular=bool(any(cross(tu,tv))) and all(rank3([vertices[int(i)] for i in tet]) for tet in tets)
    stages['beb1_kernel']=verdict('PASS' if exact_regular else 'REJECT',
        'EXACT_NONDEGENERATE_COMPLETED_BEB1_KERNEL' if exact_regular else 'EXACT_DEGENERATE_SOURCE_KERNEL',
        tetrahedra=4,critical_side_faces_remaining=completion['critical_side_faces_remaining'],
        scope='Local exact source-labelled kernel only; no ordinary-boundary gluing claim.')
    # Retain the existing TV3 selector as a separate implementation condition.
    src=source_values(row); ch=tuple(HVID(src[10+i],src[18+i]) for i in range(8))
    ct=tuple(hvs[(h.node,h.group)]['time'] for h in ch)
    record=SourceRecord(src[0],src[1],tuple(src[2:5]),*src[5:10],ch)
    cell=Cell('forest-selected',record,len(rows),ch,ct,np.asarray([[float(x) for x in hvs[(h.node,h.group)]['position']] for h in ch]),
              tuple(ct[i+4]-ct[i] for i in range(4)),'NOT_ROUTE_CERTIFIED',canonical_partition(ch),'NOT_SAMPLED',0.,0.)
    try:
        build_production_half_handle(cell,saddle,int(row['face_side']))
        stages['beb1_kernel']['existing_tv3_selector']='PASS'
    except Exception as error:
        stages['beb1_kernel']['existing_tv3_selector']='UNSUPPORTED'
        stages['beb1_kernel']['selector_reason']=str(error)
        if exact_regular: stages['beb1_kernel'].update(status='UNSUPPORTED',reason_code='EXACT_KERNEL_REGULAR_BUT_EXISTING_TV3_SELECTOR_REFUSES')
    levels=candidate_levels(ts,root); event['_levels']=levels
    raw_boundary=[time_ordered_edge((hs[i].node,hs[i].group),(hs[(i+1)%4].node,hs[(i+1)%4].group),hvs) for i in range(4)]
    boundary=[effective(pair,root,hvs)[0] for pair in raw_boundary]
    event['_raw_boundary']=raw_boundary;event['_boundary']=boundary
    stages['candidate_window']=verdict('PASS','EXPLICIT_OLD_COMPILER_CLEARANCE_HALF_WINDOW',
        lower=fj(levels[0]),root=fj(root),upper=fj(levels[2]),certified_admitted_window=False,
        formula='epsilon=min(abs(face_HV_time-root))/2; compile_critical_beb1_event_ir.py:857-864',
        no_integer_source_predicate_breakpoint=not contains_integer(levels[0],levels[2]))
    event['_box']=tuple((min(q[k] for q in p),max(q[k] for q in p)) for k in range(3))

def parse_vids(blob):
    return [((struct.unpack_from('<i',blob,i)[0],struct.unpack_from('<b',blob,i+4)[0]),
             (struct.unpack_from('<i',blob,i+8)[0],struct.unpack_from('<b',blob,i+12)[0])) for i in range(0,len(blob),16)]

def scan_halo(cache,events,reader):
    inverse=defaultdict(set); singles=defaultdict(set); roots=sorted({e['_root'] for e in events})
    for index,e in enumerate(events):
        for pair in e.get('_boundary',[]):
            inverse[e['_root'],e['element'],pair].add(index)
            if pair[0]==pair[1]: singles[e['_root'],e['element'],pair[0]].add(index)
    halo=[]; record_count=0; wanted=set(); selected_records=set()
    for path in sorted((cache/'processed_hyperpolys').glob('*.bin')):
        pieces=path.stem.split('_')
        if len(pieces)!=2 or not all(x.isdigit() for x in pieces): continue
        group,start=map(int,pieces)
        eligible_roots=[r for r in roots if (group==0 or group==1 and r>=2) and r>=start-1]
        if not eligible_roots: continue
        meta=path.with_name(path.stem+'_hpmeta.bin');reader.watch(path);reader.watch(meta)
        require(meta.stat().st_size%132==0,'BPM2_SIZE')
        count=meta.stat().st_size//132
        if not count: require(path.stat().st_size==0,'EMPTY_PRIMARY_BPM2_DISAGREE');continue
        with path.open('rb') as pf,meta.open('rb') as mf,mmap.mmap(pf.fileno(),0,access=mmap.ACCESS_READ) as primary,mmap.mmap(mf.fileno(),0,access=mmap.ACCESS_READ) as metadata:
            # Check every prefix in small numpy views, not Python record objects.
            dtype=np.dtype({'names':['magic','version','size','layout','group','start','index'],
                'formats':['<u4']*4+['<i4']*3,'offsets':[0,4,8,12,16,20,24],'itemsize':132})
            first=struct.unpack_from('<i',metadata,24)[0]
            for base in range(0,count,65536):
                reader.check();n=min(65536,count-base); rows=np.ndarray((n,),dtype=dtype,buffer=metadata,offset=base*132)
                for name,value in [('magic',0x324D5042),('version',2),('size',132),('layout',1),('group',group),('start',start)]:
                    require(bool(np.all(rows[name]==value)),'BPM2_PREFIX_SCHEMA')
                require(bool(np.all(rows['index']==np.arange(first+base,first+base+n))),'BPM2_NONCONTIGUOUS_SORTED_OWNER')
                reader.bytes+=n*28
                del rows
            offset=0
            for ri in range(count):
                if ri%16384==0:reader.check()
                begin=offset; require(offset+5<=len(primary),'PRIMARY_TRUNCATED_HEADER')
                element=struct.unpack_from('<b',primary,offset)[0];offset+=1
                nt=struct.unpack_from('<i',primary,offset)[0];offset+=4
                require(2<=nt<=128 and offset+nt<=len(primary),'PRIMARY_TIME_COUNT')
                times=struct.unpack_from('<'+'b'*nt,primary,offset);offset+=nt;reader.bytes+=5+nt
                require(all(a<b for a,b in zip(times,times[1:])),'PRIMARY_TIME_ORDER')
                expanded=(times[0]-1,*times[1:-1],times[-1]+1)
                active={r:next((i for i in range(nt-1) if expanded[i]<=r<expanded[i+1]),-1) for r in eligible_roots if times[0]<=r<=times[-1]}
                captured=False
                for interval in range(nt-1):
                    face_index=0
                    while True:
                        require(offset+4<=len(primary),'PRIMARY_TRUNCATED_FACE_COUNT')
                        nv=struct.unpack_from('<i',primary,offset)[0];offset+=4;reader.bytes+=4
                        if nv==0:break
                        require(0<nv<=12 and face_index<16 and offset+nv*16<=len(primary),'PRIMARY_FACE_BOUND')
                        current=[r for r,s in active.items() if s==interval]
                        if current:
                            blob=primary[offset:offset+nv*16];reader.bytes+=len(blob);vids=parse_vids(blob)
                            for root in current:
                                candidates=set()
                                for pair in vids:
                                    candidates.update(inverse.get((root,element,key(*pair)),()))
                                    for h in pair:candidates.update(singles.get((root,element,h),()))
                                if candidates:
                                    require(len(halo)<100000,'BOUNDARY_HALO_OBJECT_CAP')
                                    halo.append((root,element,vids,(element,group,start,first+ri,interval,face_index),tuple(sorted(candidates))))
                                    wanted.update(h for pair in vids for h in pair);captured=True
                        offset+=nv*16;face_index+=1
                if captured:
                    reader.take(path,begin,offset-begin); m=reader.take(meta,ri*132,132)
                    require(struct.unpack_from('<i',m,28+9*4)[0]==element,'BPM2_PRIMARY_ELEMENT')
                    selected_records.add((group,start,first+ri))
                record_count+=1
            require(offset==len(primary),'PRIMARY_TRAILING_OR_COUNT_MISMATCH')
    return halo,wanted,{'complete':True,'processed_records_scanned':record_count,'halo_polygons':len(halo),'halo_processed_records':len(selected_records),
        'scope':'Every raw-active primary stream at both exact roots; all BPM2 prefixes; only complete candidate boundary-incidence halo retained.'}

def finish_event(event,polygons,hvs):
    stages=event['stages']; boundary=event['_boundary']; root=event['_root']
    if contains_integer(event['_levels'][0],event['_levels'][2]):
        stages['source_owner_closure']=verdict('UNKNOWN','UNPARTITIONED_INTEGER_SOURCE_TIME_THRESHOLD')
        stages['candidate_local_geometry']=verdict('UNKNOWN','UNPARTITIONED_INTEGER_SOURCE_TIME_THRESHOLD')
        return
    require(len(set(boundary))==4,'COLLAPSED_EVENT_BOUNDARY')
    positions={token(pair):effective(raw,root,hvs)[1] for pair,raw in zip(boundary,event['_raw_boundary'])}
    grouped={}
    for _,element,vids,prefix,_ in polygons:
        values=[effective(pair,root,hvs) for pair in vids]
        byid={}
        for v,p in values:
            name=token(v)
            if name in byid:require(byid[name]==p,'SAME_NATIVE_QUOTIENT_ID_DIFFERENT_IDEAL_COORDINATES')
            byid[name]=p
        for j in range(len(vids)-2):
            ids=tuple(token(values[i][0]) for i in (0,j+1,j+2))
            if not set(ids)&set(positions):continue
            name=canon(ids); row=grouped.setdefault(name,{'element':element,'source_vertices':list(name),'owners':[],
                'positions_t0':None,'positions_t1':None})
            row['owners'].append([*prefix,j])
            coordinates=[byid[x] for x in name]
            if row['positions_t0'] is not None:require(row['positions_t0']==coordinates,'SOURCE_FACE_REPLICAS_HAVE_DIFFERENT_COORDINATES')
            row['positions_t0']=row['positions_t1']=coordinates
    cycle=[token(x) for x in boundary]
    sources=[face for face in grouped if len(set(face))==3 and set(face)<=set(cycle)]
    if len(sources)!=2:
        stages['source_owner_closure']=verdict('REJECT','SOURCE_BOUNDARY_NOT_EXACTLY_TWO_ORDINARY_FACES',found=len(sources),
            scope='Does not satisfy this two-face replacement template; not evidence of a baseline defect or impossibility of another repair.')
        return
    orientations=[cycle,[cycle[0],*reversed(cycle[1:])]]
    disk=next((c for c in orientations if original_disk(c,sources)['status']=='PASS'),None)
    if disk is None:
        stages['source_owner_closure']=verdict('REJECT','SOURCE_TWO_FACES_NOT_ORIENTED_DISK');return
    cycle=disk; owners=[o for f in sources for o in grouped[f]['owners']]
    event['_owners']={tuple(o) for o in owners};event['_cycle']=cycle
    require(len(owners)==len(event['_owners']),'DUPLICATE_RAW_OWNER')
    temporal_constant=not contains_integer(event['_levels'][0],event['_levels'][2])
    stages['source_owner_closure']=verdict('PASS','COMPLETE_RAW_ROOT_OWNER_REPLICA_QUOTIENT',
        source_faces=[list(x) for x in sources],boundary_cycle=cycle,owners=owners,
        whole_candidate_window_identity_state_constant=temporal_constant,
        temporal_proof='All raw load/group/original interval/effective-clamp thresholds are integers; the closed candidate interval contains none.',
        scope='Canonical quotient of native original-ordered-VID rules with ideal rational coordinates, NOT actual C++ global indices or binary32 coordinate evaluation.')
    retained=[row for face,row in grouped.items() if face not in sources]
    audit=audit_interface(cycle,sources,retained,element=event['element'])
    stages['source_interface']=verdict(audit['status'],'COMPLETE_BOUNDARY_HALO_INTERFACE_AUDIT',audit=audit)
    diagonal=set(sources[0])&set(sources[1]);require(len(diagonal)==2,'SOURCE_DIAGONAL_ARITY')
    rawmap={token(b):raw for b,raw in zip(boundary,event['_raw_boundary'])}
    lower,_,upper=event['_levels']
    boundary_at=lambda t:[effective(rawmap[x],t,hvs)[1] for x in cycle]
    midpoint=lambda t:tuple(sum(effective(rawmap[x],t,hvs)[1][k] for x in diagonal)/2 for k in range(3))
    checks=[certify_segment(boundary_at(lower),boundary_at(root),midpoint(lower),event['_star']),
            certify_segment(boundary_at(root),boundary_at(upper),event['_star'],midpoint(upper))]
    state='PASS' if all(x['status']=='PASS' for x in checks) else 'UNSUPPORTED'
    stages['candidate_local_geometry']=verdict(state,'IDEAL_CANDIDATE_XY_GRAPH_FAN' if state=='PASS' else 'CURRENT_XY_GRAPH_CERTIFIER_CANNOT_ADMIT_THIS_SOURCE',
        half_statuses=[x['status'] for x in checks],candidate_center='exact bilinear saddle + exact source-diagonal endpoint midpoints',
        runtime_binary32=False,full_exterior=False,
        diagnostics=[{'status':x['status'],'reason':x.get('reason'),'witness':x.get('witness')} for x in checks if x['status']!='PASS'])

def pairwise(events):
    results=[]; per=defaultdict(Counter)
    for i,a in enumerate(events):
        for j in range(i+1,len(events)):
            b=events[j]
            if a['_root']!=b['_root']:continue
            if not all('_owners' in e for e in (a,b)):
                status,reason='UNKNOWN','SOURCE_SUPPORT_NOT_RESOLVED'
            elif a['element']==b['element'] and (a['_owners']&b['_owners'] or set(a['_cycle'])&set(b['_cycle'])):
                status,reason='UNSUPPORTED','SHARED_OWNER_OR_SOURCE_BOUNDARY'
            elif all('_box' in e for e in (a,b)) and any(a['_box'][k][1]<b['_box'][k][0] or b['_box'][k][1]<a['_box'][k][0] for k in range(3)):
                status,reason='PASS','STRICT_SOURCE_HV_CONVEX_HULL_AABB_SEPARATION'
            else:status,reason='UNKNOWN','DISJOINT_IDENTITIES_BUT_SPATIAL_SUPPORT_NOT_SEPARATED'
            per[i][status]+=1;per[j][status]+=1
            results.append({'a':a['key'],'b':b['key'],'root':fj(a['_root']),'status':status,'reason_code':reason})
    for i,e in enumerate(events):
        c=per[i];status='UNSUPPORTED' if c['UNSUPPORTED'] else 'UNKNOWN' if c['UNKNOWN'] else 'PASS'
        e['stages']['same_root_compatibility']=verdict(status,'PAIRWISE_CANDIDATE_SUPPORT_SCREEN_ONLY',pair_counts=dict(c),atomic_plan_compiled=False)
    return results

def run(cache,stage64,output,seconds=1200):
    cache,stage64,output=map(lambda p:Path(p).resolve(),(cache,stage64,output))
    require(not output.exists() and cache not in output.parents and output!=cache,'FRESH_OUTPUT_OUTSIDE_CACHE_REQUIRED')
    reader=Reader(seconds);events=[];started=time.monotonic();failure=None
    script_paths=[Path(__file__),HERE/'interface_topology.py',HERE/'window_geometry.py',HERE.parent/'tv0_tv4'/'theory_audit.py']
    script_hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in script_paths}
    import csv,io
    with reader.stage('registry_and_input_binding'):
        data=reader.whole(cache/'event_registry_p1.csv')
        require(hashlib.sha256(data).hexdigest()=='0ce1a439bf33b785db9dbb3ea55dbc548933477aca64d2fca64829f660609b52','FROZEN_FOREST_REGISTRY_HASH')
        rows=list(csv.DictReader(io.StringIO(data.decode())))
        groups=defaultdict(list)
        for row in rows:groups[row['canonical_event_id']].append(row)
        require(len(rows)==440 and len(groups)==131 and len({r['logical_incidence_id'] for r in rows})==268,'FOREST_FIXED_DENOMINATORS')
        camera=json.loads(reader.whole(stage64/'camera_inputs.json'))
        native=json.loads(reader.whole(stage64/'native_registry_verification.json'))
        require(native['protocol']['time_groups']==camera['time_mapping']['temporal_group_count']==2
                and camera['time_mapping']['maximum_discrete_time']==4,'FROZEN_FOREST_TWO_GROUP_FOUR_TIME_PROFILE_REQUIRED')
        for i,(eid,group) in enumerate(sorted(groups.items(),key=lambda item:(F(int(item[1][0]['root_num']),int(item[1][0]['root_den'])),item[0]))):
            roots={F(int(r['root_num']),int(r['root_den'])) for r in group};require(len(roots)==1,'CANONICAL_ROOT_DISAGREEMENT')
            events.append({'key':f'forest-{i:03d}-{hashlib.sha256(eid.encode()).hexdigest()[:12]}','event_id':eid,
                'element':int(group[0]['element']),'root':fj(next(iter(roots))),'raw_observations':len(group),
                'logical_incidences':len({r['logical_incidence_id'] for r in group}),'_rows':group,'_root':next(iter(roots)),
                'stages':{stage:verdict('NOT_REACHED','PREREQUISITE_NOT_ESTABLISHED') for stage in STAGES}})
    try:
        with reader.stage('referenced_hv_and_source_provenance'):
            wanted={hp(row[f'source_h{i}']) for row in rows for i in range(8)}
            hvs=load_hvs(cache,wanted,reader)
            verify_sources(cache,rows,reader)
            for e in events:e['stages']['source_provenance']=verdict('PASS','ALL_REGISTRY_REFERENCED_BHP2_ROWS_MATCH')
        with reader.stage('exact_saddle_beb1_candidate_classification'):
            for e in events:
                reader.check()
                try:classify(e,hvs)
                except Exception as error:
                    e['stages']['beb1_kernel']=verdict('UNKNOWN','CLASSIFICATION_EXCEPTION',detail=str(error))
        with reader.stage('streamed_all_root_owner_interface_halo'):
            halo,wanted,scan=scan_halo(cache,events,reader)
            load_hvs(cache,wanted,reader,hvs)
        per=defaultdict(list)
        for row in halo:
            for index in row[-1]:per[index].append(row)
        with reader.stage('complete_owner_interface_and_candidate_geometry'):
            for i,e in enumerate(events):
                reader.check()
                if '_boundary' not in e:continue
                try:finish_event(e,per[i],hvs)
                except Exception as error:
                    pending=next((s for s in ('source_owner_closure','source_interface','candidate_local_geometry') if e['stages'][s]['status']=='NOT_REACHED'),'candidate_local_geometry')
                    e['stages'][pending]=verdict('UNKNOWN','SOURCE_HALO_AUDIT_EXCEPTION',detail=str(error))
    except (ValueError,TimeoutError,MemoryError,OSError,struct.error) as error:
        failure=type(error).__name__+': '+str(error);scan=locals().get('scan',{'complete':False});pairs=locals().get('pairs',[])
    with reader.stage('same_root_and_natural_candidate_schedule'):
        pairs=pairwise(events)
        delta=fr(camera['time_mapping']['delta_seconds']);origin=float(fr(camera['time_mapping']['origin_seconds']))
        for e in events:
            if '_levels' not in e:continue
            a,_,b=e['_levels'];hits=[]
            for index,t in enumerate(camera['times_seconds']):
                physical=float(float(t)-origin);tau=F.from_float(float(physical/float(delta)))
                if a<tau<b:hits.append(index+1)
            e['stages']['natural_frame_hits']=verdict('PASS','EXPLICIT_CANDIDATE_WINDOW_GRID_INTERSECTION',
                hit_count=len(hits),frame_numbers=hits,natural_frames=len(camera['times_seconds']),
                denominator='explicit unadmitted candidate windows; NOT certified windows or visibility',time_mapping_status=camera['time_mapping']['status'])
    for e in events:
        e['stages']['global_exterior']=verdict('UNKNOWN','NO_COMPLETE_FOREST_EXTERIOR_GEOMETRY_CERTIFICATE',execution='NOT_ATTEMPTED')
        e['stages']['runtime_implementation']=verdict('UNSUPPORTED','CURRENT_PUBLIC_RUNTIME_IS_SINGLE_ELEMENT_FOREST_HAS_FIVE',cache_elements=[0,1,2,3,4])
        e['stages']['runtime_admission']=verdict('UNKNOWN','NO_FOREST_RUNTIME_TRANSACTION_EXECUTED',execution='NOT_ATTEMPTED',runtime_admitted=False)
        e['structurally_eligible_candidate']=all(e['stages'][s]['status']=='PASS' for s in ('registry_saddle','source_provenance','beb1_kernel','source_owner_closure','source_interface','candidate_local_geometry'))
    with reader.stage('input_immutability_verification'):immutable=reader.verify()
    require(immutable['pass'],'FROZEN_INPUT_CHANGED_NO_PUBLICATION')
    require(all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in script_hashes.items()),'EXECUTED_SOURCE_CHANGED_NO_PUBLICATION')
    eligible=[e for e in events if e['structurally_eligible_candidate']]
    clean=[{k:v for k,v in e.items() if not k.startswith('_')} for e in events]
    summary={'schema':'forest131-readonly-eligibility-funnel-v1','status':'COMPLETE_BOUNDED_ELIGIBILITY' if failure is None else 'PARTIAL_STOP',
        'failure':failure,'scope':'One 64-frame Forest6px cache, native geometry PRE_SURFACE_DISPLACEMENT; not six scenes or completed Infinigen geometry.',
        'denominators':{'all_canonical_events':131,'raw_observations':440,'logical_incidences':268,
            'structurally_eligible_candidates':len(eligible),'eligible_candidate_windows_with_natural_hits':sum(e['stages']['natural_frame_hits'].get('hit_count',0)>0 for e in eligible),
            'certified_admitted_windows':0,'runtime_transactions_attempted':0,'actually_modified_natural_frames':0,
            'runtime_success_rate_all_131':None,'runtime_success_rate_eligible':None,'visibility_evaluated':False},
        'stage_counts':{s:dict(Counter(e['stages'][s]['status'] for e in events)) for s in STAGES},
        'root_groups':[{'root':fj(r),'canonical_events':sum(e['_root']==r for e in events)} for r in sorted({e['_root'] for e in events})],
        'same_root_pair_counts':dict(Counter(x['status'] for x in pairs)),'scan':scan,'input_immutability':immutable,
        'measurements':{'stages':reader.stages,'wall_seconds':time.monotonic()-started,'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'logical_bytes_examined':reader.bytes,'logical_bytes_are_not_OS_disk_IO':True},
        'input_full_sha256':{str(p):h for p,h in reader.full_hashes.items()},
        'script_sha256':script_hashes[str(Path(__file__).resolve())],'executed_source_hashes_start_end_identical':True,
        'executed_source_sha256':script_hashes,
        'historical_build_seconds_separate_not_free':native['worker_wall_seconds'],
        'not_claimed':['Registry saddles equal admitted BEB1 repairs','Candidate windows are certified coverage','Root-grid hits are window coverage',
            'Sampled Jacobians prove nondegeneracy','SourceVID quotient is actual binary32 runtime identity','Other elements are geometrically disjoint by namespace','Unsupported runtime is a geometric rejection']}
    payloads={'events.json':clean,'same_root_pairs.json':pairs,'summary.json':summary}
    encoded={name:(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode() for name,value in payloads.items()}
    require(sum(map(len,encoded.values()))<20*1024**2,'PERSISTENT_20_MIB_BUDGET')
    output.mkdir(parents=True)
    for name,data in encoded.items():
        with (output/name).open('xb') as stream:stream.write(data)
    print(json.dumps({'status':summary['status'],'output':str(output),'bytes':sum(map(len,encoded.values())),
        'stages':summary['stage_counts'],'denominators':summary['denominators'],'wall_seconds':summary['measurements']['wall_seconds'],
        'max_rss_bytes':summary['measurements']['max_rss_bytes']},sort_keys=True),flush=True)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache-root','stage64','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--max-seconds',type=float,default=1200)
    args=p.parse_args();require(0<args.max_seconds<=1200,'MAX_20_MINUTES')
    result=run(args.cache_root,args.stage64,args.output,args.max_seconds)
    raise SystemExit(0 if result['status']=='COMPLETE_BOUNDED_ELIGIBILITY' else 2)
