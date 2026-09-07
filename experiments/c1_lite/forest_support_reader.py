"""Bounded, read-only ordinary support data, not a method admission.

The first scan retains ALL polygons of registry-associated processed records.
The second scan retains the complete vertex stars of ALL ordinary candidate
vertices, not the four critical-face edges. This supplies the established
lower-first two-face selector with complete incidence and owner replicas.

Snapshot.raw_triangles deliberately means the LEGACY PARSER'S EXPANDED-TIME
SUPERSET. Snapshot.actual_raw_triangles applies native original-time rejection.
Neither is an actual C++ index/float32 identity proof. Ordered serialized VIDs,
ideal exact coordinates, legacy parser64 values and native-VID-rule identity
are separate fields. extra_smooth special identities are unsupported.

No function writes artifacts or initializes a native mesher. Memory is bounded
by selected records/HV plus one mmap file pair; the original 1.1 GB Forest cache
is never copied or expanded into full Python mesh objects.
"""
from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import io
import json
import mmap
from pathlib import Path
import struct
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'source_splice'))
import forest_eligibility_funnel as streamed
from processed_mesh import (HVID, SourceVID, TriangleRef, Hypervertex, RawTriangle,
    _compute_slice_vertex, _active_groups, infer_cache_shape)

MAX_SELECTED_RECORDS = 100000
MAX_SNAPSHOT_TRIANGLES = 250000
MAX_TOTAL_SNAPSHOT_TRIANGLES = 1500000
PREFIX_DTYPE = np.dtype({'names':['magic','version','size','layout','group','start','index'],
    'formats':['<u4']*4+['<i4']*3, 'offsets':[0,4,8,12,16,20,24], 'itemsize':132})


class SupportReadError(RuntimeError):
    """Incomplete/unsupported input; callers must report UNKNOWN, not no patch."""


def require(condition, message):
    if not condition:
        raise SupportReadError(message)


def _vid(pair):
    return SourceVID.canonical(HVID(*pair[0]), HVID(*pair[1]))


def _pair(source_vid):
    return ((source_vid.first.node,source_vid.first.group),
            (source_vid.second.node,source_vid.second.group))


def _legacy_order(triangle):
    r = triangle.reference
    return (-r.t_group,r.t_start,r.sorted_record_index,r.interval_index,r.face_index,r.fan_index)


def _expanded_interval(times, tau):
    expanded=(times[0]-1,*times[1:-1],times[-1]+1)
    return next((i for i in range(len(expanded)-1) if expanded[i]<=tau<expanded[i+1]), -1)


def _is_actual_raw(times, interval, tau):
    return not (interval==0 and tau<times[0] or interval==len(times)-2 and tau>times[-1])


def _stream_inventory(cache,groups,maximum):
    primary=set();metadata=set()
    for path in (cache/'processed_hyperpolys').glob('*.bin'):
        stem=path.stem
        target=metadata if stem.endswith('_hpmeta') else primary
        if stem.endswith('_hpmeta'):stem=stem[:-7]
        parts=stem.split('_')
        if len(parts)==2 and all(p.isdigit() for p in parts):target.add(tuple(map(int,parts)))
    expected={(g,s) for g in range(groups) for s in range(maximum)}
    require(primary==metadata==expected,'INCOMPLETE_PRIMARY_BPM2_STREAM_MATRIX')
    return frozenset(expected)


@dataclass(frozen=True)
class ProcessedRecord:
    element: int
    group: int
    start: int
    sorted_index: int
    times: tuple[int,...]
    intervals: tuple
    provenance_values: tuple[int,...]
    primary_path: str
    primary_offset: int
    primary_size: int
    primary_sha256: str
    metadata_path: str
    metadata_offset: int
    metadata_sha256: str

    @property
    def key(self):
        return self.group,self.start,self.sorted_index


@dataclass
class Snapshot:
    event_id: str
    tau: Fraction
    raw_triangles: tuple[RawTriangle,...]
    actual_raw_triangles: tuple[RawTriangle,...]
    details_by_owner: dict
    halo_vertex_ids: frozenset[SourceVID]
    event_candidates_complete: bool
    halo_complete: bool

    def summary(self):
        return {'event_id':self.event_id,'time':streamed.fj(self.tau),
            'legacy_expanded_triangles':len(self.raw_triangles),
            'actual_raw_triangles':len(self.actual_raw_triangles),
            'event_candidate_raw_occurrences':sum(t.event_record for t in self.raw_triangles),
            'candidate_vertex_ids':len(self.halo_vertex_ids),
            'event_candidates_complete':self.event_candidates_complete,'halo_complete':self.halo_complete,
            'native_legacy_identity_disagreements':sum(not d['native_legacy_identity_equal'] for d in self.details_by_owner.values()),
            'raw_triangles_scope':'Legacy expanded-interval parser superset, not native raw output.',
            'actual_raw_scope':'Original-time emission-filtered SourceVID quotient, not actual C++ array identities.',
            'extra_smooth':'UNSUPPORTED'}


def _decode_record(blob, meta, path, offset, metadata_path, metadata_offset):
    magic,version,size,layout,group,start,index=struct.unpack_from('<4I3i',meta)
    require((magic,version,size,layout)==(0x324D5042,2,132,1),'BPM2_SELECTED_SCHEMA')
    provenance=struct.unpack_from('<26i',meta,28)
    element=struct.unpack_from('<b',blob)[0]
    count=struct.unpack_from('<i',blob,1)[0]
    times=struct.unpack_from('<'+'b'*count,blob,5);position=5+count;intervals=[]
    require(provenance[9]==element,'BPM2_PRIMARY_ELEMENT_DISAGREEMENT')
    for _ in range(count-1):
        polygons=[]
        while True:
            n=struct.unpack_from('<i',blob,position)[0];position+=4
            if not n:break
            polygons.append(tuple(streamed.parse_vids(blob[position:position+n*16])))
            position+=n*16
        intervals.append(tuple(polygons))
    require(position==len(blob),'SELECTED_RECORD_TRAILING_BYTES')
    return ProcessedRecord(element,group,start,index,tuple(times),tuple(intervals),tuple(provenance),
        str(path),offset,len(blob),hashlib.sha256(blob).hexdigest(),str(metadata_path),metadata_offset,
        hashlib.sha256(meta).hexdigest())


def _bind_full_mmap(reader,path,mapping):
    actual=hashlib.sha256(mapping).hexdigest()
    previous=reader.full_hashes.get(path)
    require(previous is None or previous==actual,'INPUT_CHANGED_BETWEEN_SUPPORT_SCANS: '+str(path))
    reader.full_hashes[path]=actual
    reader.bytes+=len(mapping)


def _scan(cache,reader,contexts,polygon_matches=None):
    """contexts(key,element,times)->(capture_all, context); match returns queries."""
    selected={};matches=defaultdict(set);total=0
    for path in sorted((cache/'processed_hyperpolys').glob('*.bin')):
        parts=path.stem.split('_')
        if len(parts)!=2 or not all(part.isdigit() for part in parts):continue
        group,start=map(int,parts);metadata_path=path.with_name(path.stem+'_hpmeta.bin')
        reader.watch(path);reader.watch(metadata_path)
        require(metadata_path.stat().st_size%132==0,'BPM2_RECORD_ALIGNMENT')
        count=metadata_path.stat().st_size//132
        if not count:
            require(path.stat().st_size==0,'EMPTY_BPM2_NONEMPTY_PRIMARY')
            reader.full_hashes[path]=hashlib.sha256(b'').hexdigest()
            reader.full_hashes[metadata_path]=hashlib.sha256(b'').hexdigest()
            continue
        with path.open('rb') as primary_file,metadata_path.open('rb') as metadata_file, \
                mmap.mmap(primary_file.fileno(),0,access=mmap.ACCESS_READ) as primary, \
                mmap.mmap(metadata_file.fileno(),0,access=mmap.ACCESS_READ) as metadata:
            first=struct.unpack_from('<i',metadata,24)[0]
            for base in range(0,count,65536):
                reader.check();n=min(65536,count-base)
                rows=np.ndarray((n,),dtype=PREFIX_DTYPE,buffer=metadata,offset=base*132)
                for field,value in [('magic',0x324D5042),('version',2),('size',132),('layout',1),('group',group),('start',start)]:
                    require(bool(np.all(rows[field]==value)),'BPM2_PREFIX_SCHEMA')
                require(bool(np.all(rows['index']==np.arange(first+base,first+base+n))),'BPM2_NONCONTIGUOUS_SORTED_RECORDS')
                del rows
            offset=0
            for ri in range(count):
                if ri%8192==0:reader.check()
                begin=offset;require(offset+5<=len(primary),'TRUNCATED_PRIMARY_HEADER')
                element=struct.unpack_from('<b',primary,offset)[0]
                nt=struct.unpack_from('<i',primary,offset+1)[0];offset+=5
                require(2<=nt<=128 and offset+nt<=len(primary),'PRIMARY_TIME_COUNT')
                times=struct.unpack_from('<'+'b'*nt,primary,offset);offset+=nt
                require(all(a<b for a,b in zip(times,times[1:])),'PRIMARY_TIME_ORDER')
                key=(group,start,first+ri);capture,context=contexts(key,element,times)
                query_matches=set()
                for interval in range(nt-1):
                    face=0
                    while True:
                        require(offset+4<=len(primary),'TRUNCATED_PRIMARY_FACE_COUNT')
                        nv=struct.unpack_from('<i',primary,offset)[0];offset+=4
                        if not nv:break
                        require(0<nv<=12 and face<16 and offset+nv*16<=len(primary),'PRIMARY_POLYGON_BOUND')
                        if polygon_matches is not None and context and interval in context:
                            pairs=streamed.parse_vids(primary[offset:offset+nv*16])
                            query_matches.update(polygon_matches(context[interval],element,pairs))
                        offset+=nv*16;face+=1
                if capture or query_matches:
                    require(len(selected)<MAX_SELECTED_RECORDS,'SELECTED_SUPPORT_RECORD_CAP')
                    blob=reader.take(path,begin,offset-begin)
                    meta=reader.take(metadata_path,ri*132,132)
                    record=_decode_record(blob,meta,path,begin,metadata_path,ri*132)
                    require(record.key==key and key not in selected,'DUPLICATE_SELECTED_RECORD_KEY')
                    selected[key]=record;matches[key].update(query_matches)
                total+=1
            require(offset==len(primary),'PRIMARY_METADATA_COUNT_OR_TRAILING_BYTES')
            _bind_full_mmap(reader,path,primary);_bind_full_mmap(reader,metadata_path,metadata)
    return selected,matches,{'processed_records_scanned':total,'selected_records':len(selected),'complete':True}


class SupportDataset:
    def __init__(self,cache,registry,reader):
        self.cache=cache;self.registry=registry;self.reader=reader
        self.roots={eid:Fraction(int(rows[0]['root_num']),int(rows[0]['root_den'])) for eid,rows in registry.items()}
        self.event_ids=tuple(sorted(registry,key=lambda eid:(self.roots[eid],eid)))
        self.event_record_keys={eid:frozenset((int(r['t_group']),int(r['t_start']),int(r['sorted_record_index'])) for r in rows) for eid,rows in registry.items()}
        self.groups,self.maximum=infer_cache_shape(cache)
        self.stream_inventory=_stream_inventory(cache,self.groups,self.maximum)
        self.records={};self.hypervertices={};self.exact_hvs={};self.scans=[]
        self._provenance_checked={};self._header_checked={};self._sources={}
        for path in (Path(__file__),HERE/'forest_eligibility_funnel.py',HERE.parent/'source_splice'/'processed_mesh.py'):
            self._sources[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
        for eid,rows in registry.items():
            require(len({Fraction(int(r['root_num']),int(r['root_den'])) for r in rows})==1,'REGISTRY_ROOT_DISAGREEMENT')
            require(len({int(r['element']) for r in rows})==1,'REGISTRY_ELEMENT_DISAGREEMENT')

    def _add_records(self,records):
        for key,record in records.items():
            previous=self.records.get(key)
            require(previous is None or previous==record,'RECORD_CHANGED_BETWEEN_SCANS')
            values=record.provenance_values;ident=values[:2]
            if ident in self._provenance_checked:
                require(self._provenance_checked[ident]==values,'ONE_SOURCE_IDENT_MULTIPLE_VALUES')
            else:
                path=self.cache/'hyperpoly_meta'/f'{ident[0]}.bin'
                if path not in self._header_checked:
                    header=self.reader.take(path,0,24)
                    magic,version,size,layout,count=struct.unpack('<IIIIQ',header)
                    require((magic,version,size,layout)==(0x32504842,2,104,1),'BHP2_HEADER_SCHEMA')
                    require(path.stat().st_size==24+104*count,'BHP2_LENGTH')
                    self._header_checked[path]=count
                require(0<=ident[1]<self._header_checked[path],'BHP2_SOURCE_INDEX')
                actual=struct.unpack('<26i',self.reader.take(path,24+104*ident[1],104))
                require(actual==values,'BPM2_BHP2_SOURCE_DISAGREEMENT')
                self._provenance_checked[ident]=values
            self.records[key]=record
        wanted={h for r in records.values() for interval in r.intervals for polygon in interval for pair in polygon for h in pair}
        # Original eight HV are needed by TV3 and the saddle/space-position contract.
        wanted.update(streamed.hp(row[f'source_h{i}']) for rows in self.registry.values() for row in rows for i in range(8))
        streamed.load_hvs(self.cache,wanted,self.reader,self.exact_hvs)
        for (node,group),h in self.exact_hvs.items():
            self.hypervertices[HVID(node,group)]=Hypervertex(tuple(float(x) for x in h['position']),h['time'],h['span'],h['view'])

    def _evaluate(self,record,tau,event_id):
        if record.group not in _active_groups(tau,self.groups,self.maximum) or tau<record.start-1:return []
        interval=_expanded_interval(record.times,tau)
        if interval<0:return []
        result=[]
        for face,polygon in enumerate(record.intervals[interval]):
            legacy=[_compute_slice_vertex(_vid(pair),tau,self.hypervertices) for pair in polygon]
            native=[streamed.effective(pair,tau,self.exact_hvs) for pair in polygon]
            for fan in range(len(polygon)-2):
                indices=(0,fan+1,fan+2)
                ref=TriangleRef(record.element,record.group,record.start,record.sorted_index,interval,face,fan)
                triangle=RawTriangle(ref,tuple(legacy[i][0] for i in indices),tuple(legacy[i][1] for i in indices),
                    tuple(legacy[i][2] for i in indices),record.key in self.event_record_keys[event_id])
                native_ids=tuple(_vid(native[i][0]) for i in indices)
                detail={'actual_raw_emitted':_is_actual_raw(record.times,interval,tau),
                    'ordered_raw_vids':tuple(polygon[i] for i in indices),
                    'positions_exact':tuple(native[i][1] for i in indices),
                    'native_source_vertices':native_ids,
                    'native_legacy_identity_equal':native_ids==triangle.source_vertices,
                    'record_key':record.key,'original_times':record.times,
                    'provenance_values':record.provenance_values,
                    'coordinate_model':'Ideal rational interpolation of serialized HV binary32; positions separately use legacy parser binary64.',
                    'actual_cpp_identity_proven':False}
                result.append((triangle,detail))
        return result

    def event_triangles(self,event_id,tau):
        require(event_id in self.registry,'UNKNOWN_CANONICAL_EVENT_ID')
        tau=Fraction(tau)
        values=[t for key in self.event_record_keys[event_id] for t,_ in self._evaluate(self.records[key],tau,event_id)]
        return tuple(sorted(values,key=_legacy_order))

    def load_halos(self,requests):
        normalized={eid:tuple(sorted({Fraction(t) for t in times})) for eid,times in requests.items()}
        require(all(eid in self.registry for eid in normalized),'UNKNOWN_REQUESTED_EVENT')
        query_keys=[(eid,t) for eid in self.event_ids if eid in normalized for t in normalized[eid]]
        targets={q:frozenset(v for tri in self.event_triangles(*q) for v in tri.source_vertices) for q in query_keys}
        inverse=defaultdict(set);singles=defaultdict(set)
        for q,vertices in targets.items():
            element=int(self.registry[q[0]][0]['element'])
            for v in vertices:
                pair=_pair(v);inverse[element,pair].add(q)
                if pair[0]==pair[1]:singles[element,pair[0]].add(q)
        active={q:set(_active_groups(q[1],self.groups,self.maximum)) for q in query_keys}
        # Deduplicate identical time predicates to avoid events * millions of records.
        by_group=defaultdict(lambda:defaultdict(set))
        for q in query_keys:
            for group in active[q]:by_group[group][q[1]].add(q)

        def contexts(key,element,times):
            group,start,_=key;out=defaultdict(set)
            for tau,queries in by_group[group].items():
                if tau<start-1:continue
                interval=_expanded_interval(times,tau)
                if interval>=0:out[interval].update(queries)
            return False,out

        def matches(context,element,pairs):
            found=set()
            for pair in pairs:
                found.update(inverse.get((element,streamed.key(*pair)),()))
                for h in pair:found.update(singles.get((element,h),()))
            return found & context

        with self.reader.stage('support_complete_candidate_vertex_halos'):
            records,matched,scan=_scan(self.cache,self.reader,contexts,matches)
            self._add_records(records);self.scans.append(scan)
        per_query=defaultdict(set)
        for key,queries in matched.items():
            for q in queries:per_query[q].add(key)
        snapshots={};total=0
        with self.reader.stage('support_snapshot_evaluation'):
            for q in query_keys:
                self.reader.check();eid,tau=q;triangles=[];details={}
                # Include empty/degenerate event candidates even if no proper disk exists.
                keys=per_query[q]|set(self.event_record_keys[eid])
                for key in keys:
                    for triangle,detail in self._evaluate(self.records[key],tau,eid):
                        if not (set(triangle.source_vertices)&targets[q]):continue
                        owner=triangle.reference.values()
                        require(owner not in details,'DUPLICATE_RAW_OWNER_IN_HALO')
                        triangles.append(triangle);details[owner]=detail
                        require(len(triangles)<=MAX_SNAPSHOT_TRIANGLES,'SINGLE_SNAPSHOT_TRIANGLE_CAP')
                triangles.sort(key=_legacy_order)
                candidate_owners={t.reference.values() for t in self.event_triangles(eid,tau)}
                require(candidate_owners<=set(details),'EVENT_CANDIDATE_MISSING_FROM_COMPLETE_HALO')
                total+=len(triangles);require(total<=MAX_TOTAL_SNAPSHOT_TRIANGLES,'TOTAL_SNAPSHOT_TRIANGLE_CAP')
                snapshots[q]=Snapshot(eid,tau,tuple(triangles),tuple(t for t in triangles if details[t.reference.values()]['actual_raw_emitted']),
                    details,targets[q],True,True)
        return snapshots

    def verify_inputs(self):
        require(_stream_inventory(self.cache,self.groups,self.maximum)==self.stream_inventory,'SOURCE_STREAM_INVENTORY_CHANGED')
        with self.reader.stage('formal_support_input_verification'):
            result=self.reader.verify()
        require(result['pass'],'SOURCE_INPUT_CHANGED')
        require(all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in self._sources.items()),'EXECUTED_SUPPORT_READER_CHANGED')
        result.update(executed_source_sha256=self._sources,executed_sources_unchanged=True,
            input_full_sha256={str(p):h for p,h in self.reader.full_hashes.items()})
        result['scope']='Full opened primary/BPM2/registry/HV hashes and selected BHP2 ranges, plus all opened file stats; not unrelated cache files.'
        return result

    def summary(self):
        return {'schema':'forest-formal-support-reader-v1','events':len(self.event_ids),
            'group_count':self.groups,'maximum_discrete_time':self.maximum,
            'selected_records':len(self.records),'referenced_hv':len(self.hypervertices),
            'verified_BHP2_sources':len(self._provenance_checked),'scans':self.scans,
            'stages':self.reader.stages,'logical_bytes_examined':self.reader.bytes,
            'legacy_selector_scope':'All associated processed-record candidates and complete candidate vertex-star halo.',
            'actual_raw_filter_available':True,'actual_runtime_identity_proof':False,'extra_smooth':'UNSUPPORTED'}


def read_event_candidates(cache_root,event_ids=None,*,seconds=1200):
    """Read complete associated records for all/requested canonical events.

    Required caller sequence: load_halos(requests), compile/validate, then
    verify_inputs() before publishing results. No native global state is used.
    """
    require(0<seconds<=3600,'SUPPORT_READER_MAX_ONE_HOUR_COOPERATIVE_BUDGET')
    cache=Path(cache_root).resolve();reader=streamed.Reader(seconds)
    with reader.stage('formal_registry_and_all_associated_records'):
        data=reader.whole(cache/'event_registry_p1.csv')
        rows=list(csv.DictReader(io.StringIO(data.decode('utf-8'))));all_registry=defaultdict(list)
        for row in rows:all_registry[row['canonical_event_id']].append(row)
        chosen=set(all_registry) if event_ids is None else set(event_ids)
        require(chosen and chosen<=set(all_registry),'NO_OR_UNKNOWN_REQUESTED_EVENTS')
        registry={eid:tuple(all_registry[eid]) for eid in chosen}
        dataset=SupportDataset(cache,registry,reader)
        wanted={key for keys in dataset.event_record_keys.values() for key in keys}
        records,_,scan=_scan(cache,reader,lambda key,element,times:(key in wanted,None))
        require(set(records)==wanted,'REGISTRY_PROCESSED_RECORD_NOT_FOUND')
        for eid,event_rows in registry.items():
            for row in event_rows:
                key=tuple(int(row[n]) for n in ('t_group','t_start','sorted_record_index'))
                require(records[key].provenance_values==streamed.source_values(row),'REGISTRY_BPM2_FULL_SOURCE_DISAGREEMENT')
        dataset._add_records(records);dataset.scans.append(scan)
    return dataset
