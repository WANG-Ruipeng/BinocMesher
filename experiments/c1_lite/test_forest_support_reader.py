"""Small generated-cache tests for complete formal support candidates/halos."""
import csv
from fractions import Fraction as F
import hashlib
from pathlib import Path
import struct
import tempfile
import unittest

import forest_support_reader as m
from processed_mesh import trace_processed_triangles


def pack_vid(pair):
    return b''.join(struct.pack('<ib3x',*h) for h in pair)


def pack_record(element,times,intervals):
    return (struct.pack('<bi',element,len(times))+struct.pack('<'+'b'*len(times),*times)
        +b''.join(b''.join(struct.pack('<i',len(poly))+b''.join(map(pack_vid,poly)) for poly in interval)
                  +struct.pack('<i',0) for interval in intervals))


def values(index,element=0):
    return (0,index,0,0,0,0,0,0,0,element,*range(1,9),*(0 for _ in range(8)))


def registry_row(index=0,source_index=0,group=0,element=0,event='synthetic-event',root=F(3,2)):
    source=values(source_index,element)
    row={'canonical_event_id':event,'t_group':group,'t_start':0,'sorted_record_index':index,
         'root_num':root.numerator,'root_den':root.denominator,
         'source_t_group':source[0],'source_record_index':source[1]}
    row.update(zip(('edge_x','edge_y','edge_z','edge_L','edge_tcoord','edge_tL','edge_dir','element'),source[2:10]))
    row.update({f'source_h{i}':f'{source[10+i]}:{source[18+i]}' for i in range(8)})
    return row


def fixture(cache,records=None,rows=None,hv_overrides=None):
    pairs=[((1+2*i,0),(2+2*i,0)) for i in range(4)]
    outside=((9,0),(10,0))
    if records is None:
        records=[(0,0,0,[0,2],[[pairs]],0),
                 (0,1,0,[0,2],[[pairs]],0),
                 (0,2,0,[0,2],[[[pairs[1],pairs[0],outside]]],1)]
    if rows is None:rows=[registry_row()]
    processed=cache/'processed_hyperpolys';processed.mkdir()
    by_group={}
    for group,index,element,times,intervals,source_index in records:
        by_group.setdefault(group,[]).append((index,element,times,intervals,source_index))
    for group,group_records in by_group.items():
        group_records.sort();blob=b'';meta=b''
        for index,element,times,intervals,source_index in group_records:
            blob+=pack_record(element,times,intervals)
            meta+=struct.pack('<4I3i26i',0x324D5042,2,132,1,group,0,index,*values(source_index,element))
        (processed/f'{group}_0.bin').write_bytes(blob)
        (processed/f'{group}_0_hpmeta.bin').write_bytes(meta)
    for name in (f'{g}_{s}' for g in range(2) for s in range(4)):
        if not (processed/(name+'.bin')).exists():
            (processed/(name+'.bin')).write_bytes(b'')
            (processed/(name+'_hpmeta.bin')).write_bytes(b'')
    provenance=cache/'hyperpoly_meta';provenance.mkdir()
    sources={source:values(source,element) for _,_,element,_,_,source in records}
    maximum=max(sources)
    (provenance/'0.bin').write_bytes(struct.pack('<IIIIQ',0x32504842,2,104,1,maximum+1)
        +b''.join(struct.pack('<26i',*sources.get(i,values(i))) for i in range(maximum+1)))
    hvdir=cache/'hypervertices';hvdir.mkdir();payload=b''
    points=[(0,0,0),(1,0,0),(1,1,0),(0,1,0),(1,-1,0)]
    for n in range(1,11):
        row={'time':0 if n%2 else 2,'point':points[(n-1)//2],'span':0,'view':1}
        row.update((hv_overrides or {}).get(n,{}))
        b=bytearray(28);struct.pack_into('<ib',b,0,n,0);struct.pack_into('<3f',b,8,*row['point'])
        struct.pack_into('<bb',b,20,row['time'],row['span']);struct.pack_into('<b',b,24,row['view']);payload+=b
    (hvdir/'0.bin').write_bytes(struct.pack('<i',10)+payload)
    with (cache/'event_registry_p1.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return pairs,outside


class ReaderTests(unittest.TestCase):
    def test_full_candidates_and_replicas_not_critical_quad(self):
        with tempfile.TemporaryDirectory() as directory:
            cache=Path(directory);pairs,outside=fixture(cache)
            dataset=m.read_event_candidates(cache)
            self.assertEqual(len(dataset.event_triangles('synthetic-event',F(3,2))),2)
            snapshots=dataset.load_halos({'synthetic-event':[F(5,4),F(3,2),F(7,4)]})
            snapshot=snapshots['synthetic-event',F(3,2)]
            self.assertTrue(snapshot.halo_complete and snapshot.event_candidates_complete)
            self.assertEqual(len(snapshot.raw_triangles),5)
            self.assertEqual(len(snapshot.actual_raw_triangles),5)
            self.assertEqual(sum(t.event_record for t in snapshot.raw_triangles),2)
            self.assertTrue(any(m._vid(outside) in t.source_vertices for t in snapshot.raw_triangles))
            self.assertEqual(len(snapshot.details_by_owner),5)
            self.assertTrue(dataset.verify_inputs()['pass'])
            self.assertEqual(dataset.summary()['group_count'],2)
            self.assertEqual(dataset.summary()['maximum_discrete_time'],4)

    def test_snapshot_exactly_matches_full_legacy_parser_halo(self):
        with tempfile.TemporaryDirectory() as directory:
            cache=Path(directory);fixture(cache)
            dataset=m.read_event_candidates(cache);tau=F(3,2)
            snapshot=dataset.load_halos({'synthetic-event':[tau]})['synthetic-event',tau]
            full,_=trace_processed_triangles(cache,tau,event_id='synthetic-event')
            expected=tuple(t for t in full if set(t.source_vertices)&snapshot.halo_vertex_ids)
            self.assertEqual(snapshot.raw_triangles,expected)

    def test_expanded_superset_and_actual_raw_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            cache=Path(directory);fixture(cache)
            dataset=m.read_event_candidates(cache);tau=F(5,2)
            snapshot=dataset.load_halos({'synthetic-event':[tau]})['synthetic-event',tau]
            self.assertEqual(len(snapshot.raw_triangles),5)
            self.assertEqual(snapshot.actual_raw_triangles,())
            self.assertTrue(all(d['actual_raw_emitted'] is False for d in snapshot.details_by_owner.values()))

    def test_endpoint_equality_remains_actual_raw(self):
        with tempfile.TemporaryDirectory() as directory:
            cache=Path(directory);fixture(cache);dataset=m.read_event_candidates(cache)
            snapshot=dataset.load_halos({'synthetic-event':[F(2)]})['synthetic-event',F(2)]
            self.assertEqual(len(snapshot.actual_raw_triangles),5)

    def test_all_intervals_of_associated_record_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            cache=Path(directory);pairs=[((1+2*i,0),(2+2*i,0)) for i in range(4)]
            fixture(cache,records=[(0,0,0,[0,1,2],[[pairs[:3]],[pairs]],0)])
            dataset=m.read_event_candidates(cache)
            self.assertEqual(len(dataset.event_triangles('synthetic-event',F(1,2))),1)
            self.assertEqual(len(dataset.event_triangles('synthetic-event',F(3,2))),2)
            self.assertEqual(len(dataset.records[0,0,0].intervals),2)

    def test_opposite_face_and_all_replica_owners_in_halo(self):
        with tempfile.TemporaryDirectory() as directory:
            cache=Path(directory);pairs=[((1+2*i,0),(2+2*i,0)) for i in range(4)]
            fixture(cache,records=[(0,0,0,[0,2],[[pairs[:3]]],0),
                (0,1,0,[0,2],[[pairs[:3]]],0),(0,2,0,[0,2],[[list(reversed(pairs[:3]))]],1)])
            dataset=m.read_event_candidates(cache)
            snapshot=dataset.load_halos({'synthetic-event':[F(3,2)]})['synthetic-event',F(3,2)]
            self.assertEqual(len(snapshot.raw_triangles),3)
            self.assertEqual(len({t.oriented_key() for t in snapshot.raw_triangles}),2)

    def test_native_ordered_collapse_separate_from_legacy_id(self):
        with tempfile.TemporaryDirectory() as directory:
            cache=Path(directory);fixture(cache,hv_overrides={1:{'time':2},2:{'time':0}})
            dataset=m.read_event_candidates(cache)
            snapshot=dataset.load_halos({'synthetic-event':[F(2)]})['synthetic-event',F(2)]
            self.assertTrue(any(not d['native_legacy_identity_equal'] for d in snapshot.details_by_owner.values()))
            first=snapshot.details_by_owner[(0,0,0,0,0,0,0)]
            self.assertEqual(first['ordered_raw_vids'][0],((1,0),(2,0)))
            self.assertFalse(first['actual_cpp_identity_proven'])

    def test_legacy_group_order_and_event_specific_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            cache=Path(directory);pairs=[((1+2*i,0),(2+2*i,0)) for i in range(4)]
            fixture(cache,records=[(0,0,0,[0,3],[[pairs]],0),(1,0,0,[0,3],[[pairs]],0)],
                rows=[registry_row(),registry_row(group=1,event='second-event')])
            dataset=m.read_event_candidates(cache)
            snapshots=dataset.load_halos({'synthetic-event':[F(5,2)],'second-event':[F(5,2)]})
            a=snapshots['synthetic-event',F(5,2)];b=snapshots['second-event',F(5,2)]
            self.assertEqual([t.reference.t_group for t in a.raw_triangles],[1,1,0,0])
            self.assertEqual([t.event_record for t in a.raw_triangles],[False,False,True,True])
            self.assertEqual([t.event_record for t in b.raw_triangles],[True,True,False,False])

    def test_missing_registry_processed_record_is_unknown_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            cache=Path(directory);fixture(cache,rows=[registry_row(index=999)])
            with self.assertRaisesRegex(m.SupportReadError,'REGISTRY_PROCESSED_RECORD_NOT_FOUND'):
                m.read_event_candidates(cache)

    def test_missing_non_event_primary_is_incomplete_not_empty_support(self):
        with tempfile.TemporaryDirectory() as directory:
            cache=Path(directory);fixture(cache)
            (cache/'processed_hyperpolys/0_1.bin').unlink()
            with self.assertRaisesRegex(m.SupportReadError,'INCOMPLETE_PRIMARY_BPM2_STREAM_MATRIX'):
                m.read_event_candidates(cache)

    def test_missing_non_event_file_pair_is_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            cache=Path(directory);fixture(cache)
            (cache/'processed_hyperpolys/0_1.bin').unlink()
            (cache/'processed_hyperpolys/0_1_hpmeta.bin').unlink()
            with self.assertRaisesRegex(m.SupportReadError,'INCOMPLETE_PRIMARY_BPM2_STREAM_MATRIX'):
                m.read_event_candidates(cache)

    def test_conflicting_registry_full_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cache=Path(directory);row=registry_row();row['edge_x']=1;fixture(cache,rows=[row])
            with self.assertRaisesRegex(m.SupportReadError,'REGISTRY_BPM2_FULL_SOURCE_DISAGREEMENT'):
                m.read_event_candidates(cache)

    def test_changed_cache_between_phases_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            cache=Path(directory);fixture(cache);dataset=m.read_event_candidates(cache)
            path=cache/'processed_hyperpolys/0_0.bin';blob=bytearray(path.read_bytes());blob[-1]=1;path.write_bytes(blob)
            with self.assertRaises(Exception):dataset.load_halos({'synthetic-event':[F(3,2)]})

    def test_full_input_hashes_bound_and_no_output_files_created(self):
        with tempfile.TemporaryDirectory() as directory:
            cache=Path(directory);fixture(cache);before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in cache.rglob('*') if p.is_file()}
            dataset=m.read_event_candidates(cache);dataset.load_halos({'synthetic-event':[F(3,2)]});result=dataset.verify_inputs()
            after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in cache.rglob('*') if p.is_file()}
            self.assertEqual(before,after);self.assertTrue(result['executed_sources_unchanged'])
            self.assertIn(str(cache/'processed_hyperpolys/0_0.bin'),result['input_full_sha256'])


if __name__=='__main__':unittest.main()
