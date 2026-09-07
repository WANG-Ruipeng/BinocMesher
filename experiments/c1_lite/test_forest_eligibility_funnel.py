"""Synthetic bounded-parser and fail-closed tests; no scientific cache needed."""
import copy
from fractions import Fraction as F
from pathlib import Path
import struct
import tempfile
import unittest

import forest_eligibility_funnel as m


def hv(time,position=(0,0,0),view=1,span=0):
    return {'time':time,'position':tuple(map(F,position)),'view':view,'span':span}


def vid(pair):
    return b''.join(struct.pack('<ib3x',*h) for h in pair)


def primary(element,times,polygons):
    return (struct.pack('<bi',element,len(times))+struct.pack('<'+'b'*len(times),*times)
            +b''.join(b''.join(struct.pack('<i',len(face))+b''.join(map(vid,face)) for face in interval)
                       +struct.pack('<i',0) for interval in polygons))


def metadata(index,element=0,group=0,start=0):
    src=[0]*26;src[9]=element
    return struct.pack('<4I3i26i',0x324D5042,2,132,1,group,start,index,*src)


def square():
    pairs=[((10+i,0),(20+i,0)) for i in range(4)]
    positions=[(0,0,0),(1,0,0),(1,1,0),(0,1,0)]
    hvs={h:hv(j+1,p) for pair,p in zip(pairs,positions) for j,h in enumerate(pair)}
    event={'_root':F(3,2),'_levels':(F(5,4),F(3,2),F(7,4)),
           '_raw_boundary':pairs,'_boundary':[m.key(*x) for x in pairs],
           '_star':(F(1,2),F(1,2),F(0)),'element':0,
           'stages':{s:m.verdict('NOT_REACHED','TEST') for s in m.STAGES}}
    return pairs,hvs,event


class SourceTests(unittest.TestCase):
    def test_rational_window_contains_no_integer(self):
        self.assertEqual(m.candidate_levels([1,2,1,2],F(3,2)),(F(5,4),F(3,2),F(7,4)))
        self.assertFalse(m.contains_integer(F(5,4),F(7,4)))
        self.assertTrue(m.contains_integer(F(1),F(7,4)))
        self.assertTrue(m.contains_integer(F(-5,4),F(-3,4)))

    def test_exact_rank(self):
        points=[(F(0),)*4,(F(1),F(0),F(0),F(0)),(F(0),F(1),F(0),F(0)),(F(0),F(0),F(0),F(1))]
        self.assertTrue(m.rank3(points))
        points[-1]=points[1]
        self.assertFalse(m.rank3(points))

    def test_same_view_affine(self):
        a,b=(9,0),(1,0);h={a:hv(0),b:hv(2,(2,0,0))}
        out,p=m.effective((a,b),F(1),h)
        self.assertEqual(out,m.key(a,b));self.assertEqual(p,(F(1),F(0),F(0)))

    def test_native_reverse_time_does_not_swap_original_id(self):
        a,b=(9,0),(1,0);h={a:hv(2,(2,0,0)),b:hv(0)}
        out,p=m.effective((a,b),F(-1),h)
        self.assertEqual(out,(a,a));self.assertEqual(p,(F(0),)*3)

    def test_registry_boundary_time_order_not_id_order(self):
        low,high=(99,0),(1,0);h={low:hv(0,view=0),high:hv(2,(2,0,0),view=1,span=1)}
        pair=m.time_ordered_edge(high,low,h)
        self.assertEqual(pair,(low,high))
        out,p=m.effective(pair,F(3,2),h)
        self.assertEqual(out,(high,high));self.assertEqual(p,(F(2),F(0),F(0)))

    def test_equal_time_registry_edge_unsupported(self):
        with self.assertRaisesRegex(ValueError,'EQUAL_ENDPOINT_TIMES'):
            m.time_ordered_edge((1,0),(2,0),{(1,0):hv(1),(2,0):hv(1)})

    def test_ordered_vid_parser(self):
        pairs=[((9,1),(1,0)),((5,0),(5,0))]
        self.assertEqual(m.parse_vids(b''.join(map(vid,pairs))),pairs)

    def test_whole_replica_source_quotient(self):
        pairs,hvs,event=square()
        polys=[(F(3,2),0,pairs,(0,0,0,index,0,0),(0,)) for index in (7,8)]
        m.finish_event(event,polys,hvs)
        stage=event['stages']['source_owner_closure']
        self.assertEqual(stage['status'],'PASS')
        self.assertEqual(len(stage['owners']),4)
        self.assertEqual(len(stage['source_faces']),2)
        self.assertEqual(event['stages']['candidate_local_geometry']['status'],'PASS')
        self.assertEqual(event['stages']['source_interface']['status'],'REJECT')

    def test_template_mismatch_not_baseline_defect(self):
        _,hvs,event=square();m.finish_event(event,[],hvs)
        stage=event['stages']['source_owner_closure']
        self.assertEqual(stage['status'],'REJECT')
        self.assertIn('not evidence of a baseline defect',stage['scope'])

    def test_unpartitioned_window_is_unknown(self):
        pairs,hvs,event=square();event['_levels']=(F(1),F(3,2),F(7,4))
        m.finish_event(event,[(F(3,2),0,pairs,(0,0,0,7,0,0),(0,))],hvs)
        self.assertEqual(event['stages']['candidate_local_geometry']['status'],'UNKNOWN')
        self.assertNotIn('_owners',event)

    def test_stream_raw_interval_excludes_ended_candidate(self):
        pairs,_,event=square();late=copy.deepcopy(event);late['_root']=F(5,2)
        with tempfile.TemporaryDirectory() as directory:
            cache=Path(directory);processed=cache/'processed_hyperpolys';processed.mkdir()
            data=primary(0,[1,2],[[pairs]])
            (processed/'0_0.bin').write_bytes(data+data)
            (processed/'0_0_hpmeta.bin').write_bytes(metadata(7)+metadata(8))
            reader=m.Reader(10);halo,wanted,scan=m.scan_halo(cache,[event,late],reader)
            self.assertTrue(scan['complete']);self.assertEqual(scan['processed_records_scanned'],2)
            self.assertEqual(len(halo),2);self.assertTrue(all(row[0]==F(3,2) for row in halo))
            self.assertEqual({row[3][3] for row in halo},{7,8})
            self.assertEqual(wanted,set(h for pair in pairs for h in pair))
            self.assertTrue(reader.verify()['pass'])

    def test_stream_complete_collapsed_boundary_halo(self):
        a,b,c=(1,0),(2,0),(3,0)
        event={'_root':F(3,2),'element':0,'_boundary':[(a,a)]}
        with tempfile.TemporaryDirectory() as directory:
            cache=Path(directory);processed=cache/'processed_hyperpolys';processed.mkdir()
            (processed/'0_0.bin').write_bytes(primary(0,[1,2],[[[(a,b),(b,c),(c,a)]]]))
            (processed/'0_0_hpmeta.bin').write_bytes(metadata(0))
            halo,_,scan=m.scan_halo(cache,[event],m.Reader(10))
            self.assertEqual(len(halo),1);self.assertTrue(scan['complete'])

    def test_bad_owner_metadata_rejects(self):
        pairs,_,event=square()
        with tempfile.TemporaryDirectory() as directory:
            cache=Path(directory);processed=cache/'processed_hyperpolys';processed.mkdir()
            (processed/'0_0.bin').write_bytes(primary(0,[1,2],[[pairs]]))
            (processed/'0_0_hpmeta.bin').write_bytes(metadata(7,start=1))
            with self.assertRaisesRegex(ValueError,'BPM2_PREFIX_SCHEMA'):
                m.scan_halo(cache,[event],m.Reader(10))

    def test_selected_range_mutation_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'input.bin';path.write_bytes(b'abc')
            reader=m.Reader(10);reader.take(path,0,3);path.write_bytes(b'abd')
            self.assertFalse(reader.verify()['pass'])


if __name__=='__main__':unittest.main()
