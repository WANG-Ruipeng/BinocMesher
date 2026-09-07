"""Synthetic source-domain tests only; no cache or native library."""
from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import forest_source_support_inventory as m


class VID:
    def __init__(self,i):self.i=i
    def text(self):return f'{self.i}:0|{self.i+100}:0'
    def __hash__(self):return hash(self.i)
    def __eq__(self,other):return isinstance(other,VID) and self.i==other.i


class Ref:
    def __init__(self,i):self.element=0;self.row=(0,0,0,0,0,i,0)
    def values(self):return self.row


class Snapshot(NS):
    def summary(self):return {'event_candidates_complete':self.event_candidates_complete,'halo_complete':self.halo_complete}


def fixture(compiled=True):
    labels=[VID(i).text() for i in range(6)]
    shapes=[(0,1,2),(0,2,3),(0,1,2),(1,0,4),(2,4,5)]
    triangles=[NS(reference=Ref(i),source_vertices=tuple(VID(v) for v in shape),event_record=i!=3)
               for i,shape in enumerate(shapes)]
    provenance=(0,)*26
    details={t.reference.values():{'record_key':(0,0,0),'provenance_values':provenance,
        'ordered_raw_vids':tuple(((v.i,0),(v.i+100,0)) for v in t.source_vertices),
        'actual_raw_emitted':i!=4,'native_legacy_identity_equal':True}
        for i,t in enumerate(triangles)}
    record=NS(element=0,times=(1,2),provenance_values=provenance,
        primary_path='/cache/p.bin',primary_offset=12,primary_size=128,primary_sha256='a'*64,
        metadata_path='/cache/p_hpmeta.bin',metadata_offset=0,metadata_sha256='b'*64)
    dataset=NS(records={(0,0,0):record},cache=Path('/cache'))
    event={'event_id':'synthetic','element':0,'root':'3/2',
        'schedule':{'bounds':{'lower':'5/4','root':'3/2','upper':'7/4'}},
        'kernel':{'critical_position':[1,1,0]},'compiler':{}}
    corners=[(0,0,0),(2,0,0),(2,2,0),(0,2,0)]
    payload={'boundary_cycle':labels[:4],'source_faces':[[labels[v] for v in shapes[i]] for i in (0,1)],
             'owners':[list(triangles[i].reference.values()) for i in (0,1,2)]}
    source={'partition_proof':{'all_thresholds_integer':True},'segments':[],
        'breakpoint_points':[],'anchors':{k:{'position':[m.fj(x) for x in (1,1,0)]} for k in ('lower','root','upper')}}
    cells=m.cells_for_event(event);snapshots={}
    for c in cells:
        tau=m.frac(c['representative'])
        snapshots[event['event_id'],tau]=Snapshot(event_id=event['event_id'],tau=tau,raw_triangles=tuple(triangles),
            actual_raw_triangles=tuple(triangles[:4]),details_by_owner=deepcopy(details),
            halo_vertex_ids=frozenset(VID(i) for i in range(6)),event_candidates_complete=True,halo_complete=True)
        if c['kind']=='singleton':
            source['breakpoint_points'].append({**deepcopy(payload),'time':c['t0'],
                'boundary':[{'source_vid':v,'position':[m.fj(x) for x in p]} for v,p in zip(labels,corners)]})
        else:
            source['segments'].append({**deepcopy(payload),'t0':c['t0'],'t1':c['t1'],
                'boundary':[{'source_vid':v,'position_t0':[m.fj(x) for x in p],
                    'position_t1':[m.fj(x) for x in p]} for v,p in zip(labels,corners)]})
    event['compiler']={'source':source if compiled else None,
        'reason':'LEGACY_EXHAUSTIVE_SELECTOR_NO_CANDIDATE: synthetic'}
    native={'decision':'REJECTED_FIXED_POLICY','runtime':{'status':'NOT_ATTEMPTED'}}
    return event,native,snapshots,dataset


class InventoryTests(unittest.TestCase):
    def build(self,data):return m.build_event(*data,{'source_artifact':{'sha256':'c'*64}})

    def test_all_five_cells_full_classes_and_replicas(self):
        data=fixture();r=self.build(data)
        self.assertEqual(r['source_status'],'COMPLETE_FIXED_COMPILED_SOURCE_MODEL')
        self.assertEqual(len(r['cells']),5);self.assertEqual(len(r['face_class_catalog']),4)
        self.assertEqual(len(r['owner_catalog']),5);self.assertEqual(len(r['record_catalog']),1)
        self.assertEqual(r['union']['source_owner_ids'],[0,1,2])
        for c in r['cells']:
            self.assertEqual(len(c['source_face_ids']),2)
            self.assertEqual(len(c['actual_halo_face_ids']),3)
            self.assertEqual(len(c['legacy_halo_face_ids']),4)
            self.assertEqual(len(c['retained_face_ids']),1)
            self.assertEqual(c['candidate_actual_owner_ids'],[0,1,2])
            self.assertEqual(c['candidate_legacy_owner_ids'],[0,1,2,4])
        selected=[r['face_class_catalog'][i] for i in r['cells'][0]['source_face_ids']]
        self.assertEqual(sorted(len(x['owner_ids']) for x in selected),[1,2])

    def test_no_candidate_patch_is_nonempty_domain_not_empty_source(self):
        r=self.build(fixture(False))
        self.assertEqual(r['source_status'],'CANDIDATE_DOMAIN_ONLY')
        self.assertIsNone(r['union']['source_owner_ids']);self.assertIsNone(r['ideal_affine_patch_aabb'])
        self.assertTrue(r['union']['candidate_vids'])
        for c in r['cells']:
            self.assertIsNone(c['source_face_ids']);self.assertIsNone(c['source_owner_ids'])
            self.assertEqual(c['retained_face_ids'],c['actual_halo_face_ids'])
            self.assertTrue(c['candidate_vertex_stars_legacy'])

    def test_partial_owner_class_is_never_suppressed(self):
        data=fixture()
        for row in data[0]['compiler']['source']['segments']+data[0]['compiler']['source']['breakpoint_points']:
            row['owners']=row['owners'][:2]
        with self.assertRaisesRegex(ValueError,'FULL_OWNER_CLASS'):self.build(data)

    def test_missing_halo_or_source_boundary_is_not_complete(self):
        data=fixture();next(iter(data[2].values())).halo_complete=False
        with self.assertRaisesRegex(ValueError,'INCOMPLETE'):self.build(data)
        data=fixture();next(iter(data[2].values())).halo_vertex_ids=frozenset([VID(0)])
        with self.assertRaisesRegex(ValueError,'SOURCE_BOUNDARY_OUTSIDE'):self.build(data)

    def test_whole_window_integer_partition_gate(self):
        event=fixture()[0];event['schedule']['bounds']['lower']='3/4'
        with self.assertRaisesRegex(ValueError,'UNPARTITIONED_INTEGER'):m.cells_for_event(event)

    def test_ideal_aabb_includes_center_and_never_excludes_actual32(self):
        source=fixture()[0]['compiler']['source'];source['anchors']['root']['position'][2]=m.fj(F(7,3))
        bounds=m.ideal_bounds(source)
        self.assertEqual(m.frac(bounds['maximum'][2]),F(7,3))
        self.assertTrue(bounds['not_safe_for_actual32_exclusion'])
        self.assertEqual(bounds['actual_binary32_rounding_envelope'],'NOT_PROVIDED')

    def test_degenerate_retained_corners_not_silently_dropped(self):
        catalogue=[{'vids':['0/a','0/a','0/b']}]
        self.assertEqual(m.stars([0],catalogue,['0/a']),{'0/a':[0]})
        links=m.incidence_links([0],catalogue,['0/a'])
        self.assertEqual(len(links['0/a']),2)
        self.assertEqual([r[3] for r in links['0/a']],[0,1])

    def test_scoped_identity_and_oriented_faces(self):
        self.assertNotEqual(m.scoped_vid(0,'1:0|2:0'),m.scoped_vid(1,'1:0|2:0'))
        self.assertEqual(m.face_key(0,['a','b','c']),m.face_key(0,['b','c','a']))
        self.assertNotEqual(m.face_key(0,['a','b','c']),m.face_key(0,['a','c','b']))

    def test_same_owner_serialized_pair_change_stops(self):
        data=fixture();snapshot=list(data[2].values())[1]
        detail=next(iter(snapshot.details_by_owner.values()))
        detail['ordered_raw_vids']=(((99,0),(100,0)),)*3
        with self.assertRaisesRegex(ValueError,'MULTIPLE_SERIALIZED'):self.build(data)

    def test_candidate_actual_owner_api_uses_exact_cell_and_rejects_outside(self):
        r=self.build(fixture(False))
        for tau in (F(5,4),F(13,10),F(3,2),F(8,5),F(7,4)):
            self.assertEqual(m.candidate_actual_owners(r,tau),tuple(Ref(i).values() for i in (0,1,2)))
        with self.assertRaisesRegex(ValueError,'OUTSIDE'):m.candidate_actual_owners(r,F(2))

    def test_symbolic_edges_include_rejected_candidate_domain_and_all_cells(self):
        left=self.build(fixture());right=self.build(fixture(False));right['event_id']='other'
        edges=m.source_symbolic_edges([left,right])
        self.assertTrue(edges)
        self.assertEqual({(e['cell']['kind'],e['cell']['t0'],e['cell']['t1']) for e in edges},
            {(c['kind'],str(m.frac(c['t0'])),str(m.frac(c['t1']))) for c in left['cells']})
        self.assertTrue(any(e['reason']=='SOURCE_CELL_SOURCE_RETAINED_INTERFACE_DEPENDENCY' for e in edges))
        self.assertTrue(all('CANDIDATE_DOMAIN_ONLY' in e['cell']['source_statuses'] for e in edges))

    def test_symbolic_edges_ignore_space_bounds_and_never_cross_roots(self):
        left=self.build(fixture());right=deepcopy(left);right['event_id']='other'
        before=m.source_symbolic_edges([left,right])
        left['ideal_affine_patch_aabb']={'invalid':'must not be read'}
        self.assertEqual(before,m.source_symbolic_edges([left,right]))
        right['root']='5/2';self.assertEqual(m.source_symbolic_edges([left,right]),[])


if __name__=='__main__':unittest.main()
