"""Offline finite-pass audits and a pending post-displacement worker contract.

No Blender/native initialization, scene rebuild, displacement, or rendering is
implemented or run here. PASS concerns supplied arrays, not actual kernel input
provenance, a completed selected schedule, or production admission.
"""
from collections import Counter
from fractions import Fraction as F
import hashlib
from itertools import combinations
from pathlib import Path
import sys

import numpy as np

C1_LITE=Path(__file__).resolve().parents[1]/'c1_lite'
if str(C1_LITE) not in sys.path:
    sys.path.insert(0,str(C1_LITE))
from forest_exact_contact import check_triangle_contact
from interface_topology import original_disk
from window_geometry import certify_segment

WORKER_CONTRACT={
    'status':'PENDING_VISIBLE_SELECTION_AND_BOUND_BLENDER_WORKER',
    'version':'POST_DISPLACEMENT_VISIBLE_COMPONENT_PILOT_V1',
    'execution_implemented':False,
    'selection_gate':'All three preregistered segments complete screen or explicit infrastructure STOP, then the protocol qualifying-component rule. No quality-based selection.',
    'schedule':'Selected existing requested natural hit frames, immediate available neighbors, separate exact root; at most 11 unique queries. Caller binds exact absolute frame/global time/local tau; no E2/Forest time constants.',
    'inputs':[
        'The frozen protocol, complete-screen selection receipt, component membership and finite-query pre-displacement certificates.',
        'Native full ordinary baseline and certified full combined element arrays from the same query/cache, with every changed event mapping. No patch crop or forgotten component members.',
        'Build effective_inputs, actual opaque element order, original coarse scene SHA, source assets, saved .npy element parameters and loaded geometry-library SHA set.',
        'Fresh isolated Terrain state with identical seed/gin/surface registry; compare reconstructed parameter arrays and actual loaded libraries to build provenance before evaluating attributes.',
        'Actual per-element kernel queries at every corresponding vertex, including actual new-center positions, with complete returned selection/material fields and legitimate ElementTag. Do not use global SDF argmin ownership or interpolated center attributes.',
        'Saved original modifier parameters and imported SurfaceKernel values; sorted attribute execution order; same float32 conversion and current Mesh.vertex_normals mode in both methods.',
    ],
    'execution_rules':[
        'Use independent writable full-mesh/attribute copies. Element.__call__ temporarily adjusts input Z; do not pass immutable native buffers directly.',
        'Use SurfaceKernel Mesh branch. Dict branch forces vertical normals and discards X/Y offset; it is not a Forest displacement shortcut.',
        'Audit baseline/combined after EACH corresponding surface pass, before a subsequent pass. Compare old positions across methods, not against undisplaced input.',
        'No normal freezing, boundary clamping, attribute averaging, support enlargement or tolerance change to force a pass.',
        'SurfaceKernel offsets may accumulate to binary64 vertex positions. Do not route them through a native-only binary32 admission adapter or silently requantize them.',
        'If BlenderDisplacement/topology-changing modifiers remain unapplied, report partial pipeline. Applying them needs a separately defined identity/geometry contract.',
    ],
    'required_each_pass':['old boundary and all old exterior positions bit-identical across methods',
        'face/tag rows unchanged within each method', 'fixed-XY local fan validity',
        'full retained interface nondegeneracy', 'all replacement-retained contacts', 'all replacement-replacement contacts'],
    'unsupported':'Any identity/attribute/contact mismatch, unbound kernel state or incomplete proof budget leaves the result offline as POST_DISPLACEMENT_UNSUPPORTED; never expand support.',
    'outputs':'Per-pass kernel/query receipts, hashes, counters, concrete first failure; selected finite-schedule summary. No full mesh video, RGB or quality-reference claim.',
    'pending_controls':['centroid requires its own valid plan/certificate', 'schedule-only control not silently invented',
        'independent visual-quality reference remains undefined until frozen before measurement'],
}


class _Unsupported(Exception):
    def __init__(self,reason,witness=None):
        super().__init__(reason);self.witness=witness


def require(value,message,witness=None):
    if not value:raise _Unsupported(message,witness)


def array_hash(array):
    a=np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(a).cast('B') if a.size else b'').hexdigest()


def same(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and array_hash(a)==array_hash(b)


def mesh_hashes(meshes):
    return [{name:array_hash(a) for name,a in zip(('vertices','faces','tags'),mesh)} for mesh in meshes]


def _mesh(mesh):
    require(isinstance(mesh,(tuple,list)) and len(mesh)==3,'Expected element (vertices,faces,tags).')
    v,f,t=mesh
    require(all(isinstance(a,np.ndarray) and a.flags.c_contiguous for a in mesh),'Contiguous actual arrays required.')
    require(v.ndim==2 and v.shape[1]==3 and v.dtype.kind=='f' and np.isfinite(v).all(),'Finite floating vertex positions required.')
    require(f.ndim==2 and f.shape[1]==3 and f.dtype.kind in 'iu'
            and (not f.size or (int(f.min())>=0 and int(f.max())<len(v))),'Invalid actual face indices.')
    require(t.shape==(len(v),) and t.dtype.kind in 'iub','One discrete tag per vertex required.')


def _integer(x):
    require(isinstance(x,(int,np.integer)) and not isinstance(x,(bool,np.bool_)),'Expected integer identity.')
    return int(x)


def _bindings(baseline,combined,mappings):
    require(1<=len(baseline)==len(combined)<=64,'Explicit actual element count required.')
    for mesh in (*baseline,*combined):_mesh(mesh)
    require(isinstance(mappings,dict),'Explicit authorized event mapping required.')
    records=[];used_faces=set();centers=set();used_boundary=set()
    for eid,entry in sorted(mappings.items()):
        require(entry.get('event_id')==eid and isinstance(eid,str),'Event mapping key mismatch.')
        e=_integer(entry['element']);require(0<=e<len(baseline),'Element outside current scene.')
        bv,bf,_=baseline[e];cv,cf,_=combined[e]
        center=_integer(entry['new_center_id']);source=tuple(map(_integer,entry['source_face_rows']))
        rows=tuple(map(_integer,entry['fan_face_rows_by_sector']))
        fan=tuple(tuple(map(_integer,r)) for r in entry['fan_actual_vertex_ids'])
        require(len(source)==2 and len(set(source))==2 and all(0<=i<len(bf) for i in source),'Malformed authorized source faces.')
        require(len(fan)==len(rows)==4 and len(set(rows))==4 and all(0<=i<len(cf) for i in rows)
                and rows[:2]==source and all(i>=len(bf) for i in rows[2:]),'Malformed fan face-row mapping.')
        require(len(bv)<=center<len(cv) and all(len(f)==3 and f[2]==center for f in fan),'Invalid appended center identity.')
        cycle=tuple(f[0] for f in fan)
        require(len(set(cycle))==4 and all(0<=i<len(bv) for i in cycle)
                and fan==tuple((cycle[i],cycle[(i+1)%4],center) for i in range(4)),'Wrong oriented boundary cycle.')
        require((e,center) not in centers,'Center is shared by distinct events.');centers.add((e,center))
        for row in rows:
            require((e,row) not in used_faces,'Two event mappings overwrite one actual face.');used_faces.add((e,row))
        for vertex in cycle:
            require((e,vertex) not in used_boundary,'Joint shared-boundary construction is undefined in this pilot.')
            used_boundary.add((e,vertex))
        require(original_disk(cycle,[tuple(map(int,bf[i])) for i in source])['status']=='PASS','Authorized source faces are not the oriented two-triangle disk.')
        require(np.array_equal(cf[list(rows)].astype(np.int64),np.asarray(fan,dtype=np.int64)),
                'Actual combined fan differs from authorized mapping (no narrow-index wrapping).')
        records.append({'event_id':eid,'element':e,'center':center,'source':source,'rows':rows,'cycle':cycle})
    for e,((bv,bf,bt),(cv,cf,ct)) in enumerate(zip(baseline,combined)):
        local=[r for r in records if r['element']==e]
        require(len(cv)==len(bv)+len(local) and len(cf)==len(bf)+2*len(local),'Unmapped appended center/face or missing element rows.')
        require({r['center'] for r in local}==set(range(len(bv),len(cv))),'Center interval is not completely accounted for.')
        require({i for r in local for i in r['rows'][2:]}==set(range(len(bf),len(cf))),'Appended face interval is not completely accounted for.')
        require(same(bv,cv[:len(bv)]) and same(bt,ct[:len(bt)]),'Pre-displacement old vertex/tag identity already differs.')
        removed={i for r in local for i in r['source']}
        for start in range(0,len(bf),8192):
            ids=np.asarray([i for i in range(start,min(start+8192,len(bf))) if i not in removed],np.int64)
            require(same(bf[ids],cf[ids]),'Unauthorized retained face changed before displacement.')
    return records


def validate_center_attributes(positions,attributes,evaluated_positions,returned_attributes,required_names):
    """Bind fields to supplied query arrays; does not prove a real kernel ran."""
    report={'status':'POST_DISPLACEMENT_UNSUPPORTED','kernel_execution_verified':False}
    try:
        require(isinstance(positions,np.ndarray) and isinstance(evaluated_positions,np.ndarray)
                and positions.ndim==2 and positions.shape[1]==3 and positions.dtype.kind=='f'
                and np.isfinite(positions).all() and same(positions,evaluated_positions),
                'Attribute query positions differ from finite actual vertex inputs.')
        require(required_names and len(set(required_names))==len(required_names)
                and all(isinstance(k,str) and k for k in required_names),'Explicit nonempty required attribute names needed.')
        require(set(required_names)<=set(attributes) and set(required_names)<=set(returned_attributes),'Missing required center/vertex attribute.')
        for name in required_names:
            a,b=attributes[name],returned_attributes[name]
            require(isinstance(a,np.ndarray) and isinstance(b,np.ndarray) and a.ndim>=1 and len(a)==len(positions)
                    and a.dtype.kind in 'fiub' and np.isfinite(a).all() and same(a,b),'Attribute differs from supplied owner-element query output.',{'attribute':name})
        report.update(status='PASS_SUPPLIED_ATTRIBUTE_QUERY_ARRAY_BINDING_ONLY',
            positions_sha256=array_hash(positions),attributes_sha256={k:array_hash(attributes[k]) for k in required_names})
    except (_Unsupported,ValueError,TypeError,KeyError,AttributeError) as error:
        report.update(reason=str(error),witness=getattr(error,'witness',None))
    return report


def _nondegenerate(triangle):
    a,b,c=[[F(float(x)) for x in p] for p in triangle]
    u=[b[i]-a[i] for i in range(3)];v=[c[i]-a[i] for i in range(3)]
    return any(u[(i+1)%3]*v[(i+2)%3]-u[(i+2)%3]*v[(i+1)%3] for i in range(3))


def audit_post_pass(baseline_before,combined_before,baseline_after,combined_after,mappings,
                    *,max_exact_pairs=10000,max_face_aabb_tests=100_000_000):
    """Audit one corresponding Mesh-kernel pass on supplied full element arrays.

    Old boundary/exterior equality is measured between the displaced baseline
    and displaced combined output. Ordinary motion common to both is allowed.
    The support is never expanded when their old positions differ.
    """
    report={'status':'POST_DISPLACEMENT_UNSUPPORTED','scope':'ONE_SUPPLIED_FINITE_PASS_OFFLINE_ARRAY_AND_GEOMETRY_AUDIT',
        'production_admitted':False,'post_displacement_pilot_complete':False,'actual_kernel_execution_verified':False,
        'support_expanded':False,'new_contact_relative_to_baseline_proven':False}
    counts=Counter(face_aabb_tests=0,exact_contact_pairs=0,retained_pairs_aabb_excluded=0,replacement_pairs=0)
    try:
        require(type(max_exact_pairs)is int and max_exact_pairs>0 and type(max_face_aabb_tests)is int and max_face_aabb_tests>0,'Positive explicit proof budgets required.')
        records=_bindings(baseline_before,combined_before,mappings)
        require(len(baseline_after)==len(baseline_before) and len(combined_after)==len(combined_before),'Post-pass element count changed.')
        snapshots=[mesh_hashes(m) for m in (baseline_before,combined_before,baseline_after,combined_after)]
        boundary={e:{v for r in records if r['element']==e for v in r['cycle']} for e in range(len(baseline_before))}
        for e,(bb,cb,ba,ca) in enumerate(zip(baseline_before,combined_before,baseline_after,combined_after)):
            _mesh(ba);_mesh(ca)
            require(bb[0].shape==ba[0].shape and cb[0].shape==ca[0].shape
                    and bb[0].dtype==ba[0].dtype and cb[0].dtype==ca[0].dtype
                    and same(bb[1],ba[1]) and same(cb[1],ca[1]) and same(bb[2],ba[2]) and same(cb[2],ca[2]),
                    'Pass changed within-method topology, tags or vertex count.',{'element':e})
            old=len(bb[0]);require(ba[0].dtype==ca[0].dtype,'Post-method coordinate dtypes differ.')
            if not same(ba[0],ca[0][:old]):
                a,b=ba[0].view(np.uint8).reshape(old,-1),ca[0][:old].view(np.uint8).reshape(old,-1)
                wrong=np.flatnonzero(np.any(a!=b,axis=1));first=int(wrong[0])
                raise _Unsupported('OLD_BOUNDARY_OR_EXTERIOR_POSITION_IDENTITY_LOST',
                    {'element':e,'changed_old_vertices':len(wrong),'changed_boundary_vertices':sum(int(i) in boundary[e] for i in wrong),
                     'first_vertex':first,'first_is_boundary':first in boundary[e],
                     'baseline_position':ba[0][first].tolist(),'combined_position':ca[0][first].tolist(),
                     'former_predisplacement_certificate_transfer':'NOT_APPLICABLE','support_was_not_enlarged':True})
        fans=[];fan_rows={e:set() for e in range(len(combined_after))}
        for record in records:
            e=record['element'];v,f,_=combined_after[e]
            b=v[list(record['cycle'])];c=v[record['center']]
            local=certify_segment(b.tolist(),b.tolist(),c.tolist(),c.tolist())
            require(local['status']=='PASS','POST_FIXED_XY_LOCAL_FAN_UNSUPPORTED',{'event_id':record['event_id'],'local':local})
            for row in record['rows']:
                ids=tuple((e,int(i)) for i in f[row]);tri=v[f[row]]
                require(_nondegenerate(tri),'POST_FAN_DEGENERATE',{'event_id':record['event_id'],'row':row})
                fans.append((e,row,ids,tri));fan_rows[e].add(row)
        def exact(a,b,relation):
            require(counts['exact_contact_pairs']<max_exact_pairs,'EXACT_CONTACT_PROOF_BUDGET_EXHAUSTED')
            proof=check_triangle_contact(a[3],b[3],a[2],b[2]);counts['exact_contact_pairs']+=1
            require(proof['status']=='PASS','POST_CONTACT_FORBIDDEN_OR_UNPROVEN',
                {'relation':relation,'first_element':a[0],'first_face':a[1],'second_element':b[0],'second_face':b[1],
                 'proof':proof,'new_defect_relative_to_displaced_baseline_proven':False})
        for a,b in combinations(fans,2):
            exact(a,b,'REPLACEMENT_REPLACEMENT');counts['replacement_pairs']+=1
        expected_retained=len(fans)*(sum(len(m[1]) for m in combined_after)-len(fans))
        retained_exact=0
        for fan in fans:
            low,high=fan[3].min(axis=0),fan[3].max(axis=0)
            for e,(v,f,_) in enumerate(combined_after):
                for start in range(0,len(f),8192):
                    rows=np.asarray([i for i in range(start,min(start+8192,len(f))) if i not in fan_rows[e]],np.int64)
                    counts['face_aabb_tests']+=len(rows)
                    require(counts['face_aabb_tests']<=max_face_aabb_tests,'FULL_RETAINED_SCAN_PROOF_BUDGET_EXHAUSTED')
                    if not len(rows):continue
                    triangles=v[f[rows]]
                    separated=np.any(triangles.max(axis=1)<low,axis=1)|np.any(triangles.min(axis=1)>high,axis=1)
                    counts['retained_pairs_aabb_excluded']+=int(separated.sum())
                    for row,tri in zip(rows[~separated],triangles[~separated]):
                        ids=tuple((e,int(i)) for i in f[row])
                        if set(ids)&set(fan[2]):
                            require(_nondegenerate(tri),'DISPLACED_BASELINE_RETAINED_INTERFACE_DEGENERATE',
                                {'element':e,'face_row':int(row),'actual_ids':list(map(list,ids)),'coordinates':tri.tolist(),
                                 'not_claimed_new_defect':True})
                        exact(fan,(e,int(row),ids,tri),'REPLACEMENT_RETAINED');retained_exact+=1
        require(counts['retained_pairs_aabb_excluded']+retained_exact==expected_retained,'Incomplete retained contact denominator.')
        require([mesh_hashes(m) for m in (baseline_before,combined_before,baseline_after,combined_after)]==snapshots,'Auditor input mutated during evaluation.')
        report.update(status='PASS_OFFLINE_FINITE_PASS_ARRAY_AND_GEOMETRY_CONTRACT',old_boundary_and_exterior_bit_identical=True,
            within_method_faces_and_tags_bit_identical=True,all_retained_elements_scanned=True,
            replacement_faces=len(fans),expected_retained_triangle_pairs=expected_retained,
            expected_replacement_triangle_pairs=len(fans)*(len(fans)-1)//2,input_array_hashes=snapshots,
            source_owner_identity_scope='Inherited from caller-bound pre-displacement certificates and explicit mappings, not re-established by coordinate differences.',
            interface_incidence_scope='Within-method face IDs/order unchanged; declared original disks and fan correspondence checked. Caller must bind previously certified full source-interface incidence.')
    except (_Unsupported,ValueError,TypeError,KeyError,IndexError,ArithmeticError) as error:
        report.update(reason=str(error),witness=getattr(error,'witness',None))
    report['counts']=dict(counts)
    report['proof_budgets']={'max_exact_pairs':max_exact_pairs,'max_face_aabb_tests':max_face_aabb_tests}
    return report
