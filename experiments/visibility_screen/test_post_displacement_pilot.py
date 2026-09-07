import copy
import unittest
import numpy as np
from post_displacement_pilot import audit_post_pass,validate_center_attributes,WORKER_CONTRACT


def fixture(remote=True):
    v=np.array([[0.,0.,0.],[2.,0.,0.],[2.,2.,0.],[0.,2.,0.], [10.,10.,0.],[11.,10.,0.],[10.,11.,0.]])
    f=np.array([[0,1,2],[0,2,3]]+([[4,5,6]] if remote else []),np.int32)
    t=np.ones(len(v),np.int32);center=len(v)
    cv=np.concatenate((v,[[1.,1.,.2]]));ct=np.append(t,np.int32(1))
    fan=np.array([[0,1,center],[1,2,center],[2,3,center],[3,0,center]],np.int32)
    cf=np.concatenate((f,fan[2:]));cf[:2]=fan[:2]
    mapping={'e0':{'event_id':'e0','component_id':'c0','element':0,'new_center_id':center,
        'source_face_rows':[0,1],'fan_face_rows_by_sector':[0,1,len(f),len(f)+1],'fan_actual_vertex_ids':fan.tolist()}}
    return [(v,f,t)],[(cv,cf,ct)],mapping


def shifted(meshes,delta):
    return [(v+delta,f.copy(),t.copy()) for v,f,t in meshes]


class PostDisplacementTests(unittest.TestCase):
    def test_identity_pass_still_not_production(self):
        b,c,m=fixture();r=audit_post_pass(b,c,b,c,m)
        self.assertTrue(r['status'].startswith('PASS'),r)
        self.assertFalse(r['production_admitted']);self.assertFalse(r['actual_kernel_execution_verified'])

    def test_common_displacement_is_allowed(self):
        b,c,m=fixture();r=audit_post_pass(b,c,shifted(b,.1),shifted(c,.1),m)
        self.assertTrue(r['status'].startswith('PASS'),r)

    def test_boundary_change_stops_without_expanding_support(self):
        b,c,m=fixture();after=shifted(c,0.);after[0][0][0,2]=.1
        r=audit_post_pass(b,c,b,after,m)
        self.assertEqual(r['reason'],'OLD_BOUNDARY_OR_EXTERIOR_POSITION_IDENTITY_LOST')
        self.assertTrue(r['witness']['first_is_boundary']);self.assertFalse(r['support_expanded'])

    def test_remote_old_vertex_change_rejected(self):
        b,c,m=fixture();after=shifted(c,0.);after[0][0][5,2]=.1
        r=audit_post_pass(b,c,b,after,m)
        self.assertEqual(r['witness']['first_vertex'],5);self.assertFalse(r['witness']['first_is_boundary'])

    def test_signed_zero_is_bitwise_identity_failure(self):
        b,c,m=fixture();after=shifted(c,0.);after[0][0][0,2]=-0.
        self.assertEqual(audit_post_pass(b,c,b,after,m)['reason'],'OLD_BOUNDARY_OR_EXTERIOR_POSITION_IDENTITY_LOST')

    def test_face_topology_change_rejected(self):
        b,c,m=fixture();after=shifted(c,0.);after[0][1][0]=after[0][1][1]
        self.assertEqual(audit_post_pass(b,c,b,after,m)['status'],'POST_DISPLACEMENT_UNSUPPORTED')

    def test_tag_change_rejected(self):
        b,c,m=fixture();after=shifted(c,0.);after[0][2][0]=0
        self.assertEqual(audit_post_pass(b,c,b,after,m)['status'],'POST_DISPLACEMENT_UNSUPPORTED')

    def test_unmapped_remote_face_change_rejected(self):
        b,c,m=fixture();c[0][1][2]=[4,6,5]
        self.assertIn('Unauthorized retained face',audit_post_pass(b,c,b,c,m)['reason'])

    def test_mapping_cannot_authorize_wrong_fan(self):
        b,c,m=fixture();m['e0']['fan_actual_vertex_ids'][0]=[1,0,7]
        self.assertEqual(audit_post_pass(b,c,b,c,m)['status'],'POST_DISPLACEMENT_UNSUPPORTED')

    def test_center_can_have_actual_binary64_position(self):
        b,c,m=fixture();after=shifted(c,0.);after[0][0][-1,2]=.1234567890123
        self.assertTrue(audit_post_pass(b,c,b,after,m)['status'].startswith('PASS'))

    def test_center_on_edge_is_local_failure(self):
        b,c,m=fixture();after=shifted(c,0.);after[0][0][-1]=[1.,0.,0.]
        self.assertEqual(audit_post_pass(b,c,b,after,m)['reason'],'POST_FIXED_XY_LOCAL_FAN_UNSUPPORTED')

    def test_no_changes_no_replacement(self):
        b,c,m=fixture();r=audit_post_pass(b,b,b,b,{})
        self.assertTrue(r['status'].startswith('PASS'));self.assertEqual(r['replacement_faces'],0)

    def test_exact_budget_is_not_contact_rejection(self):
        b,c,m=fixture();r=audit_post_pass(b,c,b,c,m,max_exact_pairs=1)
        self.assertEqual(r['reason'],'EXACT_CONTACT_PROOF_BUDGET_EXHAUSTED')

    def test_scan_budget_no_partial_pass(self):
        b,c,m=fixture();r=audit_post_pass(b,c,b,c,m,max_face_aabb_tests=1)
        self.assertEqual(r['reason'],'FULL_RETAINED_SCAN_PROOF_BUDGET_EXHAUSTED')

    def test_other_element_collision_not_ignored(self):
        b,c,m=fixture()
        other=(np.array([[.5,.5,.1],[1.5,.5,.1],[1.,1.5,.1]]),np.array([[0,1,2]],np.int32),np.ones(3,np.int32))
        b.append(other);c.append(other)
        r=audit_post_pass(b,c,b,c,m)
        self.assertEqual(r['reason'],'POST_CONTACT_FORBIDDEN_OR_UNPROVEN')
        self.assertFalse(r['witness']['new_defect_relative_to_displaced_baseline_proven'])

    def test_nonfinite_post_position_is_unsupported(self):
        b,c,m=fixture();after=shifted(c,0.);after[0][0][-1,2]=np.nan
        self.assertEqual(audit_post_pass(b,c,b,after,m)['status'],'POST_DISPLACEMENT_UNSUPPORTED')

    def test_attribute_query_coordinates_must_match(self):
        p=np.array([[1.,2.,3.]]);a={'eroded':np.array([.5],np.float32)}
        self.assertEqual(validate_center_attributes(p,a,p+1,a,['eroded'])['status'],'POST_DISPLACEMENT_UNSUPPORTED')

    def test_interpolated_attribute_not_equal_owner_query(self):
        p=np.array([[1.,2.,3.]]);a={'eroded':np.array([.5],np.float32)};b={'eroded':np.array([.6],np.float32)}
        self.assertEqual(validate_center_attributes(p,a,p,b,['eroded'])['status'],'POST_DISPLACEMENT_UNSUPPORTED')

    def test_attribute_binding_not_actual_kernel_execution(self):
        p=np.array([[1.,2.,3.]]);a={'eroded':np.array([.5],np.float32)}
        r=validate_center_attributes(p,a,p,a,['eroded'])
        self.assertTrue(r['status'].startswith('PASS'));self.assertFalse(r['kernel_execution_verified'])

    def test_empty_element_is_preserved(self):
        b,c,m=fixture();empty=(np.empty((0,3),np.float64),np.empty((0,3),np.int32),np.empty(0,np.int32))
        b.append(empty);c.append(empty)
        self.assertTrue(audit_post_pass(b,c,b,c,m)['status'].startswith('PASS'))

    def test_two_face_source_without_retained_faces(self):
        b,c,m=fixture(remote=False)
        self.assertTrue(audit_post_pass(b,c,b,c,m)['status'].startswith('PASS'))

    def test_silent_position_quantization_rejected(self):
        b,c,m=fixture();ba=[(v.astype(np.float32),f,t) for v,f,t in b];ca=[(v.astype(np.float32),f,t) for v,f,t in c]
        self.assertEqual(audit_post_pass(b,c,ba,ca,m)['status'],'POST_DISPLACEMENT_UNSUPPORTED')

    def test_nonfinite_attribute_query_rejected(self):
        p=np.array([[np.nan,2.,3.]]);a={'eroded':np.array([.5],np.float32)}
        self.assertEqual(validate_center_attributes(p,a,p,a,['eroded'])['status'],'POST_DISPLACEMENT_UNSUPPORTED')

    def test_no_execution_entrypoint_claim(self):
        self.assertFalse(WORKER_CONTRACT['execution_implemented'])
        self.assertIn('PENDING',WORKER_CONTRACT['status'])


if __name__=='__main__':unittest.main()
