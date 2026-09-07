"""Directed synthetic JSON tests; no runtime, cache, or production imports."""
from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path
import tempfile
import unittest

import assemble_forest_sequence_manifest as m
import compare_forest_sequences as c
from test_compare_forest_sequences import fixture


def population():
    events=[]; components=[]; per_root={}
    profiles=[('3/2',3,6,24,6),('5/2',2,9,23,6)]
    for root,ap,asol,rp,rsol in profiles:
        group=[]
        for admitted,sizes in [(True,[2]*ap+[1]*asol),(False,[2]*rp+[1]*rsol)]:
            for count in sizes:
                ids=[]
                for _ in range(count):
                    eid=f'e{len(events):03d}';ids.append(eid)
                    events.append({'event_id':eid,'root':root,
                        'native_individual_decision':'ADMITTED_REQUESTED_SCHEDULE' if admitted else 'REJECTED_FIXED_POLICY',
                        'window':{'lower':str(F(root)-F(1,4)),'upper':str(F(root)+F(1,4))}})
                group.append({'component_id':'component-'+m.digest(ids)[:16],'root':root,'events':ids,
                    'independently_rejected_members':[] if admitted else ids,
                    'decision':'ADMITTED_COMPONENT_REQUESTED_SCHEDULE' if admitted else 'FAIL_CLOSED_REJECTED_MEMBER'})
        components.extend(group)
        per_root[root]={'events':sum(len(x['events']) for x in group),'components':len(group),
            'admitted_components':ap+asol,'jointly_admitted_events':2*ap+asol}
    inventory={'status':'PASS_COMPLETE_FIXED_SOURCE_CANDIDATE_INVENTORY',
        'full_input_hash_set_equal_frozen':True,'event_count':131,'events':events}
    graph={'status':'COMPLETE_CONSERVATIVE_REQUESTED_SUPPORT_GRAPH','event_count':131,
        'events_sha256':m.digest(sorted(e['event_id'] for e in events)),'components':components}
    cert={'per_root':per_root,'canonical_event_denominator':131,'component_denominator':79,
        'admitted_components':20,'jointly_admitted_events':25,'query_count':18}
    return inventory,graph,cert


def frames_and_queries():
    inv,graph,cert=population(); events,admitted,_=m.check_population(inv,graph,cert)
    _,base=fixture(); frames=[]; queries={}
    for item in base:
        q=item['query'];tau=F(q['evaluation_tau'])
        selected={e:cid for e,cid in admitted.items() if F(events[e]['window']['lower'])<tau<F(events[e]['window']['upper'])}
        records=[{'event_id':e,'component_id':cid,'plan':{'example':e}} for e,cid in sorted(selected.items())]
        frame={'query':q,'records':records,'baseline':['synthetic_receipt'],
            'root_support_replay_sha256':None,'union':{'output':['synthetic_receipt']}}
        if records:
            root=events[next(iter(selected))]['root']
            rows=[{'event_id':e,'status':'COMPLETE_ACTUAL_REQUESTED_SUPPORT','full_interface_star_enumerated':True}
                  for e in events if events[e]['root']==root]
            queries[q['key']]={'query':deepcopy(q),'root':root,'events':rows,'baseline':frame['baseline'],
                'certified_independent_plans':deepcopy(records)}
            frame['root_support_replay_sha256']=m.digest(rows)
        frames.append(frame)
    return frames,queries,events,admitted


class ManifestTests(unittest.TestCase):
    def test_fixed_population_is_131_not_only_25(self):
        events,admitted,roots=m.check_population(*population())
        self.assertEqual((len(events),len(admitted)),(131,25))
        self.assertEqual(roots['3/2']['jointly_admitted_events'],12)

    def test_missing_rejected_node_cannot_leave_same_denominator(self):
        inv,graph,cert=population();inv['events'].pop()
        with self.assertRaisesRegex(c.InvalidSequence,'DENOMINATOR'):
            m.check_population(inv,graph,cert)

    def test_rejected_component_cannot_be_relabelled_admitted(self):
        inv,graph,cert=population()
        next(x for x in graph['components'] if x['independently_rejected_members'])['decision']='ADMITTED_COMPONENT_REQUESTED_SCHEDULE'
        with self.assertRaisesRegex(c.InvalidSequence,'REJECTED_MEMBER_WAS_ADMITTED'):
            m.check_population(inv,graph,cert)

    def test_fixed_schedule_counts_are_recomputed(self):
        result=m.check_frame_links(*frames_and_queries())
        self.assertEqual(result['natural_event_frame_replacements'],200)
        self.assertEqual(result['unchanged_natural_frames'],48)
        self.assertEqual(result['complete_root_support_replay_queries'],18)

    def test_partial_component_membership_is_not_a_full_schedule(self):
        frames,queries,events,admitted=frames_and_queries();frames[20]['records'].pop()
        with self.assertRaisesRegex(c.InvalidSequence,'FULL_MEMBERSHIP'):
            m.check_frame_links(frames,queries,events,admitted)

    def test_changed_root_support_replay_is_not_accepted(self):
        frames,queries,events,admitted=frames_and_queries();frames[20]['root_support_replay_sha256']='a'*64
        with self.assertRaisesRegex(c.InvalidSequence,'ROOT_SUPPORT_REPLAY_BINDING'):
            m.check_frame_links(frames,queries,events,admitted)

    def test_plan_changed_since_certification_is_not_accepted(self):
        frames,queries,events,admitted=frames_and_queries();frames[20]['records'][0]['plan']['example']='changed'
        with self.assertRaisesRegex(c.InvalidSequence,'FRESH_PLAN_CHANGED'):
            m.check_frame_links(frames,queries,events,admitted)

    def test_outside_mesh_receipt_must_equal_baseline(self):
        frames,queries,events,admitted=frames_and_queries();frames[0]['union']['output']=['changed']
        with self.assertRaisesRegex(c.InvalidSequence,'OUTSIDE_BASELINE_CHANGED'):
            m.check_frame_links(frames,queries,events,admitted)

    def test_comparison_pass_string_without_exact_audited_bindings_fails(self):
        a,af=fixture('1');b,bf=fixture('8');replay=c.compare_sequences(a,af,b,bf)
        bindings={str(i):'a'*64 for i in range(134)}
        report={**replay,'audited_files_sha256':deepcopy(bindings),'input_artifacts_unchanged':True,
            'executed_script_sha256':m.sha(c.__file__)}
        m.check_comparison(report,replay,bindings)
        report['audited_files_sha256']['0']='b'*64
        with self.assertRaisesRegex(c.InvalidSequence,'134_FILES'):
            m.check_comparison(report,replay,bindings)

    def test_child_path_cannot_escape_evidence_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(c.InvalidSequence,'ESCAPES'):
                m.child(Path(directory),{'path':'../other.json','sha256':'a'*64})


if __name__=='__main__':unittest.main()
