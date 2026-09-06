import unittest
from interface_topology import original_disk, audit_interface


def face(ids, element=0):
    return {'element': element, 'source_vertices': ids, 'owners': []}


class InterfaceTests(unittest.TestCase):
    cycle = ['a', 'b', 'c', 'd']
    source = [['a', 'b', 'c'], ['a', 'c', 'd']]

    def collar(self):
        # Outer apex gives one valid retained-link path at each boundary vertex.
        return [face([b, a, 'o']) for a, b in zip(self.cycle, self.cycle[1:]+self.cycle[:1])]

    def test_valid_disk_and_collar(self):
        self.assertEqual(original_disk(self.cycle, self.source)['status'], 'PASS')
        result = audit_interface(self.cycle, self.source, self.collar())
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['runtime_identity_equivalence'], 'UNKNOWN')

    def test_wrong_original_orientation_or_diagonal(self):
        self.assertEqual(original_disk(self.cycle, [self.source[0], ['a', 'd', 'c']])['status'], 'REJECT')
        self.assertEqual(original_disk(self.cycle, [self.source[0], self.source[0]])['status'], 'REJECT')

    def test_missing_extra_or_reversed_attachment(self):
        self.assertEqual(audit_interface(self.cycle, self.source, self.collar()[:-1])['status'], 'REJECT')
        extra = self.collar()+[face(['b', 'a', 'p'])]
        self.assertEqual(audit_interface(self.cycle, self.source, extra)['status'], 'REJECT')
        reverse = self.collar()
        reverse[0] = face(['a', 'b', 'o'])
        self.assertEqual(audit_interface(self.cycle, self.source, reverse)['status'], 'REJECT')

    def test_vertex_bowtie_is_not_hidden_by_edge_counts(self):
        retained = self.collar()+[face(['a', 'p', 'q']), face(['a', 'q', 'r']), face(['a', 'r', 'p'])]
        result = audit_interface(self.cycle, self.source, retained)
        self.assertTrue(all(x['status'] == 'PASS' for x in result['boundary_edge_checks']))
        self.assertEqual(result['status'], 'REJECT')
        self.assertTrue(any(x['kind'] == 'RETAINED_LINK_NOT_ONE_PATH' for x in result['witnesses']))

    def test_inherited_interface_degenerate_rejected_remote_retained(self):
        self.assertEqual(audit_interface(self.cycle, self.source, self.collar()+[face(['x', 'x', 'a'])])['status'], 'REJECT')
        self.assertEqual(audit_interface(self.cycle, self.source, self.collar()+[face(['x', 'x', 'y'])])['status'], 'PASS')

    def test_other_element_cannot_complete_collar(self):
        other = self.collar()
        other[0]['element'] = 1
        self.assertEqual(audit_interface(self.cycle, self.source, other)['status'], 'REJECT')

    def test_retained_cannot_reuse_original_interior_diagonal(self):
        retained = [face(['a', 'c', 'b']), face(['a', 'd', 'c'])]
        result = audit_interface(self.cycle, self.source, retained)
        self.assertEqual(result['status'], 'REJECT')
        self.assertTrue(any(x['kind'] == 'RETAINED_USES_ORIGINAL_INTERNAL_DIAGONAL' for x in result['witnesses']))

    def test_internal_link_orientation_is_checked(self):
        retained = self.collar()[1:]+[face(['b', 'a', 'p']), face(['a', 'o', 'p']), face(['o', 'b', 'p'])]
        self.assertEqual(audit_interface(self.cycle, self.source, retained)['status'], 'PASS')
        retained[-2] = face(['a', 'p', 'o'])
        result = audit_interface(self.cycle, self.source, retained)
        self.assertEqual(result['status'], 'REJECT')
        self.assertTrue(all(x['status'] == 'PASS' for x in result['boundary_edge_checks']))

    def test_duplicate_and_witness_budget(self):
        result = audit_interface(self.cycle, self.source, self.collar()+self.collar(), max_witnesses=1)
        self.assertEqual(result['status'], 'REJECT')
        self.assertGreater(result['witness_count'], 1)
        self.assertEqual(len(result['witnesses']), 1)


if __name__ == '__main__':
    unittest.main()
