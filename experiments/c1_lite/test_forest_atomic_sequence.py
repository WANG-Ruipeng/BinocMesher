import unittest
from pathlib import Path
from unittest.mock import patch

from run_forest_atomic_sequence import load_component_certificate


class AtomicSequenceInputTests(unittest.TestCase):
    def test_path_objects_join_query_artifact_paths(self):
        root = Path('/unit-component').resolve()
        rows = [{'path': f'queries/{i}.json', 'sha256': 'hash'} for i in range(18)]
        summary = {'status': 'PASS_FOREST_COMPONENT_REQUESTED_CERTIFICATION',
            'final_input_verification': {'status': 'PASS'}, 'private_cache_removed': True,
            'support_graph': {'path': 'support_graph.json', 'sha256': 'hash'},
            'query_artifacts': rows, 'input_bindings_sha256': {}, 'executed_sources_sha256': {},
            'query_count': 18}
        documents = {root/'summary.json': summary, root/'support_graph.json': {'components': []}}
        documents.update({root/r['path']: {'query': {'key': str(i)}} for i, r in enumerate(rows)})
        with patch('run_forest_atomic_sequence.read', side_effect=lambda p: documents[Path(p)]), \
             patch('run_forest_atomic_sequence.file_sha', return_value='hash'):
            result, graph, queries, bindings = load_component_certificate(root)
        self.assertEqual(len(queries), 18)
        self.assertEqual(result, summary)
        self.assertEqual(len(bindings), 20)


if __name__ == '__main__':
    unittest.main()
