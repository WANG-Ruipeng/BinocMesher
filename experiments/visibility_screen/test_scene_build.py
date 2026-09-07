import json
from pathlib import Path
import tempfile
import unittest
from scene_build import load_selection,time_mapping,coarse_command,fresh_path,environment

PROTOCOL=Path(__file__).with_name('protocol_20260906.json')


class SceneBuildTests(unittest.TestCase):
    def test_all_frozen_segments(self):
        expected={'forest_b':(0,97,160),'cave':(1,1,64),'mountain':(1,1,64)}
        for name,values in expected.items():
            p,s,o=load_selection(PROTOCOL,name)
            self.assertEqual((s['seed'],s['first_frame'],s['last_frame']),values)
            self.assertIn('BinocMesher.pixels_per_cube=6',o)
            self.assertIn('Terrain.device="cpu"',o)

    def test_unknown_segment_is_not_replaced(self):
        with self.assertRaises(ValueError):load_selection(PROTOCOL,'better_scene')

    def test_cave_keeps_official_relaxation(self):
        p,s,o=load_selection(PROTOCOL,'cave')
        self.assertIn('BinocMesher.relax_iters=6',o)
        self.assertIn("surface.registry.ground_collection=[('mountain',1)]",o)

    def test_mountain_asset_seed_not_scene_seed(self):
        p,s,o=load_selection(PROTOCOL,'mountain')
        self.assertEqual(s['seed'],1)
        self.assertIn('LandTiles.assets_seed_override=64090700628785987370221272729015798609',o)

    def test_coarse_only_no_pipeline_or_render(self):
        p,s,o=load_selection(PROTOCOL,'cave');cmd=coarse_command('/python',s,o,Path('/fresh/coarse'))
        self.assertEqual(cmd[cmd.index('--task')+1],'coarse')
        self.assertNotIn('infinigen.datagen.manage_jobs',cmd)
        self.assertNotIn('render',cmd);self.assertNotIn('fine_terrain',cmd)

    def test_same_length_new_absolute_time_origin(self):
        first=time_mapping([(f-.5)/24 for f in range(1,65)])
        later=time_mapping([(f-.5)/24 for f in range(97,161)])
        for value in (first,later):
            self.assertEqual(value['maximum_discrete_time'],4)
            self.assertEqual(value['temporal_group_count'],2)
        self.assertNotEqual(first['origin_seconds'],later['origin_seconds'])

    def test_reject_non_64_times(self):
        with self.assertRaises(ValueError):time_mapping([0.,1.])

    def test_duplicate_override_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=json.loads(PROTOCOL.read_text());p['common']['shared_overrides'].append('BinocMesher.pixels_per_cube=3')
            target=Path(tmp)/'p.json';target.write_text(json.dumps(p))
            with self.assertRaises(ValueError):load_selection(target,'forest_b')

    def test_changed_budget_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=json.loads(PROTOCOL.read_text());p['budgets']['cache_per_scene_wall_seconds']=99999
            target=Path(tmp)/'p.json';target.write_text(json.dumps(p))
            with self.assertRaises(ValueError):load_selection(target,'forest_b')

    def test_fresh_directory_and_protected_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir()
            self.assertEqual(fresh_path(root/'new',[source]),root/'new')
            for target in (source,source/'child',root):
                with self.assertRaises(ValueError):fresh_path(target,[source])

    def test_native_environment_enables_sidecars_not_splice(self):
        env=environment(Path('/repo'),True)
        self.assertEqual(env['BINOC_EVENT_MODE'],'1');self.assertEqual(env['BINOC_PROVENANCE_V2'],'1')
        self.assertFalse(any(k.startswith('BINOC_SOURCE_SPLICE') for k in env))

    def test_coarse_environment_not_registry_run(self):
        env=environment(Path('/repo'),False)
        self.assertNotIn('BINOC_EVENT_MODE',env)


if __name__=='__main__':unittest.main()
