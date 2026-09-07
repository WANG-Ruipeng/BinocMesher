"""Bounded preregistered coarse/assets + native opaque PRE-displacement build.

Supervisor only uses stdlib. Blender lives in isolated child processes. This
module never calls terrain.mesh_extraction, fine_terrain, export, or render.
"""
from __future__ import annotations
import argparse
import csv
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

DEFAULT_REPO = Path('/home/warpwang/src/BinocMesher')
GIB = 1024**3


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')


def regular_files(root):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), 'Expected nonsymlink input directory: '+str(root))
    for directory, folders, files in os.walk(root, followlinks=False):
        for name in folders+files:
            require(not (Path(directory)/name).is_symlink(), 'Symlink is unsupported: '+str(Path(directory)/name))
        for name in sorted(files):
            yield Path(directory)/name


def inventory(root):
    root = Path(root)
    return {str(p.relative_to(root)): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in regular_files(root)}


def tree_bytes(root):
    total = 0
    for directory, folders, files in os.walk(root, followlinks=False):
        folders[:] = [n for n in folders if not (Path(directory)/n).is_symlink()]
        for name in files:
            try:
                total += (Path(directory)/name).lstat().st_size
            except FileNotFoundError:
                pass
    return total


def load_selection(protocol_path, segment_id):
    protocol = json.loads(Path(protocol_path).read_text())
    require(protocol['schema'] == 'preregistered-paper-scene-visibility-screen-v1'
            and protocol['registration_state'].startswith('LOCKED_BEFORE_'), 'Protocol is not preregistered.')
    rows = [s for s in protocol['segments'] if s['segment_id'] == segment_id]
    require(len(rows) == 1, 'Unknown/duplicate registered segment.')
    segment, common, budget = rows[0], protocol['common'], protocol['budgets']
    require(segment['last_frame']-segment['first_frame']+1 == common['frames_per_segment'] == 64,
            'Only the preregistered 64-frame profile is supported.')
    require(common['fps'] == segment['fps'] == 24 and common['resolution'] == [960, 540], 'Unexpected camera profile.')
    require([common[k] for k in ('pixels_per_cube','pixels_per_cube_coarse','pixels_per_cube_outview','min_dist','fading_time')]
            == [6,30,120,.1,1] and common['geometry_device'] == 'cpu', 'Unsupported build profile.')
    require(budget['coarse_per_scene_wall_seconds'] == 3600 and budget['cache_per_scene_wall_seconds'] == 2700
            and budget['coarse_per_scene_output_bytes'] == 8*GIB and budget['cache_per_scene_output_bytes'] == 4*GIB
            and budget['build_peak_rss_bytes'] == 16*GIB, 'Unregistered resource profile.')
    overrides = [*common['shared_overrides'], *segment['scene_overrides'],
        'compose_nature.load_cameras='+json.dumps(segment['camera_path']),
        f"execute_tasks.frame_range=[{segment['first_frame']},{segment['last_frame']}]"]
    keys = [x.split('=', 1)[0].strip() for x in overrides]
    require(len(keys) == len(set(keys)), 'Duplicate gin assignment makes the frozen policy ambiguous.')
    return protocol, segment, overrides


def time_mapping(times, fading=1.):
    require(len(times) == 64 and all(math.isfinite(t) for t in times), 'Bad camera times.')
    origin = min(times); shifted = [float(t-origin) for t in times]
    duration = max(shifted)+1e-5
    effective = max(fading, min(b-a for a,b in zip(shifted,shifted[1:])))
    level = 0
    while effective*(1 << (level+1)) < duration:
        level += 1
    maximum = 2 << level
    def rational(value):
        f=Fraction.from_float(value); return {'numerator':f.numerator,'denominator':f.denominator}
    return {'status':'RECONSTRUCTED_FROM_EXECUTED_CAMERA_INPUT_NOT_CPP_SERIALIZED',
        'origin_seconds':rational(origin),'duration_seconds':rational(duration),
        'delta_seconds':rational(duration/maximum),'maximum_discrete_time':maximum,
        'temporal_group_count':1 << level,'natural_frame_formula':'(absolute_frame_number - 0.5) / 24',
        'min_t_offset':0,'use_alignment':False,'effective_fading_seconds':effective,
        'limitation':'Input-value reconstruction, not exact-root or continuous admission.'}


def coarse_command(python, segment, overrides, output):
    return [str(python), '-B', '-m', 'infinigen_examples.generate_nature', '--',
        '--output_folder', str(output), '--seed', str(segment['seed']), '--task', 'coarse',
        '--task_uniqname', 'coarse', '-g', *segment['configs'], '-p', *overrides]


def fresh_path(path, protected):
    p = Path(path).resolve()
    require(not p.exists(), 'Fresh output required: '+str(p))
    for item in protected:
        q = Path(item).resolve()
        require(p != q and not p.is_relative_to(q) and not q.is_relative_to(p), 'Output overlaps protected input: '+str(q))
    return p


def environment(repo, native):
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(('BINOC_SOURCE_SPLICE','BINOC_EVENT_','BINOC_PROVENANCE')):
            del env[key]
    env.update(PYTHONPATH=str(repo/'infinigen_binocmesher'), PYTHONDONTWRITEBYTECODE='1',
        PYTHONUNBUFFERED='1', OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        BINOC_MAX_SERIALIZED_CACHE_RECORDS='100000000', BINOC_MAX_SERIALIZED_CACHE_PAYLOAD_BYTES=str(GIB),
        LD_LIBRARY_PATH='/usr/lib/wsl/lib'+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else ''))
    if native:
        env.update(BINOC_EVENT_MODE='1', BINOC_PROVENANCE_V2='1')
    return env


def group_rss(pgid):
    total = 0
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat=(entry/'stat').read_text(); fields=stat[stat.rfind(')')+2:].split()
            if int(fields[2]) != pgid:
                continue
            for line in (entry/'status').read_text().splitlines():
                if line.startswith('VmRSS:'):
                    total += int(line.split()[1])*1024
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return total


def terminate(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=5)


def monitor(command, cwd, env, directory, wall, bytes_limit, rss_limit):
    started=time.monotonic(); peak_rss=peak_bytes=0; stop=None
    # A monitoring reserve is not an increased quota; the report distinguishes
    # this earlier safety stop from native algorithm failure.
    stop_bytes=bytes_limit-256*1024**2
    with (directory/'worker.log').open('x', encoding='utf-8') as log:
        process=subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                used=tree_bytes(directory); rss=group_rss(process.pid)
                peak_bytes=max(peak_bytes,used); peak_rss=max(peak_rss,rss)
                if time.monotonic()-started > wall: stop='REGISTERED_WALL_LIMIT'
                elif used > stop_bytes: stop='REGISTERED_DISK_SAFETY_RESERVE'
                elif rss > rss_limit: stop='REGISTERED_PROCESS_GROUP_RSS_LIMIT'
                if stop:
                    terminate(process); break
                time.sleep(.25)
        finally:
            terminate(process)
    final_bytes=tree_bytes(directory)
    if final_bytes > bytes_limit:
        stop='REGISTERED_FINAL_OUTPUT_LIMIT'
    return {'exit_code':process.returncode, 'infrastructure_stop':stop,
        'wall_seconds':time.monotonic()-started,'peak_process_group_rss_bytes':peak_rss,
        'peak_monitored_output_bytes':max(peak_bytes,final_bytes),
        'wall_limit_seconds':wall,'output_limit_bytes':bytes_limit,'early_stop_output_bytes':stop_bytes,
        'rss_limit_bytes':rss_limit,'single_file_limit_bytes':GIB,
        'all_children_exit_expected':True,'directory_quota_is_polled_not_filesystem_enforced':True}


def worker(args):
    import resource
    resource.setrlimit(resource.RLIMIT_FSIZE,(GIB,GIB))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    protocol, segment, overrides=load_selection(args.protocol,args.segment)
    repo=args.repo.resolve(); output=args.output.resolve(); infinigen=repo/'infinigen_binocmesher'
    os.chdir(infinigen); sys.path.insert(0,str(infinigen))
    if args.phase == 'coarse':
        command=coarse_command(sys.executable,segment,overrides,output)
        os.execve(sys.executable,command,os.environ.copy())
    source=args.source.resolve()
    before=inventory(source)
    require('scene.blend' in before and 'MaskTag.json' in before and 'assets/info.pickle' in before,
            'Coarse source is missing saved scene, tags or asset metadata.')
    save(output/'source_inputs_before.json',before)
    shutil.copytree(source/'assets',output/'assets')
    import bpy
    import gin
    import numpy as np
    from infinigen_examples import generate_nature  # noqa: F401; registers fixed gin selectors, no main execution
    from infinigen.core import init,surface
    from infinigen.core.placement import camera as cam_util
    from infinigen.core.tagging import tag_system
    from infinigen.core.util.organization import Transparency
    from infinigen.terrain.core import Terrain,UntexturedBinocMesher
    from infinigen.terrain.utils import get_caminfo,Vars
    seed=init.apply_scene_seed(str(segment['seed']))
    mandatory=[Path('infinigen_examples/configs_nature/scene_types')]
    init.apply_gin_configs(configs=['base_nature.gin',*segment['configs']],overrides=overrides,
        config_folders='infinigen_examples/configs_nature',mandatory_folders=mandatory,mutually_exclusive_folders=mandatory)
    bpy.ops.wm.open_mainfile(filepath=str(source/'scene.blend'),load_ui=False,use_scripts=False)
    tag_system.load_tag(path=str(source/'MaskTag.json'))
    scene=bpy.context.scene; fs,fe=segment['first_frame'],segment['last_frame']
    scene.frame_start=fs;scene.frame_end=fe;scene.render.fps=24;scene.render.fps_base=1.
    scene.render.resolution_x=960;scene.render.resolution_y=540;scene.render.resolution_percentage=100
    scene.frame_set(fs);bpy.context.view_layer.update()
    surface.registry.initialize_from_gin();init.configure_blender();cam_util.set_active_camera(0,0)
    cameras=[cam_util.get_camera(i,j) for i,j in cam_util.get_cameras_ids()]
    require(len(cameras)==1, 'Expected original one-camera scene, no substituted view.')
    caminfo=get_caminfo(cameras,fs=fs,fe=fe)[0]
    require(len(caminfo[4])==64 and list(caminfo[4])==[(f-.5)/24 for f in range(fs,fe+1)],'Natural camera schedule differs.')
    camdoc={'camera_ids':[list(x) for x in cam_util.get_cameras_ids()], 'poses':np.asarray(caminfo[0]).tolist(),
        'intrinsics':np.asarray(caminfo[1]).tolist(),'heights':list(caminfo[2]),'widths':list(caminfo[3]),
        'times_seconds':list(caminfo[4]),'time_mapping':time_mapping(caminfo[4]),'get_caminfo_relax':1.05,
        'absolute_frames':list(range(fs,fe+1)),'frame_range':[fs,fe],'segment_id':args.segment}
    save(output/'camera_inputs.json',camdoc)
    actual=cameras[0]; d=actual.data
    require(d.type=='PERSP' and d.sensor_fit in ('AUTO','HORIZONTAL') and
            scene.render.pixel_aspect_x==scene.render.pixel_aspect_y==1 and d.shift_x==d.shift_y==0,
            'Original image calibration lies outside this declared pinhole adapter; do not silently approximate.')
    focal=float(d.lens)/float(d.sensor_width)*960
    image_poses=[]
    for frame in range(fs,fe+1):
        scene.frame_set(frame);bpy.context.view_layer.update()
        image_poses.append((np.asarray(actual.matrix_world,dtype=np.float64)@np.diag([1,-1,-1,1])).tolist())
    scene.frame_set(fs);bpy.context.view_layer.update()
    calibration={'type':d.type,'lens_mm':float(d.lens),'sensor_width_mm':float(d.sensor_width),
        'sensor_height_mm':float(d.sensor_height),'sensor_fit':d.sensor_fit,'shift':[float(d.shift_x),float(d.shift_y)],
        'pixel_aspect':[float(scene.render.pixel_aspect_x),float(scene.render.pixel_aspect_y)],
        'near':float(d.clip_start),'far':float(d.clip_end),'active_camera':actual.name,
        'pose_vs_lod_max_abs_difference':float(np.max(np.abs(np.asarray(image_poses)-np.asarray(caminfo[0])))),
        'calibration':'Ideal unshifted square-pixel horizontal-sensor pinhole; not float32 Cycles jitter equivalence.'}
    image_doc={'schema':'registered-original-image-camera-v1','intrinsics_role':'UNRELAXED_IMAGE_CAMERA',
        'poses':image_poses,'image_intrinsics':[[[focal,0.,480.],[0.,focal,270.],[0.,0.,1.]] for _ in range(64)],
        'widths':[960]*64,'heights':[540]*64,'times_seconds':list(caminfo[4]),'absolute_frames':list(range(fs,fe+1)),
        'near':float(d.clip_start),'far':float(d.clip_end),'pixel_centers':'x+0.5,y+0.5','depth_kind':'CAMERA_Z_NOT_RAY_LENGTH',
        'camera_calibration':calibration,'provenance':{'scene_sha256':before['scene.blend']['sha256'],
            'lod_camera_sha256':sha(output/'camera_inputs.json'),'protocol_sha256':sha(args.protocol),'segment_id':args.segment}}
    save(output/'original_render_camera.json',image_doc)
    # info.pickle is the explicitly selected local coarse artifact, not downloaded input.
    import pickle
    with (output/'assets/info.pickle').open('rb') as handle: info=pickle.load(handle)
    require(Terrain.instance is None,'Unexpected preexisting singleton Terrain state.')
    terrain=Terrain(seed,surface.registry,task=['fine_terrain'],on_the_fly_asset_folder=output/'assets',
        height_offset=info['height_offset'],whole_bbox=info['whole_bbox'])
    terrain.sample_surface_templates();terrain.surfaces_into_sdf()
    opaque=[element for element in terrain.elements_list if element.transparency==Transparency.Opaque]
    require(0<len(opaque)<=64,'Opaque element count exceeds declared native profile.')
    parameter_dir=output/'kernel_parameters';parameter_dir.mkdir()
    element_docs=[]
    for i,element in enumerate(terrain.elements_list):
        record={'index':i,'name':element.__class__.name,'python_class':type(element).__module__+'.'+type(element).__name__,
            'opaque':element in opaque,'attributes':list(element.attributes),'height_offset':float(element.height_offset),
            'whole_bbox':None if element.whole_bbox is None else np.asarray(element.whole_bbox).tolist(),
            'parameters':{},'sdf_surface_kernels':[]}
        for name in ('meta_params','int_params','float_params','int_params2','float_params2','int_params3','float_params3'):
            if hasattr(element,name):
                array=np.asarray(getattr(element,name));require(array.dtype.kind in 'fiub','Non-numeric element parameter.')
                path=parameter_dir/f'{i}_{name}.npy';np.save(path,array,allow_pickle=False)
                record['parameters'][name]={'path':str(path.relative_to(output)),'shape':list(array.shape),'dtype':str(array.dtype),'sha256':sha(path)}
        for j,kernel in enumerate(element.displacement):
            kr={'name':kernel.name,'attribute':kernel.attribute,'use_normal':kernel.use_normal,'use_position':kernel.use_position,'imported_values':{}}
            for dtype,value in kernel.imp_values_of_type.items():
                array=np.asarray(value);path=parameter_dir/f'{i}_sdfsurface_{j}_{str(dtype)}.npy';np.save(path,array,allow_pickle=False)
                kr['imported_values'][str(dtype)]={'path':str(path.relative_to(output)),'shape':list(array.shape),'dtype':str(array.dtype),'sha256':sha(path)}
            record['sdf_surface_kernels'].append(kr)
        element_docs.append(record)
    core_path=Path(sys.modules[UntexturedBinocMesher.__module__].__file__).resolve()
    params={k:protocol['common'][k] for k in ('pixels_per_cube','pixels_per_cube_coarse','pixels_per_cube_outview','min_dist','fading_time')}
    params.update(relax_iters=6 if segment['scene']=='Cave' else 0,use_alignment=False,min_t_offset=0)
    effective={'schema':'registered-opaque-build-effective-v1','segment_id':args.segment,'scene':segment['scene'],'seed':seed,
        'frame_range':[fs,fe],'configs':segment['configs'],'overrides':overrides,
        'terrain_elements':[e.__class__.name for e in terrain.elements_list],
        'opaque_elements':[e.__class__.name for e in opaque],'bounds':list(terrain.bounds),
        'mesher_parameters':params,'mesher_python':str(core_path),'mesher_python_sha256':sha(core_path),
        'core_so':str(core_path.parent/'lib/core.so'),'core_so_sha256':sha(core_path.parent/'lib/core.so'),
        'element_parameters':element_docs,'sampled_surfaces':{k:v.__name__ for k,v in terrain.surfaces.items()},
        'geometry_stage':'PRE_SURFACE_DISPLACEMENT_OPAQUE','surface_displacement_executed':False,
        'occupancy_sdf_perturbations_preserved':True,'rendering_executed':False}
    save(output/'effective_inputs.json',effective)
    calls=[{'calls':0,'positions':0} for _ in opaque]
    def query(i,positions):
        calls[i]['calls']+=1;calls[i]['positions']+=len(positions)
        return opaque[i](positions)[Vars.SDF]
    cache=output/'HyperMesh/OpaqueTerrain'
    mesher=UntexturedBinocMesher(caminfo,terrain.bounds,slicing_time=caminfo[4][0],path=cache,**params)
    meshes,tags=mesher([(lambda positions,i=i:query(i,positions)) for i in range(len(opaque))])
    counts=[{'element':i,'vertices':len(m.vertices),'faces':len(m.faces)} for i,m in enumerate(meshes)]
    del meshes,tags
    required=('slicing_preprocess.finish','slicing_preprocess.manifest.json','event_registry_p1.csv','event_registry_p1_summary.json')
    for name in required:
        require((cache/name).is_file(),'Native completion missing required artifact: '+name)
    require((cache/'hyperpoly_meta').is_dir(),'Native provenance sidecar directory missing.')
    with (cache/'event_registry_p1.csv').open(newline='') as handle:
        rows=list(csv.DictReader(handle))
    registry={'raw_observations':len(rows),'unique_raw_ids':len({r['raw_id'] for r in rows}),
        'canonical_events':len({r['canonical_event_id'] for r in rows}),
        'exact_roots':sorted({str(Fraction(int(r['root_num']),int(r['root_den']))) for r in rows}),
        'classification_or_admission':'NOT_ATTEMPTED'}
    (output/'operative_gin.txt').write_text(gin.operative_config_str(),encoding='utf-8')
    loaded={}
    for line in Path('/proc/self/maps').read_text().splitlines():
        name=line.split()[-1]
        if name.startswith('/') and name.endswith('.so') and ('/terrain/lib/' in name or name==effective['core_so']):
            loaded[name]=sha(name)
    save(output/'loaded_geometry_libraries.json',loaded)
    source_hashes={}
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name and name.endswith('.py') and Path(name).resolve().is_relative_to(infinigen):
            source_hashes[str(Path(name).resolve())]=sha(name)
    source_hashes[str(core_path)]=sha(core_path)
    save(output/'loaded_python_sources.json',source_hashes)
    after=inventory(source);require(before==after,'Original coarse source changed during native build.')
    save(output/'source_inputs_after.json',after)
    save(output/'cache_inventory.json',inventory(cache))
    save(output/'worker_complete.json',{'status':'COMPLETE_PRE_DISPLACEMENT_OPAQUE_CACHE',
        'source_inputs_unchanged':True,'registry':registry,'initial_slice_counts':counts,
        'initial_default_smooth_slice_discarded_not_scientific_treatment':True,
        'occupancy_query_cost':calls,'cache':str(cache),'whole_frame_mesh_saved':False,
        'camera_inputs_sha256':sha(output/'camera_inputs.json'),'effective_inputs_sha256':sha(output/'effective_inputs.json'),
        'original_render_camera_sha256':sha(output/'original_render_camera.json')})


def supervise(args):
    protocol,segment,overrides=load_selection(args.protocol,args.segment)
    repo=args.repo.resolve(); prereg=args.protocol.resolve(); infinigen=repo/'infinigen_binocmesher'
    source=Path(segment['source_scene_root'])/'coarse' if 'source_scene_root' in segment else None
    protected=[repo,prereg.parent]+([source] if source else [])
    output=fresh_path(args.output,protected);reports=fresh_path(args.report_dir,[repo]+([source] if source else []))
    require(output!=reports and not output.is_relative_to(reports) and not reports.is_relative_to(output),'Data/report paths overlap.')
    camera_path=(infinigen/segment['camera_path']).resolve()
    require(camera_path.is_file(),'Missing official camera trajectory.')
    if 'camera_path_sha256' in segment:require(sha(camera_path)==segment['camera_path_sha256'],'Registered camera path hash mismatch.')
    if source:require(sha(source/'scene.blend')==segment['source_scene_sha256'],'Registered saved scene mismatch.')
    output.mkdir(parents=True);reports.mkdir(parents=True)
    snapshot=output/'driver_snapshot.py';shutil.copyfile(Path(__file__).resolve(),snapshot)
    shutil.copyfile(prereg,reports/'preregistration.json')
    initial={'schema':'registered-scene-build-attempt-v1','segment':segment,'overrides':overrides,'output':str(output),
        'repo':str(repo),'protocol_sha256':sha(prereg),'driver_snapshot_sha256':sha(snapshot),
        'camera_path':str(camera_path),'camera_path_sha256':sha(camera_path),'production_algorithm_modified':False,
        'native_only_scope':'Opaque occupancy/cache, no post-surface-displacement, no exported/rendered scene.',
        'resource_reserve_bytes':256*1024**2,'per_file_limit_bytes':GIB}
    save(reports/'build_protocol.json',initial)
    stages={};summary={'status':'STOP_INFRASTRUCTURE_BUILD_INCOMPLETE'}
    try:
        for phase in (('native',) if source else ('coarse','native')):
            directory=output/phase;directory.mkdir()
            command=[sys.executable,'-B',str(snapshot),'--worker','--phase',phase,
                '--protocol',str(prereg),'--segment',args.segment,'--repo',str(repo),'--output',str(directory)]
            if phase=='native':command+=['--source',str(source or output/'coarse')]
            native_before=inventory(source or output/'coarse') if phase=='native' else None
            if native_before is not None:save(reports/'native_source_before.json',native_before)
            prefix='cache' if phase=='native' else 'coarse'
            b=protocol['budgets']
            save(reports/f'{phase}_command.json',{'argv':command,'coarse_exec_argv':coarse_command(sys.executable,segment,overrides,directory) if phase=='coarse' else None})
            result=monitor(command,infinigen,environment(repo,phase=='native'),directory,
                b[f'{prefix}_per_scene_wall_seconds'],b[f'{prefix}_per_scene_output_bytes'],b['build_peak_rss_bytes'])
            stages[phase]=result;save(reports/f'{phase}_process.json',result)
            if native_before is not None:
                native_after=inventory(source or output/'coarse');save(reports/'native_source_after.json',native_after)
                require(native_before==native_after,'Original coarse inputs changed, including on failed attempt.')
            require(result['exit_code']==0 and result['infrastructure_stop'] is None,'Bounded '+phase+' stage did not complete: '+str(result))
            if phase=='coarse':
                for name in ('scene.blend','MaskTag.json','assets/info.pickle'):
                    require((directory/name).is_file(),'Fresh coarse result missing '+name)
                save(reports/'coarse_inventory.json',inventory(directory))
            else:
                complete=json.loads((directory/'worker_complete.json').read_text())
                require(complete['status']=='COMPLETE_PRE_DISPLACEMENT_OPAQUE_CACHE','Native worker lacks PRE-only completion.')
                for name in ('worker_complete.json','camera_inputs.json','original_render_camera.json','effective_inputs.json',
                    'loaded_geometry_libraries.json','loaded_python_sources.json','source_inputs_before.json','source_inputs_after.json','cache_inventory.json'):
                    shutil.copyfile(directory/name,reports/name)
                summary.update(status='COMPLETE_PRE_DISPLACEMENT_OPAQUE_CACHE',native_complete=complete)
        require(sha(prereg)==initial['protocol_sha256'] and sha(snapshot)==initial['driver_snapshot_sha256']
                and sha(camera_path)==initial['camera_path_sha256'],'Protocol, driver snapshot or camera file changed.')
    except Exception as error:
        summary.update(status='STOP_INFRASTRUCTURE_BUILD_INCOMPLETE',reason=type(error).__name__+': '+str(error),traceback=traceback.format_exc())
    summary.update(segment_id=args.segment,stages=stages,output=str(output),build_protocol_sha256=sha(reports/'build_protocol.json'),
        geometry_stage='PRE_SURFACE_DISPLACEMENT_OPAQUE',full_infinigen_or_post_displacement_complete=False,
        no_rendering=True,no_method_admission_attempted=True,failed_build_is_not_zero_events=True,
        data_left_for_readonly_diagnosis=True,cleanup_performed=False)
    save(reports/'summary.json',summary)
    print(json.dumps({k:summary.get(k) for k in ('status','segment_id','reason','output')},sort_keys=True),flush=True)
    return 0 if summary['status']=='COMPLETE_PRE_DISPLACEMENT_OPAQUE_CACHE' else 2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--segment',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--report-dir',type=Path)
    p.add_argument('--repo',type=Path,default=DEFAULT_REPO)
    p.add_argument('--worker',action='store_true');p.add_argument('--phase',choices=('coarse','native'));p.add_argument('--source',type=Path)
    a=p.parse_args()
    if a.worker:
        require(a.phase is not None and (a.phase!='native' or a.source is not None),'Private worker arguments missing.')
        return worker(a)
    require(a.report_dir is not None,'Supervisor requires a fresh report directory.')
    return supervise(a)


if __name__=='__main__':
    raise SystemExit(main())
