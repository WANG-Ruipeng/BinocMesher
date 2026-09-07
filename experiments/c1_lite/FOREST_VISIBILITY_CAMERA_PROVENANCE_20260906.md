# Forest original-camera geometry triage: provenance and limits

This note records read-only camera/environment inspection and the new experimental
CPU raster contract. It does not change the frozen LOD camera, source cache,
production mesher, admission certificates, or original Blender scene. No Forest
image was rasterized while developing this module; its tests use synthetic meshes.

## Original image camera versus the frozen LOD camera

The [frozen Forest64 camera document](artifacts/repair_20260906/forest/stage64/camera_inputs.json)
contains 64 **already OpenCV** camera-to-world poses, 960 x 540 dimensions, and
frame times `(frame_number - 0.5) / 24` (one-based frame numbers). The first and
last times are `1/48` and `2.6458333333333335` seconds. Its focal length
`1423.6682378151224` is the **relaxed LOD** calibration, not the original image
calibration. Existing cache generation and certification continue using it.

The actual WSL Infinigen source is
`/home/warpwang/src/BinocMesher/infinigen_binocmesher/infinigen/terrain/utils/camera.py`:
`get_caminfo` defaults to `relax=1.05`, multiplies the **field of view** by that
factor, converts Blender local axes using `diag(1,-1,-1,1)`, and evaluates the
camera rig for each frame. Scaling focal length directly by 1.05 is incorrect.
The frozen poses must not receive a second Blender/OpenCV axis conversion.

The existing original scene was inspected without saving, rendering, loading UI,
or enabling embedded scripts:

`/home/warpwang/runs/forest96-four-baseline-pilot-20260905/binoc/0/coarse/scene.blend`

It contains the active camera `CameraRigs/0/0`, parent `CameraRigs/0`, with:

| Quantity | Saved value |
| --- | --- |
| Image dimensions and scale | 960 x 540, 100% |
| Frame rate | 24, `fps_base=1` |
| Lens and sensor | 50 mm; 32 x 18 mm; `sensor_fit=AUTO` |
| Shift and pixel aspect | shifts zero; aspect 1:1 |
| Near / far | `0.10000000149011612` / `10000.0` |
| Ideal image pinhole K | `[[1500,0,480],[0,1500,270],[0,0,1]]` |

For **all 64 frames**, saved `camera.matrix_world @ diag(1,-1,-1,1)` equals
the frozen pose numerically with maximum absolute difference **0.0**. Scene
SHA-256 was unchanged after inspection. This validates pose reuse, not reuse of
the relaxed intrinsics. The ideal pinhole K above derives from saved lens/sensor
settings. Blender's float32 projection matrix gives `fx=1500` and
`fy=1499.999942779541`; inverse-FOV recovery from the relaxed K gives
`1500.0000605002188`. These are representation-level differences, not alternative
camera fitting parameters. The CPU triage explicitly uses the ideal K; it does
not claim bit-identical Cycles projection, jitter, antialiasing, or sampling.

[forest.txt](../../infinigen_example_scenes/forest.txt) supplies five rig keyframes,
not the complete evaluated 64-frame trajectory. Do not replace the frozen poses
with linear interpolation of those keys. The placement camera helper's debug
sensor-coordinate samples also use a different pixel offset; the raster instead
declares pixel centers `(x+0.5,y+0.5)`. Synthetic E2 diagnostic cameras are not
used for Forest.

| Provenance item | SHA-256 |
| --- | --- |
| Frozen camera JSON file bytes | `40535546e900f9fc9c6d6efab3e1538da8eb9a88021c9078ebec4697a53f23b7` |
| Frozen camera canonical JSON | `5e986d2e2280e053d5ab4324bc58f4f7a36a3371c87ec75c634b11b929accd1d` |
| Original coarse `scene.blend` (43,310,196 bytes) | `ba63ee3b72e800d67ebf81fa7a35997cb2366a7f3889c4dd243e0614c50902fe` |
| Original `forest.txt` | `78000978c1910ff7b99eb76d8074b9960cd53a93d3bbb23042362ccdfd5a01a7` |
| WSL terrain camera helper | `cd0336dcf3d9645cef667bb967947d9ac4c94bfb5450391d56312bc021dfd3d5` |

Canonical JSON here means UTF-8 `json.dumps(document, sort_keys=True,
separators=(',', ':'), allow_nan=False)`. The adapter in
[forest_raster.py](forest_raster.py) checks that digest before constructing a
**new** image-camera document; it never overwrites the LOD document.

## Raster interface, reproducibility, and budget

`render_scene(meshes, camera_doc, frame_index, event_face_ids=None)` accepts an
ordered sequence of element `(vertices, faces[, tags])` arrays. Tags are ignored:
an `OutOfView` LOD tag is not an occlusion test. It returns camera-Z depth, oriented
world-space flat geometric normals, foreground mask, original element/face IDs,
optional scalar face labels, and cost/count statistics. Background depth/normals
are NaN and IDs are -1. Mesh arrays are never modified.

The Numba CPU kernel is serial float64, `fastmath=False`, `cache=False`; it uses
near/far clipping, perspective-correct depth, no backface culling, a top-left
pixel-coverage rule, and strict depth comparison in element/original-face order.
Exact depth ties retain the first face. A tie winner does not establish that the
other face is strictly occluded. Degenerate triangles are counted and skipped
**for display only**, never repaired or removed from the scientific mesh.
Pixel-work exhaustion raises an exception and returns no partial image.

The installed Miniforge environment already provides NumPy 1.26.4, Numba 0.67.0,
llvmlite 0.49.0, SciPy 1.11.4, Pillow, bpy 3.6.0, PyOpenGL 3.1.0, pyrender 0.1.45,
and trimesh 4.5.3. A Numba JIT smoke and the raster's 20 synthetic tests passed.
Torch, CuPy, pycuda, Open3D, PyTorch3D, nvdiffrast, and moderngl are absent. No
packages were installed and no GPU/OpenGL context or hardware renderer was tested.

At 960 x 540 the six returned arrays occupy 27,475,200 bytes per image, about
55 MB for baseline and combined images together. Transformation/index scratch
is per element; the kernel avoids constructing an all-triangle float64 `v[f]`
array (about 194 MB per copy for 2.7 million faces). Native full-scene arrays are
still owned and budgeted by the caller. Do not persist 64 whole meshes: consume
one verified native query, form the combined output, rasterize both, save compact
summaries/selected diagnostic PNGs, and release arrays. A conservative first-frame
measurement should establish real wall time before extrapolating all frames;
synthetic-test time is not a Forest throughput benchmark. The default raster work
cap is two billion bounding-box pixel tests per call; the outer campaign retains
its independent RSS, wall-time, and output-size caps.

## Visibility and scientific denominators

- Keep the fixed population of **131 events**, including rejected events. A
  missing or unresolved support is unknown, never automatically invisible.
- Baseline source support exists for 130 compiled events; the remaining empty
  fixed-selector event needs a separately labeled broader support proxy. Report
  homogeneous 130-event source-support statistics separately from that proxy.
- Candidate support visibility is not artifact visibility or improvement. Use
  the full five-element opaque scene for occlusion; projected boxes and isolated
  patches alone cannot establish visibility. Zero covered pixel centers is not
  a proof of zero continuous projected area or absence from all camera rays.
- Joint component admission, visible admitted support, visible replacement, and
  changed depth/normal/mask pixels are distinct quantities. One scalar component
  label may represent overlapping event faces; it cannot provide unique event
  attribution without the caller's membership map.
- Separate exact nonzero buffer differences from a predeclared numerical
  threshold. Retriangulation can change final floating-point bits without a
  meaningful geometric image difference. A depth change is not depth-error
  improvement without an independent reference.
- Natural frames use the saved half-frame timestamp convention. Exact-root
  diagnostics are separate samples, not extra natural 24 FPS frames. Do not
  relabel individual-event success as success of a jointly committed mesh.

## Later post-displacement pilot: valid inputs and remaining risks

The reusable Forest cache is **pre-surface-displacement**. The existing stage64
wrapper failure was `KeyError: 'eroded'` during displacement after attributes had
been disabled, not evidence of a finished Infinigen scene. See the saved
[native registry verification](artifacts/repair_20260906/forest/stage64/native_registry_verification.json).

Read-only inspection of the existing WSL Infinigen implementation establishes a
possible input path, but not a completed or certified post-displacement method:

1. `infinigen/terrain/core.py`'s BinocMesher wrapper calls
   `write_attributes(kernels, None, meshes)` on the per-element meshes when
   attribute writing is enabled. `terrain/utils/mesh.py:441-492` evaluates the
   **corresponding element kernel at every vertex position** and supplies its
   non-SDF attributes plus `ElementTag`. Thus a new center can receive defined
   material/selection attributes by evaluating the same owner-element kernel at
   its actual pre-displacement position. Arbitrary copying, zero-filling, or
   interpolating attributes is not equivalent. The merged-mesh global SDF-argmin
   branch is a different path and should not silently replace this behavior.
2. `terrain/core.py:415-438` subsequently applies sampled surface-displacement
   kernels and records Blender-displacement modifiers. The saved sampled choices
   include `eroded -> cracked_ground`; all required attributes must exist before
   invoking these kernels. Re-evaluation cost must be reported and shared fairly
   by baseline, center control, and proposed method.
3. `terrain/surface_kernel/core.py:119-140` converts positions/normals to float32,
   uses the **current mesh vertex normals**, multiplies output by the selected
   vertex attribute, and applies offsets. `terrain/utils/mesh.py:250-270` derives
   vertex normals from mesh faces (default mean weighting). A fan changes the
   boundary's incident faces, so it can change boundary normals and therefore
   displacement of old boundary vertices despite exact pre-displacement position
   identity. Later passes can spread normal-dependent changes through adjacent
   retained faces. Existing pre-displacement interface/exterior certificates do
   not prove post-displacement safety.
4. Saved kernels, pass order, material attributes, normal mode, float32 conversion,
   and any Blender modifiers must be fixed and logged for a versioned pilot.
   SDF perturbations already baked into the meshing query must not be applied
   twice. A post-displacement geometry audit and newly defined comparison domain
   are necessary before interpreting RGB or claiming geometric non-regression.

The inspected WSL `terrain/utils/mesh.py` SHA-256 is
`ea523d03aeed0429f29777d08b80d11d046acda8a3230738764d6213cf531a27`;
`terrain/surface_kernel/core.py` SHA-256 is
`7234a8b77ea9c6f5a9429e65365e40225df26e4897fe02b7d4b7b3367b5aab22`.
Those WSL source paths are provenance references, not claims that the Windows
submodule currently contains identical checked-out files. No new displacement
feature, RGB rendering, environment change, or scene/cache rebuild is part of
this note or the raster implementation.
