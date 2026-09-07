# Forest post-displacement pilot: read-only feasibility

Status: **DESIGN ONLY / NOT RUN / NOT PRODUCTION ADMISSION**.

The existing Infinigen interfaces can assign defined inputs to an added center
without inventing interpolated attributes. They do **not** automatically preserve
the current pre-displacement geometric certificates. Wait for original-camera
visibility before deciding whether to implement a versioned, single-frame pilot.
No scene was rebuilt, displacement executed, package installed, or production
function changed for this analysis.

## Authoritative source and saved inputs

The code links below refer to the existing **WSL** source checkout under
`/home/warpwang/src/BinocMesher/infinigen_binocmesher`. Open them from the WSL
workspace; the Windows submodule may not contain these files. They are inspected
local sources, not links to an assumed upstream implementation.

| Source | SHA-256 |
| --- | --- |
| `infinigen/terrain/core.py` | `84520ef794dc6ea9a2b7d14f6914af9fa8e719ce1106bd9701745cd9f96c4b13` |
| `infinigen/terrain/utils/mesh.py` | `ea523d03aeed0429f29777d08b80d11d046acda8a3230738764d6213cf531a27` |
| `infinigen/terrain/surface_kernel/core.py` | `7234a8b77ea9c6f5a9429e65365e40225df26e4897fe02b7d4b7b3367b5aab22` |

The original `scene.blend`, camera calibration, unchanged scene hash and frame
convention are recorded in the [camera provenance note](FOREST_VISIBILITY_CAMERA_PROVENANCE_20260906.md).
Use the same seed 0 scene, saved assets, gin overrides, and opaque-element order;
do not generate a new scene or build a new LOD cache. The saved
[effective inputs](artifacts/repair_20260906/forest/stage64/effective_inputs.json)
record six scene elements, of which the existing opaque filter gives the five
native mesh namespaces. Do not infer element identity from an arbitrary sorted
mesh filename list.

The [stage64 worker log](artifacts/repair_20260906/forest/stage64/worker_full.log)
records the surface choices: `ground_collection -> mud`,
`mountain_collection -> mountain`, `eroded -> cracked_ground`,
`tree_collection -> sdftree`, and atmosphere's separate surface. CPU libraries
`mud.so`, `mountain.so`, and `cracked_ground.so` already exist. Existence is not
an ABI/parameter-parity or execution-success test; a pilot must bind their actual
hashes and the loaded modifier parameters.

The old stage64 wrapper did not finish displacement: it disabled attributes and
then failed on missing `eroded`. Its native cache is **pre-surface-displacement**;
see the [corrected native verification](artifacts/repair_20260906/forest/stage64/native_registry_verification.json).
The older note in `effective_inputs.json` must not be interpreted as a successful
complete Infinigen pipeline. Nor is today's mutable census driver a byte-exact
replay of the earlier worker; use the recorded driver snapshot/protocol when
reconstructing provenance.

## Defined attribute inputs for the new center

The relevant ordinary path is
[BinocMesher's per-element wrapper](/home/warpwang/src/BinocMesher/infinigen_binocmesher/infinigen/terrain/core.py:76),
not `CollectiveBinocMesher`:

1. The native slicer supplies one mesh and in-view tag vector per opaque element.
2. With attributes enabled, the wrapper calls `write_attributes(kernels, None,
   meshes)` before concatenation.
3. In the [per-element branch](/home/warpwang/src/BinocMesher/infinigen_binocmesher/infinigen/terrain/utils/mesh.py:471),
   `elements[i](meshes[i].vertices)` evaluates the corresponding element at **all
   actual vertex positions**. Every returned field except `Vars.SDF` and
   `Vars.Offset` becomes a vertex attribute; `ElementTag` is added when present.
4. The wrapper sets `OutOfView = ~in_view_tag`, then calls `Mesh.cat`.

Therefore, for a new center in element `i`, valid ordinary-style inputs are that
element's returned fields at the actual rounded center position, with compatible
shape/dtype and the same element tag. Evaluate required attributes on unchanged
vertices too, or reuse them only after proving they were evaluated with the same
kernel state and identical coordinates. Do not copy a neighboring material mask,
zero-fill a missing selection attribute, interpolate categorical tags, or choose
the lowest-SDF element merely because the point lies geometrically nearby.

The [single merged-mesh branch](/home/warpwang/src/BinocMesher/infinigen_binocmesher/infinigen/terrain/utils/mesh.py:443)
evaluates all elements and masks attributes by a global SDF argmin. That is a
different contract. Substituting it changes center ownership/material semantics.

Use independent writable `Mesh`/attribute copies for each treatment: the native
certified arrays remain read-only. [Mesh.cat](/home/warpwang/src/BinocMesher/infinigen_binocmesher/infinigen/terrain/utils/mesh.py:338)
reshapes some input attribute arrays and fills missing fields when combining
elements. Surface kernels also mutate vertices/attributes. Aliasing the baseline
and treatment copies would invalidate the comparison even without touching the
native source cache.

## Parameter source and execution order

- [Terrain initialization](/home/warpwang/src/BinocMesher/infinigen_binocmesher/infinigen/terrain/core.py:224)
  creates the ordered elements from the fixed scene/asset state. Its singleton
  shortcut can reuse earlier state: use an isolated fresh worker, not an
  accidentally pre-initialized `Terrain.instance` from another experiment.
- [Surface sampling](/home/warpwang/src/BinocMesher/infinigen_binocmesher/infinigen/terrain/core.py:455)
  uses a fixed seed derived from `['terrain surface', scene_seed]` and the surface
  registry. Freeze the gin registry and require the resulting attribute-to-surface
  mapping to match the saved scene; do not resample to improve the result.
- [mesh_extraction's surface stage](/home/warpwang/src/BinocMesher/infinigen_binocmesher/infinigen/terrain/core.py:409)
  chooses `OpaqueTerrain_unapplied` if present, otherwise `OpaqueTerrain`, then
  iterates **sorted attribute names**. Each eligible displacement uses that
  saved object's modifier with the selected surface's `mod_name`.
- [SurfaceKernel construction](/home/warpwang/src/BinocMesher/infinigen_binocmesher/infinigen/terrain/surface_kernel/core.py:27)
  kernelizes the modifier into inputs/outputs and imported parameter values,
  loads `terrain/lib/<device>/surfaces/<name>.so`, and constructs its ABI in sorted
  input/type order. The `.so` alone is insufficient provenance: bind selected
  modifier identity, imported values, declared inputs/outputs, device, and source.
- [get_surface_type](/home/warpwang/src/BinocMesher/infinigen_binocmesher/infinigen/terrain/core.py:67)
  can degrade SDF perturbation into displacement. Preserve the saved gin setting.
  [surfaces_into_sdf](/home/warpwang/src/BinocMesher/infinigen_binocmesher/infinigen/terrain/core.py:488)
  registers actual SDF perturbation kernels on elements; do not duplicate this
  registration or apply a perturbation twice to geometry already defined by it.
- [BlenderDisplacement bookkeeping](/home/warpwang/src/BinocMesher/infinigen_binocmesher/infinigen/terrain/core.py:432)
  records additional modifiers separately. [fine_terrain](/home/warpwang/src/BinocMesher/infinigen_binocmesher/infinigen/terrain/core.py:558)
  also exports and copies/saves Blender data. A CPU surface-kernel pilot must not
  be described as the complete final Blender pipeline unless those additional
  operations are explicitly included and audited.

This is a reconstruction plan using existing interfaces, not an instruction to
call `fine_terrain` or `mesh_extraction` now: those larger entry points can mesh,
export, generate assets, or modify Blender state beyond the proposed small pilot.

## Mesh and dict branches are not interchangeable

| Existing branch | Position/normal inputs | Offset behavior | Suitable scope |
| --- | --- | --- | --- |
| [Mesh](/home/warpwang/src/BinocMesher/infinigen_binocmesher/infinigen/terrain/surface_kernel/core.py:121) | Actual positions and current mesh vertex normals, cast to float32 | Selection-weighted full 3D offset added to vertices | The existing Forest surface-displacement path |
| [dict](/home/warpwang/src/BinocMesher/infinigen_binocmesher/infinigen/terrain/surface_kernel/core.py:99) | Supplied positions but forced normal `(0,0,1)` | Selection-weighted offset reduced to Z and returned | A different height-oriented interface, not a Forest shortcut |

Both branches multiply each output by the selected attribute. Missing attributes
are errors, not zero-displacement evidence. Forest is not a global XY height
graph, and E2's height-reference machinery cannot supply a full Forest reference
by merely passing positions through the dict branch.

## Why the current certificates do not transfer

[Mesh defaults](/home/warpwang/src/BinocMesher/infinigen_binocmesher/infinigen/terrain/utils/mesh.py:99)
to mean vertex-normal weighting. Its
[vertex_normals property](/home/warpwang/src/BinocMesher/infinigen_binocmesher/infinigen/terrain/utils/mesh.py:249)
derives normals from the current incident faces. Replacing two triangles by four
changes those incident faces at the old boundary vertices. Consequently:

- Identical old XYZ/SourceVID values before displacement do not imply identical
  normals or identical normal-dependent displacement afterward.
- A later surface pass may see already moved geometry and changed normals;
  effects are not necessarily limited to the four boundary vertices.
- Cropping only the patch changes incident-face normal sums. A small geometric
  crop is not an equivalent full-mesh displacement unless a sufficient dependency
  neighborhood is separately proved for every pass.
- Pre-displacement local embedding, interface topology, exterior separation, and
  retained-row/position identities are not post-displacement certificates. A
  failure to preserve the old identity contract is not automatically a new
  collision, but it does invalidate reuse of that particular contract.

For the first pilot, retaining the full five-element frame in memory is simpler
and less ambiguous than introducing a new truncated-normal method. Do not clamp
boundary displacement, freeze normals, or blend attributes merely to recover
the old certificate: each would define a different method requiring a new design.

## Minimal single-frame pilot, only after a visibility decision

**Predeclare inputs.** Choose one natural frame by an explicit rule based solely
on pre-displacement original-camera visibility, before inspecting any displacement
result. Use its certified full ordinary baseline and existing atomic combined
output, unchanged source cache/LOD, and fixed camera. Identify the selected visible
component ROI without deleting other members from the certified joint output.
If no admitted support is visible, do not silently choose a diagnostic camera or
new scene under the name of this original-camera pilot.

**Minimum comparison.** Run ordinary baseline and the already certified combined
output through the same frozen attribute/surface stages on independent copies.
A centroid control is useful later, but needs its own valid pre-displacement
query plan/certificate; the BEB1 certificate cannot authorize a different center.
Record full-query and added-center attribute query counts, normal/offset timing,
peak RSS, source/library hashes, and all required field shapes/dtypes. Treat
initialization as shared setup only if it truly is shared and immutable.

**Minimum saved outputs.** Save a small versioned manifest, per-pass input/output
array hashes, finite-value checks, counts and displacement magnitudes for old
boundary/exterior/new-center vertices, changed retained-vertex counts, and final
geometry audits. Reuse the original-camera depth/flat-normal/mask convention for
diagnostic images; distinguish changed pixels from reference error reduction.
Store no long whole-mesh sequence. A CPU-kernel-only run must explicitly enumerate
unapplied Blender modifiers and remain labeled partial pipeline.

**Review gates and exits.**

1. Any scene/kernel/parameter mismatch, unresolved owner attribute, missing field,
   unexpected dtype/shape, nonfinite value, source mutation, or budget excess stops
   the pilot without publishing a successful method output. Keep the small log
   and exact reason; do not substitute a different kernel, center, or tolerance.
2. Check baseline and combined geometry after every applied pass. Preserve face/ID
   correspondence in the audit, report changed old vertices, and independently
   test local degeneracy, interface continuity and relevant full-scene contacts.
   Old baseline defects must be distinguished from newly introduced defects.
3. If old exterior/boundary position identity is lost, report the observed domain
   of change and mark the former certificate **not applicable**. A different
   bounded-deformation certificate would be new work, not a repaired report flag.
4. Failure or inconclusive geometry evidence keeps the versioned result offline
   and unsupported; no production wiring or RGB quality claim. Successful finite
   single-frame evidence still does not admit a continuous time window or all
   Forest frames.

Use an explicitly bounded fresh output directory and isolated worker. Keep the
existing campaign's total-storage/RSS protections, and choose a wall-time/output
budget before any execution. No runtime or disk-cost prediction is established
by this read-only analysis. The decision to start this pilot remains pending the
original-camera visibility triage and user/parent approval of the concrete scope.
