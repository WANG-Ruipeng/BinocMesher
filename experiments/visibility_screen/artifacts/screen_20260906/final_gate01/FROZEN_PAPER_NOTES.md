# Frozen evidence notes

Status: `DRAFT_FOR_PAPER_NOT_FINAL_METHOD_OR_RESULTS`

These notes freeze only the established problem framing, implemented contract,
and completed Forest A/Mountain measurements. They are not a title, abstract,
teaser, expanded theorem, or final campaign conclusion.

## Problem

A temporal mesher can encounter exact shared events whose local treatment must
remain compatible with the ordinary mesh's source ownership and retained
interface. A locally plausible replacement is insufficient: duplicate source
occurrences, native vertex identities, neighboring retained triangles, and
simultaneous event dependencies also constrain whether that replacement can be
committed. The research problem is therefore to define and validate a conservative
partial replacement transaction, including a precise baseline-preserving outcome
when the declared construction cannot be certified. Event prevalence, certified
mesh modification, and camera-visible impact are distinct measurements; none
substitutes for the next.

## Implemented finite-schedule contract

The current pipeline retains the complete canonical registry denominator and
compiles source support under a fixed selector policy. Exact event roots and
source-cell descriptions are kept separate from the actual floating-point query
times and rounded native coordinates. Native provenance resolves source-owner
classes and effective SourceVIDs; coordinate coincidence is not an identity
oracle. Source-ready is an intermediate state, not runtime admission. See the
[source adapter](../../../scene_source.py) and
[native per-query checks](../../../../c1_lite/forest_native_patch.py).

For the supported two-face replacement domain, the pipeline evaluates a
BEB1-anchored window fan and checks the actual queried geometry, source interface,
and replacement/retained contacts under the frozen policy. A conservative
same-root support graph includes all events, including rejected events, and
records owner, face, boundary/interface, retained-star, and spatial dependencies.
Missing support is not an empty reservation. Candidate-only support is explicitly
bounded to the current selector's candidate domain, not a hypothetical future
closure. Graph edges need not be collisions, and components need not be minimal.
See the [support-graph implementation](../../../../c1_lite/forest_component_graph.py).

Component admission is all-or-none: independently admissible members may fall
back with rejected peers or an unsupported joint configuration. Each requested
time starts from a fresh ordinary baseline. Accepted plans are combined once,
with canonical identities and single consumption of selected owner classes;
retained data and fallback source faces are checked for preservation. A complete
requested-sequence receipt is published only after the required checks and input
bindings succeed. See the
[Forest transaction](../../../../c1_lite/run_forest_atomic_sequence.py) and
[multi-scene driver](../../../run_screen.py).

The defensible claim is **conditional and baseline-relative over the finite
queried schedule**: successful transactions satisfy the recorded source,
interface, contact, and preservation gates within this supported domain. This
does not establish arbitrary-time admission, a globally defect-free baseline,
complete coverage of all event types, or a general shared-boundary solver.
Existing global defects are not silently repaired. Ideal rational source models
are not substituted for actual binary32 runtime evidence. No post-displacement
guarantee follows.

## Completed coverage and negative visibility results

Both completed segments use 64 natural frames, 24 FPS, 960×540, 6 px LOD, and
pre-surface-displacement opaque geometry. Forest A is retained earlier evidence;
Mountain is one of the three newly preregistered segments.

| Segment | Joint events / all canonical | Joint components / all | Modified natural frames | Event-frame replacements |
| --- | ---: | ---: | ---: | ---: |
| Forest A | 25/131 (19.08%) | 20/79 | 16/64 | 200 |
| Mountain | 33/46 (71.74%) | 30/35 | 16/64 | 264 |

Each has two exact roots, with nonempty admitted subsets at both roots. OMP 1/8
agree on all 64 natural outputs plus two separate exact-root diagnostics and
their bound transaction/support records. These are joint-admission fractions,
not image-quality scores or full-root-event coverage. Evidence:
[Forest A structural manifest](../../../../c1_lite/artifacts/forest_sequence_20260906/structural_coherence.json),
[Mountain confirmed summary](../mountain/screen03_omp8/combined_summary.json).

Original-camera geometric visibility tested all candidates: 1,048 event/frame
cases for Forest A and 368 for Mountain. Both measured zero visible candidate
source events, zero visible admitted events, and zero replacement pixels. The
visible-support admission ratio is undefined (`null`, 0/0), not 0%. Across all
16 modified natural frames, baseline/combined depth, flat-normal, mask,
element-ID, face-ID, and component-ID buffer hashes agree. This is nonempty-scene,
pinhole pixel-center geometric triage, not RGB, SSIM, perceptual improvement, or
post-displacement validation. Mountain's 366 no-isolated-pixel cases are not all
proven out of frustum. See the
[Forest visibility evidence](../../../../c1_lite/artifacts/forest_sequence_20260906/visibility01/summary.json)
and [Mountain buffer audit](../mountain/visibility_buffer_audit01.json).

Cave and Forest B final retry outcomes remain unmeasured here. Their prior
resource stops are unknown scientific outcomes, not zero events or visibility.
No conclusion about all three new segments, temporal smoothing, popping
reduction, or improved natural-frame images is currently frozen.

中文范围：仅冻结已完成的有限查询、相对 baseline 的条件性结构证据及两段可见性负结果；
不提前解释未完成片段，不扩定理，不声称位移后或画质通过。
