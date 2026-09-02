# Production source-face suppression and exact boundary gluing

This experiment extends the read-only TV0--TV4 work with a generic runtime seam
for replacing one production-derived source patch.  It does not add a
shape-specific event handler.

The runtime plan identifies every raw source triangle owner by

```text
(element, t_group, t_start, sorted_record_index,
 interval_index, face_index, fan_index)
```

and requires every owner to be consumed and suppressed exactly once.  Boundary
vertices are not spatially welded: each replacement boundary label is an exact
ordinary source-edge `VID`, resolved only after the ordinary global merge.

Two positive plans are used on the same real event-star patch:

1. **Identity round trip.** Suppress all raw owners and re-emit the same two
   oriented triangles.  The complete final mesh arrays must be byte-exact.
2. **Shared-edge subdivision.** Insert one point on the old shared source edge
   and replace two triangles by four.  The PL surface and all external boundary
   edges are unchanged; the four boundary vertices must reuse pre-existing
   ordinary global IDs.

Negative plans cover wrong exact time, a missing raw owner, an unknown
boundary source VID, and a replacement whose internal-edge orientations do
not cancel.  Every negative case must fail closed.

Run only this experiment:

```bash
bash experiments/source_splice/run_source_splice.sh \
  --repo "$PWD" --output /tmp/binoc-source-splice
```

Run the authoritative engineering baseline, TV0--TV4, and this experiment:

```bash
bash experiments/source_splice/run_full_validation.sh \
  --repo "$PWD" --output /tmp/binoc-source-splice-full
```

Expected final marker:

```text
PASS_CERTIFIED_SOURCE_SPLICE_FULL_VALIDATION
```

The event-free identity/subdivision campaign remains as a regression seam.  The
full wrapper additionally compiles and runs the production-selected critical
event at exact time `104/5`.  It does not claim production endpoint blocks, a
natural mixed batch, or an arbitrary global gluing theorem.

## Critical BEB1 admission compiler

After TV0--TV4 has produced a fresh cache and theory directory, compile the
closed event star and its SSP1 plan with:

```bash
python3 experiments/source_splice/compile_critical_beb1_event_ir.py \
  --cache-root /tmp/tv0-tv4/cache \
  --theory-root /tmp/tv0-tv4/theory \
  --output /tmp/critical-beb1-event-ir.json \
  --plan-output /tmp/critical-beb1.ssp1 \
  --expected-root 104/5 \
  --require-whole-mesh-ready
```

The compiler independently ties the selected registry batch to the TV3
two-tetrahedron relative half-handle.  It then:

1. completes the critical link disk to the four-tetrahedron core sphere;
2. proves that the four former critical side edges are internal;
3. extracts the same closed ordinary source patch at lower/root/upper times;
4. freezes its four SourceVID trajectories as the prescribed affine side trace
   `S_B`;
5. builds an explicit double mapping cylinder with 15 spacetime vertices and
   24 tetrahedra from the lower disk through the critical four-face fan to the
   upper disk;
6. audits positive 4D Gram volumes, relative 3-manifold incidence, a
   16-triangle side wall disjoint from all three disk centers, and an internal
   root critical vertex; and
7. emits SSP1 only when every boundary, geometry, and mapping-cylinder audit
   passes.

Successful admission prints:

```text
PASS_CRITICAL_BEB1_EVENT_IR
READY_FOR_WHOLE_MESH_SPLICE
```

The full wrapper next runs the emitted plan at exact `104/5` with OMP 1 and 8.
It requires exact-once source suppression, four reused ordinary boundary IDs,
one internal critical vertex, four replacement faces, unchanged outside
oriented faces and global topology, and no new external intersection partner.
Only then does it print:

```text
PASS_CRITICAL_BEB1_EVENT_STAR_CLOSURE
PASS_CRITICAL_BEB1_WHOLE_MESH_SPLICE
```

The original relative half-handle and its four unresolved side edges remain in
the Event IR as an explicit before-closure witness; they are not exposed as the
replacement boundary.  The full production closure is globally a
disk-to-disk local retriangulation: the relative seed's two-component-to-one
transition is not promoted to a whole-mesh topology-change claim.  SSP1 also
rejects malformed plans whose boundary
contains internal vertices, whose internal orientations do not cancel, or
whose edge/face incidence is nonmanifold.

The small exact-combinatorics regression does not need a production cache:

```bash
python3 experiments/source_splice/test_critical_beb1_event_ir.py
```

Expected marker:

```text
PASS_CRITICAL_BEB1_EVENT_STAR_CLOSURE_COMBINATORICS
```
