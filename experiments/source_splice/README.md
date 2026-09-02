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

Scope: this validates a real event-star source patch at an exact event-free
rational probe.  The next step is to feed a certified TV3/BEB1 critical-time
slice through the same source-boundary interface.  This experiment does not yet
claim production endpoint blocks, a natural mixed batch, or an arbitrary global
gluing theorem.

## Critical BEB1 admission compiler

The first critical-time integration stage is intentionally fail-closed.  After
TV0--TV4 has produced a fresh cache and theory directory, run:

```bash
python3 experiments/source_splice/compile_critical_beb1_event_ir.py \
  --cache-root /tmp/tv0-tv4/cache \
  --theory-root /tmp/tv0-tv4/theory \
  --output /tmp/critical-beb1-event-ir.json \
  --expected-root 104/5
```

The compiler independently ties the selected registry batch to the TV3
two-tetrahedron block, reconstructs exact lower/critical/upper slices, and
labels every block-slice vertex by source provenance.  The current TV3 object
is a relative half-handle, so its critical side seams are reported explicitly:

```text
PASS_CRITICAL_BEB1_EVENT_IR
SINGULAR_UNRESOLVED_SIDE_TRACE
```

This is not a whole-mesh success marker.  Passing
`--require-whole-mesh-ready` makes the command fail until event-star closure
cancels every internal side seam and all remaining patch boundary edges are
ordinary SourceVID trajectories.  SSP1 also rejects malformed replacement
plans whose boundary contains internal vertices, whose internal orientations
do not cancel, or whose edge/face incidence is nonmanifold.

For the current 104/5 witness the labelled half-handle has four unresolved
critical side edges in each of K-minus, K-zero, and K-plus.  Its direct
source-edge boundary agrees with the production face trace on the lower probe
but not on the upper probe.  This is machine-recorded evidence that TV3 is
still a relative half-handle and must not be emitted as a finished global
patch.

The small exact-combinatorics regression does not need a production cache:

```bash
python3 experiments/source_splice/test_critical_beb1_event_ir.py
```
