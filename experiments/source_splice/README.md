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

Negative plans cover wrong exact time, a missing raw owner, and an unknown
boundary source VID.  Every negative case must fail closed.

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
