# C1-lite pilot protocol

Scope approved: freeze C0, read-only census, then one isolated reference event.
No production runtime edits, full-scene remeshing, rendering, global schedule
rewrite, or GitHub push in this pilot. Earlier source-splice guard changes stay
separate and are not treated as continuous-geometry fixes.

## Fixed inputs and comparisons

- Existing `full-ubuntu26-smoke` C0 campaign; archive and hash before evaluation.
- Census all four canonical demo events; do not present them as paper scenes.
- Reference event stays index 0, `event-00-6d2d6dc2cdd7`, root `120/11` in internal
  units. Use the original admitted window, not an outcome-tuned wider window.
- Compare ordinary source patch, exact-event-driven fan, and the same-cost fan
  whose center interpolates directly between endpoint diagonal centers.
- Reference fidelity is the original static terrain function. The event fan
  itself is never ground truth. Height error is not Euclidean Hausdorff error.
- No observer-mode toggle is relabeled as a schedule/pairing-only algorithm.
  Raw and `extra_smooth` production comparisons require a later admitted run.

## Gates

1. Preserve original files and record provenance. The currently available
   core binary is not silently claimed to be the historical validation binary.
2. Census the old sidewall mismatch, interval breakpoints, and natural-frame
   coverage with the actual internal-time conversion. Sampled gaps are not
   certified maxima; unproven source/owner stability is UNKNOWN, not PASS.
3. At one fixed event, compare ideal boundary/root/endpoint equality and the
   prospective binary32 implementation contract separately. Matching SourceVID
   names is not a substitute for matching ordinary runtime vertices.
4. Invalid geometry, inconsistent source/owner identity, a broken required
   coordinate contract, or a fidelity regression stops advancement. Report the
   failure and leave production unchanged; do not retune/select another event.
5. A reference-only PASS is not whole-window admission. Full-interval source
   stability, nondegeneracy, embedding and exterior compatibility remain
   prerequisites for any later runtime overlay.

If a natural sample misses the window, report NOT_OBSERVED rather than calling
it a mathematical failure. Do not choose a favorable camera phase afterward.
No claim of a topological saddle repair follows from an embedded fixed-disk
fan family. The project label C1-lite does not mean C1 differentiability.

Storage: C0 snapshot stays in ignored `tmp/c1_lite_c0_freeze_20260905`; compact
JSON and logs only under this experiment. Do not duplicate long render runs.
