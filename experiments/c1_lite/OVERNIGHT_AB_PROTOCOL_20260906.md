# A/B research run, authorized overnight 2026-09-06 (Asia/Shanghai)

Start: approximately 2026-09-05 16:34 UTC. The user's nine-hour return time is
an upper work horizon, not a requirement to consume compute for nine hours.

## Scope

A: source/owner window contracts and exact local geometry certificates, with
known-answer positive/negative tests. B: fixed-window anchor comparisons on
the existing four canonical demo events, retaining negative results. A tiny
raw/extra_smooth runtime baseline diagnostic supports the interpretation.

No general production window integration, global schedule rewrite, full six-
scene benchmark, commit, or push. Small targeted census/render work is allowed
only if A/B and actual applicability justify it. A diagram or diagnostic render
cannot be described as a paper-quality video comparison.

## Fixed B design

- All four existing canonical demo events, sorted by their existing keys.
- Original windows; no widening, rephasing, or event reselection for benefit.
- Each half-window has 17 equally spaced exact rational times: 33 in total.
- XY grids 32/64/128, original independent terrain reference.
- Ordinary source, naive endpoint-chord fan, root-centroid fan, BEB1-root fan;
  root-only C0 is a separate control, not a continuous trajectory.
- Same endpoint anchors, boundary input, one center/four faces for the fans.
  Centroid control does not have to reproduce the BEB1 root anchor.
- Runtime baseline diagnostics explicitly compare extra_smooth=False/True.
- Preserve all negative quality results. Invalid geometry stops the affected
  intervention and is reported, not silently excluded. Dataset-level quality
  is evaluated separately from certificate admission.
- Time averages use the physical-time measure; a root-only modification has
  zero measure and must not acquire artificial duration from quadrature.

This is an explicitly approved extension of the prior single-event preflight.
Old scripts/results and their stronger fail-fast quality rule stay historical.

## A certificate scopes

Distinguish ideal rational trajectories derived from serialized binary32
hypervertices, finite parser-binary64 quality measurements, prospective
binary32 endpoint contracts, and actual C++ outputs. Do not transfer a proof
between these coordinate models without an explicit justification.

Source/owner branch partitions require complete relevant discrete branch
coverage, not five matching samples. Local geometry may use exact polynomial
positivity for the restricted convex-XY graph class. Local embedding is not
patch/exterior compatibility. Unproved claims remain UNKNOWN; proof timeout
or unsupported geometry is not PASS or proof of a geometric failure.

Do not emit production plans for uncertified windows. Keep any root-only C0
fallback separate from a continuous-window claim.

## Storage and provenance

Measured at start (allocated Linux du unless noted):

- Windows repository: 51,437,568 bytes.
- WSL source checkout: 775,598,080 bytes.
- WSL binoc-runs: 40,378,368 bytes.
- WSL render runs: 14,665,302,016 bytes.
- Miniforge: 5,477,117,952 bytes.
- Entire Ubuntu VHDX host file: 31,615,614,976 bytes.
- Migration backup: 8,600,965,120 bytes.

The latter two plus the Windows repository conservatively account for about
40.27 GB of host files (do not add the Linux contents again). User limit:
400,000,000,000 bytes. This run's new-data budget: 2,000,000,000 bytes; stop
new artifact production before either limit is approached. Runtime mutable
cache operations use a small fresh copy, never the frozen original cache.

Keep compact JSON, certificate evidence, summary tables, and only necessary
plots. No full mesh/flow/EXR dump by default. Do not delete environments,
effective baseline runs, or the migration backup. No deletion is needed now.
