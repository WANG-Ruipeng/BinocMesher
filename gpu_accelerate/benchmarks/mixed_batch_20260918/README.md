# Historical offline mixed-batch evidence

This is the complete small export of the sealed 256-arm distinct-task study,
not the older 224-arm replicated-size study. All eight verdicts, 32 A/A
controls, 256 arm statuses/times and full frozen schedule are included.
SUMMARY.json distinguishes historical performance from new integration smoke.

EVIDENCE_MANIFEST.json identifies the unedited originals separately from the
derived public files. Original data and binaries remain outside Git; the
manifest uses relative file names under three owner-supplied external roots.
PROTOCOL.json is byte-identical to original PLAN_SCHEDULE.json.
REFERENCE_HASHES.json is byte-identical to the original ten-task eleven-array
manifest. It contains hashes/lengths only, never scene output arrays.

The CPU-only exporter rechecks the entire original seal and all quartet ratios.
Supply the original evidence, backend assets and original build records
explicitly, with a new output directory:

    python3 -B export_history.py \
      --external-evidence-root /path/to/sealed-distinct-task-evidence \
      --external-backend-root /path/to/historical-backend-evidence \
      --external-build-root /path/to/historical-build-evidence \
      --output /path/to/new-public-export

The exporter loads no CUDA library. Obtaining the nonpublic originals is an
owner-mediated step; scene assets and third-party trees are not downloaded by
this package. No benchmark rerun or new performance claim follows from export.
