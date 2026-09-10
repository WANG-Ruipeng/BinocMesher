# Stage 0: historical identity audit

Status: **UNRECOVERED_WITHIN_BOUNDED_SEARCH** (2026-09-07).
Parameter center definition: **UNRESOLVED**.
Historical 120-case metrics: **UNVERIFIED_REPORT_ONLY**.

Read-only search covered the current E:/BinocMesher source/documentation and
relevant filenames (excluding Infinigen and bulky binary data); all 23 reachable
Git commits, including messages, historical filenames and graph-lift/star-polygon
text changes; the WSL source copy /home/warpwang/src/BinocMesher at ae81991;
directory listings through depth 3 and candidate filenames through depth 4 under
/home/warpwang/src, /home/warpwang/runs and /home/warpwang/binoc-runs.

No executable Graph-Lifted Star Polygonization, original 120-case generator or
per-case metric data was recovered. This is not proof of absence elsewhere.
An existing cleanup script was inspected read-only; its explicit targets were
Forest render intermediates, not an identifiable historical GL experiment.

The attachment's formula is compatible with both parameter vertex mean and
parameter area centroid. It does not establish a separate optimized point rule.
Current experiments/c1_lite/reference.py:center_at interpolates three anchors;
anchor_ablation.py:candidate_centers uses the XYZ boundary mean before that
interpolation. Neither establishes the historical GL point-selection rule.

This probe implements two explicitly NEW model-sampling variants:
`rebuilt_lifted_vertex_mean` and `rebuilt_lifted_area_centroid`.
Neither is labeled historical/original GL or `ours`. No extra unexplained GL
strategy is compared. The two definitions coincide on a square; the asymmetric
cropped hexagon distinguishes them. Agreement does not recover old identity.
