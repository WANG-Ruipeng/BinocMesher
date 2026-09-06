# Approved endpoint-order correction, before the second real run

The user approved continuing after the initial endpoint stop. Scope remains
offline; do not change production code or start rendering.

Only change the prospective binary32 center's evaluation order: first round
the source endpoints to binary32, then compute their exact rational midpoint,
then round that midpoint to binary32. Check exact collinearity against those
same rounded endpoints. Do not add a tolerance or a per-event special case.

The binary64 reference center and boundary remain unchanged. Independently
require that the binary64 midpoint remains on its own diagonal; passing the
binary32 gate does not stand in for this check. Report explicitly that interior
height measurements use the binary64 reference, not a C++ runtime execution.

Retain event 00, root 120/11, original window [114/11,126/11], five probes
114/11,117/11,120/11,123/11,126/11, grids 32/64/128, and all existing quality
gates. Do not add natural-frame probes in this rerun. A new failure stops
advancement; do not tune the inputs or relax the quality gate after observing
results. The strong quality gate is a pilot screen, not a necessary condition
for any scientific contribution.

The old census and reference result must remain unchanged. Write the new
reference result to the fresh sibling directory `reference_v2`.

Pre-run verification: all 16 synthetic/regression tests passed. Added fixtures
cover the old lower endpoint's evaluation-order failure and a case where the
binary32 gate passes but the independent binary64 gate fails. The existing
nonrepresentable binary32 midpoint case remains rejected.
