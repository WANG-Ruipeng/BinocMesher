"""Fail-closed orchestration contract; not a production mesh emitter.

Evidence producers must supply the gates. Missing or ideal-only evidence can
never authorize a rounded runtime window. Old C0 root-only behavior is not a
fallback for the continuous-window mode.
"""
from fractions import Fraction

REQUIRED_GATES = (
    'source_ownership', 'complete_temporal_partition',
    'original_patch_disk', 'replacement_disk', 'ideal_temporal_continuity',
    'source_incidence_contact', 'interface_topology',
    'interface_degeneracy_policy', 'event_isolation',
    'endpoint_realization', 'compiler_selector_conditioning',
    'raw_emission_equivalence', 'binary32_geometry_contact',
    'runtime_identity_and_unchanged_exterior',
)


def decide_admission(evidence):
    if not isinstance(evidence, dict):
        raise ValueError('Evidence must be a named gate mapping.')
    gates = {}
    for name in REQUIRED_GATES:
        record = evidence.get(name, {'status': 'UNKNOWN', 'reason': 'No evidence supplied.'})
        if not isinstance(record, dict) or record.get('status') not in ('PASS', 'UNKNOWN', 'REJECT'):
            raise ValueError('Malformed gate: '+name)
        gates[name] = dict(record)
    rejected = [name for name, gate in gates.items() if gate['status'] == 'REJECT']
    unresolved = [name for name, gate in gates.items() if gate['status'] == 'UNKNOWN']
    admitted = not rejected and not unresolved
    return {
        'schema': 'c1-lite-baseline-relative-admission-v1',
        'status': 'ADMIT' if admitted else 'FAIL_CLOSED',
        'runtime_plan_authorized': admitted,
        'rejected_gates': rejected, 'unresolved_gates': unresolved, 'gates': gates,
        'fallback': 'UNCHANGED_BASELINE_ENTIRE_WINDOW_INCLUDING_ROOT',
        'fallback_is_a_continuity_improvement': False,
        'root_only_c0_fallback_allowed': False,
        'scope': 'Decision contract only; no runtime plan is emitted by this module.',
    }


def select_action(tau, lower, root, upper, decision):
    """Pure selection regression. Endpoints use the original baseline arrays."""
    tau, lower, root, upper = map(Fraction, (tau, lower, root, upper))
    if not lower < root < upper:
        raise ValueError('Expected an ordered, nonzero-width event window.')
    # Recompute instead of trusting a possibly stale ADMIT flag.
    verified = decide_admission(decision.get('gates', {}))
    if verified['status'] != 'ADMIT' or not lower < tau < upper:
        return 'BASELINE'
    return 'CERTIFIED_WINDOW_ROOT' if tau == root else 'CERTIFIED_WINDOW_INTERIOR'
