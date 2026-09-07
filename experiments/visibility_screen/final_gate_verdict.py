"""Classify the final fixed-population visibility evidence, never select images.

Pure JSON classification. Resource stops and missing measurements are not zeros;
the pre-registered pixel/frame qualification cannot be weakened by this report.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from coverage_report import (read_bound, require, file_sha, _stage_stop,
                             scene_row, verify_confirmed_summaries)
from final_gate_amendment import load_amendment


def classify(rows):
    rows = [r for r in rows if r['segment_id'] != 'forest_a_previous']
    require(len(rows) == 3 and {r['segment_id'] for r in rows} == {'cave','forest_b','mountain'},
            'FIXED_THREE_SEGMENT_POPULATION_REQUIRED')
    complete = [r for r in rows if r['screen_complete']]
    for row in complete:
        require(type(row.get('unknown_support_visibility_events')) is int and row['unknown_support_visibility_events'] >= 0,
                'COMPLETE_SCREEN_UNKNOWN_VISIBILITY_COUNT_REQUIRED')
    incomplete = [r['segment_id'] for r in rows if not r['screen_complete']]
    terminal = all(r['screen_complete'] or r.get('terminal_infrastructure_stop') for r in rows)
    unknown = [r['segment_id'] for r in complete if r.get('unknown_support_visibility_events', 0)]
    selected = [r['qualified_components'][0] for r in complete if r.get('qualified_components')]
    selected.sort(key=lambda r: (r['segment_id'],r['earliest_qualifying_absolute_frame'],r['component_id']))
    visible = [r['segment_id'] for r in complete if r['visible_candidate_events'] > 0]
    visible_replacements = [r['segment_id'] for r in complete if r['visible_replacement_events'] > 0]
    if not terminal:
        decision = 'WAIT_FIXED_SECOND_ATTEMPTS_NOT_ALL_TERMINAL'
    elif selected:
        decision = 'GO_FIRST_PREREGISTERED_QUALIFYING_COMPONENT'
    elif incomplete or unknown:
        decision = 'RESOURCE_OR_EVIDENCE_UNRESOLVED_STOP_VISIBILITY_CAMPAIGN'
    elif not visible and not visible_replacements:
        decision = 'STOP_ORIGINAL_CAMERA_VISUAL_HEADLINE'
    elif visible and not visible_replacements:
        decision = 'VISIBLE_CANDIDATES_WITHOUT_VISIBLE_REPLACEMENT_FIXED_DIAGNOSIS_ONLY'
    else:
        decision = 'VISIBLE_REPLACEMENT_BELOW_PREREGISTERED_PILOT_QUALIFICATION'
    return {'decision': decision, 'all_fixed_segments_terminal': bool(terminal),
        'completed_new_segments': sorted(r['segment_id'] for r in complete),
        'uncompleted_new_segments': sorted(incomplete), 'unknown_visibility_segments': sorted(unknown),
        'measured_visible_candidate_segments': sorted(visible),
        'measured_visible_replacement_segments': sorted(visible_replacements),
        'first_qualifying_component': selected[0] if terminal and selected else None,
        'ranking_scope': 'Frozen rank among completed segments; stopped/unmeasured segments remain unknown.' if incomplete else
                         'Frozen rank among all three completed preregistered segments.',
        'post_displacement_or_attribute_gate_still_required': True,
        'quality_reference_and_controls_still_required': True,
        'visual_improvement_established': False, 'stop_method': False,
        'new_camera_seed_segment_lod_or_budget_search_authorized': False,
        'automatic_method_extension_authorized': False}


def _resolved_ref(ref, base):
    path = Path(ref['path'])
    if not path.is_absolute():
        path = Path(base)/path
    return {'path': str(path.resolve()), 'sha256': ref['sha256']}


def validate_retry_build(summary, contract, segment, amendment):
    """Terminal second-build policy only, not successful source/screen evidence."""
    expected = {'path': amendment['amendment_path'], 'sha256': amendment['amendment_sha256']}
    require(summary['segment_id'] == segment and summary.get('budget_amendment') == expected
            and contract.get('budget_amendment') == expected, 'RETRY_NOT_BOUND_TO_FINAL_APPROVED_BUDGET')
    require(summary['status'] in ('COMPLETE_PRE_DISPLACEMENT_OPAQUE_CACHE',
                'STOP_INFRASTRUCTURE_BUILD_INCOMPLETE', 'STOP_INPUT_INTEGRITY_FAILED'),
            'SECOND_BUILD_NOT_TERMINAL')
    require(Path(summary['output']).name == segment+'_attempt02'
            and contract['output'] == summary['output'], 'NOT_FIXED_SECOND_BUILD_ATTEMPT')
    for key in ('started_at_utc', 'finished_at_utc'):
        require(isinstance(summary.get(key), str), 'SECOND_BUILD_HAS_NO_TERMINAL_TIMESTAMPS')
    from datetime import datetime
    started, finished = (datetime.fromisoformat(summary[k]) for k in ('started_at_utc', 'finished_at_utc'))
    require(started.tzinfo is not None and finished.tzinfo is not None and finished >= started,
            'INVALID_SECOND_BUILD_TERMINAL_TIMESTAMPS')
    require(contract['effective_budgets'] == amendment['effective_budgets']
            and contract['source_reuse'] == amendment['retry_sources'][segment],
            'SECOND_BUILD_RESOURCE_OR_COARSE_BINDING_CHANGED')
    selected = next(s for s in amendment['effective_protocol']['segments'] if s['segment_id'] == segment)
    require(contract['segment'] == selected and contract['protocol_sha256'] == amendment['protocol_sha256']
            and contract.get('worker_uses_original_protocol_unchanged') is True
            and contract.get('only_supervisor_native_wall_and_output_ceiling_amended') is True,
            'SECOND_BUILD_FIXED_METHOD_BINDING_CHANGED')
    if summary['status'] == 'COMPLETE_PRE_DISPLACEMENT_OPAQUE_CACHE':
        native = summary.get('stages', {}).get('native', {})
        require(native.get('exit_code') == 0 and native.get('infrastructure_stop') is None
                and summary.get('native_complete', {}).get('status') == 'COMPLETE_PRE_DISPLACEMENT_OPAQUE_CACHE'
                and summary.get('original_source_inputs_unchanged') is True
                and summary.get('final_bound_inputs_unchanged') is True,
                'SECOND_BUILD_COMPLETION_NOT_VERIFIED')
    return summary['status'] == 'COMPLETE_PRE_DISPLACEMENT_OPAQUE_CACHE'


def validate_outcome_binding(row, outcome, outcome_ref, build, build_ref, contract_ref, amendment):
    """Reject first-attempt STOP/new-build mixtures and unbound stage outcomes."""
    segment = row['segment_id']
    expected = {'path': amendment['amendment_path'], 'sha256': amendment['amendment_sha256']}
    require(outcome['segment_id'] == segment and outcome.get('budget_amendment') == expected,
            'COVERAGE_OUTCOME_NOT_FROM_FINAL_AMENDMENT')
    stage = row.get('upstream_stage')
    if stage == 'build':
        require(outcome_ref == build_ref and build['status'].startswith('STOP')
                and not row['screen_complete'] and row.get('terminal_infrastructure_stop') is True,
                'COVERAGE_BUILD_STOP_IS_NOT_THIS_SECOND_BUILD')
        require(row.get('upstream_failure_artifact') == build_ref, 'COVERAGE_UPSTREAM_BUILD_REFERENCE_CHANGED')
    else:
        require(build['status'] == 'COMPLETE_PRE_DISPLACEMENT_OPAQUE_CACHE',
                'DOWNSTREAM_OUTCOME_WITHOUT_COMPLETED_SECOND_BUILD')
        require(stage in (None, 'source'), 'UNSUPPORTED_FINAL_OUTCOME_STAGE')
        if stage == 'source':
            require(outcome['status'].startswith('STOP') and not row['screen_complete']
                    and row.get('terminal_infrastructure_stop') is True,
                    'SOURCE_NOT_A_TERMINAL_STOP')
            links = outcome.get('amendment_input_sha256', {})
        else:
            links = outcome.get('input_and_code_sha256', {})
            require((row['screen_complete'] and outcome['status'] == 'PASS_PREREGISTERED_SEGMENT_SCREEN')
                    or (not row['screen_complete'] and outcome['status'].startswith('STOP')
                        and row.get('terminal_infrastructure_stop') is True), 'SCREEN_OUTCOME_NOT_TERMINAL')
        require(all(links.get(ref['path']) == ref['sha256'] for ref in (build_ref, contract_ref)),
                'DOWNSTREAM_OUTCOME_NOT_BOUND_TO_THIS_SECOND_BUILD')
    return True


def verify_final_outcomes(coverage, coverage_path, retry_documents, amendment, bindings, external_stops=None):
    """Rebuild coverage rows from their bound outcomes and exact second-build chain."""
    base = Path(coverage_path).resolve().parent
    rows = {r['segment_id']: r for r in coverage['rows'] if r['segment_id'] != 'forest_a_previous'}
    require(len(rows) == 3 and set(rows) == {'cave', 'forest_b', 'mountain'}
            and set(retry_documents) == {'cave', 'forest_b'}, 'FIXED_THREE_SEGMENT_POPULATION_REQUIRED')
    external_stops = {} if external_stops is None else external_stops
    require(set(external_stops) <= {'cave', 'forest_b'}, 'UNEXPECTED_EXTERNAL_STOP_SEGMENT')
    evidence = {}
    for segment in ('cave', 'forest_b'):
        entry = retry_documents[segment]; build, bref = entry['summary'], entry['ref']
        contract_path = Path(bref['path']).parent/'build_protocol.json'
        contract, cref = read_bound(contract_path, bindings, build['build_protocol_sha256'])
        validate_retry_build(build, contract, segment, amendment)
        row = rows[segment]
        ref = _resolved_ref(row['artifact_references']['summary'], base)
        outcome, ref = read_bound(ref['path'], bindings, ref['sha256'])
        external = external_stops.get(segment)
        if external is not None:
            require(not row['screen_complete'] and row.get('terminal_infrastructure_stop') is True
                    and not row.get('upstream_stage') and outcome['status'] == 'STOP_SCREEN_ATTEMPT_NOT_COMPLETE',
                    'EXTERNAL_BINDING_ONLY_FOR_ACTUAL_TERMINAL_SCREEN_STOP')
            from final_stop_binding import verify_external_stop
            verify_external_stop(external['summary'], ref, bref, cref, amendment, bindings)
        else:
            validate_outcome_binding(row, outcome, ref, build, bref, cref, amendment)

        selected = next(s for s in amendment['effective_protocol']['segments'] if s['segment_id'] == segment)
        if row.get('upstream_stage'):
            shaped, _ = _stage_stop(ref['path'], row['upstream_stage'], selected,
                amendment['protocol_sha256'], amendment['preregistration_seal_sha256'], bindings)
            confirmation = None
        else:
            shaped = outcome; confirmation = None
            if row['screen_complete']:
                oref = _resolved_ref(outcome['omp_confirmation']['artifact'], Path(ref['path']).parent)
                confirmation, _ = read_bound(oref['path'], bindings, oref['sha256'])
                confirmed = verify_confirmed_summaries(outcome, confirmation, oref['path'], bindings)
                expected = {'path': amendment['amendment_path'], 'sha256': amendment['amendment_sha256']}
                require(confirmation.get('budget_amendment') == expected, 'CONFIRMED_RETRY_AMENDMENT_CHANGED')
                for original in confirmed:
                    document, _ = read_bound(original['path'], bindings, original['sha256'])
                    require(document.get('budget_amendment') == expected
                            and all(document.get('input_and_code_sha256', {}).get(x['path']) == x['sha256']
                                    for x in (bref, cref)), 'CONFIRMED_RETRY_BUILD_BINDING_CHANGED')
        rebuilt = scene_row(shaped, selected, amendment['protocol_sha256'], amendment['preregistration_seal_sha256'],
                            amendment['effective_protocol']['visibility']['qualifying_component_rule'], confirmation)
        require({k:v for k,v in row.items() if k != 'artifact_references'} == rebuilt,
                'COVERAGE_ROW_NOT_EQUAL_BOUND_FINAL_OUTCOME')
        evidence[segment] = {'second_build': bref, 'second_build_contract': cref, 'outcome': ref,
                             'outcome_stage': row.get('upstream_stage', 'screen')}
        if external is not None:
            evidence[segment]['external_stop_binding'] = external['ref']
            evidence[segment]['screen_self_certification'] = False
    mountain = rows['mountain']
    mref = _resolved_ref(mountain['artifact_references']['summary'], base)
    require(mref == amendment['mountain_completed_evidence'] and mountain['screen_complete'],
            'MOUNTAIN_MUST_USE_ORIGINAL_CONFIRMED_SUMMARY')
    document, mref = read_bound(mref['path'], bindings, mref['sha256'])
    oref = _resolved_ref(document['omp_confirmation']['artifact'], Path(mref['path']).parent)
    confirmation, _ = read_bound(oref['path'], bindings, oref['sha256'])
    originals = verify_confirmed_summaries(document, confirmation, oref['path'], bindings)
    selected = next(s for s in amendment['effective_protocol']['segments'] if s['segment_id'] == 'mountain')
    rebuilt = scene_row(document, selected, amendment['protocol_sha256'], amendment['preregistration_seal_sha256'],
                        amendment['effective_protocol']['visibility']['qualifying_component_rule'], confirmation)
    require({k:v for k,v in mountain.items() if k != 'artifact_references'} == rebuilt,
            'MOUNTAIN_COVERAGE_ROW_CHANGED')
    evidence['mountain'] = {'outcome': mref, 'confirmed_omp_summaries': originals, 'reused_not_rerun': True}
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','seal','budget-amendment','coverage','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--retry-build', action='append', required=True, help='segment=build02/summary.json')
    parser.add_argument('--external-screen-stop', action='append', default=[],
                        help='segment=outer_stop_binding.json; raw coverage/STOP remain unchanged')
    args = parser.parse_args()
    amendment = load_amendment(args.budget_amendment,args.protocol,args.seal)
    bindings = dict(amendment['input_sha256'])
    coverage, cref = read_bound(args.coverage, bindings)
    bindings.update(coverage['input_json_sha256'])
    require(coverage['status'] == 'PASS_HASH_BOUND_COVERAGE_LEDGER' and
            coverage['protocol']['sha256'] == amendment['protocol_sha256'] and
            coverage['preregistration_seal']['sha256'] == amendment['preregistration_seal_sha256'],
            'FINAL_COVERAGE_PROTOCOL_MISMATCH')
    _, aref = read_bound(args.budget_amendment, bindings)
    retries = {}
    for entry in args.retry_build:
        segment, path = entry.split('=',1)
        require(segment in ('cave','forest_b') and segment not in retries, 'UNEXPECTED_RETRY_BUILD_POPULATION')
        summary, ref = read_bound(path,bindings)
        require(summary['segment_id'] == segment and summary['budget_amendment']['sha256'] == aref['sha256'],
                'RETRY_NOT_BOUND_TO_FINAL_APPROVED_BUDGET')
        retries[segment] = {'summary': summary, 'ref': ref}
    require(set(retries) == {'cave','forest_b'}, 'BOTH_SECOND_BUILD_ATTEMPTS_MUST_BE_REPORTED')
    external_stops = {}
    for entry in args.external_screen_stop:
        segment, path = entry.split('=', 1)
        require(segment in ('cave', 'forest_b') and segment not in external_stops, 'UNEXPECTED_EXTERNAL_STOP_SEGMENT')
        document, ref = read_bound(path, bindings)
        external_stops[segment] = {'summary': document, 'ref': ref}
    outcome_bindings = verify_final_outcomes(coverage, args.coverage, retries, amendment, bindings, external_stops)
    result = {'schema': 'final-preregistered-visibility-verdict-v1',
        'status': 'HASH_BOUND_FINAL_VISIBILITY_DECISION_NOT_QUALITY_EVIDENCE',
        'coverage': cref, 'budget_amendment': aref, 'retry_builds': {k:v['ref'] for k,v in retries.items()},
        'validated_final_outcome_chains': outcome_bindings, 'executed_script_sha256': file_sha(__file__),
        **classify(coverage['rows']), 'input_sha256': bindings}
    require(not args.output.exists(), 'FINAL_VERDICT_OUTPUT_MUST_BE_FRESH')
    for path, expected in bindings.items():
        import hashlib
        require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, 'VERDICT_BOUND_INPUT_CHANGED')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:result[k] for k in ('decision','completed_new_segments','uncompleted_new_segments')},sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
