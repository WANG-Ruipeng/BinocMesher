# Preregistered coverage ledger

Source admission and visible source-support coverage are distinct. N/A means unmeasured, unresolved, or an empty denominator—not 0%.

| Segment | Status | Events | Joint events / components | Modified natural frames | Visible source / admitted | Visible rate |
| --- | --- | ---: | --- | ---: | --- | --- |
| forest_a_previous | PREVIOUS_COMPLETED_NEGATIVE | 131 | 25 / 20 | 16 | 0 / 0 | N/A |
| forest_b | STOP_UPSTREAM_BUILD | N/A | N/A / N/A | N/A | N/A / N/A | N/A |
| cave | STOP_UPSTREAM_BUILD | N/A | N/A / N/A | N/A | N/A / N/A | N/A |
| mountain | NOT_RUN_OR_REPORT_MISSING | N/A | N/A / N/A | N/A | N/A / N/A | N/A |

The previous Forest negative is retained. A STOP or missing segment is never counted as zero events or zero visibility.

## Frozen selection

First by the frozen rank among completed segments only; not a claimed first across stopped or unmeasured segments.

Pilot selection ready: false. Post-displacement checks remain required.

- forest_b: ValueError: Bounded native stage did not complete: {'exit_code': -15, 'infrastructure_stop': 'REGISTERED_DISK_SAFETY_RESERVE', 'wall_seconds': 2363.932227059, 'peak_process_group_rss_bytes': 4292042752, 'peak_monitored_output_bytes': 4072805034, 'wall_limit_seconds': 2700, 'output_limit_bytes': 4294967296, 'early_stop_output_bytes': 4026531840, 'rss_limit_bytes': 17179869184, 'single_file_limit_bytes': 1073741824, 'all_children_exit_expected': True, 'directory_quota_is_polled_not_filesystem_enforced': True}
- cave: ValueError: Bounded native stage did not complete: {'exit_code': -15, 'infrastructure_stop': 'REGISTERED_WALL_LIMIT', 'wall_seconds': 2700.2017811250003, 'peak_process_group_rss_bytes': 2766114816, 'peak_monitored_output_bytes': 426612159, 'wall_limit_seconds': 2700, 'output_limit_bytes': 4294967296, 'early_stop_output_bytes': 4026531840, 'rss_limit_bytes': 17179869184, 'single_file_limit_bytes': 1073741824, 'all_children_exit_expected': True, 'directory_quota_is_polled_not_filesystem_enforced': True}
- mountain: No completed or STOP summary supplied.

No quality delta was read for ranking. Coverage is not evidence of visual improvement, artifact-impact coverage, all-time admission, or post-displacement safety.
