#!/usr/bin/env python3
"""Plot saved anchor-ablation JSON only; never load meshes or recompute geometry.

Four-event mean-height figure retains unsupported events. A separate regret
figure exposes negative outcomes and zero-gain cases, including an automatic
near-zero zoom for a small positive maximum-error regression.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, FormatStrFormatter
import numpy as np

GRID = 128
MAX_OUTPUT_BYTES = 5 * 1024 * 1024
STYLES = {
    'baseline': {'label': 'Ordinary baseline', 'color': '#334155', 'linestyle': '-', 'linewidth': 2.1},
    'naive': {'label': 'Endpoint chord (null)', 'color': '#a4aebc', 'linestyle': (0, (5, 4)), 'linewidth': 3.1},
    'centroid': {'label': 'Centroid-root fan', 'color': '#d97706', 'linestyle': (0, (4, 2, 1, 2)), 'linewidth': 2.0},
    'beb1': {'label': 'BEB1-root fan', 'color': '#007f86', 'linestyle': '-', 'linewidth': 2.2},
}


def rational(value):
    return Fraction(int(value['numerator']), int(value['denominator']))


def phase(tau, levels):
    lower, root, upper = levels
    if not lower < root < upper or not lower <= tau <= upper:
        raise ValueError('Malformed original window or out-of-window measurement.')
    return float((tau-root)/(root-lower if tau <= root else upper-root))


def series(event, method, metric):
    data = event['methods'][method]
    if data['status'] != 'COMPLETE_REFERENCE_ONLY':
        raise ValueError(f'{event["key"]}/{method}: incomplete curves must not be plotted as complete.')
    levels = tuple(map(rational, event['levels_internal']))
    rows = sorted(data['measurements'], key=lambda row: rational(row['time']))
    times = [rational(row['time']) for row in rows]
    if len(times) != 33 or len(set(times)) != 33 or times[0] != levels[0] or times[-1] != levels[2] or levels[1] not in times:
        raise ValueError('The saved study must contain all 33 fixed probes, including endpoint/root anchors.')
    values = []
    for row in rows:
        matches = [item for item in row['grids'] if item['grid'] == GRID]
        if len(matches) != 1:
            raise ValueError('Expected exactly one saved grid-128 measurement at each time.')
        values.append(float(matches[0][metric]))
    if not np.all(np.isfinite(values)) or any(value < 0 for value in values):
        raise ValueError('Height-error measurements must be finite and nonnegative.')
    return times, np.asarray([phase(tau, levels) for tau in times]), np.asarray(values)


def event_label(event):
    return 'E'+str(int(event['key'].split('-')[1]))


def clean_axes(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#cbd5e1')
    ax.tick_params(colors='#475569', labelsize=9)
    ax.grid(axis='y', color='#e2e8f0', linewidth=.65)
    ax.axvline(0, color='#cbd5e1', linewidth=.8, linestyle=':', zorder=0)
    ax.set_xlim(-1.04, 1.04)
    ax.set_xticks([-1, -.5, 0, .5, 1])


def complete(event):
    return event['status'] == 'COMPLETE_REFERENCE_ONLY' and all(
        event['methods'][name]['status'] == 'COMPLETE_REFERENCE_ONLY' for name in STYLES)


def draw_means(report):
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.8))
    fig.subplots_adjust(left=.085, right=.97, bottom=.13, top=.80, wspace=.25, hspace=.46)
    fig.suptitle('Event-window anchor ablation', x=.07, y=.975, ha='left', fontsize=20, fontweight='bold', color='#152238')
    fig.text(.07, .93, 'Saved binary64 reference geometry  |  128-grid height error  |  all four fixed demo events', fontsize=11, color='#475569')
    handles = [Line2D([], [], **STYLES[name]) for name in ('baseline', 'naive', 'centroid', 'beb1')]
    handles.append(Line2D([], [], color='#a03d87', marker='D', linestyle='None', markersize=5, label='C0 (root only)'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.52, .898), ncol=5, frameon=False, fontsize=9)
    for ax, event in zip(axes.ravel(), report['events']):
        name = event_label(event)
        if not complete(event):
            ax.set_axis_off()
            ax.set_facecolor('#f8fafc')
            ax.text(0, 1.06, name+'  |  Unsupported', transform=ax.transAxes, fontsize=13, fontweight='bold', color='#7f4a22')
            reason = event.get('stop_reason') or 'The saved method curve did not complete.'
            message = ('No curve was evaluated.\n\n'+textwrap.fill(reason, width=48)+
                       '\n\nOriginal event and endpoint contract retained;\nnot replaced by a more favorable event.')
            ax.text(.04, .84, message, transform=ax.transAxes, va='top', fontsize=11, color='#475569',
                    bbox={'boxstyle': 'round,pad=.8', 'facecolor': '#f8fafc', 'edgecolor': '#e2e8f0'})
            continue
        clean_axes(ax)
        reference_times = None
        # Draw the thicker null curve first: its agreement with baseline is
        # visible as a dashed grey surround, not an artificial position offset.
        for method in ('naive', 'baseline', 'centroid', 'beb1'):
            times, x, values = series(event, method, 'mean_absolute_height_error')
            if reference_times is not None and reference_times != times:
                raise ValueError('Comparison curves do not share exact saved time probes.')
            reference_times = times
            ax.plot(x, values, **STYLES[method])
        c0_times, c0_x, c0_values = series(event, 'c0_root_only', 'mean_absolute_height_error')
        root = rational(event['levels_internal'][1])
        root_index = c0_times.index(root)
        ax.plot(c0_x[root_index], c0_values[root_index], marker='D', markersize=5,
                color='#a03d87', markeredgecolor='white', markeredgewidth=.6, linestyle='None', zorder=5)
        rows = [row for row in event.get('comparison_summaries', []) if row.get('grid') == GRID and row.get('opponent') == 'centroid']
        note = 'No integrated comparison available'
        if len(rows) == 1 and rows[0].get('normalized_integral_relative_change') is not None:
            change = 100*rows[0]['normalized_integral_relative_change']
            note = f'Time-integrated mean vs centroid: {change:+.2f}%'
        ax.set_title(f'{name}  |  {event["window_seconds"]*1000:.2f} ms window', loc='left', pad=13, fontsize=13, fontweight='bold')
        ax.text(0, 1.015, note, transform=ax.transAxes, fontsize=9, color='#475569')
        ax.set_ylabel('Mean absolute height error\n(model-space units)', fontsize=10, color='#334155')
        ax.set_xlabel('Normalized window time (root = 0)', fontsize=10, color='#334155')
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        ax.margins(y=.14)
    fig.text(.07, .065, 'Lower is better. Endpoints are -1 / +1; each half-window has 17 saved probes (33 total). Grey/null curves may overlap baseline.', fontsize=9, color='#475569')
    fig.text(.07, .037, 'Local evidence from one demo family only. No production-window admission, continuous-error bound, rendering, or SSIM claim.', fontsize=9, color='#475569')
    return fig


def draw_regret(report):
    events = [event for event in report['events'] if complete(event)]
    if not events:
        raise ValueError('No complete saved events for a regret plot.')
    fig, axes = plt.subplots(1, len(events), figsize=(12.6, 4.85), squeeze=False)
    fig.subplots_adjust(left=.08, right=.97, bottom=.25, top=.74, wspace=.32)
    fig.suptitle('Worst-height-error regret: BEB1 minus centroid', x=.07, y=.965, ha='left', fontsize=19, fontweight='bold', color='#152238')
    fig.text(.07, .90, 'Positive = BEB1 is worse at that saved time. Negative = BEB1 is better. Each panel has its own y scale.', fontsize=11, color='#475569')
    witness_records = []
    for ax, event in zip(axes.ravel(), events):
        times, x, ours = series(event, 'beb1', 'maximum_absolute_height_error')
        other_times, other_x, centroid = series(event, 'centroid', 'maximum_absolute_height_error')
        if times != other_times or not np.array_equal(x, other_x):
            raise ValueError('Regret requires identical saved time samples.')
        regret = ours-centroid
        clean_axes(ax)
        ax.axhline(0, color='#64748b', linewidth=1.)
        ax.plot(x, regret, color='#007f86', linewidth=2.)
        ax.fill_between(x, 0, regret, where=regret < 0, color='#9ad3ce', alpha=.45, interpolate=True)
        ax.fill_between(x, 0, regret, where=regret > 0, color='#f4a4a4', alpha=.6, interpolate=True)
        peak_index = int(np.argmax(regret))
        peak = float(regret[peak_index])
        ax.set_title(event_label(event), loc='left', pad=12, fontsize=13, fontweight='bold')
        ax.text(0, 1.015, f'Maximum saved-time regret: {peak:+.3e}', transform=ax.transAxes,
                fontsize=8.5, color='#aa3030' if peak > 0 else '#475569')
        ax.set_xlabel('Normalized window time', fontsize=10, color='#334155')
        scale_power = int(np.floor(np.log10(np.max(np.abs(regret))))) if np.any(regret != 0) else -5
        scale = 10.**scale_power
        ax.set_ylabel('Max-height-error difference\n'+r'($\times 10^{%d}$ model units)' % scale_power, fontsize=9, color='#334155')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, position, unit=scale: f'{value/unit:.1f}'))
        if np.all(regret == 0):
            ax.set_ylim(-1e-5, 1e-5)
            ax.text(.5, .35, 'Error curves coincide\nat all 33 saved probes.', transform=ax.transAxes,
                    ha='center', fontsize=10, color='#475569')
        elif peak > 0:
            ax.plot(x[peak_index], peak, 'o', color='#b63131', markersize=4, zorder=6)
            # An explicit near-zero zoom makes a small regression visible even
            # when larger negative values dominate the main y-axis range.
            inset = ax.inset_axes([.36, .13, .60, .34])
            inset.set_facecolor('#fffafb')
            inset.plot(x, regret, color='#aa3030', linewidth=1.1)
            inset.axhline(0, color='#64748b', linewidth=.6)
            inset.plot(x[peak_index], peak, 'o', color='#b63131', markersize=3)
            inset.set_ylim(-.25*peak, 1.35*peak)
            inset.set_xlim(max(-1., x[peak_index]-.35), min(1., x[peak_index]+.35))
            inset.tick_params(labelsize=6, pad=1)
            inset.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
            inset.set_title('Near-zero zoom: positive loss', fontsize=6.5, pad=3)
        witness_records.append({'event': event['key'], 'maximum_sampled_max_error_regret': peak,
                                'time': {'numerator': times[peak_index].numerator, 'denominator': times[peak_index].denominator},
                                'strictly_positive_regret_observed': bool(peak > 0)})
    missing = ', '.join(event_label(event) for event in report['events'] if not complete(event))
    fig.text(.07, .12, f'Grid 128; errors and differences come only from saved JSON. Unsupported and not omitted from the study: {missing or "none"}.', fontsize=9, color='#475569')
    fig.text(.07, .077, 'These are sampled maximum-height errors, not certified continuous maxima. A lower time-integrated mean can coexist with local regression.', fontsize=9, color='#475569')
    return fig, witness_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError('Plot output must be a fresh directory: '+str(output))
    payload = source.read_bytes()
    report = json.loads(payload)
    if report.get('schema') != 'binoc-c1-lite-anchor-ablation-v1' or len(report.get('events', [])) != 4:
        raise ValueError('Expected the saved four-event anchor-ablation report schema.')
    if len({event['key'] for event in report['events']}) != 4:
        raise ValueError('Duplicate event keys are not allowed.')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'svg.fonttype': 'none', 'axes.unicode_minus': True,
                         'savefig.facecolor': 'white'})
    first = draw_means(report)
    second, witnesses = draw_regret(report)
    output.mkdir(parents=True)
    files = []
    for name, figure in (('mean_height_all_four_events', first), ('max_height_regret_vs_centroid', second)):
        for extension in ('svg', 'png'):
            path = output/(name+'.'+extension)
            figure.savefig(path, dpi=160)
            files.append({'file': path.name, 'bytes': path.stat().st_size,
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        plt.close(figure)
    manifest = {'schema': 'binoc-anchor-ablation-plots-v1', 'input': str(source),
                'input_sha256': hashlib.sha256(payload).hexdigest(), 'grid': GRID,
                'geometry_recomputed': False, 'meshes_read': False, 'renderer_used': 'matplotlib Agg',
                'files': files, 'regret_witnesses': witnesses,
                'scope': 'Restricted geometry evidence from the first demo family; no runtime/video/paper-superiority claim.',
                'c0_display': 'Single exact-root marker only, never connected into a smoothed window curve.',
                'total_figure_bytes': sum(row['bytes'] for row in files)}
    manifest_text = json.dumps(manifest, indent=2, allow_nan=False)+'\n'
    if manifest['total_figure_bytes']+len(manifest_text.encode()) > MAX_OUTPUT_BYTES:
        raise RuntimeError('Generated figures exceeded the fixed 5 MiB budget.')
    (output/'manifest.json').write_text(manifest_text, encoding='utf-8')
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
