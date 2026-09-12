"""Read saved Nsight CSV; no process launch or GPU access."""
import csv
import io
import math

class NcuCsvError(ValueError):
    pass

IDENTITY = ('ID', 'Process ID', 'Kernel Name', 'Context', 'Stream')

def parse_ncu_csv(raw):
    if 'ERR_' in raw or '==ERROR==' in raw:
        raise NcuCsvError('Nsight error in raw report')
    lines = raw.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith('"ID",') and '"Kernel Name"' in line), None)
    if start is None:
        raise NcuCsvError('Missing Nsight CSV header')
    rows = list(csv.reader(io.StringIO('\n'.join(lines[start:]))))
    header = rows[0]
    if len(header) != len(set(header)) or not all(k in header for k in IDENTITY):
        raise NcuCsvError('Duplicate columns or missing kernel identity')
    data = [r for r in rows[1:] if r]
    if any(len(r) != len(header) for r in data):
        raise NcuCsvError('Ragged CSV row')
    if 'Metric Name' in header:
        if not {'Metric Unit', 'Metric Value'}.issubset(header):
            raise NcuCsvError('Incomplete long metric columns')
        records = [dict(zip(header, r)) for r in data]
    else:
        if len(data) < 2:
            raise NcuCsvError('Raw wide CSV requires units and kernel rows')
        units = dict(zip(header, data[0]))
        if any(units[k] for k in IDENTITY):
            raise NcuCsvError('Missing raw wide CSV units row')
        metrics = [k for k in header if '__' in k or k == 'inst_executed']
        if not metrics:
            raise NcuCsvError('No metric columns in raw wide CSV')
        records, identities = [], set()
        for values in data[1:]:
            row = dict(zip(header, values))
            identity = tuple(row[k] for k in IDENTITY)
            if identity in identities:
                raise NcuCsvError('Duplicate wide kernel row')
            identities.add(identity)
            for metric in metrics:
                records.append({**{k: row[k] for k in IDENTITY}, 'Metric Name': metric,
                                'Metric Unit': units[metric], 'Metric Value': row[metric]})
    if not records or any(not r['ID'].isdigit() or not r['Kernel Name'] or not r['Metric Name'] for r in records):
        raise NcuCsvError('Empty report or invalid metric/kernel identity')
    return records

def single_kernel_metrics(raw):
    records = parse_ncu_csv(raw)
    identities = {tuple(r[k] for k in IDENTITY) for r in records}
    if len(identities) != 1:
        raise NcuCsvError('Expected exactly one profiled kernel')
    metrics = {}
    for r in records:
        name = r['Metric Name']
        value = {'unit': r['Metric Unit'], 'raw_value': r['Metric Value']}
        if name in metrics and metrics[name] != value:
            raise NcuCsvError('Conflicting duplicate metric: ' + name)
        metrics[name] = value
    return {'identity': dict(zip(IDENTITY, next(iter(identities)))), 'metrics': metrics}

def finite_metric(metrics, name, expected_unit=None):
    if name not in metrics:
        raise NcuCsvError('Missing metric: ' + name)
    row = metrics[name]
    if expected_unit is not None and row['unit'] != expected_unit:
        raise NcuCsvError('Wrong unit for ' + name)
    try:
        value = float(row['raw_value'].replace(',', ''))
    except ValueError as exc:
        raise NcuCsvError('Nonnumeric metric: ' + name) from exc
    if not math.isfinite(value):
        raise NcuCsvError('Nonfinite metric: ' + name)
    return value