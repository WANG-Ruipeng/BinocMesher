#!/usr/bin/env python3
"""Production-derived TV0--TV4 theory audit for BinocMesher.

This module deliberately does not modify production slicing.  It consumes the
read-only BHP2/HV/event-registry artifacts emitted by the authoritative P1
observer and builds falsifiable, source-labeled offline certificates.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import itertools
import json
import math
import random
import struct
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

# Binary ABI pinned by the existing proof-ready smoke on Linux/GCC/x86-64.
HV_RECORD_SIZE = 28
SOURCE_HEADER = struct.Struct('<IIIIQ')
SOURCE_RECORD = struct.Struct('<26i')
SOURCE_MAGIC = 0x32504842  # BHP2
PROVENANCE_VERSION = 2
PROVENANCE_LAYOUT_VERSION = 1

TEMPORAL_FACE_SLOTS = {
    0: (0, 1, 3, 2),
    1: (4, 5, 7, 6),
}
FACE_CORNERS = (
    (0, 1, 3, 2),  # temporal side 0
    (4, 5, 7, 6),  # temporal side 1
    (0, 1, 5, 4),  # j side 0
    (2, 3, 7, 6),  # j side 1
    (0, 2, 6, 4),  # i side 0
    (1, 3, 7, 5),  # i side 1
)
CUBE_EDGES = tuple(sorted({
    tuple(sorted((face[i], face[(i + 1) % 4])))
    for face in FACE_CORNERS for i in range(4)
}))


class AuditError(RuntimeError):
    pass


@dataclass(frozen=True, order=True)
class HVID:
    node: int
    group: int

    def text(self) -> str:
        return f'{self.node}:{self.group}'


@dataclass(frozen=True)
class HV:
    hvid: HVID
    position: tuple[float, float, float]
    time: int
    halfspan: int
    in_view: int


@dataclass(frozen=True)
class SourceRecord:
    source_t_group: int
    source_record_index: int
    edge_coords: tuple[int, int, int]
    edge_L: int
    edge_tcoord: int
    edge_tL: int
    edge_dir: int
    element: int
    hvids: tuple[HVID, ...]

    def logical_key(self) -> tuple[Any, ...]:
        # Cache t_group/index are replication provenance, not source identity.
        return (
            self.edge_coords, self.edge_L, self.edge_tcoord, self.edge_tL,
            self.edge_dir, self.element,
            tuple((h.node, h.group) for h in self.hvids),
        )

    def source_edge_key(self) -> tuple[int, ...]:
        return (*self.edge_coords, self.edge_L, self.edge_tcoord,
                self.edge_tL, self.edge_dir, self.element)


@dataclass
class Cell:
    cell_id: str
    representative: SourceRecord
    raw_records: int
    hvids: tuple[HVID, ...]
    times: tuple[int, ...]
    positions: np.ndarray
    gaps: tuple[int, int, int, int]
    route: str
    equality_partition: tuple[int, ...]
    sampled_jacobian_class: str
    sampled_jacobian_min: float
    sampled_jacobian_max: float


@dataclass(frozen=True)
class Saddle:
    element: int
    face_hvids: tuple[HVID, HVID, HVID, HVID]
    face_times: tuple[int, int, int, int]
    root: Fraction
    u: Fraction
    v: Fraction
    A: int
    B: int

    def key(self) -> tuple[Any, ...]:
        return (
            self.element,
            tuple((h.node, h.group) for h in self.face_hvids),
            self.root.numerator,
            self.root.denominator,
        )

    def event_id(self) -> str:
        face = '|'.join(h.text() for h in self.face_hvids)
        return f'element={self.element};role=temporal_neighbour;face={face}'


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), default=str)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode('utf-8')).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + '\n')


def write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True, default=str) + '\n')


def fraction_json(value: Fraction) -> dict[str, int]:
    return {'numerator': value.numerator, 'denominator': value.denominator}


def parse_hvid(text: str) -> HVID:
    node, group = text.split(':', 1)
    return HVID(int(node), int(group))


def parse_hypervertices(root: Path) -> dict[HVID, HV]:
    result: dict[HVID, HV] = {}
    files = sorted((root / 'hypervertices').glob('*.bin'))
    if not files:
        raise AuditError('no hypervertex cache files')
    for path in files:
        data = path.read_bytes()
        if len(data) < 4:
            raise AuditError(f'{path}: truncated vector header')
        count = struct.unpack_from('<i', data, 0)[0]
        if count < 0 or len(data) != 4 + count * HV_RECORD_SIZE:
            raise AuditError(f'{path}: invalid hypervertex vector size')
        for index in range(count):
            offset = 4 + index * HV_RECORD_SIZE
            node = struct.unpack_from('<i', data, offset)[0]
            group = struct.unpack_from('<b', data, offset + 4)[0]
            position = struct.unpack_from('<3f', data, offset + 8)
            time = struct.unpack_from('<b', data, offset + 20)[0]
            halfspan = struct.unpack_from('<b', data, offset + 21)[0]
            in_view = struct.unpack_from('<b', data, offset + 24)[0]
            hvid = HVID(node, group)
            hv = HV(hvid, tuple(float(x) for x in position), time, halfspan, in_view)
            previous = result.get(hvid)
            if previous is not None and previous != hv:
                raise AuditError(f'inconsistent duplicate hypervertex {hvid.text()}')
            result[hvid] = hv
    return result


def parse_source_records(root: Path) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    files = sorted((root / 'hyperpoly_meta').glob('*.bin'))
    if not files:
        raise AuditError('no BHP2 source provenance files')
    for path in files:
        data = path.read_bytes()
        if len(data) < SOURCE_HEADER.size:
            raise AuditError(f'{path}: truncated BHP2 header')
        magic, version, record_size, layout, count = SOURCE_HEADER.unpack_from(data, 0)
        if (magic, version, record_size, layout) != (
            SOURCE_MAGIC, PROVENANCE_VERSION, SOURCE_RECORD.size,
            PROVENANCE_LAYOUT_VERSION,
        ):
            raise AuditError(f'{path}: BHP2 schema mismatch')
        expected = SOURCE_HEADER.size + count * SOURCE_RECORD.size
        if len(data) != expected:
            raise AuditError(f'{path}: BHP2 payload/trailing byte mismatch')
        for index in range(count):
            values = SOURCE_RECORD.unpack_from(data, SOURCE_HEADER.size + index * SOURCE_RECORD.size)
            if values[1] != index:
                raise AuditError(f'{path}: source record index mismatch at {index}')
            hvids = tuple(HVID(values[10 + i], values[18 + i]) for i in range(8))
            records.append(SourceRecord(
                source_t_group=values[0],
                source_record_index=values[1],
                edge_coords=(values[2], values[3], values[4]),
                edge_L=values[5], edge_tcoord=values[6], edge_tL=values[7],
                edge_dir=values[8], element=values[9], hvids=hvids,
            ))
    return records


def parse_registry(root: Path) -> list[dict[str, str]]:
    path = root / 'event_registry_p1.csv'
    if not path.is_file():
        raise AuditError('event registry CSV missing')
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def canonical_partition(values: Sequence[HVID]) -> tuple[int, ...]:
    mapping: dict[HVID, int] = {}
    return tuple(mapping.setdefault(value, len(mapping)) for value in values)


def canonical_cycle(
    hvids: Sequence[HVID], times: Sequence[int]
) -> tuple[tuple[HVID, ...], tuple[int, ...]]:
    if len(hvids) != 4 or len(times) != 4:
        raise ValueError('face cycle must have four entries')
    variants = []
    hs = tuple(hvids); ts = tuple(int(x) for x in times)
    for reversed_cycle in (False, True):
        base_h = hs if not reversed_cycle else (hs[0], hs[3], hs[2], hs[1])
        base_t = ts if not reversed_cycle else (ts[0], ts[3], ts[2], ts[1])
        for offset in range(4):
            vh = base_h[offset:] + base_h[:offset]
            vt = base_t[offset:] + base_t[:offset]
            variants.append((vh, vt))
    return min(variants, key=lambda x: tuple((h.node, h.group) for h in x[0]))


def admissible_saddle(
    hvids: Sequence[HVID], times: Sequence[int], element: int
) -> Saddle | None:
    if len(set(hvids)) != 4:
        return None
    hcanon, tcanon = canonical_cycle(hvids, times)
    t00, t10, t11, t01 = tcanon
    A = t00 + t11 - t10 - t01
    B = t00 * t11 - t10 * t01
    if A == 0:
        return None
    root = Fraction(B, A)
    u = Fraction(t00 - t01, A)
    v = Fraction(t00 - t10, A)
    if not (min(tcanon) < root < max(tcanon)):
        return None
    if root in {Fraction(value, 1) for value in tcanon}:
        return None
    signs = tuple(Fraction(value, 1) - root for value in tcanon)
    if not (signs[0] * signs[2] > 0 and
            signs[1] * signs[3] > 0 and signs[0] * signs[1] < 0):
        return None
    if not (0 < u < 1 and 0 < v < 1):
        return None
    return Saddle(element, hcanon, tcanon, root, u, v, A, B)


def trilinear_position(positions: np.ndarray, u: float, v: float, w: float) -> np.ndarray:
    out = np.zeros(3, dtype=np.float64)
    for k in (0, 1):
        for j in (0, 1):
            for i in (0, 1):
                slot = i + 2 * j + 4 * k
                weight = (u if i else 1-u) * (v if j else 1-v) * (w if k else 1-w)
                out += weight * positions[slot]
    return out


def trilinear_derivatives(positions: np.ndarray, u: float, v: float, w: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    du = np.zeros(3); dv = np.zeros(3); dw = np.zeros(3)
    for k in (0, 1):
        for j in (0, 1):
            for i in (0, 1):
                slot = i + 2*j + 4*k
                bu = u if i else 1-u
                bv = v if j else 1-v
                bw = w if k else 1-w
                du += (1 if i else -1) * bv * bw * positions[slot]
                dv += bu * (1 if j else -1) * bw * positions[slot]
                dw += bu * bv * (1 if k else -1) * positions[slot]
    return du, dv, dw


def sampled_jacobian(positions: np.ndarray) -> tuple[str, float, float]:
    values = []
    for u in np.linspace(0, 1, 5):
        for v in np.linspace(0, 1, 5):
            for w in np.linspace(0, 1, 5):
                du, dv, dw = trilinear_derivatives(positions, float(u), float(v), float(w))
                values.append(float(np.linalg.det(np.column_stack((du, dv, dw)))))
    low, high = min(values), max(values)
    scale = max(1.0, max(abs(low), abs(high)))
    eps = 1e-9 * scale
    if low > eps:
        kind = 'positive_sampled'
    elif high < -eps:
        kind = 'orientation_reversed_sampled'
    elif low < -eps and high > eps:
        kind = 'mixed_sampled'
    else:
        kind = 'near_singular_sampled'
    return kind, low, high


def make_cells(source_records: Sequence[SourceRecord], hv: Mapping[HVID, HV]) -> tuple[list[Cell], dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[SourceRecord]] = collections.defaultdict(list)
    for record in source_records:
        grouped[record.logical_key()].append(record)
    cells: list[Cell] = []
    missing_hvids = 0
    for index, (key, raw) in enumerate(sorted(grouped.items(), key=lambda item: repr(item[0]))):
        rep = raw[0]
        missing = [h for h in rep.hvids if h.node >= 0 and h not in hv]
        if missing:
            missing_hvids += len(missing)
            times = tuple(0 for _ in range(8))
            positions = np.zeros((8, 3))
            gaps = (0, 0, 0, 0)
            route = 'unresolved_missing_hvid'
            jac_kind, jac_min, jac_max = 'not_applicable', math.nan, math.nan
        else:
            # Negative nodes are explicit unresolved placeholders.
            if any(h.node < 0 for h in rep.hvids):
                times = tuple(0 if h.node < 0 else hv[h].time for h in rep.hvids)
                positions = np.asarray([
                    (0.0, 0.0, 0.0) if h.node < 0 else hv[h].position
                    for h in rep.hvids], dtype=float)
                gaps = tuple(times[i+4]-times[i] for i in range(4))  # type: ignore[assignment]
                route = 'unresolved_placeholder'
                jac_kind, jac_min, jac_max = 'not_applicable', math.nan, math.nan
            else:
                times = tuple(hv[h].time for h in rep.hvids)
                positions = np.asarray([hv[h].position for h in rep.hvids], dtype=float)
                gaps = tuple(times[i+4]-times[i] for i in range(4))  # type: ignore[assignment]
                distinct = len(set(rep.hvids))
                if any(value < 0 for value in gaps):
                    route = 'invalid_temporal_order'
                    jac_kind, jac_min, jac_max = 'not_applicable', math.nan, math.nan
                elif distinct == 8 and any(value > 0 for value in gaps):
                    route = 'regular_monotone'
                    jac_kind, jac_min, jac_max = sampled_jacobian(positions)
                else:
                    route = 'quotient_singular'
                    jac_kind, jac_min, jac_max = 'not_applicable', math.nan, math.nan
        cell_id = 'cell-' + hashlib.sha256(repr(key).encode()).hexdigest()[:16]
        cells.append(Cell(
            cell_id=cell_id, representative=rep, raw_records=len(raw),
            hvids=rep.hvids, times=times, positions=positions, gaps=gaps,
            route=route, equality_partition=canonical_partition(rep.hvids),
            sampled_jacobian_class=jac_kind,
            sampled_jacobian_min=jac_min, sampled_jacobian_max=jac_max,
        ))
    return cells, {
        'raw_source_records': len(source_records),
        'logical_cells': len(cells),
        'replication_factor_mean': len(source_records)/max(1, len(cells)),
        'missing_hvid_references': missing_hvids,
    }


def face_saddles_for_cell(cell: Cell) -> list[Saddle]:
    out = []
    for side, slots in TEMPORAL_FACE_SLOTS.items():
        candidate = admissible_saddle(
            [cell.hvids[i] for i in slots], [cell.times[i] for i in slots],
            cell.representative.element,
        )
        if candidate is not None:
            out.append(candidate)
    return out


def sign(value: Fraction) -> int:
    return (value > 0) - (value < 0)


def face_segments(
    corners: Sequence[int], times: Sequence[int], tau: Fraction,
    order_token: int = 0,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    g = [Fraction(times[c], 1) - tau for c in corners]
    if any(value == 0 for value in g):
        raise AuditError('probe hit a vertex time')
    edge_nodes: list[tuple[int, int]] = []
    edge_indices: list[int] = []
    for i in range(4):
        a, b = corners[i], corners[(i+1)%4]
        if sign(g[i]) != sign(g[(i+1)%4]):
            edge_nodes.append(tuple(sorted((a,b))))
            edge_indices.append(i)
    if len(edge_nodes) == 0:
        return []
    if len(edge_nodes) == 2:
        return [(edge_nodes[0], edge_nodes[1])]
    if len(edge_nodes) != 4:
        raise AuditError(f'invalid face crossing count {len(edge_nodes)}')
    q = g[0]*g[2] - g[1]*g[3]
    if q == 0:
        raise AuditError('probe hit a face saddle')
    # e_i lies between corners i and i+1.  q>0 isolates negative diagonal;
    # q<0 isolates positive diagonal.  Either convention is valid if it flips
    # exactly at q=0; this one matches the source-local oracle.
    if q > 0:
        pairs = ((0,1),(2,3)) if g[0] > 0 else ((3,0),(1,2))
    else:
        pairs = ((3,0),(1,2)) if g[0] > 0 else ((0,1),(2,3))
    return [(edge_nodes[a], edge_nodes[b]) for a,b in pairs]


def canonical_hvid_partition(hvids: Sequence[HVID]) -> dict[HVID, str]:
    mapping: dict[HVID, str] = {}
    for h in hvids:
        if h not in mapping:
            mapping[h] = f'h{len(mapping)}'
    return mapping


def crossing_label(edge: tuple[int,int], hvids: Sequence[HVID], canonical=True) -> str:
    mapping = canonical_hvid_partition(hvids) if canonical else {h:h.text() for h in hvids}
    a,b = edge
    ha,hb = mapping[hvids[a]], mapping[hvids[b]]
    return 'E:' + '|'.join(sorted((ha,hb)))


def graph_signature(cell: Cell, tau: Fraction, face_order: Sequence[int] | None = None) -> dict[str, Any]:
    if face_order is None:
        face_order = range(len(FACE_CORNERS))
    edges: list[tuple[str,str]] = []
    for index in face_order:
        for first, second in face_segments(FACE_CORNERS[index], cell.times, tau):
            a = crossing_label(first, cell.hvids)
            b = crossing_label(second, cell.hvids)
            edges.append(tuple(sorted((a,b))))
    edges.sort()
    nodes = sorted({x for edge in edges for x in edge})
    parent = {x:x for x in nodes}
    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a,b in edges:
        ra,rb=find(a),find(b)
        if ra != rb: parent[rb]=ra
    components: dict[str,list[str]] = collections.defaultdict(list)
    for node in nodes: components[find(node)].append(node)
    comp_values = sorted(tuple(sorted(value)) for value in components.values())
    payload = {'nodes':nodes,'edges':edges,'components':comp_values}
    payload['hash'] = sha256_json(payload)
    return payload


def exact_schedule(cell: Cell) -> list[Fraction]:
    values = {Fraction(t,1) for t in cell.times}
    values.update(s.root for s in face_saddles_for_cell(cell))
    return sorted(values)


def registry_saddles(rows: Sequence[Mapping[str,str]]) -> dict[tuple[Any,...], list[Mapping[str,str]]]:
    result: dict[tuple[Any,...], list[Mapping[str,str]]] = collections.defaultdict(list)
    for row in rows:
        if row.get('face_axis_role') != 'temporal_neighbour':
            continue
        hvids = tuple(parse_hvid(row[f'h{i}']) for i in range(4))
        root = Fraction(int(row['root_num']), int(row['root_den']))
        key = (int(row['element']), tuple((h.node,h.group) for h in hvids), root.numerator, root.denominator)
        result[key].append(row)
    return result


def event_limit_label(cell: Cell, edge: tuple[int,int], event: Fraction) -> str:
    partition = canonical_hvid_partition(cell.hvids)
    a,b=edge; ta,tb=cell.times[a],cell.times[b]
    ha,hb=cell.hvids[a],cell.hvids[b]
    if Fraction(ta,1) == event:
        return 'V:' + partition[ha]
    if Fraction(tb,1) == event:
        return 'V:' + partition[hb]
    lo,hi = (a,b) if ta < tb else (b,a)
    tlo,thi=cell.times[lo],cell.times[hi]
    if not (Fraction(tlo,1) < event < Fraction(thi,1)):
        return 'X:' + crossing_label(edge, cell.hvids)
    lam = (event - tlo) / (thi - tlo)
    h0,h1=partition[cell.hvids[lo]],partition[cell.hvids[hi]]
    return f'I:{h0}|{h1}@{lam.numerator}/{lam.denominator}'


def quotient_signature(cell: Cell, event: Fraction, tau: Fraction, face_order: Sequence[int] | None=None) -> dict[str,Any]:
    if face_order is None: face_order=range(6)
    coefficients: collections.Counter[tuple[str,str]] = collections.Counter()
    collapsed: collections.Counter[str] = collections.Counter()
    for fi in face_order:
        segments=face_segments(FACE_CORNERS[fi],cell.times,tau)
        for e0,e1 in segments:
            a=event_limit_label(cell,e0,event); b=event_limit_label(cell,e1,event)
            if a==b:
                collapsed[a]+=1
            else:
                coefficients[tuple(sorted((a,b)))]+=1
    payload={
        'edges':sorted((a,b,count) for (a,b),count in coefficients.items()),
        'collapsed':sorted(collapsed.items()),
    }
    payload['hash']=sha256_json(payload)
    return payload


def subdivide_contract_signature(payload: Mapping[str,Any]) -> str:
    # Insert one synthetic degree-2 vertex per edge copy and contract it again.
    recovered: collections.Counter[tuple[str,str]] = collections.Counter()
    for index,(a,b,count) in enumerate(payload['edges']):
        for copy in range(int(count)):
            synthetic=f'S:{index}:{copy}'
            adjacency={synthetic:[a,b]}
            neighbours=adjacency[synthetic]
            recovered[tuple(sorted(neighbours))]+=1
    normalized={'edges':sorted((a,b,c) for (a,b),c in recovered.items()),'collapsed':payload['collapsed']}
    return sha256_json(normalized)


def source_record_dict(cell: Cell) -> dict[str,Any]:
    rep=cell.representative
    return {
        'cell_id':cell.cell_id,'raw_records':cell.raw_records,
        'source_edge':list(rep.source_edge_key()),'element':rep.element,
        'hvids':[h.text() for h in cell.hvids],
        'times':list(cell.times),'positions':cell.positions.tolist(),
        'gaps':list(cell.gaps),'route':cell.route,
        'equality_partition':list(cell.equality_partition),
        'sampled_jacobian':{
            'class':cell.sampled_jacobian_class,
            'minimum':cell.sampled_jacobian_min,
            'maximum':cell.sampled_jacobian_max,
            'scope':'5x5x5 sampled diagnostic; not a Bernstein proof',
        },
    }


def mesh_topology(vertices: np.ndarray, faces: np.ndarray) -> dict[str,int]:
    if len(faces)==0:
        return {'V':0,'E':0,'F':0,'chi':0,'components':0,'boundary_edges':0,'boundary_loops':0,'nonmanifold_edges':0,'duplicate_faces':0}
    incidence: collections.Counter[tuple[int,int]]=collections.Counter()
    face_count: collections.Counter[tuple[int,int,int]]=collections.Counter()
    for tri in faces:
        a,b,c=(int(x) for x in tri)
        face_count[tuple(sorted((a,b,c)))]+=1
        for x,y in ((a,b),(b,c),(c,a)): incidence[tuple(sorted((x,y)))]+=1
    used=sorted(set(int(x) for x in faces.ravel()))
    parent={x:x for x in used}
    def find(x:int)->int:
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    for tri in faces:
        a,b,c=(int(x) for x in tri)
        for x,y in ((a,b),(b,c)):
            rx,ry=find(x),find(y)
            if rx!=ry: parent[ry]=rx
    boundary=[e for e,n in incidence.items() if n==1]
    bnodes=sorted(set(x for e in boundary for x in e)); bp={x:x for x in bnodes}
    def bf(x:int)->int:
        while bp[x]!=x: bp[x]=bp[bp[x]]; x=bp[x]
        return x
    for a,b in boundary:
        ra,rb=bf(a),bf(b)
        if ra!=rb: bp[rb]=ra
    return {
        'V':len(used),'E':len(incidence),'F':len(faces),
        'chi':len(used)-len(incidence)+len(faces),
        'components':len({find(x) for x in used}),
        'boundary_edges':len(boundary),
        'boundary_loops':len({bf(x) for x in bnodes}) if bnodes else 0,
        'nonmanifold_edges':sum(n>2 for n in incidence.values()),
        'duplicate_faces':sum(n-1 for n in face_count.values() if n>1),
    }


def slice_tet_complex(vertices4: np.ndarray, tets: np.ndarray, tau: float) -> tuple[np.ndarray,np.ndarray]:
    cache: dict[tuple[str,int,int],int]={}; vertices=[]; faces=[]
    tet_edges=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))
    for tet in tets:
        crossing=[]
        for raw_vertex in tet:
            vertex=int(raw_vertex)
            if vertices4[vertex,3] != tau:
                continue
            key=('v',vertex,-1)
            if key not in cache:
                cache[key]=len(vertices)
                vertices.append(vertices4[vertex,:3].copy())
            crossing.append(cache[key])
        for ea,eb in tet_edges:
            ia,ib=int(tet[ea]),int(tet[eb]); ta,tb=vertices4[ia,3],vertices4[ib,3]
            if (ta<tau<tb) or (tb<tau<ta):
                first,second=sorted((ia,ib))
                key=('e',first,second)
                if key not in cache:
                    w=(tau-ta)/(tb-ta)
                    cache[key]=len(vertices); vertices.append((1-w)*vertices4[ia,:3]+w*vertices4[ib,:3])
                crossing.append(cache[key])
        crossing=list(dict.fromkeys(crossing))
        if len(crossing)==3: faces.append(tuple(crossing))
        elif len(crossing)==4:
            p=np.asarray([vertices[i] for i in crossing]); center=p.mean(axis=0)
            _,_,vt=np.linalg.svd(p-center); u,v=vt[0],vt[1]
            angles=np.arctan2((p-center)@v,(p-center)@u)
            ordered=[crossing[i] for i in np.argsort(angles)]
            faces.extend(((ordered[0],ordered[1],ordered[2]),(ordered[0],ordered[2],ordered[3])))
    return np.asarray(vertices,dtype=float),np.asarray(faces,dtype=np.int64)


def tet_gram_volume(vertices4: np.ndarray, tet: Sequence[int]) -> float:
    e=(vertices4[np.asarray(tet)[1:]]-vertices4[int(tet[0])]).T
    det=float(np.linalg.det(e.T@e)); return math.sqrt(max(det,0.0))/6.0


def critical_link_audit(tets: np.ndarray, critical=0) -> dict[str,Any]:
    faces=[]
    for tet in tets:
        if critical not in tet: continue
        faces.append(tuple(int(x) for x in tet if int(x)!=critical))
    # Link is a 2D triangulated disk.
    verts=sorted(set(x for f in faces for x in f)); edges=collections.Counter()
    for a,b,c in faces:
        for x,y in ((a,b),(b,c),(c,a)): edges[tuple(sorted((x,y)))]+=1
    boundary=[e for e,n in edges.items() if n==1]
    chi=len(verts)-len(edges)+len(faces)
    boundary_count=len(boundary)
    return {
        'vertices':len(verts),'edges':len(edges),'faces':len(faces),
        'chi':chi,
        'boundary_edges':boundary_count,'nonmanifold_edges':sum(n>2 for n in edges.values()),
        'is_disk':chi==1 and boundary_count>0 and not any(n>2 for n in edges.values()),
        'is_sphere':chi==2 and boundary_count==0 and not any(n>2 for n in edges.values()),
    }


def complete_production_event_star(
    tets: np.ndarray,
    critical: int = 0,
) -> tuple[np.ndarray,dict[str,Any]]:
    '''Complete the two-tet relative BEB1 kernel to one closed event star.

    The TV3 half-handle is the cone over two adjacent faces of the tetrahedron
    on its four source branches.  The other two branch faces are forced: their
    cones are the unique no-new-vertex completion whose critical link is the
    boundary of that tetrahedron.  Consequently every face containing the
    critical vertex is paired and the only boundary faces are source-labelled.
    '''
    rows=[tuple(int(value) for value in row) for row in np.asarray(tets)]
    if len(rows)!=2 or any(critical not in row for row in rows):
        raise AuditError('BEB1 completion requires two critical tetrahedra')
    branches=sorted({value for row in rows for value in row if value!=critical})
    if len(branches)!=4:
        raise AuditError('BEB1 completion requires exactly four source branches')
    existing={tuple(sorted(value for value in row if value!=critical)) for row in rows}
    sphere_faces={tuple(face) for face in itertools.combinations(branches,3)}
    if len(existing)!=2 or not existing<sphere_faces:
        raise AuditError('BEB1 half-handle is not a two-face branch disk')
    complement=sorted(sphere_faces-existing)
    if len(complement)!=2:
        raise AuditError('BEB1 branch-disk complement is not two faces')
    completed=np.asarray(
        rows+[tuple([critical,*face]) for face in complement],
        dtype=np.int64)
    facet_counts=collections.Counter(
        tuple(sorted(face))
        for tet in completed
        for face in itertools.combinations(map(int,tet),3)
    )
    boundary=sorted(face for face,count in facet_counts.items() if count==1)
    internal=sorted(face for face,count in facet_counts.items() if count==2)
    if any(count>2 for count in facet_counts.values()):
        raise AuditError('completed BEB1 event star has a nonmanifold face')
    if any(critical in face for face in boundary):
        raise AuditError('completed BEB1 event star leaves a critical side face')
    link=critical_link_audit(completed,critical)
    if not link['is_sphere']:
        raise AuditError('completed BEB1 critical link is not a sphere')
    return completed,{
        'completion_kind':'STELLAR_BRANCH_TETRAHEDRON',
        'added_tetrahedra':[list(map(int,[critical,*face])) for face in complement],
        'boundary_faces':[list(map(int,face)) for face in boundary],
        'internal_faces':[list(map(int,face)) for face in internal],
        'critical_link':link,
        'critical_side_faces_remaining':sum(critical in face for face in boundary),
    }


def build_production_half_handle(cell: Cell, saddle: Saddle, face_side: int) -> tuple[np.ndarray,np.ndarray,dict[str,Any]]:
    slots=TEMPORAL_FACE_SLOTS[face_side]
    # Reorder the source face to saddle canonical order.
    source_h=[cell.hvids[i] for i in slots]; source_t=[cell.times[i] for i in slots]
    canonical_h,canonical_t=canonical_cycle(source_h,source_t)
    position_by_h={h:cell.positions[i] for i,h in enumerate(cell.hvids)}
    p=np.asarray([position_by_h[h] for h in canonical_h],dtype=float)
    u=float(saddle.u); v=float(saddle.v)
    pstar=(1-u)*(1-v)*p[0]+u*(1-v)*p[1]+u*v*p[2]+(1-u)*v*p[3]
    tu=(1-v)*(p[1]-p[0])+v*(p[2]-p[3])
    tv=(1-u)*(p[3]-p[0])+u*(p[2]-p[1])
    if np.linalg.norm(tu)<1e-8 or np.linalg.norm(tv)<1e-8 or np.linalg.norm(np.cross(tu,tv))<1e-10:
        raise AuditError('selected production saddle has degenerate spatial tangents')
    tu=tu/np.linalg.norm(tu); tv=tv-tu*np.dot(tu,tv); tv=tv/np.linalg.norm(tv)
    root=float(saddle.root)
    lower_branches=[(h, root-t, t) for h,t in zip(canonical_h,canonical_t) if t<root]
    upper_branches=[(h, t-root, t) for h,t in zip(canonical_h,canonical_t) if t>root]
    if len(lower_branches)!=2 or len(upper_branches)!=2:
        raise AuditError('saddle does not have two lower and two upper branches')
    scale=max(
        [distance for _,distance,_ in lower_branches] +
        [distance for _,distance,_ in upper_branches]
    )
    spatial_scale=max(1e-3,float(np.mean([np.linalg.norm(p[(i+1)%4]-p[i]) for i in range(4)]))/2)
    vertices=np.zeros((5,4),dtype=float)
    branch_hvids = [
        lower_branches[0][0], lower_branches[1][0],
        upper_branches[0][0], upper_branches[1][0],
    ]
    branch_times = [
        lower_branches[0][2], lower_branches[1][2],
        upper_branches[0][2], upper_branches[1][2],
    ]
    # Whole-mesh gluing needs both exact source identity and the corresponding
    # production geometry.  The former abstract tangent-frame coordinates were
    # useful for a local combinatorial witness but could not certify beta_e.
    vertices[0,:3]=pstar
    vertices[0,3]=root
    for index,(hvid,time) in enumerate(zip(branch_hvids,branch_times),start=1):
        vertices[index,:3]=position_by_h[hvid]
        vertices[index,3]=time
    tets=np.asarray([(0,3,1,4),(0,3,4,2)],dtype=np.int64)
    metadata={
        'geometry_kind':'PRODUCTION_SOURCE_HVID_SPACETIME',
        'critical_position':pstar.tolist(),
        'tangent_u':tu.tolist(),
        'tangent_v':tv.tolist(),
        'spatial_scale':spatial_scale,
        'time_scale':scale,
        'block_vertex_roles':['critical'] + [
            'lower_source_branch', 'lower_source_branch',
            'upper_source_branch', 'upper_source_branch',
        ],
        'block_vertex_source_hvids':[None] + [h.text() for h in branch_hvids],
        'block_vertex_exact_times':[
            fraction_json(saddle.root),
            *[fraction_json(Fraction(value,1)) for value in branch_times],
        ],
    }
    return vertices,tets,metadata
