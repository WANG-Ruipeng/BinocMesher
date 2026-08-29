#!/usr/bin/env python3
"""Official-demo-geometry P0 for the unified Binoc event compiler.

This experiment uses the exact procedural terrain function and camera model from
BinocMesher's public standalone demo, then injects controlled flat-star and
bilinear-saddle events into visible, curved terrain patches.

It is deliberately *not* a real Binoc cache census: the public repository does
not ship generated hypervertices/hyperpolys.  It asks the next narrower question:
do the local event constructions remain conforming and visually relevant when
embedded in the official demo geometry and viewed through the official camera?
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import find_contours

# -----------------------------------------------------------------------------
# Exact vectorized Perlin implementation used by the public vnoise dependency.
# We only need Noise.noise2 for the official BinocMesher demo geometry.
# -----------------------------------------------------------------------------
GRAD3 = np.array(
    ((1,1,0),(-1,1,0),(1,-1,0),(-1,-1,0),(1,0,1),(-1,0,1),
     (1,0,-1),(-1,0,-1),(0,1,1),(0,-1,1),(0,1,-1),(0,-1,-1),
     (1,0,-1),(-1,0,-1),(0,-1,1),(0,1,1)), dtype=int)
PERM = np.array((
151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
), dtype=np.uint8)

class Noise2:
    def __init__(self) -> None:
        self.perm = np.concatenate([PERM, PERM])

    @staticmethod
    def _lerp(t: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + t * (b - a)

    @staticmethod
    def _grad2(hash_value: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        g = GRAD3[:, :2][hash_value & 15]
        return x * g[..., 0] + y * g[..., 1]

    def _impl(self, x: np.ndarray, y: np.ndarray, repeat_x: int, repeat_y: int, base: int) -> np.ndarray:
        i = np.floor(np.fmod(x, repeat_x)).astype(int)
        j = np.floor(np.fmod(y, repeat_y)).astype(int)
        ii = np.fmod(i + 1, repeat_x).astype(int)
        jj = np.fmod(j + 1, repeat_y).astype(int)
        i = (i & 255) + base
        j = (j & 255) + base
        ii = (ii & 255) + base
        jj = (jj & 255) + base
        xf = x - np.floor(x)
        yf = y - np.floor(y)
        x1 = xf - 1.0
        y1 = yf - 1.0
        fx = xf**3 * (xf * (xf * 6.0 - 15.0) + 10.0)
        fy = yf**3 * (yf * (yf * 6.0 - 15.0) + 10.0)
        A = self.perm[i]
        AA = self.perm[A + j]
        AB = self.perm[A + jj]
        B = self.perm[ii]
        BA = self.perm[B + j]
        BB = self.perm[B + jj]
        return self._lerp(
            fy,
            self._lerp(fx, self._grad2(self.perm[AA], xf, yf), self._grad2(self.perm[BA], x1, yf)),
            self._lerp(fx, self._grad2(self.perm[AB], xf, y1), self._grad2(self.perm[BB], x1, y1)),
        )

    def noise2(self, x: np.ndarray, y: np.ndarray, octaves: int = 1,
               persistence: float = 0.5, lacunarity: float = 2.0,
               repeat_x: int = 1024, repeat_y: int = 1024, base: int = 0) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.shape != y.shape:
            x, y = np.broadcast_arrays(x, y)
        freq = 1.0
        ampl = 1.0
        max_ampl = 0.0
        total = np.zeros_like(x, dtype=float)
        for _ in range(octaves):
            total += self._impl(x * freq, y * freq, int(repeat_x * freq), int(repeat_y * freq), base) * ampl
            max_ampl += ampl
            freq *= lacunarity
            ampl *= persistence
        return total / max_ampl

NOISE = Noise2()

def official_height(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Exact public demo formula: scale=10, height=5, octaves=4.
    return 5.0 * NOISE.noise2(np.asarray(x) / 10.0, np.asarray(y) / 10.0, octaves=4)

# -----------------------------------------------------------------------------
# Official standalone-demo camera
# -----------------------------------------------------------------------------
W, H = 1280, 720
FX = FY = 2000.0
K = np.array([[FX, 0.0, W/2.0], [0.0, FY, H/2.0], [0.0, 0.0, 1.0]])

def official_camera_pose(frame: int) -> np.ndarray:
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, frame * 0.3],
        [0.0, -1.0, 0.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

def project(points: np.ndarray, frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(points, dtype=float)
    ext = np.linalg.inv(official_camera_pose(frame))[:3, :]
    pc = (ext[:, :3] @ p.T + ext[:, 3:4]).T
    depth = pc[:, 2]
    q = (K @ pc.T).T
    uv = q[:, :2] / q[:, 2:3]
    return uv, depth, pc

# -----------------------------------------------------------------------------
# Terrain patches
# -----------------------------------------------------------------------------
@dataclass
class Patch:
    frame: int
    grid_i: int
    grid_j: int
    points: np.ndarray  # order 00,10,11,01
    projected: np.ndarray
    projected_area: float
    curvature: float
    score: float


def polygon_area_2d(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def tri_area3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return 0.5 * float(np.linalg.norm(np.cross(b-a, c-a)))


def quad_curvature(p: np.ndarray) -> float:
    # Twist plus normal disagreement across the two official-diagonal triangles.
    twist = abs(float(p[0,2] - p[1,2] + p[2,2] - p[3,2]))
    n0 = np.cross(p[1]-p[0], p[2]-p[0])
    n1 = np.cross(p[2]-p[0], p[3]-p[0])
    n0 /= max(np.linalg.norm(n0), 1e-12)
    n1 /= max(np.linalg.norm(n1), 1e-12)
    return twist + 0.5 * (1.0 - float(np.clip(np.dot(n0, n1), -1.0, 1.0)))


def collect_patches(frames: Sequence[int], per_frame: int = 40) -> list[Patch]:
    out: list[Patch] = []
    for frame in frames:
        cy = frame * 0.3
        xs = np.linspace(-12.0, 12.0, 65)
        ys = np.linspace(cy + 4.0, cy + 34.0, 81)
        xx, yy = np.meshgrid(xs, ys, indexing='ij')
        zz = official_height(xx, yy)
        candidates: list[Patch] = []
        for i in range(len(xs)-1):
            for j in range(len(ys)-1):
                p = np.array([
                    [xs[i],   ys[j],   zz[i,j]],
                    [xs[i+1], ys[j],   zz[i+1,j]],
                    [xs[i+1], ys[j+1], zz[i+1,j+1]],
                    [xs[i],   ys[j+1], zz[i,j+1]],
                ])
                uv, depth, _ = project(p, frame)
                if np.min(depth) <= 0.25:
                    continue
                if np.any(uv[:,0] < -64) or np.any(uv[:,0] > W+64) or np.any(uv[:,1] < -64) or np.any(uv[:,1] > H+64):
                    continue
                area = polygon_area_2d(uv)
                if area < 8.0:
                    continue
                curv = quad_curvature(p)
                score = math.sqrt(area) * (0.02 + curv)
                candidates.append(Patch(frame, i, j, p, uv, area, curv, score))
        candidates.sort(key=lambda x: x.score, reverse=True)
        # Greedy separation avoids selecting essentially the same local patch repeatedly.
        selected: list[Patch] = []
        for c in candidates:
            if all(abs(c.grid_i-s.grid_i) + abs(c.grid_j-s.grid_j) >= 4 for s in selected):
                selected.append(c)
                if len(selected) >= per_frame:
                    break
        out.extend(selected)
    return out

# -----------------------------------------------------------------------------
# Flat-star / official per-face pyramid experiment
# -----------------------------------------------------------------------------
def canonical_edge(a: int, b: int) -> tuple[int,int]:
    return (a,b) if a < b else (b,a)


def surface_signature(faces: Sequence[Sequence[int]]) -> dict:
    edge_count: collections.Counter[tuple[int,int]] = collections.Counter()
    verts: set[int] = set()
    adj: dict[int,set[int]] = collections.defaultdict(set)
    for f in faces:
        a,b,c = map(int,f)
        verts.update((a,b,c))
        for x,y in ((a,b),(b,c),(c,a)):
            e = canonical_edge(x,y)
            edge_count[e] += 1
            adj[x].add(y); adj[y].add(x)
    seen=set(); comps=0
    for v in verts:
        if v in seen: continue
        comps += 1; stack=[v]; seen.add(v)
        while stack:
            q=stack.pop()
            for r in adj[q]:
                if r not in seen: seen.add(r); stack.append(r)
    return {
        'components': comps,
        'vertices': len(verts),
        'edges': len(edge_count),
        'faces': len(faces),
        'boundary_edges': sum(n==1 for n in edge_count.values()),
        'nonmanifold_edges': sum(n>2 for n in edge_count.values()),
        'euler': len(verts)-len(edge_count)+len(faces),
    }


def flat_star_metrics(patch: Patch, progress_values: Sequence[float]) -> dict:
    p = patch.points
    tri0 = p[[0,1,2]]
    tri1 = p[[0,2,3]]
    c0 = tri0.mean(axis=0)
    c1 = tri1.mean(axis=0)
    # Area-weighted patch centroid gives one shared affine map for the connected disk.
    a0 = tri_area3(*tri0)
    a1 = tri_area3(*tri1)
    cp = (a0*c0 + a1*c1) / max(a0+a1, 1e-15)
    rows=[]
    for s in progress_values:
        copies0 = c0 + s*(p[[0,2]]-c0)
        copies1 = c1 + s*(p[[0,2]]-c1)
        uv0,d0,_ = project(copies0, patch.frame)
        uv1,d1,_ = project(copies1, patch.frame)
        px_gap = np.linalg.norm(uv0-uv1, axis=1)
        depth_gap = np.abs(d0-d1)
        world_gap = np.linalg.norm(copies0-copies1, axis=1)
        shared0 = cp + s*(p[[0,2]]-cp)
        shared1 = cp + s*(p[[0,2]]-cp)
        suv0,sd0,_ = project(shared0, patch.frame)
        suv1,sd1,_ = project(shared1, patch.frame)
        rows.append({
            's': float(s),
            'pixel_gap_max': float(px_gap.max()),
            'pixel_gap_mean': float(px_gap.mean()),
            'depth_gap_max': float(depth_gap.max()),
            'world_gap_max': float(world_gap.max()),
            'shared_pixel_gap_max': float(np.linalg.norm(suv0-suv1,axis=1).max()),
            'shared_depth_gap_max': float(np.abs(sd0-sd1).max()),
        })
    # Exact area-law check for the shared map.
    area_orig = a0+a1
    area_errors=[]
    for s in progress_values:
        q = cp + s*(p-cp)
        area_s = tri_area3(q[0],q[1],q[2]) + tri_area3(q[0],q[2],q[3])
        area_errors.append(abs(area_s-s*s*area_orig)/max(area_orig,1e-15))
    # Area-weighted integral of squared displacement to collapse apex.  For an
    # affine triangle, E[||X-c||^2] is evaluated by deterministic barycentric samples.
    bary = np.array([[i/10,j/10,1-(i+j)/10] for i in range(11) for j in range(11-i)])
    def tri_energy(tri: np.ndarray, apex: np.ndarray) -> float:
        samples = bary @ tri
        return float(np.mean(np.sum((samples-apex)**2,axis=1)))
    e_face = (a0*tri_energy(tri0,c0)+a1*tri_energy(tri1,c1))/max(a0+a1,1e-15)
    e_shared = (a0*tri_energy(tri0,cp)+a1*tri_energy(tri1,cp))/max(a0+a1,1e-15)
    half = min(rows, key=lambda r: abs(r['s']-0.5))
    return {
        'frame': patch.frame,
        'grid_i': patch.grid_i,
        'grid_j': patch.grid_j,
        'projected_area_px2': patch.projected_area,
        'curvature': patch.curvature,
        'pixel_gap_s05': half['pixel_gap_max'],
        'depth_gap_s05': half['depth_gap_max'],
        'world_gap_s05': half['world_gap_max'],
        'pixel_gap_max_window': max(r['pixel_gap_max'] for r in rows),
        'shared_pixel_gap_max': max(r['shared_pixel_gap_max'] for r in rows),
        'shared_depth_gap_max': max(r['shared_depth_gap_max'] for r in rows),
        'area_law_relerr_max': max(area_errors),
        'motion_energy_per_face': e_face,
        'motion_energy_shared': e_shared,
        'motion_energy_ratio_shared_over_face': e_shared/max(e_face,1e-15),
        'progress_rows': rows,
    }


# -----------------------------------------------------------------------------
# Multi-face connected event-star stress on a 2x2 official terrain block
# -----------------------------------------------------------------------------
def official_grid_vertex(frame: int, i: int, j: int) -> np.ndarray:
    cy=frame*0.3
    xs=np.linspace(-12.0,12.0,65); ys=np.linspace(cy+4.0,cy+34.0,81)
    x=float(xs[i]); y=float(ys[j]); z=float(official_height(np.array([x]),np.array([y]))[0])
    return np.array([x,y,z])


def multiface_star_metrics(patch: Patch, s: float=.5) -> dict | None:
    i,j=patch.grid_i,patch.grid_j
    if i+2>=65 or j+2>=81:
        return None
    verts=np.array([official_grid_vertex(patch.frame,i+a,j+b) for a in range(3) for b in range(3)])
    def vid(a:int,b:int)->int: return a*3+b
    faces=[]
    for a in range(2):
        for b in range(2):
            q0,q1,q2,q3=vid(a,b),vid(a+1,b),vid(a+1,b+1),vid(a,b+1)
            faces.extend([(q0,q1,q2),(q0,q2,q3)])
    areas=np.array([tri_area3(*verts[list(f)]) for f in faces])
    cents=np.array([verts[list(f)].mean(0) for f in faces])
    cp=np.average(cents,axis=0,weights=areas)
    edge_faces: dict[tuple[int,int],list[int]]=collections.defaultdict(list)
    for fi,(a,b,c) in enumerate(faces):
        for x,y in ((a,b),(b,c),(c,a)):
            edge_faces[canonical_edge(x,y)].append(fi)
    internal=[(e,fs) for e,fs in edge_faces.items() if len(fs)==2]
    gaps=[]; depth_gaps=[]; world_gaps=[]
    for (a,b),(f0,f1) in internal:
        p0=cents[f0]+s*(verts[[a,b]]-cents[f0])
        p1=cents[f1]+s*(verts[[a,b]]-cents[f1])
        uv0,d0,_=project(p0,patch.frame); uv1,d1,_=project(p1,patch.frame)
        gaps.extend(np.linalg.norm(uv0-uv1,axis=1).tolist())
        depth_gaps.extend(np.abs(d0-d1).tolist())
        world_gaps.extend(np.linalg.norm(p0-p1,axis=1).tolist())
    # Proposed map is identical for all incident faces by construction.
    shared=cp+s*(verts-cp)
    sig_base=surface_signature([(3*k,3*k+1,3*k+2) for k in range(len(faces))])
    sig_shared=surface_signature(faces)
    # Deterministic barycentric motion energy.
    bary=np.array([[a/10,b/10,1-(a+b)/10] for a in range(11) for b in range(11-a)])
    e_face=0.0; e_shared=0.0
    for area,face,c in zip(areas,faces,cents):
        tri=verts[list(face)]; samples=bary@tri
        e_face += area*float(np.mean(np.sum((samples-c)**2,axis=1)))
        e_shared += area*float(np.mean(np.sum((samples-cp)**2,axis=1)))
    e_face/=areas.sum(); e_shared/=areas.sum()
    return {
        'frame':patch.frame,'grid_i':i,'grid_j':j,
        'triangles':len(faces),'internal_edges':len(internal),
        'baseline_components':sig_base['components'],'shared_components':sig_shared['components'],
        'baseline_boundary_edges':sig_base['boundary_edges'],'shared_boundary_edges':sig_shared['boundary_edges'],
        'pixel_gap_internal_median':float(np.median(gaps)),
        'pixel_gap_internal_p95':float(np.percentile(gaps,95)),
        'pixel_gap_internal_max':float(np.max(gaps)),
        'depth_gap_internal_max':float(np.max(depth_gaps)),
        'world_gap_internal_max':float(np.max(world_gaps)),
        'shared_pixel_gap_max':0.0,
        'motion_energy_ratio_shared_over_face':e_shared/max(e_face,1e-15),
    }

# -----------------------------------------------------------------------------
# Bilinear saddle / static triangulation experiment
# -----------------------------------------------------------------------------
UV_CORNERS = np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
BOUNDARY_EDGES = [(0,1),(1,2),(2,3),(3,0)]
TRIS_DIAG_02 = [(0,1,2),(0,2,3)]

def bilinear(vals: np.ndarray, uv: np.ndarray) -> np.ndarray:
    u=uv[...,0]; v=uv[...,1]
    return ((1-u)*(1-v)*vals[0] + u*(1-v)*vals[1] + u*v*vals[2] + (1-u)*v*vals[3])


def bilinear_xyz(points: np.ndarray, uv: np.ndarray) -> np.ndarray:
    u=uv[...,0,None]; v=uv[...,1,None]
    return ((1-u)*(1-v)*points[0] + u*(1-v)*points[1] + u*v*points[2] + (1-u)*v*points[3])


def saddle_time(vals: np.ndarray) -> float:
    t00,t10,t11,t01 = map(float, vals)
    den=t00+t11-t10-t01
    return (t00*t11-t10*t01)/den


def q_decider(vals: np.ndarray, tau: float) -> float:
    t00,t10,t11,t01 = map(float, vals)
    return (t00-tau)*(t11-tau)-(t10-tau)*(t01-tau)


def boundary_intersections(vals: np.ndarray, tau: float) -> dict[int,np.ndarray]:
    f=vals-tau
    out={}
    for ei,(a,b) in enumerate(BOUNDARY_EDGES):
        fa,fb=f[a],f[b]
        if fa==0: out[ei]=UV_CORNERS[a].copy()
        elif fb==0: out[ei]=UV_CORNERS[b].copy()
        elif fa*fb<0:
            w=fa/(fa-fb)
            out[ei]=(1-w)*UV_CORNERS[a]+w*UV_CORNERS[b]
    return out


def decider_pairing(vals: np.ndarray, tau: float) -> tuple[tuple[int,int],tuple[int,int]]:
    # For corners (00,10,11,01): Q<0 joins around corners 00 and 11.
    if q_decider(vals,tau) < 0:
        return ((0,3),(1,2))
    return ((0,1),(2,3))


def reference_contours(vals: np.ndarray, tau: float, n: int=129) -> list[np.ndarray]:
    us=np.linspace(0,1,n); vs=np.linspace(0,1,n)
    uu,vv=np.meshgrid(us,vs,indexing='xy')
    z=bilinear(vals,np.stack([uu,vv],axis=-1))-tau
    curves=[]
    for c in find_contours(z,0.0):
        # c[:,0] is row(v), c[:,1] is column(u)
        uv=np.stack([c[:,1]/(n-1),c[:,0]/(n-1)],axis=-1)
        if len(uv)>=2: curves.append(uv)
    return curves


def edge_id_from_uv(q: np.ndarray, tol: float=2e-2) -> int:
    u,v=q
    ds=[abs(v),abs(u-1),abs(v-1),abs(u)]
    return int(np.argmin(ds))


def pairing_from_curves(curves: Sequence[np.ndarray]) -> tuple[tuple[int,int],...]:
    pairs=[]
    for c in curves:
        e0=edge_id_from_uv(c[0]); e1=edge_id_from_uv(c[-1])
        if e0!=e1: pairs.append(tuple(sorted((e0,e1))))
    return tuple(sorted(set(pairs)))


def sample_pairing_curves(inter: dict[int,np.ndarray], pairing: Sequence[tuple[int,int]], n: int=80) -> list[np.ndarray]:
    curves=[]
    for a,b in pairing:
        aa=inter[a]; bb=inter[b]
        w=np.linspace(0,1,n)[:,None]
        curves.append((1-w)*aa+w*bb)
    return curves


def triangle_contour_curves(vals: np.ndarray, tau: float, n: int=80) -> list[np.ndarray]:
    # Piecewise-linear contour on the fixed 00-11 diagonal.  Curves are sampled
    # in parameter space and later mapped by piecewise-linear spatial triangles.
    f=vals-tau
    segments=[]
    for tri in TRIS_DIAG_02:
        hits=[]
        for ia,ib in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
            fa,fb=f[ia],f[ib]
            if fa==0: hits.append(UV_CORNERS[ia])
            elif fb==0: hits.append(UV_CORNERS[ib])
            elif fa*fb<0:
                w=fa/(fa-fb); hits.append((1-w)*UV_CORNERS[ia]+w*UV_CORNERS[ib])
        # Unique intersections.
        uniq=[]
        for h in hits:
            if not any(np.linalg.norm(h-u)<1e-9 for u in uniq): uniq.append(h)
        if len(uniq)==2:
            w=np.linspace(0,1,n)[:,None]
            segments.append((1-w)*uniq[0]+w*uniq[1])
    return segments


def pairing_from_segments(segments: Sequence[np.ndarray]) -> tuple[tuple[int,int],...]:
    # Merge segment endpoints that lie on the internal diagonal, then identify
    # connected components by endpoint proximity.
    if not segments: return tuple()
    nodes=[]; edges=[]
    def get_node(q: np.ndarray) -> int:
        for i,p in enumerate(nodes):
            if np.linalg.norm(q-p)<1e-6: return i
        nodes.append(q.copy()); return len(nodes)-1
    for s in segments:
        a=get_node(s[0]); b=get_node(s[-1]); edges.append((a,b))
    adj=collections.defaultdict(set)
    for a,b in edges: adj[a].add(b); adj[b].add(a)
    seen=set(); pairs=[]
    for v in range(len(nodes)):
        if v in seen: continue
        stack=[v]; seen.add(v); comp=[]
        while stack:
            q=stack.pop(); comp.append(q)
            for r in adj[q]:
                if r not in seen: seen.add(r); stack.append(r)
        bes=[]
        for q in comp:
            u,vv=nodes[q]
            if min(abs(vv),abs(u-1),abs(vv-1),abs(u))<1e-5:
                bes.append(edge_id_from_uv(nodes[q]))
        bes=sorted(set(bes))
        if len(bes)==2: pairs.append(tuple(bes))
    return tuple(sorted(pairs))


def map_fixed_triangle(points: np.ndarray, uv: np.ndarray) -> np.ndarray:
    # Use the same fixed diagonal as the official terrain mesh.
    u=uv[:,0]; v=uv[:,1]
    out=np.empty((len(uv),3))
    lower = v <= u  # triangle 00-10-11
    # tri 00,10,11: uv = b*(1,0)+c*(1,1), c=v, b=u-v
    b=u[lower]-v[lower]; c=v[lower]; a=1-u[lower]
    out[lower]=a[:,None]*points[0]+b[:,None]*points[1]+c[:,None]*points[2]
    # tri 00,11,01: uv = b*(1,1)+c*(0,1), b=u, c=v-u
    b2=u[~lower]; c2=v[~lower]-u[~lower]; a2=1-v[~lower]
    out[~lower]=a2[:,None]*points[0]+b2[:,None]*points[2]+c2[:,None]*points[3]
    return out


def flatten_curves(curves: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate(curves,axis=0) if curves else np.zeros((0,2))


def symmetric_chamfer(a: np.ndarray,b: np.ndarray) -> tuple[float,float]:
    if len(a)==0 or len(b)==0: return float('inf'),float('inf')
    ta=cKDTree(a); tb=cKDTree(b)
    da=tb.query(a,k=1)[0]; db=ta.query(b,k=1)[0]
    all_d=np.concatenate([da,db])
    return float(all_d.mean()),float(np.percentile(all_d,95))


def random_checkerboard_times(rng: np.random.Generator) -> np.ndarray:
    low0=float(rng.uniform(0.0,1.5)); low1=float(rng.uniform(0.0,2.0))
    high0=float(rng.uniform(4.5,7.5)); high1=float(rng.uniform(3.5,7.0))
    # Ensure a strict checkerboard interval.
    lo=max(low0,low1); hi=min(high0,high1)
    if hi-lo<1.0:
        high0 += 2.0; high1 += 2.0
    return np.array([low0,high0,low1,high1])  # 00,10,11,01


def saddle_metrics(patch: Patch, rng: np.random.Generator) -> list[dict]:
    vals=random_checkerboard_times(rng)
    ts=saddle_time(vals)
    interval_lo=max(vals[0],vals[2]); interval_hi=min(vals[1],vals[3])
    delta=0.42*min(ts-interval_lo,interval_hi-ts)
    if not (delta>1e-5):
        return []
    rows=[]
    for side,tau in [('before',ts-delta),('after',ts+delta)]:
        ref_uv=reference_contours(vals,tau,n=129)
        ref_pair=pairing_from_curves(ref_uv)
        inter=boundary_intersections(vals,tau)
        exact_pair=tuple(sorted(tuple(sorted(x)) for x in decider_pairing(vals,tau)))
        exact_uv=sample_pairing_curves(inter,exact_pair,n=100)
        fixed_uv=triangle_contour_curves(vals,tau,n=100)
        fixed_pair=pairing_from_segments(fixed_uv)
        ref_xyz=[bilinear_xyz(patch.points,c) for c in ref_uv]
        exact_xyz=[bilinear_xyz(patch.points,c) for c in exact_uv]
        fixed_xyz=[map_fixed_triangle(patch.points,c) for c in fixed_uv]
        ref3=np.concatenate(ref_xyz,axis=0); ex3=np.concatenate(exact_xyz,axis=0); fx3=np.concatenate(fixed_xyz,axis=0)
        ref_px,ref_d,_=project(ref3,patch.frame)
        ex_px,ex_d,_=project(ex3,patch.frame)
        fx_px,fx_d,_=project(fx3,patch.frame)
        ex_ch,ex_p95=symmetric_chamfer(ref_px,ex_px)
        fx_ch,fx_p95=symmetric_chamfer(ref_px,fx_px)
        # Nearest-neighbour depth discrepancy in screen space.
        ex_tree=cKDTree(ex_px); fx_tree=cKDTree(fx_px)
        ex_idx=ex_tree.query(ref_px,k=1)[1]; fx_idx=fx_tree.query(ref_px,k=1)[1]
        ex_depth=float(np.mean(np.abs(ref_d-ex_d[ex_idx])))
        fx_depth=float(np.mean(np.abs(ref_d-fx_d[fx_idx])))
        rows.append({
            'frame': patch.frame,'grid_i':patch.grid_i,'grid_j':patch.grid_j,
            'side':side,'tau':float(tau),'saddle_time':float(ts),'q':float(q_decider(vals,tau)),
            'ref_pairing':str(ref_pair),'exact_pairing':str(exact_pair),'fixed_pairing':str(fixed_pair),
            'exact_pairing_correct': int(exact_pair==ref_pair),
            'fixed_pairing_correct': int(fixed_pair==ref_pair),
            'exact_screen_chamfer_px':ex_ch,'fixed_screen_chamfer_px':fx_ch,
            'exact_screen_p95_px':ex_p95,'fixed_screen_p95_px':fx_p95,
            'exact_depth_mae':ex_depth,'fixed_depth_mae':fx_depth,
            'projected_area_px2':patch.projected_area,'curvature':patch.curvature,
            'times':vals.tolist(),
        })
    return rows

# -----------------------------------------------------------------------------
# Reporting / plots
# -----------------------------------------------------------------------------
def qtile(x: Sequence[float],q: float) -> float:
    return float(np.quantile(np.asarray(x,dtype=float),q))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows: return
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader()
        for r in rows:
            rr={k:(json.dumps(v) if isinstance(v,(list,dict)) else v) for k,v in r.items()}
            w.writerow(rr)


def make_plots(out: Path, patches: list[Patch], flat_rows: list[dict], multiface_rows: list[dict], saddle_rows: list[dict]) -> None:
    # Crack histogram.
    vals=np.array([r['pixel_gap_s05'] for r in flat_rows])
    fig,ax=plt.subplots(figsize=(7.5,4.6))
    ax.hist(vals,bins=30)
    ax.axvline(np.median(vals),linestyle='--',label=f'median={np.median(vals):.2f}px')
    ax.set_xlabel('Per-face centroid crack at s=0.5 (pixels)')
    ax.set_ylabel('Visible official-demo patches')
    ax.set_title('Official demo terrain: independent face pyramids break shared edges')
    ax.legend(); fig.tight_layout(); fig.savefig(out/'flat_star_crack_pixels.png',dpi=180); plt.close(fig)

    # Multi-face event-star internal-edge crack distribution.
    mv=np.array([r['pixel_gap_internal_max'] for r in multiface_rows])
    fig,ax=plt.subplots(figsize=(7.5,4.6))
    ax.hist(mv,bins=30)
    ax.axvline(np.median(mv),linestyle='--',label=f'median max={np.median(mv):.2f}px')
    ax.set_xlabel('Max internal-edge crack in 8-triangle star at s=0.5 (px)')
    ax.set_ylabel('Official-demo 2x2 terrain stars')
    ax.set_title('Cracks accumulate across a connected multi-face event star')
    ax.legend(); fig.tight_layout(); fig.savefig(out/'multiface_star_cracks.png',dpi=180); plt.close(fig)

    # Saddle topology and error.
    fixed=[r['fixed_screen_chamfer_px'] for r in saddle_rows]
    exact=[r['exact_screen_chamfer_px'] for r in saddle_rows]
    fig,ax=plt.subplots(figsize=(7.5,4.6))
    ax.boxplot([fixed,exact],tick_labels=['Fixed diagonal','Exact event pairing'],showfliers=False)
    ax.set_ylabel('Screen-space symmetric Chamfer (px)')
    ax.set_title('Controlled saddle events embedded in official demo terrain')
    fig.tight_layout(); fig.savefig(out/'saddle_screen_error.png',dpi=180); plt.close(fig)

    # Motion penalty vs crack removal.
    x=np.array([r['pixel_gap_s05'] for r in flat_rows])
    y=np.array([r['motion_energy_ratio_shared_over_face'] for r in flat_rows])
    fig,ax=plt.subplots(figsize=(7.5,4.8))
    ax.scatter(x,y,s=14,alpha=.65)
    ax.axhline(1.0,linestyle='--')
    ax.set_xlabel('Per-face crack at s=0.5 (px)')
    ax.set_ylabel('Shared-apex / per-face motion energy')
    ax.set_title('Conformity benefit versus geometric-motion cost')
    fig.tight_layout(); fig.savefig(out/'crack_vs_motion_cost.png',dpi=180); plt.close(fig)

    # Representative official geometry case.
    ridx=int(np.argmax([r['pixel_gap_s05'] for r in flat_rows]))
    patch=patches[ridx]; r=flat_rows[ridx]; p=patch.points
    tri0=p[[0,1,2]]; tri1=p[[0,2,3]]; c0=tri0.mean(0); c1=tri1.mean(0)
    a0=tri_area3(*tri0); a1=tri_area3(*tri1); cp=(a0*c0+a1*c1)/(a0+a1)
    s=.5
    base0=c0+s*(p[[0,2]]-c0); base1=c1+s*(p[[0,2]]-c1)
    sh=cp+s*(p[[0,2]]-cp)
    uv0,_,_=project(base0,patch.frame); uv1,_,_=project(base1,patch.frame); uvs,_,_=project(sh,patch.frame)

    # Saddle representative: largest fixed/exact error ratio among wrong fixed cases.
    wrong=[q for q in saddle_rows if not q['fixed_pairing_correct']]
    sr=max(wrong,key=lambda q:q['fixed_screen_chamfer_px']/max(q['exact_screen_chamfer_px'],1e-9)) if wrong else saddle_rows[0]
    sp=next(pp for pp in patches if pp.frame==sr['frame'] and pp.grid_i==sr['grid_i'] and pp.grid_j==sr['grid_j'])
    vals=np.array(sr['times']); tau=sr['tau']
    ref=reference_contours(vals,tau,129); inter=boundary_intersections(vals,tau)
    ex=sample_pairing_curves(inter,decider_pairing(vals,tau),100); fx=triangle_contour_curves(vals,tau,100)
    ref_px=[project(bilinear_xyz(sp.points,c),sp.frame)[0] for c in ref]
    ex_px=[project(bilinear_xyz(sp.points,c),sp.frame)[0] for c in ex]
    fx_px=[project(map_fixed_triangle(sp.points,c),sp.frame)[0] for c in fx]

    fig,axs=plt.subplots(1,3,figsize=(15,4.6))
    axs[0].plot(p[[0,1,2,3,0],0],p[[0,1,2,3,0],1],'-o')
    axs[0].set_aspect('equal'); axs[0].set_title(f'Official terrain quad, frame {patch.frame}')
    axs[0].set_xlabel('world x'); axs[0].set_ylabel('world y')
    axs[1].plot(uv0[:,0],uv0[:,1],'o-',label='face A copy')
    axs[1].plot(uv1[:,0],uv1[:,1],'o-',label='face B copy')
    axs[1].plot(uvs[:,0],uvs[:,1],'x--',label='shared event star')
    axs[1].invert_yaxis(); axs[1].set_aspect('equal'); axs[1].legend(fontsize=8)
    axs[1].set_title(f's=0.5 crack: {r["pixel_gap_s05"]:.2f}px')
    for c in ref_px: axs[2].plot(c[:,0],c[:,1],linewidth=3,label='dense bilinear reference' if c is ref_px[0] else None)
    for c in fx_px: axs[2].plot(c[:,0],c[:,1],'--',label='fixed diagonal' if c is fx_px[0] else None)
    for c in ex_px: axs[2].plot(c[:,0],c[:,1],':',linewidth=2.5,label='exact event pairing' if c is ex_px[0] else None)
    axs[2].invert_yaxis(); axs[2].set_aspect('equal'); axs[2].legend(fontsize=8)
    axs[2].set_title(f'Saddle {sr["side"]}: fixed wrong, exact correct')
    fig.tight_layout(); fig.savefig(out/'official_geometry_examples.png',dpi=200); plt.close(fig)



def export_assets(out: Path, patches: list[Patch], flat_rows: list[dict], saddle_rows: list[dict]) -> None:
    assets=out/'assets'; assets.mkdir(parents=True,exist_ok=True)
    def write_obj(path: Path, verts: np.ndarray, faces=None, lines=None) -> None:
        with path.open('w',encoding='utf-8') as f:
            for v in verts:
                f.write(f"v {v[0]:.9f} {v[1]:.9f} {v[2]:.9f}\n")
            if faces:
                for face in faces:
                    f.write('f '+' '.join(str(int(i)+1) for i in face)+'\n')
            if lines:
                for line in lines:
                    f.write('l '+' '.join(str(int(i)+1) for i in line)+'\n')

    idx=int(np.argmax([r['pixel_gap_s05'] for r in flat_rows]))
    patch=patches[idx]; q=patch.points
    tri0=q[[0,1,2]]; tri1=q[[0,2,3]]; c0=tri0.mean(0); c1=tri1.mean(0)
    a0=tri_area3(*tri0); a1=tri_area3(*tri1); cp=(a0*c0+a1*c1)/(a0+a1); ss=.5
    baseline=np.vstack([c0+ss*(tri0-c0),c1+ss*(tri1-c1)])
    shared=cp+ss*(q-cp)
    write_obj(assets/'official_per_face_s05.obj',baseline,[(0,1,2),(3,4,5)])
    write_obj(assets/'shared_event_star_s05.obj',shared,[(0,1,2),(0,2,3)])
    write_obj(assets/'original_official_quad.obj',q,[(0,1,2),(0,2,3)])

    wrong=[r for r in saddle_rows if not r['fixed_pairing_correct']]
    sr=max(wrong,key=lambda r:r['fixed_screen_chamfer_px']/max(r['exact_screen_chamfer_px'],1e-12)) if wrong else saddle_rows[0]
    lookup={(p.frame,p.grid_i,p.grid_j):p for p in patches}
    sp=lookup[(int(sr['frame']),int(sr['grid_i']),int(sr['grid_j']))]
    vals=np.array(sr['times'],dtype=float); tau=float(sr['tau'])
    ref=reference_contours(vals,tau,129); inter=boundary_intersections(vals,tau)
    exact=sample_pairing_curves(inter,decider_pairing(vals,tau),100)
    fixed=triangle_contour_curves(vals,tau,100)
    def curve_obj(path: Path, curves: Sequence[np.ndarray], mapper) -> None:
        verts=[]; lines=[]
        for c in curves:
            w=mapper(c); st=len(verts); verts.extend(w); lines.append(list(range(st,st+len(w))))
        write_obj(path,np.asarray(verts),lines=lines)
    curve_obj(assets/'saddle_dense_reference.obj',ref,lambda uv:bilinear_xyz(sp.points,uv))
    curve_obj(assets/'saddle_exact_pairing.obj',exact,lambda uv:bilinear_xyz(sp.points,uv))
    curve_obj(assets/'saddle_fixed_diagonal.obj',fixed,lambda uv:map_fixed_triangle(sp.points,uv))
    meta={'flat_patch':{k:v for k,v in flat_rows[idx].items() if k!='progress_rows'},'saddle_case':sr}
    (assets/'representative_metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')

def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'results')
    ap.add_argument('--per-frame',type=int,default=40)
    ap.add_argument('--seed',type=int,default=20260827)
    args=ap.parse_args(); out=args.out; out.mkdir(parents=True,exist_ok=True)
    frames=[0,80,160,240,320,400]
    patches=collect_patches(frames,args.per_frame)
    if len(patches)<60:
        raise RuntimeError(f'not enough visible patches: {len(patches)}')
    progress=[.1,.25,.5,.75,.9]
    flat=[flat_star_metrics(p,progress) for p in patches]
    multiface=[r for p in patches if (r:=multiface_star_metrics(p,.5)) is not None]
    rng=np.random.default_rng(args.seed)
    saddle=[]
    for p in patches:
        saddle.extend(saddle_metrics(p,rng))
    if not saddle:
        raise RuntimeError('no saddle trials')

    flat_sig_official=surface_signature([(0,1,2),(3,4,5)])
    flat_sig_shared=surface_signature([(0,1,2),(0,2,3)])
    fixed_correct=np.array([r['fixed_pairing_correct'] for r in saddle],dtype=float)
    exact_correct=np.array([r['exact_pairing_correct'] for r in saddle],dtype=float)
    wrong=[r for r in saddle if not r['fixed_pairing_correct']]
    summary={
        'experiment':'Official standalone-demo geometry controlled-event P0',
        'scope':{
            'geometry':'exact public demo vnoise height field',
            'camera':'exact public demo intrinsics and 480-frame pose formula',
            'frames':frames,
            'patches':len(patches),
            'saddle_side_trials':len(saddle),
            'multiface_star_trials':len(multiface),
            'real_binoc_cache':False,
            'controlled_events_injected':True,
        },
        'flat_star':{
            'per_face_signature':flat_sig_official,
            'shared_star_signature':flat_sig_shared,
            'pixel_gap_s05_median':qtile([r['pixel_gap_s05'] for r in flat],.5),
            'pixel_gap_s05_p95':qtile([r['pixel_gap_s05'] for r in flat],.95),
            'pixel_gap_s05_max':max(r['pixel_gap_s05'] for r in flat),
            'depth_gap_s05_median':qtile([r['depth_gap_s05'] for r in flat],.5),
            'shared_pixel_gap_max':max(r['shared_pixel_gap_max'] for r in flat),
            'shared_depth_gap_max':max(r['shared_depth_gap_max'] for r in flat),
            'area_law_relerr_max':max(r['area_law_relerr_max'] for r in flat),
            'motion_energy_ratio_median':qtile([r['motion_energy_ratio_shared_over_face'] for r in flat],.5),
            'motion_energy_ratio_p95':qtile([r['motion_energy_ratio_shared_over_face'] for r in flat],.95),
        },
        'multiface_star':{
            'trials':len(multiface),
            'triangles_per_star':8,
            'baseline_components':int(multiface[0]['baseline_components']),
            'shared_components':int(multiface[0]['shared_components']),
            'pixel_gap_internal_max_median':qtile([r['pixel_gap_internal_max'] for r in multiface],.5),
            'pixel_gap_internal_max_p95':qtile([r['pixel_gap_internal_max'] for r in multiface],.95),
            'pixel_gap_internal_global_max':max(r['pixel_gap_internal_max'] for r in multiface),
            'shared_pixel_gap_max':max(r['shared_pixel_gap_max'] for r in multiface),
            'motion_energy_ratio_median':qtile([r['motion_energy_ratio_shared_over_face'] for r in multiface],.5),
        },
        'saddle':{
            'exact_pairing_accuracy':float(exact_correct.mean()),
            'fixed_diagonal_pairing_accuracy':float(fixed_correct.mean()),
            'wrong_fixed_trials':len(wrong),
            'fixed_screen_chamfer_median_px':qtile([r['fixed_screen_chamfer_px'] for r in saddle],.5),
            'exact_screen_chamfer_median_px':qtile([r['exact_screen_chamfer_px'] for r in saddle],.5),
            'wrong_fixed_screen_chamfer_median_px':qtile([r['fixed_screen_chamfer_px'] for r in wrong],.5),
            'wrong_exact_screen_chamfer_median_px':qtile([r['exact_screen_chamfer_px'] for r in wrong],.5),
            'wrong_fixed_p95_median_px':qtile([r['fixed_screen_p95_px'] for r in wrong],.5),
            'wrong_exact_p95_median_px':qtile([r['exact_screen_p95_px'] for r in wrong],.5),
            'wrong_fixed_depth_mae_median':qtile([r['fixed_depth_mae'] for r in wrong],.5),
            'wrong_exact_depth_mae_median':qtile([r['exact_depth_mae'] for r in wrong],.5),
        },
        'verdict':'GO_OFFICIAL_GEOMETRY_EMBEDDING; NEXT_TRUE_CACHE_CENSUS',
    }
    write_csv(out/'flat_star_trials.csv',[{k:v for k,v in r.items() if k!='progress_rows'} for r in flat])
    write_csv(out/'multiface_star_trials.csv',multiface)
    write_csv(out/'saddle_trials.csv',saddle)
    with (out/'summary.json').open('w',encoding='utf-8') as f: json.dump(summary,f,indent=2,ensure_ascii=False)
    # Derived statistics kept separate from the primary endpoint summary.
    wrong_rows=[r for r in saddle if not r['fixed_pairing_correct']]
    analysis={
        'flat':{
            'pixel_gap_s05_by_frame':{str(fr):float(np.median([r['pixel_gap_s05'] for r in flat if r['frame']==fr])) for fr in frames},
            'pixel_gap_projected_area_correlation':float(np.corrcoef([r['pixel_gap_s05'] for r in flat],[r['projected_area_px2'] for r in flat])[0,1]),
            'pixel_gap_curvature_correlation':float(np.corrcoef([r['pixel_gap_s05'] for r in flat],[r['curvature'] for r in flat])[0,1]),
            'global_max_window_gap_px':float(max(r['pixel_gap_max_window'] for r in flat)),
            'motion_penalty_median_percent':float((np.median([r['motion_energy_ratio_shared_over_face'] for r in flat])-1)*100),
            'motion_penalty_p95_percent':float((np.quantile([r['motion_energy_ratio_shared_over_face'] for r in flat],.95)-1)*100),
        },
        'multiface':{
            'pixel_gap_internal_max_median':float(np.median([r['pixel_gap_internal_max'] for r in multiface])),
            'pixel_gap_internal_max_p95':float(np.quantile([r['pixel_gap_internal_max'] for r in multiface],.95)),
            'motion_penalty_median_percent':float((np.median([r['motion_energy_ratio_shared_over_face'] for r in multiface])-1)*100),
        },
        'saddle':{
            'overall_chamfer_reduction_percent':float((1-np.median([r['exact_screen_chamfer_px'] for r in saddle])/np.median([r['fixed_screen_chamfer_px'] for r in saddle]))*100),
            'wrong_chamfer_reduction_percent':float((1-np.median([r['exact_screen_chamfer_px'] for r in wrong_rows])/np.median([r['fixed_screen_chamfer_px'] for r in wrong_rows]))*100),
            'wrong_p95_reduction_percent':float((1-np.median([r['exact_screen_p95_px'] for r in wrong_rows])/np.median([r['fixed_screen_p95_px'] for r in wrong_rows]))*100),
            'wrong_depth_reduction_percent':float((1-np.median([r['exact_depth_mae'] for r in wrong_rows])/np.median([r['fixed_depth_mae'] for r in wrong_rows]))*100),
        },
    }
    (out/'analysis.json').write_text(json.dumps(analysis,indent=2,ensure_ascii=False),encoding='utf-8')
    make_plots(out,patches,flat,multiface,saddle)
    export_assets(out,patches,flat,saddle)
    print(json.dumps(summary,indent=2,ensure_ascii=False))

if __name__=='__main__':
    main()
