#!/usr/bin/env python3
"""Exact shared-face saddle registry and source-driven universal event construction."""
from __future__ import annotations

import dataclasses
import itertools
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Hashable, Iterable, Mapping, Sequence, Tuple

import numpy as np

import event_complex as ec
import official_geometry_source as og
import quotient_cobordism as qc
import run_universal_quotient_p0 as ru


LOWER_FACE_SLOTS = (0, 2, 3, 1)  # 00,10,11,01 in official slot order
UPPER_FACE_SLOTS = (4, 6, 7, 5)


def canonical_cycle(values: Sequence[int]) -> Tuple[int, ...]:
    cycle = tuple(map(int, values))
    variants = []
    for oriented in (cycle, tuple(reversed(cycle))):
        for offset in range(len(cycle)):
            variants.append(oriented[offset:] + oriented[:offset])
    return min(variants)


def rotate_face(ids: Sequence[int], times: Sequence[int], offset: int, reverse: bool = False):
    pairs = list(zip(map(int, ids), map(int, times)))
    if reverse:
        pairs = list(reversed(pairs))
    pairs = pairs[offset:] + pairs[:offset]
    return tuple(pair[0] for pair in pairs), tuple(pair[1] for pair in pairs)


def q_fraction(values: Sequence[int], tau: Fraction) -> Fraction:
    t00, t10, t11, t01 = map(Fraction, values)
    return (t00 - tau) * (t11 - tau) - (t10 - tau) * (t01 - tau)


def saddle_fraction(values: Sequence[int]) -> Fraction | None:
    t00, t10, t11, t01 = map(int, values)
    denominator = t00 + t11 - t10 - t01
    if denominator == 0:
        return None
    root = Fraction(t00 * t11 - t10 * t01, denominator)
    unique = sorted(set(map(Fraction, values)))
    if len(unique) < 2 or not (unique[0] < root < unique[-1]):
        return None
    span = unique[-1] - unique[0]
    epsilon = min(Fraction(1, 1000), span / 1000)
    for tau in (root - epsilon, root + epsilon):
        signs = [Fraction(v) > tau for v in values]
        if not (signs[0] == signs[2] and signs[1] == signs[3] and signs[0] != signs[1]):
            return None
    if q_fraction(values, root) != 0:
        raise AssertionError("inexact rational saddle")
    return root


def bilinear_saddle_uv(values: Sequence[float]) -> np.ndarray:
    t00, t10, t11, t01 = map(float, values)
    mixed = t00 - t10 - t01 + t11
    if abs(mixed) < 1e-14:
        raise ValueError("no finite bilinear saddle")
    u = -(t01 - t00) / mixed
    v = -(t10 - t00) / mixed
    return np.asarray([u, v], dtype=float)


@dataclass(frozen=True)
class RegistryKey:
    element: int
    hvid_cycle: Tuple[int, ...]


@dataclass
class SaddleEvent:
    key: RegistryKey
    exact_time: Fraction
    incidents: list[str]
    insertions: int = 0


class SaddleRegistry:
    def __init__(self) -> None:
        self.events: Dict[RegistryKey, SaddleEvent] = {}
        self.root_mismatches = 0

    def add(
        self,
        hvid_cycle: Sequence[int],
        time_cycle: Sequence[int],
        incident: str,
        element: int = 0,
    ) -> SaddleEvent | None:
        root = saddle_fraction(time_cycle)
        if root is None:
            return None
        key = RegistryKey(int(element), canonical_cycle(hvid_cycle))
        existing = self.events.get(key)
        if existing is None:
            existing = SaddleEvent(key, root, [], 0)
            self.events[key] = existing
        elif existing.exact_time != root:
            self.root_mismatches += 1
            raise ValueError(f"shared-face root mismatch {existing.exact_time} != {root}")
        existing.incidents.append(str(incident))
        existing.insertions += 1
        return existing

    def schedule(self) -> Tuple[Fraction, ...]:
        return tuple(sorted({event.exact_time for event in self.events.values()}))


@dataclass
class EmbeddedSaddleEvent:
    values: Tuple[int, int, int, int]
    exact_time: Fraction
    saddle_uv: np.ndarray
    source_vertices4: np.ndarray
    source_tetrahedra: np.ndarray
    lower_points: np.ndarray
    lower_triangles: Tuple[Tuple[int, int, int], ...]
    lower_quotient: Dict[int, int]
    upper_points: np.ndarray
    upper_triangles: Tuple[Tuple[int, int, int], ...]
    upper_quotient: Dict[int, int]
    critical_points: np.ndarray
    universal_tetrahedra: Tuple[Tuple[Hashable, Hashable, Hashable, Hashable], ...]


def _segment_midpoints(values: np.ndarray, tau: float) -> list[np.ndarray]:
    intersections = og.boundary_intersections(values, tau)
    pairing = og.decider_pairing(values, tau)
    output = []
    for a, b in pairing:
        output.append(0.5 * (intersections[a] + intersections[b]))
    return output


def build_embedded_saddle_event(
    patch: og.Patch,
    values: Sequence[int],
    side_fraction: float = 0.22,
) -> EmbeddedSaddleEvent:
    values_arr = np.asarray(values, dtype=float)
    exact = saddle_fraction(tuple(map(int, values)))
    if exact is None:
        raise ValueError("not a checkerboard saddle schedule")
    root = float(exact)
    ordered = sorted(values_arr)
    lower_gap = root - ordered[1]
    upper_gap = ordered[2] - root
    delta = side_fraction * min(lower_gap, upper_gap)
    if delta <= 1e-8:
        delta = side_fraction * min(root - ordered[0], ordered[-1] - root)
    lower_uv = _segment_midpoints(values_arr, root - delta)
    upper_uv = _segment_midpoints(values_arr, root + delta)
    if len(lower_uv) != 2 or len(upper_uv) != 2:
        raise RuntimeError("expected two contour components on both sides")
    saddle_uv = bilinear_saddle_uv(values_arr)
    center = og.bilinear_xyz(patch.points, saddle_uv[None, :])[0]
    lower_xyz = og.bilinear_xyz(patch.points, np.asarray(lower_uv))
    upper_xyz = og.bilinear_xyz(patch.points, np.asarray(upper_uv))

    # One source-driven minimal PL-Morse star.  The compiler never receives a
    # shape label; it later sees only the two one-sided complexes and quotient maps.
    vertices4 = np.asarray([
        [center[0], center[1], center[2], 0.0],
        [lower_xyz[0, 0], lower_xyz[0, 1], lower_xyz[0, 2], -1.0],
        [lower_xyz[1, 0], lower_xyz[1, 1], lower_xyz[1, 2], -1.0],
        [upper_xyz[0, 0], upper_xyz[0, 1], upper_xyz[0, 2], 1.0],
        [upper_xyz[1, 0], upper_xyz[1, 1], upper_xyz[1, 2], 1.0],
    ], dtype=float)
    source_tetrahedra = np.asarray(((0, 3, 1, 4), (0, 3, 4, 2)), dtype=int)

    tau_lower, tau_upper = -0.45, 0.45
    lower_positions, lower_polygons = ru.slice_polygons_with_edge_keys(
        vertices4, source_tetrahedra, tau_lower
    )
    upper_positions, upper_polygons = ru.slice_polygons_with_edge_keys(
        vertices4, source_tetrahedra, tau_upper
    )
    lower_q_keys, upper_q_keys, targets = ru.critical_quotients(
        lower_positions, upper_positions, vertices4
    )
    lower_triangles_keys = ru.triangulate_polygons(
        lower_polygons, lower_q_keys, lower_positions
    )
    upper_triangles_keys = ru.triangulate_polygons(
        upper_polygons, upper_q_keys, upper_positions
    )
    lower_points, lower_triangles, lower_index = ru.relabel_surface_keys(
        lower_positions, lower_triangles_keys
    )
    upper_points, upper_triangles, upper_index = ru.relabel_surface_keys(
        upper_positions, upper_triangles_keys
    )
    lower_q = {lower_index[key]: label for key, label in lower_q_keys.items()}
    upper_q = {upper_index[key]: label for key, label in upper_q_keys.items()}
    universal = qc.double_mapping_cobordism(
        lower_triangles, upper_triangles, lower_q, upper_q
    )
    if not ec.audit_volume(universal).valid_relative_3_manifold:
        raise RuntimeError("universal saddle cobordism is not a relative 3-manifold")
    return EmbeddedSaddleEvent(
        values=tuple(map(int, values)),
        exact_time=exact,
        saddle_uv=saddle_uv,
        source_vertices4=vertices4,
        source_tetrahedra=source_tetrahedra,
        lower_points=lower_points,
        lower_triangles=lower_triangles,
        lower_quotient=lower_q,
        upper_points=upper_points,
        upper_triangles=upper_triangles,
        upper_quotient=upper_q,
        critical_points=targets,
        universal_tetrahedra=universal,
    )


def embedded_positions4(event: EmbeddedSaddleEvent) -> Dict[Hashable, np.ndarray]:
    output: Dict[Hashable, np.ndarray] = {}
    for index, point in enumerate(event.lower_points):
        output[("lower", index)] = np.r_[point, -1.0]
    for index, point in enumerate(event.upper_points):
        output[("upper", index)] = np.r_[point, 1.0]
    for index, point in enumerate(event.critical_points):
        output[("critical", index)] = np.r_[point, 0.0]
    return output
