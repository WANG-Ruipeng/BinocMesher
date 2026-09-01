#!/usr/bin/env python3
"""Canonical exact-time parser for BinocMesher processed source triangles.

This module mirrors the production run_slicing record traversal without
modifying the cache. It exposes every raw fan triangle together with stable
BPM2 provenance and the effective source-edge VID used by the upstream merge.
"""
from __future__ import annotations

import csv
import json
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np

HV_RECORD_SIZE = 28
PROCESSED_RECORD_SIZE = 132
PROCESSED_PREFIX = struct.Struct("<4I3i")
PROCESSED_MAGIC = 0x324D5042  # BPM2
PROVENANCE_VERSION = 2
PROVENANCE_LAYOUT_VERSION = 1


class ProcessedMeshError(RuntimeError):
    pass


@dataclass(frozen=True, order=True)
class HVID:
    node: int
    group: int

    def text(self) -> str:
        return f"{self.node}:{self.group}"


@dataclass(frozen=True, order=True)
class SourceVID:
    first: HVID
    second: HVID

    @classmethod
    def canonical(cls, first: HVID, second: HVID) -> "SourceVID":
        return cls(*sorted((first, second)))

    def tuple(self) -> tuple[int, int, int, int]:
        return (
            self.first.node,
            self.first.group,
            self.second.node,
            self.second.group,
        )

    def text(self) -> str:
        return (
            f"{self.first.node}:{self.first.group}|"
            f"{self.second.node}:{self.second.group}"
        )


@dataclass(frozen=True, order=True)
class TriangleRef:
    element: int
    t_group: int
    t_start: int
    sorted_record_index: int
    interval_index: int
    face_index: int
    fan_index: int

    def values(self) -> tuple[int, ...]:
        return (
            self.element,
            self.t_group,
            self.t_start,
            self.sorted_record_index,
            self.interval_index,
            self.face_index,
            self.fan_index,
        )


@dataclass(frozen=True)
class Hypervertex:
    position: tuple[float, float, float]
    time: int
    halfspan: int
    in_view: int


@dataclass(frozen=True)
class RawTriangle:
    reference: TriangleRef
    source_vertices: tuple[SourceVID, SourceVID, SourceVID]
    positions: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    in_view: tuple[bool, bool, bool]
    event_record: bool

    def oriented_key(self) -> tuple[int, tuple[SourceVID, ...]]:
        values = self.source_vertices
        candidates = tuple(values[offset:] + values[:offset] for offset in range(3))
        return self.reference.element, min(candidates)

    def unoriented_key(self) -> tuple[int, tuple[SourceVID, ...]]:
        return self.reference.element, tuple(sorted(self.source_vertices))


def _read_counted_vector(
    payload: bytes,
    offset: int,
    item_size: int,
) -> tuple[list[memoryview], int]:
    if offset + 4 > len(payload):
        raise ProcessedMeshError("truncated counted-vector header")
    count = struct.unpack_from("<i", payload, offset)[0]
    offset += 4
    if count < 0:
        raise ProcessedMeshError("negative counted-vector length")
    end = offset + count * item_size
    if end > len(payload):
        raise ProcessedMeshError("truncated counted-vector payload")
    view = memoryview(payload)
    return [
        view[offset + index * item_size: offset + (index + 1) * item_size]
        for index in range(count)
    ], end


def parse_hypervertices(cache_root: Path) -> dict[HVID, Hypervertex]:
    result: dict[HVID, Hypervertex] = {}
    files = sorted((cache_root / "hypervertices").glob("*.bin"))
    if not files:
        raise ProcessedMeshError("hypervertex cache is empty")
    for path in files:
        payload = path.read_bytes()
        if len(payload) < 4:
            raise ProcessedMeshError(f"{path}: truncated record-count header")
        count = struct.unpack_from("<i", payload, 0)[0]
        if count < 0 or len(payload) != 4 + count * HV_RECORD_SIZE:
            raise ProcessedMeshError(f"{path}: invalid hypervertex payload")
        for index in range(count):
            offset = 4 + index * HV_RECORD_SIZE
            hvid = HVID(
                struct.unpack_from("<i", payload, offset)[0],
                struct.unpack_from("<b", payload, offset + 4)[0],
            )
            vertex = Hypervertex(
                tuple(float(value) for value in struct.unpack_from(
                    "<3f", payload, offset + 8
                )),
                struct.unpack_from("<b", payload, offset + 20)[0],
                struct.unpack_from("<b", payload, offset + 21)[0],
                struct.unpack_from("<b", payload, offset + 24)[0],
            )
            previous = result.get(hvid)
            if previous is not None and previous != vertex:
                raise ProcessedMeshError(
                    f"inconsistent duplicate HVID {hvid.text()}"
                )
            result[hvid] = vertex
    return result


def parse_serialized_vid(blob: memoryview) -> SourceVID:
    if len(blob) != 16:
        raise ProcessedMeshError("serialized VID has the wrong size")
    first = HVID(
        struct.unpack_from("<i", blob, 0)[0],
        struct.unpack_from("<b", blob, 4)[0],
    )
    second = HVID(
        struct.unpack_from("<i", blob, 8)[0],
        struct.unpack_from("<b", blob, 12)[0],
    )
    return SourceVID.canonical(first, second)


def _compute_slice_vertex(
    source_vid: SourceVID,
    exact_time: Fraction,
    hypervertices: Mapping[HVID, Hypervertex],
) -> tuple[SourceVID, tuple[float, float, float], bool]:
    first_hvid = source_vid.first
    second_hvid = source_vid.second
    try:
        first = hypervertices[first_hvid]
        second = hypervertices[second_hvid]
    except KeyError as error:
        raise ProcessedMeshError(
            f"processed VID references missing HVID {error.args[0]}"
        ) from error

    if first.time > second.time:
        first_hvid, second_hvid = second_hvid, first_hvid
        first, second = second, first

    t1 = Fraction(first.time, 1)
    t2 = Fraction(second.time, 1)
    clamped = max(t1, min(exact_time, t2))
    effective = clamped
    effective_t1 = t1
    effective_t2 = t2
    if (
        first.in_view != second.in_view
        and first.time + first.halfspan
            <= second.time - second.halfspan
    ):
        if second.in_view:
            effective_t2 -= second.halfspan
        else:
            effective_t1 += first.halfspan
        effective = max(effective_t1, min(clamped, effective_t2))

    p1 = np.asarray(first.position, dtype=np.float64)
    p2 = np.asarray(second.position, dtype=np.float64)
    if effective_t1 != effective_t2:
        weight = float(
            (effective - effective_t1) / (effective_t2 - effective_t1)
        )
        position = (1.0 - weight) * p1 + weight * p2
        if effective == effective_t1:
            effective_vid = SourceVID.canonical(first_hvid, first_hvid)
        elif effective == effective_t2:
            effective_vid = SourceVID.canonical(second_hvid, second_hvid)
        else:
            effective_vid = SourceVID.canonical(first_hvid, second_hvid)
    else:
        if clamped < effective_t1:
            position = p1
            effective_vid = SourceVID.canonical(first_hvid, first_hvid)
        else:
            position = p2
            effective_vid = SourceVID.canonical(second_hvid, second_hvid)

    in_view = not (
        (not first.in_view and clamped < effective_t2)
        or (not second.in_view and clamped > effective_t1)
        or (not first.in_view and not second.in_view)
    )
    return (
        effective_vid,
        tuple(float(value) for value in position),
        bool(in_view),
    )


def _active_groups(
    exact_time: Fraction,
    group_count: int,
    maximum_discrete_time: int,
) -> list[int]:
    if not 0 <= exact_time <= maximum_discrete_time:
        raise ProcessedMeshError("exact time is outside the cache")
    if exact_time == maximum_discrete_time:
        current = group_count - 1
    else:
        current = int(exact_time * group_count // maximum_discrete_time)
    result = []
    while True:
        result.append(current)
        if current == 0:
            break
        current -= current & -current
    return result


def infer_cache_shape(cache_root: Path) -> tuple[int, int]:
    identities = []
    for path in (cache_root / "processed_hyperpolys").glob("*_hpmeta.bin"):
        pieces = path.name.removesuffix("_hpmeta.bin").split("_")
        if len(pieces) != 2:
            continue
        identities.append(tuple(map(int, pieces)))
    if not identities:
        raise ProcessedMeshError("no BPM2 processed streams")
    return max(group for group, _ in identities) + 1, max(
        start for _, start in identities
    ) + 1


def selected_event_rows(cache_root: Path) -> tuple[str, list[dict[str, str]]]:
    selected_path = cache_root / "event_registry_selected_event.json"
    csv_path = cache_root / "event_registry_p1.csv"
    if not selected_path.is_file() or not csv_path.is_file():
        raise ProcessedMeshError("event registry sidecars are missing")
    selected = json.loads(selected_path.read_text())
    if not selected.get("selected"):
        raise ProcessedMeshError("event registry selected no event")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    event_id = str(selected["event_id"])
    group = [row for row in rows if row["canonical_event_id"] == event_id]
    if not group:
        raise ProcessedMeshError("selected event has no registry observations")
    return event_id, group


def _parse_processed_metadata(
    blob: bytes,
    expected_t_group: int,
    expected_t_start: int,
) -> list[int]:
    if len(blob) % PROCESSED_RECORD_SIZE:
        raise ProcessedMeshError("truncated BPM2 stream")
    result = []
    for offset in range(0, len(blob), PROCESSED_RECORD_SIZE):
        magic, version, record_size, layout, t_group, t_start, sorted_index = (
            PROCESSED_PREFIX.unpack_from(blob, offset)
        )
        if (
            magic != PROCESSED_MAGIC
            or version != PROVENANCE_VERSION
            or record_size != PROCESSED_RECORD_SIZE
            or layout != PROVENANCE_LAYOUT_VERSION
            or t_group != expected_t_group
            or t_start != expected_t_start
        ):
            raise ProcessedMeshError("BPM2 schema/file identity mismatch")
        result.append(sorted_index)
    return result


def trace_processed_triangles(
    cache_root: Path,
    exact_time: Fraction,
) -> tuple[list[RawTriangle], dict[str, object]]:
    hypervertices = parse_hypervertices(cache_root)
    event_id, event_rows = selected_event_rows(cache_root)
    group_count, maximum_discrete_time = infer_cache_shape(cache_root)
    active_groups = _active_groups(
        exact_time, group_count, maximum_discrete_time
    )
    active_event_records = {
        (
            int(row["t_group"]),
            int(row["t_start"]),
            int(row["sorted_record_index"]),
        )
        for row in event_rows
        if int(row["t_group"]) in active_groups
    }

    triangles: list[RawTriangle] = []
    processed_root = cache_root / "processed_hyperpolys"
    for t_group in active_groups:
        for t_start in range(maximum_discrete_time):
            if exact_time < t_start - 1:
                break
            primary_path = processed_root / f"{t_group}_{t_start}.bin"
            metadata_path = processed_root / f"{t_group}_{t_start}_hpmeta.bin"
            if not primary_path.is_file():
                continue
            if not metadata_path.is_file():
                raise ProcessedMeshError(
                    f"missing BPM2 stream for {primary_path.name}"
                )
            primary = primary_path.read_bytes()
            metadata = _parse_processed_metadata(
                metadata_path.read_bytes(), t_group, t_start
            )
            offset = 0
            record_index = 0
            while offset < len(primary):
                if record_index >= len(metadata):
                    raise ProcessedMeshError(
                        f"primary stream outlives BPM2 in {primary_path.name}"
                    )
                element = struct.unpack_from("<b", primary, offset)[0]
                offset += 1
                time_blobs, offset = _read_counted_vector(primary, offset, 1)
                times = [struct.unpack_from("<b", value, 0)[0]
                         for value in time_blobs]
                if len(times) < 2 or any(
                    first >= second for first, second in zip(times, times[1:])
                ):
                    raise ProcessedMeshError("invalid processed time sequence")
                expanded = list(times)
                expanded[0] -= 1
                expanded[-1] += 1
                slice_group = -1
                for index, value in enumerate(expanded):
                    if exact_time >= value:
                        slice_group = index
                    else:
                        break
                relevant = 0 <= slice_group < len(expanded) - 1
                sorted_index = metadata[record_index]
                for interval_index in range(len(expanded) - 1):
                    face_index = 0
                    while True:
                        face_blobs, offset = _read_counted_vector(
                            primary, offset, 16
                        )
                        if not face_blobs:
                            break
                        source_vids = [parse_serialized_vid(value)
                                       for value in face_blobs]
                        if relevant and interval_index == slice_group:
                            effective = [
                                _compute_slice_vertex(
                                    source_vid, exact_time, hypervertices
                                )
                                for source_vid in source_vids
                            ]
                            for fan_index in range(len(source_vids) - 2):
                                indices = (0, fan_index + 1, fan_index + 2)
                                triangles.append(RawTriangle(
                                    TriangleRef(
                                        element,
                                        t_group,
                                        t_start,
                                        sorted_index,
                                        interval_index,
                                        face_index,
                                        fan_index,
                                    ),
                                    tuple(effective[index][0]
                                          for index in indices),
                                    tuple(effective[index][1]
                                          for index in indices),
                                    tuple(effective[index][2]
                                          for index in indices),
                                    (t_group, t_start, sorted_index)
                                    in active_event_records,
                                ))
                        face_index += 1
                record_index += 1
            if offset != len(primary) or record_index != len(metadata):
                raise ProcessedMeshError(
                    f"primary/BPM2 record mismatch in {primary_path.name}"
                )
    return triangles, {
        "event_id": event_id,
        "active_groups": active_groups,
        "active_event_records": len(active_event_records),
        "raw_triangles": len(triangles),
        "event_raw_triangles": sum(t.event_record for t in triangles),
        "group_count": group_count,
        "maximum_discrete_time": maximum_discrete_time,
    }


def canonical_face_groups(
    triangles: Sequence[RawTriangle],
) -> dict[tuple[int, tuple[SourceVID, ...]], list[RawTriangle]]:
    grouped: dict[
        tuple[int, tuple[SourceVID, ...]], list[RawTriangle]
    ] = defaultdict(list)
    for triangle in triangles:
        grouped[triangle.oriented_key()].append(triangle)
    return dict(grouped)


def edge_incidence(
    face_keys: Iterable[tuple[int, tuple[SourceVID, ...]]],
) -> Counter[tuple[int, SourceVID, SourceVID]]:
    incidence: Counter[tuple[int, SourceVID, SourceVID]] = Counter()
    for element, vertices in face_keys:
        for first, second in ((0, 1), (1, 2), (2, 0)):
            a, b = sorted((vertices[first], vertices[second]))
            incidence[(element, a, b)] += 1
    return incidence
