#!/usr/bin/env python3
"""Independent regression validator for the production provenance-v2 registry.

The validator intentionally does not trust the counts or rational values in
the JSON files.  It rebuilds raw observations, logical incidences, canonical
events, and the exact saddle solution from the CSV rows, then compares that
independent result with both JSON reports.

The binary parsers in this module mirror the documented packed provenance-v2
ABI.  They are used by the self-test to make truncated, over-sized, or
misaligned metadata a fail-closed condition without loading the C++ library.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import struct
import sys
import tempfile
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPECTED_COUNTS = {
    "raw_observations": 5,
    "logical_incidences": 2,
    "canonical_events": 1,
}
EXPECTED_HVIDS = ("10:4", "38:0", "76:0", "205:0")
EXPECTED_TIMES = (12, 8, 16, 8)
EXPECTED_ROOT = Fraction(32, 3)
EXPECTED_U = Fraction(1, 3)
EXPECTED_V = Fraction(1, 3)

AXIS_ROLES = {
    0: "temporal_neighbour",
    1: "spatial_ring_j",
    2: "spatial_ring_i",
}
TEMPORAL_AXIS = 0
TEMPORAL_ROLE = AXIS_ROLES[TEMPORAL_AXIS]
LAYOUT_VERSION = 1
PRODUCER_MAPPING = "dual_contouring hp_slot=i+2*j+4*t"

SOURCE_MAGIC = 0x32504842  # "BHP2"
PROCESSED_MAGIC = 0x324D5042  # "BPM2"
PROVENANCE_VERSION = 2
SOURCE_HEADER = struct.Struct("<4IQ")
SOURCE_RECORD = struct.Struct("<26i")
PROCESSED_PREFIX = struct.Struct("<4I3i")
SOURCE_HEADER_SIZE = 24
SOURCE_RECORD_SIZE = 104
PROCESSED_RECORD_SIZE = 132
DEFAULT_MAX_RECORDS = 10_000_000

CSV_COLUMNS = (
    "raw_id",
    "t_group",
    "t_start",
    "sorted_record_index",
    "source_t_group",
    "source_record_index",
    "element",
    "edge_x",
    "edge_y",
    "edge_z",
    "edge_L",
    "edge_tcoord",
    "edge_tL",
    "edge_dir",
    "source_h0",
    "source_h1",
    "source_h2",
    "source_h3",
    "source_h4",
    "source_h5",
    "source_h6",
    "source_h7",
    "face_axis",
    "face_axis_role",
    "face_side",
    "h0",
    "h1",
    "h2",
    "h3",
    "t0",
    "t1",
    "t2",
    "t3",
    "A",
    "B",
    "root_num",
    "root_den",
    "u_num",
    "u_den",
    "v_num",
    "v_den",
    "logical_incidence_id",
    "canonical_event_id",
)


class ValidationError(RuntimeError):
    """Raised when provenance evidence violates the v2 contract."""


@dataclass(frozen=True)
class SourceMetadata:
    source_t_group: int
    source_record_index: int
    edge_coords: tuple[int, int, int]
    edge_L: int
    edge_tcoord: int
    edge_tL: int
    edge_dir: int
    element: int
    hvid_node: tuple[int, ...]
    hvid_group: tuple[int, ...]

    @property
    def hvids(self) -> tuple[str, ...]:
        return tuple(
            f"{node}:{group}"
            for node, group in zip(self.hvid_node, self.hvid_group, strict=True)
        )


@dataclass(frozen=True)
class ProcessedMetadata:
    t_group: int
    t_start: int
    sorted_record_index: int
    source: SourceMetadata


@dataclass(frozen=True)
class DerivedObservation:
    raw_id: str
    logical_incidence_id: str
    canonical_event_id: str
    element: int
    face_axis: int
    face_axis_role: str
    face_side: int
    hvids: tuple[str, str, str, str]
    times: tuple[int, int, int, int]
    A: int
    B: int
    root: Fraction
    u: Fraction
    v: Fraction


def _fail(message: str) -> None:
    raise ValidationError(message)


def _integer(row: Mapping[str, str], field: str) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValidationError(f"CSV field {field!r} is not an integer") from error


def _fraction(row: Mapping[str, str], numerator: str, denominator: str) -> Fraction:
    n = _integer(row, numerator)
    d = _integer(row, denominator)
    if d <= 0:
        _fail(f"CSV fraction {numerator}/{denominator} has non-positive denominator")
    result = Fraction(n, d)
    if result.numerator != n or result.denominator != d:
        _fail(f"CSV fraction {numerator}/{denominator} is not normalized")
    return result


def _parse_hvid(text: str) -> tuple[int, int]:
    pieces = text.split(":")
    if len(pieces) != 2:
        _fail(f"invalid HVID {text!r}")
    try:
        return int(pieces[0]), int(pieces[1])
    except ValueError as error:
        raise ValidationError(f"invalid HVID {text!r}") from error


def _canonical_cycle(
    hvids: Sequence[str], times: Sequence[int]
) -> tuple[tuple[str, str, str, str], tuple[int, int, int, int]]:
    if len(hvids) != 4 or len(times) != 4:
        _fail("a face must contain exactly four HVID/time pairs")
    parsed = tuple(_parse_hvid(value) for value in hvids)
    if len(set(parsed)) != 4:
        _fail("a saddle face must contain four distinct HVIDs")

    paired = tuple(zip(parsed, hvids, times, strict=True))
    candidates: list[tuple[tuple[tuple[int, int], ...], tuple[Any, ...]]] = []
    for cycle in (paired, (paired[0], paired[3], paired[2], paired[1])):
        for offset in range(4):
            rotated = cycle[offset:] + cycle[:offset]
            candidates.append((tuple(item[0] for item in rotated), rotated))
    _, best = min(candidates, key=lambda item: item[0])
    return (
        tuple(item[1] for item in best),  # type: ignore[return-value]
        tuple(int(item[2]) for item in best),  # type: ignore[return-value]
    )


def _event_id(element: int, role: str, hvids: Sequence[str]) -> str:
    return f"element={element};role={role};face={'|'.join(hvids)}"


def _logical_id(
    event_id: str, element: int, edge_fields: Sequence[int]
) -> str:
    return f"{event_id};edge={','.join(str(value) for value in (*edge_fields, element))}"


def _raw_id(row: Mapping[str, str], face_axis: int, face_side: int) -> str:
    return (
        f"cache={_integer(row, 't_group')}:{_integer(row, 't_start')}:"
        f"{_integer(row, 'sorted_record_index')};source="
        f"{_integer(row, 'source_t_group')}:{_integer(row, 'source_record_index')};"
        f"face={face_axis}:{face_side}"
    )


def _layout_corner_index(i: int, j: int, temporal_side: int) -> int:
    """Independent implementation of the producer's documented slot map."""

    if i not in (0, 1) or j not in (0, 1) or temporal_side not in (0, 1):
        _fail("hypercube corner coordinates must be binary")
    return i + 2 * j + 4 * temporal_side


def _temporal_face_slots(side: int) -> tuple[int, int, int, int]:
    return (
        _layout_corner_index(0, 0, side),
        _layout_corner_index(1, 0, side),
        _layout_corner_index(1, 1, side),
        _layout_corner_index(0, 1, side),
    )


def derive_observation(row: Mapping[str, str]) -> DerivedObservation:
    missing = [column for column in CSV_COLUMNS if column not in row]
    if missing:
        _fail(f"CSV is missing columns: {', '.join(missing)}")

    element = _integer(row, "element")
    face_axis = _integer(row, "face_axis")
    face_side = _integer(row, "face_side")
    if face_axis not in AXIS_ROLES:
        _fail(f"invalid typed face axis {face_axis}")
    if face_side not in (0, 1):
        _fail(f"invalid face side {face_side}")
    role = row["face_axis_role"]
    if role != AXIS_ROLES[face_axis]:
        _fail(f"axis/role disagreement: {face_axis} != {role!r}")

    source_hvids = tuple(row[f"source_h{corner}"] for corner in range(8))
    if any(not value for value in source_hvids):
        _fail("CSV lacks the producer's complete eight-slot HVID provenance")
    face_slots = _temporal_face_slots(face_side) if face_axis == TEMPORAL_AXIS else None

    hvids = tuple(row[f"h{corner}"] for corner in range(4))
    times = tuple(_integer(row, f"t{corner}") for corner in range(4))
    canonical_hvids, canonical_times = _canonical_cycle(hvids, times)
    if hvids != canonical_hvids or times != canonical_times:
        _fail("CSV face is not in canonical HVID cycle order")
    if face_slots is not None:
        producer_face = tuple(source_hvids[slot] for slot in face_slots)
        producer_canonical, _ = _canonical_cycle(producer_face, (0, 1, 2, 3))
        if producer_canonical != hvids:
            _fail("typed temporal face is not derivable from producer HVID slots")

    t00, t10, t11, t01 = times
    A = t00 + t11 - t10 - t01
    B = t00 * t11 - t10 * t01
    if A == 0:
        _fail("selected CSV row does not have a finite algebraic root")
    root = Fraction(B, A)
    u = Fraction(t00 - t01, A)
    v = Fraction(t00 - t10, A)
    if not (min(times) < root < max(times)):
        _fail("saddle root is outside the open corner-time envelope")
    if root in (Fraction(value, 1) for value in times):
        _fail("saddle root coincides with a corner time")
    signs = tuple(Fraction(value, 1) - root for value in times)
    if not (signs[0] * signs[2] > 0 and signs[1] * signs[3] > 0 and signs[0] * signs[1] < 0):
        _fail("corner signs do not define an alternating saddle")
    if not (0 < u < 1 and 0 < v < 1):
        _fail("exact critical coordinates are outside the open face")

    if _integer(row, "A") != A or _integer(row, "B") != B:
        _fail("CSV A/B disagree with independently recomputed values")
    if _fraction(row, "root_num", "root_den") != root:
        _fail("CSV root disagrees with independently recomputed value")
    if _fraction(row, "u_num", "u_den") != u:
        _fail("CSV u disagrees with independently recomputed value")
    if _fraction(row, "v_num", "v_den") != v:
        _fail("CSV v disagrees with independently recomputed value")

    event_id = _event_id(element, role, hvids)
    edge = tuple(
        _integer(row, name)
        for name in (
            "edge_x",
            "edge_y",
            "edge_z",
            "edge_L",
            "edge_tcoord",
            "edge_tL",
            "edge_dir",
        )
    )
    logical_id = _logical_id(event_id, element, edge)
    raw_id = _raw_id(row, face_axis, face_side)
    if row["canonical_event_id"] != event_id:
        _fail("canonical_event_id is not derivable from element/typed role/HVID face")
    if row["logical_incidence_id"] != logical_id:
        _fail("logical_incidence_id is not derivable from stable source-edge metadata")
    if row["raw_id"] != raw_id:
        _fail("raw_id is not derivable from cache/source/face provenance")

    return DerivedObservation(
        raw_id=raw_id,
        logical_incidence_id=logical_id,
        canonical_event_id=event_id,
        element=element,
        face_axis=face_axis,
        face_axis_role=role,
        face_side=face_side,
        hvids=hvids,  # type: ignore[arg-type]
        times=times,  # type: ignore[arg-type]
        A=A,
        B=B,
        root=root,
        u=u,
        v=v,
    )


def analyze_rows(rows: Iterable[Mapping[str, str]]) -> dict[str, Any]:
    observations = [derive_observation(row) for row in rows]
    if not observations:
        _fail("CSV contains no accepted saddle observations")
    raw_ids = [observation.raw_id for observation in observations]
    if len(set(raw_ids)) != len(raw_ids):
        _fail("duplicate raw observation identity")

    events: dict[str, dict[str, Any]] = {}
    for observation in observations:
        signature = (
            observation.element,
            observation.face_axis,
            observation.face_axis_role,
            observation.hvids,
            observation.times,
            observation.A,
            observation.B,
            observation.root,
            observation.u,
            observation.v,
        )
        aggregate = events.setdefault(
            observation.canonical_event_id,
            {"signature": signature, "raw": 0, "logical": set()},
        )
        if aggregate["signature"] != signature:
            _fail("one canonical event has inconsistent provenance or exact mathematics")
        aggregate["raw"] += 1
        aggregate["logical"].add(observation.logical_incidence_id)

    logical = {observation.logical_incidence_id for observation in observations}
    return {
        "raw_observations": len(observations),
        "logical_incidences": len(logical),
        "canonical_events": len(events),
        "events": events,
        "observations": observations,
    }


def read_registry_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as source:
        parsed = list(csv.reader(source))
    if not parsed:
        _fail("registry CSV is empty")
    header = tuple(parsed[0])
    if header != CSV_COLUMNS:
        _fail("registry CSV schema does not match provenance-v2")

    rows: list[dict[str, str]] = []
    logical_index = CSV_COLUMNS.index("logical_incidence_id")
    for line_number, values in enumerate(parsed[1:], start=2):
        if len(values) == len(CSV_COLUMNS):
            normalized = values
        elif len(values) > len(CSV_COLUMNS):
            # Older C++ writers did not CSV-quote the comma-separated edge in
            # logical_incidence_id.  Its position is penultimate, so recover it
            # without weakening any semantic validation.
            normalized = (
                values[:logical_index]
                + [",".join(values[logical_index:-1])]
                + [values[-1]]
            )
        else:
            _fail(f"registry CSV line {line_number} is truncated")
        if len(normalized) != len(CSV_COLUMNS):
            _fail(f"registry CSV line {line_number} has ambiguous field alignment")
        rows.append(dict(zip(CSV_COLUMNS, normalized, strict=True)))
    return rows


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValidationError(f"invalid JSON in {path}") from error
    if not isinstance(value, dict):
        _fail(f"{path} must contain one JSON object")
    return value


def _json_fraction(value: Any, field: str) -> Fraction:
    if not isinstance(value, dict):
        _fail(f"selected-event {field} must be an object")
    if set(value) != {"numerator", "denominator"}:
        _fail(f"selected-event {field} has unexpected fields")
    try:
        numerator = int(value["numerator"])
        denominator = int(value["denominator"])
    except (TypeError, ValueError) as error:
        raise ValidationError(f"selected-event {field} is not an integer fraction") from error
    if denominator <= 0:
        _fail(f"selected-event {field} denominator is non-positive")
    result = Fraction(numerator, denominator)
    if (result.numerator, result.denominator) != (numerator, denominator):
        _fail(f"selected-event {field} is not normalized")
    return result


def _json_integer(value: Any, field: str) -> int:
    if isinstance(value, bool):
        _fail(f"JSON field {field} must be an integer, not a boolean")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValidationError(f"JSON field {field} is not an integer") from error
    if isinstance(value, float) and value != result:
        _fail(f"JSON field {field} is not an exact integer")
    return result



def validate_evidence_generic(
    summary_path: Path, selected_path: Path, csv_path: Path
) -> dict[str, Any]:
    """Validate one provenance-v2 run without assuming a historical scene.

    The original regression contract is intentionally scene-specific
    (5 raw -> 2 logical -> 1 canonical at 32/3).  Production provenance is a
    read-only instrumentation layer, so a different deterministic scene may
    legitimately yield different counts and exact events.  This validator
    independently reconstructs every row and checks internal consistency,
    typed temporal provenance, selected-event aggregation, and exact rational
    mathematics without silently treating the historical pilot as universal.
    """

    summary = _read_json_object(summary_path)
    selected = _read_json_object(selected_path)
    analysis = analyze_rows(read_registry_csv(csv_path))

    for field in ("raw_observations", "logical_incidences", "canonical_events"):
        try:
            reported_value = summary[field]
        except KeyError as error:
            raise ValidationError(f"summary lacks integer field {field}") from error
        reported = _json_integer(reported_value, f"summary.{field}")
        if reported != analysis[field]:
            _fail(
                f"summary {field}={reported} but CSV independently gives "
                f"{analysis[field]}"
            )

    if "canonical_shared_events" in summary and _json_integer(
        summary["canonical_shared_events"], "summary.canonical_shared_events"
    ) != analysis["canonical_events"]:
        _fail("canonical_shared_events disagrees with independently rebuilt events")
    if "accepted_saddle_occurrences" in summary and _json_integer(
        summary["accepted_saddle_occurrences"],
        "summary.accepted_saddle_occurrences",
    ) != analysis["raw_observations"]:
        _fail("accepted_saddle_occurrences disagrees with raw observations")
    if _json_integer(
        summary.get("shared_root_mismatches", -1),
        "summary.shared_root_mismatches",
    ) != 0:
        _fail("shared_root_mismatches must equal zero")

    if selected.get("selected") is not True:
        _fail("selected-event JSON does not select an event")
    if selected.get("face_axis") != TEMPORAL_AXIS:
        _fail("selected event is not on the typed temporal-neighbour axis")
    if selected.get("face_axis_role") != TEMPORAL_ROLE:
        _fail("selected event has the wrong typed face role")
    temporal = selected.get("temporal_provenance")
    if not isinstance(temporal, dict):
        _fail("selected event lacks temporal_provenance")
    if temporal.get("layout_version") != LAYOUT_VERSION:
        _fail("selected event has the wrong producer layout version")
    if temporal.get("producer_mapping") != PRODUCER_MAPPING:
        _fail("selected event lacks the dual-contouring slot mapping contract")
    if temporal.get("verified_temporal_face") is not True:
        _fail("selected event was not verified as a temporal face")
    expected_slots = {
        "side_0": list(_temporal_face_slots(0)),
        "side_1": list(_temporal_face_slots(1)),
    }
    if selected.get("producer_temporal_face_slots") != expected_slots:
        _fail("selected event lacks the independently checkable producer face-slot map")

    event_id = selected.get("event_id")
    if not isinstance(event_id, str) or event_id not in analysis["events"]:
        _fail("selected event_id is absent from independently rebuilt events")
    aggregate = analysis["events"][event_id]
    (
        element,
        face_axis,
        face_axis_role,
        hvids,
        times,
        A,
        B,
        root,
        u,
        v,
    ) = aggregate["signature"]

    if _json_integer(selected.get("element"), "selected.element") != element:
        _fail("selected event element disagrees with CSV")
    if selected.get("face_axis") != face_axis or selected.get("face_axis_role") != face_axis_role:
        _fail("selected event face role disagrees with CSV")
    if tuple(selected.get("canonical_hvids", ())) != hvids:
        _fail("selected event HVID face disagrees with CSV")
    if tuple(selected.get("corner_times", ())) != times:
        _fail("selected event corner times disagree with CSV")
    if _json_integer(selected.get("A"), "selected.A") != A or _json_integer(
        selected.get("B"), "selected.B"
    ) != B:
        _fail("selected event A/B disagree with CSV")
    if _json_fraction(selected.get("root"), "root") != root:
        _fail("selected event exact root disagrees with CSV")
    if _json_fraction(selected.get("u"), "u") != u:
        _fail("selected event exact u disagrees with CSV")
    if _json_fraction(selected.get("v"), "v") != v:
        _fail("selected event exact v disagrees with CSV")

    selected_counts = {
        "raw_observations": aggregate["raw"],
        "logical_incidences": len(aggregate["logical"]),
        # This JSON describes one selected canonical event, not the global count.
        "canonical_events": 1,
    }
    for field, expected in selected_counts.items():
        if _json_integer(selected.get(field), f"selected.{field}") != expected:
            _fail(f"selected-event {field} must equal {expected}")

    observations = [
        observation
        for observation in analysis["observations"]
        if observation.canonical_event_id == event_id
    ]
    expected_raw_ids = sorted(observation.raw_id for observation in observations)
    expected_logical_ids = sorted(
        {observation.logical_incidence_id for observation in observations}
    )
    if "raw_ids" in selected and sorted(selected["raw_ids"]) != expected_raw_ids:
        _fail("selected event raw_ids disagree with CSV")
    if "logical_incidence_ids" in selected and sorted(
        selected["logical_incidence_ids"]
    ) != expected_logical_ids:
        _fail("selected event logical_incidence_ids disagree with CSV")

    return {
        "status": "PASS_PROVENANCE_V2_EVIDENCE",
        "global_counts": {
            field: analysis[field]
            for field in ("raw_observations", "logical_incidences", "canonical_events")
        },
        "selected_counts": selected_counts,
        "event_id": event_id,
        "canonical_hvids": list(hvids),
        "corner_times": list(times),
        "root": {"numerator": root.numerator, "denominator": root.denominator},
        "u": {"numerator": u.numerator, "denominator": u.denominator},
        "v": {"numerator": v.numerator, "denominator": v.denominator},
        "temporal_provenance": {
            "layout_version": LAYOUT_VERSION,
            "producer_mapping": PRODUCER_MAPPING,
            "axis": TEMPORAL_AXIS,
            "role": TEMPORAL_ROLE,
            "side_0_slots": list(_temporal_face_slots(0)),
            "side_1_slots": list(_temporal_face_slots(1)),
        },
    }


def validate_historical_evidence(summary_path: Path, selected_path: Path, csv_path: Path) -> dict[str, Any]:
    summary = _read_json_object(summary_path)
    selected = _read_json_object(selected_path)
    analysis = analyze_rows(read_registry_csv(csv_path))

    for field in EXPECTED_COUNTS:
        try:
            reported_value = summary[field]
        except KeyError as error:
            raise ValidationError(f"summary lacks integer field {field}") from error
        reported = _json_integer(reported_value, f"summary.{field}")
        if reported != analysis[field]:
            _fail(f"summary {field}={reported} but CSV independently gives {analysis[field]}")
        if reported != EXPECTED_COUNTS[field]:
            _fail(f"required {field}={EXPECTED_COUNTS[field]}, got {reported}")
    if "canonical_shared_events" in summary and _json_integer(
        summary["canonical_shared_events"], "summary.canonical_shared_events"
    ) != 1:
        _fail("canonical_shared_events must equal one")
    if _json_integer(
        summary.get("shared_root_mismatches", -1), "summary.shared_root_mismatches"
    ) != 0:
        _fail("shared_root_mismatches must equal zero")

    if selected.get("selected") is not True:
        _fail("selected-event JSON does not select an event")
    if selected.get("face_axis") != TEMPORAL_AXIS:
        _fail("selected event is not on the typed temporal-neighbour axis")
    if selected.get("face_axis_role") != TEMPORAL_ROLE:
        _fail("selected event has the wrong typed face role")
    temporal = selected.get("temporal_provenance")
    if not isinstance(temporal, dict):
        _fail("selected event lacks temporal_provenance")
    if temporal.get("layout_version") != LAYOUT_VERSION:
        _fail("selected event has the wrong producer layout version")
    if temporal.get("producer_mapping") != PRODUCER_MAPPING:
        _fail("selected event lacks the dual-contouring slot mapping contract")
    if temporal.get("verified_temporal_face") is not True:
        _fail("selected event was not verified as a temporal face")
    if _temporal_face_slots(0) != (0, 1, 3, 2) or _temporal_face_slots(1) != (4, 5, 7, 6):
        _fail("independent temporal slot derivation is inconsistent")
    if selected.get("producer_temporal_face_slots") != {
        "side_0": [0, 1, 3, 2],
        "side_1": [4, 5, 7, 6],
    }:
        _fail("selected event lacks the independently checkable producer face-slot map")

    selected_hvids = tuple(selected.get("canonical_hvids", ()))
    selected_times = tuple(selected.get("corner_times", ()))
    if selected_hvids != EXPECTED_HVIDS:
        _fail(f"selected HVID face is {selected_hvids}, expected {EXPECTED_HVIDS}")
    if selected_times != EXPECTED_TIMES:
        _fail(f"selected corner times are {selected_times}, expected {EXPECTED_TIMES}")
    if _json_integer(selected.get("A"), "selected.A") != 12 or _json_integer(
        selected.get("B"), "selected.B"
    ) != 128:
        _fail("selected A/B must be 12/128")
    if _json_fraction(selected.get("root"), "root") != EXPECTED_ROOT:
        _fail("selected exact root must be 32/3")
    if _json_fraction(selected.get("u"), "u") != EXPECTED_U:
        _fail("selected exact u must be 1/3")
    if _json_fraction(selected.get("v"), "v") != EXPECTED_V:
        _fail("selected exact v must be 1/3")

    event_id = selected.get("event_id")
    if not isinstance(event_id, str) or event_id not in analysis["events"]:
        _fail("selected event_id is absent from independently rebuilt events")
    aggregate = analysis["events"][event_id]
    if aggregate["raw"] != 5 or len(aggregate["logical"]) != 2:
        _fail("selected event does not independently reproduce 5 raw -> 2 logical")
    for field, expected in EXPECTED_COUNTS.items():
        if _json_integer(selected.get(field), f"selected.{field}") != expected:
            _fail(f"selected-event {field} must equal {expected}")

    observations: list[DerivedObservation] = analysis["observations"]
    if any(observation.face_axis != TEMPORAL_AXIS for observation in observations):
        _fail("a target observation is not typed as temporal-neighbour")
    if any(observation.hvids != EXPECTED_HVIDS for observation in observations):
        _fail("CSV contains an unexpected canonical face")
    if any(observation.times != EXPECTED_TIMES for observation in observations):
        _fail("CSV contains unexpected corner times")
    if any(
        (observation.root, observation.u, observation.v)
        != (EXPECTED_ROOT, EXPECTED_U, EXPECTED_V)
        for observation in observations
    ):
        _fail("CSV exact saddle solution differs across raw observations")

    return {
        "status": "PASS_PROVENANCE_V2_REGRESSION",
        "counts": dict(EXPECTED_COUNTS),
        "event_id": event_id,
        "canonical_hvids": list(EXPECTED_HVIDS),
        "corner_times": list(EXPECTED_TIMES),
        "root": {"numerator": 32, "denominator": 3},
        "u": {"numerator": 1, "denominator": 3},
        "v": {"numerator": 1, "denominator": 3},
        "temporal_provenance": {
            "layout_version": LAYOUT_VERSION,
            "producer_mapping": PRODUCER_MAPPING,
            "axis": TEMPORAL_AXIS,
            "role": TEMPORAL_ROLE,
            "side_0_slots": list(_temporal_face_slots(0)),
            "side_1_slots": list(_temporal_face_slots(1)),
        },
    }


def _decode_source_record(values: Sequence[int]) -> SourceMetadata:
    return SourceMetadata(
        source_t_group=values[0],
        source_record_index=values[1],
        edge_coords=(values[2], values[3], values[4]),
        edge_L=values[5],
        edge_tcoord=values[6],
        edge_tL=values[7],
        edge_dir=values[8],
        element=values[9],
        hvid_node=tuple(values[10:18]),
        hvid_group=tuple(values[18:26]),
    )


def parse_source_sidecar(
    data: bytes, *, expected_t_group: int | None = None, max_records: int = DEFAULT_MAX_RECORDS
) -> list[SourceMetadata]:
    if len(data) < SOURCE_HEADER_SIZE:
        _fail("source provenance header is truncated")
    magic, version, record_size, layout_version, count = SOURCE_HEADER.unpack_from(data)
    if magic != SOURCE_MAGIC or version != PROVENANCE_VERSION:
        _fail("source provenance magic/version mismatch")
    if record_size != SOURCE_RECORD_SIZE or layout_version != LAYOUT_VERSION:
        _fail("source provenance record/layout size mismatch")
    if count > max_records:
        _fail("source provenance record count exceeds the schema limit")
    expected_bytes = SOURCE_HEADER_SIZE + count * SOURCE_RECORD_SIZE
    if len(data) != expected_bytes:
        _fail("source provenance payload is truncated or has trailing bytes")

    records: list[SourceMetadata] = []
    offset = SOURCE_HEADER_SIZE
    for index in range(count):
        record = _decode_source_record(SOURCE_RECORD.unpack_from(data, offset))
        if record.source_record_index != index:
            _fail("source provenance record-index alignment mismatch")
        if expected_t_group is not None and record.source_t_group != expected_t_group:
            _fail("source provenance t_group alignment mismatch")
        records.append(record)
        offset += SOURCE_RECORD_SIZE
    return records


def parse_processed_sidecar(
    data: bytes, *, max_records: int = DEFAULT_MAX_RECORDS
) -> list[ProcessedMetadata]:
    if len(data) % PROCESSED_RECORD_SIZE != 0:
        _fail("processed provenance record is truncated")
    count = len(data) // PROCESSED_RECORD_SIZE
    if count > max_records:
        _fail("processed provenance record count exceeds the schema limit")
    records: list[ProcessedMetadata] = []
    for index in range(count):
        offset = index * PROCESSED_RECORD_SIZE
        magic, version, record_size, layout_version, t_group, t_start, sorted_index = (
            PROCESSED_PREFIX.unpack_from(data, offset)
        )
        if magic != PROCESSED_MAGIC or version != PROVENANCE_VERSION:
            _fail("processed provenance magic/version mismatch")
        if record_size != PROCESSED_RECORD_SIZE or layout_version != LAYOUT_VERSION:
            _fail("processed provenance record/layout size mismatch")
        values = SOURCE_RECORD.unpack_from(data, offset + PROCESSED_PREFIX.size)
        records.append(
            ProcessedMetadata(
                t_group=t_group,
                t_start=t_start,
                sorted_record_index=sorted_index,
                source=_decode_source_record(values),
            )
        )
    return records


def validate_processed_alignment(
    metadata: Sequence[ProcessedMetadata], expected: Sequence[Mapping[str, Any]]
) -> None:
    if len(metadata) != len(expected):
        _fail("primary/metadata processed-record counts differ")
    for index, (record, wanted) in enumerate(zip(metadata, expected, strict=True)):
        for field in ("t_group", "t_start", "sorted_record_index"):
            if getattr(record, field) != int(wanted[field]):
                _fail(f"processed metadata {field} mismatch at record {index}")
        if record.source.element != int(wanted["element"]):
            _fail(f"processed metadata element mismatch at record {index}")
        if record.source.hvids != tuple(wanted["hvids"]):
            _fail(f"processed metadata HVID alignment mismatch at record {index}")


def validate_source_processed_join(
    sources: Sequence[SourceMetadata], processed: Sequence[ProcessedMetadata]
) -> None:
    """Require one exact BPM2 join for every BHP2 source record.

    The embedded source payload is compared as a complete schema object, not
    merely by HVID or record index.  This catches a BPM2 stream that is
    internally well formed but was copied from the wrong BHP2 producer row.
    """

    source_by_key: dict[tuple[int, int], SourceMetadata] = {}
    for source in sources:
        key = (source.source_t_group, source.source_record_index)
        if key in source_by_key:
            _fail("duplicate BHP2 source identity")
        source_by_key[key] = source

    joined: set[tuple[int, int]] = set()
    for record in processed:
        key = (
            record.source.source_t_group,
            record.source.source_record_index,
        )
        source = source_by_key.get(key)
        if source is None:
            _fail("BPM2 record references a missing BHP2 source identity")
        if record.source != source:
            _fail("BPM2 embedded source payload differs from BHP2")
        if key in joined:
            _fail("duplicate BPM2 join for one BHP2 source identity")
        joined.add(key)

    if joined != set(source_by_key):
        _fail("BHP2 source record is missing from the BPM2 join")



def validate_cache_root(cache_root: Path) -> dict[str, Any]:
    """Independently validate all BHP2/BPM2 sidecars in one cache root."""

    source_root = cache_root / "hyperpoly_meta"
    processed_root = cache_root / "processed_hyperpolys"
    source_files = sorted(source_root.glob("*.bin")) if source_root.is_dir() else []
    processed_files = (
        sorted(processed_root.glob("*_hpmeta.bin"))
        if processed_root.is_dir()
        else []
    )
    if not source_files or not processed_files:
        _fail("cache root does not contain both BHP2 and BPM2 provenance sidecars")

    sources: list[SourceMetadata] = []
    for path in source_files:
        try:
            expected_t_group = int(path.stem)
        except ValueError as error:
            raise ValidationError(f"invalid BHP2 file name: {path.name}") from error
        sources.extend(
            parse_source_sidecar(
                path.read_bytes(), expected_t_group=expected_t_group
            )
        )

    processed: list[ProcessedMetadata] = []
    for path in processed_files:
        stem = path.name.removesuffix("_hpmeta.bin")
        pieces = stem.split("_")
        if len(pieces) != 2:
            _fail(f"invalid BPM2 file name: {path.name}")
        try:
            expected_t_group, expected_t_start = map(int, pieces)
        except ValueError as error:
            raise ValidationError(f"invalid BPM2 file name: {path.name}") from error
        records = parse_processed_sidecar(path.read_bytes())
        previous_index: int | None = None
        for record in records:
            if record.t_group != expected_t_group or record.t_start != expected_t_start:
                _fail(f"BPM2 file identity mismatch in {path.name}")
            if previous_index is not None and record.sorted_record_index != previous_index + 1:
                _fail(f"BPM2 sorted record index is not consecutive in {path.name}")
            previous_index = record.sorted_record_index
        processed.extend(records)

    validate_source_processed_join(sources, processed)
    return {
        "status": "PASS_BHP2_BPM2_CACHE_JOIN",
        "source_files": len(source_files),
        "processed_files": len(processed_files),
        "source_records": len(sources),
        "processed_records": len(processed),
        "join_failures": 0,
    }



def _fixture_event_id(element: int = 0) -> str:
    return _event_id(element, TEMPORAL_ROLE, EXPECTED_HVIDS)


def _fixture_row(
    raw_index: int,
    edge: tuple[int, int, int, int, int, int, int],
    *,
    element: int = 0,
) -> dict[str, str]:
    event_id = _fixture_event_id(element)
    side = raw_index % 2
    source_hvids = [f"{300 + index}:0" for index in range(8)]
    for slot, hvid in zip(_temporal_face_slots(side), EXPECTED_HVIDS, strict=True):
        source_hvids[slot] = hvid
    row = {
        "t_group": str(raw_index % 3),
        "t_start": str(7 + raw_index),
        "sorted_record_index": str(100 + raw_index),
        "source_t_group": str(raw_index % 2),
        "source_record_index": str(20 + raw_index),
        "element": str(element),
        "edge_x": str(edge[0]),
        "edge_y": str(edge[1]),
        "edge_z": str(edge[2]),
        "edge_L": str(edge[3]),
        "edge_tcoord": str(edge[4]),
        "edge_tL": str(edge[5]),
        "edge_dir": str(edge[6]),
        **{f"source_h{index}": value for index, value in enumerate(source_hvids)},
        "face_axis": "0",
        "face_axis_role": TEMPORAL_ROLE,
        "face_side": str(side),
        "h0": EXPECTED_HVIDS[0],
        "h1": EXPECTED_HVIDS[1],
        "h2": EXPECTED_HVIDS[2],
        "h3": EXPECTED_HVIDS[3],
        "t0": "12",
        "t1": "8",
        "t2": "16",
        "t3": "8",
        "A": "12",
        "B": "128",
        "root_num": "32",
        "root_den": "3",
        "u_num": "1",
        "u_den": "3",
        "v_num": "1",
        "v_den": "3",
        "canonical_event_id": event_id,
    }
    edge_fields = edge
    row["logical_incidence_id"] = _logical_id(event_id, element, edge_fields)
    row["raw_id"] = _raw_id(row, 0, raw_index % 2)
    return row


def _fixture_rows() -> list[dict[str, str]]:
    edge_a = (1, 2, 3, 4, 5, 6, 0)
    edge_b = (9, 8, 7, 4, 5, 6, 1)
    return [
        _fixture_row(0, edge_a),
        _fixture_row(1, edge_a),
        _fixture_row(2, edge_a),
        _fixture_row(3, edge_b),
        _fixture_row(4, edge_b),
    ]


def _fixture_source_record() -> SourceMetadata:
    return SourceMetadata(
        source_t_group=4,
        source_record_index=0,
        edge_coords=(1, 2, 3),
        edge_L=4,
        edge_tcoord=5,
        edge_tL=6,
        edge_dir=0,
        element=0,
        hvid_node=(10, 38, 76, 205, 300, 301, 302, 303),
        hvid_group=(4, 0, 0, 0, 0, 0, 0, 0),
    )


def _pack_source(record: SourceMetadata) -> bytes:
    return SOURCE_RECORD.pack(
        record.source_t_group,
        record.source_record_index,
        *record.edge_coords,
        record.edge_L,
        record.edge_tcoord,
        record.edge_tL,
        record.edge_dir,
        record.element,
        *record.hvid_node,
        *record.hvid_group,
    )


def _expect_rejected(label: str, operation: Any) -> None:
    try:
        operation()
    except ValidationError:
        return
    _fail(f"self-test expected rejection: {label}")


def run_self_test() -> dict[str, Any]:
    rows = _fixture_rows()
    baseline = analyze_rows(rows)
    if {key: baseline[key] for key in EXPECTED_COUNTS} != EXPECTED_COUNTS:
        _fail("synthetic 5 -> 2 -> 1 fixture did not reproduce")

    shuffled = list(rows)
    random.Random(20260830).shuffle(shuffled)
    reordered = analyze_rows(shuffled)
    if {key: reordered[key] for key in EXPECTED_COUNTS} != EXPECTED_COUNTS:
        _fail("registry aggregation depends on input order")
    if set(baseline["events"]) != set(reordered["events"]):
        _fail("canonical event identity depends on input order")

    split = [dict(row) for row in rows]
    split[-1] = _fixture_row(4, (9, 8, 7, 4, 5, 6, 1), element=1)
    separated = analyze_rows(split)
    if separated["canonical_events"] != 2:
        _fail("identical HVID faces from different elements were incorrectly merged")

    misaligned_row = [dict(row) for row in rows]
    misaligned_row[0]["t0"] = "13"
    _expect_rejected("CSV exact metadata mismatch", lambda: analyze_rows(misaligned_row))

    source = _fixture_source_record()
    source_payload = _pack_source(source)
    source_blob = SOURCE_HEADER.pack(
        SOURCE_MAGIC,
        PROVENANCE_VERSION,
        SOURCE_RECORD_SIZE,
        LAYOUT_VERSION,
        1,
    ) + source_payload
    if parse_source_sidecar(source_blob, expected_t_group=4) != [source]:
        _fail("source sidecar round trip failed")
    _expect_rejected(
        "source header truncation", lambda: parse_source_sidecar(source_blob[:17])
    )
    _expect_rejected(
        "source payload truncation", lambda: parse_source_sidecar(source_blob[:-1])
    )
    huge_header = SOURCE_HEADER.pack(
        SOURCE_MAGIC,
        PROVENANCE_VERSION,
        SOURCE_RECORD_SIZE,
        LAYOUT_VERSION,
        DEFAULT_MAX_RECORDS + 1,
    )
    _expect_rejected("source record-count cap", lambda: parse_source_sidecar(huge_header))

    processed_blob = PROCESSED_PREFIX.pack(
        PROCESSED_MAGIC,
        PROVENANCE_VERSION,
        PROCESSED_RECORD_SIZE,
        LAYOUT_VERSION,
        4,
        7,
        11,
    ) + source_payload
    processed = parse_processed_sidecar(processed_blob)
    expected = [
        {
            "t_group": 4,
            "t_start": 7,
            "sorted_record_index": 11,
            "element": 0,
            "hvids": source.hvids,
        }
    ]
    validate_processed_alignment(processed, expected)
    validate_source_processed_join([source], processed)
    _expect_rejected(
        "processed record truncation", lambda: parse_processed_sidecar(processed_blob[:-1])
    )
    bad_source = replace(source, hvid_node=(11,) + source.hvid_node[1:])
    bad_processed = [replace(processed[0], source=bad_source)]
    _expect_rejected(
        "processed HVID misalignment",
        lambda: validate_processed_alignment(bad_processed, expected),
    )
    _expect_rejected(
        "BHP2/BPM2 embedded-source mismatch",
        lambda: validate_source_processed_join([source], bad_processed),
    )
    _expect_rejected(
        "BHP2/BPM2 missing processed record",
        lambda: validate_source_processed_join([source], []),
    )
    _expect_rejected(
        "BHP2/BPM2 duplicate processed record",
        lambda: validate_source_processed_join([source], [processed[0], processed[0]]),
    )

    with tempfile.TemporaryDirectory(prefix="binoc-provenance-v2-") as directory_text:
        directory = Path(directory_text)
        summary_path = directory / "event_registry_p1_summary.json"
        selected_path = directory / "event_registry_selected_event.json"
        csv_path = directory / "event_registry_p1.csv"
        summary_path.write_text(
            json.dumps(
                {
                    **EXPECTED_COUNTS,
                    "canonical_shared_events": 1,
                    "accepted_saddle_occurrences": 5,
                    "shared_root_mismatches": 0,
                }
            ),
            encoding="utf-8",
        )
        selected_path.write_text(
            json.dumps(
                {
                    "selected": True,
                    "event_id": _fixture_event_id(),
                    "element": 0,
                    "face_axis": 0,
                    "face_axis_role": TEMPORAL_ROLE,
                    "temporal_provenance": {
                        "layout_version": LAYOUT_VERSION,
                        "producer_mapping": PRODUCER_MAPPING,
                        "verified_temporal_face": True,
                    },
                    "producer_temporal_face_slots": {
                        "side_0": [0, 1, 3, 2],
                        "side_1": [4, 5, 7, 6],
                    },
                    "canonical_hvids": list(EXPECTED_HVIDS),
                    "corner_times": list(EXPECTED_TIMES),
                    "A": 12,
                    "B": 128,
                    "root": {"numerator": 32, "denominator": 3},
                    "u": {"numerator": 1, "denominator": 3},
                    "v": {"numerator": 1, "denominator": 3},
                    **EXPECTED_COUNTS,
                }
            ),
            encoding="utf-8",
        )
        with csv_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        validate_historical_evidence(summary_path, selected_path, csv_path)

        # Exercise the as-written C++ encoding too: logical_incidence_id has
        # embedded commas and older writers emit it without RFC 4180 quoting.
        with csv_path.open("w", newline="", encoding="utf-8") as output:
            output.write(",".join(CSV_COLUMNS) + "\n")
            for row in rows:
                output.write(",".join(row[column] for column in CSV_COLUMNS) + "\n")
        validate_historical_evidence(summary_path, selected_path, csv_path)

    return {
        "status": "PASS_PROVENANCE_V2_SELF_TEST",
        "input_order_invariance": True,
        "different_elements_separated": True,
        "csv_metadata_misalignment_rejected": True,
        "source_header_truncation_rejected": True,
        "source_payload_truncation_rejected": True,
        "source_record_count_cap_enforced": True,
        "processed_record_truncation_rejected": True,
        "processed_hvid_misalignment_rejected": True,
        "binary_join_embedded_source_mismatch_rejected": True,
        "binary_join_missing_processed_record_rejected": True,
        "binary_join_duplicate_processed_record_rejected": True,
        "legacy_unquoted_logical_id_validated": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, help="event_registry_p1_summary.json")
    parser.add_argument("--selected", type=Path, help="event_registry_selected_event.json")
    parser.add_argument("--csv", type=Path, help="event_registry_p1.csv")
    parser.add_argument("--cache-root", type=Path, help="cache root containing BHP2/BPM2 sidecars")
    parser.add_argument(
        "--expect-historical-contract",
        action="store_true",
        help="also require the archived 5 -> 2 -> 1 event at root 32/3",
    )
    parser.add_argument("--self-test", action="store_true", help="run isolated synthetic regressions")
    arguments = parser.parse_args(argv)

    supplied = (arguments.summary, arguments.selected, arguments.csv)
    if any(value is not None for value in supplied) and not all(value is not None for value in supplied):
        parser.error("--summary, --selected, and --csv must be supplied together")
    if arguments.expect_historical_contract and not all(value is not None for value in supplied):
        parser.error("--expect-historical-contract requires all three evidence paths")
    if not arguments.self_test and not all(value is not None for value in supplied) and arguments.cache_root is None:
        parser.error("provide --self-test, --cache-root, and/or all three evidence paths")

    try:
        results: list[dict[str, Any]] = []
        if arguments.self_test:
            results.append(run_self_test())
        if arguments.cache_root is not None:
            results.append(validate_cache_root(arguments.cache_root))
        if all(value is not None for value in supplied):
            if arguments.expect_historical_contract:
                results.append(validate_historical_evidence(arguments.summary, arguments.selected, arguments.csv))
            else:
                results.append(validate_evidence_generic(arguments.summary, arguments.selected, arguments.csv))
        for result in results:
            print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, ValidationError) as error:
        print(f"STOP_PROVENANCE_V2_REGRESSION: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
