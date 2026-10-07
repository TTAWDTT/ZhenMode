"""Identity-bound GSHHG point classification for native geometry preparation.

Raster fractions are sampled approximations, never exact polygon areas. The
Antarctic ice-front convention excludes shelf cavities; freshwater lakes are
not marine. These choices must accompany the resulting binary geometry.
"""

from __future__ import annotations

import heapq
import struct
from collections import deque
from pathlib import Path

import numpy as np
from scipy.ndimage import label

from zhenmode.provenance.sources import sha256_file


def _scan_polygon(levels, lon, lat, vertices, level):
    previous = np.roll(vertices, 1, axis=0)
    first_y, second_y = vertices[:, 1], previous[:, 1]
    for j in np.flatnonzero((lat >= first_y.min()) & (lat <= first_y.max())):
        y = lat[j]
        crossing = (first_y > y) != (second_y > y)
        a, b = vertices[crossing], previous[crossing]
        cuts = np.sort(a[:, 0] + (y - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1]))
        if len(cuts) % 2:
            raise ValueError("shoreline polygon has odd scanline intersection count")
        for left, right in cuts.reshape(-1, 2):
            for shift in (-360.0, 0.0, 360.0):
                lo = int(np.searchsorted(lon + shift, left, side="left"))
                hi = int(np.searchsorted(lon + shift, right, side="left"))
                if hi > lo:
                    levels[lo:hi, j] = np.maximum(levels[lo:hi, j], level)


def rasterize_gshhg(path, lon, lat, *, expected_sha256):
    """Classify ordered lon/lat point centres from original binary polygons.

    GSHHG 2.3.7 binary hierarchy: marine=0, land=1, lake=2, island=3,
    pond=4; level5 ice-front Antarctica acts as land and level6 is skipped.
    Greenwich vertices unwrap at 270 degrees for Eurasia, 180 for other
    Greenwich-crossing polygons. Header east bounds are not wrap thresholds:
    lower-resolution vertices may slightly exceed those bounds.

    Returns uint8 levels with lon,lat axis order. Source identity is checked
    before and after streaming; no unverified source or implicit cache reuse.
    """
    path = Path(path)
    x, y = np.asarray(lon), np.asarray(lat)
    for values, lower, upper in ((x, 0.0, 360.0), (y, -90.0, 90.0)):
        if (
            values.ndim != 1
            or not len(values)
            or values.dtype.kind not in "fiu"
            or not np.isfinite(values).all()
            or not np.all(np.diff(values.astype(float)) > 0)
            or np.any(values < lower)
            or np.any(values >= upper)
        ):
            raise ValueError("shoreline point coordinates must be real, finite, ordered in range")
    x, y = x.astype(float), y.astype(float)
    if sha256_file(path) != expected_sha256:
        raise ValueError("shoreline source identity mismatch")
    size = path.stat().st_size
    levels = np.zeros((len(x), len(y)), dtype=np.uint8)
    with path.open("rb") as handle:
        while header := handle.read(44):
            if len(header) != 44:
                raise ValueError("truncated GSHHG header")
            identifier, count, flag, *metadata = struct.unpack(">11i", header)
            level = flag & 255
            if count < 3 or not 1 <= level <= 6 or count > (size - handle.tell()) // 8:
                raise ValueError("invalid/truncated GSHHG polygon")
            raw = handle.read(count * 8)
            if len(raw) != count * 8:
                raise ValueError("truncated GSHHG coordinates")
            vertices = np.frombuffer(raw, dtype=">i4").reshape(count, 2).astype(float) / 1e6
            if np.any(np.abs(vertices[:, 0]) > 360) or np.any(np.abs(vertices[:, 1]) > 90):
                raise ValueError("GSHHG vertices outside geographic range")
            if level == 6:
                continue
            if ((flag >> 16) & 3) & 1:
                threshold = 270.0 if identifier == 0 else 180.0
                vertices[:, 0] = np.where(
                    vertices[:, 0] > threshold, vertices[:, 0] - 360.0, vertices[:, 0]
                )
            if level == 5 and metadata[2] == -90000000:
                if vertices[0, 0] != 180.0 or vertices[-1, 0] != -180.0:
                    raise ValueError("unexpected Antarctic cap endpoints")
                vertices = np.vstack((vertices, [-180.0, -90.0], [180.0, -90.0]))
            _scan_polygon(levels, x, y, vertices, 1 if level == 5 else level)
    if sha256_file(path) != expected_sha256:
        raise ValueError("shoreline source changed during rasterization")
    return levels


def wet_components(wet):
    """Native four-neighbour components: periodic longitude, closed latitude."""
    visited = np.zeros(wet.shape, dtype=bool)
    nx, ny = wet.shape
    groups = []
    for i, j in zip(*np.nonzero(wet), strict=True):
        if visited[i, j]:
            continue
        pending, cells = deque([(i, j)]), []
        visited[i, j] = True
        while pending:
            x, y = pending.popleft()
            cells.append(int(x * ny + y))
            for xx, yy in (((x + 1) % nx, y), ((x - 1) % nx, y), (x, y + 1), (x, y - 1)):
                if 0 <= yy < ny and wet[xx, yy] and not visited[xx, yy]:
                    visited[xx, yy] = True
                    pending.append((xx, yy))
        groups.append(cells)
    return sorted(groups, key=len, reverse=True)


def marine_regions(marine, *, periodic_longitude):
    """Fine sample graph labels, with an optional longitude seam union."""
    regions, count = label(marine, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    parents = np.arange(count + 1)

    def find(k):
        while parents[k] != k:
            parents[k] = parents[parents[k]]
            k = parents[k]
        return k

    if periodic_longitude:
        for left, right in zip(regions[0], regions[-1], strict=True):
            if left and right:
                parents[find(right)] = find(left)
    canonical = np.array([find(k) for k in range(count + 1)], dtype=np.int32)[regions]
    sizes = np.bincount(canonical.ravel())
    sizes[0] = 0
    if not np.any(sizes):
        raise ValueError("shoreline graph has no marine support")
    return canonical, int(np.argmax(sizes))


def marine_native_edges(marine, subdivisions, *, periodic_longitude):
    """Aggregate aligned fine centres into possible cells and face witnesses."""
    if (
        type(subdivisions) is not int
        or subdivisions < 1
        or marine.ndim != 2
        or any(n % subdivisions for n in marine.shape)
    ):
        raise ValueError("fine centres must align exactly with native cell bounds")
    nx, ny = (n // subdivisions for n in marine.shape)
    possible = marine.reshape(nx, subdivisions, ny, subdivisions).any(axis=(1, 3))
    x = (
        (
            marine[subdivisions - 1 :: subdivisions]
            & np.roll(marine, -1, axis=0)[subdivisions - 1 :: subdivisions]
        )
        .reshape(nx, ny, subdivisions)
        .any(-1)
    )
    y = (
        (
            marine[:, subdivisions - 1 :: subdivisions]
            & np.roll(marine, -1, axis=1)[:, subdivisions - 1 :: subdivisions]
        )
        .reshape(nx, subdivisions, ny)
        .any(1)
    )
    y[:, -1] = False
    if not periodic_longitude:
        x[-1] = False
    return possible, x, y


def connect_binary_channels(
    wet, depth_support, possible, x_faces, y_faces, lon, lat, *, targets=None
):
    """Minimum added binary cells then shortest spherical graph route.

    An edge must have a paired fine marine sample witness. This does not give
    horizontal partial areas/openings and cannot certify subcell topology
    preserved when an entire coarse cell merges distinct marine patches.
    """
    wet = np.asarray(wet, dtype=bool).copy()
    possible = np.asarray(possible, dtype=bool) & (depth_support > 0)
    groups = wet_components(wet)
    if not groups:
        raise ValueError("binary geometry has no native wet column")
    nx, ny = wet.shape
    cost, distance = np.full(wet.shape, np.inf), np.full(wet.shape, np.inf)
    parent = np.full(wet.shape, -1, dtype=np.int64)
    heap = []
    for flat in groups[0]:
        i, j = divmod(flat, ny)
        if possible[i, j]:
            cost[i, j] = distance[i, j] = 0.0
            heapq.heappush(heap, (0.0, 0.0, i, j))
    while heap:
        old, length, i, j = heapq.heappop(heap)
        if (old, length) != (cost[i, j], distance[i, j]):
            continue
        for ii, jj, allowed in (
            ((i + 1) % nx, j, x_faces[i, j]),
            ((i - 1) % nx, j, x_faces[(i - 1) % nx, j]),
            (i, j + 1, y_faces[i, j]),
            (i, j - 1, y_faces[i, j - 1] if j > 0 else False),
        ):
            if not allowed or not 0 <= jj < ny or not possible[ii, jj]:
                continue
            count = old + (not wet[ii, jj])
            step = (
                abs(lat[jj] - lat[j])
                if j != jj
                else min(abs(lon[ii] - lon[i]), 360 - abs(lon[ii] - lon[i]))
                * np.cos(np.deg2rad(lat[j]))
            )
            total = length + 6371000.0 * np.deg2rad(step)
            if (count, total) < (cost[ii, jj], distance[ii, jj]):
                cost[ii, jj], distance[ii, jj], parent[ii, jj] = count, total, i * ny + j
                heapq.heappush(heap, (count, total, ii, jj))
    selected = (
        groups[1:] if targets is None else [g for g in groups[1:] if any(t in g for t in targets)]
    )
    records, added = [], set()
    for group in selected:
        candidates = [
            (cost.flat[k], distance.flat[k], k) for k in group if np.isfinite(cost.flat[k])
        ]
        if not candidates:
            records.append({"component_cells": group, "status": "unresolved_no_source_graph_path"})
            continue
        _, _, cell = min(candidates)
        route, seen = [], set()
        while parent.flat[cell] >= 0:
            if cell in seen:
                raise ValueError("cycle in native channel provenance")
            seen.add(cell)
            route.append(int(cell))
            cell = int(parent.flat[cell])
        route.append(int(cell))
        new = [k for k in route if not wet.flat[k]]
        added.update(new)
        records.append(
            {
                "component_cells": group,
                "status": "source_graph_channel",
                "route_native_cells": route,
                "added_cells": new,
            }
        )
    for k in added:
        wet.flat[k] = True
    return wet, records
