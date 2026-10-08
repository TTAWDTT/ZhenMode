"""Independent exact contact oracle, archived with naming/formatting cleanup.

No product geometry/operator helpers are imported. Invoke through the resource
runner. Fraction inputs are integer pairs or strings, never constructed floats.
"""

import hashlib
import json
import time
from fractions import Fraction as F
from itertools import product
from pathlib import Path

ZERO = F(0)
checks = 0
cases = 0
start = time.monotonic()


def serial(x):
    if isinstance(x, F):
        return str(x)
    if isinstance(x, dict):
        return {str(k): serial(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [serial(v) for v in x]
    return x


def emit(label, value):
    print(json.dumps({"label": label, "value": serial(value)}, sort_keys=True), flush=True)


def equal(label, got, want, show=False):
    global checks
    if got != want:
        emit("FAIL", {"check": label, "got": got, "want": want})
        raise AssertionError(label)
    checks += 1
    if show:
        emit(label, got)


def cells(depths, bed):
    # Full physical intervals, independent of stored slots.
    if bed == 0:
        return {}
    K = max(k for k, d in enumerate(depths) if d <= bed)
    return {
        k: (
            ZERO if k == 0 else (depths[k - 1] + depths[k]) / 2,
            bed if k == K else (depths[k] + depths[k + 1]) / 2,
        )
        for k in range(K + 1)
    }


def full_intersections(depths, left, right):
    a, b = cells(depths, left), cells(depths, right)
    result = {}
    for k, interval_map in a.items():
        for recipient_level, J in b.items():
            lo = max(interval_map[0], J[0])
            hi = min(interval_map[1], J[1])
            if hi > lo:
                result[k, recipient_level] = (hi - lo, (hi + lo) / 2)
    return result


def slot_intersections(depths, left, right):
    a, b = cells(depths, left), cells(depths, right)
    result = {}
    for k in range(len(depths)):
        for offset in (0, 1, -1):
            recipient_level = k + offset
            if k not in a or recipient_level not in b:
                result[k, recipient_level] = (ZERO, ZERO)
                continue
            top = max(a[k][0], b[recipient_level][0])
            bottom = min(a[k][1], b[recipient_level][1])
            result[k, recipient_level] = (
                (bottom - top, (top + bottom) / 2) if bottom > top else (ZERO, ZERO)
            )
    return result


def geometry_check(depths, left, right):
    global cases
    full = full_intersections(depths, left, right)
    slots = slot_intersections(depths, left, right)
    positive = {p: v for p, v in slots.items() if v[0] > 0}
    equal("C1/full-slots", (full), positive)
    equal("C1/total", sum((v[0] for v in full.values()), ZERO), min(left, right))
    equal(
        "C1/reversal",
        full,
        {
            (recipient_level, k): v
            for (k, recipient_level), v in full_intersections(depths, right, left).items()
        },
    )
    segments = sorted((mid - w / 2, mid + w / 2) for w, mid in full.values())
    cursor = ZERO
    for lo, hi in segments:
        equal("C1/gap-or-overlap", lo, cursor)
        cursor = hi
    equal("C1/coverage-end", cursor, min(left, right))
    for bed in (left, right):
        interval_map = cells(depths, bed)
        equal("C1/width-sum", sum((hi - lo for lo, hi in interval_map.values()), ZERO), bed)
        for lo, hi in interval_map.values():
            equal("C1/positive-width", hi > lo, True)
    cases += 1
    return full


class Grid:
    def __init__(self, depths, beds, nx, ny, c, L=F(4), dy=F(3)):
        self.depths = depths
        self.beds = beds
        self.nx = nx
        self.ny = ny
        self.c = c
        self.L = L
        self.dy = dy
        self.cols = [(i, j) for j in range(ny) for i in range(nx)]
        self.nodes = [(i, j, k) for i, j in self.cols for k in range(len(depths))]
        self.dx = {a: L * c[a[1]] for a in self.cols}
        self.area = {a: self.dx[a] * dy for a in self.cols}
        self.h = {}
        self.safe = {}
        self.vol = {}
        for i, j, k in self.nodes:
            interval_map = cells(depths, beds[i, j])
            width = interval_map[k][1] - interval_map[k][0] if k in interval_map else ZERO
            self.h[i, j, k] = width
            self.safe[i, j, k] = width if width else F(1)
            self.vol[i, j, k] = self.area[i, j] * width
        self.edges = []
        for axis in ("x", "y"):
            for i, j in self.cols:
                if axis == "y" and j == ny - 1:
                    continue
                other = ((i + 1) % nx, j) if axis == "x" else (i, j + 1)
                # Full-pair intersections, BEFORE considering the three slots.
                full = full_intersections(depths, beds[i, j], beds[other])
                for (k, recipient_level), (w, mid) in full.items():
                    assert abs(k - recipient_level) <= 1
                    f = F(1) if axis == "x" else (c[j] + c[j + 1]) / 2
                    self.edges.append((axis, (i, j, k), (*other, recipient_level), w, mid, f))

    def blank(self):
        return {n: ZERO for n in self.nodes}

    def div(self, quantities):
        E = self.blank()
        for edge, q in zip(self.edges, quantities):
            axis, a, b, w, mid, f = edge
            for n, sign in ((a, 1), (b, -1)):
                den = self.dx[n[:2]] if axis == "x" else self.dy * self.c[n[1]]
                E[n] += sign * q / den
        return E, {n: E[n] / self.safe[n] for n in self.nodes}

    def transports(self, u, v):
        return [
            w * f * ((u if axis == "x" else v)[a] + (u if axis == "x" else v)[b]) / 2
            for axis, a, b, w, mid, f in self.edges
        ]

    def grad(self, phi, plant=False):
        gx = self.blank()
        gy = self.blank()
        for axis, a, b, w, mid, f in self.edges:
            delta = phi[b] - phi[a]
            for n, sign in ((a, 1), (b, -1 if plant else 1)):
                den = self.dx[n[:2]] if axis == "x" else self.dy * self.c[n[1]]
                dest = gx if axis == "x" else gy
                dest[n] += sign * w * f * delta / (2 * den * self.safe[n])
        return gx, gy

    def broadcast(self, psi):
        return {n: psi[n[:2]] if self.h[n] else ZERO for n in self.nodes}

    def projection_direct(self, psi):
        gx, gy = self.grad(self.broadcast(psi))
        E, D = self.div(self.transports(gx, gy))
        return {
            a: -self.area[a] * sum((self.safe[n] * D[n] for n in self.nodes if n[:2] == a), ZERO)
            if self.beds[a]
            else ZERO
            for a in self.cols
        }

    def gram_matrix(self):
        # Independently accumulate column coefficients from each oriented edge.
        coefficients = {
            (axis, n): {a: ZERO for a in self.cols} for axis in ("x", "y") for n in self.nodes
        }
        for axis, left, right, w, mid, f in self.edges:
            for n in (left, right):
                den = self.dx[n[:2]] if axis == "x" else self.dy * self.c[n[1]]
                z = w * f / (2 * den * self.safe[n])
                coefficients[axis, n][right[:2]] += z
                coefficients[axis, n][left[:2]] -= z
        matrix = []
        for a in self.cols:
            matrix.append(
                [
                    sum(
                        (
                            self.vol[n] * coefficients[axis, n][a] * coefficients[axis, n][b]
                            for axis in ("x", "y")
                            for n in self.nodes
                        ),
                        ZERO,
                    )
                    for b in self.cols
                ]
            )
        return matrix

    def basis_matrix(self):
        columns = []
        for a in self.cols:
            psi = {b: F(b == a) for b in self.cols}
            p = self.projection_direct(psi)
            columns.append([p[b] for b in self.cols])
        return [list(row) for row in zip(*columns)]

    def candidate_diag(self):
        answer = {a: ZERO for a in self.cols}
        for axis in ("x", "y"):
            p = self.blank()
            m = self.blank()
            t = self.blank()
            r = self.blank()
            for axis_e, left, right, w, mid, f in self.edges:
                if axis_e != axis:
                    continue
                weight = w * f
                p[left] += weight
                m[right] += weight
                # align recipient index onto source column and vice versa
                t[(*left[:2], right[2])] += weight
                r[(*right[:2], left[2])] += weight
            for n in self.nodes:
                i, j, k = n
                a = (i, j)
                if axis == "x":
                    if self.nx <= 2:
                        continue
                    nxt = ((i + 1) % self.nx, j, k)
                    prev = ((i - 1) % self.nx, j, k)
                    z = (
                        (p[n] - m[n]) ** 2 / self.safe[n]
                        + t[n] ** 2 / self.safe[nxt]
                        + r[n] ** 2 / self.safe[prev]
                    ) / (4 * self.dx[a] ** 2)
                else:
                    nxt = (i, (j + 1) % self.ny, k)
                    prev = (i, (j - 1) % self.ny, k)
                    z = (
                        (p[n] - m[n]) ** 2 / (self.safe[n] * self.c[j])
                        + t[n] ** 2 / (self.safe[nxt] * self.c[nxt[1]])
                        + r[n] ** 2 / (self.safe[prev] * self.c[prev[1]])
                    ) / (4 * self.dy**2 * self.c[j])
                if self.beds[a]:
                    answer[a] += self.area[a] * z
        return [answer[a] for a in self.cols]


def inner(grid, a, b):
    return sum((grid.vol[n] * a[n] * b[n] for n in grid.nodes), ZERO)


def audit_grid(g):
    global cases
    u = {n: F((idx % 7) - 3, 3) if g.h[n] else ZERO for idx, n in enumerate(g.nodes)}
    v = {n: F((idx % 5) - 2, 2) if g.h[n] else ZERO for idx, n in enumerate(g.nodes)}
    phi = {n: F((idx % 11) - 5, 7) if g.h[n] else ZERO for idx, n in enumerate(g.nodes)}
    # arbitrary signed supported contact fluxes, not necessarily transports
    q = [F((idx % 13) - 6, 5) for idx in range(len(g.edges))]
    E, D = g.div(q)
    equal("C2/supported-flux", sum((g.vol[n] * D[n] for n in g.nodes), ZERO), ZERO)
    E, D = g.div(g.transports(u, v))
    gx, gy = g.grad(phi)
    equal("C3/adjoint", inner(g, phi, D) + inner(g, u, gx) + inner(g, v, gy), ZERO)
    basis = g.basis_matrix()
    gram = g.gram_matrix()
    equal("C4/direct-basis-vs-Gram", basis, gram)
    equal("C4/candidate-vs-basis", g.candidate_diag(), [basis[a][a] for a in range(len(g.cols))])
    equal(
        "P/constant-null",
        list(g.projection_direct({a: F(1) for a in g.cols}).values()),
        [ZERO] * len(g.cols),
    )
    equal(
        "P/zero-null",
        list(g.projection_direct({a: ZERO for a in g.cols}).values()),
        [ZERO] * len(g.cols),
    )
    cases += 1
    return basis


def t1():
    ds = list(map(F, (0, 5, 15, 30)))
    intervals = {
        0: {},
        3: {0: (F(0), F(3))},
        14: {0: (F(0), F(5, 2)), 1: (F(5, 2), F(14))},
        27: {0: (F(0), F(5, 2)), 1: (F(5, 2), F(10)), 2: (F(10), F(27))},
        30: {0: (F(0), F(5, 2)), 1: (F(5, 2), F(10)), 2: (F(10), F(45, 2)), 3: (F(45, 2), F(30))},
    }
    widths = {
        0: [0, 0, 0, 0],
        3: [3, 0, 0, 0],
        14: [F(5, 2), F(23, 2), 0, 0],
        27: [F(5, 2), F(15, 2), 17, 0],
        30: [F(5, 2), F(15, 2), F(25, 2), F(15, 2)],
    }
    for H, interval_map in intervals.items():
        got = cells(ds, F(H))
        equal("T1g/intervals/" + str(H), got, interval_map, True)
        h = [got[k][1] - got[k][0] if k in got else ZERO for k in range(4)]
        equal("T1g/widths/" + str(H), h, list(map(F, widths[H])), True)
        equal(
            "T1g/safe-divisors/" + str(H),
            [z or F(1) for z in h],
            [z or F(1) for z in map(F, widths[H])],
            True,
        )
    pairs = {
        (3, 14): {(0, 0): (F(5, 2), F(5, 4)), (0, 1): (F(1, 2), F(11, 4))},
        (14, 27): {(0, 0): (F(5, 2), F(5, 4)), (1, 1): (F(15, 2), F(25, 4)), (1, 2): (F(4), F(12))},
        (27, 30): {
            (0, 0): (F(5, 2), F(5, 4)),
            (1, 1): (F(15, 2), F(25, 4)),
            (2, 2): (F(25, 2), F(65, 4)),
            (2, 3): (F(9, 2), F(99, 4)),
        },
        (0, 30): {},
    }
    for pair, want in pairs.items():
        equal("T1g/contact/" + str(pair), geometry_check(ds, *map(F, pair)), want, True)
        equal(
            "T1g/reverse/" + str(pair),
            geometry_check(ds, *map(F, pair[::-1])),
            {(recipient_level, k): v for (k, recipient_level), v in want.items()},
            True,
        )
        for p, (w, mid) in slot_intersections(ds, *map(F, pair)).items():
            if w == 0:
                equal("T1g/zero-midpoint", mid, ZERO)
    boundary = list(map(F, ("0", "5/2", "5", "10", "15", "45/2", "30")))
    probes = sorted(
        {
            h + delta
            for h in boundary
            for delta in (ZERO, F(-1, 10), F(1, 10))
            if 0 <= h + delta <= 30
        }
    )
    for left, right in product(probes, repeat=2):
        geometry_check(ds, left, right)
    emit("T1g/boundary-probe-corpus", {"beds": probes, "ordered_pairs": len(probes) ** 2})
    g = Grid(ds, {(i, 0): F(3) for i in range(3)}, 3, 1, [F(1, 2)])
    wet = [(i, 0, 0) for i in range(3)]
    equal(
        "T1x/metrics",
        (
            g.c,
            g.L,
            list(g.dx.values()),
            g.dy,
            list(g.area.values()),
            [g.h[n] for n in wet],
            [g.vol[n] for n in wet],
        ),
        ([F(1, 2)], F(4), [F(2)] * 3, F(3), [F(6)] * 3, [F(3)] * 3, [F(18)] * 3),
        True,
    )
    u = g.blank()
    v = g.blank()
    phi = g.blank()
    for n, z, p in zip(wet, (1, 2, 4), (3, -1, 2)):
        u[n] = F(z)
        phi[n] = F(p)
    q = g.transports(u, v)
    E, D = g.div(q)
    gx, gy = g.grad(phi)
    equal("T1x/flux", q, [F(9, 2), F(9), F(15, 2)], True)
    equal("T1x/E", [E[n] for n in wet], [F(-3, 2), F(9, 4), F(-3, 4)], True)
    equal("T1x/D", [D[n] for n in wet], [F(-1, 2), F(3, 4), F(-1, 4)], True)
    equal("T1x/Gx", [gx[n] for n in wet], [F(-3, 4), F(-1, 4), F(1)], True)
    equal("T1x/Gy", list(gy.values()), [ZERO] * len(g.nodes), True)
    equal("T1x/conservation", sum((g.vol[n] * D[n] for n in g.nodes), ZERO), ZERO, True)
    equal("T1x/pairings", (inner(g, phi, D), inner(g, u, gx)), (F(-99, 2), F(99, 2)), True)
    want = [[F(9, 8) * F(2 if i == j else -1) for j in range(3)] for i in range(3)]
    equal("T1x/P", audit_grid(g), want, True)
    equal("T1x/diag", g.candidate_diag(), [F(9, 4)] * 3, True)
    # Copy control: same T1x input, only incoming gradient scatter sign changed.
    badx, bady = g.grad(phi, plant=True)
    equal("PLANT/gradient", [badx[n] for n in wet], [F(-5, 4), F(7, 4), F(-1, 2)], True)
    residual = inner(g, phi, D) + inner(g, u, badx) + inner(g, v, bady)
    equal("PLANT/nonzero-residual", residual, F(-45), True)
    caught = False
    try:
        equal("PLANT/false-step-claims-zero", residual, ZERO)
    except AssertionError:
        caught = True
    equal("PLANT/caught", caught, True, True)
    caught = False
    try:
        equal("COMPARATOR/deliberate-mismatch", F(1, 3), F(1, 2))
    except AssertionError:
        caught = True
    equal("COMPARATOR/caught", caught, True, True)
    g = Grid(ds, {(0, j): F(3) for j in range(2)}, 1, 2, [F(1, 2), F(3, 4)])
    wet = [(0, j, 0) for j in range(2)]
    equal(
        "T1y/metrics",
        (g.c, g.L, list(g.dx.values()), g.dy, list(g.area.values()), [g.vol[n] for n in wet]),
        ([F(1, 2), F(3, 4)], F(4), [F(2), F(3)], F(3), [F(6), F(9)], [F(18), F(27)]),
        True,
    )
    equal("T1y/cface", [e[5] for e in g.edges if e[0] == "y"], [F(5, 8)], True)
    u = g.blank()
    v = g.blank()
    phi = g.blank()
    for n, z, p in zip(wet, (2, 4), (3, -1)):
        v[n] = F(z)
        phi[n] = F(p)
    q = g.transports(u, v)
    E, D = g.div(q)
    gx, gy = g.grad(phi)
    equal("T1y/flux", [z for e, z in zip(g.edges, q) if e[0] == "y"], [F(45, 8)], True)
    equal(
        "T1y/closed-wall-flux",
        [
            sum((z for e, z in zip(g.edges, q) if e[0] == "y" and e[1][1] == g.ny - 1), ZERO),
            sum((z for e, z in zip(g.edges, q) if e[0] == "y" and e[2][1] == 0), ZERO),
        ],
        [ZERO, ZERO],
        True,
    )
    equal("T1y/E", [E[n] for n in wet], [F(15, 4), F(-5, 2)], True)
    equal("T1y/D", [D[n] for n in wet], [F(5, 4), F(-5, 6)], True)
    equal("T1y/Gy", [gy[n] for n in wet], [F(-5, 6), F(-5, 9)], True)
    equal("T1y/Gx", list(gx.values()), [ZERO] * len(g.nodes), True)
    equal("T1y/conservation", sum((g.vol[n] * D[n] for n in g.nodes), ZERO), ZERO, True)
    equal("T1y/pairings", (inner(g, phi, D), inner(g, v, gy)), (F(90), F(-90)), True)
    equal("T1y/P", audit_grid(g), [[F(125, 96), F(-125, 96)], [F(-125, 96), F(125, 96)]], True)
    equal("T1y/diag", g.candidate_diag(), [F(125, 96)] * 2, True)
    g = Grid(ds, {(0, 0): F(3), (1, 0): F(14)}, 2, 1, [F(1, 2)])
    psi = {(0, 0): F(7, 3), (1, 0): F(-11, 5)}
    gx, gy = g.grad(g.broadcast(psi))
    equal("T1small/nx2-G", list(gx.values()) + list(gy.values()), [ZERO] * (2 * len(g.nodes)), True)
    equal("T1small/nx2-P", audit_grid(g), [[ZERO, ZERO], [ZERO, ZERO]], True)
    equal("T1small/nx2-diag", g.candidate_diag(), [ZERO, ZERO], True)
    g = Grid(ds, {(0, 0): F(14)}, 1, 1, [F(1, 2)])
    equal("T1small/nx1-P", audit_grid(g), [[ZERO]], True)
    equal("T1small/nx1-diag", g.candidate_diag(), [ZERO], True)


def hostile():
    # Fixed before first execution: finite domain, not universal proof.
    depth_sets = [
        list(map(F, (0, 5, 15, 30))),
        list(map(F, (0, 1, 2, 3))),
        [F(0), F(1, 7), F(23, 5), F(47, 3)],
        list(map(F, (0, 1))),
    ]
    geo_pairs = 0
    grid_cases = 0
    for ds in depth_sets:
        values = sorted(
            {
                ZERO,
                F(1, 100),
                ds[-1],
                *ds,
                *((a + b) / 2 for a, b in zip(ds, ds[1:])),
                *(
                    d + F(sign, 100)
                    for d in ds
                    for sign in (-1, 1)
                    if 0 <= d + F(sign, 100) <= ds[-1]
                ),
            }
        )
        for a, b in product(values, repeat=2):
            geometry_check(ds, a, b)
            geo_pairs += 1
        for nx, ny in product(range(1, 5), range(1, 4)):
            cols = [(i, j) for j in range(ny) for i in range(nx)]
            # Uniform cases plus mixed cyclic/asymmetric dry/terminal patterns.
            patterns = [{a: H for a in cols} for H in values]
            patterns += [
                {a: values[(idx * stride + shift) % len(values)] for idx, a in enumerate(cols)}
                for stride in (1, 3, 7)
                for shift in range(4)
            ]
            for beds in patterns:
                c = [F(2 + j, 5 + j) for j in range(ny)]
                audit_grid(Grid(ds, beds, nx, ny, c, L=F(7, 3), dy=F(11, 4)))
                grid_cases += 1
    emit(
        "HOSTILE/finite-domain",
        {
            "depth_sets": depth_sets,
            "geometry_pairs": geo_pairs,
            "grid_cases": grid_cases,
            "nx": [1, 2, 3, 4],
            "ny": [1, 2, 3],
            "row_cosines": "(2+j)/(5+j)",
            "L": "7/3",
            "dy": "11/4",
        },
    )


if __name__ == "__main__":
    emit(
        "BIRTH",
        {
            "script": str(Path(__file__).resolve()),
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "origin": "pin v1 only, saved before arm review",
            "imports": ["fractions", "itertools", "json", "hashlib", "pathlib", "time"],
            "arithmetic": "exact Fraction",
            "budget": "parent runner 1 CPU/180 seconds/4 GiB",
        },
    )
    t1()
    hostile()
    equal("NONVACUOUS/check-count", checks > 1000, True)
    equal("NONVACUOUS/case-count", cases > 1000, True)
    emit(
        "SUMMARY",
        {
            "status": "PASS-FRESH-EXACT",
            "checks": checks,
            "cases": cases,
            "elapsed_seconds": time.monotonic() - start,
            "scope": "T1 and finite hostile corpus; universal proof separately reviewed",
        },
    )
