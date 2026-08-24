"""Pathfinding + rendering reference for the Luminary strut-graph family.

This is a LIBRARY AND FIELD GUIDE, not a pattern. It collects the graph,
distance, movement, and slew-safe-rendering idioms that pacman.py,
serpent.py, and constellations.py each carry private copies of, in their
best-known form (including the 2026-08-23 hub fixes that cured the
one-sided blip spokes). Import from it when prototyping; copy into a
pattern when shipping (patterns stay self-contained by convention).

VOCABULARY — the words the codebase and records use for this geometry
=====================================================================

light      One LED: a row of the (n, cols) lights array. Canonical order
           is (controller, channel, index).
strip      All lights sharing (controller, channel) — one physical LED
           string. On this build every strip is 60 lights and its FIRST
           light sits at a hub center (see build_runs' forward merge).
run        A maximal straight stretch of one strip: split where the
           polyline bends more than TURN_DEG (28°) or the spacing jumps
           more than GAP_FACTOR (4x) the strip's median. Runs are the
           atoms of graph extraction.
vertex     A cluster of run endpoints (union-find within `tol`). Where
(hub)      struts physically meet. "Hub" is used for the physical
           junction, "vertex" for its graph node; same thing here.
corridor   One graph edge between two vertices. ALL runs whose endpoints
(lane)     cluster to the same vertex pair merge into ONE corridor — the
           beams either side of a panel seam become a single lane with a
           shared arclength coordinate. Rendering positions live on
           corridors as arclength `s` in [0, clen].
through-run  Two near-collinear struts read as ONE run by the bend test.
           The hub between them then has a corridor passing through it
           with no incidence — a radial glow there lights every arm
           EXCEPT the straight-through pair (the one-sided blip bug).
           split_runs_at_junctions repairs this; measured on the star,
           genuine passes sit 0.03–0.08 units from their hub centroid
           and the nearest false candidate at 0.635, so the 0.25-unit
           threshold has ~3x margin both ways. Raw run-ENDPOINT distance
           is NOT a usable criterion — inset inner-beam corners sit
           within cluster-tol of strut interiors everywhere, and a first
           attempt with it shredded 422 runs into 1098 pieces.
unit       The median run chord length — the natural yardstick. Sigmas,
           reaches, band widths, and speeds are all expressed in units
           so they transfer across geometries (world coords ≈ inches on
           this build; the hex demo's unit differs from the star's).
incenter   A degree-3 vertex whose three corridors leave ~120° apart
           (max angular gap < 150°). One per structural triangle panel;
           the hex demo has none.
spoke      A corridor incident to an incenter — the three short bars
           joining a panel's incenter to its edge midpoints. pacman
           drops most spokes to make a maze; serpent/constellations
           keep them.
border vertex  Degree-3 vertex on the net's outer boundary: its angular
           gaps run ~90/90/180, so the same 150° cap that finds
           incenters separates borders cleanly.
triangle-  A 3-cycle of vertices — a panel's perimeter (or an inner
perimeter  triangle formed by edge midpoints). Enumerate with
           `triangles()`.
seam       A degree-4 (or higher) vertex where one panel joins another.

THE WIRE, AND WHY TAPERING EXISTS
=================================
The preview client decodes REAL codec frames — what you see at :8082 is
what the LEDs get. The decoder rate-limits each light per frame; the
measured caps at 30 fps are SLEW_CAP_L/C/H below. Consequences, all
measured, all now doctrine (see records/):

- A moving hard edge is impossible to transmit. The decoder walks the
  falling light's hue toward the next color under the hue cap WHILE its
  luminance is still high — a bright yellow disc leaving a light drags
  H 95°→266° and paints a cyan/blue streak behind it.
- Therefore every moving bright thing needs a taper sized to its speed
  and amplitude: gaussian heads (slew_safe_sigma), exponential or
  linear tails (slew_safe_tail), and collapse/retract rates bounded in
  units/s (retract_duration). The shipped pacman blur is not aesthetic:
  its sigma (0.19 x unit) is exactly wire-sized for its speed.
- Hue itself must only move fast where composite L is low ("dark before
  turn"). Design to HUE_DESIGN_DEG (59°/frame) rather than the 89° hard
  cap; the margin absorbs compositing surprises.
- Formulas lie; scans don't. Every first pass of every bright element in
  this codebase violated a cap in some frame a formula missed. Run
  scan_slew() over windows that contain the event (a crash, an eat, a
  draw completing) before believing any number.

STATELESSNESS
=============
Patterns are pure functions of (lights, t) — bit-identical across
processes and PYTHONHASHSEED (spec §9.1.3). Never use Python's salted
hash(); use fnv()/frac()/seeded_unit() here. Heavy choreography is
simulated once per epoch/round keyed on (content_key(lights), epoch
index) and memoized; render is then an O(1) lookup at any t.
"""

from __future__ import annotations

import heapq
import zlib
from collections import deque
from typing import Dict, List, NamedTuple, Optional, Sequence, Set, Tuple

import numpy as np

from luminary.geometry.lights import LightColumns

# --- wire limits and design targets ----------------------------------

FPS = 30.0  # the stream rate all slew math assumes

# Measured per-light per-frame caps of the wire codec at 30 fps.
SLEW_CAP_L = 0.24  # luminance
SLEW_CAP_C = 0.09  # chroma
SLEW_CAP_H = 89.0  # hue, degrees (only meaningful while C is visible)

# Design targets — build to these, not the caps. Compositing (overlaps,
# OKLab vector cancellation between differently-hued elements) spends
# the difference for you.
SLEW_TARGET_L = 0.08  # constellations ships on this and reads punchy
HUE_DESIGN_DEG = 59.0  # the house limit for hue motion while L is high
HERO_ENVELOPE_L = 0.36  # measured max of shipped transient hero events
# (gulp flashes, blip pops). Rising edges this size just bloom a frame
# slower; FALLING edges this size are the blue-smear hazard — taper them.

# Hue changes read as motion opposite to hue-angle travel: serpent's
# stripes flow tailward by ADVANCING hue with arclength and time in
# opposite signs. If a shape moves at +v, drifting its hue field at a
# negative rate strengthens perceived motion; same-sign drift fights it.

# --- graph extraction constants ---------------------------------------

TURN_DEG = 28.0  # split a strip where it bends more than this
GAP_FACTOR = 4.0  # ... or jumps more than this multiple of its spacing
MIN_RUN = 4  # runs shorter than this merge into a neighbor or drop
SPLIT_EPS = 0.25  # x unit: run interior passing this close to a hub
# centroid is a through-run and gets split there (see vocabulary).
INCENTER_GAP_DEG = 150.0  # max angular gap separating incenters (<150°)
# from border vertices (~180° gap); one incenter per panel on 4A-33/37,
# zero on the hex demo.

# --- deterministic hashing (PYTHONHASHSEED-proof) ---------------------


def fnv(*vals: int) -> int:
    """FNV-1a over integers — pure arithmetic, identical in every
    process. The ONLY acceptable hash for choreography decisions."""
    h = 2166136261
    for v in vals:
        h = ((h ^ (int(v) & 0xFFFFFFFF)) * 16777619) & 0xFFFFFFFF
    return h


def frac(*vals: int) -> float:
    """Deterministic uniform in [0, 1) from integer keys."""
    return fnv(*vals) / 4294967296.0


def seeded_unit(key: str, n: int) -> np.ndarray:
    """n deterministic uniforms in [0, 1) from a string key — for
    per-light or per-cell stagger fields. crc32-keyed RandomState is
    stable across platforms and hash seeds."""
    rs = np.random.RandomState(zlib.crc32(key.encode()) & 0x7FFFFFFF)
    return rs.random_sample(n)


def content_key(lights: np.ndarray) -> Tuple[int, int]:
    """Cheap content fingerprint of a lights array for memoization —
    (row count, crc32 of a strided identity+position sample). Keying
    caches on this instead of id(lights) survives the server handing
    you a fresh array each frame."""
    sample = lights[
        :: max(1, lights.shape[0] // 64),
        [
            LightColumns.CONTROLLER,
            LightColumns.CHANNEL,
            LightColumns.INDEX,
            LightColumns.X,
            LightColumns.Y,
        ],
    ]
    return (
        lights.shape[0],
        zlib.crc32(np.ascontiguousarray(np.nan_to_num(sample)).tobytes()),
    )


# --- graph extraction --------------------------------------------------


class Graph(NamedTuple):
    """The strut graph recovered from a lights array.

    Light rows of corridor c are rows[ptr[c]:ptr[c+1]], sorted by their
    arclength arc[...] measured from the cu[c] end. All runs between the
    same vertex pair share the corridor (seam lanes merge); their
    arclengths are rescaled to the common clen[c]."""

    cu: np.ndarray  # (nc,) corridor endpoint vertex (arc = 0 end)
    cv: np.ndarray  # (nc,) corridor endpoint vertex (arc = clen end)
    clen: np.ndarray  # (nc,) corridor arclength
    rows: np.ndarray  # concatenated light rows per corridor
    arc: np.ndarray  # arclength of each row from the cu end
    ptr: np.ndarray  # (nc+1,) corridor slices into rows/arc
    adj: List[List[Tuple[int, int]]]  # adj[v] = [(other_vertex, corridor)]
    vxy: np.ndarray  # (nv, 2) vertex centroid positions
    unit: float  # median run chord — the geometry yardstick
    spacing: float  # median light spacing
    nv: int
    nc: int


def build_runs(a: np.ndarray) -> Tuple[List[np.ndarray], float]:
    """Split each strip into straight runs; return (runs, median spacing).

    Includes the forward-merge pass: a short leading piece (every strip's
    first light sits at a hub center and bends away from its strut) has
    no previous piece to merge into and was historically dropped —
    83 permanently dark hub-center lights on the star. Attach it to the
    FOLLOWING piece when spatially contiguous."""
    keys = a[:, LightColumns.CONTROLLER].astype(np.int64) * 8 + a[
        :, LightColumns.CHANNEL
    ].astype(np.int64)
    runs: List[np.ndarray] = []
    spacings = []
    for k in np.unique(keys):
        rows = np.flatnonzero(keys == k)
        xy = a[np.ix_(rows, np.array([LightColumns.X, LightColumns.Y], np.intp))]
        finite = ~np.isnan(xy).any(axis=1)
        rows, xy = rows[finite], xy[finite]
        if len(rows) < 2:
            continue
        d = np.diff(xy, axis=0)
        seg = np.hypot(d[:, 0], d[:, 1])
        spacings.append(seg)
        med = float(np.median(seg))
        ang = np.abs(np.diff(np.arctan2(d[:, 1], d[:, 0])))
        ang = np.degrees(np.minimum(ang, 2.0 * np.pi - ang))
        cut = seg > GAP_FACTOR * med
        cut[1:] |= ang > TURN_DEG
        starts = np.concatenate([[0], np.flatnonzero(cut) + 1, [len(rows)]])
        pieces = [rows[s:e] for s, e in zip(starts[:-1], starts[1:])]
        merged: List[np.ndarray] = []
        for p in pieces:  # backward merge
            if merged and len(p) < MIN_RUN:
                prev = merged[-1]
                gap = float(
                    np.hypot(
                        a[p[0], LightColumns.X] - a[prev[-1], LightColumns.X],
                        a[p[0], LightColumns.Y] - a[prev[-1], LightColumns.Y],
                    )
                )
                if gap < GAP_FACTOR * med:
                    merged[-1] = np.concatenate([prev, p])
                    continue
            merged.append(p)
        fwd: List[np.ndarray] = []  # forward merge (see docstring)
        pend: Optional[np.ndarray] = None
        for p in merged:
            if pend is not None:
                gap = float(
                    np.hypot(
                        a[p[0], LightColumns.X] - a[pend[-1], LightColumns.X],
                        a[p[0], LightColumns.Y] - a[pend[-1], LightColumns.Y],
                    )
                )
                if gap < GAP_FACTOR * med:
                    p = np.concatenate([pend, p])
                pend = None
            if len(p) < MIN_RUN:
                pend = p
                continue
            fwd.append(p)
        runs.extend(fwd)
    med_all = float(np.median(np.concatenate(spacings))) if spacings else 1.0
    return runs, med_all


def cluster_endpoints(a: np.ndarray, runs: List[np.ndarray], tol: float) -> np.ndarray:
    """Union-find run endpoints into vertex labels; run i's ends are
    out[2i] (first light) and out[2i+1] (last). Chains: endpoints A-B and
    B-C within tol union A with C even if A-C exceeds tol — this is what
    folds a freshly split through-run's new ends into the hub cluster."""
    pts = np.array(
        [
            [a[r[i], LightColumns.X], a[r[i], LightColumns.Y]]
            for r in runs
            for i in (0, -1)
        ]
    )
    n = len(pts)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        d = np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1])
        for j in np.flatnonzero(d < tol):
            ri, rj = find(i), find(int(j))
            if ri != rj:
                parent[ri] = rj
    roots = sorted({find(i) for i in range(n)})
    remap = {r: k for k, r in enumerate(roots)}
    return np.array([remap[find(i)] for i in range(n)], np.int64)


def split_runs_at_junctions(
    a: np.ndarray, runs: List[np.ndarray], tol: float, unit: float
) -> List[np.ndarray]:
    """Split runs whose interior passes within SPLIT_EPS*unit of a hub
    (vertex centroid of a provisional clustering) — the through-run
    repair. See the vocabulary entry for why the threshold must be hub
    CENTROIDS, never raw run endpoints."""
    labels = cluster_endpoints(a, runs, tol)
    nv = int(labels.max()) + 1
    pts = np.array(
        [
            [a[r[i], LightColumns.X], a[r[i], LightColumns.Y]]
            for r in runs
            for i in (0, -1)
        ]
    )
    cent = np.zeros((nv, 2))
    cnt = np.zeros(nv)
    for j, lab in enumerate(labels):
        cent[lab] += pts[j]
        cnt[lab] += 1
    cent /= np.maximum(cnt, 1)[:, None]

    eps = SPLIT_EPS * unit
    xcols = np.array([LightColumns.X, LightColumns.Y], np.intp)
    out: List[np.ndarray] = []
    for e, r in enumerate(runs):
        own = {int(labels[2 * e]), int(labels[2 * e + 1])}
        foreign = np.array([v for v in range(nv) if v not in own], np.intp)
        stack = [r]
        while stack:
            rr = stack.pop()
            if len(foreign) and len(rr) >= 2 * MIN_RUN:
                xy = a[np.ix_(rr, xcols)]
                d = np.hypot(
                    xy[:, None, 0] - cent[None, foreign, 0],
                    xy[:, None, 1] - cent[None, foreign, 1],
                ).min(axis=1)
                inner = d[MIN_RUN : len(rr) - MIN_RUN]
                if len(inner) and inner.min() < eps:
                    k = MIN_RUN + int(inner.argmin())
                    stack.append(rr[: k + 1])
                    stack.append(rr[k + 1 :])
                    continue
            out.append(rr)
    return out


def build_graph(a: np.ndarray) -> Optional[Graph]:
    """Full extraction: runs -> hub split -> cluster -> merge seam lanes
    into corridors -> largest connected component. Returns None when the
    lights don't form a usable graph (fewer than 6 runs / 4 corridors).

    Degenerate runs whose two ends cluster to the SAME vertex (tiny
    stubs curling back to their hub) are dropped, as all three patterns
    drop them; on the star that is 2 four-light stubs."""
    runs, spacing = build_runs(a)
    if len(runs) < 6:
        return None
    chords = [
        float(
            np.hypot(
                a[r[-1], LightColumns.X] - a[r[0], LightColumns.X],
                a[r[-1], LightColumns.Y] - a[r[0], LightColumns.Y],
            )
        )
        for r in runs
    ]
    unit = float(np.median(chords))
    tol = max(3.0 * spacing, 0.3 * unit)
    runs = split_runs_at_junctions(a, runs, tol, unit)
    labels = cluster_endpoints(a, runs, tol)

    by_pair: Dict[Tuple[int, int], List[int]] = {}
    flip: Dict[int, bool] = {}
    for e in range(len(runs)):
        u, v = int(labels[2 * e]), int(labels[2 * e + 1])
        if u == v:
            continue  # degenerate stub
        by_pair.setdefault((min(u, v), max(u, v)), []).append(e)
        flip[e] = u > v
    if len(by_pair) < 4:
        return None

    order_keys = sorted(by_pair)
    cu = np.array([p[0] for p in order_keys], np.int64)
    cv = np.array([p[1] for p in order_keys], np.int64)
    clen = np.zeros(len(order_keys))
    rows_l, arc_l, counts = [], [], []
    for ci, pair in enumerate(order_keys):
        members = by_pair[pair]
        alongs, lens = [], []
        for e in members:
            r = runs[e]
            xy = a[np.ix_(r, np.array([LightColumns.X, LightColumns.Y], np.intp))]
            seg = np.hypot(*np.diff(xy, axis=0).T)
            along = np.concatenate([[0.0], np.cumsum(seg)])
            lens.append(max(1e-6, float(along[-1])))
            if flip[e]:
                along = along[-1] - along
            alongs.append(along)
        clen[ci] = float(np.mean(lens))
        rows_l.append(np.concatenate([runs[e] for e in members]))
        arc_l.append(
            np.concatenate([al * (clen[ci] / ln) for al, ln in zip(alongs, lens)])
        )
        counts.append(sum(len(runs[e]) for e in members))
    rows = np.concatenate(rows_l).astype(np.int64)
    arc = np.concatenate(arc_l)
    ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    for ci in range(len(order_keys)):
        lo, hi = int(ptr[ci]), int(ptr[ci + 1])
        order = np.argsort(arc[lo:hi], kind="stable")
        rows[lo:hi] = rows[lo:hi][order]
        arc[lo:hi] = arc[lo:hi][order]

    nv = int(labels.max()) + 1
    vsum = np.zeros((nv, 2))
    vcnt = np.zeros(nv)
    for e, r in enumerate(runs):
        for i, end in ((0, 0), (-1, 1)):
            lab = int(labels[2 * e + end])
            vsum[lab] += (a[r[i], LightColumns.X], a[r[i], LightColumns.Y])
            vcnt[lab] += 1
    vxy = vsum / np.maximum(vcnt, 1)[:, None]

    # largest connected component
    adj0 = corridor_adj(cu, cv, nv)
    seen = np.zeros(nv, bool)
    best: List[int] = []
    for s in range(nv):
        if seen[s] or not adj0[s]:
            continue
        comp = [s]
        seen[s] = True
        q = deque([s])
        while q:
            u = q.popleft()
            for w, _ in adj0[u]:
                if not seen[w]:
                    seen[w] = True
                    comp.append(w)
                    q.append(w)
        if len(comp) > len(best):
            best = comp
    keep_v = np.zeros(nv, bool)
    keep_v[best] = True
    keep_c = keep_v[cu] & keep_v[cv]
    remap = np.full(nv, -1, np.int64)
    remap[np.flatnonzero(keep_v)] = np.arange(int(keep_v.sum()))
    kept = np.flatnonzero(keep_c)
    rows2, arc2, counts2 = [], [], []
    for ci in kept:
        lo, hi = int(ptr[ci]), int(ptr[ci + 1])
        rows2.append(rows[lo:hi])
        arc2.append(arc[lo:hi])
        counts2.append(hi - lo)
    cu2, cv2 = remap[cu[kept]], remap[cv[kept]]
    nv2 = int(keep_v.sum())
    return Graph(
        cu=cu2,
        cv=cv2,
        clen=clen[kept],
        rows=np.concatenate(rows2),
        arc=np.concatenate(arc2),
        ptr=np.concatenate([[0], np.cumsum(counts2)]).astype(np.int64),
        adj=corridor_adj(cu2, cv2, nv2),
        vxy=vxy[keep_v],
        unit=unit,
        spacing=spacing,
        nv=nv2,
        nc=len(kept),
    )


def corridor_adj(
    cu: np.ndarray, cv: np.ndarray, nv: int
) -> List[List[Tuple[int, int]]]:
    """adj[v] = [(other_vertex, corridor), ...] sorted by corridor index
    so every traversal is deterministic."""
    adj: List[List[Tuple[int, int]]] = [[] for _ in range(nv)]
    for ci in range(len(cu)):
        adj[int(cu[ci])].append((int(cv[ci]), ci))
        adj[int(cv[ci])].append((int(cu[ci]), ci))
    for lst in adj:
        lst.sort(key=lambda oc: oc[1])
    return adj


# --- vocabulary in code: classification --------------------------------


def corridor_dir_at_vertex(g: Graph, a: np.ndarray, c: int, v: int) -> np.ndarray:
    """Unit direction corridor c LEAVES vertex v with: from the vertex
    toward the mean position of the corridor's middle third. Never aim
    at the nearest lights — a corridor merges the parallel beams either
    side of a panel seam, and a single near light sits laterally offset
    by up to a lane's width, skewing the angle by tens of degrees
    (measured 41° on the star). The middle-third mean averages the lanes
    back onto the corridor axis. v must be an endpoint of c."""
    lo, hi = int(g.ptr[c]), int(g.ptr[c + 1])
    arcs = g.arc[lo:hi]
    d = arcs if int(g.cu[c]) == v else (float(g.clen[c]) - arcs)
    ln = float(g.clen[c])
    mid = (d >= ln / 3.0) & (d <= 2.0 * ln / 3.0)
    if not mid.any():
        mid = np.ones(len(d), bool)
    rows_m = g.rows[lo:hi][mid]
    target = np.array(
        [
            float(np.mean(a[rows_m, LightColumns.X])),
            float(np.mean(a[rows_m, LightColumns.Y])),
        ]
    )
    vec = target - g.vxy[v]
    n = float(np.hypot(*vec))
    return vec / n if n > 1e-9 else vec


def external_angle(g: Graph, a: np.ndarray, ci: int, cj: int, v: int) -> float:
    """Angle in degrees between the departure directions of two corridors
    at their shared vertex v. ~180° means straight-through (a snake
    "goes straight"); ~60° is a sharp panel corner. Checks like
    `external_angle(...) > 150` find collinear continuations."""
    di = corridor_dir_at_vertex(g, a, ci, v)
    dj = corridor_dir_at_vertex(g, a, cj, v)
    cosang = float(np.clip(np.dot(di, dj), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def vertex_angular_gaps(g: Graph, v: int) -> List[float]:
    """Sorted angular gaps (degrees) between the corridors leaving v,
    computed from neighbor-vertex bearings — the incenter/border
    discriminator (works without light positions)."""
    p = g.vxy[v]
    ang = sorted(
        float(np.arctan2(g.vxy[w][1] - p[1], g.vxy[w][0] - p[0])) for w, _ in g.adj[v]
    )
    if len(ang) < 2:
        return [360.0]
    gaps = [np.degrees(b - a) for a, b in zip(ang, ang[1:])]
    gaps.append(360.0 - sum(gaps))
    return sorted(gaps)


def incenters(g: Graph) -> List[int]:
    """Degree-3 vertices whose corridors leave ~120° apart (max angular
    gap < INCENTER_GAP_DEG). One per structural triangle panel on the
    star family; none on the hex demo."""
    out = []
    for v in range(g.nv):
        if len(g.adj[v]) != 3:
            continue
        if max(vertex_angular_gaps(g, v)) < INCENTER_GAP_DEG:
            out.append(v)
    return out


def spokes(g: Graph) -> List[int]:
    """Corridors incident to an incenter — the three inner bars of each
    panel. (pacman drops these to make a maze; radial glows love them.)"""
    inc = set(incenters(g))
    return sorted({c for v in inc for _w, c in g.adj[v]})


def border_vertices(g: Graph) -> List[int]:
    """Degree-3 vertices with a ~180° gap: edge midpoints on the net's
    outer boundary."""
    out = []
    for v in range(g.nv):
        if len(g.adj[v]) != 3:
            continue
        if max(vertex_angular_gaps(g, v)) >= INCENTER_GAP_DEG:
            out.append(v)
    return out


def triangles(g: Graph) -> List[Tuple[int, int, int]]:
    """All 3-cycles of vertices, each once, sorted. NOTE: on the star
    family this returns NOTHING — edge midpoints subdivide every panel
    side, so a panel's perimeter is a 6-cycle (corner, mid, corner, mid,
    corner, mid); use panel_perimeters() there. The hex demo's rim IS
    made of true 3-cycles, which this finds."""
    nbr: List[Set[int]] = [set(w for w, _ in g.adj[v]) for v in range(g.nv)]
    out = []
    for u in range(g.nv):
        for v in nbr[u]:
            if v <= u:
                continue
            for w in nbr[u] & nbr[v]:
                if w > v:
                    out.append((u, v, w))
    return sorted(out)


def panel_perimeters(g: Graph) -> List[Tuple[List[int], List[int]]]:
    """The star family's triangle-perimeters: for each incenter, the
    6-cycle (vertices, corridors) around its panel — alternating edge
    midpoints (the incenter's spoke neighbors) and panel corners (the
    unique common neighbor of each midpoint pair that is not the
    incenter). Panels whose corners cannot be resolved (boundary
    irregularities) are skipped rather than guessed."""
    out: List[Tuple[List[int], List[int]]] = []
    nbr: List[Dict[int, int]] = [dict((w, c) for w, c in g.adj[v]) for v in range(g.nv)]
    for inc in incenters(g):
        mids = [w for w, _ in g.adj[inc]]
        if len(mids) != 3:
            continue
        corners = []
        ok = True
        for i in range(3):
            m0, m1 = mids[i], mids[(i + 1) % 3]
            common = (set(nbr[m0]) & set(nbr[m1])) - {inc}
            if len(common) != 1:
                ok = False
                break
            corners.append(common.pop())
        if not ok:
            continue
        vs: List[int] = []
        cs: List[int] = []
        for i in range(3):
            vs.extend([mids[i], corners[i]])
            cs.append(nbr[mids[i]][corners[i]])
            cs.append(nbr[corners[i]][mids[(i + 1) % 3]])
        out.append((vs, cs))
    return out


def straight_through_pairs(
    g: Graph, a: np.ndarray, min_deg: float = 150.0
) -> List[Tuple[int, int, int]]:
    """(v, ci, cj) triples where corridors ci and cj continue nearly
    straight through v (external angle >= min_deg). Useful for walkers
    that prefer momentum, and for auditing where the bend test ALMOST
    produced a through-run."""
    out = []
    for v in range(g.nv):
        cs = [c for _w, c in g.adj[v]]
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                if external_angle(g, a, cs[i], cs[j], v) >= min_deg:
                    out.append((v, cs[i], cs[j]))
    return out


# --- distances (pick the cheapest that answers the question) -----------
#
# hop_distances : int16 all-pairs BFS. O(nv * (nv + nc)). The workhorse —
#                 spawn spacing, avoid-radii, k-hop balls. 140 vertices
#                 -> instant, ~40 KB.
# arc_distances : float all-pairs by Dijkstra per source. Use when
#                 corridor LENGTHS matter (travel time, "within 4
#                 triangle-edges" style locality that must not treat a
#                 long border bar like a short spoke).
# dijkstra      : single-source with predecessors, for actual paths.
# vertex_euclid : straight-line xy distance — cheap pre-filter only;
#                 the cloth has holes, so never use it as a path length.


def hop_distances(g: Graph) -> Tuple[np.ndarray, int]:
    """All-pairs hop distance by BFS; `far` marks unreachable."""
    far = g.nv + 1
    dist = np.full((g.nv, g.nv), far, np.int16)
    for s in range(g.nv):
        dist[s, s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for w, _ in g.adj[u]:
                if dist[s, w] == far:
                    dist[s, w] = dist[s, u] + 1
                    q.append(w)
    return dist, far


def dijkstra(
    g: Graph, source: int, banned_c: Optional[Set[int]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-source shortest arclength paths. Returns (dist, pred_v,
    pred_c); reconstruct with shortest_path(). `banned_c` excludes
    corridors (constellations' non-intersection restriction)."""
    dist = np.full(g.nv, np.inf)
    dist[source] = 0.0
    pred_v = np.full(g.nv, -1, np.int64)
    pred_c = np.full(g.nv, -1, np.int64)
    visited = np.zeros(g.nv, bool)
    heap: List[Tuple[float, int]] = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        for w, c in g.adj[u]:
            if banned_c and c in banned_c:
                continue
            nd = d + float(g.clen[c])
            if nd < dist[w] - 1e-9:
                dist[w] = nd
                pred_v[w] = u
                pred_c[w] = c
                heapq.heappush(heap, (nd, w))
    return dist, pred_v, pred_c


def shortest_path(
    dist: np.ndarray, pred_v: np.ndarray, pred_c: np.ndarray, target: int
) -> Tuple[List[int], List[int]]:
    """(vertices, corridors) from source to target given dijkstra's
    outputs; ([], []) when unreachable. The source is vs[0]."""
    if not np.isfinite(dist[target]):
        return [], []
    vs, cs = [target], []
    v = target
    while pred_v[v] >= 0:
        cs.append(int(pred_c[v]))
        v = int(pred_v[v])
        vs.append(v)
    return vs[::-1], cs[::-1]


def arc_distances(g: Graph) -> np.ndarray:
    """All-pairs arclength distance (float). O(nv) Dijkstras — fine for
    hundreds of vertices; precompute once per geometry and memoize on
    content_key."""
    out = np.full((g.nv, g.nv), np.inf)
    for s in range(g.nv):
        out[s] = dijkstra(g, s)[0]
    return out


def vertex_euclid(g: Graph) -> np.ndarray:
    """All-pairs straight-line vertex distance. Pre-filter only."""
    dx = g.vxy[:, 0][:, None] - g.vxy[:, 0][None, :]
    dy = g.vxy[:, 1][:, None] - g.vxy[:, 1][None, :]
    return np.hypot(dx, dy)


def k_hop_ball(g: Graph, center: int, k: int) -> List[int]:
    """Vertices within k hops of center — the neighborhood primitive for
    LOCAL figure construction ("only connect stars within ~4 triangle
    edges"), as opposed to farthest-point spreads, which produce
    graph-spanning snakey shapes."""
    dist = np.full(g.nv, -1, np.int32)
    dist[center] = 0
    q = deque([center])
    while q:
        u = q.popleft()
        if dist[u] == k:
            continue
        for w, _ in g.adj[u]:
            if dist[w] < 0:
                dist[w] = dist[u] + 1
                q.append(w)
    return [int(v) for v in np.flatnonzero(dist >= 0)]


def spread_vertices(g: Graph, k: int, seed: int = 0) -> List[int]:
    """Farthest-point sampling: k vertices maximally spread in hop
    distance. Good for placing independent things far apart (pacman's
    energizers, serpent's snake starts). The WRONG tool for constellation
    figures — spreading anchors then chaining them is exactly what reads
    'snakey'; use k_hop_ball for compact shapes."""
    dist, _far = hop_distances(g)
    first = fnv(seed, 1) % g.nv
    picked = [first]
    while len(picked) < k:
        dmin = dist[picked].min(axis=0)
        picked.append(int(dmin.argmax()))
    return picked


def free_space(
    g: Graph, start_v: int, banned_c: int, occ: np.ndarray, threshold: float
) -> float:
    """Flood-filled corridor arclength reachable from start_v through
    corridors that are neither occupied (occ, bool per corridor) nor
    banned_c, early-exiting once `threshold` is exceeded. The snake
    survival check: entering a corridor is safe if the space reachable
    beyond it exceeds your own body length — worst case you circle that
    space until your tail frees more. Conservative (ignores the tail
    receding during traversal), which only ever understates safety."""
    seen_v = np.zeros(g.nv, bool)
    seen_c = np.zeros(g.nc, bool)
    seen_v[start_v] = True
    q = deque([start_v])
    total = 0.0
    while q:
        u = q.popleft()
        for w, c in g.adj[u]:
            if seen_c[c] or c == banned_c or occ[c]:
                continue
            seen_c[c] = True
            total += float(g.clen[c])
            if total >= threshold:
                return total
            if not seen_v[w]:
                seen_v[w] = True
                q.append(w)
    return total


def arrival_heading(g: Graph, c_prev: int, c_next: int) -> Optional[int]:
    """The vertex a walker passes through going c_prev -> c_next: their
    shared endpoint. THE robust way to know travel direction at a
    junction. The naive test ("is my next position further along the
    same corridor?") is a coin flip on the tick before every corridor
    switch — the pacman ghost-skirt flicker bug. Returns None if the
    corridors share no endpoint (or share both, i.e. parallel lanes:
    then the caller must disambiguate by arclength)."""
    ends_p = {int(g.cu[c_prev]), int(g.cv[c_prev])}
    ends_n = {int(g.cu[c_next]), int(g.cv[c_next])}
    shared = ends_p & ends_n
    if len(shared) == 1:
        return shared.pop()
    return None


# --- rendering on the graph -------------------------------------------


def smoothstep(v, lo: float, hi: float):
    u = np.clip((v - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def gradient_ab(hue0: float, hue1: float, chroma: float, u):
    """OKLab-vector lerp between two hues at fraction u. NEVER lerp hue
    angles directly: the (a, b) straight line dips toward the origin at
    the midpoint of a wide span — a deliberate pearl-dip (desaturation),
    not a mud sweep through intermediate hues. If you WANT the hue sweep,
    step the angle and rebuild (a, b) per step instead."""
    a0, b0 = chroma * np.cos(np.radians(hue0)), chroma * np.sin(np.radians(hue0))
    a1, b1 = chroma * np.cos(np.radians(hue1)), chroma * np.sin(np.radians(hue1))
    return a0 + (a1 - a0) * u, b0 + (b1 - b0) * u


def corridor_blob(
    g: Graph, c: int, s: float, sigma: float, spill: bool = True
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Gaussian glow centered at arclength s on corridor c, spilling
    around the corner onto adjacent corridors when it sits near a vertex.
    Agents spill; static dots don't (they'd pile into a blot at every
    junction). Returns (row arrays, weight arrays) to feed a bincount
    compositor."""
    rows: List[np.ndarray] = []
    ws: List[np.ndarray] = []
    cut = 3.0 * sigma
    inv = -0.5 / (sigma * sigma)
    lo, hi = int(g.ptr[c]), int(g.ptr[c + 1])
    d = g.arc[lo:hi] - s
    keep = np.abs(d) < cut
    if keep.any():
        rows.append(g.rows[lo:hi][keep])
        ws.append(np.exp(inv * d[keep] ** 2))
    if not spill:
        return rows, ws
    ln = float(g.clen[c])
    for v, back in ((int(g.cu[c]), s), (int(g.cv[c]), ln - s)):
        if back >= cut:
            continue
        for _w, c2 in g.adj[v]:
            if c2 == c:
                continue
            lo2, hi2 = int(g.ptr[c2]), int(g.ptr[c2 + 1])
            dv = g.arc[lo2:hi2]
            if int(g.cv[c2]) == v:
                dv = float(g.clen[c2]) - dv
            d2 = dv + back
            keep2 = d2 < cut
            if keep2.any():
                rows.append(g.rows[lo2:hi2][keep2])
                ws.append(np.exp(inv * d2[keep2] ** 2))
    return rows, ws


def vertex_blob(
    g: Graph, v: int, sigma: float, max_reach: Optional[float] = None
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Radial glow sitting ON a vertex, reaching down every incident
    corridor. `max_reach` caps the reach down EVERY arm equally (radial
    symmetry): without it a short spoke gets fully swallowed while a
    long border bar shows a partial stripe. serpent uses
    max_reach = 0.75 * (shortest incident corridor)."""
    rows: List[np.ndarray] = []
    ws: List[np.ndarray] = []
    cut = 3.0 * sigma if max_reach is None else min(3.0 * sigma, max_reach)
    inv = -0.5 / (sigma * sigma)
    for _w, c in g.adj[v]:
        lo, hi = int(g.ptr[c]), int(g.ptr[c + 1])
        d = g.arc[lo:hi]
        if int(g.cv[c]) == v:
            d = float(g.clen[c]) - d
        keep = d < cut
        if keep.any():
            rows.append(g.rows[lo:hi][keep])
            ws.append(np.exp(inv * d[keep] ** 2))
    return rows, ws


def composite(
    n: int,
    rows: Sequence[np.ndarray],
    lum: Sequence[np.ndarray],
    a_ok: Sequence[np.ndarray],
    b_ok: Sequence[np.ndarray],
    floor_l: float = 0.045,
    floor_c: float = 0.020,
    floor_h: float = 268.0,
) -> np.ndarray:
    """The house energy compositor (serpent's exact math). Overlapping
    contributions ADD in (lum, lum*a, lum*b); luminance saturates
    softly, chroma is gated so dim light stays near-neutral, hue is the
    energy-weighted OKLab angle. Returns (n, 3) OKLCH.

    Hazard to know: two overlapping elements with near-opposite hues
    cancel in (a, b) — composite chroma dives toward grey and recovers,
    and its per-frame delta scales with the elements' chroma. This is
    where dC budgets get spent; scan for it."""
    out = np.zeros((n, 3))
    out[:, 0] = floor_l
    out[:, 1] = floor_c
    out[:, 2] = floor_h
    if not rows:
        return out
    rows_all = np.concatenate(list(rows))
    lum_acc = np.bincount(rows_all, weights=np.concatenate(list(lum)), minlength=n)
    a_acc = np.bincount(rows_all, weights=np.concatenate(list(a_ok)), minlength=n)
    b_acc = np.bincount(rows_all, weights=np.concatenate(list(b_ok)), minlength=n)
    out[:, 0] = np.clip(out[:, 0] + 0.92 * (1.0 - np.exp(-1.9 * lum_acc)), 0.0, 1.0)
    chroma_mag = np.hypot(a_acc, b_acc) / np.maximum(lum_acc, 1e-6)
    add_c = np.clip(chroma_mag, 0.0, 0.37) * (1.0 - np.exp(-1.4 * lum_acc))
    out[:, 1] = np.clip(out[:, 1] + add_c, 0.0, 0.4)
    hue_field = np.degrees(np.arctan2(b_acc, a_acc)) % 360.0
    out[:, 2] = np.where(lum_acc > 1e-6, hue_field, out[:, 2])
    return out


# --- slew-fit tapering: shapes the wire can carry ----------------------


def slew_safe_sigma(
    unit: float, amplitude: float, speed: float, target: float = SLEW_TARGET_L
) -> float:
    """Minimum gaussian sigma (arclength) so a bump of peak L `amplitude`
    swept at `speed` (arclength/s) keeps any fixed light's per-frame L
    delta under `target`. A swept gaussian is a temporal gaussian of
    width sigma/speed; its steepest slope is 0.6065*A*speed/sigma.
    Floored at the craft minimum 0.19*unit — the shipped pacman blur,
    which sits exactly at the wire cap for pacman's speed."""
    needed = 0.6065 * amplitude * speed / (FPS * target)
    return max(0.19 * unit, needed)


def slew_safe_tail(
    unit: float, amplitude: float, speed: float, target: float = SLEW_TARGET_L
) -> float:
    """Minimum exponential tail length for a moving element's trailing
    edge (peak slope of A*exp(-u/tail) swept at speed is A*speed/tail).
    For pacman-style shapes prefer a tail shaped so RENDERED L falls
    linearly: energy plunges early and the hue crossover lands only
    after the light is dim — the 'dark before turn' rule."""
    needed = amplitude * speed / (FPS * target)
    return max(0.6 * unit, needed)


def max_sweep_speed(
    sigma: float, amplitude: float, target: float = SLEW_TARGET_L
) -> float:
    """Inverse of slew_safe_sigma: fastest a gaussian of this sigma and
    peak may travel."""
    return target * FPS * sigma / (0.6065 * amplitude)


def retract_duration(
    delta_arclength: float, unit: float, rate_units_per_s: float = 2.5
) -> float:
    """Minimum duration for a tail/window retraction of the given
    arclength (smoothstep-eased; the 1.5 factor budgets for smoothstep's
    peak rate). A FIXED retract time is a trap: serpent's 0.65s crash
    collapse swept a 158-unit body's tail at 3x the L cap and would have
    smeared blue down the whole dying snake. 2.5 units/s measured clean
    (falling edges 0.29, inside the 0.36 hero envelope)."""
    return 1.5 * delta_arclength / (rate_units_per_s * unit)


def scan_slew(
    render,
    lights: np.ndarray,
    t0: float,
    t1: float,
    fps: float = FPS,
    chroma_floor: float = 0.02,
) -> Dict[str, float]:
    """The mandatory verification harness: per-frame worst-case deltas of
    a pattern's render over [t0, t1]. Every first pass of every bright
    element in this codebase violated a cap in a frame that formula
    sizing missed — pop-ins, normalization unlocks, crash collapses.
    Scan windows that CONTAIN the events (a crash, an eat, a figure
    completing); a quiet window proves nothing.

    Hue deltas count only where BOTH frames carry chroma above
    `chroma_floor`: hue read off a near-grey light is noise (a light
    crossing C 0.002 -> 0.02 'swings' 80° while never being visibly
    anything but grey), and 180° flips at the OKLab chroma-null are
    benign by the same argument."""
    prev = render(lights, t0)
    n = int((t1 - t0) * fps)
    m = {"dL": 0.0, "dC": 0.0, "dH": 0.0, "dL_t": t0, "dC_t": t0, "dH_t": t0}
    for i in range(1, n + 1):
        t = t0 + i / fps
        cur = render(lights, t)
        dL = float(np.abs(cur[:, 0] - prev[:, 0]).max())
        dC = float(np.abs(cur[:, 1] - prev[:, 1]).max())
        dh = np.abs(cur[:, 2] - prev[:, 2]) % 360.0
        dh = np.minimum(dh, 360.0 - dh)
        sig = (cur[:, 1] > chroma_floor) & (prev[:, 1] > chroma_floor)
        dH = float(np.where(sig, dh, 0.0).max())
        for key, val in (("dL", dL), ("dC", dC), ("dH", dH)):
            if val > m[key]:
                m[key] = val
                m[key + "_t"] = t
        prev = cur
    return m


# --- self-check --------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    for path in sys.argv[1:] or [
        "store/lights/8da8a7abca.lights.json",
        "store/lights/e1375ee84d.lights.json",
    ]:
        from luminary.geometry.lights import LightsGeometry

        arr = LightsGeometry.load(path).array
        g = build_graph(arr)
        if g is None:
            print(f"{path}: no graph")
            continue
        deg = np.bincount([len(a_) for a_ in g.adj])
        inc = incenters(g)
        tri = triangles(g)
        dist, far = hop_distances(g)
        diam = int(dist[dist < far].max())
        # audit: no through-runs survive extraction
        thr = 0
        xy = arr[:, [LightColumns.X, LightColumns.Y]].astype(float)
        for c in range(g.nc):
            lo, hi = int(g.ptr[c]), int(g.ptr[c + 1])
            pts = xy[g.rows[lo:hi]]
            for v in range(g.nv):
                if v == int(g.cu[c]) or v == int(g.cv[c]):
                    continue
                d = np.hypot(pts[:, 0] - g.vxy[v, 0], pts[:, 1] - g.vxy[v, 1])
                k = int(d.argmin())
                if d[k] < 0.35 * g.unit and 2 <= k <= len(pts) - 3:
                    thr += 1
        print(
            f"{path}: nv={g.nv} nc={g.nc} unit={g.unit:.2f} "
            f"degree_histogram={deg.tolist()} incenters={len(inc)} "
            f"spokes={len(spokes(g))} triangles={len(tri)} "
            f"panels={len(panel_perimeters(g))} "
            f"hop_diameter={diam} through_runs={thr} "
            f"borders={len(border_vertices(g))}"
        )
        print(
            "  slew examples: sigma(A=1.0, v=2u/s) = "
            f"{slew_safe_sigma(g.unit, 1.0, 2.0 * g.unit) / g.unit:.2f}u, "
            f"max speed at craft sigma = "
            f"{max_sweep_speed(0.19 * g.unit, 1.0) / g.unit:.2f}u/s, "
            f"retract(7u of body) = "
            f"{retract_duration(7.0 * g.unit, g.unit):.1f}s"
        )
