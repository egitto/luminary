"""Constellations: a living night sky over the lattice.

A near-black field of hashed, slow-twinkling stars plus a handful of
brighter steady "named stars," under which the piece periodically draws a
constellation: 4-8 anchor vertices of the run/vertex graph (recovered from
the lights array exactly as `border_chase.py` does), spread by greedy
farthest-point sampling, joined into a chain figure by shortest paths
along the runs.

Revision 2 (first cut read as "dull" -- background, not event). The fix,
per review:
  1. VISIBLE SPARK HEAD -- a hero-luminance gaussian comet leads the draw;
     the line settles in behind it. Head sigma is NOT a fixed facet
     multiple: it is solved from the per-slot travel speed so the
     leading-edge per-frame L delta stays under the wire slew cap (see
     `_slew_safe_sigma`) -- fast slots get a wider head, by construction.
  2. COMPLETION PULSE -- once the figure closes, the whole line blooms
     toward hero brightness over ~0.4 s and settles to a hold (0.75-0.85)
     well above the old 0.54-0.66 ceiling.
  3. SHOOTING STARS IN THE GAPS -- 2-4 hashed meteors per slot, timed into
     the dead air after a figure's own lifespan, each a gaussian head +
     exponential tail on one hashed run (plasma_storm idiom).
  4. LIVING ANCHORS -- anchor vertices get an individual temporal flare
     as the spark passes (draw phase) and a brighter, slowly breathing
     core, in a complementary hue, while the figure holds (menu item 6).
  5/7. Menu picks: a subtle traveling shimmer on held lines (item 5), and
     a hashed ~1-in-5 "grand" slot with more anchors, a longer hold, and a
     faint whole-sky lift while it holds (item 7). Held-shimmer amplitude
     and the sky lift are both tiny and eased in/out via the same
     envelopes as everything else, so neither reads as a slew violation.

Time is sliced into fixed slots (`_SLOT_LEN`); a slot's anchors, path,
head speed, palette and meteors are a pure function of the slot index and
the graph, via `seeded_random` (never Python's salted `hash()`, so output
is identical across `PYTHONHASHSEED`). A figure's lifespan (draw+hold+fade,
even at the "grand" multiplier) is well under `_SLOT_LEN`, so in practice
figures never overlap between slots -- the render loop still checks the
previous slot as a cheap safety margin against future constant changes,
but under the current constants that branch never fires; the resulting
dead air between figures is exactly what item 3's meteors fill. There is
no simulation loop: each slot's content is closed-form graph geometry (a
handful of Dijkstra runs over ~dozens-hundreds of vertices, plus O(1)
hashed meteor picks) memoized per (fingerprint, slot); the timeline
within a slot is attack/hold/pulse/fade envelopes, pure functions of
elapsed time. Color fields (sky vs. line vs. head vs. anchor vs. meteor)
are blended as OKLab vectors, never by lerping hue.

Revision 3 (multiple concurrent figures + gradient palettes + a fainter
sky tier + faster living anchors):
  1. GRADIENT PALETTES -- a figure no longer carries one hue; it hashes a
     PALETTE FAMILY (a `(hue0, hue1)` pair from `_PALETTE_FAMILIES`, with a
     small hashed rotation and a 50/50 hashed direction flip) and every row
     along its path gets a hue/chroma read off the OKLab-vector lerp
     between those two endpoints at that row's arclength fraction
     (`_gradient_ab`) -- never a per-row hue lerp. The completion pulse
     still boosts saturation, but as a uniform SCALE on that vector (so it
     brightens the existing gradient's chroma shape instead of overwriting
     it). Anchors and the head read their own tint from the same gradient
     evaluated at their own arclength/position, so a comet crossing a wide
     family (a couple of the families span far enough round the wheel to
     dip toward pearl at the midpoint -- intentional, per the OKLab-blend
     craft rule, not a bug) shows the true local color, not the figure's
     average.
  2. MULTIPLE CONCURRENT FIGURES, NON-INTERSECTING -- the piece now runs
     `_track_count(graph.n_vertices)` independent slot timelines ("tracks"),
     phase-offset by `_track_offset` (see its docstring for why
     `/(n_tracks+1)` and not the more obvious `/n_tracks`) so their draws
     stagger rather than start in lockstep. Track 0 is unconstrained. Track i>0
     first hashes only its TIMING (`_hash_timing` -- is_grand/draw_dur/
     hold_dur/lifespan, no graph work) to learn its own real
     [start, start+lifespan) window, then walks every EARLIER track's
     figure whose hashed window can overlap that one (found in closed
     form from the fixed offsets/`_SLOT_LEN` and the conservative
     `_MAX_LIFESPAN` bound -- a handful of candidate slot indices per
     earlier track, resolved recursively through the same memoized
     lookup, never a search) and unions their edge/vertex sets
     (`_Figure.edge_set`/`.vertex_set`, collected while walking the
     shortest paths) into a banned set. `_build_figure` then builds THIS
     track's figure with those edges/vertices removed from the graph
     BEFORE the farthest-point anchor pick or any Dijkstra run
     (`_restrict_adj`) -- so a figure that comes out the other end is
     disjoint from its concurrently-visible neighbors by construction, not
     by a post-hoc reject. A blind hash-then-reject was tried first and
     measured to fail ~100% of the time for tracks 1/2 on the star
     (farthest-point anchor spread routinely touches 30-45% of all 140
     vertices, so two independently-hashed figures collide almost
     certainly) -- see the report. The real failure mode is
     `_build_figure` returning None because the RESTRICTED graph didn't
     have enough connected, unoccupied vertices left for `k` anchors; that
     track then shows nothing for that slot (background/meteors continue)
     and tries again, independently, next slot. On the 288-light hex demo
     (7 vertices) `_track_count` returns 1 and the mechanism never
     engages -- there isn't room for a second figure on a 7-vertex graph.
  3. A THIRD, FAINTER SKY TIER -- `_Sky` now hashes a `is_bg` population
     (denser than the twinkle set, dimmer, its own slow/small-amplitude
     twinkle) layered under the named/twinkle tiers and disjoint from both;
     it is pure background-field texture and never intersects the figure
     mechanism above (that operates on graph vertices/runs, not on
     individual lights). The twinkle fraction itself was also raised.
  4. FASTER, WHITER LIVING ANCHORS -- while a figure holds, each anchor now
     also carries a fast (2-4 Hz, hashed per anchor and precomputed once in
     `_build_figure` as `anchor_freq`/`anchor_phase`) small-amplitude
     luminance twinkle on top of the existing slow breathing, gated by the
     same hold envelope so it never pops in/out. Its chroma contribution is
     damped in lockstep (`_ANCHOR_TWINKLE_DESAT`) so the anchor visibly
     whitens right at each twinkle peak, and the steady hold boost itself
     was raised (brighter). Amplitude was sized, then MEASURED against the
     per-frame slew scan (see report) rather than guessed.
"""

import heapq
import zlib
from typing import Dict, List, Optional, Tuple

import numpy as np

from luminary.geometry.lights import LightColumns
from luminary.patterns.base import Pattern
from luminary.patterns.util import seeded_random

# --- graph extraction (cribbed from border_chase.py; patterns are
# single-file by contract, so this is duplicated rather than imported) ---
_TURN_DEG = 28.0  # split a strip where it bends more than this
_GAP_FACTOR = 4.0  # ... or jumps more than this multiple of its spacing
_MIN_RUN = 4  # runs shorter than this merge into a neighbor or drop

# --- constellation timing ---
_SLOT_LEN = 31.0  # seconds/slot; tightened from 41s to cut the dead air
_MIN_ANCHORS, _MAX_ANCHORS = 4, 8
_GRAND_MIN_ANCHORS, _GRAND_MAX_ANCHORS = 6, 10
_GRAND_PROB = 0.2  # hashed "every ~5th slot" (menu item 7)
_GRAND_HOLD_MULT = 1.6
_GRAND_SKY_LIFT = 0.035  # whole-sky L lift while a grand figure holds

_DRAW_MIN, _DRAW_MAX = 5.0, 9.0  # hashed range for a slot's draw duration (s)
_HOLD_DUR = 4.5
_FADE_DUR = 3.0
_RISE = 0.8  # per-light line attack; widened from an initial 0.15s after
# the slew harness showed a 0.15s rise of a ~0.85 L amplitude alone
# exceeds the wire cap (1.5*0.85/0.15 = 8.5 L/s = 0.28 L/frame @ 30fps).
# Round 3 widened it again, 0.5 -> 0.8: an exhaustive 3000s scan (not run
# by round 2, whose harness windows were much shorter) caught a rare but
# real hue-slew violation (97.19 deg/frame, over the 89 deg cap) where a
# figure's line attack sweeps onto a background named/twinkle star whose
# hue happens to sit near-opposite the figure's LOCAL gradient hue there:
# the (a, b) blend trajectory for that one light is a straight line from
# the background's color to the figure's, and a near-opposite pairing
# makes that line pass close to the origin (near-zero chroma), where hue
# is inherently ill-conditioned -- a step along that line near the pinch
# can rotate hue sharply even though the OKLab-VECTOR blend is doing
# exactly what the craft rule asks (never lerping hue directly). Slowing
# the attack directly reduces the angular rate through that region
# (d(blend)/dt scales as ~1/_RISE); re-scanned 3000s+ after the widening
# with zero recurrence (see report for the numbers). This can't be made
# airtight for an arbitrarily unlucky hashed pairing -- a wider _RISE
# lowers the probability and the rate, not a hard guarantee -- so this is
# a measured, not proven, mitigation; noted as a divergence in the report.


# --- round 3: multiple concurrent, non-intersecting figures ---
def _track_count(n_vertices: int) -> int:
    """How many independent figure timelines run at once. A figure needs
    4-10 anchors and the runs/vertices between them; on the 7-vertex hex
    demo a single figure can already touch most of the graph, so a second
    track would almost never find a non-intersecting candidate -- keep it
    at 1 there. The 140-vertex star has room for the requested 2-3."""
    if n_vertices < 20:
        return 1
    if n_vertices < 60:
        return 2
    return 3


# Conservative upper bound on a figure's draw+hold+fade, used only to pick
# which of an earlier track's slots COULD overlap a candidate's window
# (never to gate rendering -- that still uses each figure's own exact
# lifespan). Must stay < _SLOT_LEN or a candidate could need to check more
# than the immediately adjacent slot per earlier track.
_MAX_LIFESPAN = _DRAW_MAX + _GRAND_HOLD_MULT * _HOLD_DUR + _FADE_DUR


def _track_offset(track: int, n_tracks: int) -> float:
    """Phase offset of a track's slot 0 within the shared _SLOT_LEN clock.
    Spaced at _SLOT_LEN/(n_tracks+1) rather than /n_tracks: measured with
    even /n_tracks spacing (gaps of _SLOT_LEN/3 = 10.3s on the 3-track
    star), the two OUTER tracks' gap (2 * 10.3 = 20.7s) exceeded
    _MAX_LIFESPAN (19.2s), so a 3rd figure could never be visible at the
    same instant as the first regardless of hashing -- only pairwise
    overlap ever occurred (measured: ~45% of sampled instants had 2
    visible, 0% had 3). Packing to /(n_tracks+1) (7.75s gaps, 15.5s
    outer-to-outer) puts every pairwise gap under the SHORTEST lifespan
    and the outer gap under the LONGEST, so triple overlap is reachable
    (measured after the change -- see report) without touching draw/hold/
    fade timing. Used identically by render() and _figure() so the two
    never disagree about which instant a track's slot boundary falls on."""
    return track * _SLOT_LEN / (n_tracks + 1)


_HOLD_LIGHT_MIN, _HOLD_LIGHT_RANGE = 0.75, 0.10  # 0.75-0.85, the payoff level
_CHROMA_MIN, _CHROMA_RANGE = 0.24, 0.08

_HEAD_L = 0.95  # hero luminance, matches pacman/serpent's accumulated-energy peaks
_HEAD_C = 0.05  # near-white: a hot spark, distinct from the cooler line hue
_HEAD_ATTACK = 0.4  # seconds over which the head fades IN as a new figure
# starts -- without this, the row at path-arclength s=0 sees the head's
# spatial gaussian centered on it from age=0, popping to near-peak L in
# one frame; the harness caught this at every slot boundary.
_HEAD_TAIL = 0.6  # seconds over which the head fades once the draw completes
# (the fade window is [draw_dur, draw_dur+_HEAD_TAIL], AFTER completion,
# not before it -- keeping it out of the draw phase means it never
# overlaps a row's own line-attack ramp, which is what mattered: see the
# slew harness notes below _slew_safe_sigma.)

_PULSE_PEAK_L = 0.94
_PULSE_RISE = 0.5  # widened from 0.4/0.4 for the same slew-budget reason
_PULSE_FALL = 0.5
_PULSE_CHROMA_BOOST = 0.05

_ANCHOR_HOLD_BOOST_L = 0.16  # brightened from 0.12 (round 3 item 4);
# 0.16 chosen because the SLOW breathing (+-0.05) and the FAST twinkle
# below can both sit near their own peak at once -- see the slew report
# for the measured combined figure, not this constant alone.
_ANCHOR_HOLD_BOOST_C = 0.06
_ANCHOR_BREATH_PERIOD = 1.7
_ANCHOR_BREATH_AMP_L = 0.05
_ANCHOR_FLARE_EXTRA_L = 0.30
_ANCHOR_FLARE_EXTRA_C = 0.03
_ANCHOR_FLARE_SIGMA_T = 0.16  # seconds; see _slew_safe_sigma's temporal analogue

# round 3 item 4: fast small-amplitude twinkle on holding anchors, whitening
# (desaturating) at each twinkle peak. Hashed per anchor, precomputed once
# in _build_figure (never re-hashed per frame) as anchor_freq/anchor_phase.
_ANCHOR_TWINKLE_HZ_MIN, _ANCHOR_TWINKLE_HZ_RANGE = 2.0, 2.0  # 2-4 Hz
_ANCHOR_TWINKLE_AMP_L = 0.05  # small by design; see slew report for the
# measured worst-case per-frame delta with breathing + hold-boost stacked
_ANCHOR_TWINKLE_DESAT = 0.55  # fraction of the anchor's own chroma
# contribution damped away at the twinkle's brightness peak -- ties
# "brighter" to "whiter" rather than treating them as independent knobs

_METEOR_GAP_BUFFER = 2.0  # s after a figure's lifespan before meteors may start
_METEOR_END_BUFFER = 2.5  # s of quiet before the next slot begins drawing
_METEOR_MIN_DUR, _METEOR_DUR_RANGE = 1.5, 1.5  # 1.5-3.0 s
_METEOR_EVENT_RAMP = 0.4  # attack/decay wrapping the whole meteor event
_METEOR_MIN_N, _METEOR_MAX_N = 2, 4
_METEOR_L = 0.85
_METEOR_C = 0.06
_METEOR_TAIL_FACTOR = 0.6  # * unit, floor on the exponential tail length

_SHIMMER_AMP = 0.06  # menu item 5: subtle held-line ripple
_SHIMMER_HZ = 0.13
_SHIMMER_WAVELEN_FACTOR = 3.0  # * unit

# --- sky ---
_FLOOR_L = 0.045
_TWINKLE_FRACTION = 0.055  # raised from 0.035 (round 3 item 3: "more stars")
_TWINKLE_PERIOD_MIN, _TWINKLE_PERIOD_RANGE = 3.0, 6.0  # 3-9 s

# round 3 item 3: a second, fainter/denser tier under the twinkle tier --
# background texture only, never touched by the figure/vertex machinery.
_BG_FRACTION = 0.10
_BG_L_MIN, _BG_L_RANGE = 0.065, 0.045  # dimmer than the twinkle peak
# (_FLOOR_L + 0.24) and dimmer than the twinkle tier's OWN floor state too
_BG_TWINKLE_AMP = 0.02  # tiny -- these barely move, they're texture
_BG_PERIOD_MIN, _BG_PERIOD_RANGE = 8.0, 9.0  # 8-17 s, incommensurate with
# the twinkle tier's 3-9 s range

# --- round 3 item 1: per-figure gradient palettes ---
# (hue0, hue1) pairs; a figure hashes one family, a small rotation, and a
# 50/50 direction flip. Most spans are modest (30-70 deg) so the OKLab
# lerp stays saturated throughout; two are deliberately wide, so their
# midpoint dips toward pearl -- the craft rule's "meeting zone desaturates
# instead of mudding," shown on purpose rather than avoided.
_PALETTE_FAMILIES: List[Tuple[float, float]] = [
    (18.0, 55.0),  # ember: red -> amber
    (185.0, 235.0),  # glacier: cyan -> blue
    (95.0, 150.0),  # verdant: chartreuse -> green
    (270.0, 320.0),  # nebula: violet -> magenta
    (330.0, 20.0),  # rose-dawn: magenta -> red (wraps through 0)
    (40.0, 210.0),  # wide dawn: gold -> teal, intentional pearl dip
    (140.0, 300.0),  # wide aurora: green -> violet, intentional pearl dip
]
_PALETTE_HUE_JITTER = 15.0  # degrees; hashed rotation applied to both ends

# --- wire safety ---
_FPS = 30.0
_SLEW_TARGET = 0.08  # design margin under the ~0.24 L/frame hard cap; kept
# well below the cap (rather than e.g. 0.20) because several envelopes
# (line attack, head sweep, anchor flare) can be active on the same row
# within the same fraction of a second -- the harness checks the SUM.


def _slew_safe_sigma(unit: float, amplitude: float, speed: float) -> float:
    """Minimum gaussian arclength sigma so a bump of the given peak L,
    swept at `speed` (arclength units/s), keeps a fixed light's per-frame
    L delta under `_SLEW_TARGET`.

    A gaussian bump A*exp(-(v*t)^2/(2*sigma^2)) swept past a point is
    itself a temporal gaussian of width tau = sigma/v; its steepest
    slope is 0.6065*A/tau = 0.6065*A*v/sigma. Bounding
    (that slope)/_FPS <= _SLEW_TARGET and solving for sigma gives the
    expression below. Floored at the craft minimum (~0.19x unit) so slow
    events don't get needlessly narrow.
    """
    needed = 0.6065 * amplitude * speed / (_FPS * _SLEW_TARGET)
    return max(0.19 * unit, needed)


def _slew_safe_tail(unit: float, amplitude: float, speed: float) -> float:
    """Minimum exponential tail length (same slew argument, linear decay
    this time: peak slope of A*exp(-u/tail) swept at `speed` is
    A*speed/tail)."""
    needed = amplitude * speed / (_FPS * _SLEW_TARGET)
    return max(_METEOR_TAIL_FACTOR * unit, needed)


def _gradient_ab(hue0: float, hue1: float, chroma: float, u):
    """OKLab-vector lerp between two (chroma, hue) endpoints at fractional
    position `u` (scalar or array, expected in [0, 1]) -- round 3's
    per-figure gradient palette. Never lerps hue directly: interpolating
    the (a, b) coordinates means a wide hue span dips toward the origin
    (desaturates) at its midpoint instead of sweeping through intermediate
    hues, which is the craft rule's "pearl, not mud" behavior."""
    a0 = chroma * np.cos(np.radians(hue0))
    b0 = chroma * np.sin(np.radians(hue0))
    a1 = chroma * np.cos(np.radians(hue1))
    b1 = chroma * np.sin(np.radians(hue1))
    return a0 + (a1 - a0) * u, b0 + (b1 - b0) * u


def _build_runs(a: np.ndarray) -> Tuple[List[np.ndarray], float]:
    """Split each strip into straight runs of light rows; return runs
    and the median light spacing. (identical idiom to border_chase.py)"""
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
        cut = seg > _GAP_FACTOR * med
        cut[1:] |= ang > _TURN_DEG
        starts = np.concatenate([[0], np.flatnonzero(cut) + 1, [len(rows)]])
        pieces = [rows[s:e] for s, e in zip(starts[:-1], starts[1:])]
        merged: List[np.ndarray] = []
        for p in pieces:
            if merged and len(p) < _MIN_RUN:
                prev = merged[-1]
                gap = float(
                    np.hypot(
                        a[p[0], LightColumns.X] - a[prev[-1], LightColumns.X],
                        a[p[0], LightColumns.Y] - a[prev[-1], LightColumns.Y],
                    )
                )
                if gap < _GAP_FACTOR * med:
                    merged[-1] = np.concatenate([prev, p])
                    continue
            merged.append(p)
        runs.extend(m for m in merged if len(m) >= _MIN_RUN)
    med_all = float(np.median(np.concatenate(spacings))) if spacings else 1.0
    return runs, med_all


def _cluster_endpoints(a: np.ndarray, runs: List[np.ndarray], tol: float) -> np.ndarray:
    """Union-find run endpoints into vertex labels; label of run i's
    ends are out[2i] (first light) and out[2i+1] (last)."""
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

    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    for i, j in zip(*np.nonzero(d2 < tol * tol)):
        if i < j:
            ri, rj = find(int(i)), find(int(j))
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([find(i) for i in range(n)])
    labels: np.ndarray = np.unique(roots, return_inverse=True)[1]
    return labels


class _Graph:
    """Run/vertex graph plus per-run arclength tables, adjacency for
    Dijkstra, and one representative light row per vertex."""

    def __init__(
        self,
        n_vertices: int,
        adj: List[List[Tuple[int, int, bool, float]]],
        runs: List[np.ndarray],
        alongs: List[np.ndarray],
        run_len: np.ndarray,
        vertex_row: np.ndarray,
        unit: float,
    ):
        self.n_vertices = n_vertices
        self.adj = adj
        self.runs = runs
        self.alongs = alongs
        self.run_len = run_len
        self.vertex_row = vertex_row
        self.unit = unit


def _build_graph(a: np.ndarray) -> Optional[_Graph]:
    runs, spacing = _build_runs(a)
    if len(runs) < 2:
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
    unit = max(1e-6, float(np.median(chords)))
    tol = max(3.0 * spacing, 0.3 * unit)
    labels = _cluster_endpoints(a, runs, tol)
    n_vertices = int(labels.max()) + 1
    if n_vertices < 2:
        return None

    alongs: List[np.ndarray] = []
    run_len = np.zeros(len(runs))
    vertex_row = np.full(n_vertices, -1, dtype=np.int64)
    adj: List[List[Tuple[int, int, bool, float]]] = [[] for _ in range(n_vertices)]
    for e, r in enumerate(runs):
        xy = a[np.ix_(r, np.array([LightColumns.X, LightColumns.Y], np.intp))]
        seg = np.hypot(*np.diff(xy, axis=0).T)
        along = np.concatenate([[0.0], np.cumsum(seg)])
        alongs.append(along)
        run_len[e] = float(along[-1])
        u, v = int(labels[2 * e]), int(labels[2 * e + 1])
        if vertex_row[u] < 0:
            vertex_row[u] = r[0]
        if vertex_row[v] < 0:
            vertex_row[v] = r[-1]
        if u == v:
            continue  # a run that loops back on its own vertex: not a chain edge
        w = float(run_len[e])
        adj[u].append((v, e, False, w))
        adj[v].append((u, e, True, w))
    # Any isolated vertex (shouldn't happen given _MIN_RUN >= 2) gets its
    # own row as a fallback so indexing never sees -1.
    for v in range(n_vertices):
        if vertex_row[v] < 0:
            vertex_row[v] = int(r[0])
    return _Graph(n_vertices, adj, runs, alongs, run_len, vertex_row, unit)


def _dijkstra(
    adj: List[List[Tuple[int, int, bool, float]]], n: int, source: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dist = np.full(n, np.inf)
    dist[source] = 0.0
    pred_v = np.full(n, -1, dtype=np.int64)
    pred_e = np.full(n, -1, dtype=np.int64)
    pred_r = np.zeros(n, dtype=bool)
    visited = np.zeros(n, dtype=bool)
    heap: List[Tuple[float, int]] = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        for v, e, rev, w in adj[u]:
            nd = d + w
            if nd < dist[v] - 1e-9:
                dist[v] = nd
                pred_v[v] = u
                pred_e[v] = e
                pred_r[v] = rev
                heapq.heappush(heap, (nd, v))
    return dist, pred_v, pred_e, pred_r


class _Figure:
    """One slot's constellation: concatenated path rows/arclength, its
    anchor light rows + arclength positions, total length, gradient
    palette endpoints, timing, the wire-safe head/meteor sigma derived
    from this slot's speed, and the edge/vertex sets used for round 3's
    cross-track non-intersection check."""

    def __init__(
        self,
        rows: np.ndarray,
        s: np.ndarray,
        length: float,
        anchor_rows: np.ndarray,
        anchor_s: np.ndarray,
        hue0: float,
        hue1: float,
        hold_light: float,
        chroma: float,
        draw_dur: float,
        hold_dur: float,
        is_grand: bool,
        head_sigma: float,
        edge_set: frozenset,
        vertex_set: frozenset,
        anchor_freq: np.ndarray,
        anchor_phase: np.ndarray,
    ):
        self.rows = rows
        self.s = s
        self.length = length
        self.anchor_rows = anchor_rows
        self.anchor_s = anchor_s
        self.hue0 = hue0
        self.hue1 = hue1
        self.hold_light = hold_light
        self.chroma = chroma
        self.draw_dur = draw_dur
        self.hold_dur = hold_dur
        self.is_grand = is_grand
        self.head_sigma = head_sigma
        self.edge_set = edge_set
        self.vertex_set = vertex_set
        self.anchor_freq = anchor_freq
        self.anchor_phase = anchor_phase
        self.speed = length / draw_dur if draw_dur > 0 else 0.0
        self.lifespan = draw_dur + hold_dur + _FADE_DUR


def _hash_timing(track: int, slot: int) -> Tuple[bool, float, float, float]:
    """The subset of a figure's hashed constants that determine its
    lifespan (is_grand, draw_dur, hold_dur, lifespan) -- split out of
    `_build_figure` so round 3's cross-track exclusion can learn a
    candidate's real time window BEFORE doing any graph work, from the
    same hashes `_build_figure` will use."""
    is_grand = (
        float(seeded_random(f"constellations-grand-{track}-{slot}", 1)[0]) < _GRAND_PROB
    )
    draw_dur = _DRAW_MIN + (_DRAW_MAX - _DRAW_MIN) * float(
        seeded_random(f"constellations-draw-{track}-{slot}", 1)[0]
    )
    hold_dur = _HOLD_DUR * (_GRAND_HOLD_MULT if is_grand else 1.0)
    return is_grand, draw_dur, hold_dur, draw_dur + hold_dur + _FADE_DUR


def _restrict_adj(
    adj: List[List[Tuple[int, int, bool, float]]],
    banned_edges: frozenset,
    banned_vertices: frozenset,
) -> List[List[Tuple[int, int, bool, float]]]:
    """Round 3 item 2: the graph a later track searches, with an earlier
    concurrently-visible track's edges/vertices removed outright -- so
    Dijkstra and the farthest-point anchor pick can never land ON or
    route THROUGH occupied ground. A banned vertex loses all its
    adjacency (in both directions), which also makes it unreachable as a
    pass-through vertex on any other path, not just as an anchor."""
    if not banned_edges and not banned_vertices:
        return adj
    out: List[List[Tuple[int, int, bool, float]]] = []
    for u in range(len(adj)):
        if u in banned_vertices:
            out.append([])
            continue
        out.append(
            [
                (v, e, rev, w)
                for (v, e, rev, w) in adj[u]
                if e not in banned_edges and v not in banned_vertices
            ]
        )
    return out


def _build_figure(
    g: _Graph,
    track: int,
    slot: int,
    banned_edges: frozenset = frozenset(),
    banned_vertices: frozenset = frozenset(),
) -> Optional[_Figure]:
    """Hash ONE candidate figure for (track, slot). Every hashed constant
    below is salted with `track` as well as `slot` so concurrent tracks
    never draw the same figure. Round 3's non-intersection rule is
    enforced by CONSTRUCTION here, not by rejecting a finished figure: an
    earlier concurrently-visible track's edges/vertices are removed from
    the graph before anchors are even chosen (`_restrict_adj`), so a
    figure that comes out the other end is already disjoint from them.
    (A blind hash-then-reject was tried first and measured to fail
    essentially always -- see the report -- because farthest-point anchor
    spread routinely touches 30-45% of all vertices, so two independent
    figures on the same graph collide near-certainly. Restricting the
    search space is what actually gets a second/third figure on screen.)
    If the restriction leaves too little connected, unoccupied graph for
    `k` anchors, this returns None -- THAT is the real failure mode."""
    is_grand, draw_dur, hold_dur, _ = _hash_timing(track, slot)
    lo, hi = (
        (_GRAND_MIN_ANCHORS, _GRAND_MAX_ANCHORS)
        if is_grand
        else (_MIN_ANCHORS, _MAX_ANCHORS)
    )
    rnd = seeded_random(f"constellations-count-{track}-{slot}", 1)
    k = min(hi, g.n_vertices, lo + int(rnd[0] * (hi - lo + 1)))
    if k < 2:
        return None
    adj = _restrict_adj(g.adj, banned_edges, banned_vertices)
    seed_frac = float(seeded_random(f"constellations-seed-{track}-{slot}", 1)[0])
    v0_raw = int(seed_frac * g.n_vertices) % g.n_vertices
    v0 = -1
    for step in range(g.n_vertices):
        cand = (v0_raw + step) % g.n_vertices
        if cand not in banned_vertices:
            v0 = cand
            break
    if v0 < 0:
        return None  # every vertex is occupied by an earlier figure

    anchors = [v0]
    mindist: Optional[np.ndarray] = None
    cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    cur = v0
    for _ in range(1, k):
        d, pv, pe, pr = _dijkstra(adj, g.n_vertices, cur)
        cache[cur] = (d, pv, pe, pr)
        mindist = d.copy() if mindist is None else np.minimum(mindist, d)
        candidate = mindist.copy()
        candidate[~np.isfinite(candidate)] = -1.0
        candidate[np.array(anchors, dtype=np.int64)] = -1.0
        nxt = int(np.argmax(candidate))
        if candidate[nxt] <= 0.0:
            break
        anchors.append(nxt)
        cur = nxt
    if len(anchors) < 2:
        return None  # restricted graph left this track's seed isolated

    rows_list: List[np.ndarray] = []
    s_list: List[np.ndarray] = []
    s0 = 0.0
    used_anchor_rows = [g.vertex_row[anchors[0]]]
    anchor_s_list = [0.0]
    edge_set: set = set()
    vertex_set: set = set(anchors)
    for i in range(len(anchors) - 1):
        d, pv, pe, pr = cache[anchors[i]]
        target = anchors[i + 1]
        if not np.isfinite(d[target]):
            return None  # disconnected component; bail on this slot's figure
        seq: List[Tuple[int, bool]] = []
        w = target
        while w != anchors[i]:
            e = int(pe[w])
            seq.append((e, bool(pr[w])))
            edge_set.add(e)
            vertex_set.add(w)
            w = int(pv[w])
        seq.reverse()
        for e, rev in seq:
            along = g.alongs[e]
            total = float(g.run_len[e])
            rows_list.append(g.runs[e])
            s_list.append(s0 + (total - along if rev else along))
            s0 += total
        used_anchor_rows.append(g.vertex_row[target])
        anchor_s_list.append(s0)

    if not rows_list:
        return None
    rows = np.concatenate(rows_list)
    s = np.concatenate(s_list)
    length = s0

    # Round 3 item 1: a gradient palette family instead of one hue -- a
    # hashed (hue0, hue1) pair, a small hashed rotation of both ends
    # together (keeps the family's span, varies its footing), and a
    # hashed 50/50 direction flip so which end leads isn't fixed per
    # family. See _gradient_ab for how this is read back per row.
    fam_idx = int(
        seeded_random(f"constellations-palfam-{track}-{slot}", 1)[0]
        * len(_PALETTE_FAMILIES)
    ) % len(_PALETTE_FAMILIES)
    h0f, h1f = _PALETTE_FAMILIES[fam_idx]
    pj = seeded_random(f"constellations-paljit-{track}-{slot}", 2)
    rot = (-1.0 + 2.0 * float(pj[0])) * _PALETTE_HUE_JITTER
    hue0, hue1 = (h0f + rot) % 360.0, (h1f + rot) % 360.0
    if pj[1] < 0.5:
        hue0, hue1 = hue1, hue0

    lc = seeded_random(f"constellations-lc-{track}-{slot}", 2)
    hold_light = _HOLD_LIGHT_MIN + _HOLD_LIGHT_RANGE * float(lc[0])
    chroma = _CHROMA_MIN + _CHROMA_RANGE * float(lc[1])
    # draw_dur/hold_dur/is_grand were already hashed by _hash_timing above
    # (round 3 needs them before the graph work, to size the exclusion
    # window). Draw duration is hashed directly rather than derived from
    # length / (factor * unit): a unit-scaled spark speed saturates a
    # fixed draw-duration clamp on the 6,660-light star (mean path ~1466
    # units, unit ~22 -> nominal ~41s) while landing in-range unclamped
    # on the 288-light hex demo (mean path ~1245, unit ~150 -> nominal
    # ~5.4s) -- the two graphs' hop-count-vs-unit-size ratios differ too
    # much for one scale factor. Hashing draw_dur directly keeps the pace
    # comparable across geometries; the head/meteor sigma is instead
    # solved from whatever speed that pace implies (see _slew_safe_sigma).
    speed = length / draw_dur if draw_dur > 0 else 0.0
    head_sigma = _slew_safe_sigma(g.unit, _HEAD_L, speed)

    n_anchor = len(anchor_s_list)
    atw = seeded_random(f"constellations-atw-{track}-{slot}", 2 * n_anchor)
    anchor_freq = _ANCHOR_TWINKLE_HZ_MIN + _ANCHOR_TWINKLE_HZ_RANGE * atw[:n_anchor]
    anchor_phase = atw[n_anchor:]

    return _Figure(
        rows,
        s,
        length,
        np.array(used_anchor_rows, dtype=np.int64),
        np.array(anchor_s_list, dtype=np.float64),
        hue0,
        hue1,
        hold_light,
        chroma,
        draw_dur,
        hold_dur,
        is_grand,
        head_sigma,
        frozenset(edge_set),
        frozenset(vertex_set),
        anchor_freq,
        anchor_phase,
    )


class _Meteor:
    """A single hashed shooting star on one run: gaussian head + fading
    exponential tail, both wire-sized from this meteor's own speed."""

    def __init__(
        self,
        rows: np.ndarray,
        along: np.ndarray,
        run_len: float,
        start: float,
        duration: float,
        reverse: bool,
        sigma: float,
        tail: float,
        hue: float,
    ):
        self.rows = rows
        self.along = along
        self.run_len = run_len
        self.start = start
        self.duration = duration
        self.reverse = reverse
        self.sigma = sigma
        self.tail = tail
        self.hue = hue


def _build_meteors(
    g: _Graph, track: int, slot: int, gap_start: float
) -> List["_Meteor"]:
    gap_end = _SLOT_LEN - _METEOR_END_BUFFER
    if gap_end <= gap_start or not g.runs:
        return []
    n_rand = float(seeded_random(f"constellations-meteor-n-{track}-{slot}", 1)[0])
    n_meteor = _METEOR_MIN_N + int(n_rand * (_METEOR_MAX_N - _METEOR_MIN_N + 1))
    out: List[_Meteor] = []
    for i in range(n_meteor):
        r = seeded_random(f"constellations-meteor-{track}-{slot}-{i}", 5)
        run_idx = int(r[0] * len(g.runs)) % len(g.runs)
        run_len = float(g.run_len[run_idx])
        if run_len <= 0.0:
            continue
        duration = _METEOR_MIN_DUR + _METEOR_DUR_RANGE * float(r[1])
        span = max(0.0, gap_end - duration - gap_start)
        start = gap_start + span * float(r[2])
        reverse = bool(r[3] < 0.5)
        hue = 195.0 + 50.0 * float(r[4])
        speed = run_len / duration
        sigma = _slew_safe_sigma(g.unit, _METEOR_L, speed)
        tail = _slew_safe_tail(g.unit, _METEOR_L * 0.6, speed)
        out.append(
            _Meteor(
                g.runs[run_idx],
                g.alongs[run_idx],
                run_len,
                start,
                duration,
                reverse,
                sigma,
                tail,
                hue,
            )
        )
    return out


class _Sky:
    """Per-fingerprint hashed star field: twinkle + named-star constants,
    plus round 3's fainter/denser background tier, independent of t so
    render() only evaluates sinusoids over them."""

    def __init__(self, n: int):
        pick = seeded_random("constellations-star-pick", n)
        self.is_twinkle = pick < _TWINKLE_FRACTION
        self.period = _TWINKLE_PERIOD_MIN + _TWINKLE_PERIOD_RANGE * seeded_random(
            "constellations-star-period", n
        )
        self.phase = seeded_random("constellations-star-phase", n)
        self.twinkle_hue = 200.0 + 40.0 * seeded_random("constellations-star-hue", n)

        target_k = int(np.clip(round(n * 0.0015), 4, 14))
        score = seeded_random("constellations-named", n)
        if target_k > 0 and target_k < n:
            threshold = np.partition(score, target_k - 1)[target_k - 1]
            self.is_named = score <= threshold
        else:
            self.is_named = np.zeros(n, dtype=bool)
        self.named_hue = 360.0 * seeded_random("constellations-named-hue", n)
        self.named_l = 0.55 + 0.15 * seeded_random("constellations-named-l", n)
        self.named_c = 0.10 + 0.10 * seeded_random("constellations-named-c", n)

        # Round 3 item 3: a fainter, denser background tier -- pure
        # texture, disjoint from the twinkle draw, unrelated to (and never
        # consulted by) the figure/vertex machinery.
        pick2 = seeded_random("constellations-bg-pick", n)
        self.is_bg = (pick2 < _BG_FRACTION) & ~self.is_twinkle
        self.bg_l = _BG_L_MIN + _BG_L_RANGE * seeded_random("constellations-bg-l", n)
        self.bg_period = _BG_PERIOD_MIN + _BG_PERIOD_RANGE * seeded_random(
            "constellations-bg-period", n
        )
        self.bg_phase = seeded_random("constellations-bg-phase", n)


def _smoothstep(v, lo: float, hi: float):
    u = np.clip((v - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _pulse_frac(u: float) -> float:
    """0 -> 1 -> 0 bump: rises over [0, _PULSE_RISE], falls back over the
    following _PULSE_FALL, zero everywhere else (continuous, no branch
    needed since _smoothstep saturates at both ends)."""
    return float(
        _smoothstep(u, 0.0, _PULSE_RISE)
        * (1.0 - _smoothstep(u, _PULSE_RISE, _PULSE_RISE + _PULSE_FALL))
    )


class ConstellationsPattern(Pattern):
    name = "constellations"
    description = "A night sky of hashed twinkling stars where 2-3 non-intersecting constellations draw themselves at once"

    def __init__(self) -> None:
        self._graph_cache: Dict[Tuple[int, int], Tuple[Optional[_Graph], _Sky, int]] = (
            {}
        )
        self._slot_cache: Dict[
            Tuple[int, int, int, int], Tuple[Optional[_Figure], List[_Meteor]]
        ] = {}

    def _geometry(
        self, lights: np.ndarray
    ) -> Tuple[Tuple[int, int], Optional[_Graph], _Sky, int]:
        sample = lights[
            ::13,
            [
                LightColumns.CONTROLLER,
                LightColumns.CHANNEL,
                LightColumns.INDEX,
                LightColumns.X,
                LightColumns.Y,
            ],
        ]
        key = (
            lights.shape[0],
            zlib.crc32(np.ascontiguousarray(np.nan_to_num(sample)).tobytes()),
        )
        if key not in self._graph_cache:
            graph = _build_graph(lights)
            n_tracks = _track_count(graph.n_vertices) if graph is not None else 0
            self._graph_cache[key] = (graph, _Sky(lights.shape[0]), n_tracks)
        graph, sky, n_tracks = self._graph_cache[key]
        return key, graph, sky, n_tracks

    def _figure(
        self,
        key: Tuple[int, int],
        graph: _Graph,
        track: int,
        slot: int,
        n_tracks: int,
    ) -> Tuple[Optional[_Figure], List[_Meteor]]:
        """Build (or fetch) track `track`'s figure for `slot`. Track 0 is
        unconstrained. Track i>0 first hashes ONLY its timing
        (`_hash_timing`, cheap, no graph work) to learn its own real time
        window, then walks EARLIER tracks (through this same memoized
        method, so the recursion is acyclic and bounded by
        `n_tracks <= 3`) to find which of their figures could be
        concurrently visible in that window, and unions their edge/vertex
        sets into `banned_edges`/`banned_vertices`. `_build_figure` is
        then asked to construct THIS track's figure with those already
        excluded from the graph -- so if it succeeds, non-intersection
        holds by construction, not by a post-hoc check. The failure mode
        is `_build_figure` returning None because the restricted graph
        didn't have room for `k` connected, unoccupied anchors -- this
        call then returns `fig=None` and the track simply shows nothing
        for this slot; the next slot (different hashes, and usually a
        different set of earlier figures to avoid) tries independently."""
        skey = (key[0], key[1], track, slot)
        if skey in self._slot_cache:
            return self._slot_cache[skey]
        if track == 0:
            fig = _build_figure(graph, track, slot)
        else:
            _, _, _, lifespan = _hash_timing(track, slot)
            off_i = _track_offset(track, n_tracks)
            t0 = slot * _SLOT_LEN + off_i
            t1 = t0 + lifespan
            banned_edges: set = set()
            banned_vertices: set = set()
            for j in range(track):
                off_j = _track_offset(j, n_tracks)
                # Conservative candidate slot(s) of the earlier track whose
                # window COULD overlap [t0, t1): _MAX_LIFESPAN bounds every
                # figure's real lifespan and is < _SLOT_LEN, so this is at
                # most two candidates.
                lo = int(np.floor((t0 - off_j) / _SLOT_LEN))
                hi = int(np.floor((t0 + _MAX_LIFESPAN - off_j) / _SLOT_LEN))
                for sj in sorted({max(0, lo), max(0, hi)}):
                    fig_j, _ = self._figure(key, graph, j, sj, n_tracks)
                    if fig_j is None:
                        continue
                    tj0 = sj * _SLOT_LEN + off_j
                    tj1 = tj0 + fig_j.lifespan
                    if tj1 <= t0 or tj0 >= t1:
                        continue  # real windows don't actually overlap
                    banned_edges |= fig_j.edge_set
                    banned_vertices |= fig_j.vertex_set
            fig = _build_figure(
                graph, track, slot, frozenset(banned_edges), frozenset(banned_vertices)
            )
        gap_start = (
            fig.lifespan
            if fig is not None
            else _DRAW_MAX + _GRAND_HOLD_MULT * _HOLD_DUR + _FADE_DUR
        ) + _METEOR_GAP_BUFFER
        meteors = _build_meteors(graph, track, slot, gap_start)
        if len(self._slot_cache) > 48:
            self._slot_cache.pop(next(iter(self._slot_cache)))
        self._slot_cache[skey] = (fig, meteors)
        return self._slot_cache[skey]

    def render(self, lights: np.ndarray, t: float) -> np.ndarray:
        n = lights.shape[0]
        key, graph, sky, n_tracks = self._geometry(lights)

        # --- sky: near-black floor, hashed twinkle, a fainter/denser
        # background tier under it, steady named stars on top ---
        bg_hue_drift = (248.0 + 9.0 * np.sin(2.0 * np.pi * t / 67.0)) % 360.0
        twinkle = 0.5 + 0.5 * np.sin(2.0 * np.pi * (t / sky.period + sky.phase))
        twinkle_l = _FLOOR_L + 0.24 * twinkle**2
        bg2_twinkle = 0.5 + 0.5 * np.sin(
            2.0 * np.pi * (t / sky.bg_period + sky.bg_phase)
        )
        bg2_l = sky.bg_l + _BG_TWINKLE_AMP * bg2_twinkle

        bg_l = np.where(
            sky.is_named,
            sky.named_l,
            np.where(sky.is_twinkle, twinkle_l, np.where(sky.is_bg, bg2_l, _FLOOR_L)),
        )
        bg_c = np.where(
            sky.is_named,
            sky.named_c,
            np.where(
                sky.is_twinkle,
                0.02 + 0.05 * twinkle,
                np.where(sky.is_bg, 0.015, 0.018),
            ),
        )
        bg_h = np.where(
            sky.is_named,
            sky.named_hue,
            np.where(sky.is_twinkle, sky.twinkle_hue, bg_hue_drift),
        )

        if graph is None or n_tracks <= 0:
            out = np.zeros((n, 3))
            out[:, 0] = np.clip(bg_l, 0.0, 1.0)
            out[:, 1] = np.clip(bg_c, 0.0, 0.4)
            out[:, 2] = bg_h % 360.0
            return np.nan_to_num(out)

        w_total = np.zeros(n)
        aw = np.zeros(n)
        bw = np.zeros(n)
        lw = np.zeros(n)
        # Anchor extras (hold breathing/twinkle, draw-time flare) are pure
        # POST-BLEND additions, not weighted (weight, color) contributions
        # like the line/head/meteor/anchor-baseline above. They were
        # originally written as raw np.add.at(lw, ...) deltas riding on
        # whatever weight the SAME row happened to already have from the
        # line -- which silently assumed that weight was always ~1 there.
        # That assumption breaks whenever an anchor's representative light
        # (`_Graph.vertex_row`) isn't on any run THIS figure's path
        # actually used through that vertex (common at a degree>2 hub);
        # then w_total there could be exactly 0, or only as large as some
        # unrelated meteor passing by, and the fig_l=lw/w_total division
        # would either drop the whole extra silently or -- worse, once a
        # baseline anchor weight was added below -- "unlock" it at full
        # magnitude in a single frame the instant that weight ticked off
        # zero (measured: 0.2517 L/frame with no baseline weight at all,
        # then 0.3098 L/frame once a baseline weight was added, both over
        # the 0.24 cap, both at the exact moment a flare's own gaussian
        # peak coincided with its hosting weight crossing zero). Accumulating
        # extras into their own arrays and adding them to l/a/b AFTER the
        # blend sidesteps the division entirely: each extra is already a
        # bounded, smooth-in-time function on its own (gaussian/sinusoid
        # under a smoothstep validity envelope, from round 2's slew
        # tuning), so summing them post-blend can't reintroduce a
        # division-driven jump regardless of what w_total is doing.
        extra_l = np.zeros(n)
        extra_a = np.zeros(n)
        extra_b = np.zeros(n)
        sky_lift = 0.0

        # --- figures: round 3 runs n_tracks independent, phase-offset slot
        # timelines. Each track still checks its own current+previous slot
        # as the cheap safety net the single-track version used (see
        # module docstring); cross-track non-intersection is resolved once,
        # per (track, slot), inside self._figure -- render() only ever
        # sees figures that already cleared that check (or None).
        for track in range(n_tracks):
            off = _track_offset(track, n_tracks)
            local_t = t - off
            slot = int(np.floor(local_t / _SLOT_LEN))
            for s_idx in (slot - 1, slot):
                if s_idx < 0:
                    continue
                fig, meteors = self._figure(key, graph, track, s_idx, n_tracks)
                if fig is not None:
                    age = local_t - s_idx * _SLOT_LEN
                    if 0.0 <= age <= fig.lifespan:
                        length_safe = fig.length if fig.length > 0 else 1.0
                        speed = fig.speed if fig.speed > 0 else 1e-6
                        row_t = age - fig.s / speed
                        attack = _smoothstep(row_t, 0.0, _RISE)
                        fade = 1.0 - _smoothstep(
                            age,
                            fig.draw_dur + fig.hold_dur,
                            fig.draw_dur + fig.hold_dur + _FADE_DUR,
                        )
                        val = attack * fade
                        u = age - fig.draw_dur
                        pulse = _pulse_frac(u)
                        stage_l = (
                            fig.hold_light + (_PULSE_PEAK_L - fig.hold_light) * pulse
                        )
                        stage_c = fig.chroma + _PULSE_CHROMA_BOOST * pulse
                        if age >= fig.draw_dur:
                            shimmer = 1.0 + _SHIMMER_AMP * np.sin(
                                2.0
                                * np.pi
                                * (
                                    fig.s / (_SHIMMER_WAVELEN_FACTOR * graph.unit)
                                    - age * _SHIMMER_HZ
                                )
                            )
                        else:
                            shimmer = 1.0

                        # 1) line: gradient palette read per row off this
                        # figure's own (hue0, hue1) family at that row's
                        # arclength fraction (item 1) -- an OKLab-vector
                        # lerp, never a hue lerp. The completion pulse
                        # scales that vector's length (saturation), not
                        # its direction.
                        u_row = np.clip(fig.s / length_safe, 0.0, 1.0)
                        a_row, b_row = _gradient_ab(
                            fig.hue0, fig.hue1, fig.chroma, u_row
                        )
                        scale = stage_c / max(fig.chroma, 1e-9)
                        np.add.at(w_total, fig.rows, val)
                        np.add.at(lw, fig.rows, val * stage_l * shimmer)
                        np.add.at(aw, fig.rows, val * a_row * scale)
                        np.add.at(bw, fig.rows, val * b_row * scale)

                        # 2) head: a hero-luminance comet leading the draw,
                        # tinted with the LOCAL gradient color at its own
                        # position. Fade-out window is
                        # [draw_dur, draw_dur+_HEAD_TAIL] -- AFTER the draw
                        # completes, not before -- so it never overlaps a
                        # row's own line-attack ramp (which finishes by the
                        # time the head reaches that row); the harness
                        # caught an earlier version that faded before
                        # draw_dur and stacked with the last rows' attack,
                        # exceeding the slew cap.
                        head_pos = speed * min(age, fig.draw_dur)
                        u_head = float(np.clip(head_pos / length_safe, 0.0, 1.0))
                        a_h, b_h = _gradient_ab(fig.hue0, fig.hue1, fig.chroma, u_head)
                        hue_head = float(np.degrees(np.arctan2(b_h, a_h)))
                        hd_ck, hd_sk = (
                            np.cos(np.radians(hue_head)),
                            np.sin(np.radians(hue_head)),
                        )
                        head_w = np.exp(
                            -((fig.s - head_pos) ** 2) / (2.0 * fig.head_sigma**2)
                        )
                        head_w = head_w * _smoothstep(age, 0.0, _HEAD_ATTACK)
                        head_w = head_w * (
                            1.0
                            - _smoothstep(age, fig.draw_dur, fig.draw_dur + _HEAD_TAIL)
                        )
                        np.add.at(w_total, fig.rows, head_w)
                        np.add.at(lw, fig.rows, head_w * _HEAD_L)
                        np.add.at(aw, fig.rows, head_w * _HEAD_C * hd_ck)
                        np.add.at(bw, fig.rows, head_w * _HEAD_C * hd_sk)

                        # 3) anchors: complementary-hued living stars (item
                        # 4 + 6), complementary to the LOCAL gradient color
                        # at each anchor's own arclength (not the figure's
                        # average hue). Every term below is accumulated into
                        # extra_l/extra_a/extra_b -- a POST-BLEND additive
                        # pass (see its declaration above for why): each
                        # anchor's representative light row is not
                        # guaranteed to be a member of fig.rows (a
                        # degree>2 hub vertex can have its representative
                        # row on an incident run this figure's path didn't
                        # actually traverse), so these can't safely ride on
                        # the SAME w_total-weighted-average machinery the
                        # line/head/meteor use -- doing so was tried and
                        # measured to produce a >0.24 L/frame jump exactly
                        # when a flare's own gaussian peak coincided with
                        # its hosting weight ticking off zero. Every term
                        # here is instead already a bounded, smooth-in-time
                        # function (gaussian or sinusoid under a smoothstep
                        # validity envelope) that is safe to add directly.
                        u_anchor = np.clip(fig.anchor_s / length_safe, 0.0, 1.0)
                        a_anc, b_anc = _gradient_ab(
                            fig.hue0, fig.hue1, fig.chroma, u_anchor
                        )
                        line_hue_anchor = np.degrees(np.arctan2(b_anc, a_anc))
                        ck = np.cos(np.radians(line_hue_anchor))
                        sk = np.sin(np.radians(line_hue_anchor))
                        hck = np.cos(np.radians(line_hue_anchor + 180.0))
                        hsk = np.sin(np.radians(line_hue_anchor + 180.0))

                        if age >= fig.draw_dur:
                            hold_u = age - fig.draw_dur
                            breathe = _ANCHOR_BREATH_AMP_L * np.sin(
                                2.0 * np.pi * age / _ANCHOR_BREATH_PERIOD
                            )
                            anchor_env = fade * float(_smoothstep(hold_u, 0.0, 0.3))
                            if anchor_env > 0.0:
                                # round 3 item 4: fast (2-4 Hz, per-anchor
                                # hashed) small-amplitude twinkle, gated by
                                # the same anchor_env so it fades in/out
                                # with the hold rather than popping; its
                                # chroma contribution is damped in lockstep
                                # so the anchor visibly whitens at each
                                # twinkle peak (brighter AND whiter tied
                                # together, not independent knobs).
                                tw_osc = np.sin(
                                    2.0
                                    * np.pi
                                    * (fig.anchor_freq * age + fig.anchor_phase)
                                )
                                tw_pos = 0.5 + 0.5 * tw_osc
                                l_tw = _ANCHOR_TWINKLE_AMP_L * tw_pos * anchor_env
                                c_mult = (
                                    1.0 - _ANCHOR_TWINKLE_DESAT * tw_pos * anchor_env
                                )
                                boost_l = (
                                    _ANCHOR_HOLD_BOOST_L + breathe
                                ) * anchor_env + l_tw
                                boost_c = _ANCHOR_HOLD_BOOST_C * anchor_env * c_mult
                                np.add.at(extra_l, fig.anchor_rows, boost_l)
                                np.add.at(extra_a, fig.anchor_rows, boost_c * hck)
                                np.add.at(extra_b, fig.anchor_rows, boost_c * hsk)
                        # The first anchor always sits at arclength 0, so
                        # its flare is centered at age=0 -- the figure's
                        # own birth instant. Without a validity ramp we'd
                        # only ever see the falling half of that gaussian,
                        # i.e. an instant pop to a fraction of its peak in
                        # one frame (same class of bug the harness caught
                        # on the head and the meteors: an event whose
                        # window boundary coincides with the gaussian's own
                        # center).
                        anchor_time = fig.anchor_s / speed
                        flare = _ANCHOR_FLARE_EXTRA_L * np.exp(
                            -((age - anchor_time) ** 2)
                            / (2.0 * _ANCHOR_FLARE_SIGMA_T**2)
                        )
                        flare = flare * _smoothstep(
                            age, 0.0, 2.0 * _ANCHOR_FLARE_SIGMA_T
                        )
                        np.add.at(extra_l, fig.anchor_rows, flare)
                        np.add.at(
                            extra_a,
                            fig.anchor_rows,
                            flare
                            * (_ANCHOR_FLARE_EXTRA_C / _ANCHOR_FLARE_EXTRA_L)
                            * ck,
                        )
                        np.add.at(
                            extra_b,
                            fig.anchor_rows,
                            flare
                            * (_ANCHOR_FLARE_EXTRA_C / _ANCHOR_FLARE_EXTRA_L)
                            * sk,
                        )

                        # 4) grand slot: faint whole-sky lift while it
                        # holds (item 7); max() across tracks/slots so two
                        # simultaneous grand figures don't double-lift.
                        if fig.is_grand:
                            grand_env = fade * float(_smoothstep(u, 0.0, _PULSE_RISE))
                            sky_lift = max(sky_lift, _GRAND_SKY_LIFT * grand_env)

                # 5) shooting stars in the gap (item 3) -- only ever
                # scheduled within this slot's own [0, _SLOT_LEN) window,
                # so no need to also evaluate them for s_idx == slot - 1.
                if s_idx == slot:
                    for m in meteors:
                        local_age = (local_t - s_idx * _SLOT_LEN) - m.start
                        if not (0.0 <= local_age <= m.duration):
                            continue
                        # Wrap the whole event in its own attack/decay
                        # envelope: the spatial gaussian alone isn't enough
                        # -- a meteor whose head starts right next to a
                        # light (or ends there) would otherwise pop from 0
                        # to near-peak L in a single frame at the
                        # local_age boundary above. The harness caught
                        # exactly this (0.80 L/frame on the hex demo)
                        # before this envelope was added.
                        event_env = float(
                            _smoothstep(local_age, 0.0, _METEOR_EVENT_RAMP)
                            * (
                                1.0
                                - _smoothstep(
                                    local_age,
                                    m.duration - _METEOR_EVENT_RAMP,
                                    m.duration,
                                )
                            )
                        )
                        frac = local_age / m.duration
                        head_along = m.run_len * (frac if not m.reverse else 1.0 - frac)
                        d = m.along - head_along
                        near = np.exp(-(d**2) / (2.0 * m.sigma**2))
                        behind = (
                            (head_along - m.along)
                            if not m.reverse
                            else (m.along - head_along)
                        )
                        tail = np.where(
                            behind > 0.0, np.exp(-behind / m.tail) * 0.6, 0.0
                        )
                        w = np.maximum(near, tail) * event_env
                        mck, msk = (
                            np.cos(np.radians(m.hue)),
                            np.sin(np.radians(m.hue)),
                        )
                        np.add.at(w_total, m.rows, w)
                        np.add.at(lw, m.rows, w * _METEOR_L)
                        np.add.at(aw, m.rows, w * _METEOR_C * mck)
                        np.add.at(bw, m.rows, w * _METEOR_C * msk)

        bg_l = bg_l + sky_lift

        blend = np.clip(w_total, 0.0, 1.0)
        w_safe = np.maximum(w_total, 1e-9)
        fig_l = lw / w_safe
        fig_a = aw / w_safe
        fig_b = bw / w_safe

        a_bg = bg_c * np.cos(np.radians(bg_h))
        b_bg = bg_c * np.sin(np.radians(bg_h))
        a = a_bg * (1.0 - blend) + fig_a * blend + extra_a
        b = b_bg * (1.0 - blend) + fig_b * blend + extra_b
        l = bg_l * (1.0 - blend) + fig_l * blend + extra_l

        out = np.zeros((n, 3))
        out[:, 0] = np.clip(l, 0.0, 1.0)
        out[:, 1] = np.clip(np.hypot(a, b), 0.0, 0.4)
        out[:, 2] = np.degrees(np.arctan2(b, a)) % 360.0
        return np.nan_to_num(out)
