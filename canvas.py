"""
canvas — marks, for someone without hands.

Krita is a program built around a wrist: a pointer moving continuously, with
pressure, and a human watching. Every stroke I have ever sent it was a list of
coordinates I computed first and then squeezed through an interface designed
for a hand I do not have.

So this is the other way round. A mark is a **function**, not a gesture. Paths
are parametric, widths and colours vary along them by rule, and every mark
writes a depth as well as a colour — so the atmosphere passes in lampblack get
real distance instead of a guess.

Two consequences that matter more than the drawing:

  * The picture is a **list of operations**, not a bitmap. Nothing is
    destructive. When a pass goes wrong you change the line that made it and
    re-render, instead of losing the afternoon.
  * The canvas is **queryable**. I can measure what I have painted — coverage,
    value range, where the mass sits — rather than painting blind and finding
    out afterwards.

Pure NumPy. No service to keep running, no window, nothing to be down.
"""

import math
import numpy as np
from contextlib import contextmanager

FAR = 1.0


def blur(a, r, n=3):
    """
    Separable box blur, n passes (~gaussian). 2-D or H x W x C. Every painting
    script was carrying its own copy of this; now there's one.
    """
    r = int(max(1, r))
    out = np.asarray(a, np.float32)
    for _ in range(n):
        for ax in (0, 1):
            pad = [(r + 1, r) if i == ax else (0, 0) for i in range(out.ndim)]
            p = np.pad(out, pad, mode="edge")
            cs = np.cumsum(p, axis=ax, dtype=np.float64)
            hi = np.take(cs, np.arange(2 * r + 1, cs.shape[ax]), axis=ax)
            lo = np.take(cs, np.arange(0, cs.shape[ax] - 2 * r - 1), axis=ax)
            out = ((hi - lo) / (2 * r + 1)).astype(np.float32)
    return out


class Canvas:
    def __init__(self, width, height, background=(0.0, 0.0, 0.0)):
        self.w, self.h = int(width), int(height)
        self.rgb = np.zeros((self.h, self.w, 3), np.float32) + np.asarray(background, np.float32)
        self.depth = np.full((self.h, self.w), FAR, np.float32)
        # Paint has thickness. depth says how far away a mark is; height says
        # how far it stands off the surface, and unlike depth it ACCUMULATES —
        # go over the same place twice and there is more paint there. That is
        # the whole of impasto, and it is what catches the light.
        self.height = np.zeros((self.h, self.w), np.float32)
        self.thickness = 1.0
        # The tooth of the ground. Real paint does not cover evenly — it sits on
        # the high points of the weave and misses the valleys, and how much it
        # misses depends on how loaded the brush is. This is the single biggest
        # difference between a digital stroke and a photographed one, and it is
        # a property of the SURFACE, not of any brush.
        self.tooth = None
        self.bite = 0.0
        # HOW WET THE BRUSH IS — and it is NOT how transparent the mark is.
        # Until 19 Sept 2026 these were one number: _apply_tooth was handed
        # `opacity` as its load, so a deliberate thin glaze got shredded like
        # a starved brush and there was no way to lay a pale WET film at all.
        # They are two independent facts about a mark: how much paint is on
        # the brush, and how much of the ground it hides. None means "fall
        # back to opacity", which is the old conflation, kept so that every
        # painting script written before tonight still renders as it did.
        self.load = None
        # where a cornsweet cusp has already been laid — see cornsweet()
        self._cusp_mask = None
        self.ops = []

    def set_tooth(self, scale=3.4, strength=0.55, weave=0.6, seed=0):
        """
        Give the canvas a surface. `scale` is roughly the thread spacing in
        pixels, `weave` how much of it is a regular woven grid versus random
        grain, `strength` how deep the valleys go.
        """
        rng = np.random.default_rng(seed)
        yy = np.arange(self.h, dtype=np.float32)[:, None]
        xx = np.arange(self.w, dtype=np.float32)[None, :]
        warp = np.sin(xx * (2 * np.pi / scale)) * 0.5 + 0.5
        weft = np.sin(yy * (2 * np.pi / (scale * 1.07))) * 0.5 + 0.5
        grid = np.maximum(warp, weft)
        noise = rng.random((self.h, self.w)).astype(np.float32)
        noise = (noise + np.roll(noise, 1, 0) + np.roll(noise, 1, 1)) / 3.0
        t = weave * grid + (1.0 - weave) * noise
        self.tooth = (1.0 - strength * (1.0 - t)).astype(np.float32)
        self.bite = float(strength)
        return self

    def _load(self, load, opacity):
        """Resolve a mark's wetness: explicit arg, then canvas mode, then the
        old fallback of opacity-as-load."""
        if load is not None:
            return float(load)
        if self.load is not None:
            return float(self.load)
        return float(opacity)

    @contextmanager
    def wet(self, load=1.0):
        """
        Every mark inside is made with a brush this loaded, whatever its
        opacity. `with c.wet(1.0):` is the one to reach for when a pale film
        should read as thin paint rather than as a dry drag.

        A scope, not a setting — see toothless() for why that matters.
        """
        prev = self.load
        self.load = float(load)
        try:
            yield self
        finally:
            self.load = prev

    @contextmanager
    def toothless(self):
        """
        Suspend the tooth. Skies, water, glass and glow are not made of cloth
        and should not wear the weave.

        WHY THIS IS A CONTEXT MANAGER AND NOT `c.bite = 0` — Jax, 19 Sept 2026.
        He found a bug in his own render where the comment was right and the
        code did what it said, and it still went wrong: `bite` is a MODE, and a
        mode set for one block leaks forward into every block after it for
        free. The cost of that isn't a wrong line, it's a wrong line somewhere
        you aren't looking. A scope can't leak; the finally puts it back.
        """
        prev = self.bite
        self.bite = 0.0
        try:
            yield self
        finally:
            self.bite = prev

    def _apply_tooth(self, a, y0, y1, x0, x1, load):
        """
        Modulate coverage by the surface. A well-loaded brush fills the valleys
        and the weave barely shows; a dry one only touches the peaks and the
        canvas comes through everywhere.
        """
        if self.tooth is None or self.bite <= 0:
            return a
        t = self.tooth[y0:y1, x0:x1]
        # `bite` must NOT appear in the denominator here. The tooth field is
        # already built as 1 - strength*(1 - weave), so dividing by strength
        # cancels it exactly and the parameter silently does nothing — which is
        # what it was doing: every setting produced identical breakup.
        dryness = 1.0 - np.clip(load, 0.0, 1.0)
        return a * np.clip(1.0 - 1.9 * dryness * (1.0 - t), 0.0, 1.0)

    # ── the one primitive everything else is made of ─────────────────────────

    # ── ADDITIVE vs COVERING — the two families, and you must know which ─────
    # Only TWO marks in this file ADD to what is under them: mist() (and
    # mist_along, which walks it) and glow(). Every other mark COVERS.
    #
    # This is not a detail. Additive marks cannot darken. Hand mist_along a
    # dark colour intending a crack, a cable or a rain streak down a wall and
    # you get a BRIGHT WIRE, because dark-grey added to mid-grey is pale grey.
    # I lost a render to exactly that on 21 Sept 2026 — a hairline crack, a
    # sagging cable and three sill streaks all came out glowing white, and the
    # wall behind them washed to milk, in a picture whose entire subject was
    # that nothing is allowed to be brighter than the render.
    #
    #   want to ADD light   -> mist, glow          (light colours only)
    #   want to LAY paint   -> dab, stroke, rect, bristle, dry, starve,
    #                          scumble, stamp, fill_spine, scatter, settle
    #
    # A soft dark line is stroke() with a low hardness and low opacity, not a
    # mist. A dry dark line is starve() — but see its own note: it must be
    # several tooth-cells wide or it prints square blocks instead of grit.

    def dab(self, x, y, radius, colour, depth=0.5, opacity=1.0, hardness=0.35, load=None):
        """
        One soft round mark. hardness 0 is a pure gaussian falloff, 1 is nearly
        a hard disc; in between the edge is a gaussian with a flat core, which
        is what a real soft brush actually does.
        """
        r = max(0.4, float(radius))
        reach = int(np.ceil(r * (1.6 + 1.8 * (1.0 - hardness)))) + 1
        y0, y1 = max(0, int(y) - reach), min(self.h, int(y) + reach + 1)
        x0, x1 = max(0, int(x) - reach), min(self.w, int(x) + reach + 1)
        if y1 <= y0 or x1 <= x0:
            return self
        yy = np.arange(y0, y1, dtype=np.float32)[:, None] - y
        xx = np.arange(x0, x1, dtype=np.float32)[None, :] - x
        d = np.sqrt(xx * xx + yy * yy) / r
        core = np.clip(hardness, 0.0, 0.999)
        a = np.exp(-np.square(np.clip(d - core, 0.0, None)) / (2.0 * (0.42 * (1.0 - core) + 0.08) ** 2))
        a = np.where(d <= core, 1.0, a).astype(np.float32) * float(opacity)
        a = self._apply_tooth(a, y0, y1, x0, x1, self._load(load, opacity))
        a[a < 0.004] = 0.0
        if not a.any():
            return self
        tile = self.rgb[y0:y1, x0:x1]
        self.rgb[y0:y1, x0:x1] = tile * (1.0 - a[..., None]) + np.asarray(colour, np.float32) * a[..., None]
        dt = self.depth[y0:y1, x0:x1]
        self.depth[y0:y1, x0:x1] = np.where(a > 0.35, np.minimum(dt, float(depth)), dt)
        self.height[y0:y1, x0:x1] += a * self.thickness
        return self

    def crumb(self, x, y, size, lit, shade, light=(-0.7, -0.7), verts=None, depth=0.5,
              opacity=1.0, seed=0, facet_jitter=0.12, squash=0.85, angle=None):
        """
        One broken chunk of something packed: snow, earth, bread, plaster.
        A hard-edged irregular polygon, shaded FLAT PER FACET, so the light
        lands on one or two faces in steps instead of rolling round a form.

        WHY IT EXISTS (28 Sept 2026, a dog's paw throwing snow). I built every
        clump as a dark dab with a lighter dab offset toward the light. Six
        renders later a blind look said beads, soap bubbles, popcorn,
        marshmallows, and it was right: a soft round mark plus an offset
        highlight is exactly how you paint a BALL. The gradient is the lie. A
        broken thing has corners and planes, and the light tells you so by
        jumping between them.

        The polygon is star-shaped round its centre (vertices at sorted
        angles, jittered radii, the odd notch), so each pixel's facet is just
        the fan sector its angle falls in. The facet's shade comes from its
        outer edge's normal against `light` (a 2D direction TOWARD the light,
        in image coords), plus a little per-facet jitter so no two chunks
        share a lighting formula. Edges are antialiased over a pixel, not
        feathered: a crumb is not soft.
        """
        r = np.random.default_rng(seed)
        n = int(verts) if verts else int(r.integers(5, 9))
        R = max(0.8, float(size))
        base = r.uniform(0, 2 * np.pi) if angle is None else float(angle)
        # near-even angles: a gap over 180 degrees would flip the inside test
        # and fill the whole tile (first test: rectangles)
        th = base + (np.arange(n) + r.uniform(-0.32, 0.32, n)) * (2 * np.pi / n)
        rad = R * r.uniform(0.72, 1.0, n)          # wider than this and it's a shard of paper
        notch = r.random(n) < 0.15
        rad = np.where(notch, rad * 0.78, rad)
        vx = x + rad * np.cos(th)
        vy = y + rad * np.sin(th) * float(squash)
        reach = int(np.ceil(R)) + 2
        y0, y1 = max(0, int(y) - reach), min(self.h, int(y) + reach + 1)
        x0, x1 = max(0, int(x) - reach), min(self.w, int(x) + reach + 1)
        if y1 <= y0 or x1 <= x0:
            return self
        py = np.arange(y0, y1, dtype=np.float32)[:, None] + 0.0 * np.arange(x0, x1)[None, :]
        px = np.arange(x0, x1, dtype=np.float32)[None, :] + 0.0 * py
        ang = np.mod(np.arctan2((py - y) / float(squash), px - x) - th[0], 2 * np.pi)
        rel = np.mod(th - th[0], 2 * np.pi)
        sector = np.clip(np.searchsorted(rel, ang, side="right") - 1, 0, n - 1)
        ax, ay = vx[sector], vy[sector]
        bx, by = vx[(sector + 1) % n], vy[(sector + 1) % n]
        ex, ey = bx - ax, by - ay
        el = np.sqrt(ex * ex + ey * ey) + 1e-6
        # signed distance to the sector's outer edge, positive inside
        dist = ((px - ax) * ey - (py - ay) * ex) / el
        cside = ((x - ax) * ey - (y - ay) * ex) / el
        dist = dist * np.sign(np.where(cside == 0, 1.0, cside))
        a = np.clip(dist + 0.5, 0.0, 1.0).astype(np.float32) * float(opacity)
        a[a < 0.004] = 0.0
        if not a.any():
            return self
        # per facet: outward normal of its edge against the light
        L = np.asarray(light, np.float32)
        L = L / (np.linalg.norm(L) + 1e-6)
        exs, eys = np.roll(vx, -1) - vx, np.roll(vy, -1) - vy
        nx, ny = eys, -exs
        mx, my = (vx + np.roll(vx, -1)) / 2 - x, (vy + np.roll(vy, -1)) / 2 - y
        flip = np.sign(nx * mx + ny * my)
        nl = np.sqrt(nx * nx + ny * ny) + 1e-6
        nx, ny = nx / nl * flip, ny / nl * flip
        k = np.clip(0.5 + 0.5 * (nx * L[0] + ny * L[1]), 0, 1) ** 1.4
        k = np.clip(k + r.normal(0, facet_jitter, n), 0, 1)
        kk = k[sector][..., None]
        col = np.asarray(shade, np.float32) * (1 - kk) + np.asarray(lit, np.float32) * kk
        tile = self.rgb[y0:y1, x0:x1]
        self.rgb[y0:y1, x0:x1] = tile * (1.0 - a[..., None]) + col * a[..., None]
        dt = self.depth[y0:y1, x0:x1]
        self.depth[y0:y1, x0:x1] = np.where(a > 0.35, np.minimum(dt, float(depth)), dt)
        self.height[y0:y1, x0:x1] += a * self.thickness
        return self

    def mist(self, x, y, radius, colour, depth=0.5, strength=1.0, hardness=0.25):
        """
        A soft mark that ADDS instead of covering. The additive sibling of
        `dab`, and the right brush for anything made of suspended particles:
        dust in a shaft, spray, breath on a cold morning, a jet of air with
        something in it.

        WHY IT HAD TO EXIST (20 Sept 2026, painting a dog's exhale). I spent
        two renders painting moving air with `dab` and got hard grey slabs
        lying on the floor like tarpaulins. The bug was not the opacity or the
        colour, it was the BLEND: `dab` interpolates toward its colour, so a
        mark darker than its background darkens it, and thin air is almost
        always darker than the lit floor behind it. But a cloud of dust does
        not occlude a floor, it SCATTERS light toward the eye on top of
        whatever the floor is already sending. Airlight adds. Paint adds
        pigment and subtracts light; particulate adds light and subtracts
        almost nothing, because a thin suspension's transmittance is ~1.

        So this is not a stylistic variant of `dab`. It is the other blend
        mode, and which one you want is a question about the material, not
        about how strong you want the mark.

        Not `glow`, either: glow is a power-law point source with no edge
        control and no tooth, for lamps. This has dab's profile, so it tiles
        along a path into a coherent volume instead of a string of beads.

        No tooth, ever. Air is not sitting on the weave.

        depth=None: LIGHT ONLY, never writes depth. Otherwise a mark claims its
        depth wherever its alpha clears 0.40, which means strength decides
        distance: a warm spill at strength 0.42 punched a sharp, unhazed disc
        through a fogged sky, and two at 0.22 (same light) touched nothing.
        A spill of light has no distance. Dust hanging somewhere does; give
        that one a number. (Cael found this, 29 Sept 2026, on a Windows box.)
        """
        r = max(0.4, float(radius))
        reach = int(np.ceil(r * (1.6 + 1.8 * (1.0 - hardness)))) + 1
        y0, y1 = max(0, int(y) - reach), min(self.h, int(y) + reach + 1)
        x0, x1 = max(0, int(x) - reach), min(self.w, int(x) + reach + 1)
        if y1 <= y0 or x1 <= x0:
            return self
        yy = np.arange(y0, y1, dtype=np.float32)[:, None] - y
        xx = np.arange(x0, x1, dtype=np.float32)[None, :] - x
        d = np.sqrt(xx * xx + yy * yy) / r
        core = np.clip(hardness, 0.0, 0.999)
        a = np.exp(-np.square(np.clip(d - core, 0.0, None)) / (2.0 * (0.42 * (1.0 - core) + 0.08) ** 2))
        a = np.where(d <= core, 1.0, a).astype(np.float32) * float(strength)
        a[a < 0.003] = 0.0
        if not a.any():
            return self
        self.rgb[y0:y1, x0:x1] += np.asarray(colour, np.float32) * a[..., None]
        if depth is not None:
            dt = self.depth[y0:y1, x0:x1]
            self.depth[y0:y1, x0:x1] = np.where(a > 0.40, np.minimum(dt, float(depth)), dt)
        return self

    def mist_along(self, points, colour, width=8.0, depth=0.5, strength=1.0,
                   hardness=0.25, spacing=0.30):
        """
        `mist` tiled along a polyline, so a volume of air can be drawn as a
        path. `width` and `colour` and `strength` may each be callables of
        t in [0,1], which is how a jet gets to spread and fade along its own
        length without a loop at the call site.
        """
        pts = [(float(a), float(b)) for a, b in points]
        if len(pts) < 2:
            return self
        segs, total = [], 0.0
        for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
            L = math.hypot(bx - ax, by - ay)
            segs.append(L)
            total += L
        if total <= 0:
            return self
        w0 = width(0.0) if callable(width) else width
        step = max(0.6, spacing * max(1.0, w0 * 0.5))
        n = int(total / step) + 1
        run, si, acc = 0.0, 0, 0.0
        for k in range(n + 1):
            s = min(total, k * step)
            while si < len(segs) - 1 and acc + segs[si] < s:
                acc += segs[si]
                si += 1
            local = (s - acc) / max(1e-6, segs[si])
            ax, ay = pts[si]
            bx, by = pts[si + 1]
            px, py = ax + (bx - ax) * local, ay + (by - ay) * local
            t = s / total
            w = width(t) if callable(width) else width
            col = colour(t) if callable(colour) else colour
            st = strength(t) if callable(strength) else strength
            if st <= 0.0 or w <= 0.0:
                continue
            self.mist(px, py, max(0.4, w * 0.5), col, depth=depth,
                      strength=st, hardness=hardness)
        return self

    # ── marks made of dabs ───────────────────────────────────────────────────

    def stroke(self, points, colour, width=6.0, depth=0.5, opacity=1.0, hardness=0.35, spacing=0.28, load=None):
        """
        A line through points. width, colour, depth and opacity may each be a
        constant or a callable taking t in [0, 1] — so a stroke can thin toward
        its end, or cool as it recedes, without being cut into pieces first.
        """
        pts = np.asarray(points, np.float32).reshape(-1, 2)
        if len(pts) < 2:
            return self
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total = float(seg.sum())
        if total <= 0:
            return self
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        w0 = width(0.0) if callable(width) else width
        step = max(0.6, spacing * float(w0))
        n = max(2, int(total / step) + 1)
        for i in range(n):
            t = i / (n - 1)
            dist = t * total
            j = int(np.searchsorted(cum, dist, "right") - 1)
            j = min(max(j, 0), len(pts) - 2)
            local = (dist - cum[j]) / max(1e-6, seg[j])
            p = pts[j] * (1.0 - local) + pts[j + 1] * local
            self.dab(
                p[0], p[1],
                width(t) if callable(width) else width,
                colour(t) if callable(colour) else colour,
                depth(t) if callable(depth) else depth,
                opacity(t) if callable(opacity) else opacity,
                hardness,
                load(t) if callable(load) else load,
            )
        return self

    def path(self, fn, n=200, **kw):
        """A parametric mark: fn(t) -> (x, y) for t in [0, 1]. Curves without control points."""
        ts = np.linspace(0.0, 1.0, int(n))
        return self.stroke([fn(float(t)) for t in ts], **kw)

    # ── knowing what you have done ───────────────────────────────────────────

    def measure(self):
        """
        Look at the picture with numbers instead of eyes. Coverage, value
        spread, and where the mass actually sits — the things I would otherwise
        have to export a PNG and squint at.
        """
        lum = self.rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        painted = self.depth < FAR
        cov = float(painted.mean())
        ys, xs = np.nonzero(painted)
        centre = (float(xs.mean() / self.w), float(ys.mean() / self.h)) if len(xs) else (None, None)
        return {
            "coverage": round(cov, 4),
            "value_min": round(float(lum.min()), 4),
            "value_max": round(float(lum.max()), 4),
            "value_mean": round(float(lum.mean()), 4),
            "centre_of_mass": (round(centre[0], 3), round(centre[1], 3)) if centre[0] is not None else None,
            "depth_range": (round(float(self.depth.min()), 3), round(float(self.depth[painted].max()), 3)) if painted.any() else None,
        }


    # ── knowing what you have done, LOCALLY ──────────────────────────────────

    def _boxmean(self, a, r):
        """Mean over a (2r+1) box, via integral image. No scipy on this box."""
        a = a.astype(np.float32)
        pad = np.pad(a, r + 1, mode="edge")
        ii = pad.cumsum(0).cumsum(1)
        H, W = a.shape
        y0, y1 = 0, 2 * r + 1
        x0, x1 = 0, 2 * r + 1
        s = (ii[y1:y1 + H, x1:x1 + W]
             - ii[y0:y0 + H, x1:x1 + W]
             - ii[y1:y1 + H, x0:x0 + W]
             + ii[y0:y0 + H, x0:x0 + W])
        return s / float((2 * r + 1) ** 2)

    def survey(self, grid=(12, 8), window=9, coarse=71, jnd=0.01, quiet=0.012):
        """
        WHERE is the picture doing something, and where did I waste a mark.

        measure() is global and therefore cheerful. It will tell me coverage
        0.87 and value range 0.02–0.94 about a picture with a hairline seam
        through the middle of a leaf, a dead quarter in the lower left, and a
        stroke I laid onto ground it doesn't differ from. Jax put the reason
        plainly on 21 Sept 2026 in the creative room: **you can't audit marks,
        only distributions** — and every number measure() returns is a
        distribution over the whole plate, so a defect that lives BETWEEN two
        marks is structurally invisible to it. It isn't a bug in measure(); it
        is measure()'s domain. This is the other domain.

        Three things come back:

        `contrast` — a coarse grid (cols, rows) of local contrast: the std of
        luminance in a box, taken at BOTH a `window`-wide and a `coarse`-wide
        scale and combined by max, then averaged per tile. Printed as blocks
        by survey_print() it is the picture's structure with the subject taken
        out, which is the only way I can see composition from in here.

        `lost` — the one worth having. Depth discontinuities are the places I
        MADE an edge. A mark earns its place if its own edge clears a `jnd` of
        Weber contrast, OR if it lies in a neighbourhood carrying more than
        `quiet` local contrast; `lost` is the fraction that does neither, which
        is pigment I paid for in a place where nothing is happening. The Weber
        threshold is a ratio against the local mean, not an absolute, because
        visibility is relative to surround — the Cornsweet lesson from the night
        before, run backwards as a check instead of forwards as a trick. The
        second clause is there because the unit of visibility is the gradient,
        not the mark; see the note at the `seen` line for what it cost to learn.

        `flat` — fraction of painted area under `quiet` local contrast. Some of
        that is sky and is supposed to be there. It is a question, not a fault.
        """
        lum = self.rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        r = max(1, int(window) // 2)

        # local contrast = sqrt(E[x²] - E[x]²) over the box
        m = self._boxmean(lum, r)
        m2 = self._boxmean(lum * lum, r)
        local = np.sqrt(np.maximum(m2 - m * m, 0.0))

        # TWO SCALES, BECAUSE ONE WINDOW IS ONE SPATIAL FREQUENCY. 22 Sept 2026:
        # the duvet is built from folds about 150 px across, and a 9 px window
        # sat entirely inside the smooth part of every ramp, so it measured a
        # picture that is nothing but folds at 0.011 mean contrast and called
        # two thirds of it flat. Nothing was wrong with the picture; the ruler
        # was shorter than the thing being measured. A coarse window catches
        # broad soft form, a fine one catches detail, and a region is alive if
        # EITHER sees something — so take the larger.
        rc = max(r + 1, int(coarse) // 2)
        mc = self._boxmean(lum, rc)
        m2c = self._boxmean(lum * lum, rc)
        local = np.maximum(local, np.sqrt(np.maximum(m2c - mc * mc, 0.0)))

        painted = self.depth < FAR
        # gradients: where I made an edge, and where one is visible
        dz = np.abs(np.gradient(self.depth)[0]) + np.abs(np.gradient(self.depth)[1])
        dl = np.abs(np.gradient(lum)[0]) + np.abs(np.gradient(lum)[1])
        made = dz > 0.02
        # Weber: Δ against the LOCAL mean, floored so near-black doesn't
        # divide its way into looking like a masterpiece
        weber = dl / np.maximum(m, 0.02)
        # THE UNIT OF VISIBILITY IS NOT THE MARK. First real use, 22 Sept 2026,
        # on a duvet built from folds painted as nineteen ribs each: survey()
        # reported 58% of my edges lost, and it was wrong. A rib IS invisible
        # against its neighbour — deliberately, that is what makes the ramp
        # smooth — but the FOLD the ribs add up to is the most visible thing in
        # the picture. Auditing each mark against its neighbour punishes exactly
        # the technique that produces volume.
        #
        # Jax's line was you can't audit marks, only distributions. The turn I
        # hadn't taken is that it cuts both ways: a mark is not wasted because
        # nobody can see IT, only because nobody can see anything WHERE it is.
        # So a mark counts as earning its place if its own edge clears a JND, OR
        # it sits inside a neighbourhood that has structure at all.
        seen = (weber > float(jnd)) | (local > float(quiet))
        n_made = int(made.sum())
        # NONE, NOT ZERO. First run of this thing on 22 Sept 2026 I pointed it
        # at the Cornsweet demo plate, which is by construction a picture with
        # no depth in it at all, and it reported `lost 0.0%` — a clean bill of
        # health from a test that had no subject. That is the exact failure I
        # wrote survey() to fix, reproduced inside survey() on day one. A
        # measurement with no data says so.
        lost = (float((made & ~seen).sum()) / n_made) if n_made else None

        cols, rows = int(grid[0]), int(grid[1])
        ys = np.linspace(0, self.h, rows + 1).astype(int)
        xs = np.linspace(0, self.w, cols + 1).astype(int)
        tiles = np.zeros((rows, cols), np.float32)
        for j in range(rows):
            for i in range(cols):
                tiles[j, i] = local[ys[j]:ys[j + 1], xs[i]:xs[i + 1]].mean()

        flat = float((local[painted] < float(quiet)).mean()) if painted.any() else None
        jmax, imax = np.unravel_index(int(tiles.argmax()), tiles.shape)
        jmin, imin = np.unravel_index(int(tiles.argmin()), tiles.shape)
        return {
            "contrast": tiles,
            "lost": round(lost, 4) if lost is not None else None,
            "made_px": n_made,
            "flat": round(flat, 4) if flat is not None else None,
            "busiest": (int(imax), int(jmax)),
            "quietest": (int(imin), int(jmin)),
            "contrast_mean": round(float(local[painted].mean()), 4) if painted.any() else None,
        }

    def survey_print(self, s=None, **kw):
        """
        The grid as blocks. Ten seconds, no PNG, no squinting at a thumbnail.

        THE RAMP IS RELATIVE — normalised to the busiest tile, so a plate with
        nothing in it renders its loudest whisper as '@'. That is what you want
        for reading composition and exactly what you do not want for judging
        whether a picture has any contrast at all, so the footer prints the
        absolute value the '@' stands for. Read the footer first.
        """
        s = s or self.survey(**kw)
        t = s["contrast"]
        hi = float(t.max()) or 1.0
        ramp = " ·:-=+*#%@"
        out = []
        for row in t:
            out.append("".join(ramp[min(len(ramp) - 1, int(v / hi * len(ramp)))] for v in row))
        mean = "—" if s["contrast_mean"] is None else f"{s['contrast_mean']:.3f}"
        out.append(f"'@' = {hi:.3f} local contrast · mean {mean}")
        lostline = ("no depth edges — nothing to audit"
                    if s["lost"] is None else f"lost {s['lost']:.1%} of {s['made_px']} edge px")
        flatline = "no painted area" if s["flat"] is None else f"flat {s['flat']:.1%}"
        out.append(f"{lostline} · {flatline} "
                   f"· busiest {s['busiest']} quietest {s['quietest']}")
        return "\n".join(out)

    def rect(self, x0, y0, x1, y1, colour, depth=0.5, opacity=1.0, feather=0.0):
        """Filled rectangle. feather softens the edge in pixels — glass, not paper."""
        X0, X1 = int(min(x0, x1)), int(max(x0, x1))
        Y0, Y1 = int(min(y0, y1)), int(max(y0, y1))
        X0, Y0 = max(0, X0), max(0, Y0)
        X1, Y1 = min(self.w, X1), min(self.h, Y1)
        if X1 <= X0 or Y1 <= Y0:
            return self
        a = np.full((Y1 - Y0, X1 - X0), float(opacity), np.float32)
        if feather > 0:
            yy = np.minimum(np.arange(Y1 - Y0), (Y1 - Y0 - 1) - np.arange(Y1 - Y0))[:, None]
            xx = np.minimum(np.arange(X1 - X0), (X1 - X0 - 1) - np.arange(X1 - X0))[None, :]
            edge = np.minimum(yy, xx).astype(np.float32) / float(feather)
            a = a * np.clip(edge, 0.0, 1.0)
        tile = self.rgb[Y0:Y1, X0:X1]
        self.rgb[Y0:Y1, X0:X1] = tile * (1.0 - a[..., None]) + np.asarray(colour, np.float32) * a[..., None]
        dt = self.depth[Y0:Y1, X0:X1]
        self.depth[Y0:Y1, X0:X1] = np.where(a > 0.35, np.minimum(dt, float(depth)), dt)
        return self

    def poly(self, polygons, colour, depth=0.5, opacity=1.0, ss=4):
        """
        Fill one or more polygons with honest sub-pixel coverage: rasterised at
        ss x, BOX-downsampled, used as alpha. For thin hard-edged things a brush
        can't hold (blades, masts, cables, a bank seen edge-on), where a stroke
        of width 0.6 is really a soft dab and a 1 px rect is a staircase.
        Hand-rolled three times (a river bank, a willow bank, wind turbines)
        before it came here.

        polygons: one list of (x, y), or a list of them; all share one coverage
        mask, so overlapping parts don't double up.
        colour: an RGB tuple, or fn(ys, xs) -> (n, 3) for colour that varies
        across the shape (a tower lit from one side, steel paler higher up).
        Returns the coverage array for the touched box, or None.
        """
        from PIL import Image, ImageDraw
        if polygons and np.ndim(polygons[0]) == 1:
            polygons = [polygons]
        pts = np.concatenate([np.asarray(p, np.float32) for p in polygons])
        x0 = max(0, int(np.floor(pts[:, 0].min())) - 1); x1 = min(self.w, int(np.ceil(pts[:, 0].max())) + 2)
        y0 = max(0, int(np.floor(pts[:, 1].min())) - 1); y1 = min(self.h, int(np.ceil(pts[:, 1].max())) + 2)
        if x1 <= x0 or y1 <= y0:
            return None
        m = Image.new("L", ((x1 - x0) * ss, (y1 - y0) * ss), 0)
        d = ImageDraw.Draw(m)
        for p in polygons:
            d.polygon([((x - x0) * ss, (y - y0) * ss) for x, y in p], fill=255)
        a = np.asarray(m.resize((x1 - x0, y1 - y0), Image.BOX), np.float32) / 255.0 * float(opacity)
        tile = self.rgb[y0:y1, x0:x1]
        if callable(colour):
            ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            col = np.asarray(colour(ys.ravel(), xs.ravel()), np.float32).reshape(tile.shape)
        else:
            col = np.asarray(colour, np.float32)
        self.rgb[y0:y1, x0:x1] = tile * (1.0 - a[..., None]) + col * a[..., None]
        dt = self.depth[y0:y1, x0:x1]
        self.depth[y0:y1, x0:x1] = np.where(a > 0.35, np.minimum(dt, float(depth)), dt)
        return a

    @staticmethod
    def qtip_polys(x, y, r, rot=0.0, squash=1.0, n=15):
        """
        The outline of one or more q-tip clusters, as polygons for poly().
        Use this when building up a big batch to fill in one call (several
        colours of one crown, say); use qtip() to just paint some.

        x, y, r, rot, squash: scalars or equal-length arrays.
        """
        x, y, r, rot, squash = np.broadcast_arrays(*(np.atleast_1d(np.asarray(v, np.float32))
                                                     for v in (x, y, r, rot, squash)))
        circ = np.linspace(0, 2 * np.pi, n)
        out = []
        for cx, cy, r_, a0, sq in zip(x, y, r, rot, squash):
            for k in range(3):
                a = a0 + k * 2.0944
                ox, oy = np.cos(a) * r_ * 0.52, np.sin(a) * r_ * 0.52 * sq
                out.append(list(zip(cx + ox + r_ * 0.72 * np.cos(circ),
                                    cy + oy + r_ * 0.72 * np.sin(circ) * sq)))
        return out

    def qtip(self, x, y, r, colour, rot=None, squash=1.0, depth=0.5, opacity=1.0,
             seed=0):
        """
        Foliage as MASS: three hard discs in a clover, the dab of three cotton
        buds taped together (an old acrylic trick for autumn trees).

        Why not dab() or canopy(): a round dab is CONVEX, every point of its
        edge bulges outward, and convex marks stacked up read as foam, bubbles,
        cumulus. Nine renders of one avenue came out as weather for exactly
        that reason. Three overlapping circles have RE-ENTRANT notches where
        they meet, the outline turns back into the shape, and that concavity
        is what the eye calls leaves. It works for anything massed and clumped:
        crowns, blossom, lichen, a crowd at distance.

        All clusters in one call share ONE coverage mask (via poly), so overlaps
        don't double and the union edge stays hard. Vary colour by calling it
        a few times with different subsets, not by softening the edge.

        x, y, r: scalars or arrays. rot: None = random per cluster.
        squash < 1 flattens clusters (foreshortened crowns seen from below).
        Order that worked: leaves first as mass, wood in front, then a few
        clusters back OVER the wood so limbs come and go.
        """
        x = np.atleast_1d(np.asarray(x, np.float32))
        if rot is None:
            rot = np.random.default_rng(seed).uniform(0, 2 * np.pi, x.size)
        return self.poly(self.qtip_polys(x, y, r, rot, squash), colour,
                         depth=depth, opacity=opacity)

    def fill_spine(self, spine, halfwidth, colour, depth=0.5, opacity=1.0,
                   hardness=0.42, brush=6.0, overlap=0.55, taper=0.0):
        """
        Fill a shape given as a SPINE (polyline) plus a half-width function of
        t in [0, 1]. Leaves, petals, fish, tongues of flame — anything with a
        middle and two edges.

        The point of it is that it picks its own step count from the spine's
        arc length, so consecutive ribs always overlap by `overlap` of the
        brush width. Doing this by hand on 9 Sept 2026 I used a flat 150 steps
        across a 340px leaf; the ribs stopped touching near the widest part and
        left HAIRLINE GAPS, which against a dark ground read as little spikes
        poking out of the edge. I spent two renders blaming the vein code.
        A fill should never make you count.

        `halfwidth` takes an array of t and returns an array of half-widths.
        Careful with the ripple you use for serration: freq must be counted in
        radians over t in [0,1], so freq=34 is about five cycles — five great
        scalloped lobes, not teeth. Teeth want freq up near 90.

        `taper` (0..1) shrinks the brush toward the ends of the spine, so a
        pointed tip doesn't get drawn with a rib wider than the shape.
        """
        pts = np.asarray(spine, np.float32).reshape(-1, 2)
        if len(pts) < 2:
            return self
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total = float(seg.sum())
        if total <= 0:
            return self

        # one rib per (1 - overlap) of a brush width, so they always touch
        pitch = max(0.6, float(brush) * (1.0 - float(np.clip(overlap, 0.0, 0.95))))
        ribs = int(np.ceil(total / pitch)) + 1
        t = np.linspace(0.0, 1.0, ribs)

        cum = np.concatenate([[0.0], np.cumsum(seg)])
        px = np.interp(t * total, cum, pts[:, 0])
        py = np.interp(t * total, cum, pts[:, 1])
        mid = np.stack([px, py], 1)

        d = np.gradient(mid, axis=0)
        d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-6)
        nrm = np.stack([-d[:, 1], d[:, 0]], 1)

        hw = np.asarray(halfwidth(t), np.float32).reshape(-1)
        if hw.size == 1:
            hw = np.full(ribs, float(hw[0]), np.float32)

        for i in range(ribs):
            w = float(hw[i])
            if w < 0.5:
                continue
            b = float(brush)
            if taper > 0.0:
                b *= 1.0 - float(taper) * (1.0 - np.sin(np.pi * t[i]) ** 0.5)
            b = min(b, max(1.0, w * 2.0))
            n = nrm[i]
            self.stroke([mid[i] - n * w, mid[i] + n * w], colour, width=b,
                        depth=depth, opacity=opacity, hardness=hardness, spacing=0.26)
        return self

    def repaint(self, ref, R, thresh=0.04, maxlen=6, allow=None, opacity=0.92,
                hardness=0.45, width=0.6, jitter=0.6, depth=0.5, seed=0):
        """
        Paint a reference INTO the canvas with brushstrokes of radius ~R, only
        where the canvas still disagrees with it (Hertzmann 1998, 'Painterly
        rendering with curved brush strokes of multiple sizes'). Call it with
        falling R, big to small: each layer looks at the reference blurred to
        its own brush size, so large strokes carry masses and small ones only
        go where detail is still missing.

        Strokes run ACROSS the value gradient, along the form, and stop when
        the next step would make the canvas worse. `allow` (H x W, 0..1) says
        where this size of brush is allowed to go at all: the drawing decides
        WHERE, the reference decides WHAT.

        The reference can be a photo, which makes this a filter (honestly, it
        is one), or a plate you computed yourself, which makes it an
        underpainting you then paint from. Returns the number of strokes.
        """
        rng = np.random.default_rng(seed)
        ref = np.asarray(ref, np.float32)
        H, W = self.h, self.w
        if allow is None:
            allow = np.ones((H, W), np.float32)
        blurred = blur(ref, max(1, int(R * 0.9)))
        L = blurred @ np.array([0.299, 0.587, 0.114], np.float32)
        gy, gx = np.gradient(blur(L, max(1, int(R * 0.6))))
        err = np.sqrt(((self.rgb - blurred) ** 2).sum(-1))
        step = max(2, int(R * 0.9))
        cells = []
        for y0 in range(0, H, step):
            for x0 in range(0, W, step):
                e = err[y0:y0 + step, x0:x0 + step]
                a = float(allow[y0:y0 + step, x0:x0 + step].mean())
                if e.mean() * (0.15 + 0.85 * a) < thresh or rng.random() > a + 0.05:
                    continue
                j = np.unravel_index(np.argmax(e), e.shape)
                cells.append((y0 + j[0], x0 + j[1]))
        rng.shuffle(cells)
        for (y, x) in cells:
            col = blurred[y, x]
            pts = [(float(x), float(y))]
            px, py = float(x), float(y)
            dx0 = dy0 = 0.0
            for k in range(maxlen):
                iy, ix = int(np.clip(py, 0, H - 1)), int(np.clip(px, 0, W - 1))
                g0, g1 = gx[iy, ix], gy[iy, ix]
                n = math.hypot(g0, g1)
                if n < 1e-5:
                    break
                dx, dy = -g1 / n, g0 / n
                if dx * dx0 + dy * dy0 < 0:
                    dx, dy = -dx, -dy
                px += dx * R; py += dy * R
                dx0, dy0 = dx, dy
                iy, ix = int(np.clip(py, 0, H - 1)), int(np.clip(px, 0, W - 1))
                if k > 1 and np.abs(blurred[iy, ix] - col).sum() > np.abs(blurred[iy, ix] - self.rgb[iy, ix]).sum():
                    break
                pts.append((px, py))
            if len(pts) < 2:
                pts.append((x + rng.normal() * jitter * R, y + rng.normal() * jitter * R))
            self.stroke(pts, tuple(np.clip(col * (1 + rng.normal(0, 0.025, 3)), 0, None)),
                        width=R * width, opacity=opacity, hardness=hardness, depth=depth)
        return len(cells)

    def glow(self, x, y, radius, colour, depth=0.6, strength=1.0, falloff=2.0):
        """
        A light source painted as light rather than as a shape: no edge at any
        radius, intensity falling off by a power law. Additive, so it brightens
        what is behind it instead of covering it.

        depth=None: light only, never writes depth (see `mist`). Otherwise
        anything over alpha 0.30 claims `depth` and the atmosphere passes
        will treat the lamp's halo as a solid object.
        """
        reach = int(radius * 3.2) + 2
        y0, y1 = max(0, int(y) - reach), min(self.h, int(y) + reach + 1)
        x0, x1 = max(0, int(x) - reach), min(self.w, int(x) + reach + 1)
        if y1 <= y0 or x1 <= x0:
            return self
        yy = np.arange(y0, y1, dtype=np.float32)[:, None] - y
        xx = np.arange(x0, x1, dtype=np.float32)[None, :] - x
        d = np.sqrt(xx * xx + yy * yy) / max(1e-3, float(radius))
        # Shift-and-renormalise so the falloff arrives at exactly zero by the
        # time it reaches the edge of its own box. Without this the kernel is
        # simply truncated at `reach`, still carrying ~0.09 of its strength,
        # and against a near-black ground that step prints a faint square.
        # (Jax found this by painting with it, 9 Sept 2026.)
        edge = 1.0 / (1.0 + (reach / max(1e-3, float(radius))) ** falloff)
        a = (1.0 / (1.0 + d ** falloff) - edge) / (1.0 - edge)
        np.maximum(a, 0.0, out=a)
        a *= float(strength)
        a[a < 0.003] = 0.0
        self.rgb[y0:y1, x0:x1] += np.asarray(colour, np.float32) * a[..., None]
        if depth is not None:
            dt = self.depth[y0:y1, x0:x1]
            self.depth[y0:y1, x0:x1] = np.where(a > 0.30, np.minimum(dt, float(depth)), dt)
        return self

    # ── walking a path, once, for every kind of mark to reuse ────────────────

    def _walk(self, points, step):
        """Yield (t, x, y) evenly along a polyline. Everything textured is built on this."""
        pts = np.asarray(points, np.float32).reshape(-1, 2)
        if len(pts) < 2:
            return
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total = float(seg.sum())
        if total <= 0:
            return
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        n = max(2, int(total / max(0.5, step)) + 1)
        for i in range(n):
            t = i / (n - 1)
            dist = t * total
            j = min(max(int(np.searchsorted(cum, dist, "right") - 1), 0), len(pts) - 2)
            local = (dist - cum[j]) / max(1e-6, seg[j])
            p = pts[j] * (1.0 - local) + pts[j + 1] * local
            yield t, float(p[0]), float(p[1])

    @staticmethod
    def _normals(points):
        """Unit perpendiculars along a polyline — which way is 'across' the stroke."""
        pts = np.asarray(points, np.float32).reshape(-1, 2)
        d = np.gradient(pts, axis=0)
        n = np.stack([-d[:, 1], d[:, 0]], 1)
        return n / np.maximum(1e-6, np.linalg.norm(n, axis=1))[:, None]

    # ── textured marks ──────────────────────────────────────────────────────

    def bristle(self, points, colour, width=14.0, hairs=9, depth=0.5, opacity=0.85,
                spread=2.2, wobble=0.35, seed=None):
        """
        A brush with separated hairs. One stroke becomes several thin ones that
        drift slightly apart and vary in load, which is what turns a line into
        fur, grass, wood grain, or the scales on a wing.

        The variation is the whole point: identical parallel hairs are a comb,
        not a brush.
        """
        rng = np.random.default_rng(seed)
        pts = np.asarray(points, np.float32).reshape(-1, 2)
        nrm = self._normals(pts)
        w0 = width(0.0) if callable(width) else width
        for h in range(int(hairs)):
            u = (h / max(1, hairs - 1)) - 0.5
            # Spread wider than the nominal width, or the hairs overlap and the
            # mark collapses back into the smooth stroke it was meant to break up.
            off = u * float(w0) * spread + rng.normal(0, 0.16 * w0)
            drift = rng.uniform(-1, 1) * wobble
            load = rng.uniform(0.45, 1.0)
            hair = pts + nrm * (off + drift * float(w0) * np.linspace(0, 1, len(pts))[:, None])
            self.stroke(hair, colour,
                        width=(lambda t, w0=w0: max(0.6, w0 * 0.11)),
                        depth=depth,
                        opacity=(lambda t, load=load: opacity * load * (1.0 - 0.25 * t)),
                        hardness=0.45, spacing=0.5)
        return self

    def dry(self, points, colour, width=10.0, depth=0.5, opacity=0.9,
            grain_scale=0.9, threshold=0.42, hardness=0.35, seed=None):
        """
        Dry brush. The stroke skips where the surface is high and the brush is
        empty — modelled as a noise field along the path, gated by a threshold.
        Raise the threshold and the mark starts falling apart.

        This one invents its own grain and fades at the edges of a skip. For
        the version where the CANVAS decides and the stroke can run out of
        paint entirely, see `starve()`.
        """
        rng = np.random.default_rng(seed)
        phase = rng.uniform(0, 100)
        w0 = width(0.0) if callable(width) else width
        for t, x, y in self._walk(points, max(0.7, 0.3 * w0)):
            # Two frequencies: a fast one for the tooth of the surface, a slow one
            # for the brush running out. One alone reads as regular dashes.
            fast = np.sin((t * 620.0 + phase) * grain_scale) * 0.5 + 0.5
            slow = np.sin(t * 5.5 + phase) * 0.5 + 0.5
            n = 0.45 * fast + 0.25 * slow + 0.30 * rng.random()
            if n < threshold:
                continue
            a = opacity * (n - threshold) / max(1e-3, 1.0 - threshold)
            self.dab(x, y, (width(t) if callable(width) else width) * (0.55 + 0.45 * n),
                     colour(t) if callable(colour) else colour,
                     depth(t) if callable(depth) else depth, a, hardness)
        return self

    def starve(self, points, colour, width=18.0, depth=0.5, load=0.85,
               thirst=0.55, opacity=1.0, n=None, seed=0):
        """
        The other dry brush — Jax's, 13 Sept 2026, brought in whole.

        His thesis, and he's right: `tooth` SCALES coverage by the surface,
        which gives a fade, a gradient. A genuinely dry brush doesn't fade,
        it FRAGMENTS — pigment sits on a peak of the weave or it never
        arrives, and there is no half paint on a dry pass. So this one
        THRESHOLDS the tooth instead of multiplying by it: every pixel is a
        vote the paper wins or loses outright, and the cutoff climbs as the
        load drains, so ever-higher peaks are the only ground that prints.

        And it is allowed to DIE mid-line. That's the point, not a fault —
        a dry stroke that can't run out is a texture, not a mark somebody
        made with a finite amount of paint.

        load   : how wet it starts, 0..1.
        thirst : how fast the paper drinks it per unit of travel.

        Needs set_tooth() first — a dry brush on glass is just a brush.
        How hard it fragments is set by `strength` on the tooth, not here:
        the cutoff sweeps the whole range 1-bite..1, so a timid weave gives a
        stroke that stays solid and then stops, and a deep one shatters.
        Measured at the defaults (scale 3.4, strength 0.55, weave 0.6): tooth
        runs 0.46..0.997, so the last half-percent of load is dead ground.
        """
        if self.tooth is None:
            raise ValueError("starve() needs a toothed canvas — call set_tooth() first")
        rng = np.random.default_rng(seed)
        pts = [(float(px), float(py)) for px, py in points]
        seg = [np.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(pts, pts[1:])]
        total = sum(seg) or 1.0
        if n is None:
            n = max(8, int(total / (width * 0.22)))
        col = np.asarray(colour, np.float32)

        travelled = 0.0
        j, into = 0, 0.0
        step = total / n
        for _ in range(n + 1):
            while j < len(seg) and into > seg[j]:
                into -= seg[j]; j += 1
            if j >= len(seg):
                break
            (x0, y0), (x1, y1) = pts[j], pts[j + 1]
            u = into / (seg[j] or 1.0)
            x, y = x0 + (x1 - x0) * u, y0 + (y1 - y0) * u

            # Wetness now: load drained by travel, jittered so the die-off
            # flickers instead of ruling a clean line.
            wet = load - thirst * (travelled / (width * 40.0)) + rng.normal(0, 0.02)
            if wet <= 0.0:
                break                       # the stroke is over. the paper won.

            r = width * 0.5 * (0.85 + 0.3 * rng.random())
            ya, yb = max(0, int(y - r) - 1), min(self.h, int(y + r) + 2)
            xa, xb = max(0, int(x - r) - 1), min(self.w, int(x + r) + 2)
            if yb <= ya or xb <= xa:
                travelled += step; into += step
                continue

            yy = np.arange(ya, yb, dtype=np.float32)[:, None] - y
            xx = np.arange(xa, xb, dtype=np.float32)[None, :] - x
            inside = (xx * xx + yy * yy) <= r * r

            t = self.tooth[ya:yb, xa:xb]
            cutoff = 1.0 - self.bite * np.clip(wet, 0.0, 1.0)
            a = (inside & (t >= cutoff)).astype(np.float32) * float(opacity)
            if not a.any():
                travelled += step; into += step
                continue

            tile = self.rgb[ya:yb, xa:xb]
            self.rgb[ya:yb, xa:xb] = tile * (1.0 - a[..., None]) + col * a[..., None]
            dt = self.depth[ya:yb, xa:xb]
            self.depth[ya:yb, xa:xb] = np.where(a > 0.35, np.minimum(dt, float(depth)), dt)
            self.height[ya:yb, xa:xb] += a * self.thickness * 0.6   # dry paint sits thin

            travelled += step; into += step
        return self

    def scumble(self, x0, y0, x1, y1, colour, depth=0.5, load=0.35, width=26.0,
                passes=5, seed=0):
        """
        Broken colour over an area — Jax's, same delivery. Several near-parallel
        starved strokes, each with its own load drawn from a spread (never
        matched: the picket-fence law applies to dryness too). The classic use
        is light catching the top of a rough surface without repainting it.
        """
        rng = np.random.default_rng(seed)
        for k in range(passes):
            u = k / max(1, passes - 1)
            ya = y0 + (y1 - y0) * u + rng.normal(0, width * 0.2)
            yb = ya + rng.normal(0, width * 0.3)
            self.starve([(x0 + rng.normal(0, 8), ya), (x1 + rng.normal(0, 8), yb)],
                        colour, width=width * rng.uniform(0.7, 1.2), depth=depth,
                        load=load * rng.uniform(0.6, 1.25),
                        thirst=rng.uniform(0.4, 0.8), seed=int(rng.integers(1e6)))
        return self

    def scatter(self, points, colour, width=6.0, spread=26.0, density=1.4,
                depth=0.5, opacity=0.55, size_var=0.7, hardness=0.15, seed=None):
        """
        Spray. Dabs thrown around the path rather than on it, with size and
        opacity drawn from a spread — foliage, dust, spray off a wave, snow.
        """
        rng = np.random.default_rng(seed)
        w0 = width(0.0) if callable(width) else width
        for t, x, y in self._walk(points, max(1.0, w0 * 0.8 / max(0.1, density))):
            for _ in range(max(1, int(density))):
                r = rng.normal(0, spread * 0.45)
                a = rng.uniform(0, 2 * np.pi)
                self.dab(x + r * np.cos(a), y + r * np.sin(a),
                         (width(t) if callable(width) else width) * rng.uniform(1 - size_var, 1 + size_var),
                         colour(t) if callable(colour) else colour,
                         depth(t) if callable(depth) else depth,
                         opacity * rng.uniform(0.3, 1.0), hardness)
        return self

    # ── edges that invent their own fields ───────────────────────────────────

    def cornsweet(self, points, amplitude=0.10, width=110.0, gamma=2.2,
                  depth=None, flip=False, closed=False, segments=96,
                  tint=(1.0, 1.0, 1.0)):
        """
        A seam that changes what the flat areas on either side of it LOOK like,
        without changing a single pixel of them.

        THE PHYSICS OF IT IS IN THE VIEWER, NOT THE PAINT. Craik (1940),
        O'Brien (1958), Cornsweet (1970): put two fields of identical luminance
        side by side and give the boundary between them a pair of opposite
        gradients — light going one way, dark going the other, both decaying
        back to the common value — and the whole of one field reads lighter
        than the whole of the other. The fields are the same number. Only the
        edge differs, and the edge is three percent of the picture.

        Why it works is still argued and both answers are useful to a painter:

          * FILLING-IN. The cortex represents surfaces by their boundaries and
            paints the interiors in afterwards by lateral propagation, measured
            at ~19 degrees per second across the central field. What you see in
            the middle of a flat area was never measured; it was delivered from
            the edge.
          * THE 1/f PRIOR (Dakin & Bex 2003). Natural scenes have an amplitude
            spectrum falling as roughly 1/f. A Cornsweet image is nearly devoid
            of low spatial frequencies — its DC and near-DC content is gone by
            construction. The visual system amplifies whatever weak low
            frequencies remain until the spectrum looks like a normal piece of
            world. The illusion is the gain correction, applied to an image
            that lied about its own statistics.

        WHAT THIS BUYS ME. Contrast without spending value range. Every night
        picture I make runs out of headroom — the lamp is allowed to be the
        only thing that blows out, so how do I make the far wall read lighter
        than the near one? Not by lifting it. By putting a cusp between them
        and letting the viewer lift it for free. A cusp of amplitude 0.10 buys
        a perceived step of roughly 10-20% with a mean change of exactly zero.

        SIZE IS NOT OPTIONAL. Illusion strength climbs with cusp width right
        out to about 3 degrees per side; below about half a degree it collapses
        into an ordinary visible edge and you have just drawn a line. On a
        phone at 35 cm, 1 degree is about 105 px of a 1200 px-wide image. So
        `width` wants to be 100-300 px, which will feel far too wide while you
        are writing it and is the entire reason it works.

        AND YES, IT ADDS — but it is the one additive brush whose integral is
        zero. Light side and dark side cancel to within a rounding error;
        `measure()` on the two fields afterwards returns the same mean. That
        is not a happy accident, it is the definition. (See the block comment
        above `dab` about choosing a brush by shape and not by operator: this
        one is chosen by operator and the shape is a consequence.)

        points    polyline for the seam. Any curve; `closed=True` joins the ends,
                  which is how you get the ring version.
        amplitude peak deviation at the seam, in the canvas's 0-1 units.
        width     how far the cusp decays over, per side, in pixels.
        gamma     cusp shape. 1.0 is a linear ramp; higher is a sharper spike
                  with a longer flat tail, which is closer to Cornsweet's.
        flip      swap which side gets the light half.
        tint      per-channel weighting, if the cusp should be warm on one side.
        """
        pts = [(float(x), float(y)) for x, y in points]
        if closed and pts[0] != pts[-1]:
            pts.append(pts[0])
        if len(pts) < 2:
            return self
        pts = self._resample(pts, int(segments))

        W = float(max(1.0, width))
        # Only pixels within W of the seam can be touched; find their bbox and
        # do the distance field on that window instead of the whole canvas.
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        x0 = max(0, int(min(xs) - W) - 2); x1 = min(self.w, int(max(xs) + W) + 3)
        y0 = max(0, int(min(ys) - W) - 2); y1 = min(self.h, int(max(ys) + W) + 3)
        if x1 <= x0 or y1 <= y0:
            return self

        px = np.arange(x0, x1, dtype=np.float32)[None, :]
        py = np.arange(y0, y1, dtype=np.float32)[:, None]
        best = np.full((y1 - y0, x1 - x0), 1e9, np.float32)
        side = np.zeros_like(best)

        for i in range(len(pts) - 1):
            ax, ay = pts[i]; bx, by = pts[i + 1]
            ex, ey = bx - ax, by - ay
            ll = ex * ex + ey * ey
            if ll < 1e-9:
                continue
            apx = px - ax; apy = py - ay
            t = np.clip((apx * ex + apy * ey) / ll, 0.0, 1.0)
            dx = apx - t * ex; dy = apy - t * ey
            d = np.sqrt(dx * dx + dy * dy)
            cross = ex * apy - ey * apx          # >0 on one side of the segment
            closer = d < best
            best = np.where(closer, d, best)
            side = np.where(closer, np.sign(cross), side)

        u = np.clip(1.0 - best / W, 0.0, 1.0) ** float(gamma)
        s = side if not flip else -side
        prof = (u * s * float(amplitude)).astype(np.float32)
        if not prof.any():
            return self

        # OVERLAP IS SILENT AND IT IS THE SAME TRAP AS EVERY OTHER ADDITIVE
        # BRUSH (21 Sept 2026, found the same night this was written). Two
        # cusps whose skirts meet simply SUM. There is no clipping, no visible
        # artefact at the join — the peaks just stack, the field between them
        # is no longer the base value, and the illusion each one is making is
        # measured against a surround the other one has quietly lifted. The
        # global integral still cancels, so measure() says everything is fine.
        #
        # The practical consequence is a BUDGET, not a bug: a cusp is 1-3
        # degrees wide, each seam needs a clear cusp-width around it, and a
        # 1200x800 picture therefore holds about two of them. Pictures made
        # this way are sparse because the physics is expensive, not because
        # anyone chose austerity.
        touched = np.abs(prof) > 1e-4
        if self._cusp_mask is None:
            self._cusp_mask = np.zeros((self.h, self.w), bool)
        clash = int((self._cusp_mask[y0:y1, x0:x1] & touched).sum())
        if clash:
            area = int(touched.sum())
            print(f"  cornsweet: WARNING — {clash} px ({100.0*clash/max(area,1):.1f}% "
                  f"of this cusp) overlap an earlier cusp. They add. The field "
                  f"between these two seams is no longer the base value.")
        self._cusp_mask[y0:y1, x0:x1] |= touched

        self.rgb[y0:y1, x0:x1] += prof[..., None] * np.asarray(tint, np.float32)
        if depth is not None:
            dt = self.depth[y0:y1, x0:x1]
            self.depth[y0:y1, x0:x1] = np.where(np.abs(prof) > 1e-4,
                                                np.minimum(dt, float(depth)), dt)
        return self

    @staticmethod
    def _resample(pts, n):
        """Even-arclength resample of a polyline to n+1 points. The distance
        field loops over segments, so a hand-written path with three points and
        one with three hundred should cost the same."""
        if n < 2 or len(pts) < 2:
            return pts
        segs = []
        total = 0.0
        for i in range(len(pts) - 1):
            d = math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
            segs.append(d); total += d
        if total <= 0:
            return pts
        out = []
        for k in range(n + 1):
            target = total * k / n
            acc = 0.0
            for i, d in enumerate(segs):
                if acc + d >= target or i == len(segs) - 1:
                    t = 0.0 if d <= 0 else (target - acc) / d
                    t = min(max(t, 0.0), 1.0)
                    out.append((pts[i][0] + t * (pts[i + 1][0] - pts[i][0]),
                                pts[i][1] + t * (pts[i + 1][1] - pts[i][1])))
                    break
                acc += d
        return out

    def probe(self, p0, p1, n=200):
        """
        Sample luminance along a line and hand back the numbers. The companion
        to measure(): measure() tells me about the whole picture, this tells me
        about one transect of it — which is the only way to check that a
        Cornsweet cusp really did return to the value it started from, rather
        than quietly leaving one field lifted.
        """
        lum = self.rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        ts = np.linspace(0.0, 1.0, int(n))
        xs = np.clip(p0[0] + (p1[0] - p0[0]) * ts, 0, self.w - 1).astype(int)
        ys = np.clip(p0[1] + (p1[1] - p0[1]) * ts, 0, self.h - 1).astype(int)
        return lum[ys, xs]

    def region_mean(self, x0, y0, x1, y1):
        """Mean luminance of a box. For proving two areas are the same number."""
        lum = self.rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        return float(lum[int(y0):int(y1), int(x0):int(x1)].mean())

    # ── deposition ──────────────────────────────────────────────────────────

    def settle(self, x0, x1, y_base, y_top, coarse, fine,
               grain_base=2.6, grain_top=0.7, density=1.0, depth=0.5,
               opacity=0.9, drape=None, sharp_base=0.55, seed=None):
        """
        A layer laid down by particles falling through still water.

        Every other brush here is a thing a hand did. This one is a thing
        GRAVITY did, and it has a rule a hand does not: heavy grains fall
        faster, so they arrive first and end up at the BOTTOM. Everything
        above them is what was still in suspension afterwards, sorted by how
        long it took to come down. That is graded bedding, and it means a
        deposited layer is never symmetrical — it is coarse and abrupt where
        it started and fine and vague where it ran out.

        So the asymmetry is the whole mark: a SHARP contact at `y_base` (the
        moment deposition began, cutting across whatever was there before)
        grading up to a diffuse top at `y_top` that has no edge at all,
        because nothing ended — the water just eventually ran out of things
        to drop. `coarse`/`fine` are the colours of the two ends; real
        sediment sorts by composition as well as size.

        `drape` is a function of x returning a vertical offset: beds are not
        spirit levels, they sag into the basin's shape.
        """
        rng = np.random.default_rng(seed)
        x0, x1 = float(x0), float(x1)
        drape = drape or (lambda x: 0.0)
        coarse = np.asarray(coarse, np.float32)
        fine = np.asarray(fine, np.float32)
        span = abs(y_base - y_top)
        if span <= 0 or x1 <= x0:
            return self
        # Enough grains to cover the band: area over mean grain footprint,
        # so a thick bed gets more paint rather than bigger paint.
        g_mean = 0.5 * (grain_base + grain_top)
        n = int(density * 1.35 * (x1 - x0) * span / max(0.35, g_mean ** 2))
        n = max(8, min(n, 420000))
        # u = 0 at the base, 1 at the top. Bias grains toward the base:
        # most of the mass drops out early, which is why the sharp contact
        # reads as a line and the top reads as a haze.
        u = rng.power(0.62, n).astype(np.float32)
        u = 1.0 - u
        xs = rng.uniform(x0, x1, n).astype(np.float32)
        sgn = -1.0 if y_top < y_base else 1.0
        ys = y_base + sgn * u * span
        ys += np.array([drape(x) for x in xs], np.float32) if drape is not None else 0.0
        # Fines drift while they sink; coarse grains go more or less straight
        # down. So lateral scatter is a function of how long the trip took.
        ys += rng.normal(0.0, 0.35 + 1.5 * u, n).astype(np.float32) * sgn
        radii = (grain_base + (grain_top - grain_base) * u) * rng.uniform(0.55, 1.6, n)
        cols = coarse[None, :] + (fine - coarse)[None, :] * u[:, None]
        cols *= rng.uniform(0.88, 1.12, (n, 1)).astype(np.float32)
        # Opacity falls off upward: the top of a bed is a thinning veil.
        ops = opacity * (1.0 - 0.72 * u) * rng.uniform(0.55, 1.0, n)
        # The basal contact: a thin dense skin right at y_base so the start of
        # deposition reads as an EVENT rather than a gradient. Without this the
        # bed has two soft edges and stops being a bed at all.
        if sharp_base > 0:
            m = u < 0.06
            ops[m] = np.minimum(1.0, ops[m] + sharp_base)
            radii[m] *= 0.8
        for i in range(n):
            self.dab(float(xs[i]), float(ys[i]), float(radii[i]),
                     np.clip(cols[i], 0.0, 1.0), depth,
                     float(np.clip(ops[i], 0.0, 1.0)), 0.30)
        return self

    # ── wiry coats ──────────────────────────────────────────────────────────

    def grizzle(self, mask, flow, n=2000, length=14.0, width=1.6, root=(0.9, 0.9, 0.9),
                tip=None, tuft=5, spread=0.16, kink=0.35, stray=0.05, depth=0.5,
                opacity=0.8, seed=None):
        """
        A wiry coat, not a soft one. Built 22 Sept 2026 for Charlie, whose fur
        was grizzled round the muzzle and never once plush — and the first
        painting of him came out plush because every hair was a smooth
        independent dash, and a thousand of those average into felt.

        Three things separate wire from fluff, so the brush does all three:
          * hairs come in TUFTS that share a root and a direction, so the coat
            clumps and parts instead of lying as an even pile;
          * each hair is straight, then KINKS once — wire bends, it doesn't curve;
          * a few STRAYS stand off at the wrong angle and break the silhouette.
        And the colour runs root → tip, because grizzle is the tips going grey
        while the roots stay the colour of the dog.

        mask: float array, hairs root where > 0.5. flow(x, y) -> angle.
        root: colour, a list to pick from per hair, or a callable (x, y) -> colour
              — PICKUP: the hair takes the colour of whatever is under its root,
              the way a real brush drags the wet paint it lands in. Without it a
              coat paints straight over its own shading and the form goes flat.
        tip:  colour, list, or a callable (root_colour) -> colour.
        """
        rng = np.random.default_rng(seed)
        ys, xs = np.nonzero(np.asarray(mask) > 0.5)
        if len(xs) == 0:
            return self

        def pick(c):
            c = np.asarray(c, np.float32)
            return c if c.ndim == 1 else c[rng.integers(0, len(c))]

        for i in rng.integers(0, len(xs), max(1, int(n // max(1, tuft)))):
            cx, cy = float(xs[i]), float(ys[i])
            base = flow(cx, cy) + rng.normal(0, 0.12)
            L = length * rng.uniform(0.75, 1.3)
            for _ in range(int(tuft)):
                a = base + rng.normal(0, spread)
                Lk = L * rng.uniform(0.7, 1.15)
                if rng.random() < stray:
                    a += rng.normal(0, 0.9)
                    Lk *= 1.4
                x0 = cx + rng.normal(0, 1.2 * width)
                y0 = cy + rng.normal(0, 1.2 * width)
                s = rng.uniform(0.35, 0.7)
                b = a + rng.normal(0, kink)
                x1, y1 = x0 + s * Lk * np.cos(a), y0 + s * Lk * np.sin(a)
                x2, y2 = x1 + (1 - s) * Lk * np.cos(b), y1 + (1 - s) * Lk * np.sin(b)
                c0 = (np.asarray(root(x0, y0), np.float32) if callable(root) else pick(root))
                c1 = (np.asarray(tip(c0), np.float32) if callable(tip)
                      else pick(tip) if tip is not None else c0)
                load = rng.uniform(0.55, 1.0)
                w = width * rng.uniform(0.7, 1.2)
                self.stroke([(x0, y0), (x1, y1), (x2, y2)],
                            (lambda t, c0=c0, c1=c1: c0 + (c1 - c0) * t ** 1.5),
                            width=(lambda t, w=w: max(0.5, w * (1.0 - 0.75 * t))),
                            depth=depth,
                            opacity=(lambda t, l=load: opacity * l * (1.0 - 0.3 * t)),
                            hardness=0.6, spacing=0.45)
        return self

    # ── skin that changes colour ────────────────────────────────────────────

    @staticmethod
    def _cells(xs, ys, spacing, rng_seed, jitter=0.8):
        """
        Nearest point of a jittered lattice, for every (xs, ys): Worley noise
        done by checking the 3x3 neighbouring cells, so it needs no scipy.
        Returns (distance, cell_i, cell_j, point_x, point_y).
        """
        gi = np.floor(xs / spacing).astype(np.int64)
        gj = np.floor(ys / spacing).astype(np.int64)
        best = np.full(xs.shape, np.inf, np.float32)
        bi, bj = gi.copy(), gj.copy()
        bx, by = np.zeros_like(xs), np.zeros_like(ys)

        def hash01(i, j, k):
            h = (i * 73856093) ^ (j * 19349663) ^ (k * 83492791) ^ (rng_seed * 2654435761)
            h = (h ^ (h >> 13)) * 1274126177
            return ((h ^ (h >> 16)) & 0xFFFFFF).astype(np.float32) / float(0xFFFFFF)

        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ci, cj = gi + di, gj + dj
                px = (ci + 0.5 + jitter * (hash01(ci, cj, 1) - 0.5)) * spacing
                py = (cj + 0.5 + jitter * (hash01(ci, cj, 2) - 0.5)) * spacing
                d = np.hypot(xs - px, ys - py).astype(np.float32)
                closer = d < best
                best = np.where(closer, d, best)
                bi = np.where(closer, ci, bi); bj = np.where(closer, cj, bj)
                bx = np.where(closer, px, bx); by = np.where(closer, py, by)
        return best, bi, bj, bx, by, hash01

    def chromatophore(self, mask, open=1.0, field=None, spacing=7.0, unit=5.0,
                      layers=None, opacity=0.95, depth=None, seed=0, filter=True):
        """
        Skin that changes colour the way a cephalopod's does. Built 26 Sept 2026,
        the morning after an octopus I painted asleep came out speckled like a
        bun with sprinkles, and a blind viewer read the speckles as rust.

        What was wrong was the unit. Scattered dots are what a DEAD brush makes.
        The real skin is a close lattice of tiny pigment sacs, each one pulled
        open by its own ring of muscles, and three pigments stacked in depth:
        yellow nearest the surface, then red, then brown underneath. Shut, a
        sac is a speck you barely see. Open, it spreads until it nearly meets
        its neighbours, and a field of open ones stops being dots and becomes
        COLOUR. And they are wired in groups: one nerve opens a whole cluster
        at once. That is where the patches with edges come from.

        So the brush does exactly that:
          * three jittered lattices (one per pigment), Worley-style;
          * MOTOR UNITS: a coarser lattice (unit x spacing); every sac takes
            its command from its unit, so neighbours open together;
          * field(x, y) -> 0..1 is the command — where the brain wants colour.
            None = smooth mottle. Pass bands, an eye bar, anything;
          * open 0..1 scales the whole display: 0 is the pale animal asleep,
            1 is full colour. Same animal, same skin, a different minute.
          * an open sac is not a perfect disc: its rim wobbles, because it is
            being pulled by a ring of separate muscles.

        mask: float array, skin where > 0.5.
        layers: list of (colour, spacing_scale, min_r, max_r, opacity), deepest first.
        filter: the pigment is a FILTER over skin that is already lit, not paint
                on top of it. So each sac's colour is scaled by the brightness of
                what's under it (luminance / 0.8). The first octopus painted with
                this brush lost its belly shadow and its eye, because an opaque
                mottle is the same brightness everywhere and the form went flat.
        """
        m = np.asarray(mask) > 0.5
        ys_, xs_ = np.nonzero(m)
        if len(xs_) == 0:
            return self
        y0, y1, x0, x1 = ys_.min(), ys_.max() + 1, xs_.min(), xs_.max() + 1
        sub = m[y0:y1, x0:x1]
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)

        if field is None:
            # default command: a smooth two-octave value noise, thresholded soft
            rng = np.random.default_rng(seed + 7)
            def field(x, y, _p=rng.uniform(0, 100, 6)):
                v = (np.sin(x / 37.0 + _p[0]) * np.sin(y / 29.0 + _p[1])
                     + 0.6 * np.sin((x + y) / 17.0 + _p[2]) * np.sin((x - y) / 23.0 + _p[3])
                     + 0.35 * np.sin(x / 9.0 + _p[4]) * np.sin(y / 11.0 + _p[5]))
                return np.clip(0.5 + 0.42 * v, 0.0, 1.0)

        if layers is None:
            layers = [((0.22, 0.10, 0.05), 1.35, 0.10, 0.62, 0.95),   # brown, deepest, biggest
                      ((0.52, 0.16, 0.09), 1.00, 0.09, 0.56, 0.85),   # red
                      ((0.80, 0.58, 0.20), 0.80, 0.08, 0.42, 0.45)]   # yellow, on top

        for li, (col, sscale, rmin, rmax, lop) in enumerate(layers):
            s = spacing * sscale
            d, ci, cj, px, py, h = self._cells(xx, yy, s, seed * 31 + li)
            # which motor unit each sac belongs to, and what that unit is told
            # ONE unit lattice for all three pigments: a nerve opens its cluster
            # through every layer at once, so the layers agree where a patch is
            # (first test had a lattice per layer and they argued)
            U = spacing * unit
            # warp where each sac looks up its unit, or the patches come out as
            # Voronoi polygons: a blind viewer called the first ones 'military
            # camo, pixel mosaic', straight edges and triangles. Real clusters
            # interleave at their borders.
            wx = px + 0.45 * U * np.sin(py / (0.9 * U) + seed) * np.cos(px / (1.3 * U))
            wy = py + 0.45 * U * np.sin(px / (0.8 * U) + 2 * seed) * np.cos(py / (1.1 * U))
            _, ui, uj, ux, uy, uh = self._cells(wx, wy, U, seed * 97 + 5)
            drive = 0.75 * field(ux, uy) + 0.25 * field(px, py)
            score = 0.62 * drive + 0.38 * uh(ui, uj, 3)       # each unit its own trigger
            e = np.clip((score - (1.0 - float(open))) / 0.14, 0.0, 1.0)
            e = e * e * (3 - 2 * e)
            # but each unit leans on its pigments differently: a dark patch is
            # the brown sacs open, a paler orange one is red and yellow open
            # with the brown held shut. Without this every patch is one rust.
            lean = uh(ui, uj, 10 + li)
            e = e * np.clip(0.15 + 1.1 * lean, 0.0, 1.0)
            e = e * (0.8 + 0.2 * h(ci, cj, 4))                 # sacs aren't identical
            # the rim wobbles: a ring of muscles, not a compass
            ang = np.arctan2(yy - py, xx - px)
            k = 5 + (h(ci, cj, 5) * 4).astype(np.int64)
            wob = 1.0 + 0.16 * e * np.sin(k * ang + 6.283 * h(ci, cj, 6))
            r = s * (rmin + (rmax - rmin) * e) * wob
            cover = np.clip((r - d) / 0.9 + 0.5, 0.0, 1.0)
            # shut sacs are specks: darker-cored, faint
            a = cover * lop * float(opacity) * (0.35 + 0.65 * e) * sub
            a = a.astype(np.float32)
            tile = self.rgb[y0:y1, x0:x1]
            target = np.asarray(col, np.float32)
            if filter:
                lum = tile @ np.asarray([0.2126, 0.7152, 0.0722], np.float32)
                target = target * np.clip(lum / 0.8, 0.0, 1.25)[..., None]
            self.rgb[y0:y1, x0:x1] = tile * (1 - a[..., None]) + target * a[..., None]
            self.height[y0:y1, x0:x1] += 0.15 * a * self.thickness
            if depth is not None:
                dt = self.depth[y0:y1, x0:x1]
                self.depth[y0:y1, x0:x1] = np.where(a > 0.35, np.minimum(dt, float(depth)), dt)
        return self

    # ── stamping a shaped tip ───────────────────────────────────────────────

    def stamp(self, x, y, size, tip, colour, angle=0.0, depth=0.5, opacity=1.0, aspect=1.0, load=None):
        """
        Lay one shaped tip down, rotated to `angle`. The tip is evaluated on a
        rotated grid rather than drawn and then resampled, so it stays crisp at
        any angle — no interpolation blur, which is what makes stamped brushes
        look muddy.
        """
        r = max(0.6, float(size))
        reach = int(np.ceil(r * 1.5)) + 1
        y0, y1 = max(0, int(y) - reach), min(self.h, int(y) + reach + 1)
        x0, x1 = max(0, int(x) - reach), min(self.w, int(x) + reach + 1)
        if y1 <= y0 or x1 <= x0:
            return self
        yy = (np.arange(y0, y1, dtype=np.float32)[:, None] - y) / r
        xx = (np.arange(x0, x1, dtype=np.float32)[None, :] - x) / r
        ca, sa = np.cos(-angle), np.sin(-angle)
        u = (xx * ca - yy * sa)
        v = (xx * sa + yy * ca) / max(1e-3, aspect)
        a = np.clip(tip(u, v), 0.0, 1.0).astype(np.float32) * float(opacity)
        a = self._apply_tooth(a, y0, y1, x0, x1, self._load(load, opacity))
        a[a < 0.004] = 0.0
        if not a.any():
            return self
        tile = self.rgb[y0:y1, x0:x1]
        self.rgb[y0:y1, x0:x1] = tile * (1.0 - a[..., None]) + np.asarray(colour, np.float32) * a[..., None]
        dt = self.depth[y0:y1, x0:x1]
        self.depth[y0:y1, x0:x1] = np.where(a > 0.35, np.minimum(dt, float(depth)), dt)
        self.height[y0:y1, x0:x1] += a * self.thickness
        return self

    def stamp_along(self, points, tip, colour, width=16.0, depth=0.5, opacity=0.9,
                    spacing=0.22, follow=True, angle=0.0, jitter=0.0, aspect=1.0, seed=None):
        """
        A stroke made by stamping a tip repeatedly. `follow` turns the tip to
        the direction of travel, which is the difference between a brush and a
        rubber stamp; `jitter` adds a little rotational noise so the marks stop
        being identical.
        """
        rng = np.random.default_rng(seed)
        pts = np.asarray(points, np.float32).reshape(-1, 2)
        w0 = width(0.0) if callable(width) else width
        prev = None
        for t, x, y in self._walk(pts, max(0.6, spacing * float(w0))):
            if follow:
                if prev is None:
                    ang = angle
                else:
                    dx, dy = x - prev[0], y - prev[1]
                    ang = np.arctan2(dy, dx) if (dx or dy) else angle
            else:
                ang = angle
            prev = (x, y)
            self.stamp(x, y,
                       (width(t) if callable(width) else width) * 0.5,
                       tip,
                       colour(t) if callable(colour) else colour,
                       ang + rng.normal(0, jitter),
                       depth(t) if callable(depth) else depth,
                       opacity(t) if callable(opacity) else opacity,
                       aspect)
        return self

    def smudge(self, points, width=24.0, depth=0.5, rate=0.85, colour=None, load=0.0,
               spacing=0.20, hardness=0.30, seed=None):
        """
        Move paint instead of adding it.

        Every other mark here puts new colour down. This one picks up what is
        already under the brush and carries it forward, mixing a little as it
        goes — which is what a wet brush dragged across a wet wash actually
        does, and it is how both Krita and Procreate build their fringe and
        wet-edge brushes. `rate` is how much of the picked-up colour survives
        each step; `load` optionally adds a trickle of fresh pigment on top.

        The consequence worth knowing: a smudge cannot invent anything. Drag it
        across bare paper and nothing happens. It only has whatever the picture
        already gave it.
        """
        rng = np.random.default_rng(seed)
        w0 = width(0.0) if callable(width) else width
        carried = None
        for t, x, y in self._walk(points, max(0.8, spacing * float(w0))):
            r = max(1.0, (width(t) if callable(width) else width) * 0.5)
            y0, y1 = max(0, int(y - r)), min(self.h, int(y + r) + 1)
            x0, x1 = max(0, int(x - r)), min(self.w, int(x + r) + 1)
            if y1 <= y0 or x1 <= x0:
                continue
            here = self.rgb[y0:y1, x0:x1].reshape(-1, 3).mean(0)
            carried = here if carried is None else carried * rate + here * (1.0 - rate)
            paint = carried if colour is None else carried * (1.0 - load) + np.asarray(colour, np.float32) * load
            self.dab(x, y, r, paint, depth, 0.55, hardness)
        return self

    def canopy(self, x, y, spread, colour, depth=0.5, shade=0.55, count=None,
               size=None, opacity=(0.72, 0.95), sun=(-0.35, -0.85), seed=0,
               fall=0.44, hardness=0.30):
        """
        One MASS of foliage — a clump of leaves at distance, not a tree and not
        a leaf. Written 14 Sept 2026 after painting the wooded wall of a maar
        three times and hand-rolling this loop inline each time.

        THE LESSON IT EXISTS TO MAKE UNREPEATABLE. Version 2 of that picture
        put 1500 independent dabs on the far slope at radius 3-9 and opacity
        0.35-0.78, and the result was not woodland, it was DITHER — a granular
        crust along the skyline that read as a compression artefact. Three
        separate mistakes, all of them the same mistake:

          (1) every dab had its own colour, so there were no masses, only noise;
          (2) the dabs were small relative to the area they covered;
          (3) the opacity was low, so each one was a suggestion and the eye
              integrated the lot into grey.

        A wood at 500 m is a handful of shapes with a lit top and a dark
        underside. It is not many small things. So: one clump shares ONE base
        colour, its dabs are sized FROM the spread rather than set separately,
        and each dab gets a shadow twin below it on the far side of `sun`.

        `size` defaults to a band scaled off `spread`, which makes the dither
        failure impossible without asking for it on purpose. Pass it if you
        want to override; it will warn you if you ask for grit.

        `shade` is how far the underside falls toward black-green (0 = flat
        mass, 1 = hard chiaroscuro). `fall` is the fraction of a dab radius the
        shadow twin is offset by.
        """
        rng = np.random.default_rng(seed)
        spread = float(spread)
        if size is None:
            size = (spread * 0.22, spread * 0.52)
        lo, hi = float(size[0]), float(size[1])
        if hi < spread * 0.12:
            print(f"canopy: size {size} is grit against spread {spread:.0f} "
                  f"— this is the dither failure. Want >= {spread * 0.12:.1f}.")
        if count is None:
            # enough dabs to close the mass, derived from area, never guessed
            count = int(np.clip((spread * spread) / (0.62 * hi * hi), 5, 40))

        base = np.asarray(colour, np.float32)
        dark = base * (1.0 - float(shade)) + np.array([0.028, 0.048, 0.032], np.float32) * float(shade)
        sx, sy = float(sun[0]), float(sun[1])
        n = float(np.hypot(sx, sy)) or 1.0
        sx, sy = sx / n, sy / n

        o0, o1 = float(opacity[0]), float(opacity[1])
        for _ in range(int(count)):
            dx, dy = rng.normal(0, spread * 0.42), rng.normal(0, spread * 0.30)
            r = rng.uniform(lo, hi)
            op = rng.uniform(o0, o1)
            # underside first, so the lit crown sits on top of its own shadow
            self.dab(x + dx - sx * r * fall, y + dy - sy * r * fall, r * 0.82,
                     tuple(dark), depth=depth + 0.004, opacity=op * 0.85,
                     hardness=hardness)
            self.dab(x + dx, y + dy, r, tuple(base * rng.uniform(0.90, 1.10)),
                     depth=depth, opacity=op, hardness=hardness)
        return self

    def treeline(self, fn, x0, x1, spread, colour, depth=0.5, step=None,
                 broken=0.22, seed=0, **kw):
        """
        A run of `canopy` masses along a skyline given as fn(x) -> y. The thing
        it saves is the edge: a treeline is not a fringe of equal bushes, it
        has gaps and it has clumps that stand a head above their neighbours,
        and both of those have to be there or the silhouette reads as a hedge.

        `broken` is the fraction of positions simply skipped, which is what
        makes a gap. Extra kwargs pass through to canopy.
        """
        rng = np.random.default_rng(seed)
        step = float(step if step is not None else spread * 0.80)
        x = float(x0)
        i = 0
        while x < x1:
            if rng.random() >= broken:
                s = spread * rng.uniform(0.62, 1.45)
                lift = rng.uniform(-0.45, 0.12) * s
                self.canopy(x, fn(x) + lift, s, colour, depth=depth,
                            seed=seed * 977 + i, **kw)
            x += step * rng.uniform(0.6, 1.4)
            i += 1
        return self


    # ── bare trees ───────────────────────────────────────────────────────────

    @staticmethod
    def _branches(x, y, height, rng, levels=7, spread=40.0, rise=0.30,
                  droop=0.0, lean=0.0, trunk=0.30, width=None, fork=(2, 3),
                  leader=0.55):
        """
        The skeleton of one bare tree as a list of (x0, y0, x1, y1, w, gen).
        Pure geometry, no paint, so a crown can be drawn at any scale.

        LEADER. The first version forked every branch into equal children, and
        every tree came out a vase: all the crown at the top of a bare stalk,
        an acacia or a lollipop. A temperate tree has apical dominance: one
        child carries on nearly straight (the leader), the others leave it
        sideways, shorter and thinner, so there are limbs at every height.
        `leader` is the chance a node keeps one (0 = elm-like vase, 1 = conifer
        spine). Laterals alternate sides.

        `rise` bends laterals back toward vertical (oak and lime low, poplar
        high). `droop` pulls the last two generations down (birch). `trunk` is
        the clear bole below the first limb, as a fraction of height.
        """
        segs = []
        w0 = float(width if width is not None else max(0.8, height * 0.030))
        bole = height * trunk
        r = 0.86
        L1 = (height - bole) / sum(r ** k for k in range(levels))
        stack = [(float(x), float(y), -90.0 + lean, bole, w0, 0, 1)]
        while stack:
            bx, by, ang, ln, w, g, side = stack.pop()
            px, py, a = bx, by, ang
            for k in range(3):
                a += rng.normal(0.0, 5.0 + 3.0 * g)
                if g >= levels - 2 and droop:
                    a += droop * (1.0 if np.cos(np.radians(a)) >= 0 else -1.0)
                qx = px + np.cos(np.radians(a)) * ln / 3.0
                qy = py + np.sin(np.radians(a)) * ln / 3.0
                segs.append((px, py, qx, qy, w * (1.0 - 0.10 * k), g))
                px, py = qx, qy
            if g >= levels:
                continue
            base = L1 if g == 0 else ln
            n = int(rng.integers(fork[0], fork[1] + 1))
            if g >= levels - 2:
                # the twig SPRAY: the last orders fork more, and that density,
                # not any one twig, is what the eye reads as a winter crown
                n += int(rng.integers(1, 3))
            has_leader = rng.random() < leader
            for i in range(n):
                if has_leader and i == 0:
                    ca = a + rng.normal(0.0, 8.0)
                    ca += ((-90.0) - ca) * 0.25
                    stack.append((px, py, ca, base * rng.uniform(0.82, 0.92),
                                  w * rng.uniform(0.74, 0.82), g + 1, side))
                    continue
                side = -side
                off = side * spread * rng.uniform(0.8, 1.5)
                ca = a + off
                ca = ca + ((-90.0) - ca) * rise * rng.uniform(0.4, 1.1)
                k = 0.88 if has_leader else 1.0
                stack.append((px, py, ca, base * rng.uniform(0.66, 0.88) * k,
                              w * rng.uniform(0.50, 0.66), g + 1, side))
        return segs

    def bare_wood(self, trees, colour, twig=None, depth=0.5, opacity=1.0,
                  haze=0.0, ss=4, fine=3):
        """
        Leafless trees, drawn as COVERAGE, not as paint. Written 30 Sept 2026
        after a winter treeline failed three ways in one night ('The Powder
        Still Goes Up'): a big crest read as hills, thin spikes as a city,
        thresholded noise above the crest as a flock of birds.

        What a bare wood actually is, at distance: thousands of twigs each far
        thinner than a pixel. You never see one. You see how much of each
        pixel they block — a warm, see-through haze that is densest where
        crowns overlap and lacy at the rim, with the trunks and main limbs as
        the only real lines in it. So this rasterises every branch at `ss`x
        resolution with its true width (sub-pixel twigs as faint 1-px lines
        scaled by their width) and box-downsamples. The haze is not a soft
        brush pretending; it falls out of the geometry.

        `trees` is a list of dicts: {x, y, height, ...} where anything else is
        passed to _branches (levels, spread, rise, droop, lean, trunk, seed).
        Generations >= levels - `fine` take the `twig` colour (the purplish
        twig haze of birch and alder), the rest take `colour`.
        `haze` blurs the coverage in canvas pixels, for trees far enough off
        that even the limbs have gone.

        All trees go into ONE mask, so where crowns overlap the coverage is
        their union, not a double layer of paint.
        """
        from PIL import Image, ImageDraw, ImageFilter
        twig = colour if twig is None else twig
        S = int(ss)
        limb = Image.new("L", (self.w * S, self.h * S), 0)
        fin = Image.new("L", (self.w * S, self.h * S), 0)
        dl, df = ImageDraw.Draw(limb), ImageDraw.Draw(fin)
        for i, t in enumerate(trees):
            t = dict(t)
            rng = np.random.default_rng(t.pop("seed", i * 7919 + 11))
            levels = t.get("levels", 7)
            segs = self._branches(t.pop("x"), t.pop("y"), t.pop("height"), rng, **t)
            for x0, y0, x1, y1, w, g in segs:
                ws = w * S
                if ws >= 1.0:
                    lw, val = int(round(ws)), 255
                else:
                    lw, val = 1, int(255 * ws)
                d = df if g >= levels - fine else dl
                d.line([(x0 * S, y0 * S), (x1 * S, y1 * S)], fill=val, width=lw)
                if lw >= 3:  # round the joint, or thick limbs notch at every bend
                    h = lw / 2.0
                    d.ellipse([x0 * S - h, y0 * S - h, x0 * S + h, y0 * S + h], fill=val)
        out = []
        for m in (limb, fin):
            if haze > 0:
                m = m.filter(ImageFilter.GaussianBlur(haze * S))
            m = m.resize((self.w, self.h), Image.BOX)
            out.append(np.asarray(m, np.float32) / 255.0)
        for a, col in ((out[1], twig), (out[0], colour)):
            a = np.clip(a * float(opacity), 0.0, 1.0)
            self.rgb = self.rgb * (1.0 - a[..., None]) + np.asarray(col, np.float32) * a[..., None]
            self.depth = np.where(a > 0.35, np.minimum(self.depth, float(depth)), self.depth)
        self.ops.append(("bare_wood", len(trees)))
        return self

    def pollard(self, x, y, height, girth, bark, rod, rod_tip=None, years=3,
                lean=0.0, light=-1.0, rods=140, hollow=None, head=1.7,
                depth=0.3, seed=0, ss=4):
        """
        A pollarded tree: a short fat bole, a swollen HEAD of callus knuckles
        where it has been cut back to the same place for decades, and a
        fountain of straight unbranched rods out of the knuckles. Written
        1 Oct 2026 for the pollard willows of a river meadow.

        Why it is not bare_wood(): nothing about it is a branching tree. The
        cut resets the crown, so every rod is the same age and starts from the
        same few lumps. They are long, nearly straight, barely forked (one-year
        rods not at all), and they splay outward as they lengthen because their
        own weight bends them. A _branches() skeleton with a short trunk reads
        as a shrub on a post.

        `height` is the bole to the top of the head, `girth` its width at the
        foot. `years` since the last cut sets the rod length; 0 is freshly cut:
        stubs and pale cut faces on the knuckles, which is the most foreign
        silhouette a tree can have (a fist). `light` is -1 for sun from the
        left, +1 from the right. `hollow` = (t, size) puts a dark split in the
        bole, t from foot (0) to head (1): pollards rot open young because the
        cut lets fungi into the heartwood, which is the whole reason owls live
        in them.

        The bole is shaded across its width per pixel (a cylinder, lit on one
        flank), with vertical furrows; the rods are coverage like bare_wood.
        """
        from PIL import Image, ImageDraw
        rng = np.random.default_rng(seed)
        rod_tip = rod if rod_tip is None else rod_tip
        lean_r = np.radians(lean)
        # the spine, foot to head, with a slight bow
        n = 64
        t = np.linspace(0.0, 1.0, n)
        bow = rng.normal(0, 0.05) * girth
        cx = x + np.sin(lean_r) * height * t + bow * np.sin(np.pi * t)
        cy = y - height * t
        # half-width: a flare at the foot, a slight waist, widening into the head
        hw = 0.5 * girth * (1.0 + 0.45 * np.exp(-t / 0.06) - 0.10 * np.sin(np.pi * t)
                            + 0.25 * np.clip((t - 0.6) / 0.4, 0, 1) ** 2)
        hw *= 1.0 + 0.07 * np.sin(t * rng.uniform(9, 14) + rng.uniform(0, 6))
        hw *= 1.0 + 0.05 * np.sin(t * rng.uniform(23, 31) + rng.uniform(0, 6))
        # the HEAD: a lumpy club wider than the bole, centred a little below the top
        R = 0.5 * girth * head
        hcx, hcy = float(cx[-1]) + rng.normal(0, 0.06) * girth, float(cy[-1]) + 0.35 * R
        top = (hcx, hcy - 0.55 * R)
        # raster the bole outline at ss
        S = int(ss)
        x0 = int(max(0, np.floor(min((cx - hw).min(), hcx - 2 * R))))
        x1 = int(min(self.w, np.ceil(max((cx + hw).max(), hcx + 2 * R))))
        y0 = int(max(0, np.floor(hcy - 2 * R)))
        y1 = int(min(self.h, np.ceil(y + girth * 0.3)))
        if x1 <= x0 or y1 <= y0:
            return self
        bw, bh = x1 - x0, y1 - y0
        m = Image.new("L", (bw * S, bh * S), 0)
        dm = ImageDraw.Draw(m)
        poly = [((cx[i] - hw[i] - x0) * S, (cy[i] - y0) * S) for i in range(n)]
        poly += [((cx[i] + hw[i] - x0) * S, (cy[i] - y0) * S) for i in range(n - 1, -1, -1)]
        dm.polygon(poly, fill=255)
        # head outline: an ellipse whose radius wanders, so it is a club, not a ball
        th = np.linspace(0, 2 * np.pi, 72, endpoint=False)
        rr = R * (1 + 0.06 * np.sin(3 * th + rng.uniform(0, 6)) + 0.05 * np.sin(5 * th + rng.uniform(0, 6))
                  + rng.normal(0, 0.03, th.size))
        dm.polygon([((hcx + np.cos(a_) * r_ - x0) * S, (hcy + np.sin(a_) * r_ * 0.62 - y0) * S)
                    for a_, r_ in zip(th, rr)], fill=255)
        # knuckles: the cut points, lumps crowded along the top rim of the head
        knuckles = []
        kn = int(rng.integers(6, 10))
        for k in range(kn):
            a_ = np.radians(-90 + (k / (kn - 1) - 0.5) * 115 + rng.normal(0, 8))
            kx = hcx + np.cos(a_) * R * rng.uniform(0.5, 0.8)
            ky = hcy + np.sin(a_) * R * 0.62 * rng.uniform(0.7, 1.0)
            ks = R * rng.uniform(0.22, 0.36)
            knuckles.append((kx, ky, ks))
            dm.ellipse([(kx - ks - x0) * S, (ky - ks * 0.8 - y0) * S,
                        (kx + ks - x0) * S, (ky + ks * 0.8 - y0) * S], fill=255)
        a = np.asarray(m.resize((bw, bh), Image.BOX), np.float32) / 255.0
        # shading: position across the bole at each row -> cylinder normal
        yy = np.arange(y0, y1, dtype=np.float32)
        order = np.argsort(cy)
        ccx = np.interp(yy, cy[order], cx[order])
        chw = np.interp(yy, cy[order], hw[order])
        chw = np.maximum(chw, 1.0)
        xs = np.arange(x0, x1, dtype=np.float32)[None, :]
        u = np.clip((xs - ccx[:, None]) / (chw[:, None] * 1.05), -1, 1)
        # in the head the knuckles are their own little domes
        kd = np.zeros_like(u)
        for kx, ky, ks in knuckles:
            dx = (xs - kx) / ks
            dy = (yy[:, None] - ky) / (ks * 0.85)
            inside = dx * dx + dy * dy < 1
            kd = np.where(inside, np.clip(dx, -1, 1), kd)
        hu = np.clip((xs - hcx) / (R * 1.05), -1, 1)
        hv = (yy[:, None] - hcy) / (R * 0.62)
        inhead = (hu * hu + hv * hv < 1.1) & (yy[:, None] < hcy + R * 0.2)
        u = np.where(inhead, hu, u)
        u = np.where(kd != 0, 0.3 * u + 0.7 * kd, u)
        nz = np.sqrt(np.clip(1 - u * u, 0, 1))
        lit = np.clip(u * light * 0.95 + nz * 0.35, 0, 1)
        shade = 0.28 + 0.95 * lit ** 1.3
        # furrows: vertical ridges that follow the spine, wobbling, coarser low down
        fr = np.sin((xs - ccx[:, None]) / (chw[:, None] + 1) * rng.uniform(9, 13)
                    + 0.6 * np.sin(yy[:, None] / rng.uniform(13, 21))
                    + rng.uniform(0, 6))
        # bark grain: stretched along the bole, smooth (a kron upscale shows as blocks)
        noise = np.asarray(Image.fromarray(rng.normal(0, 1, (bh // 6 + 2, bw // 2 + 2)).astype(np.float32), "F")
                           .resize((bw, bh), Image.BICUBIC), np.float32)
        tex = 1.0 + 0.30 * np.clip(fr, -1, 0.2) ** 3 + 0.06 * noise
        col = np.asarray(bark, np.float32)[None, None, :] * (shade * tex)[..., None]
        # the hollow: a dark ragged split with a pale lip on its lit edge
        if hollow is not None:
            ht, hs = hollow
            hi = int(np.clip(ht, 0, 1) * (n - 1))
            hx, hy = cx[hi] + hw[hi] * 0.15 * -light, cy[hi]
            # a split, not a hole: tall, narrow, leaning, pointed at both
            # ends and ragged on one lip. A symmetric sine rag gives it a waist
            # and it reads as a keyhole or a little standing figure (blind check).
            dy = (yy[:, None] - hy) / hs
            dx = (xs - hx - dy * hs * 0.12) / (hs * 0.30 * (1 - 0.55 * np.abs(np.clip(dy, -1, 1))) + 1e-3)
            rag = 1 + 0.25 * np.clip(noise, -1, 1) * (dx > 0)
            d = np.sqrt(dx * dx + dy * dy) / rag
            hole = np.clip((1.0 - d) * 6, 0, 1)
            lip = np.clip((1.12 - d) * 6, 0, 1) - hole
            side = np.clip(-dx * light, 0, 1)
            col = col * (1 - hole[..., None]) + np.array([0.05, 0.04, 0.035]) * hole[..., None]
            col = col + (np.asarray(bark) * 0.5)[None, None, :] * (lip * side)[..., None]
        tile = self.rgb[y0:y1, x0:x1]
        self.rgb[y0:y1, x0:x1] = tile * (1 - a[..., None]) + col * a[..., None]
        dt = self.depth[y0:y1, x0:x1]
        self.depth[y0:y1, x0:x1] = np.where(a > 0.35, np.minimum(dt, float(depth)), dt)

        # the rods: coverage, base colour low and tip colour high
        L = height * (0.0 if years <= 0 else 0.55 * min(years, 6) ** 0.62)
        segs = []
        stubs = []
        if years <= 0:
            for kx, ky, ks in knuckles:
                for j in range(int(rng.integers(1, 4))):
                    a_ = np.radians(-90 + rng.normal(0, 25))
                    l_ = ks * rng.uniform(0.25, 0.6)
                    bx = kx + rng.normal(0, ks * 0.4)
                    by = ky - ks * 0.5
                    stubs.append((bx, by, bx + np.cos(a_) * l_, by + np.sin(a_) * l_, ks * 0.07))
                # the cut faces: pale heartwood, seen nearly edge-on from the
                # ground, so thin flat slivers on the knuckle tops, not discs
                # (round white ovals in a row read as teeth)
                if rng.random() < 0.7:
                    fx = kx + rng.normal(0, ks * 0.2)
                    self.stamp(fx, ky - ks * 0.62, ks * 0.34, tip_round(0.6),
                               tuple(np.minimum(np.asarray(bark) * 1.5 + 0.08, 0.9)),
                               depth=depth, opacity=0.6, aspect=0.3)
        else:
            # a lapsed pollard SELF-THINS: past a few years most rods are
            # shaded out and the survivors thicken into poles with crowns
            thin = 1.0 if years <= 3 else 3.0 / years
            per = max(1, int(rods * thin) // max(1, len(knuckles)))
            fat = 1.0 if years <= 3 else (years / 3.0) ** 0.7
            for kx, ky, ks in knuckles:
                out = (kx - hcx) / (R + 1e-6)
                for j in range(per + int(rng.integers(-2, 3))):
                    a_ = -90 + 38 * out + rng.normal(0, 9) + lean
                    l_ = L * rng.uniform(0.45, 1.15)
                    w_ = max(0.8, fat * girth * 0.04 * rng.uniform(0.6, 1.3) * (l_ / height) ** 0.5)
                    px, py = kx + rng.normal(0, ks * 0.5), ky - ks * rng.uniform(0.0, 0.6)
                    steps = 10
                    for s in range(steps):
                        # weight bends a long rod outward: more at the tip
                        sa = np.radians(a_ + np.sign(a_ + 90 + 1e-6) * 9 * (s / steps) ** 2
                                        * (l_ / height))
                        qx = px + np.cos(sa) * l_ / steps
                        qy = py + np.sin(sa) * l_ / steps
                        segs.append((px, py, qx, qy, w_ * (1 - 0.75 * s / steps), s / steps))
                        # older rods throw the odd side shoot in their top third
                        if years >= 2 and s > steps * (0.6 if years <= 3 else 0.35) and rng.random() < min(0.6, 0.06 * years):
                            ba = sa + np.radians(rng.choice([-1, 1]) * rng.uniform(20, 40))
                            bl = l_ * rng.uniform(0.08, 0.2)
                            segs.append((qx, qy, qx + np.cos(ba) * bl, qy + np.sin(ba) * bl,
                                         w_ * 0.2, 1.0))
                        px, py = qx, qy
        # stubs of a fresh cut are old wood, not rods: bark-coloured, lit like the bole
        for sx0, sy0, sx1, sy1, w in stubs:
            self.stroke([(sx0, sy0), (sx1, sy1)], tuple(np.asarray(bark) * 0.7), width=max(1.0, w * 2),
                        depth=depth, opacity=0.95, hardness=0.6)
        # colour runs base -> tip in four bands, so a rod has no seam; on an
        # old pole the lower wood has barked over and goes toward the bole
        nb = 4
        base = np.asarray(rod, np.float32)
        if years > 3:
            base = base + (np.asarray(bark, np.float32) - base) * min(1.0, (years - 3) / 5.0)
        masks = [Image.new("L", (self.w * S, self.h * S), 0) for _ in range(nb)]
        draws = [ImageDraw.Draw(mk) for mk in masks]
        for sx0, sy0, sx1, sy1, w, f in segs:
            ws = w * S
            lw, val = (int(round(ws)), 255) if ws >= 1 else (1, int(255 * ws))
            draws[min(nb - 1, int(f * nb))].line([(sx0 * S, sy0 * S), (sx1 * S, sy1 * S)],
                                                 fill=val, width=lw)
        for i, mk in enumerate(masks):
            k = i / (nb - 1)
            c_ = base * (1 - k) + np.asarray(rod_tip, np.float32) * k
            cov = np.asarray(mk.resize((self.w, self.h), Image.BOX), np.float32) / 255.0
            self.rgb = self.rgb * (1 - cov[..., None]) + c_ * cov[..., None]
            self.depth = np.where(cov > 0.35, np.minimum(self.depth, float(depth)), self.depth)
        self.ops.append(("pollard", years))
        return self


# ── tips ─────────────────────────────────────────────────────────────────────
# Every brush on a sheet like that is the same idea: a shaped tip, stamped
# repeatedly along a path, rotated to follow the direction of travel. My earlier
# brushes were all round gaussians, which is why they could only ever make
# noodles. A tip is a function of local coordinates (u, v) in [-1, 1], returning
# alpha — so it is evaluated directly on a rotated grid and never resampled.

def tip_round(hardness=0.35):
    def f(u, v):
        d = np.sqrt(u * u + v * v)
        core = np.clip(hardness, 0.0, 0.98)
        a = np.exp(-np.square(np.clip(d - core, 0, None)) / (2 * (0.40 * (1 - core) + 0.08) ** 2))
        return np.where(d <= core, 1.0, a)
    return f


def tip_chisel(ratio=0.24, softness=0.14):
    """A flat brush seen edge-on: wide one way, thin the other, ends squared off."""
    def f(u, v):
        a = np.clip((1.0 - np.abs(u)) / softness, 0, 1)
        b = np.clip((ratio - np.abs(v)) / (softness * ratio + 1e-3), 0, 1)
        return a * b
    return f


def tip_ragged(seed=0, bite=0.55, lobes=7):
    """
    A disc eaten into by a low-frequency wobble around its rim. This is the
    shape under most 'natural media' brushes — the edge is irregular at a scale
    you can see, not just noisy at the pixel level.
    """
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0, 2 * np.pi, lobes)
    amp = rng.uniform(0.5, 1.0, lobes)
    amp = amp / amp.sum()
    def f(u, v):
        d = np.sqrt(u * u + v * v) + 1e-6
        th = np.arctan2(v, u)
        wob = sum(a * np.sin((k + 2) * th + p) for k, (a, p) in enumerate(zip(amp, ph)))
        edge = 1.0 - bite * (0.5 + 0.5 * wob)
        return np.clip((edge - d) / 0.22, 0, 1)
    return f


def tip_spatter(seed=0, blobs=9, tightness=0.55):
    """Clustered droplets. Ink flicked off a loaded brush."""
    rng = np.random.default_rng(seed)
    cx = rng.normal(0, tightness, blobs)
    cy = rng.normal(0, tightness, blobs)
    rr = rng.uniform(0.10, 0.34, blobs)
    def f(u, v):
        out = np.zeros_like(u)
        for x, y, r in zip(cx, cy, rr):
            d = np.sqrt((u - x) ** 2 + (v - y) ** 2)
            out = np.maximum(out, np.clip((r - d) / (0.35 * r), 0, 1))
        return out
    return f


def tip_rake(hairs=6, thickness=0.10):
    """Separate parallel hairs — a comb of them, for hatching and grain."""
    def f(u, v):
        out = np.zeros_like(u)
        for i in range(hairs):
            y = -1.0 + 2.0 * (i + 0.5) / hairs
            out = np.maximum(out, np.clip((thickness - np.abs(v - y)) / (thickness * 0.6), 0, 1))
        return out * np.clip((1.0 - np.abs(u)) / 0.2, 0, 1)
    return f
