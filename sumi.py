"""
sumi — soot, glue, water, and paper that drinks.

The tool is named after lampblack and in two weeks it never once painted in
it. This is the fix.

What an ink stick is (read 23 Sept 2026): soot from burning pine or oil,
floated in water so the fine particles separate from the coarse, beaten into
animal glue, pressed, dried for weeks, and then left for years because the
glue slowly hydrolyses and the ink gets better. Old ink — koboku — is prized.

What makes it look like itself, and what this module does:

  * ONE BLACK, SEVERAL COLOURS. "Sumi ni gosai ari" — ink has five colours.
    Dense, it is black. Diluted, what shows is the particle size: soot made
    at uneven temperatures has mixed particle sizes, and the small ones tint
    the wash. Blue-black (ao-zumi) or brown-black (cha-boku) depending on
    the stick. So each stroke here carries TWO populations — fine and
    coarse — and they are rendered with different absorption per channel.
  * THE HALO IS A DIFFERENT COLOUR FROM THE CORE. Fine particles ride the
    water further than coarse ones. On absorbent paper the bleed edge is a
    pale tinted ghost around a neutral black middle — a chromatography you
    get for free. Nothing I drew; it falls out of two mobilities.
  * PAPER WITH A GRAIN. Xuan paper is fibre, and water runs along fibres,
    so the bleed is feathered, not round. `conduct` is that fibre field.
  * DRY BRUSH (feibai, "flying white"). A starved brush skips in streaks
    along its own direction and leaves the paper showing through.

Sources I read: Wikipedia 'Inkstick'; pigment.tokyo on koboku and on sumi
colours. Honest note: sources disagree on WHICH soot is which colour —
one says lamp-oil soot is the finer particle, retail lore often calls pine
soot the blue one. I made it a parameter rather than pretend to know.
"""

import numpy as np
import lampblack as lb

# per-channel absorption (R, G, B). Higher = that channel is eaten harder.
COARSE = np.array([2.45, 2.40, 2.30], np.float32)       # near-neutral, faintly warm black
AO     = np.array([2.60, 1.95, 1.35], np.float32)       # fine particles, blue-black
CHA    = np.array([1.35, 1.80, 2.45], np.float32)       # fine particles, brown-black

# The only colours allowed near ink (24 Sept 2026, for the plum). Each is
# (absorption per channel, mobility in water). The two classic reds for plum
# behave OPPOSITELY on wet paper, and that is the whole reason to have both:
#   yanzhi 胭脂 — rouge, a dye from safflower. Dissolved, so it rides the
#     water like the fine soot does: soft, feathered, pink at the bleed.
#   zhusha 朱砂 — cinnabar, ground mineral. Heavy grains; it sits exactly
#     where the brush put it. Seal paste is cinnabar.
TINTS = {
    "yanzhi": (np.array([0.20, 2.25, 1.40], np.float32), 0.75),
    "zhusha": (np.array([0.10, 1.85, 2.35], np.float32), 0.06),
}


def _lap(a):
    return (np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1) - 4 * a)


class Sheet:
    def __init__(self, width, height, paper=(0.94, 0.915, 0.86), fibre=1.1, seed=0):
        self.w, self.h = int(width), int(height)
        self.rng = np.random.default_rng(seed)
        self.paper = np.array(paper, np.float32)
        self.fine = np.zeros((self.h, self.w), np.float32)
        self.coarse = np.zeros((self.h, self.w), np.float32)
        self.water = np.zeros((self.h, self.w), np.float32)
        self.conduct = self._fibres(fibre)
        self.tints = {}

    def _fibres(self, strength):
        # long thin streaks at random angles, summed: a mat of fibres
        f = np.zeros((self.h, self.w), np.float32)
        yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        for _ in range(900):
            x0, y0 = self.rng.uniform(0, self.w), self.rng.uniform(0, self.h)
            ang = self.rng.uniform(0, np.pi)
            L = self.rng.uniform(20, 90)
            x1, y1 = x0 + np.cos(ang) * L, y0 + np.sin(ang) * L
            xa, xb = int(max(0, min(x0, x1) - 2)), int(min(self.w, max(x0, x1) + 3))
            ya, yb = int(max(0, min(y0, y1) - 2)), int(min(self.h, max(y0, y1) + 3))
            if xb <= xa or yb <= ya:
                continue
            X, Y = xx[ya:yb, xa:xb], yy[ya:yb, xa:xb]
            dx, dy = x1 - x0, y1 - y0
            t = np.clip(((X - x0) * dx + (Y - y0) * dy) / (dx * dx + dy * dy), 0, 1)
            d = np.hypot(X - (x0 + t * dx), Y - (y0 + t * dy))
            f[ya:yb, xa:xb] += np.exp(-(d / 0.9) ** 2)
        f = lb.blur(f[..., None], 1)[..., 0]
        noise = lb.blur(self.rng.random((self.h, self.w, 1)).astype(np.float32), 6)[..., 0]
        noise = (noise - noise.mean()) / (noise.std() + 1e-6)
        c = 0.30 + 0.18 * noise + strength * np.clip(f, 0, 1.5)
        return np.clip(c, 0.08, 1.6).astype(np.float32)

    # ── laying ink ──────────────────────────────────────────────────────────

    def _walk(self, pts, step):
        pts = np.asarray(pts, np.float32)
        seg = np.hypot(*np.diff(pts, axis=0).T)
        s = np.concatenate([[0], np.cumsum(seg)])
        n = max(2, int(s[-1] / step))
        u = np.linspace(0, s[-1], n)
        return np.stack([np.interp(u, s, pts[:, 0]), np.interp(u, s, pts[:, 1])], 1), u / max(s[-1], 1e-6)

    def stroke(self, points, width=10.0, ink=0.8, water=0.6, fine=0.35,
               press=None, dry=0.0, side=None, seed=None, tint=None):
        """ink: total soot per unit (0..1+). water: how wet the brush is.
        fine: share of the soot that is fine particle. press(u)->0..1 width
        profile along the stroke. dry: 0 wet, 1 flying white — the brush
        skips in streaks along its direction, worse toward the end.
        side: brush laid on its side (cefeng). None is a round brush, evenly
        loaded. A number 0..1 is how much PALER the middle is than the
        edges — the tip dragged along one edge carries the most ink, the
        heel along the other a little less. A bamboo culm is this stroke.
        tint: a name in TINTS — the brush carries that pigment instead of
        soot. Returns the coverage, so a picture can know where it painted."""
        rng = np.random.default_rng(self.rng.integers(1 << 30) if seed is None else seed)
        P, U = self._walk(points, max(0.5, width * 0.06))
        tang = np.gradient(P, axis=0)
        tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-6
        norm = np.stack([-tang[:, 1], tang[:, 0]], 1)
        if press is None:
            press = lambda u: np.sin(np.clip(u, 0, 1) * np.pi) ** 0.35
        hairs = max(6, int(width * 0.9))
        # each hair runs out of ink at its own point when dry, and skips in runs
        runout = 1.0 - dry * rng.uniform(0.1, 1.0, hairs)
        skip = rng.random((hairs, 64)) < dry * 0.35
        cover = np.zeros((self.h, self.w), np.float32)
        for (x, y), u, nv in zip(P, U, norm):
            r = max(0.6, width * 0.5 * press(u))
            xa, xb = int(max(0, x - r - 2)), int(min(self.w, x + r + 3))
            ya, yb = int(max(0, y - r - 2)), int(min(self.h, y + r + 3))
            if xb <= xa or yb <= ya:
                continue
            Y, X = np.mgrid[ya:yb, xa:xb].astype(np.float32)
            d = np.hypot(X - x, Y - y) / r
            m = np.clip((1.0 - d) * 3.0, 0, 1)
            o = ((X - x) * nv[0] + (Y - y) * nv[1]) / r
            if side is not None:
                # tip edge (o=+1) full, heel edge (o=-1) a touch less, belly pale
                m = m * (1.0 - side * np.exp(-(o / 0.45) ** 2)) * (0.9 + 0.1 * o)
            if dry > 0:
                k = np.clip(((o + 1) * 0.5 * hairs).astype(int), 0, hairs - 1)
                live = (u <= runout) & ~skip[:, min(63, int(u * 63))]
                m = m * live[k]
            cover[ya:yb, xa:xb] = np.maximum(cover[ya:yb, xa:xb], m)
        load = ink * (1.0 - 0.45 * dry * np.clip(cover, 0, 1))
        if tint is not None:
            t = self.tints.setdefault(tint, np.zeros((self.h, self.w), np.float32))
            t += cover * load
        else:
            self.fine += cover * load * fine
            self.coarse += cover * load * (1 - fine)
        self.water += cover * water * (1.0 - 0.8 * dry)
        return cover

    def wash(self, alpha, ink=0.15, water=0.8, fine=0.7):
        """A broad thin wash laid wherever alpha says, reserving the rest.
        This is how snow gets into an ink painting: not painted — the sky is
        washed grey AROUND it and the bare paper is left to be the snow
        (烘托, 'setting off'). Wang Mian's ink plums do it to the petals."""
        a = np.clip(alpha, 0, None).astype(np.float32)
        self.fine += a * ink * fine
        self.coarse += a * ink * (1 - fine)
        self.water += a * water

    def lift(self, mask):
        """Take pigment back off where mask is 1 (0..1). Not a real-paper
        move — real snow is reserved from the start — but it lets a picture
        lay a branch first and then decide where the snow sits on it."""
        k = 1.0 - np.clip(mask, 0, 1)
        self.fine *= k; self.coarse *= k
        for t in self.tints.values():
            t *= k

    def spill(self, x, y, radius, water=1.0, ink=0.0, fine=0.6):
        """A drop of water (or thin ink) set down on its own."""
        Y, X = np.mgrid[0:self.h, 0:self.w]
        m = np.clip(1 - np.hypot(X - x, Y - y) / radius, 0, 1) ** 0.6
        self.water += m * water
        self.fine += m * ink * fine
        self.coarse += m * ink * (1 - fine)

    # ── the paper drinks it ──────────────────────────────────────────────────

    def flow(self, steps=60, spread=0.15, mobility_fine=0.9, mobility_coarse=0.18,
             absorb=0.012):
        """Water diffuses along the fibres and soaks away; particles move only
        where it is wet, the fine ones far more readily than the coarse."""
        c = self.conduct
        def diff(a, rate):
            return rate * (lb.blur(a[..., None], 2, passes=1)[..., 0] - a)
        for _ in range(steps):
            wet = np.clip(self.water * 2.5, 0, 1)
            k = spread * 4 * c * wet
            k = np.clip(k, 0, 0.95)
            self.water += np.clip(spread * 4 * c, 0, 0.95) * (lb.blur(self.water[..., None], 2, passes=1)[..., 0] - self.water)
            self.fine += diff(self.fine, mobility_fine * k)
            self.coarse += diff(self.coarse, mobility_coarse * k)
            for name, t in self.tints.items():
                t += diff(t, TINTS[name][1] * k)
            self.water = np.maximum(self.water - absorb * c, 0)
        self.water[:] = 0

    # ── what you see ────────────────────────────────────────────────────────

    def render(self, kind="ao", age=0.0):
        """kind: 'ao' blue-black or 'cha' brown-black. age 0..1: koboku —
        old glue lets the ink sink into the paper; the gradations widen."""
        tint = AO if kind == "ao" else CHA
        g = 1.0 - 0.35 * age          # older ink: dilute tones go further before they go black
        f = np.clip(self.fine, 0, None)[..., None] ** g
        co = np.clip(self.coarse, 0, None)[..., None]
        A = f * tint + co * COARSE
        for name, t in self.tints.items():
            A = A + np.clip(t, 0, None)[..., None] * TINTS[name][0]
        T = np.exp(-A * 1.25)
        img = self.paper[None, None, :] * T
        # paper texture: fibres catch a little light
        img *= (0.985 + 0.02 * np.clip(self.conduct - 0.6, -0.5, 0.8))[..., None]
        return img.astype(np.float32)
