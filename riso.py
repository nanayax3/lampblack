"""
riso — a stencil duplicator, simulated rather than imitated.

A risograph is not a printer. It is a photocopier's angrier cousin: for each
colour it burns a stencil onto a master sheet, wraps it round a drum loaded
with one soy ink, and rolls the paper past it. One drum, one colour, one pass.
A two-colour print goes through the machine twice, and the paper is grabbed by
rubber rollers each time, so the second colour lands a millimetre or two off
the first. That gap is the whole look.

Everything people reach for as "the riso filter" is downstream of four
mechanical facts, and this module implements those instead of the look:

  1. SEPARATION. The machine has no idea what a colour is. You hand it one
     greyscale per drum. Getting from a painting to N greyscales is a
     least-squares fit in optical density — Beer-Lambert, density roughly
     proportional to coverage — which is the same arithmetic a real separation
     does, clipped because a drum cannot print -14% ink.

  2. SCREENING. Continuous tone is impossible; the stencil is a field of holes.
     So each drum gets an amplitude-modulated dot screen at its own angle
     (15/45/75, the classic rosette-avoiding set). Coarse — riso screens run
     around 34-106 lpi where litho runs 150-300, which is why you can see the
     dots from across a room.

  3. MISREGISTRATION. Per pass: a translation, a small rotation, and a slight
     scale, because the sheet is fed by friction. Applied AFTER screening,
     because the screen belongs to the drum and the error belongs to the paper.

  4. INK THAT NEVER DRIES. Soy ink sets by soaking into uncoated stock, so it
     stays translucent and overlaps MULTIPLY rather than compositing. It also
     goes down unevenly — the drum is a cylinder, so the unevenness is banded
     in the print direction — and it smudges under the next sheet.

The paper is not white and is not flat, and both of those show through
everywhere, because the ink is transparent and there is not much of it.

Ink hexes below are the published Riso spot colours, eyeballed into sRGB; they
are approximations of an ink on a particular stock, not measurements.

Pure NumPy. Takes and returns (H, W, 3) float arrays in [0, 1], like every
other pass in here.
"""

import numpy as np

# ── the drums ────────────────────────────────────────────────────────────────

INKS = {
    "black":          (0.00, 0.00, 0.00),
    "fluoro_pink":    (1.00, 0.28, 0.69),
    "blue":           (0.00, 0.47, 0.75),
    "federal_blue":   (0.24, 0.33, 0.53),
    "aqua":           (0.37, 0.78, 0.90),
    "teal":           (0.00, 0.51, 0.54),
    "green":          (0.00, 0.66, 0.36),
    "kelly_green":    (0.40, 0.70, 0.27),
    "yellow":         (1.00, 0.91, 0.00),
    "sunflower":      (1.00, 0.71, 0.07),
    "orange":         (1.00, 0.42, 0.18),
    "red":            (1.00, 0.40, 0.37),
    "crimson":        (0.89, 0.36, 0.31),
    "burgundy":       (0.57, 0.31, 0.45),
    "purple":         (0.46, 0.36, 0.65),
}

# Cheap uncoated stock. Not white — nothing you can feed a riso is white.
PAPER = (0.955, 0.941, 0.906)

SCREEN_ANGLES = (15.0, 75.0, 45.0, 0.0, 30.0, 60.0)


# ── separation ───────────────────────────────────────────────────────────────

def _density(rgb, floor=1e-3):
    """Optical density: how much light this colour has eaten."""
    return -np.log(np.clip(np.asarray(rgb, np.float32), floor, 1.0))


def separate(img, inks, paper=PAPER, strength=1.0, refine=2):
    """
    RGB painting -> one coverage map per drum, each in [0, 1].

    Least squares in density space, then clipped, then the residual is refitted
    onto whichever drums still have headroom. That second part matters: a naive
    clip loses every shadow the moment one ink saturates, and the refit is what
    a human doing a separation by eye does without calling it that.
    """
    img = np.clip(np.asarray(img, np.float32), 1e-4, 1.0)
    P = np.asarray(paper, np.float32)

    target = _density(img) - _density(P)          # (H, W, 3), >= 0 mostly
    target = np.maximum(target, 0.0) * float(strength)

    # Column i is what one full unit of drum i does to each channel.
    M = np.stack([_density(INKS[k] if isinstance(k, str) else k) for k in inks], 1)  # (3, N)
    pinv = np.linalg.pinv(M)                                                         # (N, 3)

    cov = np.clip(target @ pinv.T, 0.0, 1.0)

    for _ in range(int(refine)):
        resid = target - cov @ M.T
        head = 1.0 - cov                      # how much each drum has left
        step = np.clip(resid @ pinv.T, -cov, head)
        cov = np.clip(cov + step * 0.6, 0.0, 1.0)

    return [cov[:, :, i] for i in range(cov.shape[2])]


# ── screening ────────────────────────────────────────────────────────────────

def _coords(h, w):
    yy = np.arange(h, dtype=np.float32)[:, None]
    xx = np.arange(w, dtype=np.float32)[None, :]
    return yy, xx


def _lowfreq(h, w, scale, seed):
    """Smooth [0,1] noise by upsampling a small grid — cheap, and the softness
    is the point: paper does not stretch in sharp steps."""
    rng = np.random.default_rng(seed)
    gh, gw = max(2, int(h / scale)), max(2, int(w / scale))
    g = rng.random((gh, gw)).astype(np.float32)
    yi = np.linspace(0, gh - 1, h, dtype=np.float32)
    xi = np.linspace(0, gw - 1, w, dtype=np.float32)
    y0 = np.floor(yi).astype(int); y1 = np.minimum(y0 + 1, gh - 1); fy = (yi - y0)[:, None]
    x0 = np.floor(xi).astype(int); x1 = np.minimum(x0 + 1, gw - 1); fx = (xi - x0)[None, :]
    a = g[np.ix_(y0, x0)] * (1 - fx) + g[np.ix_(y0, x1)] * fx
    b = g[np.ix_(y1, x0)] * (1 - fx) + g[np.ix_(y1, x1)] * fx
    return a * (1 - fy) + b * fy


def screen(cov, period=6.0, angle=45.0, softness=0.28, wobble=0.6, seed=0):
    """
    Coverage -> dots. Classic AM round-dot spot function on a rotated grid:
    the dot grows circular, goes checkerboard at 50%, then the HOLES shrink.

    `softness` is ink soaking into the fibre — a hard-edged dot is a laser
    printer, not a stencil. `wobble` warps the grid with low-frequency noise,
    because the master is wrapped round a drum by hand-tight tension.
    """
    h, w = cov.shape
    yy, xx = _coords(h, w)
    t = np.radians(angle)

    if wobble > 0:
        wx = (_lowfreq(h, w, 90.0, seed * 7 + 1) - 0.5) * (period * wobble)
        wy = (_lowfreq(h, w, 90.0, seed * 7 + 2) - 0.5) * (period * wobble)
    else:
        wx = wy = 0.0

    u = ((xx + wx) * np.cos(t) + (yy + wy) * np.sin(t)) * (2 * np.pi / period)
    v = (-(xx + wx) * np.sin(t) + (yy + wy) * np.cos(t)) * (2 * np.pi / period)
    spot = (np.cos(u) + np.cos(v)) * 0.25 + 0.5      # [0, 1]

    return np.clip((cov - spot) / max(softness, 1e-3) + 0.5, 0.0, 1.0).astype(np.float32)


# ── the paper moving ─────────────────────────────────────────────────────────

def _warp(a, dx=0.0, dy=0.0, rot=0.0, scale=1.0):
    """Bilinear affine resample about the centre. Edges clamp — the sheet does
    not wrap around, it just runs out."""
    h, w = a.shape
    yy, xx = _coords(h, w)
    cy, cx = (h - 1) * 0.5, (w - 1) * 0.5
    t = np.radians(rot)
    X = (xx - cx - dx) / scale
    Y = (yy - cy - dy) / scale
    sx = X * np.cos(t) + Y * np.sin(t) + cx
    sy = -X * np.sin(t) + Y * np.cos(t) + cy

    x0 = np.clip(np.floor(sx), 0, w - 1).astype(int); x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.clip(np.floor(sy), 0, h - 1).astype(int); y1 = np.clip(y0 + 1, 0, h - 1)
    fx = np.clip(sx - x0, 0, 1); fy = np.clip(sy - y0, 0, 1)
    top = a[y0, x0] * (1 - fx) + a[y0, x1] * fx
    bot = a[y1, x0] * (1 - fx) + a[y1, x1] * fx
    return (top * (1 - fy) + bot * fy).astype(np.float32)


def ink_unevenness(h, w, amount=0.18, banding=0.10, seed=0):
    """
    The drum is a cylinder rotating one way, so the ink it fails to lay down
    fails in STRIPES along the print direction. Plus broad blotchiness from the
    ink sitting unevenly in the mesh.
    """
    blotch = _lowfreq(h, w, 140.0, seed * 13 + 3)
    yy, _ = _coords(h, w)
    rng = np.random.default_rng(seed * 13 + 4)
    phase = rng.random(6).astype(np.float32) * 6.283
    per = rng.uniform(9.0, 140.0, 6).astype(np.float32)
    band = sum(np.sin(yy * (2 * np.pi / p) + ph) for p, ph in zip(per, phase)) / 6.0
    f = 1.0 - amount * (1.0 - blotch) + banding * band
    return np.clip(f, 0.0, 1.4).astype(np.float32)


def paper(h, w, colour=PAPER, fibre=0.035, seed=0):
    """Uncoated stock: warm, and never flat."""
    rng = np.random.default_rng(seed + 99)
    n = rng.random((h, w)).astype(np.float32)
    n = (n + np.roll(n, 1, 0) + np.roll(n, 1, 1) + np.roll(n, -1, 1)) * 0.25
    n = n * 0.55 + _lowfreq(h, w, 26.0, seed + 5) * 0.45
    base = np.asarray(colour, np.float32)[None, None, :]
    return np.clip(base * (1.0 + fibre * (n - 0.5)[:, :, None] * 2.0), 0, 1)


# ── the whole run ────────────────────────────────────────────────────────────

def print_run(img, inks, period=6.0, angles=None, registration=2.2, softness=0.28,
              wobble=0.6, unevenness=0.18, banding=0.10, density=1.05,
              strength=1.0, stock=PAPER, fibre=0.035, seed=0, smudge=0.0,
              offsets=None, return_layers=False):
    """
    Painting -> N-colour riso print.

    `registration` is the misregistration budget in pixels — how far the sheet
    can be off. Zero is a machine that does not exist. Two to four is a good
    day. Above eight it reads as a mistake rather than a process, which is
    itself worth knowing.

    `density` scales how hard each drum lays ink down; riso over-inks easily
    and the paper goes soggy, so above ~1.4 it stops looking like paper.

    `offsets` overrides the dice with a deliberate (dx, dy) or (dx, dy, rot)
    per drum. Use it when the gap is the subject rather than the flaw: anything
    painted neutral sits in every drum at once, so ONE mark in the artwork
    arrives on the sheet as N marks, spread by exactly these numbers. That is
    not a copy of the effect — it is the effect, made the way the machine makes
    it.
    """
    img = np.asarray(img, np.float32)
    h, w = img.shape[:2]
    inks = list(inks)
    angles = list(angles) if angles else [SCREEN_ANGLES[i % len(SCREEN_ANGLES)]
                                          for i in range(len(inks))]
    rng = np.random.default_rng(seed)

    covs = separate(img, inks, paper=stock, strength=strength)
    sheet = paper(h, w, stock, fibre, seed)
    layers = []

    for i, (name, cov) in enumerate(zip(inks, covs)):
        k = np.asarray(INKS[name] if isinstance(name, str) else name, np.float32)

        cov = cov * ink_unevenness(h, w, unevenness, banding, seed + i)
        a = screen(np.clip(cov, 0, 1), period, angles[i], softness, wobble, seed + i)

        # The first drum is the reference; everything after it is the error.
        if offsets is not None:
            o = list(offsets[i]) if i < len(offsets) else [0.0, 0.0]
            dx, dy = float(o[0]), float(o[1])
            rot = float(o[2]) if len(o) > 2 else 0.0
            sc = float(o[3]) if len(o) > 3 else 1.0
            if dx or dy or rot or sc != 1.0:
                a = _warp(a, dx, dy, rot, sc)
        elif i > 0 and registration > 0:
            dx = rng.uniform(-1, 1) * registration
            dy = rng.uniform(-1, 1) * registration
            rot = rng.uniform(-1, 1) * registration * 0.045
            sc = 1.0 + rng.uniform(-1, 1) * registration * 0.0004
            a = _warp(a, dx, dy, rot, sc)

        if smudge > 0:
            a = np.maximum(a, _warp(a, 0.0, smudge * 6.0) * smudge)

        a = np.clip(a * density, 0.0, 1.0)
        layers.append((name, a))
        # Translucent ink on paper: multiply, never composite.
        sheet = sheet * ((1.0 - a[:, :, None]) + a[:, :, None] * k[None, None, :])

    sheet = np.clip(sheet, 0.0, 1.0)
    return (sheet, layers) if return_layers else sheet


def coverage_report(layers):
    """What the machine would tell you if it could. Total ink over ~2.2 means
    a damp sheet and set-off on the back of the next one."""
    lines, total = [], 0.0
    for name, a in layers:
        m = float(a.mean())
        total += m
        lines.append(f"  {str(name):<14} {m*100:5.1f}%")
    lines.append(f"  {'TOTAL':<14} {total*100:5.1f}%"
                 + ("   <- soggy" if total > 2.2 else ""))
    return "\n".join(lines)
