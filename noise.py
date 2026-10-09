"""
noise — the irregularity under everything else.

Nearly every picture needs a field that varies without repeating: turf, stone,
glaze, a wash that isn't flat. For a while each painting carried its own copy
of the same thirty lines, and the copies drifted (one upscaled a grid and
showed blocks, one hashed differently, one only worked in pixel space). This is
the one copy.

Everything here is a pure function of coordinates. Pass pixel coordinates and
you get image-space texture; pass world coordinates (metres on a floor, a
ray-hit position) and the texture stays glued to the surface whatever the
camera does. That second use is the reason this is not a pass in lampblack.py:
a pass only ever sees pixels.

    import noise as nz
    yy, xx = nz.grid(W, H)
    turf = nz.fbm(xx, yy, scale=40, seed=3, octaves=4)          # 0..1
    f1, f2, ids = nz.worley(xx, yy, cell=60, seed=7)            # cells
    crack = np.exp(-(f2 - f1) / 1.5)                            # their edges
    tint = nz.cellrand(ids, salt=1)                             # one value per cell

All outputs are float32. Seeds are small ints. Same inputs, same picture.
"""

import numpy as np


def grid(width, height, ss=1):
    """(yy, xx) pixel-centre coordinates, optionally supersampled ss times."""
    yy, xx = np.mgrid[0:height * ss, 0:width * ss].astype(np.float32)
    return (yy + 0.5) / ss, (xx + 0.5) / ss


def hash2(i, j, seed):
    """A float in [0, 1) for each integer lattice point (i, j)."""
    i = np.asarray(i).astype(np.int64)
    j = np.asarray(j).astype(np.int64)
    h = (i * 73856093) ^ (j * 19349663) ^ (int(seed) * 83492791)
    h = (h ^ (h >> 13)) * 1274126177
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF).astype(np.float32) / float(0xFFFFFF)


def vn(x, y, scale, seed):
    """Value noise, smoothstep-interpolated, 0..1. `scale` is the feature size
    in the same units as x and y."""
    u = np.asarray(x, np.float32) / scale
    v = np.asarray(y, np.float32) / scale
    i, j = np.floor(u), np.floor(v)
    fu, fv = u - i, v - j
    fu = fu * fu * (3 - 2 * fu)
    fv = fv * fv * (3 - 2 * fv)
    a = hash2(i, j, seed)
    b = hash2(i + 1, j, seed)
    c = hash2(i, j + 1, seed)
    d = hash2(i + 1, j + 1, seed)
    return (a + (b - a) * fu) * (1 - fv) + (c + (d - c) * fu) * fv


# Value noise lives on a square lattice, and stacking octaves on the SAME
# lattice lines all their creases up: the result shows horizontals and
# verticals that nothing in the picture asked for (worst in ridged()). Turning
# each octave by an angle that never comes round again hides the lattice.
_TURN = 2.39996  # the golden angle, radians


def _turn(x, y, o):
    if o == 0:
        return x, y
    a = _TURN * o
    ca, sa = np.cos(a), np.sin(a)
    return x * ca - y * sa, x * sa + y * ca


def fbm(x, y, scale, seed, octaves=3, gain=0.5, lacunarity=2.0, rotate=True):
    """Octaves of vn, each finer and quieter, normalised back to 0..1.
    gain near 0.5 is soft (stone, cloud); higher is rougher (bark, rust).
    rotate=False reproduces the unrotated copy older paintings carry."""
    acc, amp, tot, s = 0.0, 1.0, 0.0, float(scale)
    for o in range(octaves):
        u, v = _turn(x, y, o) if rotate else (x, y)
        acc = acc + amp * vn(u, v, s, seed + 17 * o)
        tot += amp
        amp *= gain
        s /= lacunarity
    return (acc / tot).astype(np.float32)


def ridged(x, y, scale, seed, octaves=4, gain=0.5):
    """1 - |2n - 1| per octave: thin bright ridges where the noise crosses its
    middle. Veins in marble, crests, the creases in a sheet."""
    acc, amp, tot, s = 0.0, 1.0, 0.0, float(scale)
    for o in range(octaves):
        u, v = _turn(x, y, o + 1)
        n = vn(u, v, s, seed + 31 * o)
        acc = acc + amp * (1.0 - np.abs(2.0 * n - 1.0)) ** 2
        tot += amp
        amp *= gain
        s /= 2.0
    return (acc / tot).astype(np.float32)


def field(width, height, scale, seed, octaves=3, gain=0.5):
    """fbm over the whole image in pixel space: the common case, in one call."""
    yy, xx = grid(width, height)
    return fbm(xx, yy, scale, seed, octaves, gain)


def warp(x, y, amount, scale, seed, octaves=2):
    """Push the coordinates around by a noise field before sampling something
    else with them. Anything regular (a grid of setts, a row of planks, a
    Worley net) stops looking machined once its coordinates are warped by a
    few percent of its own size."""
    dx = fbm(x, y, scale, seed, octaves) - 0.5
    dy = fbm(x, y, scale, seed + 101, octaves) - 0.5
    return x + 2 * amount * dx, y + 2 * amount * dy


def streak(x, y, angle, along, across, seed, octaves=3, gain=0.5):
    """Oriented noise: features `along` long and `across` wide, lying at `angle`
    (radians, 0 = pointing along +x). Wood grain, brushed metal, wind on a
    field, rain-streaks down glass, the direction a floor gets scrubbed."""
    ca, sa = np.cos(angle), np.sin(angle)
    u = x * ca + y * sa
    v = -x * sa + y * ca
    return fbm(u * (across / along), v, across, seed, octaves, gain, rotate=False)


def worley(x, y, cell, seed, jitter=0.9, spread=0.0):
    """Cellular noise on a jittered grid of feature points.

    Returns (f1, f2, ids): distance to the nearest point, to the second
    nearest, and an int id for the nearest point's cell. f2 - f1 is ~0 along
    the borders between cells, so exp(-(f2 - f1) / w) draws the net: cracks,
    cobbles, caustics, scales, dried mud.

    spread > 0 makes it ADDITIVELY WEIGHTED: each point's distance has a
    per-point amount (up to spread * cell) taken off it, so some cells grow
    at their neighbours' expense and the borders turn into curves. Uniform
    cells read as a pattern; uneven ones read as stones somebody laid."""
    x = np.asarray(x, np.float32)
    y = np.asarray(y, np.float32)
    gx = np.floor(x / cell).astype(np.int64)
    gy = np.floor(y / cell).astype(np.int64)
    f1 = np.full(x.shape, 1e9, np.float32)
    f2 = f1.copy()
    ids = np.zeros(x.shape, np.int64)
    # spread can let a point two cells away win, so look further when it's on
    reach = 1 if spread <= 0.25 else 2
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            cx, cy = gx + dx, gy + dy
            jx = (hash2(cx, cy, seed) - 0.5) * jitter + 0.5
            jy = (hash2(cx, cy, seed + 7) - 0.5) * jitter + 0.5
            d = np.hypot(x - (cx + jx) * cell, y - (cy + jy) * cell).astype(np.float32)
            if spread > 0:
                d = d - hash2(cx, cy, seed + 13) * spread * cell
            cid = (cx * 0x1F1F1F1F) ^ (cy * 0x2E2E2E2F) ^ int(seed)
            closer = d < f1
            f2 = np.where(closer, f1, np.minimum(f2, d))
            ids = np.where(closer, cid, ids)
            f1 = np.where(closer, d, f1)
    return f1, f2, ids


def cellrand(ids, salt=0):
    """A float in [0, 1) per cell id (or per any int label): one tint, one
    tilt, one height per stone. Different salts give independent values."""
    h = (np.asarray(ids, np.int64) * 2654435761 + int(salt) * 40503) & 0xFFFFFFFF
    h = (h ^ (h >> 15)) * 2246822519 & 0xFFFFFFFF
    h = h ^ (h >> 13)
    return ((h & 0xFFFFFF).astype(np.float32) / float(0xFFFFFF))
