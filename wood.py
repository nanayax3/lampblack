"""
wood — boards, as two fields: an albedo and a height.

Promoted out of two paintings that each grew their own: a weathered fence
post (grey, eroded, cracked along the grain) and an oiled oak table (planks
of different widths, each from its own log, silver rays). Same machinery
both times, so here it is once.

How a board is built:

  * A GRAIN COORDINATE. Everything follows one warped "across" coordinate,
    gx. Rings are cos() of it, fine figure is noise streaked along it,
    cracks are narrow bands of it. Warp it and the whole board bends
    together, which is what wood does.
  * KNOTS ARE OBSTACLES IN A FLOW. A branch was there first and the trunk
    grew round it, so grain lines around a knot are streamlines past a
    cylinder: psi = x * (1 - R^2 / d^2). Using that as the grain coordinate
    makes the rings part and close around the knot for free. Inside R the
    knot shows its own end grain: tight concentric rings, darker, a check
    or two. Knots are stretched along the board (they meet the saw at an
    angle).
  * PLANKS ARE SEPARATE LOGS. Each plank gets its own ring spacing, offset
    and tone, and a dark seam between them.
  * LATEWOOD STANDS PROUD when the softer earlywood weathers away, so ring
    goes into height as well as colour. On oiled wood keep the height low.

Coordinates are the caller's job. Pass X (across the grain) and Z (along it)
in board units, already put through whatever perspective you want; for a
flat, face-on board just pass the pixel grid (X = xx, Z = yy for vertical
grain). The albedo then wants lighting: lb.relight(alb, height, ...).

    alb, h = wood.board(xx, yy, "oak", seed=3, planks=[300, 620],
                        knots=[(450, 900, 30)])
    img = lb.relight(alb, h, direction=(-0.6, -0.5), strength=0.5)

Species are only presets for colour and figure; every number can be
overridden by keyword.
"""

import numpy as np

import noise


def _sstep(a, b, x):
    u = np.clip((x - a) / (b - a), 0, 1)
    return u * u * (3 - 2 * u)


SPECIES = {
    # light, dark: albedo at earlywood and latewood. ring: mean ring spacing.
    # sharp: how narrow the latewood band is. rays: oak's silver medullary rays.
    "oak": dict(light=(0.60, 0.42, 0.25), dark=(0.36, 0.23, 0.12), ring=52,
                sharp=3.0, rays=0.08, fine=0.45, blotch=0.6, relief=4.0),
    "pine": dict(light=(0.80, 0.62, 0.40), dark=(0.55, 0.30, 0.14), ring=40,
                 sharp=4.5, rays=0.0, fine=0.25, blotch=0.35, relief=5.0),
    "walnut": dict(light=(0.42, 0.28, 0.18), dark=(0.20, 0.12, 0.08), ring=60,
                   sharp=2.2, rays=0.0, fine=0.5, blotch=0.7, relief=3.0),
    # the fence post: sun and rain have taken the colour and most of the earlywood
    "weathered": dict(light=(0.74, 0.71, 0.67), dark=(0.20, 0.165, 0.14), ring=96,
                      sharp=2.2, rays=0.0, fine=0.75, blotch=0.7, relief=8.0,
                      checks=5),
}


def board(X, Z, species="oak", seed=0, planks=None, knots=(), wander=1.0,
          cut="quarter", taper=2.0, **override):
    """Albedo (H, W, 3) and height (H, W) for wood whose grain runs along Z.

    planks: across-grain positions of the seams, in X units (None = one board).
    knots:  (x, z, radius) in X/Z units; grain flows round them.
    wander: how much the grain swims (0 = ruler-straight, 1 = a normal board).
    cut:    "quarter" = straight stripes; "flat" = cathedrals. A flat-sawn face
            is a plane cutting a tapering stack of cones, so a ring sits where
            sqrt(x^2 + depth^2) is constant, and depth grows along the board
            with the taper: the rings come out as nested arches pointing
            towards +Z. Per plank, its own pith and depth.
    taper:  (flat cut) how many rings deeper the face reaches from one end of
            Z to the other. Low = long narrow arches (a real board is about
            1-3); high = a garland of short ones.
    override: any SPECIES key, plus checks=N for long drying cracks.
    """
    p = dict(SPECIES[species])
    p.update(override)
    X = np.asarray(X, np.float32)
    Z = np.asarray(Z, np.float32)
    s = seed * 101
    light = np.array(p["light"], np.float32)
    dark = np.array(p["dark"], np.float32)

    # ── planks: each its own log ────────────────────────────────────────────
    if planks:
        edges = np.sort(np.asarray(planks, np.float32))
        pid = np.searchsorted(edges, X).astype(np.int32)
        rng = np.random.default_rng(seed + 711)
        n = len(edges) + 1
        spacing = (p["ring"] * rng.uniform(0.55, 1.4, n)).astype(np.float32)[pid]
        offset = rng.uniform(0, 400, n).astype(np.float32)[pid]
        tone = rng.uniform(-0.10, 0.10, n).astype(np.float32)[pid]
        dseam = np.min(np.abs(X[..., None] - edges), -1)
        lo = np.concatenate([[X.min()], edges])[pid]
        hi = np.concatenate([edges, [X.max()]])[pid]
        pith = lo + (hi - lo) * rng.uniform(0.3, 0.7, n).astype(np.float32)[pid]
        depth0 = rng.uniform(2.0, 5.0, n).astype(np.float32)[pid]
    else:
        spacing, offset, tone = np.float32(p["ring"]), np.float32(0), np.float32(0)
        dseam = None
        pith = np.float32((X.min() + X.max()) / 2)
        depth0 = np.float32(3.0)

    # ── the grain coordinate, bent round the knots ──────────────────────────
    gx = X + wander * (60 * (noise.fbm(X * 0.25, Z * 0.03, 220, s + 1) - 0.5)
                       + 14 * (noise.fbm(X, Z * 0.1, 50, s + 2) - 0.5))
    inside = np.zeros(X.shape, np.float32)       # 1 inside a knot
    kring = np.zeros(X.shape, np.float32)         # the knot's own end grain
    for i, (kx, kz, r) in enumerate(knots):
        dx = gx - kx
        dz = (Z - kz) / 2.6                       # stretched along the board
        d2 = dx * dx + dz * dz
        R2 = r * r
        # streamline past a cylinder, faded out a few radii away so it stays local
        fade = np.exp(-d2 / (R2 * 30))
        gx = gx - dx * np.minimum(R2 / np.maximum(d2, 1e-3), 1.0) * fade
        d = np.sqrt(d2) / r
        wob = 1 + 0.15 * (noise.fbm(X, Z, r * 0.8, s + 40 + i) - 0.5)
        k = _sstep(1.05, 0.9, d * wob)
        inside = np.maximum(inside, k)
        kring = np.maximum(kring, k * (0.5 + 0.5 * np.cos(2 * np.pi * d * 5.5)) ** 2)

    if cut == "flat":
        # depth of the face below the bark, in rings, growing along +Z (the
        # taper), so the arches close and point towards +Z. Pass -Z to flip.
        zspan = float(Z.max() - Z.min()) or 1.0
        depth = spacing * (depth0 + taper * (Z - Z.min()) / zspan)
        radial = np.sqrt((gx - pith) ** 2 + depth ** 2)
        phase = (radial + offset) / spacing + 1.2 * wander * noise.fbm(gx, Z * 0.015, 260, s + 3, octaves=4)
    else:
        phase = (gx + offset) / spacing + 3.2 * wander * noise.fbm(gx + offset, Z * 0.015, 260, s + 3, octaves=4)
    # a flat face cuts the rings obliquely, so near an arch's apex one ring
    # smears across a wide band: sharpen it, or oak goes to fish scales
    sharp = p["sharp"] * (1.8 if cut == "flat" else 1.0)
    ring = (0.5 + 0.5 * np.cos(2 * np.pi * phase)) ** sharp
    fine = noise.streak(gx * 3, Z, np.pi / 2, 180, 3.0, s + 4, octaves=3)
    mid = noise.streak(gx, Z, np.pi / 2, 420, 13, s + 5, octaves=3)
    blot = noise.fbm(X * 0.5, Z * 0.2, 200, s + 6, octaves=3)

    t = np.clip(0.62 - 0.55 * ring + p["fine"] * (fine - 0.5) + p["blotch"] * (blot - 0.5) + tone, 0, 1)
    alb = dark + (light - dark) * t[..., None] ** 0.9
    if p["rays"]:
        ray = _sstep(0.80, 0.88, noise.streak(gx * 2, Z * 1.5, np.pi / 2, 30, 6, s + 7, octaves=2))
        alb = alb * (1 + p["rays"] * ray[..., None])
    height = p["relief"] * (ring + 0.8 * mid + 0.4 * fine)

    # ── knots: darker, resinous, their own rings, sitting a hair proud ──────
    if knots:
        kcol = dark * 0.62
        kt = (0.55 + 0.45 * kring)[..., None]
        alb = alb + (kcol * (1.2 - 0.5 * kt) - alb) * inside[..., None] * 0.92
        # a dark halo where the trunk grain crowds against the branch
        halo = np.zeros(X.shape, np.float32)
        for kx, kz, r in knots:
            d = np.sqrt((X - kx) ** 2 + ((Z - kz) / 2.6) ** 2) / r
            halo = np.maximum(halo, np.exp(-((d - 1.05) / 0.12) ** 2))
        alb = alb * (1 - 0.18 * halo)[..., None]
        height = height * (1 - inside) + (p["relief"] * 1.2 + 2.5 * kring) * inside

    # ── checks: long drying cracks along the grain ──────────────────────────
    n_checks = int(p.get("checks", 0))
    if n_checks:
        rng = np.random.default_rng(seed + 19)
        x0, x1 = float(X.min()), float(X.max())
        z0, z1 = float(Z.min()), float(Z.max())
        crack = np.zeros(X.shape, np.float32)
        for _ in range(n_checks):
            g = rng.uniform(x0, x1)
            a = rng.uniform(z0 - 0.1 * (z1 - z0), z1)
            b = a + rng.uniform(0.2, 0.9) * (z1 - z0)
            w = rng.uniform(4, 11)
            u = np.clip((Z - a) / (b - a), 0, 1)
            env = np.maximum(np.sin(np.pi * u), 0) ** 0.6
            wob = 1 + 0.5 * (noise.fbm(X * 0 + g, Z, 60, s + 8) - 0.5)
            crack = np.maximum(crack, _sstep(1.0, 0.25, np.abs(gx - g) / np.maximum(w * env * wob, 0.3)) * (env > 0.01))
        height = height - 26 * crack
        alb = alb + (np.float32(0.03) - alb) * crack[..., None] * 0.96

    # ── seams between planks ────────────────────────────────────────────────
    if dseam is not None:
        gap = _sstep(2.2, 0.4, dseam)
        alb = alb * (1 - 0.6 * gap)[..., None]
        height = height - 6 * gap

    return np.clip(alb, 0, 1).astype(np.float32), height.astype(np.float32)
