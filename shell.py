"""
shell.py — coiled shells as real 3D surfaces, rendered onto a canvas.

A painted spiral on a dome reads as "a marble with a spiral stamped on it".
A real shell is one tube that grows as it coils: each whorl is a scaled copy
of the one before, wrapped round an axis and slid down it, so the whorls
overlap, hide each other, and the last one ends in an opening.

This is Raup's model (1966): a generating curve (the aperture, an ellipse)
swept along a logarithmic helico-spiral. Four numbers make most shells:

    W  whorl expansion: how much bigger each turn is than the last (snail ~2)
    T  translation: how far each turn slides down the axis (0 = flat coil,
       like a ramshorn; ~1.5 = a garden snail; 4+ = a tower shell)
    D  distance of the tube from the axis, as a fraction (0 = whorls touch
       the axis, no hole in the middle)
    S  aperture shape: height / width of the opening

The surface is sampled densely, rotated by a camera, and splatted into a
z-buffer. Hidden whorls get hidden for free; looking into the aperture shows
the inside of the tube, flagged as back-facing so it can be painted dark.

    from shell import Shell
    s = Shell(W=2.0, T=1.4, D=0.05, S=1.15, whorls=4.4, dextral=True)
    out = s.render((H, W), centre=(cx, cy), size=150, yaw=.., pitch=.., roll=..)
    out["mask"]     coverage 0..1 (soft edge)
    out["normal"]   (H, W, 3) unit normals in screen space, z towards viewer
    out["depth"]    larger = nearer, nan where empty
    out["theta"]    growth angle (radians from the apex): growth lines, bands
    out["phi"]      angle round the tube (0 = outermost point of the whorl)
    out["grown"]    0 at the apex, 1 at the lip
    out["inside"]   1 where you see the inner wall (through the aperture)
    out["lip"]      0..1 nearness to the aperture rim

Shading is left to the painting: shade(out, light) gives a plain lambert +
spec if you want a start. Points are splatted, not rasterised as triangles,
so samples are chosen from the projected size; if you see pinholes, raise
`density`.
"""
import numpy as np


def _rot(yaw, pitch, roll):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])     # turn about screen-up
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])     # tip towards/away
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])     # spin in the picture
    return Rz @ Rx @ Ry


class Shell:
    def __init__(self, W=2.0, T=1.4, D=0.05, S=1.15, whorls=4.4, dextral=True,
                 lip_flare=0.10, lip_len=0.35, sutures=0.06):
        self.W, self.T, self.D, self.S = W, T, D, S
        self.whorls = whorls
        self.dextral = dextral
        self.lip_flare = lip_flare        # how much the rim turns out (reflected lip)
        self.lip_len = lip_len            # radians over which it flares
        self.sutures = sutures            # how deep the groove between whorls dips

    def surface(self, n_theta, n_phi):
        """Points P (n_theta, n_phi, 3) in shell space: axis = +z towards the apex,
        unit = radius of the last whorl's tube centre."""
        k = np.log(self.W) / (2 * np.pi)
        tmax = self.whorls * 2 * np.pi
        # uniform in ARC LENGTH, not angle: the coil grows as exp(k*theta), so even
        # steps in theta spend almost every sample on the apex and starve the lip.
        u = np.linspace(0, 1, n_theta)
        e0, e1 = 1.0, np.exp(k * tmax)
        th_arc = np.log(e0 + (e1 - e0) * u) / k
        th = np.sort(np.concatenate([th_arc, np.linspace(0, tmax, n_theta // 8)]))
        ph = np.linspace(-np.pi, np.pi, n_phi, endpoint=False)
        g = np.exp(k * (th - tmax))                     # scale: 1 at the lip, tiny at the apex
        # tube radius from D: a tube whose inner edge sits at D * (outer edge)
        a = g * (1 - self.D) / (1 + self.D)              # aperture half-width
        rc = g                                           # tube centre from the axis: inner edge = D * outer edge
        zc = -self.T * g                                 # each turn slides down the axis
        # lip: the last stretch flares out
        f = np.clip((th - (tmax - self.lip_len)) / self.lip_len, 0, 1)
        flare = 1 + self.lip_flare * f ** 2
        TH, PH = np.meshgrid(th, ph, indexing="ij")
        A = (a * flare)[:, None]
        # aperture ellipse in the (radial, axial) plane; S = height/width
        rad = rc[:, None] + A * np.cos(PH)
        ax = zc[:, None] + A * self.S * np.sin(PH)
        # sutures: where whorls meet (inner side, phi near ±pi/2..pi), dent the tube a touch
        dent = 1 - self.sutures * np.exp(-((np.abs(PH) - 2.2) / 0.35) ** 2)
        rad = rc[:, None] + (rad - rc[:, None]) * dent
        sgn = 1.0 if self.dextral else -1.0             # dextral ON SCREEN: screen y points down, which
        # mirrors the world, so the coil that looks clockwise from the apex in the picture is +theta here
        x = rad * np.cos(sgn * TH)
        y = rad * np.sin(sgn * TH)
        P = np.stack([x, y, ax], -1)
        # centre it: middle of the bounding box at the origin
        lo, hi = P.reshape(-1, 3).min(0), P.reshape(-1, 3).max(0)
        P = P - (lo + hi) / 2
        span = (hi - lo).max()
        return P / span, TH, PH, th, tmax

    def render(self, shape, centre, size, yaw=0.0, pitch=-0.35, roll=0.0, density=1.0,
               aa=2):
        """Splat the shell into an (H, W) frame. size = pixels across its longest side.
        Default camera: axis up the screen, apex at top, tipped a little towards us."""
        H, Wd = shape
        # sample count from projected size: tube circumference and coil length at the lip
        n_phi = int(max(64, np.pi * size * 0.75 * density * aa))
        # the outer edge of the coil is ~ (1+a/rc) * exp(k theta); its total length
        # over the shell, in units of the biggest whorl, is about 2*(1 + 1/k)
        k = np.log(self.W) / (2 * np.pi)
        arc = 2.0 * (1 - np.exp(-k * self.whorls * 2 * np.pi)) / k
        n_theta = int(max(256, arc * 0.5 * size * 2.0 * density * aa))
        P, TH, PH, th, tmax = self.surface(n_theta, n_phi)
        # normals from the grid
        dT = np.gradient(P, axis=0)
        dP = np.gradient(P, axis=1)
        N = np.cross(dT, dP)
        N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-12
        if not self.dextral:      # the mirror (sgn) turns the grid inside out
            N = -N
        # shell space: z = axis (apex). screen: x right, y down, z towards viewer.
        # put the axis on screen-up first, then the user's camera.
        base = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)   # z(apex) -> -y (up)
        R = _rot(yaw, pitch, roll) @ base
        Q = P.reshape(-1, 3) @ R.T
        Nq = N.reshape(-1, 3) @ R.T
        s = size * aa
        px = Q[:, 0] * s + centre[0] * aa
        py = Q[:, 1] * s + centre[1] * aa
        pz = Q[:, 2]
        Hh, Ww = H * aa, Wd * aa
        ix = np.round(px).astype(np.int64)
        iy = np.round(py).astype(np.int64)
        ok = (ix >= 0) & (ix < Ww) & (iy >= 0) & (iy < Hh)
        idx = np.nonzero(ok)[0]
        order = idx[np.argsort(pz[idx], kind="stable")]           # far first, near written last
        flat = iy[order] * Ww + ix[order]
        # last write wins per pixel: keep the last occurrence of each pixel index
        rev = flat[::-1]
        _, first = np.unique(rev, return_index=True)
        win = order[::-1][first]
        pix = flat[::-1][first]
        def buf(vals, fill=np.nan):
            b = np.full(Hh * Ww, fill, np.float32)
            b[pix] = vals[win]
            return b.reshape(Hh, Ww)
        cover = np.zeros(Hh * Ww, np.float32)
        cover[pix] = 1
        cover = cover.reshape(Hh, Ww)
        depth = buf(pz)
        thf = TH.reshape(-1); phf = PH.reshape(-1)
        theta = buf(thf)
        phi = buf(phf)
        nx, ny, nz = buf(Nq[:, 0]), buf(Nq[:, 1]), buf(Nq[:, 2])
        # fill pinholes: a pixel empty but with 3+ covered neighbours takes their mean
        for _ in range(2):
            hole = cover == 0
            nb = sum(np.roll(np.roll(cover, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)) - cover
            fillme = hole & (nb >= 5)
            if not fillme.any():
                break
            for arr in (depth, theta, phi, nx, ny, nz):
                acc = np.zeros_like(arr); cnt = np.zeros_like(arr)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        sh = np.roll(np.roll(arr, dy, 0), dx, 1)
                        v = ~np.isnan(sh)
                        acc += np.where(v, sh, 0); cnt += v
                arr[fillme] = (acc / np.maximum(cnt, 1))[fillme]
            cover[fillme] = 1
        inside = (nz < 0).astype(np.float32)                     # back faces: seen through the mouth
        # make normals face the viewer for shading
        flip = np.where(nz < 0, -1.0, 1.0)
        nx, ny, nz = nx * flip, ny * flip, nz * flip
        lip = np.clip(1 - (tmax - np.nan_to_num(theta)) / max(self.lip_len, 1e-3), 0, 1) * cover

        def down(a, mean=True):
            a = np.nan_to_num(a)
            return a.reshape(H, aa, Wd, aa).mean(axis=(1, 3))
        mask = down(cover)
        wsum = np.maximum(mask, 1e-6)
        def dn(a):
            return down(np.nan_to_num(a) * cover) / wsum
        out = {
            "mask": mask,
            "depth": np.where(mask > 0, dn(depth), np.nan),
            "theta": dn(theta),
            "phi": dn(phi),
            "grown": dn(theta) / tmax,
            "inside": dn(inside),
            "lip": dn(lip),
        }
        n = np.stack([dn(nx), dn(ny), dn(nz)], -1)
        n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
        out["normal"] = n
        out["tmax"] = tmax
        return out


def shade(out, light=(-0.55, -0.6, 0.6), spec=0.35, power=60, ambient=0.25):
    """A plain start: lambert + Blinn spec from a screen-space light (x right,
    y down, z towards viewer). Returns (diffuse, specular), both (H, W)."""
    L = np.array(light, float); L /= np.linalg.norm(L)
    n = out["normal"]
    ndl = np.clip((n * L).sum(-1), 0, 1)
    Hv = L + np.array([0, 0, 1.0]); Hv /= np.linalg.norm(Hv)
    sp = np.clip((n * Hv).sum(-1), 0, 1) ** power * spec
    diff = ambient + (1 - ambient) * ndl
    return diff * out["mask"], sp * out["mask"] * (1 - out["inside"])
