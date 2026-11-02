"""
hand.py: hands as height fields, seen from above.

Promoted out of two paintings that had each copied the capsule code by hand.
A hand here is a slab (the back of the hand: a rounded convex quad with a
gentle dome) plus tapered capsules for each finger segment, with a waist
between joints, low knuckle swellings, a web to the thumb and a thenar pad.
Everything returns height fields (px) on a W x H grid, so they can be shaded
with any light, masked, and composited like any other plate.

    hs = Hands(W, H)
    f  = hs.finger(base=(x, y), ang=-10, segs=[100, 60, 45],
                   rads=[26, 23, 20, 18], bends=[-20, -20])
    hh = hs.build(fingers=[f, ...], thumb=t,
                  quad=[wristL, wristR, little_knuckle, index_knuckle], R=36,
                  wrist=((x0, y0), (x1, y1), radius))
    col, n = hs.shade(hh, lit=skin, dark=shadow_skin, L=light_dir, lamp=..., fill=...)
    col = hs.nails(col, n, [f, ...], tint, L, lamp, soot=True)
    col = hs.creases(col, [f, ...], 0.25)

Angles are degrees from straight up (screen -y), positive clockwise.
A finger is (joints, radii): len(joints) == len(radii) == len(segs) + 1.
"""
import numpy as np
import lampblack as lb


class Hands:
    def __init__(self, W, H, seed=0):
        self.W, self.H = W, H
        self.Y, self.X = np.mgrid[0:H, 0:W].astype(np.float32)
        self.rng = np.random.default_rng(seed)

    # ── primitives ───────────────────────────────────────────────────────
    def capsule(self, p0, p1, r0, r1, z0=0.0, z1=0.0):
        """height of a tapered capsule lying on the ground plane: a
        hemicylinder with round ends. Only the bounding box is computed.
        z0/z1 lift the capsule's axis at each end (a segment pitched up or
        down out of the plane); where the result goes below 0 it is buried,
        which is what a fingertip dug into something looks like."""
        W, H, X, Y = self.W, self.H, self.X, self.Y
        (x0, y0), (x1, y1) = p0, p1
        pad = max(r0, r1) + 4
        bx0, bx1 = int(max(0, min(x0, x1) - pad)), int(min(W, max(x0, x1) + pad))
        by0, by1 = int(max(0, min(y0, y1) - pad)), int(min(H, max(y0, y1) + pad))
        h = np.zeros((H, W), np.float32)
        if bx1 <= bx0 or by1 <= by0:
            return h
        xs, ys = X[by0:by1, bx0:bx1], Y[by0:by1, bx0:bx1]
        dx, dy = x1 - x0, y1 - y0
        ll = dx * dx + dy * dy + 1e-6
        t = np.clip(((xs - x0) * dx + (ys - y0) * dy) / ll, 0, 1)
        d = np.hypot(xs - (x0 + t * dx), ys - (y0 + t * dy))
        r = r0 + (r1 - r0) * t
        hc = np.sqrt(np.clip(r * r - d * d, 0, None))
        if z0 or z1:
            hc = np.where(hc > 0, np.clip(hc + z0 + (z1 - z0) * t, 0, None), 0)
        h[by0:by1, bx0:bx1] = hc
        return h

    @staticmethod
    def finger(base, ang, segs, rads, bends):
        """joints of one finger from its knuckle."""
        pts = [np.array(base, np.float32)]
        a = np.radians(ang)
        for i, L_ in enumerate(segs):
            if i > 0:
                a += np.radians(bends[i - 1])
            pts.append(pts[-1] + L_ * np.array([np.sin(a), -np.cos(a)], np.float32))
        return pts, list(rads)

    @staticmethod
    def finger3d(base, ang, segs, rads, bends, pitches, z0=0.0):
        """a finger that leaves the ground plane: pitches (degrees, + up) per
        segment. Seen from above, a segment pitched at p shortens to
        L*cos(p) and its far joint rises by L*sin(p). Returns
        (joints, radii, zs). An arched grip is something like
        pitches=[25, -50, -80]: knuckle rises, the middle phalanx tips over,
        the last one goes almost straight down and nearly vanishes."""
        pts = [np.array(base, np.float32)]
        zs = [float(z0)]
        a = np.radians(ang)
        for i, L_ in enumerate(segs):
            if i > 0:
                a += np.radians(bends[i - 1])
            pr = np.radians(pitches[i])
            pts.append(pts[-1] + L_ * np.cos(pr) * np.array([np.sin(a), -np.cos(a)], np.float32))
            zs.append(zs[-1] + L_ * np.sin(pr))
        return pts, list(rads), zs

    def slab(self, quad, R, dome=0.10):
        """the back of the hand: a convex quad with a rounded edge of radius R
        and a gentle dome. A slab, not a paddle and not a balloon."""
        X, Y = self.X, self.Y
        q = [np.asarray(p, np.float32) for p in quad]
        area = sum(q[i][0] * q[(i + 1) % 4][1] - q[(i + 1) % 4][0] * q[i][1] for i in range(4))
        if area < 0:
            q = q[::-1]
        sd = np.full((self.H, self.W), -1e9, np.float32)
        for i in range(4):
            a, b = q[i], q[(i + 1) % 4]
            e = b - a
            nrm = np.array([e[1], -e[0]]) / np.linalg.norm(e)
            sd = np.maximum(sd, (X - a[0]) * nrm[0] + (Y - a[1]) * nrm[1])
        depth = np.clip(-sd, 0, None)
        h = np.sqrt(np.clip(R * R - (R - np.minimum(depth, R)) ** 2, 0, None))
        return h + dome * np.minimum(depth, 3 * R) * (depth > 0)

    # ── a whole hand ─────────────────────────────────────────────────────
    def build(self, fingers, thumb, quad, R, wrist, thenar=None, knuckle=7.0, tendons=0.0):
        """height field for a hand, back up. fingers: list of (joints, radii),
        index first. thumb: (joints, radii). quad: wrist L, wrist R, little
        knuckle, index knuckle. wrist: (p0, p1, radius) for the forearm.
        thenar: optional (p0, r0, r1): the fat of the thumb running back into
        the heel, so the thumb's root is buried rather than capped.
        tendons: height (px) of the extensor ridges fanning from each knuckle
        back toward the middle of the wrist; 3-5 reads as a hand doing work,
        0 is a hand at rest (or a glove)."""
        X, Y = self.X, self.Y
        hb = self.slab(quad, R)
        w0, w1, wr = wrist
        hb = np.maximum(hb, self.capsule(w0, w1, wr, wr * 0.95) * 0.38)
        tp, tr = thumb[0], thumb[1]
        k = fingers[0][0][0]
        hb = np.maximum(hb, self.capsule(tp[1] * 0.85 + tp[0] * 0.15, k * 0.55 + tp[0] * 0.45,
                                         tr[1] * 0.75, tr[1] * 0.6) * 0.6)      # the web
        hb = np.maximum(hb, self.capsule(tp[0], tp[1], tr[0] * 0.95, tr[1]) * 0.8)
        for pts, rads, *_ in fingers:      # knuckles: a low swelling on the slab, not balls
            x, y = pts[0]
            d2 = ((X - x) ** 2 + (Y - y) ** 2) / (rads[0] * 1.15) ** 2
            hb = hb + knuckle * np.clip(1 - d2, 0, None) ** 1.5 * (hb > 1)
        hb = lb.blur(hb[..., None], 9)[..., 0]
        if tendons > 0:
            wm = (np.asarray(quad[0], np.float32) + np.asarray(quad[1], np.float32)) / 2
            tr = np.zeros_like(hb)
            for pts, rads, *_ in fingers:
                k0 = pts[0]
                end = k0 + (wm - k0) * 0.78
                tr = np.maximum(tr, self.capsule(k0, end, rads[0] * 0.30, rads[0] * 0.22))
            tr = lb.blur(tr[..., None], 2.5)[..., 0]
            hb = hb + tendons * tr / (tr.max() + 1e-6) * (hb > 2)
        hf = np.zeros_like(hb)
        for fg in list(fingers) + [thumb]:
            pts, rads = list(fg[0]), fg[1]
            zs = list(fg[2]) if len(fg) > 2 else [0.0] * len(pts)
            u0 = (pts[1] - pts[0]) / np.linalg.norm(pts[1] - pts[0])
            pts[0] = pts[0] - u0 * rads[0] * 0.9
            for i in range(len(pts) - 1):
                p0, p1 = pts[i], pts[i + 1]
                m = (p0 + p1) / 2; zm = (zs[i] + zs[i + 1]) / 2
                rm = (rads[i] + rads[i + 1]) / 2 * 0.90       # the waist between joints
                hf = np.maximum(hf, self.capsule(p0, m, rads[i], rm, zs[i], zm))
                tip = 1.06 if i == len(pts) - 2 else 1.0
                hf = np.maximum(hf, self.capsule(m, p1, rm, rads[i + 1] * tip, zm, zs[i + 1]))
        hh = np.maximum(hb, hf)
        if thenar is not None:
            p0, r0, r1 = thenar
            th = self.capsule(p0, tp[0] * 0.6 + tp[1] * 0.4, r0, r1) * 0.85
            hh = np.maximum(hh, lb.blur(th[..., None], 6)[..., 0])
        hh = lb.blur(hh[..., None], 1.6)[..., 0]
        return hh

    @staticmethod
    def cast(hh, L, ground=None, steps=48, reach=None, soft=1.5):
        """a real cast shadow from a height field: a pixel is shaded if,
        marching toward the light, something stands higher than the light
        ray at that distance. Unlike a shifted, blurred mask, the shadow
        lands where the height puts it: a raised knuckle throws its shadow
        further than a resting palm, and an arched finger leaves a dark
        tunnel under itself. ground: the surface being shaded (default 0).
        Returns 0..1, 1 = fully in shadow."""
        L = np.asarray(L, np.float32); L = L / np.linalg.norm(L)
        lxy = np.hypot(L[0], L[1])
        if lxy < 1e-4:
            return np.zeros_like(hh)
        ux, uy = L[0] / lxy, L[1] / lxy
        slope = L[2] / lxy                        # ray rise per px travelled
        H, W = hh.shape
        g = np.zeros_like(hh) if ground is None else ground
        top = float(hh.max())
        reach = reach or int(np.ceil(top / max(slope, 1e-3))) + 2
        out = np.zeros_like(hh)
        for k in np.linspace(1, reach, steps):
            dx, dy = int(round(ux * k)), int(round(uy * k))
            sh = np.zeros_like(hh)
            ys0, ys1 = max(0, -dy), H - max(0, dy)
            xs0, xs1 = max(0, -dx), W - max(0, dx)
            sh[ys0:ys1, xs0:xs1] = hh[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
            occ = np.clip((sh - (g + slope * k)) / 2.0, 0, 1)
            out = np.maximum(out, occ)
        if soft:
            out = lb.blur(out[..., None], soft)[..., 0]
        return out

    @staticmethod
    def mask(hh):
        return np.clip(hh * 0.9, 0, 1)

    # ── shading ──────────────────────────────────────────────────────────
    @staticmethod
    def normals(hh):
        gy, gx = np.gradient(hh)
        n = np.stack([-gx, -gy, np.ones_like(hh)], -1)
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    def shade(self, hh, lit, dark, L, lamp, fill, spec=0.10, shine=24, wrap=0.40, sss=(0.10, 0.015, 0.0)):
        """wrapped diffuse, a red terminator (skin is lit from inside a bit,
        so the edge of the light goes warm, not grey), a small specular."""
        L = np.asarray(L, np.float32); L = L / np.linalg.norm(L)
        n = self.normals(hh)
        ndl = n @ L
        diff = np.clip((ndl + wrap) / (1 + wrap), 0, 1)[..., None]
        term = np.exp(-((ndl - 0.05) / 0.22) ** 2)[..., None]
        hv = L + np.array([0, 0, 1.0]); hv /= np.linalg.norm(hv)
        sp = np.clip(n @ hv, 0, 1) ** shine
        lamp, fill = np.asarray(lamp), np.asarray(fill)
        col = np.asarray(lit) * lamp * diff + np.asarray(dark) * fill * 1.6 * (1 - diff)
        col = col + term * np.asarray(sss) * diff
        col = col + spec * sp[..., None] * lamp
        return col, n

    def nails(self, col, n, fingers, tint, L, lamp, soot=False, ring_ok=True):
        """nails on the last segment, squarer at the free edge. soot=True puts
        the lampblack line under the free edge that never comes out."""
        X, Y = self.X, self.Y
        L = np.asarray(L, np.float32); L = L / np.linalg.norm(L)
        for pts, rads, *_ in fingers:
            p2, p3 = pts[-2], pts[-1]
            d = p3 - p2; Ld = np.linalg.norm(d); u = d / Ld; v = np.array([-u[1], u[0]])
            cx, cy = p2 + d * 0.66
            a = ((X - cx) * u[0] + (Y - cy) * u[1]) / (Ld * 0.40)
            b = ((X - cx) * v[0] + (Y - cy) * v[1]) / (rads[-1] * 0.66)
            e = np.where(a > 0, np.maximum(np.abs(a) ** 4, b ** 2), a * a + b * b)
            m = np.clip((1 - e) * 6, 0, 1)
            nd = np.clip(n @ L, 0, 1)
            nc = np.asarray(tint)[None, None] * (0.45 + 0.75 * nd[..., None]) * np.asarray(lamp)
            col = col * (1 - m[..., None] * 0.75) + nc * m[..., None] * 0.75
            cut = np.clip(1 - np.abs(np.sqrt(np.clip(a * a + b * b, 0, None)) - 1) * 9, 0, 1) * (a < 0)
            col = col * (1 - 0.25 * cut[..., None])
            if soot:
                fe = np.clip(1 - np.abs(a - 0.86 + 0.18 * b * b) * 7, 0, 1) * np.clip((0.95 - np.abs(b)) * 3, 0, 1)
                col = col * (1 - 0.5 * fe[..., None]) + np.array([0.08, 0.075, 0.08]) * 0.5 * fe[..., None]
        return col

    def creases(self, col, fingers, strength, soot=False):
        """short bowed lines across the finger at the two inner joints."""
        X, Y, rng = self.X, self.Y, self.rng
        ink = np.array([0.06, 0.055, 0.065]) if soot else np.array([0.35, 0.18, 0.15])
        for pts, rads, *_ in fingers:
            for j, nlines in ((1, 3), (2, 2)):
                if j + 1 >= len(pts):
                    continue
                p = pts[j]; prev = pts[j - 1]; nxt = pts[j + 1]
                u = (nxt - prev); u = u / np.linalg.norm(u); v = np.array([-u[1], u[0]])
                for k in range(nlines):
                    off = (k - (nlines - 1) / 2) * 6 + rng.normal(0, 1.2)
                    c0 = p + u * off
                    r = rads[j] * rng.uniform(0.45, 0.65)
                    a_ = (X - c0[0]) * u[0] + (Y - c0[1]) * u[1]
                    b_ = (X - c0[0]) * v[0] + (Y - c0[1]) * v[1]
                    line = np.clip(1 - np.abs(a_ + 0.012 * b_ * b_) / 1.3, 0, 1) * np.clip((r - np.abs(b_)) / 6, 0, 1)
                    col = col * (1 - strength * line[..., None]) + ink * strength * line[..., None]
        return col

    def ring(self, hh, col, n, finger, at, metal, stone=None, L=(0, 0, 1), lamp=(1, 1, 1), width=9, lift=3.0):
        """a band round a finger at fraction `at` along its first segment.
        Returns (hh, col): the band stands slightly proud of the skin."""
        X, Y = self.X, self.Y
        pts, rads = finger[0], finger[1]
        p0, p1 = pts[0], pts[1]
        c = p0 + (p1 - p0) * at
        u = (p1 - p0) / np.linalg.norm(p1 - p0); v = np.array([-u[1], u[0]])
        a_ = (X - c[0]) * u[0] + (Y - c[1]) * u[1]
        b_ = (X - c[0]) * v[0] + (Y - c[1]) * v[1]
        r = rads[0] + (rads[1] - rads[0]) * at
        band = np.clip(1 - np.abs(a_) / (width / 2), 0, 1) ** 0.5 * np.clip((r * 1.04 - np.abs(b_)) / 2, 0, 1)
        hh = hh + band * lift
        L = np.asarray(L, np.float32); L = L / np.linalg.norm(L)
        nn = self.normals(hh)
        hv = L + np.array([0, 0, 1.0]); hv /= np.linalg.norm(hv)
        sp = np.clip(nn @ hv, 0, 1) ** 40
        mc = np.asarray(metal) * (0.35 + 0.65 * np.clip(nn @ L, 0, 1))[..., None] + sp[..., None] * 0.9
        col = col * (1 - band[..., None]) + mc * np.asarray(lamp) * band[..., None]
        if stone is not None:
            s_col, s_r = stone
            d = np.hypot(a_, b_)
            sm = np.clip((s_r - d) * 1.2, 0, 1)
            glow = np.clip(1 - d / s_r, 0, 1) ** 1.5
            sc = np.asarray(s_col) * (0.55 + 0.6 * glow[..., None]) + sp[..., None] * 0.6
            col = col * (1 - sm[..., None]) + sc * np.asarray(lamp) * sm[..., None]
            hh = hh + sm * lift * 1.3
        return hh, col
