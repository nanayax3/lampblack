"""
watercolour — pigment, water, paper.

Every other brush in this toolkit puts colour where I said to put it. This one
does not, and that is the entire reason it exists. What makes a watercolour
recognisable is a list of things the painter did not draw:

  * the DARK RIM. A wash dries from the middle outward; the water retreating
    toward the edge carries pigment with it and abandons it there. Almost every
    stroke on a watercolour sheet is darker at its boundary than in its middle,
    and nothing else in painting does that.
  * BLEED. Pigment laid on wet paper keeps travelling after the brush has gone.
  * GRANULATION. Heavy pigments fall into the valleys of the paper and light
    ones stay suspended on the peaks, so a flat wash comes out speckled.
  * LAYERING that darkens rather than covers, because the paint is transparent
    and the paper is what you are looking at through it.

So this is a small physics, not a brush shape. I lay down water and pigment,
let it move, let it dry, and then find out what it did.
"""

import numpy as np
import canvas as cv
import lampblack as lb


class Wash:
    """One layer of transparent paint: how much pigment ended up where."""

    def __init__(self, width, height, tooth=None, seed=0):
        self.w, self.h = int(width), int(height)
        self._mottle = None
        self._seed = int(seed)
        self.pigment = np.zeros((self.h, self.w), np.float32)
        self.water = np.zeros((self.h, self.w), np.float32)
        self.tooth = tooth

    # ── laying paint down ───────────────────────────────────────────────────

    def lay(self, draw, wetness=1.0, load=1.0):
        """
        `draw` is a function taking a scratch Canvas and marking on it — so the
        whole brush library (tips, stamping, bristle, scatter) can describe the
        *shape* of a wash, while what happens afterwards is up to the water.
        """
        scratch = cv.Canvas(self.w, self.h, background=(0, 0, 0))
        draw(scratch)
        mask = np.clip(scratch.rgb[..., 0], 0.0, 1.0)

        # A wash is uneven BEFORE it starts drying. The brush releases in
        # surges, the paper drinks at different rates, and the pigment settles
        # in drifts. Starting from a flat mask was my mistake: with a uniform
        # interior the physics has nothing to work on except the boundary, so
        # everything came out as a shape with an outline round it.
        #
        # Two scales of unevenness, because that is what it looks like: broad
        # drifts across the whole wash, and a finer mottle within them.
        if self._mottle is None:
            rng = np.random.default_rng(self._seed)
            n = rng.random((self.h, self.w)).astype(np.float32)
            broad = lb.blur(n[..., None], 26, passes=2)[..., 0]
            fine = lb.blur(n[..., None], 5, passes=2)[..., 0]
            broad = (broad - broad.mean()) / max(1e-5, broad.std())
            fine = (fine - fine.mean()) / max(1e-5, fine.std())
            self._mottle = np.clip(1.0 + 0.52 * broad + 0.30 * fine, 0.12, 2.2).astype(np.float32)

        self.pigment += mask * float(load) * self._mottle
        self.water = np.maximum(self.water, mask * float(wetness))
        return self

    # ── what the water does next ────────────────────────────────────────────

    def settle(self, bleed=2.2, rim=0.55, rim_width=2.0, granulation=0.45, blooms=0):
        """
        Dry the wash. Order matters and mirrors what actually happens: the paint
        spreads while it is wet, then the retreating edge concentrates pigment
        at the boundary, then what is left settles into the paper.
        """
        # 1. bleed — pigment travels while the paper is wet, and only there
        if bleed > 0:
            spread = lb.blur(self.pigment[..., None], bleed, passes=2)[..., 0]
            wet = np.clip(self.water, 0.0, 1.0)
            self.pigment = self.pigment * (1.0 - wet) + spread * wet

        # 2. the rim. The wash boundary is where the water finally retreats to,
        #    so pigment piles up there. Found as the difference between the wash
        #    and a slightly shrunken copy of itself.
        if rim > 0:
            m = np.clip(self.pigment / max(1e-5, self.pigment.max()), 0, 1)
            inner = lb.blur(m[..., None], rim_width, passes=1)[..., 0]
            edge = np.clip(m - inner, 0.0, None)
            edge = edge / max(1e-5, edge.max())
            # Pigment is CONSERVED. The rim is dark because the interior gave it
            # up, not because more paint appeared — so take from the middle what
            # you add to the edge. Adding without subtracting was my first
            # version, and it produced a stroke dark in the centre with a pale
            # halo: precisely inside-out from a real wash.
            taken = rim * m * float(self.pigment.max())
            self.pigment = self.pigment - taken
            deposit = edge * float(taken.sum() / max(1e-5, edge.sum()))
            self.pigment = np.clip(self.pigment + deposit, 0.0, None)

        # 3. granulation — heavy pigment falls into the tooth of the paper
        if granulation > 0 and self.tooth is not None:
            g = 1.0 + granulation * (1.0 - self.tooth) * 2.0 - granulation
            self.pigment = self.pigment * g

        # 4. blooms — a drop of clean water into a damp wash pushes pigment
        #    outward and leaves a pale centre with a hard scalloped edge
        for _ in range(int(blooms)):
            self._bloom()
        return self

    def _bloom(self, rng=np.random.default_rng()):
        ys, xs = np.nonzero(self.pigment > self.pigment.max() * 0.25)
        if not len(xs):
            return
        i = rng.integers(len(xs))
        cx, cy = float(xs[i]), float(ys[i])
        r = rng.uniform(0.05, 0.13) * min(self.w, self.h)
        yy = np.arange(self.h, dtype=np.float32)[:, None] - cy
        xx = np.arange(self.w, dtype=np.float32)[None, :] - cx
        d = np.sqrt(xx * xx + yy * yy) / r
        wobble = 1.0 + 0.16 * np.sin(np.arctan2(yy, xx) * rng.integers(5, 11))
        push = np.clip(1.0 - np.abs(d - wobble) / 0.30, 0, 1)
        clear = np.clip(1.0 - d / wobble, 0, 1) ** 0.6
        self.pigment = self.pigment * (1.0 - 0.75 * clear) + push * self.pigment.max() * 0.45

    # ── seeing it ───────────────────────────────────────────────────────────

    def over(self, rgb, colour, strength=1.0):
        """
        Composite onto an image the way transparent paint works: it multiplies
        what is underneath instead of replacing it, so two washes crossing make
        a third, darker colour and the paper keeps showing through.
        """
        p = np.clip(self.pigment * float(strength), 0.0, 3.0)[..., None]
        transmit = np.exp(-p * (1.0 - np.asarray(colour, np.float32)) * 3.2)
        return np.clip(rgb, 0, 1) * transmit
