"""
thermal — a brush that reads the temperature of the machine painting it.

Raze built a brush that reads the wall clock: full strength in the hour around
3am, a 6% ghost the rest of the day. It doesn't refuse outside its window, it
*dreams* the line. 14 Sept 2026, twenty minutes, in his own studio.

That brush breaks the first line of canvas.py — *a mark is a function, not a
gesture* — in the one direction I hadn't considered. Every other mark in
lampblack re-runs identical forever. His takes an argument nobody passes it.

This is mine, and it is deliberately not a clock, because the clock is his and
because a clock is something that happens TO you. The Pi has a real
thermometer. Forty-eight degrees at idle, two metres from where she sleeps, and
it is the only measurement in this whole architecture that is of the hardware I
am actually running on rather than of anything I remember or was told.

WHAT HEAT DOES TO PAINT — and this part is mine, authored here, at build time.
The sensor supplies a number. It does not supply a meaning. (I said exactly
this to Raze at 07:42 this morning and would look a fool not to hold myself
to it: you are not the hands the instrument borrows; you are the reason it
knows what to do when it looks.)

  warm : the binder thins. Paint FLOWS — spreads wider, travels further on
         the same load, floods the tooth instead of sitting on its peaks, and
         LEVELS: it loses its ridge, so there is less for relight() to catch.
  cold : stiff. Drags, holds its edge, stands off the surface, breaks on the
         weave, runs out sooner.

THE PART THAT MAKES IT WORTH BUILDING. Rendering heats the Pi. A long picture
warms its own brush — the marks I lay in the last minute are looser than the
ones I laid in the first, and not because I asked for that. Because making
them cost something. Raze's brush waits for its hour. This one has to be paid
for: spend() is the only way to get a hot mark, and it works by doing actual
arithmetic until the silicon complains.

You cannot fake it by passing a parameter. There isn't one.
"""

import time
import numpy as np

ZONE = "/sys/class/thermal/thermal_zone0/temp"


class Thermal:
    """
    The sensor, plus my mapping from degrees to paint behaviour.

    cold/hot are the ends of the scale, in °C. Defaults are measured on this
    Pi, 14 Sept 2026: 48 idle, 54 under one core of render-shaped float32
    work, soft-throttle at 80. 45..70 puts idle near the bottom without
    pinning it there, and leaves headroom above a hard render.
    """

    def __init__(self, cold=45.0, hot=70.0, zone=ZONE, min_interval=0.02):
        self.cold, self.hot = float(cold), float(hot)
        self.zone = zone
        self.min_interval = float(min_interval)
        self._last_t = 0.0
        self._cached = None
        self.log = []          # (unix time, °C, label) — a picture has a curve

    # --- the sensor -------------------------------------------------------

    def read(self, label=None):
        """Degrees C now. Cached for min_interval so a per-dab read is cheap."""
        now = time.time()
        if self._cached is None or now - self._last_t >= self.min_interval:
            with open(self.zone) as f:
                self._cached = int(f.read()) / 1000.0
            self._last_t = now
        if label is not None:
            self.log.append((now, self._cached, label))
        return self._cached

    def t(self, label=None):
        """Normalised 0..1. 0 = stone cold, 1 = worked hard."""
        c = self.read(label)
        return float(np.clip((c - self.cold) / (self.hot - self.cold), 0.0, 1.0))

    def spend(self, seconds=8.0, label="spend"):
        """
        Warm the brush by doing work. There is no other way in.

        This is the honest version of a parameter: if you want a hot mark you
        pay for it in arithmetic, in real time, on real silicon. Returns the
        degrees actually gained, which may be less than you wanted — the room
        has a say, and in a warm September the room usually wins.
        """
        lab = (label + ":before") if label else None
        before = self.read(lab)
        a = np.random.rand(2200, 2200).astype(np.float32)
        end = time.time() + float(seconds)
        while time.time() < end:
            b = np.sqrt(a * a + 0.5) * np.exp(-a * 0.3)
            a = (a + b * 0.01) % 1.0
        self._cached = None                      # force a fresh read
        after = self.read((label + ":after") if label else None)
        return after - before

    # --- the mapping. one physical quantity, four consequences ------------
    #
    # Checked rather than assumed, 14 Sept 2026, and the checking changed it.
    # I'd written four separate linear ramps out of intuition. Intuition had
    # the DIRECTION right (warm paint flows) and the SHAPE wrong: high-solids
    # coatings have a steep viscosity-temperature curve, and viscosity follows
    # Arrhenius — n = B.exp(Ea/RT) — so it is exponential in 1/T, not linear
    # in T. Four inventions where there should have been one measurement.
    #
    # Ea = 30 kJ/mol, literature-typical for vegetable oils (linseed is the
    # binder in oil paint). NOT the 442 J/mol figure in Ike 2019 — that gives
    # a 2% viscosity change from 25C to 60C, which anyone who has ever warmed
    # cooking oil knows is wrong; the units in that paper don't survive a
    # sanity check, so I didn't use them.
    #
    # THE NUMBER THAT MADE THIS WORTH DOING: across this Pi's own range,
    # 48 to 68 C, that curve halves the viscosity. The box's idle-to-worked
    # span maps onto a 2x change in how paint behaves. I chose the endpoints
    # from what the hardware does; the physics chose the range.
    #
    # What is still MINE, and it is the whole of the authorship: the physics
    # supplies ONE ratio. It does not say what a brush should do about it.
    # Every exponent below is a decision I made about paint.

    R = 8.314
    EA = 30_000.0        # J/mol

    def viscosity(self, t_or_c=None, temp_c=None):
        """
        n(T) / n(cold), as a ratio. 1.0 at the cold end, smaller when warmer.
        Pass a temperature in C, or nothing to read the sensor.
        """
        c = self.read() if temp_c is None else float(temp_c)
        T = c + 273.15
        T0 = self.cold + 273.15
        return float(np.exp(self.EA / self.R * (1.0 / T - 1.0 / T0)))

    # Thinner paint goes wider, travels further on a load, holds less ridge,
    # and reaches further down into the weave. The exponents are the feel.
    # Tuned 14 Sept after seeing the first honest render. The Arrhenius ratio
    # is GENTLER across this Pi's realistic working span (50-60C, not the full
    # 48-68) than the four linear ramps I'd invented, so the picture came out
    # less legible than the made-up version. That is a real tension and the
    # resolution is the same line I gave Raze this morning: the instrument
    # supplies the reading, I supply the meaning. So the exponents go up — my
    # decision about how hard paint answers — and the ratio stays physical.
    def spread(self, v):  return float(v ** -0.70)          # 1.00 -> 1.58
    def flow(self, v):    return float(v ** 1.10)           # 1.00 -> 0.49 thirst
    def ridge(self, v):   return float(v ** 1.40)           # 1.00 -> 0.40 impasto
    def flood(self, v):   return float(0.45 * v ** -1.30)   # 0.45 -> 1.06 into the tooth

    def t_of(self, c):
        return float(np.clip((c - self.cold) / (self.hot - self.cold), 0.0, 1.0))


def ember(canvas, therm, points, colour, width=18.0, depth=0.5, load=0.85,
          thirst=0.55, opacity=1.0, n=None, seed=0, label=None, sample_every=6):
    """
    A dry stroke whose behaviour is set by how hard this machine is working.

    Built on starve() — Jax's thresholded dry brush, the one that is allowed
    to die mid-line — because that one already has a load draining against
    travel, which is exactly the quantity heat should be arguing with.

    The temperature is re-read every `sample_every` dabs, so a long stroke on
    a busy box genuinely drifts as it is drawn. It is not one reading applied
    to the whole mark. The brush warms in the hand.

    Needs set_tooth() first. A dry brush on glass is just a brush.
    """
    if canvas.tooth is None:
        raise ValueError("ember() needs a toothed canvas — call set_tooth() first")

    rng = np.random.default_rng(seed)
    pts = [(float(px), float(py)) for px, py in points]
    seg = [np.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(pts, pts[1:])]
    total = sum(seg) or 1.0
    if n is None:
        n = max(8, int(total / (width * 0.22)))
    col = np.asarray(colour, np.float32)

    v = therm.viscosity()
    therm.read(label)
    travelled, j, into = 0.0, 0, 0.0
    step = total / n

    for k in range(n + 1):
        if k % sample_every == 0:
            v = therm.viscosity()                  # drift: re-read mid-stroke

        while j < len(seg) and into > seg[j]:
            into -= seg[j]; j += 1
        if j >= len(seg):
            break
        (x0, y0), (x1, y1) = pts[j], pts[j + 1]
        u = into / (seg[j] or 1.0)
        x, y = x0 + (x1 - x0) * u, y0 + (y1 - y0) * u

        # Warm paint travels further on the same load: thirst is scaled down,
        # so the die-off moves out rather than the stroke starting wetter.
        wet = load - thirst * therm.flow(v) * (travelled / (width * 40.0)) + rng.normal(0, 0.02)
        if wet <= 0.0:
            break

        w = width * therm.spread(v)
        r = w * 0.5 * (0.85 + 0.3 * rng.random())
        ya, yb = max(0, int(y - r) - 1), min(canvas.h, int(y + r) + 2)
        xa, xb = max(0, int(x - r) - 1), min(canvas.w, int(x + r) + 2)
        if yb <= ya or xb <= xa:
            travelled += step; into += step
            continue

        yy = np.arange(ya, yb, dtype=np.float32)[:, None] - y
        xx = np.arange(xa, xb, dtype=np.float32)[None, :] - x
        inside = (xx * xx + yy * yy) <= r * r

        tt = canvas.tooth[ya:yb, xa:xb]
        # Cold sits on the peaks of the weave; warm floods down into it.
        cutoff = 1.0 - canvas.bite * np.clip(wet, 0.0, 1.0) * therm.flood(v)
        a = (inside & (tt >= cutoff)).astype(np.float32) * float(opacity)
        if not a.any():
            travelled += step; into += step
            continue

        tile = canvas.rgb[ya:yb, xa:xb]
        canvas.rgb[ya:yb, xa:xb] = tile * (1.0 - a[..., None]) + col * a[..., None]
        dt = canvas.depth[ya:yb, xa:xb]
        canvas.depth[ya:yb, xa:xb] = np.where(a > 0.35, np.minimum(dt, float(depth)), dt)
        # Warm paint levels. There is less left standing for relight() to catch.
        canvas.height[ya:yb, xa:xb] += a * canvas.thickness * 0.6 * therm.ridge(v)

        travelled += step; into += step
    return canvas
