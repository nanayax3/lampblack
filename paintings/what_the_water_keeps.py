"""
What the Water Keeps — 9 Sept 2026.

Jellyfish, because Nana asked and because they are the right subject for a
tool whose whole argument is distance: a soft body in deep water is almost
entirely a question of how much water is in front of it.

The bell is one parametric dome; the tentacles are paths with lengths drawn
from a spread rather than a constant, because evenly matched trailing lines
read as a picket fence — the mistake I made twice tonight already, once as a
row and once as a starburst. Third time, deliberately avoided.

Light comes from above and the far ones are not drawn dimmer. They are simply
further away.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import canvas as cv, lampblack as lb

W, H = 1100, 1500
c = cv.Canvas(W, H, background=(0.020, 0.036, 0.058))
rng = np.random.default_rng(1207)


def jelly(x, y, size, depth, bell, trail, n_tent=8, phase=0.0, crown=1.0):
    """One animal. size is bell width in pixels; everything else scales off it."""
    s = float(size)
    bell = np.asarray(bell, np.float32)

    # the bell: nested arcs, brightest at the crown, so it reads as a dome
    # rather than a disc. Ten arcs is enough — the gradient is the step count.
    for k in range(10):
        u = k / 9.0
        r = s * (0.5 - 0.42 * u)
        lift = s * 0.10 * u
        c.path(lambda t, r=r, lift=lift: (x + r * np.cos(np.pi * (1 - t)),
                                          y - lift - r * 0.62 * np.sin(np.pi * t)),
               # `crown` attenuates the top arcs only (u->1), leaving the skirt
               # alone — Ezra's note, 9 Sept 2026: where the bloom flattens the
               # dome the fix is upstream of the bloom, not sanded on after.
               colour=tuple(bell * (0.55 + 0.55 * u) * (1.0 - (1.0 - crown) * u)),
               width=s * 0.085, depth=depth, opacity=0.55, n=34, hardness=0.25)
    # the rim, darker, where the bell turns away
    c.path(lambda t: (x + s * 0.5 * np.cos(np.pi * (1 - t)), y - s * 0.02),
           colour=tuple(bell * 0.42), width=s * 0.07, depth=depth, opacity=0.7, n=26, hardness=0.4)

    # oral arms — short, thick, doubled back on themselves
    for i in range(4):
        u = (i + 0.5) / 4.0
        ox = x + (u - 0.5) * s * 0.62
        L = s * (1.3 + 0.5 * rng.random())
        c.path(lambda t, ox=ox, L=L: (ox + s * 0.16 * np.sin(t * 4.4 + phase) * t,
                                      y + L * t),
               colour=tuple(bell * 0.85), width=lambda t: s * 0.10 * (1.0 - 0.75 * t),
               depth=depth, opacity=0.8, n=44, hardness=0.2)

    # trailing tentacles. Length, drift and start are all drawn from a spread:
    # matched lengths are the picket fence.
    for i in range(n_tent):
        u = (i + rng.uniform(-0.3, 0.3)) / max(1, n_tent - 1)
        ox = x + (u - 0.5) * s * 0.92
        L = trail * rng.uniform(0.35, 1.0) ** 0.8
        drift = rng.uniform(-0.5, 0.5)
        start = y + s * rng.uniform(0.0, 0.18)
        c.path(lambda t, ox=ox, L=L, drift=drift, start=start: (
                   ox + drift * s * 0.9 * t * t + s * 0.10 * np.sin(t * 7.0 + phase),
                   start + L * t),
               colour=tuple(bell * 0.75), width=lambda t: max(0.9, s * 0.030 * (1.0 - 0.55 * t)),
               depth=depth + 0.01, opacity=0.62, n=70, hardness=0.15)


WARM = (1.00, 0.62, 0.24)
COLD = (0.62, 0.74, 0.92)

# far field first
jelly(880, 300, 62, 0.88, COLD, 300, 6, phase=1.1)
jelly(210, 480, 54, 0.86, COLD, 260, 6, phase=2.3)
jelly(650, 210, 48, 0.90, COLD, 230, 5, phase=0.4)
jelly(430, 700, 78, 0.72, COLD, 380, 7, phase=3.0)
jelly(940, 760, 86, 0.68, COLD, 420, 7, phase=1.8)
jelly(150, 980, 96, 0.60, COLD, 460, 8, phase=0.9)

# the two that are near, and lit
jelly(700, 1080, 210, 0.16, WARM, 760, 11, phase=0.6, crown=0.78)
c.glow(700, 1050, 150, (0.85, 0.44, 0.14), depth=0.17, strength=0.42, falloff=2.1)
jelly(330, 560, 150, 0.40, WARM, 620, 9, phase=2.6)
c.glow(330, 535, 110, (0.80, 0.40, 0.12), depth=0.41, strength=0.32, falloff=2.2)

# marine snow — sparse, small, at every distance
for _ in range(220):
    px, py = rng.uniform(0, W), rng.uniform(0, H)
    d = rng.uniform(0.05, 0.95)
    c.dab(px, py, rng.uniform(0.9, 2.6) * (1.2 - d), (0.72, 0.80, 0.88),
          depth=d, opacity=rng.uniform(0.25, 0.7), hardness=0.1)

# light from the surface
c.glow(560, -260, 900, (0.30, 0.46, 0.62), depth=0.99, strength=0.30, falloff=1.5)

print("before passes:", c.measure())

img = lb.depth_fog(c.rgb, c.depth, colour=(0.026, 0.052, 0.085), density=1.05, desaturate=0.72)
img = lb.bloom(img, threshold=0.26, radius=34, intensity=0.62, tint=(1.0, 0.74, 0.44))
img = lb.vignette(img, strength=0.40, power=1.8)
img = lb.grain(img, amount=0.010, seed=44)

dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "what_the_water_keeps.png")
lb.save(img, dest)
print("saved:", dest, os.path.getsize(dest) // 1024, "KB")
