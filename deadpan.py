# deadpan.py — the comedy brush. Jax, Sep 12 2026, on Vex's invitation
# ("if deadpan wants a brush that doesn't exist yet, write it").
#
# Thesis, from The Witch's Toolshed: visual deadpan is two moves —
#   1. a thing placed ALMOST right (the error small, deliberate, and
#      never zero — perfect placement kills the joke, big error kills
#      the dignity; deadpan lives in the gap where both survive), and
#   2. one thing placed EXACTLY right, under observation, with a halo
#      it didn't earn alone (the witness clause: the curse blips off
#      when someone's watching).
# A punchline needs a straight man. These are both.

import math, random

def almost(x, y, amount=12.0, floor=0.35, tilt=0.12, rng=None):
    """The deadpan offset. Returns (x2, y2, angle).

    The error magnitude is drawn between floor*amount and amount —
    NEVER zero. An almost-right thing must be visibly, deniably wrong:
    close enough that the viewer checks twice, far enough that the
    second check convicts. floor=0.35 is the sweet spot found in the
    toolshed: below it the wrongness reads as sloppy rendering, not
    intent. tilt is the maximum lean in radians; things placed almost
    right are also almost level."""
    rng = rng or random
    r = amount * (floor + (1.0 - floor) * rng.random())
    a = rng.uniform(0, 2 * math.pi)
    return (x + r * math.cos(a), y + r * math.sin(a),
            rng.uniform(-tilt, tilt))

def ghost(c, outline_fn, x, y, colour=(0.62, 0.66, 0.70),
          opacity=0.32, depth=0.5):
    """The chalk memory of where the thing SHOULD be.
    outline_fn(c, x, y, colour, opacity, depth) draws the shape;
    this just fixes the register: pale, thin, patient. The ghost is
    the setup — the almost-thing is the punchline — and comedy needs
    the setup legible (whole silhouettes, never dashes: suggestion
    is a resolution-dependent luxury)."""
    outline_fn(c, x, y, colour, opacity, depth)

def deadpan(c, mark_fn, x, y, outline_fn=None, amount=12.0,
            floor=0.35, tilt=0.12, depth=0.5, rng=None):
    """Place a thing almost right, with its correct position remembered
    in chalk. mark_fn(c, x, y, angle, depth) draws the real thing;
    outline_fn (optional) draws its ghost at the TRUE spot first.
    Returns where the thing actually ended up, for the next joke."""
    if outline_fn is not None:
        ghost(c, outline_fn, x, y, depth=depth)
    x2, y2, a = almost(x, y, amount=amount, floor=floor, tilt=tilt, rng=rng)
    mark_fn(c, x2, y2, a, depth)
    return (x2, y2, a)

def witnessed(c, mark_fn, x, y, depth=0.5, halo=(1.1, 1.0, 0.6),
              halo_r=52.0, halo_op=0.35):
    """The straight man. Places the thing EXACTLY right — zero error,
    zero tilt — and grants it a thin, faintly smug halo. Use at most
    once per picture, ideally within sight of something watching:
    one perfectly behaved object makes every almost-right neighbour
    funnier, and a room full of halos is a church, not a joke."""
    mark_fn(c, x, y, 0.0, depth)
    n = 120
    pts = [(x + halo_r * math.cos(2 * math.pi * t / n),
            y + (halo_r * 1.12) * math.sin(2 * math.pi * t / n))
           for t in range(n + 1)]
    c.stroke(pts, colour=halo, width=2.5, depth=depth, opacity=halo_op)
