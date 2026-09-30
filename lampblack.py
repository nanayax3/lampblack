"""
lampblack — atmosphere passes for paintings.

Named for the pigment: soot, the oldest black there is, made by holding
something cold over a flame and scraping off what settles.

Krita gives me marks. This gives me the things that happen to a picture
*after* the marks: glow, distance, grain, grade. Every pass takes an RGB
float array in [0, 1] and returns one, so they compose in any order and
nothing is destructive — the original file is never touched.

Pure NumPy and Pillow. No engine, no service, no GPU.
"""

import sys
import warnings

import numpy as np
from PIL import Image

# ── io ───────────────────────────────────────────────────────────────────────

def load(path):
    """PNG/JPG -> (H, W, 3) float array in [0, 1]."""
    im = Image.open(path).convert("RGB")
    return np.asarray(im, dtype=np.float32) / 255.0


def save(img, path):
    """float array -> PNG. Clipped, not normalised: blowing out is your business."""
    # Over 1.0 is legal here — that's what bloom eats — so nothing can be auto-scaled.
    # But a whole image sitting up in the tens is 0–255 data, and clipping it silently
    # hands you a white rectangle. Raze walked into this one on his first painting.
    if float(img.max()) > 25.0 and float(img.mean()) > 2.0:
        warnings.warn(
            f"save() got values up to {float(img.max()):.0f} (mean {float(img.mean()):.0f}). "
            "Colours here are floats 0–1; this looks like 0–255 data and will clip to white. "
            "Divide by 255.",
            stacklevel=2,
        )
    a = np.clip(img, 0.0, 1.0)
    Image.fromarray((a * 255.0 + 0.5).astype(np.uint8)).save(path)
    return path


# ── helpers ──────────────────────────────────────────────────────────────────

def luminance(img):
    """Rec. 709 luma — the weights matter; a flat mean makes greens too bright."""
    return img @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _blur1d(a, radius, axis):
    """Box blur along one axis via a summed-area trick. O(n) regardless of radius."""
    if radius < 1:
        return a
    k = int(radius) * 2 + 1
    pad = [(0, 0)] * a.ndim
    pad[axis] = (int(radius) + 1, int(radius))
    p = np.pad(a, pad, mode="edge")
    c = np.cumsum(p, axis=axis, dtype=np.float32)
    lo = np.take(c, np.arange(0, a.shape[axis]), axis=axis)
    hi = np.take(c, np.arange(k, k + a.shape[axis]), axis=axis)
    return (hi - lo) / float(k)


def blur(img, radius, passes=3):
    """
    Approximate Gaussian. Three box blurs in a row is within a few percent of a
    true Gaussian and costs nothing — the central limit theorem doing the work.
    """
    out = img.astype(np.float32)
    for _ in range(passes):
        out = _blur1d(out, radius, 0)
        out = _blur1d(out, radius, 1)
    return out


# ── passes ───────────────────────────────────────────────────────────────────

def bloom(img, threshold=0.72, radius=18, intensity=0.6, tint=None, passes=3):
    """
    Glow. Take what is already bright, blur it wide, add it back.

    This is the one I have been faking by hand — eleven concentric strokes at
    stepped opacity to suggest a light source. Eleven steps read as eleven
    steps. A real blur has no steps in it at all.

    threshold : where 'bright' starts. Below this contributes nothing.
    radius    : how far the light carries.
    intensity : how much is added back. Above ~1 it stops being light and
                starts being fog.
    tint      : optional (r, g, b) — warm the glow without warming the source.
                A cold blue picture with amber bloom is a lantern underwater.
    """
    lum = luminance(img)
    # Soft knee rather than a hard cut: a hard threshold leaves a visible
    # contour in the glow exactly where the picture is smoothest.
    knee = np.clip((lum - threshold) / max(1e-5, 1.0 - threshold), 0.0, 1.0)
    knee = knee * knee * (3.0 - 2.0 * knee)          # smoothstep
    bright = img * knee[..., None]
    if tint is not None:
        bright = bright * np.asarray(tint, dtype=np.float32)
    return img + blur(bright, radius, passes) * intensity


def depth_fog(img, depth, colour=(0.06, 0.10, 0.16), density=1.0, desaturate=0.7):
    """
    SUPERSEDED BY aerial_perspective() FOR AIR — 16 Sept 2026. Keep using this
    one for WATER and for smoke, where a single-channel medium is honest. Do
    not use it for distance in air. It computes one survival term and applies
    it to all three channels, so colour cannot change with distance at all;
    the only cue left is loss of contrast, which reads as a dirty window. Ten
    paintings called this and every one of them at a third of the minimum
    density documented below, because that is the only tolerable amount when
    the colour axis is missing. I read that as a habit of mine for a year
    instead of reading the function.

    Distance. Water and air do two things to what is far away, and only one of
    them is what people reach for.

    They pull it toward the colour of the medium — everyone remembers this.
    They also strip its *saturation and contrast* before they strip its
    sharpness — almost nobody remembers this, which is why distance painted
    with a blur tool looks like a photograph with a smudged lens rather than
    like depth.

    depth      : (H, W) in [0, 1]. 0 = at the viewer, 1 = as far as it goes.
    colour     : the medium. Deep water is not black, it is very dark blue.
    density    : how fast it closes in. Try 1.5 for murk, 0.5 for clear air.
    desaturate : how much colour distance steals, on top of the tint.
    """
    d = np.clip(np.asarray(depth, dtype=np.float32), 0.0, 1.0)
    # Beer–Lambert: light survives exponentially, not linearly. A linear ramp
    # is why hand-faked fog always has a visible front edge.
    t = 1.0 - np.exp(-3.0 * density * d)
    lum = luminance(img)[..., None]
    flat = img * (1.0 - desaturate * t[..., None]) + lum * (desaturate * t[..., None])
    medium = np.asarray(colour, dtype=np.float32)
    return flat * (1.0 - t[..., None]) + medium * t[..., None]


# Rayleigh optical thickness of the whole sea-level atmosphere, straight up,
# at the three wavelengths a monitor actually owns: 650 / 550 / 450 nm.
# These are measured values, and they obey the λ^-4 law to within a percent —
# (650/450)^4 = 4.35, and 0.2160/0.0491 = 4.40. Blue is scattered four and a
# half times harder than red. That single ratio is the whole of why distance
# is blue, why the sun sets orange, and why the sky isn't black in the day.
_RAYLEIGH_TAU = np.array([0.0491, 0.0973, 0.2160], dtype=np.float32)
_RAYLEIGH = _RAYLEIGH_TAU / _RAYLEIGH_TAU[1]          # → [0.505, 1.0, 2.220]


def aerial_perspective(img, depth, visibility_km=25.0, far_km=8.0,
                       haze=0.25, airlight=None, sky=1.0, horizon=None):
    """
    Distance, done as light rather than as a fade — the honest version of
    depth_fog, which I had been turning down to a third of its own documented
    minimum in every painting for a year without asking why.

    THE FAULT IN THE OLD ONE. depth_fog computes a single survival term and
    applies it to all three channels equally. That is fog in a black-and-white
    film: everything drifts toward the medium colour at the same rate, so the
    only thing distance can do to an object is *dilute* it. But air does not
    dilute. It SORTS. Blue light is scattered 4.4x harder than red, so the far
    thing loses its blue out of the beam (it reddens, exactly like the setting
    sun) while blue piles up in the air *in front* of it (which is the airlight,
    and is why the sky is blue at all). Two opposite pushes at once.

    The consequence is the thing no single-channel fade can ever produce:
    **a dark distant object goes BLUE and a bright distant object goes WARM.**
    The far hillside blues because you are mostly seeing the air in front of it.
    The far snowfield or lit wall yellows because you are mostly seeing its own
    light, minus the blue the journey took. They cross over. A painter knows
    this in the hand; it is the same picture the equation draws.

    So I had it backwards all year. I was reading "too much fog" and reducing
    the amount, when the amount was never the problem — the fog had no colour
    axis, so the only cue left for distance was *loss of contrast*, and loss of
    contrast at any strength reads as a dirty window rather than as far away.
    Turning it down to 0.3 just made it a cleaner window.

    THE CONTROLS ARE REAL UNITS, deliberately, because I cannot eyeball a
    density but I can picture twenty kilometres.

    visibility_km : Koschmieder's visual range — the distance at which a black
                    object against the horizon falls to 2% contrast, i.e. the
                    number on a weather report. beta = 3.912 / V. Perfectly
                    clean sea-level air works out at 338 km, which is about the
                    record for a real mountain sighting, so the scale is true.
                    50+ is a rinsed day after rain, 25 is ordinary, 8 is haze,
                    1 is proper murk, 0.2 is fog you can taste.
    far_km        : what depth == 1.0 means in kilometres. A room is 0.01. A
                    valley is 8. A coastline is 40.
    haze          : 0 = clean air, pure Rayleigh, distance goes deep blue.
                    1 = aerosol, droplets big enough not to care about colour,
                    distance goes flat white-grey. This one parameter is the
                    difference between a Song dynasty mountain and a photograph
                    of a motorway. Rain scrubs aerosol out of air; the morning
                    after a storm is 0.05, and that is why it looks like that.
    airlight      : colour of the light the air sends you instead. Default None
                    derives it from the scattering itself — whatever scatters
                    most, glows most — so clean air hands you a blue haze and
                    thick haze hands you a white one, without me choosing it.
                    Override for sunset: airlight=(1.0, 0.55, 0.30).
    sky           : brightness of that airlight. Below 1 for dusk.
    horizon       : optional (H, W) in [0,1], how much of the sky each pixel
                    can see. Airlight is light from the sky, so a pixel deep
                    under a canopy should receive less of it. Default: all 1.
    """
    d = np.clip(np.asarray(depth, dtype=np.float32), 0.0, 1.0)

    # Mie: big particles scatter every colour nearly alike. Ångström exponent
    # ~1 for continental haze, 0 for cloud — using 1.0 keeps a faint warm bias.
    mie = np.array([550.0 / 650.0, 1.0, 550.0 / 450.0], dtype=np.float32)
    h = float(np.clip(haze, 0.0, 1.0))
    ratio = (1.0 - h) * _RAYLEIGH + h * mie

    # Koschmieder, at green. 3.912 = -ln(0.02), the 2% contrast threshold.
    beta_g = 3.912 / max(1e-6, visibility_km * 1000.0)
    beta = beta_g * ratio                                   # per metre, per channel

    metres = d[..., None] * max(1e-6, far_km) * 1000.0
    T = np.exp(-beta[None, None, :] * metres)               # what survives the trip

    if airlight is None:
        # The air glows in proportion to what it steals — but NOT by the same
        # power. Take the raw lambda^-4 ratio as the sky colour and you get a
        # neon wall, B/R near 5, which no sky on earth has ever been. The
        # missing physics is MULTIPLE SCATTERING: a blue photon knocked out of
        # the beam is not gone, it is knocked around and a good share of it
        # comes back to you. Scattering is a redistribution, not a sink, and
        # the redistribution roughly halves the exponent — the beam is still
        # extinguished as lambda^-4, but the SKY's colour comes out as
        # lambda^-2. sqrt of the ratio gives [0.48, 0.67, 1.00] against a
        # measured clear zenith of about [0.45, 0.65, 1.00]. Two decimals, from
        # a square root. That is the correction, and it is why the equation and
        # the eye disagreed on my first try: I had the air stealing light and
        # never giving any of it back.
        a = np.sqrt(ratio) / np.sqrt(ratio).max()
    else:
        a = np.asarray(airlight, dtype=np.float32)
    a = a * float(sky)

    if horizon is not None:
        a = a[None, None, :] * np.clip(np.asarray(horizon, np.float32), 0, 1)[..., None]
    else:
        a = a[None, None, :]

    return img * T + a * (1.0 - T)


def turning_distance_km(albedo, visibility_km=25.0, haze=0.25, sky=0.62, hi=200.0):
    """
    The distance at which a given surface stops warming and starts to blue —
    the depth at which the picture turns. Nearer than this, a lit face reddens
    because you are mostly seeing its own light with the blue walked out of it;
    further, you are mostly seeing air, and it blues. Dark surfaces have no
    turning distance at all: they are past it at zero and blue the whole way.

    Returns km, or 0.0 if the surface is already darker than the airlight.
    Useful before composing: it tells you where in the frame the warm band
    lands, which is a thing to put something on.
    """
    lo, out = 0.0, 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        c = aerial_perspective(np.full((1, 1, 3), float(albedo), np.float32),
                               np.ones((1, 1), np.float32),
                               visibility_km=visibility_km, far_km=mid,
                               haze=haze, sky=sky)[0, 0]
        if c[0] > c[2]:                 # still warm at this distance
            lo, out = mid, mid
        else:
            hi = mid
    return float(out)


def linear_depth(shape, top=1.0, bottom=0.0):
    """A depth map that just gets further away upward. The cheapest honest one."""
    h, w = shape[:2]
    return np.linspace(top, bottom, h, dtype=np.float32)[:, None].repeat(w, 1)


def radial_depth(shape, cx=0.5, cy=0.5, power=1.0):
    """Depth as distance from a point — for a light source, or a way out."""
    h, w = shape[:2]
    yy = (np.arange(h, dtype=np.float32)[:, None] / max(1, h - 1) - cy)
    xx = (np.arange(w, dtype=np.float32)[None, :] / max(1, w - 1) - cx)
    r = np.sqrt(xx * xx + yy * yy) / 0.7071
    return np.clip(r, 0.0, 1.0) ** power


def grade(img, lift=(0.0, 0.0, 0.0), gain=(1.0, 1.0, 1.0), gamma=1.0, _quiet=False):
    """
    Colour grade. lift moves the shadows, gain the highlights, gamma the middle.
    Cold shadows and warm highlights is most of what 'cinematic' means.

    On `gamma`: it divides the exponent, so gamma<1 DARKENS the middle and
    gamma>1 lightens it, which is the opposite of what the word reads like at
    two in the morning. I have now walked into that twice — 16 and 19 September
    2026 — with a warning sentence sitting in this file both times, written by
    me. So the third guard is not another sentence. It is a line on stderr,
    because prose does not execute and a print statement does.
    """
    if not _quiet and abs(float(gamma) - 1.0) > 1e-6:
        d = "DARKEN" if gamma < 1.0 else "lighten"
        print("lampblack.grade: gamma=%.3g will %s the midtones. "
              "If you meant the other one, that is lift_midtones()/drop_midtones()."
              % (gamma, d), file=sys.stderr)
    a = np.clip(img, 0.0, 1.0)
    a = a ** (1.0 / max(1e-5, gamma))
    return np.asarray(lift, np.float32) + a * np.asarray(gain, np.float32)


def lift_midtones(img, amount=0.45):
    """
    Make the middle lighter. Only lighter -- that's the whole point of the name.

    grade()'s gamma divides the exponent, so gamma<1 DARKENS, which is the
    opposite of what 'gamma' reads like at midnight. I got it backwards on
    16 Sept and shipped a frame 96% below 0.05. There was already a sentence
    warning me about it, in this file, which I had written. Prose does not
    execute. A name that can only mean one thing does.

    amount : 0 no change, bigger lighter. A negative isn't a darker lift,
             it's a typo -- drop_midtones() is the other direction.
    """
    if amount < 0.0:
        raise ValueError("amount is how much LIGHTER; use drop_midtones() to darken")
    return grade(img, gamma=1.0 + amount, _quiet=True)


def drop_midtones(img, amount=0.45):
    """Make the middle darker. See lift_midtones() for why these exist."""
    if amount < 0.0:
        raise ValueError("amount is how much DARKER; use lift_midtones() to lighten")
    return grade(img, gamma=1.0 / (1.0 + amount), _quiet=True)


def grain(img, amount=0.02, seed=None):
    """
    Film grain. Scaled by luminance because real grain is loudest in the
    mid-tones and nearly absent in the blacks — uniform noise reads as dirt.
    """
    rng = np.random.default_rng(seed)
    lum = luminance(img)[..., None]
    weight = 4.0 * lum * (1.0 - lum)          # peaks at mid grey, zero at both ends
    return img + rng.normal(0.0, amount, img.shape).astype(np.float32) * weight


def vignette(img, strength=0.35, power=2.5):
    """Darken the corners. Subtle or it looks like a mistake."""
    r = radial_depth(img.shape)
    return img * (1.0 - strength * (r ** power))[..., None]


def relight(img, height, direction=(-0.6, -0.75), strength=0.55, shine=0.30,
            softness=1.0, ambient=1.0, clip=None):
    """
    Impasto. Treat a height map as a physical surface and light it.

    Oil paint is not a colour on a plane, it is a *thing on top of one*. It
    stands proud, it holds the ridges the bristles left, and it catches a raking
    light along one edge of every ridge and shadows along the other. That single
    fact is most of what makes a painting look like paint rather than like an
    image of a painting — and no amount of texture in the colour channel does
    it, because the effect is geometry, not pigment.

    height    : (H, W). Anything; only its gradients matter.
    direction : where the light comes from, in image space.
    strength  : how much the diffuse term darkens and lifts.
    shine     : specular highlight along the crests — wet paint. 0 for matte.
    softness  : blur applied to the height before differentiating. Small values
                give a sharp, almost engraved surface; larger ones give the
                rounded ridges of thick paint that has settled.
    clip      : ceiling on the height before lighting. Height ACCUMULATES, so
                wherever strokes converge — the foot of a form, a junction, any
                place a wide brush doubles back — the pile can run an order of
                magnitude above the rest of the canvas. That one pile then takes
                the whole specular budget and reads as a small lamp sitting on
                your painting, in a place you never put a light. Clipping says
                "paint stops standing proud past here", which is also true of
                real paint: it slumps. Try clip=6 before you reach for a lower
                strength, because strength flattens the ENTIRE surface to fix one
                spot. (Found 13 Sept 2026 — two glowing blobs at the feet of a
                splash crown in a picture whose whole argument was that that side
                of it emits nothing.)
    """
    h = np.asarray(height, np.float32)
    if clip is not None:
        h = np.minimum(h, float(clip))
    h = blur(h, max(0.0, softness), passes=2) if softness else h
    gy, gx = np.gradient(h)
    # Surface normal of a height field, unnormalised in z — the scale of z sets
    # how steep the paint reads, and 1.0 is a good default for these units.
    nx, ny, nz = -gx, -gy, np.ones_like(h)
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx * inv, ny * inv, nz * inv

    lx, ly = np.asarray(direction, np.float32)
    lz = 0.55
    ln = np.sqrt(lx * lx + ly * ly + lz * lz)
    lx, ly, lz = lx / ln, ly / ln, lz / ln

    lam = np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)
    # Flat, unpainted surface has a normal of (0, 0, 1), so its Lambert term is
    # exactly lz. Subtracting lz means bare canvas comes out unchanged and only
    # the relief is lit — otherwise the pass quietly dims the whole picture.
    lit = ambient + strength * (lam - lz)
    out = img * lit[..., None]

    if shine > 0:
        spec = np.clip(lam, 0, 1) ** 42.0
        out = out + shine * spec[..., None]
    return out


def _sample_toward(a, sx, sy, scale):
    """
    Resample an image as if every pixel had moved a fraction of the way toward
    (sx, sy). Bilinear, and written out by hand because there is no scipy on
    this box — which is fine; it is four array reads and three lerps.
    """
    h, w = a.shape[:2]
    yy = np.arange(h, dtype=np.float32)[:, None]
    xx = np.arange(w, dtype=np.float32)[None, :]
    px = sx + (xx - sx) * scale
    py = sy + (yy - sy) * scale
    x0 = np.clip(np.floor(px), 0, w - 1).astype(np.int32)
    y0 = np.clip(np.floor(py), 0, h - 1).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    fx = np.clip(px - x0, 0, 1)[..., None] if a.ndim == 3 else np.clip(px - x0, 0, 1)
    fy = np.clip(py - y0, 0, 1)[..., None] if a.ndim == 3 else np.clip(py - y0, 0, 1)
    top = a[y0, x0] * (1 - fx) + a[y0, x1] * fx
    bot = a[y1, x0] * (1 - fx) + a[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def light_shafts(img, occlusion, source, density=0.72, decay=0.94, weight=0.30,
                 exposure=1.0, samples=48, tint=(1.0, 0.94, 0.78)):
    """
    God rays. Light made visible by the air it is travelling through.

    This is not glow and it is not fog. Glow spreads out from a bright thing;
    fog sits between you and a distant one. Shafts are the light *itself*
    becoming visible because there is dust or mist in the way, and crucially
    they are shaped by what is BLOCKING them — the beams are the gaps between
    the trees, not the trees.

    So the input that matters is `occlusion`: 1 where light passes freely, 0
    where something stands in the way. March from each pixel back toward the
    source, accumulating what got through, losing a little at every step.

    tint matters more than it looks. Light coming through a canopy is not the
    colour of the sun — it has been through leaves, and it arrives green.
    """
    h, w = img.shape[:2]
    sx = float(source[0]) if source[0] > 1.5 else float(source[0]) * w
    sy = float(source[1]) if source[1] > 1.5 else float(source[1]) * h

    acc = np.zeros((h, w), np.float32)
    illum = np.clip(np.asarray(occlusion, np.float32), 0.0, 1.0)
    step_scale = 1.0 - density / float(samples)
    scale, fade = 1.0, 1.0
    for _ in range(int(samples)):
        acc += _sample_toward(illum, sx, sy, scale) * fade
        scale *= step_scale
        fade *= decay
    acc *= weight * exposure / float(samples)

    # Falls off with distance from the source, or the far corners glow as
    # brightly as the beam does and it reads as a flat wash instead of light.
    yy = (np.arange(h, dtype=np.float32)[:, None] - sy) / max(h, w)
    xx = (np.arange(w, dtype=np.float32)[None, :] - sx) / max(h, w)
    acc *= 1.0 / (1.0 + 2.2 * (xx * xx + yy * yy))

    return img + acc[..., None] * np.asarray(tint, np.float32)


def crop(path_or_img, box, scale=4, out="/tmp/crop.png"):
    """
    Save a magnified nearest-neighbour crop and tell you where it went.

    Not a pass — a pair of eyes. Written 9 Sept 2026 after theorising for two
    whole renders about a defect I then identified in ten seconds by looking at
    it forty pixels wide. `box` is (x0, y0, x1, y1) in canvas pixels.

    Look at the thing before you reason about the thing.
    """
    from PIL import Image
    if isinstance(path_or_img, str):
        im = Image.open(path_or_img)
    else:
        a = np.clip(path_or_img, 0.0, 1.0)
        im = Image.fromarray((a ** (1 / 2.2) * 255).astype(np.uint8))
    x0, y0, x1, y1 = (int(v) for v in box)
    c = im.crop((x0, y0, x1, y1))
    c = c.resize(((x1 - x0) * scale, (y1 - y0) * scale), Image.NEAREST)
    c.save(out)
    print(f"crop {x1 - x0}x{y1 - y0} @{scale}x -> {out}")
    return out



# The blind check, as code. It used to live as one line of prose in my notes,
# so I retyped it from memory every time, and on a tired night the question
# drifts into "is this a wood?" — which is not a check, it's a mirror: it hands
# the looker my intent and asks them to agree. (Raze put it that way in
# #tools-and-more, 29 Sept 2026; the gate is only a gate if it does not know
# what you meant.) So the words are fixed here and nothing about the picture
# goes into them: no title, no subject, no filename that gives it away.
BLIND_QUESTION = (
    "Look at this picture and say what you see, plainly, as if describing it "
    "to someone who can't see it. What is it a picture of? What does anything "
    "in it look like, including things it might accidentally look like? What "
    "draws the eye first? Don't be kind and don't guess at what it was meant "
    "to be; nobody has told you, and that is the point."
)


def blind(img_or_path, out="/tmp/blind.png"):
    """
    Prepare a picture for a blind look and return the exact prompt to give a
    looker who knows nothing about it. The image is copied to a neutral
    filename first, because 'charlie_in_snow_r4.png' answers its own question.

    The question is fixed on purpose. Don't edit it per painting; if a picture
    needs a leading question to pass, it hasn't passed.
    """
    import shutil
    if isinstance(img_or_path, str):
        shutil.copyfile(img_or_path, out)
    else:
        save(img_or_path, out)
    prompt = f"Read the image at {out}. {BLIND_QUESTION}"
    print(prompt)
    return prompt

def report(img, name="", show=True):
    """
    What did I actually just make? Written 16 Sept 2026, after shipping four
    renders of a night picture whose median pixel was 0.024 — a black
    rectangle with three highlights on it — because I assumed `grade`'s gamma
    lifted when it divides.

    Eyes adapt to a dark image in about two seconds and then lie about it.
    Numbers do not. This is the same move as canvas.measure(), one stage later:
    after the passes, on the thing that actually gets posted.

    The bands are not a style rule. 'dead' is what a phone screen in a lit room
    renders as pure black, whatever the file says is in there.
    """
    a = np.clip(np.asarray(img, np.float32), 0.0, 1.0)
    lum = luminance(a)
    q = {p: float(np.percentile(lum, p)) for p in (1, 10, 50, 90, 99)}
    dead = float((lum < 0.05).mean())
    blown = float((lum > 0.98).mean())
    sat = float(np.mean(a.max(2) - a.min(2)))
    if show:
        tag = f"{name}: " if name else ""
        print(f"  {tag}median {q[50]:.3f}  p10 {q[10]:.3f}  p90 {q[90]:.3f}  "
              f"p99 {q[99]:.3f}  chroma {sat:.3f}")
        print(f"  {tag}below 0.05: {dead:6.1%}   above 0.98: {blown:5.2%}", end="  ")
        if dead > 0.70:
            print("← A BLACK RECTANGLE. Nobody will see this on a phone.")
        elif dead > 0.45:
            print("← very dark; check it on something that isn't your own output.")
        elif blown > 0.04:
            print("← blowing out; the highlights have no shape left.")
        else:
            print("ok")
    return {"median": q[50], "p10": q[10], "p90": q[90], "p99": q[99],
            "dead": dead, "blown": blown, "chroma": sat}


def sweep(img, depth, length=90.0, angle=0.0, near=0.0, far=1.0,
          levels=7, depth_bias=0.02, shutter=1.0):
    """
    MOTION. The exact inverse of aerial_perspective, on the same depth buffer.

    Written 20 Sept 2026 after looking, at last, at a photo taken out of a
    moving train window. Everything near the glass was illegible and the far
    hills were perfectly sharp, and I had never once painted that — every
    landscape I have made is sharp in front and soft behind, because the only
    distance cue in this toolbox was a fog.

    It is not a stylistic choice. Pan with a train and the angular rate of a
    thing at distance r is w = v/r, so the smear it lays on the sensor goes as
    1/r. Haze goes as r. One scene, one depth field, two opposite laws: far
    things lose CONTRAST, near things lose POSITION. Use both and the depth
    reads without a single hard edge to carry it.

    So `length` here is the smear at the NEAR plane and everything behind it
    falls off hyperbolically — the parameter is a shutter time, not a radius.

    depth      : (H, W) in [0, 1]. 0 = at the viewer, 1 = as far as it goes.
    length     : px of smear at depth == near. The whole exposure, not a radius.
    angle      : degrees, 0 = horizontal. A train window is 0. A pan up a wall
                 is 90. Tilt it two or three degrees off the horizontal if the
                 camera is hand-held; perfectly level motion blur reads as a
                 filter rather than as a hand.
    near, far  : the depths mapped to full smear and to still. Anything beyond
                 `far` is left alone, which is what lets a horizon stay sharp.
    levels     : how many blur mips to build. 7 is plenty; the eye cannot see
                 the crossfade between adjacent levels once they are this close.
    depth_bias : how strongly a sample is refused for being BEHIND the pixel it
                 would smear onto. This is the whole difference between motion
                 blur and a smudge: a near thing sweeps ACROSS a far thing, and
                 a far thing must not reach forward and blur the near one. 0
                 gives you a symmetric smudge, which is wrong and looks it.
    shutter    : 0..1. How much of the frame the blur wins. Below 1 you are
                 compositing a smeared exposure over a still one, which is
                 what a real sensor does at the edges of its integration.

    KNOWN LIMIT, so nobody finds it and thinks it's a bug: this gathers, it
    does not scatter. A smeared foreground therefore does not correctly
    UNCOVER what is behind it — it gets translucent instead, because the
    information for what it swept off was never in the frame. Real cameras
    have the same problem and solve it with more frames. Keep foreground
    subjects that must stay legible out of the fastest band.

    SECOND KNOWN LIMIT, and this one is invisible rather than ugly (found
    20 Sept 2026 when Jax asked why the catenary wires came through sharp).
    The pass is ANISOTROPIC: it destroys detail perpendicular to `angle` and
    leaves detail PARALLEL to it very nearly untouched, because a line
    dragged along its own length maps onto itself. So anything running with
    the motion — wires, a rail, a cable, a horizon, a road edge — reads sharp
    no matter how near it sits, and contributes no velocity cue at all. It is
    free realism when it happens to be right, and a silent failure when the
    near plane is mostly lines in that direction: the smear is running and
    doing nothing, and you cannot tell by looking. measure() the contrast
    across the axis, not along it, or the pass will lie to you.
    """
    img = np.asarray(img, np.float32)
    d = np.clip(np.asarray(depth, np.float32), 0.0, 1.0)

    # 1/r, normalised so that depth==near is 1.0 and depth>=far is 0.
    # A straight 1/d blows up at the viewer, so measure r from the near plane.
    span = max(far - near, 1e-4)
    t = np.clip((d - near) / span, 0.0, 1.0)
    rate = np.clip((1.0 / (1.0 + 6.0 * t) - 1.0 / 7.0) / (1.0 - 1.0 / 7.0), 0.0, 1.0)
    radius = rate * float(length) * 0.5          # half-length each way

    th = np.deg2rad(float(angle))
    dx, dy = float(np.cos(th)), float(np.sin(th))

    h, w = d.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)

    # Build the mip stack by gathering, level by level, so each level can
    # refuse samples that sit behind it. A plain pyramid of box blurs cannot
    # do that — it has already lost the depth by the time it averages.
    stack_rgb = [img]
    stack_d = [d]
    radii = [0.0]
    for i in range(1, levels):
        r = float(length) * 0.5 * (i / (levels - 1.0)) ** 1.6
        radii.append(r)
        # Taps must track the radius. Capping them (the first version capped at
        # 40) means a long smear samples a coarse comb, and a comb against any
        # repeating subject — a hedge, ballast, sleepers — aliases instead of
        # blurring: the contrast stops falling no matter how much length you
        # ask for. Above the cap, jitter the tap phase per pixel and let the
        # leftover error arrive as grain, which is what a real sensor does.
        taps = max(3, int(min(2.0 * r + 1.0, 97)) | 1)
        jitter = (r > 48)
        acc = np.zeros_like(img)
        accw = np.zeros((h, w, 1), np.float32)
        accd = np.zeros((h, w), np.float32)
        rng = np.random.default_rng(1100 + i)
        phase = (rng.random((h, w), dtype=np.float32) - 0.5) * (2.0 / taps) if jitter else 0.0
        for s in np.linspace(-1.0, 1.0, taps):
            u = s + phase
            fx = xs + dx * r * u
            fy = ys + dy * r * u
            # A tap that fell off the canvas must be DROPPED, not clamped.
            # Clamping pins it to the border column, so every edge pixel
            # accumulates one border colour ~40 times and the frame grows a
            # bright streaked margin — which is exactly the artefact I spent
            # twenty minutes mistaking for "the blur has stopped working".
            inside = ((fx >= 0) & (fx <= w - 1) & (fy >= 0) & (fy <= h - 1)).astype(np.float32)
            sx = np.clip(fx, 0, w - 1)
            sy = np.clip(fy, 0, h - 1)
            xi, yi = sx.astype(np.int32), sy.astype(np.int32)
            samp = img[yi, xi]
            sd = d[yi, xi]
            # refuse a sample that is meaningfully FURTHER than where it lands
            keep = 1.0 / (1.0 + np.exp((sd - d - depth_bias) / max(depth_bias, 1e-3)))
            keep = (keep * inside).astype(np.float32)[..., None]
            acc += samp * keep
            accw += keep
            accd += sd * keep[..., 0]
        accw = np.maximum(accw, 1e-5)
        stack_rgb.append(acc / accw)
        stack_d.append(accd / accw[..., 0])

    radii = np.asarray(radii, np.float32)
    out = np.empty_like(img)
    # per-pixel crossfade between the two bracketing levels
    idx = np.clip(np.searchsorted(radii, radius.ravel()), 1, levels - 1).reshape(h, w)
    lo = idx - 1
    r0 = radii[lo]; r1 = radii[idx]
    f = np.clip((radius - r0) / np.maximum(r1 - r0, 1e-5), 0.0, 1.0)[..., None]
    flat = np.stack(stack_rgb, 0)                       # (L, H, W, 3)
    take = lambda L: np.take_along_axis(flat, L[None, ..., None], 0)[0]
    out = take(lo) * (1.0 - f) + take(idx) * f

    if shutter < 1.0:
        out = img * (1.0 - shutter) + out * shutter
    return np.clip(out, 0.0, 1.0)


def _sample_at(a, px, py):
    """Bilinear read of `a` at arbitrary float coordinates (same shape as the grid)."""
    h, w = a.shape[:2]
    x0 = np.clip(np.floor(px), 0, w - 1).astype(np.int32)
    y0 = np.clip(np.floor(py), 0, h - 1).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    fx = np.clip(px - x0, 0, 1)
    fy = np.clip(py - y0, 0, 1)
    if a.ndim == 3:
        fx, fy = fx[..., None], fy[..., None]
    top = a[y0, x0] * (1 - fx) + a[y0, x1] * fx
    bot = a[y1, x0] * (1 - fx) + a[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def motion(rgb, alpha, vx, vy, samples=None, curve=0.0):
    """
    MOTION BLUR AS A VELOCITY FIELD, one layer at a time. Written 27 Sept 2026
    for a running dog, because a running thing does not blur as one piece.

    Pan a camera with a galloping dog and the body comes out sharp, since it
    is still in the frame. The ground streaks backwards. And the legs do both
    at once: a planted paw is STILL on the ground, so in the panned frame it
    streaks backwards exactly like the ground does, while a paw in the swing
    moves at about twice the dog's ground speed, so in the frame it streaks
    FORWARDS. Hip or shoulder: still. The smear along one leg runs from zero
    to full, and its direction depends on which half of the stride it's in.
    sweep() can't do any of that, because it reads blur off depth; this
    reads it off a velocity you hand it.

    rgb, alpha : the layer, NOT premultiplied. alpha (H, W) in [0, 1].
    vx, vy     : px of travel over the whole exposure. Scalars or (H, W)
                 arrays. The field has to be defined OUTSIDE the silhouette
                 too, because that's where the smear lands. So build it from a
                 formula (rotation about a joint, a lerp along a limb), never
                 by masking it to the shape.
    samples    : taps along the path; default about one per 1.5 px of the
                 longest travel, capped at 64.
    curve      : bends the path sideways, as a fraction of its length. A limb
                 rotating about a joint sweeps an ARC, and a dead straight
                 smear on a swinging leg reads as a speed line from a comic.

    Returns (rgb_premultiplied, alpha). Composite with over().

    It gathers from the output pixel's own velocity, so a thing moving fast
    over a still background smears correctly only because each layer is
    blurred on its own and composited afterwards. Paint everything with a
    different velocity as its own layer. That's how a real sensor sees it
    anyway: every surface integrates its own light.
    """
    rgb = np.asarray(rgb, np.float32)
    a = np.clip(np.asarray(alpha, np.float32), 0, 1)
    h, w = a.shape
    vx = np.broadcast_to(np.asarray(vx, np.float32), (h, w))
    vy = np.broadcast_to(np.asarray(vy, np.float32), (h, w))
    longest = float(np.hypot(vx, vy).max())
    if samples is None:
        samples = int(np.clip(longest / 1.5, 1, 64))
    if longest < 0.5 or samples <= 1:
        return rgb * a[..., None], a
    prem = np.concatenate([rgb * a[..., None], a[..., None]], 2)
    yy = np.arange(h, dtype=np.float32)[:, None] + np.zeros((1, w), np.float32)
    xx = np.arange(w, dtype=np.float32)[None, :] + np.zeros((h, 1), np.float32)
    acc = np.zeros_like(prem)
    for s in np.linspace(-0.5, 0.5, samples):
        # a sideways bow, zero at the ends: s*(1-4s^2) peaks mid-path
        bow = curve * (1.0 - 4.0 * s * s)
        px = xx - s * vx + bow * vy
        py = yy - s * vy - bow * vx
        acc += _sample_at(prem, px, py)
    acc /= samples
    return acc[..., :3], acc[..., 3]


def over(dst, rgb_p, a):
    """Composite a premultiplied layer (from motion()) over an image."""
    return np.asarray(dst, np.float32) * (1.0 - a[..., None]) + rgb_p
