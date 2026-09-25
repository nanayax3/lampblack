"""Pine, In Its Own Soot — 23 Sept 2026, Pi morning. First painting in sumi.py."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, sumi, lampblack as lb

W, H = 900, 1300
s = sumi.Sheet(W, H, seed=23)
rng = np.random.default_rng(9)

# far wash: a low hill in very thin wet ink, laid first so it bleeds most
Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
ridge = 1080 + 22 * np.sin(X / 150 + 1.3) - 70 * np.exp(-((X - 600) / 180) ** 2)
body = np.clip((Y - ridge) / 30, 0, 1) * np.clip(1 - (Y - ridge) / 260, 0, 1)
s.water += body * 1.6
s.fine += body * 0.10
s.coarse += body * 0.035
s.flow(120)

# the branch: one dry, confident stroke from the top-left, then a second limb
def curve(p0, p1, p2, n=40):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * np.array(p0) + 2 * (1 - t) * t * np.array(p1) + t ** 2 * np.array(p2)

trunk = curve((-30, 180), (330, 120), (560, 430))
s.stroke(trunk, width=30, ink=1.5, water=0.5, fine=0.3, dry=0.8,
         press=lambda u: 1.0 - 0.55 * u)
limb = curve((352, 230), (435, 420), (335, 640))
s.stroke(limb, width=15, ink=1.3, water=0.4, fine=0.3, dry=0.45,
         press=lambda u: 1.0 - 0.6 * u)
twig = curve((520, 370), (640, 380), (720, 330))
s.stroke(twig, width=9, ink=1.2, water=0.4, dry=0.35, press=lambda u: 1.0 - 0.6 * u)

# needle clusters: fans of short strokes from a point, darker near, paler behind
def cluster(cx, cy, n, length, ink, water, spread=np.pi * 0.75, turn=-np.pi / 2):
    for a in np.linspace(turn - spread / 2, turn + spread / 2, n) + rng.normal(0, 0.06, n):
        L = length * rng.uniform(0.75, 1.1)
        end = (cx + np.cos(a) * L, cy + np.sin(a) * L)
        s.stroke([(cx, cy), ((cx + end[0]) / 2, (cy + end[1]) / 2 + 3), end], width=3.2,
                 ink=ink, water=water, fine=0.4, press=lambda u: 1.0 - 0.8 * u)

# back layer: pale and wet, so it blurs into the mist
for cx, cy in [(250, 120), (470, 300), (400, 560), (690, 290), (150, 220)]:
    cluster(cx, cy, 16, 95, ink=0.35, water=1.4, turn=rng.uniform(-2.2, -0.9))
s.flow(40)
# front layer: dense and nearly dry, so it keeps its edges
for cx, cy in [(300, 170), (540, 410), (330, 630), (720, 330), (430, 250), (380, 470)]:
    cluster(cx, cy, 19, 80, ink=1.4, water=0.25, turn=rng.uniform(-2.0, -1.0))
s.flow(12)

img = s.render("ao", age=0.6)

# the seal — cinnabar, lower right, the only colour on the sheet
x0, y0, sz = 760, 1150, 54
seal = np.zeros((H, W), np.float32); seal[y0:y0 + sz, x0:x0 + sz] = 1
chew = lb.blur(np.random.default_rng(4).random((H, W, 1)).astype(np.float32), 1)[..., 0]
cx, cy = x0 + sz / 2, y0 + sz / 2
def seg(p, q, w=3.2):
    px, py = X - p[0], Y - p[1]; dx, dy = q[0] - p[0], q[1] - p[1]
    t = np.clip((px * dx + py * dy) / (dx * dx + dy * dy), 0, 1)
    return np.hypot(px - t * dx, py - t * dy) < w
inner = (seg((cx, y0 + 9), (cx, y0 + sz - 8)) | seg((x0 + 11, y0 + 20), (x0 + sz - 11, y0 + 20)) |
         seg((cx, y0 + 22), (x0 + 10, y0 + sz - 12)) | seg((cx, y0 + 22), (x0 + sz - 10, y0 + sz - 12)))
seal = seal * (chew > 0.38) * (~inner)
red = np.array([0.72, 0.16, 0.10], np.float32)
img = img * (1 - 0.85 * seal[..., None]) + red * 0.85 * seal[..., None] * img / np.maximum(img, 1e-3) * 0.9

img = lb.grain(img, amount=0.012, seed=2)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pine_in_its_own_soot.png")
lb.save(img, out)
print(out)
