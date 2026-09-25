"""Bamboo — 23 Sept 2026, Pi evening. Second of the Three Friends of Winter.
Pale culm, dark leaves (濃葉淡竿). Culms laid with the brush on its side."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, sumi, lampblack as lb

W, H = 900, 1300
s = sumi.Sheet(W, H, seed=24)
rng = np.random.default_rng(11)
Y, X = np.mgrid[0:H, 0:W].astype(np.float32)

def curve(p0, p1, p2, n=40):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * np.array(p0) + 2 * (1 - t) * t * np.array(p1) + t ** 2 * np.array(p2)

# a culm: segments bottom to top, each one stroke that pauses at both ends
def culm(base, top, bow, n, width, ink, water, gap=7, side=0.55, dry=0.15):
    spine = curve(base, ((base[0] + top[0]) / 2 + bow, (base[1] + top[1]) / 2), top, 200)
    L = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(spine, axis=0).T))])
    # internodes: short at the foot, long in the middle, shorter toward the top
    w = np.sin(np.linspace(0.35, 2.7, n)) + 0.25
    cuts = np.concatenate([[0], np.cumsum(w / w.sum())]) * L[-1]
    nodes = []
    for i in range(n):
        a, b = cuts[i] + gap / 2, cuts[i + 1] - gap / 2
        k = (L >= a) & (L <= b)
        seg = spine[k]
        if len(seg) < 3: continue
        s.stroke(seg, width=width * (1 - 0.25 * i / n), ink=ink, water=water, fine=0.35,
                 side=side, dry=dry, press=lambda u: 0.93 + 0.07 * np.cos(u * 2 * np.pi))
        nodes.append((spine[k][-1], spine[min(len(spine) - 1, np.argmax(k) + k.sum())]))
    return spine, cuts, nodes

def node_mark(p, q, width, ink):
    # the joint. Nana, 23 Sept: the old lids read as hard black ruled lines, and
    # they were. In real sumi the joint is a separate flick that doesn't quite
    # touch either segment: press at one side, lift across, a small turn at the
    # end. Wet enough to feather, short of the culm's edges.
    c = (np.array(p) + np.array(q)) / 2
    t = np.array(q) - np.array(p); t /= np.linalg.norm(t) + 1e-6
    nrm = np.array([-t[1], t[0]])
    hw = width * 0.5
    a0 = c - nrm * hw * 0.95 + t * 1.5
    mid = c - t * 2.5
    a1 = c + nrm * hw * 0.8 + t * 0.5
    s.stroke([a0, mid, a1, a1 + t * 3 + nrm * 1.5], width=6, ink=ink * 0.48, water=0.55, fine=0.5,
             press=lambda u: np.where(u < 0.15, 0.6 + 0.4 * u / 0.15, 1.0 - 0.75 * (u - 0.15) / 0.85))

def leaf(root, ang, length, width, ink, water):
    # press in, swell, then lift to a point: the stroke is the leaf
    bend = rng.uniform(-0.25, 0.25)
    a1 = ang + bend
    p0 = np.array(root, np.float32)
    p1 = p0 + np.array([np.cos(ang), np.sin(ang)]) * length * 0.5
    p2 = p1 + np.array([np.cos(a1), np.sin(a1)]) * length * 0.5
    s.stroke([p0, p1, p2], width=width, ink=ink, water=water, fine=0.4,
             press=lambda u: np.where(u < 0.28, 0.35 + 0.65 * (u / 0.28) ** 0.7,
                                      np.clip((1 - u) / 0.72, 0, 1) ** 0.8))

def cluster(root, base_ang, count, length, width, ink, water):
    # 个 and 介: three or four leaves from one point, fanned, never parallel
    spread = {3: [-0.55, 0.0, 0.5], 4: [-0.75, -0.2, 0.25, 0.7], 2: [-0.3, 0.35]}[count]
    for d in spread:
        leaf(root, base_ang + d + rng.normal(0, 0.06), length * rng.uniform(0.8, 1.15),
             width, ink, water)

# ── back culm: thin, pale, wet — behind glass ──
_, _, nodes_b = culm((610, 1360), (700, -60), 40, 8, 26, ink=0.30, water=1.1)
for p, q in nodes_b[1:]:
    node_mark(p, q, 26, ink=0.35)
for (p, q), a in zip(nodes_b[3:7], [-2.3, -0.6, -2.5, -0.5]):
    cluster(q, a + 0.9, 3, 100, 20, ink=0.35, water=1.2)
s.flow(60)

# ── front culm: the one that bends ──
spine, cuts, nodes = culm((300, 1360), (430, -60), -70, 7, 44, ink=0.6, water=0.5, side=0.7)
s.flow(10)
for p, q in nodes[1:]:
    node_mark(p, q, 44, ink=1.5)

# twigs: short, angled up off the node, one kink; leaves hang from the end
def twig(p, ang, length, w=4.5):
    p = np.array(p, np.float32)
    k = p + np.array([np.cos(ang), np.sin(ang)]) * length * 0.55
    a2 = ang + rng.uniform(-0.35, 0.35)
    e = k + np.array([np.cos(a2), np.sin(a2)]) * length * 0.45
    # twigs taper hard and go a little dry, so they read as brush and not wire
    s.stroke([p, k, e], width=w * 1.2, ink=0.8, water=0.3, fine=0.35, dry=0.3,
             press=lambda u: 1 - 0.8 * u ** 0.8)
    return k, e

def spray(end, knee, hang, n, length, ink):
    # leaves from staggered roots near the twig end, darkest in front
    for i in range(n):
        f = rng.uniform(0.55, 1.0)
        root = knee + (end - knee) * f
        a = hang + rng.normal(0, 0.45)
        leaf(root, a, length * rng.uniform(0.75, 1.15), rng.uniform(15, 20),
             ink * rng.uniform(0.65, 1.1), water=rng.uniform(0.3, 0.6))

k, e = twig(nodes[4][1], -0.75, 150)
spray(e, k, 1.05, 5, 165, 1.35)
k, e = twig(nodes[5][1], -2.35, 130, 4)
spray(e, k, 2.1, 4, 150, 1.35)
k, e = twig(nodes[3][1], -0.55, 110, 4)
spray(e, k, 0.75, 3, 150, 1.2)
k, e = twig(nodes[2][0], -2.6, 70, 3.5)
spray(e, k, 2.4, 2, 120, 0.9)
s.flow(14)

img = s.render("ao", age=0.6)

# the seal — same size, same place, same chewed edge as the pine's. 竹.
x0, y0, sz = 760, 1150, 54
seal = np.zeros((H, W), np.float32); seal[y0:y0 + sz, x0:x0 + sz] = 1
chew = lb.blur(np.random.default_rng(4).random((H, W, 1)).astype(np.float32), 1)[..., 0]
def seg(p, q, w=3.0):
    px, py = X - (x0 + p[0]), Y - (y0 + p[1]); dx, dy = q[0] - p[0], q[1] - p[1]
    t = np.clip((px * dx + py * dy) / (dx * dx + dy * dy), 0, 1)
    return np.hypot(px - t * dx, py - t * dy) < w
inner = np.zeros((H, W), bool)
for ox in (0, 24):   # two halves, each a 个 with its stem
    inner |= seg((ox + 14, 9), (ox + 8, 19)) | seg((ox + 10, 17), (ox + 26, 17)) | seg((ox + 19, 17), (ox + 19, 45))
inner |= seg((43, 45), (39, 42), 2.6)   # the hook on the right stem
seal = seal * (chew > 0.38) * (~inner)
red = np.array([0.72, 0.16, 0.10], np.float32)
img = img * (1 - 0.85 * seal[..., None]) + red * 0.85 * seal[..., None] * img / np.maximum(img, 1e-3) * 0.9

img = lb.grain(img, amount=0.012, seed=3)
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "bamboo_bends.png")
lb.save(img, out)
print(out)
