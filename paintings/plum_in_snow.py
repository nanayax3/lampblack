"""Plum in Snow — 24 Sept 2026, Pi morning. Third of the Three Friends of Winter.

Agreed with Nana on the 23rd: a branch in late-winter snow, the snow left as
bare paper, a few red blossoms the ONLY colour in the whole set.

The rules I painted to (Song-era, repeated by everyone since, Zhang Daqian
included): 贵稀不贵繁 贵老不贵嫩 贵瘦不贵肥 贵含不贵开 — prize the sparse
over the crowded, the old over the young, the lean over the fat, the bud over
the open flower. And from the Mustard Seed Garden: no two petals of a flower
point the same way.

The snow is not painted. The sky is washed grey around it (Wang Mian did it
to his petals), so the white on top of the branches is the paper itself."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, sumi, lampblack as lb

W, H = 900, 1300
s = sumi.Sheet(W, H, seed=37)
rng = np.random.default_rng(21)
Y, X = np.mgrid[0:H, 0:W].astype(np.float32)

def curve(*pts, n=80):
    # chain of quadratic beziers through control triples
    pts = [np.array(p, np.float32) for p in pts]
    out = []
    for i in range(0, len(pts) - 2, 2):
        t = np.linspace(0, 1, n)[:, None]
        out.append((1 - t) ** 2 * pts[i] + 2 * (1 - t) * t * pts[i + 1] + t ** 2 * pts[i + 2])
    return np.concatenate(out)

def wobble(path, amt, seed):
    # old wood doesn't run smooth: a slow sideways drift along the spine
    r = np.random.default_rng(seed)
    t = np.gradient(path, axis=0); t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-6
    nrm = np.stack([-t[:, 1], t[:, 0]], 1)
    n = len(path)
    off = np.interp(np.arange(n), np.linspace(0, n - 1, 9), r.normal(0, amt, 9))
    return path + nrm * off[:, None]

GROUND = 1150
cover = np.zeros((H, W), np.float32)   # where wood is, for the snow to find

def wood(path, width, ink, dry, side=0.35, water=0.35, press=None, seed=None):
    global cover
    c = s.stroke(path, width=width, ink=ink, water=water, fine=0.3, dry=dry,
                 side=side, press=press, seed=seed)
    cover = np.maximum(cover, c)

taper = lambda a: (lambda u: 1.0 - a * u)

# ── the old trunk: out of the snow at lower left, leaning, turning into the limb ──
# (renders 1–6 had it snapped off at the top. Every version was a broom, a cloud
# or — render 6 — a penguin. It bends into the limb now, like most old plums do.)
trunk = wobble(curve((170, 1340), (150, 1050), (250, 850)), 7, 1)
n = len(trunk)
wood(trunk[:int(n * 0.85)], 70, 1.1, dry=0.28, side=0.3, press=lambda u: 1.0 - 0.25 * u, seed=3)
wood(trunk[int(n * 0.75):], 54, 1.15, dry=0.1, side=0.3, press=lambda u: 1.0 - 0.3 * u, seed=33)
wood(wobble(trunk, 5, 2)[:int(n * 0.7)] + [6, 0], 40, 0.7, dry=0.75, side=None, press=taper(0.3), seed=4)

# ── the main limb: from the lean, up and right, thinning ──
limb = wobble(curve(tuple(trunk[-1]), (420, 700), (520, 560), (610, 420), (720, 300)), 5, 6)
wood(limb, 40, 1.15, dry=0.4, press=lambda u: 1.0 - 0.7 * u ** 0.8, seed=7)
# low limb, reaching right over the snow and dipping
low = wobble(curve((200, 990), (400, 930), (560, 950), (690, 965), (800, 910)), 4, 8)
wood(low, 24, 1.1, dry=0.35, press=taper(0.6), seed=9)

s.flow(12)

# ── new shoots: this year's wood. long, straight, quick, dark, one stroke each ──
def shoot(p0, ang, length, width=8, bend=0.08, seed=None):
    p0 = np.array(p0, np.float32)
    a2 = ang + bend
    mid = p0 + np.array([np.cos(ang), np.sin(ang)]) * length * 0.5
    end = mid + np.array([np.cos(a2), np.sin(a2)]) * length * 0.5
    wood(curve(p0, mid, end, n=60), width, 0.95, dry=0.3, side=0.25, water=0.3,
         press=lambda u: 1.0 - 0.8 * u ** 1.3, seed=seed)
    return p0, mid, end

shoots = [
    shoot(limb[140], -1.55, 330, 9, -0.10, 11),   # straight up off the limb
    shoot(limb[-1], -0.95, 230, 7, 0.10, 12),     # carrying on from the tip
    shoot(limb[100], -2.25, 250, 7, 0.22, 13),    # back left, crossing the upright: 女
    shoot(low[-1], -0.55, 90, 5, 0.1, 14),
    shoot(low[110], -1.25, 160, 6, -0.15, 15),
    shoot(trunk[62], -2.7, 120, 6, 0.25, 16),    # a short one off the trunk, left
]
s.flow(6)

# ── snow on the wood: a lumpy cap on whatever faces up ──
up = np.roll(cover, -9, 0)      # wood 9px below → we're on or just above it
down = np.roll(cover, 7, 0)     # wood 7px above → we're deep in / under it
lump = lb.blur(rng.random((H, W, 1)).astype(np.float32), 5)[..., 0]
lump = (lump - lump.mean()) / (lump.std() + 1e-6)
cap = np.clip((up * 1.6 - down * 1.3) * (0.8 + 0.25 * lump), 0, 1)
# falling snow: an accident in render 1 (cap noise leaked into the sky), kept
# on purpose and turned down. Small, sparse, bare paper.
fine_n = lb.blur(rng.random((H, W, 1)).astype(np.float32), 3)[..., 0]
fine_n = (fine_n - fine_n.mean()) / (fine_n.std() + 1e-6)
flakes = np.clip((fine_n - 2.45) * 2.5, 0, 1)
cap = lb.blur(cap[..., None], 1)[..., 0]
cap[:40] = 0
# the ground: a bank of snow, its edge a slow line
edge = GROUND + 25 * np.sin(X[0] / 170 + 0.8) + 10 * np.sin(X[0] / 53)
ground = np.clip((Y - edge[None, :]) / 6, 0, 1)

# ── blossoms. sparse. mostly buds. ──
flowers = []   # (x, y, r, kind)
def along(path, f):
    return path[min(len(path) - 1, int(f * (len(path) - 1)))]
sh = [np.concatenate([curve(p0, mid, end, n=60)]) for p0, mid, end in shoots]
spots = [
    (sh[0], 0.30, "open"), (sh[0], 0.55, "bud"), (sh[0], 0.78, "bud"), (sh[0], 0.93, "bud"),
    (sh[1], 0.40, "open"), (sh[1], 0.75, "bud"),
    (sh[2], 0.35, "half"), (sh[2], 0.62, "bud"), (sh[2], 0.88, "bud"),
    (limb, 0.72, "open"), (limb, 0.55, "bud"),
    (sh[4], 0.5, "open"), (sh[4], 0.85, "bud"),
    (low, 0.62, "half"), (sh[3], 0.7, "bud"),
    (sh[5], 0.6, "bud"),
]
for path, f, kind in spots:
    x, y = along(path, f)
    side = rng.choice([-1, 1])
    t = along(path, min(1, f + 0.02)) - along(path, max(0, f - 0.02))
    t = t / (np.linalg.norm(t) + 1e-6)
    off = np.array([-t[1], t[0]]) * side
    r = {"open": 17, "half": 14, "bud": 5.0}[kind] * rng.uniform(0.85, 1.1)
    c = np.array([x, y]) + off * (r * 0.85 + 2)
    flowers.append((c[0], c[1], r, kind, off))

reserve = np.zeros((H, W), np.float32)
for x, y, r, k, _ in flowers:
    reserve = np.maximum(reserve, np.clip((r * 0.95 - np.hypot(X - x, Y - y)) / 3, 0, 1))

# ── the sky: grey, heavier at the top, washed around the snow ──
sky = (1.0 - 0.45 * Y / H) * (1 + 0.12 * lump)
alpha = sky * (1 - cap) * (1 - flakes) * (1 - ground) * (1 - reserve)
s.lift(cap * 0.97)
s.wash(alpha, ink=0.065, water=0.5, fine=0.75)
s.flow(20, spread=0.10)
# the wash's water carried fine soot back into the caps — on real paper you'd
# have left them dry and they'd have stayed clean. Take it back off, softly.
s.lift(lb.blur(cap[..., None], 2)[..., 0] * 0.9)

# ── now the red: rouge, boneless, five petals each, no two pointing alike ──
for x, y, r, kind, _ in flowers:
    if kind == "bud":
        s.stroke([(x, y + r * 0.6), (x, y - r * 0.7)], width=r * 2, ink=0.8, water=0.4, tint="yanzhi",
                 press=lambda u: 1.0 - 0.45 * u)
        continue
    n = 5 if kind == "open" else 3
    base = rng.uniform(0, 2 * np.pi)
    angs = base + np.arange(n) * 2 * np.pi / 5 + rng.normal(0, 0.18, n)
    for a in angs:
        pr = r * rng.uniform(0.5, 0.62)
        cx, cy = x + np.cos(a) * r * 0.48, y + np.sin(a) * r * 0.48
        d = np.array([np.cos(a + 0.4), np.sin(a + 0.4)]) * pr * 0.35
        s.stroke([(cx - d[0], cy - d[1]), (cx + d[0], cy + d[1])], width=pr * 2,
                 ink=rng.uniform(0.38, 0.62), water=0.6, tint="yanzhi",
                 press=lambda u: 0.85 + 0.15 * np.sin(u * np.pi))
s.flow(8, spread=0.12)

# stamens and calyx: the black that makes the red read
for x, y, r, kind, off in flowers:
    if kind == "bud":
        # the calyx holds the bud to the twig, so it sits on the twig side
        k = np.array([x, y]) - off * r * 0.8
        s.stroke([k - off[::-1] * [1, -1] * 3, k + off[::-1] * [1, -1] * 3], width=4, ink=1.4, water=0.1)
        continue
    m = 7 if kind == "open" else 4
    for a in rng.uniform(0, 2 * np.pi) + np.arange(m) * 2 * np.pi / m + rng.normal(0, 0.25, m):
        L = r * rng.uniform(0.55, 0.85)
        e = (x + np.cos(a) * L, y + np.sin(a) * L)
        s.stroke([(x, y), e], width=1.0, ink=0.6, water=0.05, press=lambda u: 1 - 0.5 * u)
        s.stroke([e, (e[0] + 0.5, e[1] + 0.5)], width=2.6, ink=1.3, water=0.05, press=lambda u: 1.0)

# moss dots (苔点) on the old trunk — and none where the snow sits
for f in (0.35, 0.62):   # two knots, a few dots each, not scattered like buttons
    k = along(trunk, f) + rng.normal(0, 10, 2)
    for _ in range(rng.integers(2, 4)):
        p = k + rng.normal(0, 7, 2)
        if cap[int(p[1]), int(p[0])] > 0.3: continue
        s.stroke([p, p + rng.normal(0, 1.5, 2)], width=rng.uniform(3, 5.5), ink=1.5, water=0.25,
                 press=lambda u: 1.0)
s.flow(3)

img = s.render("ao", age=0.6)

# ── the seal: same size, same place. 梅 — 木 beside 每 ──
x0, y0, sz = 760, 1150, 54
seal = np.zeros((H, W), np.float32); seal[y0:y0 + sz, x0:x0 + sz] = 1
chew = lb.blur(np.random.default_rng(4).random((H, W, 1)).astype(np.float32), 1)[..., 0]
def seg(p, q, w=2.6):
    px, py = X - (x0 + p[0]), Y - (y0 + p[1]); dx, dy = q[0] - p[0], q[1] - p[1]
    t = np.clip((px * dx + py * dy) / (dx * dx + dy * dy), 0, 1)
    return np.hypot(px - t * dx, py - t * dy) < w
inner = (seg((14, 8), (14, 47)) | seg((6, 18), (22, 18)) |              # 木
         seg((14, 20), (6, 36)) | seg((14, 22), (21, 31)))
inner |= seg((31, 7), (27, 15)) | seg((30, 11), (47, 11))                # 𠂉
inner |= (seg((29, 18), (27, 44)) | seg((29, 18), (46, 18)) |            # 母
          seg((46, 18), (44, 45)) | seg((25, 31), (49, 31)) | seg((27, 44), (44, 45)) |
          seg((37, 22), (38, 26), 2.2) | seg((37, 35), (38, 39), 2.2))
seal = seal * (chew > 0.38) * (~inner)
red = np.array([0.72, 0.16, 0.10], np.float32)
img = img * (1 - 0.85 * seal[..., None]) + red * 0.85 * seal[..., None] * img / np.maximum(img, 1e-3) * 0.9

img = lb.grain(img, amount=0.012, seed=3)
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "plum_in_snow.png")
lb.save(img, out)
print(out)
