"""Low-level drawing helpers: noise textures, blur, masks, silhouettes, props.

Everything here works at the *work* resolution (w x h); masks are drawn at
SS x supersampling and box-downsampled for antialiasing.
"""
import math

import numpy as np
from PIL import Image, ImageDraw

w, h = 720, 1280          # work resolution (upscaled to 1080x1920 at output)
SS = 2                    # mask supersampling

YY, XX = np.mgrid[0:h, 0:w].astype(np.float32)


# ---------------------------------------------------------------- utilities
def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def sm(x):
    """Scalar smoothstep."""
    return float(smooth(x))


def lerp(a, b, t):
    return a + (b - a) * t


def col(c):
    return np.asarray(c, np.float32)


def fbm(hh, ww, a=1.2, seed=0, ax=1.0, ay=1.0):
    """Tileable fractal noise via 1/f^a spectral shaping, normalised to 0..1."""
    r = np.random.default_rng(seed)
    n = r.standard_normal((hh, ww))
    fy = np.fft.fftfreq(hh)[:, None] * ay
    fx = np.fft.fftfreq(ww)[None, :] * ax
    f = np.sqrt(fx ** 2 + fy ** 2)
    f[0, 0] = 1.0
    out = np.real(np.fft.ifft2(np.fft.fft2(n) / f ** a))
    lo, hi = np.percentile(out, [1, 99])
    return np.clip((out - lo) / (hi - lo), 0, 1).astype(np.float32)


ROCK = fbm(h, w, 1.15, 1)
ROCK2 = fbm(h, w, 0.9, 2)
CLOUD = fbm(h, w, 1.35, 3, ax=0.45)
SEA = fbm(512, 512, 1.05, 4, ax=3.5)


def _box1(a, r, axis):
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r + 1, r)
    c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis, dtype=np.float32)
    n = a.shape[axis]
    sl_hi = [slice(None)] * a.ndim
    sl_lo = [slice(None)] * a.ndim
    sl_hi[axis] = slice(2 * r + 1, 2 * r + 1 + n)
    sl_lo[axis] = slice(0, n)
    return (c[tuple(sl_hi)] - c[tuple(sl_lo)]) / (2 * r + 1)


def blur(a, r):
    r = int(r)
    if r < 1:
        return a
    for _ in range(2):
        a = _box1(a, r, 0)
        a = _box1(a, r, 1)
    return a


def vgrad(c1, c2, y0=0.0, y1=1.0, gamma=1.0):
    t = np.clip((np.linspace(0, 1, h, dtype=np.float32) - y0) / (y1 - y0), 0, 1) ** gamma
    g = col(c1)[None, :] * (1 - t[:, None]) + col(c2)[None, :] * t[:, None]
    return np.repeat(g[:, None, :], w, axis=1)


def radial(cx, cy, rx, ry=None):
    ry = rx if ry is None else ry
    return np.exp(-(((XX - cx) / rx) ** 2 + ((YY - cy) / ry) ** 2)).astype(np.float32)


def comp(img, m, c):
    m = m[..., None]
    return img * (1 - m) + col(c) * m


def add(img, m, c):
    return img + m[..., None] * col(c)


def shade(mask, L, r=6, k=None):
    """Directional edge light on a silhouette. L points toward the light (screen coords)."""
    b = blur(mask, r)
    gy, gx = np.gradient(b)
    k = 6.0 * r if k is None else k
    return np.clip(-(gx * L[0] + gy * L[1]) * k, 0, 1) * mask


def splat(xs, ys, vals, r=2):
    """Additive soft points."""
    buf = np.zeros((h, w), np.float32)
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    np.add.at(buf, (ys[ok].astype(int), xs[ok].astype(int)), np.asarray(vals, np.float32)[ok])
    return blur(buf, r) * (2 * r + 1) ** 2 * 0.35 if r > 0 else buf


def beam(x0, y0, x1, y1, w0, w1):
    """Soft light shaft from (x0,y0) to (x1,y1), widening from w0 to w1."""
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    px, py = XX - x0, YY - y0
    along = (px * ux + py * uy) / L
    perp = np.abs(px * -uy + py * ux)
    wid = w0 + (w1 - w0) * np.clip(along, 0, 1)
    m = np.exp(-(perp / wid) ** 2) * smooth(along * 4) * np.clip(1.3 - along, 0, 1)
    return m.astype(np.float32)


def rays(cx, cy, n=18, seed=0, sharp=6.0):
    ang = np.arctan2(YY - cy, XX - cx)
    r = np.random.default_rng(seed)
    v = np.zeros_like(ang)
    for k, ph in zip(r.integers(5, 30, n), r.random(n) * 6.28):
        v += np.cos(ang * k + ph)
    v = (v / n * 2 + 0.5)
    return np.clip(v, 0, 1) ** sharp


# ---------------------------------------------------------------- canvas
class Canvas:
    def __init__(self):
        self.im = Image.new("L", (w * SS, h * SS), 0)
        self.d = ImageDraw.Draw(self.im)

    def line(self, a, b, wd, fill=255, caps=True):
        self.d.line([a[0] * SS, a[1] * SS, b[0] * SS, b[1] * SS], fill=fill, width=max(1, int(wd * SS)))
        if caps:
            self.ell(a[0], a[1], wd / 2, wd / 2, fill)
            self.ell(b[0], b[1], wd / 2, wd / 2, fill)

    def ell(self, cx, cy, rx, ry, fill=255):
        self.d.ellipse([(cx - rx) * SS, (cy - ry) * SS, (cx + rx) * SS, (cy + ry) * SS], fill=fill)

    def ring(self, cx, cy, rx, ry, wd, fill=255):
        self.d.ellipse([(cx - rx) * SS, (cy - ry) * SS, (cx + rx) * SS, (cy + ry) * SS],
                       outline=fill, width=max(1, int(wd * SS)))

    def poly(self, pts, fill=255):
        self.d.polygon([(x * SS, y * SS) for x, y in pts], fill=fill)

    def rect(self, x0, y0, x1, y1, fill=255):
        self.d.rectangle([x0 * SS, y0 * SS, x1 * SS, y1 * SS], fill=fill)

    def arr(self):
        return np.asarray(self.im.resize((w, h), Image.BOX), np.float32) / 255.0


def ell_pts(cx, cy, rx, ry, ang=0.0, n=48, a0=0.0, a1=2 * math.pi):
    th = np.linspace(a0, a1, n)
    x, y = rx * np.cos(th), ry * np.sin(th)
    ca, sa = math.cos(ang), math.sin(ang)
    return [(cx + xi * ca - yi * sa, cy + xi * sa + yi * ca) for xi, yi in zip(x, y)]


def rot(p, ang, c=(0, 0)):
    ca, sa = math.cos(ang), math.sin(ang)
    x, y = p[0] - c[0], p[1] - c[1]
    return (c[0] + x * ca - y * sa, c[1] + x * sa + y * ca)


# ---------------------------------------------------------------- figures
# Poses in units of figure height, x forward (facing direction), y up from the feet.
_SIDE = dict(head=(0.0, 0.93), neck=(0.0, 0.85), lsh=(0.0, 0.81), rsh=(0.0, 0.81), hip=(0.0, 0.52),
             lel=(0.03, 0.65), lha=(0.04, 0.49), rel=(-0.02, 0.65), rha=(-0.01, 0.49),
             lkn=(0.03, 0.27), lft=(0.02, 0.0), rkn=(-0.02, 0.27), rft=(-0.04, 0.0))
POSES = {
    "side": _SIDE,
    "tall": dict(_SIDE, head=(0.015, 0.935)),
    "front": dict(head=(0.0, 0.93), neck=(0.0, 0.85), lsh=(-0.11, 0.81), rsh=(0.11, 0.81), hip=(0.0, 0.52),
                  lel=(-0.125, 0.65), lha=(-0.125, 0.49), rel=(0.125, 0.65), rha=(0.125, 0.49),
                  lkn=(-0.05, 0.27), lft=(-0.06, 0.0), rkn=(0.05, 0.27), rft=(0.06, 0.0)),
    "pick_up": dict(head=(0.06, 0.86), neck=(0.03, 0.79), lsh=(0.02, 0.76), rsh=(0.02, 0.76), hip=(-0.05, 0.50),
                    lel=(-0.02, 0.92), lha=(-0.06, 1.0), rel=(0.0, 0.90), rha=(-0.05, 0.99),
                    lkn=(0.07, 0.27), lft=(0.10, 0.0), rkn=(-0.10, 0.26), rft=(-0.16, 0.0)),
    "pick_down": dict(head=(0.22, 0.70), neck=(0.16, 0.66), lsh=(0.15, 0.64), rsh=(0.15, 0.64), hip=(-0.03, 0.47),
                      lel=(0.24, 0.52), lha=(0.30, 0.38), rel=(0.22, 0.50), rha=(0.28, 0.37),
                      lkn=(0.10, 0.25), lft=(0.10, 0.0), rkn=(-0.10, 0.24), rft=(-0.16, 0.0)),
    "carry": dict(head=(0.03, 0.91), neck=(0.02, 0.83), lsh=(0.02, 0.80), rsh=(0.02, 0.80), hip=(0.0, 0.51),
                  lel=(0.09, 0.92), lha=(0.04, 1.02), rel=(-0.05, 0.92), rha=(-0.01, 1.02),
                  lkn=(0.03, 0.27), lft=(0.02, 0.0), rkn=(-0.02, 0.27), rft=(-0.04, 0.0)),
}


def blend(p1, p2, t):
    return {k: (lerp(p1[k][0], p2[k][0], t), lerp(p1[k][1], p2[k][1], t)) for k in p1}


def walk(pose, ph, amp=1.0):
    p = dict(pose)
    s = math.sin(ph) * amp
    p["lft"] = (0.13 * s, 0.03 * max(0.0, math.cos(ph)))
    p["rft"] = (-0.13 * s, 0.03 * max(0.0, -math.cos(ph)))
    p["lkn"] = (0.08 * s + 0.03, 0.27)
    p["rkn"] = (-0.08 * s + 0.03, 0.27)
    p["hip"] = (p["hip"][0], p["hip"][1] + 0.01 * abs(math.cos(ph)))
    if "lel" in p and pose is not POSES["carry"]:
        p["lel"] = (0.02 - 0.05 * s, 0.65)
        p["lha"] = (0.02 - 0.11 * s, 0.50)
        p["rel"] = (0.02 + 0.05 * s, 0.65)
        p["rha"] = (0.02 + 0.11 * s, 0.50)
    return p


def figure(c, x, yb, hh, pose, face=1, front=False, tilt=0.0, profile=False, fill=255, build=1.0):
    """Draw a human silhouette; returns joint positions in screen coords."""
    P = {k: (x + face * v[0] * hh, yb - v[1] * hh) for k, v in pose.items()}
    b = build
    if front:
        hl = (P["hip"][0] - 0.045 * hh, P["hip"][1])
        hr = (P["hip"][0] + 0.045 * hh, P["hip"][1])
    else:
        hl = hr = P["hip"]
    for hp, kn, ft in ((hl, "lkn", "lft"), (hr, "rkn", "rft")):
        c.line(hp, P[kn], 0.09 * hh * b, fill)
        c.line(P[kn], P[ft], 0.06 * hh * b, fill)
        c.ell(P[kn][0], P[kn][1] - 0.03 * hh, 0.05 * hh * b, 0.06 * hh, fill)   # loose trousers
    if not front:
        for k in ("lft", "rft"):
            fx, fy = P[k]
            c.line((fx - 0.015 * hh * face, fy), (fx + 0.06 * hh * face, fy), 0.035 * hh, fill)
    c.line(P["neck"], P["hip"], 0.15 * hh * b, fill)
    if front:
        wy = P["hip"][1] - 0.12 * hh
        c.poly([(P["lsh"][0], P["lsh"][1] - 0.02 * hh), (P["rsh"][0], P["rsh"][1] - 0.02 * hh),
                (P["hip"][0] + 0.075 * hh * b, wy), (P["hip"][0] + 0.095 * hh * b, P["hip"][1] + 0.03 * hh),
                (P["hip"][0] - 0.095 * hh * b, P["hip"][1] + 0.03 * hh), (P["hip"][0] - 0.075 * hh * b, wy)], fill)
        for k in ("lsh", "rsh"):
            c.ell(P[k][0], P[k][1] + 0.005 * hh, 0.04 * hh * b, 0.035 * hh, fill)
        c.poly([(P["neck"][0] - 0.03 * hh, P["neck"][1]), (P["neck"][0] + 0.03 * hh, P["neck"][1]),
                (P["rsh"][0] - 0.02 * hh, P["rsh"][1] - 0.025 * hh), (P["lsh"][0] + 0.02 * hh, P["lsh"][1] - 0.025 * hh)], fill)
    else:
        c.line(P["lsh"], P["rsh"], 0.09 * hh * b, fill)
    c.line(P["lsh"], P["lel"], 0.056 * hh * b, fill)
    c.line(P["lel"], P["lha"], 0.048 * hh * b, fill)
    c.line(P["rsh"], P["rel"], 0.056 * hh * b, fill)
    c.line(P["rel"], P["rha"], 0.048 * hh * b, fill)
    c.line(P["neck"], P["head"], 0.045 * hh, fill)
    hx, hy = P["head"]
    rx, ry = 0.05 * hh, 0.064 * hh
    a = tilt * face
    c.poly(ell_pts(hx, hy, rx, ry, a), fill)
    if profile:
        # brow, nose and chin give the head a direction without caricature
        pts = [(0.80, -0.30), (1.16, 0.02), (0.92, 0.20), (0.95, 0.42), (0.75, 0.80), (0.30, 0.95)]
        pts = [rot((hx + face * px * rx, hy + py * ry), a, (hx, hy)) for px, py in pts]
        c.poly([(hx, hy)] + pts, fill)
    return P


def pick(c, P, t_up, face, hh, fill=255):
    """Pickaxe held in both hands; t_up=1 raised, 0 struck. Returns tip position."""
    hx = (P["lha"][0] + P["rha"][0]) / 2
    hy = (P["lha"][1] + P["rha"][1]) / 2
    a_up, a_dn = math.radians(200), math.radians(-70)   # handle angle, facing right
    a = lerp(a_dn, a_up, t_up)
    dx, dy = math.cos(a) * face, -math.sin(a)
    L = 0.34 * hh
    ex, ey = hx + dx * L, hy + dy * L
    c.line((hx - dx * 0.04 * hh, hy - dy * 0.04 * hh), (ex, ey), 0.018 * hh, fill)
    px, py = -dy, dx
    c.poly([(ex + px * 0.10 * hh, ey + py * 0.10 * hh), (ex + dx * 0.02 * hh, ey + dy * 0.02 * hh),
            (ex - px * 0.10 * hh, ey - py * 0.10 * hh), (ex - dx * 0.02 * hh, ey - dy * 0.02 * hh)], fill)
    return ex + px * 0.10 * hh, ey + py * 0.10 * hh


def bust(c, cx, cy, s, variant=0, fill=255):
    """Frontal head-and-shoulders silhouette. s = head height in px."""
    rx, ry = s * (0.40, 0.38, 0.41)[variant % 3], s * 0.5
    c.poly(ell_pts(cx, cy, rx, ry), fill)                                   # cranium/face
    c.poly(ell_pts(cx, cy + ry * 0.45, rx * 0.78, ry * 0.62), fill)          # jaw
    c.ell(cx - rx * 0.98, cy + ry * 0.05, rx * 0.13, ry * 0.2, fill)         # ears
    c.ell(cx + rx * 0.98, cy + ry * 0.05, rx * 0.13, ry * 0.2, fill)
    if variant == 2:                                                        # elder: short beard
        c.poly(ell_pts(cx, cy + ry * 0.70, rx * 0.62, ry * 0.42), fill)
    if variant == 1:                                                        # close-cropped hair line
        c.poly(ell_pts(cx, cy - ry * 0.18, rx * 1.04, ry * 0.86, 0, 48, math.pi, 2 * math.pi), fill)
    nw = rx * 0.62
    c.rect(cx - nw, cy + ry * 0.6, cx + nw, cy + ry * 1.7, fill)             # neck
    sw = s * (1.55, 1.4, 1.5)[variant % 3]
    sy = cy + ry * (1.65, 1.6, 1.75)[variant % 3]
    c.poly(ell_pts(cx, sy + s * 0.9, sw, s * 1.1, 0, 64, math.pi, 2 * math.pi) +
           [(cx + sw, h * 2), (cx - sw, h * 2)], fill)                      # shoulders
    c.poly([(cx - nw * 1.05, cy + ry * 1.2), (cx + nw * 1.05, cy + ry * 1.2),
            (cx + sw * 0.75, sy + s * 0.05), (cx - sw * 0.75, sy + s * 0.05)], fill)  # trapezius


# ---------------------------------------------------------------- props
def jangada(c, x, yb, L, face=1, sail=True, mast=True, sail_c=None, sway=0.0):
    """Jangada (raft with a tall triangular sail). L = raft length px."""
    th = L * 0.045
    pts = [(x - L / 2, yb - th), (x + L * 0.42, yb - th), (x + L / 2, yb - th * 2.2),
           (x + L * 0.47, yb), (x - L / 2, yb)]
    c.poly([(x + (px - x) * face, py) for px, py in pts])
    for k in range(3):                     # bench / log lines read as texture via the gaps
        pass
    mx = x + face * L * 0.12
    Hm = L * 1.05
    top = (mx + face * L * 0.04 + sway, yb - th - Hm)
    if mast:
        c.line((mx, yb - th), top, max(1.2, L * 0.016))
    if sail and sail_c is not None:
        boom = (mx - face * L * 0.72, yb - th - Hm * 0.13)
        belly = (mx - face * L * 0.38 + sway * 0.5, yb - th - Hm * 0.62)
        sail_c.poly([top, (mx, yb - th - Hm * 0.10), boom, belly])
        c.line((mx, yb - th - Hm * 0.11), boom, max(1.0, L * 0.012))
        c.line(top, (x + face * L * 0.48, yb - th * 2.2), max(0.6, L * 0.004), caps=False)


def ship(c, x, yb, s, face=1, sails=0.0):
    """Square-rigged sailing ship silhouette. s = hull length px."""
    hh = s * 0.16
    pts = [(-0.5, -0.25), (-0.46, 0.55), (0.40, 0.55), (0.52, -0.05), (0.60, -0.35), (0.30, -0.15), (-0.40, -0.15)]
    c.poly([(x + face * px * s, yb - hh * 0.55 + py * hh) for px, py in pts])
    c.line((x + face * 0.48 * s, yb - hh * 0.6), (x + face * 0.85 * s, yb - hh * 1.6), max(1, s * 0.012))  # bowsprit
    for mx, mh in ((-0.28, 1.55), (0.05, 1.85), (0.32, 1.5)):
        bx = x + face * mx * s
        c.line((bx, yb - hh * 0.7), (bx, yb - s * mh * 0.62), max(1, s * 0.013))
        for k, yy in enumerate((0.33, 0.48, 0.60)):
            yw = s * (0.20 - k * 0.045)
            ty = yb - s * mh * yy
            c.line((bx - yw, ty), (bx + yw, ty), max(1, s * 0.008), caps=False)
            if sails > 0:
                c.poly([(bx - yw * 0.95, ty), (bx + yw * 0.95, ty),
                        (bx + yw * 0.85, ty + s * 0.11 * sails), (bx - yw * 0.85, ty + s * 0.11 * sails)])
    c.line((x + face * 0.85 * s, yb - hh * 1.6), (x + face * 0.05 * s, yb - s * 1.85 * 0.62), max(0.6, s * 0.004), caps=False)
    c.line((x - face * 0.5 * s, yb - hh * 0.8), (x - face * 0.28 * s, yb - s * 1.55 * 0.62), max(0.6, s * 0.004), caps=False)


def chain(c, x0, y0, x1, y1, link, wd, sag=0.0, phase=0):
    """Chain of alternating links between two points with a parabolic sag."""
    L = math.hypot(x1 - x0, y1 - y0)
    n = max(1, int(L / (link * 1.55)))
    for i in range(n + 1):
        t = i / n if n else 0
        px = lerp(x0, x1, t)
        py = lerp(y0, y1, t) + sag * 4 * t * (1 - t)
        tx = (x1 - x0) / L
        ty = (y1 - y0 + sag * 4 * (1 - 2 * t)) / L
        a = math.atan2(ty, tx)
        if (i + phase) % 2 == 0:
            pts = ell_pts(px, py, link, link * 0.58, a, 32)
            c.d.line([(p[0] * SS, p[1] * SS) for p in pts], fill=255, width=max(1, int(wd * SS)), joint="curve")
        else:
            c.line(rot((px - link * 0.8, py), a, (px, py)), rot((px + link * 0.8, py), a, (px, py)), wd * 1.1)


def skyline(c, y, seed=5, x0=0, x1=w, hmax=0.22):
    r = np.random.default_rng(seed)
    x = x0
    tops = []
    while x < x1:
        bw = r.uniform(22, 64)
        bh = h * hmax * r.choice([0.25, 0.4, 0.55, 0.7, 1.0], p=[0.25, 0.25, 0.2, 0.2, 0.1]) * r.uniform(0.8, 1.1)
        gap = r.uniform(1, 14)
        c.rect(x, y - bh, x + bw - gap, y)
        if r.random() < 0.35:                        # setback / rooftop
            c.rect(x + bw * 0.25, y - bh - r.uniform(8, 30), x + bw * 0.6, y)
        if r.random() < 0.15:
            c.line((x + bw * 0.4, y - bh), (x + bw * 0.4, y - bh - 40), 2)
        tops.append((x, y - bh, x + bw - gap, y))
        x += bw
    return tops


# Side-profile head-and-shoulders, facing +x. Units: head height = 1, origin = head centre, y down.
_PROFILE = [(0.0, -0.5), (0.30, -0.42), (0.40, -0.25), (0.43, -0.12), (0.41, -0.05), (0.53, 0.08), (0.52, 0.12),
            (0.44, 0.15), (0.465, 0.20), (0.43, 0.245), (0.455, 0.29), (0.39, 0.35), (0.42, 0.43), (0.36, 0.49),
            (0.26, 0.53), (0.17, 0.62), (0.19, 0.95), (0.55, 1.22), (0.78, 1.6), (0.86, 3.2), (-0.95, 3.2),
            (-0.92, 1.45), (-0.62, 1.07), (-0.27, 0.80), (-0.27, 0.50), (-0.40, 0.22), (-0.46, -0.10), (-0.34, -0.40)]


def profile_bust(c, cx, cy, s, face=1, tilt=0.0, fill=255):
    """Large side-profile silhouette for close-ups. s = head height px; tilt<0 lifts the chin."""
    pts = []
    for px, py in _PROFILE:
        x, y = px * face, py
        if py < 0.75:
            x, y = rot((x, y), tilt * face, (0.0, 0.7))
        pts.append((cx + x * s, cy + y * s))
    c.poly(pts, fill)


def sample(tex, u, v):
    """Bilinear, wrapping texture lookup."""
    n, m = tex.shape
    u0 = np.floor(u)
    v0 = np.floor(v)
    fu = (u - u0).astype(np.float32)
    fv = (v - v0).astype(np.float32)
    u0 = u0.astype(np.int32) % m
    v0 = v0.astype(np.int32) % n
    u1 = (u0 + 1) % m
    v1 = (v0 + 1) % n
    return (tex[v0, u0] * (1 - fu) + tex[v0, u1] * fu) * (1 - fv) + (tex[v1, u0] * (1 - fu) + tex[v1, u1] * fu) * fv


# Standing person seen from the front/back: right-half outline, mirrored. Units of height, y up.
_PERSON = [(0.025, 0.865), (0.07, 0.838), (0.125, 0.82), (0.155, 0.795), (0.168, 0.74), (0.172, 0.62),
           (0.168, 0.52), (0.163, 0.45), (0.168, 0.42), (0.152, 0.385), (0.13, 0.395), (0.127, 0.45),
           (0.127, 0.55), (0.12, 0.66), (0.112, 0.70), (0.104, 0.62), (0.100, 0.53), (0.106, 0.47),
           (0.102, 0.40), (0.088, 0.27), (0.072, 0.12), (0.066, 0.04), (0.08, 0.0), (0.02, 0.0),
           (0.024, 0.05), (0.026, 0.25), (0.0, 0.44)]


def person(c, x, yb, hh, width=1.0, stride=0.0, fill=255):
    """Natural standing silhouette (front or back view). stride shifts the feet for walking."""
    right = []
    for px, py in _PERSON:
        dx = stride * (0.25 - py) * 0.5 if py < 0.44 else 0.0
        right.append((x + (px * width + dx) * hh, yb - py * hh))
    left = []
    for px, py in reversed(_PERSON):
        dx = -stride * (0.25 - py) * 0.5 if py < 0.44 else 0.0
        left.append((x + (-px * width + dx) * hh, yb - py * hh))
    c.poly(right + left, fill)
    c.poly(ell_pts(x, yb - 0.93 * hh, 0.052 * hh * width ** 0.5, 0.066 * hh), fill)
    c.line((x, yb - 0.90 * hh), (x, yb - 0.83 * hh), 0.05 * hh, fill)
