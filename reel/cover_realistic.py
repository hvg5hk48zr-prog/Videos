#!/usr/bin/env python3
"""Higher-fidelity cover: smooth anatomical profile, skin/hair/fabric shading,
bump-lit rock wall with depth of field and a volumetric light shaft."""
import math
import os

import numpy as np
from PIL import Image, ImageDraw

import gfx
import render
import timeline as T

W, H = 1080, 1920
SS = 3
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
TITLES = {
    "en": ("cover_realistic.jpg", "BRAZIL’S\nREAL HEROES", "WERE FIGHTING\nFOR FREEDOM."),
    "pt": ("capa_realista_pt.jpg", "OS VERDADEIROS\nHERÓIS DO BRASIL", "LUTAVAM PELA\nLIBERDADE."),
}


def catmull(pts, n=12, closed=True):
    P = list(pts)
    m = len(P)
    out = []
    for i in range(m if closed else m - 1):
        p0, p1, p2, p3 = P[(i - 1) % m], P[i], P[(i + 1) % m], P[(i + 2) % m]
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(tuple(0.5 * ((2 * p1[k]) + (-p0[k] + p2[k]) * t + (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t2
                                    + (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * t3) for k in (0, 1)))
    return out


def mask_from(polys, blur_px=0):
    im = Image.new("L", (W * SS, H * SS), 0)
    d = ImageDraw.Draw(im)
    for pts, fill in polys:
        d.polygon([(x * SS, y * SS) for x, y in pts], fill=fill)
    m = np.asarray(im.resize((W, H), Image.BOX), np.float32) / 255
    return gfx.blur(m, blur_px) if blur_px else m


def ellipse(cx, cy, rx, ry, ang=0.0):
    return gfx.ell_pts(cx, cy, rx, ry, ang, 64)


# Profile facing right. Units: head height, origin head centre, y down.
PROFILE = [(-0.05, -0.52), (0.15, -0.50), (0.33, -0.38), (0.40, -0.22), (0.435, -0.10), (0.41, -0.035),
           (0.47, 0.04), (0.535, 0.115), (0.525, 0.16), (0.455, 0.18), (0.458, 0.215), (0.478, 0.25),
           (0.447, 0.283), (0.472, 0.318), (0.443, 0.36), (0.428, 0.385), (0.452, 0.45), (0.42, 0.515),
           (0.30, 0.56), (0.20, 0.60), (0.17, 0.70), (0.19, 0.86), (0.25, 0.98), (0.45, 1.12), (0.62, 1.35),
           (0.70, 1.7), (0.74, 2.6), (0.74, 3.6), (-1.1, 3.6), (-1.06, 2.0), (-0.95, 1.36), (-0.70, 1.06),
           (-0.42, 0.92), (-0.27, 0.72), (-0.25, 0.52), (-0.38, 0.28), (-0.47, 0.02), (-0.44, -0.25), (-0.30, -0.45)]
HAIRLINE = [(-0.05, -0.56), (0.18, -0.54), (0.36, -0.40), (0.36, -0.30), (0.20, -0.30), (0.05, -0.22),
            (-0.02, -0.05), (-0.12, 0.05), (-0.30, 0.20), (-0.42, 0.30), (-0.52, 0.0), (-0.48, -0.30), (-0.32, -0.50)]
SHIRT = [(0.21, 0.94), (0.30, 1.03), (0.20, 1.10), (0.05, 1.02), (-0.15, 0.95), (-0.32, 0.86), (-0.45, 0.93),
         (-0.75, 1.06), (-1.2, 1.3), (-1.2, 3.7), (0.9, 3.7), (0.8, 1.3), (0.62, 1.20), (0.42, 1.05)]


def place(pts, cx, cy, s, tilt):
    out = []
    for px, py in pts:
        if py < 0.75:
            px, py = gfx.rot((px, py), tilt, (0.0, 0.72))
        out.append((cx + px * s, cy + py * s))
    return out


def radial(cx, cy, rx, ry=None):
    ry = rx if ry is None else ry
    return np.exp(-(((XX - cx) / rx) ** 2 + ((YY - cy) / ry) ** 2)).astype(np.float32)


def beam(x0, y0, x1, y1, w0, w1):
    dx, dy = x1 - x0, y1 - y0
    ln = math.hypot(dx, dy)
    ux, uy = dx / ln, dy / ln
    px, py = XX - x0, YY - y0
    along = (px * ux + py * uy) / ln
    perp = np.abs(-px * uy + py * ux)
    wid = w0 + (w1 - w0) * np.clip(along, 0, 1)
    return (np.exp(-(perp / wid) ** 2) * gfx.smooth(along * 4) * np.clip(1.3 - along, 0, 1)).astype(np.float32)


def fbm(a, seed, ax=1.0):
    return gfx.fbm(H, W, a, seed, ax=ax)


def main(lang="en"):
    name, title, sub = TITLES[lang]
    cx, cy, s, tilt = 0.30 * W, 0.30 * H, 0.215 * H, -0.20
    L = np.array([0.82, -0.57])                      # toward the light shaft (upper right)

    # ---- background: bump-lit rock wall, out of focus
    hgt = fbm(1.25, 11) * 0.7 + fbm(0.8, 12) * 0.3
    gy, gx = np.gradient(gfx.blur(hgt, 2))
    lam = np.clip(0.55 + (-gx * L[0] - gy * L[1]) * 260, 0, 1.4)
    spill = radial(0.75 * W, 0.45 * H, 0.55 * W, 0.7 * H)
    torch = radial(0.02 * W, 0.80 * H, 0.35 * W, 0.25 * H)
    rock = (0.35 + 0.65 * hgt)[..., None] * lam[..., None]
    bg = rock * (gfx.col((0.16, 0.11, 0.07)) * (0.15 + 1.4 * spill[..., None]) + gfx.col((0.5, 0.22, 0.06)) * torch[..., None] * 0.8)
    bg = np.stack([gfx.blur(bg[..., k], 7) for k in range(3)], -1)

    # ---- volumetric shaft with streaks, and the opening above
    sh = beam(0.76 * W, -0.06 * H, 0.64 * W, 1.0 * H, 70, 260)
    streak = gfx.blur(gfx.fbm(H, W, 1.0, 21, ax=0.08), 3)
    bg += (sh * (0.55 + 0.6 * streak))[..., None] * gfx.col((1.0, 0.74, 0.40)) * 0.75
    bg += radial(0.76 * W, -0.01 * H, 0.22 * W, 0.05 * H)[..., None] * gfx.col((1.0, 0.92, 0.75)) * 1.6

    # ---- figure masks
    body = mask_from([(catmull(place(PROFILE, cx, cy, s, tilt)), 255)])
    hair = mask_from([(catmull(place(HAIRLINE, cx, cy, s, tilt)), 255)], 2) * body
    shirt = mask_from([(catmull(place(SHIRT, cx, cy, s, tilt)), 255)], 1) * body
    skin = np.clip(body - shirt, 0, 1)

    # soft form shading from the mask's own curvature, plus a crisp rim
    broad = gfx.shade(body, L, 45, 300)
    mid = gfx.shade(body, L, 14, 110)
    rim = gfx.shade(body, L, 3, 22)
    fill = gfx.shade(body, (-0.9, 0.2), 30, 200) * 0.25    # warm bounce from the torch side

    skin_c = gfx.col((0.36, 0.20, 0.12))
    img = bg * (1 - body[..., None])
    tone = 0.05 + 0.55 * broad + 0.8 * mid
    pores = 0.92 + 0.08 * gfx.fbm(H, W, 0.4, 31)
    img += (skin[..., None] * skin_c * (tone * pores)[..., None])
    img += (skin[..., None] * gfx.col((1.0, 0.70, 0.40)) * (rim * 1.25)[..., None])
    img += (skin[..., None] * gfx.col((0.6, 0.25, 0.08)) * fill[..., None])
    # specular sheen on forehead, nose, cheekbone, lips
    for (px, py, rx, ry, a) in ((0.30, -0.28, 0.06, 0.10, 0.3), (0.47, 0.08, 0.04, 0.03, 0.6),
                                (0.30, 0.10, 0.06, 0.04, 0.4), (0.42, 0.30, 0.03, 0.015, 0.4)):
        x, y = place([(px, py)], cx, cy, s, tilt)[0]
        img += (skin * radial(x, y, rx * s, ry * s) * mid)[..., None] * gfx.col((1.0, 0.82, 0.6)) * a

    # eye: socket shadow, lid line, a small catchlight from the shaft
    ex, ey = place([(0.33, -0.015)], cx, cy, s, tilt)[0]
    img *= (1 - (radial(ex - 0.02 * s, ey, 0.07 * s, 0.045 * s) * 0.65 * skin))[..., None]
    img += (radial(ex + 0.025 * s, ey - 0.004 * s, 0.006 * s, 0.005 * s) * 0.9)[..., None] * gfx.col((1.0, 0.9, 0.75))
    # mouth seam and nostril
    for (px, py, rx, ry) in ((0.43, 0.284, 0.035, 0.005), (0.47, 0.165, 0.018, 0.008)):
        x, y = place([(px, py)], cx, cy, s, tilt)[0]
        img *= (1 - radial(x, y, rx * s, ry * s) * 0.7 * skin)[..., None]
    # ear
    ex2, ey2 = place([(-0.08, 0.06)], cx, cy, s, tilt)[0]
    ear = mask_from([(ellipse(ex2, ey2, 0.07 * s, 0.12 * s, 0.25), 255)], 2) * skin
    img += (gfx.shade(ear, L, 4, 30) * 0.35)[..., None] * gfx.col((1.0, 0.7, 0.45))
    img *= (1 - mask_from([(ellipse(ex2 + 0.01 * s, ey2, 0.03 * s, 0.06 * s, 0.25), 255)], 3) * skin * 0.5)[..., None]

    # short tightly coiled hair: dark, matte, with a broken rim
    coil = gfx.fbm(H, W, 0.15, 41)
    hair_tone = 0.03 + 0.10 * coil + 0.35 * mid * (coil > 0.55)
    img = img * (1 - hair[..., None]) + hair[..., None] * gfx.col((0.10, 0.07, 0.05)) * hair_tone[..., None]
    img += (hair * rim * (coil > 0.4))[..., None] * gfx.col((1.0, 0.75, 0.45)) * 0.8

    # rough linen shirt with folds
    folds = gfx.blur(gfx.fbm(H, W, 1.1, 51, ax=0.35), 4)
    weave = 0.9 + 0.1 * gfx.fbm(H, W, 0.2, 52)
    stone = (0.06 + 0.5 * broad + 0.6 * mid) * (0.6 + 0.6 * folds) * weave
    img = img * (1 - shirt[..., None]) + shirt[..., None] * gfx.col((0.42, 0.36, 0.28)) * stone[..., None]
    img += (shirt * rim)[..., None] * gfx.col((1.0, 0.82, 0.55)) * 0.7

    # light wrap and foreground dust (some in focus, some as bokeh)
    img += (gfx.blur(sh, 20) * gfx.blur(body, 12) * (1 - body))[..., None] * gfx.col((1.0, 0.75, 0.4)) * 0.25
    r = np.random.default_rng(3)
    n = 900
    xs, ys = r.uniform(0, W, n), r.uniform(0, H, n)
    v = r.uniform(0.3, 1.0, n) * sh[ys.astype(int), xs.astype(int)]
    buf = np.zeros((H, W), np.float32)
    np.add.at(buf, (ys.astype(int), xs.astype(int)), v)
    img += (gfx.blur(buf, 1) * 6)[..., None] * gfx.col((1.0, 0.85, 0.6))
    xs, ys = r.uniform(0.45 * W, W, 40), r.uniform(0.1 * H, 0.9 * H, 40)
    for x, y in zip(xs, ys):
        img += (radial(x, y, r.uniform(12, 30)) * r.uniform(0.05, 0.18) * sh[int(y), int(x)] * 3)[..., None] * gfx.col((1.0, 0.8, 0.5))

    # ---- grade, vignette, grain, title
    img = np.clip(img, 0, None)
    img = img / (1 + img * 0.35) * 1.25                      # soft highlight roll-off
    img = np.clip(img, 0, 1)
    img = img + (img * img * (3 - 2 * img) - img) * 0.35
    img *= render.VIGNETTE
    img += render.GRAIN[0] * 0.035
    img *= 0.85
    for txt, font, size, color, y, tr, glow in (
            (title, T.SERIF, 128 if lang == "en" else 104, T.GOLDC, 0.56, 0.05, 0.6),
            (sub, T.SANS_B, 66, T.WHITE, 0.695, 0.12, 0.25),
            ("CHICO REI  ·  DRAGÃO DO MAR", T.SANS_M, 32, (0.9, 0.85, 0.75), 0.775, 0.22, 0.0)):
        render.draw_text(img, render.text_layer(txt, font, size, color, tr, glow), y, 1.0)
    out = os.path.join(OUT_DIR, name)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(out, quality=93)
    print(out)


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "en")
