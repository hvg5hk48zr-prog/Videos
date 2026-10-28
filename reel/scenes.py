"""Shot renderers. Each returns a float32 (h, w, 3) linear-ish RGB frame in 0..1+.

Signature: shot(lt, d, gt) where lt = time into the shot, d = shot length,
gt = global reel time (used where motion must stay in sync with the audio).
"""
import math

import numpy as np

from gfx import (CLOUD, person, profile_bust, sample, POSES, ROCK, ROCK2, SEA, Canvas, XX, YY, add, beam, blend, blur, bust, chain,
                 col, comp, ell_pts, figure, h, jangada, lerp, pick, radial, rays, rot, shade, ship,
                 skyline, sm, smooth, splat, vgrad, w, walk)

SEA_PERIOD = 2.9          # seconds between breaking waves (shared with audio)
SEA_CRASH = 0.78          # phase of the crash within a period


def sea_crash_times(t0, t1):
    k0 = int(t0 // SEA_PERIOD) - 1
    out = []
    for k in range(k0, k0 + 20):
        tc = (k + SEA_CRASH) * SEA_PERIOD
        if t0 <= tc < t1:
            out.append(tc)
    return out


# ================================================================ scene 1 flashes
def town(lt, d, gt):
    img = vgrad((0.05, 0.03, 0.05), (0.75, 0.30, 0.10), 0.0, 0.7)
    img = add(img, radial(0.62 * w, 0.62 * h, 0.4 * w), (0.6, 0.22, 0.04))
    img *= (0.7 + 0.45 * np.roll(CLOUD, int(lt * 30), 1))[..., None]
    c = Canvas()
    xs = np.linspace(-10, w + 10, 60)
    ridge = 0.66 * h - 0.07 * h * np.sin(xs / w * math.pi * 1.1 + 0.2) - 6 * np.sin(xs / 37)
    c.poly(list(zip(xs, ridge)) + [(w + 10, h), (-10, h)])
    cx, top = 0.52 * w, 0.585 * h
    c.rect(cx - 0.13 * w, top - 0.09 * h, cx + 0.13 * w, top + 0.02 * h)
    c.poly([(cx - 0.08 * w, top - 0.09 * h), (cx, top - 0.14 * h), (cx + 0.08 * w, top - 0.09 * h)])
    for sx in (-1, 1):
        tx = cx + sx * 0.11 * w
        c.rect(tx - 0.035 * w, top - 0.20 * h, tx + 0.035 * w, top)
        c.poly(ell_pts(tx, top - 0.20 * h, 0.035 * w, 0.025 * h, 0, 24, math.pi, 2 * math.pi))
        c.line((tx, top - 0.225 * h), (tx, top - 0.26 * h), 4)
        c.line((tx - 0.012 * w, top - 0.248 * h), (tx + 0.012 * w, top - 0.248 * h), 3)
    r = np.random.default_rng(11)
    for _ in range(26):
        hx = r.uniform(0, w)
        base = float(np.interp(hx, xs, ridge)) + r.uniform(10, 120)
        bw, bh = r.uniform(30, 70), r.uniform(25, 45)
        c.rect(hx, base - bh, hx + bw, base)
        c.poly([(hx - 6, base - bh), (hx + bw / 2, base - bh - 18), (hx + bw + 6, base - bh)])
    m = c.arr()
    return comp(img, m, (0.02, 0.012, 0.012))


def mine_bg(lt, torch=(0.80, 0.40), dark=1.0, tunnel=(0.36, 0.47)):
    fl = 0.86 + 0.08 * math.sin(lt * 23) + 0.06 * math.sin(lt * 37 + 1.3)
    glow = radial(torch[0] * w, torch[1] * h, 0.62 * w, 0.5 * h) * fl
    tex = ROCK * 0.7 + ROCK2 * 0.3
    img = (tex[..., None] * col((0.20, 0.12, 0.07))) * (0.18 + 1.9 * glow[..., None]) * dark
    img *= (1 - 0.88 * radial(tunnel[0] * w, tunnel[1] * h, 0.20 * w, 0.17 * h))[..., None]
    floor = smooth((YY - 0.80 * h) / (0.2 * h))
    img = img * (1 - floor[..., None] * 0.4) + floor[..., None] * col((0.10, 0.065, 0.04)) * (0.4 + glow[..., None])
    img = add(img, radial(torch[0] * w, torch[1] * h, 14, 22) * fl * 2.5, (1.0, 0.6, 0.2))
    img = add(img, radial(torch[0] * w, torch[1] * h, 60) * fl * 0.6, (1.0, 0.45, 0.1))
    return img


WORKERS = [  # x, feet y, height, facing, swing rate (Hz), phase offset
    (0.25, 0.90, 0.46, 1, 0.95, 0.0),
    (0.64, 0.83, 0.33, -1, 0.80, 0.37),
    (0.84, 0.76, 0.21, -1, 1.10, 0.66),
]


def swing_state(gt, rate, off):
    ph = (gt * rate + off) % 1.0
    if ph < 0.12:
        b = 0.0
    elif ph < 0.62:
        b = sm((ph - 0.12) / 0.5)
    elif ph < 0.72:
        b = 1 - ((ph - 0.62) / 0.10) ** 2
    else:
        b = 0.0
    since = ((ph - 0.72) % 1.0) / rate
    return b, since


def pick_strike_times(t0, t1):
    out = []
    for i, (_, _, hh, _, rate, off) in enumerate(WORKERS):
        k = math.floor(t0 * rate + off)
        while True:
            t = (k + 0.72 - off) / rate
            if t >= t1:
                break
            if t >= t0:
                out.append((t, i))
            k += 1
    return sorted(out)


def mine_workers(lt, d, gt):
    img = mine_bg(lt)
    sparks_x, sparks_y, sparks_v = [], [], []
    for i in (2, 1, 0):
        x, yb, hh, face, rate, off = WORKERS[i]
        x, yb, hh = x * w, yb * h, hh * h
        b, since = swing_state(gt, rate, off)
        c = Canvas()
        P = figure(c, x, yb, hh, blend(POSES["pick_down"], POSES["pick_up"], b), face, profile=True)
        tip = pick(c, P, b, face, hh)
        m = c.arr()
        haze = (0.10, 0.065, 0.04) if i else (0.012, 0.008, 0.006)
        img = comp(img, m, col(haze) * (1.6 if i == 2 else 1.0))
        img = add(img, shade(m, (0.8, -0.3), 4) * (0.55 if i == 0 else 0.3), (1.0, 0.55, 0.22))
        if since < 0.35:
            r = np.random.default_rng(int(gt * rate) * 7 + i)
            n = 40
            ang = r.uniform(-math.pi, 0, n)
            sp = r.uniform(60, 260, n) * hh / (0.46 * h)
            sparks_x += list(tip[0] + np.cos(ang) * sp * since)
            sparks_y += list(tip[1] + np.sin(ang) * sp * since + 600 * since ** 2)
            sparks_v += list(np.full(n, (1 - since / 0.35) * 1.6))
    if sparks_x:
        img = add(img, splat(sparks_x, sparks_y, sparks_v, 1), (1.0, 0.7, 0.3))
    return img


def dust(lt, n, seed, region, drift=(6, -10), mask=None, size=1, bright=1.0):
    r = np.random.default_rng(seed)
    x0, y0, x1, y1 = region
    xs = (r.uniform(x0, x1, n) + drift[0] * lt + 8 * np.sin(lt * r.uniform(0.3, 1.0, n) + r.uniform(0, 6, n))) % w
    ys = (r.uniform(y0, y1, n) + drift[1] * lt + 6 * np.cos(lt * r.uniform(0.3, 1.0, n))) % h
    v = r.uniform(0.2, 1.0, n) * bright
    if mask is not None:
        v = v * mask[ys.astype(int).clip(0, h - 1), xs.astype(int).clip(0, w - 1)]
    return splat(xs, ys, v, size)


def mine_carry(lt, d, gt):
    img = mine_bg(lt, torch=(0.12, 0.30), dark=0.8, tunnel=(0.70, 0.5))
    bm = beam(0.82 * w, -0.05 * h, 0.40 * w, 1.0 * h, 30, 170)
    img = add(img, bm * 0.55, (1.0, 0.72, 0.38))
    for i, (x0, v, hh, yb, ph0) in enumerate(((-0.10, 0.11, 0.40, 0.86, 0.0), (0.22, 0.10, 0.44, 0.90, 1.7),
                                              (0.50, 0.12, 0.30, 0.80, 3.1))):
        x = (x0 + v * lt) * w
        c = Canvas()
        P = figure(c, x, yb * h, hh * h, walk(POSES["carry"], lt * 5.2 + ph0, 0.8), 1, profile=True)
        bx, by = (P["lha"][0] + P["rha"][0]) / 2, P["lha"][1] - 0.02 * hh * h
        c.poly(ell_pts(bx, by, 0.11 * hh * h, 0.045 * hh * h))
        c.poly(ell_pts(bx, by - 0.02 * hh * h, 0.09 * hh * h, 0.05 * hh * h, 0, 24, math.pi, 2 * math.pi))
        m = c.arr()
        img = comp(img, m, (0.015, 0.01, 0.008) if i < 2 else (0.07, 0.045, 0.03))
        img = add(img, shade(m, (0.4, -0.9), 4) * bm * 1.6, (1.0, 0.75, 0.4))
        img = add(img, shade(m, (-0.9, -0.2), 3) * 0.25, (1.0, 0.5, 0.15))
    img = add(img, dust(lt, 700, 21, (0, 0, w, h), mask=bm, bright=1.4), (1.0, 0.8, 0.5))
    return img


def chico(lt, d, gt):
    img = mine_bg(lt, torch=(0.10, 0.62), dark=0.55, tunnel=(0.22, 0.55))
    bm = beam(0.74 * w, -0.08 * h, 0.66 * w, 0.95 * h, 40, 150)
    img = add(img, bm * 0.75, (1.0, 0.78, 0.42))
    img = add(img, radial(0.74 * w, -0.02 * h, 0.28 * w, 0.08 * h) * 1.6, (1.0, 0.9, 0.7))
    # far workers in the dark, still bent over their labour
    c = Canvas()
    for x, ph in ((0.14, 0.0), (0.27, 0.4)):
        b, _ = swing_state(gt * 0.6, 1.0, ph)
        P = figure(c, x * w, 0.80 * h, 0.17 * h, blend(POSES["pick_down"], POSES["pick_up"], b * 0.7), 1, profile=True)
        pick(c, P, b * 0.7, 1, 0.17 * h)
    img = comp(img, c.arr(), (0.05, 0.032, 0.022))
    # Chico Rei: upright, head lifted toward the light
    c = Canvas()
    rise = sm(lt / 2.5)
    profile_bust(c, 0.34 * w, 0.28 * h, 0.20 * h, 1, tilt=-0.10 - 0.12 * rise)
    m = c.arr()
    img = comp(img, m, (0.008, 0.006, 0.005))
    img = add(img, shade(m, (0.85, -0.5), 7) * (1.4 + 0.5 * rise), (1.0, 0.78, 0.40))
    img = add(img, shade(m, (-0.9, 0.1), 5) * 0.18, (0.9, 0.45, 0.15))
    img = add(img, dust(lt, 900, 33, (0, 0, w, h), drift=(-3, 14), mask=bm, bright=1.6), (1.0, 0.85, 0.55))
    return img


def gold_hands(lt, d, gt):
    img = vgrad((0.20, 0.13, 0.06), (0.02, 0.012, 0.01), 0.0, 0.8)
    img = add(img, radial(0.5 * w, -0.1 * h, 0.7 * w, 0.5 * h), (0.55, 0.35, 0.12))
    cx, cy = 0.5 * w, 0.80 * h
    c = Canvas()
    c.poly(ell_pts(cx, cy, 0.29 * w, 0.10 * h, 0, 48, 0, math.pi))         # cupped palms
    c.poly(ell_pts(cx - 0.12 * w, cy, 0.17 * w, 0.035 * h, -0.08))
    c.poly(ell_pts(cx + 0.12 * w, cy, 0.17 * w, 0.035 * h, 0.08))
    c.line((cx - 0.15 * w, cy + 0.05 * h), (cx - 0.10 * w, 1.10 * h), 0.19 * w)  # forearms
    c.line((cx + 0.15 * w, cy + 0.05 * h), (cx + 0.10 * w, 1.10 * h), 0.19 * w)
    c.poly(ell_pts(cx - 0.27 * w, cy - 0.02 * h, 0.045 * w, 0.03 * h, -0.6))   # thumbs
    c.poly(ell_pts(cx + 0.27 * w, cy - 0.02 * h, 0.045 * w, 0.03 * h, 0.6))
    m = c.arr()
    img = comp(img, m, (0.03, 0.018, 0.012))
    img = add(img, shade(m, (0.0, -1.0), 8) * 0.9, (1.0, 0.72, 0.35))
    img = add(img, shade(m, (0.0, -1.0), 30, 120) * 0.25, (1.0, 0.72, 0.35))
    pile = radial(cx, cy - 0.012 * h, 0.21 * w, 0.025 * h) * (1 - m * 0.6)
    img = add(img, pile * (0.9 + 0.3 * ROCK2), (1.2, 0.8, 0.25))
    r = np.random.default_rng(5)
    n = 420
    age = (lt * r.uniform(0.18, 0.4, n) + r.random(n)) % 1.0
    px = cx + r.normal(0, 0.09 * w, n) + np.sin(age * 7 + r.random(n) * 6) * 30 * age
    py = cy - 0.02 * h - age * 0.75 * h
    v = np.sin(age * math.pi) * r.uniform(0.6, 1.6, n) * sm(lt / 0.6)
    img = add(img, splat(px, py, v, 1), (1.2, 0.85, 0.35))
    img = add(img, splat(px, py, v * age, 4) * 0.5, (1.0, 0.9, 0.7))
    return img


SHACKLE_OPEN = 0.8


def gold_to_freedom(lt, d, gt):
    glow = 0.4 + 1.6 * sm((lt - SHACKLE_OPEN) / 1.6)
    img = vgrad((0.30, 0.20, 0.09), (0.015, 0.01, 0.01), 0.0, 0.85) * glow
    img = add(img, radial(0.5 * w, 0.05 * h, 0.55 * w, 0.35 * h) * glow, (0.9, 0.65, 0.3))
    cx = 0.5 * w
    c = Canvas()
    c.line((cx + 10, 1.1 * h), (cx, 0.47 * h), 0.13 * w)                     # forearm
    c.poly(ell_pts(cx, 0.405 * h, 0.075 * w, 0.06 * h))                         # palm
    for k, (dx, ln) in enumerate(((-0.055, 0.10), (-0.018, 0.12), (0.02, 0.115), (0.055, 0.095))):
        c.line((cx + dx * w, 0.39 * h), (cx + dx * w * 1.4, (0.39 - ln) * h), 0.03 * w)
    c.line((cx - 0.06 * w, 0.43 * h), (cx - 0.14 * w, 0.37 * h), 0.034 * w)    # thumb
    m = c.arr()
    img = comp(img, m, (0.02, 0.012, 0.01))
    img = add(img, shade(m, (0.0, -1.0), 7) * (0.6 + 0.6 * glow), (1.0, 0.8, 0.45))
    # iron cuff: closed band, then two halves fall away
    mc = Canvas()
    tt = max(0.0, lt - SHACKLE_OPEN)
    by = 0.53 * h
    if tt == 0:
        mc.poly([(cx - 0.085 * w, by - 0.022 * h), (cx + 0.085 * w, by - 0.022 * h),
                 (cx + 0.085 * w, by + 0.022 * h), (cx - 0.085 * w, by + 0.022 * h)])
        chain(mc, cx + 0.07 * w, by + 0.02 * h, cx + 0.09 * w, by + 0.22 * h, 13, 6)
    else:
        for side in (-1, 1):
            ox = side * (40 * tt + 60 * tt ** 2)
            oy = 0.5 * 1800 * tt ** 2
            a = side * 1.6 * tt
            pts = [(cx + ox, by - 0.022 * h + oy), (cx + side * 0.085 * w + ox, by - 0.022 * h + oy),
                   (cx + side * 0.085 * w + ox, by + 0.022 * h + oy), (cx + ox, by + 0.022 * h + oy)]
            pc = (cx + side * 0.04 * w + ox, by + oy)
            mc.poly([rot(p, a, pc) for p in pts])
            if side == 1:
                e = rot((cx + 0.07 * w + ox, by + 0.02 * h + oy), a, pc)
                chain(mc, e[0], e[1], e[0] + 0.02 * w + 30 * tt, e[1] + 0.2 * h, 13, 6)
    mm = mc.arr()
    img = comp(img, mm, (0.10, 0.095, 0.09))
    img = add(img, shade(mm, (0.0, -1.0), 3) * 0.8, (0.8, 0.75, 0.65))
    r = np.random.default_rng(9)
    n = 600
    age = (lt * r.uniform(0.25, 0.55, n) + r.random(n)) % 1.0
    px = cx + r.normal(0, 0.16 * w, n) * (0.4 + age)
    py = 1.0 * h - age * 1.1 * h
    v = np.sin(age * math.pi) * r.uniform(0.5, 1.5, n)
    img = add(img, splat(px, py, v, 1), (1.2, 0.85, 0.35))
    img = add(img, radial(cx, 0.36 * h, 0.35 * w) * sm((lt - 1.4) / 1.0) * 1.2, (1.0, 0.9, 0.7))
    return img


def ship_storm(lt, d, gt):
    img = sky(lt, (0.03, 0.04, 0.06), (0.16, 0.10, 0.10), 0.55, storm=1.0)
    flash = 1.0 if 0.05 < lt < 0.12 else 0.0
    img = img * (1 + flash * 2.5)
    img = ocean(img, gt, 0.55, None, (0.02, 0.03, 0.04), (0.25, 0.25, 0.27), speed=1.6, chop=1.4)
    c = Canvas()
    roll = 0.05 * math.sin(lt * 3)
    ship(c, 0.5 * w, 0.62 * h, 0.75 * w, 1, sails=0.6)
    m = c.arr()
    img = comp(img, m, (0.01, 0.01, 0.012))
    img = add(img, shade(m, (0.0, -1.0), 3) * 0.3, (0.6, 0.4, 0.4))
    return img


def shackled(lt, d, gt):
    img = vgrad((0.07, 0.015, 0.012), (0.01, 0.005, 0.005))
    img = add(img, radial(0.5 * w, 0.45 * h, 0.5 * w), (0.35, 0.08, 0.04))
    c = Canvas()
    c.line((-0.1 * w, 0.95 * h), (0.47 * w, 0.48 * h), 0.15 * w)
    c.line((1.1 * w, 0.95 * h), (0.53 * w, 0.48 * h), 0.15 * w)
    c.poly(ell_pts(0.45 * w, 0.44 * h, 0.07 * w, 0.065 * h, 0.6))
    c.poly(ell_pts(0.55 * w, 0.44 * h, 0.07 * w, 0.065 * h, -0.6))
    m = c.arr()
    img = comp(img, m, (0.02, 0.01, 0.008))
    img = add(img, shade(m, (0.0, -1.0), 6) * 0.6, (0.9, 0.4, 0.25))
    mc = Canvas()
    for sx in (-1, 1):
        cx = 0.5 * w + sx * 0.16 * w
        mc.poly([rot(p, -sx * 0.69, (cx, 0.58 * h)) for p in
                 [(cx - 0.1 * w, 0.565 * h), (cx + 0.1 * w, 0.565 * h), (cx + 0.1 * w, 0.595 * h), (cx - 0.1 * w, 0.595 * h)]])
    chain(mc, 0.36 * w, 0.62 * h, 0.64 * w, 0.62 * h, 16, 7, sag=0.07 * h + 6 * math.sin(lt * 9))
    mm = mc.arr()
    img = comp(img, mm, (0.08, 0.075, 0.07))
    img = add(img, shade(mm, (0.0, -1.0), 2) * 0.9, (0.9, 0.7, 0.6))
    return img


# ================================================================ sea
def sky(lt, top, horizon_c, hz, storm=0.0, sun=None, drift=12):
    img = vgrad(top, horizon_c, 0.0, hz, 0.8)
    cl = np.roll(CLOUD, int(lt * drift), 1)
    if storm:
        img = img * (0.55 + 0.9 * cl[..., None] * storm)
    else:
        img = img * (0.85 + 0.3 * cl[..., None])
    if sun is not None:
        sx, sy, sc, sr = sun
        img = add(img, radial(sx * w, sy * h, sr * w, sr * 0.7 * w), sc)
        img = add(img, radial(sx * w, sy * h, sr * 0.15 * w), (1.4, 1.2, 0.9))
    return img


def ocean(img, gt, hz, shore, deep, light, speed=1.0, chop=1.0, foam=1.0, sun_x=None, sun_c=(1, 0.7, 0.4)):
    hzp = int(hz * h)
    sy = int((shore if shore is not None else 1.02) * h)
    sy = min(sy, h)
    rows = np.arange(hzp, sy, dtype=np.float32)
    if len(rows) == 0:
        return img
    p = (rows - hzp) / max(1, (sy - hzp))
    z = 1.0 / (p * 0.9 + 0.035)
    X = (XX[hzp:sy] - w / 2) / w * z[:, None]
    Z = np.broadcast_to(z[:, None], X.shape)
    v = sample(SEA, X * 140 + gt * 6, Z * 70 + gt * 40 * speed)
    crest = np.sin(Z * 1.6 + gt * 2.2 * speed + v * 2.5 * chop)
    crest = np.clip((crest - 0.55) / 0.45, 0, 1) * np.clip(p * 3, 0, 1)[:, None]
    hcol = img[max(0, hzp - 2)]
    v = v * 0.75 + sample(SEA, X * 400 + gt * 9, Z * 210 + gt * 70 * speed) * 0.25
    sea = col(deep) * (0.6 + 0.7 * v[..., None]) + col(light) * (crest * v)[..., None] * 0.9 * foam
    refl = np.clip(1 - p * 2.2, 0, 1)[:, None, None]
    sea = sea * (1 - refl * 0.6) + hcol[None, :, :] * refl * 0.6
    if sun_x is not None:
        path = np.exp(-((XX[hzp:sy] - sun_x * w) / (40 + 260 * p[:, None])) ** 2) * (v > 0.55)
        sea = sea + path[..., None] * col(sun_c) * 1.4
    if shore is not None:
        c = (gt % SEA_PERIOD) / SEA_PERIOD
        yb = lerp(0.45, 0.97, c ** 1.4)
        thick = 0.02 + 0.06 * c
        band = np.exp(-((p - yb) / thick) ** 2)[:, None] * (v > 0.35)
        white = np.clip((c - 0.6) / 0.25, 0, 1) * np.clip(p - 0.75, 0, 1)[:, None] * 4 * (v > 0.3)
        f = np.clip(band * (0.4 + 0.8 * c) + white, 0, 1)[..., None]
        sea = sea * (1 - f) + col(light) * 1.5 * f * foam
    img = img.copy()
    img[hzp:sy] = sea
    if shore is not None and sy < h:
        n = h - sy
        q = np.linspace(0, 1, n, dtype=np.float32)[:, None, None]
        sand = col((0.16, 0.13, 0.10)) * (0.6 + 0.4 * q) * (0.85 + 0.25 * ROCK2[sy:, :, None])
        wet = np.flipud(img[max(hzp, 2 * sy - h):sy])
        if wet.shape[0] < n:
            wet = np.concatenate([wet, np.repeat(wet[-1:], n - wet.shape[0], 0)], 0)
        c = (gt % SEA_PERIOD) / SEA_PERIOD
        wash = 0.06 + 0.10 * math.sin(math.pi * np.clip((c - 0.72) / 0.5, 0, 1)) if c > 0.72 else 0.06
        wm = np.clip(1 - q / wash, 0, 1) if wash > 0 else 0
        foamline = np.exp(-((q - wash) / 0.01) ** 2) * (ROCK[sy:, :, None] > 0.4)
        img[sy:] = sand * (1 - 0.6 * np.exp(-q * 6)) + wet[:n] * 0.45 * np.exp(-q * 5) + wm * col(light) * 0.4 + foamline * col(light) * 0.9
    return img


def spray(gt, x0, x1, y, n=260, seed=1, height=0.25):
    out_x, out_y, out_v = [], [], []
    for tc in sea_crash_times(gt - 1.6, gt + 0.001):
        age = gt - tc
        r = np.random.default_rng(int(tc * 100) + seed)
        vx = r.normal(0, 40, n)
        vy = -r.uniform(0.3, 1.0, n) * height * h * 2.2
        out_x += list(r.uniform(x0, x1, n) + vx * age)
        out_y += list(y + vy * age + 0.5 * 700 * age ** 2)
        out_v += list(np.full(n, max(0.0, 1 - age / 1.6)) * r.uniform(0.3, 1.0, n))
    return out_x, out_y, out_v


STORM = dict(top=(0.05, 0.07, 0.09), hor=(0.42, 0.40, 0.36), deep=(0.03, 0.06, 0.07), light=(0.55, 0.60, 0.60))


def sea_storm(lt, d, gt):
    hz, shore = 0.42, 0.70
    img = sky(lt, STORM["top"], STORM["hor"], hz, storm=1.0, sun=(0.7, hz - 0.02, (0.35, 0.25, 0.15), 0.5))
    img = ocean(img, gt, hz, shore, STORM["deep"], STORM["light"], speed=1.3, chop=1.3)
    c, sc = Canvas(), Canvas()
    jangada(c, 0.24 * w, 0.82 * h, 0.40 * w, 1, sail=True, sail_c=sc, sway=4 * math.sin(lt * 2))
    jangada(c, 0.80 * w, 0.88 * h, 0.52 * w, -1, sail=True, sail_c=sc, sway=5 * math.sin(lt * 2 + 1))
    jangada(c, 0.55 * w, 0.735 * h, 0.22 * w, 1, sail=False)
    ship(c, 0.62 * w, hz * h + 3, 0.12 * w, -1)
    ms, m = sc.arr(), c.arr()
    img = comp(img, ms * 0.92, (0.30, 0.28, 0.24))
    img = add(img, shade(ms, (0.6, -0.8), 6) * 0.5, (0.7, 0.62, 0.5))
    img = comp(img, m, (0.015, 0.016, 0.018))
    sx, sy, sv = spray(gt, 0.0, w, shore * h)
    img = add(img, splat(sx, sy, sv, 2), (0.7, 0.75, 0.75))
    return img


def jangada_close(lt, d, gt):
    hz, shore = 0.50, 0.74
    img = sky(lt, STORM["top"], STORM["hor"], hz, storm=1.0)
    img = ocean(img, gt, hz, shore, STORM["deep"], STORM["light"], speed=1.5, chop=1.6)
    c, sc = Canvas(), Canvas()
    jangada(c, 0.55 * w, 0.86 * h, 1.05 * w, 1, sail=True, sail_c=sc, sway=8 * math.sin(lt * 1.7))
    ms, m = sc.arr(), c.arr()
    img = comp(img, ms * 0.9, (0.33, 0.30, 0.25))
    img = add(img, shade(ms, (0.3, -0.9), 9) * 0.6, (0.75, 0.68, 0.55))
    img = comp(img, m, (0.03, 0.022, 0.016))
    img = add(img, shade(m, (0.2, -1.0), 3) * 0.6, (0.6, 0.55, 0.5))
    sx, sy, sv = spray(gt, 0.0, w, 0.80 * h, n=500, seed=3, height=0.35)
    img = add(img, splat(sx, sy, sv, 2), (0.8, 0.85, 0.85))
    img = add(img, splat(sx, sy, np.asarray(sv) * 0.5, 6), (0.5, 0.55, 0.55))
    return img


PORT = dict(top=(0.06, 0.08, 0.11), hor=(0.62, 0.52, 0.42), deep=(0.03, 0.06, 0.08), light=(0.6, 0.62, 0.6))


def port_horizon(c, hz, side=1):
    """Distant port: anchored slave ship and low buildings on the far shore."""
    ship(c, (0.5 + 0.12 * side) * w, hz * h + 2, 0.16 * w, -side)
    r = np.random.default_rng(3)
    x = 0.0 if side < 0 else 0.72 * w
    for _ in range(14):
        bw, bh = r.uniform(8, 22), r.uniform(6, 18)
        c.rect(x, hz * h - bh, x + bw, hz * h + 2)
        x += bw + r.uniform(0, 6)


def shore_hero(lt, d, gt):
    hz, shore = 0.52, 0.66
    img = sky(lt, PORT["top"], PORT["hor"], hz, storm=0.7, sun=(0.42, hz - 0.03, (0.45, 0.28, 0.14), 0.45))
    img = ocean(img, gt, hz, shore, PORT["deep"], PORT["light"], speed=1.0, chop=1.0)
    far = Canvas()
    port_horizon(far, hz)
    img = comp(img, far.arr(), (0.10, 0.09, 0.10))
    c = Canvas()
    person(c, 0.5 * w, 0.99 * h, 0.56 * h, 1.08)
    arrivals = ((0.18, 0.95, 0.47, 0.5, -1), (0.82, 0.96, 0.48, 1.0, 1), (0.04, 0.93, 0.40, 1.6, -1),
                (0.96, 0.93, 0.41, 2.1, 1))
    for x, yb, hh, t0, side in arrivals:
        k = sm((lt - t0) / 1.3)
        if lt < t0:
            continue
        xx = lerp(x + side * 0.35, x, k) * w
        person(c, xx, (yb - 0.004 * abs(math.sin(lt * 6)) * (1 - k)) * h, hh * h, 1.0, 0.25 * math.sin(lt * 6) * (1 - k))
    m = c.arr()
    img = comp(img, m, (0.012, 0.012, 0.014))
    img = add(img, shade(m, (-0.7, -0.7), 5) * 0.40, (0.95, 0.75, 0.55))
    img = add(img, shade(m, (-0.9, -0.2), 3) * 0.25, (0.95, 0.65, 0.45))
    return img


def line_of_men(lt, d, gt):
    hz, shore = 0.47, 0.60
    img = sky(lt, PORT["top"], PORT["hor"], hz, storm=0.8, sun=(0.3, hz - 0.03, (0.4, 0.26, 0.14), 0.5))
    img = ocean(img, gt, hz, shore, PORT["deep"], PORT["light"], speed=0.8, chop=0.9)
    far = Canvas()
    ship(far, 0.66 * w, hz * h + 4, 0.22 * w, -1)
    ship(far, 0.30 * w, hz * h + 2, 0.10 * w, 1)
    img = comp(img, far.arr(), (0.07, 0.07, 0.08))
    c = Canvas()
    jangada(c, 0.13 * w, 0.88 * h, 0.40 * w, 1, sail=False)
    jangada(c, 0.90 * w, 0.92 * h, 0.46 * w, -1, sail=False)
    r = np.random.default_rng(4)
    for i in range(8):
        x = (0.12 + i * 0.108) * w
        person(c, x, (0.80 + r.uniform(-0.005, 0.01)) * h, r.uniform(0.20, 0.23) * h, r.uniform(0.92, 1.1))
    m = c.arr()
    img = comp(img, m, (0.012, 0.012, 0.014))
    img = add(img, shade(m, (0.0, -1.0), 3) * 0.45, (0.95, 0.75, 0.55))
    return img


def dragao_close(lt, d, gt):
    hz = 0.86
    img = sky(lt, (0.04, 0.05, 0.08), (0.55, 0.42, 0.30), hz, storm=1.0)
    lx, ly = 0.20 * w, 0.10 * h
    img = add(img, rays(lx, ly, seed=2) * radial(lx, ly, 0.9 * w) * 0.55, (1.0, 0.8, 0.55))
    img = add(img, radial(lx, ly, 0.10 * w) * 1.4, (1.0, 0.9, 0.75))
    img = ocean(img, gt, hz, None, PORT["deep"], PORT["light"], speed=1.2)
    far = Canvas()
    ship(far, 0.16 * w, hz * h + 2, 0.16 * w, 1)
    sc = Canvas()
    jangada(far, 1.02 * w, 1.0 * h, 0.55 * w, -1, sail=True, sail_c=sc, sway=4 * math.sin(lt * 1.4))
    img = comp(img, sc.arr() * 0.8, (0.30, 0.26, 0.22))
    img = comp(img, far.arr(), (0.03, 0.03, 0.035))
    c = Canvas()
    push = sm(lt / d)
    profile_bust(c, 0.62 * w, 0.30 * h, 0.21 * h, -1, tilt=-0.08 - 0.04 * push)
    m = c.arr()
    img = comp(img, m, (0.01, 0.01, 0.012))
    img = add(img, shade(m, (-0.8, -0.6), 7) * (1.3 + 0.3 * push), (1.0, 0.82, 0.58))
    return img


# ================================================================ legacy
def ocean_dawn(lt, d, gt):
    hz = 0.56
    img = sky(lt, (0.10, 0.14, 0.24), (1.0, 0.62, 0.36), hz, sun=(0.5, hz - 0.01, (1.0, 0.55, 0.25), 0.55), drift=20)
    img = ocean(img, gt, hz, None, (0.05, 0.07, 0.11), (0.9, 0.7, 0.5), speed=0.7, chop=0.7, sun_x=0.5)
    return img


def chain_break(lt, d, gt, slow=False):
    tb = d * (0.5 if slow else 0.45)
    img = vgrad((0.08, 0.04, 0.025), (0.015, 0.01, 0.01))
    img = add(img, radial(0.5 * w, 0.48 * h, 0.55 * w, 0.35 * h), (0.45, 0.22, 0.08))
    r = np.random.default_rng(2)
    for _ in range(9):
        img = add(img, radial(r.uniform(0, w), r.uniform(0.2, 0.8) * h, r.uniform(20, 60)) * 0.25, (1.0, 0.6, 0.3))
    s = 1.35 if slow else 1.0
    tt = max(0.0, lt - tb)
    pull = sm(lt / tb) * 0.012 + (0.10 * (1 - math.exp(-tt * 5)))
    shake = (3 * math.sin(lt * 70) if lt < tb else 0)
    cy = 0.50 * h
    gap = (0.21 + pull) * w * s
    c = Canvas()
    for side in (-1, 1):
        fx = 0.5 * w + side * gap
        c.line((fx + side * 0.6 * w, cy + 0.05 * h), (fx + side * 0.05 * w, cy + 0.01 * h), 0.16 * w * s)
        c.poly(ell_pts(fx + shake * side, cy, 0.085 * w * s, 0.07 * h * s * 0.75, side * 0.2))
        c.poly(ell_pts(fx - side * 0.05 * w * s, cy - 0.035 * h * s, 0.05 * w * s, 0.02 * h * s, side * 0.2))
    m = c.arr()
    img = comp(img, m, (0.02, 0.012, 0.01))
    img = add(img, shade(m, (0.0, -1.0), 7) * 0.8, (1.0, 0.65, 0.35))
    mc = Canvas()
    link = 28 * s
    if tt == 0:
        chain(mc, 0.5 * w - gap + 0.06 * w * s, cy, 0.5 * w + gap - 0.06 * w * s, cy + shake, link, 11 * s)
    else:
        for side in (-1, 1):
            a0 = 0.5 * w + side * (gap - 0.06 * w * s)
            fall = 0.5 * 2600 * tt ** 2
            whip = side * 0.5 * w * (1 - math.exp(-tt * 9)) * 0.25
            chain(mc, a0, cy, 0.5 * w + side * 0.04 * w + whip, cy + min(fall, 0.4 * h) + 30 * math.sin(tt * 14) * math.exp(-tt * 3),
                  link, 11 * s, phase=0 if side < 0 else 1)
    mm = mc.arr()
    img = comp(img, mm, (0.12, 0.11, 0.10))
    img = add(img, shade(mm, (0.0, -1.0), 2) * 0.9, (1.0, 0.8, 0.6))
    if 0 < tt < 1.2:
        rr = np.random.default_rng(int(tb * 1000))
        n = 300
        ang = rr.uniform(0, 2 * math.pi, n)
        sp = rr.uniform(80, 520, n) * (0.6 if slow else 1.0)
        px = 0.5 * w + np.cos(ang) * sp * tt
        py = cy + np.sin(ang) * sp * tt + 500 * tt ** 2
        img = add(img, splat(px, py, np.full(n, 1.8 * max(0.0, 1 - tt / 1.2)), 1), (1.0, 0.75, 0.35))
        img = add(img, radial(0.5 * w, cy, 0.16 * w) * max(0.0, 1 - tt * 4) * 1.0, (1.0, 0.85, 0.6))
    return img


FACES = {
    0: dict(bg=(0.20, 0.10, 0.04), rim=(1.0, 0.70, 0.40), L=(-0.9, -0.3), x=0.50, s=0.36),
    1: dict(bg=(0.05, 0.10, 0.13), rim=(0.70, 0.85, 0.95), L=(0.9, -0.3), x=0.52, s=0.34),
    2: dict(bg=(0.16, 0.08, 0.05), rim=(1.0, 0.62, 0.35), L=(0.8, -0.5), x=0.48, s=0.35),
    3: dict(bg=(0.06, 0.05, 0.05), rim=(1.0, 0.80, 0.60), L=(-0.85, -0.4), x=0.50, s=0.56),
}


def face(lt, d, gt, variant=0):
    f = FACES[variant]
    img = vgrad(col(f["bg"]) * 0.4, col(f["bg"]) * 0.08)
    img = add(img, radial(0.5 * w, 0.40 * h, 0.55 * w), f["bg"])
    c = Canvas()
    s = f["s"] * h * 0.5
    bust(c, f["x"] * w, 0.42 * h, s, variant % 3)
    m = c.arr()
    img = comp(img, m, col(f["bg"]) * 0.06)
    img = add(img, shade(m, f["L"], 10) * 1.1, f["rim"])
    img = add(img, shade(m, f["L"], 30, 90) * 0.12, f["rim"])
    img = add(img, shade(m, (0.0, -1.0), 4) * 0.2, f["rim"])
    img = add(img, dust(lt, 120, 40 + variant, (0, 0, w, h), drift=(4, -6), size=1, bright=0.4), f["rim"])
    return img


def crowd(lt, d, gt):
    hz = 0.78
    img = sky(lt, (0.14, 0.12, 0.20), (1.0, 0.66, 0.38), hz, sun=(0.5, hz - 0.03, (1.0, 0.6, 0.3), 0.6), drift=18)
    img = add(img, (rays(0.5 * w, hz * h, seed=7) * radial(0.5 * w, hz * h, 0.9 * w)) * 0.35, (1.0, 0.75, 0.45))
    c = Canvas()
    xs = np.linspace(-10, w + 10, 40)
    c.poly(list(zip(xs, 0.785 * h - 8 * np.sin(xs / 90))) + [(w + 10, h), (-10, h)])
    r = np.random.default_rng(8)
    n = 9
    for i in range(n):
        x = (0.06 + i * 0.111) * w + r.uniform(-8, 8)
        person(c, x, 0.86 * h + r.uniform(0, 6), r.uniform(0.25, 0.32) * h, r.uniform(0.88, 1.1))
    m = c.arr()
    img = comp(img, m, (0.03, 0.02, 0.025))
    img = add(img, shade(m, (0.0, -1.0), 4) * 0.8, (1.0, 0.72, 0.45))
    return img


def modern(lt, d, gt):
    hz = 0.60
    img = sky(lt, (0.12, 0.16, 0.30), (1.0, 0.64, 0.40), hz, sun=(0.30, hz - 0.06, (1.0, 0.6, 0.3), 0.5), drift=16)
    back = Canvas()
    skyline(back, hz * h + 2, seed=21, hmax=0.15)
    img = comp(img, back.arr(), (0.30, 0.22, 0.22))
    c = Canvas()
    tops = skyline(c, hz * h + 2, seed=12, hmax=0.22)
    m = c.arr()
    img = comp(img, m, (0.07, 0.07, 0.10))
    img = add(img, shade(m, (-0.9, -0.2), 2) * 0.5, (1.0, 0.65, 0.4))
    r = np.random.default_rng(13)
    wx, wy, wv = [], [], []
    for x0, y0, x1, y1 in tops:
        for _ in range(int((x1 - x0) * (y1 - y0) / 500)):
            wx.append(r.uniform(x0 + 3, x1 - 3))
            wy.append(r.uniform(y0 + 4, y1 - 4))
            wv.append(r.uniform(0.2, 1.0) * (r.random() > 0.55))
    img = add(img, splat(wx, wy, wv, 1) * 1.2, (1.0, 0.8, 0.5))
    img = ocean(img, gt, hz + 0.003, None, (0.05, 0.07, 0.12), (0.9, 0.7, 0.5), speed=0.6, chop=0.6, sun_x=0.30)
    fg, sc = Canvas(), Canvas()
    jangada(fg, (0.62 + 0.03 * lt) * w, 0.80 * h, 0.36 * w, -1, sail=True, sail_c=sc, sway=3 * math.sin(lt * 2))
    img = comp(img, sc.arr() * 0.85, (0.55, 0.40, 0.30))
    img = comp(img, fg.arr(), (0.03, 0.03, 0.04))
    return img


def black(lt, d, gt):
    return np.zeros((h, w, 3), np.float32)
