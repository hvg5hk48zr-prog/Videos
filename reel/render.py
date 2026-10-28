"""Frame renderer: shot -> camera -> grade -> grain/vignette -> text -> RGB24."""
import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import gfx
import timeline as T

FPS = 30
DUR = 60.0
NF = int(FPS * DUR)
W, H = 1080, 1920

_r = np.random.default_rng(99)
GRAIN = [np.asarray(Image.fromarray(((_r.standard_normal((H // 2, W // 2)) * 40 + 128).clip(0, 255)).astype(np.uint8))
                    .resize((W, H), Image.BILINEAR), np.float32)[..., None] / 255.0 - 0.5 for _ in range(10)]
_yy, _xx = np.mgrid[0:H, 0:W].astype(np.float32)
VIGNETTE = (1 - 0.62 * (((_xx - W / 2) / (W * 0.62)) ** 2 + ((_yy - H / 2) / (H * 0.60)) ** 2) ** 1.4).clip(0.15, 1)[..., None]
del _yy, _xx


# ---------------------------------------------------------------- text
_cache = {}


def text_layer(s, font, size, color, tracking, glow):
    key = (s, font, size, color, tracking, glow)
    if key in _cache:
        return _cache[key]
    f = ImageFont.truetype(font, size)
    lines = s.split("\n")
    lh = int(size * 1.12)
    tr = tracking * size
    widths = [sum(f.getlength(ch) for ch in ln) + tr * max(0, len(ln) - 1) for ln in lines]
    pad = int(size * 0.8)
    tw, th = int(max(widths)) + 2 * pad, lh * len(lines) + 2 * pad
    im = Image.new("L", (tw, th), 0)
    d = ImageDraw.Draw(im)
    asc = f.getmetrics()[0]
    for i, ln in enumerate(lines):
        x = pad + (tw - 2 * pad - widths[i]) / 2
        y = pad + i * lh + (lh - asc) * 0.5
        for ch in ln:
            d.text((x, y), ch, font=f, fill=255)
            x += f.getlength(ch) + tr
    a = np.asarray(im, np.float32) / 255.0
    shadow = np.clip(gfx.blur(a, max(2, size // 8)) * 1.6, 0, 1)
    gl = gfx.blur(a, max(2, size // 6)) * glow
    layer = dict(a=a, shadow=shadow, glow=gl, color=np.asarray(color, np.float32), w=tw, h=th)
    _cache[key] = layer
    return layer


def draw_text(frame, L, yc, alpha, dy=0.0):
    x0 = (W - L["w"]) // 2
    y0 = int(yc * H - L["h"] / 2 + dy)
    ys, ye = max(0, y0), min(H, y0 + L["h"])
    xs, xe = max(0, x0), min(W, x0 + L["w"])
    if ys >= ye or xs >= xe:
        return
    sl = (slice(ys - y0, ye - y0), slice(xs - x0, xe - x0))
    reg = frame[ys:ye, xs:xe]
    reg *= (1 - L["shadow"][sl][..., None] * 0.55 * alpha)
    reg += L["glow"][sl][..., None] * L["color"] * 0.6 * alpha
    a = L["a"][sl][..., None] * alpha
    reg[:] = reg * (1 - a) + L["color"] * a


def text_items(t):
    out = []
    for (t0, t1, s, font, size, color, y, tr, fi, fo, glow) in T.TEXT:
        if t0 <= t <= t1:
            a = min(gfx.sm((t - t0) / fi) if fi else 1.0, gfx.sm((t1 - t) / fo) if fo else 1.0)
            if s.startswith("ARTISTIC"):
                a *= 0.55
            out.append((s, font, size, color, y, tr, glow, a, (1 - gfx.sm((t - t0) / max(fi, 0.01) * 0.7)) * 18))
    for (t0, t1, s) in narration_subs():
        if t0 <= t <= t1:
            a = min(gfx.sm((t - t0) / 0.15), gfx.sm((t1 - t) / 0.2))
            out.append((s, T.SANS_SB, 44, (1.0, 0.98, 0.94), 0.70, 0.0, 0.0, a, 0.0))
    return out


_subs = None


def narration_subs():
    global _subs
    if _subs is None:
        p = os.path.join(os.path.dirname(__file__), "build", "narration.json")
        _subs = []
        if os.path.exists(p):
            for seg in json.load(open(p)):
                _subs.append((seg["start"], seg["end"] + 0.25, seg["sub"]))
    return _subs


# ---------------------------------------------------------------- frames
def shot_at(t):
    for sh in T.SHOTS:
        if sh[0] <= t < sh[1]:
            return sh
    return T.SHOTS[-1]


def render_frame(i, with_text=True):
    t = i / FPS
    t0, t1, fn, cam, grade, flash_in = shot_at(t)
    lt, d = t - t0, t1 - t0
    img = fn(lt, d, t)
    u = gfx.sm(lt / d) * 0.6 + (lt / d) * 0.4
    z0, z1, cx0, cy0, cx1, cy1 = cam
    z = gfx.lerp(z0, z1, u)
    cx, cy = gfx.lerp(cx0, cx1, u), gfx.lerp(cy0, cy1, u)
    r = np.random.default_rng(i)
    jx, jy = r.normal(0, 0.6, 2)                     # gate weave
    cw, ch = gfx.w / z, gfx.h / z
    x0 = min(max(cx * gfx.w - cw / 2 + jx, 0), gfx.w - cw)
    y0 = min(max(cy * gfx.h - ch / 2 + jy, 0), gfx.h - ch)
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    f = np.asarray(pil.resize((W, H), Image.BICUBIC, box=(x0, y0, x0 + cw, y0 + ch)), np.float32) / 255.0
    gain, tint, contrast = grade
    f = f * gain * np.asarray(tint, np.float32)
    f = np.clip(f, 0, 1)
    f = f + (f - f * f * (3 - 2 * f)) * (1 - contrast) * 2      # s-curve
    f *= VIGNETTE
    if fn is not T.S.black:
        f *= 1 + 0.03 * math.sin(t * 47) * math.sin(t * 13)      # projector flicker
        lum = f.mean(axis=2, keepdims=True)
        f += GRAIN[i % len(GRAIN)] * (0.05 + 0.10 * lum * (1 - lum))
        if r.random() < 0.12:                                    # occasional scratch
            x = int(r.uniform(0.05, 0.95) * W)
            f[:, x:x + 2] *= 1 + r.uniform(0.15, 0.35)
    else:
        f += GRAIN[i % len(GRAIN)] * 0.025
    flash = 0.0
    for ft, fs in T.FLASHES:
        if 0 <= t - ft < 0.2:
            flash = max(flash, fs * (1 - (t - ft) / 0.2) ** 2)
    if flash_in and lt < 0.12 and t < 4:
        flash = max(flash, 0.8 * (1 - lt / 0.12))
    env = 1.0
    for a, b, v0, v1 in T.FADES:
        if b > a and a <= t <= b:
            env = gfx.lerp(v0, v1, (t - a) / (b - a))
        elif b > a and t > b and t < 53.25:
            env = v1 if a > 50 else env
    f = f * env + flash * (1 - f * 0.3)
    if with_text:
        for (s, font, size, color, y, tr, glow, a, dy) in text_items(t):
            draw_text(f, text_layer(s, font, size, color, tr, glow), y, a, dy)
    return (np.clip(f, 0, 1) * 255).astype(np.uint8)
