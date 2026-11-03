"""On-screen cards (hook, section titles, theory note, outro).

Times are seconds in the source video, matched to what is said (from the
burned-in captions). Each line can appear later than its card (`at`), so the
key words pop in as they are spoken.
"""
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = "/usr/share/fonts/opentype/inter/"
EMOJI = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"

YELLOW = (255, 210, 63)
GREEN = (52, 211, 120)
WHITE = (255, 255, 255)
SOFT = (225, 225, 225)
BLACK = (17, 17, 17)

FRAME_W = 1170
TOP = 200  # below the Instagram/TikTok top bar, above the speaker's head


def L(at, *segs, gap=14):
    """A line: (text, size, color[, weight]) segments; emoji segments are (emoji, size, 'emoji')."""
    return {"at": at, "segs": segs, "gap": gap}


CARDS = [
    # Hook (0–3s)
    {"start": 0.0, "end": 3.35, "plate": "light", "lines": [
        L(0.0, ("Why do we name our", 62, BLACK, "Black")),
        L(0.0, ("capoeira rhythms", 62, BLACK, "Black")),
        L(0.0, ("after a ", 62, BLACK, "Black"), ("SAINT?", 62, (0, 140, 70), "Black"), (" 🤔", 62, "emoji")),
    ]},
    # Part 1: who is São Bento
    {"start": 3.4, "end": 15.4, "lines": [
        L(3.4, ("SÃO BENTO", 108, YELLOW, "Black"), (" 🐍", 96, "emoji")),
        L(3.4, ("= Saint Benedict", 52, WHITE, "ExtraBold")),
        L(6.0, ("protector against ", 44, SOFT, "Bold"), ("SNAKES", 44, GREEN, "Black")),
    ]},
    # Part 2: the capoeira link
    {"start": 15.7, "end": 30.7, "lines": [
        L(15.7, ("CAPOEIRISTA", 96, YELLOW, "Black")),
        L(18.7, ("= ", 96, WHITE, "Black"), ("🐍", 96, "emoji")),
        L(22.8, ("LOW", 56, WHITE, "Black"), (" · ", 56, SOFT, "Bold")),
        L(24.0, ("WAIT", 56, WHITE, "Black"), (" · ", 56, SOFT, "Bold")),
        L(25.2, ("STRIKE", 56, GREEN, "Black")),
    ], "inline_from": 2},
    {"start": 30.85, "end": 38.3, "lines": [
        L(30.85, ("SÃO BENTO", 96, YELLOW, "Black"), (" 🐍", 86, "emoji")),
        L(32.3, ("where the snakes", 50, WHITE, "ExtraBold")),
        L(32.3, ("come out to play", 50, WHITE, "ExtraBold")),
        L(33.0, ("⚠️", 34, "emoji"), (" One of the most popular theories.", 34, SOFT, "SemiBoldItalic"), gap=22),
        L(33.0, ("Mestres tell different stories!", 34, SOFT, "SemiBoldItalic")),
    ]},
    # Part 3: Angola
    {"start": 38.45, "end": 50.4, "lines": [
        L(38.45, ("ANGOLA", 124, YELLOW, "Black")),
        L(38.45, ("São Bento Grande de Angola", 46, WHITE, "ExtraBold")),
        L(41.0, ("traditional · old rodas", 42, SOFT, "Bold"), gap=22),
        L(47.4, ("faster · ", 42, SOFT, "Bold"), ("AGGRESSIVE", 42, GREEN, "Black"), (" ⚡", 42, "emoji")),
    ]},
    # Part 4: Regional
    {"start": 50.85, "end": 64.15, "lines": [
        L(50.85, ("REGIONAL", 124, YELLOW, "Black")),
        L(50.85, ("São Bento Grande da Regional", 46, WHITE, "ExtraBold")),
        L(52.4, ("Mestre Bimba · 1930s", 42, SOFT, "Bold"), gap=22),
        L(58.3, ("different rhythm · ", 42, SOFT, "Bold"), ("FAST", 42, GREEN, "Black"), (" & ", 42, SOFT, "Bold"),
          ("OBJECTIVE", 42, GREEN, "Black")),
    ]},
    # Outro
    {"start": 64.4, "end": 71.35, "lines": [
        L(64.4, ("ANGOLA", 84, YELLOW, "Black"), ("  vs  ", 52, SOFT, "Bold"), ("REGIONAL", 84, GREEN, "Black")),
        L(64.4, ("same name · different beat · different game", 36, SOFT, "Bold"), gap=22),
        L(67.3, ("SAME SAINT ", 52, WHITE, "Black"), ("🐍", 50, "emoji"), gap=18),
        L(68.4, ("2 RHYTHMS · 2 WAYS TO PLAY", 52, WHITE, "Black")),
    ]},
    {"start": 71.4, "end": 99.0, "plate": "light", "lines": [
        L(71.4, ("Can you hear the difference?", 54, BLACK, "Black")),
        L(71.4, ("Tell me in the comments ", 46, (60, 60, 60), "Bold"), ("👇", 46, "emoji")),
        L(75.05, ("AXÉ!", 110, (0, 140, 70), "Black"), gap=20),
    ]},
]


@lru_cache(None)
def font(weight, size):
    return ImageFont.truetype(f"{FONT_DIR}Inter-{weight}.otf", size)


@lru_cache(None)
def emoji_img(ch, size):
    f = ImageFont.truetype(EMOJI, 109)
    im = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((10, 10), ch, font=f, embedded_color=True)
    im = im.crop(im.getbbox())
    h = int(size * 0.95)
    return im.resize((max(1, round(im.width * h / im.height)), h), Image.LANCZOS)


def seg_img(seg):
    text, size = seg[0], seg[1]
    if seg[2] == "emoji":
        lead = len(text) - len(text.lstrip(" "))
        e = emoji_img(text.strip(), size)
        pad = int(size * 0.28) * lead
        im = Image.new("RGBA", (e.width + pad, int(size * 1.22)), (0, 0, 0, 0))
        im.alpha_composite(e, (pad, int(size * 1.22 - e.height) // 2 + int(size * 0.02)))
        return im
    color, weight = seg[2], seg[3]
    f = font(weight, size)
    asc, desc = f.getmetrics()
    w = int(f.getlength(text)) + 2
    im = Image.new("RGBA", (max(1, w), int(size * 1.22)), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((0, int(size * 1.22) - asc - desc + int(size * 0.0)), text, font=f, fill=color + (255,))
    return im


def line_img(segs):
    ims = [seg_img(s) for s in segs]
    h = max(i.height for i in ims)
    im = Image.new("RGBA", (sum(i.width for i in ims), h), (0, 0, 0, 0))
    x = 0
    for s in ims:
        im.alpha_composite(s, (x, h - s.height))
        x += s.width
    return im


@lru_cache(None)
def layout(ci):
    """Pre-render a card: plate + positioned line images."""
    card = CARDS[ci]
    lines = card["lines"]
    inline_from = card.get("inline_from")
    rows = []  # each row: list of (line_index, image)
    for k, ln in enumerate(lines):
        im = line_img(ln["segs"])
        if inline_from is not None and k > inline_from:
            rows[-1].append((k, im))
        else:
            rows.append([(k, im)])
    pad_x, pad_y = 54, 34
    row_w = [sum(i.width for _, i in r) for r in rows]
    row_h = [max(i.height for _, i in r) for r in rows]
    gaps = [lines[r[0][0]]["gap"] if n else 0 for n, r in enumerate(rows)]
    w = min(FRAME_W - 60, max(row_w) + 2 * pad_x)
    h = sum(row_h) + sum(gaps) + 2 * pad_y
    plate = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(plate)
    if card.get("plate") == "light":
        d.rounded_rectangle((0, 0, w - 1, h - 1), 38, fill=(255, 255, 255, 240))
    else:
        d.rounded_rectangle((0, 0, w - 1, h - 1), 38, fill=(16, 16, 16, 212))
        d.rounded_rectangle((w // 2 - 70, 0, w // 2 + 70, 9), 4, fill=YELLOW + (255,))
    placed = []
    y = pad_y
    for n, r in enumerate(rows):
        y += gaps[n]
        x = (w - row_w[n]) // 2
        for k, im in r:
            placed.append((k, np.asarray(im).astype(np.float32), x, y + row_h[n] - im.height))
            x += im.width
        y += row_h[n]
    return np.asarray(plate).astype(np.float32), placed, w, h


def ease(x):
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


def render_card(card, t):
    if not (card["start"] <= t < card["end"]):
        return None
    ci = CARDS.index(card)
    plate, placed, w, h = layout(ci)
    canvas = plate.copy()
    lines = card["lines"]
    for k, im, x, y in placed:
        p = (t - max(lines[k]["at"], card["start"] + 0.05)) / 0.22
        if p <= 0:
            continue
        a = ease(p)
        dy = int(round((1 - a) * 18))
        ih, iw = im.shape[:2]
        y2 = y + dy
        src = im[: max(0, min(ih, h - y2))]
        sa = src[..., 3:4] / 255.0 * a
        dst = canvas[y2:y2 + src.shape[0], x:x + iw]
        dst[..., :3] = dst[..., :3] * (1 - sa) + src[..., :3] * sa
        dst[..., 3:4] = np.maximum(dst[..., 3:4], src[..., 3:4] * a)
    # card in / out
    a_in = ease((t - card["start"]) / 0.25)
    a_out = min(1.0, (card["end"] - t) / 0.18)
    alpha = a_in * a_out
    scale = 0.92 + 0.08 * a_in
    img = canvas
    if scale < 0.999:
        pil = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), "RGBA")
        pil = pil.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
        img = np.asarray(pil).astype(np.float32)
    img = img.copy()
    img[..., 3] *= alpha
    x = (FRAME_W - img.shape[1]) // 2
    y = TOP + (h - img.shape[0]) // 2
    return img, x, y
