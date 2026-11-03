"""Replace the burned-in English captions with Portuguese ones (same place, same look).

  python3 analyze.py SRC.mp4 caps.json          # caption timeline (English words + per-frame boxes)
  python3 subtitles_pt.py SRC.mp4 caps.json OUT.mp4

Each Portuguese sentence is shown over the time the matching English sentence is
spoken, one or two words at a time like the original, and only while the English
caption was on screen (so pauses stay pauses).
"""
import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from captions import blue_mask

FONT = str(Path(__file__).with_name("fonts") / "Fraunces.ttf")
BLUE = (3, 143, 251)

# (start seconds, Portuguese sentence) – each runs until the next start.
SEGMENTS = [
    (0.0, "São Bento é São Benedito."),
    (1.65, "Na fé popular brasileira, ele é o protetor contra todo perigo, por exemplo, as cobras."),
    (7.6, "O povo rezava para ele pedindo proteção contra todo perigo."),
    (12.15, "Por exemplo, contra uma picada de cobra."),
    (15.6, "E o que é um capoeirista?"),
    (17.25, "Um capoeirista é alguém que pode ser perigoso, como uma cobra!"),
    (22.7, "Eles jogam baixo, esperam, e aí dão o bote."),
    (26.5, "É por isso que tantas músicas de capoeira falam da cobra."),
    (30.9, "Então São Bento virou o toque do jogo em que as cobras saem para brincar, "
           "em que os capoeiristas saem para jogar."),
    (38.4, "São Bento Grande de Angola é o toque tradicional das rodas antigas, "
           "mais antigo que qualquer estilo ou escola."),
    (46.4, "Geralmente é tocado mais rápido, para um jogo mais ligeiro e mais agressivo."),
    (50.6, "Por outro lado, temos a Capoeira Regional de Mestre Bimba, nos anos 30."),
    (55.5, "E ele fez sua própria versão do São Bento Grande, com outro ritmo."),
    (59.7, "É o toque principal do jogo rápido e objetivo da Regional."),
    (64.3, "Mesmo nome, outra batida, outro jogo."),
    (67.15, "Mesmo santo, dois ritmos, dois jeitos de jogar."),
    (70.3, "Você consegue ouvir a diferença?"),
    (71.45, "Me conta nos comentários."),
    (72.45, "Se quiser aprender a segunda parte com mais detalhes…"),
    (75.1, "Axé!"),
]


def chunks(sentence):
    """One word per caption, but glue short words (o, de, é, ...) to the next one."""
    words, out, buf = sentence.split(), [], ""
    for i, w in enumerate(words):
        buf = f"{buf} {w}".strip()
        bare = w.strip(",.!?…")
        nxt = words[i + 1] if i < len(words) - 1 else ""
        if (len(bare) <= 3 and nxt and len(buf) + 1 + len(nxt) <= 15
                and not w.endswith((",", ".", "?", "!"))):
            continue
        out.append(buf)
        buf = ""
    if buf:
        out.append(buf)
    return out


def timeline(boxes, fps):
    """Per frame: Portuguese chunk to show (or None)."""
    n = len(boxes)
    vis = np.array([b is not None for b in boxes])
    # bridge detector dropouts of a few frames
    for i in range(n):
        if not vis[i]:
            j = i
            while j < n and not boxes[j]:
                j += 1
            if 0 < i and j < n and j - i <= 4 and vis[i - 1]:
                vis[i:j] = True
    out = [None] * n
    for k, (start, text) in enumerate(SEGMENTS):
        end = SEGMENTS[k + 1][0] if k + 1 < len(SEGMENTS) else n / fps
        frames = [i for i in range(int(start * fps), min(n, int(end * fps))) if vis[i]]
        if not frames:
            continue
        cs = chunks(text)
        weights = np.array([len(c) + 4 for c in cs], float)
        edges = np.concatenate([[0], np.cumsum(weights) / weights.sum()]) * len(frames)
        for c, (a, b) in zip(cs, zip(edges[:-1], edges[1:])):
            for i in frames[int(round(a)):int(round(b))]:
                out[i] = c
    return out


@lru_cache(None)
def caption_img(text, size=86):
    f = ImageFont.truetype(FONT, size)
    f.set_variation_by_axes([72, 900, 100, 0])  # opsz, wght, SOFT, WONK
    l, t, r, b = f.getbbox(text, stroke_width=5)
    pad = 24
    w, h = r - l + 2 * pad, b - t + 2 * pad
    pos = (pad - l, pad - t)
    shadow = Image.new("L", (w, h), 0)
    ImageDraw.Draw(shadow).text((pos[0] + 2, pos[1] + 4), text, font=f, fill=150, stroke_width=5, stroke_fill=150)
    shadow = shadow.filter(ImageFilter.GaussianBlur(5))
    im = Image.new("RGBA", (w, h), (20, 40, 80, 0))
    im.putalpha(shadow)
    ImageDraw.Draw(im).text(pos, text, font=f, fill=BLUE + (255,), stroke_width=5, stroke_fill=(255, 255, 255, 255))
    a = np.asarray(im).astype(np.float32)
    return a, (-l + pad, -t + pad)


def remove_caption(frame, box):
    x, y, w, h = box
    x0, y0 = max(0, x - 25), max(0, y - 25)
    x1, y1 = min(frame.shape[1], x + w + 25), min(frame.shape[0], y + h + 40)
    roi = np.ascontiguousarray(frame[y0:y1, x0:x1])
    m = blue_mask(roi)
    m[:max(0, y - 8 - y0)] = 0  # keep descenders (g, y, p, ç) below the box
    if m.sum() == 0:
        return
    hole = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    frame[y0:y1, x0:x1] = cv2.inpaint(roi, hole * 255, 7, cv2.INPAINT_TELEA)


def draw(frame, text, cx, base_y):
    img, (ox, oy) = caption_img(text)
    h, w = img.shape[:2]
    x, y = int(cx - w / 2), int(base_y - img.shape[0] + 24)  # baseline-ish anchor
    xs, ys, xe, ye = max(0, x), max(0, y), min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    sub = img[ys - y:ye - y, xs - x:xe - x]
    a = sub[..., 3:4] / 255.0
    d = frame[ys:ye, xs:xe].astype(np.float32)
    frame[ys:ye, xs:xe] = (d * (1 - a) + sub[..., 2::-1] * a).astype(np.uint8)


def main():
    src, caps, out = sys.argv[1:4]
    data = json.load(open(caps))
    boxes, fps = data["frames"], data["fps"]
    words = timeline(boxes, fps)
    det = [b for b in boxes if b]
    cx = 1170 / 2
    base_y = int(np.median([b[1] + b[3] for b in det]))  # bottom of the English words

    cap = cv2.VideoCapture(src)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps),
         "-i", "-", "-i", src, "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", "-movflags", "+faststart", out],
        stdin=subprocess.PIPE)
    n = len(boxes)
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        # remove the English word; if the detector dropped this frame, use a neighbour's box
        box = boxes[i] or next((boxes[j] for d in (1, -1, 2, -2, 3, -3)
                                if 0 <= (j := i + d) < n and boxes[j]), None)
        if box:
            remove_caption(frame, box)
        if words[i]:
            draw(frame, words[i], cx, base_y)
        enc.stdin.write(frame.tobytes())
    enc.stdin.close()
    sys.exit(enc.wait())


if __name__ == "__main__":
    main()
