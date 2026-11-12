"""Ungu Capoeira School Brunswick - beginners reel (1080x1920, 30 fps).

Usage: python3 build.py CLIP1.mov CLIP2.mov [--preview]

Cuts two phone clips into a vertical reel, timed to the berimbau that is
playing in the room (the audio of clip 2 is used as one continuous bed).
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS = 1080, 1920, 30
SW, SH = 1440, 1920                      # source size after rotation
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FONT = "/usr/share/fonts/opentype/inter/InterDisplay-Black.otf"
FONT_B = "/usr/share/fonts/opentype/inter/Inter-Bold.otf"
YELLOW, GREEN, WHITE = (255, 212, 0), (0, 168, 89), (255, 255, 255)
HANDLE = "@ungucapoeira.brunswick"

BED_START = 7.5                          # where in clip 2's audio the bed starts
# clip, start (s), duration (s), crop centre x, centre y, zoom, text lines
# durations are snapped to berimbau notes in the bed
SHOTS = [
    (1, 0.55, 2.50, 690, 800, 1.15, [("NEVER TRIED", WHITE), ("CAPOEIRA?", YELLOW)]),
    (2, 14.90, 2.48, 900, 960, 1.00, [("START", WHITE), ("HERE.", YELLOW)]),
    (1, 8.40, 1.95, 850, 790, 1.55, [("MOVE.", WHITE)]),
    (2, 1.90, 1.86, 800, 960, 1.00, [("PLAY.", WHITE)]),
    (1, 15.20, 1.88, 740, 800, 1.40, [("GROW.", YELLOW)]),
    (2, 5.90, 1.80, 620, 960, 1.00, [("NO EXPERIENCE", WHITE), ("NEEDED", YELLOW)]),
    (1, 4.00, 1.72, 740, 820, 1.50, [("EVERYONE STARTS", WHITE), ("AT ZERO", YELLOW)]),
    (2, 18.70, 2.16, 780, 960, 1.00, [("YOUR TURN.", WHITE)]),
    (2, 25.10, 4.85, 720, 960, 1.00, None),          # end card
]


def font(size, path=FONT):
    return ImageFont.truetype(path, size)


def text_layer(lines, max_w=940):
    """Stacked, centred headline with a soft shadow. Returns an RGBA image."""
    n = len(lines)
    size = 230 if n == 1 else 150
    while True:
        f = font(size)
        widths = [f.getbbox(t)[2] - f.getbbox(t)[0] for t, _ in lines]
        if max(widths) <= max_w:
            break
        size -= 4
    lh = int(size * 1.02)
    pad = 60
    img = Image.new("RGBA", (W, lh * n + pad * 2), (0, 0, 0, 0))
    sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d, ds = ImageDraw.Draw(img), ImageDraw.Draw(sh)
    for i, (t, col) in enumerate(lines):
        y = pad + i * lh
        ds.text((W // 2 + 6, y + 10), t, font=f, fill=(0, 0, 0, 230), anchor="ma")
        d.text((W // 2, y), t, font=f, fill=col + (255,), anchor="ma",
               stroke_width=5, stroke_fill=(0, 0, 0, 255))
    sh = sh.filter(ImageFilter.GaussianBlur(14))
    return Image.alpha_composite(sh, img)


def handle_layer():
    f = font(40, FONT_B)
    img = Image.new("RGBA", (W, 90), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((W // 2 + 2, 22), HANDLE, font=f, fill=(0, 0, 0, 170), anchor="ma")
    d.text((W // 2, 20), HANDLE, font=f, fill=(255, 255, 255, 215), anchor="ma")
    return img


def end_card():
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    y = 470
    d.text((W // 2, y), "BEGINNERS WELCOME", font=font(84), fill=WHITE, anchor="ma")
    y += 130
    d.text((W // 2, y), "FIRST CLASS", font=font(150), fill=YELLOW, anchor="ma")
    d.text((W // 2, y + 150), "FREE", font=font(300), fill=YELLOW, anchor="ma")
    y -= 40
    y += 530
    d.rectangle((W // 2 - 150, y, W // 2, y + 10), fill=GREEN)
    d.rectangle((W // 2, y, W // 2 + 150, y + 10), fill=YELLOW)
    y += 60
    d.text((W // 2, y), "UNGU CAPOEIRA SCHOOL", font=font(78), fill=WHITE, anchor="ma")
    d.text((W // 2, y + 100), "BRUNSWICK", font=font(78), fill=WHITE, anchor="ma")
    y += 260
    f = font(52, FONT_B)
    tw = f.getbbox(HANDLE)[2]
    d.rounded_rectangle((W // 2 - tw // 2 - 44, y - 22, W // 2 + tw // 2 + 44, y + 88),
                        radius=55, fill=WHITE)
    d.text((W // 2, y), HANDLE, font=f, fill=(15, 15, 15), anchor="ma")
    d.text((W // 2, y + 140), "DM us to book yours", font=font(56, FONT_B), fill=WHITE, anchor="ma")
    return img


def vignette():
    """Darkening gradient behind the headline area, for legibility."""
    g = np.zeros((H, W), np.float32)
    ys = np.arange(H, dtype=np.float32)
    g += (np.clip((ys - 1000) / 500, 0, 1) * np.clip((1900 - ys) / 300, 0.55, 1) * 0.55)[:, None]
    g += (np.clip((330 - ys) / 330, 0, 1) * 0.35)[:, None]
    return (255 * (1 - g)).astype(np.uint8)


def read_frames(path, start, dur):
    n = round(dur * FPS)
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start}", "-i", path, "-frames:v", str(n),
           "-vf", f"fps={FPS},eq=contrast=1.08:saturation=1.22:brightness=0.01",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = SW * SH * 3
    last = None
    for _ in range(n):
        buf = p.stdout.read(size)
        if len(buf) == size:
            last = Image.frombuffer("RGB", (SW, SH), buf)
        yield last                       # hold the last frame if the clip runs out
    p.stdout.close()
    p.wait()


def crop(frame, cx, cy, zoom):
    cw, ch = W / zoom, H / zoom
    x0 = min(max(cx - cw / 2, 0), SW - cw)
    y0 = min(max(cy - ch / 2, 0), SH - ch)
    return frame.resize((W, H), Image.BICUBIC, box=(x0, y0, x0 + cw, y0 + ch))


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def paste_scaled(base, layer, cy, scale, alpha):
    if alpha <= 0:
        return
    lw, lh = int(layer.width * scale), int(layer.height * scale)
    lay = layer if scale == 1 else layer.resize((lw, lh), Image.BILINEAR)
    if alpha < 1:
        lay = lay.copy()
        lay.putalpha(lay.getchannel("A").point(lambda v: int(v * alpha)))
    base.paste(lay, ((W - lw) // 2, int(cy - lh / 2)), lay)


def main():
    clips = {1: sys.argv[1], 2: sys.argv[2]}
    preview = "--preview" in sys.argv
    os.makedirs(OUT, exist_ok=True)
    total = sum(round(s[2] * FPS) for s in SHOTS) / FPS
    silent = os.path.join(OUT, "_video.mp4")
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast" if preview else "slow",
         "-crf", "26" if preview else "17", "-pix_fmt", "yuv420p", "-colorspace", "bt709",
         "-color_primaries", "bt709", "-color_trc", "bt709", silent], stdin=subprocess.PIPE)
    vig = Image.fromarray(vignette())
    black = Image.new("RGB", (W, H), (0, 0, 0))
    white = Image.new("RGB", (W, H), (255, 255, 255))
    handle = handle_layer()
    card = end_card()
    cover_done = False
    for si, (clip, start, dur, cx, cy, zoom, lines) in enumerate(SHOTS):
        n = round(dur * FPS)
        layer = text_layer(lines) if lines else None
        for i, fr in enumerate(read_frames(clips[clip], start, dur)):
            t = i / FPS
            z = zoom * (1 + 0.07 * i / n)                       # slow push-in
            img = crop(fr, cx, cy, z)
            if lines:
                img = Image.composite(img, black, vig)
                a = ease(t / 0.16)
                out = ease((t - (dur - 0.10)) / 0.10)
                paste_scaled(img, layer, 1330, 1.22 - 0.22 * a, a * (1 - out))
                paste_scaled(img, handle, 285, 1, 1)
            else:
                k = ease((t - 1.1) / 0.6)                        # laugh first, then the card
                img = Image.blend(img, black, 0.68 * k)
                if k > 0:
                    paste_scaled(img, card, H / 2 + 30 * (1 - k), 1, k)
                else:
                    paste_scaled(img, handle, 285, 1, 1)
                if t > dur - 0.4:
                    img = Image.blend(img, black, ease((t - (dur - 0.4)) / 0.4))
            if si and i < 3:                                     # flash on the cut
                img = Image.blend(img, white, (0.35, 0.18, 0.07)[i])
            if not cover_done and si == 0 and i == 24:
                img.save(os.path.join(OUT, "cover.jpg"), quality=93)
                cover_done = True
            enc.stdin.write(img.tobytes())
    enc.stdin.close()
    enc.wait()

    final = os.path.join(OUT, "ungu_capoeira_beginners_reel.mp4")
    af = (f"atrim={BED_START}:{BED_START + total},asetpts=PTS-STARTPTS,highpass=f=70,"
          "afftdn=nr=8:nf=-45,loudnorm=I=-14:TP=-1.5:LRA=11,"
          f"afade=t=in:d=0.08,afade=t=out:st={total - 0.9}:d=0.9,aresample=48000")
    subprocess.check_call(
        ["ffmpeg", "-v", "error", "-y", "-i", silent, "-i", clips[2], "-filter_complex",
         f"[1:a]{af}[a]", "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac",
         "-b:a", "192k", "-ac", "2", "-movflags", "+faststart", "-shortest", final])
    os.remove(silent)
    print(final, total)


if __name__ == "__main__":
    main()
