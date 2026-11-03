"""Render the São Bento Grande reel.

  python3 render.py SRC.mp4 OUT.mp4 [--no-mirror] [--broll broll.json] [--preview SECONDS]

* Mirrors the speaker while keeping the burned-in captions readable: the caption
  is detected in the original frame, the mirrored (backwards) caption is
  inpainted away, and the original caption is pasted back on top.
* Adds the animated section cards from cards.py (hook, SÃO BENTO 🐍, ANGOLA,
  REGIONAL, outro/CTA).
* Optional full-screen B-roll cut-aways (berimbau, roda footage...) listed in a
  JSON file; your captions are carried over onto the B-roll.
"""
import argparse
import json
import subprocess
import sys

import cv2
import numpy as np

from captions import find_caption
from cards import CARDS, render_card

W, H = 1170, 2080


def mirror_keep_caption(frame):
    mask, bb = find_caption(frame)
    out = cv2.flip(frame, 1)
    if mask is None:
        return out
    x, y, w, h = bb
    keep = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    hole = cv2.dilate(cv2.flip(keep, 1), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    x0 = max(0, min(x, W - x - w) - 30)
    x1 = min(W, max(x + w, W - x) + 30)
    y0, y1 = max(0, y - 30), min(H, y + h + 30)
    roi = np.ascontiguousarray(out[y0:y1, x0:x1])
    out[y0:y1, x0:x1] = cv2.inpaint(roi, hole[y0:y1, x0:x1] * 255, 6, cv2.INPAINT_TELEA)
    paste_caption(out, frame, keep, (y0, y1, x0, x1))
    return out


def paste_caption(dst, src, keep, box):
    y0, y1, x0, x1 = box
    a = cv2.GaussianBlur(keep[y0:y1, x0:x1].astype(np.float32), (0, 0), 1.6)[..., None]
    a = np.clip(a * 1.4, 0, 1)
    d = dst[y0:y1, x0:x1].astype(np.float32)
    dst[y0:y1, x0:x1] = (d * (1 - a) + src[y0:y1, x0:x1].astype(np.float32) * a).astype(np.uint8)


def caption_onto(dst, frame):
    mask, bb = find_caption(frame)
    if mask is None:
        return
    x, y, w, h = bb
    keep = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    paste_caption(dst, frame, keep, (max(0, y - 30), min(H, y + h + 30), max(0, x - 30), min(W, x + w + 30)))


def cover(img):
    h, w = img.shape[:2]
    s = max(W / w, H / h)
    img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    oy, ox = (img.shape[0] - H) // 2, (img.shape[1] - W) // 2
    return img[oy:oy + H, ox:ox + W]


class Broll:
    def __init__(self, slot):
        self.cap = cv2.VideoCapture(slot["file"])
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.cap.set(cv2.CAP_PROP_POS_MSEC, 1000 * slot.get("in", 0.0))
        self.frame, self.k = None, 0

    def at(self, local_t):
        target = int(local_t * self.fps)
        while self.k <= target:
            ok, f = self.cap.read()
            if not ok:
                break
            self.frame, self.k = cover(f), self.k + 1
        return self.frame


def overlay(frame, rgba, x, y):
    h, w = rgba.shape[:2]
    if w == 0 or h == 0:
        return
    xs, ys = max(0, x), max(0, y)
    xe, ye = min(W, x + w), min(H, y + h)
    if xe <= xs or ye <= ys:
        return
    sub = rgba[ys - y:ye - y, xs - x:xe - x].astype(np.float32)
    a = sub[..., 3:4] / 255.0
    d = frame[ys:ye, xs:xe].astype(np.float32)
    frame[ys:ye, xs:xe] = (d * (1 - a) + sub[..., 2::-1] * a).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--broll")
    ap.add_argument("--preview", type=float, help="only render the first N seconds")
    args = ap.parse_args()

    slots = json.load(open(args.broll)) if args.broll else []
    cap = cv2.VideoCapture(args.src)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.preview:
        n = min(n, int(args.preview * fps))
    dur = n / fps

    cmd = ["ffmpeg", "-v", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           "-i", args.src]
    mix = ["[1:a]volume=1.0[v0]"]
    labels = ["[v0]"]
    for k, s in enumerate(slots):
        if s.get("audio", 0) > 0:
            cmd += ["-ss", str(s.get("in", 0.0)), "-t", str(s["end"] - s["start"]), "-i", s["file"]]
            idx = 1 + len(labels)
            ms = int(s["start"] * 1000)
            mix.append(f"[{idx}:a]volume={s['audio']},afade=t=in:d=0.15,"
                       f"afade=t=out:st={max(0, s['end'] - s['start'] - 0.2)}:d=0.2,adelay={ms}|{ms}[b{k}]")
            labels.append(f"[b{k}]")
            if s.get("duck_voice"):
                mix[0] = (f"[1:a]volume='if(between(t,{s['start']},{s['end']}),{s['duck_voice']},1)'"
                          f":eval=frame[v0]")
    mix.append("".join(labels) + f"amix=inputs={len(labels)}:normalize=0:duration=first[a]")
    cmd += ["-filter_complex", ";".join(mix), "-map", "0:v", "-map", "[a]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-t", f"{dur:.3f}", "-movflags", "+faststart", args.out]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    brolls = {}
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        slot = next((s for s in slots if s["start"] <= t < s["end"]), None)
        if slot is not None:
            key = id(slot)
            if key not in brolls:
                brolls[key] = Broll(slot)
            b = brolls[key].at(t - slot["start"])
            if b is None:
                out = frame if args.no_mirror else mirror_keep_caption(frame)
            else:
                out = b.copy()
                caption_onto(out, frame)
        elif args.no_mirror:
            out = frame
        else:
            out = mirror_keep_caption(frame)
        for card in CARDS:
            r = render_card(card, t)
            if r is not None:
                img, x, y = r
                overlay(out, img, x, y)
        enc.stdin.write(out.tobytes())
        if i % 150 == 0:
            print(f"  {t:5.1f}s / {dur:.1f}s", file=sys.stderr, flush=True)
    enc.stdin.close()
    sys.exit(enc.wait())


if __name__ == "__main__":
    main()
