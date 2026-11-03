"""Locate the burned-in caption (blue word, centred, lower third) in a frame."""
import cv2
import numpy as np

BAND = (1380, 1800)  # y-range where the caption lives
CX = 585             # captions are horizontally centred (1170 px wide)


def blue_mask(img):
    b, g, r = [img[..., i].astype(np.int16) for i in range(3)]
    return ((b > 200) & (r < 120) & (b - r > 110)).astype(np.uint8)


def find_caption(img):
    """Return (mask_full_frame uint8 0/1, bbox) or (None, None)."""
    y0, y1 = BAND
    m = blue_mask(img[y0:y1])
    # Caption rows are dense with blue; the light-blue shirt only gives sparse specks.
    rows = cv2.blur(m[:, 80:-80].sum(1).astype(np.float32)[:, None], (1, 5)).ravel()
    peak = int(rows.argmax())
    if rows[peak] < 40:
        return None, None
    thr = max(12, 0.15 * rows[peak])
    a = peak
    while a > 0 and rows[a - 1] > thr:
        a -= 1
    b = peak
    while b < len(rows) - 1 and rows[b + 1] > thr:
        b += 1
    a, b = max(0, a - 4), min(len(rows), b + 5)
    if not (30 <= b - a <= 140):
        return None, None
    strip = m[a:b]
    merged = cv2.morphologyEx(strip, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (41, 1)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(merged, 8)
    best = None
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if w < 30 or abs(x + w / 2 - CX) > 60:
            continue
        if strip[y:y + h, x:x + w].mean() < 0.15:
            continue
        if best is None or area > best[4]:
            best = (x, y, w, h, area)
    if best is None:
        return None, None
    x, y, w, h, _ = best
    full = np.zeros(img.shape[:2], np.uint8)
    full[y0 + a + y:y0 + a + y + h, x:x + w] = strip[y:y + h, x:x + w]
    return full, (x, y0 + a + y, w, h)
