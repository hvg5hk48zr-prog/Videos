"""OCR the burned-in captions to get a word timeline (JSON)."""
import json, sys, cv2, numpy as np
from rapidocr_onnxruntime import RapidOCR
from captions import find_caption

src, out = sys.argv[1], sys.argv[2]
ocr = RapidOCR()
cap = cv2.VideoCapture(src)
fps = cap.get(cv2.CAP_PROP_FPS)
frames, events, prev, i = [], [], None, 0
while True:
    ok, f = cap.read()
    if not ok:
        break
    m, bb = find_caption(f)
    frames.append(None if bb is None else list(map(int, bb)))
    key = None if bb is None else (bb[0] // 8, bb[2] // 8)
    if key != prev:
        txt = None
        if bb is not None:
            x, y, w, h = bb
            # black text on white, upscaled: much easier for OCR than blue-on-shirt
            crop = 255 - m[y - 10:y + h + 10, x - 10:x + w + 10] * 255
            crop = cv2.copyMakeBorder(crop, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
            crop = cv2.resize(crop, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            res, _ = ocr(cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR))
            txt = " ".join(r[1] for r in res) if res else "?"
        events.append({"t": round(i / fps, 3), "f": i, "w": txt})
        prev = key
    i += 1
json.dump({"fps": fps, "frames": frames, "events": events}, open(out, "w"))
print(i, "frames", len(events), "events")
