"""Pass 1: OCR the burned-in captions to get a word timeline (captions.json)."""
import json, sys, cv2
from rapidocr_onnxruntime import RapidOCR
from captions import find_caption

src, out = sys.argv[1], sys.argv[2]
ocr = RapidOCR()
cap = cv2.VideoCapture(src)
fps = cap.get(cv2.CAP_PROP_FPS)
words, prev, i = [], None, 0
while True:
    ok, f = cap.read()
    if not ok:
        break
    m, bb = find_caption(f)
    key = None if bb is None else (bb[0] // 6, bb[2] // 6)
    if key != prev:
        if bb is not None:
            x, y, w, h = bb
            crop = f[max(0, y - 25):y + h + 25, max(0, x - 25):x + w + 25]
            res, _ = ocr(crop)
            txt = " ".join(r[1] for r in res) if res else "?"
            words.append({"t": round(i / fps, 3), "w": txt, "bbox": list(map(int, bb))})
        else:
            words.append({"t": round(i / fps, 3), "w": None})
        prev = key
    i += 1
json.dump({"fps": fps, "frames": i, "words": words}, open(out, "w"), indent=1)
print(i, "frames", len(words), "events")
