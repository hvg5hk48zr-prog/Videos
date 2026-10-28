import sys, time
from PIL import Image
import render
ts = [float(x) for x in sys.argv[2].split(",")]
ims = []
t0 = time.time()
for t in ts:
    ims.append(Image.fromarray(render.render_frame(int(t * render.FPS))).resize((270, 480)))
print("per frame", (time.time() - t0) / len(ts))
cols = 6
rows = (len(ims) + cols - 1) // cols
sheet = Image.new("RGB", (270 * cols, 480 * rows))
for k, im in enumerate(ims):
    sheet.paste(im, ((k % cols) * 270, (k // cols) * 480))
sheet.save(sys.argv[1])
