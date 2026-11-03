# São Bento Grande reel

Edit of the talking-head reel "Why do we name our capoeira rhythms after a saint?".

| File | What it is |
|---|---|
| `out/sao_bento_grande_mirrored.mp4` | Speaker mirrored, burned-in captions kept readable, plus the cards below |
| `out/sao_bento_grande.mp4` | Same edit, not mirrored |

## What was added (times match what is said)

| Time | Card |
|---|---|
| 0–3.3s | Hook: *Why do we name our capoeira rhythms after a SAINT? 🤔* |
| 3.4–15.4s | **SÃO BENTO 🐍** = Saint Benedict / protector against **SNAKES** |
| 15.7–30.7s | **CAPOEIRISTA = 🐍**, then LOW · WAIT · STRIKE as each word is said |
| 30.9–38.3s | **SÃO BENTO 🐍**, where the snakes come out to play + ⚠️ *One of the most popular theories. Mestres tell different stories!* |
| 38.5–50.4s | **ANGOLA**: São Bento Grande de Angola / traditional · old rodas / faster · AGGRESSIVE ⚡ |
| 50.9–64.2s | **REGIONAL**: São Bento Grande da Regional / Mestre Bimba · 1930s / FAST & OBJECTIVE |
| 64.4–71.3s | **ANGOLA vs REGIONAL** / SAME SAINT 🐍 / 2 RHYTHMS · 2 WAYS TO PLAY |
| 71.4s–end | Can you hear the difference? Tell me in the comments 👇 → **AXÉ!** |

## Mirroring without flipping the captions

The captions are burned into the footage, so a plain flip would make them read backwards.
`render.py` finds the blue caption word in each original frame (`captions.py`), flips the frame,
inpaints away the reversed word, and pastes the original word back on top.

## Adding your B-roll / berimbau toques

Copy `broll.example.json` to `broll.json`, point it at your clips, and run:

    python3 render.py SOURCE.mp4 out/sao_bento_grande_mirrored.mp4 --broll broll.json

Each slot is a full-screen cut-away between `start` and `end` (seconds in the source video).
Your captions are carried onto the B-roll. `audio` mixes in the clip's sound at that volume;
`duck_voice` lowers your voice during the slot (good for playing the two toques back to back).

Requirements: `ffmpeg`, `pip install opencv-python-headless pillow numpy`. Re-run
`analyze.py` to OCR the caption timeline if you swap the source video.
