#!/usr/bin/env python3
"""40-second Portuguese cut, hook-first.

Re-times shots from the 60 s master (timeline.py) through an edit list,
overlays Portuguese text and subtitles, and builds a matching score with
Portuguese narration.

    PIPER_VOICE_PT=/path/pt-br-edresson-low.onnx python3 cut_pt.py [audio|video|mux]
"""
import json
import os
import subprocess
import sys
import time
import wave
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import audio as A          # noqa: E402
import build as B          # noqa: E402
import scenes as S         # noqa: E402
import timeline as T       # noqa: E402

DUR = 40.0
FPS = 30
NF = int(DUR * FPS)
BUILD = os.path.join(ROOT, "build", "pt")
OUT = os.path.join(ROOT, "out")
os.makedirs(BUILD, exist_ok=True)
VOICE = os.environ.get("PIPER_VOICE_PT", os.path.join(ROOT, "build", "voice", "pt-br-edresson-low.onnx"))

# (out start, out end, source start, source end) — source times refer to timeline.SHOTS
EDIT = [
    # gancho: the question is on screen from the first frame
    (0.00, 0.55, 1.23, 1.355), (0.55, 1.10, 1.80, 2.055), (1.10, 1.65, 2.45, 2.705),
    (1.65, 2.20, 3.00, 3.255), (2.20, 2.60, 4.00, 4.40),
    # Chico Rei
    (2.60, 4.60, 5.00, 7.00), (4.60, 7.80, 11.00, 14.20), (7.80, 9.60, 15.20, 17.00), (9.60, 11.60, 17.60, 19.60),
    # Dragão do Mar
    (11.60, 13.40, 20.00, 21.80), (13.40, 14.40, 23.00, 24.00), (14.40, 17.20, 25.20, 28.00),
    (17.20, 19.20, 29.50, 31.50), (19.20, 22.60, 33.00, 36.40),
    # legado
    (22.60, 23.80, 38.00, 39.20), (23.80, 25.10, 39.30, 40.60), (25.10, 26.20, 40.60, 41.69),
    (26.20, 27.30, 41.70, 42.79), (27.30, 28.40, 42.80, 43.89), (28.40, 29.70, 43.90, 45.19),
    (29.70, 30.90, 45.20, 46.39), (30.90, 32.10, 46.40, 47.59), (32.10, 35.60, 47.60, 49.99),
    # a pergunta final
    (35.60, 37.60, 50.00, 52.00), (37.60, 40.00, 54.00, 56.40),
]


def src_time(t):
    for o0, o1, s0, s1 in EDIT:
        if o0 <= t < o1:
            return s0 + (t - o0) * (s1 - s0) / (o1 - o0)
    return EDIT[-1][3]


def out_time(ts):
    """Map a source time to output time, or None if it was cut."""
    for o0, o1, s0, s1 in EDIT:
        if s0 <= ts < s1:
            return o0 + (ts - s0) * (o1 - o0) / (s1 - s0)
    return None


LABEL = "RECONSTITUIÇÃO ARTÍSTICA — NÃO SÃO IMAGENS DE ARQUIVO"
TEXT = [
    (0.00, 2.60, "QUEM SÃO OS\nVERDADEIROS\nHERÓIS\nDO BRASIL?", T.SERIF, 104, T.WHITE, 0.47, 0.05, 0.0, 0.15, 0.5),
    (2.80, 4.50, "VILA RICA · MINAS GERAIS · SÉCULO XVIII", T.SANS_M, 28, (0.95, 0.85, 0.7), 0.22, 0.2, 0.4, 0.4, 0.0),
    (5.40, 9.60, "CHICO REI", T.SERIF, 150, T.GOLDC, 0.54, 0.08, 0.5, 0.3, 0.8),
    (5.90, 9.60, "LIBERDADE, DIGNIDADE,\nRESISTÊNCIA.", T.SANS_B, 50, T.WHITE, 0.635, 0.12, 0.4, 0.3, 0.3),
    (7.90, 11.40, "Diz a tradição oral que ele comprou\nsua liberdade com o ouro das minas —\ne ajudou a libertar outros.",
     T.SANS_M, 33, (0.95, 0.9, 0.82), 0.25, 0.0, 0.4, 0.4, 0.0),
    (11.80, 13.30, "FORTALEZA · CEARÁ · 1881", T.SANS_M, 30, T.SEAC, 0.22, 0.22, 0.4, 0.3, 0.0),
    (14.60, 17.10, "Os jangadeiros se recusaram\na levar pessoas escravizadas\naos navios negreiros.",
     T.SANS_M, 36, T.WHITE, 0.30, 0.0, 0.35, 0.3, 0.0),
    (17.40, 19.10, "Nenhuma jangada partiu.\nO tráfico no porto parou.", T.SANS_SB, 40, T.WHITE, 0.30, 0.0, 0.35, 0.3, 0.0),
    (19.50, 22.60, "DRAGÃO\nDO MAR", T.SERIF, 150, T.SEAC, 0.585, 0.08, 0.4, 0.25, 0.7),
    (20.20, 22.60, "O HOMEM QUE DISSE NÃO.", T.SANS_B, 50, T.WHITE, 0.71, 0.12, 0.4, 0.25, 0.3),
    (20.80, 22.50, "Francisco José do Nascimento", T.SANS_M, 32, (0.85, 0.9, 0.95), 0.755, 0.04, 0.4, 0.3, 0.0),
    (35.80, 37.40, "QUEM\nCHAMAMOS\nDE HERÓI?", T.SERIF, 112, T.WHITE, 0.46, 0.06, 0.4, 0.25, 0.5),
    (37.75, 40.00, "LEMBRE-SE\nDE QUEM\nLUTOU PELA\nLIBERDADE.", T.SERIF, 100, T.GOLDC, 0.46, 0.06, 0.4, 0.0, 0.6),
    (38.40, 40.00, "CHICO REI  ·  DRAGÃO DO MAR", T.SANS_M, 30, (0.85, 0.8, 0.7), 0.71, 0.24, 0.4, 0.0, 0.0),
    (2.60, 35.60, LABEL, T.SANS_M, 19, (0.8, 0.8, 0.8), 0.115, 0.18, 0.5, 0.3, 0.0),
]
NARRATION = [
    ("Um lutou para reconquistar\na liberdade.", "Um lutou para reconquistar a liberdade."),
    ("O outro transformou a recusa\nem resistência coletiva.", "O outro transformou a recusa em resistência coletiva."),
    ("Eles nos lembram: a história também\nfoi feita por quem lutou\ncontra a opressão.",
     "Eles nos lembram: a história também foi feita por quem lutou contra a opressão."),
]
NARR_START, NARR_GAP, NARR_LIMIT = 22.75, 0.22, 35.5
FADE = (37.25, 37.60)


# ---------------------------------------------------------------- narration
def narration():
    SR = A.SR
    segs, t = [], NARR_START
    track = np.zeros(int(DUR * SR), np.float32)
    for i, (sub, line) in enumerate(NARRATION):
        raw, rs = os.path.join(BUILD, f"narr{i}.wav"), os.path.join(BUILD, f"narr{i}_48k.wav")
        B.run(["piper", "-m", VOICE, "--length_scale", "0.88", "--sentence_silence", "0", "-f", raw],
              input=line.encode(), stderr=subprocess.DEVNULL)
        B.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", raw, "-af",
               "silenceremove=start_periods=1:start_threshold=-40dB,areverse,"
               "silenceremove=start_periods=1:start_threshold=-40dB,areverse,"
               "aresample=48000:resampler=soxr,equalizer=f=150:t=q:w=1:g=3,equalizer=f=3000:t=q:w=1:g=2,"
               "highpass=f=70", "-ar", "48000", "-ac", "1", rs])
        with wave.open(rs) as wf:
            v = np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float32) / 32768
        i0 = int(t * SR)
        track[i0:i0 + len(v)] += v
        segs.append(dict(start=round(t, 3), end=round(t + len(v) / SR, 3), sub=sub, text=line))
        t += len(v) / SR + NARR_GAP
    assert segs[-1]["end"] < NARR_LIMIT, f"narration overruns: {segs[-1]['end']:.2f}s"
    track /= max(1e-6, np.abs(track).max()) / 0.9
    with wave.open(os.path.join(BUILD, "narration.wav"), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((track * 32767).astype(np.int16).tobytes())
    json.dump(segs, open(os.path.join(BUILD, "narration.json"), "w"), indent=1, ensure_ascii=False)
    print("narração:", [(s["start"], s["end"]) for s in segs])
    return segs


# ---------------------------------------------------------------- score
def score(narr_wav, segs, path):
    A.N = int(DUR * A.SR)
    SR, N = A.SR, A.N
    music, sfx, amb = A.Bus(), A.Bus(), A.Bus()
    tt = A.tt

    # gancho — a hit on frame one, heartbeat and chains under the flashes
    for k, t in enumerate((0.0, 0.55, 1.10, 1.65)):
        sfx.add(t, A.boom(0.7 if k else 1.0, 120, 40, 1.4))
        if k:
            sfx.add(t - 0.3, A.whoosh(0.3, 0.25))
    for t in (0.05, 0.75, 1.45, 2.15):
        sfx.add(t, A.heartbeat(0.95))
    for t, p in ((0.2, -0.4), (0.9, 0.4), (1.7, 0.0)):
        sfx.add(t, A.rattle(7, 0.45, 0.35), pan=p)
    sfx.add(1.10, A.filt(A.noise(int(1.0 * SR)), 100, 4000) * A.env(int(1.0 * SR), 0.01, 0.35) * 0.3)
    music.add(0.0, A.pad(2.6, [26, 38, 45], 0.6, a=0.02, r=0.4, bright=4))
    sfx.add(2.3, A.whoosh(0.3, 0.4))

    # Chico Rei
    sfx.add(2.6, A.boom(0.9, 90, 30, 2.5))
    amb.add(2.6, A.filt(A.noise(int(9 * SR), "brown"), 30, 300) * A.adsr(9, 0.2, 0.6) * 0.35)
    for td in A.rng.uniform(2.8, 11.2, 6):
        amb.add(td, A.drip(0.12), pan=A.rng.uniform(-0.7, 0.7))
    for ts, i in S.pick_strike_times(5.0, 7.0):
        sfx.add(out_time(ts), A.pick_strike((0.8, 0.45, 0.3)[i]), pan=(-0.3, 0.3, 0.6)[i])
    music.add(2.6, A.pad(2.2, [26, 38, 50, 53, 57], 0.7, a=0.6, r=0.4, bright=6))
    music.add(4.6, A.pad(3.2, [26, 38, 50, 53, 57, 62], 0.75, a=0.8, r=0.3, bright=9, vib=0.002))
    sfx.add(5.4, A.boom(0.7, 70, 30, 2.5))
    sfx.add(5.4, A.bell(74, 0.25, 3.0), pan=0.2)
    music.add(7.8, A.pad(1.8, [34, 46, 50, 53, 58, 62], 0.75, a=0.2, r=0.3, bright=9, vib=0.002))
    music.add(9.6, A.pad(2.1, [29, 41, 53, 57, 60, 65], 0.8, a=0.2, r=0.4, bright=10, vib=0.002))
    for k, tb in enumerate(np.sort(A.rng.uniform(7.9, 11.4, 16))):
        sfx.add(tb, A.bell(int(A.rng.choice([74, 77, 79, 81, 84, 86])), 0.08 + 0.004 * k), pan=A.rng.uniform(-0.8, 0.8))
    t_open = out_time(17.6 + S.SHACKLE_OPEN)
    sfx.add(t_open, A.metal_hit(0.9, 380))
    sfx.add(t_open, A.rattle(5, 0.3, 0.4, 1200))
    sfx.add(t_open + 0.55, A.metal_hit(0.35, 300), pan=0.3)
    sfx.add(11.1, A.whoosh(0.5, 0.5))

    # Dragão do Mar
    sfx.add(11.6, A.boom(1.0, 85, 28, 3.0))
    n = int(11.0 * SR)
    sea = A.filt(A.noise(n, "pink"), 60, 1800) * 0.25
    wind = A.filt(A.noise(n), 300, 1400) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.13 * tt(11.0) + 1))
    amb.add(11.6, (sea + wind * 0.12) * A.adsr(11.0, 0.05, 0.3))
    crashes = []
    for o0, o1, s0, s1 in EDIT:
        if 11.6 <= o0 < 22.6 or 26.2 <= o0 < 27.3:
            crashes += [out_time(tc) for tc in S.sea_crash_times(s0, s1)]
    for tc in crashes:
        L = 2.2
        e = np.concatenate([np.linspace(0, 1, int(0.45 * SR)) ** 2, np.exp(-np.arange(int((L - 0.45) * SR)) / SR / 0.6)])
        amb.add(tc - 0.45, A.filt(A.noise(len(e)), 150, 7000) * e * 0.5, pan=A.rng.uniform(-0.4, 0.4))
    music.add(11.6, A.pad(5.6, [26, 38, 45], 0.8, a=0.4, r=0.4, bright=6))
    music.add(14.4, A.pad(2.8, [50, 53, 57], 0.35, a=1.0, r=0.3, bright=6))
    music.add(17.2, A.pad(2.0, [31, 43, 50, 55, 58], 0.6, a=0.4, r=0.2, bright=8))
    beat = 0.6
    for k in range(int((19.2 - 14.4) / beat)):
        tb = 14.4 + k * beat
        sfx.add(tb, A.drum(0.3 + 0.5 * k / 8 + (0.2 if k % 4 == 0 else 0), 70))
        if k % 2 == 1 and tb > 17.2:
            sfx.add(tb + beat / 2, A.drum(0.25, 110), pan=0.3)
    sfx.add(18.7, A.whoosh(0.5, 0.6))
    sfx.add(19.2, A.boom(1.2, 75, 26, 3.5))
    sfx.add(19.2, A.drum(1.0, 60))
    music.add(19.2, A.pad(1.9, [34, 46, 53, 58, 62, 65], 0.85, a=0.05, r=0.4, bright=10, vib=0.002))
    music.add(21.1, A.pad(1.6, [33, 45, 52, 57, 61, 64], 0.85, a=0.2, r=0.2, bright=10, vib=0.002))
    for k in range(4):
        sfx.add(20.2 + k * 0.6, A.drum(0.5 + k * 0.1, 66))
    sfx.add(22.2, A.whoosh(0.4, 0.6))

    # legado, under the narration
    sfx.add(22.6, A.boom(0.9, 90, 30, 2.5))
    music.add(22.6, A.pad(6.0, [26, 38, 50, 53, 57, 62], 0.7, a=0.3, r=0.3, bright=9, vib=0.002))
    music.add(28.6, A.pad(3.5, [34, 46, 50, 53, 58, 62], 0.75, a=0.3, r=0.3, bright=9, vib=0.002))
    music.add(32.1, A.pad(3.5, [29, 41, 48, 53, 57, 60, 65], 0.85, a=0.3, r=0.0, bright=10, vib=0.002))
    for k in range(int(13.0 / 0.5)):
        sfx.add(22.6 + k * 0.5, A.drum(0.22 + (0.18 if k % 4 == 0 else 0) + 0.25 * k / 26, 68))
    for tb in (out_time(39.3 + 1.3 * 0.45), out_time(43.9 + 1.3 * 0.5)):
        sfx.add(tb - 0.4, A.whoosh(0.4, 0.4))
        sfx.add(tb, A.metal_hit(1.0, 520))
        sfx.add(tb, A.rattle(8, 0.6, 0.6, 1500))
        sfx.add(tb, A.boom(0.5, 140, 50, 1.0))
    for tcut in (25.1, 26.2, 27.3, 29.7, 30.9, 32.1):
        sfx.add(tcut, A.boom(0.3, 110, 45, 0.8))
    amb.add(26.2, A.filt(A.noise(int(1.1 * SR), "pink"), 60, 1800) * 0.3 * A.adsr(1.1, 0.02, 0.1))
    sfx.add(34.9, A.whoosh(0.7, 0.5))

    # a pergunta final — silence, a heartbeat, then the answer
    sfx.add(35.8, A.boom(0.8, 70, 26, 2.0))
    sfx.add(36.6, A.heartbeat(0.5))
    sfx.add(37.75, A.boom(1.0, 70, 26, 2.2))
    music.add(37.75, A.pad(2.25, [26, 38, 50, 54, 57, 62, 66], 0.8, a=0.4, r=0.9, bright=8, vib=0.002))
    sfx.add(37.75, A.bell(74, 0.25, 2.2))
    sfx.add(38.0, A.bell(81, 0.12, 2.0), pan=0.4)

    wet = A.reverb(music.x * 0.8 + sfx.x * 0.35)
    mix = music.x * 0.55 + sfx.x * 0.9 + amb.x * 0.75 + wet * 0.45
    duck = np.ones(N, np.float32)
    voice = np.zeros((N, 2), np.float32)
    if narr_wav:
        with wave.open(narr_wav) as wf:
            v = np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float32) / 32768
        v = A.filt(v, 80, 8000)
        voice[:len(v)] = np.stack([v, v], 1)[:N]
        for seg in segs:
            duck[int((seg["start"] - 0.25) * SR):int((seg["end"] + 0.3) * SR)] = 0.42
        k = int(0.12 * SR)
        duck = np.convolve(duck, np.ones(k) / k, mode="same").astype(np.float32)
        voice = voice + A.reverb(voice, 1.6, 0.35, 5000) * 0.12
    mix = mix * duck[:, None] + voice * 1.25
    # hard silence after the legacy section; only the dry hits/heartbeat remain
    i0, i1, i2 = int(35.6 * SR), int(35.64 * SR), int(37.7 * SR)
    mix[i0:i1] *= np.linspace(1, 0, i1 - i0)[:, None]
    mix[i1:i2] = sfx.x[i1:i2] * 0.9
    fo = int(0.25 * SR)
    mix[-fo:] *= np.linspace(1, 0, fo)[:, None]
    mix = np.tanh(mix * 0.9) / 0.9
    mix /= max(1e-6, np.abs(mix).max()) / 0.89
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((np.clip(mix, -1, 1) * 32767).astype(np.int16).tobytes())


# ---------------------------------------------------------------- picture
_subs = None


def items(t):
    global _subs
    if _subs is None:
        p = os.path.join(BUILD, "narration.json")
        _subs = json.load(open(p)) if os.path.exists(p) else []
    out = []
    for (t0, t1, s, font, size, color, y, tr, fi, fo, glow) in TEXT:
        if t0 <= t <= t1:
            a = min(B_sm((t - t0) / fi) if fi else 1.0, B_sm((t1 - t) / fo) if fo else 1.0)
            a *= 0.55 if s is LABEL else 1.0
            dy = (1 - B_sm((t - t0) / max(fi, 0.01) * 0.7)) * 18 if fi else 0.0
            out.append((s, font, size, color, y, tr, glow, a, dy))
    for seg in _subs:
        t0, t1 = seg["start"], seg["end"] + 0.25
        if t0 <= t <= t1:
            a = min(B_sm((t - t0) / 0.15), B_sm((t1 - t) / 0.2))
            out.append((seg["sub"], T.SANS_SB, 44, (1.0, 0.98, 0.94), 0.70, 0.0, 0.0, a, 0.0))
    return out


def B_sm(x):
    import gfx
    return gfx.sm(x)


def frame(i):
    import render
    t = i / FPS
    f = render.render_frame(i, with_text=False, t=src_time(t)).astype(np.float32) / 255
    if FADE[0] <= t < FADE[1]:
        f *= 1 - (t - FADE[0]) / (FADE[1] - FADE[0])
    for (s, font, size, color, y, tr, glow, a, dy) in items(t):
        render.draw_text(f, render.text_layer(s, font, size, color, tr, glow), y, a, dy)
    return (np.clip(f, 0, 1) * 255).astype(np.uint8).tobytes()


def video():
    path = os.path.join(BUILD, "picture.mp4")
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1080x1920",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "18",
                           "-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "+faststart", path],
                          stdin=subprocess.PIPE)
    t0 = time.time()
    with Pool(os.cpu_count()) as pool:
        for k, buf in enumerate(pool.imap(frame, range(NF), chunksize=4)):
            ff.stdin.write(buf)
            if k % 150 == 0:
                print(f"frame {k}/{NF}  {time.time() - t0:.0f}s", flush=True)
    ff.stdin.close()
    ff.wait()


def mux():
    pic = os.path.join(BUILD, "picture.mp4")
    for mix, name in (("mix_voz.wav", "herois_do_brasil_40s_pt.mp4"),
                      ("mix_sem_narracao.wav", "herois_do_brasil_40s_pt_sem_narracao.mp4")):
        norm = os.path.join(BUILD, "norm_" + mix)
        B.loudnorm(os.path.join(BUILD, mix), norm)
        B.run(["ffmpeg", "-v", "error", "-y", "-i", pic, "-i", norm, "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
               "-shortest", "-movflags", "+faststart", os.path.join(OUT, name)])


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("audio", "all"):
        segs = narration()
        score(os.path.join(BUILD, "narration.wav"), segs, os.path.join(BUILD, "mix_voz.wav"))
        score(None, [], os.path.join(BUILD, "mix_sem_narracao.wav"))
    if step in ("video", "all"):
        video()
    if step in ("mux", "all"):
        mux()
