"""Synthesised score and sound design, mixed with the narration. All events are
timed from the same constants the picture uses, so hits land on the cuts."""
import json
import math
import os
import wave

import numpy as np

import scenes as S

SR = 48000
DUR = 60.0
N = int(SR * DUR)
rng = np.random.default_rng(2024)


# ---------------------------------------------------------------- dsp
def filt(x, lo=None, hi=None, order=2):
    X = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(x.shape[0], 1 / SR)
    g = np.ones_like(f)
    if lo:
        g *= 1 / np.sqrt(1 + (lo / np.maximum(f, 1e-3)) ** (2 * order))
    if hi:
        g *= 1 / np.sqrt(1 + (f / hi) ** (2 * order))
    if x.ndim == 2:
        g = g[:, None]
    return np.fft.irfft(X * g, n=x.shape[0], axis=0).astype(np.float32)


def noise(n, color="white"):
    x = rng.standard_normal(n).astype(np.float32)
    if color == "pink":
        X = np.fft.rfft(x)
        f = np.fft.rfftfreq(n, 1 / SR)
        f[0] = 1
        x = np.fft.irfft(X / np.sqrt(f), n=n).astype(np.float32)
    elif color == "brown":
        X = np.fft.rfft(x)
        f = np.fft.rfftfreq(n, 1 / SR)
        f[0] = 1
        x = np.fft.irfft(X / f, n=n).astype(np.float32)
    return x / (np.abs(x).max() + 1e-9)


def tt(dur):
    return np.arange(int(dur * SR), dtype=np.float32) / SR


class Bus:
    def __init__(self):
        self.x = np.zeros((N, 2), np.float32)

    def add(self, t0, sig, gain=1.0, pan=0.0):
        i0 = int(t0 * SR)
        if sig.ndim == 1:
            l, r = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
            sig = np.stack([sig * l, sig * r], 1) * 1.414
        j0, j1 = max(0, i0), min(N, i0 + len(sig))
        if j1 > j0:
            self.x[j0:j1] += sig[j0 - i0:j1 - i0] * gain


def env(n, a, d, sustain_to=None):
    t = np.arange(n, dtype=np.float32) / SR
    e = np.minimum(1, t / max(a, 1e-4)) * np.exp(-np.maximum(0, t - a) / d)
    return e


def adsr(dur, a, r):
    t = tt(dur)
    return (np.minimum(1, t / a) * np.minimum(1, np.maximum(0, dur - t) / max(r, 1e-3))).astype(np.float32)


# ---------------------------------------------------------------- instruments
def boom(amp=1.0, f0=95, f1=32, dur=2.2):
    t = tt(dur)
    f = f1 + (f0 - f1) * np.exp(-t * 6)
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) * env(len(t), 0.004, 0.7)
    s += filt(noise(len(t)), hi=180) * env(len(t), 0.002, 0.25) * 1.2
    s += filt(noise(len(t)), 200, 3000) * env(len(t), 0.001, 0.05) * 0.4
    return s * amp


def heartbeat(amp=1.0):
    out = np.zeros(int(0.6 * SR), np.float32)
    for dt, a in ((0.0, 1.0), (0.21, 0.7)):
        t = tt(0.3)
        f = 38 + 30 * np.exp(-t * 25)
        s = np.sin(2 * np.pi * np.cumsum(f) / SR) * env(len(t), 0.006, 0.07)
        s += filt(noise(len(t)), hi=120) * env(len(t), 0.003, 0.04) * 0.5
        i = int(dt * SR)
        out[i:i + len(s)] += s * a
    return out * amp


def clink(f0, dur=0.35, amp=1.0):
    t = tt(dur)
    s = np.zeros_like(t)
    for k, a in ((1.0, 1.0), (2.76, 0.6), (5.40, 0.4), (8.93, 0.25), (13.3, 0.12)):
        s += a * np.sin(2 * np.pi * f0 * k * t + rng.random() * 6) * np.exp(-t * (8 + k * 4))
    s += filt(noise(len(t)), 2000, 9000) * env(len(t), 0.0005, 0.006) * 1.5
    return s * amp * 0.5


def rattle(n=6, spread=0.45, amp=1.0, base=1700):
    out = np.zeros(int((spread + 0.5) * SR), np.float32)
    for _ in range(n):
        s = clink(base * rng.uniform(0.8, 1.4), 0.3, rng.uniform(0.4, 1.0))
        i = int(rng.uniform(0, spread) * SR)
        out[i:i + len(s)] += s
    return out * amp


def metal_hit(amp=1.0, f0=420):
    t = tt(1.6)
    s = np.zeros_like(t)
    for k, a in ((1.0, 1.0), (2.32, 0.7), (4.25, 0.5), (6.63, 0.3), (9.38, 0.2)):
        s += a * np.sin(2 * np.pi * f0 * k * t) * np.exp(-t * (2.5 + k * 1.5))
    s += filt(noise(len(t)), 800, 8000) * env(len(t), 0.0005, 0.03) * 1.2
    return s * amp * 0.5


def pick_strike(amp=1.0):
    t = tt(0.5)
    s = filt(noise(len(t)), 900, 6000) * env(len(t), 0.0005, 0.025) * 1.2
    s += clink(rng.uniform(1100, 1500), 0.5, 0.6)
    s += np.sin(2 * np.pi * 90 * t) * env(len(t), 0.002, 0.05) * 0.8
    return s * amp


def drum(amp=1.0, f0=72):
    t = tt(1.0)
    f = f0 * (0.65 + 0.35 * np.exp(-t * 18))
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * env(len(t), 0.002, 0.28)
    s += filt(noise(len(t)), 80, 900) * env(len(t), 0.001, 0.05) * 0.7
    return s * amp


def whoosh(dur=0.8, amp=1.0, rev=True):
    t = tt(dur)
    n = filt(noise(len(t)), 300, 5000)
    e = (t / dur) ** 3 if rev else (1 - t / dur) ** 3
    return n * e * amp


def drip(amp=1.0):
    t = tt(0.18)
    f = 900 + 1400 * (t / 0.18)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(len(t), 0.001, 0.03) * amp


def note(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def pad(dur, notes, amp=1.0, a=1.5, r=1.5, bright=8, detune=0.004, vib=0.0):
    t = tt(dur)
    out = np.zeros((len(t), 2), np.float32)
    for m in notes:
        f = note(m)
        for v, pan in ((-1, -0.6), (0, 0.0), (1, 0.6)):
            ff = f * (1 + v * detune) * (1 + vib * np.sin(2 * np.pi * 5.2 * t + rng.random() * 6))
            ph = 2 * np.pi * np.cumsum(np.broadcast_to(ff, t.shape)) / SR + rng.random() * 6
            s = np.zeros_like(t)
            for k in range(1, bright + 1):
                if f * k > 9000:
                    break
                s += np.sin(ph * k) / k ** 1.25
            l, rr = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
            out[:, 0] += s * l
            out[:, 1] += s * rr
    out *= (adsr(dur, a, r) * (1 + 0.12 * np.sin(2 * np.pi * 0.25 * t)))[:, None]
    return out * amp / (len(notes) * 3) * 0.9


def bell(m, amp=1.0, dur=2.5):
    t = tt(dur)
    f = note(m)
    s = np.sin(2 * np.pi * f * t) * np.exp(-t * 1.8) + 0.4 * np.sin(2 * np.pi * f * 2.0 * t) * np.exp(-t * 3)
    s += 0.2 * np.sin(2 * np.pi * f * 3.01 * t) * np.exp(-t * 5)
    return s * np.minimum(1, t / 0.004) * amp


def reverb(x, secs=2.8, decay=0.55, hi=6500):
    n = int(secs * SR)
    t = np.arange(n) / SR
    ir = np.stack([noise(n) * np.exp(-t / decay), noise(n) * np.exp(-t / decay)], 1).astype(np.float32)
    ir = filt(ir, 120, hi)
    ir /= np.sqrt((ir ** 2).sum(0))
    L = 1 << int(math.ceil(math.log2(len(x) + n)))
    out = np.zeros_like(x)
    for ch in range(2):
        y = np.fft.irfft(np.fft.rfft(x[:, ch], L) * np.fft.rfft(ir[:, ch], L), L)[:len(x)]
        out[:, ch] = y
    return out


# ---------------------------------------------------------------- score
def build(narr_wav, narr_segments, out_path):
    music, sfx, amb = Bus(), Bus(), Bus()

    # ---- scene 1: heartbeat, chains, flashes
    for k, tb in enumerate((0.20, 1.02, 1.80, 2.52, 3.18, 3.80, 4.40)):
        sfx.add(tb, heartbeat(0.9 + 0.05 * k))
    for tr, p in ((0.55, -0.4), (1.45, 0.5), (2.20, -0.2), (2.95, 0.3), (4.0, 0.0)):
        sfx.add(tr, rattle(7, 0.5, 0.35), pan=p)
    for tf in (1.10, 1.80, 2.45, 3.00):
        sfx.add(tf - 0.35, whoosh(0.35, 0.25))
        sfx.add(tf, boom(0.55, 120, 40, 1.4))
    sfx.add(2.45, filt(noise(int(1.2 * SR)), 100, 4000) * env(int(1.2 * SR), 0.01, 0.4) * 0.35)   # storm
    sfx.add(3.30, boom(1.0, 80, 28, 2.5))
    music.add(3.30, pad(1.75, [38, 45], 0.5, a=0.05, r=0.6, bright=4))

    # ---- scene 2: the mine
    sfx.add(4.6, whoosh(0.4, 0.4))
    sfx.add(5.0, boom(0.9, 90, 30, 2.5))
    n = int(15 * SR)
    amb.add(5.0, filt(noise(n, "brown"), 30, 300) * adsr(15, 0.3, 1.0) * 0.35)
    for td in rng.uniform(5.3, 19.5, 9):
        amb.add(td, drip(0.12), pan=rng.uniform(-0.7, 0.7))
    for ts, i in S.pick_strike_times(5.0, 8.0):
        sfx.add(ts, pick_strike((0.8, 0.45, 0.3)[i]), pan=(-0.3, 0.3, 0.6)[i])
    for ts, i in S.pick_strike_times(1.80, 2.06):
        sfx.add(ts, pick_strike(0.4))
    music.add(5.0, pad(6.0, [26, 38], 0.9, a=1.0, r=1.0, bright=6))                 # low D drone
    music.add(5.0, pad(6.0, [50, 53, 57], 0.35, a=2.5, r=1.0, bright=5))           # Dm
    music.add(10.8, pad(4.6, [26, 38, 50, 53, 57, 62], 0.75, a=1.2, r=0.4, bright=9, vib=0.002))
    sfx.add(12.0, boom(0.7, 70, 30, 2.5))
    sfx.add(12.0, bell(74, 0.25, 3.0), pan=0.2)
    music.add(15.2, pad(2.5, [34, 46, 50, 53, 58, 62], 0.75, a=0.3, r=0.4, bright=9, vib=0.002))   # Bb
    music.add(17.6, pad(2.6, [29, 41, 53, 57, 60, 65], 0.8, a=0.3, r=0.5, bright=10, vib=0.002))  # F
    for k, tb in enumerate(np.sort(rng.uniform(15.3, 19.8, 22))):
        sfx.add(tb, bell(int(rng.choice([74, 77, 79, 81, 84, 86])), 0.08 + 0.004 * k), pan=rng.uniform(-0.8, 0.8))
    sfx.add(17.6 + S.SHACKLE_OPEN, metal_hit(0.9, 380))
    sfx.add(17.6 + S.SHACKLE_OPEN, rattle(5, 0.3, 0.4, 1200))
    sfx.add(17.6 + S.SHACKLE_OPEN + 0.55, metal_hit(0.35, 300), pan=0.3)
    sfx.add(19.4, whoosh(0.6, 0.5))

    # ---- scene 3: the sea at Fortaleza
    sfx.add(20.0, boom(1.0, 85, 28, 3.0))
    n = int(18.2 * SR)
    sea = filt(noise(n, "pink"), 60, 1800) * 0.25
    wind = filt(noise(n), 300, 1400) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.13 * tt(18.2) + 1))
    amb.add(20.0, (sea + wind * 0.12) * adsr(18.2, 0.05, 0.4))
    amb.add(41.7, filt(noise(int(1.1 * SR), "pink"), 60, 1800) * 0.3 * adsr(1.1, 0.02, 0.1))
    amb.add(38.0, filt(noise(int(1.3 * SR), "pink"), 60, 900) * 0.12 * adsr(1.3, 0.05, 0.2))
    for tc in S.sea_crash_times(19.0, 38.0) + S.sea_crash_times(41.7, 42.8):
        L = 2.6
        e = np.concatenate([np.linspace(0, 1, int(0.45 * SR)) ** 2, np.exp(-np.arange(int((L - 0.45) * SR)) / SR / 0.7)])
        amb.add(tc - 0.45, filt(noise(len(e)), 150, 7000) * e * 0.5, pan=rng.uniform(-0.4, 0.4))
    music.add(20.0, pad(9.0, [26, 38, 45], 0.8, a=0.5, r=0.6, bright=6))
    music.add(25.0, pad(4.0, [50, 53, 57], 0.35, a=1.5, r=0.3, bright=6))
    music.add(29.0, pad(4.0, [31, 43, 50, 55, 58], 0.6, a=0.6, r=0.3, bright=8))       # Gm
    beat = 0.75
    for k in range(int((33.0 - 25.0) / beat)):
        tb = 25.0 + k * beat
        sfx.add(tb, drum(0.25 + 0.5 * (k / 11) + (0.2 if k % 4 == 0 else 0), 70), pan=0.0)
        if k % 2 == 1 and tb > 29:
            sfx.add(tb + beat / 2, drum(0.25, 110), pan=0.3)
    sfx.add(32.4, whoosh(0.6, 0.6))
    sfx.add(33.0, boom(1.2, 75, 26, 3.5))
    sfx.add(33.0, drum(1.0, 60))
    music.add(33.0, pad(2.8, [34, 46, 53, 58, 62, 65], 0.85, a=0.05, r=0.5, bright=10, vib=0.002))   # Bb
    music.add(35.8, pad(2.4, [33, 45, 52, 57, 61, 64], 0.85, a=0.3, r=0.2, bright=10, vib=0.002))    # A
    for k in range(6):
        sfx.add(34.4 + k * 0.6, drum(0.5 + k * 0.08, 66))
    sfx.add(37.5, whoosh(0.5, 0.6))

    # ---- scene 4: legacy, under the narration
    sfx.add(38.0, boom(0.9, 90, 30, 2.5))
    music.add(38.0, pad(6.0, [26, 38, 50, 53, 57, 62], 0.7, a=0.4, r=0.3, bright=9, vib=0.002))   # Dm
    music.add(44.0, pad(3.2, [34, 46, 50, 53, 58, 62], 0.75, a=0.3, r=0.3, bright=9, vib=0.002))   # Bb
    music.add(47.2, pad(2.8, [29, 41, 48, 53, 57, 60, 65], 0.85, a=0.3, r=0.0, bright=10, vib=0.002))  # F
    for k in range(int(12 / 0.5)):
        tb = 38.0 + k * 0.5
        sfx.add(tb, drum(0.22 + (0.18 if k % 4 == 0 else 0) + 0.25 * k / 24, 68))
    for tb in (39.30 + 1.3 * 0.45, 43.90 + 1.3 * 0.5):
        sfx.add(tb - 0.4, whoosh(0.4, 0.4))
        sfx.add(tb, metal_hit(1.0, 520))
        sfx.add(tb, rattle(8, 0.6, 0.6, 1500))
        sfx.add(tb, boom(0.5, 140, 50, 1.0))
    for tcut in (40.6, 41.7, 42.8, 45.2, 46.4, 47.6):
        sfx.add(tcut, boom(0.3, 110, 45, 0.8))
    sfx.add(49.2, whoosh(0.8, 0.5))

    # ---- scene 5: silence, then the question
    sfx.add(51.0, heartbeat(0.55))
    sfx.add(52.0, heartbeat(0.40))
    sfx.add(53.7, boom(1.0, 70, 26, 3.0))
    music.add(53.7, pad(3.3, [26, 38], 0.6, a=0.2, r=1.2, bright=4))
    sfx.add(57.1, boom(0.9, 70, 26, 3.0))
    music.add(57.1, pad(2.9, [26, 38, 50, 54, 57, 62, 66], 0.8, a=0.6, r=1.8, bright=8, vib=0.002))   # D major
    sfx.add(57.1, bell(74, 0.25, 3.0))
    sfx.add(57.4, bell(81, 0.12, 2.6), pan=0.4)

    # ---- mix
    wet_src = music.x * 0.8 + sfx.x * 0.35
    wet = reverb(wet_src)
    mix = music.x * 0.55 + sfx.x * 0.9 + amb.x * 0.75 + wet * 0.45

    # voice
    voice = np.zeros((N, 2), np.float32)
    duck = np.ones(N, np.float32)
    if narr_wav is not None:
        with wave.open(narr_wav) as wf:
            v = np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float32) / 32768
        v = filt(v, 80, 9000)
        voice[:len(v)] = np.stack([v, v], 1)[:N] * 1.0
        for seg in narr_segments:
            a, b = int((seg["start"] - 0.25) * SR), int((seg["end"] + 0.3) * SR)
            duck[a:b] = 0.42
        duck = np.convolve(duck, np.ones(int(0.12 * SR)) / int(0.12 * SR), mode="same").astype(np.float32)
        vwet = reverb(voice, 1.6, 0.35, 5000)
        voice = voice + vwet * 0.12
    mix = mix * duck[:, None] + voice * 1.25

    # hard silence at 50s (with a 40 ms tail so it doesn't click)
    # Until the final hit only the dry heartbeats remain: no music or reverb tails.
    i0, i1, i2 = int(50.0 * SR), int(50.04 * SR), int(53.65 * SR)
    mix[i0:i1] *= np.linspace(1, 0, i1 - i0)[:, None]
    mix[i1:i2] = sfx.x[i1:i2] * 0.9
    # fade out end
    fo = int(0.4 * SR)
    mix[-fo:] *= np.linspace(1, 0, fo)[:, None]

    mix = np.tanh(mix * 0.9) / 0.9
    mix /= max(1e-6, np.abs(mix).max()) / 0.89
    pcm = (np.clip(mix, -1, 1) * 32767).astype(np.int16)
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())
