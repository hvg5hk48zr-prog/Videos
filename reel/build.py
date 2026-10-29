#!/usr/bin/env python3
"""Build the reel: narration -> soundtrack -> frames -> final MP4s + cover.

    python3 build.py            # everything
    python3 build.py audio      # narration + soundtrack only
    python3 build.py video      # frames only (needs build/narration.json for subtitles)
    python3 build.py mux        # combine existing picture + soundtracks

Narration uses Piper TTS (pip install piper-tts) with the en-us-ryan-high voice;
set PIPER_VOICE to the .onnx path.
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
BUILD = os.path.join(ROOT, "build")
OUT = os.path.join(ROOT, "out")
os.makedirs(BUILD, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, ROOT)

import timeline as T  # noqa: E402

VOICE = os.environ.get("PIPER_VOICE", os.path.join(BUILD, "voice", "en-us-ryan-high.onnx"))
SR = 48000


def run(cmd, **kw):
    print("+", " ".join(cmd) if isinstance(cmd, list) else cmd, flush=True)
    return subprocess.run(cmd, check=True, **kw)


def narration():
    segs, t = [], T.NARRATION_START
    track = np.zeros(int(60 * SR), np.float32)
    for i, (sub, line) in enumerate(T.NARRATION):
        raw = os.path.join(BUILD, f"narr{i}.wav")
        rs = os.path.join(BUILD, f"narr{i}_48k.wav")
        run(["piper", "-m", VOICE, "--length_scale", "1.12", "--sentence_silence", "0", "-f", raw],
            input=line.encode(), stderr=subprocess.DEVNULL)
        # trim edge silence, gently lower pitch for gravitas, resample to 48 kHz
        run(["ffmpeg", "-v", "error", "-y", "-i", raw, "-af",
             "silenceremove=start_periods=1:start_threshold=-45dB,areverse,"
             "silenceremove=start_periods=1:start_threshold=-45dB,areverse,"
             "asetrate=22050*0.94,aresample=48000,atempo=1/0.94,"
             "equalizer=f=140:t=q:w=1:g=3,equalizer=f=3500:t=q:w=1:g=2",
             "-ar", "48000", "-ac", "1", rs])
        with wave.open(rs) as wf:
            v = np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float32) / 32768
        dur = len(v) / SR
        i0 = int(t * SR)
        track[i0:i0 + len(v)] += v
        segs.append(dict(start=round(t, 3), end=round(t + dur, 3), sub=sub, text=line))
        t += dur + T.NARRATION_GAP
    assert segs[-1]["end"] < 50.0, f"narration overruns scene 4: {segs[-1]['end']:.2f}s"
    track /= max(1e-6, np.abs(track).max()) / 0.9
    with wave.open(os.path.join(BUILD, "narration.wav"), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((track * 32767).astype(np.int16).tobytes())
    json.dump(segs, open(os.path.join(BUILD, "narration.json"), "w"), indent=1)
    print("narration:", [(s["start"], s["end"]) for s in segs])
    return segs


def soundtrack():
    import audio
    segs = narration()
    audio.build(os.path.join(BUILD, "narration.wav"), segs, os.path.join(BUILD, "mix_voice.wav"))
    audio.build(None, [], os.path.join(BUILD, "mix_music_only.wav"))


def _frame(i):
    import render
    return render.render_frame(i).tobytes()


def video():
    import render
    path = os.path.join(BUILD, "picture.mp4")
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{render.W}x{render.H}", "-r", str(render.FPS), "-i", "-",
                           "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
                           "-profile:v", "high", "-movflags", "+faststart", path], stdin=subprocess.PIPE)
    t0 = time.time()
    with Pool(os.cpu_count()) as pool:
        for k, buf in enumerate(pool.imap(_frame, range(render.NF), chunksize=4)):
            ff.stdin.write(buf)
            if k % 150 == 0:
                print(f"frame {k}/{render.NF}  {time.time() - t0:.0f}s", flush=True)
    ff.stdin.close()
    ff.wait()


def loudnorm(src, dst):
    """Two-pass EBU R128 normalisation to -14 LUFS (Instagram-friendly)."""
    p = subprocess.run(["ffmpeg", "-hide_banner", "-i", src, "-af",
                        "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True)
    js = json.loads(p.stderr[p.stderr.rindex("{"):p.stderr.rindex("}") + 1])
    af = (f"loudnorm=I=-14:TP=-1.5:LRA=11:measured_I={js['input_i']}:measured_TP={js['input_tp']}:"
          f"measured_LRA={js['input_lra']}:measured_thresh={js['input_thresh']}:offset={js['target_offset']}:linear=true")
    run(["ffmpeg", "-v", "error", "-y", "-i", src, "-af", af, "-ar", "48000", dst])


def mux():
    pic = os.path.join(BUILD, "picture.mp4")
    for mix, name in (("mix_voice.wav", "brazils_real_heroes_reel.mp4"),
                      ("mix_music_only.wav", "brazils_real_heroes_reel_no_voiceover.mp4")):
        norm = os.path.join(BUILD, "norm_" + mix)
        loudnorm(os.path.join(BUILD, mix), norm)
        run(["ffmpeg", "-v", "error", "-y", "-i", pic, "-i", norm, "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
             "-shortest", "-movflags", "+faststart", os.path.join(OUT, name)])
    cover()


def cover(lang="en"):
    """Cover for the Reels grid: the Chico Rei photograph with the reel's title,
    kept inside the centre 3:4 area the profile grid shows."""
    import render
    from PIL import Image
    f = render.render_frame(int(12.0 * render.FPS), with_text=False).astype(np.float32) / 255
    yy = np.linspace(0, 1, f.shape[0], dtype=np.float32)[:, None, None]
    f *= 1 - 0.55 * np.clip((yy - 0.48) / 0.25, 0, 1)            # darken under the title
    if lang == "pt":
        name, lines = "capa_pt.jpg", (("OS VERDADEIROS\nHERÓIS DO BRASIL", T.SERIF, 90, T.GOLDC, 0.62, 0.05, 0.6),
                                      ("LUTAVAM PELA\nLIBERDADE.", T.SANS_B, 62, T.WHITE, 0.725, 0.12, 0.25))
    else:
        name, lines = "cover.jpg", (("BRAZIL’S\nREAL HEROES", T.SERIF, 124, T.GOLDC, 0.62, 0.05, 0.6),
                                    ("WERE FIGHTING\nFOR FREEDOM.", T.SANS_B, 62, T.WHITE, 0.735, 0.12, 0.25))
    for s, font, size, color, y, tr, glow in lines + (("CHICO REI  ·  DRAGÃO DO MAR", T.SANS_M, 30, (0.9, 0.85, 0.75), 0.80, 0.22, 0.0),):
        render.draw_text(f, render.text_layer(s, font, size, color, tr, glow), y, 1.0)
    Image.fromarray((np.clip(f, 0, 1) * 255).astype(np.uint8)).save(os.path.join(OUT, name), quality=92)


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("audio", "all"):
        soundtrack()
    if step in ("video", "all"):
        video()
    if step in ("mux", "all"):
        mux()
    if step == "cover":
        cover()
        cover("pt")
