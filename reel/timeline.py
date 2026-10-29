"""Edit decision list for the reel: shots, camera moves, grades and on-screen text."""
from functools import partial

import scenes as S

# (start, end, renderer, camera(z0, z1, cx0, cy0, cx1, cy1), grade, flash-in)
#   grade: (gain, (r, g, b) tint, contrast)
WARM = (1.0, (1.06, 0.97, 0.86), 1.15)
GOLD = (1.05, (1.08, 0.96, 0.80), 1.12)
STEEL = (1.0, (0.92, 1.0, 1.06), 1.2)
DAWN = (1.0, (1.05, 0.98, 0.92), 1.05)
NEUTRAL = (1.0, (1.0, 1.0, 1.0), 1.1)
PHOTO = (1.0, (1.0, 1.0, 1.0), 1.04)

C = lambda z0=1.0, z1=1.0, cx0=0.5, cy0=0.5, cx1=None, cy1=None: (z0, z1, cx0, cy0, cx0 if cx1 is None else cx1, cy0 if cy1 is None else cy1)

SHOTS = [
    # ---- Scene 1: the question
    (0.00, 1.10, S.black, C(), NEUTRAL, 0),
    (1.10, 1.36, S.town, C(1.10, 1.16), WARM, 1),
    (1.36, 1.80, S.black, C(), NEUTRAL, 0),
    (1.80, 2.06, S.mine_workers, C(1.15, 1.2), WARM, 1),
    (2.06, 2.45, S.black, C(), NEUTRAL, 0),
    (2.45, 2.71, S.ship_storm, C(1.1, 1.18), STEEL, 1),
    (2.71, 3.00, S.black, C(), NEUTRAL, 0),
    (3.00, 3.26, S.shackled, C(1.05, 1.12), WARM, 1),
    (3.26, 5.00, S.black, C(), NEUTRAL, 0),
    # ---- Scene 2: Chico Rei
    (5.00, 8.00, S.mine_workers, C(1.0, 1.10, 0.5, 0.55, 0.48, 0.58), WARM, 1),
    (8.00, 10.80, S.mine_carry, C(1.05, 1.12), WARM, 0),
    (10.80, 15.20, S.CHICO_PHOTO, C(1.0, 1.10, 0.5, 0.5, 0.52, 0.44), PHOTO, 0),
    (15.20, 17.60, S.gold_hands, C(1.0, 1.12, 0.5, 0.58, 0.5, 0.5), GOLD, 0),
    (17.60, 20.00, S.gold_to_freedom, C(1.0, 1.08, 0.5, 0.5, 0.5, 0.45), GOLD, 0),
    # ---- Scene 3: Dragão do Mar
    (20.00, 23.00, S.sea_storm, C(1.0, 1.08, 0.5, 0.55, 0.5, 0.6), STEEL, 1),
    (23.00, 25.00, S.jangada_close, C(1.06, 1.14), STEEL, 0),
    (25.00, 29.00, S.shore_hero, C(1.0, 1.08, 0.5, 0.6, 0.5, 0.55), STEEL, 0),
    (29.00, 33.00, S.line_of_men, C(1.12, 1.0, 0.5, 0.6, 0.5, 0.55), STEEL, 0),
    (33.00, 38.00, S.DRAGAO_PHOTO, C(1.0, 1.12, 0.5, 0.5, 0.47, 0.44), PHOTO, 1),
    # ---- Scene 4: the legacy
    (38.00, 39.30, S.ocean_dawn, C(1.0, 1.06), DAWN, 1),
    (39.30, 40.60, S.chain_break, C(1.0, 1.06), WARM, 0),
    (40.60, 41.70, partial(S.face, variant=0), C(1.0, 1.06), WARM, 0),
    (41.70, 42.80, S.sea_storm, C(1.1, 1.16), STEEL, 0),
    (42.80, 43.90, partial(S.face, variant=1), C(1.0, 1.06), STEEL, 0),
    (43.90, 45.20, partial(S.chain_break, slow=True), C(1.08, 1.14), WARM, 0),
    (45.20, 46.40, S.crowd, C(1.0, 1.05), DAWN, 0),
    (46.40, 47.60, partial(S.face, variant=2), C(1.0, 1.06), WARM, 0),
    (47.60, 50.00, S.modern, C(1.0, 1.08), DAWN, 0),
    # ---- Scene 5: the final question
    (50.00, 53.20, S.DRAGAO_CLOSE, C(1.45, 1.75, 0.45, 0.42, 0.43, 0.41), PHOTO, 0),
    (53.20, 60.00, S.black, C(), NEUTRAL, 0),
]

# global brightness envelope: fades to black (t0, t1, from, to)
FADES = [(52.6, 53.2, 1.0, 0.0), (0.0, 0.0, 1, 1)]

SERIF = "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"
SANS_B = "/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf"
SANS_SB = "/usr/share/fonts/opentype/inter/InterDisplay-SemiBold.otf"
SANS_M = "/usr/share/fonts/opentype/inter/InterDisplay-Medium.otf"

WHITE = (0.97, 0.95, 0.92)
GOLDC = (1.0, 0.80, 0.42)
SEAC = (0.86, 0.93, 1.0)

# text: (t0, t1, string, font, size, color, y-centre as fraction of H, tracking, fade in, fade out, glow)
TEXT = [
    (3.30, 5.00, "WHO ARE\nBRAZIL’S\nREAL HEROES?", SERIF, 112, WHITE, 0.47, 0.06, 0.25, 0.15, 0.5),
    (5.30, 9.60, "VILA RICA · MINAS GERAIS · 1700s", SANS_M, 30, (0.95, 0.85, 0.7), 0.22, 0.22, 0.5, 0.5, 0.0),
    (12.00, 17.60, "CHICO REI", SERIF, 150, GOLDC, 0.63, 0.08, 0.6, 0.3, 0.8),
    (13.20, 17.60, "FREEDOM, DIGNITY,\nRESISTANCE.", SANS_B, 52, WHITE, 0.72, 0.14, 0.5, 0.3, 0.3),
    (15.60, 19.80, "Oral tradition says he bought his\nfreedom with gold from the mines —\nthen helped free others.",
     SANS_M, 34, (0.95, 0.9, 0.82), 0.25, 0.0, 0.5, 0.4, 0.0),
    (20.40, 24.60, "FORTALEZA · CEARÁ · 1881", SANS_M, 30, SEAC, 0.22, 0.22, 0.5, 0.5, 0.0),
    (25.60, 28.80, "The jangadeiros of Fortaleza\nrefused to carry enslaved people\nto the slave ships.",
     SANS_M, 36, WHITE, 0.30, 0.0, 0.45, 0.35, 0.0),
    (29.30, 32.80, "No jangada sailed.\nThe port’s slave trade stopped.", SANS_SB, 40, WHITE, 0.30, 0.0, 0.45, 0.35, 0.0),
    (33.40, 38.00, "DRAGÃO\nDO MAR", SERIF, 150, SEAC, 0.585, 0.08, 0.5, 0.25, 0.7),
    (34.40, 38.00, "THE MAN WHO SAID NO.", SANS_B, 52, WHITE, 0.71, 0.14, 0.5, 0.25, 0.3),
    (35.40, 37.90, "Francisco José do Nascimento", SANS_M, 32, (0.85, 0.9, 0.95), 0.755, 0.04, 0.5, 0.3, 0.0),
    (53.70, 56.70, "WHO DO WE\nCALL A HERO?", SERIF, 118, WHITE, 0.46, 0.06, 0.6, 0.4, 0.5),
    (57.10, 60.00, "REMEMBER\nTHOSE WHO\nFOUGHT FOR\nFREEDOM.", SERIF, 104, GOLDC, 0.46, 0.06, 0.6, 0.0, 0.6),
    (58.20, 60.00, "CHICO REI  ·  DRAGÃO DO MAR", SANS_M, 30, (0.85, 0.8, 0.7), 0.71, 0.24, 0.6, 0.0, 0.0),
    (5.00, 50.00, "ARTISTIC RECONSTRUCTION — NOT ARCHIVAL FOOTAGE", SANS_M, 20, (0.8, 0.8, 0.8), 0.115, 0.2, 0.6, 0.4, 0.0),
]

# Narration subtitles: filled from the synthesized voice durations by build.py
NARRATION = [
    ("One represents the fight\nto reclaim freedom.", "One represents the fight to reclaim freedom."),
    ("The other helped turn refusal\ninto collective resistance.", "The other helped turn refusal into collective resistance."),
    ("They remind us that history\nwas also made by those\nwho fought against oppression.",
     "They remind us that history was also made by those who fought against oppression."),
]
NARRATION_START = 38.45
NARRATION_GAP = 0.40

# flashes of white at cut points (time, strength)
FLASHES = [(5.0, 0.9), (20.0, 1.0), (33.0, 0.7), (38.0, 0.9), (39.3, 0.4), (41.7, 0.35), (43.9, 0.4), (45.2, 0.3), (47.6, 0.35)]
