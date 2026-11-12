"""Generates the show's intro/outro music from scratch (no samples, so there are no licensing issues).
Upbeat morning groove, ~112 bpm: punchy drums, bouncy bass, bright plucked chords and a bell hook.
Run once: python make_music.py  (then convert the wavs to intro.mp3 / outro.mp3 with ffmpeg)"""
import numpy as np, wave

SR = 44100
BPM = 112
BEAT = 60 / BPM
rng = np.random.default_rng(7)


def hz(m):
    return 440 * 2 ** ((m - 69) / 12)


def t_(dur):
    return np.arange(int(SR * dur)) / SR


def pluck(f, dur, vel):
    t = t_(dur)
    env = (1 - np.exp(-t * 400)) * np.exp(-t * 5.5)
    tone = sum(a * np.sin(2 * np.pi * f * h * t) for h, a in ((1, 1), (2, .5), (3, .25), (4, .12), (5, .06)))
    tone += .4 * np.sin(2 * np.pi * f * 1.004 * t)
    return vel * env * tone


def bass(f, dur, vel=.55):
    t = t_(dur)
    env = (1 - np.exp(-t * 300)) * np.exp(-t * 3.0)
    return vel * env * (np.sin(2 * np.pi * f * t) + .3 * np.sin(2 * np.pi * 2 * f * t))


def bell(f, dur, vel=.16):
    t = t_(dur)
    return vel * np.exp(-t * 4) * (np.sin(2 * np.pi * f * t) + .35 * np.sin(2 * np.pi * f * 3.01 * t) * np.exp(-t * 8))


def kick():
    t = t_(.35)
    f = 110 * np.exp(-t * 18) + 45
    return .9 * np.exp(-t * 9) * np.sin(2 * np.pi * np.cumsum(f) / SR)


def clap():
    t = t_(.25)
    n = rng.standard_normal(len(t))
    n = n - np.concatenate([[0], n[:-1]]) * .6
    return .35 * np.exp(-t * 22) * n


def hat(open_=False):
    t = t_(.18 if open_ else .05)
    n = rng.standard_normal(len(t))
    n = n - np.concatenate([[0], n[:-1]])
    return (.09 if open_ else .07) * np.exp(-t * (18 if open_ else 80)) * n


# Fmaj7 - G6 - Am7 - Cadd9: bright and rising
CHORDS = [[53, 57, 60, 64], [55, 59, 62, 64], [57, 60, 64, 67], [48, 55, 60, 62]]
ROOTS = [41, 43, 45, 36]
HOOK = [[72, 74, 76, 72], [74, 76, 79, 74], [76, 79, 81, 76], [79, 76, 74, 72]]


def add(out, x, at):
    s = int(SR * at)
    if s >= len(out):
        return
    out[s:s + len(x)] += x[:len(out) - s]


def render(bars, tail, drums_from=0):
    out = np.zeros(int(SR * (bars * 4 * BEAT + tail)))
    for b in range(bars):
        bar = b * 4 * BEAT
        ch, root = CHORDS[b % 4], ROOTS[b % 4]
        for pos in (0, 1.5, 2.5, 3.5):  # syncopated plucked chords
            for i, m in enumerate(ch):
                add(out, pluck(hz(m), 1.2, .16), bar + pos * BEAT + .006 * i)
        for pos, m in ((0, root), (1.5, root), (2, root + 12), (3, root), (3.5, root + 7)):
            add(out, bass(hz(m), .45), bar + pos * BEAT)
        for k, m in enumerate(HOOK[b % 4]):
            add(out, bell(hz(m), 1.0), bar + k * .5 * BEAT)
        if b >= drums_from:
            for q in range(4):
                if q in (0, 2):
                    add(out, kick(), bar + q * BEAT)
                if q in (1, 3):
                    add(out, clap(), bar + q * BEAT)
                add(out, hat(), bar + q * BEAT)
                add(out, hat(open_=(q == 3)), bar + (q + .5) * BEAT)
            if b % 2:
                add(out, kick(), bar + 2.75 * BEAT)
    end = bars * 4 * BEAT  # final hit so the music lands instead of just stopping
    for i, m in enumerate(CHORDS[0] + [65]):
        add(out, pluck(hz(m), tail + .5, .2), end + .006 * i)
    add(out, bass(hz(ROOTS[0]), tail, .6), end)
    add(out, kick(), end)
    for d, g in ((BEAT * .75, .18), (BEAT * 1.5, .08)):  # tempo-synced echo
        k = int(SR * d)
        out[k:] += g * out[:-k]
    out = np.tanh(out * 1.4)
    return out / np.max(np.abs(out)) * .85


def save(x, name, fade_out=1.5):
    fo = int(SR * fade_out)
    x[-fo:] *= np.linspace(1, 0, fo)
    x[:200] *= np.linspace(0, 1, 200)
    st = np.stack([x, np.roll(x, int(SR * .007))], 1)
    with wave.open(name, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((st * 32767).astype("<i2").tobytes())


save(render(4, 1.2, drums_from=1), "intro.wav", fade_out=1.0)  # one bar build, then the full groove
save(render(6, 2.2, drums_from=0), "outro.wav", fade_out=2.0)
