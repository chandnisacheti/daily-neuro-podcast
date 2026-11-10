"""Generates the show's intro/outro music from scratch (no samples, so there are no licensing issues).
Warm electric-piano chords + a soft bell arpeggio, ~84 bpm. Run once: python make_music.py"""
import numpy as np, wave
SR = 44100
def note(f, dur, vel=1.0):
    t = np.arange(int(SR * dur)) / SR
    env = (1 - np.exp(-t * 60)) * np.exp(-t * 1.6)            # soft attack, long decay
    tone = sum(a * np.sin(2 * np.pi * f * h * t + 0.3 * np.sin(2 * np.pi * 5 * t) * (h > 1))
               for h, a in ((1, 1), (2, .35), (3, .12), (4, .05)))
    tone += .5 * np.sin(2 * np.pi * f * 1.003 * t)            # slight detune = chorus warmth
    return vel * env * tone
def bell(f, dur):
    t = np.arange(int(SR * dur)) / SR
    return np.exp(-t * 3.5) * (np.sin(2*np.pi*f*t) + .4*np.sin(2*np.pi*f*2.76*t)*np.exp(-t*6))
def hz(m): return 440 * 2 ** ((m - 69) / 12)
beat = 60 / 84
chords = [[53, 57, 60, 64, 67], [57, 60, 64, 67], [50, 53, 57, 60, 64], [46, 50, 53, 57, 62]]  # Fmaj9 Am7 Dm9 Bbmaj9
def render(bars, tail):
    out = np.zeros(int(SR * (bars * 4 * beat + tail)))
    for b in range(bars):
        ch = chords[b % 4]; start = int(SR * b * 4 * beat)
        for i, m in enumerate(ch):
            n = note(hz(m), 4 * beat + tail, .22 if m > 52 else .3)
            s = start + int(SR * .012 * i)                    # gentle strum
            out[s:s + len(n)] += n[:len(out) - s]
        for k, m in enumerate([ch[-1] + 12, ch[-2] + 12, ch[-3] + 12, ch[-2] + 12]):  # bell arpeggio
            n = bell(hz(m), 2.5) * .10
            s = start + int(SR * k * beat)
            out[s:s + len(n)] += n[:len(out) - s]
    # room: two short echoes
    for d, g in ((0.11, .25), (0.23, .12)):
        k = int(SR * d); out[k:] += g * out[:-k]
    out /= np.max(np.abs(out)) / 0.8
    return out
def save(x, name, fade_in=.05, fade_out=2.5):
    n = len(x); fi = int(SR * fade_in); fo = int(SR * fade_out)
    x[:fi] *= np.linspace(0, 1, fi); x[-fo:] *= np.linspace(1, 0, fo)
    st = np.stack([x, np.roll(x, int(SR * .008))], 1)       # tiny stereo width
    with wave.open(name, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((st * 32767).astype("<i2").tobytes())
save(render(2, 1.5), "intro.wav", fade_out=2.0)
save(render(4, 2.5), "outro.wav", fade_out=4.0)
