"""Build one daily episode: gather news -> Gemini script -> Gemini two-voice audio -> MP3 -> RSS feed.

Env:
  GEMINI_API_KEY      required (free key from aistudio.google.com)
  GEMINI_TEXT_MODEL   optional, default gemini-2.5-flash
  GEMINI_TTS_MODEL    optional, default gemini-2.5-flash-preview-tts
  SITE_URL            public base URL of the docs/ folder (GitHub Pages)
  DRY_RUN=1           skip Gemini; use a tiny fake script and a silent clip (for testing)
"""
import base64
import calendar
import datetime as dt
import html
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import feedparser
import requests

from sources import FEEDS
from feed import write_feed

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
EPISODES = DOCS / "episodes"
KEEP_EPISODES = 21
HOSTS = ("Maya", "Leo")
VOICES = {"Maya": "Kore", "Leo": "Puck"}
EDGE_VOICES = {"Maya": "en-US-AvaNeural", "Leo": "en-US-AndrewNeural"}
API = "https://generativelanguage.googleapis.com/v1beta/models"
KEY = os.environ.get("GEMINI_API_KEY", "")
TEXT_MODEL = os.environ.get("GEMINI_TEXT_MODEL", "gemini-2.5-flash")
TTS_MODEL = os.environ.get("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts")
DRY = os.environ.get("DRY_RUN") == "1"


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# ---------- 1. news ----------
def gather_news(hours=30):
    cutoff = time.time() - hours * 3600
    seen, items = set(), []
    for label, url, cap in FEEDS:
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": "daily-neuro-podcast/1.0"})
            r.raise_for_status()
            parsed = feedparser.parse(r.content)
        except Exception as e:  # noqa: BLE001
            log(f"skip {label}: {e}")
            continue
        n = 0
        for e in parsed.entries:
            t = e.get("published_parsed") or e.get("updated_parsed")
            if t and calendar.timegm(t) < cutoff:
                continue
            title = html.unescape(re.sub(r"<[^>]+>", "", e.get("title", ""))).strip()
            key = re.sub(r"\W+", "", title.lower())[:80]
            if not title or key in seen:
                continue
            seen.add(key)
            summary = html.unescape(re.sub(r"<[^>]+>", " ", e.get("summary", "")))
            summary = re.sub(r"\s+", " ", summary).strip()[:500]
            items.append({"source": label, "title": title, "summary": summary, "link": e.get("link", "")})
            n += 1
            if n >= cap:
                break
        log(f"{label}: {n} items")
    return items


# ---------- 2. script ----------
def gemini(model, body, tries=4):
    for i in range(tries):
        r = requests.post(f"{API}/{model}:generateContent", json=body,
                          headers={"x-goog-api-key": KEY}, timeout=300)
        if r.status_code == 200:
            return r.json()
        log(f"{model} HTTP {r.status_code}: {r.text[:300]}")
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(30 * (i + 1))
            continue
        break
    raise RuntimeError(f"Gemini call to {model} failed")


PROMPT = """You write "Neuro Morning", a daily ~10 minute news podcast for Chandni, an MS Biomedical
Engineering student at Columbia (NYC) who works on biosignal ML (EEG, EMG, HRV, EDA) and closed-loop
neuromodulation, and wants a career in neurotech, wearables and healthtech.

Two hosts: Maya and Leo. Warm, smart, conversational, like a good tech podcast: they react,
ask each other quick questions, explain jargon in a sentence, and never read headlines robotically.

From the news items below (last 24 hours), pick the 5 to 8 most important. Priority order:
1. Neurotech and brain-computer interfaces (implants, non-invasive BCI, neuromodulation, EEG)
2. Wearables and biosensing
3. Funding rounds, acquisitions and FDA clearances in neuro/health/medtech (say the amount and stage)
4. Notable research results
5. Anything involving New York companies or labs (call this out)
For each story say who, what, why it matters, and, when natural, one line on what it signals about
the job market or which companies are growing. Do not invent facts, numbers or names that are not
in the items. If there is little news, go deeper on fewer stories rather than padding.

Structure: a 2-line cold open, quick "today's rundown", the stories, then a 30-second wrap with
"one thing to watch". Today is {date}. About 1,400 words total.

Output ONLY dialogue lines, each starting with "Maya:" or "Leo:". No stage directions, no markdown.
Also, on the very first line before the dialogue, write "TITLE: " followed by a short episode title.

NEWS ITEMS (JSON):
{items}
"""


def write_script(items, date):
    if DRY:
        return "Test episode", [("Maya", "Good morning! This is a test."), ("Leo", "Everything works.")]
    body = {"contents": [{"parts": [{"text": PROMPT.format(date=date, items=json.dumps(items[:90], ensure_ascii=False))}]}],
            "generationConfig": {"temperature": 0.8}}
    out = gemini(TEXT_MODEL, body)
    text = "".join(p.get("text", "") for p in out["candidates"][0]["content"]["parts"])
    title, lines = f"Neuro Morning, {date}", []
    for raw in text.splitlines():
        raw = raw.strip().replace("**", "")
        if raw.upper().startswith("TITLE:"):
            title = raw.split(":", 1)[1].strip() or title
            continue
        m = re.match(r"^(Maya|Leo)\s*:\s*(.+)$", raw)
        if m:
            lines.append((m.group(1), m.group(2)))
    if len(lines) < 6:
        raise RuntimeError("Script came back too short:\n" + text[:1000])
    return title, lines


# ---------- 3. audio ----------
def chunks(lines, size=14):
    for i in range(0, len(lines), size):
        yield lines[i:i + size]


def tts_gemini(chunk, wav_path):
    convo = "\n".join(f"{s}: {t}" for s, t in chunk)
    body = {
        "contents": [{"parts": [{"text": "Read this as a lively, natural morning podcast conversation:\n" + convo}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"multiSpeakerVoiceConfig": {"speakerVoiceConfigs": [
                {"speaker": s, "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": VOICES[s]}}} for s in HOSTS
            ]}},
        },
    }
    out = gemini(TTS_MODEL, body)
    pcm = base64.b64decode(out["candidates"][0]["content"]["parts"][0]["inlineData"]["data"])
    raw = wav_path.with_suffix(".pcm")
    raw.write_bytes(pcm)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "s16le", "-ar", "24000", "-ac", "1",
                    "-i", str(raw), str(wav_path)], check=True)


def tts_edge(chunk, wav_path, tmp):
    """Free fallback with no key (Microsoft Edge voices)."""
    parts = []
    for j, (s, t) in enumerate(chunk):
        p = tmp / f"{wav_path.stem}_{j}.mp3"
        subprocess.run([sys.executable, "-m", "edge_tts", "--voice", EDGE_VOICES[s], "--text", t,
                        "--write-media", str(p)], check=True)
        parts.append(p)
    concat(parts, wav_path)


def concat(paths, out):
    lst = out.with_suffix(".txt")
    lst.write_text("".join(f"file '{p}'\n" for p in paths))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-ar", "24000", "-ac", "1", str(out)], check=True)


def make_audio(lines, mp3_path):
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        wavs = []
        use_edge = False
        for i, ch in enumerate(chunks(lines)):
            w = tmp / f"part{i:02d}.wav"
            if DRY:
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                                "sine=frequency=440:sample_rate=24000", "-t", "2", str(w)], check=True)
            else:
                if not use_edge:
                    try:
                        tts_gemini(ch, w)
                        time.sleep(20)  # stay under free-tier rate limits
                    except Exception as e:  # noqa: BLE001
                        log(f"Gemini TTS failed ({e}); switching to free Edge voices")
                        use_edge = True
                if use_edge:
                    tts_edge(ch, w, tmp)
            wavs.append(w)
        joined = tmp / "all.wav"
        concat(wavs, joined)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(joined), "-af", "loudnorm=I=-16",
                        "-codec:a", "libmp3lame", "-b:a", "64k", str(mp3_path)], check=True)
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                          str(mp3_path)], capture_output=True, text=True, check=True).stdout
    return int(float(dur))


# ---------- main ----------
def main():
    if not KEY and not DRY:
        sys.exit("GEMINI_API_KEY is not set")
    now = dt.datetime.now(dt.timezone.utc)
    date_str = now.astimezone(dt.timezone(dt.timedelta(hours=-5))).strftime("%A, %B %-d, %Y")
    slug = now.strftime("%Y-%m-%d")
    EPISODES.mkdir(parents=True, exist_ok=True)

    items = gather_news()
    log(f"{len(items)} news items")
    if not items and not DRY:
        sys.exit("No news found; skipping today")
    title, lines = write_script(items, date_str)
    log(f"script: {len(lines)} lines, title: {title}")

    mp3 = EPISODES / f"{slug}.mp3"
    seconds = make_audio(lines, mp3)

    meta_path = DOCS / "episodes.json"
    eps = json.loads(meta_path.read_text()) if meta_path.exists() else []
    eps = [e for e in eps if e["slug"] != slug]
    links = "\n".join(f"- {it['title']} ({it['source']}): {it['link']}" for it in items[:40])
    eps.insert(0, {
        "slug": slug, "title": title, "date": now.strftime("%a, %d %b %Y %H:%M:%S +0000"),
        "seconds": seconds, "bytes": mp3.stat().st_size,
        "notes": "Sources from the last 24 hours:\n" + links,
    })
    for old in eps[KEEP_EPISODES:]:
        (EPISODES / f"{old['slug']}.mp3").unlink(missing_ok=True)
    eps = eps[:KEEP_EPISODES]
    meta_path.write_text(json.dumps(eps, indent=1))
    (DOCS / "scripts").mkdir(exist_ok=True)
    (DOCS / "scripts" / f"{slug}.txt").write_text(title + "\n\n" + "\n".join(f"{s}: {t}" for s, t in lines))
    write_feed(eps, DOCS)
    log(f"done: {mp3.name}, {seconds}s")


if __name__ == "__main__":
    main()
