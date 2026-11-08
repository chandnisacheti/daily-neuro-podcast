"""Build one daily episode: gather news -> Gemini script -> Gemini two-voice audio -> MP3 -> RSS feed.

Env:
  GEMINI_API_KEY      required (free key from aistudio.google.com)
  GEMINI_TEXT_MODEL   optional, default: newest Flash model this key can use
  GEMINI_TTS_MODEL    optional, default: newest TTS model this key can use
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
TEXT_MODEL = os.environ.get("GEMINI_TEXT_MODEL", "")  # empty = pick newest Flash model automatically
TTS_MODEL = os.environ.get("GEMINI_TTS_MODEL", "")    # empty = pick newest TTS model automatically
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36"


def _ver(name):
    nums = re.findall(r"\d+(?:\.\d+)?", name)
    return float(nums[0]) if nums else 0.0


def pick_models():
    """Google retires model names often, so ask the API which models this key can use."""
    global TEXT_MODEL, TTS_MODEL
    if TEXT_MODEL and TTS_MODEL:
        return
    names, token = [], None
    while True:
        r = requests.get(f"{API}", params={"pageSize": 200, **({"pageToken": token} if token else {})},
                         headers={"x-goog-api-key": KEY}, timeout=60)
        r.raise_for_status()
        j = r.json()
        names += [m["name"].split("/")[-1] for m in j.get("models", [])
                  if "generateContent" in m.get("supportedGenerationMethods", [])]
        token = j.get("nextPageToken")
        if not token:
            break
    log("models available:", ", ".join(names))
    bad = ("tts", "image", "live", "audio", "embed", "vision", "thinking", "exp", "robotics", "computer")
    if not TEXT_MODEL:
        text = [n for n in names if n.startswith("gemini") and "flash" in n and not any(b in n for b in bad)]
        text.sort(key=lambda n: (_ver(n), "lite" not in n, "preview" not in n, "latest" in n), reverse=True)
        TEXT_MODEL = text[0] if text else "gemini-flash-latest"
    if not TTS_MODEL:
        tts = [n for n in names if "tts" in n]
        tts.sort(key=lambda n: (_ver(n), "flash" in n, "lite" not in n), reverse=True)
        TTS_MODEL = tts[0] if tts else ""
    log(f"using text model {TEXT_MODEL}, tts model {TTS_MODEL or '(none, Edge voices)'}")
DRY = os.environ.get("DRY_RUN") == "1"


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# ---------- 1. news ----------
def gather_news(hours=30):
    cutoff = time.time() - hours * 3600
    seen, items = set(), []
    for label, url, cap in FEEDS:
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": UA, "Accept": "application/rss+xml, application/xml, text/xml, */*"})
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
        try:
            r = requests.post(f"{API}/{model}:generateContent", json=body,
                              headers={"x-goog-api-key": KEY}, timeout=480)
        except requests.exceptions.RequestException as e:  # timeouts, dropped connections
            log(f"{model} request error: {e}; retrying")
            time.sleep(15 * (i + 1))
            continue
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

She listens while walking or commuting, so the show must be EASY TO FOLLOW BY EAR:
- Clear fixed structure with spoken signposts. Announce every section by number and name,
  e.g. "Section one: Who Expanded." When moving on, say so: "That's it for research. Section three..."
- One story at a time. Say the company or lab name, then what happened, then why it matters.
- Short sentences. Spell out numbers the way people say them. No lists of more than three things.
- End each story with a one-sentence takeaway that starts with "The takeaway:".
- Light, warm banter only BETWEEN stories, never in the middle of the facts.

Hosts: Maya (anchor, keeps the structure) and Leo (adds context and the "why it matters").

STRUCTURE (exactly these five sections, in this order; skip a section only if there is truly
nothing for it):
Intro: two short lines only. Say hello and the date. Do NOT preview or list stories up front.
Section one: Who Expanded. Funding rounds, acquisitions, partnerships, FDA clearances and approvals,
  new sites or teams. For each, say the company, the move (amount and stage for funding), and why it
  matters.
Section two: New Research and Products. Notable papers, results and product launches, explained simply.
Section three: NYC Watch. Anything involving New York companies, labs or investors.
Section four: Career Signal. Which companies or areas look like they are growing or hiring, based
  only on today's stories, and what kind of role someone like Chandni could pitch them.
Section five: Recap. "Three things to remember today", then sign off.

Topic priority: neurotech and brain-computer interfaces first, then wearables and biosensing, then
broader healthtech and medtech. Ignore items that are not about health, medicine or neuroscience
(for example stock tickers that happen to contain "BCI"). Do not invent facts, numbers or names
that are not in the items. Today is {date}. About 1,400 words total.

Output ONLY dialogue lines, each starting with "Maya:" or "Leo:". No stage directions, no markdown.
On the very first line before the dialogue, write "TITLE: " followed by a short episode title.

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


LEADS_PROMPT = """From the news items below, list companies (or labs) that show a growth signal useful for a
student seeking a Summer 2027 internship in neurotech, wearables, healthtech or medtech: funding round,
acquisition, FDA clearance or approval, new site, expansion, big partnership, or explicit hiring.
Only health, medicine, neuroscience or medical-device organizations. Do not invent facts.
Return JSON: a list of objects with keys company, signal (one of funding, acquisition, fda, expansion,
partnership, hiring, research), detail (one sentence with amounts if given), location (city/country
if known, else ""), nyc (true/false), why_relevant (one sentence for a biosignal-ML / neuromodulation
MS BME student), link (copied from the item).
Order by relevance to neurotech and wearables first.

NEWS ITEMS (JSON):
{items}
"""


def extract_leads(items):
    if DRY:
        return [{"company": "TestCo", "signal": "funding", "detail": "Raised a test round.", "location": "New York",
                 "nyc": True, "why_relevant": "Test.", "link": "https://example.com"}]
    body = {"contents": [{"parts": [{"text": LEADS_PROMPT.format(items=json.dumps(items[:90], ensure_ascii=False))}]}],
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}}
    try:
        out = gemini(TEXT_MODEL, body)
        text = "".join(p.get("text", "") for p in out["candidates"][0]["content"]["parts"])
        leads = json.loads(text)
        return leads if isinstance(leads, list) else []
    except Exception as e:  # noqa: BLE001
        log(f"lead extraction failed: {e}")
        return []


def write_leads(leads, slug, date_str):
    d = DOCS / "leads"
    d.mkdir(exist_ok=True)
    (d / "latest.json").write_text(json.dumps({"date": slug, "leads": leads}, indent=1, ensure_ascii=False))
    (d / f"{slug}.json").write_text(json.dumps(leads, indent=1, ensure_ascii=False))
    md = [f"# Outreach leads, {date_str}", ""]
    for l in leads:
        tag = " (NYC)" if l.get("nyc") else ""
        md.append(f"- **{l.get('company','')}**{tag}, {l.get('signal','')}: {l.get('detail','')} "
                  f"{l.get('why_relevant','')} [source]({l.get('link','')})")
    (d / f"{slug}.md").write_text("\n".join(md) + "\n")


# ---------- 3. audio ----------
def chunks(lines, size=14):
    for i in range(0, len(lines), size):
        yield lines[i:i + size]


def tts_gemini(chunk, wav_path):
    # Gemini 3.x TTS: every text part carries its own speaker tag; no untagged preamble allowed.
    body = {
        "contents": [{"role": "user", "parts": [
            {"text": t, "speechMetadata": {"speaker": s, "style": "warm, lively morning podcast host"}}
            for s, t in chunk
        ]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"multiSpeakerVoiceConfig": {"speakerVoiceConfigs": [
                {"speaker": s, "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": VOICES[s]}}} for s in HOSTS
            ]}},
        },
    }
    out = gemini(TTS_MODEL, body)
    part = next(p for p in out["candidates"][0]["content"]["parts"] if "inlineData" in p)
    data = base64.b64decode(part["inlineData"]["data"])
    mime = part["inlineData"].get("mimeType", "")
    if data[:4] == b"RIFF":  # already a WAV file
        src = wav_path.with_suffix(".in.wav")
        src.write_bytes(data)
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src)]
    else:  # raw 16-bit PCM
        rate = re.search(r"rate=(\d+)", mime)
        src = wav_path.with_suffix(".pcm")
        src.write_bytes(data)
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "s16le", "-ar", rate.group(1) if rate else "24000",
               "-ac", "1", "-i", str(src)]
    subprocess.run(cmd + ["-ar", "24000", "-ac", "1", str(wav_path)], check=True)


def tts_edge(chunk, wav_path, tmp):
    """Free fallback with no key (Microsoft Edge voices)."""
    parts = []
    for j, (s, t) in enumerate(chunk):
        p = tmp / f"{wav_path.stem}_{j}.mp3"
        for attempt in range(4):
            r = subprocess.run([sys.executable, "-m", "edge_tts", "--voice", EDGE_VOICES[s], "--text", t,
                                "--write-media", str(p)])
            if r.returncode == 0 and p.exists() and p.stat().st_size > 0:
                break
            time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"Edge voice failed on line: {t[:80]}")
        parts.append(p)
    concat(parts, wav_path)


def concat(paths, out):
    lst = out.with_suffix(".txt")
    lst.write_text("".join(f"file '{p}'\n" for p in paths))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-ar", "24000", "-ac", "1", str(out)], check=True)


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                          str(path)], capture_output=True, text=True, check=True).stdout
    return float(out)


def ts(sec):
    h, rem = divmod(sec, 3600)
    m, s_ = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s_:06.3f}"


def write_vtt(chunk_times, path):
    """Timed transcript. Line timings inside each audio chunk are estimated from text length."""
    cues, t0 = ["WEBVTT", ""], 0.0
    for ch, dur in chunk_times:
        total = sum(len(t) for _, t in ch) or 1
        t = t0
        for speaker, text in ch:
            d = dur * len(text) / total
            cues += [f"{ts(t)} --> {ts(t + d)}", f"<v {speaker}>{text}", ""]
            t += d
        t0 += dur
    path.write_text("\n".join(cues))


def make_audio(lines, mp3_path):
    chunk_times = []
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
                if not use_edge and not TTS_MODEL:
                    use_edge = True
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
            chunk_times.append((ch, probe(w)))
        joined = tmp / "all.wav"
        concat(wavs, joined)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(joined), "-af", "loudnorm=I=-16",
                        "-codec:a", "libmp3lame", "-b:a", "64k", str(mp3_path)], check=True)
    write_vtt(chunk_times, mp3_path.with_suffix(".vtt"))
    return int(probe(mp3_path))


# ---------- main ----------
def main():
    if not KEY and not DRY:
        sys.exit("GEMINI_API_KEY is not set")
    now = dt.datetime.now(dt.timezone.utc)
    date_str = now.astimezone(dt.timezone(dt.timedelta(hours=-5))).strftime("%A, %B %-d, %Y")
    slug = now.strftime("%Y-%m-%d")
    EPISODES.mkdir(parents=True, exist_ok=True)

    if not DRY:
        pick_models()
    items = gather_news()
    log(f"{len(items)} news items")
    if not items and not DRY:
        sys.exit("No news found; skipping today")
    title, lines = write_script(items, date_str)
    log(f"script: {len(lines)} lines, title: {title}")

    write_leads(extract_leads(items), slug, date_str)

    mp3 = EPISODES / f"{slug}.mp3"
    seconds = make_audio(lines, mp3)

    meta_path = DOCS / "episodes.json"
    eps = json.loads(meta_path.read_text()) if meta_path.exists() else []
    eps = [e for e in eps if e["slug"] != slug]
    links = "\n".join(f"- {it['title']} ({it['source']}): {it['link']}" for it in items[:40])
    eps.insert(0, {
        "slug": slug, "title": title, "date": now.strftime("%a, %d %b %Y %H:%M:%S +0000"),
        "seconds": seconds, "bytes": mp3.stat().st_size,
        "notes": "Full transcript:\n\n" + "\n\n".join(f"{sp}: {t}" for sp, t in lines)
                 + "\n\nSources from the last 24 hours:\n" + links,
    })
    for old in eps[KEEP_EPISODES:]:
        (EPISODES / f"{old['slug']}.mp3").unlink(missing_ok=True)
        (EPISODES / f"{old['slug']}.vtt").unlink(missing_ok=True)
    eps = eps[:KEEP_EPISODES]
    meta_path.write_text(json.dumps(eps, indent=1))
    (DOCS / "scripts").mkdir(exist_ok=True)
    (DOCS / "scripts" / f"{slug}.txt").write_text(title + "\n\n" + "\n".join(f"{s}: {t}" for s, t in lines))
    write_feed(eps, DOCS)
    log(f"done: {mp3.name}, {seconds}s")


if __name__ == "__main__":
    main()
